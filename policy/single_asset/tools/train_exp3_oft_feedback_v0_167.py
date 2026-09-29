"""Train feedback-aware VLA head on immutable success sequences with augmentation."""
import argparse,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
import exp3_oft_feedback_adapter_v0_167 as adapter
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.configuration.read_text());fr=json.loads((a.configuration.parent/'freeze_receipt.json').read_text())
    if sha(a.configuration)!=fr['training_configuration_sha256']:raise ValueError('configuration_changed')
    for path,digest in cfg['code_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('code_changed:'+path)
    adapter.require_dataset(cfg['dataset_manifest']);cache_path=Path(cfg['feature_cache_manifest']);cm=json.loads(cache_path.read_text())
    if sha(cache_path)!=cfg['feature_cache_manifest_sha256'] or cm['episode_count']!=1000 or cm['heldout_used'] is not False:raise ValueError('feature_cache_not_frozen_success_only')
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.set_device(0);torch.manual_seed(cfg['adapter_initialization_seed']);rng=random.Random(cfg['seed']);torch.set_num_threads(4)
    model=adapter.RecurrentHead(cfg['feature_dim'],cfg['memory_width'],cfg['proprio_mean'],cfg['proprio_std']).cuda();initial=a.output/'adapter_initial.safetensors';save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(initial));opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    groups={};loaded={}
    for r in cm['records']:
        if sha(r['path'])!=r['sha256']:raise ValueError('cached_feature_changed')
        with np.load(r['path']) as x:loaded[r['episode_job']]={k:x[k].copy() for k in ['features','proprio','targets','valid']}
        groups.setdefault(r['asset_id'],[]).append(r['episode_job'])
    if len(groups)!=10 or any(len(x)!=100 for x in groups.values()):raise ValueError('ten_by_100_missing')
    assets=sorted(groups);weights=torch.tensor([5.]*7+[1.,1.],device='cuda');scales=torch.tensor(adapter.HORIZON_ARM_SCALE,device='cuda')[None,None,:,None];peak=0
    for step in range(cfg['max_steps']):
        tick=time.monotonic();jobs=[rng.choice(groups[assets[(step*cfg['batch_size']+j)%10]]) for j in range(cfg['batch_size'])];n=max(len(loaded[j]['features']) for j in jobs);b=len(jobs)
        feat=np.zeros((b,n,cfg['feature_dim']),np.float32);q=np.zeros((b,n,9),np.float32);target=np.zeros((b,n,8,9),np.float32);mask=np.zeros((b,n,8),np.float32)
        for j,job in enumerate(jobs):
            d=loaded[job];t=len(d['features']);feat[j,:t]=d['features'];q[j,:t]=d['proprio'];target[j,:t]=d['targets'];mask[j,:t]=d['valid']
        augmented,targets,aug=adapter.augment_proprio_and_targets(q,target,mask,cfg,step)
        # Motion weighting uses original training actions solely in the loss.
        motion=np.abs(target[:,:,0,:7]*adapter.HORIZON_ARM_SCALE[0]).max(-1)
        motion_weight=1.+cfg['motion_loss_gain']*np.minimum(motion/cfg['motion_loss_scale_rad'],1.)
        ft=torch.from_numpy(feat).cuda();qt=torch.from_numpy(augmented).cuda();tt=torch.from_numpy(targets).cuda();valid=torch.from_numpy(mask[:,:,:,None]).cuda();mw=torch.from_numpy(motion_weight[:,:,None,None]).cuda();weighted=valid*mw;opt.zero_grad(set_to_none=True);pred,_=model(ft,qt);error=pred-tt
        loss=((error.abs()+.1*error.square())*weights*weighted).sum()/(weighted.sum()*weights.sum())
        pv=weighted[:,:,1:]*valid[:,:,:-1];de=(pred[:,:,1:]-pred[:,:,:-1])-(tt[:,:,1:]-tt[:,:,:-1]);temporal=((de.abs()+.1*de.square())*weights*pv).sum()/(pv.sum()*weights.sum()).clamp_min(1.)
        absolute=qt[:,:,:7][:,:,None,:]+pred[:,:,:,:7]*scales;desired=qt[:,:,:7][:,:,None,:]+tt[:,:,:,:7]*scales
        ov=valid[:,:-1,4:]*valid[:,1:,:4];difference=(absolute[:,:-1,4:]-absolute[:,1:,:4])-(desired[:,:-1,4:]-desired[:,1:,:4]);overlap_loss=(difference.abs()*ov).sum()/(ov.sum()*7).clamp_min(1.)
        total=loss+cfg['temporal_consistency_lambda']*temporal+cfg['query_overlap_lambda']*overlap_loss
        if not torch.isfinite(total):raise RuntimeError('nonfinite_training_loss')
        total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);factor=min(1.,(step+1)/cfg['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*step/cfg['max_steps'])))
        for g in opt.param_groups:g['lr']=cfg['learning_rate']*factor
        opt.step();torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_allocated());arm_mae=(error[:,:,:,:7].abs()*scales*valid).sum()/(valid.sum()*7);finger_mae=(error[:,:,:,7:].abs()*.02*valid).sum()/(valid.sum()*2)
        log=dict(step=step+1,loss=float(total),action_loss=float(loss),temporal_loss=float(temporal),query_overlap_loss_rad=float(overlap_loss),arm_delta_mae_rad=float(arm_mae),finger_target_mae_m=float(finger_mae),gradient_norm=float(norm),learning_rate=opt.param_groups[0]['lr'],runtime_seconds=time.monotonic()-tick,peak_memory_bytes=peak,training_episode_count=b,heldout_used=False,augmentation=aug)
        with (a.output/'training_log_rank_0.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        with (a.output/'training_sampling_rank_0.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,episodes=jobs,augmentation_seed=cfg['augmentation_seed']+step,heldout_used=False))+'\n')
        if (step+1)%20==0 or step==0:print(json.dumps(log),flush=True)
        if (step+1)%cfg['save_every']==0:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(a.output/f'adapter_step_{step+1:06d}.safetensors'))
    final=a.output/'adapter_final.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(final));adapter.RecurrentHead(cfg['feature_dim'],cfg['memory_width']).load_state_dict(load_file(str(final)),strict=True)
    if sha(initial)==sha(final):raise RuntimeError('checkpoint_not_trained')
    emit(a.output/'checkpoint_manifest.json',dict(base_checkpoint=cfg['base_checkpoint'],checkpoint_inventory_sha256=sha(cfg['checkpoint_inventory']),initial_adapter=str(initial.resolve()),initial_adapter_sha256=sha(initial),final_adapter=str(final.resolve()),final_adapter_sha256=sha(final),completed_steps=cfg['max_steps'],checkpoint_reload_passed=True,fixed_final_checkpoint_not_heldout_selected=True,dataset_manifest_sha256=sha(cfg['dataset_manifest']),feature_cache_manifest_sha256=sha(cache_path),training_code_sha256=sha(__file__),**ROBOT))
    emit(a.output/'rank_0_execution_receipt.json',dict(status='training_completed_in_process',steps=cfg['max_steps'],runtime_seconds=time.monotonic()-start,peak_memory_bytes=peak,gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,python_version=sys.version,command=[sys.executable,*sys.argv],scripted_actions_at_inference=False,**ROBOT));emit(a.output/'training_config.json',cfg);emit(a.output/'independent_training_gate.json',dict(training_verified=True,success_only_episode_count=1000,heldout_used=False,cache_files_verified=True,checkpoint_reload_passed=True,initial_and_final_distinct=True,formal_evaluation_allowed=True,**ROBOT))


if __name__=='__main__':main()
