"""Training-only controlled proprio perturbation; no simulator or heldout."""
import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from exp3_oft_recurrent_adapter_v0_166 import RecurrentHead, HORIZON_ARM_SCALE
from run_exp3_remote_pilot_jobs_v0_125 import emit, sha, ROBOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    read=lambda p:json.loads(Path(p).read_text());tc=read(a.registration/'training_configuration.json');cp=read(a.run/'training/checkpoint_manifest.json');cache=read(a.run/'features/manifest.json');rng=np.random.default_rng(167031);model=RecurrentHead(tc['feature_dim'],tc['memory_width']);model.load_state_dict(load_file(cp['final_adapter']),strict=True);model.eval();scale=np.asarray(HORIZON_ARM_SCALE,np.float32)[None,:,None];groups=collections.defaultdict(list)
    config=dict(seed=167031,arm_proprio_perturbation_std_rad=.005,correlation_rho=.8,feature_source='frozen training-only RGB/instruction',all_training_episodes=True,heldout_used=False,diagnostic_only=True,projection_into_simulator=False,checkpoint_sha256=cp['final_adapter_sha256'],cache_manifest_sha256=sha(a.run/'features/manifest.json'),**ROBOT);emit(a.output/'configuration.json',config)
    for n,r in enumerate(cache['records']):
        if sha(r['path'])!=r['sha256']:raise ValueError('cache_changed')
        with np.load(r['path']) as d:feat=d['features'].astype(np.float32);q=d['proprio'].copy();target=d['targets'].copy();valid=d['valid'].astype(bool)
        target=np.concatenate([q[:,None,:7]+target[:,:,:7]*scale,(target[:,:,7:]+1)*.02],-1);noise=np.zeros_like(q)
        for i in range(len(q)):
            previous=noise[i-1,:7] if i else np.zeros(7)
            noise[i,:7]=.8*previous+.005*np.sqrt(1-.8**2)*rng.standard_normal(7)
        outputs=[]
        with torch.inference_mode():
            for prop in [q,q+noise]:
                pred,_=model(torch.from_numpy(feat)[None],torch.from_numpy(prop)[None]);pred=pred[0].numpy();outputs.append(np.concatenate([prop[:,None,:7]+pred[:,:,:7]*scale,(pred[:,:,7:]+1)*.02],-1))
        prefix=valid.copy();prefix[:,4:]=False
        for joint in range(7):
            record=dict(asset_id=r['asset_id'],joint=joint,normal_mae_rad=float(np.abs(outputs[0]-target)[:,:,joint][prefix].mean()),perturbed_mae_rad=float(np.abs(outputs[1]-target)[:,:,joint][prefix].mean()),target_sensitivity=float(np.abs(outputs[1]-outputs[0])[:,:,joint][prefix].mean()),mean_abs_input_perturbation_rad=float(np.broadcast_to(np.abs(noise[:,None,joint]),valid.shape)[prefix].mean()))
            groups[(r['asset_id'],joint)].append(record)
        if (n+1)%100==0:print(json.dumps(dict(episodes=n+1,runtime_seconds=time.monotonic()-start)),flush=True)
    rows=[]
    for (asset,joint),items in sorted(groups.items()):
        row=dict(asset_id=asset,joint=joint,episodes=len(items));row.update({k:sum(x[k] for x in items)/len(items) for k in ['normal_mae_rad','perturbed_mae_rad','target_sensitivity','mean_abs_input_perturbation_rad']});row['absolute_target_sensitivity_per_input_delta']=row['target_sensitivity']/row['mean_abs_input_perturbation_rad'];rows.append(row)
    with (a.output/'feedback_sensitivity.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(episodes=len(cache['records']),rows=rows,all_joint_normal_mae_rad=sum(r['normal_mae_rad'] for r in rows)/len(rows),all_joint_perturbed_mae_rad=sum(r['perturbed_mae_rad'] for r in rows)/len(rows),mean_absolute_target_sensitivity_per_input_delta=sum(r['absolute_target_sensitivity_per_input_delta'] for r in rows)/len(rows),interpretation='A gain near one means observation perturbation largely passes into the predicted absolute target on teacher RGB; it is not a closed-loop success result.',heldout_used=False,raw_training_trajectory_not_modified=True,**ROBOT)
    emit(a.output/'summary.json',summary);emit(a.output/'execution_receipt.json',dict(command=[sys.executable,*sys.argv],runtime_seconds=time.monotonic()-start,code_sha256=sha(__file__),gpu_used=False,**ROBOT));emit(a.output/'freeze_receipt.json',dict(files_sha256={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}));print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
