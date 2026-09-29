"""Independent training/split/cache/normalization admission before Isaac."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from exp3_oft_feedback_adapter_v0_167 import RecurrentHead,require_dataset
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();cfg=json.loads(a.configuration.read_text());errors=[];require_dataset(cfg['dataset_manifest']);cache=json.loads(Path(cfg['feature_cache_manifest']).read_text());cp=json.loads((a.run/'training/checkpoint_manifest.json').read_text());root=Path(cfg['dataset_manifest']).parent
    train=[json.loads(l) for l in (root/'train_split.jsonl').open()];allowed={r['job'] for r in train};cache_jobs={r['episode_job'] for r in cache['records']}
    if len(allowed)!=1000 or allowed!=cache_jobs or cache['heldout_used'] is not False or sha(cfg['feature_cache_manifest'])!=cfg['feature_cache_manifest_sha256']:errors.append('cache_training_split_invalid')
    for r in cache['records']:
        if sha(r['path'])!=r['sha256'] or r['heldout_used'] is not False:errors.append('cache_hash_or_split')
    q=np.asarray([json.loads(l)['proprio'] for l in (root/'frame_index.jsonl').open()],np.float64)
    if not np.allclose(q.mean(0),cfg['proprio_mean'],atol=1e-9) or not np.allclose(np.maximum(q.std(0),[.1]*7+[.01,.01]),cfg['proprio_std'],atol=1e-9):errors.append('normalization_not_frozen_training_statistics')
    logs=[json.loads(l) for l in (a.run/'training/training_log_rank_0.jsonl').open()];samples=[json.loads(l) for l in (a.run/'training/training_sampling_rank_0.jsonl').open()]
    if [x['step'] for x in logs]!=list(range(1,cfg['max_steps']+1)) or len(samples)!=cfg['max_steps']:errors.append('incomplete_optimization')
    for x in logs:
        if not all(math.isfinite(x[k]) for k in ['loss','arm_delta_mae_rad','finger_target_mae_m','runtime_seconds']) or x['heldout_used'] is not False:errors.append('invalid_training_log')
    for x in samples:
        if x['heldout_used'] is not False or not set(x['episodes'])<=allowed or x['augmentation_seed']!=cfg['augmentation_seed']+x['step']-1:errors.append('unregistered_training_input_or_seed')
    for key in ['initial_adapter','final_adapter']:
        if sha(cp[key])!=cp[key+'_sha256']:errors.append('checkpoint_hash')
        state=load_file(cp[key]);RecurrentHead(cfg['feature_dim'],cfg['memory_width']).load_state_dict(state,strict=True)
        if not np.allclose(state['proprio_mean'].numpy(),cfg['proprio_mean'],atol=1e-7) or not np.allclose(state['proprio_std'].numpy(),cfg['proprio_std'],atol=1e-7):errors.append('saved_normalization_changed')
    if cp['initial_adapter_sha256']==cp['final_adapter_sha256'] or cp['completed_steps']!=cfg['max_steps']:errors.append('checkpoint_not_trained')
    report=dict(training_verified=not errors,paired_evaluation_allowed=not errors,errors=sorted(set(errors)),training_episodes=1000,heldout_used=False,optimization_steps=len(logs),feature_cache_sha256=sha(cfg['feature_cache_manifest']),checkpoint_manifest_sha256=sha(a.run/'training/checkpoint_manifest.json'),total_augmented_sequences=sum(r['augmentation']['augmented_sequences'] for r in logs),augmented_label_projection_fraction=sum(r['augmentation']['projected_targets'] for r in logs)/sum(r['augmentation']['valid_targets'] for r in logs),runtime_seconds=time.monotonic()-start,command=[sys.executable,*sys.argv],verifier_sha256=sha(__file__),**ROBOT);emit(a.output/'independent_verification.json',report);print(json.dumps(report));return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
