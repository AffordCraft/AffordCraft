"""Cache only frozen training RGB/instruction features at action query cadence."""
import argparse,json,os,sys,time
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT
import exp3_oft_recurrent_adapter_v0_166 as adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.configuration.read_text());fr=json.loads((a.configuration.parent/'freeze_receipt.json').read_text())
    if sha(a.configuration)!=fr['training_configuration_sha256']:raise ValueError('configuration_changed')
    for path,digest in cfg['code_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('code_changed:'+path)
    adapter.require_dataset(cfg['dataset_manifest'])
    import numpy as np,torch,torch.distributed as dist
    from PIL import Image
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl')
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    dist.barrier();os.environ['HF_MODULES_CACHE']=str(a.output/f'hf_rank_{rank}');start=time.monotonic();base,processor=adapter.build_features(cfg,torch.device('cuda',local));load=time.monotonic()-start
    groups={}
    for line in (Path(cfg['dataset_manifest']).parent/'frame_index.jsonl').open():
        row=json.loads(line);groups.setdefault(row['episode_job'],[]).append(row)
    if len(groups)!=1000:raise ValueError('training_episode_denominator_changed')
    records=[]
    for epi,(job,rows) in enumerate(sorted(groups.items())):
        if epi%world!=rank:continue
        rows.sort(key=lambda x:x['index']);vectors=[];proprio=[];targets=[];masks=[];indices=[];hashes=[];tick=time.monotonic()
        for i in range(0,len(rows),adapter.EXECUTE_PREFIX):
            row=rows[i]
            if sha(row['path'])!=row['sha256']:raise ValueError('training_rgb_changed')
            with Image.open(row['path']) as im:inputs=processor(adapter.prompt(row['instruction']),im.convert('RGB'),return_tensors='pt')
            inputs={k:v.to(local) for k,v in inputs.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):feature=adapter.features(base,**inputs)
            vectors.append(feature[0].cpu().numpy().astype(np.float16));proprio.append(row['proprio']);targets.append(adapter.encode_targets([rows[min(i+k,len(rows)-1)]['action'] for k in range(8)],row['proprio']));masks.append([i+k<len(rows) for k in range(8)]);indices.append(i);hashes.append(row['sha256'])
        path=a.output/f'episode_{epi:04d}.npz';np.savez_compressed(path,features=np.asarray(vectors),proprio=np.asarray(proprio,np.float32),targets=np.asarray(targets,np.float32),valid=np.asarray(masks,np.float32),indices=np.asarray(indices))
        record=dict(episode_job=job,asset_id=rows[0]['asset_id'],query_count=len(indices),path=str(path.resolve()),sha256=sha(path),source_frame_indices=indices,source_rgb_sha256=hashes,supervision_not_model_input=True,heldout_used=False,runtime_seconds=time.monotonic()-tick);records.append(record)
        print(json.dumps(dict(rank=rank,episode=epi,asset_id=rows[0]['asset_id'],queries=len(indices),runtime_seconds=record['runtime_seconds'])),flush=True)
    emit(a.output/f'rank_{rank}_manifest.json',dict(records=records,model_load_seconds=load,runtime_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(local),gpu_name=torch.cuda.get_device_name(local),torch_version=torch.__version__,cuda_version=torch.version.cuda,**ROBOT));dist.barrier()
    if rank==0:
        all_rows=[r for i in range(world) for r in json.loads((a.output/f'rank_{i}_manifest.json').read_text())['records']]
        if len(all_rows)!=1000 or len({r['episode_job'] for r in all_rows})!=1000:raise ValueError('cache_coverage_invalid')
        emit(a.output/'manifest.json',dict(episode_count=1000,query_stride=4,records=sorted(all_rows,key=lambda x:x['episode_job']),configuration_sha256=sha(a.configuration),dataset_manifest_sha256=sha(cfg['dataset_manifest']),frozen_base_features_only=True,heldout_used=False,command=[sys.executable,*sys.argv],**ROBOT))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
