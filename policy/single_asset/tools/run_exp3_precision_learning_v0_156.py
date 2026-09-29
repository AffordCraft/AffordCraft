"""Detached lifecycle: final dataset gate -> real training -> paired simulation.

All child processes are owned, private, and logged. Unrelated jobs are never
stopped. Checkpoint/physics success is never inferred from exit code alone.
"""
import argparse,json,os,subprocess,sys,time,traceback,tempfile
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,probe,ROBOT

def event(root,kind,**value):
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=time.time(),kind=kind,**value),sort_keys=True)+'\n')

def sample(root,phase):
    data=dict(timestamp=time.time(),phase=phase,loadavg=list(os.getloadavg()),gpu=probe(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader']),processes=probe(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
    with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(data,sort_keys=True)+'\n')
    return data

def read(path):return json.loads(Path(path).read_text())

def check_definition(config):
    cfg=read(config);fr=read(config.parent/'freeze_receipt.json')
    if sha(config)!=fr['lifecycle_configuration_sha256']:raise ValueError('lifecycle_definition_changed')
    for name,key in [('training_configuration','training_configuration_sha256'),('evaluation_configuration','evaluation_configuration_sha256')]:
        if sha(cfg[name])!=fr[key]:raise ValueError('registered_configuration_changed:'+name)
    for path,digest in fr['code_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('registered_code_changed:'+path)
    return cfg

def wait_idle(root):
    waiting=False
    while True:
        data=sample(root,'waiting_for_private_gpus')
        if data['processes']['returncode']==0 and not data['processes']['stdout'].strip():return
        if not waiting:event(root,'waiting_for_unrelated_or_finishing_gpu_jobs_no_preemption');waiting=True
        time.sleep(30)

def environment(root,repo,isaac_python,isaac=False):
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    tmp=root/'tmp';tmp.mkdir(parents=True,exist_ok=True)
    env.update(PYTHONPATH=str(repo/'src')+':'+str(repo/'tools'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(tmp),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    if isaac:
        lib=Path(isaac_python).parent.parent/'lib';env.update(LD_PRELOAD=str(lib/'libstdc++.so.6'),LD_LIBRARY_PATH=str(lib),ACCEPT_EULA='Y',OMNI_KIT_ACCEPT_EULA='YES')
    return env

def run_owned(command,root,name,env,repo,resources):
    emit(root/(name+'_command.json'),dict(argv=command,environment={k:env[k] for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TMPDIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','LD_PRELOAD','LD_LIBRARY_PATH','ACCEPT_EULA','OMNI_KIT_ACCEPT_EULA'] if k in env}))
    tick=time.monotonic()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=repo,env=env,start_new_session=True)
        emit(root/(name+'_pid.json'),dict(pid=child.pid,command=command,purpose=name));last=0.
        while child.poll() is None:
            if resources and time.monotonic()-last>=30:sample(root,name);last=time.monotonic()
            time.sleep(5)
    result=dict(subprocess_returncode=child.returncode,pid=child.pid,runtime_seconds=time.monotonic()-tick,command=command,**ROBOT)
    emit(root/(name+'_execution_receipt.json'),result);return result

def finish_worker(run,worker,returncode,elapsed):
    result=worker/'in_process_batch_result.json';receipt=worker/'execution_receipt.json'
    if not worker.exists():worker.mkdir(parents=True)
    emit(receipt,dict(subprocess_returncode=returncode,in_process_result_present=result.exists(),in_process_result_sha256=sha(result) if result.exists() else None,runtime_seconds=elapsed,**ROBOT))
    for job in sorted((worker/'episodes').glob('*/*')):
        call=job/'call_receipt.json'
        if not call.exists():continue
        value=read(call);emit(job/'execution_receipt.json',dict(**{k:v for k,v in value.items() if k not in ['subprocess_returncode','process_exit_pending']},subprocess_returncode=returncode,process_exit_pending=False,batch_execution_receipt=str(receipt),batch_execution_receipt_sha256=sha(receipt),episode_call_receipt_sha256=sha(call)))
    emit(worker/'completion_marker.json',dict(episode_evidence_requires_final_independent_verification=True,execution_receipt_sha256=sha(receipt)))

def paired_variant(cfg,root,variant):
    repo=Path(cfg['repo']);ec=read(cfg['evaluation_configuration']);tc=Path(cfg['training_configuration']);cp=root/'training/checkpoint_manifest.json';variant_root=root/'evaluation'/variant;variant_root.mkdir(parents=True,exist_ok=False)
    wait_idle(root);active={};pending={gpu:[x['asset_id'] for x in ec['assets'] if x['gpu_index']==gpu] for gpu in range(8)};outcomes=[]
    for gpu,aids in pending.items():
        if not aids:continue
        parent=variant_root/f'gpu_{gpu}';parent.mkdir();service=parent/'policy';socket=Path(tempfile.mkdtemp(prefix=f'exp3-v156-{os.getpid()}-{variant[:1]}-{gpu}-'))/'policy.sock'
        # Socket basename depends on this supervisor, not another process or
        # case. Existing paths are never reused or overwritten.
        command=[cfg['vla_python'],str(repo/'tools/serve_exp3_precision_vla_v0_156.py'),'--training-configuration',str(tc),'--checkpoint-manifest',str(cp),'--variant',variant,'--gpu',str(gpu),'--output',str(service),'--socket',str(socket)]
        env=environment(parent,repo,cfg['isaac_python']);log=(parent/'policy_service.log').open('x');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=repo,env=env)
        emit(parent/'policy_launch.json',dict(pid=child.pid,command=command,environment={k:env[k] for k in ['PYTHONPATH','TMPDIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']},asset_ids=aids))
        active[gpu]=dict(parent=parent,service=service,socket=socket,process=child,log=log,started=time.monotonic(),asset_ids=aids,remaining_asset_ids=list(aids),worker=None)
    last=0.
    while active:
        for gpu,state in list(active.items()):
            parent=state['parent'];service=state['service'];server=state['process'];worker=state['worker']
            if worker is None:
                if server.poll() is not None or (time.monotonic()-state['started']>900 and not (service/'ready.json').exists()):
                    reason='policy_service_load_failed' if server.poll() is not None else 'policy_service_load_timeout'
                    if server.poll() is None:server.terminate();server.wait(timeout=60)
                    state['log'].close();emit(parent/'policy_terminal_receipt.json',dict(subprocess_returncode=server.returncode,status='blocked',failure_reason=reason,asset_ids=state['asset_ids']))
                    outcomes.append(dict(gpu_index=gpu,asset_ids=state['asset_ids'],status='blocked',failure_reason=reason));del active[gpu];continue
                if not (service/'ready.json').exists():continue
                aid=state['remaining_asset_ids'].pop(0);worker_parent=parent/'assets'/aid;worker_parent.mkdir(parents=True,exist_ok=False)
                wc=dict(dataset_manifest=cfg['dataset_manifest'],checkpoint_manifest=str(cp),model_variant=variant,asset_ids=[aid],gpu_index=gpu,policy_socket=str(state['socket']),policy_identity=str(service/'policy_identity.json'),preregistered_evaluation_configuration=cfg['evaluation_configuration'],preregistered_evaluation_configuration_sha256=sha(cfg['evaluation_configuration']))
                config=worker_parent/'worker_configuration.json';emit(config,wc);out=worker_parent/'worker';command=[cfg['isaac_python'],str(repo/'tools/run_exp3_paired_policy_batch_v0_134.py'),'--configuration',str(config),'--output',str(out)]
                env=environment(worker_parent/'isaac_private',repo,cfg['isaac_python'],True);log=(worker_parent/'isaac_worker.log').open('x');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=repo,env=env)
                emit(worker_parent/'worker_launch.json',dict(pid=child.pid,command=command,configuration_sha256=sha(config),environment={k:env[k] for k in ['PYTHONPATH','TMPDIR','LD_PRELOAD','LD_LIBRARY_PATH','ACCEPT_EULA','OMNI_KIT_ACCEPT_EULA']}))
                state.update(worker=child,worker_log=log,worker_started=time.monotonic(),worker_output=out,current_asset_id=aid);event(root,'paired_worker_started',variant=variant,gpu_index=gpu,asset_ids=[aid],pid=child.pid)
            elif worker.poll() is not None:
                state['worker_log'].close();finish_worker(root,state['worker_output'],worker.returncode,time.monotonic()-state['worker_started'])
                outcomes.append(dict(gpu_index=gpu,asset_ids=[state['current_asset_id']],worker_returncode=worker.returncode,worker_output=str(state['worker_output'])))
                event(root,'paired_worker_finished',variant=variant,gpu_index=gpu,asset_id=state['current_asset_id'],returncode=worker.returncode)
                if state['remaining_asset_ids']:
                    state['worker']=None
                    continue
                # Only our own now-unneeded model service receives SIGTERM.
                if server.poll() is None:server.terminate()
                try:server.wait(timeout=60)
                except subprocess.TimeoutExpired:raise RuntimeError('owned_policy_service_did_not_stop_no_other_job_touched')
                state['log'].close();emit(parent/'policy_terminal_receipt.json',dict(subprocess_returncode=server.returncode,status='service_finished_after_worker',runtime_seconds=time.monotonic()-state['started'],asset_ids=state['asset_ids']))
                del active[gpu]
        if time.monotonic()-last>=30:sample(root,variant);last=time.monotonic()
        if active:time.sleep(5)
    emit(variant_root/'execution_receipt.json',dict(model_variant=variant,workers=outcomes,expected_episode_count=200,success_not_inferred=True,**ROBOT))

def execute(a):
    cfg=check_definition(a.configuration);root=Path(a.output or cfg['output']);root.mkdir(parents=True,exist_ok=False);repo=Path(cfg['repo']);start=time.monotonic();emit(root/'configuration.json',dict(**cfg,actual_output=str(root),invocation=[sys.executable,*sys.argv]));sample(root,'lifecycle_started_no_model')
    try:
        admission=Path(cfg['admission']);collection=Path(cfg['collection']);event(root,'waiting_for_final_independent_1000_success_admission')
        while not (admission/'completion_marker.json').exists():time.sleep(15)
        gate=read(admission/'execution_receipt.json')
        if gate.get('training_allowed') is not True or gate.get('dataset_freeze_returncode')!=0:raise RuntimeError('training_blocked_final_success_dataset_shortfall_or_verification_failure')
        if gate.get('all_selected_episodes_process_exit_verified') is not True:raise ValueError('selected_episode_process_exit_not_verified')
        # The original failed run remains failed. The new dataset may use only
        # independently verified, cleanly exited same-definition recollections.
        cfg=check_definition(a.configuration);event(root,'final_dataset_admitted',dataset_manifest_sha256=sha(cfg['dataset_manifest']))
        wait_idle(root);sample(root,'training_before');env=environment(root,repo,cfg['isaac_python']);env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),NCCL_DEBUG='WARN')
        command=[cfg['vla_python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(repo/'tools/train_exp3_oft_precision_v0_156.py'),'--configuration',cfg['training_configuration'],'--output',str(root/'training')]
        if a.resume_training_receipt:command.extend(['--resume-receipt',str(a.resume_training_receipt)])
        result=run_owned(command,root,'training',env,repo,True);sample(root,'training_after')
        if result['subprocess_returncode']!=0 or not (root/'training/checkpoint_manifest.json').exists():raise RuntimeError('training_failed_no_task_success_claim')
        command=[cfg['vla_python'],str(repo/'tools/verify_exp3_actual_training_v0_138.py'),'--configuration',cfg['training_configuration'],'--run',str(root),'--output',str(root/'training_independent')]
        result=run_owned(command,root,'training_verifier',env,repo,False)
        if result['subprocess_returncode'] or read(root/'training_independent/independent_verification.json').get('paired_evaluation_allowed') is not True:raise RuntimeError('actual_training_verifier_failed')
        for variant in read(cfg['evaluation_configuration'])['variants']:paired_variant(cfg,root,variant)
        command=[cfg['isaac_python'],str(repo/'tools/summarize_exp3_paired_learning_v0_141.py'),'--configuration',cfg['evaluation_configuration'],'--run',str(root),'--output',str(root/'final_admission')]
        result=run_owned(command,root,'paired_verifier',environment(root/'cpu_verifier',repo,cfg['isaac_python']),repo,False)
        if result['subprocess_returncode']:raise RuntimeError('paired_independent_verifier_or_report_failed')
        status='training_and_paired_evaluation_completed_requires_report_review';failure=None
    except Exception as exc:
        status='blocked';failure=type(exc).__name__+':'+str(exc);emit(root/'failure.json',dict(failure_reason=failure,traceback=traceback.format_exc(),history_unchanged=True,**ROBOT));event(root,'blocked',failure_reason=failure)
    emit(root/'execution_receipt.json',dict(status=status,failure_reason=failure,runtime_seconds=time.monotonic()-start,code_sha256=sha(__file__),configuration_sha256=sha(a.configuration),training_checkpoint_present=(root/'training/checkpoint_manifest.json').exists(),**ROBOT))
    emit(root/'completion_marker.json',dict(status=status,execution_receipt_sha256=sha(root/'execution_receipt.json'),robot_real_world='not_evaluated'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--resume-training-receipt',type=Path);a=p.parse_args();execute(a)

if __name__=='__main__':main()

