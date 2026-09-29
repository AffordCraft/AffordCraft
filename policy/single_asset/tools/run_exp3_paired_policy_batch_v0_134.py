"""One private Isaac worker; every frozen heldout episode, no teacher fallback."""
import argparse,copy,gc,json,os,sys,time,traceback
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT

def execute(config_path,output):
    cfg=json.loads(config_path.read_text());output.mkdir(parents=True,exist_ok=False)
    dataset=Path(cfg['dataset_manifest']);dm=json.loads(dataset.read_text())
    if not dm.get('training_allowed') or dm.get('training_episode_count')!=1000:raise ValueError('full_dataset_gate_locked')
    cp=json.loads(Path(cfg['checkpoint_manifest']).read_text());field='initial_adapter' if cfg['model_variant']=='untrained_vla' else 'final_adapter'
    if sha(cp[field])!=cp[field+'_sha256']:raise ValueError('policy_checkpoint_changed')
    identity=json.loads(Path(cfg['policy_identity']).read_text())
    if identity['adapter_sha256']!=cp[field+'_sha256']:raise ValueError('policy_server_checkpoint_mismatch')
    plan=[json.loads(line) for line in (dataset.parent/'heldout_split.jsonl').open()]
    wanted=set(cfg['asset_ids']);plan=[x for x in plan if x['asset_id'] in wanted]
    if any(sum(x['asset_id']==aid for x in plan)!=20 for aid in wanted):raise ValueError('twenty_heldout_per_asset_required')
    emit(output/'configuration.json',cfg);emit(output/'input_hashes.json',dict(configuration_sha256=sha(config_path),dataset_manifest_sha256=sha(dataset),heldout_split_sha256=sha(dataset.parent/'heldout_split.jsonl'),policy_identity_sha256=sha(cfg['policy_identity'])))
    start=time.monotonic();original=sys.argv[:];sys.argv=[sys.argv[0]]
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'hide_ui':True,'width':320,'height':240,'renderer':'RayTracedLighting','anti_aliasing':0,'multi_gpu':False,'active_gpu':cfg['gpu_index'],'physics_gpu':cfg['gpu_index'],'fast_shutdown':True});sys.argv=original
    import run_exp3_complete_task_local_policy_v0_134 as driver
    from verify_exp3_paired_robot_episode_v0_134 import inspect
    from isaacsim.core.api import World
    import omni.usd
    load=time.monotonic()-start;records=[]
    try:
        for item in sorted(plan,key=lambda x:(x['asset_id'],x['index'])):
            job=output/'episodes'/item['asset_id']/f"episode_{item['index']:03d}";job.mkdir(parents=True,exist_ok=False)
            spec=copy.deepcopy(item['base_task_spec']);spec['camera_reset_delta']=item['camera_reset_delta'];cm=item['base_case']
            if 'initial_joint_position_si' in item:spec['workcell']['articulated_initial_joint_rad']=item['initial_joint_position_si']
            emit(job/'task_spec.json',spec);emit(job/'case_manifest.json',cm);emit(job/'initial_state_registration.json',item)
            command=item['predecessor_registration']['command'].copy();command[1]=str(Path(driver.__file__).resolve())
            replace={'--case-manifest':str(job/'case_manifest.json'),'--output-dir':str(job/'episode'),'--gpu-index':str(cfg['gpu_index']),'--controller':'local_robot_policy','--model-variant':cfg['model_variant'],'--policy-socket':cfg['policy_socket'],'--policy-identity':cfg['policy_identity'],'--control-steps':str(item['action_budget_steps'])}
            if 'initial_joint_position_si' in item:replace['--articulated-initial-joint-rad']=str(item['initial_joint_position_si'])
            for key,value in replace.items():
                if key in command:command[command.index(key)+1]=value
                else:command += [key,value]
            os.environ['EXP3_COMPLETE_TASK_SPEC']=str(job/'task_spec.json');os.environ['EXP3_COLLECTION_SEED']=str(item['seed']);center=item.get('initial_center_world_m',[-.15,0.]);os.environ['EXP3_INITIAL_X_M']=str(center[0]);os.environ['EXP3_INITIAL_Y_M']=str(center[1])
            emit(job/'exact_command.json',dict(argv=command,batch_invocation=original,model_variant=cfg['model_variant'],teacher_enabled=False,environment={k:os.environ[k] for k in ['EXP3_COMPLETE_TASK_SPEC','EXP3_COLLECTION_SEED','EXP3_INITIAL_X_M','EXP3_INITIAL_Y_M']}))
            sys.argv=command[1:];args=driver.parse_args();sys.argv=original;tick=time.monotonic();failure=None
            try:driver.run_case(args,app)
            except Exception as exc:
                failure=type(exc).__name__+':'+str(exc);emit(job/'runtime_failure.json',dict(failure_reason=failure,traceback=traceback.format_exc()))
            emit(job/'call_receipt.json',dict(episode_call_completed_normally=failure is None,in_process_result_present=(job/'episode/case_result.json').exists(),subprocess_returncode=None,process_exit_pending=True,runtime_seconds=time.monotonic()-tick,output_hashes={str(p.relative_to(job)):sha(p) for p in job.rglob('*') if p.is_file()},**ROBOT))
            try:verification=inspect(job,require_process_exit=False) if failure is None else dict(sim_task_success=None,evidence_valid=False,failure_reasons=[failure])
            except Exception as exc:verification=dict(sim_task_success=None,evidence_valid=False,failure_reasons=[type(exc).__name__+':'+str(exc)])
            row=dict(asset_id=item['asset_id'],task_id=spec['task_id'],model_variant=cfg['model_variant'],seed=item['seed'],heldout_index=item['index'],job=str(job),runtime_seconds=time.monotonic()-tick,independent_provisional_verification=verification,process_exit_pending=True,**ROBOT)
            emit(job/'provisional_verification.json',row);records.append(row);print(json.dumps(dict(asset_id=item['asset_id'],index=item['index'],model_variant=cfg['model_variant'],success=verification.get('sim_task_success'),reason=verification.get('failure_reasons',[]))),flush=True)
            world=World.instance()
            if world is not None:world.stop();world.clear();World.clear_instance()
            omni.usd.get_context().close_stage()
            for _ in range(3):app.update()
            gc.collect()
        emit(output/'in_process_batch_result.json',dict(model_variant=cfg['model_variant'],records=records,expected_episode_count=len(plan),model_load_seconds=load,runtime_seconds=time.monotonic()-start,process_exit_pending=True,**ROBOT))
    finally:app.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();execute(a.configuration,a.output)

if __name__=='__main__':main()
