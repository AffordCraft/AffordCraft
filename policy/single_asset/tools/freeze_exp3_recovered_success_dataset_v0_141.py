"""Same frozen cohort; assemble normal original or explicit infrastructure reruns.

No model, candidate or TaskSpec selection. A rerun is eligible only for the same
asset, original seed/reset/camera plan, unchanged robot teacher and physics.
"""
import argparse,copy,json,sys,time
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT
from freeze_exp3_full_task_success_dataset_v0_133 import audit_actions
from exp3_training_instruction_contract_v0_134 import instruction,release_evidence
from i2ia.tasks.full_task_dataset_coverage import coverage

def register(a):
    a.output.mkdir(parents=True,exist_ok=False);original=json.loads((a.collection/'configuration.json').read_text());members=[];paths=json.loads(a.recovery_path_map.read_text()) if a.recovery_path_map else {}
    for row in original['jobs']:
        aid=row['asset_id'];batch=Path(row['batch']);base=json.loads((batch/'configuration.json').read_text());repair=Path(paths.get(aid,str(a.recovery_root/(aid+'-descriptor-recovery-001'))))
        members.append(dict(asset_id=aid,asset_role=base['asset_role'],original_configuration=str(batch/'configuration.json'),original_configuration_sha256=sha(batch/'configuration.json'),original_verification=str(a.original_admission/'per_asset'/aid/'independent_verification.json'),registered_recovery=str(repair),candidate_id=base['base_spec']['candidate_id'],candidate_sha256=base['source_asset_sha256']))
    cfg=dict(collection=str(a.collection),collection_configuration_sha256=sha(a.collection/'configuration.json'),cohort_release=original['cohort_release'],cohort_release_sha256=original['cohort_release_sha256'],assets=members,selection_rule='normal original final independent proof with exactly 100 successes; only for an invalid original process, same-definition preregistered segmented infrastructure recollection; never best-of methods/candidates/tasks',target_per_asset=100,heldout_per_asset=20,required_asset_count=10,source_seed_state_camera_rules_changed=False,physics_teacher_changed=False,original_result_files_modified=False,code_sha256=sha(__file__),command=[sys.executable,*sys.argv],**ROBOT)
    emit(a.output/'configuration.json',cfg);emit(a.output/'freeze_receipt.json',dict(configuration_sha256=sha(a.output/'configuration.json'),source_collection_configuration_sha256=sha(a.collection/'configuration.json')))

def select_sources(cfg):
    selected=[];pending=[];errors=[]
    for row in cfg['assets']:
        proofpath=Path(row['original_verification']);proof=json.loads(proofpath.read_text()) if proofpath.exists() else None
        if proof and proof.get('target_met') is True and proof.get('training_count')==100 and proof.get('process_exit_verified') is True:
            selected.append(dict(**row,selected_kind='original_normal_completed',verification=str(proofpath),per_episode=str(proofpath.parent/'per_episode.jsonl')));continue
        repair=Path(row['registered_recovery']);rp=repair/'independent_verification.json'
        if not proof or not rp.exists():pending.append(row['asset_id']);continue
        original_exit=json.loads((Path(row['original_configuration']).parent/'execution_receipt.json').read_text());rc=original_exit.get('subprocess_returncode');registered=json.loads((repair/'configuration.json').read_text());repaired=json.loads(rp.read_text())
        if rc==0:errors.append('normal_source_cannot_be_replaced_by_descriptor_rerun:'+row['asset_id']);continue
        if registered['source_configuration_sha256']!=row['original_configuration_sha256'] or registered['asset_sha256']!=row['candidate_sha256'] or registered['physical_task_changed'] is not False or registered['teacher_policy_changed'] is not False:errors.append('recovery_changes_frozen_definition:'+row['asset_id']);continue
        if repaired.get('target_met') is not True or repaired.get('training_count')!=100:errors.append('recovery_has_success_shortfall:'+row['asset_id']);continue
        selected.append(dict(**row,selected_kind='explicit_same_definition_infrastructure_recollection',verification=str(rp),per_episode=str(repair/'per_episode.jsonl'),original_failed_exit=rc,recovery_configuration_sha256=sha(repair/'configuration.json')))
    return selected,pending,errors

def freeze(cfg,selected,output):
    output.mkdir(parents=True,exist_ok=False);episodes=[];heldout=[];frames=[];errors=[];bindings={}
    collection=Path(cfg['collection']);gate=json.loads(Path(cfg['cohort_release']).read_text())
    if sha(collection/'configuration.json')!=cfg['collection_configuration_sha256'] or sha(cfg['cohort_release'])!=cfg['cohort_release_sha256']:raise ValueError('original_cohort_changed')
    for row in selected:
        source=Path(row['original_configuration']);base=json.loads(source.read_text());proof=json.loads(Path(row['verification']).read_text());pr=Path(row['per_episode'])
        if sha(source)!=row['original_configuration_sha256'] or sha(pr)!=proof['per_episode_sha256']:raise ValueError('selected_source_proof_changed')
        if sha(base['base_case']['asset']['usd_path'])!=row['candidate_sha256']:raise ValueError('original_asset_changed')
        bindings[str(source)]=sha(source);bindings[row['verification']]=sha(row['verification']);bindings[str(pr)]=sha(pr)
        if row['selected_kind']=='explicit_same_definition_infrastructure_recollection':
            for segment in proof['segments']:
                root=Path(segment['path']);config=json.loads((root/'configuration.json').read_text())
                if sha(root/'configuration.json')!=segment['configuration_sha256'] or sha(root/'independent/independent_verification.json')!=segment['verification_sha256'] or not segment['normal_exit']:raise ValueError('recovery_segment_binding_or_exit_invalid')
                for key in ['base_spec','base_case','source_asset_sha256','predecessor_registration','source_dependencies']:
                    if config[key]!=base[key]:raise ValueError('recovery_task_or_robot_teacher_changed:'+key)
                original_seeds={x['index']:x for x in base['episodes'] if x['split']=='train'}
                if any(x!=original_seeds.get(x['index']) for x in config['episodes']):raise ValueError('recovery_seed_or_initial_state_changed')
        heldout.extend(dict(**item,base_task_spec=base['base_spec'],base_case=base['base_case'],predecessor_registration=base['predecessor_registration'],action_budget_steps=360) for item in base['episodes'] if item['split']=='heldout')
        scheduled={x['index']:x for x in base['episodes'] if x['split']=='train'}
        for entry in map(json.loads,pr.read_text().splitlines()):
            if not entry['training_admitted']:continue
            job=Path(entry['job']);spec=json.loads((job/'task_spec.json').read_text());initial=json.loads((job/'initial_state_registration.json').read_text());prior=json.loads((job/'provisional_verification.json').read_text());verification=entry['verification']
            try:
                if initial!=scheduled[entry['index']]:raise ValueError('initial_seed_state_camera_changed')
                expected=copy.deepcopy(base['base_spec']);expected['camera_reset_delta']=initial['camera_reset_delta']
                if 'initial_joint_position_si' in initial:expected['workcell']['articulated_initial_joint_rad']=initial['initial_joint_position_si']
                if spec!=expected:raise ValueError('task_spec_changed')
                if verification.get('process_exit_verified') is not True or verification['full_task_evaluation']['sim_task_success'] is not True:raise ValueError('full_task_and_process_exit_not_verified')
                observation,state_sha=audit_actions(job,spec);wording=instruction(spec);release=release_evidence(spec,json.loads((job/'episode/complete_task_raw_trace.json').read_text())['states'])
                if not release['passed']:raise ValueError('release_then_hold_clause_not_verified')
            except Exception as exc:errors.append(row['asset_id']+':'+str(entry['index'])+':'+str(exc));continue
            record=dict(asset_id=row['asset_id'],asset_role=row['asset_role'],seed=initial['seed'],split='train',task_id=spec['task_id'],task_instruction=wording['training_and_paired_evaluation_instruction'],instruction_lineage=wording,job=str(job),sim_task_success=True,independent_admission_passed=True,model_variant='demonstration_teacher',physical_trajectory_fingerprint=prior['physical_trajectory_fingerprint'],initial_state_camera_fingerprint=initial['initial_state_camera_fingerprint'],task_spec_sha256=sha(job/'task_spec.json'),state_trace_sha256=state_sha,raw_trace_sha256=verification['trace_sha256'],videos=verification['media']['videos'],action_count=len(observation),source_collection_kind=row['selected_kind'],release_evidence=release,**ROBOT)
            episodes.append(record)
            for frame in observation:frames.append(dict(**frame,episode_job=str(job),asset_id=row['asset_id'],task_id=spec['task_id'],instruction=record['task_instruction']))
    result=coverage(gate['assets'],episodes,heldout);result['errors']=sorted(set(result['errors']+errors));result['training_allowed']=result['training_allowed'] and not result['errors']
    for name,rows in [('train_split.jsonl',episodes),('heldout_split.jsonl',heldout),('frame_index.jsonl',frames),('source_selection_lineage.jsonl',selected)]:
        with (output/name).open('x') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    emit(output/'dataset_quality_report.json',result)
    emit(output/'training_input_contract.json',dict(observation='one raw 320x240 Isaac RGB, frozen language instruction and measured nine robot joint positions',action_dim=9,proprio_dim=9,control_hz=10,physics_hz=60,object_state_in_model_input=False,teacher_action_in_model_input=False,heldout_seed_state_camera_unchanged=True,raw_isaac_video_required=True,**ROBOT))
    hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()};manifest=dict(training_allowed=result['training_allowed'],training_episode_count=len(episodes),heldout_episode_count=len(heldout),frame_count=len(frames),required_assets=10,required_successes_per_asset=100,dataset_files_sha256=hashes,source_proof_hashes=bindings,all_selected_episodes_process_exit_verified=not errors,collector_success_used=False,training_started=False,**ROBOT)
    emit(output/'successful_trajectory_dataset_manifest.json',manifest);emit(output/'freeze_receipt.json',dict(manifest_sha256=sha(output/'successful_trajectory_dataset_manifest.json'),dataset_files_sha256=hashes,training_allowed=result['training_allowed']))
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--original-admission',type=Path,required=True);p.add_argument('--recovery-root',type=Path,required=True);p.add_argument('--recovery-path-map',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();a.collection=a.collection.resolve();a.original_admission=a.original_admission.resolve();a.recovery_root=a.recovery_root.resolve();a.output=a.output.resolve();start=time.monotonic()
    if not a.resume:register(a)
    cfg=json.loads((a.output/'configuration.json').read_text())
    if sha(a.output/'configuration.json')!=json.loads((a.output/'freeze_receipt.json').read_text())['configuration_sha256'] or sha(__file__)!=cfg['code_sha256']:raise ValueError('recovered_dataset_definition_changed')
    previous=None
    while True:
        selected,pending,errors=select_sources(cfg);state=(len(selected),tuple(pending),tuple(errors))
        if state!=previous:print(json.dumps(dict(ready_assets=len(selected),pending=pending,errors=errors)),flush=True);previous=state
        if errors or not pending:break
        time.sleep(15)
    result={}
    try:
        if errors:raise RuntimeError(';'.join(errors))
        result=freeze(cfg,selected,a.output/'frozen_dataset')
    except Exception as exc:emit(a.output/'failure.json',dict(failure_reason=type(exc).__name__+':'+str(exc)))
    allowed=result.get('training_allowed',False)
    emit(a.output/'execution_receipt.json',dict(status='dataset_admitted' if allowed else 'dataset_blocked',training_allowed=allowed,all_selected_episodes_process_exit_verified=result.get('all_selected_episodes_process_exit_verified',False),dataset_freeze_returncode=0 if allowed else 1,dataset_manifest=str(a.output/'frozen_dataset/successful_trajectory_dataset_manifest.json'),runtime_seconds=time.monotonic()-start,command=[sys.executable,*sys.argv],training_started=False,**ROBOT));emit(a.output/'completion_marker.json',dict(training_allowed=allowed,execution_receipt_sha256=sha(a.output/'execution_receipt.json')))

if __name__=='__main__':main()
