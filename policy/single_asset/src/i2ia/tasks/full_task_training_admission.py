"""Independent dataset admission, stricter than a collector's success boolean."""
import math
from .complete_robot_precision import evaluate


def audit_contact_identity(trace):
    errors=[];matched=0
    for i,state in enumerate(trace.get('states',[])):
        for contact in state.get('contacts',[]):
            found=False
            for event in state.get('raw_contacts',[]):
                paths=event.get('paths',[])[:2]
                if contact.get('object_link') not in paths or contact.get('robot_link') not in paths:continue
                for sample in event.get('samples',[]):
                    if sample.get('position')==contact.get('point_world_m') and sample.get('impulse')==contact.get('impulse_world_ns') and sample.get('separation')==contact.get('separation_m'):found=True;break
                if found:break
            if not found:errors.append({'state_index':i,'reason':'derived_contact_not_supported_by_raw_actor_pair'})
            else:matched+=1
    return {'passed':not errors,'errors':errors,'verified_contacts':matched}


def decide(spec,trace,physics,media,framing):
    evaluation=evaluate(spec,trace);errors=[]
    if trace.get('model_variant')!='demonstration_teacher':errors.append('not_a_demonstration_teacher')
    if evaluation['sim_task_success'] is not True:errors.extend(evaluation['failure_reasons'])
    identity=audit_contact_identity(trace)
    if not identity['passed']:errors.append('contact_identity_inconsistent')
    if physics.get('sim_ready_status')!='sim_ready_only':errors.append('sim_ready_not_verified')
    if any(physics.get(k) is not True for k in ['collision_valid','physical_sanity','articulation_valid']):errors.append('physics_gate_not_passed')
    if physics.get('format_load',{}).get('isaac_stage_opened') is not True:errors.append('stage_load_not_passed')
    pe=physics.get('physics_evidence',{})
    for key in ['initial_penetration_status','gravity_settle_status','stability_status','contact_test_status']:
        if pe.get(key)!='passed':errors.append(key+'_not_passed')
    if pe.get('object_transform_writes_after_play')!=0:errors.append('transform_writeback_or_missing_record')
    body=physics.get('asset_inspection',{}).get('body_physics',[])
    dynamic=[b for b in body if b.get('motion_type')=='dynamic']
    if not dynamic:errors.append('dynamic_body_missing')
    for b in dynamic:
        mass=b.get('mass_kg');com=b.get('center_of_mass_values_m');inertia=b.get('diagonal_inertia_values_kg_m2')
        if not isinstance(mass,(int,float)) or not math.isfinite(mass) or mass<=0:errors.append('mass_invalid')
        if not isinstance(com,list) or len(com)!=3 or not all(math.isfinite(v) for v in com):errors.append('center_of_mass_invalid')
        if not isinstance(inertia,list) or len(inertia)!=3 or not all(math.isfinite(v) and v>0 for v in inertia):errors.append('inertia_invalid')
        elif max(inertia)>sum(inertia)-max(inertia)+1e-7:errors.append('inertia_triangle_inequality')
    if media.get('passed') is not True:errors.append('raw_video_or_clock_not_verified')
    if framing.get('passed') is not True:errors.append('robot_and_object_framing_not_verified')
    if physics.get('robot_task_success') is not None:errors.append('physical_robot_success_mislabel')
    return {'training_admitted':not errors,'failure_reasons':sorted(set(errors)),'full_task_evaluation':evaluation,'contact_identity':identity,
            'collector_success_boolean_used':False,'teacher_is_not_vla':True,'robot_real_world':'not_evaluated','robot_control_status':'simulated_robot_only','robot_task_success':None}
