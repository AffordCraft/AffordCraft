"""Fail-closed, simulator-neutral full-task evaluator over raw timed evidence.

No model, controller or collection success boolean participates in a decision.
The adapter must supply physics timestamps, world geometry and link-local
contact coordinates; omissions are evidence failures, not inferred defaults.
"""
from __future__ import annotations
import math
from typing import Any

SHORTCUTS=('fallback_used','teleport_used','post_play_transform_writeback',
           'direct_object_command','invisible_attachment_used')

def finite(values):
    return isinstance(values,(list,tuple)) and bool(values) and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values)

def inside(point,lo,hi,tolerance=0.):
    return len(point)==len(lo)==len(hi) and all(a-tolerance<=p<=b+tolerance for p,a,b in zip(point,lo,hi))

def in_region(state,spec):
    b=state['object_bounds_world_m'];r=spec['target_region_world_m'];t=spec['region_tolerance_m']
    return inside(b['min'][:2],r['min'],r['max'],t) and inside(b['max'][:2],r['min'],r['max'],t)

def operation_contact(contact,spec):
    roi=spec.get('operation_region')
    if not roi or contact.get('object_link')!=spec['target_link']:return False
    if contact.get('robot_link') not in spec['allowed_robot_contact_links']:return False
    p=contact.get('point_link_local_m');imp=contact.get('impulse_world_ns')
    return (finite(p) and len(p)==3 and finite(imp) and len(imp)==3
            and inside(p,roi['min'],roi['max'],spec['contact_roi_tolerance_m'])
            and math.sqrt(sum(x*x for x in imp))>=spec['minimum_contact_impulse_ns']
            and isinstance(contact.get('separation_m'),(int,float))
            and contact['separation_m']<=spec['maximum_contact_separation_m'])

def evaluate(spec:dict[str,Any],trace:dict[str,Any])->dict[str,Any]:
    """Pure independent recomputation; never consumes trace.success."""
    result={'task_id':spec['task_id'],'asset_id':spec['asset_id'],'sim_task_success':False,
            'status':'failed','failure_reasons':[],'phase_events':[],
            'robot_real_world':'not_evaluated','robot_control_status':'simulated_robot_only','robot_task_success':None}
    def fail(reason,status='failed'):
        result['failure_reasons'].append(reason);result['status']=status;return result
    if spec.get('admission_status')!='ready_for_pilot':return fail('task_preflight_blocked','blocked')
    if trace.get('asset_id')!=spec['asset_id'] or trace.get('candidate_id')!=spec['candidate_id']:return fail('asset_identity_mismatch')
    if trace.get('asset_sha256')!=spec['asset_sha256']:return fail('asset_hash_mismatch')
    if trace.get('model_variant') not in ('demonstration_teacher','untrained_vla','trained_vla'):return fail('unregistered_controller_identity')
    if any(trace.get(k) is not False for k in SHORTCUTS):return fail('shortcut_or_missing_instrumentation')
    states=trace.get('states',[])
    if len(states)<2:return fail('missing_raw_state_trace','insufficient_evidence')
    required=('timestamp_s','object_bounds_world_m','object_linear_velocity_m_s','object_angular_velocity_rad_s',
              'robot_proprio','robot_action','contacts','max_penetration_m','forbidden_collision','support_contact','joint_positions')
    for s in states:
        if any(k not in s for k in required):return fail('missing_raw_state_fields','insufficient_evidence')
        if not all(finite(s[k]) for k in ('object_linear_velocity_m_s','object_angular_velocity_rad_s','robot_proprio','robot_action')):return fail('nonfinite_action_or_state')
        if len(s['robot_action'])!=9 or len(s['robot_proprio'])!=9:return fail('action_or_proprio_dimension')
        if not inside(s['robot_action'],spec['action_limits']['min'],spec['action_limits']['max']):return fail('robot_action_out_of_range')
        b=s['object_bounds_world_m']
        if not all(finite(b.get(k)) and len(b[k])==3 for k in ('min','max')) or any(x>y for x,y in zip(b['min'],b['max'])):return fail('invalid_geometry_bounds')
        if not isinstance(s['timestamp_s'],(int,float)) or not math.isfinite(s['timestamp_s']):return fail('invalid_timestamp')
        if not isinstance(s['max_penetration_m'],(int,float)) or not math.isfinite(s['max_penetration_m']):return fail('missing_penetration_evidence')
        if s['forbidden_collision'] is not False:return fail('forbidden_collision')
        if s['max_penetration_m']>spec['max_penetration_m']:return fail('excessive_penetration')
        if b['min'][2]<spec['support_top_z_m']-spec['drop_tolerance_m']:return fail('object_dropped')
        for j,limits in spec['joint_limits'].items():
            q=s['joint_positions'].get(j)
            if not isinstance(q,(int,float)) or not math.isfinite(q):return fail('missing_joint_state','insufficient_evidence')
            if not limits[0]-spec['joint_limit_tolerance']<=q<=limits[1]+spec['joint_limit_tolerance']:return fail('joint_out_of_limits')
    times=[s['timestamp_s'] for s in states]
    if abs(times[0])>1e-6 or any(b<=a or b-a>spec['maximum_trace_gap_s'] for a,b in zip(times,times[1:])):return fail('trace_time_coverage_invalid')
    if times[-1]>spec['time_budget_s']+spec['maximum_trace_gap_s']:return fail('episode_budget_exceeded')
    task=spec['task_type'];joint=spec.get('target_joint')
    def joint_in(s,key):return spec[key][0]<=s['joint_positions'][joint]<=spec[key][1]
    first=states[0]
    if task in ('push_to_region','grasp_lift_place'):
        center=[(a+b)/2 for a,b in zip(first['object_bounds_world_m']['min'],first['object_bounds_world_m']['max'])]
        if not inside(center[:2],spec['initial_center_range_world_m']['min'],spec['initial_center_range_world_m']['max']):return fail('initial_state_out_of_range')
        if in_region(first,spec):return fail('initial_state_already_satisfies_goal')
    else:
        if not joint_in(first,'initial_joint_range'):return fail('initial_state_out_of_range')
        if task!='open_then_close' and joint_in(first,'target_joint_range'):return fail('initial_state_already_satisfies_goal')
    contact_seen=False;hold_start=None;open_done=False;lift_done=False;grasp_seen=False
    result['phase_events'].append({'phase':'initial_valid','timestamp_s':0.})
    for s in states:
        t=s['timestamp_s'];matched=[c for c in s['contacts'] if operation_contact(c,spec)]
        if matched and not contact_seen:
            contact_seen=True;result['phase_events'].append({'phase':'correct_operation_contact','timestamp_s':t})
        if task=='grasp_lift_place':
            touching={c['robot_link'] for c in matched}
            grasp_seen=grasp_seen or set(spec['grasp_finger_links']).issubset(touching)
            raised=grasp_seen and s['object_bounds_world_m']['min'][2]>=spec['support_top_z_m']+spec['lift_height_m']
            if not lift_done:
                hold_start=(hold_start if hold_start is not None else t) if raised else None
                if hold_start is not None and t-hold_start+1e-9>=spec['lift_hold_s']:
                    lift_done=True;hold_start=None;result['phase_events'].append({'phase':'grasp_and_lift_held','timestamp_s':t})
                continue
        goal= in_region(s,spec) if task in ('push_to_region','grasp_lift_place') else joint_in(s,'open_joint_range' if task=='open_then_close' and not open_done else 'target_joint_range')
        if task=='grasp_lift_place':goal=goal and min(s['robot_proprio'][-2:])>=spec['release_gripper_min_m'] and s['support_contact'] is True
        stable=(math.sqrt(sum(x*x for x in s['object_linear_velocity_m_s']))<=spec['max_hold_linear_speed_m_s'] and math.sqrt(sum(x*x for x in s['object_angular_velocity_rad_s']))<=spec['max_hold_angular_speed_rad_s'])
        if task in ('push_to_region','grasp_lift_place'):stable=stable and s['support_contact'] is True
        good=contact_seen and goal and stable
        if good and hold_start is None:
            hold_start=t;result['phase_events'].append({'phase':'goal_hold_started','timestamp_s':t})
        if not good:hold_start=None
        if task=='open_then_close' and not open_done and hold_start is not None and t-hold_start+1e-9>=spec['hold_duration_s']:
            open_done=True;hold_start=None;result['phase_events'].append({'phase':'open_stage_complete','timestamp_s':t})
    if not contact_seen:return fail('correct_operation_contact_missing')
    if task=='grasp_lift_place' and not lift_done:return fail('grasp_lift_stage_incomplete')
    if task=='open_then_close' and not open_done:return fail('open_stage_incomplete')
    if hold_start is None:return fail('final_goal_not_reached_or_not_stable')
    if times[-1]-hold_start+1e-9<spec['hold_duration_s']:return fail('final_hold_duration_short')
    result.update(status='passed',sim_task_success=True,hold_duration_s=times[-1]-hold_start)
    result['phase_events'].append({'phase':'full_task_complete','timestamp_s':times[-1]})
    return result

def training_admissible(spec,trace,media_verified):
    return (trace.get('model_variant')=='demonstration_teacher' and media_verified is True
            and evaluate(spec,trace)['sim_task_success'] is True)
