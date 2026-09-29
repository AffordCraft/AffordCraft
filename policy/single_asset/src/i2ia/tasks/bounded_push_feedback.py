"""Explicit demonstration-only robot waypoint feedback; never object commands."""
import numpy as np

POLICY={'max_waypoint_step_m':.004,'position_gain':.20,'velocity_damping_seconds':.10,
        'scope':'demonstration_teacher_only','vla_fallback_allowed':False,
        'inputs':['object_bounds_world_m','object_linear_velocity_m_s','robot_ee_position','frozen_goal_region'],
        'output':'robot_ee_waypoint_on_registered_push_segment'}


def waypoint(ee, object_bounds, velocity, goal_region, segment_start, segment_end):
    ee=np.asarray(ee,dtype=float);v=np.asarray(velocity,dtype=float);start=np.asarray(segment_start,dtype=float);end=np.asarray(segment_end,dtype=float)
    lo=np.asarray(object_bounds['min'],dtype=float);hi=np.asarray(object_bounds['max'],dtype=float)
    goal=(np.asarray(goal_region['min'],dtype=float)+np.asarray(goal_region['max'],dtype=float))*.5
    if any(x.shape!=(3,) for x in (ee,v,start,end,lo,hi)) or goal.shape!=(2,) or not all(np.isfinite(x).all() for x in (ee,v,start,end,lo,hi,goal)):raise ValueError('invalid_feedback_state')
    axis=end-start;axis[2]=0;length=float(np.linalg.norm(axis))
    if length<=0:raise ValueError('empty_registered_push_segment')
    axis/=length;center=(lo+hi)*.5;error=float((goal-center[:2])@axis[:2]);speed=float(v@axis)
    delta=float(np.clip(POLICY['position_gain']*error-POLICY['velocity_damping_seconds']*speed,-POLICY['max_waypoint_step_m'],POLICY['max_waypoint_step_m']))
    progress=float(np.clip((ee-start)@axis+delta,0,length));target=start+progress*axis
    return {'robot_ee_waypoint_m':target.tolist(),'longitudinal_goal_error_m':error,'observed_object_velocity_m_s':speed,
            'waypoint_delta_m':delta,'segment_progress_m':progress,'direct_object_command':False,'teacher_is_not_vla':True}
