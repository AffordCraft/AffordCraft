"""Bounded robot-reference integration for an explicitly registered teacher.

Unlike a waypoint repeatedly anchored at measured EE+4mm, this allows the
robot reference to build finite tracking error against contact resistance.
It never commands the object or changes the registered segment or task goal.
"""
import numpy as np
from .bounded_push_feedback import POLICY as PREDECESSOR

POLICY={**PREDECESSOR,'reference':'previous commanded waypoint, initialized at measured EE',
        'anti_windup':'clip reference to original preflighted push segment',
        'uses_sim_task_success_as_input':False}


def waypoint(ee, object_bounds, velocity, goal_region, segment_start, segment_end, previous_command=None):
    from .bounded_push_feedback import waypoint as measured_feedback
    feedback=measured_feedback(ee,object_bounds,velocity,goal_region,segment_start,segment_end)
    start=np.asarray(segment_start,dtype=float);end=np.asarray(segment_end,dtype=float)
    axis=end-start;axis[2]=0;length=float(np.linalg.norm(axis));axis/=length
    previous=np.asarray(ee if previous_command is None else previous_command,dtype=float)
    if previous.shape!=(3,) or not np.isfinite(previous).all():raise ValueError('invalid_previous_robot_reference')
    progress=float(np.clip((previous-start)@axis,0,length))
    new_progress=float(np.clip(progress+feedback['waypoint_delta_m'],0,length))
    return {**feedback,'robot_ee_waypoint_m':(start+new_progress*axis).tolist(),
        'segment_progress_m':new_progress,'previous_reference_progress_m':progress,
        'reference_update_m':new_progress-progress,'reference_policy':'bounded_command_integration_v0_1'}
