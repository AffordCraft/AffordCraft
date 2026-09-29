"""Explicit robot-only time parameterization of an IK joint target."""
import numpy as np


def limit_target(current, requested, dt_s=.1, maximum_velocity_rad_s=.6):
    current=np.asarray(current,dtype=float);requested=np.asarray(requested,dtype=float)
    if current.shape!=(7,) or requested.shape!=(7,) or not np.isfinite(np.r_[current,requested]).all():
        raise ValueError("invalid_robot_joint_target")
    if not 0.<dt_s<=.2 or not 0.<maximum_velocity_rad_s<=2.:
        raise ValueError("invalid_registered_robot_rate")
    delta=requested-current;maximum=float(np.max(np.abs(delta)))
    factor=min(1.,maximum_velocity_rad_s*dt_s/maximum) if maximum else 1.
    sent=current+factor*delta
    return sent,dict(requested_target=requested.tolist(),measured_robot_joints=current.tolist(),
                    sent_target=sent.tolist(),interpolation_factor=factor,dt_s=dt_s,
                    maximum_joint_velocity_rad_s=maximum_velocity_rad_s,object_actuation=False)
