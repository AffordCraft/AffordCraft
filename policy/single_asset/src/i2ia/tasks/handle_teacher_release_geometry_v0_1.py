"""Robot-only handle arc, release, retreat and hold teacher protocol."""
import numpy as np
from .handle_teacher_triangle_geometry import handle_open_waypoints as contact_arc


def handle_open_waypoints(points_world, body_center, pivot, axis, initial_q,
                          target_q, approach_distance=.12,
                          panel_points_world=None, handle_triangles=None):
    plan = contact_arc(points_world, body_center, pivot, axis, initial_q, target_q,
                       approach_distance, panel_points_world, handle_triangles)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = target_q - initial_q
    rotation = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
    outward = rotation @ np.asarray(plan["outward_world"])
    endpoint = plan["waypoints"][-2]
    retreat = np.asarray(endpoint["position_m"]) + .12*outward
    plan["waypoints"] = plan["waypoints"][:-1] + [
        {**endpoint, "phase": "robot_handle_release", "finger_position_m": .04},
        {**endpoint, "phase": "robot_handle_retreat", "position_m": retreat.tolist(), "finger_position_m": .04},
        {**endpoint, "phase": "robot_joint_hold", "position_m": retreat.tolist(), "finger_position_m": .04},
    ]
    plan["release_protocol"] = dict(version="source_normal_release_retreat_v0_1", retreat_distance_m=.12,
                                    task_threshold_changed=False, object_commands=False,
                                    model_evaluation_fallback=False)
    return plan
