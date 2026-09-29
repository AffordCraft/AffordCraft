"""Source-defined linear handle motion. Outputs robot poses only."""
import numpy as np
from .handle_teacher_triangle_geometry import handle_open_waypoints as surface_contact


def handle_open_waypoints(points_world,body_center,pivot,axis,initial_q,target_q,
                          approach_distance=.12,panel_points_world=None,handle_triangles=None):
    direction=np.asarray(axis,dtype=float)
    if not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1.)>1e-5:
        raise ValueError('invalid_prismatic_axis')
    plan=surface_contact(points_world,body_center,pivot,direction,0.,0.,approach_distance,
                         panel_points_world,handle_triangles)
    rows=plan['waypoints'][:3];grasp=rows[-1]
    for i in range(1,5):
        rows.append({**grasp,'phase':f'robot_joint_arc_{i}',
                     'position_m':(np.asarray(grasp['position_m'])+direction*(target_q-initial_q)*i/4).tolist()})
    end=rows[-1];retreat=np.asarray(end['position_m'])+approach_distance*np.asarray(plan['outward_world'])
    rows += [{**end,'phase':'robot_handle_release','finger_position_m':.04},
             {**end,'phase':'robot_handle_retreat','finger_position_m':.04,'position_m':retreat.tolist()},
             {**end,'phase':'robot_joint_hold','finger_position_m':.04,'position_m':retreat.tolist()}]
    plan.update(waypoints=rows,joint_motion_type='prismatic',joint_state_units='m',
                release_protocol={'object_commands':False,'translation_m':target_q-initial_q,
                                  'task_threshold_changed':False,'retreat_distance_m':approach_distance})
    return plan
