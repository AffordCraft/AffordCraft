"""Contact-point feasibility from retained real tool geometry, no task changes."""
import math


def support_clear_contact_height(object_min_z, object_max_z, support_z, initial_ee_z, robot_geometry, clearance_m=.005):
    values=[object_min_z,object_max_z,support_z,initial_ee_z,clearance_m]
    if not all(math.isfinite(v) for v in values) or clearance_m<=0 or object_max_z<=object_min_z:
        raise ValueError('invalid_contact_height_geometry')
    links={'panda_hand','panda_leftfinger','panda_rightfinger'}
    found={r['link']:r for r in robot_geometry if r['link'] in links}
    if set(found)!=links:raise ValueError('tool_collision_envelope_missing')
    lowest=min(r['min'][2] for r in found.values())
    if not math.isfinite(lowest):raise ValueError('nonfinite_tool_envelope')
    depth=max(0.,initial_ee_z-lowest)
    nominal=object_min_z+.35*(object_max_z-object_min_z)
    height=max(nominal,support_z+depth+clearance_m)
    if height>=object_max_z-1e-4:raise ValueError('no_surface_contact_height_with_tool_clearance')
    return {'contact_ray_height_m':height,'tool_lowest_point_below_ee_m':depth,
            'support_clearance_m':clearance_m,'source':'actual initial robot mesh envelope, same registered downward orientation',
            'task_goal_modified':False,'collision_verification_still_required':True}
