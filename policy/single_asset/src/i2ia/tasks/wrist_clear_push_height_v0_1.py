"""Source-mesh envelope clearance; robot pose only, never object modification."""
import math

def choose(object_min_z, object_max_z, initial_ee_z, robot_mesh_bounds, nominal, margin=.020):
    offsets=[row['min'][2]-initial_ee_z for row in robot_mesh_bounds
             if row['link'] not in ('panda_hand','panda_leftfinger','panda_rightfinger')]
    if not offsets or not all(math.isfinite(x) for x in offsets) or min(offsets)<=0:
        raise ValueError('forbidden_wrist_vertical_envelope_uncertain')
    required=max(nominal,object_max_z-min(offsets)+margin)
    if not object_min_z+.015<=required<=object_max_z-.015:
        raise ValueError('no_source_side_contact_with_wrist_clearance')
    return dict(contact_ray_height_m=required,nominal_contact_height_m=nominal,
                source_forbidden_link_min_offset_z_m=min(offsets),clearance_m=margin,
                policy='source_forbidden_wrist_envelope_v0_1',object_geometry_modified=False)
