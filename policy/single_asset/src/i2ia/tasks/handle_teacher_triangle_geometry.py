"""Geometric handle-open teacher emits robot tool poses, never object commands."""
import numpy as np

def rotation_quaternion(m):
    a,b,c=m[0];d,e,f=m[1];g,h,i=m[2]
    k=np.array([[a-e-i,b+d,c+g,h-f],[b+d,e-a-i,f+h,c-g],[c+g,f+h,i-a-e,d-b],[h-f,c-g,d-b,a+e+i]])/3
    _,v=np.linalg.eigh(k);q=v[:,-1]
    if q[3]<0:q=-q
    return [float(q[3]),*q[:3].tolist()]

def handle_open_waypoints(points_world, body_center, pivot, axis, initial_q, target_q, approach_distance=.12, panel_points_world=None, handle_triangles=None):
    points=np.asarray(points_world,dtype=float);pivot=np.asarray(pivot);axis=np.asarray(axis,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():raise ValueError('handle_geometry_missing')
    if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:raise ValueError('joint_axis_invalid')
    axis=axis/np.linalg.norm(axis);center=points.mean(0);_,_,vectors=np.linalg.svd(points-center,full_matrices=False);bar=vectors[0]
    panel=np.asarray(panel_points_world,dtype=float)
    if panel.ndim!=2 or panel.shape[1]!=3 or len(panel)<4 or not np.isfinite(panel).all():raise ValueError("source_panel_geometry_missing")
    panel_center=panel.mean(0);_,singular,axes=np.linalg.svd(panel-panel_center,full_matrices=False)
    if singular[-1]/max(singular[-2],1e-12)>.2:raise ValueError("operation_panel_normal_uncertain")
    outward=axes[-1];signed=float((center-panel_center)@outward)
    if abs(signed)<.001:raise ValueError("handle_protrusion_direction_uncertain")
    if signed<0:outward=-outward
    outward-=bar*np.dot(outward,bar);outward/=np.linalg.norm(outward)
    from .contact_geometry import first_surface_intersection
    ray_start=center+outward*(float(np.linalg.norm(np.ptp(points,axis=0)))+approach_distance)
    if handle_triangles is not None and len(handle_triangles):
        hit=first_surface_intersection(points,handle_triangles,ray_start,-outward)
        contact=np.asarray(hit['point'])
        contact_method='source_triangle_intersection'
    else:
        # Some source-link correspondence manifests intentionally retain only
        # sampled handle vertices.  Use the measured protruding extreme as an
        # explicit point-cloud contact, never a box/proxy or an object command.
        # The method is recorded so downstream admission can require human
        # review when triangle evidence is unavailable.
        projections=(points-center)@outward
        if not np.isfinite(projections).all() or len(points)<4:
            raise ValueError('source_handle_point_cloud_invalid')
        contact=np.asarray(points[int(np.argmax(projections))])
        hit={'point':contact.tolist(),'method':'source_handle_point_cloud_extreme','triangle_evidence':False}
        contact_method='source_handle_point_cloud_extreme'
    z=-outward;y=np.cross(bar,z);y/=np.linalg.norm(y);x=np.cross(y,z);rotation=np.column_stack([x,y,z])
    def pose(phase,point,rot,finger):
        return {'phase':phase,'position_m':np.asarray(point).tolist(),'orientation_wxyz':rotation_quaternion(rot),'finger_position_m':finger}
    rows=[pose('robot_approach',contact+outward*approach_distance,rotation,.04),pose('robot_handle_contact',contact,rotation,.04),pose('robot_handle_grasp',contact,rotation,0.)]
    for i in range(1,5):
        x,y,z=axis;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);angle=(target_q-initial_q)*i/4
        r=np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)
        rows.append(pose(f'robot_joint_arc_{i}',pivot+r@(contact-pivot),r@rotation,0.))
    rows.append({**rows[-1],'phase':'robot_joint_hold'})
    return {'waypoints':rows,'source_surface_intersection':hit,'contact_surface_world_m':contact.tolist(),'handle_axis_world':bar.tolist(),'outward_world':outward.tolist(),'precomputed_from_authored_geometry':True,'contact_method':contact_method,'object_commands':False,'normal_source':'source-matched door panel smallest-variance axis, oriented toward protruding handle','panel_flatness_ratio':float(singular[-1]/singular[-2]),'handle_protrusion_m':abs(signed)}

