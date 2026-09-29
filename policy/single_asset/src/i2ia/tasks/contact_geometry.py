"""Read-only geometric contact planning utilities; never commands an object."""
from __future__ import annotations
import numpy as np

def first_surface_intersection(vertices,triangles,origin,direction):
    """Moller-Trumbore intersection, requiring a real triangle surface hit."""
    v=np.asarray(vertices,dtype=float);f=np.asarray(triangles,dtype=int);o=np.asarray(origin,dtype=float);d=np.asarray(direction,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(v).all() or not np.isfinite(o).all() or not np.isfinite(d).all():raise ValueError('invalid_contact_geometry')
    length=np.linalg.norm(d)
    if length<1e-12:raise ValueError('zero_ray_direction')
    d=d/length;t=v[f];e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);valid=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[valid]=1/det[valid];s=o-t[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);w=inv*(q@d);distance=inv*np.einsum('ij,ij->i',e2,q);valid &= (u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)&(distance>1e-8)
    if not valid.any():raise ValueError('no_actual_surface_on_registered_push_ray')
    indices=np.flatnonzero(valid);index=int(indices[np.argmin(distance[valid])]);point=o+d*distance[index]
    return {'point':point.tolist(),'triangle_index':index,'distance':float(distance[index]),'barycentric':[float(1-u[index]-w[index]),float(u[index]),float(w[index])],'method':'exact_triangle_ray_intersection_no_proxy'}
