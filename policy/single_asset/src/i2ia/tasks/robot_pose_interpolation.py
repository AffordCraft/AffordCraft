"""Shortest-arc robot orientation interpolation; no object control."""
import numpy as np

def slerp(start,end,progress):
    a=np.asarray(start,dtype=float).reshape(-1);b=np.asarray(end,dtype=float).reshape(-1)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all() or min(np.linalg.norm(a),np.linalg.norm(b))<1e-10 or not 0<=progress<=1:raise ValueError('invalid_quaternion_interpolation_contract')
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);dot=float(a@b)
    if dot<0:b=-b;dot=-dot
    if dot>.9995:q=a+progress*(b-a)
    else:
        angle=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-progress)*angle)*a+np.sin(progress*angle)*b)/np.sin(angle)
    return q/np.linalg.norm(q)
