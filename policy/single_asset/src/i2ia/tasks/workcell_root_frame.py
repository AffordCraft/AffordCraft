"""Preserve authored fixed-root frame while applying a pre-episode workcell pose."""
import numpy as np

def normalize(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-10:raise ValueError('invalid_root_quaternion')
    return q/np.linalg.norm(q)

def multiply(a,b):
    a=normalize(a);b=normalize(b);w=a[0]*b[0]-a[1:]@b[1:];v=a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:]);return normalize([w,*v])

def compose(workcell,asset_root,local0,local1):
    q=normalize(local1);inverse=np.array([q[0],*-q[1:]])
    return multiply(multiply(multiply(workcell,asset_root),local0),inverse).tolist()
