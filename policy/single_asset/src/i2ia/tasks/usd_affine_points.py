"""Vectorized double-precision USD row-vector affine geometry readback."""
import numpy as np


def transform_points(points,matrix):
    p=np.asarray(points,dtype=np.float64);m=np.asarray(matrix,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or m.shape!=(4,4):raise ValueError('invalid_affine_geometry_shape')
    if not np.isfinite(p).all() or not np.isfinite(m).all():raise ValueError('nonfinite_affine_geometry')
    if not np.allclose(m[:,3],[0,0,0,1],atol=1e-12,rtol=0):raise ValueError('non_affine_usd_transform')
    return p@m[:3,:3]+m[3,:3]
