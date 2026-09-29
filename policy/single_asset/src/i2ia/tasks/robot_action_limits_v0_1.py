"""Numerical action boundary adapter; never changes object state or task goals.

The actor API stores position targets as float32. Decimal USD limits can lie
between adjacent float32 numbers. Clamp numerical round-off to the nearest
interior float32 value *before* sending the action, and reject material errors.
"""
import numpy as np

ROUND_OFF_BOUND_RAD = 1e-6

def safe_position_targets(values, lower, upper):
    q = np.asarray(values, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if q.shape != lo.shape or q.shape != hi.shape or q.ndim != 1 or q.size != 7:
        raise ValueError("robot_arm_action_shape_invalid")
    if not all(np.isfinite(v).all() for v in (q, lo, hi)) or np.any(lo >= hi):
        raise ValueError("robot_arm_action_nonfinite_or_limits_invalid")
    excess = np.maximum(np.maximum(lo - q, q - hi), 0.)
    if np.any(excess > ROUND_OFF_BOUND_RAD):
        raise ValueError("robot_action_materially_out_of_range_no_fallback")
    low32 = np.nextafter(lo.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    high32 = np.nextafter(hi.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    sent = np.clip(q, low32, high32)
    return sent, dict(raw_position_targets=q.tolist(), sent_position_targets=sent.tolist(),
                      maximum_roundoff_correction_rad=float(np.max(np.abs(sent - q))),
                      policy="nearest_interior_float32_before_robot_command", object_commands=False,
                      task_thresholds_changed=False)
