"""Per-horizon action-chunk normalization for Exp3 (append-only v0.164).

The predecessor v0.160 used one-radian normalization, and the exploratory
v0.163 used a single 0.06-rad scale even for future chunk targets, which
correctly rejected multi-tick targets as out of range.  This revision freezes
training-set-derived *uniform per-horizon* scales and decodes them identically
at runtime.  No non-uniform joint scaling is used; each horizon has one scalar
rad/tick envelope shared by all seven arm joints.
"""
import torch
from torch import nn
import exp3_oft_precision_adapter_v0_160 as _prev

ADAPTER_ID = "openvla_oft_franka_token_preserving_per_horizon_scale_v0_164"
CHUNK = _prev.CHUNK
EXECUTE_PREFIX = _prev.EXECUTE_PREFIX
FINGER_MAX = _prev.FINGER_MAX
# Frozen before this test run from the immutable training frame index. Values
# exceed every observed cumulative future target (including the 8-step max).
HORIZON_ARM_SCALE = (0.06, 0.17, 0.30, 0.40, 0.48, 0.55, 0.62, 0.68)
ARM_DELTA_SCALE = HORIZON_ARM_SCALE[0]
require_dataset = _prev.require_dataset
validate_precision_configuration = _prev.validate_precision_configuration


def encode_targets(actions, proprio):
    import numpy as np
    actions = np.asarray(actions, dtype=np.float32)
    q = np.asarray(proprio, dtype=np.float32)
    if actions.shape != (CHUNK, 9) or q.shape != (9,):
        raise ValueError("robot_action_training_dimensions")
    scales = np.asarray(HORIZON_ARM_SCALE, dtype=np.float32)[:, None]
    result = np.concatenate([
        (actions[:, :7] - q[None, :7]) / scales,
        actions[:, 7:] / FINGER_MAX * 2.0 - 1.0,
    ], axis=-1)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1.0001:
        raise ValueError("registered_action_normalization_out_of_range")
    return result


def make_action_head(hidden, config):
    width = int(config["precision_head"]["hidden_width"])

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())

        def forward(self, x):
            return x + self.net(x)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_language = nn.Sequential(nn.LayerNorm(hidden * 7), nn.Linear(hidden * 7, width), nn.GELU())
            self.precise_robot_state = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, width))
            self.residual = nn.Sequential(Residual(), Residual(), nn.LayerNorm(width))
            self.output = nn.Linear(width, 9)

        def forward(self, latent, proprio):
            if latent.shape[-1] != hidden * 7 or proprio.shape[-1] != 9:
                raise ValueError("token_preserving_head_shape")
            with torch.autocast(device_type=latent.device.type, enabled=False):
                x = self.visual_language(latent.float()) + self.precise_robot_state(proprio.float())[:, None, :]
                return torch.tanh(self.output(self.residual(x)))

    return Head()


def build(config, dataset_manifest, device):
    _prev.make_action_head = make_action_head
    _prev.ADAPTER_ID = ADAPTER_ID
    return _prev.build(config, dataset_manifest, device)

