"""Corrected precision adapter v0.160.

The v0.156 branch is retained as an invalid implementation because it scaled
the output as well as the regression loss.  This module patches no historical
file: it reuses its dataset/checkpoint loader and replaces only the new action
head with a full-range tanh output.  Training-only RMS is used in the loss by
the v0.156 trainer; it is never applied to the action range.
"""
import torch
from torch import nn
import torch.nn.functional as F

import exp3_oft_precision_adapter_v0_156 as _base

ADAPTER_ID = 'openvla_oft_franka_token_preserving_precision_adapter_v0_160'
CHUNK = _base.CHUNK
EXECUTE_PREFIX = _base.EXECUTE_PREFIX
ARM_DELTA_SCALE = _base.ARM_DELTA_SCALE
FINGER_MAX = _base.FINGER_MAX
require_dataset = _base.require_dataset
validate_precision_configuration = _base.validate_precision_configuration
encode_targets = _base.encode_targets


class _Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())

    def forward(self, x):
        return x + self.net(x)


def make_action_head(hidden, config):
    scale = validate_precision_configuration(config)
    width = int(config['precision_head']['hidden_width'])

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_language = nn.Sequential(nn.LayerNorm(hidden * 7), nn.Linear(hidden * 7, width), nn.GELU())
            self.precise_robot_state = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, width))
            self.residual = nn.Sequential(_Residual(width), _Residual(width), nn.LayerNorm(width))
            self.output = nn.Linear(width, 9)
            # scale is deliberately not an output multiplier: normalized
            # actions retain the registered [-1, 1] range.
            self.register_buffer('training_arm_scale', torch.tensor(scale, dtype=torch.float32), persistent=False)

        def forward(self, latent, proprio):
            if latent.shape[-1] != hidden * 7 or proprio.shape[-1] != 9:
                raise ValueError('token_preserving_head_shape')
            with torch.autocast(device_type=latent.device.type, enabled=False):
                x = self.visual_language(latent.float()) + self.precise_robot_state(proprio.float())[:, None, :]
                z = self.output(self.residual(x))
                return torch.tanh(z)

    return Head()


def build(config, dataset_manifest, device):
    _base.make_action_head = make_action_head
    _base.ADAPTER_ID = ADAPTER_ID
    return _base.build(config, dataset_manifest, device)


def patch_loaded_module():
    _base.make_action_head = make_action_head
    _base.ADAPTER_ID = ADAPTER_ID
    return _base
