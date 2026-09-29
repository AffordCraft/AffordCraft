"""Learned feedback-aware recurrent head; robot observations/actions only."""
import numpy as np
import torch
from torch import nn
from exp3_oft_recurrent_adapter_v0_166 import build_features, features, prompt, require_dataset, HORIZON_ARM_SCALE, FINGER_MAX

ADAPTER_ID='openvla_oft_franka_feedback_denoising_v0_167'
CHUNK=8
EXECUTE_PREFIX=4


class RecurrentHead(nn.Module):
    def __init__(self, hidden=4096, width=512, proprio_mean=None, proprio_std=None):
        super().__init__()
        self.register_buffer('proprio_mean',torch.tensor(proprio_mean if proprio_mean is not None else [0.]*9,dtype=torch.float32))
        self.register_buffer('proprio_std',torch.tensor(proprio_std if proprio_std is not None else [1.]*9,dtype=torch.float32))
        self.vision_language=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,width),nn.GELU())
        self.proprio=nn.Sequential(nn.Linear(9,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.memory=nn.GRU(width+128,width,num_layers=2,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(width+128),nn.Linear(width+128,width),nn.GELU(),nn.Linear(width,CHUNK*9))

    def forward(self, features, proprio, memory=None):
        robot=self.proprio((proprio.float()-self.proprio_mean)/self.proprio_std)
        x=torch.cat([self.vision_language(features.float()),robot],-1)
        y,memory=self.memory(x,memory)
        return torch.tanh(self.head(torch.cat([y,robot],-1))).reshape(*y.shape[:2],CHUNK,9),memory


def augment_proprio_and_targets(proprio, encoded, valid, cfg, step):
    """Independent sensor/state noise; targets remain teacher desired positions.

    Augmented targets are explicitly projected through each registered uniform
    arm increment envelope. No original dataset record is written or relabeled.
    Teacher actions are supervision only, never an input to the neural network.
    """
    rng=np.random.default_rng(cfg['augmentation_seed']+step)
    q=np.asarray(proprio,np.float32);target=np.asarray(encoded,np.float32);mask=np.asarray(valid)>0
    noise=np.zeros_like(q);selected=rng.random(len(q))<cfg['proprio_augmentation_probability'];rho=cfg['proprio_noise_correlation']
    for t in range(q.shape[1]):
        previous=noise[:,t-1,:7] if t else np.zeros((len(q),7))
        noise[:,t,:7]=rho*previous+cfg['proprio_noise_std_rad']*np.sqrt(1-rho*rho)*rng.standard_normal((len(q),7))
    noise*=selected[:,None,None];noise*=mask.any(axis=2).astype(np.float32)[:,:,None]
    augmented=q+noise
    scale=np.asarray(HORIZON_ARM_SCALE,np.float32)[None,None,:,None]
    desired=q[:,:,None,:7]+target[:,:,:,:7]*scale
    delta=desired-augmented[:,:,None,:7]
    ratio=np.abs(delta).max(axis=-1,keepdims=True)/scale
    factor=np.maximum(1.,ratio)
    repaired=np.concatenate([delta/factor/scale,target[:,:,:,7:]],-1)
    projected=(factor[...,0]>1.000001)&mask
    return augmented,repaired,dict(augmented_sequences=int(selected.sum()),projected_targets=int(projected.sum()),valid_targets=int(mask.sum()),maximum_added_proprio_noise_rad=float(np.abs(noise[:,:,:7]).max()))
