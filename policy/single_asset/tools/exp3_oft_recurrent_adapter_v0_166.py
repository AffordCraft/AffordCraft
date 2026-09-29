"""Frozen OpenVLA features with learned observation-history robot head.

No task phase, time, object state, contact, or teacher action is an input. The
GRU memory is initialized to zero at each episode and updated only from the
same RGB/proprio/instruction observations available to both policy variants.
"""
import json, os
from pathlib import Path
import torch
from torch import nn
from exp3_oft_joint_adapter_v0_134 import require_dataset, namespace_source, sha
from exp3_oft_precision_adapter_v0_164 import encode_targets, HORIZON_ARM_SCALE, FINGER_MAX

ADAPTER_ID = "openvla_oft_franka_observation_memory_v0_166"
CHUNK = 8
EXECUTE_PREFIX = 4


class RecurrentHead(nn.Module):
    def __init__(self, hidden=4096, width=512):
        super().__init__()
        self.vision_language = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU())
        self.proprio = nn.Sequential(nn.Linear(9,128), nn.GELU(), nn.Linear(128,128), nn.GELU())
        self.memory = nn.GRU(width+128, width, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,width), nn.GELU(), nn.Linear(width,CHUNK*9))

    def forward(self, features, proprio, memory=None):
        x=torch.cat([self.vision_language(features.float()), self.proprio(proprio.float()/proprio.new_tensor([3.]*7+[.04,.04]))],-1)
        y,memory=self.memory(x,memory)
        return torch.tanh(self.head(y)).reshape(*y.shape[:2],CHUNK,9),memory


def build_features(config, device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    checkpoint=Path(config['base_checkpoint'])
    for row in json.loads(Path(config['checkpoint_inventory']).read_text())['files']:
        if sha(checkpoint/row['path'])!=row['sha256']:raise ValueError('base_checkpoint_changed:'+row['path'])
    for rel,expected in config['official_source_files_sha256'].items():
        if sha(Path(config['official_source'])/rel)!=expected:raise ValueError('official_code_changed:'+rel)
    namespace_source(config['official_source']);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    processor=AutoProcessor.from_pretrained(str(checkpoint),trust_remote_code=True,local_files_only=True)
    base,loading=AutoModelForVision2Seq.from_pretrained(str(checkpoint),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',trust_remote_code=True,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('base_load_not_exact')
    base.requires_grad_(False);base.vision_backbone.set_num_images_in_input(1);base.config.use_cache=False;base.language_model.config.use_cache=False;base.to(device).eval()
    return base,processor


@torch.inference_mode()
def features(base, input_ids, attention_mask, pixel_values):
    from prismatic.vla.constants import IGNORE_INDEX,ACTION_DIM,NUM_ACTIONS_CHUNK
    if (ACTION_DIM,NUM_ACTIONS_CHUNK)!=(7,8):raise ValueError('native_latent_shape_changed')
    if not torch.all(input_ids[:,-1]==29871):
        input_ids=torch.cat([input_ids,input_ids.new_full((1,1),29871)],1);attention_mask=torch.cat([attention_mask,attention_mask.new_ones((1,1))],1)
    labels=input_ids.clone().fill_(IGNORE_INDEX);prompt_tokens=input_ids.shape[-1]-1
    ids,mask=base._prepare_input_for_action_prediction(input_ids,attention_mask);labels=base._prepare_labels_for_action_prediction(labels,ids)
    embeddings=base.get_input_embeddings()(ids);actions_mask=base._process_action_masks(labels);hidden=embeddings.shape[-1]
    language=embeddings[~actions_mask].reshape(1,-1,hidden);vision=base._process_vision_features(pixel_values,language,False)
    embeddings=embeddings*(~actions_mask.unsqueeze(-1));fused,fused_mask=base._build_multimodal_attention(embeddings,vision,mask)
    out=base.language_model.model(input_ids=None,attention_mask=fused_mask,inputs_embeds=fused,use_cache=False,output_hidden_states=False,return_dict=True)
    first=vision.shape[1]+prompt_tokens
    return out.last_hidden_state[:,first:first+56].float().mean(1)


def prompt(instruction):return 'In: What action should the robot take to '+instruction.strip().lower()+'?\nOut:'
