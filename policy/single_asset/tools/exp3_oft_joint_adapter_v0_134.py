"""Explicit adapted OpenVLA-OFT baseline for 9D Franka robot actions.

NOT a native LIBERO 7D action reinterpretation. The original vision/language
checkpoint is retained; a new supervised robot joint head and proprio encoder,
and zero-output LoRA on the last eight language layers, are registered. Both
paired variants start from exactly the same saved adapter initialization.
"""
import hashlib,json,math,os,sys,types
from pathlib import Path
import numpy as np

ADAPTER_ID='openvla_oft_franka_joint_chunk_adapter_v0_134'
CHUNK=8
EXECUTE_PREFIX=4
ARM_DELTA_SCALE=1.0
FINGER_MAX=.04

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def require_dataset(path):
    p=Path(path);manifest=json.loads(p.read_text());root=p.parent
    if manifest.get('training_allowed') is not True or manifest.get('training_episode_count')!=1000 or manifest.get('heldout_episode_count')!=200:raise ValueError('independent_1000_success_dataset_gate_locked')
    for name,digest in manifest['dataset_files_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('dataset_changed:'+name)
    quality=json.loads((root/'dataset_quality_report.json').read_text())
    if not quality['training_allowed'] or len(quality['per_asset_accepted_count'])!=10 or any(n!=100 for n in quality['per_asset_accepted_count'].values()):raise ValueError('ten_by_100_independent_coverage_missing')
    return manifest

def namespace_source(source):
    source=Path(source)
    # The upstream aggregating __init__ imports unrelated RLDS/TensorFlow
    # trainers. Only unchanged functions used by the checkpoint are loaded.
    for name,rel in [('prismatic','prismatic'),('prismatic.training','prismatic/training'),('prismatic.vla','prismatic/vla')]:
        if name in sys.modules:raise ValueError('ambiguous_existing_prismatic_namespace')
        m=types.ModuleType(name);m.__path__=[str(source/rel)];m.__package__=name;sys.modules[name]=m

def robot_target(normalized_chunk,chunk_proprio,current_proprio,lo,hi,chunk_index):
    a=np.asarray(normalized_chunk,dtype=np.float64);q0=np.asarray(chunk_proprio,dtype=np.float64);q=np.asarray(current_proprio,dtype=np.float64)
    if a.shape!=(CHUNK,9) or q0.shape!=(9,) or q.shape!=(9,) or not np.isfinite(a).all() or not np.isfinite(q).all():raise ValueError('invalid_vla_robot_action_contract')
    if not 0<=chunk_index<EXECUTE_PREFIX:raise ValueError('action_chunk_prefix_out_of_range')
    raw=a[chunk_index];target=np.r_[q0[:7]+np.clip(raw[:7],-1,1)*ARM_DELTA_SCALE,(np.clip(raw[7:],-1,1)+1)*.5*FINGER_MAX]
    bounded=np.clip(target,np.asarray(lo),np.asarray(hi));delta=bounded[:7]-q[:7];maximum=float(np.max(np.abs(delta)));factor=min(1.,.06/maximum) if maximum else 1.;bounded[:7]=q[:7]+factor*delta
    bounded=np.clip(bounded,np.asarray(lo),np.asarray(hi))
    return bounded.tolist(),dict(adapter_id=ADAPTER_ID,raw_prediction=raw.tolist(),unlimited_target=target.tolist(),maximum_arm_reference_velocity_rad_s=.6,control_hz=10,normalization='future arm target relative to chunk-start measured proprio / 1.0 rad; finger absolute 0..0.04 m',safety_clamping_applied=not np.allclose(target,bounded),fallback_used=False,object_commands_emitted=False)

def encode_targets(actions,proprio):
    actions=np.asarray(actions,dtype=np.float32);q=np.asarray(proprio,dtype=np.float32)
    if actions.shape!=(CHUNK,9) or q.shape!=(9,):raise ValueError('robot_action_training_dimensions')
    result=np.concatenate([(actions[:,:7]-q[None,:7])/ARM_DELTA_SCALE,actions[:,7:]/FINGER_MAX*2.-1.],axis=-1)
    if not np.isfinite(result).all() or np.max(np.abs(result))>1.0001:raise ValueError('registered_action_normalization_out_of_range')
    return result

def build(config,dataset_manifest,device):
    require_dataset(dataset_manifest)  # Before checkpoint/model construction.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForVision2Seq,AutoProcessor
    checkpoint=Path(config['base_checkpoint']);inventory=json.loads(Path(config['checkpoint_inventory']).read_text())
    for row in inventory['files']:
        if sha(checkpoint/row['path'])!=row['sha256']:raise ValueError('base_checkpoint_hash_changed:'+row['path'])
    for rel,expected in config['official_source_files_sha256'].items():
        if sha(Path(config['official_source'])/rel)!=expected:raise ValueError('official_source_changed:'+rel)
    namespace_source(config['official_source'])
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    processor=AutoProcessor.from_pretrained(str(checkpoint),trust_remote_code=True,local_files_only=True)
    base,loading=AutoModelForVision2Seq.from_pretrained(str(checkpoint),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',trust_remote_code=True,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('base_checkpoint_not_exactly_loaded:'+str(loading))
    base.requires_grad_(False);base.vision_backbone.set_num_images_in_input(1);base.config.use_cache=False;base.language_model.config.use_cache=False
    hidden=base.language_model.config.hidden_size
    torch.manual_seed(config['adapter_initialization_seed'])
    class LoRALinear(nn.Module):
        def __init__(self,original):
            super().__init__();self.original=original;self.rank=config['lora_rank'];self.scale=config['lora_alpha']/self.rank
            self.A=nn.Parameter(torch.empty(self.rank,original.in_features,dtype=torch.float32));self.B=nn.Parameter(torch.zeros(original.out_features,self.rank,dtype=torch.float32));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        @property
        def weight(self):return self.original.weight
        def forward(self,x):return self.original(x)+F.linear(F.linear(x,self.A),self.B)*self.scale
    layers=base.language_model.model.layers
    for i in config['lora_layers']:
        for name in ['q_proj','v_proj']:
            attn=layers[i].self_attn;setattr(attn,name,LoRALinear(getattr(attn,name)))
    class Policy(nn.Module):
        def __init__(self):
            super().__init__();self.base=base
            self.robot_proprio=nn.Sequential(nn.Linear(9,hidden),nn.GELU(),nn.Linear(hidden,hidden))
            self.joint_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,512),nn.GELU(),nn.Linear(512,9),nn.Tanh())
            # A deterministic nonzero ordinary neural initialization is saved
            # before learning. It is not a scripted or zero-action controller.
        def forward(self,input_ids,attention_mask,pixel_values,proprio):
            from prismatic.vla.constants import IGNORE_INDEX,ACTION_DIM,NUM_ACTIONS_CHUNK
            if (ACTION_DIM,NUM_ACTIONS_CHUNK)!=(7,8):raise ValueError('upstream_native_latent_shape_changed')
            if input_ids.shape[0]!=1:raise ValueError('registered_microbatch_one_required')
            if not torch.all(input_ids[:,-1]==29871):
                input_ids=torch.cat([input_ids,input_ids.new_full((1,1),29871)],dim=1);attention_mask=torch.cat([attention_mask,attention_mask.new_ones((1,1))],dim=1)
            labels=input_ids.clone().fill_(IGNORE_INDEX);prompt_tokens=input_ids.shape[-1]-1
            ids,mask=base._prepare_input_for_action_prediction(input_ids,attention_mask);labels=base._prepare_labels_for_action_prediction(labels,ids)
            embeddings=base.get_input_embeddings()(ids);actions_mask=base._process_action_masks(labels)
            language=embeddings[~actions_mask].reshape(1,-1,hidden)
            with torch.no_grad():vision=base._process_vision_features(pixel_values,language,False)
            scales=proprio.new_tensor([3.]*7+[.04,.04]);normalized_proprio=proprio/scales
            vision=base._process_proprio_features(vision,normalized_proprio,self.robot_proprio)
            embeddings=embeddings*(~actions_mask.unsqueeze(-1));fused,fused_mask=base._build_multimodal_attention(embeddings,vision,mask)
            output=base.language_model.model(input_ids=None,attention_mask=fused_mask,inputs_embeds=fused,use_cache=False,output_hidden_states=False,return_dict=True)
            first=vision.shape[1]+prompt_tokens;latent=output.last_hidden_state[:,first:first+56,:].reshape(1,8,7,hidden).mean(2)
            return self.joint_head(latent)
        def adapter_state(self):
            train_names={n for n,p in self.named_parameters() if p.requires_grad}
            return {n:t.detach().cpu().contiguous() for n,t in self.state_dict().items() if n in train_names}
        def load_adapter(self,path):
            from safetensors.torch import load_file
            expected=set(self.adapter_state());value=load_file(str(path))
            if set(value)!=expected:raise ValueError('adapter_checkpoint_keys_mismatch')
            self.load_state_dict(value,strict=False)
    model=Policy().to(device)
    base.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,processor
