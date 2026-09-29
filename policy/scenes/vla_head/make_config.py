"""Write the frozen training configuration for the scene VLA, verifying the official
OpenVLA-OFT source copy against the v0.167 file hashes and the checkpoint inventory.

REF is the registered single-asset v0.167 training configuration (shipped under policy/single_asset/configs); the
head code of this folder imports the adapter modules of policy/single_asset/tools (TOOLS). The official openvla-oft
checkout (OPENVLA_OFT_HOME), the LIBERO-Spatial checkpoint (OPENVLA_OFT_CHECKPOINT) and the checkpoint inventory
(OPENVLA_OFT_CHECKPOINT_INVENTORY: JSON {"files": [{"path", "sha256"}, ...]}) are read from the environment.
Usage: make_config.py OUT_JSON [MAX_STEPS]"""

import json, hashlib, os, sys
from pathlib import Path

_POLICY = Path(__file__).resolve().parents[2]
REF = str(_POLICY / "single_asset" / "configs" / "training_configuration.json")
SRC = os.environ.get("OPENVLA_OFT_HOME", "openvla-oft")
CKPT = os.environ.get("OPENVLA_OFT_CHECKPOINT", "libero_spatial")
INV = os.environ.get("OPENVLA_OFT_CHECKPOINT_INVENTORY", "checkpoint_inventory.json")
TOOLS = str(_POLICY / "single_asset" / "tools")
out = Path(sys.argv[1])
ref = json.load(open(REF))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


local = {}
mismatch = []
for rel, exp in ref["official_source_files_sha256"].items():
    p = Path(SRC) / rel
    if not p.exists():
        mismatch.append((rel, "missing"))
        continue
    h = sha(p)
    local[rel] = h
    if h != exp:
        mismatch.append((rel, "differs"))
cfg = dict(
    method_id="affordcraft_scenes_vla_v0_1",
    parent_method_id=ref["method_id"],
    base_checkpoint=CKPT,
    checkpoint_inventory=INV,
    official_source=SRC,
    official_commit=ref["official_commit"],
    official_source_files_sha256=local,
    tools_dir=TOOLS,
    feature_dim=4096,
    memory_width=512,
    batch_size=16,
    max_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 24000,
    learning_rate=3e-4,
    weight_decay=1e-4,
    warmup_steps=250,
    save_every=4000,
    seed=138000001,
    adapter_initialization_seed=138000002,
    augmentation_seed=167031,
    proprio_augmentation_probability=0.75,
    proprio_noise_correlation=0.8,
    proprio_noise_std_rad=0.005,
    motion_loss_gain=4.0,
    motion_loss_scale_rad=0.01,
    temporal_consistency_lambda=0.25,
    query_overlap_lambda=0.1,
    prompt_template=ref["prompt_template"],
    model_inputs=ref["model_inputs"],
    forbidden_model_inputs=ref["forbidden_model_inputs"],
    action_contract=ref["action_contract"],
    robot_real_world="not_evaluated",
    robot_control_status="simulated_robot_only",
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=1, sort_keys=True))
print(json.dumps(dict(files=len(local), mismatch=mismatch[:10], n_mismatch=len(mismatch), written=str(out))))
