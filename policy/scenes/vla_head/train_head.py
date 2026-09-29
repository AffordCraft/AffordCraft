"""Train the feedback-aware recurrent robot head (v0.167 recipe) on cached frozen OpenVLA-OFT features
of the success-only scene dataset. Scene-balanced sampling, proprio-noise augmentation with target
projection, motion-weighted L1+0.1MSE, temporal and query-overlap consistency. Saves initial and
final adapters and a checkpoint manifest (no held-out checkpoint selection)."""

import argparse, json, math, random, sys, time, hashlib
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--features", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--output", required=True)
p.add_argument("--gpu", type=int, default=0)
a = p.parse_args()
cfg = json.load(open(a.config))
sys.path.insert(0, cfg["tools_dir"])
import exp3_oft_feedback_adapter_v0_167 as adapter


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


man = json.load(open(Path(a.dataset) / "manifest.json"))
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
SCALES = tuple(man["horizon_arm_scale_rad"])
adapter.HORIZON_ARM_SCALE = SCALES
import exp3_oft_precision_adapter_v0_164 as p164

p164.HORIZON_ARM_SCALE = SCALES
records = []
for m in sorted(Path(a.features).glob("shard_*_manifest.json")):
    records += json.load(open(m))["records"]
if not records:
    raise SystemExit("no cached features")
torch.cuda.set_device(a.gpu)
torch.manual_seed(cfg["adapter_initialization_seed"])
rng = random.Random(cfg["seed"])
torch.set_num_threads(4)
start = time.time()
model = adapter.RecurrentHead(cfg["feature_dim"], cfg["memory_width"], man["proprio_mean"], man["proprio_std"]).cuda()
initial = out / "adapter_initial.safetensors"
save_file({k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(initial))
opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
groups = {}
loaded = {}
for r in records:
    if sha(r["path"]) != r["sha256"]:
        raise ValueError("cached_feature_changed")
    with np.load(r["path"]) as x:
        loaded[r["episode_job"]] = {k: x[k].copy() for k in ["features", "proprio", "targets", "valid"]}
    groups.setdefault(r["scene"], []).append(r["episode_job"])
scenes = sorted(groups)
print(json.dumps(dict(scenes={s: len(groups[s]) for s in scenes}, total=len(loaded))), flush=True)
weights = torch.tensor([5.0] * 7 + [1.0, 1.0], device="cuda")
scales = torch.tensor(SCALES, device="cuda")[None, None, :, None]
peak = 0
acfg = dict(cfg, **{"augmentation_seed": cfg["augmentation_seed"]})
for step in range(cfg["max_steps"]):
    tick = time.time()
    jobs = [rng.choice(groups[scenes[(step * cfg["batch_size"] + j) % len(scenes)]]) for j in range(cfg["batch_size"])]
    n = max(len(loaded[j]["features"]) for j in jobs)
    b = len(jobs)
    feat = np.zeros((b, n, cfg["feature_dim"]), np.float32)
    q = np.zeros((b, n, 9), np.float32)
    target = np.zeros((b, n, 8, 9), np.float32)
    mask = np.zeros((b, n, 8), np.float32)
    for j, job in enumerate(jobs):
        d = loaded[job]
        t = len(d["features"])
        feat[j, :t] = d["features"]
        q[j, :t] = d["proprio"]
        target[j, :t] = d["targets"]
        mask[j, :t] = d["valid"]
    augmented, targets, aug = adapter.augment_proprio_and_targets(q, target, mask, acfg, step)
    motion = np.abs(target[:, :, 0, :7] * SCALES[0]).max(-1)
    motion_weight = 1.0 + cfg["motion_loss_gain"] * np.minimum(motion / cfg["motion_loss_scale_rad"], 1.0)
    ft = torch.from_numpy(feat).cuda()
    qt = torch.from_numpy(augmented).cuda()
    tt = torch.from_numpy(targets).cuda()
    valid = torch.from_numpy(mask[:, :, :, None]).cuda()
    mw = torch.from_numpy(motion_weight[:, :, None, None]).cuda()
    weighted = valid * mw
    opt.zero_grad(set_to_none=True)
    pred, _ = model(ft, qt)
    error = pred - tt
    loss = ((error.abs() + 0.1 * error.square()) * weights * weighted).sum() / (weighted.sum() * weights.sum())
    pv = weighted[:, :, 1:] * valid[:, :, :-1]
    de = (pred[:, :, 1:] - pred[:, :, :-1]) - (tt[:, :, 1:] - tt[:, :, :-1])
    temporal = ((de.abs() + 0.1 * de.square()) * weights * pv).sum() / (pv.sum() * weights.sum()).clamp_min(1.0)
    absolute = qt[:, :, :7][:, :, None, :] + pred[:, :, :, :7] * scales
    desired = qt[:, :, :7][:, :, None, :] + tt[:, :, :, :7] * scales
    ov = valid[:, :-1, 4:] * valid[:, 1:, :4]
    difference = (absolute[:, :-1, 4:] - absolute[:, 1:, :4]) - (desired[:, :-1, 4:] - desired[:, 1:, :4])
    overlap = (difference.abs() * ov).sum() / (ov.sum() * 7).clamp_min(1.0)
    total = loss + cfg["temporal_consistency_lambda"] * temporal + cfg["query_overlap_lambda"] * overlap
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite_training_loss")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    factor = min(1.0, (step + 1) / cfg["warmup_steps"]) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / cfg["max_steps"]))
    )
    for g in opt.param_groups:
        g["lr"] = cfg["learning_rate"] * factor
    opt.step()
    torch.cuda.synchronize()
    peak = max(peak, torch.cuda.max_memory_allocated())
    arm_mae = (error[:, :, :, :7].abs() * scales * valid).sum() / (valid.sum() * 7)
    finger_mae = (error[:, :, :, 7:].abs() * 0.02 * valid).sum() / (valid.sum() * 2)
    log = dict(
        step=step + 1,
        loss=float(total),
        action_loss=float(loss),
        temporal_loss=float(temporal),
        overlap_loss_rad=float(overlap),
        arm_delta_mae_rad=float(arm_mae),
        finger_target_mae_m=float(finger_mae),
        grad_norm=float(norm),
        lr=opt.param_groups[0]["lr"],
        s=round(time.time() - tick, 3),
        augmentation=aug,
    )
    with (out / "training_log.jsonl").open("a") as f:
        f.write(json.dumps(log) + "\n")
    if (step + 1) % 50 == 0 or step == 0:
        print(json.dumps({k: v for k, v in log.items() if k != "augmentation"}), flush=True)
    if (step + 1) % cfg["save_every"] == 0:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(out / f"adapter_step_{step+1:06d}.safetensors"),
        )
final = out / "adapter_final.safetensors"
save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(final))
adapter.RecurrentHead(cfg["feature_dim"], cfg["memory_width"]).load_state_dict(load_file(str(final)), strict=True)
if sha(initial) == sha(final):
    raise RuntimeError("checkpoint_not_trained")
(out / "checkpoint_manifest.json").write_text(
    json.dumps(
        dict(
            base_checkpoint=cfg["base_checkpoint"],
            checkpoint_inventory_sha256=sha(cfg["checkpoint_inventory"]),
            initial_adapter=str(initial.resolve()),
            initial_adapter_sha256=sha(initial),
            final_adapter=str(final.resolve()),
            final_adapter_sha256=sha(final),
            completed_steps=cfg["max_steps"],
            horizon_arm_scale_rad=list(SCALES),
            proprio_mean=man["proprio_mean"],
            proprio_std=man["proprio_std"],
            dataset_manifest_sha256=sha(Path(a.dataset) / "manifest.json"),
            config_sha256=sha(a.config),
            training_code_sha256=sha(__file__),
            episodes=len(loaded),
            per_scene={s: len(groups[s]) for s in scenes},
            runtime_s=round(time.time() - start, 1),
            peak_memory_bytes=peak,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            heldout_used=False,
            fixed_final_checkpoint=True,
        ),
        indent=1,
    )
)
print("training done", round(time.time() - start, 1), "s")
