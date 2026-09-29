"""Cache frozen OpenVLA-OFT features (v0.166 extraction: mean of the 56 action-token hidden states)
for every training query (stride 4) of the frozen scene dataset. One process per GPU shard; no
training here. Writes episode_XXXX.npz (features fp16, proprio, encoded targets, valid mask, indices)."""

import argparse, json, os, sys, time, hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--output", required=True)
p.add_argument("--gpu", type=int, required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument(
    "--reuse-features",
    default="",
    help="feature directory of an earlier run (same frames, possibly different dataset/scales): its per-episode feature arrays are reused for episodes whose frame sha256s match; targets are re-encoded with the scales of this dataset",
)
a = p.parse_args()
cfg = json.load(open(a.config))
ds = Path(a.dataset)
man = json.load(open(ds / "manifest.json"))
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, cfg["tools_dir"])
os.environ["HF_MODULES_CACHE"] = str(out / f"hf_rank_{a.shard}")
import numpy as np, torch
from PIL import Image
import exp3_oft_recurrent_adapter_v0_166 as adapter


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


SCALES = np.asarray(man["horizon_arm_scale_rad"], np.float32)
FINGER_MAX = 0.04
C = man["chunk"]
STRIDE = man["execute_prefix"]


def encode(actions, q):
    actions = np.asarray(actions, np.float32)
    q = np.asarray(q, np.float32)
    r = np.concatenate([(actions[:, :7] - q[None, :7]) / SCALES[:, None], actions[:, 7:] / FINGER_MAX * 2 - 1], -1)
    if not np.isfinite(r).all() or np.max(np.abs(r)) > 1.0001:
        raise ValueError("normalization_out_of_range:%.4f" % float(np.max(np.abs(r))))
    return r


NIMG = int(man.get("num_images", 1))
if int(cfg["feature_dim"]) != 4096 * NIMG:
    raise SystemExit("feature_dim_mismatch:%d vs %d images" % (int(cfg["feature_dim"]), NIMG))
torch.cuda.set_device(a.gpu)
device = torch.device("cuda", a.gpu)
t0 = time.time()
base, processor = adapter.build_features(cfg, device)
load = time.time() - t0
print(
    json.dumps(dict(model_load_s=round(load, 1), peak_gb=round(torch.cuda.max_memory_allocated(a.gpu) / 1e9, 2))),
    flush=True,
)
groups = {}
for line in (ds / "frame_index.jsonl").open():
    r = json.loads(line)
    groups.setdefault(r["episode_job"], []).append(r)
jobs = sorted(groups)
records = []
MSHA = sha(ds / "manifest.json")
resumed = 0
reused = 0
prior = {}
if a.reuse_features:
    pd = Path(a.reuse_features)
    for m in sorted(pd.glob("shard_*_manifest.json")):
        for r in json.load(open(m))["records"]:
            prior[r["episode_job"]] = r
    pfi = {}
    for line in (pd.parent / "dataset" / "frame_index.jsonl").open():
        r = json.loads(line)
        pfi.setdefault(r["episode_job"], {})[r["index"]] = (r["sha256"], r.get("sha256_wrist"))
    print(json.dumps(dict(reuse_dir=str(pd), prior_episodes=len(prior))), flush=True)


def reuse_prior(job, rows):
    """feature array of the same episode from the earlier run, if every queried frame (and wrist frame) has the same sha256."""
    r = prior.get(job)
    if not r:
        return None
    idx = list(range(0, len(rows), STRIDE))
    pf = pfi.get(job, {})
    for i in idx:
        p_ = pf.get(rows[i]["index"])
        if not p_ or p_[0] != rows[i]["sha256"] or p_[1] != rows[i].get("sha256_wrist"):
            return None
    try:
        with np.load(r["path"]) as x:
            f = x["features"]
            ix = x["indices"]
            if f.shape != (len(idx), 4096 * NIMG) or list(ix) != idx:
                return None
            return f.copy()
    except Exception:
        return None


def reusable(path, rows):
    """an existing episode file from an interrupted run of the same frozen dataset (manifest sha stored inside) is reused."""
    try:
        with np.load(path) as x:
            if "manifest_sha" in x.files:
                if str(x["manifest_sha"]) != MSHA:
                    return False
            elif os.path.getmtime(path) < os.path.getmtime(ds / "manifest.json"):
                return False  # legacy file (no sha inside): accept only if written after this dataset's manifest
            n = len(range(0, len(rows), STRIDE))
            ok = (
                x["features"].shape == (n, 4096 * NIMG)
                and x["proprio"].shape == (n, 9)
                and x["targets"].shape == (n, C, 9)
                and x["valid"].shape == (n, C)
                and x["indices"].shape == (n,)
            )
            return ok and bool(
                np.allclose(
                    x["proprio"][:3],
                    np.asarray([rows[i]["proprio"] for i in range(0, min(len(rows), 3 * STRIDE), STRIDE)], np.float32),
                )
            )
    except Exception:
        return False


for epi, job in enumerate(jobs):
    if epi % a.num_shards != a.shard:
        continue
    rows = sorted(groups[job], key=lambda x: x["index"])
    vec = []
    pro = []
    tgt = []
    msk = []
    idx = []
    tick = time.time()
    path = out / f"episode_{epi:04d}.npz"
    pf = reuse_prior(job, rows) if prior else None
    if pf is not None and not (path.exists() and reusable(path, rows)):
        for k, i in enumerate(range(0, len(rows), STRIDE)):
            row = rows[i]
            vec.append(pf[k])
            pro.append(row["proprio"])
            tgt.append(encode([rows[min(i + kk, len(rows) - 1)]["action"] for kk in range(C)], row["proprio"]))
            msk.append([i + kk < len(rows) for kk in range(C)])
            idx.append(i)
        tmp = out / f"episode_{epi:04d}.tmp.npz"
        np.savez_compressed(
            tmp,
            features=np.asarray(vec),
            proprio=np.asarray(pro, np.float32),
            targets=np.asarray(tgt, np.float32),
            valid=np.asarray(msk, np.float32),
            indices=np.asarray(idx),
            manifest_sha=np.array(MSHA),
        )
        os.replace(tmp, path)
        rec = dict(
            episode_job=job,
            scene=rows[0]["scene"],
            query_count=len(idx),
            path=str(path.resolve()),
            sha256=sha(path),
            runtime_s=round(time.time() - tick, 1),
            reused_features=True,
        )
        records.append(rec)
        reused += 1
        print(json.dumps(dict(shard=a.shard, episode=epi, scene=rec["scene"], reused_features=True)), flush=True)
        continue
    if path.exists() and reusable(path, rows):
        rec = dict(
            episode_job=job,
            scene=rows[0]["scene"],
            query_count=len(range(0, len(rows), STRIDE)),
            path=str(path.resolve()),
            sha256=sha(path),
            runtime_s=0.0,
            resumed=True,
        )
        records.append(rec)
        resumed += 1
        print(json.dumps(dict(shard=a.shard, episode=epi, scene=rec["scene"], resumed=True)), flush=True)
        continue
    for i in range(0, len(rows), STRIDE):
        row = rows[i]
        parts = []
        for key in ["path", "path_wrist"][:NIMG]:
            with Image.open(row[key]) as im:
                inputs = processor(adapter.prompt(row["instruction"]), im.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = adapter.features(base, **inputs)
            parts.append(f[0].cpu().numpy().astype(np.float16))
        vec.append(np.concatenate(parts))
        pro.append(row["proprio"])
        tgt.append(encode([rows[min(i + k, len(rows) - 1)]["action"] for k in range(C)], row["proprio"]))
        msk.append([i + k < len(rows) for k in range(C)])
        idx.append(i)
    tmp = out / f"episode_{epi:04d}.tmp.npz"
    np.savez_compressed(
        tmp,
        features=np.asarray(vec),
        proprio=np.asarray(pro, np.float32),
        targets=np.asarray(tgt, np.float32),
        valid=np.asarray(msk, np.float32),
        indices=np.asarray(idx),
        manifest_sha=np.array(MSHA),
    )
    os.replace(tmp, path)
    rec = dict(
        episode_job=job,
        scene=rows[0]["scene"],
        query_count=len(idx),
        path=str(path.resolve()),
        sha256=sha(path),
        runtime_s=round(time.time() - tick, 1),
    )
    records.append(rec)
    print(
        json.dumps(dict(shard=a.shard, episode=epi, scene=rec["scene"], queries=len(idx), s=rec["runtime_s"])),
        flush=True,
    )
(out / f"shard_{a.shard}_manifest.json").write_text(
    json.dumps(
        dict(
            records=records,
            num_images=NIMG,
            feature_dim=4096 * NIMG,
            resumed_episodes=resumed,
            reused_feature_episodes=reused,
            reuse_dir=a.reuse_features,
            model_load_s=load,
            peak_memory_bytes=torch.cuda.max_memory_allocated(a.gpu),
            gpu=torch.cuda.get_device_name(a.gpu),
            torch=torch.__version__,
            config_sha256=sha(a.config),
            dataset_manifest_sha256=sha(ds / "manifest.json"),
        ),
        indent=1,
    )
)
print("shard done", a.shard, len(records))
