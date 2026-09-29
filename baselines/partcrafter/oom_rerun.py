#!/usr/bin/env python
"""Amendment 002 for an external-baseline revision: re-execute, once, the cases whose recorded failure is a CUDA out-of-memory
caused by GPU co-tenancy. Never touches any other failed case. For each such case the original attempt directory is moved to
lane-<k>/interrupted/<source_id>-oom-<ts>/ (all files kept) so the lane runner (append-only, skips cases with result.json)
recomputes exactly that case on the exclusive GPU given by CUDA_VISIBLE_DEVICES. Writes amendment_002_oom_rerun.json (open 'x').
usage: oom_rerun.py --revision-root R --gpu N [--dry-run]"""
import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--revision-root", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
R = os.path.abspath(a.revision_root)
cfg = json.load(open(os.path.join(R, "configuration.json")))
is_t2 = cfg["method_id"].startswith("trellis2")
runner = "run_trellis2_lane.py" if is_t2 else "run_partcrafter_lane.py"
py = (
    os.environ.get("TRELLIS2_PYTHON", sys.executable) if is_t2 else os.environ.get("PARTCRAFTER_PYTHON", sys.executable)
)
oom = []
for p in sorted(glob.glob(os.path.join(R, "lane-*", "cases", "*", "result.json"))):
    r = json.load(open(p))
    fr = r.get("failure_reason") or ""
    if "out of memory" in fr.lower():
        oom.append((os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))), r["source_id"], fr[:200]))
print(f"{len(oom)} CUDA-OOM cases: {[s for _, s, _ in oom]}")
if a.dry_run or not oom:
    sys.exit(0)
ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
moved = []
for lane, sid, fr in oom:
    src = os.path.join(R, lane, "cases", sid)
    dst = os.path.join(R, lane, "interrupted", f"{sid}-oom-{ts}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    moved.append({"lane": lane, "source_id": sid, "original_failure_reason": fr, "original_attempt_kept_at": dst})
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(a.gpu),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PYTHONUNBUFFERED="1",
    PYTHONDONTWRITEBYTECODE="1",
    OPENCV_IO_ENABLE_OPENEXR="1",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    PYOPENGL_PLATFORM="egl",
)
if is_t2:
    env.update(
        HF_HOME=os.environ.get(
            "TRELLIS2_HF_HOME", os.path.join(os.environ.get("AFFORDCRAFT_MODELS", "models"), "hfcache-trellis2")
        ),
        ATTN_BACKEND="flash_attn",
        SPARSE_ATTN_BACKEND="flash_attn",
        SPARSE_CONV_BACKEND="flex_gemm",
    )
else:
    env.update(
        EVAL_API_BASE=os.environ["EVAL_API_BASE"],
        EVAL_API_KEY=os.environ["EVAL_API_KEY"],
        SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt",
        PARTCRAFTER_API_MODEL=cfg["part_suggest"]["model"],
    )
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    env.pop(k, None)
runs = []
for lane in sorted({m["lane"] for m in moved}):
    k = lane.split("-")[1]
    tag = f"oomrerun{a.gpu}l{k}"
    logp = os.path.join(R, lane, "lane.log")
    with open(logp, "a") as lf:
        lf.write(
            f"{_dt.datetime.now().astimezone().isoformat(timespec='seconds')} amendment 002: OOM rerun of {[m['source_id'] for m in moved if m['lane'] == lane]} on exclusive GPU {a.gpu}\n"
        )
        cmd = [py, os.path.join(R, "code", runner), "--revision-root", R, "--lane", k, "--tag", tag, "--render"]
        if not is_t2:
            cmd += ["--vlm-workers", "1"]
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=os.path.join(R, lane)).returncode
        rc2 = subprocess.run(
            [
                py,
                os.path.join(R, "code", "write_receipt.py"),
                "--revision-root",
                R,
                "--lane",
                k,
                "--tag",
                tag,
                "--gpu",
                str(a.gpu),
                "--exit-code",
                str(rc),
            ],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        ).returncode
    runs.append({"lane": lane, "tag": tag, "runner_exit": rc, "receipt_exit": rc2})
after = []
for m in moved:
    p = os.path.join(R, m["lane"], "cases", m["source_id"], "result.json")
    after.append({"source_id": m["source_id"], "rerun_result": json.load(open(p)) if os.path.exists(p) else None})
rec = {
    "amendment": 2,
    "revision": os.path.basename(R),
    "applied": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "exclusive_gpu": a.gpu,
    "policy": "one re-execution of cases whose only recorded failure was a CUDA out-of-memory produced while the GPU was shared with other lane processes "
    "(infrastructure, not the method); original attempt directories kept under interrupted/; no other failed case touched; no selection among outputs.",
    "moved": moved,
    "runs": runs,
    "rerun_outcome": [
        {
            "source_id": x["source_id"],
            "asset_emitted": (x["rerun_result"] or {}).get("asset_emitted"),
            "failure_reason": (x["rerun_result"] or {}).get("failure_reason"),
        }
        for x in after
    ],
}
with open(os.path.join(R, "amendment_002_oom_rerun.json"), "x") as f:
    json.dump(rec, f, indent=2)
print(json.dumps(rec["rerun_outcome"], indent=1))
