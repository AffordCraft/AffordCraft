#!/usr/bin/env python
"""Write lane-<k>/execution_receipt.json (append-only: numbered if one already exists) for an external-baseline lane."""
import argparse
import datetime as _dt
import glob
import json
import os
import socket
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--revision-root", required=True)
ap.add_argument("--lane", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--exit-code", type=int, required=True)
a = ap.parse_args()
lane_dir = os.path.join(a.revision_root, f"lane-{a.lane}")
rows = [json.loads(l) for l in open(os.path.join(lane_dir, "input_manifest.jsonl")) if l.strip()]
results = []
for r in rows:
    p = os.path.join(lane_dir, "cases", r["source_id"], "result.json")
    if os.path.exists(p):
        results.append(json.load(open(p)))
timings = []
for r in results:
    p = os.path.join(lane_dir, "cases", r["source_id"], "timing.json")
    if os.path.exists(p):
        timings.append(json.load(open(p)))


def med(key):
    v = sorted(t[key] for t in timings if isinstance(t.get(key), (int, float)))
    return v[len(v) // 2] if v else None


try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
except Exception as e:
    smi = f"unavailable: {e}"
receipt = {
    "revision": os.path.basename(a.revision_root),
    "lane": a.lane,
    "tag": a.tag,
    "host": socket.gethostname(),
    "cuda_visible_devices": a.gpu,
    "nvidia_smi": smi,
    "python": sys.executable,
    "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "runner_exit_code": a.exit_code,
    "lane_rows": len(rows),
    "results_written": len(results),
    "asset_emitted": sum(1 for r in results if r.get("asset_emitted")),
    "failed": sum(1 for r in results if r.get("failure_reason")),
    "failure_reasons": {r["source_id"]: r["failure_reason"] for r in results if r.get("failure_reason")},
    "interrupted_dirs": sorted(os.path.basename(p) for p in glob.glob(os.path.join(lane_dir, "interrupted", "*"))),
    "median_seconds": {
        k.replace("_seconds", ""): med(k) for k in sorted({k for t in timings for k in t if k.endswith("_seconds")})
    },
    "max_peak_gpu_mem_allocated_bytes": max([t.get("peak_gpu_mem_allocated_bytes", 0) for t in timings] or [0]),
    "complete": len(results) == len(rows),
}
out = os.path.join(lane_dir, "execution_receipt.json")
if os.path.exists(out):
    n = len(glob.glob(os.path.join(lane_dir, "execution_receipt*.json")))
    out = os.path.join(lane_dir, f"execution_receipt_{n}.json")
with open(out, "x") as f:
    json.dump(receipt, f, indent=2)
print(
    f"[receipt] {out}: results {len(results)}/{len(rows)} emitted {receipt['asset_emitted']} failed {receipt['failed']} complete={receipt['complete']}"
)
