#!/usr/bin/env python
"""Lane-level execution_receipt.json (written once per lane run; a later re-run appends execution_receipt_<n>.json)."""
import os, sys, json, glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    lane_dir, lane, lanes, gpu = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rows = rows_of(lane_dir / "input_manifest.jsonl")
    ids = [r["source_id"] for r in rows]
    results = {}
    for sid in ids:
        rp = lane_dir / "cases" / sid / "result.json"
        if rp.exists():
            results[sid] = read_json(rp)
    vlm_ok = sum(1 for sid in ids if (lane_dir / "cases" / sid / "vlm_result.json").exists())
    counts = dict(
        lane_rows=len(ids),
        attempted=sum(1 for sid in ids if (lane_dir / "cases" / sid).exists()),
        with_result=len(results),
        vlm_ok=vlm_ok,
        decoder_ok=sum(1 for r in results.values() if r["stage_reached"] in ("decoder", "split", "export")),
        split_ok=sum(1 for r in results.values() if r["stage_reached"] in ("split", "export")),
        exported=sum(1 for r in results.values() if r["asset_emitted"]),
        mujoco_load_ok=sum(1 for r in results.values() if r.get("mjcf_mujoco_load_ok")),
        failed=sum(1 for r in results.values() if not r["asset_emitted"]),
        pending=len(ids) - len(results),
    )
    stage_seconds = {}
    for k in ("vlm_seconds", "decoder_seconds", "split_seconds", "export_seconds", "total_runtime_seconds"):
        vals = sorted(
            float(read_json(p)[k])
            for p in glob.glob(str(lane_dir / "cases" / "*" / "timing.json"))
            if read_json(p).get(k) is not None
        )
        stage_seconds[k] = dict(
            n=len(vals),
            median=(vals[len(vals) // 2] if vals else None),
            min=(vals[0] if vals else None),
            max=(vals[-1] if vals else None),
        )
    receipt = dict(
        schema="affordcraft.external_baseline_execution.v1",
        method_id=METHOD_ID,
        input_set=INPUT_SET,
        lane=lane,
        lanes=lanes,
        gpu=gpu,
        host=os.uname().nodename,
        gpu_name=capture(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i", str(gpu)]
        ).strip(),
        lane_manifest=lock(lane_dir / "input_manifest.jsonl"),
        registered_subset=dict(path=str(SUBSET), sha256=sha256(SUBSET)),
        configuration=lock(REV / "configuration.json"),
        counts=counts,
        stage_seconds=stage_seconds,
        failures={sid: r["failure_reason"] for sid, r in results.items() if not r["asset_emitted"]},
        interrupted=sorted(os.listdir(lane_dir / "interrupted")) if (lane_dir / "interrupted").is_dir() else [],
        completed_utc=utc(),
        no_result_selection=True,
        no_paper_write=True,
    )
    n = 0
    path = lane_dir / "execution_receipt.json"
    while path.exists():
        n += 1
        path = lane_dir / ("execution_receipt_%d.json" % n)
    write_json_once(path, receipt)
    print(json.dumps(dict(path=str(path), counts=counts), indent=1))


if __name__ == "__main__":
    main()
