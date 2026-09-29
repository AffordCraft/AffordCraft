#!/usr/bin/env python
"""lane_receipt.py -- lane-level execution_receipt.json (written once per lane run; a later re-run appends
execution_receipt_<n>.json). Usage: lane_receipt.py <lane_dir> <lane> <lanes>"""
import glob
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    GPU,
    METHOD_ID,
    PATCH_DIFF,
    REV,
    SUBSET,
    SUBSET_SHA,
    VLM_MODEL,
    capture,
    lock,
    read_json,
    rows_of,
    utc,
    write_json_once,
)


def med(vals):
    vals = [v for v in vals if v is not None]
    return dict(
        n=len(vals),
        median=(statistics.median(vals) if vals else None),
        min=(min(vals) if vals else None),
        max=(max(vals) if vals else None),
        sum=(sum(vals) if vals else None),
    )


def main():
    lane_dir, lane, lanes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rows = rows_of(lane_dir / "input_manifest.jsonl")
    ids = [r["source_id"] for r in rows]
    results = {
        sid: read_json(lane_dir / "cases" / sid / "result.json")
        for sid in ids
        if (lane_dir / "cases" / sid / "result.json").exists()
    }
    counts = dict(
        lane_rows=len(ids),
        attempted=sum(1 for sid in ids if (lane_dir / "cases" / sid).exists()),
        with_result=len(results),
        retrieval_ok=sum(1 for r in results.values() if r.get("selected_object_id")),
        link_ok=sum(1 for r in results.values() if r["stage_reached"] in ("joint_prediction", "export")),
        exported=sum(1 for r in results.values() if r["asset_emitted"]),
        exported_with_movable_joint=sum(
            1 for r in results.values() if r["asset_emitted"] and (r.get("joint_count") or 0) > 0
        ),
        failed=sum(1 for r in results.values() if not r["asset_emitted"]),
        pending=len(ids) - len(results),
        timeouts=sum(1 for r in results.values() if (r.get("failure_reason") or "").startswith("timeout")),
    )
    stats = dict(
        runtime_seconds=med([r.get("runtime_seconds") for r in results.values()]),
        vlm_calls=med([r.get("vlm_calls") for r in results.values()]),
        vlm_tokens=med([r.get("vlm_tokens") for r in results.values()]),
        vlm_prompt_tokens=med([r.get("vlm_prompt_tokens") for r in results.values()]),
        vlm_completion_tokens=med([r.get("vlm_completion_tokens") for r in results.values()]),
        vlm_reasoning_tokens=med([r.get("vlm_reasoning_tokens") for r in results.values()]),
        part_count=med([r.get("part_count") for r in results.values() if r["asset_emitted"]]),
        joint_count=med([r.get("joint_count") for r in results.values() if r["asset_emitted"]]),
    )
    stage_seconds = {}
    for k in ("retrieval_seconds", "link_seconds", "joint_seconds", "total_runtime_seconds", "vlm_latency_seconds"):
        stage_seconds[k] = med([read_json(p).get(k) for p in glob.glob(str(lane_dir / "cases" / "*" / "timing.json"))])
    receipt = dict(
        schema="affordcraft.external_baseline_execution.v1",
        method_id=METHOD_ID,
        vlm_model=VLM_MODEL,
        input_set="registered_subset_200_seed20260916",
        lane=lane,
        lanes=lanes,
        gpu=GPU,
        host=os.uname().nodename,
        gpu_name=capture(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i", str(GPU)]
        ).strip(),
        lane_manifest=lock(lane_dir / "input_manifest.jsonl"),
        registered_subset=dict(path=str(SUBSET), sha256=SUBSET_SHA),
        configuration=lock(REV / "configuration.json"),
        patch_diff=lock(PATCH_DIFF),
        counts=counts,
        stats=stats,
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
    print(path, counts)


if __name__ == "__main__":
    main()
