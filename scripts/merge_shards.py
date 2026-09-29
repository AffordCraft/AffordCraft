#!/usr/bin/env python3
"""Merge the shard runs of one frozen definition into a single run directory.

Formal runs are sharded (`run_pipeline.py --shard-index k --shard-count n`, input i goes to shard i mod n), and a shard
that was interrupted by an infrastructure fault is continued with `--resume-from`, whose new run keeps the earlier
terminal records by reference (inherited_terminal_references.json). This script collects every terminal record once and
writes `per_input.jsonl` and `summary.json` in the layout that `export_final_results.py` and `evaluate_scenes.py` read.

It refuses to merge unless every given run is a complete formal run, all runs share one frozen definition and shard
count, every shard index is covered by a complete run, no input appears twice, and the inputs equal the manifest.
Give only the final (complete) attempt of each shard.

Example:
    python scripts/merge_shards.py --inputs runs/inputs/single_inputs.json --runs runs/exp1/shard-* --output runs/exp1/merged
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="input manifest the shards were run on")
    ap.add_argument("--runs", nargs="+", required=True, help="output folders of the final attempt of every shard")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.inputs).read_text())
    cases = raw["inputs"] if isinstance(raw, dict) else raw
    expected = [c["input_id"] for c in cases]
    records, shards = {}, []
    for folder in map(Path, args.runs):
        summary = json.loads((folder / "summary.json").read_text())
        process = json.loads((folder / "process.json").read_text())
        if summary.get("scope") != "formal" or summary.get("complete") is not True:
            raise SystemExit(f"not a complete formal run: {folder}")
        rows = [ref["record"] for ref in json.loads((folder / "inherited_terminal_references.json").read_text())]
        rows += [json.loads(line) for line in (folder / "per_input.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != summary["completed"]:
            raise SystemExit(f"record count differs from the run summary: {folder}")
        for r in rows:
            if r["input_id"] in records:
                raise SystemExit("input recorded twice: " + r["input_id"])
            records[r["input_id"]] = r
        shards.append(
            {
                "run": folder.name,
                "shard_index": process["shard_index"],
                "shard_count": process["shard_count"],
                "freeze_sha256": process["freeze_sha256"],
                "summary_sha256": sha(folder / "summary.json"),
                "records": len(rows),
            }
        )
    if len({(s["shard_count"], s["freeze_sha256"]) for s in shards}) != 1:
        raise SystemExit("runs come from different frozen definitions or shard counts")
    if sorted(s["shard_index"] for s in shards) != list(range(shards[0]["shard_count"])):
        raise SystemExit("every shard index must be covered exactly once")
    if set(records) != set(expected):
        raise SystemExit("merged inputs differ from the manifest")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "per_input.jsonl").open("x") as stream:
        for input_id in expected:
            stream.write(json.dumps(records[input_id], allow_nan=False) + "\n")
    passes = sum(records[i]["physical_pass"] is True for i in expected)
    summary = {
        "scope": "formal",
        "complete": True,
        "expected": len(expected),
        "completed": len(expected),
        "successes": passes,
        "success_rate": passes / len(expected),
        "input_sha256": sha(args.inputs),
        "merged_from": sorted(shards, key=lambda s: s["shard_index"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("expected", "successes", "success_rate")}))


if __name__ == "__main__":
    main()
