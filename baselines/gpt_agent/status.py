"""Status summary of a run root: counts, medians, in-flight cases. usage: python status.py <run_root>"""

from __future__ import annotations
import json, statistics, sys, time
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    results = []
    for p in sorted(root.glob("lane-*/cases/*/result.json")):
        try:
            results.append(json.loads(p.read_text()))
        except Exception:
            pass
    prog = {}
    if (root / "progress.json").exists():
        prog = json.loads((root / "progress.json").read_text())
    emitted = [r for r in results if r["asset_emitted"]]
    reasons = {}
    for r in results:
        k = r.get("failure_reason") or "none"
        k = k.split(":")[0] + (":" + k.split(":")[1] if k.startswith("budget_exhausted") else "")
        reasons[k] = reasons.get(k, 0) + 1

    def med(key, rows):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 1) if vals else None

    out = {
        "finished": len(results),
        "emitted": len(emitted),
        "reasons": reasons,
        "median_rounds": med("rounds", results),
        "median_completion_tokens": med("completion_tokens", results),
        "median_prompt_tokens": med("prompt_tokens", results),
        "median_runtime_s": med("runtime_seconds", results),
        "median_parts_emitted": med("part_count", emitted),
        "median_joints_emitted": med("joint_count", emitted),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "max_runtime_s": max([r["runtime_seconds"] for r in results], default=None),
        "progress_json": {k: prog.get(k) for k in ("t", "in_flight", "concurrency", "api_stats")},
        "interrupted": len(list((root / "interrupted").glob("*"))) if (root / "interrupted").exists() else 0,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
