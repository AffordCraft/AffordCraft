from pathlib import Path
import argparse, json, sys, traceback, os

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    a = argparse.ArgumentParser()
    a.add_argument("--request", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--budget-seconds", type=float, default=500.0)
    args = a.parse_args()
    from affordcraft.build import construct_asset
    import affordcraft.build as build_module

    request = json.loads(Path(args.request).read_text())
    out = Path(args.output)
    if request.get("variant") == "without_decomposition_cascade":
        # One-factor variant: only level 0 of the cascade is available. Cache keys,
        # cleaning, snapping and audits are unchanged, so level-0 entries are shared with the full method.
        build_module.DECOMPOSITION_CASCADE = build_module.DECOMPOSITION_CASCADE[:1]
    try:
        result = construct_asset(
            request["project"],
            request["candidate"],
            request["grounding"],
            request["task_metadata"],
            out,
            args.cache,
            request.get("repair_mode"),
            budget_seconds=args.budget_seconds,
        )
    except Exception as exc:
        blocked = isinstance(exc, (ImportError, MemoryError)) or "immutable_source_changed" in str(exc)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "exported": False,
            "status": "infrastructure_blocked" if blocked else "construction_failed",
            "candidate_id": request["candidate"]["candidate_id"],
            "reason": repr(exc),
            "traceback": traceback.format_exc(),
        }
        audit = out / "geometry_audit.json"
        if audit.exists():
            try:
                result["diagnostics"] = json.loads(audit.read_text())
            except ValueError:
                pass
    (out / "build_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"exported": result["exported"], "reason": result.get("reason")}))


if __name__ == "__main__":
    main()
