"""Calibration or complete fixed-input run. Never combines old and new rates."""

from pathlib import Path
import argparse, json, os, sys, time, hashlib, traceback

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--inputs", required=True)
    a.add_argument("--index", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--calibration", action="store_true")
    a.add_argument("--freeze")
    a.add_argument("--shard-index", type=int, default=0)
    a.add_argument("--shard-count", type=int, default=1)
    a.add_argument("--resume-from", action="append", default=[])
    a.add_argument("--geometry-cache")
    a.add_argument("--scene-mode", action="store_true")
    a.add_argument(
        "--variant", default=None, help="calibration only; formal runs take the variant from the frozen definition"
    )
    args = a.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from affordcraft.backend import RuntimeBackend, PhysicsServices, SceneRuntimeBackend
    from affordcraft.search import BudgetedConstruction
    from affordcraft.contracts import ConstructionPolicy
    from affordcraft.catalog import file_sha
    from affordcraft.execution import completion, atomic_status, save_new, health_probe
    from affordcraft import paths

    raw = json.loads(Path(args.inputs).read_text())
    cases = raw["inputs"] if isinstance(raw, dict) else raw
    total_expected = len(cases)
    policy = ConstructionPolicy()
    if args.limit:
        if not args.calibration:
            raise RuntimeError("Formal limits cannot shrink the denominator")
        cases = cases[: args.limit]
    if not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("Invalid deterministic shard")
    if not args.calibration:
        if not args.freeze:
            raise RuntimeError("Formal execution requires a frozen definition")
        lock = json.loads(Path(args.freeze).read_text())
        if not lock.get("calibration_passed") or file_sha(args.inputs) != lock["input_sha256"]:
            raise RuntimeError("Invalid formal freeze")
        for ref in lock.get("code_and_model_locks", []):
            if file_sha(ref["path"]) != ref["sha256"]:
                raise RuntimeError("Frozen code/model source changed: " + ref["path"])
        expected_shards = lock.get("execution_policy", {}).get("shard_count", 1)
        if args.shard_count != expected_shards:
            raise RuntimeError("Shard count differs from frozen definition")
        if args.scene_mode != (lock.get("scope") == "exp2_automatic_objects"):
            raise RuntimeError("Input scope differs from frozen definition")
        # The frozen definition, not the CLI, decides the study variant and construction budget.
        policy = ConstructionPolicy.from_definition(lock.get("construction"))
    elif args.variant:
        policy = ConstructionPolicy(
            variant=args.variant, repairs_per_candidate=0 if args.variant == "without_repair" else 1
        )
    if policy.variant == "encoder_replacement" and "clip" not in Path(args.index).name:
        raise RuntimeError("encoder_replacement requires a CLIP-built index")
    if policy.variant != "encoder_replacement" and "clip" in Path(args.index).name:
        raise RuntimeError("CLIP index is only valid for encoder_replacement")
    cases = [c for i, c in enumerate(cases) if i % args.shard_count == args.shard_index]
    inherited = []
    seen = set()
    for previous in args.resume_from:
        prev = Path(previous)
        process = json.loads((prev / "process.json").read_text())
        if (
            process.get("freeze_sha256") != (file_sha(args.freeze) if args.freeze else None)
            or process.get("shard_index") != args.shard_index
            or process.get("shard_count") != args.shard_count
        ):
            raise RuntimeError("Incompatible resume definition")
        prior_summary = json.loads((prev / "summary.json").read_text())
        prior_outcomes = prior_summary.get("processes", {})
        if set(prior_outcomes) != {"construction", "final"} or not all(
            x.get("exit_code") == 0 and x.get("in_process_completed") is True for x in prior_outcomes.values()
        ):
            raise RuntimeError("Cannot resume scientific results with unresolved process exits")
        case_ids = {c["input_id"] for c in cases}
        for line in (prev / "per_input.jsonl").read_text().splitlines():
            r = json.loads(line)
            if type(r.get("physical_pass")) is bool:
                if r["input_id"] not in case_ids or r["input_id"] in seen:
                    raise RuntimeError("Resume duplicate or changed input")
                case_file = prev / "cases" / r["input_id"] / "result.json"
                if json.loads(case_file.read_text()) != r:
                    raise RuntimeError("Resume record mismatch")
                seen.add(r["input_id"])
                inherited.append({"record": r, "path": str(case_file), "sha256": file_sha(case_file)})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "process.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "gpu": args.gpu,
                "scope": "calibration" if args.calibration else "formal",
                "input_sha256": file_sha(args.inputs),
                "expected_inputs": len(cases),
                "global_denominator": total_expected,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "freeze_sha256": file_sha(args.freeze) if args.freeze else None,
                "construction_budget": policy.candidates,
                "build_threads": int(os.environ.get("AFFORDCRAFT_BUILD_THREADS", "2")),
                "variant": policy.variant,
                "independent_final_process": True,
            },
            indent=2,
        )
    )
    save_new(out / "inherited_terminal_references.json", inherited)
    services = None
    records = [x["record"] for x in inherited]
    t = time.perf_counter()
    error = None
    try:
        probe = health_probe(R, paths.RUNTIME_PYTHON, out / "coacd_start_health")
        if not probe["passed"]:
            raise RuntimeError("coacd_start_health_failed")
        services = PhysicsServices(R, out / "services", args.gpu)
        model_start = time.perf_counter()
        backend_type = SceneRuntimeBackend if args.scene_mode else RuntimeBackend
        backend = backend_type(
            paths.PROJECT_ROOT, args.index, out / "runtime", services, R, variant=policy.variant
        )
        method = BudgetedConstruction(backend, policy)
        if args.scene_mode:
            backend.detection_root = Path(args.inputs).parent
        if args.geometry_cache:
            backend.geometry_cache = Path(args.geometry_cache)
        save_new(out / "model_load_timing.json", {"measured_runtime_seconds": time.perf_counter() - model_start})
        with (out / "per_input.jsonl").open("x") as stream:
            for i, c in enumerate(cases):
                if c["input_id"] in seen:
                    continue
                result = method.run(c)
                result["input_fingerprint"] = file_sha(c["image"]["path"])
                if args.scene_mode:
                    result.update(
                        scene_id=c["scene_id"],
                        relation_evaluation="not_in_scope",
                        scene_relation_success=None,
                        reference_object_denominator=237,
                    )
                records.append(result)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                per_case = out / "cases" / c["input_id"]
                per_case.mkdir(parents=True, exist_ok=False)
                (per_case / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
                atomic_status(
                    out / "progress.json",
                    {
                        "completed": len(records),
                        "expected": len(cases),
                        "global_denominator": total_expected,
                        "physical_passes": sum(x["physical_pass"] is True for x in records),
                        "infrastructure_blocked": sum(x["physical_pass"] is None for x in records),
                        "elapsed_seconds": time.perf_counter() - t,
                    },
                )
                print(
                    json.dumps(
                        {
                            "input": c["input_id"],
                            "status": result["status"],
                            "considered": result["considered_candidates"],
                            "completed": len(records),
                            "expected": len(cases),
                        }
                    ),
                    flush=True,
                )
                if result["status"] == "infrastructure_blocked":
                    break
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        save_new(out / "failure.json", error)
    finally:
        outcomes = services.close() if services else {}
        complete = completion(records, len(cases), outcomes) and error is None
        summary = {
            "scope": "calibration" if args.calibration else "formal",
            "complete": complete,
            "expected": len(cases),
            "completed": len(records),
            "successes": sum(x["physical_pass"] is True for x in records),
            "success_rate": (
                sum(x["physical_pass"] is True for x in records) / len(cases)
                if complete and not args.calibration
                else None
            ),
            "processes": outcomes,
            "seconds": time.perf_counter() - t,
            "no_historical_result_merge": True,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if complete:
            save_new(out / "completion.json", {"complete": True, "summary_sha256": file_sha(out / "summary.json")})
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
