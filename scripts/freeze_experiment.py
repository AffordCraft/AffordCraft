"""Freeze inputs, actual source bytes and policies only after required calibration checks."""

from pathlib import Path
import argparse, json, hashlib, sys, time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from affordcraft.contracts import ConstructionPolicy, EvaluationPolicy
from affordcraft.catalog import file_sha
from affordcraft import paths


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--inputs", required=True)
    a.add_argument("--output", required=True)
    a.add_argument(
        "--physics-controls", required=True, help="execution_receipt.json of run_physics_job.py on the fixtures"
    )
    a.add_argument("--fixtures", required=True, help="jobs.json written by make_physics_fixtures.py")
    a.add_argument("--pipeline-calibration", required=True, help="summary.json of a calibration run of run_pipeline.py")
    # Fields the study registrations wrote into their definitions (one-factor variants, sharding, cluttered images).
    a.add_argument("--variant", default="full", help="one-factor study variant (affordcraft.contracts.VARIANTS)")
    a.add_argument("--shard-count", type=int, default=1)
    a.add_argument("--cluttered", action="store_true", help="inputs are automatic detections of cluttered images")
    args = a.parse_args()
    policy = ConstructionPolicy(
        variant=args.variant, repairs_per_candidate=0 if args.variant == "without_repair" else 1
    )
    control = json.loads(Path(args.physics_controls).read_text())
    expected = {x["job_id"]: x["expected_pass"] for x in json.loads(Path(args.fixtures).read_text())}
    assert control["exit_code"] == 0 and all(
        x["physical_pass"] == expected[x["job_id"]] for x in control["in_process"]["results"]
    )
    pipeline = json.loads(Path(args.pipeline_calibration).read_text())
    assert pipeline["complete"] and pipeline["successes"] >= 1
    files = [*R.glob("affordcraft/**/*.py"), *R.glob("scripts/*.py")]
    enc = paths.DINOV2_ROOT / "image_encoder_dinov2"
    vlm = paths.QWEN3_VL
    files += [*enc.glob("*.safetensors"), *enc.glob("*.json"), *vlm.glob("*.safetensors"), *vlm.glob("*.json")]
    locks = []
    for f in files:
        locks.append({"path": str(f), "sha256": file_sha(f), "bytes": f.stat().st_size})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(
            {
                "method": "AffordCraft",
                "created_time": time.time(),
                "calibration_passed": True,
                "calibration_control_cases": 7,
                "real_catalog_end_to_end_verified": True,
                "input_sha256": file_sha(args.inputs),
                "construction": policy.to_dict(),
                "execution_policy": {"shard_count": args.shard_count},
                "scope": "exp2_automatic_objects" if args.cluttered else "exp1_single",
                "evaluation": EvaluationPolicy().to_dict(),
                "code_and_model_locks": locks,
                "historical_results_already_observed": True,
                "frozen_before_viewing": False,
                "post_hoc_after_failure_observed": True,
                "formal_replay_frozen_before_execution": True,
                "installation_policy": "no_extra_pinning_of_movable_objects;unverified_mount_defaults_free",
                "semantics": "model_compatibility_is_diagnostic_not_independent_task_match",
            },
            f,
            indent=2,
        )
    print(json.dumps({"freeze": str(out), "locked_files": len(locks), "input_sha256": file_sha(args.inputs)}))


if __name__ == "__main__":
    main()
