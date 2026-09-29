#!/usr/bin/env python3
"""Join immutable audit and raw rows, recheck evidence, and freeze robot actions.

Historical audit v0.2 renamed success fields to historical_*; dataset v0.1
mistook their omission for a failed episode. This version joins by exact
attempt path, keeps both originals, and never manufactures a success flag.
Recorded images/proprioception are post-action: image[t] predicts action[t+1],
not the action that produced image[t]. Gripper labels are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shlex
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOT = {"robot_real_world": "not_evaluated", "robot_control_status": "simulated_robot_only", "robot_task_success": None}
SOURCES = {
    "runs/remote-demo/interaction-first-openimages-formal-2000-quality-v3-semantic-manufactured/openimages_quality_2000_manifest.json": "4415b07224f14f084bf5f80d94a5e3d50568ea7a88d37756965c4cffdaa00d45",
    "runs/remote-demo/interaction-first-exp2-sim-ready-v0-8-coco-50-clean-first-covered/source_manifest.json": "9fcbca38307bff23e3fb5978094bf445f5bf020df77bf119b9e265a8ea0317d8",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def emit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def emit_rows(path, values):
    with Path(path).open("x") as f:
        for value in values:
            f.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def finite(values):
    return isinstance(values, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)


def recover_row(audit, raw_row):
    """Missing enriched fields are not false; raw contradictions still fail."""
    if audit.get("audit_decision") != "accepted" or audit.get("accepted_for_training") is not True:
        raise ValueError("historical_audit_rejected")
    for field in ("attempt_directory", "asset_id", "candidate_id"):
        if str(audit.get(field)) != str(raw_row.get(field)):
            raise ValueError("audit_raw_identity_mismatch:" + field)
    if audit.get("historical_sim_task_success") is not True or raw_row.get("sim_task_success") is not True:
        raise ValueError("sim_task_success_not_true")
    return {**raw_row, "prior_audit_sha256": digest(audit), "original_collection_row_sha256": digest(raw_row)}


def check_actions(trace):
    actions, states = trace.get("actions", []), trace.get("states", [])
    failures = []
    if not actions or len(actions) != len(states):
        return ["action_state_count_mismatch"]
    for a, s in zip(actions, states):
        if not finite(a.get("action")) or len(a["action"]) != 7:
            failures.append("arm_action_dimension_or_finiteness")
        g = a.get("gripper_action")
        if not finite(g) or len(g) != 2 or any(x < 0 or x > 0.04 for x in g):
            failures.append("gripper_action_invalid")
        p = s.get("robot_proprioception", {})
        if not finite(p.get("joint_position")) or len(p["joint_position"]) != 9 or not finite(p.get("joint_velocity")) or len(p["joint_velocity"]) != 9:
            failures.append("proprio_dimension_or_finiteness")
        if s.get("action") != a.get("action") or s.get("gripper_action") != g or s.get("timestamp_s") != a.get("timestamp_s"):
            failures.append("trace_alignment_mismatch")
        if s.get("fallback_used") is not False or s.get("replay_used") is not False or s.get("object_transform_writes_after_play") != 0:
            failures.append("trace_shortcut_or_missing_declaration")
        if not isinstance(s.get("contact_evidence"), dict):
            failures.append("contact_trace_missing")
    if not any(any(abs(x) > 1e-8 for x in a.get("action", [])) for a in actions):
        failures.append("placeholder_actions")
    return sorted(set(failures))


def make_windows(trace, observations, row, frame_hashes):
    # No future observation, object state, outcome, or controller phase is input.
    for i in range(0, len(trace["actions"]) - 8, 8):
        yield {
            "sample_id": f"{row['asset_id']}__{row['seed']}__{i:05d}",
            "asset_id": row["asset_id"], "task_id": row["task_id"],
            "instruction": f"perform the registered {row['task_id']} task on {row['category']}",
            "observation_path": str(observations[i]), "observation_sha256": frame_hashes[i]["sha256"],
            "proprio": trace["states"][i]["robot_proprioception"]["joint_position"],
            "action_chunk": [a["action"] for a in trace["actions"][i + 1:i + 9]],
            "gripper_chunk": [a["gripper_action"] for a in trace["actions"][i + 1:i + 9]],
            "observation_index": i, "first_target_action_index": i + 1,
            "trajectory_path": row["attempt_directory"],
            "trajectory_sha256": row["strict_evidence"]["state_trace_sha256"],
        }


def coverage(accepted, cases):
    split, counts = {}, {}
    for case in cases:
        aid = case["asset_id"]
        selected = sorted((r for r in accepted if r["asset_id"] == aid), key=lambda r: (r["seed"], r["attempt_directory"]))
        for i, row in enumerate(selected):
            split[row["attempt_directory"]] = "train" if i < 100 else "heldout_reference" if i < 120 else "excess_not_used"
        counts[aid] = {"category": case["category"], "candidate_id": case["candidate_id"], "asset_role": case["asset_role"], "accepted_count": len(selected), "train_count": min(100, len(selected)), "heldout_count": min(20, max(0, len(selected) - 100)), "target_met": len(selected) >= 120}
    return split, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--collection-root", type=Path, action="append", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("strict_v01_readonly", ROOT / "tools/freeze_exp3_franka_real_contact_dataset_v0_1.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    case_path = args.registration_root / "collection_cases.jsonl"
    cases = list(rows(case_path)); by_id = {c["asset_id"]: c for c in cases}
    if len(cases) != 10 or len(by_id) != 10 or Counter(c["asset_role"] for c in cases) != {"rigid": 5, "articulated": 5}:
        raise ValueError("registration_must_have_5_rigid_5_articulated")
    source_hashes = {path: sha(ROOT / path) for path in SOURCES}
    if source_hashes != SOURCES:
        raise ValueError("frozen_source_manifest_changed")
    audit_path = args.audit_root / "accepted_trajectory_manifest.jsonl"
    source_files = [case_path, audit_path, args.audit_root / "rejected_trajectory_manifest.jsonl"]
    source_files += [r / "per_attempt.jsonl" for r in args.collection_root]
    file_locks = {str(p.resolve()): sha(p) for p in source_files}
    emit(out / "configuration.json", {"registration_root": str(args.registration_root.resolve()), "history_read_only": True, "prior_audit": str(audit_path.resolve()), "train_per_asset": 100, "heldout_reference_per_asset": 20, "split_rule": "ascending immutable seed then attempt path; first 100 train, next 20 heldout reference", "new_heldout_evaluation_required": True, "action_contract": "7 Franka joint position targets radians + 2 gripper position targets metres", "frame_alignment": "post-action RGB/proprio at t predict actions t+1..t+8", "window_stride": 8, "chunk_length": 8, "video_evidence": "reuse prior independent decode only after rehashing the exact MP4", "full_coverage_required_before_training": True, "source_manifest_hashes": source_hashes, "history_file_locks": file_locks, "command": shlex.join([sys.executable, *sys.argv]), **ROBOT})
    emit(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "gpu_used": False})
    raw_rows, excluded = {}, []
    for path in source_files[3:]:
        for row in rows(path):
            if row.get("asset_id") not in by_id:
                excluded.append({"asset_id": row.get("asset_id"), "attempt_directory": row.get("attempt_directory"), "source_row_sha256": digest(row), "reason": "outside_frozen_replacement_cohort", "historical_failure_reason": row.get("failure_reason"), "historical_success": row.get("sim_task_success")})
                continue
            key = row["attempt_directory"]
            if key in raw_rows and raw_rows[key] != row:
                raise ValueError("conflicting_collection_rows")
            raw_rows[key] = row
    old_audits = {r["attempt_directory"]: r for r in rows(audit_path)}
    accepted, rejected, warnings, sample_cache = [], [], [], {}
    seen_seeds = set()
    input_locks = dict(file_locks)
    from PIL import Image
    for index, (key, original) in enumerate(sorted(raw_rows.items())):
        errors = []; evidence = {}; a = old_audits.get(key, {})
        row = dict(original); directory = Path(key)
        try:
            row = recover_row(a, original)
            case = by_id[row["asset_id"]]
            if row["candidate_id"] != case["candidate_id"] or row["asset_sha256"] != case["asset_sha256"]:
                errors.append("frozen_candidate_mismatch")
            unique_seed = (row["asset_id"], row["seed"])
            if unique_seed in seen_seeds:
                errors.append("duplicate_asset_seed")
            seen_seeds.add(unique_seed)
            ok, reasons, evidence = legacy.strict_check(row)
            errors += reasons
            raw = read(directory / "case_result.json")
            trace = read(directory / "state_trace.json")
            errors += check_actions(trace)
            for name, path in {"case_result": directory / "case_result.json", "state_trace": directory / "state_trace.json", "video": directory / "rollout_raw.mp4"}.items():
                actual = sha(path); input_locks[str(path)] = actual
                if a.get("artifact_hashes", {}).get(name) != actual:
                    errors.append("prior_decode_or_audit_hash_mismatch:" + name)
            source_record = a.get("source_record")
            if source_record and sha(source_record) != a.get("source_record_sha256"):
                errors.append("historical_audit_source_changed")
            re = raw["reproducibility"]
            runner = ROOT / "tools/run_exp3_franka_contact_episode_v0_2.py"
            for name, path, expected in (("runner", runner, re.get("runner_sha256")), ("asset", Path(raw["asset_inspection"]["usd_path"]), row["asset_sha256"]), ("robot", Path(re["franka_usd_path"]), re["franka_usd_sha256"]), ("manifest", Path(raw["artifact_paths"]["case_manifest"]), re["case_manifest_sha256"])):
                observed = sha(path); input_locks[str(path)] = observed
                if expected != observed:
                    errors.append("input_hash_mismatch:" + name)
            pe = raw["physics_evidence"]; robot = raw["robot_evidence"]
            if any(v is not True for v in pe["physical_checks"].values()):
                errors.append("physical_gate_not_passed")
            if robot.get("ik_failure_count") != 0 or robot.get("failure") is not None or raw.get("failure_reason"):
                errors.append("controller_or_episode_failure")
            if robot.get("waypoint_tracking_failure_count", 0):
                warnings.append({"asset_id": row["asset_id"], "attempt_directory": key, "reason": "waypoint_tracking_diagnostic", "details": robot["waypoint_tracking_failures"], "task_predicate_unchanged": True})
            if not raw.get("simulator_success_predicate"):
                errors.append("success_predicate_missing")
            if row["asset_role"] == "articulated":
                states = trace["states"]
                if not all(isinstance(s.get("object_joint_state"), dict) and finite(s["object_joint_state"].get("joint_position")) for s in states):
                    errors.append("joint_state_missing")
                measured = (sum((x-y)**2 for x,y in zip(pe["tracked_child_end_position_m"], pe["tracked_child_start_position_m"])) ** .5 >= .005 or pe.get("tracked_child_rotation_rad", 0) >= .035)
                if not measured:
                    errors.append("success_predicate_not_reproduced")
            # Hash all observations; no source/result is ever rewritten.
            observations = sorted((directory / "observations").glob("frame_*.jpg"))
            frame_hashes = []
            for f in observations:
                with Image.open(f) as im:
                    im.verify()
                frame_hashes.append({"path": str(f), "sha256": sha(f)})
            emit_rows(out / f"frame_hashes_{row['asset_id']}_{row['attempt_index']:04d}.jsonl", frame_hashes)
            row["strict_evidence"] = evidence
            sample_cache[key] = (trace, observations, frame_hashes)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
        row.update({"audit_decision": "rejected" if errors else "accepted", "audit_reasons": sorted(set(errors)), "strict_evidence": evidence})
        (rejected if errors else accepted).append(row)
        if index % 100 == 0:
            print(json.dumps({"checked": index+1, "accepted": len(accepted), "rejected": len(rejected)}), flush=True)
    split, per_asset = coverage(accepted, cases)
    for row in accepted:
        row["frozen_split"] = split[row["attempt_directory"]]
    train = [r for r in accepted if r["frozen_split"] == "train"]
    heldout = [r for r in accepted if r["frozen_split"] == "heldout_reference"]
    missing = [aid for aid, c in per_asset.items() if c["train_count"] < 100]
    shortfall = [aid for aid, c in per_asset.items() if not c["target_met"]]
    samples = []
    for row in train:
        trace, observations, frame_hashes = sample_cache[row["attempt_directory"]]
        samples.extend(make_windows(trace, observations, row, frame_hashes))
    if samples:
        mins = [min(a[d] for s in samples for a in s["action_chunk"]) for d in range(7)]
        maxs = [max(a[d] for s in samples for a in s["action_chunk"]) for d in range(7)]
        for s in samples:
            s["normalized_action_chunk"] = [[2*(a[d]-mins[d])/(maxs[d]-mins[d])-1 if maxs[d] > mins[d] else 0 for d in range(7)] for a in s["action_chunk"]]
            s["normalized_gripper_chunk"] = [[2*x/.04-1 for x in a] for a in s["gripper_chunk"]]
    else:
        mins = maxs = None
    for name, values in (("accepted_trajectory_manifest.jsonl", accepted), ("rejected_trajectory_manifest.jsonl", rejected), ("excluded_historical_rows.jsonl", excluded), ("train_split.jsonl", train), ("heldout_split.jsonl", heldout), ("training_samples.jsonl", samples), ("warnings.jsonl", warnings)):
        emit_rows(out / name, values)
    unchanged = {p: sha(p) == value for p, value in input_locks.items()}
    summary = {"method_id": "exp3_franka_training_dataset_v0_2", "assets": sorted(by_id), "accepted_count": len(accepted), "rejected_count": len(rejected), "excluded_out_of_cohort_count": len(excluded), "train_count": len(train), "heldout_count": len(heldout), "training_sample_count": len(samples), "per_asset": per_asset, "missing_train_assets": missing, "shortfall_assets": shortfall, "training_blocked_missing_success_trajectory": bool(shortfall), "full_coverage_ready": not shortfall and all(unchanged.values()), "warning_count": len(warnings), "rejection_reasons": dict(Counter(r for x in rejected for r in x["audit_reasons"])), **ROBOT}
    emit(out / "successful_trajectory_dataset_manifest.json", summary)
    emit(out / "training_input_contract.json", {"action_normalization": {"min": mins, "max": maxs}, "action_dimension": 7, "gripper_dimension": 2, "proprio_dimension": 9, "action_chunk": 8, "frame_alignment": "post-action observation at t predicts t+1..t+8", "forbidden_input": ["object_state", "task_outcome", "scripted_controller_phase", "heldout_trajectories"], **ROBOT})
    emit(out / "history_hash_lock.json", input_locks)
    emit(out / "history_verification.json", {"passed": all(unchanged.values()), "files_checked": len(unchanged), "mismatches": [p for p,v in unchanged.items() if not v], "frozen_sources": source_hashes})
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    emit(out / "dataset_sha256.json", hashes)
    emit(out / "freeze_receipt.json", {"status": "frozen_ready" if summary["full_coverage_ready"] else "frozen_blocked", "manifest_sha256": sha(out / "successful_trajectory_dataset_manifest.json"), "configuration_sha256": sha(out / "configuration.json"), "dataset_hashes_sha256": sha(out / "dataset_sha256.json"), "command": shlex.join([sys.executable, *sys.argv]), "runtime_seconds": time.monotonic()-started, **ROBOT})
    emit(out / "dataset_freeze_completion_marker.json", {"status": "ready" if summary["full_coverage_ready"] else "blocked", "freeze_receipt_sha256": sha(out / "freeze_receipt.json"), **ROBOT})
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["full_coverage_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
