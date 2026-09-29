#!/usr/bin/env python3
"""Append-only, detached-safe pilot runner with explicit infrastructure receipts.

This diagnostic does not collect training data or change task thresholds. Only
the demonstration robot's waypoints may differ from a predecessor. Runtime
failure, task failure, media failure and subprocess exit are separate fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROBOT = dict(robot_real_world="not_evaluated", robot_control_status="simulated_robot_only", robot_task_success=None)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def emit(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

def probe(command):
    r = subprocess.run(command, text=True, capture_output=True, timeout=30)
    return dict(command=command, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)

def register(a):
    a.output.mkdir(parents=True, exist_ok=False)
    repo = a.repo.resolve()
    definition = a.definition.resolve()
    wanted = set(a.asset_ids.split(",")) if a.asset_ids else None
    cases = [json.loads(line) for line in (definition / "collection_cases.jsonl").read_text().splitlines() if line.strip()]
    jobs = []
    for i, case in enumerate(cases):
        aid = case["asset_id"]
        if wanted is not None and aid not in wanted:
            continue
        base_spec = definition / "task_specs" / (aid + ".json")
        spec = json.loads(base_spec.read_text())
        job = a.output / aid
        job.mkdir()
        wc = spec.get("workcell", {})
        changes = {}
        spec["support_contact_observation_contract"] = "physx_lifecycle_sleep_v0_1"
        spec["contact_coordinate_convention"] = "usd_row_vector_world_to_link_inverse_including_uniform_scale"
        spec["forbidden_object_fixture_paths"] = ["/World/Exp3RobotPedestal"]
        # All source body IDs, not just the currently operated link.
        if spec["asset_role"]=="articulated" and not spec.get("object_body_paths"):
            raise ValueError("static_body_inventory_required_for_fixture_guard")
        spec.setdefault("object_body_paths",[spec["target_link"]])
        spec["episode_termination_policy"] = "independent_full_task_predicate_at_control_boundary_v0_1"
        spec["episode_termination_applies_to"] = ["demonstration_teacher", "untrained_vla", "trained_vla"]
        emit(job / "task_spec.json", spec)
        cm = Path(case["case_manifest_path"])
        c = json.loads(cm.read_text())
        emit(job / "case_manifest.json", c)
        overrides = {k: v for k, v in {
            "position_only_ik": a.position_only_ik,
            "finger_contact_compensation": a.finger_contact_compensation,
            "articulated_orientation": a.articulated_orientation,
            "asset_uniform_scale": a.asset_uniform_scale,
        }.items() if v is not None}
        options = dict(attempt_robot="true", controller="lula_ik", physics_hz=60, fps=10,
                       width=320, height=240, settle_steps=60, control_steps=360,
                       save_observations="true", renderer="RayTracedLighting",
                       franka_usd_path=a.franka, gpu_index=a.gpu_index,
                       support_top_z=wc.get("support_top_z", .9),
                       franka_base_x=wc.get("franka_base_x", -.66),
                       franka_base_y=wc.get("franka_base_y", 0.),
                       franka_base_z=wc.get("franka_base_z", .922),
                       asset_uniform_scale=wc.get("asset_uniform_scale", 1.),
                       articulated_placement="actuated_child_center",
                       rigid_push_distance=a.teacher_push_distance if a.teacher_push_distance is not None else spec.get("teacher_push_distance_m", .08),
                       rigid_contact_standoff=.025,
                       position_only_ik=wc.get("position_only_ik", "false"))
        for k, v in wc.items():
            if k.startswith("articulated_") or k.startswith("finger_contact_"):
                options[k] = v
        options.update(overrides)
        command = [a.python, str(repo / "tools/run_exp3_complete_task_contact_teacher_v0_125.py"),
                   "--case-manifest", str(job / "case_manifest.json"), "--repo-root", str(repo), "--output-dir", str(job / "episode")]
        for k, v in options.items():
            command.extend(["--" + k.replace("_", "-"), str(v)])
        start = spec.get("nominal_start_center_world_m", [-.15, 0.])
        env = dict(EXP3_COMPLETE_TASK_SPEC=str(job / "task_spec.json"), EXP3_INITIAL_X_M=str(start[0]), EXP3_INITIAL_Y_M=str(start[1]), EXP3_COLLECTION_SEED=str(a.seed + i), ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES")
        native_lib = Path(a.python).parent.parent / "lib"
        env.update(LD_PRELOAD=str(native_lib / "libstdc++.so.6"), LD_LIBRARY_PATH=str(native_lib), TMPDIR=str(job / "tmp"))
        spec_hash = sha(job / "task_spec.json")
        input_asset = Path(c["asset"]["usd_path"])
        if sha(input_asset) != c["asset"]["usd_sha256"]:
            raise ValueError("candidate_hash_changed:" + aid)
        row = dict(asset_id=aid, asset_role=spec["asset_role"], directory=str(job), command=command, environment=env,
                   task_spec_sha256=spec_hash, candidate_sha256=sha(input_asset), task_identity_corrections=changes,
                   predecessor_task_spec=str(base_spec), predecessor_task_spec_sha256=sha(base_spec),
                   teacher_overrides=overrides, task_thresholds_changed=False, frozen_before_viewing=False,
                   post_hoc_after_failure_observed=True, formal_replay_frozen_before_execution=True, **ROBOT)
        emit(job / "registration.json", row)
        jobs.append(row)
    if wanted and {j["asset_id"] for j in jobs} != wanted:
        raise ValueError("missing_registered_asset")
    cfg = dict(version="v0.125", phase="diagnostic_pilot_not_training", jobs=jobs,
               invocation=[sys.executable, *sys.argv], total_required_assets=10, required_successes_per_asset=100,
               runner_sha256=sha(repo / "tools/run_exp3_complete_task_contact_teacher_v0_125.py"),
               launcher_sha256=sha(__file__), timeout_seconds=600,
               source_dependencies={str(p):sha(p) for p in (repo/"src/i2ia/tasks").glob("*.py")},
               libstdcpp_sha256=sha(Path(a.python).parent.parent / "lib/libstdc++.so.6"), **ROBOT)
    emit(a.output / "configuration.json", cfg)
    emit(a.output / "freeze_receipt.json", dict(configuration_sha256=sha(a.output / "configuration.json"), status="pilot_definition_frozen", **ROBOT))

def execute(output):
    cfg = json.loads((output / "configuration.json").read_text())
    receipt = json.loads((output / "freeze_receipt.json").read_text())
    if sha(output / "configuration.json") != receipt["configuration_sha256"]:
        raise ValueError("definition_changed")
    for path, expected in cfg["source_dependencies"].items():
        if sha(path) != expected:
            raise ValueError("source_dependency_changed_after_freeze:" + path)
    summaries = []
    for job in cfg["jobs"]:
        directory = Path(job["directory"])
        terminal = directory / "execution_receipt.json"
        if terminal.exists():
            summaries.append(json.loads(terminal.read_text()))
            continue
        if (directory / "episode").exists():
            # No in-place retry: preserve interrupted files and require a new
            # append-only registration for the same frozen input.
            emit(terminal, dict(asset_id=job["asset_id"], status="blocked", failure_reason="interrupted_attempt_requires_new_directory", **ROBOT))
            summaries.append(json.loads(terminal.read_text()))
            continue
        if sha(job["command"][1]) != cfg["runner_sha256"]:
            raise ValueError("runner_changed_after_freeze")
        if sha(directory / "task_spec.json") != job["task_spec_sha256"]:
            raise ValueError("task_changed_after_freeze")
        Path(job["environment"]["TMPDIR"]).mkdir()
        emit(directory / "gpu_before.json", probe(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv"]))
        env = os.environ.copy()
        # Do not mask CUDA enumeration for Kit; active_gpu and physics_gpu are
        # explicit SimulationApp indices. No unrelated process is modified.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(job["environment"])
        emit(directory / "exact_command.json", dict(argv=job["command"], environment=job["environment"]))
        start = time.monotonic()
        with (directory / "launcher.log").open("x") as log:
            process = subprocess.Popen(job["command"], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            emit(directory / "pid.json", dict(pid=process.pid, command=job["command"], **ROBOT))
            timed_out = False
            try:
                rc = process.wait(timeout=cfg["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    rc = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    rc = process.wait()
        result_file = directory / "episode/case_result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        runtime_failure = directory / "episode/runtime_failure.json"
        failure = json.loads(runtime_failure.read_text()) if runtime_failure.exists() else {}
        emit(directory / "gpu_after.json", probe(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"]))
        status = dict(asset_id=job["asset_id"], status="terminal", subprocess_returncode=rc, timed_out=timed_out,
                      runtime_seconds=time.monotonic() - start, in_process_result_present=bool(result),
                      sim_task_success=result.get("sim_task_success"), sim_ready_status=result.get("sim_ready_status", "blocked"),
                      failure_reason=result.get("failure_reason", [failure.get("failure_detail", "initialization_failed_no_case_result")]),
                      video_validation=result.get("video_validation"), runtime_failure=failure,
                      output_hashes={str(p.relative_to(directory)): sha(p) for p in directory.rglob("*") if p.is_file() and "tmp" not in p.relative_to(directory).parts}, **ROBOT)
        emit(terminal, status)
        summaries.append(status)
    if not (output / "execution_receipt.json").exists():
        emit(output / "execution_receipt.json", dict(status="pilots_terminal", cases=summaries, claimed_training_trajectories=0, **ROBOT))
        emit(output / "completion_marker.json", dict(status="pilots_terminal", execution_sha256=sha(output / "execution_receipt.json")))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--definition", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python")
    p.add_argument("--franka")
    p.add_argument("--asset-ids")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=860000)
    p.add_argument("--teacher-push-distance", type=float)
    p.add_argument("--position-only-ik", choices=["true", "false"])
    p.add_argument("--finger-contact-compensation", choices=["true", "false"])
    p.add_argument("--articulated-orientation")
    p.add_argument("--asset-uniform-scale", type=float)
    p.add_argument("--register-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.resume:
        if not all([a.definition, a.python, a.franka]):
            p.error("registration requires definition, python and franka")
        register(a)
    if not a.register_only:
        execute(a.output)

if __name__ == "__main__":
    main()
