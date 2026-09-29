#!/usr/bin/env python3
"""Independent read-only task/media/contact/hash verifier for v86+ pilot jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import copy
import math
import hashlib,base64
import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_exp3_remote_pilot_jobs_v0_87 import emit, sha, ROBOT
from verify_complete_task_clock_and_video_v0_1 import check_clock
from i2ia.tasks.raw_camera_framing import verify as verify_framing
from i2ia.tasks.full_task_training_admission_v0_2 import decide
from exp3_training_instruction_contract_v0_134 import release_evidence,instruction


def recompute_safety(spec,trace,unused_scope):
    normalized=copy.deepcopy(trace);mismatches=[]
    for index,state in enumerate(normalized["states"]):
        forbidden=False;penetration=0.
        for event in state["raw_contacts"]:
            pair=event["paths"][:2]
            robot=[p for p in pair if p.startswith("/World/Exp1Franka/")]
            fixture=any(p in spec.get("forbidden_object_fixture_paths",[]) for p in pair)
            asset=any(p in spec.get("object_body_paths",[spec["target_link"]]) for p in pair)
            for contact in event["samples"]:
                penetration=max(penetration,-float(contact["separation"]))
                physical=(contact["separation"]<=spec["maximum_contact_separation_m"] and
                          math.sqrt(sum(float(x)**2 for x in contact["impulse"]))>=spec["minimum_contact_impulse_ns"])
                if not physical:continue
                if fixture and asset:forbidden=True
                if robot:
                    legal=len(robot)==1 and robot[0] in spec["allowed_robot_contact_links"] and spec["target_link"] in pair
                    forbidden=forbidden or not legal
        if forbidden!=state["forbidden_collision"] or abs(penetration-state["max_penetration_m"])>1e-8:mismatches.append(index)
        state["forbidden_collision"]=forbidden;state["max_penetration_m"]=penetration
    return normalized,mismatches

def inspect(job, require_process_exit=True):
    errors = []
    ep = job / "episode"
    read = lambda p: json.loads(Path(p).read_text())
    spec = read(job / "task_spec.json")
    trace = read(ep / "complete_task_raw_trace.json")
    physics = read(ep / "case_result.json")
    process = read(job / ("execution_receipt.json" if require_process_exit else "call_receipt.json"))
    if require_process_exit:
        if process.get("subprocess_returncode") != 0 or not process.get("in_process_result_present"):
            errors.append("process_exit_requires_review")
    elif process.get("episode_call_completed_normally") is not True:
        errors.append("episode_call_not_completed")
    for name, digest in process["output_hashes"].items():
        if not (job / name).is_file() or sha(job / name) != digest:
            errors.append("output_hash_changed:" + name)
    if sha(job / "task_spec.json") != physics["full_task_spec_sha256"]:
        errors.append("spec_hash_changed")
    if sha(ep / "complete_task_raw_trace.json") != physics["full_task_trace_sha256"]:
        errors.append("trace_hash_changed")
    normalized, mismatches = recompute_safety(spec, trace, True)
    if mismatches:
        errors.append("raw_contact_safety_mismatch")
    local_frame_errors = []
    for i, state in enumerate(trace["states"]):
        matrix = state.get("target_body_world_matrix_row_major")
        if matrix is None:
            if state["contacts"]:
                local_frame_errors.append(i)
            continue
        inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
        for contact in state["contacts"]:
            expected = (np.r_[contact["point_world_m"], 1.] @ inverse)[:3]
            if not np.allclose(expected, contact["point_link_local_m"], atol=1e-7, rtol=0.):
                local_frame_errors.append(i)
    if local_frame_errors:
        errors.append("contact_local_frame_not_independently_verified")
    sync = read(ep / "synchronized_video_receipt.json")
    media = check_clock(trace, sync, fps=spec.get("control_hz", 10), physics_hz=60)
    videos = []
    for name in ["rollout_raw.mp4", "evidence_raw.mp4"]:
        path = ep / name
        count = nonblack = 0
        try:
            with imageio.get_reader(path) as reader:
                for frame in reader:
                    count += 1
                    nonblack += int(frame.mean() > 1 and frame.max() > 8 and frame.std() > 1)
            if count < 2 or count != media["frame_count"] or nonblack != count:
                media["errors"].append(name + ":frame_or_content_failure")
        except Exception as exc:
            media["errors"].append(name + ":decode_failure:" + str(exc))
        videos.append(dict(path=str(path.resolve()), sha256=sha(path), frames=count, nonblack=nonblack, camera_capture_type="raw_isaac_rgb"))
    media.update(passed=not media["errors"], videos=videos)
    framing = verify_framing(read(ep / "live_framing_trace.json"), sync)
    decision = decide(spec, normalized, physics, media, framing)
    # This verifier evaluates genuine VLA episodes, never training admission.
    # Reuse the same physical/media checks, omit only the teacher-identity
    # requirement that is specific to a teacher training dataset.
    outcome_errors=[x for x in decision['failure_reasons'] if x!='not_a_demonstration_teacher']
    released=release_evidence(spec,normalized['states'])
    if not released['passed']:outcome_errors.append('release_then_hold_clause_not_verified')
    variant=trace.get('model_variant')
    identity=physics.get('policy_identity',{})
    if variant not in ('untrained_vla','trained_vla') or physics.get('model_variant')!=variant or identity.get('model_variant')!=variant:errors.append('model_variant_mismatch')
    if physics.get('teacher_used') is not False or trace.get('teacher_used') is not False:errors.append('teacher_used_or_missing_in_vla_evaluation')
    responses=read(ep/'policy_responses.json')
    if responses.get('scripted_teacher_used') is not False or responses.get('object_commands_sent') is not False:errors.append('policy_shortcut_or_missing_instrumentation')
    for response in responses['responses']:
        if response.get('model_variant')!=variant or response.get('adapter_sha256')!=identity.get('adapter_sha256') or response.get('fallback_used') is not False:errors.append('policy_response_binding_failure')
        recorded=dict(response);digest=recorded.pop('response_sha256',None)
        if hashlib.sha256(json.dumps(recorded,sort_keys=True).encode()).hexdigest()!=digest:errors.append('policy_response_hash_changed')
    by_input={r['input_sha256']:r for r in responses['responses']}
    actions=read(ep/'state_trace.json')['actions']
    current_request=None
    for action in actions:
        observation=action['observation'];image=Path(observation['path'])
        if sha(image)!=observation['sha256']:errors.append('policy_observation_hash_changed')
        if action['source']!=variant or action.get('fallback_used') is not False:errors.append('action_not_from_registered_model')
        if action['chunk_index']==0:
            payload=dict(rgb_png_base64=base64.b64encode(image.read_bytes()).decode(),proprio=observation['robot_proprio'],instruction=instruction(spec)['training_and_paired_evaluation_instruction'])
            current_request=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if current_request!=action['policy_input_sha256'] or current_request not in by_input:errors.append('policy_request_not_supported_by_raw_observation')
        else:
            expected=np.asarray(by_input[current_request]['joint_target_chunk'][action['chunk_index']],dtype=float)
            if not np.array_equal(expected,np.asarray(action['raw_policy_joint_target'])):errors.append('neural_action_substitution_detected')
            target=np.clip(expected,spec['action_limits']['min'],spec['action_limits']['max']);q=np.asarray(observation['robot_proprio'])
            delta=target[:7]-q[:7];maximum=float(np.max(np.abs(delta)));factor=min(1.,.06/maximum) if maximum else 1.
            target[:7]=q[:7]+factor*delta
            actual=action['action']+action['gripper_action']
            if not np.allclose(target,actual,rtol=0,atol=1e-7):errors.append('robot_command_not_registered_policy_adapter_output')
    scope = read(ep / "contact_instrumentation.json")
    if scope.get("scope") != "all_robot_bodies_object_support" or scope.get("observation_only") is not True:
        errors.append("contact_instrumentation_scope_missing")
    independent_success=not errors and not outcome_errors
    if physics.get('sim_task_success') is not independent_success:errors.append('collector_independent_success_conflict')
    # Pilots never enter the formal 1000-trajectory dataset automatically.
    return dict(asset_id=spec["asset_id"], candidate_id=spec["candidate_id"], pilot_admitted=False,
                model_variant=variant,sim_task_success=independent_success,
                evidence_valid=not errors,task_failure_reasons=sorted(set(outcome_errors)),release_evidence=released,
                process_exit_verified=require_process_exit and not errors,
                provisional_only=not require_process_exit,
                training_admitted=False, training_count=0, full_task_evaluation=decision["full_task_evaluation"],
                failure_reasons=sorted(set(errors)), media=media, framing=framing,
                contact_identity=decision["contact_identity"], raw_safety_mismatch_indices=mismatches,
                local_frame_error_indices=sorted(set(local_frame_errors)),
                task_spec_sha256=sha(job / "task_spec.json"), trace_sha256=sha(ep / "complete_task_raw_trace.json"),
                collector_success_boolean_used=False, **ROBOT)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        result = inspect(a.job)
    except Exception as exc:
        result = dict(pilot_admitted=False, training_admitted=False, failure_reasons=[type(exc).__name__ + ":" + str(exc)], **ROBOT)
    emit(a.output / "independent_verification.json", result)
    emit(a.output / "execution_receipt.json", dict(argv=[sys.executable, *sys.argv], code_sha256=sha(__file__), output_sha256=sha(a.output / "independent_verification.json")))
    print(json.dumps({k: v for k, v in result.items() if k not in ("media", "framing", "local_frame_error_indices")}, ensure_ascii=False))
    return 0 if result.get('evidence_valid') else 1

if __name__ == "__main__":
    raise SystemExit(main())
