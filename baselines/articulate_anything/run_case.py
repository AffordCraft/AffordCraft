#!/usr/bin/env python
"""run_case.py -- run ONE registered input through the official Articulate-Anything image pipeline and write the case
outputs (append-only). Must be started with cwd = the lane's copy of the (patched) source tree, PYTHONPATH containing that
tree and $AA_WORK/code, AA_OPENAI_COMPAT=1, EVAL_API_BASE/EVAL_API_KEY in the environment.

Pipeline = articulate.ArticulationPipeline stages in the official order (mesh retrieval -> link articulation ->
affordance extraction (no-op for images) -> joint articulation), stopping at the first failed stage exactly like
run_pipeline(); the only insertion is a per-case private copy of the retrieved PartNet-Mobility object between stage 1
and stage 2 (the official code renders/annotates into the object directory itself; the shared library stays untouched
and parallel lanes cannot race on it).

Config = conf/config.yaml + the image-modality switches the official demo applies (gradio_app.py setup_pipeline /
_setup_actor_critic_config and examples/articulate_image.ipynb): modality=image, joint_actor.mode=image,
joint_actor.use_cotracker=false, joint_actor.targetted_affordance=false, joint_critic.mode=image,
joint_critic.use_cotracker=false; everything else (actor_critic.max_iter=1, actor_only=false, num_seeds=1, cutoff=5,
category_selector.topk=1, obj_selector hierarchical max_images=10, simulator defaults, in_context num_examples=all,
gen temperature 0.5) at the official default. prompt = the whole photograph; additional_prompt = the category name
(the official category selector then skips its own video object detector and CLIP-matches the name against the
library categories).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CASE_TIMEOUT_SECONDS,
    DATASET_ROOT,
    METHOD_ID,
    PARTNET_EXCLUDE_DIRS,
    RUN_ID,
    VLM_MODEL,
    GPU,
    read_json,
    sha256,
    utc,
    write_json_once,
)
from urdf_native_manifest import build_native_manifest, parse_urdf, MOVABLE

CATEGORY_ALIAS = {
    "StorageFurniture": "Cabinet"
}  # the official library renames StorageFurniture -> Cabinet (partnet_utils.track_obj_types(rename=True))


def stage_log_append(case_dir, record):
    path = os.path.join(case_dir, "stage_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_object(obj_id, dst_root):
    src = os.path.join(str(DATASET_ROOT), str(obj_id))
    dst = os.path.join(dst_root, str(obj_id))
    if os.path.isdir(dst):
        return dst
    os.makedirs(dst_root, exist_ok=True)
    tmp = dst + ".copying"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(*PARTNET_EXCLUDE_DIRS, "temp_robot.urdf"))
    os.replace(tmp, dst)
    return dst


def vlm_stats(case_dir):
    path = os.path.join(case_dir, "vlm_transcript.jsonl")
    calls, ok, prompt_t, completion_t, reasoning_t, total_t, latency, attempts = 0, 0, 0, 0, 0, 0, 0.0, 0
    per_agent = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                calls += 1
                ok += 1 if r.get("status") == "ok" else 0
                prompt_t += r.get("prompt_tokens") or 0
                completion_t += r.get("completion_tokens") or 0
                reasoning_t += r.get("reasoning_tokens") or 0
                total_t += r.get("total_tokens") or 0
                latency += r.get("latency_seconds") or 0.0
                attempts += len(r.get("attempts") or [])
                a = per_agent.setdefault(r.get("agent") or "?", dict(calls=0, total_tokens=0))
                a["calls"] += 1
                a["total_tokens"] += r.get("total_tokens") or 0
    return dict(
        vlm_calls=calls,
        vlm_calls_ok=ok,
        vlm_http_attempts=attempts,
        prompt_tokens=prompt_t,
        completion_tokens=completion_t,
        reasoning_tokens=reasoning_t,
        vlm_tokens=total_t,
        vlm_latency_seconds=round(latency, 2),
        per_agent=per_agent,
    )


def collect_outputs(case_dir):
    out = os.path.join(case_dir, "aa_out")
    info = dict(aa_out=out)
    cat = os.path.join(out, "category_selector", "category_selector.json")
    sel = os.path.join(out, "obj_selector", "object_selector_result.json")
    info["category_selection"] = read_json(cat) if os.path.isfile(cat) else None
    info["object_selection"] = read_json(sel) if os.path.isfile(sel) else None
    info["selected_object_id"] = (info["object_selection"] or {}).get("obj_id")
    r0 = os.path.join(out, "obj_selector", "round_0", "obj_ids.json")
    info["candidate_count"] = len(read_json(r0)) if os.path.isfile(r0) else None
    info["selection_rounds"] = len(glob.glob(os.path.join(out, "obj_selector", "round_*", "obj_ids.json")))
    info["selection_vlm_batches"] = len(
        [d for d in glob.glob(os.path.join(out, "obj_selector", "round_*_*")) if os.path.isdir(d)]
    )
    link_iters = sorted(glob.glob(os.path.join(out, "link_placement", "iter_*", "seed_*")))
    info["link_iterations"] = len(link_iters)
    info["link_placement_dirs"] = link_iters
    info["link_urdfs"] = [
        os.path.join(d, "mobility.urdf") for d in link_iters if os.path.isfile(os.path.join(d, "mobility.urdf"))
    ]
    critics = sorted(glob.glob(os.path.join(out, "link_critic", "iter_*", "seed_*", "link_critic.json")))
    info["link_critic"] = [dict(path=p, **{k: read_json(p).get(k) for k in ("realism_rating",)}) for p in critics]
    joint_iters = sorted(glob.glob(os.path.join(out, "joint_actor", "iter_*", "seed_*")))
    info["joint_iterations"] = len(joint_iters)
    info["joint_dirs"] = joint_iters
    urdfs = [os.path.join(d, "mobility.urdf") for d in joint_iters if os.path.isfile(os.path.join(d, "mobility.urdf"))]
    info["joint_urdfs"] = urdfs
    info["final_urdf"] = urdfs[-1] if urdfs else None
    info["joint_videos"] = sorted(glob.glob(os.path.join(joint_iters[-1], "video_*.mp4"))) if joint_iters else []
    info["joint_pred_py"] = (
        os.path.join(joint_iters[-1], "joint_pred.py")
        if joint_iters and os.path.isfile(os.path.join(joint_iters[-1], "joint_pred.py"))
        else None
    )
    info["link_placement_py"] = (
        os.path.join(link_iters[-1], "link_placement.py")
        if link_iters and os.path.isfile(os.path.join(link_iters[-1], "link_placement.py"))
        else None
    )
    errs = []
    for p in sorted(
        glob.glob(os.path.join(out, "link_error", "iter_*", "seed_*", "error.txt"))
        + glob.glob(os.path.join(out, "joint_error", "iter_*", "seed_*", "error.txt"))
    ):
        txt = open(p, encoding="utf-8", errors="replace").read().strip()
        errs.append(dict(path=p, last_line=(txt.splitlines()[-1][:300] if txt else "")))
    info["actor_critic_errors"] = errs
    return info


def finalize(
    case_dir,
    source_id,
    category,
    additional_prompt,
    stages,
    forced_failure=None,
    started_utc=None,
    t0=None,
    image_path=None,
    image_sha=None,
):
    """Write timing.json, result.json, native_manifest.json (each exactly once). `stages` = list of stage records."""
    info = collect_outputs(case_dir)
    stage_ok = {s["stage"]: s.get("success") for s in stages}
    stage_err = {s["stage"]: s.get("error") for s in stages if s.get("error")}
    if info["final_urdf"]:
        stage_reached = "export"
    elif info["link_urdfs"]:
        stage_reached = "joint_prediction" if stage_ok.get("Link Articulation") else "link_placement"
    elif info["selected_object_id"]:
        stage_reached = "link_placement"
    else:
        stage_reached = "retrieval"
    asset_emitted, failure_reason, parsed = False, None, None
    if info["final_urdf"]:
        try:
            parsed = parse_urdf(info["final_urdf"])
            n_geom = sum(1 for l in parsed["links"] if any(v.get("kind") == "mesh" for v in l["visuals"]))
            asset_emitted = n_geom >= 1
            if not asset_emitted:
                failure_reason = "joint_urdf_without_mesh_links"
        except Exception as e:  # noqa
            failure_reason = "joint_urdf_parse_error: %r" % (e,)
    else:
        failed = [s for s in stages if s.get("success") is False]
        if not info["link_urdfs"] and info["actor_critic_errors"] and info["selected_object_id"]:
            # the official actor-critic loop swallows link-actor exceptions (error_handler -> link_error/.../error.txt, score -1)
            failure_reason = "link placement failed: %s" % info["actor_critic_errors"][0]["last_line"][:400]
        elif failed:
            first = failed[0]
            failure_reason = "%s failed: %s" % (first["stage"], (first.get("error_summary") or "")[:400])
        else:
            failure_reason = "no joint-stage URDF produced"
    if forced_failure:
        asset_emitted = False
        failure_reason = forced_failure
    joint_stage_error = stage_err.get("Joint Articulation")
    part_count = joint_count = None
    fixed_count = None
    if parsed:
        part_count = sum(1 for l in parsed["links"] if any(v.get("kind") == "mesh" for v in l["visuals"]))
        joint_count = sum(1 for j in parsed["joints"] if j["type"] in MOVABLE)
        fixed_count = sum(1 for j in parsed["joints"] if j["type"] == "fixed")
    vs = vlm_stats(case_dir)
    runtime = round(time.time() - t0, 2) if t0 else None
    timing = dict(
        source_id=source_id,
        case_start_utc=started_utc,
        case_end_utc=utc(),
        total_runtime_seconds=runtime,
        stages=[dict(stage=s["stage"], success=s.get("success"), seconds=s.get("seconds")) for s in stages],
        retrieval_seconds=next((s.get("seconds") for s in stages if s["stage"] == "Mesh Retrieval"), None),
        object_copy_seconds=next((s.get("seconds") for s in stages if s["stage"] == "Object Copy"), None),
        link_seconds=next((s.get("seconds") for s in stages if s["stage"] == "Link Articulation"), None),
        joint_seconds=next((s.get("seconds") for s in stages if s["stage"] == "Joint Articulation"), None),
        finalize_seconds=None,
        vlm_latency_seconds=vs["vlm_latency_seconds"],
        vlm_calls=vs["vlm_calls"],
        timeout_seconds=CASE_TIMEOUT_SECONDS,
    )
    t_fin = time.time()
    manifest = build_native_manifest(
        case_dir,
        source_id,
        METHOD_ID,
        RUN_ID,
        info["final_urdf"],
        asset_emitted,
        stage_reached,
        failure_reason,
        extra=dict(
            selected_object_id=info["selected_object_id"],
            requested_category=category,
            additional_prompt=additional_prompt,
            library_object_dir=(
                os.path.join(case_dir, "library", "dataset", str(info["selected_object_id"]))
                if info["selected_object_id"]
                else None
            ),
            joint_stage_error=(joint_stage_error[:600] if joint_stage_error else None),
            vlm_model=VLM_MODEL,
        ),
    )
    write_json_once(os.path.join(case_dir, "native_manifest.json"), manifest)
    timing["finalize_seconds"] = round(time.time() - t_fin, 2)
    write_json_once(os.path.join(case_dir, "timing.json"), timing)
    result = dict(
        source_id=source_id,
        method_id=METHOD_ID,
        run_id=RUN_ID,
        asset_emitted=asset_emitted,
        stage_reached=stage_reached,
        selected_object_id=info["selected_object_id"],
        selected_category=((info["category_selection"] or {}).get("most_similar_objects") or [None])[0],
        requested_category=category,
        additional_prompt=additional_prompt,
        candidate_count=info["candidate_count"],
        selection_vlm_batches=info["selection_vlm_batches"],
        part_count=part_count,
        joint_count=joint_count,
        fixed_joint_count=fixed_count,
        joint_types=(manifest.get("joint_counts") if asset_emitted else None),
        link_iterations=info["link_iterations"],
        joint_iterations=info["joint_iterations"],
        link_critic_ratings=[c.get("realism_rating") for c in info["link_critic"]],
        failure_reason=failure_reason,
        joint_stage_error=(joint_stage_error[:600] if joint_stage_error else None),
        actor_critic_errors=info["actor_critic_errors"],
        runtime_seconds=runtime,
        vlm_calls=vs["vlm_calls"],
        vlm_tokens=vs["vlm_tokens"],
        vlm_prompt_tokens=vs["prompt_tokens"],
        vlm_completion_tokens=vs["completion_tokens"],
        vlm_reasoning_tokens=vs["reasoning_tokens"],
        vlm_http_attempts=vs["vlm_http_attempts"],
        vlm_per_agent=vs["per_agent"],
        vlm_model=VLM_MODEL,
        urdf=info["final_urdf"],
        link_placement_py=info["link_placement_py"],
        joint_pred_py=info["joint_pred_py"],
        joint_videos=len(info["joint_videos"]),
        image=dict(path=image_path, sha256=image_sha),
        gpu=GPU,
        host=os.uname().nodename,
        finished_utc=utc(),
    )
    write_json_once(os.path.join(case_dir, "result.json"), result)
    return result


def run(args):
    case_dir = os.path.abspath(args.case_dir)
    os.makedirs(case_dir, exist_ok=True)
    t0 = time.time()
    started = utc()
    stages = []
    additional_prompt = CATEGORY_ALIAS.get(args.category, args.category)
    write_json_once(
        os.path.join(case_dir, "input.json"),
        dict(
            source_id=args.source_id,
            requested_category=args.category,
            additional_prompt=additional_prompt,
            image=dict(path=args.image, sha256_expected=args.image_sha256),
            started_utc=started,
            lane_src=os.getcwd(),
            pid=os.getpid(),
        ),
    )
    actual = sha256(args.image)
    if actual != args.image_sha256:
        stages.append(
            dict(
                stage="Input Verification",
                success=False,
                seconds=0.0,
                error="sha256 mismatch %s != %s" % (actual, args.image_sha256),
                error_summary="input image sha256 mismatch",
            )
        )
        stage_log_append(case_dir, stages[-1])
        finalize(
            case_dir,
            args.source_id,
            args.category,
            additional_prompt,
            stages,
            forced_failure="input_hash_mismatch",
            started_utc=started,
            t0=t0,
            image_path=args.image,
            image_sha=actual,
        )
        return 2
    os.environ["AA_VLM_TRANSCRIPT"] = os.path.join(case_dir, "vlm_transcript.jsonl")
    os.environ["AA_HYDRA_RUN_DIR"] = os.path.join(case_dir, "hydra_runs")

    from articulate import ArticulationPipeline  # lane src on sys.path / cwd
    from articulate_anything.utils.utils import load_config
    from omegaconf import OmegaConf

    cfg = load_config()
    cfg.prompt = args.image
    cfg.additional_prompt = additional_prompt
    cfg.modality = "image"
    cfg.out_dir = os.path.join(case_dir, "aa_out")
    cfg.dataset_dir = os.path.join(case_dir, "library", "dataset")
    cfg.gpu_id = GPU
    cfg.model_name = VLM_MODEL
    cfg.api_key = None
    cfg.joint_actor.mode = "image"
    cfg.joint_actor.use_cotracker = False
    cfg.joint_actor.targetted_affordance = False
    cfg.joint_critic.mode = "image"
    cfg.joint_critic.use_cotracker = False
    if args.max_iter is not None:
        cfg.actor_critic.max_iter = int(args.max_iter)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(case_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    pipeline = ArticulationPipeline(cfg)

    def stage(name, func):
        t = time.time()
        res = func()
        rec = dict(
            stage=name,
            success=bool(res.success),
            seconds=round(res.time_taken, 2),
            wall_seconds=round(time.time() - t, 2),
        )
        if not res.success:
            rec["error"] = res.error_msg
            rec["error_summary"] = (res.error_msg or "").strip().splitlines()[-1][:300] if res.error_msg else None
        stages.append(rec)
        stage_log_append(case_dir, rec)
        print("[stage] %s success=%s seconds=%.1f" % (name, rec["success"], rec["seconds"]), flush=True)
        return res

    r = stage("Mesh Retrieval", pipeline.process_mesh_retrieval)
    try:  # wrapper-level GPU hygiene only: the official CategorySelector leaves the CLIP model's cached blocks reserved for the rest of the case
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    if r.success:
        t = time.time()
        try:
            obj_id = pipeline.steps["Mesh Retrieval"]["Object Selection"].load_prediction()["obj_id"]
            dst = copy_object(obj_id, cfg.dataset_dir)
            rec = dict(
                stage="Object Copy", success=True, seconds=round(time.time() - t, 2), object_id=str(obj_id), dst=dst
            )
        except Exception:
            rec = dict(
                stage="Object Copy",
                success=False,
                seconds=round(time.time() - t, 2),
                error=traceback.format_exc(),
                error_summary="object copy failed",
            )
        stages.append(rec)
        stage_log_append(case_dir, rec)
        if rec["success"]:
            r = stage("Link Articulation", pipeline.process_link_articulation)
            if r.success:
                r = stage("Affordance Extraction", pipeline.process_affordance_extraction)
                if r.success:
                    r = stage("Joint Articulation", pipeline.process_joint_articulation)
    pipeline._log_processing_summary()
    # transient per-session files the official pybullet wrapper leaves behind on failure
    for p in glob.glob(os.path.join(case_dir, "library", "dataset", "*", "temp_robot.urdf")) + glob.glob(
        os.path.join(cfg.out_dir, "**", "temp_robot.urdf"), recursive=True
    ):
        try:
            os.remove(p)
        except OSError:
            pass
    res = finalize(
        case_dir,
        args.source_id,
        args.category,
        additional_prompt,
        stages,
        started_utc=started,
        t0=t0,
        image_path=args.image,
        image_sha=actual,
    )
    print(
        "[done] asset_emitted=%s stage=%s obj=%s joints=%s tokens=%s runtime=%.1fs"
        % (
            res["asset_emitted"],
            res["stage_reached"],
            res["selected_object_id"],
            res["joint_count"],
            res["vlm_tokens"],
            res["runtime_seconds"] or -1,
        ),
        flush=True,
    )
    return 0 if res["asset_emitted"] else 1


def finalize_only(args):
    """Post-mortem finalization by the lane runner (timeout / crash): everything is read from the case directory."""
    case_dir = os.path.abspath(args.case_dir)
    stages = []
    p = os.path.join(case_dir, "stage_log.jsonl")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            stages = [json.loads(l) for l in f if l.strip()]
    inp = (
        read_json(os.path.join(case_dir, "input.json")) if os.path.isfile(os.path.join(case_dir, "input.json")) else {}
    )
    additional_prompt = inp.get("additional_prompt") or CATEGORY_ALIAS.get(args.category, args.category)
    t0 = None
    if inp.get("started_utc"):
        import datetime

        t0 = (
            datetime.datetime.strptime(inp["started_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
    stages.append(
        dict(
            stage="Lane Runner",
            success=False,
            seconds=None,
            error=args.forced_failure,
            error_summary=args.forced_failure,
        )
    )
    stage_log_append(case_dir, stages[-1])
    res = finalize(
        case_dir,
        args.source_id,
        args.category,
        additional_prompt,
        stages,
        forced_failure=args.forced_failure,
        started_utc=inp.get("started_utc"),
        t0=t0,
        image_path=args.image,
        image_sha=args.image_sha256,
    )
    print("[finalized-after-failure] %s %s" % (args.source_id, res["failure_reason"]), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--image-sha256", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument(
        "--max-iter",
        default=None,
        help="override actor_critic.max_iter (official default 1); the run keeps the default",
    )
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--forced-failure", default=None)
    args = ap.parse_args()
    if args.finalize_only:
        sys.exit(finalize_only(args))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
