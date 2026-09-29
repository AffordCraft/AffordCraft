#!/usr/bin/env python
"""register_configuration.py -- write <run root>/configuration.json BEFORE any lane runs (refuses to overwrite).
Usage: register_configuration.py <lanes> '<assignment json>' [smoke_case_dir]"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    AA_RUN,
    CASE_TIMEOUT_SECONDS,
    CODE,
    DATASET_ROOT,
    FULL_MANIFEST,
    FULL_MANIFEST_SHA,
    GPU,
    MAX_OFFICIAL_ITERATIONS,
    METHOD_ID,
    OFFICIAL_VLM_MODEL,
    PATCHED_SRC,
    PATCH_DIFF,
    PY,
    REV,
    SRC_OFFICIAL,
    SUBSET,
    SUBSET_SHA,
    VLM_MODEL,
    capture,
    lock,
    sha256,
    utc,
    write_json_once,
)


def tree_lock(root, suffixes=None, exclude_dirs=(".git", "__pycache__", "partnet-mobility-v0")):
    root = Path(root)
    files = []
    for p in sorted(root.rglob("*")):
        if any(d in p.parts for d in exclude_dirs):
            continue
        if p.is_file() and not p.is_symlink() and (suffixes is None or p.suffix in suffixes):
            files.append(dict(path=str(p.relative_to(root)), sha256=sha256(p), bytes=p.stat().st_size))
    import hashlib

    h = hashlib.sha256("\n".join("%s %s" % (f["sha256"], f["path"]) for f in files).encode()).hexdigest()
    return dict(root=str(root), file_count=len(files), tree_sha256=h, files=files)


def git(args):
    """git 2.34 on the node refuses the foreign-owned repo (safe.directory is ignored on the command line); with GIT_DIR /
    GIT_WORK_TREE set explicitly the discovery-time ownership check is skipped. Falls back to the .git ref files."""
    real = os.path.realpath(str(SRC_OFFICIAL))
    env = dict(os.environ, GIT_DIR=os.path.join(real, ".git"), GIT_WORK_TREE=real)
    r = subprocess.run(["git"] + args, capture_output=True, text=True, env=env, cwd=real)
    if r.returncode == 0:
        return r.stdout.strip()
    if args[:1] == ["rev-parse"]:
        head = (SRC_OFFICIAL / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = SRC_OFFICIAL / ".git" / head[5:]
            if ref.exists():
                return ref.read_text().strip()
            for line in (SRC_OFFICIAL / ".git" / "packed-refs").read_text().splitlines():
                if line.endswith(" " + head[5:]):
                    return line.split()[0]
        return head
    return "git_failed: " + r.stderr.strip()[:200]


def main():
    lanes = int(sys.argv[1])
    from make_lane_manifests import compute_assignment

    assignment = compute_assignment(lanes, write=False)["assignment"]
    smoke = (
        json.loads(sys.argv[2])
        if len(sys.argv) > 2 and sys.argv[2].startswith("{")
        else (sys.argv[2] if len(sys.argv) > 2 else None)
    )
    REV.mkdir(parents=True, exist_ok=True)
    ds = DATASET_ROOT
    obj_dirs = sorted(d for d in os.listdir(ds) if (ds / d).is_dir())
    with_front = sum(1 for d in obj_dirs if (ds / d / "robot_frontview.png").is_file())
    with_legacy = sum(1 for d in obj_dirs if (ds / d / "mobility.urdf.legacy").is_file())
    cfg_yaml = capture(
        [
            PY,
            "-c",
            "import sys; sys.path.insert(0, %r); from articulate_anything.utils.utils import load_config; from omegaconf import OmegaConf; print(OmegaConf.to_yaml(load_config()))"
            % str(PATCHED_SRC),
        ]
    ).strip()
    cfg = dict(
        schema="affordcraft.external_baseline_configuration.v1",
        method_id=METHOD_ID,
        method="Articulate-Anything (Le et al., ICLR 2025), official code github.com/vlongle/articulate-anything, image modality (in-the-wild photograph + category name), actor-critic at official defaults",
        run_root=str(REV),
        written_utc=utc(),
        written_on_host=os.uname().nodename,
        no_result_selection=True,
        append_only=True,
        source=dict(
            path=str(SRC_OFFICIAL),
            git_head=git(["rev-parse", "HEAD"]),
            git_remote=git(["config", "--get", "remote.origin.url"]),
            git_status_non_pycache=[l for l in git(["status", "--porcelain"]).splitlines() if "__pycache__" not in l],
            official_tree=tree_lock(SRC_OFFICIAL, suffixes={".py", ".yaml", ".json", ".txt", ".csv", ".md"}),
            official_source_tree_modified=False,
            run_copy=dict(
                path=str(PATCHED_SRC),
                note="byte copy of the official tree (no .git/__pycache__) with the compatibility patches below; each lane runs its own copy under %s/lanes/lane-<k>/src"
                % AA_RUN,
            ),
            patch_diff=dict(**lock(PATCH_DIFF), text=Path(PATCH_DIFF).read_text(encoding="utf-8")),
        ),
        hydra_config=dict(
            base="conf/config.yaml + conf/*/default.yaml (official)",
            composed_defaults_yaml=cfg_yaml,
            per_case_overrides=dict(
                modality="image",
                prompt="<absolute path of the input photograph>",
                additional_prompt="<requested_category> (StorageFurniture -> Cabinet, the library alias)",
                out_dir="<case>/aa_out",
                dataset_dir="<case>/library/dataset (private copy of the retrieved object)",
                gpu_id=GPU,
                model_name=VLM_MODEL,
                api_key=None,
                **{
                    "joint_actor.mode": "image",
                    "joint_actor.use_cotracker": False,
                    "joint_actor.targetted_affordance": False,
                    "joint_critic.mode": "image",
                    "joint_critic.use_cotracker": False,
                }
            ),
            actor_critic=dict(
                max_iter=1,
                num_seeds=1,
                actor_only=False,
                conservative=True,
                cutoff=5,
                note="official conf/config.yaml defaults; link stage runs actor+critic once, joint stage runs the actor once (image mode has no joint critic in the official code)",
            ),
            category_selector=dict(topk=1),
            obj_selector=dict(name="hierarchical", max_images=10, frame_index=0),
            in_context=dict(num_examples="all", shuffle_examples=False),
            gen_config=dict(temperature=0.5, overwrite=False),
            simulator="conf/simulator/default.yaml (640x480 frontview, plain floor, no ray tracing, 50 steps per joint video)",
            image_mode_switches_source="gradio_app.py setup_pipeline/_setup_actor_critic_config and examples/articulate_image.ipynb (the official image-modality entry points)",
        ),
        vlm=dict(
            model=VLM_MODEL,
            official_default=OFFICIAL_VLM_MODEL,
            access="OpenAI-compatible chat-completions endpoint (base URL from EVAL_API_BASE; key from EVAL_API_KEY, environment only, never written)",
            client="code/api_client.py (ApiWrapper) selected by the patched prompt_utils.setup_vlm_model when AA_OPENAI_COMPAT=1",
            message_layout="system message = agent system instruction; one user message interleaving text parts and JPEG image_url data URIs (detail low), the layout of the official GPTWrapper",
            temperature=0.5,
            max_tokens="endpoint default (fallback on empty reply: reasoning_effort=low, max_tokens=32768)",
            retries="max 5 attempts, exponential backoff on 408/409/425/429/5xx/timeouts/connection errors",
            transcript="per case vlm_transcript.jsonl (every call: agent, text parts verbatim, image sha256/size, response text, usage incl. reasoning tokens, latency, attempts) + vlm_response_<n>.json in each agent directory",
        ),
        dataset=dict(
            root=str(ds),
            objects=len(obj_dirs),
            objects_with_frontview_png=with_front,
            objects_with_rotated_urdf=with_legacy,
            note="PartNet-Mobility v0 as preprocessed for Articulate-Anything (rotated mobility.urdf + mobility.urdf.legacy, robot_frontview.png per object, *_combined_mesh.obj); only objects with robot_frontview.png are retrieval candidates (official get_candidate_objs); the shared library is read-only during the run, the selected object is copied per case",
            index_files=[
                lock(SRC_OFFICIAL / "partnet_obj_types.json"),
                lock(SRC_OFFICIAL / "obj_types.json"),
                lock(SRC_OFFICIAL / "partnet_mobility_embeddings.csv"),
            ],
            index_note="partnet_obj_types.json = 46 category names (StorageFurniture renamed Cabinet by the official preprocessing) used by the CLIP category selector; partnet_mobility_embeddings.csv is the text-modality mesh index (unused in image mode)",
            clip=dict(
                model="ViT-B/32 (openai/CLIP, official)",
                weights=lock(Path(os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))),
            ),
        ),
        inputs=dict(
            frozen_manifest=dict(path=str(FULL_MANIFEST), sha256=FULL_MANIFEST_SHA),
            registered_subset=dict(path=str(SUBSET), sha256=SUBSET_SHA),
            protocol="whole RGB photograph (prompt) + the requested category name (additional_prompt); no box, no mask, no annotation, no access to our results; each input image sha256-verified before use",
            lane_partition=dict(rule="subset order, subset_index % lanes == lane", lanes=lanes, assignment=assignment),
        ),
        failure_policy=dict(
            timeout_seconds=CASE_TIMEOUT_SECONDS,
            max_official_iterations=MAX_OFFICIAL_ITERATIONS,
            rule="a case exceeding %d s or %d official actor-critic iterations is recorded as failed; a case without a joint-stage URDF (mobility.urdf compiled from the VLM joint program) is a failure and stays a failure with its logs; nothing is filled in, re-selected or re-sampled"
            % (CASE_TIMEOUT_SECONDS, MAX_OFFICIAL_ITERATIONS),
            asset_emitted="true iff the joint stage wrote joint_actor/iter_*/seed_*/mobility.urdf and it parses with >= 1 mesh link (a joint-stage exception after the URDF was compiled, e.g. no movable joint so no video for load_predicted_rendering, is recorded in joint_stage_error but the emitted URDF counts as emitted)",
        ),
        environment=dict(
            python=PY,
            note="aa5090 = byte copy of the campaign env geom5090 (venv layered on sim310: torch 2.7.0+cu128, sapien 3.0.3, pybullet 3.2.7, trimesh 4.12.2) + numpy 1.26.4, opencv 4.11 (pinned by the official setup.py), hydra-core, astor, markdown2, GPUtil, seaborn, pandas, clip (openai/CLIP git), cotracker (facebookresearch/co-tracker@5951295e, imported only), transforms3d, flow_vis; log %s/logs/env_setup.log, freeze %s/logs/aa5090_pip_freeze.txt"
            % (AA_RUN, AA_RUN),
            pip_freeze=lock(AA_RUN / "logs" / "aa5090_pip_freeze.txt"),
            env_setup_log=lock(AA_RUN / "logs" / "env_setup.log"),
        ),
        compatibility_changes=[
            "VLM: gemini-2.5-flash through an OpenAI-compatible endpoint (EVAL_API_BASE / EVAL_API_KEY) instead of the official default gemini-1.5-flash-latest via google.generativeai (Gemini 1.5 Flash is not served by the API endpoint); message layout as the official GPTWrapper (system + one user turn, JPEG images, detail low); Gemini thinking tokens are returned by the API endpoint and counted in the completion tokens.",
            "prompt_utils.py: the google-generativeai / anthropic / openai imports are optional (try/except) and setup_vlm_model() returns the API client when AA_OPENAI_COMPAT=1; agent.py passes the agent class name and output directory to the client for transcript routing only. See patch_diff.",
            "Renderer output: SAPIEN 3 shading differs slightly from the official SAPIEN 2 renders shipped with the library (object 100013: same geometry, camera and framing, mean abs pixel difference 26/255, black background in both); the selected object is re-rendered per case with the port, so the link-critic ground truth and the prediction come from the same renderer; retrieval candidates are the library renders.",
            "Renderer: sapien==2.2.2 (official pin) segfaults inside svulkan2 (Buffer::map during the first camera render) on the run node (RTX 5090, driver 595.71, Vulkan 1.4 ICD); the official articulate_anything/physics/sapien_simulate.py is replaced by code/aa_sapien3_simulate.py, a SAPIEN 3.0.3 port with the same hydra config, camera placement (pos [3,1.5,2], look-at [0,0,0.8], fovy 35 deg, 640x480), lighting, plain floor, raise-distance logic (pybullet), stationary/move logic and output files; the articulation is loaded with fix_root_link=True and zero gravity (SAPIEN 3 has no load_kinematic); the checkerboard floor is not ported (official default is plain).",
            "utils.make_cmd(): the official code launches the renderer with `conda run -n articulate-anything python ...`; here the current interpreter runs the render script (AA_RENDER_SCRIPT substitution above) and hydra.run.dir is redirected into the case directory (AA_HYDRA_RUN_DIR) with hydra.output_subdir=null.",
            "Library isolation: cfg.dataset_dir points to a per-case private copy of the retrieved object (copied after the object selection, PartNet extras images/parts_render*/point_sample skipped); the official code writes link_summary.txt, link_semantics.json, raise_distances.json, robot_frontview.png (re-rendered), link_states.json and temp_robot.urdf into that copy instead of the shared library. Candidate retrieval reads the shared library through datasets/partnet-mobility-v0 (symlink) read-only.",
            "Category prompt: requested_category is passed as additional_prompt (the official category selector skips its video object detector and CLIP-matches the text against the library categories); StorageFurniture is passed as Cabinet because the official preprocessing renames that library category (partnet_utils.track_obj_types(rename=True)); all other 30 subset categories match their library category exactly (CLIP similarity 1.0).",
            "Numerics: numpy 1.26.4 / opencv-python 4.11 in the aa5090 layer (the official setup.py pins numpy==1.26.4); torch 2.7.0+cu128 (sm_120) instead of whatever torch the official conda env would resolve; CLIP ViT-B/32 weights from the official OpenAI URL, sha256-verified, served from the local clip cache (~/.cache/clip).",
            "Nodes: RTX 5090 (sm_120, 32 GB), driver 595.71; one GPU (CUDA_VISIBLE_DEVICES) shared with a PhysX-Anything job; %d parallel lanes, each with its own copy of the source tree (lane-private obj_types.json / hydra outputs)."
            % (lanes,),
        ],
        outputs=dict(
            per_case=[
                "input.json",
                "config_used.yaml",
                "stage_log.jsonl",
                "case.log",
                "vlm_transcript.jsonl",
                "library/dataset/<obj_id>/ (private copy of the retrieved PartNet-Mobility object incl. meshes, rotated URDF, semantics, rendered robot_frontview.png, link_summary.txt)",
                "aa_out/category_selector/category_selector.json",
                "aa_out/obj_selector/round_<r>_<b>/{prompt.html,system_instruction.html,object_selector_result.json,obj_ids.json,best_image_<i>.png,vlm_response_<n>.json}",
                "aa_out/obj_selector/{object_selector_result.json,candidates.png,chosen_object.png}",
                "aa_out/link_placement/iter_<i>/seed_<s>/{response.txt,link_placement.py,mobility.urdf,robot_frontview.png,link_states.json,link_diff.json,prompt.html,vlm_response_<n>.json}",
                "aa_out/link_critic/iter_<i>/seed_<s>/{link_critic.json,prompt.html,vlm_response_<n>.json}",
                "aa_out/joint_actor/iter_<i>/seed_<s>/{response.txt,joint_pred.py,mobility.urdf,video_<joint>_frontview.mp4,prompt.html,vlm_response_<n>.json}",
                "hydra_runs/ (render subprocess logs)",
                "native/link_meshes/<link>.obj",
                "timing.json",
                "result.json",
                "native_manifest.json",
            ],
            result_json="{source_id, method_id, asset_emitted, stage_reached, selected_object_id, selected_category, requested_category, additional_prompt, candidate_count, part_count, joint_count, fixed_joint_count, joint_types, failure_reason, joint_stage_error, runtime_seconds, vlm_calls, vlm_tokens (+prompt/completion/reasoning), ...}",
            native_manifest="affordcraft.external_native_manifest.v1 parsed from the joint-stage URDF by code/urdf_native_manifest.py (per-link combined OBJ with baked visual origins; joints with axis in the parent frame; units m at library scale; physics_contract false)",
        ),
        wrappers=tree_lock(CODE, suffixes={".py", ".sh"}),
        smoke_test=smoke,
        rules=dict(
            no_paper_write=True,
            no_git_push=True,
            official_source_untouched=True,
            foreign_gpu_jobs_untouched=True,
            api_key_never_written=True,
        ),
    )
    write_json_once(REV / "configuration.json", cfg)
    print("configuration.json written", REV / "configuration.json")


if __name__ == "__main__":
    main()
