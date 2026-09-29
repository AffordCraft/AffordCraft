#!/usr/bin/env python
"""Write configuration.json for a PhysX-Anything run (official code + official weights) into PHYSX_ANYTHING_RUN
BEFORE any lane runs. Refuses to overwrite. Hashes the four official scripts, the prompt, the whole official trellis fork,
every weight file used, the auxiliary checkpoints, the processor files, the frozen inputs and these wrappers."""
import os, sys, json, subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def tree_lock(root, suffixes=None):
    root = Path(root)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and (suffixes is None or p.suffix in suffixes):
            files.append(dict(path=str(p.relative_to(root)), sha256=sha256(p), bytes=p.stat().st_size))
    h = hashlib.sha256("\n".join("%s %s" % (f["sha256"], f["path"]) for f in files).encode()).hexdigest()
    return dict(root=str(root), file_count=len(files), tree_sha256=h, files=files)


def git(args):
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(SRC)] + args, capture_output=True, text=True
    ).stdout.strip()


def main():
    lanes = int(sys.argv[1])
    assignment = json.loads(sys.argv[2])
    smoke = sys.argv[3] if len(sys.argv) > 3 else None
    code = REV / "code"
    cfg = dict(
        schema="affordcraft.external_baseline_configuration.v1",
        method_id=METHOD_ID,
        method="PhysX-Anything (Cao et al., arXiv 2511.13648), official inference code + official weights (HF Caoza/PhysX-Anything snapshot fdfdf47b)",
        run_root=str(REV),
        written_utc=utc(),
        written_on_host=os.uname().nodename,
        no_result_selection=True,
        append_only=True,
        source=dict(
            path=str(SRC),
            git_head=git(["rev-parse", "HEAD"]),
            git_remote=git(["config", "--get", "remote.origin.url"]),
            git_status_non_pycache=[l for l in git(["status", "--porcelain"]).splitlines() if "__pycache__" not in l],
            official_scripts=[
                lock(SRC / n) for n in ("1_vlm_demo.py", "2_decoder.py", "3_split.py", "4_simready_gen.py")
            ],
            prompt=lock(SRC / "dataset" / "overall_prompt.txt"),
            mjcf_skybox=lock(SRC / "mjcf_source" / "desert.png"),
            trellis_fork=tree_lock(SRC / "trellis", suffixes={".py", ".json"}),
            official_source_tree_modified=False,
        ),
        weights=dict(
            root=str(WEIGHTS),
            vlm=[lock(p) for p in sorted((WEIGHTS / "vlm").iterdir()) if p.is_file()],
            decoder=[lock(WEIGHTS / "decoder" / "pipeline.json")]
            + [lock(p) for p in sorted((WEIGHTS / "decoder" / "ckpt_new").iterdir())],
            trellis=[lock(WEIGHTS / "trellis" / "pipeline.json")]
            + [lock(p) for p in sorted((WEIGHTS / "trellis" / "ckpts").iterdir())],
            dinov2=dict(
                **lock(DINO_CKPT), hub_repo_commit="7764ea0f912e53c92e82eb78a2a1631e92725fc8", hub_cache=str(DINO_REPO)
            ),
            u2net=lock(U2NET),
            processor=[lock(p) for p in sorted(PROCESSOR.iterdir()) if p.is_file()],
            processor_note="local copy of the Qwen/Qwen2.5-VL-7B-Instruct processor files; vocab/merges/added tokens verified identical to the fine-tuned checkpoint tokenizer files",
        ),
        inputs=dict(
            frozen_manifest=dict(path=str(FULL_MANIFEST), sha256=FULL_MANIFEST_SHA),
            registered_subset=dict(path=str(SUBSET), sha256=SUBSET_SHA),
            protocol="whole RGB photograph only (no box, no mask, no annotation, no category hint); each input image sha256-verified before use",
            lane_partition=dict(rule="subset order, subset_index % lanes == lane", lanes=lanes, assignment=assignment),
        ),
        generation=dict(
            vlm=dict(
                model="fine-tuned Qwen2.5-VL-7B (official vlm checkpoint)",
                dtype="bfloat16",
                attn_implementation="sdpa",
                device_map="auto",
                image_resize=[512, 512],
                resample="LANCZOS",
                remove_bg=False,
                prompt="dataset/overall_prompt.txt (official)",
                min_pixels=65536,
                max_pixels=262144,
                decoding="greedy (do_sample=False, temperature=0, max_length=32768) as in generate_save(); checkpoint generation_config supplies repetition_penalty=1.05",
                per_part_question="official wording from 1_vlm_demo.py",
                save_part_ply=True,
                torch_manual_seed_before_each_generate=1,
            ),
            decoder=dict(
                pipeline="TrellisImageTo3DPipeline.from_pretrained(<weights>/decoder) (official pipeline.json: 25 steps, cfg 5.0, cfg_interval [0.5,1.0], rescale_t 3.0, ss encoder control)",
                voxel_grid=32,
                resolution=64,
                coord_offset="allind + 32 - 16",
                seed=1,
                image="original photograph; the official pipeline preprocess (rembg u2net + crop + 518 resize) is applied inside run_control",
                to_glb=dict(simplify=0.5, texture_size=1024),
                spconv_algo="native",
            ),
            split=dict(script="3_split.py --index 0", rotation_deg_x=90, seed_tau_ratio=0.02, min_seed_faces=20),
            export=dict(script="4_simready_gen.py", voxel_define=32, process=0, fixed_base=0, deformable=0),
        ),
        environments=dict(
            vlm_stage=dict(
                python=VLA_PY,
                note="torch 2.7.0+cu128, transformers 4.50.0, qwen-vl-utils 0.0.14, accelerate 1.5.2 (pip freeze recorded per lane in vlm_environment.json)",
            ),
            geometry_stage=dict(
                python=GEOM_PY,
                note="sm_120 copy of the campaign env: spconv-cu126 2.3.8, nvdiffrast 0.4.0 and diff_gaussian_rasterization rebuilt for sm_120, xformers 0.0.31, torch 2.7.0+cu128, trimesh 4.12.2, mujoco 3.8.1 (pip freeze recorded per lane in geom_environment.json)",
            ),
            pip_installs_for_this_run=[],
        ),
        compatibility_changes=[
            'VLM: attn_implementation="sdpa" instead of the official "flash_attention_2" (flash-attn is not installed; no sm_120 wheel).',
            'VLM: processor loaded from the local path %s instead of the hub id "Qwen/Qwen2.5-VL-7B-Instruct" (same files, same min/max pixels).'
            % PROCESSOR,
            "VLM: the fine-tuned model is loaded once per lane and the official functions generate_save/addmessage/dash_str_to_ints/voxel_decode are imported from the unmodified 1_vlm_demo.py; the per-case loop body is reproduced verbatim in code/physx_anything_vlm_stage.py; case directories are named by source_id instead of image-file stem.",
            "Decoder: the body of 2_decoder.py is reproduced verbatim in code/physx_anything_geom_stage.py with the pipeline loaded once per lane from the absolute weights path (2_decoder.py hard-codes ./pretrain/decoder and ./demo).",
            'Decoder attention: ATTN_BACKEND=xformers and SPARSE_ATTN_BACKEND=xformers (the fork defaults to flash_attn, not installed); xformers FA3 dispatch disabled (_set_use_fa3(False)) and cutlass.FwOp forced for memory_efficient_attention because the default dispatch selects a Hopper kernel that fails with "invalid argument" on sm_120 (same fix as the PhysX-Omni geometry runner).',
            "Decoder env: sm_120 environment geom5090 (spconv-cu126 2.3.8, rebuilt nvdiffrast/diff_gaussian_rasterization; torch 2.7.0+cu128 vs the official pin torch 2.1.1+cu118; trimesh 4.12.2 vs 4.0.5; numpy 2.2.6 vs 1.26.4; scipy 1.15.3 vs 1.11.4; xatlas 0.0.11 vs 0.0.9; pymeshfix 0.18.1 vs 0.17.0; pyvista 0.48.4 vs 0.44.2; kaolin 0.18.0 vs 0.15.0).",
            "Offline caches: torch.hub dinov2 repo (commit 7764ea0f) and dinov2_vitl14_reg4_pretrain.pth served from TORCH_HOME=%s; rembg u2net.onnx from U2NET_HOME=%s; no network access during the run."
            % (TORCH_HOME, U2NET_HOME),
            "Decoder: pipeline.run_control() is called under torch.no_grad() (the fork decorates run() with @torch.no_grad() but not run_control(); without it the retained autograd graph reaches 27 GB and texture baking runs out of memory on the 32 GB RTX 5090; inference numerics are unchanged). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set for the geometry stage.",
            "3_split.py and 4_simready_gen.py run unmodified as subprocesses per case from a per-case working directory whose test_demo/<source_id> and mjcf_source are symlinks (the scripts hard-code ./test_demo and mjcf_source/desert.png).",
            "Nodes: RTX 5090 (sm_120, 32 GB), driver 595.71, one GPU per lane via CUDA_VISIBLE_DEVICES.",
        ],
        outputs=dict(
            per_case=[
                "basic_info.txt",
                "coord_<k>.txt",
                "ind_<k>.npy",
                "ind_<k>.ply",
                "allind.npy",
                "sample.glb",
                "objs/<k>/<k>.obj (+ material files)",
                "basic_info.json",
                "basic.urdf",
                "basic.xml",
                "desert.png",
                "split.log",
                "export.log",
                "timing_vlm.json",
                "timing.json",
                "result.json",
                "native_manifest.json",
                "mjcf_check.json",
            ],
            result_json="{source_id, method_id, stage_reached, asset_emitted (true only when basic.xml and basic.urdf were written by 4_simready_gen.py), part_count, failure_reason, runtime_seconds, mjcf_mujoco_load_ok}",
            native_manifest="parsed from the official MJCF by code/mjcf_native_manifest.py; cross-checked against mujoco 3.8.1 by code/test_mjcf_parser.py (mjcf_check.json)",
            failure_policy="a case without basic.xml+basic.urdf is a failure and stays a failure with its logs; nothing is filled in or selected",
        ),
        wrappers=tree_lock(code, suffixes={".py", ".sh"}),
        smoke_test=smoke,
        rules=dict(
            no_paper_write=True, no_git_push=True, official_source_untouched=True, foreign_gpu_jobs_untouched=True
        ),
    )
    write_json_once(REV / "configuration.json", cfg)
    print("configuration.json written", REV / "configuration.json")


if __name__ == "__main__":
    main()
