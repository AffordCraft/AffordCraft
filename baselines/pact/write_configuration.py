#!/usr/bin/env python3
"""Write configuration.json of a PAct run (PACT_RUN) before any inference runs.

Environment (see README.md): AFFORDCRAFT_PROJECT_ROOT (the image paths of the input manifest are relative to it),
AFFORDCRAFT_RUNS (inputs/ holds the frozen 2,000-input manifest and the registered subset), AFFORDCRAFT_MODELS (default
parent of the model directories), PACT_RUN (run root), PACT_INPUT_LIST (input list in the subset schema; default the
registered 200-input subset), PACT_LANES (number of lanes), PACT_HOME (official PAct checkout), PACT_CKPT (official
checkpoint), PACT_GDINO / PACT_SAM (Grounding-DINO tiny / SAM ViT-B), DINOV2_HUB_REPO / DINOV2_CKPT (DINOv2 hub code at
commit 7764ea0f and the ViT-L/14-reg4 checkpoint), PACT_FLASH_ATTN_SHIM (directory holding the flash_attn import shim),
PACT_PYTHON (interpreter recorded in the configuration)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

P = Path(os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace"))
RUNS = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
MODELS = Path(os.environ.get("AFFORDCRAFT_MODELS", str(P / "models")))
ROOT = Path(os.environ.get("PACT_RUN", str(RUNS / "external/pact/subset200")))
SRC = Path(os.environ.get("PACT_HOME", "pact"))
CKPT = Path(os.environ.get("PACT_CKPT", str(MODELS / "pact")))
GDINO = Path(os.environ.get("PACT_GDINO", str(MODELS / "grounding-dino-tiny")))
SAM = Path(os.environ.get("PACT_SAM", str(MODELS / "sam-vit-base")))
DINO_REPO = Path(os.environ.get("DINOV2_HUB_REPO", str(MODELS / "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8")))
DINO_CKPT = Path(os.environ.get("DINOV2_CKPT", str(MODELS / "dinov2_vitl14_reg4_pretrain.pth")))
SHIM = Path(os.environ.get("PACT_FLASH_ATTN_SHIM", str(Path(__file__).resolve().parent / "flash_attn_shim")))
SUBSET = Path(os.environ.get("PACT_INPUT_LIST", str(RUNS / "inputs/subset_200_inputs.json")))
MANIFEST = RUNS / "inputs/input_manifest_2000.jsonl"
CONTRACT = Path(__file__).resolve().parents[2] / "evaluation/gate/manifest_contract.md"
TOOLS = Path(__file__).resolve().parent
PY = os.environ.get("PACT_PYTHON", sys.executable)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(root: Path, exts=None, skip_dirs=(".cache", "__pycache__")):
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(s in p.parts for s in skip_dirs):
            continue
        if exts and p.suffix not in exts:
            continue
        out[str(p.relative_to(root))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return out


def main():
    n_lanes = int(os.environ.get("PACT_LANES", "2"))
    subset = json.loads(SUBSET.read_text())
    if SUBSET.name == "subset_200_inputs.json":
        assert sha(SUBSET) == os.environ.get(
            "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
        )
    else:  # the remaining-1800 list written by make_remaining_1800_list.py
        assert subset["n"] == 1800 and len(subset["inputs"]) == 1800
    ids = [x["input_id"] for x in subset["inputs"]]
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True).stdout.strip()
    git_dirty = chr(10).join(
        l
        for l in subprocess.run(
            ["git", "status", "--short"], cwd=SRC, capture_output=True, text=True
        ).stdout.splitlines()
        if not l.rstrip().endswith(".pyc") and "__pycache__" not in l
    )
    official_files = [
        "infer_imgs.py",
        "modules/inference_utils.py",
        "modules/pact/datasets/components.py",
        "modules/pact/pipelines/pact_i23d_gen_pipe.py",
        "modules/pact/datasets/structured_latent.py",
        "modules/utils/articulation_utils.py",
        "modules/label_2d_mask/label_parts.py",
        "scripts/json_to_urdf.py",
        "scripts/batch_json_to_urdf.py",
        "modules/pact/models/sparse_structure_flow.py",
    ]
    cfg = {
        "run_id": ROOT.name,
        "method_id": "pact-v0.1",
        "method": "PAct: Part-Decomposed Single-View Articulated Object Generation (Liu et al. 2026); official code + official preview checkpoint",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "hostname": platform.node(),
            "python": PY,
            "torch": "2.7.0+cu128",
            "transformers": "4.50.0",
            "nvidia_driver": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
            ).stdout.split()[0],
        },
        "protocol": {
            "name": "full-RGB, category-name instruction, no reference box or mask",
            "inputs": {
                "manifest": str(MANIFEST),
                "manifest_sha256": sha(MANIFEST),
                "subset": str(SUBSET),
                "subset_sha256": sha(SUBSET),
                "subset_n": len(ids),
                "subset_order": subset.get("order"),
                "no_reference_boxes_or_masks": subset.get("no_reference_boxes_or_masks"),
            },
            "image_hash_verification": "every case re-hashes the image file and compares to manifest sha256 (adapter.json image_sha256_verified)",
        },
        "checkpoint": {
            "path": str(CKPT),
            "revision_hint": "pact-PAct-8c6b56c7 (preview checkpoint, cc-by-nc-sa-4.0)",
            "files": tree_hashes(CKPT),
        },
        "official": {
            "source_dir": str(SRC),
            "git_head": git_head,
            "git_status_non_pyc": git_dirty,
            "file_hashes": {f: sha(SRC / f) for f in official_files},
            "entrypoint": "infer_imgs.py (run unchanged via runpy from a per-lane workspace copy; byte-identical to source, verified at prepare)",
            "cli_args": [
                "--model",
                str(CKPT),
                "--data_dir",
                "assets/real_world_examples",
                "--outdir",
                "<lane>/official_out/attempt-NNN",
                "--batch_size",
                "1",
                "--export_arti_objects",
                "--save_glb",
            ],
            "sampling_defaults_kept": {
                "seed": 42,
                "ss_steps": 25,
                "slat_steps": 25,
                "ss_cfg_strength": 7.0,
                "slat_cfg_strength": 7.0,
                "arti_out_mode": "mean_feature_regression_steps",
                "arti_mean_num": 20,
                "mesh_simplify_ratio": 0.95,
                "texture_size": 1024,
                "textured_mesh": True,
                "save_video_grid": True,
                "save_cond_vis_grid": True,
            },
            "urdf_helper": "scripts/json_to_urdf.py --glb (unchanged) run per exported object.json at finalize",
        },
        "compat": {
            "description": "environment/loader compatibility only; no change to PAct weights, sampler settings, prompts or data semantics",
            "flash_attn_shim_dir": str(SHIM),
            "flash_attn_shim_sha256": sha(SHIM / "flash_attn/__init__.py"),
            "attn_backend_env": "ATTN_BACKEND=sdpa (torch scaled_dot_product_attention instead of flash-attn, as in the earlier PAct probe run)",
            "dinov2_repo": str(DINO_REPO),
            "dinov2_checkpoint": str(DINO_CKPT),
            "dinov2_checkpoint_sha256": sha(DINO_CKPT),
            "dinov2_loading": "torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg') intercepted -> same architecture from local repo + local checkpoint (strict load)",
            "exr_read_shim": True,
            "exr_read_shim_reason": "official loader calls imageio.v3.imread(<mask>.exr) expecting the float32 (H,W,3) label array of imageio's FreeImage EXR plugin; "
            "FreeImage is absent here and imageio silently falls back to pyav (8-bit video frames), so imread of '.exr' is routed through OpenEXR "
            "returning the identical float32 RGB array (verified on all 22 official examples: labels 0..N, channels identical)",
            "ipdb_stub": True,
            "instrumentation": "observe-only wrappers around PActPipeline.run_inference_batch/get_part_coords/get_slat_arti/animate_with_articulation/export_arti_objects and "
            "postprocessing_utils.to_glb for timing, peak memory and per-case exception capture; a case whose batch raises is recorded failed and the lane continues",
        },
        "automatic_part_mask_adapter": {
            "adapter_id": "automatic_part_mask_adapter_v1",
            "script": str(TOOLS / "mask_adapter.py"),
            "script_sha256": sha(TOOLS / "mask_adapter.py"),
            "purpose": "input-contract adapter: PAct's released loader needs an RGBA object image + per-part label map (part count read from the map). "
            "The protocol gives only the photograph and the category name, so both files are produced automatically with local models; "
            "no annotation (bbox_normalized, reference masks) is read. This is not a semantic prediction of ours and is logged as a compatibility change.",
            "stages": [
                "1 localization: Grounding-DINO tiny, prompt = category name split into lowercase words + '.', highest-scoring box; none >= 0.25 -> failure_reason automatic_localization_failed (case stays failed)",
                "2 object alpha: SAM ViT-B prompted with that box (multimask, highest predicted IoU), restricted to the box expanded by 10 %; < 500 px -> automatic_segmentation_failed",
                "3 canvas: alpha-tight crop + 10 % margin -> official resize_and_pad_to_square(518) -> <id>_processed.png (RGBA) ; white-composited RGB canvas for SAM",
                "4 part labels: SAM ViT-B automatic mask generation (transformers mask-generation pipeline; 32x32 point grid, 1 crop layer=0, pred_iou_thresh 0.88, stability 0.95, nms 0.7) "
                "merged by PAct's OFFICIAL modules/label_2d_mask/label_parts.py (get_sam_mask -> get_sam_mask(existing, skip_split) -> clean_segment_edges; size_th 2000, "
                "alpha-based background rejection, undetected-region recovery, disconnected-part split) = the app.py process_image + apply_merge path with an empty merge list",
                "5 cap: at most 32 parts (PartBasedSparseStructureFlowModel max_num_parts=32); extra smallest parts merged into their largest touching kept part; labels relabelled 1..N contiguous",
                "6 write <id>_mask.exr: float32 RGB EXR, ZIP, identical channels, 0 = background (same header/dtype/values as the official assets/real_world_examples masks), "
                "<id>_mask_preview.png, and a loader check through the official ImageConditioned_dataset code path (num_parts must equal the adapter's N)",
            ],
            "models": {
                "grounding_dino": {
                    "path": str(GDINO),
                    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
                    "files": tree_hashes(GDINO),
                },
                "sam": {
                    "path": str(SAM),
                    "revision": "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
                    "files": tree_hashes(SAM),
                },
            },
            "thresholds": {
                "gdino_box_threshold": 0.25,
                "gdino_text_threshold": 0.25,
                "crop_margin": 0.10,
                "min_alpha_pixels": 500,
                "canvas": 518,
                "sam_auto": {
                    "points_per_batch": 64,
                    "points_per_crop": 32,
                    "crops_n_layers": 0,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95,
                    "crops_nms_thresh": 0.7,
                },
                "official_size_th": 2000,
                "part_cap": 32,
            },
            "seeds": {
                "torch_manual_seed": 0,
                "numpy_seed": 0,
                "note": "all stages deterministic (no sampling); SAM point grid fixed",
            },
            "official_label_parts_sha256": sha(SRC / "modules/label_2d_mask/label_parts.py"),
            "stubbed_deps": "segment_anything (SAM ViT-H generator object replaced by the transformers SAM ViT-B mask list) and detectron2-based Visualizer (debug drawing only)",
            "part_ordering": "PAct's own loader re-orders parts bottom-up (is_sorted_mask_bottom_up=True, load_bottom_up_mask); the adapter's label ids are therefore only a partition",
        },
        "lane_runner": {"script": str(TOOLS / "lane_runner.py"), "script_sha256": sha(TOOLS / "lane_runner.py")},
        "n_lanes": n_lanes,
        "lane_partition": {
            "rule": f"round-robin by input-list order: lane k = list indices with index % {n_lanes} == k",
            "lane_gpus": {},
            "lanes": {str(k): [sid for i, sid in enumerate(ids) if i % n_lanes == k] for k in range(n_lanes)},
        },
        "subset_path": str(SUBSET),
        "manifest_path": str(MANIFEST),
        "project_root": str(P),
        "native_manifest_contract": {
            "path": str(CONTRACT),
            "sha256": sha(CONTRACT),
            "schema": "affordcraft.external_native_manifest.v1",
        },
        "no_result_selection": True,
        "append_only": True,
        "gpu_policy": "one GPU per lane (CUDA_VISIBLE_DEVICES); no foreign process touched; PIDs recorded in lane logs/pids.json",
        "failure_policy": "a failed case stays failed (never re-run with annotations); an interrupted attempt keeps its outputs under official_out/attempt-NNN and interrupted/, the case is re-run in a new attempt only if it never produced result.json",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "configuration.json", "x") as f:
        json.dump(cfg, f, indent=2)
    print("written", ROOT / "configuration.json")


if __name__ == "__main__":
    main()
