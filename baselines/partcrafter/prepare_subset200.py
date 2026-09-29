#!/usr/bin/env python
"""Prepare a PartCrafter run on the registered 200-input subset: verify inputs + weights, write configuration.json
first (open 'x'), then the fixed round-robin lane manifests, and copy the runner code into <root>/code/.
Usage: prepare_subset200.py --root <run dir> --lanes 8 [--smoke-cases N] ; refuses to touch an existing configuration.json.
Environment: AFFORDCRAFT_PROJECT_ROOT (image paths of the manifest are relative to it), AFFORDCRAFT_RUNS (inputs/),
AFFORDCRAFT_MODELS, PARTCRAFTER_HOME (official checkout), PARTCRAFTER_CKPT (wgsxm/PartCrafter snapshot),
PARTCRAFTER_RMBG (briaai/RMBG-1.4 snapshot), PARTCRAFTER_PYTHON (interpreter of the run environment).
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

P = os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace")
RUNS = os.environ.get("AFFORDCRAFT_RUNS", "runs")
MODELS = os.environ.get("AFFORDCRAFT_MODELS", os.path.join(P, "models"))
MANIFEST = f"{RUNS}/inputs/input_manifest_2000.jsonl"
SUBSET = f"{RUNS}/inputs/subset_200_inputs.json"
SRC = os.environ.get("PARTCRAFTER_HOME", "PartCrafter")
CKPT = os.environ.get("PARTCRAFTER_CKPT", f"{MODELS}/partcrafter-69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f")
RMBG = os.environ.get("PARTCRAFTER_RMBG", f"{MODELS}/rmbg-1.4-2ceba5a5")
PY = os.environ.get("PARTCRAFTER_PYTHON", sys.executable)
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--smoke-cases", type=int, default=None, help="smoke: keep only the first N subset cases")
    ap.add_argument("--vlm-model", default="gemini-3.8-flash")
    a = ap.parse_args()
    root = a.root
    os.makedirs(root, exist_ok=True)
    assert not os.path.exists(os.path.join(root, "configuration.json")), "configuration.json exists; append-only"

    # ---- inputs -------------------------------------------------------------------------------------------------
    man_sha, sub_sha = sha256(MANIFEST), sha256(SUBSET)
    assert man_sha == os.environ.get(
        "AFFORDCRAFT_MANIFEST_SHA256", "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"
    ), man_sha
    assert sub_sha == os.environ.get(
        "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
    ), sub_sha
    by_id = {}
    for l in open(MANIFEST):
        if l.strip():
            m = json.loads(l)
            by_id[m["source_id"]] = m
    subset = json.load(open(SUBSET))
    rows = []
    for s in subset["inputs"]:
        m = by_id[s["input_id"]]
        img = os.path.join(P, m["image"]["path"])
        h = sha256(img)
        assert h == m["image"]["sha256"] == s["image"]["sha256"], (s["input_id"], h)
        rows.append(
            {
                "source_id": m["source_id"],
                "requested_category": m["requested_category"],
                "image": {"path": m["image"]["path"], "sha256": h},
            }
        )
    assert len(rows) == 200
    if a.smoke_cases:
        rows = rows[: a.smoke_cases]
    lanes = a.lanes
    partition = {f"lane-{k}": [r["source_id"] for i, r in enumerate(rows) if i % lanes == k] for k in range(lanes)}

    # ---- source / weights ---------------------------------------------------------------------------------------
    commit = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", SRC, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    ck_files = {}
    for dp, _, fs in os.walk(CKPT):
        for f in fs:
            p = os.path.join(dp, f)
            ck_files[os.path.relpath(p, CKPT)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    expected = {
        "image_encoder_dinov2/model.safetensors": "aa0b83921a3339259fb1ef684ce63c9d09b5a45f9998e0789a0ad4cba318b07b",
        "transformer/diffusion_pytorch_model.safetensors": "915aba1d29c626d960305a0d36a6edb7798288a19c6724388b005c2fa53bcbf8",
        "vae/diffusion_pytorch_model.safetensors": "b43b006e5692223877427cdb568c2c1477f52a2d226db4d5eb354b4886c167a4",
    }
    for k, v in expected.items():
        assert ck_files[k]["sha256"] == v, k
    rm_files = {
        f: {"bytes": os.path.getsize(os.path.join(RMBG, f)), "sha256": sha256(os.path.join(RMBG, f))}
        for f in sorted(os.listdir(RMBG))
        if os.path.isfile(os.path.join(RMBG, f))
    }
    assert rm_files["model.safetensors"]["sha256"] == "46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f"
    pip = subprocess.run([PY, "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout
    pip = {d["name"]: d["version"] for d in json.loads(pip)}

    # ---- code copy ----------------------------------------------------------------------------------------------
    code_dir = os.path.join(root, "code")
    os.makedirs(code_dir, exist_ok=True)
    code = {}
    for f in [
        "run_partcrafter_lane.py",
        "api_part_count_provider.py",
        "pc_lane.sh",
        "write_receipt.py",
        "prepare_subset200.py",
    ]:
        shutil.copy2(os.path.join(HERE, f), os.path.join(code_dir, f))
        code[f] = sha256(os.path.join(code_dir, f))

    cfg = {
        "revision": os.path.basename(root),
        "method_id": "partcrafter-v0.1",
        "method": "PartCrafter (Lin et al. 2025), official inference code + official checkpoint wgsxm/PartCrafter",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "image-to-3D part-level generation on the registered AffordCraft 200-input subset",
        "protocol": {
            "input": "whole photograph only (full RGB); no box, mask or annotation is given to the method; the part count comes from the method's own VLM part-suggestion path; the category name is NOT passed (the official prompt takes no category)",
            "no_result_selection": True,
            "append_only": True,
            "failed_cases_stay_failed": True,
            "seeds": "single run, seed 0 (official default)",
        },
        "inputs": {
            "manifest": MANIFEST,
            "manifest_sha256": man_sha,
            "subset": SUBSET,
            "subset_sha256": sub_sha,
            "image_root": P,
            "n_cases": len(rows),
            "image_hashes_verified": True,
            "order": subset.get("order"),
            "smoke_subset": a.smoke_cases,
            "per_case_hash_check": "the lane runner re-verifies each image sha256 before use",
        },
        "source": {
            "path": SRC,
            "git_commit": commit,
            "git_status_porcelain": status,
            "official_entry": "scripts/inference_partcrafter.py (run_triposg + main flow with --part_suggest --rmbg)",
        },
        "checkpoints": {
            "partcrafter": {
                "repo": "wgsxm/PartCrafter",
                "revision": "69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f",
                "path": CKPT,
                "files": ck_files,
                "fetched": "sha256 == HF LFS oids",
            },
            "rmbg_1_4": {
                "repo": "briaai/RMBG-1.4",
                "revision": "2ceba5a5efaec153162aedea169f76caf9b46cf8",
                "path": RMBG,
                "files": rm_files,
                "fetched": "sha256 == HF LFS oid",
            },
        },
        "generation_parameters": {
            "seed": 0,
            "num_tokens": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.0,
            "max_num_expanded_coords": 1e9,
            "use_flash_decoder": False,
            "rmbg": True,
            "style_transfer": False,
            "part_suggest": True,
            "max_num_parts": 16,
            "dtype": "float16",
            "pipeline_bounds": "official default (-1.005..1.005)",
            "note": "all values are the official defaults of scripts/inference_partcrafter.py",
        },
        "part_suggest": {
            "provider": "openai_compatible (compatibility module code/api_part_count_provider.py)",
            "model": a.vlm_model,
            "official_default_model": "gemini-3-flash-preview via google-genai SDK (not reachable here); closest available Gemini 3-family flash model on the API endpoint",
            "prompt": "official src/utils/providers/gemini_provider._build_num_parts_prompt(16, mode='object') (imported, unchanged)",
            "parser": "official _parse_int_response, retry once, clamp to [1,16]",
            "raw_reply_kept": "cases/<source_id>/vlm_reply.json",
            "sampling": "endpoint defaults; no max_tokens/temperature set (the official call sets none either)",
            "prefetch": "the lane runner issues the VLM call of every pending case from a small worker-thread pool (--vlm-workers, default 4) ahead of the GPU loop and consumes the replies in subset order; the call is independent of the generation, so this only hides API latency (timing.json keeps vlm_seconds = call duration and vlm_wait_seconds = time the GPU loop actually waited)",
            "non_determinism": "the VLM reply is not deterministic (thinking model; the smoke run returned 5 and 6 parts for the same image on two calls); single run, no re-asking, the raw reply of the one call used is kept per case",
        },
        "smoke": "3-case smoke runs: 5/5, 1/1, 6/6, 1/1, 4/4 parts decoded; generation 8-53 s per case, peak 6.5 GB allocated",
        "environment": {
            "python_env": PY,
            "env_origin": "byte copy of the sm_120 geometry environment (geom5090); no package installed or changed in the copy",
            "installs_in_copy": [],
            "key_packages": {
                k: pip.get(k)
                for k in [
                    "torch",
                    "torchvision",
                    "diffusers",
                    "transformers",
                    "accelerate",
                    "trimesh",
                    "pyrender",
                    "torch-cluster",
                    "numpy",
                    "opencv-python-headless",
                    "scikit-image",
                    "huggingface-hub",
                    "xformers",
                    "triton",
                ]
            },
            "gpu": "NVIDIA GeForce RTX 5090 32 GB (sm_120)",
            "cuda_toolkit": "12.8",
            "rendering": "pyrender EGL (headless)",
        },
        "compatibility_changes": [
            "weights loaded from the local revision-pinned directories instead of snapshot_download(repo_id=...) (same HF revisions, sha256-verified)",
            "VLM provider: official gemini_provider needs the google-genai SDK + GEMINI_API_KEY; replaced at runtime by code/api_part_count_provider.py (registered as PROVIDERS['openai_compatible'] in the official registry, no official file edited) which sends the SAME prompt text and the same [image, prompt] content to an OpenAI-compatible endpoint (EVAL_API_BASE / EVAL_API_KEY; /v1/chat/completions) and applies the official integer parser and retry-once rule; transport-level retries on HTTP 5xx/timeouts added",
            "lane runner code/run_partcrafter_lane.py inlines the body of run_triposg() (identical arguments) to time the rmbg and generation stages separately and loads the models once per lane process",
            "additions per case: processed_input.png (the rmbg-cropped image fed to the pipeline), part_XX.obj export of every decoded part, vlm_reply.json, timing.json, result.json, native_manifest.json, reduced 8-view renders (rendering_strip.png / rendering.png / rendering_normal.png)",
            "official dummy-mesh rule kept: a part whose decoding fails becomes a 1-vertex dummy part_XX.glb and is NOT listed as a link",
        ],
        "lanes": {
            "n_lanes": lanes,
            "rule": "fixed partition: subset order index i -> lane i mod n_lanes; each lane runs on exactly one node/GPU at a time (recorded in the lane's execution_receipt.json and lane_assignment.json at the run root)",
            "partition": partition,
            "lane_env": "code/pc_lane.sh (PC_LANES, PC_GPU, PC_TAG)",
            "assignment_log": "lane_assignment.jsonl (appended at every launch)",
        },
        "outputs": {
            "per_case": "lane-<k>/cases/<source_id>/{part_XX.glb, part_XX.obj, object.glb, manifest.json, processed_input.png, vlm_reply.json, timing.json, result.json, native_manifest.json, case.log, renders}",
            "native_manifest_contract": "evaluation/gate/manifest_contract.md (links = decoded parts, no joints, metric_size_source none, physics_contract false)",
            "receipt": "lane-<k>/execution_receipt.json",
        },
        "code_sha256": code,
    }
    with open(os.path.join(root, "configuration.json"), "x") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for k in range(lanes):
        d = os.path.join(root, f"lane-{k}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "input_manifest.jsonl"), "x") as f:
            for r in rows:
                if r["source_id"] in partition[f"lane-{k}"]:
                    f.write(json.dumps(r) + "\n")
    print(f"prepared {root}: {len(rows)} cases, {lanes} lanes, commit {commit}")


if __name__ == "__main__":
    sys.exit(main())
