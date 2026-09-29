#!/usr/bin/env python
"""Prepare a TRELLIS.2-4B run on the registered 200-input subset: verify inputs + weights, write configuration.json first
(open 'x'), then the fixed round-robin lane manifests, and copy the runner code into <root>/code/.
Usage: prepare_subset200.py --root <run dir> --lanes 8 [--smoke-cases N]; refuses to touch an existing configuration.json.
Environment: AFFORDCRAFT_PROJECT_ROOT, AFFORDCRAFT_RUNS, AFFORDCRAFT_MODELS, TRELLIS2_HOME (official checkout),
TRELLIS2_CKPT (microsoft/TRELLIS.2-4B snapshot incl. microsoft/TRELLIS-image-large/ckpts), TRELLIS2_DINOV3 and
TRELLIS2_RMBG2 (gated image conditioner / background removal weights), TRELLIS2_HF_HOME (offline huggingface_hub cache
holding the two gated repos), TRELLIS2_PYTHON (run environment), TRELLIS2_BASE_PYTHON (environment the run environment
was copied from; package diff only), TRELLIS2_EXT_BUILD (build directory of build_env_sm120.sh; commits only).
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
SRC = os.environ.get("TRELLIS2_HOME", "TRELLIS.2")
CKPT = os.environ.get("TRELLIS2_CKPT", f"{MODELS}/trellis2-4b-af44b45f")
DINO = os.environ.get("TRELLIS2_DINOV3", f"{MODELS}/dinov3-vitl16-pretrain-lvd1689m")
RMBG2 = os.environ.get("TRELLIS2_RMBG2", f"{MODELS}/rmbg-2.0")
HFCACHE = os.environ.get("TRELLIS2_HF_HOME", f"{MODELS}/hfcache-trellis2")
PY = os.environ.get("TRELLIS2_PYTHON", sys.executable)
BASE_PY = os.environ.get("TRELLIS2_BASE_PYTHON", sys.executable)
EXT = os.environ.get("TRELLIS2_EXT_BUILD", "ext")
HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {  # HF LFS oids
    f"{CKPT}/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors": "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
    f"{CKPT}/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors": "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41",
    f"{CKPT}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors": "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
    f"{CKPT}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors": "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
    f"{CKPT}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors": "580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686",
    f"{CKPT}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors": "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d",
    f"{CKPT}/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors": "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
    f"{CKPT}/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors": "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
    f"{CKPT}/ckpts/tex_enc_next_dc_f16c32_fp16.safetensors": "dd109f75f84b90fa411554ed6b0e4a87f430841163156fc0ebda2ebdc4752493",
    f"{CKPT}/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors": "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
    f"{DINO}/model.safetensors": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    f"{RMBG2}/model.safetensors": "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(d):
    out = {}
    for dp, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, d)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    return out


def git_head(d):
    r = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or f"unavailable: {r.stderr.strip()[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--smoke-cases", type=int, default=None)
    a = ap.parse_args()
    root = a.root
    os.makedirs(root, exist_ok=True)
    assert not os.path.exists(os.path.join(root, "configuration.json")), "configuration.json exists; append-only"

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
        h = sha256(os.path.join(P, m["image"]["path"]))
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

    ck = walk_files(CKPT)
    dino = walk_files(DINO)
    rmbg2 = walk_files(RMBG2)
    for p, v in EXPECTED.items():
        base = CKPT if p.startswith(CKPT) else DINO if p.startswith(DINO) else RMBG2
        rel = os.path.relpath(p, base)
        got = (ck if base == CKPT else dino if base == DINO else rmbg2)[rel]["sha256"]
        assert got == v, (p, got)
    pipeline_json = json.load(open(os.path.join(CKPT, "pipeline.json")))
    pargs = pipeline_json["args"]
    pip = subprocess.run([PY, "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout
    pip = {d["name"]: d["version"] for d in json.loads(pip)}
    geom_pip = subprocess.run([BASE_PY, "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout
    geom_pip = {d["name"]: d["version"] for d in json.loads(geom_pip)}
    diff = {k: {"geom5090": geom_pip.get(k), "trellis2_5090": v} for k, v in pip.items() if geom_pip.get(k) != v}

    code_dir = os.path.join(root, "code")
    os.makedirs(code_dir, exist_ok=True)
    code = {}
    for f in [
        "run_trellis2_lane.py",
        "t2_lane.sh",
        "write_receipt.py",
        "prepare_subset200.py",
        "build_env_sm120.sh",
        "fix_triton_sm120.sh",
    ]:
        srcp = os.path.join(HERE, f)
        shutil.copy2(srcp, os.path.join(code_dir, f))
        code[f] = sha256(os.path.join(code_dir, f))

    cfg = {
        "revision": os.path.basename(root),
        "method_id": "trellis2-4b-v0.1",
        "method": "TRELLIS.2 (Xiang et al. 2025, 'Native and Compact Structured Latents for 3D Generation'), official inference code + official checkpoint microsoft/TRELLIS.2-4B",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "image-to-3D single textured mesh on the registered AffordCraft 200-input subset",
        "protocol": {
            "input": "whole photograph only (full RGB); no box, mask or annotation; background removal is the method's own preprocessing (RMBG-2.0 BiRefNet inside pipeline.preprocess_image); no category text (the method is image-only)",
            "no_result_selection": True,
            "append_only": True,
            "failed_cases_stay_failed": True,
            "seeds": "single run, seed 42 (official example.py default of pipeline.run)",
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
            "git_commit": git_head(SRC),
            "eigen_submodule_commit": git_head(os.path.join(SRC, "o-voxel/third_party/eigen")),
            "official_entry": "example.py (Trellis2ImageTo3DPipeline.from_pretrained -> pipeline.run -> mesh.simplify(16777216) -> o_voxel.postprocess.to_glb -> glb.export)",
        },
        "checkpoints": {
            "trellis2_4b": {
                "repo": "microsoft/TRELLIS.2-4B",
                "revision": "af44b45f2e35a493886929c6d786e563ec68364d",
                "path": CKPT,
                "files": ck,
                "fetched": "sha256 of every .safetensors == HF LFS oid; microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16 (revision 25e0d31ffbebe4b5a97464dd851910efc3002d96) is the sparse-structure decoder referenced by pipeline.json",
            },
            "dinov3_vitl16": {
                "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
                "path": DINO,
                "files": dino,
                "fetched": "HF repo is gated; files fetched from a public mirror of facebook/dinov3-vitl16-pretrain-lvd1689m; model.safetensors sha256 == HF LFS oid (byte-identical weights); exposed to the unmodified official loader through the offline huggingface_hub cache "
                + HFCACHE,
            },
            "rmbg_2_0": {
                "repo": "briaai/RMBG-2.0",
                "revision": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
                "path": RMBG2,
                "files": rmbg2,
                "fetched": "HF repo is gated; files fetched from a public mirror of briaai/RMBG-2.0; model.safetensors sha256 == HF LFS oid; birefnet.py/BiRefNet_config.py/config.json sizes equal the HF listing; offline huggingface_hub cache "
                + HFCACHE,
            },
        },
        "generation_parameters": {
            "seed": 42,
            "pipeline_type": pargs.get("default_pipeline_type", "1024_cascade"),
            "max_num_tokens": 49152,
            "num_samples": 1,
            "low_vram": pargs.get("low_vram", True),
            "sparse_structure_sampler": pargs["sparse_structure_sampler"],
            "shape_slat_sampler": pargs["shape_slat_sampler"],
            "tex_slat_sampler": pargs["tex_slat_sampler"],
            "image_cond_model": pargs["image_cond_model"],
            "rembg_model": pargs["rembg_model"],
            "simplify_target": 16777216,
            "decimation_target": 1000000,
            "texture_size": 4096,
            "remesh": True,
            "remesh_band": 1,
            "remesh_project": 0,
            "attention_backend": "flash_attn (official default)",
            "sparse_conv_backend": "flex_gemm (official default)",
            "note": "all values are the official defaults of pipeline.json / example.py / pipeline.run()",
        },
        "environment": {
            "python_env": PY,
            "env_origin": "byte copy of the sm_120 geometry environment geom5090 (torch 2.7.0+cu128, sm_120), then the TRELLIS.2 dependencies were installed into the copy (code/build_env_sm120.sh, code/fix_triton_sm120.sh)",
            "installs_in_copy": {
                "pypi": [
                    "zstandard",
                    "pandas",
                    "easydict",
                    "imageio",
                    "imageio-ffmpeg",
                    "ninja",
                    "tqdm",
                    "transformers>=4.56,<5 (DINOv3ViTModel; replaces 4.50.0)",
                ],
                "utils3d": "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 (official pin)",
                "flash_attn": "2.7.4.post1 prebuilt wheel cu12/torch2.7/cxx11abiTRUE/cp310 from github Dao-AILab releases (official setup.sh pins 2.7.3, which has no torch-2.7 wheel); verified to run on sm_120",
                "FlexGEMM": {"repo": "https://github.com/JeffreyXiang/FlexGEMM", "commit": git_head(f"{EXT}/FlexGEMM")},
                "CuMesh": {"repo": "https://github.com/JeffreyXiang/CuMesh", "commit": git_head(f"{EXT}/CuMesh")},
                "o_voxel": "from the source tree o-voxel/ (eigen submodule fetched at the pinned gitlink)",
                "nvdiffrec": {
                    "repo": "https://github.com/JeffreyXiang/nvdiffrec (branch renderutils)",
                    "commit": git_head(f"{EXT}/nvdiffrec"),
                },
                "triton": "triton==3.3.1 from pypi (code/fix_triton_sm120.sh) shadowing the 3.3.0 of the sim310 base env inside the venv copy only: Triton 3.3.0's TritonGPUAccelerateMatmul asserts 'computeCapability not supported' for sm_120, which broke every flex_gemm sparse-conv Triton kernel on the RTX 5090; 3.3.1 (the version torch 2.7.1 ships) adds the sm_120 MMA path",
                "build_flags": "TORCH_CUDA_ARCH_LIST=12.0, --no-build-isolation, CUDA 12.8 toolkit",
            },
            "package_diff_vs_geom5090": diff,
            "key_packages": {
                k: pip.get(k)
                for k in [
                    "torch",
                    "torchvision",
                    "transformers",
                    "flash_attn",
                    "flash-attn",
                    "xformers",
                    "triton",
                    "nvdiffrast",
                    "spconv-cu126",
                    "trimesh",
                    "numpy",
                    "opencv-python-headless",
                    "huggingface-hub",
                    "utils3d",
                    "cumesh",
                    "flex_gemm",
                    "flex-gemm",
                    "o_voxel",
                    "o-voxel",
                    "nvdiffrec_render",
                    "nvdiffrec-render",
                    "zstandard",
                    "easydict",
                ]
            },
            "gpu": "NVIDIA GeForce RTX 5090 32 GB (sm_120)",
            "cuda_toolkit": "12.8",
        },
        "compatibility_changes": [
            "torch 2.7.0+cu128 (sm_120) instead of the README's torch 2.6.0+cu124; CUDA extensions compiled for arch 12.0",
            "flash-attn 2.7.4.post1 prebuilt wheel instead of the pinned 2.7.3 (no torch-2.7 build of 2.7.3); same API",
            "triton 3.3.1 instead of the 3.3.0 that comes with torch 2.7.0 (3.3.0 cannot compile tl.dot kernels for compute capability 12.0; the flex_gemm sparse-conv backend is pure Triton)",
            "visualization renders read the official HDRI (assets/hdri/forest.exr) through imageio (8-bit decode) because the env's opencv-python-headless 5.0 has no OpenEXR codec; render only, the asset is unaffected",
            "weights loaded from the local revision-pinned checkpoint directory and, for the two gated HF repos (DINOv3 ViT-L/16 image conditioner, RMBG-2.0 background removal), from an offline huggingface_hub cache filled with byte-identical copies from a public mirror; the official loaders (DINOv3ViTModel.from_pretrained, AutoModelForImageSegmentation.from_pretrained) run unmodified with HF_HUB_OFFLINE=1",
            "lane runner code/run_trellis2_lane.py loads the pipeline once per lane process, calls the official pipeline.preprocess_image separately (timed) and then pipeline.run(..., preprocess_image=False) with the official seed/params, and passes verbose=False to o_voxel.postprocess.to_glb",
            "example.py's 120-frame 1024px PBR turntable video is replaced by an 8-view 512px shaded+normal strip (visualization only)",
            "additions per case: processed_input.png, mesh.obj (geometry of the exported mesh.glb), timing.json, result.json, native_manifest.json, case.log",
        ],
        "lanes": {
            "n_lanes": lanes,
            "rule": "fixed partition: subset order index i -> lane i mod n_lanes; each lane runs on exactly one node/GPU at a time (recorded in the lane's execution_receipt.json and lane_assignment.json at the run root)",
            "partition": partition,
            "lane_env": "code/t2_lane.sh (T2_LANES, T2_GPU, T2_TAG)",
            "assignment_log": "lane_assignment.jsonl (appended at every launch)",
        },
        "outputs": {
            "per_case": "lane-<k>/cases/<source_id>/{mesh.glb, mesh.obj, processed_input.png, timing.json, result.json, native_manifest.json, case.log, rendering_strip.png, rendering.png}",
            "native_manifest_contract": "evaluation/gate/manifest_contract.md (one link = the single mesh, no joints, metric_size_source none, physics_contract false)",
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
    print(f"prepared {root}: {len(rows)} cases, {lanes} lanes, commit {cfg['source']['git_commit']}")


if __name__ == "__main__":
    sys.exit(main())
