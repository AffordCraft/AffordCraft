# TRELLIS.2

Official TRELLIS.2 inference (https://github.com/microsoft/TRELLIS.2; the run configuration records the commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`, not the remote URL; checkpoint `microsoft/TRELLIS.2-4B` @ `af44b45f2e35a493886929c6d786e563ec68364d`) on the whole photograph, 2,000 inputs:
row "TRELLIS.2" of the main comparison table (`tab:main`, Section 4.2) and of the resource table
(`tab:resource-metrics`). TRELLIS.2 delivers one textured mesh without parts, joints or physical parameters, so only the
adapted gate condition applies.

## Pipeline (`run_trellis2_lane.py`)

The flow of the official `example.py`: `Trellis2ImageTo3DPipeline.from_pretrained(<checkpoint>)` (defaults of its
`pipeline.json`: `pipeline_type` `1024_cascade`, `low_vram`, the sparse-structure / shape / texture samplers, DINOv3
ViT-L/16 image conditioner, RMBG-2.0 background removal), the official `pipeline.preprocess_image` (timed separately),
`pipeline.run(..., seed=42, num_samples=1, max_num_tokens=49152, preprocess_image=False)`, `mesh.simplify(16777216)`,
`o_voxel.postprocess.to_glb(decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1,
remesh_project=0)`, `glb.export(extension_webp=True)`. Additions: `processed_input.png`, `mesh.obj` (geometry of
`mesh.glb`), 8-view 512 px renders instead of the 120-frame video, `timing.json`, `result.json`, `native_manifest.json`
(one link = the whole mesh in the normalized frame, `metric_size_source: none`, `mesh_scale` [1,1,1], no joints,
`physics_contract: false`).

## Tracks, lanes and amendments

Registered 200-input subset: 8 lanes (subset index % 8, `prepare_subset200.py`). Remaining 1,800 inputs: 72 lanes of
25 (frozen-order index % 72, `prepare_remaining1800.py`), same code, weights, environment and parameters. Amendments of
the registered-subset run, kept by the remaining run:

- 001: `torch.cuda.empty_cache()` at the start of the export stage (CuMesh allocates outside torch; GPUs were shared).
- 002 (`oom_rerun.py`): cases whose only failure is a CUDA out-of-memory caused by GPU co-tenancy are re-executed once
  on an exclusive GPU; the original attempt is kept under `interrupted/`.
- 003: after a CUDA out-of-memory the runner records the case and exits with code 3, and `t2_lane.sh` restarts a fresh
  process (up to 14 attempts, finished cases skipped), because leaked non-torch allocations made the next case fail.
- 004: at most one or two lane processes per GPU (low-VRAM mode keeps about 16 GB of weights in host memory per process).

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs AFFORDCRAFT_MODELS=/abs/models
export TRELLIS2_HOME=/abs/TRELLIS.2 TRELLIS2_CKPT=/abs/trellis2-4b-af44b45f TRELLIS2_HF_HOME=/abs/hfcache-trellis2
export TRELLIS2_DINOV3=/abs/dinov3-vitl16 TRELLIS2_RMBG2=/abs/rmbg-2.0 TRELLIS2_EXT_BUILD=/abs/ext
export TRELLIS2_BASE_PYTHON=/abs/envs/geom-sm120/bin/python TRELLIS2_PYTHON=/abs/envs/trellis2/bin/python  # a copy of the base env
bash baselines/trellis2/build_env_sm120.sh && bash baselines/trellis2/fix_triton_sm120.sh
R=$AFFORDCRAFT_RUNS/external/trellis2/subset200
$TRELLIS2_PYTHON baselines/trellis2/prepare_subset200.py --root $R --lanes 8   # configuration.json, lanes, code/
T2_ROOT=$R T2_LANES="0 1" T2_GPU=0 T2_TAG=g0 bash $R/code/t2_lane.sh
TRELLIS2_SUBSET_RUN=$R $TRELLIS2_PYTHON baselines/trellis2/prepare_remaining1800.py --root $AFFORDCRAFT_RUNS/external/trellis2/remaining1800
```

`TRELLIS2_HF_HOME` is an offline huggingface_hub cache holding the two gated repositories the official loaders request:
`facebook/dinov3-vitl16-pretrain-lvd1689m` at `ea8dc2863c51be0a264bab82070e3e8836b02d51` and `briaai/RMBG-2.0` at
`5df4c9c76d8170882c34f6986e848ee07fd0ba43` (layout `hub/models--<org>--<name>/{refs/main, snapshots/<commit>/}`; the
weights came from byte-identical public copies, sha256 equal to the Hugging Face LFS ids checked in
`prepare_subset200.py`). The checkpoint directory also holds `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`
(revision `25e0d31ffbebe4b5a97464dd851910efc3002d96`), the sparse-structure decoder named in `pipeline.json`.

Outputs per case: `mesh.glb`, `mesh.obj`, `processed_input.png`, `timing.json`, `result.json`, `native_manifest.json`,
`case.log`, renders; per lane `execution_receipt.json`.

## Compatibility changes (official code untouched)

- Environment for sm_120 (`build_env_sm120.sh`, `fix_triton_sm120.sh`): torch 2.7.0+cu128 instead of the README's
  2.6.0+cu124, CUDA extensions built for arch 12.0; flash-attn 2.7.4.post1 prebuilt wheel instead of the pinned 2.7.3
  (no torch-2.7 build); transformers >= 4.56 for `DINOv3ViTModel`; utils3d at the official pin; FlexGEMM, CuMesh,
  o-voxel (from the source tree) and nvdiffrec (branch renderutils) built from source (their commits are recorded in the
  configuration); triton 3.3.1 instead of 3.3.0 (3.3.0 cannot compile `tl.dot` for compute capability 12.0, which the
  flex_gemm sparse convolution needs).
- Weights from the local checkpoint directory and the offline cache above; `HF_HUB_OFFLINE=1`.
- Renders read the official HDRI through imageio (the environment's OpenCV has no OpenEXR codec); renders only.
- The pipeline is loaded once per lane process; `verbose=False` for `to_glb`; amendments 001 and 003 above.

## Changes from the executed version

- Locations, interpreters and run roots from environment variables; build scripts without the package-mirror and
  proxy settings of the build machine (default package index) and without scratch paths.
- `prepare_subset200.py` / `prepare_remaining1800.py` = the executed preparation scripts renamed; recorded download
  notes reduced to their technical content; the Hugging Face revision keys renamed to `revision` (the executed keys had
  an `hf` prefix); the remaining-run script copies itself instead of a lane-claim script and identifies its parent run
  by method id. `build_env_sm120.sh` / `fix_triton_sm120.sh` = the executed build scripts renamed; a second build pass
  only repeated the CuMesh clone (the first returned HTTP 503), the o-voxel rebuild and the nvdiffrec step.
- Docstrings: dates and internal run names removed. Formatting with black (AST unchanged).
- Not published: weight download script, lane launchers, GPU queue, claim loops, watchdogs, keepers, amendment
  appliers (their effect is in the published runner and lane script), mirror scripts, per-run summary.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `t2_build.sh` (renamed) | `build_env_sm120.sh` | `88ba43fab32cd5a53ffcbdcec3b78920aaecc07c2f0d1c1acb4e8c2fe52b4cc9` |
| `t2_build2.sh` (second pass (CuMesh clone retry, o-voxel rebuild, nvdiffrec); not published) | `build_env_sm120.sh` | `555ab487ea55b9a0be4db34702f2d8684bf86ab384adbe619077b6742db52a9b` |
| `t2_fix_triton.sh` (renamed) | `fix_triton_sm120.sh` | `40cb552a8d9e3016c38f7a62a2fa087d52b764336b56a3dd1a0a2eb26e91dd4a` |
| `oom_rerun.py` | `oom_rerun.py` | `ed1bdc9082b45872144988dd00eafa9976c2dc5d773152cf48af1d3aa3a69832` |
| `prepare_067.py` (renamed) | `prepare_remaining1800.py` | `a28e7e83f2cbd5de8238b372311bddd12c607ec39358e6edb5ac39bf4c59d958` |
| `prepare_055.py` (renamed) | `prepare_subset200.py` | `c9fbea4403486f785848bd906cf5fd64ec9513bdd5637a7832acb8790907cf5e` |
| `run_trellis2_lane.py` | `run_trellis2_lane.py` | `b1269d2f44c38a72084edcacfe9d0b94a6aa48732c729ea9d6ee0a85c871d4a9` |
| `t2_lane.sh` (registered-subset run; used unchanged by the remaining-1800 run) | `t2_lane.sh` | `ea8ae6aebcc1c36ddeca3e572da41171ac188eb4e848cf6d95b347d25d475cd6` |
| `write_receipt.py` | `write_receipt.py` | `8bf91edc22e0251dbb6f31ecb6b6bd667e3e3b08c85d549aa6e4b706d3618712` |
