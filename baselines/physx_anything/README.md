# PhysX-Anything

Official PhysX-Anything inference (https://github.com/ziangcao0312/PhysX-Anything, commit
`e221826e6176d940905126d1894f9c1c933b70a8`) with the official weights (HF `Caoza/PhysX-Anything`, snapshot `fdfdf47b`:
`vlm/` fine-tuned Qwen2.5-VL-7B, `decoder/` and `trellis/`) on the whole photograph. Rows "PhysX-Anything" of the main
comparison table (`tab:main`, Section 4.2) and of the resource table (`tab:resource-metrics`); 2,000 inputs.

## Pipeline (per case)

1. Stage A, `physx_anything_vlm_stage.py` (VLM environment): the official functions of `1_vlm_demo.py`
   (`generate_save`, `addmessage`, `dash_str_to_ints`, `voxel_decode`) imported from the unmodified file, the loop body of
   its `__main__` reproduced: 512x512 LANCZOS resize of the whole photograph, `remove_bg=False`, official
   `dataset/overall_prompt.txt`, greedy decoding (`do_sample=False`, `max_length=32768`; the checkpoint's
   generation_config supplies `repetition_penalty=1.05`), `torch.manual_seed(1)` before every generate call, one
   follow-up question per part (official wording), processor `min_pixels=65536`, `max_pixels=262144`, bfloat16.
2. Stage B, `physx_anything_geom_stage.py` (sm_120 geometry environment): the body of the official `2_decoder.py`
   (voxel grid 32, resolution 64, `seed=1`, `to_glb(simplify=0.5, texture_size=1024)`), then the official `3_split.py
   --index 0` and `4_simready_gen.py --voxel_define 32 --process 0 --fixed_base 0 --deformable 0`, both unmodified, as
   subprocesses in a per-case working directory whose `test_demo/<source_id>` and `mjcf_source` are symlinks.
3. Exporter, `mjcf_native_manifest.py`: parses the official `basic.xml` into the native manifest (`units: m`,
   `metric_size_source: method`); `test_mjcf_parser.py` cross-checks it against MuJoCo's compiled model
   (`mjcf_check.json`). `asset_emitted` is true only when `4_simready_gen.py` wrote `basic.xml` and `basic.urdf`.

Exporter unit conventions (verified with `parser_unit_test.sh`, which runs the unmodified official exporter on a
synthetic articulated `basic_info.txt`, `parser_unit_basic_info_synthetic.txt`): every mesh carries
`scale = max(Dimension)/100` (cm to m) and `visual_obj` stays the normalized OBJ, so `mesh_scale` must be applied; hinge
and ball pivots are already in metres; slide ranges are written in normalized units (voxels/32) without the mesh scale
and are kept verbatim, flagged by `range_units`; type-A ("move freely") groups are separate free bodies, not joints.

## Tracks and lanes

- Registered 200-input subset: 9 lanes, lane = subset index % 9 (`make_lane_manifest.py`).
- Remaining 1,800 inputs (frozen manifest order, subset removed): 10 lanes, lane = index % 10
  (`make_lane_manifest_remaining.py`). Each of these lanes stopped its VLM stage once 100 rows had a VLM result, ran
  stage B over them and wrote its receipt; the tail rows (position >= 100) of all ten lanes were computed by 9 helper
  lanes (`make_lane_manifest_helper.py`, tail-first order), so every input is computed once.

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs AFFORDCRAFT_MODELS=/abs/models
export PHYSX_ANYTHING_HOME=/abs/PhysX-Anything PHYSX_ANYTHING_CKPT=/abs/physx-anything-weights
export PHYSX_ANYTHING_PROCESSOR=/abs/qwen2.5-vl-7b-processor TORCH_HOME=/abs/torch-cache U2NET_HOME=/abs/rembg-cache
export PHYSX_ANYTHING_VLM_PYTHON=/abs/envs/vlm/bin/python PHYSX_ANYTHING_GEOM_PYTHON=/abs/envs/geom-sm120/bin/python
export PHYSX_ANYTHING_RUN=$AFFORDCRAFT_RUNS/external/physx-anything/subset200
mkdir -p $PHYSX_ANYTHING_RUN/code && cp baselines/physx_anything/* $PHYSX_ANYTHING_RUN/code/
$PHYSX_ANYTHING_VLM_PYTHON $PHYSX_ANYTHING_RUN/code/register_configuration.py 9 '{}'   # configuration.json first
PA_LANE=0 PA_LANES=9 PA_GPU=0 bash $PHYSX_ANYTHING_RUN/code/run_physx_anything_lane.sh      # one call per lane
```

The lane script calls `code/make_lane_manifest.py`; for the remaining-1800 run copy `make_lane_manifest_remaining.py`
to that name in the run's `code/` (`PHYSX_ANYTHING_RUN=.../remaining1800`, `PA_LANES=10`, `PHYSX_ANYTHING_INPUT_SET=
'frozen_manifest_2000_minus_registered_subset_200 (remaining 1800, frozen order)'`), and `make_lane_manifest_helper.py`
for the helper lanes (`PHYSX_ANYTHING_RUN=.../remaining1800-helper`, `PHYSX_ANYTHING_REMAINING_RUN=.../remaining1800`,
`PA_LANES=9`), as in the executed runs. `parser_unit_test.sh` needs `PHYSX_ANYTHING_SMOKE_CASE` (a finished case).

Inputs: the lane manifest rows (frozen manifest rows: `source_id`, `requested_category`, `image.path`, `image.sha256`);
the category is not given to the method. Outputs per case (`lane-<k>/cases/<source_id>/`): the official outputs
(`basic_info.txt`, `coord_<k>.txt`, `ind_<k>.npy/.ply`, `allind.npy`, `sample.glb`, `objs/`, `basic.xml`, `basic.urdf`,
split/export logs) plus `timing_vlm.json`, `timing.json`, `result.json`, `native_manifest.json`, `mjcf_check.json`; per
lane `execution_receipt.json`.

Dependencies: stage A: torch 2.7.0+cu128, transformers 4.50.0, qwen-vl-utils 0.0.14, accelerate 1.5.2. Stage B (sm_120):
torch 2.7.0+cu128, spconv-cu126 2.3.8, nvdiffrast 0.4.0 and diff_gaussian_rasterization rebuilt for sm_120, xformers
0.0.31, trimesh 4.12.2, mujoco 3.8.1 (official pins: torch 2.1.1+cu118, trimesh 4.0.5, numpy 1.26.4, scipy 1.11.4, xatlas
0.0.9, pymeshfix 0.17.0, pyvista 0.44.2, kaolin 0.15.0; used: numpy 2.2.6, scipy 1.15.3, xatlas 0.0.11, pymeshfix 0.18.1,
pyvista 0.48.4, kaolin 0.18.0). Offline caches: DINOv2 hub repo (commit `7764ea0f`) and `dinov2_vitl14_reg4_pretrain.pth`
under `TORCH_HOME`, `u2net.onnx` under `U2NET_HOME` (both sha256-checked).

## Compatibility changes (official source tree untouched)

- VLM: `attn_implementation="sdpa"` instead of `flash_attention_2` (no flash-attn wheel for sm_120); processor files of
  `Qwen/Qwen2.5-VL-7B-Instruct` loaded from a local copy (same files, same min/max pixels); the model is loaded once per
  lane; case directories are named by `source_id`.
- Decoder: pipeline loaded once per lane from the weights path (`2_decoder.py` hard-codes `./pretrain/decoder`);
  `ATTN_BACKEND=SPARSE_ATTN_BACKEND=xformers` (the fork defaults to flash_attn), xformers FA3 dispatch disabled and
  `cutlass.FwOp` forced (the default dispatch picks a Hopper kernel that fails on sm_120).
- Out of memory: `pipeline.run_control()` runs under `torch.no_grad()` (the fork decorates `run()` but not
  `run_control()`; the retained graph reaches 27 GB and texture baking ran out of memory on the 32 GB RTX 5090; numerics
  unchanged); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for stage B.
- `3_split.py` and `4_simready_gen.py` hard-code `./test_demo` and `mjcf_source/desert.png`: they run unmodified from a
  per-case directory with symlinks.

## Changes from the executed version

- Absolute paths, interpreters and the run root replaced by the environment variables above (`common.py`,
  `run_physx_anything_lane.sh`, `parser_unit_test.sh`, `make_lane_manifest_helper.py`); scratch directories of the
  lane script dropped; `parser_unit_test.sh` passes its output path to the embedded Python as an argument.
- `common.py` merges the three executed copies, which differed only in the run root and the input-set label (now
  `PHYSX_ANYTHING_RUN`, `PHYSX_ANYTHING_INPUT_SET`); `lane_receipt.py` is the remaining-run copy (label from
  `common.py`).
- `register_configuration.py`: registered-subset copy; the remaining-run copies only added the input-set label and their
  partition-rule text. The cloud-provider name was removed from the recorded node description.
- Docstrings: internal run names removed. Formatting with black (AST unchanged).
- Not published: node verification and launch scripts, split watcher (the rule is described above), pull/status
  scripts, per-run summary, a decoder memory diagnostic (not part of any run), a notes writer.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `common.py` (remaining-1800 run) | `common.py` | `876002dad979cdfd301a519a4976e65eb9afba784264e2f709357e5d40267167` |
| `common.py` (registered-subset run; differs only in REV / INPUT_SET) | `common.py` | `3fb2f33294610bbe076057943d80a7ed5b3d0b957b8fc287cf4635dd481f9a61` |
| `common.py` (remaining-1800 helper lanes; differs only in REV / INPUT_SET) | `common.py` | `4fa21c28097f9e3d4f17070a6103f4004dee7027e75e64cd1a662affb6f388fe` |
| `lane_receipt.py` (remaining-1800 run) | `lane_receipt.py` | `76057680fe597adb6928e2faeb43341f09c8db7e5d8a7c5a61ce55119bddca9e` |
| `lane_receipt.py` (registered-subset run; input_set label hard-coded) | `lane_receipt.py` | `2816f0f7c31477749a7431c34215f5eb6a166a55e5e4ef3f85197e271df276ed` |
| `make_lane_manifest.py` (registered-subset run) | `make_lane_manifest.py` | `dab56dd465ceb60e7063da48b22a07b590ccfcd9d3c6c14051317eb876ce0020` |
| `make_lane_manifest_helper.py` (= make_lane_manifest.py of the remaining-1800 helper lanes) | `make_lane_manifest_helper.py` | `f31f8e0c42a1a4ae45d46143418c78f2e6db63b16b65d766a3f8f52a11c4cd86` |
| `make_lane_manifest_remaining.py` (= make_lane_manifest.py of the remaining-1800 run) | `make_lane_manifest_remaining.py` | `63b221d1f821410101b48faa12fad61ff12c47c8b32e64954f2ad97d0dc7a628` |
| `mjcf_native_manifest.py` | `mjcf_native_manifest.py` | `d42207c186dd152afa4500d2e4bcb79e546b4f85f71909cc32039c3a4e5006bb` |
| `parser_unit_basic_info_synthetic.txt` | `parser_unit_basic_info_synthetic.txt` | `07cc19567338705fa3f5a90405c118189a4983978fb8141afeea158c694c1dd0` |
| `parser_unit_test.sh` | `parser_unit_test.sh` | `985f8b27ba127b1bea9a5df55114bbd30c9a2b3b4e1062cf9461b0af96f3b6fe` |
| `physx_anything_geom_stage.py` | `physx_anything_geom_stage.py` | `f3bb1999a87735438be0ddca0eb3a261daa474a426804d7a15080dd9035cf55d` |
| `physx_anything_vlm_stage.py` | `physx_anything_vlm_stage.py` | `169209e679fe186b775a38229813dabbbf63d9951dd825697a073d9e0662f9c4` |
| `register_configuration.py` (registered-subset run) | `register_configuration.py` | `ace6d7ac08daf6897fe86848e21310de9abca8d3eed76d95dc03a25e1a56b409` |
| `register_configuration.py` (remaining-1800 run; adds input_set and its partition rule text) | `register_configuration.py` | `59483dd1ad4ccc00d59e6d2e02637929a245a4a0d0aee5e00a1590e1bdd17ed2` |
| `register_configuration.py` (remaining-1800 helper lanes; adds input_set and its partition rule text) | `register_configuration.py` | `aba2c420dbdced89c4cc15d029164ca9451a6523f50fedbb582510f13687f9af` |
| `run_physx_anything_lane.sh` (registered-subset run) | `run_physx_anything_lane.sh` | `c89448fae2f9e8037b939258f94d2ba35e1d8dbeb57bc17c1b5eb6c7e68aa0ed` |
| `run_physx_anything_lane.sh` (remaining-1800 run; differs only in REV) | `run_physx_anything_lane.sh` | `cd25bffcb6b3bc522f96e4b7f150b38b6cbcb03df8b714bc78ae997afe205125` |
| `run_physx_anything_lane.sh` (remaining-1800 helper lanes; differs only in REV) | `run_physx_anything_lane.sh` | `ce529564162b3ba6aa8c899e23b9af8752774d25e71cb88b09e7b4c4918fc1ce` |
| `test_mjcf_parser.py` | `test_mjcf_parser.py` | `f2118290bf18b638b89895721561ab1b728142368126d6e6edb8d5c280c58418` |
