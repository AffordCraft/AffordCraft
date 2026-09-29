# PAct

Official PAct inference (arXiv:2602.14965; the run configuration records the commit
`955d4ef3c5a973695e5fc0a2b22667eee42ffe37` of the official code, not its remote URL; preview checkpoint `8c6b56c7`) on 2,000 inputs: rows "PAct" of the main comparison table (`tab:main`, Section 4.2) and of the
resource table (`tab:resource-metrics`); the automatic part mask is described in the appendix "Controlled
external-method and module studies".

## Input contract: automatic part mask (`mask_adapter.py`)

The released loader (`ImageConditioned_dataset`) needs an RGBA object image and a per-part label map and reads the part
count from the map; the full-RGB protocol gives only the photograph and the category name. The adapter builds both
files with local models and never reads an annotation:

1. localization: Grounding-DINO tiny (HF revision `a2bb814d`), prompt = the category name split into lowercase words +
   `.`, highest-scoring box; box and text threshold 0.25, otherwise `automatic_localization_failed`;
2. object alpha: SAM ViT-B (HF revision `70c1a07f`) prompted with that box (multimask, highest predicted IoU),
   restricted to the box expanded by 10 %; fewer than 500 pixels -> `automatic_segmentation_failed`;
3. canvas: alpha-tight crop + 10 % margin, official `resize_and_pad_to_square(518)`;
4. part labels: SAM automatic masks (32x32 points, `pred_iou_thresh` 0.88, `stability_score_thresh` 0.95, NMS 0.7)
   merged by PAct's official `label_parts.py` (`get_sam_mask`, `get_sam_mask(existing)`, `clean_segment_edges`,
   `size_th` 2000), at most 32 parts (the model's maximum; smallest parts merged into their largest touching part);
5. `<id>_mask.exr` (float32, identical channels, 0 = background) and a check through the official loader.

A case the adapter cannot localize or segment stays failed; it is never re-run with annotations.

## Inference and export (`lane_runner.py`)

`prepare` builds the lane workspace; `infer` runs the official `infer_imgs.py` unchanged (via `runpy`, byte-identical
copy verified) with `--batch_size 1 --export_arti_objects --save_glb` and the official sampling defaults (seed 42, 25
sparse-structure and 25 SLAT steps, CFG 7.0 / 7.0, articulation `mean_feature_regression_steps` over 20, mesh simplify
0.95, texture 1024); `mark-interrupted` moves an attempt that never finished to `interrupted/` (a case hard-crashed
twice is recorded failed); `finalize` runs the official `scripts/json_to_urdf.py --glb` and writes `result.json`,
`timing.json` and `native_manifest.json` from the exported `object.json`: all parts share PAct's exported object frame
(y-up, normalized extent, `metric_size_source: none`, `mesh_scale` [1,1,1]); each non-root link frame sits at its joint
pivot (as the official `json_to_urdf.py`); revolute limits in radians, prismatic in normalized units; masses are the
official URDF helper's constant fallback, not a prediction (see the notes field of every manifest).

## Tracks, lanes and environments

- Registered 200-input subset (`run_lane.sh`): 2 lanes (subset index % 2) on 80 GB GPUs (A100), python 3.10 /
  torch 2.7.0+cu128 / transformers 4.50.0; amendment 001: the mip-splatting `diff_gaussian_rasterization` (commit
  `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`, built for this environment) installed into `<run>/deps/site` and put on
  `PYTHONPATH` (the official GLB export imports it for texture baking only).
- Remaining 1,800 inputs (`run_lane_sm120.sh`): input list from `make_remaining_1800_list.py` (frozen manifest order,
  subset removed), 18 lanes (index % 18) on RTX 5090 / RTX PRO 6000 / A100 GPUs with the sm_120 geometry environment and an
  extra site directory (`PACT_PYTHONPATH`: pandas, pyfqmr, open3d). Amendment 001 of this run: `<run>/deps/site` (the
  sm_80 rasterizer) must not precede the environment's own sm_120 build; attempts made with the wrong order were moved to
  `interrupted/` and recomputed. A CUDA out-of-memory on a 32 GB lane is an infrastructure failure re-run once on a
  96 GB / 80 GB lane.

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs AFFORDCRAFT_MODELS=/abs/models
export PACT_HOME=/abs/pact PACT_CKPT=/abs/pact-checkpoint PACT_GDINO=/abs/grounding-dino-tiny PACT_SAM=/abs/sam-vit-base
export DINOV2_HUB_REPO=/abs/dinov2-7764ea0f DINOV2_CKPT=/abs/dinov2_vitl14_reg4_pretrain.pth
export PACT_PYTHON=/abs/envs/pact/bin/python PACT_RUN=$AFFORDCRAFT_RUNS/external/pact/subset200
$PACT_PYTHON baselines/pact/write_configuration.py                  # configuration.json (lane partition inside)
bash baselines/pact/run_lane.sh 0 0 & bash baselines/pact/run_lane.sh 1 1 &   # <lane> <gpu>, one GPU per lane
# remaining 1,800:
$PACT_PYTHON baselines/pact/make_remaining_1800_list.py
PACT_RUN=$AFFORDCRAFT_RUNS/external/pact/remaining1800 PACT_LANES=18 \
  PACT_INPUT_LIST=$AFFORDCRAFT_RUNS/inputs/remaining_1800_inputs.json $PACT_PYTHON baselines/pact/write_configuration.py
bash baselines/pact/run_lane_sm120.sh 0 0
```

Outputs: `masks/<source_id>/` (adapter: `adapter.json`, `<id>_processed.png`, `<id>_mask.exr`, preview), per case
`lane-<k>/cases/<source_id>/` (`official_attempt-*_{started,finished}.json`, `result.json`, `timing.json`,
`native_manifest.json`, `obj/part_<id>.obj`), official outputs under `lane-<k>/official_out/attempt-NNN/`, lane
`execution_receipt.json`.

## Compatibility changes (official code, weights and sampler settings untouched)

- `ATTN_BACKEND=sdpa`; a stub `flash_attn` package on `sys.path` (`PACT_FLASH_ATTN_SHIM`) satisfies the official
  import by delegating every flash-attention call to torch scaled-dot-product attention. It ships unchanged as
  `flash_attn_shim/flash_attn/__init__.py` (sha256 `7b5df227e60e2676aef97d2a0139cf1792f933f9dcacf3215e749c7c27ea1f6c`, as
  recorded in the run configuration) and is the default of `PACT_FLASH_ATTN_SHIM`.
- `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')` is served from the local hub code and checkpoint
  (strict load).
- `imageio.v3.imread` of `.exr` returns the float32 label array through OpenEXR (without the FreeImage plugin imageio
  silently decoded 8-bit frames); `ipdb` stubbed; observe-only timers around the pipeline stages.

## Changes from the executed version

- Renamed (run-specific prefixes dropped): `mask_adapter.py`, `lane_runner.py`, `write_configuration.py`,
  `run_lane.sh`, `run_lane_sm120.sh`. The remaining-1800 copies differed from the registered-subset tools only in the run
  id, the adapter id and (configuration writer) the run root, input list, lane count and rule text.
- `lane_runner.py`: the run id written into the records is `--run-id`, default the run root's name (the executed copies
  hard-coded their run name). `mask_adapter.py`: the adapter id label is `automatic_part_mask_adapter_v1` (executed
  labels `pact053_...` / `pact070_...`).
- `write_configuration.py`: locations from environment variables; the input list, the lane count (`PACT_LANES`) and
  a generic partition-rule text replace the per-run constants; the sha256 of the registered subset is checked, the
  remaining list is checked by its size; GPU indices of the shared machine removed from the recorded GPU policy;
  the Hugging Face revision keys renamed to `revision` / `revision_hint` (the executed keys had an `hf` prefix).
- `make_remaining_1800_list.py`: the input-list part of the executed preparation script (the rest derived per-run tool
  copies and packed a bundle for the compute nodes).
- Drivers: locations from environment variables. Formatting with black (AST unchanged).

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `pact053_lane_runner.py` (renamed) | `lane_runner.py` | `b063e92891de3ee9424146966a19fdf4e021e57c3b9b277e99edc75a5c6f3655` |
| `pact070_lane_runner.py` (remaining-1800 run copy; differs only in RUN_ID and docstring) | `lane_runner.py` | `6a8df50c14b38201b2432d5f67542c6f0a73a6affd5e7ceb791d69b2f08181f9` |
| `a100_pact070_prepare.py` (excerpt: the input-list part) | `make_remaining_1800_list.py` | `047653143c1bb142f80f5871bc4876e8034659341def9c5dfa7e2d1a7dbb1e97` |
| `pact053_mask_adapter.py` (renamed) | `mask_adapter.py` | `b5fcef76ab5b7f8d359f1bb1162702d7767f76c6d595b439e6f42d2f087bca8d` |
| `flash_attn/__init__.py` (import shim of the run environment) | `flash_attn_shim/flash_attn/__init__.py` (unchanged) | `7b5df227e60e2676aef97d2a0139cf1792f933f9dcacf3215e749c7c27ea1f6c` |
| `pact070_mask_adapter.py` (remaining-1800 run copy; differs only in ADAPTER_ID) | `mask_adapter.py` | `0a301268638106971dce20211ce0f2befe87a6e69dd21fa6732bc6cc5dcaf096` |
| `pact053_lane.sh` (renamed) | `run_lane.sh` | `d47f09460bf1fd1af767276ddeb7e7570af2e58aa95bebf362fb3544d6d13dc5` |
| `pact070_node_lane.sh` (renamed) | `run_lane_sm120.sh` | `2a04415f960df356e561ae033b557bcce06080fa5fcb7b928c497966b3bb9d2a` |
| `pact053_write_configuration.py` (renamed) | `write_configuration.py` | `8bf771882b7d7bdf11f5ac19ad151b23c87fb6dcd79599f192eb86323c51f3fc` |
| `pact070_write_configuration.py` (remaining-1800 run copy: its run root, input list, 18 lanes and rule text) | `write_configuration.py` | `aee6b71faddce20d4023d7bdbb3e5e3bd6dae43aae6d29614d618b1e0246f994` |
