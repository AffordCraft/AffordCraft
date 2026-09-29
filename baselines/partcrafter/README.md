# PartCrafter

Official PartCrafter inference (https://github.com/wgsxm/PartCrafter; the run configuration records the commit
`3d773bf02fad51c7ab31a5615573fec93b287b30`, not the remote URL; checkpoints `wgsxm/PartCrafter` @ `69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f` and `briaai/RMBG-1.4` @
`2ceba5a5efaec153162aedea169f76caf9b46cf8`) on the whole photograph, 2,000 inputs: row "PartCrafter" of the main
comparison table (`tab:main`, Section 4.2) and of the resource table (`tab:resource-metrics`). PartCrafter delivers part
meshes without joints or physical parameters, so only the adapted gate condition applies.

## Pipeline (`run_partcrafter_lane.py`)

The per-case flow of the official `scripts/inference_partcrafter.py` invoked with `--part_suggest --rmbg --seed 0
--num_tokens 1024 --num_inference_steps 50 --guidance_scale 7.0 --max_num_expanded_coords 1e9` (no flash decoder, no
style transfer), float16:

1. `set_seed(0)`; part count from the official `src.utils.vlm_utils.suggest_num_parts(image, MAX_NUM_PARTS=16,
   mode="object")` with the provider `openai_compatible` (`api_part_count_provider.py`): the official prompt
   (`gemini_provider._build_num_parts_prompt`) and integer parser imported unchanged, content order `[image, prompt]`,
   one retry on an unparsable reply, clamp to [1, 16]; model `gemini-3.8-flash` through an OpenAI-compatible endpoint
   (`EVAL_API_BASE` / `EVAL_API_KEY`); endpoint defaults for sampling (the official call sets none); transport retries on
   HTTP 5xx / timeouts (6 attempts, 240 s timeout). The calls are prefetched by 4 worker threads; the reply of the one
   call used is kept per case (`vlm_reply.json`). The reply is not deterministic; there is a single run, no re-asking.
2. Official `prepare_image` with BriaRMBG-1.4 (white background), then the body of `run_triposg`: `PartCrafterPipeline`
   with `image=[img]*num_parts`, `attention_kwargs={"num_parts": num_parts}`, generator seeded with 0; a part that
   fails to decode becomes the official 1-vertex dummy mesh and is not listed as a link.
3. Official exports (`part_XX.glb`, `object.glb`, `manifest.json`) plus `part_XX.obj` of every decoded part and the native
   manifest: links = decoded parts in the pipeline's normalized frame (bounds -1.005..1.005, `metric_size_source: none`,
   `mesh_scale` [1,1,1]), no joints, no mass or density, `physics_contract: false`.

## Tracks and lanes

Registered 200-input subset: 8 lanes (subset index % 8, `prepare_subset200.py`). Remaining 1,800 inputs: 44 lanes
(frozen-order index % 44, `prepare_remaining1800.py`), same code, checkpoints, environment, generation parameters and
part-count model; the 200 finished subset cases serve as its smoke test. Amendment 002 (`oom_rerun.py`): a case whose
only recorded failure is a CUDA out-of-memory caused by GPU co-tenancy is re-executed once on an exclusive GPU (the
original attempt directory is kept under `interrupted/`); no other failed case is touched.

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs AFFORDCRAFT_MODELS=/abs/models
export PARTCRAFTER_HOME=/abs/PartCrafter PARTCRAFTER_CKPT=/abs/partcrafter-69a0ffc1 PARTCRAFTER_RMBG=/abs/rmbg-1.4
export PARTCRAFTER_PYTHON=/abs/envs/partcrafter/bin/python EVAL_API_BASE=https://<endpoint>/v1 EVAL_API_KEY=<key>
R=$AFFORDCRAFT_RUNS/external/partcrafter/subset200
$PARTCRAFTER_PYTHON baselines/partcrafter/prepare_subset200.py --root $R --lanes 8   # configuration.json, lanes, code/
PC_ROOT=$R PC_LANES="0 1" PC_GPU=0 PC_TAG=g0 bash $R/code/pc_lane.sh              # lanes run sequentially per GPU
$PARTCRAFTER_PYTHON baselines/partcrafter/oom_rerun.py --revision-root $R --gpu 0 --dry-run   # lists co-tenancy OOM cases
PARTCRAFTER_SUBSET_RUN=$R $PARTCRAFTER_PYTHON baselines/partcrafter/prepare_remaining1800.py \
  --root $AFFORDCRAFT_RUNS/external/partcrafter/remaining1800 --lanes 44
```

`prepare_*.py` copy the wrappers into `<run root>/code/`; the lane script and `oom_rerun.py` run them from there.

Outputs per case: `part_XX.glb`, `part_XX.obj`, `object.glb`, `manifest.json`, `processed_input.png`, `vlm_reply.json`,
`timing.json`, `result.json`, `native_manifest.json`, `case.log`, 8-view renders; per lane `execution_receipt.json`.

Dependencies: a byte copy of the sm_120 geometry environment (torch 2.7.0+cu128, diffusers 0.32.2, transformers 4.50.0,
accelerate 1.5.2, trimesh 4.12.2, pyrender 0.1.45 with EGL); no package installed or changed.

## Compatibility changes (official tree untouched)

- Weights loaded from local revision-pinned directories instead of `snapshot_download` (same revisions, sha256-checked).
- The official provider needs the google-genai SDK and a Gemini key; `api_part_count_provider.py` (outside the
  official tree, registered at run time as `PROVIDERS["openai_compatible"]`) sends the same prompt and content to the
  OpenAI-compatible endpoint and applies the official parser and retry rule. The official default model,
  `gemini-3-flash-preview`, was not reachable; `gemini-3.8-flash` is the closest Gemini 3 flash model the endpoint served.
- The body of `run_triposg` is inlined (identical arguments) so that background removal and generation are timed
  separately; models are loaded once per lane process.

## Changes from the executed version

- API endpoint and key: read from `EVAL_API_BASE` / `EVAL_API_KEY` (the executed lane script exported them from
  root-only files into `OPENAI_BASE_URL` / `OPENAI_API_KEY`). Renamed to neutral names
  (imports updated): `api_part_count_provider.py`, provider name `openai_compatible` (recorded in `result.json` and
  `vlm_reply.json`), variables `PARTCRAFTER_API_MODEL`, `PARTCRAFTER_API_TRANSPORT_RETRIES`, `PARTCRAFTER_API_TIMEOUT_S`;
  texts say "API endpoint".
- Locations, interpreters and run roots from environment variables; the recorded node / smoke-test / download notes
  in the configuration were reduced to their technical content; mirror scripts removed from the copied code list.
- `prepare_subset200.py` / `prepare_remaining1800.py` = the executed preparation scripts of the two runs renamed; the
  remaining-run script no longer records the endpoint host, and it identifies its parent run by method id instead of
  its internal run name. Docstrings: dates and internal run names removed. Formatting with black (AST unchanged).
- Not published: mirror/pull scripts, lane launchers and chains, per-run summary.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| the run's part-count provider (renamed) | `api_part_count_provider.py` | `ff471ef45170b935c03be3634ab5ddbfd4fac061fa3b7ed439e0e0bbc7e1611b` |
| `oom_rerun.py` | `oom_rerun.py` | `ed1bdc9082b45872144988dd00eafa9976c2dc5d773152cf48af1d3aa3a69832` |
| `pc_lane.sh` | `pc_lane.sh` | `4a96ea5a6a7cf8cecc339978840ff55c4f68efb9eed7e3e1915ab18c533ec231` |
| `prepare_073_pc.py` (renamed) | `prepare_remaining1800.py` | `a55feb31db311c5ba94626f32a4322591e59c6a40f551758ff5d40d0df6ace0c` |
| `prepare_054.py` (renamed) | `prepare_subset200.py` | `ea84d7a38bc1bb2a2d4c0315d84c3acdd71d7e3ecdbf0e067ecb8f1548971701` |
| `run_partcrafter_lane.py` | `run_partcrafter_lane.py` | `09e8d9905b60a8304dc906b7c899d5649a69264fb88a85e32a1314dabdde42d1` |
| `write_receipt.py` | `write_receipt.py` | `8bf91edc22e0251dbb6f31ecb6b6bd667e3e3b08c85d549aa6e4b706d3618712` |
