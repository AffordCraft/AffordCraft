# External baselines

Code that ran the seven external image-to-asset routes of the paper's comparison: Section 4.2 and the main comparison
table (`tab:main`), the resource table (`tab:resource-metrics`), and the appendix "Controlled external-method and module
studies". Every folder holds the run wrapper(s) that call the official code on a list of inputs, the exporter that
writes the native manifest, the compatibility changes to the official code (as patch scripts or documented edits) and,
for the routes that call a commercial model through an API, the prompts and the harness. Official repositories are
referenced, never copied. API cost accounting is in `evaluation/api_cost/`.

## Protocol (full RGB)

Every route receives the whole photograph, with no reference box or mask. Articulate-Anything and the GPT-6 Astra agent
also receive the task instruction, which names the category; PAct receives an automatic part mask (Grounding-DINO +
SAM, `pact/mask_adapter.py`), because its released loader reads the number of parts from a part-label mask. For every
input a run writes `lane-<k>/cases/<source_id>/native_manifest.json` (schema `affordcraft.external_native_manifest.v1`,
contract `evaluation/gate/manifest_contract.md`), also for inputs without an asset (`asset_emitted: false`, the stage
reached and the failure reason), so no input leaves the denominator. The physical gate (`evaluation/gate/`) consumes
only these manifests and the meshes they point to; aggregation is in `evaluation/aggregate/`.

| Method | Official code (commit) | Checkpoint / model | Inputs given | Track | Entry point | Compatibility changes |
|---|---|---|---|---|---|---|
| PhysX-Anything | https://github.com/ziangcao0312/PhysX-Anything (`e221826e`) | official weights, HF `Caoza/PhysX-Anything` snapshot `fdfdf47b` (fine-tuned Qwen2.5-VL-7B + TRELLIS-based decoder) | photograph | 2,000 (registered 200, then the remaining 1,800) | `physx_anything/run_physx_anything_lane.sh` | SDPA instead of flash-attention-2; xformers cutlass kernel with FA3 dispatch disabled (sm_120); `torch.no_grad()` around `run_control` (OOM on 32 GB); sm_120 builds; official scripts run unmodified |
| PhysX-Omni | official PhysX-Omni release (arXiv:2605.21572); see below | fine-tuned VLM snapshot `765cd275` (representation stage); `microsoft/TRELLIS-image-large` @ `25e0d31f` (geometry stage) | photograph | 2,000 | `physx_omni/omni_representation_runner.py` (representation), `physx_omni/run_geometry_lane.sh` (geometry + export) | xformers cutlass kernel, FA3 disabled; offline DINOv2; write-isolated export copy; cases that ran out of memory on 32 GB recomputed on 80 / 96 GB GPUs |
| PAct | official PAct release (arXiv:2602.14965), commit `955d4ef3` | PAct preview checkpoint (`8c6b56c7`) | photograph + automatic part mask | 2,000 | `pact/run_lane.sh`, `pact/run_lane_sm120.sh` | SDPA attention (flash-attn import shim); local DINOv2; float32 EXR reader; mip-splatting rasterizer wheel; official `infer_imgs.py` unmodified |
| PartCrafter | https://github.com/wgsxm/PartCrafter (`3d773bf0`) | `wgsxm/PartCrafter` @ `69a0ffc1`, `briaai/RMBG-1.4` @ `2ceba5a5`; part count from `gemini-3.8-flash` | photograph | 2,000 | `partcrafter/pc_lane.sh` | provider for `--part_suggest` that calls an OpenAI-compatible endpoint (official prompt and parser); `run_triposg` inlined to time its stages |
| TRELLIS.2 | https://github.com/microsoft/TRELLIS.2 (`75fbf018`) | `microsoft/TRELLIS.2-4B` @ `af44b45f` (+ DINOv3 ViT-L/16 @ `ea8dc286`, RMBG-2.0 @ `5df4c9c7`) | photograph | 2,000 | `trellis2/t2_lane.sh` | torch 2.7 / sm_120 builds, flash-attn 2.7.4.post1, triton 3.3.1; `empty_cache` before export; fresh process after an OOM; offline Hugging Face cache |
| Articulate-Anything | https://github.com/vlongle/articulate-anything (`b311003f`) | PartNet-Mobility library as preprocessed by the official code (93 of 129 missing renders completed with the official call); `gemini-2.5-flash` | photograph + task instruction (category) | registered 200 | `articulate_anything/run_lane.sh` | `gemini-2.5-flash` instead of the default `gemini-1.5-flash`, which the API endpoint did not serve; OpenAI-compatible client; reply-format normalizer; SAPIEN 3 render port (`patch_official_tree.py`) |
| GPT-6 Astra agent | harness in `gpt_agent/`, single-image setting of https://github.com/lingxiao-guo/GPT6-real2sim | `gpt-6-astra`, reasoning `max`; AffordCraft catalog and DINOv2 index | photograph + task instruction | registered 200 | `gpt_agent/run.sh` | own harness: 32 tool rounds, 128K cumulative output tokens, 3 h per case; images returned by tools are sent as a user-role message |

## Official sources as recorded in the run configurations

Each run wrote the source path, commit and file hashes of its official checkout into `configuration.json` before any
lane ran. Recorded values:

| Method | Repository (recorded remote) | Commit (recorded) | Checkpoint (recorded) |
|---|---|---|---|
| PhysX-Anything | `https://github.com/ziangcao0312/PhysX-Anything.git` | `e221826e6176d940905126d1894f9c1c933b70a8` (tree clean) | HF `Caoza/PhysX-Anything` snapshot `fdfdf47b`; every weight file hashed |
| PhysX-Omni | official release (arXiv:2605.21572) | not recorded; the hashes of the official scripts and of every checkpoint file are | fine-tuned VLM snapshot `765cd275`; `microsoft/TRELLIS-image-large` revision `25e0d31f`; hashes in the lane configurations |
| PAct | not recorded (official release of arXiv:2602.14965) | `955d4ef3c5a973695e5fc0a2b22667eee42ffe37` (tree clean) | preview checkpoint `pact-PAct-8c6b56c7` (cc-by-nc-sa-4.0), every file hashed |
| PartCrafter | not recorded (https://github.com/wgsxm/PartCrafter, the code of the recorded checkpoint) | `3d773bf02fad51c7ab31a5615573fec93b287b30` (tree clean) | `wgsxm/PartCrafter` @ `69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f`; `briaai/RMBG-1.4` @ `2ceba5a5efaec153162aedea169f76caf9b46cf8` |
| TRELLIS.2 | not recorded (https://github.com/microsoft/TRELLIS.2, as cited in the paper) | `75fbf0183001ed9876c8dbb35de6b68552ee08bd`; eigen submodule `21e4582d1739107337a03460c81412981130373e` | `microsoft/TRELLIS.2-4B` @ `af44b45f2e35a493886929c6d786e563ec68364d`; `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc2863c51be0a264bab82070e3e8836b02d51`; `briaai/RMBG-2.0` @ `5df4c9c76d8170882c34f6986e848ee07fd0ba43` |
| Articulate-Anything | `https://github.com/vlongle/articulate-anything.git` | `b311003f8265c186314618a353089d0d86260a8f` (tree clean) | no checkpoint; PartNet-Mobility library index files and CLIP ViT-B/32 weights hashed |
| GPT-6 Astra agent | own harness (hashes of its files, tool schemas and system prompt recorded) | n/a | `gpt-6-astra` through the API; catalog, index and frozen AffordCraft modules hashed |

## Environment variables

The API endpoint and key come only from `EVAL_API_BASE` / `EVAL_API_KEY`: an OpenAI-compatible endpoint (PartCrafter,
Articulate-Anything, agent). Official checkouts and checkpoints come from `<METHOD>_HOME` / `<METHOD>_CKPT`; each
folder's README lists the rest. Shared ones:

| variable | meaning |
|---|---|
| `AFFORDCRAFT_PROJECT_ROOT` | root the image paths of the input manifest are relative to; asset sources of the catalog (agent) |
| `AFFORDCRAFT_RUNS` | run outputs; `inputs/input_manifest_2000.jsonl` (frozen 2,000-input manifest, sha256 `139981fe...`), `inputs/subset_200_inputs.json` (registered subset, sha256 `db043e72...`) |
| `AFFORDCRAFT_MODELS` | default parent directory of model files |
| `<METHOD>_PYTHON` | interpreter of a method's environment (`PHYSX_ANYTHING_VLM_PYTHON` / `PHYSX_ANYTHING_GEOM_PYTHON`, `PHYSX_OMNI_GEO_PYTHON`, `PACT_PYTHON`, `PARTCRAFTER_PYTHON`, `TRELLIS2_PYTHON`, `AA_PYTHON`, `AFFORDCRAFT_RUNTIME_PYTHON` for the agent) |

Use absolute paths: several drivers change directory. Run roots default to `$AFFORDCRAFT_RUNS/external/<method>/<track>`
with the names of `evaluation/aggregate/methods.json` (for example `physx-anything/subset200`,
`physx-anything/remaining1800`, `pact/remaining1800`). All drivers expect one GPU per lane (`CUDA_VISIBLE_DEVICES`);
the runs used RTX 5090 (32 GB, sm_120) GPUs unless a README says otherwise.

## Input files

The wrappers read two input files and check their sha256 before any lane is written, as in the executed runs.
`scripts/prepare_inputs.py baselines` writes both in the same format from the manifests of `docs/running.md`, step 2.
The checks default to the hashes of the paper's files below; the image paths inside the manifest depend on where the
images lie, so a local copy has other hashes: set `AFFORDCRAFT_MANIFEST_SHA256` and `AFFORDCRAFT_SUBSET_SHA256` to the
hashes that `prepare_inputs.py` prints. With the images at the paper's locations the regenerated manifest is
byte-identical to the paper's.

- `$AFFORDCRAFT_RUNS/inputs/input_manifest_2000.jsonl`: one JSON object per line in frozen order, with `source_id`,
  `requested_category` and `image` (`path` relative to `AFFORDCRAFT_PROJECT_ROOT`, `sha256`); sha256
  `139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df`.
- `$AFFORDCRAFT_RUNS/inputs/subset_200_inputs.json`: `{"inputs": [{"input_id", "image": {"sha256"}, "target_noun",
  "instruction", ...}], "order", "no_reference_boxes_or_masks"}` in registered order; sha256
  `db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf`.

The remaining-1,800 tracks use the frozen manifest minus the subset, in frozen order. The input ids, image hashes,
categories and instructions of both files are those listed in `data/inputs/`.

## Conventions shared by all runs

- A run writes its `configuration.json` (official commit, file and weight hashes, settings, compatibility changes)
  before any lane starts; lanes refuse to run without it. Case records are written once (`open(..., 'x')`).
- Inputs are split into lanes by a fixed rule decided before the run (index modulo the number of lanes). Every input
  image is re-hashed before use.
- A failed case stays failed; nothing is re-selected or re-sampled. Exceptions are infrastructure failures, recorded as
  amendments: a case interrupted by a killed process is recomputed from scratch (its directory is kept under
  `interrupted/`), a call refused by the API endpoint for lack of credit is rerun once credit is restored, and a CUDA
  out-of-memory caused by a shared or too small GPU is re-executed once on an exclusive or larger GPU
  (`partcrafter/oom_rerun.py`, `trellis2/oom_rerun.py`, `physx_omni/run_geometry_lane_attempt.sh`).

## Not included

- Fleet infrastructure: node setup, lane dispatch and claiming, mirroring and syncing between machines, pull loops,
  watchdogs, keepers and stale-claim sweepers, terminal-multiplexer launchers, status pollers, one-shot patchers of
  running jobs, per-run summaries (superseded by `evaluation/aggregate/`) and notes.
- Third-party code: official repositories are referenced by URL and commit; only our wrappers and patches are here.

Provenance tables (original file name, published path, sha256 of the original) are in each folder's README.
