# Articulate-Anything

Official Articulate-Anything (https://github.com/vlongle/articulate-anything, commit
`b311003f8265c186314618a353089d0d86260a8f`), image modality, on the registered 200-input subset: row
"Articulate-Anything" of the main comparison table (`tab:main`, Section 4.2), the resource table
(`tab:resource-metrics`) and the appendix "Controlled external-method and module studies". It retrieves a
PartNet-Mobility object and has a vision-language model write its link placement and joints as code; the result is a
URDF without mass or inertia, so only the adapted gate condition applies.

## Inputs and configuration (`run_case.py`)

Prompt = the whole photograph; `additional_prompt` = the requested category (the task instruction names it), with the
official library alias `StorageFurniture -> Cabinet`; the official category selector then CLIP-matches the name against
the library categories instead of running its video object detector. Configuration = the official `conf/config.yaml`
plus the switches of the official image-modality entry points (`gradio_app.py`, `examples/articulate_image.ipynb`):
`modality=image`, `joint_actor.mode=image`, `joint_actor.use_cotracker=false`, `joint_actor.targetted_affordance=false`,
`joint_critic.mode=image`, `joint_critic.use_cotracker=false`; everything else at the official defaults
(`actor_critic.max_iter=1`, one seed, `category_selector.topk=1`, hierarchical object selector with 10 images, simulator
defaults, generation temperature 0.5). The stages run in the official order (mesh retrieval, link articulation,
affordance extraction, joint articulation) and stop at the first failed stage. A case is limited to 3,600 s; a case
without a joint-stage URDF with at least one mesh link is a failure and stays one.

VLM: `gemini-2.5-flash` (the official default `gemini-1.5-flash-latest` was not served by the API endpoint) through an
OpenAI-compatible endpoint (`EVAL_API_BASE` / `EVAL_API_KEY`) with `api_client.py`: the message layout of the official GPT wrapper
(system instruction + one user turn interleaving text and JPEG images, detail low), temperature 0.5, up to 5 attempts
with backoff on transient errors; an empty reply (thinking budget used up) is retried with `reasoning_effort=low` and
`max_tokens=32768`. Every call is logged to the case's `vlm_transcript.jsonl` (the key never is).
`aa_response_normalizer.py` rewrites replies of the JSON-reply agents that fall outside the official JSON format (prose
ending in a boxed answer, trailing commas, a bare image index) into the canonical form the unchanged official parsers
read; code replies are never touched and every rewrite is saved next to the reply.

## Compatibility changes

`patch_official_tree.py` re-creates a copy of the official tree and applies exact-string patches (each must match once;
the resulting unified diff is written to `patches/official_tree.diff`); the official checkout is never modified:

- `articulate_anything/utils/prompt_utils.py`: the Google / Anthropic / OpenAI SDK imports become optional and
  `setup_vlm_model()` returns the OpenAI-compatible client when `AA_OPENAI_COMPAT=1`;
- `articulate_anything/agent/agent.py`: agent name and output directory passed to the client (transcript routing);
  replies go through the normalizer before the official `parse_response()`;
- `articulate_anything/utils/utils.py` (`make_cmd`): the renderer runs with the current interpreter instead of
  `conda run -n articulate-anything`, the render script is replaced by `aa_sapien3_simulate.py`, and hydra writes into the
  case directory.

`aa_sapien3_simulate.py` ports the official `sapien_simulate.py` to SAPIEN 3.0.3, because the pinned SAPIEN 2.2.2
segfaults in its Vulkan renderer on the RTX 5090 / driver 595.71: same hydra configuration, camera (position [3, 1.5, 2],
look-at [0, 0, 0.8], fovy 35 degrees, 640x480), lighting, plain floor, raise distance, stationary and moving logic and
output files; the articulation is loaded with a fixed root and zero gravity (SAPIEN 3 has no kinematic loader). Its
shading differs slightly from the library renders (mean absolute pixel difference 26/255 on one object); the selected
object is re-rendered per case, so the link critic compares renders of one renderer.

Library isolation: each case works on a private copy of the retrieved object (the official code writes renders and
link files into the object directory); retrieval reads the shared library read-only.

## Library completion and attempts

The first attempt stopped after 26 cases when the API balance ran out and showed two harness faults: 129 library
objects had no readable `robot_frontview.png` (the official retrieval then crashed), and some `gemini-2.5-flash`
replies fell outside the JSON format. Before the second attempt the missing renders were produced with the official
preprocessing call `render_partnet_obj(obj, gpu, cfg, "stationary")` through the SAPIEN 3 port (`list_library_gaps.py`,
`render_library_gaps.py` / `.sh`, `check_library_renders.py`): 93 of the 129 objects render, the rest stay without a
render and are therefore excluded from retrieval, as in the official `get_candidate_objs()`. The normalizer was added, the
first-attempt cases were moved to `interrupted/`, and all 200 cases ran again with the same code, model, prompts, lanes
(subset index % 6) and limits.

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs EVAL_API_BASE=https://<endpoint>/v1 EVAL_API_KEY=<key>
export ARTICULATE_ANYTHING_HOME=/abs/articulate-anything ARTICULATE_ANYTHING_CKPT=/abs/partnet-mobility-v0
export AA_WORK=/abs/aa_run AA_BASE_ENV=/abs/envs/geom-sm120 AA_ENV=/abs/envs/aa AA_PYTHON=/abs/envs/aa/bin/python AA_GPU=0
bash baselines/articulate_anything/setup_env.sh                       # environment, logs/env_setup.log, pip freeze
mkdir -p $AA_WORK/code && cp baselines/articulate_anything/* $AA_WORK/code/
$AA_PYTHON $AA_WORK/code/patch_official_tree.py                         # $AA_WORK/src + patches/official_tree.diff
$AA_PYTHON $AA_WORK/code/list_library_gaps.py && bash $AA_WORK/code/render_library_gaps.sh 8   # ends with a readability check
$AA_PYTHON $AA_WORK/code/make_lane_manifests.py 6
$AA_PYTHON $AA_WORK/code/register_configuration.py 6                     # configuration.json before any lane
bash $AA_WORK/code/run_lane.sh 0 6                                      # one call per lane
```

If the readability check lists unreadable renders, run `check_library_renders.py` without `--check-only` (with the
environment of `render_library_gaps.sh`); it re-renders them once and excludes those that stay unreadable (the executed
check found none).

Outputs per case (`lane-<k>/cases/<source_id>/`): `input.json`, `config_used.yaml`, `stage_log.jsonl`,
`vlm_transcript.jsonl`, `library/dataset/<obj_id>/` (private object copy), `aa_out/` (official outputs: category and
object selection, link placement and critic, joint actor with `joint_pred.py`, `mobility.urdf` and videos),
`native/link_meshes/<link>.obj`, `timing.json`, `result.json`, `native_manifest.json` (`urdf_native_manifest.py`: one
combined OBJ per link with the URDF visual origins baked in, joint axes rotated into the parent frame, units at the
library scale, `metric_size_source: library`, `physics_contract: false`); per lane `execution_receipt.json`.

Dependencies (`setup_env.sh`): the sm_120 geometry environment (torch 2.7.0+cu128, sapien 3.0.3, pybullet 3.2.7,
trimesh 4.12.2) plus numpy 1.26.4 and opencv-python 4.11.0.86 (official pins), hydra-core 1.3.7, astor, markdown2,
GPUtil, seaborn, pandas, OpenAI CLIP (ViT-B/32 weights, sha256 in the URL), co-tracker at `5951295e` (imported only),
transforms3d, flow_vis.

## Changes from the executed version

- API endpoint and key from `EVAL_API_BASE` / `EVAL_API_KEY` (the executed scripts exported them from root-only files
  into `OPENAI_BASE_URL` / `OPENAI_API_KEY`); the inserted comment in the patched `prompt_utils.py` names the new
  variables. Renamed to neutral names (imports updated): `api_client.py` (class
  `ApiWrapper`, response `ApiResponse`), switch variable `AA_OPENAI_COMPAT`; texts say "API endpoint".
- Locations, interpreter, working directory, run root and GPU index from environment variables (`common.py`, scripts);
  the GPU index given to the official render call is `AA_GPU` (the executed scripts hard-coded the index of the GPU they
  used). The run id is the run root's name (the executed `common.py` hard-coded the same name).
- `patch_official_tree.py`: the fresh copy of the official tree is made with `shutil.copytree` (same exclusions) instead
  of an archive-mode mirror command; `run_lane.sh` copies the lane source tree with `cp -a`.
- Renamed: `list_library_gaps.py`, `check_library_renders.py`, `render_library_gaps.sh`, `setup_env.sh` (their executed
  names carried machine names); `setup_env.sh` writes `logs/env_setup.log` and `logs/aa5090_pip_freeze.txt`, the names
  `register_configuration.py` hashes, and uses the default package index and the default CLIP cache.
- Recorded texts: node and cloud-provider names removed. Docstrings: dates and internal run names removed. Formatting
  with black (AST unchanged).
- Not published: the second-attempt bookkeeping script (it moved the first-attempt cases to `interrupted/` and wrote
  the amendment record described above), smoke-test and status scripts.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `aa_response_normalizer.py` | `aa_response_normalizer.py` | `7530a86f215fd65e22c207cd2c751358749114e1ecc56f6e26a44b40f34b9ffb` |
| `aa_sapien3_simulate.py` | `aa_sapien3_simulate.py` | `2ea7033768befb93871adcf36c7053572f8542c324df5cd714eeb82b304cc8ed` |
| the run's API client (renamed) | `api_client.py` | `b8a2978fbe1d38b542339a966f8a796199bda05fe167aa61fa664c54a200f335` |
| `aa_library_fix.py` (renamed) | `check_library_renders.py` | `66cc5dc71bbe4b1423b7761ea1048dba007686ae1080220f9623e6fa295bcef8` |
| `common.py` | `common.py` | `eadae75f6e4ca0a9ee1156309d15c11cc9e9e07b4edca167e091dc46531d80bc` |
| `lane_receipt.py` | `lane_receipt.py` | `1fce9e7c7d902a06966daf5c155778237e04d87e8c8a54e48620f9785fad312e` |
| `aa_gaps_s31.py` (renamed) | `list_library_gaps.py` | `06772bcb9cabb03025a1186e64d53b708e99272b591aaa06860d1e49256f5972` |
| `make_lane_manifests.py` | `make_lane_manifests.py` | `cca3484b7d6ab4c6a9dc087e9c5e6641da7a56d4be58770bc111a1f57c78d17a` |
| `patch_official_tree.py` (attempt-2 version (adds the agent.py normalizer hunk)) | `patch_official_tree.py` | `bb7034e28d295440d761c2c1bebf5f34b7bdd96d67e73d10fdccde87ae6613c0` |
| `patch_official_tree.py` (attempt-1 version, without the normalizer hunk) | `patch_official_tree.py` | `3452380e52a718d6656b161b433001256d1fc645410b313ce28a3cf03b55e1a5` |
| `register_configuration.py` | `register_configuration.py` | `0e7e5fa9eb980260404ed4870bac2c59c446443dcc7b038647062f4b84a51a01` |
| `render_library_gaps.py` | `render_library_gaps.py` | `53e39b1a3359b32a74f295e2836d9adca64e1a902a33556bf873dc0a2f0a633b` |
| `aa_render_gaps_s31.sh` (renamed) | `render_library_gaps.sh` | `92e97aeb4368ce15e948bc3bd14d0283b6f352cbc5e95ba44774258e77287e39` |
| `run_case.py` | `run_case.py` | `7953a68525af8ffe652fb7f6b51f65e63a62f182ad89ac30919661a3d28169be` |
| `run_lane.py` | `run_lane.py` | `aff89bd351f4d81e0c9b61aaa37077c561adb54fdeddc86b11f160bb6cfd6dc0` |
| `run_lane.sh` | `run_lane.sh` | `8185845b54ec73f1678b529d1221ff5074bc56a9a3b9f0c61aecbccce846313a` |
| `aa_env_setup_s31.sh` (attempt-2 environment rebuild; renamed) | `setup_env.sh` | `a00a9e48224e636829143cc1b923cd648c79a68952a0e263259994c0d877f091` |
| `urdf_native_manifest.py` | `urdf_native_manifest.py` | `616fef67257e16ff65a55d2b2aad1c78e0003a7b5cc7414df869bafc550db63a` |
