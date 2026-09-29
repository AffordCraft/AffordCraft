# GPT-6 Astra agent

A general-agent baseline: `gpt-6-astra` (reasoning effort `max`) calls modeling tools to assemble an interactive asset
from one photograph and the task instruction. Row "GPT-6 Astra agent" of the main comparison table (`tab:main`,
Section 4.2), the resource table (`tab:resource-metrics`) and the appendix "Controlled external-method and module
studies"; registered 200-input subset. The configuration adapts the single-image agent baseline of
https://github.com/lingxiao-guo/GPT6-real2sim and does not reproduce its multi-view episode replay.

## Protocol (`common.py`, `agent_loop.py`)

- Input per case: the photograph (original bytes, hash-verified, `detail: high`) and the instruction
  `prompts/instruction_template.txt` ("Construct an interactive simulation asset for the {requested_category} visible in
  this image. Preserve its mechanism and natural support conditions."). No box, mask, annotation, pipeline output or
  evaluation answer. One fresh conversation per case; cases share only read-only catalog caches.
- System prompt `prompts/system_prompt.txt`; tools `prompts/tools.json` (`search_catalog`, `inspect_entry`,
  `make_primitive`, `render_assembly`, `submit_asset`), both extracted verbatim from `agent_loop.py`.
- Budgets, checked before every call and after every response: 32 tool rounds, 128,000 cumulative completion tokens
  (reasoning tokens included, as reported by the API), 3 hours of wall-clock per case; exceeding one ends the case with
  `asset_emitted: false` and `failure_reason: budget_exhausted:<which>`. Per call: `max_completion_tokens` up to 16,000
  (requested; the endpoint did not truncate, so the harness enforces the cumulative budget), `tool_choice: required`,
  temperature at the provider default.
- Transport: an OpenAI-compatible endpoint (`EVAL_API_BASE` / `EVAL_API_KEY`), `/chat/completions` streamed as
  server-sent events (non-streamed calls longer than 300 s were cut by the endpoint's gateway), 600 s inactivity and
  1,800 s total timeout per call, up to 6 retries with exponential backoff on HTTP 429 / 5xx / network errors; any other
  HTTP 4xx ends the case with `api_error` (for a context-length error, up to 3 retries with fewer images in context).
- Images returned by tools are appended as one user-role message right after the tool results (the endpoint dropped
  image parts inside tool messages); tool images older than 3 rounds are replaced by text stubs, the photograph is always
  kept; previews are JPEG with a maximum side of 448 px, assembly renders 640x480. A reply without tool calls receives
  "No tool was called. Continue by calling a tool; only submit_asset ends the task." and counts as a round.
- Catalog access: `search_catalog` runs the frozen AffordCraft `CatalogIndex.retrieve` (DINOv2-large CLS index
  `visual-large-001`, 12 category-preferred slots, then category-blind neighbours) on the photograph or a crop the agent
  chooses; `inspect_entry` and imports read the catalog sources through the frozen `source_parser`. Both modules come
  from this repository's `affordcraft/` package.
- Export (`geometry.py`): the accepted spec is materialized as baked OBJ files per link (uniform scale, part pose and
  joint rotations applied; link frames world-aligned at rest with the origin at the joint pivot), a URDF (visual and
  collision mesh, convex-hull inertials) and an MJCF (mesh geoms with density or mass, hinge and slide joints, a
  freejoint root when `support` is `free_standing`), and `native_manifest.json` (`units: m`,
  `metric_size_source: method`).

## Run history

Lanes: case index % 16, up to 16 cases in flight (a semaphore; reduced to 8 after 8 or more HTTP 429 answers within 5
minutes; `control.json` can change it). The harness was amended once at a restart shortly after the launch
(`amendment_001.diff`): catalog entries whose URDF has empty helper links carrying movable joints became inspectable and
importable link by link. After 18 cases had finished, the API endpoint refused the calls of the other 182 cases with
HTTP 403 (no credit); a refused call is an infrastructure outcome, so once credit was restored those 182 cases were moved
to `interrupted/` and rerun with the same amended harness, model and protocol at 32 cases in flight.

## Usage

```bash
export AFFORDCRAFT_PROJECT_ROOT=/abs/workspace AFFORDCRAFT_RUNS=/abs/runs AFFORDCRAFT_MODELS=/abs/models
export AFFORDCRAFT_CATALOG=/abs/workspace/catalog/catalog.json AFFORDCRAFT_INDEX=/abs/workspace/index/visual-large-001
export AFFORDCRAFT_DINOV2=/abs/models/dinov2 AFFORDCRAFT_RUNTIME_PYTHON=/abs/envs/runtime/bin/python
export EVAL_API_BASE=https://<endpoint>/v1 EVAL_API_KEY=<key> AGENT_HOME=/abs/runs/external/gpt6-astra-agent
$AFFORDCRAFT_RUNTIME_PYTHON baselines/gpt_agent/offline_test.py /abs/scratch     # tools end-to-end with a scripted model
bash baselines/gpt_agent/run.sh                                                  # registers configuration.json, runs all lanes
$AFFORDCRAFT_RUNTIME_PYTHON baselines/gpt_agent/status.py $AFFORDCRAFT_RUNS/external/gpt6-astra-agent/subset200
```

Outputs: `configuration.json` (model, budgets, protocol, tool schemas and hashes, system prompt, harness hashes,
catalog identity) and a copy of the harness in the run root; per case `lane-<k>/cases/<source_id>/`: `transcript.jsonl`
(every request, response, tool call and result; images by hash), `parts/`, `renders/`, `spec.json`, `native/`
(`assembly.json`, OBJ, URDF, MJCF), `timing.json`, `result.json`, `native_manifest.json`; `events.jsonl`,
`progress.json`.

Dependencies: the runtime environment of the AffordCraft pipeline (torch, transformers for the DINOv2 encoder, trimesh,
scipy, requests, Pillow) plus pyrender with EGL for the offscreen renderer (matplotlib fallback).

## Changes from the executed version

- API endpoint and key from `EVAL_API_BASE` / `EVAL_API_KEY` (the executed client read root-only files).
- Locations from environment variables (`common.py`); the frozen `catalog.py` and `source_parser.py` are loaded from this
  repository's `affordcraft/` package. `source_parser.py` is identical to the executed one; `catalog.py` is the later
  frozen version, which adds a CLIP image encoder and a `category=None` branch in `CatalogIndex.retrieve`, neither of
  which the harness reaches (it always passes a category string).
- Catalog URDF and source-directory paths are resolved against `AFFORDCRAFT_PROJECT_ROOT` like the other catalog paths
  (the executed harness used them as given, which is the same for absolute paths).
- `run.sh` = the executed launcher reduced to the command it ran (without the terminal-multiplexer session).
- The configuration's description of a tool-free reply says "user-role message". Formatting with black (AST
  unchanged). `amendment_001.diff` compares the executed files; its line numbers refer to them.
- Not published: the one-shot patcher that produced the amended harness (its diff is `amendment_001.diff`), the resume
  bookkeeping script (described above) and a streaming patcher already contained in the registered harness.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `agent_loop.py` (amended harness) | `agent_loop.py` | `e21bef91ca85a7b1ac1d389f24a33200155374c327ca415cb1beec0fdabe4a5e` |
| `agent_loop.py` (registered harness; see amendment_001.diff) | `agent_loop.py` | `d4eb6b4f71b59df7bb61b91c3e1212aa240f4bd02e28e6b3f69d67a58195234f` |
| `patch_amend001.py` (one-shot patcher that produced the amended harness; not published) | `amendment_001.diff` | `f26c1614da76f1c33390836f17eb1d7dbdd93f1bf8a04e8e38917f83e6ab32da` |
| `common.py` | `common.py` | `97cf6e1ab1f1ecefb26b7e9b1d5f28b754431fdc35fd12eaf6171cc6648fca58` |
| `geometry.py` (amended harness) | `geometry.py` | `08eb88d095d1dc3c1f342fc7ec848f83c72afed77f8c9fb305330c9c2491a1a8` |
| `geometry.py` (registered harness; see amendment_001.diff) | `geometry.py` | `ccd9a0382af995aeb4977053d78394114ab72cc4a857db10a8857fb7e9308ef8` |
| `offline_test.py` | `offline_test.py` | `d7bfca7c7b1ea63eb340f8f9595c43be2c5e44486285dce79e6fd5424f583eae` |
| `agent_loop.py` (per-case instruction (CaseRunner.initial_messages)) | `prompts/instruction_template.txt` | `e21bef91ca85a7b1ac1d389f24a33200155374c327ca415cb1beec0fdabe4a5e` |
| `agent_loop.py` (SYSTEM_PROMPT, extracted verbatim) | `prompts/system_prompt.txt` | `e21bef91ca85a7b1ac1d389f24a33200155374c327ca415cb1beec0fdabe4a5e` |
| `agent_loop.py` (TOOLS, extracted verbatim) | `prompts/tools.json` | `e21bef91ca85a7b1ac1d389f24a33200155374c327ca415cb1beec0fdabe4a5e` |
| `render_worker.py` | `render_worker.py` | `7aaada9e91207db3c4bf611641783ca3a84966cef35e562b518ca532ce66006e` |
| `launch_full.sh` (launcher reduced to the command it ran) | `run.sh` | `810a65038887a46455d44b656a9de9a42f8613c4f99dde21d457334dcbfef759` |
| `run_lanes.py` (amended harness) | `run_lanes.py` | `e113a9b0ed726f36aa21a111f120ffe5068195ec5e0129b4ac59394c12106e28` |
| `run_lanes.py` (registered harness; see amendment_001.diff) | `run_lanes.py` | `ff2fa53ee90db0814b1b75ae2b04404447b9d22338a1f22be4a021fe95ea539c` |
| `status.py` | `status.py` | `324a41451030b54988f3a57680580e21991be0326de7424e1ca8222259934f84` |
