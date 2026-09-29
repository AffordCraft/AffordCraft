# gate/ -- physical gate for external outputs

Purpose: score one external method's outputs with AffordCraft's frozen physical gate in the native and the adapted
condition (protocol, statuses and amendments: `../README.md`). Supports Table 1 (Native / Adapted / Artic. columns) and
the external-method appendix.

## Entry points

- `external_gate_pipeline.py --manifests LIST --method-id ID --out DIR --cache DIR [--grounding JSON] [--gpu N]
  [--phases native,adapted] [--lane NAME]`: gates every `native_manifest.json` listed in LIST (one path per line).
  Example: `../README.md`, "Running the gate on one case".
- `usd_post_edit.py SPEC.json`: pxr helper called by the pipeline (runtime interpreter) to apply AffordCraft's
  construct_asset post-edits to the authored native stage (body transforms, joint axis tokens, continuous and spherical
  joints, articulation self-collision flag, support floor).
- `manifest_contract.md`: the native-manifest schema every method wrapper writes (`baselines/`).

Inputs: native manifests and the files they point to; the grounding table of the main campaign; a geometry cache
directory for the builder. Outputs: `DIR/cases/<source_id>/case_result.json` and per-condition folders,
`DIR/configuration.json`, `DIR/configuration_amendment_001_materialize_slots.json`, `DIR/lanes/<lane>/` (physics
service logs and process outcomes), `DIR/<lane>.progress.jsonl`.

Dependencies: this repository (`affordcraft/`, `scripts/physics_worker.py`, `scripts/materialize_asset.py`); the runtime
interpreter (`AFFORDCRAFT_RUNTIME_PYTHON`: numpy, scipy, trimesh, CoACD, mujoco, usd-core) and Isaac Sim 5.1
(`AFFORDCRAFT_ISAAC_PYTHON`) for the physics worker. Environment: `AFFORDCRAFT_PROJECT_ROOT`, `AFFORDCRAFT_RUNS`, `GATE_MAT_SLOTS` (concurrent builders per machine, default 3 or the value in
`$GATE_SLOTS_DIR/node_slots`), `GATE_SLOTS_DIR` (lock directory shared by all gate processes of a machine, default
`<tmp>/affordcraft_gate_slots`), `GATE_INFRA_RETRIES` (recorded in amendment 001; default 4).

## Changes from the executed version

- The frozen method code is imported from this repository (`affordcraft/` on `sys.path`, `scripts/physics_worker.py`
  and `scripts/materialize_asset.py` relative to the repository root) instead of the campaign's frozen code directory;
  `configuration.json` therefore locks `scripts/...` and `affordcraft/...` paths.
- Interpreters from `affordcraft/paths.py` (`AFFORDCRAFT_RUNTIME_PYTHON`, `AFFORDCRAFT_ISAAC_PYTHON`); the project root
  from `AFFORDCRAFT_PROJECT_ROOT`.
- New option `--grounding` (the executed version read the table from a fixed path).
- The builder semaphore directory and its per-machine default file moved from a fixed container path to
  `GATE_SLOTS_DIR` (default in the system temporary directory).
- Comments and docstrings: dates, machine and provider names removed; `usd_post_edit.py` docstring states that the
  pipeline runs it with the runtime interpreter (as the executed version did).
- Formatting with black (AST unchanged apart from the lines above).

## Provenance

| Original file | Published path | sha256 of the original |
|---|---|---|
| external_gate_pipeline.py (with amendments 001-003) | evaluation/gate/external_gate_pipeline.py | 1175a3a019924210c6d42a09b5f97623e6e9a202ff6bb5d5887c502d4265e71c |
| usd_post_edit.py | evaluation/gate/usd_post_edit.py | a13e1c0d9e603598a570d4862923d1a54e42189a087bf0caa0d043cd025afa68 |
| external_native_manifest_contract.md | evaluation/gate/manifest_contract.md | 8e93c346d35962dcb8622980e806cf4ea882197a7b50de69ab7c0d72ac416578 |

Earlier pipeline versions recorded in the gate runs: before amendment 002 `520f48add899ecda2f7dee73ea3b7de1bbc6601e7e0b68f6baf9ea74012f871b`
(registered versions of the first gate runs: `498baba5...`, `99d454f6...`, `3529f0b4...`), after amendment 002
`e0a98e75c5b9f5d825234f30bc7cf336e99c435df0629668d37b02d9cf9b53a6`, after amendment 003 `1175a3a0...` (published).
`manifest_contract.md` differs from the original only in its first line (no date), in the sentence on absolute paths
(the original named the shared directory of the executed runs) and in the added section "How the gate reads the
manifest".
