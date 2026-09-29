# aggregate/ -- verdict aggregation and finalization

Purpose: turn the method runs and gate runs of the seven external methods into the evidence file that Table 1, the
resource table and the time-to-pass figure read (`analysis/results/external/`). Per input of the target set (registered
200-input subset; the frozen 2,000-input manifest for the methods extended to it) the chain is method run ->
`native_manifest.json` -> gate verdicts (native, adapted).

## Entry points

```bash
# aggregate every method (layout in methods.json, run names relative to its runs_root)
python evaluation/aggregate/aggregate_external.py --method all --out aggregate_all.json \
    [--config methods.json] [--runs-root DIR] [--subset subset_200_inputs.json] [--manifest input_manifest_2000.jsonl] \
    [--finalization finalization_cutoff.json]
# terminal rule after failed big-memory re-gates, for one gate run (writes retry_exhausted.json markers)
python evaluation/aggregate/finalize.py retry-exhausted --gate-run $AFFORDCRAFT_RUNS/external/partcrafter/gates-remaining1800
# cutoff finalization: record + final aggregate
python evaluation/aggregate/finalize.py cutoff --out finalization_cutoff.json --final-aggregate aggregate_all.json
```

The paper's aggregate is `aggregate_external.py --finalization <the cutoff record>` after the retry-exhausted markers
were written (the executed aggregator loaded the record from its own directory). Only the Python standard library is
needed.

Inputs: `methods.json` (run layout: for every method the run directory, the gate run directories in precedence order and
a device note; the 2,000-input extension runs and their gates; the PhysX-Omni representation and geometry runs); method
runs `lane-*/cases/<source_id>/{result.json, timing.json, native_manifest.json}` and, where cases were re-executed on an
exclusive GPU after an OOM, `amendment_002_oom_rerun.json` (`{"moved": [{"source_id": ...}]}`); gate runs as written by
`evaluation/gate/` (`cases/`, `retry-bigmem/cases/`, `clone-*/cases/`, `retry_exhausted.json` markers); the subset list
and the frozen manifest. The directory names in `methods.json` are neutral placeholders; point them at your runs.

Output (`affordcraft.external_method_evidence.v2.13`): per method and track (`subset200`, `full2000`) the denominator,
counts (`stage1_emitted`, `export`, `native_pass`, `adapted_pass`, `native_not_applicable`, articulated passes and
exports, `gate_evaluated`, `infra_blocked`, `native_not_loadable`, `capacity_exhausted`, `pending`), rates with Wilson
95 % intervals, stage medians, timing (stage, gate and end-to-end medians, P95, means, worker minutes per pass, peak
GPU memory), `complete`, and one row per input.

## Rules (fixed, never outcome-based)

- Method records: the first record per input in the listed run order stands (subset run, then the extension runs);
  duplicates are listed. A failure whose reason names an API error, HTTP 403/429/5xx, CUDA OOM, a connection error or
  a timeout is infrastructure, not a method failure; a CUDA OOM that recurs on an exclusive GPU (or, for PhysX-Omni, on
  the largest available devices) is terminal capacity exhaustion (no export).
- Gate verdicts: origin `cases/` of every listed gate run, then `retry-bigmem/cases/`, then `clone-*/cases/` in name
  order. A verdict without an infrastructure block replaces a blocked one; among unblocked duplicates the first stands.
  A big-memory re-gate verdict with a `retry_exhausted.json` marker is `capacity_exhausted`; with `--finalization`, an
  input listed in the cutoff record whose chosen verdict is still blocked is `capacity_exhausted` too (gate runs listed
  under `post_finalization_gates` excepted).
- A verdict that saw no asset while the chosen generation record carries an emitted export is stale; the first
  non-stale verdict of another stage stands.
- A non-infrastructure `blocked` native verdict is a terminal method-side failure (`native_not_loadable`).
- E2E native = method stages (+ geometry stage for PhysX-Omni) + native gate seconds (0 when the native condition was
  not evaluated); E2E adapted = method stages + AffordCraft build + adapted gate. P95 = nearest rank.

## Changes from the executed version

- `aggregate_external.py`: the run layout (method runs, gate runs, extension runs, post-finalization gate runs,
  PhysX-Omni representation/geometry runs and the directory marker of its recompute lanes on the largest devices) moved
  from constants in the code to `methods.json` (`load_config`); the runs root, subset list and frozen manifest became
  options; the finalization record is passed with `--finalization` instead of being read from the script's directory.
  An unused helper that mapped paths of the executed machines to a mirror was removed. Device notes no longer name
  machines or the API provider. Formatting with black.
- `finalize.py` merges the two executed finalization scripts: `cutoff` is the first (which also patched the executed
  aggregator to add the finalization rule; the published aggregator already contains it, v2.13), `retry-exhausted` is
  the second, generalized from one gate run to any (`--gate-run`, `--third-attempt`). The earlier-attempt count of the
  cutoff record looks in the configured gate runs instead of every directory whose name contains "gates". Decision
  wording, dates and the name of the copy step were dropped from the records.

## Provenance

| Original file | Published path | sha256 of the original |
|---|---|---|
| aggregate_external.py (v2.13) | evaluation/aggregate/aggregate_external.py | 24da69642bb2ef44b429f0e574f3391857bffdf633f9a1a753f52537fff16f56 |
| finalize_20260924.py | evaluation/aggregate/finalize.py (`cutoff`) | 927bdb95d6fd1b321f9036eb3991de0fbc0757aa8aa950d7c1b5cb832ddea4fd |
| finalize_pc073_20260925.py | evaluation/aggregate/finalize.py (`retry-exhausted`) | 86731df6ece10dc5b01b99b8e578b2d020fb0cc32bdace9b1b0d7e7968fba308 |
| (constants of aggregate_external.py) | evaluation/aggregate/methods.json | -- |
