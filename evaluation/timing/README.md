# timing/ -- clean-GPU timing run

Purpose: per-input time, peak GPU memory and worker minutes per accepted asset of every local method under identical
conditions: one worker per exclusive GPU, a 32-input sample of the registered subset, the same code, weights and
settings as the registered runs. Supports the per-input columns of Table 1 (E2E median, pass within 5 min, minutes per
pass, peak GiB), the resource table, the time-to-pass figure, and the cold/warm comparison in the resource appendix.

## Protocol

- **Sample**: the first 32 inputs of the registered 200-input subset in sha256(input_id) order, kept in subset order
  (`select_sample.py`; the ids are in `sample32_ids.json`).
- **Devices**: every external-method driver refuses a GPU with 1 GiB or more already in use; the AffordCraft shards ran
  on GPUs nothing else used (their preparation record lists the GPU's processes before the start). RTX 5090 32 GB for
  every stage, except the PhysX-Omni geometry+export stage (91 GiB peak), which ran on an RTX PRO 6000 96 GB.
- **AffordCraft** (`run_affordcraft_shard.sh`): `scripts/run_pipeline.py --calibration` over the sample in 4 shards
  (input i goes to shard i % 4), each on its own GPU, with the worker environment of the main campaign (OMP/MKL/OpenBLAS
  2 threads, `MALLOC_ARENA_MAX=2`, `AFFORDCRAFT_BUILD_THREADS=8`). `prepare_affordcraft_shard.py` first checks the
  sha256 of the 65 code and model files of the main campaign's definition and of every sample image (any mismatch: the
  shard does not run), then gives the shard its own hard-linked copy of the geometry cache, in one of two regimes:
  - **cold**: only the cache entries that existed when the main campaign started (the geometry the campaign itself
    built is left out), so the run pays the first-time decomposition of library entries;
  - **warm**: the whole cache after the 2,000-input campaign, i.e. the geometry of the library entries built once and
    reused ("built once, reused"); Table 1 reports this regime, the appendix both.
  New builds go into the shard's copy only.
- **External methods** (sample index i -> lane i % n, one lane per exclusive GPU): PhysX-Anything 4 lanes (stage A
  VLM, stage B decoder + split + export), PAct 4 lanes (automatic part masks from Grounding DINO + SAM, official
  inference, finalize), TRELLIS.2 2 lanes (`--render` kept for parity, render time recorded separately), PartCrafter
  2 lanes (the VLM part count and its recorded call time are replayed from the registered run, the GPU stages are
  measured), PhysX-Omni 4 lanes (representation on RTX 5090, geometry + export on RTX PRO 6000). Every lane writes
  `lane.wall.json` (or `rep.wall.json` / `geo.wall.json`), model loading included.
- **Gates**: every output was gated with `evaluation/gate/` (both conditions) on machines with the 580 driver, with an
  empty geometry cache of this run (external meshes never hit a campaign cache entry) and `GATE_MAT_SLOTS` equal to the
  number of gate processes on the machine (no builder waits for a slot).
- **GPU monitor**: on every machine, 1-s samples
  `nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 1 >> monitor/$(hostname)_gpu.csv`;
  peak memory of a worker = maximum `memory.used` of its GPU inside its wall window, energy = sum of the 1-s power samples.
- **Not timed here**: the two API routes (Articulate-Anything, GPT-6 Astra agent); their per-input times come from their
  own registered runs (marked in Table 1).

How the paper uses the summary: minutes per accepted asset = clean median seconds per input (failed inputs included) /
pass rate of the row's registered run; "pass in 5 min" = share of the timed inputs whose asset passes the gate within
300 s of wall time; peak = nvidia-smi memory of the worker's GPU.

## Entry points

```bash
export TIMING_ROOT=$AFFORDCRAFT_RUNS/clean_timing
python evaluation/timing/select_sample.py subset_200_inputs.json exp1_per_case.jsonl 32 $TIMING_ROOT/inputs/sample32.json
# external methods: lane manifests (and the PartCrafter replay), then one driver per lane and GPU
python evaluation/timing/setup_timing_run.py --root $TIMING_ROOT lanes $TIMING_ROOT/trellis2 2
python evaluation/timing/setup_timing_run.py --root $TIMING_ROOT partcrafter-replay <registered PartCrafter subset run> $TIMING_ROOT/partcrafter
bash evaluation/timing/run_trellis2_lane.sh 0 0          # <lane> <gpu>; likewise run_physx_anything_lane.sh,
                                                         # run_pact_lane.sh, run_partcrafter_lane.sh,
                                                         # run_physx_omni_representation.sh, run_physx_omni_geometry.sh
bash evaluation/timing/run_affordcraft_shard.sh warm 0 4 0   # <cold|warm> <shard> <shard count> <gpu>
# after gating every output into $TIMING_ROOT/gates/<machine>/<method>/ with evaluation/gate/:
python evaluation/timing/summarize_clean_timing.py --root $TIMING_ROOT \
    --campaign-per-case exp1_per_case.jsonl --campaign-aggregate aggregate_all.json
```

Run root layout: `inputs/{sample32.json, input_manifest.jsonl (frozen 2,000-input manifest), locks_exp1.json}`,
`ours-{cold,warm}/`, `<method>/{configuration.json, code/, lane-<k>/}`, `gates/<machine>/<method>/`, `monitor/`,
`summary/`. `<method>/code/` is a copy of the method's wrapper code of the registered run (`baselines/`) with its run
root and input set pointed at this run; `<method>/configuration.json` is the registered run's configuration with the
sample as input set. The PAct driver reads its lane partition from that configuration (`lane_partition.lanes`).

Environment: `TIMING_ROOT`; AffordCraft: `AFFORDCRAFT_SELECTOR_PYTHON` (runs `run_pipeline.py`),
`AFFORDCRAFT_RUNTIME_PYTHON`, `AFFORDCRAFT_INDEX` (visual index), `AFFORDCRAFT_GEOMETRY_CACHE` (the main campaign's
cache); external methods: `PHYSX_ANYTHING_VLM_PYTHON`, `PHYSX_ANYTHING_GEOM_PYTHON`, `PACT_PYTHON`, `PACT_HOME`, `PACT_LANE_RUNNER`, `PACT_MASK_ADAPTER`,
`PACT_TOOLS`, `PACT_GDINO`, `PACT_SAM`, `PACT_PYTHONPATH`, `PARTCRAFTER_PYTHON`, `TRELLIS2_PYTHON`, `TRELLIS2_HF_HOME`,
`PHYSX_OMNI_REP_PYTHON`, `PHYSX_OMNI_REP_RUNNER`, `PHYSX_OMNI_REP_WORKSPACE`, `PHYSX_OMNI_GEO_PYTHON`,
`PHYSX_OMNI_GEO_RUNNER`, `PHYSX_OMNI_GEO_WORKSPACE`, `AFFORDCRAFT_PROJECT_ROOT`, `AFFORDCRAFT_MODELS`. The two PhysX-Omni
runners default to `baselines/physx_omni/`; the PAct lane runner and mask adapter default to the copy of `baselines/pact/`
in `<run root>/pact/code`.

Output: `summary/clean_timing_summary_<stamp>.json` (the paper's summary, scrubbed of paths and machine names, is
`analysis/results/clean_timing/clean_timing_summary_20260924-1335.json`).

Execution details of the paper's run: the AffordCraft cold regime has a fifth shard, a re-run
(`run_affordcraft_rerun.sh`) of the one input that shard 1 left without a record after an infrastructure abort (an empty
physics-service response); in the warm regime shard 3 was stopped after its seventh input and its eighth input ran as
shard 6 on another exclusive GPU. The summary keeps the first record per input in shard order.

## Changes from the executed version

- Drivers: absolute paths and interpreters became the variables above; the per-lane temporary and pip-cache directories
  of the executed machines were dropped (PartCrafter and TRELLIS.2 keep their per-lane `TMPDIR`); lane tags no longer
  contain the run's internal name; the two PhysX-Omni runners and workspaces must be given (no default).
- `prepare_affordcraft_shard.py`: the campaign geometry cache is an argument / `AFFORDCRAFT_GEOMETRY_CACHE`; the cold
  cutoff keeps its value (unix time of the start of the main campaign's first worker).
- `summarize_clean_timing.py`: paths became options (`--root`, `--campaign-per-case`, `--campaign-aggregate`); the gate
  machine directories are listed with `--gate-machines` in precedence order (default: sorted names; the executed version
  listed its three gate machines in a fixed order, which matters only if an input was gated twice); the monitor file of
  a worker is found through the host recorded in its wall file (the executed version mapped shards and lanes to machine
  names with a table); `--finalization` passes a cutoff record to the aggregator (in the executed run the record was
  present and changed no verdict of the sample: `capacity_exhausted` = 0 for every method); labels and device notes
  without internal run or machine names.
- `setup_timing_run.py` keeps two parts of the executed setup script: the lane partition and the PartCrafter replay
  (replay table and replay runner; the expected hashes of the manifest and the sample are options). The rest of that
  script copied the registered runs' wrapper code and rewrote their configurations for this run root; it is described
  above instead.
- Not included (orchestration): the gate dispatcher (assigned batches of four cases to gate machines, copied case
  directories between machines, collected results), the per-machine gate loop, chain scripts that sequenced drivers on
  particular GPUs, the monitor script (its measurement is the nvidia-smi command above), the result-copy script, the
  stop script of warm shard 3, the TRELLIS.2 export watchdog of the registered run (same export-timeout rule as there),
  and the PhysX-Omni native-manifest writer run before gating (the one of the registered PhysX-Omni run, `baselines/`).

## Provenance

| Original file | Published path | sha256 of the original |
|---|---|---|
| ct72_sample.py | evaluation/timing/select_sample.py | c802d2fdf5534eabfcb9842720136088bccdb4ad90faeca3ae2453ab069bfe3e |
| sample32.json | evaluation/timing/sample32_ids.json (ids only) | c4a2d8bfac3c166ad62a440fd3a09206f912401fc8591ca62ddcafc4da6d33c4 |
| ct72_prep.py | evaluation/timing/prepare_affordcraft_shard.py | 696ea4c863d7b9938c2b09d131346d1fee5e19ef76260b44c60890bc67a2a6d2 |
| ct72_ours.sh | evaluation/timing/run_affordcraft_shard.sh | 5e41b361927c0fe85445ffd76ed84f6c4f984fc38922871ae4779be3f3422a1e |
| ct72_ours_rerun.sh | evaluation/timing/run_affordcraft_rerun.sh | d58b76a559f41d369e09b60445b5f2fa0fdef15c00baad550a482e6c54b81776 |
| ct72_pa.sh | evaluation/timing/run_physx_anything_lane.sh | 73b5d1c96077e2c21e33bd249cd113caa8f4bcda3223b297ba258d7bcfe4eecb |
| ct72_pact.sh | evaluation/timing/run_pact_lane.sh | 08dde0994341df513c1569e9de1caf32e82d117e949b467e8e795af301ee015b |
| ct72_pc.sh | evaluation/timing/run_partcrafter_lane.sh | d5288af1fc4af849381261156c526cc5ba2fbb9340a0fa516bc9113f3470c883 |
| ct72_t2.sh | evaluation/timing/run_trellis2_lane.sh | b5095602f24b039623e032ba71f806871184f77e6c184acfde179ff6b80095ff |
| ct72_omni_rep.sh | evaluation/timing/run_physx_omni_representation.sh | 16ff4c0e4e5c8542bc0f29f2e053964ae8995f7650b42e9c5fd0f9ac37f4368e |
| ct72_omni_geo.sh | evaluation/timing/run_physx_omni_geometry.sh | e537a963cb23f9e43b2449cd9e82dc35d2f754d128f0ff8cb365e86dd6227de4 |
| ct72_setup.py (lane partition, PartCrafter replay) | evaluation/timing/setup_timing_run.py | ad1e4b22f72e31a55ce03fdf9c6688b3cc8de98f5f7d24cb667fb06fc342ec3e |
| ct72_summarize.py | evaluation/timing/summarize_clean_timing.py | 28ca958e60f7025d6af8048c7432bceb13c62d985725aea16e8c05d05ee2a9ad |
