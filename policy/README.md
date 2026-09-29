# Policy learning with constructed assets

Code of the manipulation study (Section 4.5, "Manipulation with constructed assets"; Appendix "Single-asset
manipulation tasks" and "Composed-scene manipulation study"). Twenty tasks are built from assets the method
constructed and run with a Franka Panda in Isaac Sim, commanded at 10 Hz over 60 Hz physics with seven arm-joint
targets and two finger widths. A scripted teacher with privileged simulator state provides demonstrations, and the only
learned component is a recurrent action-chunk head over frozen OpenVLA-OFT features (public LIBERO-Spatial checkpoint).
An independent evaluator reads every outcome from simulator state; the untrained condition is the same head at its
random initialization.

| | `single_asset/` | `scenes/` |
|---|---|---|
| Tasks | 10: push one of five rigid objects into a goal region, or close one of five articulated assets | 10 composed tabletop scenes; each chains an articulation (door, drawer, lid, laptop) with a placement |
| Teacher data | 100 successful episodes per task (1,000) | 320 teacher episodes (273 successes) plus two rounds of DAgger-style mixed rollouts labelled by the teacher |
| Observation | one 320x240 camera, 9 joint positions, instruction | third-person workspace crop and wrist camera (2 x 320x240), 9 joint positions, instruction |
| Head training | 32,000 steps, run v0.167 | 24,000 steps per round; round 2 is final |
| Held-out result | 114/200 trained vs 1/200 untrained | 11/100 trained vs 0/40 untrained (first sub-goal 66/100) |
| Paper | Table `vla-single-asset`, Figure `vla-single` | Tables `vla-scenes`, `vla-collection`, `vla-rounds`, `vla-results`, `vla-failure-modes`, `vla-hparams`; Figure `vla` |

The head is shared: each image passes with the prompt `In: What action should the robot take to {instruction}?\nOut:`
(instruction in lower case) through the frozen OpenVLA-OFT backbone, the mean of its 56 action-token states gives a
4,096-d feature per image, and a GRU head (512 units) with a learned proprioception path predicts eight future 9-D
joint targets of which four execute. The architecture, augmentation and losses were registered for the single-asset
run v0.167; the scene study reuses them: `scenes/vla_head/` imports the adapter modules of `single_asset/tools/` and
copies the prompt and action contracts from `single_asset/configs/training_configuration.json`.

## Pipeline order

1. `single_asset/` (run first; defines the head recipe): frozen feature cache -> training-only diagnostic ->
   registration of the configuration -> lifecycle (training, independent training gate, paired evaluation of the
   untrained and trained head on 200 held-out initial states, independent paired verifier).
   See `single_asset/README.md`.
2. `scenes/`: bake the scene assets and write the head configuration -> teacher collection (`collect20`) -> round 0
   (dataset, feature cache, bootstrap head) -> DAgger round 1 -> training round 1 -> evaluation -> DAgger round 2 ->
   training round 2 -> evaluation -> failure-mode analysis. See `scenes/README.md`.

## Environment

`AFFORDCRAFT_RUNS` (run root), `AFFORDCRAFT_ISAAC_PYTHON` (Isaac Sim python: 5.1 for the scenes; the registered
single-asset run used an Isaac Sim 6 installation), `OPENVLA_OFT_PYTHON`, `OPENVLA_OFT_HOME`, `OPENVLA_OFT_CHECKPOINT`
(openvla-oft checkout at commit e4287e9 from https://github.com/moojink/openvla-oft and the LIBERO-Spatial checkpoint),
`AFFORDCRAFT_RUNTIME_PYTHON`, `AFFORDCRAFT_MODELS` and `AFFORDCRAFT_PROJECT_ROOT` (placeholders in the single-asset
configurations). Study-specific variables are listed in the two READMEs. Datasets, rendered frames, videos, asset USD
files and checkpoints are not part of the repository.

## Changes from the executed version and provenance

Summarized here, detailed per file in the two READMEs.

- `single_asset/`: the code is published byte-identical to the registered files (sha256 recorded in the registered
  configuration), except that the report strings of the paired summarizer were translated to English; the data-contract
  package `i2ia.contracts`, which lies outside the registered code set, is added; configurations and summary have placeholders instead of absolute
  paths. Only the files the v0.167 chain runs or imports are published; the other registered files are listed.
- `scenes/`: the two patchers of the v20 framework edited the files in place before the runs; the published files are
  the resulting (effective) v20 files and the patchers are omitted. Scheduling on shared GPUs, marker waits, watchdog,
  rescue, installer and smoke-test scripts were removed from the drivers; paths became environment variables; Python
  files are black-formatted.
- Not published in either study: figure-rendering scripts, probes and diagnostics outside the chains, logs, receipts,
  per-episode data.
