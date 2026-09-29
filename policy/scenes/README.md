# Composed-scene manipulation study

Ten tabletop scenes built from assets the method constructed, a scripted teacher, DAgger-style mixed rollouts, and a
recurrent action-chunk head over frozen OpenVLA-OFT features. Supports Section 4.5 ("Manipulation with constructed
assets"), Appendix "Composed-scene manipulation study", Tables `vla-scenes`, `vla-collection`, `vla-rounds`,
`vla-results`, `vla-failure-modes`, `vla-hparams`, and the scene panels of Figure `vla`.

Paper numbers produced by this code: teacher 273/320 episodes (298 reach the first sub-goal); final head (round 2) on
held-out initial states 11/100 (first sub-goal 66/100) versus 0/40 (1/40) for the same head at its random
initialization; 8/60 on seen initial states.

## Contents

| Path | Role |
|---|---|
| `scene_specs.py` | the ten scenes: assets, placement regions and jitter, instruction, teacher plan, evaluator goals |
| `scene_eval.py` | independent evaluator: reads only the per-tick trace and the goals, never the teacher plan |
| `scene_runner.py` | Isaac Sim 5.1 runner: builds randomized episodes and runs `--mode teacher` (scripted teacher with privileged state), `--mode policy` (served head) or `--mode dagger` (teacher plan with policy bursts, teacher labels) |
| `bake_assets.py` | writes the articulated assets at their scene scales (run once) |
| `summarize_scenes.py` | per-scene pass counts, failure reasons and mean ticks of a collection or evaluation folder |
| `run_collect.sh` | teacher collection (`collect20`) |
| `post_train.sh` | one training round: audit, dataset, freeze receipt, feature cache, head training, completion |
| `bootstrap_round0.sh` | round-0 head on the part of the round-0 feature cache that exists (the head that drove the first DAgger round) |
| `run_dagger.sh` | DAgger round: served head of the previous round, workers in `--mode dagger` |
| `eval.sh` | held-out and seen evaluation of a trained round, trained and untrained head |
| `run_round2.sh` | round 2 (final): second DAgger round, training on all rollouts, evaluation |
| `vla_head/make_config.py` | frozen head configuration (scene configuration of Table `vla-hparams`) |
| `vla_head/pipeline_steps.py` | `config`, `audit`, `freeze`, `complete`, `eval_summary` steps |
| `vla_head/build_dataset.py` | dataset of 8-step label chunks from teacher and DAgger episodes, dynamics filter, per-horizon scales |
| `vla_head/cache_features.py` | frozen OpenVLA-OFT features of both images (2 x 4096-d) at the query stride 4, sharded, resumable |
| `vla_head/train_head.py` | recurrent head training (single-asset v0.167 recipe) |
| `vla_head/serve_policy.py` | Unix-socket policy service used by the runner (`trained_vla` or `untrained_vla`) |
| `vla_head/failure_modes_round2.py` | failure-mode classification of the round-2 held-out episodes (Table `vla-failure-modes`) |
| `data/asset_inventory_013.json` | campaign asset inventory: for each library entry the constructed asset (USD path as placeholder, hash, source input, scale, initial joint positions, links) |
| `data/asset_pool_013.json` | scene asset pool: category, role, scale, joints and USD path (placeholder) of each library entry |
| `data/handles_v1.json` | handle-part table (PartNet part samples per link) of the articulated entries |

The head code imports the adapter modules of `../single_asset/tools` (`exp3_oft_recurrent_adapter_v0_166.py`,
`exp3_oft_feedback_adapter_v0_167.py`, `exp3_oft_precision_adapter_v0_164.py` and their imports); `make_config.py`
writes that folder into the configuration (`tools_dir`) and copies the prompt, input and action contracts from
`../single_asset/configs/training_configuration.json`.

## Environment

| Variable | Meaning |
|---|---|
| `AFFORDCRAFT_RUNS` | run root; this study writes to `$AFFORDCRAFT_RUNS/scenes_vla` (below `$S`) |
| `AFFORDCRAFT_ISAAC_PYTHON` | Isaac Sim 5.1 python (runner) |
| `OPENVLA_OFT_PYTHON` | python of the openvla-oft environment (feature cache, training, service; executed with torch 2.7.1+cu128, transformers 4.40.1, timm 0.9.10) |
| `OPENVLA_OFT_HOME`, `OPENVLA_OFT_CHECKPOINT` | official openvla-oft checkout (commit e4287e9, https://github.com/moojink/openvla-oft) and the LIBERO-Spatial checkpoint |
| `OPENVLA_OFT_CHECKPOINT_INVENTORY` | JSON `{"files": [{"path", "sha256"}]}` of the checkpoint files; verified before the model loads |
| `AFFORDCRAFT_RUNTIME_PYTHON` | python with numpy for the dataset and bookkeeping steps (default `python`) |
| `AFFORDCRAFT_FRANKA_USD` | flattened local copy of the Isaac Sim Franka USD (`--franka` default) |
| `AFFORDCRAFT_RESCALE_USDA` | uniform USD rescaling helper used by `bake_assets.py` (`RESCALE SRC SCALE DST`, not included) |
| `AFFORDCRAFT_SCENE_INVENTORY`, `AFFORDCRAFT_SCENE_HANDLES` | optional overrides of `data/asset_inventory_013.json` and `data/handles_v1.json` |
| `GPUS`, `GPU_SVC`, `GPU_A`, `GPU_B`, `GPU` | GPU indices for Isaac workers, policy services, the two cache shards and training |

USD paths in `data/asset_pool_013.json` are `${AFFORDCRAFT_RUNS}/campaign/runs/<run>/runtime/builds/<build>/asset.usda`
(the campaign builds of the selected and adapted library entries) and are expanded with `os.path.expandvars`.

## Pipeline (executed order)

```bash
S=$AFFORDCRAFT_RUNS/scenes_vla
cd policy/scenes
python bake_assets.py                                              # scene-scale articulated assets -> $S/assets
python vla_head/make_config.py $S/vla/training_configuration.json 24000
GPUS="0 1 2 3" bash run_collect.sh                                 # collect20: 4 workers x 8 episodes x 10 scenes
python summarize_scenes.py $S/scenes/collect20
bash post_train.sh 0                                               # round 0: dataset, receipt, feature cache (see note)
bash bootstrap_round0.sh                                           # round-0 head (143 teacher episodes)
bash run_dagger.sh                                                 # dagger20: 200 rollouts with the round-0 head
bash post_train.sh 1                                               # round 1: 267 teacher + 82 successful rollouts
HEAD=$S/v20_vla/round1 bash eval.sh                                # round-1 evaluation (9/100 held-out)
bash run_round2.sh                                                 # dagger20r2 (240 rollouts) -> round 2 -> evaluation
mkdir -p $S/fig7_export && python vla_head/failure_modes_round2.py # Table vla-failure-modes
```

Round 0 as executed: `post_train.sh 0` built the round-0 dataset and started the two feature shards; before the cache
was complete, `bootstrap_round0.sh` trained the round-0 head on the episodes cached at that point (shard 1 complete
and the first part of shard 0: 143 of the 267 episodes) and wrote `round0/completion.json`. This bootstrap head is the
head that drove the first DAgger round. The remaining shard-0 features were completed afterwards (`cache_features.py`
resumes from existing episode files) and reused by round 1:

```bash
$OPENVLA_OFT_PYTHON vla_head/cache_features.py --config $S/v20_vla/round0/configuration.json \
    --dataset $S/v20_vla/round0/dataset --output $S/v20_vla/round0/features --gpu 0 --shard 0 --num-shards 2
```

Running `post_train.sh 0` to completion instead gives a round-0 head trained on the full cache (not what was run).

Settings (unchanged from the executed code): control 10 Hz over 60 Hz physics, arm commands clamped to 0.06 rad per
tick; episode budgets 1,200 ticks (teacher, DAgger) and 1,000 ticks (evaluation), 20 settle ticks. Seeds: collection
2000+w, DAgger rounds 2100+w and 2200+w (w = worker 0..3), held-out evaluation 2600, seen evaluation 2000; episode
seed = seed x 100000 + episode index. DAgger: executed share 0.35, bursts of 8-40 ticks, a prefix burst of up to 200
ticks in half of the episodes, teacher-only phases (grasp, close, release, arcs, lifts, press, pull, final hold),
fingers always from the teacher. Dataset: seed 20, chunk 8 with 4 executed, episodes whose executed 8-tick joint
change exceeds 0.75 rad excluded; round 2 adds failed rollouts truncated by 45 ticks with at least 120 remaining ticks
and no object off the table. Head: see `vla_head/make_config.py` (Table `vla-hparams`).

## Inputs and outputs

Inputs not in this repository: the USD builds of the library entries (paths in `data/asset_pool_013.json`), the
rescaling helper, the Franka USD, the openvla-oft checkout and LIBERO-Spatial checkpoint. Outputs under `$S`:
`assets/` (baked USDs, `baked_manifest.json`); `scenes/<tag>/<scene>_w<k>/episode_XXXX/` with `episode.json`
(placements, result, teacher status), `trace.jsonl` (per-tick observation paths, proprio, labels, executed commands,
object and joint state, contacts), `frames/` and `frames_wrist/` (320x240 policy observations), `review.mp4`, plus
`summary.jsonl` per run; `v20_vla/round<k>/` with `configuration.json`, `dataset/`, `dataset_frozen_receipt.json`,
`features/`, `training/` (initial and final adapters, `checkpoint_manifest.json`), `completion.json`, `eval/`,
`eval_seen/`, `eval_summary.json`, `eval_summary_eval_seen.json`.

## Changes from the executed version

- Patchers. `vla/patch_v20.py` and `vla/patch_round2_v20.py` were one-shot patchers that edited files in place before
  the runs; nothing is patched at run time. `patch_v20.py` turned a copy of the v19 framework into the v20 framework
  (applied to the v19 files it reproduces the v20 `serve_policy.py` and `pipeline_steps.py` byte for byte, and the v20
  runner, dataset and cache files contain all of its replacements); `patch_round2_v20.py` added the failed-rollout rule
  to `build_dataset.py` and the round-2 options to `post_train20.sh` before round 2 (both patchers applied to v19
  reproduce the v20 `build_dataset.py` exactly; the new options are off by default, so rounds 0-1 are unaffected). The
  v20 files also carry direct edits made after the patcher (wrist-camera orientation, DAgger burst rules, resumable
  feature cache with `--reuse-features`). The runner file was last modified shortly after the teacher collection had
  started; episodes recorded before and after that time store the same wrist-camera configuration (mount, pitch, field
  of view and axes in `placements/wrist_camera` of each `episode.json`). This release publishes the final v20 files,
  i.e. the effective code, and omits the patchers.
- Infrastructure removed: GPU pickers that polled `nvidia-smi` for free memory on shared GPUs (replaced by fixed GPU
  lists), waits on completion markers of the chained background launch, stop files, the installer that armed round 2,
  the hang watchdog, the relauncher of the stalled round-0 cache shard, smoke tests and the dummy policy service. The
  fill pass of the collection (`fill_collect20.sh`: resume a crashed run at its next episode index with the same seed)
  is merged into the worker loop of `run_collect.sh`; `run_dagger.sh` and `eval.sh` keep their in-place retries.
- Absolute paths, interpreters and host-specific defaults became the variables above. The executed runs resolved
  candidate ids in the campaign asset inventory first and in the asset pool second; both ship in `data/` with USD
  paths as placeholders, and the runner loads the handle table `data/handles_v1.json` as the executed runs did. `post_train.sh` passes the label `framework_v20` where the executed
  script passed the framework folder, so the recorded method id and collection tag are unchanged.
- Renames: `*20.sh` / `eval_v20.sh` -> `*.sh` / `eval.sh`, `vla/` -> `vla_head/`, `failure_modes_round2.py` moved into
  `vla_head/` (imports adjusted).
- Comments that described the host were rewritten; Python files are formatted with black (AST unchanged apart from the
  path statements listed above).
- Not published: figure scripts (`fig3_prep.py`, `fig3_render.py`, `export_fig7_frames.py`), the smoke and rescue
  scripts named above, and `noinotify.c` (an optional LD_PRELOAD workaround for an exhausted inotify budget on one
  host; the runner uses a `libnoinotify.so` next to it only if one exists).

## Provenance

| Original file | Published path | sha256 of original | Change |
|---|---|---|---|
| scene_specs.py | scenes/scene_specs.py | `305f4237c131c3857bee0e456ad81bb48e094cc8448391cd580c6fd59fd3c1f9` | paths -> variables / data folder; docstring |
| scene_eval.py | scenes/scene_eval.py | `41aac18474165b7f8bdc7c0f53849bfc8669c12c07bb65ae6bdfad243463d3a6` | black only |
| scene_runner.py | scenes/scene_runner.py | `5808ba936c24f40c80fdec2bb9c7f3658fe0979fa13c0abf82744bfc6b7663c4` | Franka and handle-table defaults; comments |
| bake_assets.py | scenes/bake_assets.py | `3c542001df76d641faf372b4a41179627c692e9c17d7d60d448beea88508dc32` | interpreter and helper paths -> variables |
| summarize_scenes.py | scenes/summarize_scenes.py | `269b5f97c4f0c9864c2400a859af5efd9a6cd25e55e9e037224fcb2c53c22467` | black only |
| make_config.py | scenes/vla_head/make_config.py | `002b9bfc23da429ae8339225ddc3c6f7ba39d76a078b420118b0b7dbcffdb7fe` | paths -> variables / repository-relative |
| pipeline_steps.py | scenes/vla_head/pipeline_steps.py | `28583825ca44aae2087d554f0cc37fcbd2c80727501dc9283d11819fb1d47161` | interpreter path string; docstring |
| build_dataset.py | scenes/vla_head/build_dataset.py | `8f91c25b297af28bc965dc586d04e6d606a8ccb6b1be5a3011400eb88df20f01` | black only |
| cache_features.py | scenes/vla_head/cache_features.py | `a002299a8ce5ea76b1eaec8c560db3fe875913790c75675bdde0c6a119f6a003` | black only |
| train_head.py | scenes/vla_head/train_head.py | `48997aeda651b3ad2dfa17b0ac1c9083c319fd16121e9fa373cf3156bbb12fc6` | black only |
| serve_policy.py | scenes/vla_head/serve_policy.py | `e2e95ad7b70cd1f57a407fd0e972a3aabe541d8b16cb0ed48411b788fed973c6` | black only |
| failure_modes_round2.py | scenes/vla_head/failure_modes_round2.py | `8687a53e53181d950ce9ca634ff94ab841a0b2ea399f435fd9840a9e7e78c75b` | run-root path -> variable; docstring |
| run_collect20.sh | scenes/run_collect.sh | `d2ebb403e1b9fa89d8275e3b223e4745066556cffd4b19f1ef79c1568f61dbd2` | GPU picking and stop file removed; fill pass merged |
| fill_collect20.sh | scenes/run_collect.sh | `edb9dcb04609e5b455ef169c4ce52cd7152902bd0a68089970aee0222b7d89f6` | resume loop merged into run_collect.sh |
| post_train20.sh | scenes/post_train.sh | `42a77025060e83f218b1e4f33485119b1db14d6cfaea2cd516135c2e5d201dcd` | marker waits and GPU picking removed |
| bootstrap_round0.sh | scenes/bootstrap_round0.sh | `b0dbe4875d7ba1b74ece5ec549b8708322651e29850677fad47b636499b2cdf1` | GPU picking removed |
| run_dagger20.sh | scenes/run_dagger.sh | `d7c23320226cf4b6613d4228200a80215197c03f34d1df08b866eabea62f803b` | head wait, markers, GPU picking removed |
| eval_v20.sh | scenes/eval.sh | `e69485353b82362a16adabdad29e7ae1c4a4a871240d272244f86b0e4bb361b1` | head wait, markers, GPU picking removed |
| run_round2.sh | scenes/run_round2.sh | `d86cc6f4e662ba43ac10abee53f7ba094e47444be888dd8c4bc94409c5352435` | waits on the previous round's markers removed |
| asset_pool_013.json | scenes/data/asset_pool_013.json | `7f6489fb15576224907fb6076be10596b7ea4157b62bf9b362c17b006f808bfe` | USD paths -> `${AFFORDCRAFT_RUNS}` placeholders |
| handles_v1.json | scenes/data/handles_v1.json | `1c60603bcf3bd8a09b045ea69330e1fcc4606eaa3be72da61080cdc836db8544` | unchanged |
| asset_inventory_013.json | scenes/data/asset_inventory_013.json | `0d7c73dcd5121df3f4cc293a96cd2b9b30ffe110f65b2539295ea79dd769bbb1` | USD paths -> `${AFFORDCRAFT_RUNS}` placeholders |
| patch_v20.py | (not published) | `864ed7824728288c593d4bb40d4307e8bdf3d5b9a2b955a91b109267dc8cad49` | one-shot patcher; its edits are in the published files |
| patch_round2_v20.py | (not published) | `5b47bfa8dd6e363d19bc8ed388d6acc479f56d3a225defb1aa589ab23621f435` | one-shot patcher; its edits are in the published files |
