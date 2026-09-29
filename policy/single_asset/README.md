# Single-asset manipulation tasks (registered run v0.167)

Ten tasks, one per constructed asset: five rigid objects must be pushed fully into a marked goal region and left
stable, five articulated assets (a microwave, three storage units, a drawer unit) must have their door or drawer
closed into the registered closed range and held there. A scripted teacher with privileged state provided 100
successful episodes per task (1,000 episodes, 245,954 frames), frozen together with 200 held-out initial states (20
per task). The only learned component is a recurrent head with proprioceptive feedback over frozen OpenVLA-OFT
features (LIBERO-Spatial checkpoint); it sees one 320x240 camera image, the nine joint positions and the instruction.
Supports Appendix "Single-asset manipulation tasks", Table `vla-single-asset`, Figure `vla-single`, and the
single-asset sentence of Section 4.5.

## Results (`results/summary.json`)

Trained head 114/200 (57.0 %, Wilson 95 % interval 50.1-63.7 %) versus 1/200 for the untrained head (the same head at
its saved random initialization); rigid 58/100, articulated 56/100. Five initial states of two articulated tasks
produced no episode record in either condition and count as failures.

| Asset | Task id | Untrained /20 | Trained /20 | No record |
|---|---|---:|---:|---:|
| rigid_00_bottle | rigid_00_bottle__full_task_v0_85 | 0 | 5 | 0 |
| rigid_compact_box_cf109672 | rigid_compact_box_cf109672__push_to_region_v0_94 | 0 | 6 | 0 |
| rigid_compact_camera_9f360f3c | rigid_compact_camera_9f360f3c__push_to_region_v0_94 | 0 | 19 | 0 |
| rigid_compact_clock_ddb96318 | rigid_compact_clock_ddb96318__push_to_region_v0_94 | 0 | 14 | 0 |
| rigid_compact_display_67ea5509 | rigid_compact_display_67ea5509__push_to_region_v0_94 | 0 | 14 | 0 |
| articulated_01_microwave | articulated_01_microwave__source_correct_close_v0_105 | 1 | 8 | 2 |
| articulated_02_storage_45526 | articulated_02_storage_45526__source_correct_close_v0_105 | 0 | 7 | 3 |
| articulated_03_storage_45638 | articulated_03_storage_45638__source_correct_close_v0_105 | 0 | 3 | 0 |
| articulated_04_storage_46889 | articulated_04_storage_46889__close_source_verified_v0_89 | 0 | 18 | 0 |
| articulated_drawer_46236 | articulated_drawer_46236__full_close_v0_120 | 0 | 20 | 0 |

`results/summary.json` is the delivery summary of the run: counts and Wilson intervals by variant, asset, role and
task, failure reasons, the blocked initial states, training metadata, and the results of the earlier head versions on
the same held-out states (the held-out set was reused during development; the paper therefore makes no claim of
generalization to unseen assets).

## Layout

- `src/i2ia/tasks/`: the task package (success predicates, episode termination, dataset coverage, contact and
  workcell geometry, robot command limits, teacher primitives). `src/i2ia/contracts/` holds the project's data-contract types, see Changes.
- `tools/`: the scripts and modules of the v0.167 chain. Their version suffixes are kept: the registered configuration
  lists every file by name and hash, and the scripts import predecessor versions by name.
- `configs/`: the registered configuration of the run (`training_configuration.json`, `evaluation_configuration.json`
  with the ten task specifications and the 200 held-out initial states, `lifecycle_configuration.json`).
- `results/summary.json`: see above.

## The v0.167 chain

Settings (registered, unchanged): head = LayerNorm/linear projection of the 4,096-d feature to 512, proprio MLP 9-128-128
on training-set normalized joints, 2-layer GRU (512), output MLP with a learned current-proprio skip, 8 x 9 targets
through tanh; arm targets are deltas to the chunk-start joints divided by per-horizon scales
(0.06, 0.17, 0.30, 0.40, 0.48, 0.55, 0.62, 0.68) rad, fingers absolute in [0, 0.04] m; 4 of 8 executed at 10 Hz over
60 Hz physics. Training: 32,000 AdamW steps, batch 16 asset-balanced episodes, learning rate 3e-4 (250 warm-up steps,
cosine to 0.1x), weight decay 1e-4, loss = motion-weighted L1 + 0.1 MSE (arm weight 5, fingers 1, motion gain 4 at
0.01 rad) + 0.25 temporal + 0.1 chunk-overlap consistency; correlated arm-proprio noise (0.005 rad, rho 0.8) on 75 %
of the sequences with targets projected into the scale envelope; seeds 138000001 (sampling), 138000002 (head
initialization), 167031 (augmentation); the step-32,000 checkpoint is evaluated, no checkpoint selection. Evaluation:
both variants on the same 200 held-out initial states, at most 360 control steps; an independent verifier recomputes
success from the recorded simulator state and contacts (the raw videos are checked as evidence, never scored).

Executed order (paths as in `configs/`; `R=$AFFORDCRAFT_RUNS`; run from this folder):

```bash
R=$AFFORDCRAFT_RUNS
# 1. frozen OpenVLA-OFT features of the 1,000 training episodes (query stride 4). Executed inside the v0.166 run;
#    v0.167 reuses this cache (feature_cache_manifest and its sha256 are registered in the training configuration).
$OPENVLA_OFT_PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 tools/cache_exp3_oft_features_v0_166.py \
    --configuration $R/exp3-complete-robot-tasks-v0-166/recurrent-training-001/training_configuration.json \
    --output $R/exp3-complete-robot-tasks-v0-166/recurrent-training-001/execution-001/features
# 2. training-only diagnostic of the v0.166 head (a required input of the registration; it sets no parameter)
$OPENVLA_OFT_PYTHON tools/diagnose_exp3_training_feedback_v0_167.py \
    --registration $R/exp3-complete-robot-tasks-v0-166/recurrent-training-001 \
    --run $R/exp3-complete-robot-tasks-v0-166/recurrent-training-001/execution-001 \
    --output $R/exp3-complete-robot-tasks-v0-166/training-feedback-diagnostic-001
# 3. registration: writes the three configurations of configs/ and freeze_receipt.json (hashes of configurations and code)
python tools/register_exp3_feedback_learning_v0_167.py \
    --base $R/exp3-complete-robot-tasks-v0-166/recurrent-training-001 \
    --diagnostic $R/exp3-complete-robot-tasks-v0-166/training-feedback-diagnostic-001 \
    --output $R/exp3-complete-robot-tasks-v0-167/feedback-denoising-training-001
# 4. lifecycle: training -> independent training gate -> paired evaluation (untrained, trained) -> paired verifier
python tools/run_exp3_feedback_learning_v0_167.py \
    --configuration $R/exp3-complete-robot-tasks-v0-167/feedback-denoising-training-001/lifecycle_configuration.json
```

What step 4 launches (with `PYTHONPATH=src:tools`, see `run_exp3_precision_learning_v0_156.environment`):

| Script | Role |
|---|---|
| `train_exp3_oft_feedback_v0_167.py` | head training on the cached features (one GPU); imports `exp3_oft_feedback_adapter_v0_167` (head, augmentation) -> `exp3_oft_recurrent_adapter_v0_166` (frozen feature extraction, prompt) -> `exp3_oft_precision_adapter_v0_164` (per-horizon scales; imports `_v0_160`, `_v0_156`) and `exp3_oft_joint_adapter_v0_134` (dataset gate, official-source loader) |
| `verify_exp3_feedback_training_v0_167.py` | independent gate: split, cache hashes, normalization statistics, complete optimization log, checkpoint reload |
| `serve_exp3_feedback_vla_v0_167.py` | policy service per GPU (the v0.166 service `serve_exp3_recurrent_vla_v0_166.py` with the v0.167 head); `run_exp3_feedback_learning_v0_167.py` reuses `paired_variant` of `run_exp3_precision_learning_v0_156.py` and swaps that function's v0.156 service command for this script at launch |
| `run_exp3_paired_policy_batch_v0_134.py` | one Isaac process per asset: its 20 held-out episodes through `run_exp3_complete_task_local_policy_v0_134.py` (episode driver; `--controller` admits only the served policy) with `exp3_local_robot_policy_client_v0_134.py`, `exp3_training_instruction_contract_v0_134.py` and provisional checks by `verify_exp3_paired_robot_episode_v0_134.py` |
| `summarize_exp3_paired_learning_v0_141.py` | independent re-verification of all 400 episodes (`verify_exp3_paired_robot_episode_v0_134.py`, `verify_complete_task_clock_and_video_v0_1.py`), fixed denominators, Wilson intervals, galleries |

Helper-only modules: `run_exp3_remote_pilot_jobs_v0_125.py` and `_v0_87.py` (imported for `emit`, `sha`, `probe`,
`ROBOT`), `freeze_exp3_franka_training_dataset_v0_2.py` (imported for `emit`, `read`, `sha`). The frozen dataset the
chain reads (`successful_trajectory_dataset_manifest.json`, `train_split.jsonl`, `heldout_split.jsonl`,
`frame_index.jsonl`, `dataset_quality_report.json`) has the layout and gate fields written by
`freeze_exp3_recovered_success_dataset_v0_141.py` (with `i2ia.tasks.full_task_dataset_coverage`).

Inputs not in this repository: the frozen dataset with its images and traces, the USD files of the constructed assets
named in the task specifications, the Franka USD named in the registered teacher commands, the openvla-oft checkout
(commit e4287e9, https://github.com/moojink/openvla-oft), the LIBERO-Spatial checkpoint with its file inventory, and the
v0.166 feature cache and head (steps 1-2).

Placeholders in `configs/` and `results/`: `${AFFORDCRAFT_RUNS}` (run root), `${AFFORDCRAFT_MODELS}` (checkpoint,
checkpoint inventory, official source copy), `${AFFORDCRAFT_PROJECT_ROOT}` (PartNet-Mobility source files),
`${AFFORDCRAFT_REPO}` (checkout of this repository), `${AFFORDCRAFT_ISAAC_PYTHON}`, `${OPENVLA_OFT_PYTHON}`.

Re-running. Every entry point re-verifies the registration before it starts: `check_definition` (lifecycle),
`train_exp3_oft_feedback_v0_167.py` and `cache_exp3_oft_features_v0_166.py` compare the sha256 of the configurations
with `freeze_receipt.json` next to them and the sha256 of every file listed in `code_files_sha256`. The registered map
lists all 97 files of the registered code set, including the ones not published here, and the summarizer differs from
its registered hash (see Provenance). To re-run: substitute the placeholders, restrict `code_files_sha256` to the
files of this release with their current hashes, and write `freeze_receipt.json` (keys `training_configuration_sha256`,
`evaluation_configuration_sha256`, `lifecycle_configuration_sha256`, `code_files_sha256`). The feature extractor also
checks the checkpoint files against the inventory and the openvla-oft files against `official_source_files_sha256`.

Environment of the executed run: training in Python 3.10 with torch 2.8.0+cu128 on one GPU (17 min); Isaac episodes
in an Isaac Sim 6 installation (the driver keeps the robot and action contract of its Isaac Sim 5.x version) with
numpy, scipy, Pillow, imageio and imageio-ffmpeg; the summarizer draws labels with the DejaVu Sans font. The paired
evaluation assigns the ten assets to GPUs 0-7 (`gpu_index` in the evaluation configuration) and waits until no
compute process runs on any GPU.

## Changes from the executed version

- Code: all published files are byte-identical to the registered files (sha256 in `configs/training_configuration.json`
  under `code_files_sha256`), except `tools/summarize_exp3_paired_learning_v0_141.py`, whose report headings and
  sentences were translated to English (AST identical apart from these strings). The files are not reformatted, so
  their hashes can be checked against the registration.
- Added `src/i2ia/contracts/`, the project's data-contract package: `i2ia/tasks/__init__.py` imports `evaluator.py`,
  which imports `TaskPredicate` and `TaskSpec` from `i2ia.contracts`. The package lies outside the code set registered
  for v0.167, so its hashes cannot be checked against the registration; both names are used only in annotations and
  no script of the chain calls the evaluator.
- Configurations and summary: absolute paths replaced by the placeholders above; the key holding the hash of an
  internal review record (under `closed_endpoint_evidence`) dropped from the evaluation configuration; re-serialized
  with the original settings (indent 2, sorted keys).
- Not published: the run's Chinese `final_report.md` (its table is reproduced above), receipts and lineage files.

Registered but not published (not used by the v0.167 chain):

- Earlier head generations, never executed by v0.167: `register_exp3_paired_learning_v0_141.py`,
  `run_exp3_paired_learning_lifecycle_v0_141.py`, `train_exp3_oft_robot_tasks_v0_134.py`, `serve_exp3_local_vla_v0_134.py`,
  `verify_exp3_actual_training_v0_138.py`, `register_exp3_precision_learning_v0_156/160/164/165.py`,
  `train_exp3_oft_precision_v0_156/160/164/165.py`, `serve_exp3_precision_vla_v0_156/160/164/165.py`,
  `run_exp3_precision_learning_v0_160/164/165.py`, `exp3_oft_precision_adapter_v0_165.py`,
  `register_exp3_recurrent_learning_v0_166.py`, `run_exp3_recurrent_learning_v0_166.py` (the lifecycle that ran step 1),
  `train_exp3_oft_recurrent_v0_166.py`, `verify_exp3_recurrent_training_v0_166.py`; diagnostic
  `diagnose_exp3_action_fit_v0_155.py`.
- Task modules not imported by any published script: `articulation_pose_evidence`, `authored_convex_collision`,
  `bounded_asset_curriculum`, `bounded_push_feedback_v0_3`, `diagnostic_budget` (and `_v0_2`, `_v0_3`),
  `dual_pad_source_contact`, `finite_joint_workcells`, `handle_teacher_geometry`, `handle_teacher_panel_geometry`,
  `inherited_asset_scope`, `joint_axis_consistency`, `pilot_protocol_family`, `planar_cover_geometry`,
  `quaternion_motion_readback`, `replacement_review_consensus`, `semantic_joint_task`, `triangle_box_collision`,
  `visual_review_admission`.

Outside the registered code set and therefore not available: the scripted-teacher collection runner of the ten tasks
and the admission step that produced the frozen dataset; `freeze_exp3_full_task_success_dataset_v0_133.py` (imported
by `freeze_exp3_recovered_success_dataset_v0_141.py`); `freeze_exp3_franka_real_contact_dataset_v0_1.py` (loaded only
by the `main` of `freeze_exp3_franka_training_dataset_v0_2.py`); `run_exp3_complete_task_contact_teacher_v0_125.py` and
`_v0_87.py` (launched only by the `register`/`execute` functions of the pilot-job helpers). The teacher primitives the
tasks share with the driver's disabled teacher branch are published in `i2ia.tasks` (`bounded_push_feedback*`,
`handle_teacher_*`, `prismatic_handle_teacher_v0_1`, `wrist_clear_push_height_v0_1`, `tool_envelope_contact`).

## Provenance

| Original file | Published path | sha256 of original | Change |
|---|---|---|---|
| cache_exp3_oft_features_v0_166.py | single_asset/tools/cache_exp3_oft_features_v0_166.py | `ba14c043bd998ff654da1c3da5979b5c2c2ca1d6cb3e6cadf24c0e142cd51126` | byte-identical |
| diagnose_exp3_training_feedback_v0_167.py | single_asset/tools/diagnose_exp3_training_feedback_v0_167.py | `5cdac62a451910edbc0adbe09edc76b47937262e7ffb933307658daff3dab140` | byte-identical |
| exp3_local_robot_policy_client_v0_134.py | single_asset/tools/exp3_local_robot_policy_client_v0_134.py | `67e42aadd94bb788eb96a7bcac3af4080b3db93c985cc0288b911f73c8936d18` | byte-identical |
| exp3_oft_feedback_adapter_v0_167.py | single_asset/tools/exp3_oft_feedback_adapter_v0_167.py | `7feb2248f97b2d1021eb23d5d4a85c0720a9972ec2503ee6943a8410ed95b969` | byte-identical |
| exp3_oft_joint_adapter_v0_134.py | single_asset/tools/exp3_oft_joint_adapter_v0_134.py | `22165a43b5f949d95d1462d5dc5d2169b34d5b66e513453629387bd63a054506` | byte-identical |
| exp3_oft_precision_adapter_v0_156.py | single_asset/tools/exp3_oft_precision_adapter_v0_156.py | `8942d313d95a177720017b83cd69d09d8ecaceda21e19707d8535fa3f6913ec6` | byte-identical |
| exp3_oft_precision_adapter_v0_160.py | single_asset/tools/exp3_oft_precision_adapter_v0_160.py | `edd41d18499753c4e1673ec7dabacbec774e6c98c229aff0da8a2b863e22bec2` | byte-identical |
| exp3_oft_precision_adapter_v0_164.py | single_asset/tools/exp3_oft_precision_adapter_v0_164.py | `fb346ddea46a9d32a326f4ea0162faa86a40eae002f309e276960ba8616c3d08` | byte-identical |
| exp3_oft_recurrent_adapter_v0_166.py | single_asset/tools/exp3_oft_recurrent_adapter_v0_166.py | `562b3b97bd7ea61ebe39846aaf09669fc302f5cfbc0b37059722669853918392` | byte-identical |
| exp3_training_instruction_contract_v0_134.py | single_asset/tools/exp3_training_instruction_contract_v0_134.py | `ee46f1c3844e67c832c25bef0ab4d7076ccc809c124e9463771211555261dc92` | byte-identical |
| freeze_exp3_franka_training_dataset_v0_2.py | single_asset/tools/freeze_exp3_franka_training_dataset_v0_2.py | `2b8a03e9f4da20fb76cc60ce01ec494f4f47c1f1687863c4ceb229a1b54deb76` | byte-identical |
| freeze_exp3_recovered_success_dataset_v0_141.py | single_asset/tools/freeze_exp3_recovered_success_dataset_v0_141.py | `e2f9e6a727934086272063370384242c5cf12a2a58a6bcd64f608d85f0f1e8cd` | byte-identical |
| register_exp3_feedback_learning_v0_167.py | single_asset/tools/register_exp3_feedback_learning_v0_167.py | `e50209d93b0669bd4ce3af93bdda3377eb015c67bcaa8a0259263b0ff2239945` | byte-identical |
| run_exp3_complete_task_local_policy_v0_134.py | single_asset/tools/run_exp3_complete_task_local_policy_v0_134.py | `47efef11ad2e6e353566de5f07d779cbbea58b9f43b3fc8fd3a186b3646233b1` | byte-identical |
| run_exp3_feedback_learning_v0_167.py | single_asset/tools/run_exp3_feedback_learning_v0_167.py | `c0d97ff5876e1f66bf6a701bd3b018d411859645ca83090bc3b2ecc7368705a0` | byte-identical |
| run_exp3_paired_policy_batch_v0_134.py | single_asset/tools/run_exp3_paired_policy_batch_v0_134.py | `1b21115a05bcbf8c73dd0a0a12ae768222a79163c04bb108ff5ad17db2c00ddd` | byte-identical |
| run_exp3_precision_learning_v0_156.py | single_asset/tools/run_exp3_precision_learning_v0_156.py | `b2187567fe07b9afca193ef9b7e89f96ba794cbb8e0db2ad0d7546c8e979947e` | byte-identical |
| run_exp3_remote_pilot_jobs_v0_125.py | single_asset/tools/run_exp3_remote_pilot_jobs_v0_125.py | `5c24d31697e8a6d82ae4fea09bd1875917e483aceed93e9d8a1e980bd272bfd0` | byte-identical |
| run_exp3_remote_pilot_jobs_v0_87.py | single_asset/tools/run_exp3_remote_pilot_jobs_v0_87.py | `ee56d3341da6d68c6e9f3302ed6bda542c6563f7924fb831c75384279d7f9b2f` | byte-identical |
| serve_exp3_feedback_vla_v0_167.py | single_asset/tools/serve_exp3_feedback_vla_v0_167.py | `5e7ace6fa8b3a599236d7123c67aed7e5ca982c267975e8980a6e699daeb80d1` | byte-identical |
| serve_exp3_recurrent_vla_v0_166.py | single_asset/tools/serve_exp3_recurrent_vla_v0_166.py | `c60d280165c268aa4cd1c86f0d65d577a9464b5e2c122be2cf4a6b049da6db23` | byte-identical |
| summarize_exp3_paired_learning_v0_141.py | single_asset/tools/summarize_exp3_paired_learning_v0_141.py | `834458642478c55c2fd30197efea5059d447cd4a57adcb24147d8304834ccdc8` | report strings translated to English |
| train_exp3_oft_feedback_v0_167.py | single_asset/tools/train_exp3_oft_feedback_v0_167.py | `a59f064b6f2393aa2c59c034957f0e9fd6f5f1844850cf400f9319f79ed24459` | byte-identical |
| verify_complete_task_clock_and_video_v0_1.py | single_asset/tools/verify_complete_task_clock_and_video_v0_1.py | `4fb97277a0f380d001e19b51e3b5bf86f1b79f13ff0b0fda3a57f0bb8098d764` | byte-identical |
| verify_exp3_feedback_training_v0_167.py | single_asset/tools/verify_exp3_feedback_training_v0_167.py | `b2dedc3577bbbc24e46e2dca3d559e93469effdb5e5b0a173e07d0d784304323` | byte-identical |
| verify_exp3_paired_robot_episode_v0_134.py | single_asset/tools/verify_exp3_paired_robot_episode_v0_134.py | `dcd26da1509645b52d2ef2f3e26784682773a9d7106d93dfbf144fae076ec622` | byte-identical |
| \_\_init\_\_.py | single_asset/src/i2ia/tasks/\_\_init\_\_.py | `0724f6ac87f88ac81cc0e0855c96c1a0b06c94445873eaee7b95dc1a0d2ce84d` | byte-identical |
| bounded_push_feedback.py | single_asset/src/i2ia/tasks/bounded_push_feedback.py | `5e7ddf7165a2e4f05d7a202f6489105b23f1294bcff2fb057be8153e5e520e2a` | byte-identical |
| bounded_push_feedback_v0_2.py | single_asset/src/i2ia/tasks/bounded_push_feedback_v0_2.py | `f491a1cb4a3e99187a4c88df1fa8d1eeed14470dff43cfd969524dd1a120ceeb` | byte-identical |
| complete_robot.py | single_asset/src/i2ia/tasks/complete_robot.py | `e094022aa4b606d4257a1d73f94bff3c7f0ce8b4d0c61697ac034256b28f554c` | byte-identical |
| complete_robot_precision.py | single_asset/src/i2ia/tasks/complete_robot_precision.py | `a727d7f78b3926e23d3a77fed12b3a9314a19f8251c1e29b0aea0de526339549` | byte-identical |
| complete_task_clock.py | single_asset/src/i2ia/tasks/complete_task_clock.py | `54c95c129aa0007915ab2e5425475b6087ae0387420c49411c08bfb0dc3df951` | byte-identical |
| contact_geometry.py | single_asset/src/i2ia/tasks/contact_geometry.py | `9c6b176b83e452e87eb34628d50a361d4fab8d0846ab3959804eaa99eeebcbea` | byte-identical |
| evaluator.py | single_asset/src/i2ia/tasks/evaluator.py | `0fafd77bc182b1a6ad7a9e28d28deba764e7d633401299625581d1bc7691bd22` | byte-identical |
| fixture_contact_guard_v0_1.py | single_asset/src/i2ia/tasks/fixture_contact_guard_v0_1.py | `45d76c0c5c7a0bd4db69e0b56a4f702f066832818af02ea5610ed128d190417f` | byte-identical |
| full_task_dataset_coverage.py | single_asset/src/i2ia/tasks/full_task_dataset_coverage.py | `d2e50040074d9176a567b4b0924312bdc9a4c16ab8be6dd8b250529398c9ead4` | byte-identical |
| full_task_episode_termination_v0_1.py | single_asset/src/i2ia/tasks/full_task_episode_termination_v0_1.py | `0c5c3ee53064a1fa98ea1d5c413d23dfa3c44027fb9c1d9d264df6407a3c259d` | byte-identical |
| full_task_training_admission.py | single_asset/src/i2ia/tasks/full_task_training_admission.py | `7eca6a726ded918f942dc74a8ebcd6269f4ef8106fbe844eff8c4e9372ba258c` | byte-identical |
| full_task_training_admission_v0_2.py | single_asset/src/i2ia/tasks/full_task_training_admission_v0_2.py | `9cd98cae04811662c541abbdd719a72264b098d89f539fb2e287fd56a76c04fc` | byte-identical |
| handle_teacher_release_geometry_v0_1.py | single_asset/src/i2ia/tasks/handle_teacher_release_geometry_v0_1.py | `1b108c79bfa1a6f0ab68bab238ea7a7d9d75eed3722bf3253f09c1cd411c689a` | byte-identical |
| handle_teacher_triangle_geometry.py | single_asset/src/i2ia/tasks/handle_teacher_triangle_geometry.py | `29a80901d98c5bc72324350b583c9e151e7761efe24cf135c852a833b04d3f8d` | byte-identical |
| prismatic_handle_teacher_v0_1.py | single_asset/src/i2ia/tasks/prismatic_handle_teacher_v0_1.py | `0200e4e9f8a3d56088541c9c07da996e7657b17f91ef76a8d66433743ec7afcc` | byte-identical |
| raw_camera_framing.py | single_asset/src/i2ia/tasks/raw_camera_framing.py | `86787501da495afd7b4ee381ca028b9c9a46bce0fdc82dc1f5099a794c17aeda` | byte-identical |
| robot_action_limits_v0_1.py | single_asset/src/i2ia/tasks/robot_action_limits_v0_1.py | `4b12374c6b1303d76b62599145b3c7aae486255eea1ee696ac9fb9439ca349f7` | byte-identical |
| robot_joint_reference_rate_v0_1.py | single_asset/src/i2ia/tasks/robot_joint_reference_rate_v0_1.py | `a942f2949bcaa72a07bab24fe0c3b8286c58b72d5a39cbdda5bb1b0a6a1d55de` | byte-identical |
| robot_pose_interpolation.py | single_asset/src/i2ia/tasks/robot_pose_interpolation.py | `e6c2cad65e30c047d0604e042d14796116631d1aa970fb7f9ba4a9a65b912b4e` | byte-identical |
| support_contact_lifecycle.py | single_asset/src/i2ia/tasks/support_contact_lifecycle.py | `0fb3badb908727fab87860415e84366924e332e7f73aa6507f497ebbc537f93f` | byte-identical |
| tool_envelope_contact.py | single_asset/src/i2ia/tasks/tool_envelope_contact.py | `e2cb2bdd1d63e3a30cf8d0529aaa108170724623f1075990a242328e4fcb6498` | byte-identical |
| usd_affine_points.py | single_asset/src/i2ia/tasks/usd_affine_points.py | `52e85bfd13f919493a22d875305133ebe132289ea14546d6813e86cb7d4a866f` | byte-identical |
| workcell_root_frame.py | single_asset/src/i2ia/tasks/workcell_root_frame.py | `25fde176e498669f4e4c8a7747dac62106ae0ea200592a2a6be21c1d1574c433` | byte-identical |
| wrist_clear_push_height_v0_1.py | single_asset/src/i2ia/tasks/wrist_clear_push_height_v0_1.py | `4f1b17e56afc42f2ca9e75f7a68dfb5b121fd1b919c098f517ae7d77e10d51e0` | byte-identical |
| training_configuration.json | single_asset/configs/training_configuration.json | `a199a8333adc07323907a10e22c6fe8faa10be6eb875491a179a3e8de40ee105` | paths -> placeholders |
| evaluation_configuration.json | single_asset/configs/evaluation_configuration.json | `8577e896099c3cd6f263573269480a0000172b8d705027fd18124fec266fb564` | paths -> placeholders; one review-record key dropped |
| lifecycle_configuration.json | single_asset/configs/lifecycle_configuration.json | `2c4afb1829c72e721169b9932a97e0f637ec0bf5de019ad823c64343f6ffd5e9` | paths -> placeholders |
| summary.json | single_asset/results/summary.json | `5b8f85f3657f4fed0ec2f8a25a6f11210c9574bd45409589703f6058c8fa9fca` | paths -> placeholders |
| contracts/__init__.py | single_asset/src/i2ia/contracts/__init__.py | `590d5aeba9b8456615f39cf017a2aaef05da7d44f56fc2d53221f7229fe16033` | unchanged (not in the registered set) |
| contracts/_validation.py | single_asset/src/i2ia/contracts/_validation.py | `9300547e63a2fbb29bdeb75a9c84e61855587481f388a7b6805d29cda50c8944` | unchanged (not in the registered set) |
| contracts/artifacts.py | single_asset/src/i2ia/contracts/artifacts.py | `0d6b20497df6d48e236c3b86ad3239b5e0bc84163795a1dc4006c207ba7484c4` | unchanged (not in the registered set) |
| contracts/packets.py | single_asset/src/i2ia/contracts/packets.py | `cac4987a54ee90c92ae893d985334fd8325a1f2aa119c94d411356971848aea1` | unchanged (not in the registered set) |
| contracts/scene.py | single_asset/src/i2ia/contracts/scene.py | `cbf2dbb694123487c0edcd2f74820295d72f3a4f83a4c5a4f747111d1c8056e9` | unchanged (not in the registered set) |
| contracts/task.py | single_asset/src/i2ia/contracts/task.py | `9f445bab254558525ce8a98bfc6afd6a9a8022e42e803a8ac5e5da489442bfc9` | unchanged (not in the registered set) |
