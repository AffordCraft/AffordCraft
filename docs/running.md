# Running AffordCraft

All commands run from the repository root with the variables of `docs/setup.md` set. `$SELECTOR` is the selector
interpreter; `run_pipeline.py` starts the construction and physics processes with the runtime and Isaac interpreters
itself. Runs never overwrite an existing output folder.

## 1. Visual index

```bash
$SELECTOR scripts/build_visual_index.py --output runs/index/dinov2 --gpu 0
$SELECTOR scripts/build_visual_index.py --encoder clip --output runs/index/clip --gpu 0   # encoder-replacement study only
```

The index holds one normalized feature per catalog preview and records the catalog and encoder hashes. The CLIP index
folder name must contain `clip`; the pipeline refuses a CLIP index for any other condition.

## 2. Inputs

```bash
python scripts/prepare_inputs.py single --open-images /datasets/openimages_v7 --output runs/inputs/single_inputs.json
python scripts/prepare_inputs.py subset --single runs/inputs/single_inputs.json --output runs/inputs/subset_200.json
python scripts/prepare_inputs.py category --single runs/inputs/single_inputs.json --output runs/inputs/category_548.json
python scripts/prepare_inputs.py cluttered --coco /datasets/coco/val2017 --output runs/inputs/scenes
```

## 3. Calibration and freezing

A formal run needs a frozen definition, written only after two calibrations pass:

```bash
# seven known positive and negative physics controls (free box, pinned free box, mounted hinge, unsupported mount,
# penetrating box, fixed free box, locked part); make_physics_fixtures.py needs pxr (Isaac Sim or usd-core)
$AFFORDCRAFT_RUNTIME_PYTHON scripts/make_physics_fixtures.py --output runs/calibration/fixtures
python scripts/run_physics_job.py --jobs runs/calibration/fixtures/jobs.json --output runs/calibration/physics --gpu 0

# end-to-end calibration on a few inputs (at least one must pass)
$SELECTOR scripts/run_pipeline.py --calibration --limit 3 --inputs runs/inputs/subset_200.json \
    --index runs/index/dinov2 --output runs/calibration/pipeline --gpu 0

# frozen definition: inputs hash, construction and evaluation policies, hashes of every code and model file
python scripts/freeze_experiment.py --inputs runs/inputs/single_inputs.json --output runs/definitions/exp1.json \
    --physics-controls runs/calibration/physics/execution_receipt.json \
    --fixtures runs/calibration/fixtures/jobs.json \
    --pipeline-calibration runs/calibration/pipeline/summary.json --shard-count 40
```

A formal run refuses to start if any locked file changed, if the inputs differ from the frozen ones, or if the shard
count or input scope differs from the definition. The construction and evaluation policies are fixed in
`affordcraft/contracts.py`: at most 20 candidates, selector batches of 5, one repair per candidate, selector confidence
>= 0.55 and geometry similarity >= 0.20, a 600 s construction budget per candidate; physical evaluation for 360 steps at
1/120 s with the tail criteria of `EvaluationPolicy`.

## 4. Single-object experiment (2,000 photographs)

```bash
for k in $(seq 0 39); do                  # shards are independent; run them in parallel on free GPUs
  $SELECTOR scripts/run_pipeline.py --inputs runs/inputs/single_inputs.json --index runs/index/dinov2 \
      --freeze runs/definitions/exp1.json --shard-index $k --shard-count 40 --gpu 0 \
      --geometry-cache runs/geometry_cache --output runs/exp1/shard-$k
done
```

Input `i` belongs to shard `i mod 40`. Each shard writes `per_input.jsonl` (one terminal record per input: the grounding,
every candidate considered with its selection verdict, construction, repair and checks, the selected asset and the
final validation), `cases/<input>/result.json`, the model calls, the construction logs and the physics evidence. A shard
stops at the first infrastructure fault (exit code 2) and is continued in a new folder, keeping its terminal records:

```bash
$SELECTOR scripts/run_pipeline.py ... --shard-index $k --shard-count 40 \
    --resume-from runs/exp1/shard-$k --output runs/exp1/shard-$k-r1
```

The geometry cache stores collision decompositions keyed by the source geometry and the decomposition settings; it
changes run time only.
CoACD is not bitwise reproducible between processes, so a cold cache can yield slightly different hulls.

Merge the final attempt of every shard and export the pass rate with its Wilson interval and per-category counts:

```bash
python scripts/merge_shards.py --inputs runs/inputs/single_inputs.json --runs <final attempt of each shard> \
    --output runs/exp1/merged
python scripts/export_final_results.py --run runs/exp1/merged --inputs runs/inputs/single_inputs.json \
    --output runs/exp1/final
```

Merging the paper's 40 shards this way reproduces its per-case records exactly (1,703 of 2,000 pass).

## 5. One-factor study

Each condition removes or replaces one component (`affordcraft.contracts.VARIANTS`): `full`, `encoder_replacement`,
`without_task_condition`, `without_multimodal_selection`, `without_scale_adaptation`,
`without_articulation_adaptation`, `without_physics_reselection`, `without_repair`, `without_installation_evidence`,
`without_decomposition_cascade`. The paper runs all ten on the 200-input subset (4 shards each) and `full`,
`without_installation_evidence` and `without_decomposition_cascade` on the 548 Window, Door and StorageFurniture
inputs (8 shards each), with the frozen `full` method re-run inside the study as the paired reference.

```bash
for v in full encoder_replacement without_task_condition without_multimodal_selection without_scale_adaptation \
         without_articulation_adaptation without_physics_reselection without_repair \
         without_installation_evidence without_decomposition_cascade; do
  index=runs/index/dinov2; [ $v = encoder_replacement ] && index=runs/index/clip
  python scripts/freeze_experiment.py --inputs runs/inputs/subset_200.json --variant $v --shard-count 4 \
      --output runs/definitions/study-$v.json --physics-controls ... --fixtures ... --pipeline-calibration ...
  for k in 0 1 2 3; do
    $SELECTOR scripts/run_pipeline.py --inputs runs/inputs/subset_200.json --index $index \
        --freeze runs/definitions/study-$v.json --shard-index $k --shard-count 4 --gpu 0 \
        --geometry-cache runs/geometry_cache --output runs/study/$v/shard-$k
  done
done
```

The gate scores physics, not identity: `analysis/ablation_task_fidelity.py` counts an input as correct only when it passes
with an asset of the requested category, and `analysis/tables/tables_ablation.py` writes Table 2 with exact McNemar
tests on the paired inputs.

## 6. Cluttered images (50 COCO images, 237 reference objects)

```bash
# object discovery on the whole image (no reference boxes); writes detected_inputs.json and every model call
$SELECTOR scripts/detect_scene_objects.py --inputs runs/inputs/scenes/model_inputs.json --catalog $AFFORDCRAFT_CATALOG \
    --output runs/exp2/detection --gpu 0

python scripts/freeze_experiment.py --inputs runs/exp2/detection/detected_inputs.json --cluttered --shard-count 8 \
    --output runs/definitions/exp2.json --physics-controls ... --fixtures ... --pipeline-calibration ...
for k in $(seq 0 7); do
  $SELECTOR scripts/run_pipeline.py --scene-mode --inputs runs/exp2/detection/detected_inputs.json \
      --index runs/index/dinov2 --freeze runs/definitions/exp2.json --shard-index $k --shard-count 8 --gpu 0 \
      --geometry-cache runs/geometry_cache --output runs/exp2/shard-$k
done
python scripts/merge_shards.py --inputs runs/exp2/detection/detected_inputs.json --runs <final attempts> \
    --output runs/exp2/merged
python scripts/evaluate_scenes.py --reference runs/inputs/scenes/evaluation_only.json \
    --detections runs/exp2/detection/detected_inputs.json --results runs/exp2/merged/per_input.jsonl \
    --output runs/exp2/scores
```

In scene mode every detection is constructed as a single-object input with its automatic box; the backend checks each
box against the recorded model call before construction. The paper's detection ran in two passes: the first pass (the
same routine without loop stopping, reply salvage and the box filters) failed on 18 images, whose replies hit the token
bound in a repetition loop; the second pass re-detected only those images (`--retain-from <first pass>`), and
`scripts/check_retained_invariance.py` shows that its rules leave the 32 retained outputs unchanged. Running
`detect_scene_objects.py` on all 50 images therefore corresponds to the second pass.

## 7. External methods, timing, tables

- External image-to-asset routes: `baselines/` (running each method under the full-RGB protocol and exporting its
  native manifest) and `evaluation/` (the common physical gate in the native and adapted conditions, aggregation,
  clean-GPU timing, API cost).
- Composed scenes and single-asset tasks with trained action heads: `policy/`.
- Every table and data figure of the paper: `analysis/`.
