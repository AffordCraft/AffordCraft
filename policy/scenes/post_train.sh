#!/usr/bin/env bash
# One training round. Usage: post_train.sh ROUND  (ROUND=0: teacher data collect20; ROUND=1: collect20 + dagger20;
# round 2 is launched by run_round2.sh with TAGS_R1 and DS_EXTRA set).
# audit -> dataset (teacher + dagger episodes, dynamics filter 0.75 rad, both frame streams) -> freeze receipt ->
# frozen OpenVLA-OFT feature cache (two images -> 8192-d, two shards) -> recurrent head 24k steps -> completion.json.
# Interpreters: numpy/json steps on AFFORDCRAFT_RUNTIME_PYTHON; feature cache / head training on OPENVLA_OFT_PYTHON.
# $S/vla/training_configuration.json is written once by vla_head/make_config.py.
set -euo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla
ROUND=${1:-0}; OUT=$S/v20_vla/round$ROUND
PYA=${AFFORDCRAFT_RUNTIME_PYTHON:-python}
PYV=${OPENVLA_OFT_PYTHON:?set OPENVLA_OFT_PYTHON to the openvla-oft environment python}
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; STEPS=${STEPS:-24000}; MAX_HDELTA=${MAX_HDELTA:-0.75}
TAGS=${TAGS:-"collect20"}; [ "$ROUND" -ge 1 ] && TAGS=${TAGS_R1:-"collect20 dagger20"}
if [ -e "$OUT/completion.json" ]; then echo "already complete: $OUT"; exit 0; fi
mkdir -p "$OUT"; echo "post_train round $ROUND started $(date -Is) tags: $TAGS" >> "$OUT/execution.log"
# the 3rd argument is recorded as scene_framework; its "_v20" suffix yields the executed method id / collection tag
$PYA "$F/vla_head/pipeline_steps.py" config "$S/vla/training_configuration.json" "$OUT/configuration.json" "${FRAMEWORK_LABEL:-framework_v20}" "$STEPS" 2 | tee -a "$OUT/execution.log"
RUNS=()
for T in $TAGS; do
  if [ "$T" = collect20 ]; then $PYA "$F/vla_head/pipeline_steps.py" audit "$S/scenes/$T" "$OUT/video_audit_$T.json" | tee -a "$OUT/execution.log"
  else $PYA "$F/vla_head/pipeline_steps.py" audit "$S/scenes/$T" "$OUT/video_audit_$T.json" 2>&1 | tee -a "$OUT/execution.log" || echo "audit coverage check non-fatal for $T $(date -Is)" >> "$OUT/execution.log"; fi
  mapfile -t R < <(find "$S/scenes/$T" -mindepth 1 -maxdepth 1 -type d -name '*_w*' | sort); RUNS+=("${R[@]}")
done
cp "$OUT/video_audit_collect20.json" "$OUT/video_audit.json"
$PYA "$F/vla_head/build_dataset.py" --runs "${RUNS[@]}" --output "$OUT/dataset" --max-per-scene 100000 --seed 20 --max-horizon-delta "$MAX_HDELTA" --modes teacher,dagger ${DS_EXTRA:-} | tee -a "$OUT/execution.log"
$PYA "$F/vla_head/pipeline_steps.py" freeze "$OUT/dataset/manifest.json" "$OUT/dataset_frozen_receipt.json" | tee -a "$OUT/execution.log"
echo "feature_cache_start $(date -Is)" >> "$OUT/execution.log"
REUSE=()
PREV=$S/v20_vla/round$((ROUND-1))
if [ "$ROUND" -ge 1 ]; then   # reuse the previous round's features for identical frames (full cache), else round 0's, else the round-0 bootstrap subset
  if [ -f "$PREV/features/shard_0_manifest.json" ] && [ -f "$PREV/features/shard_1_manifest.json" ]; then REUSE=(--reuse-features "$PREV/features")
  elif [ -f "$S/v20_vla/round0/features/shard_0_manifest.json" ] && [ -f "$S/v20_vla/round0/features/shard_1_manifest.json" ]; then REUSE=(--reuse-features "$S/v20_vla/round0/features")
  elif ls "$S/v20_vla/round0/features_bootstrap"/shard_*_manifest.json >/dev/null 2>&1; then REUSE=(--reuse-features "$S/v20_vla/round0/features_bootstrap"); fi
fi
echo "reuse: ${REUSE[*]:-none} $(date -Is)" >> "$OUT/execution.log"
$PYV "$F/vla_head/cache_features.py" --config "$OUT/configuration.json" --dataset "$OUT/dataset" --output "$OUT/features" --gpu "$GPU_A" --shard 0 --num-shards 2 "${REUSE[@]}" > "$OUT/features_shard0.log" 2>&1 &
PA=$!
$PYV "$F/vla_head/cache_features.py" --config "$OUT/configuration.json" --dataset "$OUT/dataset" --output "$OUT/features" --gpu "$GPU_B" --shard 1 --num-shards 2 "${REUSE[@]}" > "$OUT/features_shard1.log" 2>&1 &
PB=$!
wait $PA; wait $PB
test -f "$OUT/features/shard_0_manifest.json" -a -f "$OUT/features/shard_1_manifest.json"
echo "feature_cache_done $(date -Is)" >> "$OUT/execution.log"
$PYV "$F/vla_head/train_head.py" --config "$OUT/configuration.json" --features "$OUT/features" --dataset "$OUT/dataset" --output "$OUT/training" --gpu "$GPU_A" > "$OUT/training.log" 2>&1
echo "training_done $(date -Is)" >> "$OUT/execution.log"
$PYA "$F/vla_head/pipeline_steps.py" complete "$OUT" | tee -a "$OUT/execution.log"
echo "post_train round $ROUND done $(date -Is)" >> "$OUT/execution.log"
