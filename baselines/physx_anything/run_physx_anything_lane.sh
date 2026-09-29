#!/usr/bin/env bash
# One PhysX-Anything lane (official code + official weights) over the registered 200-input subset, round-robin partition
# (subset index % PA_LANES == PA_LANE). Env: PA_LANE, PA_LANES, PA_GPU (physical CUDA index), optional PA_ONLY (single
# source_id, smoke test), PA_LANE_DIR (override output dir); PHYSX_ANYTHING_RUN (run root), PHYSX_ANYTHING_VLM_PYTHON,
# PHYSX_ANYTHING_GEOM_PYTHON and the variables of common.py. Waits until the GPU shows < 1 GiB used (never kills anything),
# then stage A (VLM, vla env) over all lane cases, then stage B (decoder+split+export, geom5090 env), then the receipt.
# Re-running is idempotent: finished cases are skipped, interrupted case dirs are moved to interrupted/ (never overwritten).
set -u
LANE=${PA_LANE:?set PA_LANE}; LANES=${PA_LANES:?set PA_LANES}; GPU=${PA_GPU:?set PA_GPU}
REV=${PHYSX_ANYTHING_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/physx-anything/subset200}
CODE=$REV/code
L=${PA_LANE_DIR:-$REV/lane-$LANE}
VLA_PY=${PHYSX_ANYTHING_VLM_PYTHON:-python}
GEOM_PY=${PHYSX_ANYTHING_GEOM_PYTHON:-python}
mkdir -p "$L"
LOG=$L/lane.log
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Is) lane=$LANE/$LANES gpu=$GPU host=$(hostname) dir=$L"
[ -s "$REV/configuration.json" ] || { echo "configuration.json missing: refuse to run"; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU
"$VLA_PY" "$CODE/make_lane_manifest.py" "$L" "$LANE" "$LANES" || { echo "$(date -Is) lane manifest failed"; exit 1; }
# wait for the GPU to be free (< 1 GiB used); poll every 60 s; foreign jobs are never touched
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  [ -n "$USED" ] && [ "$USED" -lt 1024 ] && break
  echo "$(date -Is) gpu $GPU busy (${USED} MiB used), waiting"; sleep 60
done
echo "$(date -Is) gpu $GPU free (${USED} MiB); stage A"
ONLY=(); [ -n "${PA_ONLY:-}" ] && ONLY=(--only "$PA_ONLY")
"$VLA_PY" "$CODE/physx_anything_vlm_stage.py" --lane-dir "$L" "${ONLY[@]}"; echo "$(date -Is) stage A exit $?"
echo "$(date -Is) stage B"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$GEOM_PY" "$CODE/physx_anything_geom_stage.py" --lane-dir "$L" "${ONLY[@]}"; echo "$(date -Is) stage B exit $?"
"$VLA_PY" "$CODE/lane_receipt.py" "$L" "$LANE" "$LANES" "$GPU"; echo "$(date -Is) receipt exit $?"
echo "$(date -Is) lane $LANE done"
