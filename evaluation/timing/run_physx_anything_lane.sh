#!/usr/bin/env bash
# Clean-GPU timing: one PhysX-Anything lane on one exclusive RTX 5090 (the wrapper code of the registered run copied to
# <run root>/physx-anything/code with its run root and input set pointed at this run; the environments and stages of the
# registered lane script): stage A (VLM) over the lane, stage B (decoder+split+export), receipt; lane.wall.json.
# usage: run_physx_anything_lane.sh <lane> <gpu> [lanes=4]
# environment: TIMING_ROOT (run root), PHYSX_ANYTHING_VLM_PYTHON (stage A), PHYSX_ANYTHING_GEOM_PYTHON (stage B)
set -u
LANE=$1; GPU=$2; LANES=${3:-4}
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
REV=$RT/physx-anything; CODE=$REV/code; L=$REV/lane-$LANE
VLA_PY=${PHYSX_ANYTHING_VLM_PYTHON:-python}
GEOM_PY=${PHYSX_ANYTHING_GEOM_PYTHON:-python}
mkdir -p $L
exec >> $L/lane.log 2>&1
{ [ -s $REV/configuration.json ] && [ -s $L/input_manifest.jsonl ]; } || { echo "configuration or lane manifest missing: refused"; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU"; exit 3; }
echo "$(date -Is) lane=$LANE/$LANES gpu=$GPU host=$(hostname) stage A"
tA=$(date +%s.%N)
"$VLA_PY" "$CODE/physx_anything_vlm_stage.py" --lane-dir "$L"; rA=$?; echo "$(date -Is) stage A exit $rA"
tB=$(date +%s.%N)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$GEOM_PY" "$CODE/physx_anything_geom_stage.py" --lane-dir "$L"; rB=$?; echo "$(date -Is) stage B exit $rB"
tC=$(date +%s.%N)
"$VLA_PY" "$CODE/lane_receipt.py" "$L" "$LANE" "$LANES" "$GPU"; echo "$(date -Is) receipt exit $?"
echo "{\"method\": \"physx-anything\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"stage_a_start_unix\": $tA, \"stage_b_start_unix\": $tB, \"end_unix\": $tC, \"stage_a_exit\": $rA, \"stage_b_exit\": $rB}" > $L/lane.wall.json
echo "$(date -Is) lane $LANE done"
