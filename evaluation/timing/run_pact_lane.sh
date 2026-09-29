#!/bin/bash
# Clean-GPU timing: one PAct lane on one exclusive RTX 5090 = the registered run's lane driver with this run root,
# the registered lane runner and mask adapter (copied to <run root>/pact/code with the run id replaced) and the 32-input
# sample as subset: automatic part-mask adapter for the lane's cases, official infer_imgs.py in the attempt loop,
# finalize; lane.wall.json. usage: run_pact_lane.sh <lane> <gpu> [min_free_mib]
# environment: TIMING_ROOT, PACT_PYTHON, PACT_HOME (official PAct checkout), PACT_TOOLS (directory of the lane runner and
# mask adapter; PACT_LANE_RUNNER / PACT_MASK_ADAPTER override the file names),
# PACT_GDINO / PACT_SAM (Grounding DINO tiny and SAM ViT-B checkpoints), PACT_PYTHONPATH (optional extra site dir),
# AFFORDCRAFT_PROJECT_ROOT (resolves the image paths of the input manifest)
set -u
LANE=$1; GPU=$2; MINFREE=${3:-30000}
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
ROOT=$RT/pact
PY=${PACT_PYTHON:-python}
T=${PACT_TOOLS:-$ROOT/code}; LD=$ROOT/lane-$LANE; mkdir -p $LD/logs
RUNNER=${PACT_LANE_RUNNER:-$T/lane_runner.py}; ADAPTER=${PACT_MASK_ADAPTER:-$T/mask_adapter.py}   # file names of baselines/pact
MODELS=${AFFORDCRAFT_MODELS:-${AFFORDCRAFT_PROJECT_ROOT:-workspace}/models}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${PACT_PYTHONPATH:+$PACT_PYTHONPATH:}${PYTHONPATH:-}
LIST=$RT/inputs/sample32.json
MAN=$RT/inputs/input_manifest.jsonl
[ -s $ROOT/configuration.json ] || { echo "no configuration" >> $LD/logs/driver.log; exit 2; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU" >> $LD/logs/driver.log; exit 3; }
echo "{\"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"driver_pid\": $$, \"python\": \"$PY\", \"started_utc\": \"$(date -u +%FT%TZ)\", \"purpose\": \"clean-GPU timing PAct lane driver (mask adapter + official infer_imgs.py + finalize)\"}" >> $LD/logs/pids.json
T0=$(date +%s.%N)
LANE_IDS=$($PY -c "import json;print(','.join(json.load(open('$ROOT/configuration.json'))['lane_partition']['lanes']['$LANE']))")
run_adapter() {
  echo "[driver] adapter start $(date -u +%FT%TZ) unix $(date +%s.%N)" >> $LD/logs/driver.log
  $PY $ADAPTER --subset $LIST --manifest $MAN --project-root ${AFFORDCRAFT_PROJECT_ROOT:-workspace} --pact-src ${PACT_HOME:-pact} \
    --gdino ${PACT_GDINO:-$MODELS/grounding-dino-tiny} --sam ${PACT_SAM:-$MODELS/sam-vit-base} \
    --masks-root $ROOT/masks --ids "$LANE_IDS" >> $LD/logs/adapter.log 2>&1
  echo "[driver] adapter rc=$? $(date -u +%FT%TZ) unix $(date +%s.%N)" >> $LD/logs/driver.log
}
for attempt in 1 2 3 4 5 6 7 8; do
  run_adapter
  $PY $RUNNER prepare --root $ROOT --lane $LANE >> $LD/logs/driver.log 2>&1
  npend=$($PY -c "import json;print(len(json.load(open('$LD/pending.json'))['pending']))")
  echo "[driver] attempt $attempt pending=$npend $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  [ "$npend" = "0" ] && break
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1)
  while [ "$free" -lt "$MINFREE" ]; do echo "[driver] GPU $GPU free ${free} MiB < $MINFREE, waiting 60 s" >> $LD/logs/driver.log; sleep 60; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1); done
  $PY $RUNNER infer --root $ROOT --lane $LANE > $LD/logs/infer_attempt_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
  IPID=$!
  echo "{\"lane\": $LANE, \"gpu\": $GPU, \"infer_pid\": $IPID, \"attempt\": $attempt, \"started_utc\": \"$(date -u +%FT%TZ)\"}" >> $LD/logs/pids.json
  wait $IPID; rc=$?
  echo "[driver] infer attempt $attempt rc=$rc $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  $PY $RUNNER mark-interrupted --root $ROOT --lane $LANE >> $LD/logs/driver.log 2>&1
  $PY $RUNNER finalize --root $ROOT --lane $LANE --gpu $GPU >> $LD/logs/driver.log 2>&1
done
$PY $RUNNER finalize --root $ROOT --lane $LANE --gpu $GPU >> $LD/logs/driver.log 2>&1
T1=$(date +%s.%N)
echo "{\"method\": \"pact\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"start_unix\": $T0, \"end_unix\": $T1, \"results\": $(ls $LD/cases/*/result.json 2>/dev/null | wc -l)}" > $LD/lane.wall.json
echo "[driver] lane $LANE done $(date -u +%FT%TZ): results $(ls $LD/cases/*/result.json 2>/dev/null | wc -l)" >> $LD/logs/driver.log
