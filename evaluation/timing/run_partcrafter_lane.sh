#!/bin/bash
# Clean-GPU timing: one PartCrafter lane on one exclusive RTX 5090 (the wrapper code of the registered subset run copied
# to <run root>/partcrafter/code; the environment of the registered lane script without any API endpoint or key:
# run_partcrafter_lane_replay.py replays the part count and the call time the registered run recorded, see
# setup_timing_run.py partcrafter-replay), receipt; lane.wall.json. usage: run_partcrafter_lane.sh <lane> <gpu>
# environment: TIMING_ROOT, PARTCRAFTER_PYTHON
set -u
LANE=$1; GPU=$2
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
REV=$RT/partcrafter; L=$REV/lane-$LANE; TAG=clean$(hostname | tr -cd 'a-z0-9' | tail -c 6)g$GPU
PY=${PARTCRAFTER_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYOPENGL_PLATFORM=egl
unset OPENAI_BASE_URL OPENAI_API_KEY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p $L/tmp; export TMPDIR=$L/tmp; cd $L
exec >> $L/lane.log 2>&1
{ [ -s $REV/configuration.json ] && [ -s $REV/replay_table.json ] && [ -s $L/input_manifest.jsonl ]; } || { echo "configuration, replay table or lane manifest missing: refused"; exit 2; }
exec 9>$L/.lanelock; flock -n 9 || { echo "$(date -Is) lane $LANE is running elsewhere on this machine: nothing to do"; exit 0; }
n=$(grep -c . $L/input_manifest.jsonl); d=$(ls $L/cases/*/result.json 2>/dev/null | wc -l)
[ "$d" -ge "$n" ] && { echo "$(date -Is) lane $LANE already complete ($d/$n): nothing to do"; exit 0; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU"; exit 3; }
echo "$(date -Is) host=$(hostname) gpu=$GPU lane=$LANE tag=$TAG start (VLM answers replayed)"
T0=$(date +%s.%N); rc=1
for attempt in 1 2 3; do
  $PY $REV/code/run_partcrafter_lane_replay.py --revision-root $REV --lane $LANE --tag $TAG --render --vlm-workers 4; rc=$?
  echo "$(date -Is) runner exit $rc (attempt $attempt)"; [ $rc -eq 0 ] && break; sleep 20
done
T1=$(date +%s.%N)
$PY $REV/code/write_receipt.py --revision-root $REV --lane $LANE --tag $TAG --gpu $GPU --exit-code $rc
echo "{\"method\": \"partcrafter\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"start_unix\": $T0, \"end_unix\": $T1, \"exit_code\": $rc, \"vlm\": \"replayed from the registered subset run\"}" > $L/lane.wall.json
echo "$(date -Is) lane $LANE finished rc=$rc"
