#!/bin/bash
# Lane driver of the registered-subset run: runs the mask adapter (if needed), prepares the lane workspace, runs the official
# PAct inference in a loop (restarting on crash, skipping finished cases), then finalizes.
# Usage: run_lane.sh <lane> <gpu> [lane_name] [ids]
# environment: AFFORDCRAFT_PROJECT_ROOT, AFFORDCRAFT_RUNS, AFFORDCRAFT_MODELS, PACT_RUN, PACT_INPUT_LIST, PACT_PYTHON,
# PACT_HOME, PACT_GDINO, PACT_SAM, IMAGEIO_USERDIR (see README.md)
set -u
LANE=$1; GPU=$2; LNAME=${3:-lane-$1}; IDS=${4:-}
P=${AFFORDCRAFT_PROJECT_ROOT:-workspace}; RUNS=${AFFORDCRAFT_RUNS:-runs}; MODELS=${AFFORDCRAFT_MODELS:-$P/models}
ROOT=${PACT_RUN:-$RUNS/external/pact/subset200}
PY=${PACT_PYTHON:-python}
T=$(cd "$(dirname "$0")" && pwd)
LD=$ROOT/$LNAME
mkdir -p $LD/logs
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENCV_IO_ENABLE_OPENEXR=1
export IMAGEIO_USERDIR=${IMAGEIO_USERDIR:-$HOME/.imageio}
export PYTHONPATH=$ROOT/deps/site${PYTHONPATH:+:$PYTHONPATH}   # amendment 001: diff_gaussian_rasterization wheel (private dir, shared env untouched)
echo "{\"lane\": $LANE, \"gpu\": $GPU, \"driver_pid\": $$, \"started_utc\": \"$(date -u +%FT%TZ)\", \"purpose\": \"PAct lane driver (mask adapter + official infer_imgs.py + finalize)\"}" >> $LD/logs/pids.json

ids_arg=""
if [ -n "$IDS" ]; then ids_arg="--ids $IDS"; fi

run_adapter() {
# 1. adapter for this lane's cases (idempotent; skips cases with adapter.json; re-run each attempt so a crashed adapter resumes)
LANE_IDS=$($PY - <<EOF
import json
cfg=json.load(open("$ROOT/configuration.json"))
ids=cfg["lane_partition"]["lanes"]["$LANE"]
want="$IDS".split(",") if "$IDS" else None
print(",".join(i for i in ids if (want is None or i in want)))
EOF
)
echo "[driver] adapter start $(date -u +%FT%TZ)" >> $LD/logs/driver.log
$PY $T/mask_adapter.py --subset ${PACT_INPUT_LIST:-$RUNS/inputs/subset_200_inputs.json} \
  --manifest $RUNS/inputs/input_manifest_2000.jsonl \
  --project-root $P --pact-src ${PACT_HOME:-pact} \
  --gdino ${PACT_GDINO:-$MODELS/grounding-dino-tiny} --sam ${PACT_SAM:-$MODELS/sam-vit-base} \
  --masks-root $ROOT/masks --ids "$LANE_IDS" >> $LD/logs/adapter.log 2>&1
echo "[driver] adapter rc=$? $(date -u +%FT%TZ)" >> $LD/logs/driver.log

}

# 2. inference loop
for attempt in 1 2 3 4 5 6 7 8; do
  run_adapter
  $PY $T/lane_runner.py prepare --root $ROOT --lane $LANE --lane-name $LNAME $ids_arg >> $LD/logs/driver.log 2>&1
  npend=$($PY -c "import json;print(len(json.load(open('$LD/pending.json'))['pending']))")
  echo "[driver] attempt $attempt pending=$npend $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  if [ "$npend" = "0" ]; then break; fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1)
  while [ "$free" -lt 30000 ]; do
    echo "[driver] GPU $GPU free ${free} MiB < 30000, waiting 300 s" >> $LD/logs/driver.log; sleep 300
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1)
  done
  $PY $T/lane_runner.py infer --root $ROOT --lane $LANE --lane-name $LNAME > $LD/logs/infer_attempt_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
  IPID=$!
  echo "{\"lane\": $LANE, \"gpu\": $GPU, \"infer_pid\": $IPID, \"attempt\": $attempt, \"started_utc\": \"$(date -u +%FT%TZ)\", \"command\": \"$PY $T/lane_runner.py infer --root $ROOT --lane $LANE --lane-name $LNAME\", \"purpose\": \"official PAct infer_imgs.py over pending cases\"}" >> $LD/logs/pids.json
  wait $IPID; rc=$?
  echo "[driver] infer attempt $attempt rc=$rc $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  $PY $T/lane_runner.py mark-interrupted --root $ROOT --lane $LANE --lane-name $LNAME >> $LD/logs/driver.log 2>&1
  $PY $T/lane_runner.py finalize --root $ROOT --lane $LANE --lane-name $LNAME --gpu $GPU >> $LD/logs/finalize.log 2>&1
  npend=$($PY -c "import json;print(len(json.load(open('$LD/pending.json'))['pending']))")
  if [ "$npend" = "0" ]; then break; fi
  sleep 20
done
$PY $T/lane_runner.py finalize --root $ROOT --lane $LANE --lane-name $LNAME --gpu $GPU --receipt >> $LD/logs/finalize.log 2>&1
echo "[driver] done $(date -u +%FT%TZ)" >> $LD/logs/driver.log
