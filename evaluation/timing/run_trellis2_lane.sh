#!/bin/bash
# Clean-GPU timing: one TRELLIS.2 lane on one exclusive RTX 5090 (the wrapper code of the registered run copied to
# <run root>/trellis2/code, the environment of the registered lane script, --render kept for parity), receipt;
# lane.wall.json. usage: run_trellis2_lane.sh <lane> <gpu>
# environment: TIMING_ROOT, TRELLIS2_PYTHON, TRELLIS2_HF_HOME (Hugging Face cache holding the TRELLIS.2-4B weights)
set -u
LANE=$1; GPU=$2
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
REV=$RT/trellis2; L=$REV/lane-$LANE; TAG=clean$(hostname | tr -cd 'a-z0-9' | tail -c 6)g$GPU
PY=${TRELLIS2_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${TRELLIS2_HF_HOME:-${AFFORDCRAFT_MODELS:-models}/hfcache-trellis2} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENCV_IO_ENABLE_OPENEXR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p $L/tmp; export TMPDIR=$L/tmp; cd $L
exec >> $L/lane.log 2>&1
{ [ -s $REV/configuration.json ] && [ -s $L/input_manifest.jsonl ]; } || { echo "configuration or lane manifest missing: refused"; exit 2; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU"; exit 3; }
echo "$(date -Is) host=$(hostname) gpu=$GPU lane=$LANE tag=$TAG start"
T0=$(date +%s.%N); rc=1
for attempt in 1 2 3; do
  $PY $REV/code/run_trellis2_lane.py --revision-root $REV --lane $LANE --tag $TAG --render; rc=$?
  echo "$(date -Is) runner exit $rc (attempt $attempt)"; [ $rc -eq 0 ] && break; sleep 20
done
T1=$(date +%s.%N)
$PY $REV/code/write_receipt.py --revision-root $REV --lane $LANE --tag $TAG --gpu $GPU --exit-code $rc
echo "{\"method\": \"trellis2\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"start_unix\": $T0, \"end_unix\": $T1, \"exit_code\": $rc}" > $L/lane.wall.json
echo "$(date -Is) lane $LANE finished rc=$rc"
