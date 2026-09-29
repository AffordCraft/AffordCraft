#!/bin/bash
# TRELLIS.2 lanes on one GPU, sequentially (same conventions as pc_lane.sh): T2_LANES, T2_GPU, T2_TAG,
# optional T2_ROOT (default: the registered-subset run). Weights come from the local checkpoint dir and the offline HF
# cache (HF_HOME = TRELLIS2_HF_HOME) only. TRELLIS2_PYTHON: interpreter of the run environment.
set -u
LANES=${T2_LANES:?}; GPU=${T2_GPU:?}; TAG=${T2_TAG:?}
REV=${T2_ROOT:-${AFFORDCRAFT_RUNS:-runs}/external/trellis2/subset200}
PY=${TRELLIS2_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${TRELLIS2_HF_HOME:-${AFFORDCRAFT_MODELS:-models}/hfcache-trellis2} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENCV_IO_ENABLE_OPENEXR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=${T2_ATTN_BACKEND:-flash_attn} SPARSE_ATTN_BACKEND=${T2_ATTN_BACKEND:-flash_attn} SPARSE_CONV_BACKEND=${T2_CONV_BACKEND:-flex_gemm}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for LANE in $LANES; do
  L=$REV/lane-$LANE; mkdir -p $L/tmp; export TMPDIR=$L/tmp
  if [ -e $L/SKIP_ON_$TAG ]; then echo "$(date -Is) lane $LANE skipped (SKIP_ON_$TAG)" | tee -a $L/lane.log; continue; fi
  if [ -s $L/input_manifest.jsonl ] && [ "$(grep -c . $L/input_manifest.jsonl)" -eq "$(ls $L/cases/*/result.json 2>/dev/null | wc -l)" ]; then echo "$(date -Is) lane $LANE already complete; skipped" | tee -a $L/lane.log; continue; fi
  exec 9>$L/.running
  if ! flock -n 9; then echo "$(date -Is) lane $LANE already running elsewhere on this node; skipped" | tee -a $L/lane.log; continue; fi
  cd $L
  echo "$(date -Is) host=$(hostname) gpu=$GPU lane=$LANE tag=$TAG attn=$ATTN_BACKEND conv=$SPARSE_CONV_BACKEND start" | tee -a $L/lane.log
  rc=1
  for attempt in $(seq 1 ${T2_ATTEMPTS:-14}); do
    $PY $REV/code/run_trellis2_lane.py --revision-root $REV --lane $LANE --tag $TAG --render ${T2_EXTRA:-} >> $L/lane.log 2>&1; rc=$?
    echo "$(date -Is) runner exit $rc (attempt $attempt)" | tee -a $L/lane.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
  $PY $REV/code/write_receipt.py --revision-root $REV --lane $LANE --tag $TAG --gpu $GPU --exit-code $rc >> $L/lane.log 2>&1
  echo "$(date -Is) lane $LANE finished rc=$rc" | tee -a $L/lane.log
  flock -u 9; exec 9>&-
done
echo "$(date -Is) all lanes [$LANES] done on $TAG"
