#!/bin/bash
# PartCrafter lanes on one GPU, sequentially. PC_LANES = space-separated lane indices to run in order,
# PC_GPU = CUDA index, PC_TAG = alphanumeric host-gpu tag. A lane whose directory holds a SKIP_ON_<TAG> marker (or a
# .running lock held by another process) is skipped, so lanes can be handed to other GPUs/nodes while this loop runs.
# PC_ROOT may override the run root (default: the registered-subset run). The API endpoint/key come from EVAL_API_BASE /
# EVAL_API_KEY and are passed through the environment only (never echoed, never on a command line). PARTCRAFTER_PYTHON:
# interpreter of the run environment.
set -u
LANES=${PC_LANES:?}; GPU=${PC_GPU:?}; TAG=${PC_TAG:?}
REV=${PC_ROOT:-${AFFORDCRAFT_RUNS:-runs}/external/partcrafter/subset200}
PY=${PARTCRAFTER_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYOPENGL_PLATFORM=egl SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export EVAL_API_BASE="${EVAL_API_BASE:?set EVAL_API_BASE}" EVAL_API_KEY="${EVAL_API_KEY:?set EVAL_API_KEY}"
export PARTCRAFTER_API_MODEL=${PARTCRAFTER_API_MODEL:-gemini-3.8-flash}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for LANE in $LANES; do
  L=$REV/lane-$LANE; mkdir -p $L/tmp; export TMPDIR=$L/tmp
  if [ -e $L/SKIP_ON_$TAG ]; then echo "$(date -Is) lane $LANE skipped (SKIP_ON_$TAG)" | tee -a $L/lane.log; continue; fi
  if [ -s $L/input_manifest.jsonl ] && [ "$(grep -c . $L/input_manifest.jsonl)" -eq "$(ls $L/cases/*/result.json 2>/dev/null | wc -l)" ]; then echo "$(date -Is) lane $LANE already complete; skipped" | tee -a $L/lane.log; continue; fi
  exec 9>$L/.running
  if ! flock -n 9; then echo "$(date -Is) lane $LANE already running elsewhere on this node; skipped" | tee -a $L/lane.log; continue; fi
  cd $L
  echo "$(date -Is) host=$(hostname) gpu=$GPU lane=$LANE tag=$TAG start" | tee -a $L/lane.log
  rc=1
  for attempt in 1 2 3 4 5 6; do
    $PY $REV/code/run_partcrafter_lane.py --revision-root $REV --lane $LANE --tag $TAG --render --vlm-workers ${PC_VLM_WORKERS:-4} ${PC_EXTRA:-} >> $L/lane.log 2>&1; rc=$?
    echo "$(date -Is) runner exit $rc (attempt $attempt)" | tee -a $L/lane.log
    [ $rc -eq 0 ] && break
    sleep 30
  done
  $PY $REV/code/write_receipt.py --revision-root $REV --lane $LANE --tag $TAG --gpu $GPU --exit-code $rc >> $L/lane.log 2>&1
  echo "$(date -Is) lane $LANE finished rc=$rc" | tee -a $L/lane.log
  flock -u 9; exec 9>&-
done
echo "$(date -Is) all lanes [$LANES] done on $TAG"
