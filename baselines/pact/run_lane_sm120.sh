#!/bin/bash
# Lane driver of the remaining-1800 run on sm_120 GPUs (RTX 5090 / RTX PRO 6000; the registered-subset driver run_lane.sh
# with the run root and input list of that run): mask adapter for the lane's cases, official infer_imgs.py in an
# attempt loop, finalize. Environment: the sm_120 geometry environment (geom5090: torch 2.7 / spconv / nvdiffrast /
# diff_gaussian_rasterization built for sm_120; ATTN_BACKEND=sdpa as in the registered-subset run) + an extra site
# directory PACT_PYTHONPATH (pandas, pyfqmr, open3d). Other variables as in run_lane.sh.
# usage: run_lane_sm120.sh <lane> <gpu> [min_free_mib]
set -u
LANE=$1; GPU=$2; MINFREE=${3:-30000}
P=${AFFORDCRAFT_PROJECT_ROOT:-workspace}; RUNS=${AFFORDCRAFT_RUNS:-runs}; MODELS=${AFFORDCRAFT_MODELS:-$P/models}
ROOT=${PACT_RUN:-$RUNS/external/pact/remaining1800}
PY=${PACT_PYTHON:-python}
T=$(cd "$(dirname "$0")" && pwd); LD=$ROOT/lane-$LANE; mkdir -p $LD/logs
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENCV_IO_ENABLE_OPENEXR=1
export IMAGEIO_USERDIR=${IMAGEIO_USERDIR:-$HOME/.imageio} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# amendment 001: the sm_120 environment ships its own diff_gaussian_rasterization build; $ROOT/deps/site (the
# wheel built for sm_80, needed only by the sm_80 environment) must NOT precede it: with deps/site first every
# official attempt on the sm_120 GPUs failed (garbage allocation sizes, spconv int32 assert) while the smoke tests passed.
export PYTHONPATH=${PACT_PYTHONPATH:+$PACT_PYTHONPATH:}${PYTHONPATH:-}
LIST=${PACT_INPUT_LIST:-$RUNS/inputs/remaining_1800_inputs.json}
MAN=$RUNS/inputs/input_manifest_2000.jsonl
echo "{\"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"driver_pid\": $$, \"python\": \"$PY\", \"started_utc\": \"$(date -u +%FT%TZ)\", \"purpose\": \"PAct lane driver (mask adapter + official infer_imgs.py + finalize)\"}" >> $LD/logs/pids.json
run_adapter() {
  LANE_IDS=$($PY - <<EOF
import json
cfg=json.load(open("$ROOT/configuration.json")); print(",".join(cfg["lane_partition"]["lanes"]["$LANE"]))
EOF
)
  echo "[driver] adapter start $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  $PY $T/mask_adapter.py --subset $LIST --manifest $MAN --project-root $P --pact-src ${PACT_HOME:-pact} \
    --gdino ${PACT_GDINO:-$MODELS/grounding-dino-tiny} --sam ${PACT_SAM:-$MODELS/sam-vit-base} \
    --masks-root $ROOT/masks --ids "$LANE_IDS" >> $LD/logs/adapter.log 2>&1
  echo "[driver] adapter rc=$? $(date -u +%FT%TZ)" >> $LD/logs/driver.log
}
for attempt in 1 2 3 4 5 6 7 8; do
  run_adapter
  $PY $T/lane_runner.py prepare --root $ROOT --lane $LANE >> $LD/logs/driver.log 2>&1
  npend=$($PY -c "import json;print(len(json.load(open('$LD/pending.json'))['pending']))")
  echo "[driver] attempt $attempt pending=$npend $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  [ "$npend" = "0" ] && break
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1)
  while [ "$free" -lt "$MINFREE" ]; do echo "[driver] GPU $GPU free ${free} MiB < $MINFREE, waiting 300 s" >> $LD/logs/driver.log; sleep 300; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | head -1); done
  $PY $T/lane_runner.py infer --root $ROOT --lane $LANE > $LD/logs/infer_attempt_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
  IPID=$!
  echo "{\"lane\": $LANE, \"gpu\": $GPU, \"infer_pid\": $IPID, \"attempt\": $attempt, \"started_utc\": \"$(date -u +%FT%TZ)\"}" >> $LD/logs/pids.json
  wait $IPID; rc=$?
  echo "[driver] infer attempt $attempt rc=$rc $(date -u +%FT%TZ)" >> $LD/logs/driver.log
  $PY $T/lane_runner.py mark-interrupted --root $ROOT --lane $LANE >> $LD/logs/driver.log 2>&1
  $PY $T/lane_runner.py finalize --root $ROOT --lane $LANE --gpu $GPU >> $LD/logs/driver.log 2>&1
done
$PY $T/lane_runner.py finalize --root $ROOT --lane $LANE --gpu $GPU >> $LD/logs/driver.log 2>&1
echo "[driver] lane $LANE done $(date -u +%FT%TZ): results $(ls $LD/cases/*/result.json 2>/dev/null | wc -l)" >> $LD/logs/driver.log
