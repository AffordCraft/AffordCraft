#!/usr/bin/env bash
# run_lane.sh <lane> <lanes> [--only sid1,sid2]  -- Articulate-Anything registered-subset-200 lane runner (one GPU).
# The API endpoint and key come from EVAL_API_BASE / EVAL_API_KEY; they are never echoed or written.
# environment: AA_WORK (code/ = this folder, src/ = patched official tree), AA_REV, AA_PYTHON, AA_GPU, AFFORDCRAFT_*
set -u
LANE=$1; LANES=$2; shift 2
AA=$(cd "${AA_WORK:-aa_run}" && pwd)
REV=${AA_REV:-${AFFORDCRAFT_RUNS:-runs}/external/articulate-anything/subset200}; mkdir -p "$REV"; REV=$(cd "$REV" && pwd)
export AA_REV=$REV
export EVAL_API_BASE="${EVAL_API_BASE:?set EVAL_API_BASE}"
export EVAL_API_KEY="${EVAL_API_KEY:?set EVAL_API_KEY}"
export AA_OPENAI_COMPAT=1
export AA_RENDER_SCRIPT=$AA/code/aa_sapien3_simulate.py
export MPLBACKEND=Agg XDG_RUNTIME_DIR=/tmp CUDA_VISIBLE_DEVICES=${AA_GPU:-0} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export AA_VLM_TIMEOUT=300 AA_VLM_MAX_ATTEMPTS=5
LANE_SRC=$AA/lanes/lane-$LANE/src
mkdir -p "$AA/lanes/lane-$LANE" "$REV/lane-$LANE"
if [ ! -d "$LANE_SRC" ]; then cp -a "$AA/src" "$LANE_SRC" || { echo "lane src copy failed"; exit 1; }; fi
export AA_SRC=$LANE_SRC
export PYTHONPATH=$LANE_SRC:$AA/code
cd "$LANE_SRC" || exit 1
echo "$(date -Is) lane $LANE/$LANES start host=$(hostname) src=$LANE_SRC" >> "$REV/lane-$LANE/lane.log"
${AA_PYTHON:-python} "$AA/code/run_lane.py" --lane "$LANE" --lanes "$LANES" "$@" >> "$REV/lane-$LANE/lane.log" 2>&1
echo "$(date -Is) lane $LANE exit=$?" >> "$REV/lane-$LANE/lane.log"
