#!/usr/bin/env bash
# Clean-GPU timing: the same as run_affordcraft_shard.sh for a re-run of inputs that an aborted shard left without a
# record (or of the last input of a shard that is stopped early so that the regime completes sooner): the pipeline runs
# the given inputs file as one shard on its own exclusive GPU.
# usage: run_affordcraft_rerun.sh <cold|warm> <new shard number> <gpu> <inputs json>
set -u
REG=$1; K=$2; N=1; G=$3; INPUTS=$4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
INDEX=${AFFORDCRAFT_INDEX:-${AFFORDCRAFT_PROJECT_ROOT:-workspace}/index/visual-large-001}
D=$RT/ours-$REG; mkdir -p $D; LOG=$D/shard$K.log
say(){ echo "$(date -Is) $*" >> $LOG; }
say "prep $REG shard $K/$N gpu $G on $(hostname)"
${AFFORDCRAFT_RUNTIME_PYTHON:-python} $REPO/evaluation/timing/prepare_affordcraft_shard.py $RT $REG $K $G >> $LOG 2>&1 || { say "prep failed, not running"; exit 3; }
t0=$(date +%s.%N); say "run_pipeline start"
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \
    TOKENIZERS_PARALLELISM=false AFFORDCRAFT_BUILD_THREADS=8 PYTHONUNBUFFERED=1 \
  ${AFFORDCRAFT_SELECTOR_PYTHON:-python} $REPO/scripts/run_pipeline.py --calibration \
    --inputs $INPUTS --index $INDEX --output $D/shard$K --gpu $G \
    --shard-index 0 --shard-count 1 --geometry-cache $D/cache-shard$K >> $LOG 2>&1
rc=$?; t1=$(date +%s.%N)
echo "{\"regime\": \"$REG\", \"shard\": $K, \"shard_count\": $N, \"gpu\": $G, \"host\": \"$(hostname)\", \"start_unix\": $t0, \"end_unix\": $t1, \"exit_code\": $rc, \"inputs\": \"$INPUTS\"}" > $D/shard$K.wall.json
say "run_pipeline end rc=$rc"
