#!/usr/bin/env bash
# Clean-GPU timing: one AffordCraft shard (scripts/run_pipeline.py --calibration, the visual index of the main campaign,
# the main campaign's worker environment: OMP/MKL/OPENBLAS 2 threads, MALLOC_ARENA_MAX 2, AFFORDCRAFT_BUILD_THREADS 8)
# on one GPU of this machine that nothing else uses, over the 32-input sample (inputs/sample32.json,
# i % <shard count> == <shard>). prepare_affordcraft_shard.py checks the code/model locks and the images and links the
# shard's own geometry cache first. usage: run_affordcraft_shard.sh <cold|warm> <shard> <shard count> <gpu>
# environment: TIMING_ROOT (run root), AFFORDCRAFT_INDEX (visual index directory), AFFORDCRAFT_SELECTOR_PYTHON
# (interpreter of run_pipeline.py), AFFORDCRAFT_RUNTIME_PYTHON, AFFORDCRAFT_GEOMETRY_CACHE (main-campaign geometry cache)
set -u
REG=$1; K=$2; N=$3; G=$4
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
    --inputs $RT/inputs/sample32.json --index $INDEX --output $D/shard$K --gpu $G \
    --shard-index $K --shard-count $N --geometry-cache $D/cache-shard$K >> $LOG 2>&1
rc=$?; t1=$(date +%s.%N)
echo "{\"regime\": \"$REG\", \"shard\": $K, \"shard_count\": $N, \"gpu\": $G, \"host\": \"$(hostname)\", \"start_unix\": $t0, \"end_unix\": $t1, \"exit_code\": $rc}" > $D/shard$K.wall.json
say "run_pipeline end rc=$rc"
