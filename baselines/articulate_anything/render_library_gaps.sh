#!/bin/bash
# render_library_gaps.sh: library completion before the second attempt of the run -- render_library_gaps.py in
# <shards> parallel shards on one GPU (the official render call through the SAPIEN-3 port), then the PIL readability
# check of every robot_frontview.png (check_library_renders.py --check-only). Run list_library_gaps.py first.
# usage: render_library_gaps.sh [shards]; environment: AA_WORK, AA_PYTHON, AA_GPU, ARTICULATE_ANYTHING_CKPT
AA=$(cd "${AA_WORK:-aa_run}" && pwd); N=${1:-8}
export AA_SRC=$AA/src PYTHONPATH=$AA/src:$AA/code AA_OPENAI_COMPAT=1 AA_RENDER_SCRIPT=$AA/code/aa_sapien3_simulate.py
export CUDA_VISIBLE_DEVICES=${AA_GPU:-0} XDG_RUNTIME_DIR=/tmp MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd $AA/src || exit 1
echo "$(date -Is) start, $N shards"
for s in $(seq 0 $((N - 1))); do
  ${AA_PYTHON:-python} $AA/code/render_library_gaps.py $s $N > $AA/logs/render_gaps_$s.out 2>&1 &
done
wait
echo "$(date -Is) shards done: $(grep -h SHARD_DONE $AA/logs/render_gaps_*.out | wc -l)/$N; ok $(cat $AA/logs/library_render_gaps_shard*.jsonl | grep -c '"ok": true') of $(cat $AA/logs/library_render_gaps_shard*.jsonl | wc -l)"
${AA_PYTHON:-python} $AA/code/check_library_renders.py --check-only
echo "$(date -Is) GAPS_DONE"
