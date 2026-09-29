#!/usr/bin/env bash
# Round-0 teacher collection: wrist camera + workspace-crop observations. NW Isaac workers; worker W runs EP teacher
# episodes per scene with --seed SEED0+W (episode seed = seed*100000 + episode index): seeds 2000-2003, disjoint from
# the held-out evaluation seed 2600 (the "seen" evaluation pass replays seed 2000). Videos (with wrist
# picture-in-picture) and both frame streams are kept. Worker W runs on GPU ${GPUS[W % len]}.
# Executed as collect20: 4 workers x 8 episodes x 10 scenes = 320 teacher episodes. A run that ends early (simulator
# crash) is resumed at the next episode index with the same seed, which reproduces the same initial states; the
# executed collection did this in a separate fill pass, merged here into the worker loop.
set -uo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla; TAG=${TAG:-collect20}; EP=${EP:-8}; NW=${NW:-4}; OUT=$S/scenes/$TAG; mkdir -p "$OUT"
SEED0=${SEED0:-2000}; MODE=${MODE:-teacher}; EXTRA=${EXTRA:-}
PYI=${AFFORDCRAFT_ISAAC_PYTHON:?set AFFORDCRAFT_ISAAC_PYTHON to the Isaac Sim 5.1 python}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
count(){ [ -f "$1/summary.jsonl" ] && wc -l < "$1/summary.jsonl" || echo 0; }
worker(){ local W=$1 GPU=${GPUS[$(( $1 % ${#GPUS[@]} ))]} n ATT d
  for SCENE in cabinet_cup cabinet1_box fridge_cup microwave_box drawer_cup drawer_two drawer_to_cabinet drawer_to_fridge laptop_cup trashcan_box; do
    local DEST="$OUT/${SCENE}_w${W}"
    for ATT in 1 2 3; do
      n=$(count "$DEST"); [ "$n" -ge "$EP" ] && break
      for d in "$DEST"/episode_*; do [ -d "$d" ] && [ ! -f "$d/episode.json" ] && { mkdir -p "$OUT/partial"; mv "$d" "$OUT/partial/${SCENE}_w${W}_$(basename "$d")_$(date +%H%M%S)"; }; done
      env OMP_NUM_THREADS=4 "$PYI" "$F/scene_runner.py" --gpu "$GPU" --scene "$SCENE" --episodes $((EP-n)) --start-index "$n" --seed $((SEED0+W)) --output "$DEST" --mode "$MODE" --settle-ticks 20 --keep-frames true --video true --budget-ticks 1200 $EXTRA > "$DEST.a$ATT.log" 2>&1
      echo "[$TAG] w$W $SCENE attempt $ATT start_index $n gpu=$GPU exit=$? episodes=$(count "$DEST") $(date +%T)" >> "$OUT/runner.log"
    done
  done; }
for w in $(seq 0 $((NW-1))); do worker "$w" & done
wait
echo "[$TAG] all done $(date +%T)" >> "$OUT/runner.log"
