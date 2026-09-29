#!/usr/bin/env bash
# DAgger round: mixed rollouts with the head of the previous round (HEAD) served on GPU_S; NW Isaac workers in
# --mode dagger (the teacher plan executes, the served policy takes random bursts of 8-40 ticks at a target executed
# share BETA=0.35, in half of the episodes also a prefix of up to 200 ticks; labels = teacher targets).
# Executed as dagger20: HEAD=round0 (bootstrap head), seeds 2100+worker, 4 workers x 5 episodes x 10 scenes = 200.
# run_round2.sh reuses this script for dagger20r2 (HEAD=round1, seeds 2200+worker, 6 episodes: 240 rollouts).
set -uo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla; TAG=${TAG:-dagger20}; EP=${EP:-5}; OUT=$S/scenes/$TAG; HEAD=${HEAD:-$S/v20_vla/round0}
GPU_S=${GPU_S:-0}; SEED0=${SEED0:-2100}; BETA=${BETA:-0.35}; NW=${NW:-4}
PYV=${OPENVLA_OFT_PYTHON:?set OPENVLA_OFT_PYTHON to the openvla-oft environment python}
PYI=${AFFORDCRAFT_ISAAC_PYTHON:?set AFFORDCRAFT_ISAAC_PYTHON to the Isaac Sim 5.1 python}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
mkdir -p "$OUT"
SOCK=/tmp/affordcraft_dagger_$$.sock; rm -f "$SOCK"; rm -rf "$OUT/service"
"$PYV" "$F/vla_head/serve_policy.py" --config "$HEAD/configuration.json" --checkpoint-manifest "$HEAD/training/checkpoint_manifest.json" --variant trained_vla --gpu "$GPU_S" --output "$OUT/service" --socket "$SOCK" > "$OUT/service.log" 2>&1 &
SPID=$!
for i in $(seq 1 300); do [ -f "$OUT/service/ready.json" ] && break; sleep 2; done
[ -f "$OUT/service/ready.json" ] || { echo "[$TAG] service_not_ready gpu $GPU_S $(date +%T)" >> "$OUT/runner.log"; kill $SPID 2>/dev/null; exit 1; }
echo "[$TAG] service ready on gpu $GPU_S $(date +%T)" >> "$OUT/runner.log"
count(){ [ -f "$1/summary.jsonl" ] && wc -l < "$1/summary.jsonl" || echo 0; }
worker(){ local W=$1 GPU=${GPUS[$(( $1 % ${#GPUS[@]} ))]} n ATT d
  for SCENE in cabinet_cup cabinet1_box fridge_cup microwave_box drawer_cup drawer_two drawer_to_cabinet drawer_to_fridge laptop_cup trashcan_box; do
    local DEST="$OUT/${SCENE}_w${W}"
    for ATT in 1 2 3; do   # a crashed run is resumed at the next episode index (same seed; partial episode dirs moved aside)
      n=$(count "$DEST"); [ "$n" -ge "$EP" ] && break
      for d in "$DEST"/episode_*; do [ -d "$d" ] && [ ! -f "$d/episode.json" ] && { mkdir -p "$OUT/partial"; mv "$d" "$OUT/partial/${SCENE}_w${W}_$(basename "$d")_$(date +%H%M%S)"; }; done
      env OMP_NUM_THREADS=4 "$PYI" "$F/scene_runner.py" --gpu "$GPU" --scene "$SCENE" --episodes $((EP-n)) --start-index "$n" --seed $((SEED0+W)) --output "$DEST" --mode dagger --policy-socket "$SOCK" --variant trained_vla --dagger-beta "$BETA" --settle-ticks 20 --keep-frames true --video true --budget-ticks 1200 > "$DEST.a$ATT.log" 2>&1
      echo "[$TAG] w$W $SCENE attempt $ATT start_index $n gpu=$GPU exit=$? episodes=$(count "$DEST") $(date +%T)" >> "$OUT/runner.log"
    done
  done; }
WPIDS=(); for w in $(seq 0 $((NW-1))); do worker "$w" & WPIDS+=($!); done
wait "${WPIDS[@]}"   # workers only: a bare `wait` would also wait for the background policy service
kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; rm -f "$SOCK"
echo "[$TAG] all done $(date +%T)" >> "$OUT/runner.log"
