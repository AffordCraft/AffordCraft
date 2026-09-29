#!/usr/bin/env bash
# Closed-loop evaluation of a trained round (HEAD dir with completion.json) with stage-wise reporting.
# Pass "eval" (held-out seed 2600, never used in collection): trained head EP_T=10 episodes per scene (4 runners),
# untrained head (same head at its random initialization) EP_U=4 per scene (2 runners); the untrained service is
# started once the trained service is ready. Pass "eval_seen" (collect20 worker-0 seed 2000, i.e. the first six
# initial states of that worker): trained head EP_S=6 per scene (4 runners). Budget 1000 control ticks per episode.
# Services run on GPU_SVC, Isaac runners on the GPUs listed in GPUS (round-robin over the runner groups).
set -uo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla
HEAD=${HEAD:-$S/v20_vla/round1}; OUT=${OUT:-$HEAD}; EP_T=${EP_T:-10}; EP_U=${EP_U:-4}; EP_S=${EP_S:-6}; BUDGET=${BUDGET:-1000}
PYV=${OPENVLA_OFT_PYTHON:?set OPENVLA_OFT_PYTHON to the openvla-oft environment python}
PYI=${AFFORDCRAFT_ISAAC_PYTHON:?set AFFORDCRAFT_ISAAC_PYTHON to the Isaac Sim 5.1 python}
PYA=${AFFORDCRAFT_RUNTIME_PYTHON:-python}
GPU_SVC=${GPU_SVC:-0}; read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
[ -f "$HEAD/completion.json" ] || { echo "no completion.json in $HEAD"; exit 1; }
G1="cabinet_cup fridge_cup drawer_cup"; G2="drawer_to_cabinet laptop_cup"; G3="cabinet1_box microwave_box drawer_two"; G4="drawer_to_fridge trashcan_box"
count(){ ls "$1"/episode_*/episode.json 2>/dev/null | wc -l; }
run(){ local VAR=$1 SOCK=$2 EDIR=$3 SEED=$4 EP=$5 GPU=$6; shift 6; local SCENE DEST ATT
  for SCENE in "$@"; do DEST=$EDIR/$VAR/$SCENE
    for ATT in 1 2 3; do   # a crashed runner start is retried (partial output moved to <pass>/crashed/)
      [ "$(count "$DEST")" -ge "$EP" ] && break
      if [ -e "$DEST" ]; then mkdir -p "$EDIR/crashed/$VAR"; mv "$DEST" "$EDIR/crashed/$VAR/${SCENE}_$(date +%H%M%S)"; fi
      env OMP_NUM_THREADS=4 "$PYI" "$F/scene_runner.py" --gpu "$GPU" --scene "$SCENE" --episodes "$EP" --seed "$SEED" --output "$DEST" --mode policy --policy-socket "$SOCK" --variant "$VAR" --settle-ticks 20 --keep-frames true --video true --budget-ticks "$BUDGET" > "$DEST.log" 2>&1
      echo "[$(basename "$EDIR")] $VAR $SCENE attempt $ATT gpu=$GPU exit=$? episodes=$(count "$DEST") $(date +%T)" >> "$EDIR/runner.log"
    done
  done; }
gpu(){ echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }
variant(){ local VAR=$1 EDIR=$2 SEED=$3 EP=$4 NR=$5
  local VOUT=$EDIR/$VAR; mkdir -p "$VOUT"; local SOCK=/tmp/affordcraft_${VAR}_$(basename "$EDIR")_$$.sock; rm -f "$SOCK"; rm -rf "$VOUT/service"
  "$PYV" "$F/vla_head/serve_policy.py" --config "$HEAD/configuration.json" --checkpoint-manifest "$HEAD/training/checkpoint_manifest.json" --variant "$VAR" --gpu "$GPU_SVC" --output "$VOUT/service" --socket "$SOCK" > "$VOUT/service.log" 2>&1 &
  local SPID=$! i
  for i in $(seq 1 300); do [ -f "$VOUT/service/ready.json" ] && break; sleep 2; done
  if [ ! -f "$VOUT/service/ready.json" ]; then echo "service_not_ready $VAR $(basename "$EDIR") $(date -Is)" >> "$OUT/execution.log"; kill $SPID 2>/dev/null; return 1; fi
  echo "service $VAR ready on gpu $GPU_SVC $(date +%T)" >> "$EDIR/runner.log"
  local PIDS=()
  if [ "$NR" -ge 4 ]; then
    run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 0)" $G1 & PIDS+=($!); run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 1)" $G3 & PIDS+=($!)
    run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 2)" $G2 & PIDS+=($!); run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 3)" $G4 & PIDS+=($!)
  else
    run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 0)" $G1 $G2 & PIDS+=($!); run "$VAR" "$SOCK" "$EDIR" "$SEED" "$EP" "$(gpu 1)" $G3 $G4 & PIDS+=($!)
  fi
  wait "${PIDS[@]}"
  kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; rm -f "$SOCK"; }
# pass 1: held-out seed; trained (4 runners) first, untrained (2 runners) once the trained service is up
EDIR=$OUT/eval; mkdir -p "$EDIR"; echo "eval pass unseen seed 2600 trained $EP_T untrained $EP_U budget $BUDGET start $(date -Is)" >> "$OUT/execution.log"
variant trained_vla "$EDIR" 2600 "$EP_T" 4 & PT=$!
for i in $(seq 1 400); do [ -f "$EDIR/trained_vla/service/ready.json" ] && break; sleep 3; done
variant untrained_vla "$EDIR" 2600 "$EP_U" 2 & PU=$!
wait $PT; wait $PU
"$PYA" "$F/vla_head/pipeline_steps.py" eval_summary "$OUT" eval | tee -a "$OUT/execution.log"
# pass 2: seen seed, trained only (4 runners)
EDIR=$OUT/eval_seen; mkdir -p "$EDIR"; echo "eval pass seen seed 2000 trained $EP_S start $(date -Is)" >> "$OUT/execution.log"
variant trained_vla "$EDIR" 2000 "$EP_S" 4
"$PYA" "$F/vla_head/pipeline_steps.py" eval_summary "$OUT" eval_seen | tee -a "$OUT/execution.log"
echo "eval done $(date -Is)" >> "$OUT/execution.log"
