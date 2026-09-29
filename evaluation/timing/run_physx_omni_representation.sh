#!/usr/bin/env bash
# Clean-GPU timing: PhysX-Omni native representation, one lane on one exclusive RTX 5090. The frozen representation runner
# of the registered 2,000-input run with its workspace and the command of its lane script: registration, then inference
# over physx-omni/lane-<k>/input_manifest.jsonl; rep.wall.json. usage: run_physx_omni_representation.sh <lane> <gpu>
# environment: TIMING_ROOT, PHYSX_OMNI_REP_PYTHON, PHYSX_OMNI_REP_RUNNER (the registered representation runner),
# PHYSX_OMNI_REP_WORKSPACE (its workspace), AFFORDCRAFT_PROJECT_ROOT (resolves the image paths of the manifest)
set -u
LANE=$1; GPU=$2
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
NEW=$RT/physx-omni/lane-$LANE
PY=${PHYSX_OMNI_REP_PYTHON:-python}
SCRIPT=${PHYSX_OMNI_REP_RUNNER:-$(cd "$(dirname "$0")/../.." && pwd)/baselines/physx_omni/omni_representation_runner.py}
WS=${PHYSX_OMNI_REP_WORKSPACE:?set PHYSX_OMNI_REP_WORKSPACE to the runner workspace}
[ -s $NEW/input_manifest.jsonl ] || { echo "no lane manifest $NEW"; exit 2; }
exec >> $NEW/omni_lane.log 2>&1
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU"; exit 3; }
echo "$(date -Is) start lane=$LANE gpu=$GPU host=$(hostname) runner_sha256=$(sha256sum $SCRIPT | cut -c1-64)"
t0=$(date +%s.%N)
"$PY" "$SCRIPT" --workspace "$WS" --input-manifest "$NEW/input_manifest.jsonl" --input-root "${AFFORDCRAFT_PROJECT_ROOT:-workspace}" --output "$NEW/native_representation" --registration || { echo "$(date -Is) registration_failed"; exit 1; }
t1=$(date +%s.%N)
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT" --workspace "$WS" --input-manifest "$NEW/input_manifest.jsonl" --input-root "${AFFORDCRAFT_PROJECT_ROOT:-workspace}" --output "$NEW/native_representation"; rc=$?
t2=$(date +%s.%N)
echo "{\"stage\": \"physx-omni native_representation\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"registration_start_unix\": $t0, \"run_start_unix\": $t1, \"end_unix\": $t2, \"exit_code\": $rc}" > $NEW/rep.wall.json
echo "$(date -Is) done lane=$LANE rc=$rc"
