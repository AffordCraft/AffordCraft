#!/usr/bin/env bash
# Clean-GPU timing: PhysX-Omni geometry+export, one lane on one exclusive RTX PRO 6000 (the stage peaks at 91 GiB, it does
# not fit in a 5090). The frozen geometry runner of the registered subset run (its code and workspace, the command of its
# lane script) over physx-omni/lane-<k>: representation from native_representation/ (computed on 5090s by
# run_physx_omni_representation.sh), output native_geometry/; geo.wall.json. usage: run_physx_omni_geometry.sh <lane> <gpu>
# environment: TIMING_ROOT, PHYSX_OMNI_GEO_PYTHON, PHYSX_OMNI_GEO_RUNNER (the registered geometry runner; its directory
# goes on PYTHONPATH), PHYSX_OMNI_GEO_WORKSPACE (its workspace with the links checked below)
set -u
LANE=$1; GPU=$2
RT=${TIMING_ROOT:-${AFFORDCRAFT_RUNS:-runs}/clean_timing}
OUT=$RT/physx-omni/lane-$LANE; REP=$OUT/native_representation; QUERY=$OUT/input_manifest.jsonl; TAG=cleanomni$LANE
PY=${PHYSX_OMNI_GEO_PYTHON:-python}
RUNNER=${PHYSX_OMNI_GEO_RUNNER:-$(cd "$(dirname "$0")/../.." && pwd)/baselines/physx_omni/omni_geometry_runner.py}
W=${PHYSX_OMNI_GEO_WORKSPACE:?set PHYSX_OMNI_GEO_WORKSPACE to the runner workspace}
{ [ -s $REP/configuration.json ] && [ -s $QUERY ] && [ -s $RUNNER ]; } || { echo "missing representation/query/runner for lane $LANE"; exit 2; }
for x in physx-omni dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 dinov2_vitl14_reg4_pretrain.pth.part u2net.onnx trellis-image-large-25e0d31f; do
  [ -e $W/$x ] || { echo "workspace link $x missing"; exit 2; }
done
mkdir -p $OUT/tmp; cd $OUT
exec >> $OUT/geometry.log 2>&1
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
{ [ -n "$USED" ] && [ "$USED" -lt 1024 ]; } || { echo "$(date -Is) gpu $GPU in use (${USED} MiB): refused, clean timing needs an exclusive GPU"; exit 3; }
export PYTHONPATH=$(dirname $RUNNER) CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=$OUT/tmp
CMD="$PY $RUNNER --workspace $W --representation-root $REP --query-manifest $QUERY --output $OUT/native_geometry --resource-lock-tag $TAG"
echo "$(date -Is) host=$(hostname) gpu=$GPU lane=$LANE runner_sha256=$(sha256sum $RUNNER | cut -c1-64) register"
t0=$(date +%s.%N)
$CMD --register
t1=$(date +%s.%N); echo "$(date -Is) run"
$CMD; rc=$?
t2=$(date +%s.%N)
echo "{\"stage\": \"physx-omni geometry+export\", \"lane\": $LANE, \"gpu\": $GPU, \"host\": \"$(hostname)\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU)\", \"registration_start_unix\": $t0, \"run_start_unix\": $t1, \"end_unix\": $t2, \"exit_code\": $rc}" > $OUT/geo.wall.json
echo "$(date -Is) lane exit $rc"
