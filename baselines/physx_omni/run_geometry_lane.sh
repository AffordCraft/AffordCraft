#!/usr/bin/env bash
# Generic official PhysX-Omni geometry+export lane (frozen runner omni_geometry_runner.py under the
# sm_120 env). GEO_REP = representation root (has configuration.json + cases/), GEO_QUERY = query manifest (jsonl rows in
# frozen order), GEO_OUT = lane output dir, GEO_GPU = CUDA index, GEO_TAG = alphanumeric lock tag. The runner waits per
# case until its native representation exists, then decodes + exports; every result stays in GEO_OUT/native_geometry.
# Environment: PHYSX_OMNI_RUN (run root; code/ receives the runner), PHYSX_OMNI_GEO_WORKSPACE (runner workspace, default
# $PHYSX_OMNI_RUN/workspace), PHYSX_OMNI_GEO_PYTHON, and for the workspace links PHYSX_OMNI_HOME (official checkout),
# DINOV2_HUB_REPO (facebookresearch/dinov2 at commit 7764ea0f), DINOV2_CKPT (dinov2_vitl14_reg4_pretrain.pth),
# U2NET_ONNX (rembg u2net.onnx) and TRELLIS_IMAGE_LARGE (microsoft/TRELLIS-image-large at revision 25e0d31f).
set -u
REP=${GEO_REP:?}; QUERY=${GEO_QUERY:?}; OUT=${GEO_OUT:?}; GPU=${GEO_GPU:?}; TAG=${GEO_TAG:?}
REV=${PHYSX_OMNI_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/physx-omni/geometry-subset200}; W=${PHYSX_OMNI_GEO_WORKSPACE:-$REV/workspace}
PY=${PHYSX_OMNI_GEO_PYTHON:-python}
mkdir -p $OUT/tmp $REV/code $W
HERE=$(cd "$(dirname "$0")" && pwd)
cp -n $HERE/omni_geometry_runner.py $HERE/execution_utils.py $REV/code/ 2>/dev/null
[ -e $W/physx-omni ] || ln -s ${PHYSX_OMNI_HOME:?set PHYSX_OMNI_HOME} $W/physx-omni
[ -e $W/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 ] || ln -s ${DINOV2_HUB_REPO:?set DINOV2_HUB_REPO} $W/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8
[ -e $W/dinov2_vitl14_reg4_pretrain.pth.part ] || ln -s ${DINOV2_CKPT:?set DINOV2_CKPT} $W/dinov2_vitl14_reg4_pretrain.pth.part
[ -e $W/u2net.onnx ] || ln -s ${U2NET_ONNX:?set U2NET_ONNX} $W/u2net.onnx
[ -e $W/trellis-image-large-25e0d31f ] || ln -s ${TRELLIS_IMAGE_LARGE:?set TRELLIS_IMAGE_LARGE} $W/trellis-image-large-25e0d31f
for i in $(seq 1 120); do [ -s $REP/configuration.json ] && [ -s $QUERY ] && break; sleep 30; done
export PYTHONPATH=$REV/code CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=$OUT/tmp
cd $OUT
CMD="$PY $REV/code/omni_geometry_runner.py --workspace $W --representation-root $REP --query-manifest $QUERY --output $OUT/native_geometry --resource-lock-tag $TAG"
echo "$(date -Is) host=$(hostname) gpu=$GPU rep=$REP query=$QUERY register" | tee -a $OUT/geometry.log
$CMD --register 2>&1 | tee -a $OUT/geometry.log
echo "$(date -Is) run" | tee -a $OUT/geometry.log
$CMD 2>&1 | tee -a $OUT/geometry.log
echo "$(date -Is) lane exit ${PIPESTATUS[0]}" | tee -a $OUT/geometry.log
