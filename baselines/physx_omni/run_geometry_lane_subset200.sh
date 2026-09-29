#!/usr/bin/env bash
# One official PhysX-Omni geometry lane on a 5090 node. GEO_LANE / GEO_LANES: round-robin split of the registered
# 200-input subset in subset order; GEO_GPU: CUDA index; GEO_TAG: lane name (default host-gpuN). The frozen runner
# omni_geometry_runner.py waits per case until its native representation exists. Environment: PHYSX_OMNI_RUN (run root
# with representation_union/ and code/), PHYSX_OMNI_GEO_WORKSPACE (workspace, see run_geometry_lane.sh), PHYSX_OMNI_GEO_PYTHON.
set -u
LANE=${GEO_LANE:?}; LANES=${GEO_LANES:?}; GPU=${GEO_GPU:?}; TAG=${GEO_TAG:-$(hostname | cut -c1-16)-gpu$GPU}
REV=${PHYSX_OMNI_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/physx-omni/geometry-subset200}; L=$REV/lane-$TAG; W=${PHYSX_OMNI_GEO_WORKSPACE:-$REV/workspace}
PY=${PHYSX_OMNI_GEO_PYTHON:-python}
mkdir -p $L/tmp
$PY - "$REV/representation_union/manifest_subset200.jsonl" "$L/query_manifest.jsonl" "$LANE" "$LANES" <<'PY'
import sys,os
src,dst,lane,lanes=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
rows=[l for l in open(src) if l.strip()];sel=[l for k,l in enumerate(rows) if k%lanes==lane]
if os.path.exists(dst):assert open(dst).read()==''.join(sel),'lane manifest drift'
else:open(dst,'w').write(''.join(sel))
print('lane rows',len(sel))
PY
export PYTHONPATH=$REV/code CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=$L/tmp
cd $L
CMD="$PY $REV/code/omni_geometry_runner.py --workspace $W --representation-root $REV/representation_union --query-manifest $L/query_manifest.jsonl --output $L/native_geometry --resource-lock-tag $TAG"
echo "$(date -Is) host=$(hostname) gpu=$GPU register" | tee -a $L/geometry.log
$CMD --register 2>&1 | tee -a $L/geometry.log
echo "$(date -Is) run" | tee -a $L/geometry.log
$CMD 2>&1 | tee -a $L/geometry.log
echo "$(date -Is) lane exit ${PIPESTATUS[0]}" | tee -a $L/geometry.log
