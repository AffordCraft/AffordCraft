#!/usr/bin/env bash
# Restart a geometry lane as a NEW attempt directory (the frozen runner's environment.json is append-only, so a lane
# cannot be resumed in place). Finished cases of earlier attempts are linked into the new attempt so they are skipped;
# cases whose earlier attempts ended in an infrastructure failure (CUDA OOM) or were interrupted are recomputed.
# usage: run_geometry_lane_attempt.sh <lane dir> <gpu> <tag> <attempt number>
# environment: PHYSX_OMNI_RUN, PHYSX_OMNI_GEO_WORKSPACE, PHYSX_OMNI_GEO_PYTHON (as in run_geometry_lane.sh)
set -u
L=$1; GPU=$2; TAG=$3; N=$4
REV=${PHYSX_OMNI_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/physx-omni/geometry-subset200}; W=${PHYSX_OMNI_GEO_WORKSPACE:-$REV/workspace}
PY=${PHYSX_OMNI_GEO_PYTHON:-python}
NEW=$L/native_geometry_a$(printf %02d $N); mkdir -p $NEW/cases $L/tmp
$PY - "$L" "$NEW" <<'PY'
import json,os,sys,glob,shutil
L,NEW=sys.argv[1],sys.argv[2];lineage={'attempt':os.path.basename(NEW),'inherited':[],'recomputed':[]}
prev=[d for d in sorted(glob.glob(L+'/native_geometry*')) if os.path.isdir(d) and d!=NEW]
for d in prev:
    for c in sorted(glob.glob(d+'/cases/*')):
        sid=os.path.basename(c);link=os.path.join(NEW,'cases',sid)
        if os.path.lexists(link):continue
        rp=os.path.join(c,'native_asset_result.json')
        if not os.path.exists(rp):
            lineage['recomputed'].append({'source_id':sid,'from':c,'reason':'interrupted_no_result'});continue
        r=json.load(open(rp));fr=' '.join(str(x) for x in (r.get('failure_reason') or []))
        if r.get('asset_emitted') or 'OutOfMemoryError' not in fr:
            os.symlink(c,link);lineage['inherited'].append({'source_id':sid,'from':c,'asset_emitted':r.get('asset_emitted')})
        else:
            lineage['recomputed'].append({'source_id':sid,'from':c,'reason':'cuda_oom_infrastructure_failure'})
json.dump(lineage,open(os.path.join(NEW,'attempt_lineage.json'),'w'),indent=1)
print('inherited',len(lineage['inherited']),'recomputed',len(lineage['recomputed']))
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REV/code CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=$L/tmp
cd $L
REP=$(python3 -c "import json;print(json.load(open('$L/native_geometry/configuration.json'))['representation_configuration']['path'].rsplit('/',1)[0])" 2>/dev/null || echo $REV/representation_union)
Q=$L/query_manifest.jsonl
CMD="$PY $REV/code/omni_geometry_runner.py --workspace $W --representation-root $REP --query-manifest $Q --output $NEW --resource-lock-tag $TAG"
echo "$(date -Is) attempt $N host=$(hostname) gpu=$GPU register" | tee -a $L/geometry.log
$CMD --register 2>&1 | tee -a $L/geometry.log
echo "$(date -Is) run" | tee -a $L/geometry.log
$CMD 2>&1 | tee -a $L/geometry.log
echo "$(date -Is) attempt $N exit ${PIPESTATUS[0]}" | tee -a $L/geometry.log
