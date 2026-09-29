#!/usr/bin/env bash
# Round-0 bootstrap. The round-0 head that drove the first DAgger round (dagger20) was trained on the teacher episodes
# whose features the round-0 cache of post_train.sh 0 had already written (shard 1 complete plus the finished part of
# shard 0; 143 of the 267 successful teacher episodes), so that the DAgger collection could start before the full
# cache existed. This script writes round0/features_bootstrap/{shard_0,shard_1}_manifest.json (records pointing at the
# existing npz files, sha-verified), trains into round0/training with the same configuration (24k steps) and writes
# round0/completion.json with bootstrap flags; run_dagger.sh serves that head. The full round-0 cache, finished
# afterwards, is reused by round 1 (post_train.sh 1). Run while or after post_train.sh 0 caches features.
set -euo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla; OUT=$S/v20_vla/round0
PYA=${AFFORDCRAFT_RUNTIME_PYTHON:-python}
PYV=${OPENVLA_OFT_PYTHON:?set OPENVLA_OFT_PYTHON to the openvla-oft environment python}
G=${GPU:-0}
log(){ echo "$(date -Is) $*" | tee -a "$OUT/execution.log"; }
[ -e "$OUT/completion.json" ] && { echo already_complete; exit 0; }
"$PYV" - "$OUT" <<'EOF' | tee -a "$OUT/execution.log"
import json,sys,hashlib,os
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]);ds=out/'dataset';feat=out/'features';boot=out/'features_bootstrap';boot.mkdir(exist_ok=True)
man=json.load(open(ds/'manifest.json'));MSHA=hashlib.sha256(open(ds/'manifest.json','rb').read()).hexdigest();C=man['chunk'];STRIDE=man['execute_prefix'];NIMG=int(man.get('num_images',1))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
groups={}
for line in (ds/'frame_index.jsonl').open():
    r=json.loads(line);groups.setdefault(r['episode_job'],[]).append(r)
jobs=sorted(groups);recs={0:[],1:[]};skipped=[]
for epi,job in enumerate(jobs):
    rows=sorted(groups[job],key=lambda x:x['index']);p=feat/f'episode_{epi:04d}.npz';n=len(range(0,len(rows),STRIDE))
    if not p.exists():skipped.append(epi);continue
    try:
        with np.load(p) as x:
            ok=x['features'].shape==(n,4096*NIMG) and x['proprio'].shape==(n,9) and x['targets'].shape==(n,C,9) and x['valid'].shape==(n,C) and x['indices'].shape==(n,) and bool(np.allclose(x['proprio'][:3],np.asarray([rows[i]['proprio'] for i in range(0,min(len(rows),3*STRIDE),STRIDE)],np.float32)))
            if 'manifest_sha' in x.files and str(x['manifest_sha'])!=MSHA:ok=False
    except Exception:ok=False
    if not ok:skipped.append(epi);continue
    recs[epi%2].append(dict(episode_job=job,scene=rows[0]['scene'],query_count=n,path=str(p.resolve()),sha256=sha(p),runtime_s=None,bootstrap_partial=True))
per={}
for k in (0,1):
    (boot/f'shard_{k}_manifest.json').write_text(json.dumps(dict(records=recs[k],num_images=NIMG,feature_dim=4096*NIMG,bootstrap_partial=True,source_features=str(feat),dataset_manifest_sha256=MSHA),indent=1))
    for r in recs[k]:per[r['scene']]=per.get(r['scene'],0)+1
print(json.dumps(dict(bootstrap_episodes=len(recs[0])+len(recs[1]),of=len(jobs),per_scene=per,skipped=len(skipped))))
if len(per)<10:raise SystemExit('bootstrap_scene_coverage_blocked')
EOF
log "bootstrap training on gpu $G"
[ -d "$OUT/training" ] && [ ! -f "$OUT/training/checkpoint_manifest.json" ] && mv "$OUT/training" "$OUT/training_partial_$(date +%H%M%S)"
"$PYV" "$F/vla_head/train_head.py" --config "$OUT/configuration.json" --features "$OUT/features_bootstrap" --dataset "$OUT/dataset" --output "$OUT/training" --gpu "$G" > "$OUT/training.log" 2>&1
log "training_done (bootstrap)"
"$PYA" - "$OUT" <<'EOF' | tee -a "$OUT/execution.log"
import json,sys,time
from pathlib import Path
out=Path(sys.argv[1]);cm=json.load(open(out/'training/checkpoint_manifest.json'))
json.dump({'completed':True,'created':time.time(),'bootstrap':True,'note':'DAgger bootstrap head trained on the episodes whose features were already cached (shard 1 complete, shard 0 partial); the full round-0 feature cache is completed later and reused by round 1','episodes_trained':cm['episodes'],'per_scene_trained':cm['per_scene'],'dataset_manifest':json.load(open(out/'dataset/manifest.json')),'checkpoint_manifest':cm,'video_audit_summary':{k:v for k,v in json.load(open(out/'video_audit.json')).items() if k!='rows'},'robot_real_world':'not_evaluated','robot_control_status':'simulated_robot_only','robot_task_success':None},open(out/'completion.json','w'),indent=1)
print('completion.json written (bootstrap, %d episodes)'%cm['episodes'])
EOF
log "round 0 bootstrap done"
