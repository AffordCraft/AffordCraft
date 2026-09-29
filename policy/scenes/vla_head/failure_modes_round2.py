"""Read-only failure-mode analysis of the round-2 held-out trained-head episodes. For every failed episode:
stage-1 status, articulation excursion, object lift, closest end-effector approach, gripper closure, final placement.
Writes <scene run root>/fig7_export/failure_modes_round2.json (numpy only, no Isaac)."""

import json, glob, os, sys, math
import numpy as np

S = os.path.join(os.environ.get("AFFORDCRAFT_RUNS", "runs"), "scenes_vla")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_specs import SCENES


def analyse(ep_dir):
    e = json.load(open(ep_dir + "/episode.json"))
    sc = e["scene"]
    spec = SCENES[sc]
    tr = [json.loads(l) for l in open(ep_dir + "/trace.jsonl")]
    goals = e["result"]["goals"]
    s1 = bool(goals and goals[0].get("ok"))
    obj = [r["key"] for r in spec["rigid"]][0]
    z = np.array([t["objects"][obj]["center"][2] for t in tr])
    lift = float(z.max() - z[0])
    ee = np.array([t["ee"] for t in tr])
    oc = np.array([t["objects"][obj]["center"] for t in tr])
    dist = np.linalg.norm(ee - oc, axis=1)
    mind = float(dist.min())
    argmin = int(dist.argmin())
    fin = np.array([t["proprio"][7] for t in tr])
    closed_near = bool(((fin < 0.02) & (dist < 0.06)).any())
    contact = bool(any(t["contacts"].get(obj, 0) > 0 for t in tr))
    g0 = goals[0]
    art = None
    if g0["kind"] in ("joint_min", "joint_le", "joint_max"):
        q = np.array([t["joints"][g0["asset"]][g0["joint"]] for t in tr])
        art = dict(q0=float(q[0]), qmax=float(q.max()), qmin=float(q.min()), threshold=g0.get("deg", g0.get("m")))
    final_goal = [g for g in goals if not g.get("ok")]
    reason = e["result"].get("failure_reason")
    if e["result"]["status"] == "passed":
        mode = "passed"
    elif not s1:
        if g0["kind"].startswith("joint"):
            moved = abs(art["qmax"] - art["q0"]) if g0["kind"] != "joint_le" else abs(art["qmin"] - art["q0"])
            mode = "stage1_articulation_untouched" if moved < 0.02 else "stage1_articulation_partial"
        else:
            mode = "stage1_transfer_not_placed" + (
                "_lifted" if lift > 0.03 else ("_grasp_missed" if closed_near else "_not_approached")
            )
    else:
        if lift > 0.03:
            mode = "after_stage1_lifted_not_placed"
        elif closed_near or contact:
            mode = "after_stage1_grasp_missed"
        elif mind < 0.10:
            mode = "after_stage1_approached_no_grasp"
        else:
            mode = "after_stage1_not_approached"
    return dict(
        scene=sc,
        episode=e["index"],
        status=e["result"]["status"],
        reason=reason,
        stage1=s1,
        lift_m=round(lift, 3),
        min_ee_obj_m=round(mind, 3),
        min_at_tick=argmin,
        closed_near=closed_near,
        contact=contact,
        art=art,
        ticks=e["ticks"],
        mode=mode,
    )


rows = []
for d in sorted(glob.glob(S + "/v20_vla/round2/eval/trained_vla/*/episode_*")):
    if os.path.exists(d + "/trace.jsonl"):
        rows.append(analyse(d))
from collections import Counter

modes = Counter(r["mode"] for r in rows)
per = Counter((r["scene"], r["mode"]) for r in rows)
out = dict(n=len(rows), modes=dict(modes), per_scene={f"{k[0]}|{k[1]}": v for k, v in sorted(per.items())}, rows=rows)
json.dump(out, open(S + "/fig7_export/failure_modes_round2.json", "w"), indent=1)
print(json.dumps(dict(n=len(rows), modes=dict(modes))))
for k, v in sorted(per.items()):
    print(k, v)
