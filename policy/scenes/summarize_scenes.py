"""Aggregate scene_runner outputs: per-scene pass counts, failure reasons, teacher errors, mean ticks."""

import json, sys, collections
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for sm in root.glob("**/summary.jsonl"):
    scene = sm.parent.name.rsplit("_w", 1)[0] if "_w" in sm.parent.name else sm.parent.name
    for l in sm.open():
        r = json.loads(l)
        r["scene"] = scene
        r["dir"] = str(sm.parent)
        rows.append(r)
by = collections.defaultdict(list)
for r in rows:
    by[r["scene"]].append(r)
total = 0
passed = 0
for sc in sorted(by):
    rs = by[sc]
    n = len(rs)
    p = sum(r["status"] == "passed" for r in rs)
    total += n
    passed += p
    reasons = collections.Counter((r.get("reason") or "ok") for r in rs if r["status"] != "passed")
    errs = collections.Counter((r.get("error") or "")[:40] for r in rs if r.get("error"))
    ticks = [r["ticks"] for r in rs if r["status"] == "passed"]
    wall = [r["wall_s"] for r in rs]
    print(
        f"{sc:20s} {p:3d}/{n:<3d} pass  mean_ticks_pass={sum(ticks)/len(ticks) if ticks else 0:6.1f} wall={sum(wall)/max(len(wall),1):5.1f}s  fail={dict(reasons.most_common(3))} err={dict(errs.most_common(3))}"
    )
print(f"TOTAL {passed}/{total}")
