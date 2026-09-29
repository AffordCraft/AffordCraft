#!/usr/bin/env python
"""list_library_gaps.py: list the library objects without a usable robot_frontview.png for render_library_gaps.py
(the library completion added after the first attempt hit empty renders, e.g. objects 45443 / 101399): 'missing' = no
PNG, 'bad' = zero-byte PNG; categories from meta.json (logging only). Writes $AA_WORK/logs/library_render_gaps.json
(open 'x'). Environment: ARTICULATE_ANYTHING_CKPT (partnet-mobility-v0 library), AA_WORK."""
import json, os, time

AA_WORK = os.path.abspath(os.environ.get("AA_WORK", "aa_run"))
LIB = os.path.join(os.path.abspath(os.environ.get("ARTICULATE_ANYTHING_CKPT", "partnet-mobility-v0")), "dataset")
OUT = os.path.join(AA_WORK, "logs/library_render_gaps.json")
objs = sorted(
    (d for d in os.listdir(LIB) if os.path.isdir(os.path.join(LIB, d))), key=lambda x: int(x) if x.isdigit() else 0
)
png = lambda o: os.path.join(LIB, o, "robot_frontview.png")
missing = [o for o in objs if not os.path.exists(png(o))]
bad = [o for o in objs if os.path.exists(png(o)) and os.path.getsize(png(o)) == 0]
cats = {}
for o in missing + bad:
    try:
        cats[o] = json.load(open(os.path.join(LIB, o, "meta.json"))).get("model_cat")
    except Exception:
        cats[o] = None
rec = {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "library": LIB,
    "objects": len(objs),
    "missing": missing,
    "bad": bad,
    "categories": cats,
}
with open(OUT, "x") as f:
    json.dump(rec, f, indent=1)
print(json.dumps({"objects": len(objs), "missing": len(missing), "bad": bad}))
