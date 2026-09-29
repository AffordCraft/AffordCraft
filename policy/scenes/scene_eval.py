"""Independent evaluator for AffordCraft multi-asset scene episodes (pure python, no Isaac).
Reads the per-tick trace written by scene_runner.py and the scene goals; never reads teacher plans."""

import math

HOLD_TICKS = 10


def _last(trace, n):
    return trace[-n:] if len(trace) >= n else trace


def _speed(rows, key):
    c = [r["objects"][key]["center"] for r in rows]
    return max((math.dist(c[i], c[i - 1]) for i in range(1, len(c))), default=0.0)


def _inside(center, aabb, margin=0.005):
    lo, hi = aabb
    return all(lo[i] + margin <= center[i] <= hi[i] - margin for i in range(3))


def _released(rows, obj):
    return all(min(r["proprio"][7:]) >= 0.03 and r["contacts"].get(obj, 0) == 0 for r in rows)


def evaluate(trace, goals, final_aabbs, joint_meta, table_z=0.9):
    """trace: list of ticks {proprio, joints{asset:{joint:q}}, objects{key:{center,aabb}}, contacts{name:n}, links{asset:{link:{far_edge_z}}}}
    final_aabbs: {asset:{link_key: [lo,hi]}} measured at the final tick (link_key='body' or joint name for its child link).
    joint_meta: {asset:{joint:{'type':'revolute'|'prismatic','closed':q_closed}}}"""
    result = dict(status="failed", goals=[], failure_reason=None, hold_ticks=HOLD_TICKS)
    if len(trace) < HOLD_TICKS + 2:
        result["failure_reason"] = "trace_too_short"
        return result
    tail = _last(trace, HOLD_TICKS)
    events = []
    for g in goals:
        kind = g["kind"]
        ok = False
        detail = {}
        if kind == "joint_min":
            thr = math.radians(g["deg"]) if "deg" in g else g["m"]
            qs = [r["joints"][g["asset"]][g["joint"]] for r in trace]
            hit = next((i for i, q in enumerate(qs) if q >= thr), None)
            ok = hit is not None
            detail = dict(threshold=thr, max_q=max(qs), first_tick=hit)
            if ok:
                events.append((hit, kind))
        elif kind == "joint_max":
            thr = math.radians(g["deg"]) if "deg" in g else g["m"]
            qs = [r["joints"][g["asset"]][g["joint"]] for r in tail]
            ok = all(q <= thr for q in qs)
            detail = dict(threshold=thr, final_q=qs[-1])
            if g.get("keep") == "inside":
                prev = [x for x in result["goals"] if x["kind"] == "inside"]
                ok = ok and bool(prev) and all(x["ok"] for x in prev)
        elif kind in ("joint_le", "joint_ge"):
            thr = math.radians(g["deg"]) if "deg" in g else g["m"]
            qs = [r["joints"][g["asset"]][g["joint"]] for r in tail]
            ok = all((q <= thr) if kind == "joint_le" else (q >= thr) for q in qs)
            detail = dict(threshold=thr, final_q=qs[-1])
        elif kind == "joint_closed":
            zs = [r["links"][g["asset"]][g["joint"]]["far_edge_z"] for r in tail]
            base_top = final_aabbs[g["asset"]]["body"][1][2]
            ok = all(z <= base_top + g.get("tol_m", 0.04) for z in zs)
            detail = dict(far_edge_z=zs[-1], base_top=base_top)
        elif kind == "inside":
            aabb = final_aabbs[g["asset"]][g["link"]]
            obj = g["obj"]
            inside = all(_inside(r["objects"][obj]["center"], aabb) for r in tail)
            rel = _released(tail, obj) if g.get("released") else True
            stab = _speed(tail, obj) <= 0.01 if g.get("stable") else True
            ok = inside and rel and stab
            detail = dict(
                inside=inside, released=rel, stable=stab, center=tail[-1]["objects"][obj]["center"], aabb=aabb
            )
        elif kind == "on_top":
            if g.get("link") in (None, "any"):
                boxes = list(final_aabbs[g["asset"]].values())
                aabb = [
                    [min(b[0][i] for b in boxes) for i in range(3)],
                    [max(b[1][i] for b in boxes) for i in range(3)],
                ]
            else:
                aabb = final_aabbs[g["asset"]][g["link"]]
            obj = g["obj"]
            rows = tail
            top = aabb[1][2]

            def supported(r):
                h = r.get("support_hit", {}).get(obj, {})
                ok_link = (h.get("link") == g["link"]) if g.get("link") not in (None, "body", "any") else True
                return h.get("body") == g["asset"] and ok_link

            on = all(
                aabb[0][0] - 0.02 <= r["objects"][obj]["center"][0] <= aabb[1][0] + 0.02
                and aabb[0][1] - 0.02 <= r["objects"][obj]["center"][1] <= aabb[1][1] + 0.02
                and r["objects"][obj]["aabb"][0][2] >= table_z + 0.015
                and (supported(r) or top - 0.01 <= r["objects"][obj]["aabb"][0][2] <= top + 0.04)
                for r in rows
            )
            rel = _released(rows, obj) if g.get("released") else True
            stab = _speed(rows, obj) <= 0.01 if g.get("stable") else True
            ok = on and rel and stab
            detail = dict(on=on, released=rel, stable=stab, bottom_z=rows[-1]["objects"][obj]["aabb"][0][2], top=top)
        else:
            detail = dict(error="unknown_goal_kind")
        result["goals"].append(
            dict(kind=kind, ok=bool(ok), **{k: v for k, v in g.items() if k != "kind"}, detail=detail)
        )
    # every manipulated object must still be on/above the table
    for key, o in trace[-1]["objects"].items():
        if o["center"][2] < table_z - 0.05:
            result["goals"].append(dict(kind="object_on_table", ok=False, obj=key, detail=dict(center=o["center"])))
    failed = [g for g in result["goals"] if not g["ok"]]
    if failed:
        result["failure_reason"] = failed[0]["kind"] + ":" + failed[0].get("asset", failed[0].get("obj", ""))
        return result
    result["status"] = "passed"
    result["sim_task_success"] = True
    return result
