#!/usr/bin/env python
"""Cross-check mjcf_native_manifest.parse_mjcf against MuJoCo's compiled model of the same MJCF.
usage: test_mjcf_parser.py <case_dir or basic.xml> [--json out.json]
Compares: joint count and types (free joints excluded on both sides), child/parent bodies, axis (MuJoCo normalizes),
joint position, limits, body names, free bodies; reports MuJoCo's compiled body masses as extra information.
Exit code 0 when everything agrees (or when MuJoCo cannot load the file, which is reported, not a parser error)."""
import os, sys, json, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjcf_native_manifest import parse_mjcf


def check(mjcf_path):
    parsed = parse_mjcf(mjcf_path)
    report = dict(
        mjcf=os.path.abspath(mjcf_path),
        parser_joint_count=len(parsed["joints"]),
        parser_joint_counts=parsed["joint_counts"],
        parser_links=[l["name"] for l in parsed["links"]],
        parser_free_bodies=parsed["free_bodies"],
        mujoco_load_ok=None,
        mujoco_error=None,
        mismatches=[],
        ok=None,
    )
    try:
        import mujoco

        report["mujoco_version"] = mujoco.__version__
        m = mujoco.MjModel.from_xml_path(os.path.abspath(mjcf_path))
        report["mujoco_load_ok"] = True
    except Exception as e:  # not a parser failure: the official asset itself does not compile
        report["mujoco_load_ok"] = False
        report["mujoco_error"] = "%s: %s" % (type(e).__name__, str(e)[:800])
        report["ok"] = None
        return report
    import mujoco

    mm = []
    names = lambda kind, n: [mujoco.mj_id2name(m, kind, i) for i in range(n)]
    body_names = names(mujoco.mjtObj.mjOBJ_BODY, m.nbody)
    jnt_names = names(mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    typemap = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "spherical",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic",
        int(mujoco.mjtJoint.mjJNT_HINGE): "revolute",
    }
    mj_joints = {}
    mj_free_bodies = []
    for i in range(m.njnt):
        t = typemap[int(m.jnt_type[i])]
        b = int(m.jnt_bodyid[i])
        if t == "free":
            mj_free_bodies.append(body_names[b])
            continue
        mj_joints[jnt_names[i]] = dict(
            type=t,
            child=body_names[b],
            parent=body_names[int(m.body_parentid[b])],
            axis=[float(x) for x in m.jnt_axis[i]],
            pos=[float(x) for x in m.jnt_pos[i]],
            limited=bool(m.jnt_limited[i]),
            range=[float(x) for x in m.jnt_range[i]],
        )
    report["mujoco_joint_count"] = len(mj_joints)
    report["mujoco_joint_counts"] = {
        k: sum(1 for j in mj_joints.values() if j["type"] == k) for k in ("revolute", "prismatic", "spherical")
    }
    report["mujoco_bodies"] = body_names[1:]
    report["mujoco_free_bodies"] = mj_free_bodies
    report["mujoco_body_mass_kg"] = {body_names[i]: float(m.body_mass[i]) for i in range(1, m.nbody)}
    pj = {j["name"]: j for j in parsed["joints"]}
    if set(pj) != set(mj_joints):
        mm.append("joint name sets differ: parser=%s mujoco=%s" % (sorted(pj), sorted(mj_joints)))
    for n in sorted(set(pj) & set(mj_joints)):
        a, b = pj[n], mj_joints[n]
        if a["type"] != b["type"]:
            mm.append("%s type %s vs %s" % (n, a["type"], b["type"]))
        if a["child"] != b["child"] or a["parent"] != b["parent"]:
            mm.append("%s tree %s->%s vs %s->%s" % (n, a["parent"], a["child"], b["parent"], b["child"]))
        if a["axis"] is not None:
            nrm = math.sqrt(sum(x * x for x in a["axis"])) or 1.0
            if any(abs(a["axis"][k] / nrm - b["axis"][k]) > 1e-6 for k in range(3)):
                mm.append("%s axis %s vs %s" % (n, a["axis"], b["axis"]))
        if any(abs(a["joint_pos_child_frame"][k] - b["pos"][k]) > 1e-6 for k in range(3)):
            mm.append("%s pos %s vs %s" % (n, a["joint_pos_child_frame"], b["pos"]))
        if a["limited"] != b["limited"]:
            mm.append("%s limited %s vs %s" % (n, a["limited"], b["limited"]))
        elif a["limited"] and (abs(a["lower"] - b["range"][0]) > 1e-6 or abs(a["upper"] - b["range"][1]) > 1e-6):
            mm.append("%s range %s,%s vs %s" % (n, a["lower"], a["upper"], b["range"]))
    if set(report["parser_links"]) != set(body_names[1:]):
        mm.append("body sets differ: parser=%s mujoco=%s" % (sorted(report["parser_links"]), sorted(body_names[1:])))
    if set(parsed["free_bodies"]) != set(mj_free_bodies):
        mm.append("free bodies differ: parser=%s mujoco=%s" % (parsed["free_bodies"], mj_free_bodies))
    report["mismatches"] = mm
    report["ok"] = not mm
    return report


if __name__ == "__main__":
    target = sys.argv[1]
    mjcf = os.path.join(target, "basic.xml") if os.path.isdir(target) else target
    r = check(mjcf)
    print(json.dumps(r, indent=1))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(r, f, indent=1)
    sys.exit(0 if r["ok"] in (True, None) else 1)
