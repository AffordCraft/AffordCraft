#!/usr/bin/env python3
"""External-method physical gates under the frozen AffordCraft contract.

Two conditions per case, both evaluated by the FROZEN physics worker of the method (scripts/physics_worker.py +
affordcraft/physics.py of this repository, EvaluationPolicy 360 steps @ 1/120 s):

  native  : the method's own export (MJCF first, else the native manifest's links/joints) translated to the
            AffordCraft USD layout without changing geometry, joints, scale, density or mass. Collision of every
            delivered part = its convex hull (what MuJoCo does with mesh geoms); mass/CoM/inertia = MuJoCo's own
            compile of the native MJCF (density x mesh volume) when an MJCF exists, else density x hull volume from the
            manifest's density, else the case has no physics contract and the native condition is 'not_applicable'.
  adapted : the native links/joints written as a URDF candidate and passed through the FROZEN construct_asset of the
            method (uniform scale from the main campaign's grounding size estimate for the same input, CoACD decomposition
            cascade with audits, uniform density 500 kg/m3, free-standing support contract) -- i.e. exactly our common
            physical adaptation -- then the same frozen gate.

Nothing here selects among outcomes; every case record is written once.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np

# The frozen method code is this repository: affordcraft/ (method package) and scripts/ (physics worker, builder).
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from affordcraft import paths  # noqa: E402

P = paths.PROJECT_ROOT  # project root handed to the frozen construct_asset (resolves relative asset paths)
VLA_PY = paths.RUNTIME_PYTHON  # trimesh / CoACD / pxr interpreter of the frozen builder (AFFORDCRAFT_RUNTIME_PYTHON)
ISAAC_PY = paths.ISAAC_PYTHON  # Isaac Sim interpreter of the physics worker (AFFORDCRAFT_ISAAC_PYTHON)
# grounding table of the main campaign: {"grounding": {source_id: {"size_m": ..., "size_confidence": ...}}}; --grounding
GROUNDING = Path(__file__).resolve().with_name("grounding_table.json")  # shipped copy (record paths removed)
# machine-level directory of the builder semaphore (amendment 001); every gate process of a machine must share it
SLOTS_DIR = os.environ.get("GATE_SLOTS_DIR") or str(Path(tempfile.gettempdir()) / "affordcraft_gate_slots")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def save_new(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(v, f, indent=1, allow_nan=False)


def frozen_parser():
    spec = importlib.util.spec_from_file_location(
        "affordcraft_urdf_source_parser", CODE / "affordcraft/source_parser.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["affordcraft_urdf_source_parser"] = m
    spec.loader.exec_module(m)
    return m


def quat_to_mat(q):  # wxyz
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_rpy(R):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(R).as_euler("xyz"))


# ----------------------------------------------------------------------------------------------------------------------
# 1. native MJCF -> link/joint records (MuJoCo compile = the delivered physics)
# ----------------------------------------------------------------------------------------------------------------------
def parse_mjcf(mjcf_path, workdir):
    """Return (links, joints, mujoco_info). Link frames: root = root body frame; child = joint anchor frame with the child
    body's orientation (URDF convention). Meshes are returned in those link frames, in meters (MuJoCo applies mesh scale).
    """
    import mujoco, trimesh

    mjcf_path = Path(mjcf_path)
    cwd = os.getcwd()
    os.chdir(mjcf_path.parent)
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path.name))
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nb = model.nbody
    # object bodies = descendants of the first body that carries a free joint or whose parent is world (excluding world 0)
    bodies = [b for b in range(1, nb)]
    if not bodies:
        raise ValueError("mjcf_has_no_bodies")
    names = {b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}" for b in bodies}
    roots = [b for b in bodies if model.body_parentid[b] == 0]
    if not roots:
        raise ValueError("mjcf_root_missing")
    root = roots[0]

    # links in MuJoCo body frames: collect geoms (mesh) per body
    def body_meshes(b):
        out = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            m = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
            fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
            v = np.array(model.mesh_vert[va : va + vn], dtype=np.float64)
            f = np.array(model.mesh_face[fa : fa + fn], dtype=np.int64)
            R = quat_to_mat(model.geom_quat[g])
            t = np.array(model.geom_pos[g])
            v = v @ R.T + t
            out.append((trimesh.Trimesh(vertices=v, faces=f, process=False), None))
        return out

    joints_by_child = {}
    for j in range(model.njnt):
        b = model.jnt_bodyid[j]
        if b == 0:
            continue
        joints_by_child.setdefault(b, []).append(j)
    links = {}
    joints = []
    info = {"root_bodies": [names[r] for r in roots], "bodies": [], "unsupported": []}
    order = []  # topological, every top-level body and its subtree
    stack = list(reversed(roots))
    while stack:
        b = stack.pop()
        order.append(b)
        stack.extend([c for c in bodies if model.body_parentid[c] == b])
    frames = {}  # body -> (shift s in body frame: link frame origin = body origin + s ; orientation = body)
    root_pose = {}
    for b in order:
        js = joints_by_child.get(b, [])
        parent = model.body_parentid[b]
        kinds = [int(model.jnt_type[j]) for j in js]
        if parent == 0:
            frames[b] = np.zeros(3)
            root_pose[names[b]] = (
                [float(x) for x in model.body_pos[b]],
                [float(x) for x in mat_to_rpy(quat_to_mat(model.body_quat[b]))],
            )
            info["bodies"].append(
                {
                    "name": names[b],
                    "root": True,
                    "joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in js],
                    "free_root": any(k == int(mujoco.mjtJoint.mjJNT_FREE) for k in kinds),
                }
            )
            continue
        # one articulated joint per body is what the exporter writes; extra joints are recorded and the first is used
        real = [
            j
            for j in js
            if int(model.jnt_type[j])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_BALL))
        ]
        if len(real) > 1:
            info["unsupported"].append(
                {
                    "body": names[b],
                    "reason": "multiple_joints_first_used",
                    "joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in real],
                }
            )
        j = real[0] if real else None
        Rb = quat_to_mat(model.body_quat[b])
        pb = np.array(model.body_pos[b])
        anchor = np.array(model.jnt_pos[j]) if j is not None else np.zeros(3)  # child body frame
        frames[b] = anchor
        parent_shift = frames[parent]
        origin_parent = pb + Rb @ anchor - parent_shift  # joint frame origin in the parent LINK frame
        rpy = mat_to_rpy(Rb)
        if j is None:
            jt = "fixed"
            axis = None
            lo = hi = None
        else:
            k = int(model.jnt_type[j])
            ax = np.array(model.jnt_axis[j], dtype=np.float64)
            ax = ax / np.linalg.norm(ax)
            limited = bool(model.jnt_limited[j])
            rng = [float(x) for x in model.jnt_range[j]]
            if k == int(mujoco.mjtJoint.mjJNT_HINGE):
                jt = "revolute" if limited else "continuous"
                axis = tuple(ax)
                lo, hi = rng if limited else (None, None)
            elif k == int(mujoco.mjtJoint.mjJNT_SLIDE):
                jt = "prismatic"
                axis = tuple(ax)
                lo, hi = rng if limited else (-0.0, 0.0)
                if not limited:
                    info["unsupported"].append({"body": names[b], "reason": "unlimited_slide_treated_as_zero_range"})
            else:
                jt = "spherical"
                axis = None
                lo = hi = None
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) if j is not None else f"fixed_{names[b]}"
        joints.append(
            {
                "name": jn,
                "type": jt,
                "parent": names[parent],
                "child": names[b],
                "origin_xyz": [float(x) for x in origin_parent],
                "origin_rpy": [float(x) for x in rpy],
                "axis": [float(x) for x in axis] if axis else None,
                "lower": lo,
                "upper": hi,
            }
        )
        info["bodies"].append({"name": names[b], "root": False, "joint": jn, "type": jt})
    for b in order:
        ms = body_meshes(b)
        if not ms:
            links[names[b]] = None
            continue
        s = frames[b]
        parts = []
        for m, dens in ms:
            mm = m.copy()
            mm.apply_translation(-s)
            parts.append(mm)
        import trimesh as tm

        mesh = tm.util.concatenate(parts) if len(parts) > 1 else parts[0]
        ipos = np.array(model.body_ipos[b]) - s
        links[names[b]] = {
            "mesh": mesh,
            "mass": float(model.body_mass[b]),
            "com": [float(x) for x in ipos],
            "inertia_diag": [float(x) for x in model.body_inertia[b]],
            "iquat": [float(x) for x in model.body_iquat[b]],
            "density": [d for _, d in ms],
            "root_pose": root_pose.get(names[b]),
        }
    return links, joints, info


# ----------------------------------------------------------------------------------------------------------------------
# 2. native manifest (contract) -> link/joint records (methods without MJCF)
# ----------------------------------------------------------------------------------------------------------------------
def parse_manifest_links(man):
    import trimesh

    links = {}
    joints = []
    for l in man["links"]:
        # amendment 003: a manifest link without geometry (visual_obj null: the structural PartNet-Mobility
        # 'base' / 'base_helper' links of Articulate-Anything's URDF) is a None link, as parse_mjcf and normalize_tree already
        # represent geometry-less bodies; the frozen code passed None to trimesh.load ("file_type 'None' not supported"), so
        # every Articulate-Anything export was wrongly 'blocked' as native_export_not_loadable.
        if not l.get("visual_obj"):
            links[l["name"]] = None
            continue
        m = trimesh.load(l["visual_obj"], force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate([g for g in m.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)])
        sc = np.asarray(l.get("mesh_scale") or [1, 1, 1], dtype=float)
        v = np.asarray(m.vertices, dtype=np.float64) * sc
        R = quat_to_mat(rpy_to_quat(l.get("origin_rpy") or [0, 0, 0]))
        t = np.asarray(l.get("origin_xyz") or [0, 0, 0], dtype=float)
        v = v @ R.T + t
        mesh = trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces, dtype=np.int64), process=False)
        dens = l.get("density_kg_m3")
        mass = l.get("mass_kg")
        rec = {"mesh": mesh, "density": [dens], "mass": None, "com": None, "inertia_diag": None, "iquat": [1, 0, 0, 0]}
        if dens or mass:
            hull = mesh.convex_hull
            vol = abs(float(hull.volume))
            if vol > 1e-10:
                d = float(dens) if dens else float(mass) / vol
                rec["mass"] = vol * d
                rec["com"] = [float(x) for x in hull.center_mass]
                I = np.asarray(hull.moment_inertia) * d
                w, Rr = np.linalg.eigh((I + I.T) / 2)
                if np.linalg.det(Rr) < 0:
                    Rr[:, 0] *= -1
                from scipy.spatial.transform import Rotation

                q = Rotation.from_matrix(Rr).as_quat()
                rec["inertia_diag"] = [float(x) for x in w]
                rec["iquat"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        links[l["name"]] = rec
    for j in man.get("joints") or []:
        # amendment 002: a fixed joint carries no axis, as in the frozen URDF parser (axis None for kind
        # 'fixed'); PAct's official json_to_urdf writes <axis xyz="0 0 0"/> on its fixed joints, which the frozen
        # _axis_quaternion cannot normalize (ZeroDivisionError -> native condition wrongly 'blocked' for 64/200 cases)
        ax = list(j["axis"]) if (j.get("axis") and j["type"] != "fixed") else None
        joints.append(
            {
                "name": j["name"],
                "type": j["type"],
                "parent": j["parent"],
                "child": j["child"],
                "origin_xyz": list(j.get("origin_xyz") or [0, 0, 0]),
                "origin_rpy": list(j.get("origin_rpy") or [0, 0, 0]),
                "axis": ax,
                "lower": j.get("lower"),
                "upper": j.get("upper"),
            }
        )
    return links, joints


def rpy_to_quat(rpy):
    r, p, y = (x / 2 for x in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


VIRTUAL_ROOT = "structural_root"


def normalize_tree(links, joints):
    """Return (links, joints, root). Exports with several top-level bodies (e.g. PhysX-Omni 'floating' relations, part
    generators without joints) get one empty structural root joined to every top-level body by a fixed joint carrying
    that body's delivered pose -- the same representation PartNet-Mobility uses, which the frozen code handles by removing
    the boundary joints for a free-standing asset."""
    children = {j["child"] for j in joints}
    roots = [n for n in links if n not in children]
    if not roots:
        raise ValueError("tree_root_missing")
    if len(roots) == 1 and links[roots[0]] is not None:
        return links, joints, roots[0]
    if len(roots) == 1:
        return links, joints, roots[0]
    new_links = {VIRTUAL_ROOT: None}
    new_links.update(links)
    new_joints = []
    for r in roots:
        pose = (links[r] or {}).get("root_pose") or ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        new_joints.append(
            {
                "name": f"fixed_structural_{r}",
                "type": "fixed",
                "parent": VIRTUAL_ROOT,
                "child": r,
                "origin_xyz": list(pose[0]),
                "origin_rpy": list(pose[1]),
                "axis": None,
                "lower": None,
                "upper": None,
                "boundary": True,
            }
        )
    return new_links, new_joints + list(joints), VIRTUAL_ROOT


def tree_root(links, joints):
    children = {j["child"] for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"tree_root_ambiguous:{roots}")
    return roots[0]


def forward_kinematics(links, joints, root, positions=None):
    from scipy.spatial.transform import Rotation

    T = {root: np.eye(4)}
    positions = positions or {}
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    stack = [root]
    while stack:
        n = stack.pop()
        for j in by_parent.get(n, []):
            M = np.eye(4)
            M[:3, :3] = Rotation.from_euler("xyz", j["origin_rpy"]).as_matrix()
            M[:3, 3] = j["origin_xyz"]
            q = positions.get(j["name"], 0.0)
            if j["axis"] is not None and q:
                ax = np.asarray(j["axis"]) / np.linalg.norm(j["axis"])
                Q = np.eye(4)
                if j["type"] in ("revolute", "continuous"):
                    Q[:3, :3] = Rotation.from_rotvec(ax * q).as_matrix()
                elif j["type"] == "prismatic":
                    Q[:3, 3] = ax * q
                M = M @ Q
            T[j["child"]] = T[n] @ M
            stack.append(j["child"])
    return T


# ----------------------------------------------------------------------------------------------------------------------
# 3. native USD in the frozen AffordCraft layout
# ----------------------------------------------------------------------------------------------------------------------
def author_native_usd(links, joints, out, source_files):
    """Write out/asset.usda with the frozen author_usda (text), then apply the campaign's post-edits (body transforms,
    axis tokens, continuous flags, articulation self-collision flag, support floor) with pxr in the runtime env (the
    same interpreter the frozen construct_asset uses).
    Returns the job record for the frozen physics worker."""
    from scipy.spatial.transform import Rotation

    h = frozen_parser()
    links, joints, root = normalize_tree(links, joints)
    builds = {}
    physics = {}
    for name, rec in links.items():
        if rec is None:
            builds[name] = h.LinkBuild(h.LinkSpec(name, (), (), None, None, None), None, None, None, None, None)
            continue
        m = rec["mesh"]
        if rec["mass"] is None or not rec["mass"] > 0 or rec["inertia_diag"] is None:
            raise ValueError("native_export_declares_no_physical_parameters")
        w = np.asarray(rec["inertia_diag"], dtype=float)
        q = rec["iquat"]
        Rr = quat_to_mat(q)
        I = Rr @ np.diag(w) @ Rr.T
        p = {
            "mass_kg": float(rec["mass"]),
            "center_of_mass_m": [float(x) for x in rec["com"]],
            "diagonal_inertia_kg_m2": [float(x) for x in w],
            "principal_axes_wxyz": [float(x) for x in q],
            "inertia_matrix_kg_m2": I.tolist(),
            "density_kg_m3": rec["density"],
            "policy": "native_export_as_delivered",
        }
        physics[name] = p
        builds[name] = h.LinkBuild(h.LinkSpec(name, (), (), None, None, None), m, m, m, p, "native_mesh_convex_hull")
    internal = [j for j in joints if links.get(j["parent"]) is not None]
    boundary = [j for j in joints if links.get(j["parent"]) is None]
    jspecs = tuple(
        h.JointSpec(
            j["name"],
            "revolute" if j["type"] == "spherical" else j["type"],
            j["parent"],
            j["child"],
            tuple(j["origin_xyz"]),
            tuple(j["origin_rpy"]),
            tuple(j["axis"]) if j["axis"] else ((1.0, 0.0, 0.0) if j["type"] == "spherical" else None),
            j["lower"],
            j["upper"],
            None,
            None,
        )
        for j in internal
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    usd = out / "asset.usda"
    usd.write_text(h.author_usda("Asset", builds, jspecs, root, 0.0, np), encoding="utf-8")
    init = {
        j["name"]: (min(max(0.0, j["lower"]), j["upper"]) if j["lower"] is not None and j["upper"] is not None else 0.0)
        for j in joints
        if j["type"] != "fixed"
    }
    T = forward_kinematics(links, joints, root, init)
    joints = internal
    cloud = np.concatenate(
        [
            np.asarray(b.visual.vertices) @ T[n][:3, :3].T + T[n][:3, 3]
            for n, b in builds.items()
            if b.visual is not None
        ]
    )
    lo, hi = cloud.min(0), cloud.max(0)
    hull_cloud = np.concatenate(
        [
            np.asarray(b.collision.convex_hull.vertices) @ T[n][:3, :3].T + T[n][:3, 3]
            for n, b in builds.items()
            if b.collision is not None
        ]
    )
    offset = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.002 - min(lo[2], float(hull_cloud[:, 2].min()))])
    body_paths = {n: "/World/Asset/Bodies/" + h._safe(n) for n in builds}
    edits = {
        "usd": str(usd),
        "has_joints": bool(jspecs),
        "bodies": [],
        "joints": [],
        "floor": {"sx": max(5, float(hi[0] - lo[0]) * 3), "sy": max(5, float(hi[1] - lo[1]) * 3)},
    }
    for name, b in builds.items():
        tf = T[name].copy()
        tf[:3, 3] += offset
        edits["bodies"].append({"path": body_paths[name], "transform_rows": tf.tolist(), "empty": b.visual is None})
    expected_internal = 0
    axis_records = {}
    for j, js in zip(joints, jspecs):
        path = "/World/Asset/Joints/" + h._safe(js.name)
        expected_internal += 1
        e = {
            "path": path,
            "type": j["type"],
            "set_axis_x": js.axis is not None and j["type"] != "spherical",
            "continuous": js.joint_type == "continuous",
        }
        if js.axis is not None and j["type"] != "spherical":
            axis = (
                T[j["parent"]][:3, :3] @ Rotation.from_euler("xyz", j["origin_rpy"]).as_matrix() @ np.asarray(js.axis)
            )
            axis /= np.linalg.norm(axis)
            axis_records[path] = axis.tolist()
        edits["joints"].append(e)
    spec = out / "usd_post_edits.json"
    spec.write_text(json.dumps(edits, indent=1))
    p = subprocess.run(
        [VLA_PY, str(Path(__file__).with_name("usd_post_edit.py")), str(spec)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError("usd_post_edit_failed:" + (p.stderr or p.stdout)[-800:])
    rec = {
        "usd": str(usd),
        "sha256": sha(usd),
        "exported": True,
        "task_metadata": {},
        "geometry_source_verified": True,
        "expected_internal_joints": expected_internal,
        "requires_articulation": any(j["type"] != "fixed" for j in joints),
        "expected_joint_axes_world": axis_records,
        "support_surface_z": 0.0,
        "initial_translation_m": offset.tolist(),
        "initial_joint_positions": init,
        "physics_parameters": physics,
        "links": list(links),
        "joints": joints,
        "removed_boundary_joints": [j["name"] for j in boundary],
        "source_files": source_files,
        "collision_policy": "convex_hull_of_each_delivered_part_mesh_as_in_MuJoCo",
        "mass_policy": "as_delivered_by_native_export",
    }
    save_new(out / "native_asset_manifest.json", rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# 4. adapted: URDF candidate -> frozen construct_asset (materialize_asset.py) with the campaign grounding
# ----------------------------------------------------------------------------------------------------------------------
def write_urdf_candidate(links, joints, out, method_id, sid):
    import xml.etree.ElementTree as ET, trimesh

    out = Path(out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    links, joints, root = normalize_tree(links, joints)
    robot = ET.Element("robot", {"name": f"{method_id}_{sid}"})
    for name, rec in links.items():
        l = ET.SubElement(robot, "link", {"name": name})
        if rec is None:
            continue
        fn = f"meshes/{name}.obj"
        rec["mesh"].export(out / fn)
        for kind in ("visual", "collision"):
            e = ET.SubElement(l, kind)
            ET.SubElement(e, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            g = ET.SubElement(e, "geometry")
            ET.SubElement(g, "mesh", {"filename": fn, "scale": "1 1 1"})
    for j in joints:
        jt = {
            "revolute": "revolute",
            "continuous": "continuous",
            "prismatic": "prismatic",
            "fixed": "fixed",
            "spherical": "revolute",
            "floating": "fixed",
        }[j["type"]]
        je = ET.SubElement(robot, "joint", {"name": j["name"], "type": jt})
        ET.SubElement(je, "parent", {"link": j["parent"]})
        ET.SubElement(je, "child", {"link": j["child"]})
        ET.SubElement(
            je,
            "origin",
            {
                "xyz": " ".join(repr(float(x)) for x in j["origin_xyz"]),
                "rpy": " ".join(repr(float(x)) for x in j["origin_rpy"]),
            },
        )
        if jt != "fixed":
            ax = j["axis"] or [1.0, 0.0, 0.0]
            ET.SubElement(je, "axis", {"xyz": " ".join(repr(float(x)) for x in ax)})
            if jt != "continuous":
                lo = j["lower"] if j["lower"] is not None else 0.0
                hi = j["upper"] if j["upper"] is not None else 0.0
                if j["type"] == "spherical":
                    lo, hi = -math.pi, math.pi
                ET.SubElement(
                    je, "limit", {"lower": repr(float(lo)), "upper": repr(float(hi)), "effort": "10", "velocity": "2"}
                )
    urdf = out / "candidate.urdf"
    ET.ElementTree(robot).write(urdf, encoding="utf-8", xml_declaration=True)
    return urdf, root


def acquire_materialize_slot(slots_dir=SLOTS_DIR):
    """Machine-level semaphore around the frozen builder (amendment 001): the containers' memory guard SIGKILLs the
    largest process when the container's memory quota is exhausted (no cgroup oom event), which killed ~40 % of the
    adapted builds while 12+ CoACD decompositions (25-45 GB each) ran side by side. At most GATE_MAT_SLOTS (default 3)
    builder processes run per machine; the wait is recorded separately and never counted as build time."""
    import fcntl, random

    Path(slots_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ns = (
        Path(slots_dir) / "node_slots"
    )  # per-machine default written at deployment (2 on 180 GiB containers, 4 on 360 GiB)
    n = int(os.environ.get("GATE_MAT_SLOTS") or (ns.read_text().strip() if ns.exists() else "3"))
    while True:
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            f = open(f"{slots_dir}/matslot_{i}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f, i, n, time.perf_counter() - t0
            except OSError:
                f.close()
        time.sleep(5)


def run_adapted(links, joints, out, method_id, sid, grounding, cache, threads=8):
    out = Path(out)
    src = out / "source"
    urdf, root = write_urdf_candidate(links, joints, src, method_id, sid)
    candidate = {
        "candidate_id": f"{method_id}:{sid}",
        "urdf": {"path": str(urdf), "sha256": sha(urdf)},
        "source_dir": str(src),
        "source": {"units": "m_as_declared_by_source_urdf", "urdf": {"path": str(urdf), "sha256": sha(urdf)}},
        "structural": {"joint_count": len(joints)},
    }
    request = {
        "project": str(P),
        "input_id": sid,
        "candidate": candidate,
        "grounding": {"size_m": grounding.get("size_m"), "size_confidence": grounding.get("size_confidence", 0)},
        "task_metadata": {},
    }
    req = out / "build_request.json"
    req.write_text(json.dumps(request), encoding="utf-8")
    build = out / "build"
    cmd = [
        VLA_PY,
        str(CODE / "scripts/materialize_asset.py"),
        "--request",
        str(req),
        "--output",
        str(build),
        "--cache",
        str(cache),
        "--budget-seconds",
        "600",
    ]
    slot, slot_index, slot_count, slot_wait = acquire_materialize_slot()
    t = time.perf_counter()
    log = out / "materialize.log"
    with log.open("w") as f:
        try:
            p = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "AFFORDCRAFT_BUILD_THREADS": str(threads),
                },
                timeout=1500,
            )
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout_1500s"
    materialize_seconds = time.perf_counter() - t
    try:
        import fcntl

        fcntl.flock(slot, fcntl.LOCK_UN)
        slot.close()
    except Exception:
        pass
    if (build / "build_result.json").exists():
        res = json.loads((build / "build_result.json").read_text())
    elif rc in (-9, -6, -11, "timeout_1500s"):
        # the frozen builder process itself was killed or hung (host-level condition, not a candidate verdict)
        res = {"exported": False, "status": "infrastructure_blocked", "reason": f"materialize_process_{rc}"}
    else:
        res = {"exported": False, "status": "construction_failed", "reason": f"materialize_exit_{rc}"}
    res["materialize_seconds"] = materialize_seconds
    res["request"] = str(req)
    res["materialize_slot"] = {"index": slot_index, "slots": slot_count, "wait_seconds": slot_wait}
    save_new(out / "adapted_asset_manifest.json", res)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# 5. frozen physics worker over a jobs file
# ----------------------------------------------------------------------------------------------------------------------
class GateServices:
    """One frozen PhysicsServices (two Isaac/PhysX worker processes, queue mode) per lane; jobs are evaluated on the
    'final' worker exactly like the campaign's final validation. Restarted once if a worker process dies."""

    def __init__(self, out, gpu):
        self.out = Path(out)
        self.gpu = gpu
        self.n = 0
        self.services = None

    def _start(self):
        from affordcraft.backend import PhysicsServices

        while True:  # PhysicsServices needs a fresh directory; earlier lane processes may have left some behind
            self.n += 1
            if not (self.out / f"physics_services_{self.n:02d}").exists():
                break
        self.services = PhysicsServices(CODE, self.out / f"physics_services_{self.n:02d}", self.gpu)

    def evaluate(self, job):
        from affordcraft.search import InfrastructureBlocked

        for attempt in range(2):
            if self.services is None:
                self._start()
            try:
                return self.services.evaluate({**job, "exported": True}, "final")
            except InfrastructureBlocked as exc:
                reason = str(exc)
                if attempt == 0 and ("terminated" in reason or "failed_during_request" in reason or "start" in reason):
                    try:
                        self.services.close()
                    except Exception:
                        pass
                    self.services = None
                    continue
                return {"physical_pass": None, "status": "infrastructure_blocked", "error": reason}

    def close(self):
        if self.services is not None:
            try:
                return self.services.close()
            except Exception:
                return None


# ----------------------------------------------------------------------------------------------------------------------
def process_case(man_path, out_root, method_id, grounding_table, cache, gate, phases=("native", "adapted")):
    man = json.loads(Path(man_path).read_text())
    sid = man["source_id"]
    case = Path(out_root) / "cases" / sid
    if (case / "case_result.json").exists():
        return json.loads((case / "case_result.json").read_text())
    if case.exists():
        # an earlier lane process died inside this case: keep its partial evidence, recompute from scratch
        keep = Path(out_root) / "interrupted" / f"{sid}-{int(time.time())}"
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(case), str(keep))
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rec = {
        "source_id": sid,
        "method_id": method_id,
        "native_manifest": {"path": str(man_path), "sha256": sha(man_path)},
        "asset_emitted": bool(man.get("asset_emitted")),
        "native": None,
        "adapted": None,
        "phases": list(phases),
    }
    if not man.get("asset_emitted"):
        rec["native"] = {"status": "not_run", "reason": "method_exported_no_asset", "physical_pass": False}
        rec["adapted"] = {"status": "not_run", "reason": "method_exported_no_asset", "physical_pass": False}
        rec["seconds"] = time.perf_counter() - t0
        save_new(case / "case_result.json", rec)
        return rec
    links = joints = None
    parse_note = None
    try:
        if man.get("mjcf") and Path(man["mjcf"]).is_file():
            links, joints, info = parse_mjcf(man["mjcf"], case)
            parse_note = {"source": "mjcf_mujoco_compile", "info": info}
        else:
            links, joints = parse_manifest_links(man)
            parse_note = {"source": "native_manifest_links"}
        if not any(v is not None for v in links.values()):
            raise ValueError("no_geometry")
    except Exception as exc:
        err = {
            "status": "blocked",
            "reason": f"native_export_not_loadable:{type(exc).__name__}:{exc}",
            "physical_pass": False,
            "traceback": traceback.format_exc(),
        }
        rec["native"] = err
        rec["adapted"] = dict(err)
        rec["seconds"] = time.perf_counter() - t0
        save_new(case / "case_result.json", rec)
        return rec
    rec["parse"] = parse_note
    jobs = []
    if "native" in phases:
        try:
            nrec = author_native_usd(links, joints, case / "native", {"mjcf": man.get("mjcf"), "urdf": man.get("urdf")})
            jobs.append(
                {
                    **{
                        k: nrec[k]
                        for k in (
                            "usd",
                            "sha256",
                            "task_metadata",
                            "geometry_source_verified",
                            "expected_internal_joints",
                            "requires_articulation",
                            "support_surface_z",
                            "expected_joint_axes_world",
                        )
                    },
                    "job_id": "native",
                }
            )
            rec["native"] = {"status": "authored", "usd": nrec["usd"]}
        except Exception as exc:
            rec["native"] = {
                "status": "not_applicable" if "no_physical_parameters" in str(exc) else "blocked",
                "reason": f"{type(exc).__name__}:{exc}",
                "physical_pass": False if "no_physical_parameters" not in str(exc) else None,
            }
    if "adapted" in phases:
        try:
            g = grounding_table.get(sid) or {}
            ares = run_adapted(links, joints, case / "adapted", method_id, sid, g, cache)
            rec["adapted_materialize_seconds"] = ares.get("materialize_seconds")
            if ares.get("exported"):
                jobs.append(
                    {
                        **{
                            k: ares[k]
                            for k in (
                                "usd",
                                "sha256",
                                "task_metadata",
                                "geometry_source_verified",
                                "expected_internal_joints",
                                "requires_articulation",
                                "support_surface_z",
                                "expected_joint_axes_world",
                            )
                            if k in ares
                        },
                        "job_id": "adapted",
                    }
                )
                rec["adapted"] = {
                    "status": "authored",
                    "usd": ares["usd"],
                    "uniform_scale": ares.get("uniform_scale"),
                    "grounding_size_m": g.get("size_m"),
                    "grounding_size_confidence": g.get("size_confidence"),
                }
            else:
                st_ = ares.get("status", "construction_failed")
                rec["adapted"] = {
                    "status": st_,
                    "reason": str(ares.get("reason"))[:400],
                    "physical_pass": None if st_ == "infrastructure_blocked" else False,
                }
        except Exception as exc:
            rec["adapted"] = {
                "status": "blocked",
                "reason": f"{type(exc).__name__}:{exc}",
                "physical_pass": False,
                "traceback": traceback.format_exc(),
            }
    for j in jobs:
        tg = time.perf_counter()
        r = gate.evaluate(j)
        rec[j["job_id"]]["gate_seconds"] = time.perf_counter() - tg
        rec[j["job_id"]].update(
            {
                "status": "evaluated" if type(r.get("physical_pass")) is bool else "infrastructure_blocked",
                "physical_pass": r.get("physical_pass"),
                "failure_reasons": r.get("failure_reasons"),
                "gates": r.get("gates"),
                "measurements": r.get("measurements"),
                "error": r.get("error"),
                "evidence_directory": r.get("evidence_directory"),
                "usd_sha256": r.get("usd_sha256"),
            }
        )
    rec["seconds"] = time.perf_counter() - t0
    save_new(case / "case_result.json", rec)
    return rec


def main():
    global GROUNDING
    a = argparse.ArgumentParser()
    a.add_argument("--manifests", required=True, help="text file: one native_manifest.json path per line")
    a.add_argument("--method-id", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--cache", required=True)
    a.add_argument("--phases", default="native,adapted")
    a.add_argument("--lane", default="lane")
    a.add_argument(
        "--grounding", default=str(GROUNDING), help="grounding table of the main campaign (size estimate per input)"
    )
    args = a.parse_args()
    GROUNDING = Path(args.grounding)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GROUNDING.read_text())["grounding"]
    am = Path(args.out) / "configuration_amendment_001_materialize_slots.json"
    if not am.exists():
        save_new(
            am,
            {
                "amendment": "materialize_slots",
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "slots_per_node": int(
                    os.environ.get("GATE_MAT_SLOTS")
                    or (
                        ((Path(SLOTS_DIR) / "node_slots").read_text().strip())
                        if (Path(SLOTS_DIR) / "node_slots").exists()
                        else "3"
                    )
                ),
                "infra_retries": int(os.environ.get("GATE_INFRA_RETRIES", "4")),
                "reason": "container memory guard SIGKILLed concurrent builders (materialize_process_-9, no cgroup oom event); builds now serialized per node through a slot semaphore, wait time recorded outside build timing; blocked cases re-queued up to infra_retries times, every attempt kept under interrupted/",
            },
        )
    cfg = out / "configuration.json"
    if not cfg.exists():
        save_new(
            cfg,
            {
                "schema": "affordcraft.external_gate_pipeline.v1",
                "method_id": args.method_id,
                "frozen_code": str(CODE),
                "frozen_locks": {
                    p: sha(CODE / p)
                    for p in (
                        "scripts/physics_worker.py",
                        "affordcraft/physics.py",
                        "affordcraft/contracts.py",
                        "affordcraft/build.py",
                        "affordcraft/source_parser.py",
                        "scripts/materialize_asset.py",
                    )
                },
                "pipeline_sha256": sha(__file__),
                "grounding_table": {"path": str(GROUNDING), "sha256": sha(GROUNDING)},
                "evaluation_policy": "EvaluationPolicy() of the frozen method code (360 steps, dt 1/120, tail 60, penetration 1 mm, settle 5 cm / 5 deg, floor tol 12 mm)",
                "native_condition": "native export as delivered: MuJoCo-compiled mass/CoM/inertia and joints of the native MJCF, convex hull collision per delivered part",
                "adapted_condition": "frozen construct_asset (uniform scale from the campaign grounding size estimate of the same input when confidence >= 0.65, decomposition cascade v2 with audits, density 500 kg/m3, free-standing support contract), one attempt, no repair alternative when visual==collision",
                "support_role": "free for every external asset (no dataset-declared installation evidence exists for generated assets)",
                "no_result_selection": True,
                "created": time.time(),
            },
        )
    mans = [l.strip() for l in Path(args.manifests).read_text().splitlines() if l.strip()]
    log = (out / f"{args.lane}.progress.jsonl").open("a")
    gate = GateServices(out / "lanes" / args.lane, args.gpu)
    for i, m in enumerate(mans):
        try:
            r = process_case(m, out, args.method_id, gt, args.cache, gate, tuple(args.phases.split(",")))
            log.write(
                json.dumps(
                    {
                        "i": i,
                        "source_id": r["source_id"],
                        "native": (r.get("native") or {}).get("physical_pass"),
                        "adapted": (r.get("adapted") or {}).get("physical_pass"),
                        "seconds": r.get("seconds"),
                        "time": time.time(),
                    }
                )
                + "\n"
            )
            log.flush()
            print(
                json.dumps(
                    {
                        "i": i,
                        "sid": r["source_id"],
                        "native": (r.get("native") or {}).get("physical_pass"),
                        "adapted": (r.get("adapted") or {}).get("physical_pass"),
                        "s": round(r.get("seconds") or 0),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            log.write(
                json.dumps({"i": i, "manifest": m, "error": repr(exc), "traceback": traceback.format_exc()}) + "\n"
            )
            log.flush()
            print(json.dumps({"i": i, "manifest": m, "error": repr(exc)}), flush=True)
    outcomes = gate.close()
    save_new(
        out / "lanes" / args.lane / "physics_process_outcomes.json", {"outcomes": outcomes, "finished": time.time()}
    )


if __name__ == "__main__":
    main()
