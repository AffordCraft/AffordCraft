#!/usr/bin/env python
"""urdf_native_manifest.py -- parse the URDF compiled by the official Articulate-Anything joint stage
(joint_actor/iter_<i>/seed_<s>/mobility.urdf, written by articulate_anything.api.odio_urdf.compile_python_to_urdf) into the
affordcraft.external_native_manifest.v1 contract consumed by the downstream physical gate.

Facts about the official export this parser relies on:
  * links are the PartNet-Mobility links of the retrieved template object, renamed to the semantic part names from
    semantics.txt (mask_urdf); each link keeps one <visual>/<collision> pair per original OBJ (absolute mesh filenames
    into the per-case private copy of the object), origin xyz/rpy = 0 unless the joint API translated the link
    (Robot.translate_link); `base` (and `base_helper` when a joint was attached to base) carry no geometry;
  * joints: `base` -> first link is a fixed joint whose rpy comes from the template's own base joint
    (Robot.align_robot_orientation, PartNet-Mobility's y-up -> z-up rotation); every other placement joint is fixed with
    an xyz origin from place_relative_to; make_revolute_joint / make_prismatic_joint write revolute / prismatic joints
    with <axis> in the joint (= child-at-zero) frame, <origin xyz> (pivot) in the parent link frame and <limit>
    lower/upper in radians / metres (revolute limits are degrees converted with np.radians; prismatic: 0..distance);
  * no <inertial>, no mass, no density anywhere -> physics_contract is false.
Per link we write one combined OBJ (all visual meshes of that link with their visual origins baked in) so that a single
`visual_obj` per link is available, and list the source OBJs in `visual_objs`. The OBJ files of the object copy are
never modified. PartNet-Mobility meshes are used at the library's native scale (units "m", metric_size_source
"library", mesh_scale [1,1,1]).
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

MOVABLE = ("revolute", "continuous", "prismatic")


def _floats(s, default):
    if s is None:
        return list(default)
    return [float(x) for x in s.replace(",", " ").split()]


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF rpy = fixed-axis roll, pitch, yaw


def parse_urdf(urdf_path):
    urdf_path = os.path.abspath(urdf_path)
    base = os.path.dirname(urdf_path)
    root = ET.parse(urdf_path).getroot()
    links, joints = [], []
    for link in root.findall("link"):
        visuals, collisions = [], []
        for tag, store in (("visual", visuals), ("collision", collisions)):
            for el in link.findall(tag):
                geom = el.find("geometry")
                mesh = geom.find("mesh") if geom is not None else None
                if mesh is None:
                    store.append(dict(kind=(geom[0].tag if geom is not None and len(geom) else "none"), file=None))
                    continue
                fn = mesh.get("filename")
                path = fn if os.path.isabs(fn) else os.path.normpath(os.path.join(base, fn))
                origin = el.find("origin")
                store.append(
                    dict(
                        kind="mesh",
                        file=path,
                        exists=os.path.isfile(path),
                        scale=_floats(mesh.get("scale"), [1, 1, 1]),
                        origin_xyz=_floats(origin.get("xyz") if origin is not None else None, [0, 0, 0]),
                        origin_rpy=_floats(origin.get("rpy") if origin is not None else None, [0, 0, 0]),
                    )
                )
        inertial = link.find("inertial")
        mass = None
        if inertial is not None and inertial.find("mass") is not None:
            mass = float(inertial.find("mass").get("value"))
        links.append(dict(name=link.get("name"), visuals=visuals, collisions=collisions, mass_kg=mass))
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis_el = joint.find("axis")
        limit = joint.find("limit")
        jtype = joint.get("type")
        xyz = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
        rpy = _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0])
        axis_joint_frame = (
            _floats(axis_el.get("xyz") if axis_el is not None else None, [1, 0, 0]) if jtype in MOVABLE else None
        )
        axis_parent_frame = (
            (rpy_to_matrix(rpy) @ np.array(axis_joint_frame)).tolist() if axis_joint_frame is not None else None
        )
        lower = upper = None
        if limit is not None and jtype in ("revolute", "prismatic"):
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
        joints.append(
            dict(
                name=joint.get("name"),
                type=jtype,
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis_parent_frame,
                axis_joint_frame=axis_joint_frame,
                lower=lower,
                upper=upper,
                effort=(float(limit.get("effort")) if limit is not None and limit.get("effort") else None),
                velocity=(float(limit.get("velocity")) if limit is not None and limit.get("velocity") else None),
            )
        )
    children = {j["child"] for j in joints}
    roots = [l["name"] for l in links if l["name"] not in children]
    return dict(urdf=urdf_path, robot_name=root.get("name"), links=links, joints=joints, roots=roots)


def combine_link_mesh(link, out_path):
    """One OBJ per link: every visual mesh transformed by its visual origin, concatenated. Returns stats or raises."""
    import trimesh

    parts = []
    for v in link["visuals"]:
        if v.get("kind") != "mesh" or not v.get("exists"):
            continue
        m = trimesh.load(v["file"], force="mesh", process=False)
        if not isinstance(m, trimesh.Trimesh) or m.vertices.shape[0] == 0:
            continue
        m = m.copy()
        if any(abs(s - 1.0) > 1e-12 for s in v["scale"]):
            m.apply_scale(v["scale"])
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(v["origin_rpy"])
        T[:3, 3] = v["origin_xyz"]
        if not np.allclose(T, np.eye(4)):
            m.apply_transform(T)
        parts.append(m)
    if not parts:
        return None
    merged = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.export(out_path, file_type="obj")
    ext = merged.bounds.tolist() if merged.vertices.shape[0] else None
    return dict(
        vertices=int(merged.vertices.shape[0]), faces=int(merged.faces.shape[0]), bounds=ext, source_meshes=len(parts)
    )


def build_native_manifest(
    case_dir,
    source_id,
    method_id,
    run_id,
    urdf_path,
    asset_emitted,
    stage_reached,
    failure_reason=None,
    mesh_out_dir=None,
    extra=None,
    combine_meshes=True,
):
    case_dir = os.path.abspath(case_dir)
    man = dict(
        schema="affordcraft.external_native_manifest.v1",
        source_id=source_id,
        method_id=method_id,
        run_id=run_id,
        asset_emitted=bool(asset_emitted),
        stage_reached=stage_reached,
        failure_reason=failure_reason,
        units="m",
        metric_size_source="library",
        native_formats=["URDF"],
        mjcf=None,
        urdf=(os.path.abspath(urdf_path) if urdf_path and os.path.isfile(urdf_path) else None),
        root_link=None,
        links=[],
        joints=[],
        physics_contract=False,
        notes=(
            "Articulate-Anything image pipeline: links are the retrieved PartNet-Mobility template parts (semantic names) placed by the VLM link-placement "
            "program; joints come from the VLM joint-prediction program compiled by odio_urdf (revolute/prismatic via make_revolute_joint/make_prismatic_joint, "
            "fixed placements via place_relative_to, base->first link fixed joint carries the library y-up->z-up rpy). visual_obj is a per-link OBJ that "
            "concatenates the link's visual meshes with their URDF visual origins baked in (sources in visual_objs; original OBJs untouched); "
            'mesh_scale [1,1,1] and units "m" at the PartNet-Mobility library scale (metric_size_source "library"). Joint origin_xyz is the URDF joint origin '
            "in the parent link frame, axis is rotated into the parent link frame (axis_joint_frame keeps the URDF value), origin_rpy is the URDF joint "
            "rpy (child frame orientation relative to the parent; only the base joint is non-zero), lower/upper in radians (revolute) / metres (prismatic), "
            "null for fixed. base/base_helper carry no geometry (visual_obj null). No inertial/mass/density in the export -> physics_contract false."
        ),
    )
    if extra:
        man.update(extra)
    if not man["urdf"] or not asset_emitted:
        return man
    parsed = parse_urdf(man["urdf"])
    man["root_link"] = (
        "base" if "base" in [l["name"] for l in parsed["links"]] else (parsed["roots"][0] if parsed["roots"] else None)
    )
    man["urdf_roots"] = parsed["roots"]
    mesh_out_dir = mesh_out_dir or os.path.join(case_dir, "native", "link_meshes")
    for l in parsed["links"]:
        srcs = [v["file"] for v in l["visuals"] if v.get("kind") == "mesh"]
        cols = [c["file"] for c in l["collisions"] if c.get("kind") == "mesh"]
        entry = dict(
            name=l["name"],
            visual_obj=None,
            visual_objs=srcs,
            visual_objs_exist=all(v.get("exists", False) for v in l["visuals"] if v.get("kind") == "mesh"),
            collision_objs=cols,
            mesh_scale=[1.0, 1.0, 1.0],
            density_kg_m3=None,
            mass_kg=l["mass_kg"],
            origin_xyz=[0.0, 0.0, 0.0],
            origin_rpy=[0.0, 0.0, 0.0],
            has_geometry=bool(srcs),
            combined_mesh=None,
            visual_origins=[
                dict(xyz=v["origin_xyz"], rpy=v["origin_rpy"]) for v in l["visuals"] if v.get("kind") == "mesh"
            ],
        )
        if srcs and combine_meshes:
            out = os.path.join(mesh_out_dir, l["name"] + ".obj")
            try:
                stats = combine_link_mesh(l, out)
                if stats:
                    entry["visual_obj"] = out
                    entry["combined_mesh"] = stats
            except Exception as e:  # noqa - keep the manifest, report the mesh problem
                entry["combined_mesh_error"] = repr(e)[:300]
            if entry["visual_obj"] is None and srcs:
                entry["visual_obj"] = srcs[0]
                entry["visual_obj_note"] = (
                    "combined mesh unavailable; first source OBJ listed (origins in visual_origins)"
                )
                entry["origin_xyz"] = l["visuals"][0]["origin_xyz"]
                entry["origin_rpy"] = l["visuals"][0]["origin_rpy"]
        man["links"].append(entry)
    for j in parsed["joints"]:
        man["joints"].append(
            dict(
                name=j["name"],
                type=j["type"],
                parent=j["parent"],
                child=j["child"],
                axis=j["axis"],
                axis_joint_frame=j["axis_joint_frame"],
                origin_xyz=j["origin_xyz"],
                origin_rpy=j["origin_rpy"],
                lower=j["lower"],
                upper=j["upper"],
                effort=j["effort"],
                velocity=j["velocity"],
            )
        )
    man["joint_counts"] = {
        k: sum(1 for j in parsed["joints"] if j["type"] == k) for k in ("revolute", "continuous", "prismatic", "fixed")
    }
    man["part_count"] = sum(1 for l in man["links"] if l["has_geometry"])
    man["all_visual_objs_exist"] = all(l["visual_objs_exist"] for l in man["links"])
    return man


if __name__ == "__main__":
    import json

    print(json.dumps(parse_urdf(sys.argv[1]), indent=1)[:4000])
