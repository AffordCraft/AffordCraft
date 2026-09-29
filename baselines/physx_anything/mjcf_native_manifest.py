#!/usr/bin/env python
"""Parse the official PhysX-Anything MJCF (basic.xml written by 4_simready_gen.py) into the native_manifest.json contract
consumed by the downstream physics gate. Pure XML parsing; MuJoCo is only used by test_mjcf_parser.py to cross-check.

Facts about the official exporter that this parser relies on (4_simready_gen.py, generate_mjcf):
  * <compiler angle="radian">; every mesh asset carries scale = max(Dimension)/100 (cm -> m), the OBJ files are in the
    normalized TRELLIS frame ([-0.5, 0.5] units), so visual_obj is the normalized OBJ and mesh_scale must be applied.
  * one <body name="base"> (freejoint unless fixed_base) holding the geoms of group_0; one <body name="grouppart_k">
    per movable group, re-parented under its parent group's body; all grouppart bodies have pos="0 0 0" and no
    orientation, so the child body frame coincides with its parent frame: joint axis/pos are given in the parent frame.
  * joint types: slide (range in the exporter's normalized units, NOT scaled to metres), hinge (pos scaled to metres,
    range in radians; "continuous" hinges get range +-3000*pi), ball (pos scaled to metres); type-A groups ("move
    freely") are extracted to <worldbody> with their own freejoint (listed as free bodies, not as joints).
  * geom density comes from the per-part <default class> geom (kg/m^3, from the VLM's g/cm^3 value * 1000).
"""
import os, sys, json, math
import xml.etree.ElementTree as ET

JOINT_TYPE = {"slide": "prismatic", "hinge": "revolute", "ball": "spherical"}


def _floats(s):
    return None if s is None else [float(x) for x in s.split()]


def _is_identity_orientation(body):
    e = _floats(body.get("euler"))
    q = _floats(body.get("quat"))
    ax = body.get("axisangle")
    xy = body.get("xyaxes")
    z = body.get("zaxis")
    if ax is not None or xy is not None or z is not None:
        return False
    if e is not None and any(abs(v) > 0 for v in e):
        return False
    if q is not None and (abs(q[0] - 1) > 1e-12 or any(abs(v) > 0 for v in q[1:])):
        return False
    return True


def parse_mjcf(mjcf_path):
    mjcf_path = os.path.abspath(mjcf_path)
    base = os.path.dirname(mjcf_path)
    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    angle = compiler.get("angle", "degree") if compiler is not None else "degree"
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    meshes, textures, materials = {}, {}, {}
    for asset in root.findall("asset"):
        for m in asset.findall("mesh"):
            meshes[m.get("name")] = dict(file=m.get("file"), scale=_floats(m.get("scale", "1 1 1")))
        for t in asset.findall("texture"):
            if t.get("name"):
                textures[t.get("name")] = t.get("file")
        for mat in asset.findall("material"):
            materials[mat.get("name")] = mat.get("texture")
    classes = {}
    top_geom_default = {}
    for d in root.findall("default"):
        g = d.find("geom")
        if g is not None:
            top_geom_default = dict(g.attrib)
        for sub in d.iter("default"):
            cls = sub.get("class")
            g = sub.find("geom")
            if cls and g is not None:
                classes[cls] = dict(g.attrib)
    links, joints, free_bodies, warnings = [], [], [], []

    def walk(body, parent_name, depth):
        name = body.get("name")
        if name is None:
            warnings.append("unnamed body under %s" % parent_name)
            name = "<unnamed@%s>" % parent_name
        pos = _floats(body.get("pos", "0 0 0"))
        identity = _is_identity_orientation(body)
        if not identity:
            warnings.append(
                "body %s has a non-identity orientation; joint axis/origin are reported in the child body frame" % name
            )
        geoms = []
        for g in body.findall("geom"):
            cls = g.get("class")
            attrs = dict(top_geom_default)
            attrs.update(classes.get(cls, {}))
            attrs.update(g.attrib)
            mesh = attrs.get("mesh")
            mi = meshes.get(mesh, {})
            file = mi.get("file")
            obj = os.path.normpath(os.path.join(base, meshdir, file)) if file else None
            if "density" in g.attrib:
                dsrc = "mjcf_geom"
            elif "density" in classes.get(cls, {}):
                dsrc = "mjcf_default_class_geom"
            elif "density" in top_geom_default:
                dsrc = "mjcf_default_geom"
            else:
                dsrc = "mujoco_default_1000"
            density = float(attrs["density"]) if "density" in attrs else 1000.0
            tex = textures.get(materials.get(attrs.get("material")))
            geoms.append(
                dict(
                    name=g.get("name"),
                    mjcf_class=cls,
                    mesh=mesh,
                    geom_type=attrs.get("type", "sphere"),
                    visual_obj=obj,
                    visual_obj_exists=bool(obj and os.path.isfile(obj)),
                    mesh_scale=mi.get("scale"),
                    density_kg_m3=density,
                    density_source=dsrc,
                    material=attrs.get("material"),
                    texture=os.path.normpath(os.path.join(base, texturedir, tex)) if tex else None,
                )
            )
        free = body.find("freejoint") is not None or any(j.get("type") == "free" for j in body.findall("joint"))
        first = geoms[0] if geoms else {}
        link = dict(
            name=name,
            parent=parent_name,
            depth=depth,
            body_pos=pos,
            body_orientation_identity=identity,
            free_joint=free,
            visual_obj=first.get("visual_obj"),
            visual_obj_in_meters=False if first else None,
            mesh_scale=first.get("mesh_scale"),
            density_kg_m3=first.get("density_kg_m3"),
            density_source=first.get("density_source"),
            mass_kg=None,
            geom_count=len(geoms),
            geoms=geoms,
        )
        links.append(link)
        if free:
            free_bodies.append(name)
        for j in body.findall("joint"):
            jt = j.get("type", "hinge")
            if jt == "free":
                continue
            axis = _floats(j.get("axis", "0 0 1"))
            jpos = _floats(j.get("pos", "0 0 0"))
            rng = _floats(j.get("range"))
            lower = upper = None
            if rng:
                lower, upper = rng[0], rng[1]
                if jt == "hinge" and angle == "degree":
                    lower, upper = math.radians(lower), math.radians(upper)
            entry = dict(
                name=j.get("name"),
                type=JOINT_TYPE.get(jt, jt),
                mjcf_type=jt,
                parent=parent_name,
                child=name,
                axis=axis if jt != "ball" else None,
                origin_xyz=[pos[i] + jpos[i] for i in range(3)],
                joint_pos_child_frame=jpos,
                lower=lower,
                upper=upper,
                range_raw=j.get("range"),
                limited=bool(rng),
                damping=_floats(j.get("damping")),
                frictionloss=_floats(j.get("frictionloss")),
                stiffness=_floats(j.get("stiffness")),
            )
            if jt == "slide":
                entry["range_units"] = (
                    "mjcf_verbatim: exporter writes the slide range in normalized object units (voxel/32), not scaled by the mesh scale"
                )
            elif jt == "hinge":
                entry["range_units"] = "radians (compiler angle=%s)" % angle
                if rng and (upper - lower) > 4 * math.pi + 1e-6:
                    entry["unbounded_revolute"] = True
            joints.append(entry)
        for child in body.findall("body"):
            walk(child, name, depth + 1)

    for b in root.find("worldbody").findall("body"):
        walk(b, "world", 1)
    names = [l["name"] for l in links]
    root_link = "base" if "base" in names else (names[0] if names else None)
    return dict(
        model=root.get("model"),
        compiler_angle=angle,
        links=links,
        joints=joints,
        free_bodies=free_bodies,
        root_link=root_link,
        warnings=warnings,
        joint_counts={
            k: sum(1 for j in joints if j["type"] == k) for k in ("revolute", "prismatic", "spherical", "fixed")
        },
    )


def build_native_manifest(case_dir, source_id, method_id, asset_emitted, failure_reason=None, extra=None):
    case_dir = os.path.abspath(case_dir)
    mjcf = os.path.join(case_dir, "basic.xml")
    urdf = os.path.join(case_dir, "basic.urdf")
    man = dict(
        source_id=source_id,
        method_id=method_id,
        asset_emitted=bool(asset_emitted),
        units="m",
        metric_size_source="method",
        native_formats=["MJCF", "URDF"],
        mjcf=mjcf if os.path.isfile(mjcf) else None,
        urdf=urdf if os.path.isfile(urdf) else None,
        links=[],
        joints=[],
        physics_contract=True,
        root_link=None,
        failure_reason=failure_reason,
    )
    man["frame_note"] = (
        "MJCF body frames: base pos 0 0 1 in world; every grouppart body has pos 0 0 0 and identity orientation, so joint "
        "axis and origin_xyz (metres) are expressed in the parent body frame = base frame. visual_obj files are the official "
        "normalized OBJs; apply mesh_scale (metres per normalized unit) to obtain metres. Free (type-A) parts are separate free "
        "bodies listed in free_bodies, not joints."
    )
    if extra:
        man.update(extra)
    if asset_emitted and os.path.isfile(mjcf):
        parsed = parse_mjcf(mjcf)
        man["links"] = [
            dict(
                name=l["name"],
                visual_obj=l["visual_obj"],
                visual_obj_in_meters=l["visual_obj_in_meters"],
                mesh_scale=l["mesh_scale"],
                density_kg_m3=l["density_kg_m3"],
                density_source=l["density_source"],
                mass_kg=None,
                parent=l["parent"],
                free_joint=l["free_joint"],
                body_pos=l["body_pos"],
                geom_count=l["geom_count"],
                geoms=l["geoms"],
            )
            for l in parsed["links"]
        ]
        man["joints"] = [
            dict(
                name=j["name"],
                type=j["type"],
                parent=j["parent"],
                child=j["child"],
                axis=j["axis"],
                origin_xyz=j["origin_xyz"],
                lower=j["lower"],
                upper=j["upper"],
                mjcf_type=j["mjcf_type"],
                range_raw=j["range_raw"],
                range_units=j.get("range_units"),
                unbounded_revolute=j.get("unbounded_revolute", False),
                limited=j["limited"],
            )
            for j in parsed["joints"]
        ]
        man["root_link"] = parsed["root_link"]
        man["free_bodies"] = parsed["free_bodies"]
        man["joint_counts"] = parsed["joint_counts"]
        man["mjcf_model_name"] = parsed["model"]
        man["parser_warnings"] = parsed["warnings"]
        man["all_visual_objs_exist"] = all(g["visual_obj_exists"] for l in parsed["links"] for g in l["geoms"])
    return man


if __name__ == "__main__":
    print(json.dumps(parse_mjcf(sys.argv[1]), indent=1))
