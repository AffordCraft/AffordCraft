#!/usr/bin/env python3
"""Write contract native_manifest.json files for finished PhysX-Omni geometry cases (registered-subset and full-2000 geometry lanes) and a
manifest list for the external gate pipeline. Append-only: an existing native_manifest.json is left untouched."""
import argparse, json, hashlib, os, sys, xml.etree.ElementTree as ET
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def manifest_for(case_dir, run_id):
    res = json.loads((case_dir / "native_asset_result.json").read_text())
    sid = res["source_id"]
    native = case_dir / "native" / sid
    mjcf = native / "basic.xml"
    urdf = native / "basic.urdf"
    emitted = bool(res.get("asset_emitted")) and mjcf.is_file() and urdf.is_file()
    links = []
    joints = []
    root = None
    if emitted:
        t = ET.parse(mjcf).getroot()
        meshes = {m.get("name"): m for m in t.iter("mesh")}
        classes = {}
        for d in t.iter("default"):
            g = d.find("geom")
            if d.get("class") and g is not None:
                classes[d.get("class")] = g

        def body_walk(b, parent):
            nonlocal root
            name = b.get("name")
            if root is None:
                root = name
            for g in b.findall("geom"):
                cls = g.get("class")
                ref = classes.get(cls) if cls else None
                mname = g.get("mesh") or (ref.get("mesh") if ref is not None else None)
                dens = g.get("density") or (ref.get("density") if ref is not None else None)
                if mname and mname in meshes:
                    m = meshes[mname]
                    sc = [float(x) for x in (m.get("scale") or "1 1 1").split()]
                    links.append(
                        {
                            "name": name,
                            "visual_obj": str((native / m.get("file")).resolve()),
                            "mesh_scale": sc,
                            "density_kg_m3": float(dens) if dens else None,
                            "mass_kg": None,
                            "origin_xyz": [0, 0, 0],
                            "origin_rpy": [0, 0, 0],
                            "mjcf_body": name,
                            "mjcf_mesh": mname,
                        }
                    )
            for j in b.findall("joint"):
                typ = {"hinge": "revolute", "slide": "prismatic", "ball": "spherical"}.get(j.get("type"), j.get("type"))
                rng = [float(x) for x in (j.get("range") or "0 0").split()]
                joints.append(
                    {
                        "name": j.get("name"),
                        "type": typ,
                        "parent": parent,
                        "child": name,
                        "axis": [float(x) for x in (j.get("axis") or "0 0 1").split()],
                        "origin_xyz": [float(x) for x in (j.get("pos") or "0 0 0").split()],
                        "lower": rng[0],
                        "upper": rng[1],
                        "mjcf_note": "axis/pos in the child body frame of the MJCF; the gate pipeline recomputes frames from the MuJoCo compile",
                    }
                )
            if b.find("freejoint") is not None and parent is not None:
                joints.append(
                    {
                        "name": f"free_{name}",
                        "type": "floating",
                        "parent": parent,
                        "child": name,
                        "axis": None,
                        "origin_xyz": [0, 0, 0],
                        "lower": None,
                        "upper": None,
                    }
                )
            for c in b.findall("body"):
                body_walk(c, name)

        wb = t.find("worldbody")
        for b in wb.findall("body"):
            body_walk(b, None)
    man = {
        "schema": "affordcraft.external_native_manifest.v1",
        "source_id": sid,
        "method_id": "physx-omni-v0.1",
        "run_id": run_id,
        "asset_emitted": emitted,
        "stage_reached": "export" if emitted else res.get("failed_phase"),
        "failure_reason": None if emitted else res.get("failure_reason"),
        "units": "m",
        "metric_size_source": "method",
        "native_formats": ["MJCF", "URDF"] if emitted else [],
        "mjcf": str(mjcf) if emitted else None,
        "urdf": str(urdf) if emitted else None,
        "root_link": root,
        "links": links,
        "joints": joints,
        "physics_contract": emitted,
        "native_asset_result": {
            "path": str(case_dir / "native_asset_result.json"),
            "sha256": sha(case_dir / "native_asset_result.json"),
        },
        "notes": "official 3jsongen_update.py export; MJCF mesh geoms carry scale (cm->m) and per-part density from basic_info; URDF has visual-only links with placeholder inertia. Bodies with several geoms are one link each.",
    }
    return man


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--lanes", nargs="+", required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--list", required=True)
    a.parse_args()
    args = a.parse_args()
    n = 0
    paths = []
    for lane in args.lanes:
        attempts = sorted([d for d in Path(lane).glob("native_geometry*") if d.is_dir()], key=lambda d: d.name)
        latest = attempts[-1] if attempts else Path(lane) / "native_geometry"
        for case in sorted(latest.glob("cases/*")):
            if not (case / "native_asset_result.json").exists():
                continue
            mp = case / "native_manifest.json"
            if not mp.exists():
                man = manifest_for(case, args.run_id)
                with mp.open("x") as f:
                    json.dump(man, f, indent=1)
                    n += 1
            paths.append(str(mp))
    # One entry per source_id in the list. A case recomputed after a CUDA OOM has two manifests
    # (the OOM attempt: asset_emitted false; the recompute: true); the gate lane processes the first listed, so the
    # not-emitted one produced a 'method_exported_no_asset' verdict that the supersede sweeper removed every 10 min,
    # forever. Keep the emitted manifest(s) when any exists, else the first; the dropped ones stay on disk (append-only).
    by_sid = {}
    for mp in paths:
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        by_sid.setdefault(m["source_id"], []).append((0 if m.get("asset_emitted") else 1, mp))
    dedup = []
    for sid, lst in by_sid.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        if lst[0][0] == 0:
            dedup.extend(mp for pr, mp in lst if pr == 0)
        else:
            dedup.append(lst[0][1])
    Path(args.list).write_text("\n".join(dedup) + "\n")
    print(json.dumps({"written": n, "listed": len(dedup), "manifests_seen": len(paths), "list": args.list}))


if __name__ == "__main__":
    main()
