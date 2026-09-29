"""Catalog access, entry summaries, assembly-spec validation, kinematics, native export (URDF + MJCF + OBJ).

All catalog geometry is read through the frozen AffordCraft parser (source_parser.parse_urdf / _combine) and the
frozen CatalogIndex.retrieve; nothing here touches the frozen files.
"""

from __future__ import annotations
import importlib.util, json, math, os, re, subprocess, sys, threading
from collections import OrderedDict
from pathlib import Path
import numpy as np
import trimesh
from common import (
    PROJECT,
    CODE_DIR,
    CATALOG_JSON,
    INDEX_ROOT,
    MODEL_DIR,
    CACHE_ROOT,
    PYTHON,
    RENDER_SIZE,
    rpy_matrix,
    rigid_matrix,
    finite,
    write_json,
    read_json,
    sha256_file,
)

_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
PALETTE = [
    (0.85, 0.30, 0.25),
    (0.25, 0.50, 0.85),
    (0.30, 0.70, 0.35),
    (0.90, 0.65, 0.20),
    (0.60, 0.35, 0.75),
    (0.20, 0.70, 0.70),
    (0.80, 0.45, 0.65),
    (0.55, 0.55, 0.25),
    (0.45, 0.30, 0.20),
    (0.35, 0.35, 0.80),
    (0.75, 0.75, 0.35),
    (0.30, 0.55, 0.55),
]
PALETTE_NAMES = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "teal",
    "pink",
    "olive",
    "brown",
    "indigo",
    "yellow",
    "slate",
]


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


frozen_catalog = _load_module("affordcraft_frozen_catalog", CODE_DIR / "catalog.py")
frozen_parser = _load_module("affordcraft_frozen_source_parser", CODE_DIR / "source_parser.py")


class SpecError(ValueError):
    pass


def _r(x, n=4):
    return [round(float(v), n) for v in x]


# ----------------------------------------------------------------------------------------------------------------------
# Catalog
# ----------------------------------------------------------------------------------------------------------------------
class Catalog:
    def __init__(self, device="cuda:0", load_encoder=True):
        data = json.loads(CATALOG_JSON.read_text())
        self.entries = {str(e["candidate_id"]): e for e in data["entries"]}
        self.categories = sorted({str(e.get("category")) for e in self.entries.values()})
        self.category_fold = {c.casefold(): c for c in self.categories}
        self.lock = threading.Lock()
        self.encoder = None
        self.index = None
        self.device = device
        self._geom_cache = OrderedDict()
        self._geom_lock = threading.Lock()
        self._entry_locks = {}
        self._entry_locks_guard = threading.Lock()
        (CACHE_ROOT / "entry_summary").mkdir(parents=True, exist_ok=True)
        (CACHE_ROOT / "entry_render").mkdir(parents=True, exist_ok=True)
        if load_encoder:
            self.load_encoder()

    def load_encoder(self):
        with self.lock:
            if self.index is None:
                self.encoder = frozen_catalog.VisualEncoder(
                    MODEL_DIR / "image_encoder_dinov2", MODEL_DIR / "feature_extractor_dinov2", device=self.device
                )
                self.index = frozen_catalog.CatalogIndex(PROJECT, CATALOG_JSON, INDEX_ROOT, self.encoder)

    def identity(self):
        return {
            "catalog_json": str(CATALOG_JSON),
            "catalog_sha256": sha256_file(CATALOG_JSON),
            "index_root": str(INDEX_ROOT),
            "index_json": read_json(INDEX_ROOT / "index.json"),
            "features_sha256": sha256_file(INDEX_ROOT / "features.npy"),
            "candidate_ids_sha256": sha256_file(INDEX_ROOT / "candidate_ids.json"),
            "entries": len(self.entries),
            "categories": len(self.categories),
            "frozen_catalog_py_sha256": sha256_file(CODE_DIR / "catalog.py"),
            "frozen_source_parser_py_sha256": sha256_file(CODE_DIR / "source_parser.py"),
        }

    def retrieve(self, image, category, limit):
        """Frozen retrieval: 12 category-preferred + category-blind visual neighbours, top `limit`."""
        self.load_encoder()
        with self.lock:
            return self.index.retrieve(image, category, limit=limit)

    def _entry_lock(self, key):
        with self._entry_locks_guard:
            return self._entry_locks.setdefault(key, threading.Lock())

    def resolve_category(self, name):
        return self.category_fold.get(str(name).strip().casefold())

    def category_suggestions(self, *texts, limit=15):
        out = []
        for t in texts:
            t = str(t or "").strip().casefold()
            if not t:
                continue
            for c in self.categories:
                if (t in c.casefold() or c.casefold() in t) and c not in out:
                    out.append(c)
        return out[:limit]

    # ---- geometry loading -------------------------------------------------------------------------------------------
    def source_type(self, entry):
        if entry.get("urdf"):
            return "partnet_mobility_urdf"
        lib = entry.get("source_library") or (entry.get("source") or {}).get("library") or "rigid"
        return "rigid_" + str(lib)

    def load_geometry(self, entry_id):
        """Returns {'root','links':{name:{'mesh':Trimesh,'visual_meshes':n}}, 'joints':[dict], 'frame', 'units_note', 'dropped'}
        in a Z-up frame with the source units untouched. Cached (LRU 24)."""
        with self._geom_lock:
            if entry_id in self._geom_cache:
                self._geom_cache.move_to_end(entry_id)
                return self._geom_cache[entry_id]
        with self._entry_lock(entry_id):
            with self._geom_lock:
                if entry_id in self._geom_cache:
                    return self._geom_cache[entry_id]
            g = self._load_geometry_uncached(entry_id)
            with self._geom_lock:
                self._geom_cache[entry_id] = g
                while len(self._geom_cache) > 24:
                    self._geom_cache.popitem(last=False)
            return g

    def _load_geometry_uncached(self, entry_id):
        e = self.entries.get(entry_id)
        if e is None:
            raise SpecError(f"unknown entry_id {entry_id!r}")
        if e.get("urdf"):
            urdf = frozen_catalog.resolve(e["urdf"]["path"], PROJECT)
            source_dir = frozen_catalog.resolve(e["source_dir"], PROJECT) if e.get("source_dir") else urdf.parent
            specs, joints, robot = frozen_parser.parse_urdf(urdf, source_dir)
            links = {}
            for n, spec in specs.items():
                mesh = frozen_parser._combine(spec.visuals, np, trimesh) if spec.visuals else None
                links[n] = {"mesh": mesh, "visual_meshes": len(spec.visuals)}
            jlist = [
                {
                    "name": j.name,
                    "type": j.joint_type,
                    "parent": j.parent,
                    "child": j.child,
                    "xyz": list(j.origin_xyz),
                    "rpy": list(j.origin_rpy),
                    "axis": list(j.axis) if j.axis else None,
                    "lower": j.lower,
                    "upper": j.upper,
                }
                for j in joints
            ]
            roots = sorted(set(links) - {j["child"] for j in jlist})
            root = roots[0]
            # Drop empty links (no visual) that hang on fixed joints only; PartNet-Mobility roots 'base' are such frames.
            dropped = []
            helpers = []
            changed = True
            while changed:
                changed = False
                for n in list(links):
                    if links[n]["mesh"] is not None or n in helpers:
                        continue
                    outgoing = [j for j in jlist if j["parent"] == n]
                    incoming = [j for j in jlist if j["child"] == n]
                    if any(j["type"] != "fixed" for j in outgoing) or any(j["type"] != "fixed" for j in incoming):
                        # compound-joint helper frame without geometry: kept for inspection; whole-entry import is refused
                        helpers.append(n)
                        continue
                    if n == root:
                        if len(outgoing) != 1:
                            raise SpecError(
                                f"entry {entry_id}: empty root with {len(outgoing)} children; cannot import"
                            )
                        child = outgoing[0]
                        T = rigid_matrix(child["xyz"], child["rpy"])
                        new_root = child["child"]
                        jlist = [j for j in jlist if j is not child]
                        links[new_root]["root_offset"] = T @ links[new_root].get("root_offset", np.eye(4))
                        del links[n]
                        root = new_root
                        dropped.append(n)
                        changed = True
                        break
                    else:
                        inc = incoming[0]
                        Tin = rigid_matrix(inc["xyz"], inc["rpy"])
                        for j in outgoing:
                            Tj = Tin @ rigid_matrix(j["xyz"], j["rpy"])
                            j["parent"] = inc["parent"]
                            j["xyz"] = Tj[:3, 3].tolist()
                            j["rpy"] = list(_matrix_to_rpy(Tj[:3, :3]))
                        jlist = [j for j in jlist if j is not inc]
                        del links[n]
                        dropped.append(n)
                        changed = True
                        break
            units_note = (
                "PartNet-Mobility source: dataset convention is metres and Z-up, but object sizes are only approximately "
                "real-world; declare the metric size yourself with a uniform scale."
            )
            return {
                "root": root,
                "links": links,
                "joints": jlist,
                "dropped": dropped,
                "helpers": helpers,
                "units_note": units_note,
                "frame": "Z-up source frame",
            }
        # rigid entry
        src = e.get("source") or {}
        vp = frozen_catalog.resolve(src["visual"]["path"], PROJECT)
        mesh = trimesh.load(vp, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)])
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, float), faces=np.asarray(mesh.faces, np.int64), process=False
        )
        archive = str(src.get("archive", {}).get("path", ""))
        if src.get("units") == "m_as_declared_by_source_urdf":
            up = "Z"
            units_note = "YCB source: metres, Z-up."
        elif archive.lower().endswith(".glb"):
            up = "Y"
            units_note = (
                "Objaverse source: units are NOT metric (arbitrary modelling units); rotated from Y-up to Z-up; "
                "you must choose a uniform scale that gives a plausible metric size."
            )
        else:
            up = "Z"
            units_note = "Google Scanned Objects source: metres assumed (Gazebo SDF convention), Z-up."
        if up == "Y":
            T = np.eye(4)
            T[:3, :3] = rpy_matrix((math.pi / 2, 0, 0))
            mesh.apply_transform(T)
        return {
            "root": "body",
            "links": {"body": {"mesh": mesh, "visual_meshes": 1}},
            "joints": [],
            "dropped": [],
            "helpers": [],
            "units_note": units_note,
            "frame": "Z-up (rotated from source Y-up)" if up == "Y" else "Z-up source frame",
        }

    # ---- summaries ----------------------------------------------------------------------------------------------------
    def summary(self, entry_id, detailed=False):
        """Compact entry description; detailed=True adds per-link extents/centres and full joints (cached on disk)."""
        e = self.entries.get(entry_id)
        if e is None:
            raise SpecError(f"unknown entry_id {entry_id!r}")
        cache = CACHE_ROOT / "entry_summary" / f"{entry_id}.json"
        if cache.exists():
            s = read_json(cache)
        else:
            with self._entry_lock(entry_id + ":summary"):
                if cache.exists():
                    s = read_json(cache)
                else:
                    s = self._compute_summary(entry_id)
                    write_json(cache, s)
        if detailed:
            return s
        return {
            k: s[k]
            for k in (
                "entry_id",
                "category",
                "source_type",
                "extent_source_units",
                "units_note",
                "link_count",
                "joint_summary",
                "semantic_parts",
            )
            if k in s
        }

    def _compute_summary(self, entry_id):
        e = self.entries[entry_id]
        g = self.load_geometry(entry_id)
        T = fk_source(g, {})
        links = []
        allv = []
        for n, L in g["links"].items():
            m = L["mesh"]
            Tn = T[n]
            if m is None:
                links.append({"name": n, "helper_frame_without_geometry": True, "frame_origin": _r(Tn[:3, 3])})
                continue
            v = np.asarray(m.vertices) @ Tn[:3, :3].T + Tn[:3, 3]
            allv.append(v)
            lo, hi = v.min(0), v.max(0)
            links.append(
                {
                    "name": n,
                    "extent": _r(hi - lo),
                    "center": _r((lo + hi) / 2),
                    "faces": int(len(m.faces)),
                    "frame_origin": _r(Tn[:3, 3]),
                }
            )
        allv = np.concatenate(allv)
        lo, hi = allv.min(0), allv.max(0)
        joints = []
        for j in g["joints"]:
            Tc = T[j["child"]]
            axis_world = (Tc[:3, :3] @ np.asarray(j["axis"])).tolist() if j.get("axis") else None
            joints.append(
                {
                    "name": j["name"],
                    "type": j["type"],
                    "parent": j["parent"],
                    "child": j["child"],
                    "pivot_entry_frame": _r(Tc[:3, 3]),
                    "axis_entry_frame": _r(axis_world) if axis_world else None,
                    "lower": j["lower"],
                    "upper": j["upper"],
                }
            )
        sem = None
        if e.get("semantic_parts"):
            sem = [f"{p.get('link')}={p.get('part')}({p.get('joint_semantic')})" for p in e["semantic_parts"]]
        jt = [j["type"] for j in joints if j["type"] != "fixed"]
        return {
            "entry_id": entry_id,
            "category": e.get("category"),
            "source_type": self.source_type(e),
            "model_id": e.get("model_id"),
            "extent_source_units": _r(hi - lo),
            "bounds_min": _r(lo),
            "bounds_max": _r(hi),
            "units_note": g["units_note"],
            "frame": g["frame"],
            "root_link": g["root"],
            "link_count": len([l for l in links if not l.get("helper_frame_without_geometry")]),
            "links": links,
            "joints": joints,
            "dropped_empty_frames": g["dropped"],
            "compound_helper_links": g.get("helpers", []),
            "whole_entry_importable": not g.get("helpers"),
            "import_note": (
                None
                if not g.get("helpers")
                else 'this entry has empty helper frame(s) carrying movable joints (compound joints); {"type": "catalog_entry"} import is refused for it - import its geometry links one by one with {"type": "catalog_link"} and define the joints yourself, or choose another entry'
            ),
            "joint_summary": (f"{len(jt)} movable joint(s): " + ", ".join(jt)) if jt else "rigid (no movable joints)",
            "semantic_parts": sem,
            "preview_path": str(frozen_catalog.resolve(e["preview"]["path"], PROJECT)),
        }


def _matrix_to_rpy(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(R)).as_euler("xyz").tolist()


def fk_source(g, q):
    """Transforms of an entry's links in the entry frame at joint positions q (dict name->value)."""
    T = {g["root"]: g["links"][g["root"]].get("root_offset", np.eye(4)).copy()}
    pending = list(g["joints"])
    for _ in range(len(g["links"]) + 2):
        rest = []
        for j in pending:
            if j["parent"] not in T:
                rest.append(j)
                continue
            T[j["child"]] = T[j["parent"]] @ rigid_matrix(j["xyz"], j["rpy"]) @ joint_motion(j, q.get(j["name"], 0.0))
        pending = rest
        if not pending:
            break
    if pending:
        raise SpecError("entry joints do not form a tree")
    return T


def joint_motion(j, value):
    M = np.eye(4)
    if j["type"] in ("revolute", "continuous") and j.get("axis"):
        from scipy.spatial.transform import Rotation

        a = np.asarray(j["axis"], float)
        a /= np.linalg.norm(a)
        M[:3, :3] = Rotation.from_rotvec(a * float(value)).as_matrix()
    elif j["type"] == "prismatic" and j.get("axis"):
        a = np.asarray(j["axis"], float)
        a /= np.linalg.norm(a)
        M[:3, 3] = a * float(value)
    return M


# ----------------------------------------------------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------------------------------------------------
def make_primitive_mesh(kind, size):
    kind = str(kind).lower()
    if not isinstance(size, (list, tuple)) or not size or not all(finite(v) and v > 0 for v in size):
        raise SpecError("size_m must be a non-empty list of positive finite numbers")
    if kind == "box":
        if len(size) != 3:
            raise SpecError("box needs size_m=[x,y,z]")
        if max(size) > 20:
            raise SpecError("primitive larger than 20 m")
        return trimesh.creation.box(extents=[float(v) for v in size]), {
            "kind": "box",
            "size_m": [float(v) for v in size],
        }
    if kind == "cylinder":
        if len(size) != 2:
            raise SpecError("cylinder needs size_m=[radius,height]")
        if max(size) > 20:
            raise SpecError("primitive larger than 20 m")
        return trimesh.creation.cylinder(radius=float(size[0]), height=float(size[1]), sections=48), {
            "kind": "cylinder",
            "size_m": [float(size[0]), float(size[1])],
        }
    if kind == "sphere":
        if len(size) != 1:
            raise SpecError("sphere needs size_m=[radius]")
        if size[0] > 10:
            raise SpecError("primitive larger than 20 m")
        return trimesh.creation.icosphere(subdivisions=3, radius=float(size[0])), {
            "kind": "sphere",
            "size_m": [float(size[0])],
        }
    raise SpecError("kind must be 'box', 'cylinder' or 'sphere'")


def write_obj(path, mesh):
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    lines = ["# affordcraft agent baseline; metres; triangles only"]
    lines.extend(f"v {a:.9g} {b:.9g} {c:.9g}" for a, b, c in v)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------
# Spec validation + expansion
# ----------------------------------------------------------------------------------------------------------------------
def _vec3(v, field, default=None):
    if v is None:
        if default is not None:
            return list(default)
        raise SpecError(f"{field} is required")
    if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(finite(x) for x in v):
        raise SpecError(f"{field} must be a list of 3 finite numbers")
    if max(abs(float(x)) for x in v) > 1e4:
        raise SpecError(f"{field} is unreasonably large")
    return [float(x) for x in v]


def _pose(p, field):
    if p is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if not isinstance(p, dict):
        raise SpecError(f"{field} must be an object with xyz and/or rpy")
    unknown = set(p) - {"xyz", "rpy"}
    if unknown:
        raise SpecError(f"{field}: unknown keys {sorted(unknown)} (use xyz and rpy)")
    return _vec3(p.get("xyz"), field + ".xyz", (0, 0, 0)), _vec3(p.get("rpy"), field + ".rpy", (0, 0, 0))


class Assembly:
    """Expanded, validated assembly: links (meshes in their own frames + world transforms at q=0) and joints."""

    def __init__(self, catalog, spec, case_parts_dir):
        if not isinstance(spec, dict):
            raise SpecError("spec must be a JSON object")
        unknown = set(spec) - {"parts", "joints", "root", "support", "notes"}
        if unknown:
            raise SpecError(f"unknown top-level spec keys: {sorted(unknown)}")
        parts = spec.get("parts")
        joints = spec.get("joints", [])
        if not isinstance(parts, list) or not parts:
            raise SpecError("spec.parts must be a non-empty list")
        if not isinstance(joints, list):
            raise SpecError("spec.joints must be a list")
        if len(parts) > 64:
            raise SpecError("at most 64 parts")
        self.support = spec.get("support", "free_standing")
        if self.support not in ("free_standing", "fixed_root"):
            raise SpecError("spec.support must be 'free_standing' or 'fixed_root'")
        self.notes = str(spec.get("notes", ""))[:2000]
        self.catalog = catalog
        self.case_parts_dir = Path(case_parts_dir)
        self.links = OrderedDict()
        self.internal_joints = []
        self.aliases = {}
        self.part_ids = []
        for i, p in enumerate(parts):
            if not isinstance(p, dict):
                raise SpecError(f"parts[{i}] must be an object")
            pid = p.get("id")
            if not isinstance(pid, str) or not _SAFE_ID.match(pid):
                raise SpecError(f"parts[{i}].id must match ^[A-Za-z][A-Za-z0-9_-]{{0,39}}$ (got {pid!r})")
            if pid in self.part_ids:
                raise SpecError(f"duplicate part id {pid!r}")
            unknown = set(p) - {"id", "source", "scale", "density_kg_m3", "mass_kg", "pose"}
            if unknown:
                raise SpecError(f"part {pid!r}: unknown keys {sorted(unknown)}")
            self.part_ids.append(pid)
            scale = p.get("scale", 1.0)
            if not finite(scale) or not (1e-3 <= float(scale) <= 1e3):
                raise SpecError(f"part {pid!r}: scale must be a finite number in [0.001, 1000]")
            scale = float(scale)
            density = p.get("density_kg_m3")
            mass = p.get("mass_kg")
            if density is None and mass is None:
                raise SpecError(f"part {pid!r}: declare density_kg_m3 or mass_kg (physical parameters are required)")
            if density is not None and (not finite(density) or not (1.0 <= float(density) <= 30000.0)):
                raise SpecError(f"part {pid!r}: density_kg_m3 must be in [1, 30000]")
            if mass is not None and (not finite(mass) or not (1e-4 <= float(mass) <= 1e5)):
                raise SpecError(f"part {pid!r}: mass_kg must be in [0.0001, 100000]")
            density = float(density) if density is not None else None
            mass = float(mass) if mass is not None else None
            pxyz, prpy = _pose(p.get("pose"), f"part {pid!r}.pose")
            Tpose = rigid_matrix(pxyz, prpy)
            src = p.get("source")
            if not isinstance(src, dict) or src.get("type") not in ("primitive", "catalog_link", "catalog_entry"):
                raise SpecError(f"part {pid!r}: source.type must be 'primitive', 'catalog_link' or 'catalog_entry'")
            if src["type"] == "primitive":
                prim = src.get("part_id")
                meta = self.case_parts_dir / f"{prim}.json"
                if not isinstance(prim, str) or not re.match(r"^prim_\d+$", prim) or not meta.exists():
                    raise SpecError(
                        f"part {pid!r}: unknown primitive part_id {prim!r} (create it with make_primitive first)"
                    )
                mesh = trimesh.load(self.case_parts_dir / f"{prim}.obj", force="mesh", process=False)
                self.links[pid] = {
                    "mesh": mesh,
                    "scale": scale,
                    "Tpose": Tpose,
                    "density": density,
                    "mass": mass,
                    "group": None,
                    "source": {"type": "primitive", "part_id": prim, **read_json(meta)},
                    "part_id": pid,
                }
                self.aliases[pid] = pid
            elif src["type"] == "catalog_link":
                eid = str(src.get("entry_id"))
                ln = src.get("link")
                g = catalog.load_geometry(eid)
                if ln not in g["links"] or g["links"][ln]["mesh"] is None:
                    raise SpecError(
                        f'part {pid!r}: entry {eid} has no link {ln!r}; links: {[n for n in g["links"] if g["links"][n]["mesh"] is not None]}'
                    )
                self.links[pid] = {
                    "mesh": g["links"][ln]["mesh"],
                    "scale": scale,
                    "Tpose": Tpose,
                    "density": density,
                    "mass": mass,
                    "group": None,
                    "source": {
                        "type": "catalog_link",
                        "entry_id": eid,
                        "link": ln,
                        "category": catalog.entries[eid].get("category"),
                    },
                    "part_id": pid,
                }
                self.aliases[pid] = pid
            else:
                eid = str(src.get("entry_id"))
                g = catalog.load_geometry(eid)
                if g.get("helpers"):
                    raise SpecError(
                        f"part {pid!r}: entry {eid} contains empty helper link(s) {g['helpers']} carrying movable joints (compound joints); "
                        "whole-entry import is not supported for it - import its geometry links individually with catalog_link and define the joints yourself, or choose another entry"
                    )
                names = {}
                for ln in g["links"]:
                    if g["links"][ln]["mesh"] is None:
                        continue
                    names[ln] = pid if ln == g["root"] else f"{pid}.{ln}"
                for ln, nm in names.items():
                    self.links[nm] = {
                        "mesh": g["links"][ln]["mesh"],
                        "scale": scale,
                        "Tpose": np.eye(4),
                        "density": density,
                        "mass": None,
                        "group": pid,
                        "group_mass": mass,
                        "source": {
                            "type": "catalog_entry",
                            "entry_id": eid,
                            "link": ln,
                            "category": catalog.entries[eid].get("category"),
                        },
                        "part_id": pid,
                    }
                root_offset = g["links"][g["root"]].get("root_offset", np.eye(4))
                self.links[pid]["Tpose"] = Tpose @ _scaled_rigid(root_offset, scale)
                for j in g["joints"]:
                    self.internal_joints.append(
                        {
                            "name": f'{pid}.{j["name"]}',
                            "type": j["type"],
                            "parent": names[j["parent"]],
                            "child": names[j["child"]],
                            "xyz": [scale * x for x in j["xyz"]],
                            "rpy": list(j["rpy"]),
                            "axis": j.get("axis"),
                            "lower": (
                                j["lower"] * scale
                                if j["type"] == "prismatic" and j["lower"] is not None
                                else j["lower"]
                            ),
                            "upper": (
                                j["upper"] * scale
                                if j["type"] == "prismatic" and j["upper"] is not None
                                else j["upper"]
                            ),
                            "origin": "catalog_entry",
                        }
                    )
                self.aliases[pid] = pid
        if len(self.links) > 200:
            raise SpecError("assembly expands to more than 200 links")
        # ---- joints ------------------------------------------------------------------------------------------------
        self.joints = list(self.internal_joints)
        names_seen = {j["name"] for j in self.joints}
        for i, j in enumerate(joints):
            if not isinstance(j, dict):
                raise SpecError(f"joints[{i}] must be an object")
            unknown = set(j) - {"name", "type", "parent", "child", "axis", "origin", "limits"}
            if unknown:
                raise SpecError(f"joints[{i}]: unknown keys {sorted(unknown)}")
            name = j.get("name")
            if not isinstance(name, str) or not _SAFE_ID.match(name):
                raise SpecError(f"joints[{i}].name must match ^[A-Za-z][A-Za-z0-9_-]{{0,39}}$")
            if name in names_seen:
                raise SpecError(f"duplicate joint name {name!r}")
            names_seen.add(name)
            jt = j.get("type")
            if jt not in ("revolute", "prismatic", "fixed"):
                raise SpecError(f"joint {name!r}: type must be 'revolute', 'prismatic' or 'fixed'")
            parent = self._resolve_link(j.get("parent"), f"joint {name!r}.parent")
            child = self._resolve_link(j.get("child"), f"joint {name!r}.child")
            if parent == child:
                raise SpecError(f"joint {name!r}: parent and child are the same link")
            oxyz, orpy = _pose(j.get("origin"), f"joint {name!r}.origin")
            axis = None
            lower = upper = None
            if jt != "fixed":
                axis = _vec3(j.get("axis"), f"joint {name!r}.axis")
                n = math.sqrt(sum(a * a for a in axis))
                if n < 1e-9:
                    raise SpecError(f"joint {name!r}: axis must be non-zero")
                axis = [a / n for a in axis]
                lim = j.get("limits")
                if not isinstance(lim, dict) or not finite(lim.get("lower")) or not finite(lim.get("upper")):
                    raise SpecError(
                        f"joint {name!r}: limits {{lower, upper}} are required for {jt} joints (radians or metres)"
                    )
                lower, upper = float(lim["lower"]), float(lim["upper"])
                if lower > upper:
                    raise SpecError(f"joint {name!r}: lower > upper")
                if jt == "revolute" and (
                    upper - lower > 2 * math.pi + 1e-6 or abs(lower) > 4 * math.pi or abs(upper) > 4 * math.pi
                ):
                    raise SpecError(f"joint {name!r}: revolute limits are radians; range must be <= 2*pi")
                if jt == "prismatic" and (abs(lower) > 5 or abs(upper) > 5):
                    raise SpecError(f"joint {name!r}: prismatic limits are metres and must be within +-5 m")
            self.joints.append(
                {
                    "name": name,
                    "type": jt,
                    "parent": parent,
                    "child": child,
                    "xyz": oxyz,
                    "rpy": orpy,
                    "axis": axis,
                    "lower": lower,
                    "upper": upper,
                    "origin": "agent",
                }
            )
        # ---- tree ---------------------------------------------------------------------------------------------------
        root = spec.get("root")
        if root is None and len(self.part_ids) == 1:
            root = self.part_ids[0]
        self.root = self._resolve_link(root, "spec.root")
        parents = {}
        for j in self.joints:
            if j["child"] in parents:
                raise SpecError(
                    f"link {j['child']!r} has two parent joints ({parents[j['child']]!r} and {j['name']!r})"
                )
            parents[j["child"]] = j["name"]
        if self.root in parents:
            raise SpecError(f"root {self.root!r} must not be the child of a joint")
        orphans = [n for n in self.links if n != self.root and n not in parents]
        if orphans:
            raise SpecError(f"links not connected to the tree (add a joint whose child is each of them): {orphans}")
        children = {n: [] for n in self.links}
        for j in self.joints:
            children[j["parent"]].append(j["child"])
        seen = set()
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n in seen:
                raise SpecError("joint graph contains a cycle")
            seen.add(n)
            stack.extend(children[n])
        if seen != set(self.links):
            raise SpecError(f"links unreachable from root {self.root!r}: {sorted(set(self.links) - seen)}")
        self.warnings = []
        for j in self.joints:
            if j["type"] != "fixed" and j["lower"] is not None and not (j["lower"] - 1e-9 <= 0.0 <= j["upper"] + 1e-9):
                self.warnings.append(
                    f"joint {j['name']}: the rest configuration (0) lies outside its limits [{j['lower']:.3g}, {j['upper']:.3g}]"
                )
        self.T = self.forward_kinematics({})
        self.bounds()

    def _resolve_link(self, name, field):
        if not isinstance(name, str):
            raise SpecError(f"{field} must be a link name string")
        if name in self.links:
            return name
        if name in self.aliases:
            return self.aliases[name]
        raise SpecError(f"{field}: unknown link {name!r}; known links: {list(self.links)[:40]}")

    def forward_kinematics(self, q):
        T = {self.root: np.eye(4)}
        pending = list(self.joints)
        for _ in range(len(self.links) + 2):
            rest = []
            for j in pending:
                if j["parent"] not in T:
                    rest.append(j)
                    continue
                T[j["child"]] = (
                    T[j["parent"]] @ rigid_matrix(j["xyz"], j["rpy"]) @ joint_motion(j, q.get(j["name"], 0.0))
                )
            pending = rest
            if not pending:
                break
        # the root part's pose places the whole assembly in the world (translation and rotation)
        Troot = self.links[self.root]["Tpose"]
        W = np.eye(4)
        W[:3, :3] = Troot[:3, :3]
        W[:3, 3] = Troot[:3, 3]
        return {n: W @ t for n, t in T.items()}

    def world_vertices(self, name, q=None):
        T = self.T if q is None else self.forward_kinematics(q)
        L = self.links[name]
        v = np.asarray(L["mesh"].vertices, float) * L["scale"]
        if name != self.root:
            v = v @ L["Tpose"][:3, :3].T + L["Tpose"][:3, 3]
        Tn = T[name]
        return v @ Tn[:3, :3].T + Tn[:3, 3]

    def bounds(self, q=None):
        allv = np.concatenate([self.world_vertices(n, q) for n in self.links])
        self.lo, self.hi = allv.min(0), allv.max(0)
        ext = self.hi - self.lo
        if not np.isfinite(ext).all():
            raise SpecError("assembly geometry is not finite")
        if max(ext) > 20:
            raise SpecError(f"assembly extent {_r(ext)} m exceeds 20 m; check scale")
        if max(ext) < 0.005:
            raise SpecError(f"assembly extent {_r(ext)} m is smaller than 5 mm; check scale")
        return self.lo, self.hi

    def open_configuration(self):
        q = {}
        for j in self.joints:
            if j["type"] == "fixed" or j["lower"] is None:
                continue
            q[j["name"]] = j["upper"] if abs(j["upper"]) >= abs(j["lower"]) else j["lower"]
        return q

    def describe(self):
        T = self.T
        links = []
        for n in self.links:
            v = self.world_vertices(n)
            lo, hi = v.min(0), v.max(0)
            L = self.links[n]
            src = L["source"]
            s = src.get("type") + (
                ":" + str(src.get("entry_id")) + "/" + str(src.get("link"))
                if src.get("entry_id")
                else ":" + str(src.get("part_id"))
            )
            links.append(
                {
                    "link": n,
                    "source": s,
                    "world_bbox_min_m": _r(lo, 3),
                    "world_bbox_max_m": _r(hi, 3),
                    "frame_origin_world_m": _r(T[n][:3, 3], 3),
                }
            )
        joints = []
        for j in self.joints:
            Tc = T[j["child"]]
            ax = (Tc[:3, :3] @ np.asarray(j["axis"])).tolist() if j["axis"] else None
            joints.append(
                {
                    "joint": j["name"],
                    "type": j["type"],
                    "parent": j["parent"],
                    "child": j["child"],
                    "pivot_world_m": _r(Tc[:3, 3], 3),
                    "axis_world": _r(ax, 3) if ax else None,
                    "lower": j["lower"],
                    "upper": j["upper"],
                }
            )
        return {
            "root": self.root,
            "support": self.support,
            "links": links,
            "joints": joints,
            "world_bbox_min_m": _r(self.lo, 3),
            "world_bbox_max_m": _r(self.hi, 3),
            "extent_m": _r(self.hi - self.lo, 3),
            "lowest_point_z_m": round(float(self.lo[2]), 4),
            "warnings": list(self.warnings),
        }

    # ---- rendering ---------------------------------------------------------------------------------------------------
    def render(self, out_dir, tag, views=None, q=None, timeout=240):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        items = []
        arrays = {}
        names = list(self.links)
        for i, n in enumerate(names):
            v = self.world_vertices(n, q)
            f = np.asarray(self.links[n]["mesh"].faces, np.int64)
            arrays[f"v{i}"] = v.astype(np.float32)
            arrays[f"f{i}"] = f.astype(np.int32)
            items.append(
                {"v": f"v{i}", "f": f"f{i}", "transform": np.eye(4).tolist(), "color": list(PALETTE[i % len(PALETTE)])}
            )
        return run_render(
            out_dir, tag, arrays, items, views or [{"azim": 40, "elev": 25}, {"azim": 130, "elev": 25}], timeout=timeout
        )

    def colour_legend(self):
        return {n: PALETTE_NAMES[i % len(PALETTE_NAMES)] for i, n in enumerate(self.links)}


def _scaled_rigid(T, scale):
    S = np.asarray(T, float).copy()
    S[:3, 3] *= scale
    return S


def run_render(out_dir, tag, arrays, items, views, timeout=240, size=RENDER_SIZE, ground=True):
    out_dir = Path(out_dir)
    npz = out_dir / f"{tag}.npz"
    np.savez(npz, **arrays)
    outs = [str(out_dir / f"{tag}_view{i}.png") for i in range(len(views))]
    job = {"npz": str(npz), "items": items, "views": views, "ground": ground, "size": list(size), "out": outs}
    jp = out_dir / f"{tag}.job.json"
    jp.write_text(json.dumps(job))
    env = dict(
        os.environ,
        PYOPENGL_PLATFORM="egl",
        EGL_DEVICE_ID=os.environ.get("EGL_DEVICE_ID", "0"),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    r = subprocess.run(
        [PYTHON, str(Path(__file__).with_name("render_worker.py")), str(jp)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    for p in (npz, jp):
        try:
            p.unlink()
        except OSError:
            pass
    if r.returncode != 0:
        raise RuntimeError("render failed: " + (r.stderr or r.stdout)[-800:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return outs, info.get("backend")


def render_entry(catalog, entry_id, out_dir, timeout=240):
    """Two renders of a catalog entry: rest configuration and opened configuration (joints at their far limits)."""
    g = catalog.load_geometry(entry_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in g["links"] if g["links"][n]["mesh"] is not None]
    qopen = {}
    for j in g["joints"]:
        if j["type"] != "fixed" and j.get("lower") is not None and j.get("upper") is not None:
            qopen[j["name"]] = j["upper"] if abs(j["upper"]) >= abs(j["lower"]) else j["lower"]
        elif j["type"] == "continuous":
            qopen[j["name"]] = 1.0
    outs = []
    backend = None
    for tag, q in (("rest", {}), ("open", qopen)):
        T = fk_source(g, q)
        arrays = {}
        items = []
        for i, n in enumerate(names):
            m = g["links"][n]["mesh"]
            arrays[f"v{i}"] = (np.asarray(m.vertices) @ T[n][:3, :3].T + T[n][:3, 3]).astype(np.float32)
            arrays[f"f{i}"] = np.asarray(m.faces, np.int32)
            items.append(
                {"v": f"v{i}", "f": f"f{i}", "transform": np.eye(4).tolist(), "color": list(PALETTE[i % len(PALETTE)])}
            )
        o, backend = run_render(
            out_dir, f"{entry_id}_{tag}", arrays, items, [{"azim": 40, "elev": 25}], timeout=timeout, ground=False
        )
        outs.append(o[0])
        if not qopen:
            break
    legend = {n: PALETTE_NAMES[i % len(PALETTE_NAMES)] for i, n in enumerate(names)}
    return outs, legend, backend


# ----------------------------------------------------------------------------------------------------------------------
# Native export
# ----------------------------------------------------------------------------------------------------------------------
def _safe_name(n):
    return re.sub(r"[^A-Za-z0-9_-]+", "__", n)


def _mass_properties(mesh, density, mass):
    """Mass from declared mass or density x volume (watertight volume, else convex-hull volume); inertia from the hull."""
    hull = mesh.convex_hull
    vol = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
    hull_vol = abs(float(hull.volume))
    if not math.isfinite(hull_vol) or hull_vol <= 1e-12:
        raise SpecError("a link has (near) zero volume")
    use_vol = vol if vol > 1e-9 else hull_vol
    if mass is None:
        mass = float(density) * use_vol
    if not (1e-6 <= mass <= 1e6):
        raise SpecError(f"derived mass {mass:.3g} kg is outside [1e-6, 1e6]; check density/scale")
    com = np.asarray(hull.center_mass, float)
    I = np.asarray(hull.moment_inertia, float) * (
        mass / hull_vol
    )  # trimesh: unit-density inertia about the centre of mass
    I = (I + I.T) / 2
    eig = np.linalg.eigvalsh(I)
    if not np.isfinite(eig).all() or eig.min() <= 0:
        raise SpecError("derived inertia is not positive definite")
    return {
        "mass_kg": float(mass),
        "volume_m3": float(use_vol),
        "volume_source": "watertight_mesh" if vol > 1e-9 else "convex_hull",
        "com": com.tolist(),
        "inertia": I.tolist(),
        "density_kg_m3": (float(density) if density is not None else None),
    }


def export_native(asm: Assembly, native_dir):
    """Write meshes/<link>.obj (baked: uniform scale, part pose and joint rotations applied; every link frame has the
    world orientation at the rest configuration with its origin at the joint pivot), asset.urdf, asset.mjcf.xml.
    Returns the manifest link/joint lists plus bookkeeping."""
    native_dir = Path(native_dir)
    (native_dir / "meshes").mkdir(parents=True, exist_ok=False)
    T = asm.T
    names = {n: _safe_name(n) for n in asm.links}
    if len(set(names.values())) != len(names):
        raise SpecError("link names collide after sanitising")
    link_records = []
    link_props = {}
    for n, L in asm.links.items():
        v = asm.world_vertices(n) - T[n][:3, 3]
        mesh = trimesh.Trimesh(vertices=v, faces=np.asarray(L["mesh"].faces, np.int64), process=False)
        obj = native_dir / "meshes" / f"{names[n]}.obj"
        write_obj(obj, mesh)
        grouped_mass = bool(L.get("group")) and L.get("group_mass") is not None
        if grouped_mass:
            props = _mass_properties(mesh, None, 1.0)  # placeholder; the declared entry mass is split by volume below
        else:
            props = _mass_properties(mesh, L["density"], L["mass"])
        link_props[n] = props
        link_records.append(
            {
                "name": names[n],
                "visual_obj": os.path.abspath(obj),
                "mesh_scale": [1.0, 1.0, 1.0],
                "density_kg_m3": (float(L["density"]) if L["density"] is not None else None),
                "mass_kg": (float(L["mass"]) if L["mass"] is not None else None),
                "origin_xyz": [0.0, 0.0, 0.0],
                "origin_rpy": [0.0, 0.0, 0.0],
            }
        )
    # a mass declared for a whole imported entry is split over its links by volume
    groups = {}
    for n, L in asm.links.items():
        if L.get("group") and L.get("group_mass") is not None:
            groups.setdefault(L["group"], []).append(n)
    for gname, members in groups.items():
        total_v = sum(link_props[m]["volume_m3"] for m in members)
        gm = float(asm.links[members[0]]["group_mass"])
        for m in members:
            share = gm * link_props[m]["volume_m3"] / total_v
            I = np.asarray(link_props[m]["inertia"]) * (share / link_props[m]["mass_kg"])
            link_props[m]["mass_kg"] = share
            link_props[m]["inertia"] = I.tolist()
    for rec, n in zip(link_records, asm.links):
        if rec["mass_kg"] is None and rec["density_kg_m3"] is None:
            rec["mass_kg"] = float(link_props[n]["mass_kg"])
    joint_records = []
    for j in asm.joints:
        Tc = T[j["child"]]
        Tp = T[j["parent"]]
        axis = (Tc[:3, :3] @ np.asarray(j["axis"], float)).tolist() if j["axis"] else None
        joint_records.append(
            {
                "name": _safe_name(j["name"]),
                "type": j["type"],
                "parent": names[j["parent"]],
                "child": names[j["child"]],
                "axis": ([float(a) for a in axis] if axis else [0.0, 0.0, 1.0]),
                "origin_xyz": (Tc[:3, 3] - Tp[:3, 3]).tolist(),
                "lower": (float(j["lower"]) if j["lower"] is not None else None),
                "upper": (float(j["upper"]) if j["upper"] is not None else None),
            }
        )
    root = names[asm.root]
    urdf = native_dir / "asset.urdf"
    mjcf = native_dir / "asset.mjcf.xml"
    urdf.write_text(_urdf_text(asm, link_records, joint_records, link_props), encoding="utf-8")
    mjcf.write_text(_mjcf_text(asm, link_records, joint_records, root), encoding="utf-8")
    physics_contract = all((r["density_kg_m3"] is not None) or (r["mass_kg"] is not None) for r in link_records)
    return {
        "urdf": os.path.abspath(urdf),
        "mjcf": os.path.abspath(mjcf),
        "root_link": root,
        "links": link_records,
        "joints": joint_records,
        "physics_contract": physics_contract,
        "link_physics": {
            names[n]: {k: link_props[n][k] for k in ("mass_kg", "volume_m3", "volume_source", "density_kg_m3")}
            for n in asm.links
        },
        "root_origin_world": T[asm.root][:3, 3].tolist(),
        "world_bbox_min_m": asm.lo.tolist(),
        "world_bbox_max_m": asm.hi.tolist(),
        "support": asm.support,
    }


def _fmt(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def _urdf_text(asm, links, joints, props):
    L = ['<?xml version="1.0"?>', '<robot name="affordcraft_agent_asset">']
    for rec, n in zip(links, asm.links):
        p = props[n]
        I = p["inertia"]
        L.append(f'  <link name="{rec["name"]}">')
        L.append(
            f'    <inertial><origin xyz="{_fmt(p["com"])}" rpy="0 0 0"/><mass value="{p["mass_kg"]:.9g}"/>'
            f'<inertia ixx="{I[0][0]:.9g}" ixy="{I[0][1]:.9g}" ixz="{I[0][2]:.9g}" iyy="{I[1][1]:.9g}" iyz="{I[1][2]:.9g}" izz="{I[2][2]:.9g}"/></inertial>'
        )
        for tag in ("visual", "collision"):
            L.append(
                f'    <{tag}><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="meshes/{rec["name"]}.obj" scale="1 1 1"/></geometry></{tag}>'
            )
        L.append("  </link>")
    for j in joints:
        L.append(f'  <joint name="{j["name"]}" type="{j["type"]}">')
        L.append(
            f'    <parent link="{j["parent"]}"/><child link="{j["child"]}"/><origin xyz="{_fmt(j["origin_xyz"])}" rpy="0 0 0"/>'
        )
        if j["type"] != "fixed":
            L.append(f'    <axis xyz="{_fmt(j["axis"])}"/>')
            if j["type"] != "continuous":
                L.append(f'    <limit lower="{j["lower"]:.9g}" upper="{j["upper"]:.9g}" effort="100" velocity="1"/>')
        L.append("  </joint>")
    L.append("</robot>")
    return "\n".join(L) + "\n"


def _mjcf_text(asm, links, joints, root):
    by_name = {rec["name"]: rec for rec in links}
    children = {rec["name"]: [] for rec in links}
    jin = {}
    for j in joints:
        children[j["parent"]].append(j)
        jin[j["child"]] = j
    L = [
        '<mujoco model="affordcraft_agent_asset">',
        '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
        '  <option gravity="0 0 -9.81"/>',
        "  <asset>",
    ]
    for rec in links:
        L.append(f'    <mesh name="mesh_{rec["name"]}" file="{rec["name"]}.obj"/>')
    L.append("  </asset>")
    L.append("  <worldbody>")

    def geom(rec):
        if rec["density_kg_m3"] is not None:
            return f'<geom type="mesh" mesh="mesh_{rec["name"]}" density="{rec["density_kg_m3"]:.9g}"/>'
        return f'<geom type="mesh" mesh="mesh_{rec["name"]}" mass="{rec["mass_kg"]:.9g}"/>'

    def body(name, pos, depth):
        ind = "    " + "  " * depth
        L.append(f'{ind}<body name="{name}" pos="{_fmt(pos)}">')
        if name == root and asm.support == "free_standing":
            L.append(f'{ind}  <freejoint name="root_free"/>')
        j = jin.get(name)
        if j is not None and j["type"] != "fixed":
            jt = "hinge" if j["type"] in ("revolute", "continuous") else "slide"
            rng = f' range="{j["lower"]:.9g} {j["upper"]:.9g}"' if j["type"] != "continuous" else ""
            L.append(f'{ind}  <joint name="{j["name"]}" type="{jt}" axis="{_fmt(j["axis"])}"{rng}/>')
        L.append(f"{ind}  {geom(by_name[name])}")
        for cj in children[name]:
            body(cj["child"], cj["origin_xyz"], depth + 1)
        L.append(f"{ind}</body>")

    body(root, asm.T[asm.root][:3, 3].tolist(), 0)
    L.append("  </worldbody>")
    L.append("</mujoco>")
    return "\n".join(L) + "\n"
