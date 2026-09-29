"""Source-preserving mesh/URDF import and role-aware environment boundaries."""

from pathlib import Path
from dataclasses import replace
import hashlib, importlib.util, json, math, os, sys, time
import numpy as np
from .catalog import resolve, file_sha
from .contracts import SupportContract, fingerprint

# Deterministic decomposition cascade. Every level is audited with the same
# source-clearance checks and the same 2% tolerance; the first passing level
# is used. Finer levels only add hulls, they never replace a concave source
# by a sealed global hull. Level 0 is the single-level decomposition used before the cascade.
CASCADE_VERSION = "decomposition-cascade-v2"
DECOMPOSITION_CASCADE = (
    {
        "level": 0,
        "threshold": 0.05,
        "max_convex_hull": 32,
        "preprocess_resolution": 50,
        "split_components": False,
        "max_faces": None,
        "cap_seconds": 90,
    },
    {
        "level": 1,
        "threshold": 0.05,
        "max_convex_hull": 32,
        "preprocess_resolution": 50,
        "split_components": True,
        "max_faces": None,
        "cap_seconds": 90,
    },
    {
        "level": 2,
        "threshold": 0.02,
        "max_convex_hull": 64,
        "preprocess_resolution": 50,
        "split_components": False,
        "max_faces": None,
        "cap_seconds": 90,
    },
    {
        "level": 3,
        "threshold": 0.02,
        "max_convex_hull": 64,
        "preprocess_resolution": 150,
        "split_components": False,
        "max_faces": 50000,
        "cap_seconds": 120,
    },
    {
        "level": 4,
        "threshold": 0.01,
        "max_convex_hull": 64,
        "preprocess_resolution": 200,
        "split_components": False,
        "max_faces": 50000,
        "cap_seconds": 170,
    },
)
AUDIT_TOLERANCE = 0.02
MIN_LEVEL_SECONDS = 8.0
LEGACY_KEY_SALT = b"coacd-0.05-32-seed20260916-surface-clean"


def parser(project):
    p = Path(__file__).with_name("source_parser.py")
    name = "affordcraft_urdf_source_parser"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name], {"path": str(p), "sha256": file_sha(p)}


def rotation_matrix(rpy):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", rpy).as_matrix()


def forward_kinematics(link_names, joints, root, initial_positions=None):
    initial_positions = initial_positions or {}
    transforms = {root: np.eye(4)}
    pending = list(joints)
    for _ in range(len(link_names) + 1):
        next_pending = []
        for j in pending:
            if j.parent not in transforms:
                next_pending.append(j)
                continue
            if j.child in transforms:
                raise ValueError("non_tree_joint_structure")
            h = np.eye(4)
            h[:3, :3] = rotation_matrix(j.origin_rpy)
            h[:3, 3] = j.origin_xyz
            motion = np.eye(4)
            q = float(initial_positions.get(j.name, 0.0))
            if j.joint_type in ("revolute", "continuous"):
                from scipy.spatial.transform import Rotation

                axis = np.asarray(j.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
            elif j.joint_type == "prismatic":
                axis = np.asarray(j.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                motion[:3, 3] = axis * q
            transforms[j.child] = transforms[j.parent] @ h @ motion
        pending = next_pending
        if not pending:
            break
    if set(transforms) != set(link_names) or pending:
        raise ValueError("disconnected_or_cyclic_structure")
    return transforms


def mass_properties(hulls, density=500.0):
    from scipy.spatial.transform import Rotation

    volume = sum(abs(float(h.volume)) for h in hulls)
    if not math.isfinite(volume) or volume <= 1e-10:
        raise ValueError("invalid_collision_volume")
    masses = np.asarray([abs(float(h.volume)) * density for h in hulls])
    centers = np.asarray([h.center_mass for h in hulls])
    mass = float(masses.sum())
    center = (centers * masses[:, None]).sum(0) / mass
    inertia = np.zeros((3, 3))
    for h, m, c in zip(hulls, masses, centers):
        d = c - center
        inertia += np.asarray(h.moment_inertia) * density + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    inertia = (inertia + inertia.T) / 2
    v, r = np.linalg.eigh(inertia)
    if not np.isfinite(center).all() or not np.isfinite(v).all() or min(v) <= 1e-12 or 2 * max(v) > sum(v) + 1e-8:
        raise ValueError("invalid_derived_inertia")
    if np.linalg.det(r) < 0:
        r[:, 0] *= -1
    q = Rotation.from_matrix(r).as_quat()
    qw = [float(q[3]), *map(float, q[:3])]
    return {
        "mass_kg": mass,
        "center_of_mass_m": center.tolist(),
        "diagonal_inertia_kg_m2": v.tolist(),
        "principal_axes_wxyz": qw,
        "inertia_matrix_kg_m2": inertia.tolist(),
        "density_kg_m3": density,
        "volume_m3": volume,
        "policy": "declared_density_times_compound_collision_volume",
    }


def accessible_space_audit(source, parts, resolution=24):
    """Compare first surface hits; valid for open surfaces without inventing a solid interior.
    A ray counts as newly blocked or advanced only when the hull entry point lies farther than
    1% of the source extent from any source surface: an open shell without edge faces is
    missed by rays travelling in its own plane, and a hull that merely closes that zero-width
    edge occupies no new space."""
    lo, hi = source.bounds
    extent = hi - lo
    span = max(float(max(extent)), 1e-6)
    errors = []
    raw = []
    counts = []
    flagged = []

    def first_hit(mesh, origins, directions):
        loc, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        depth = np.full(len(origins), np.inf)
        if len(rays):
            np.minimum.at(depth, rays, np.linalg.norm(loc - origins[rays], axis=1))
        return depth

    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        u, v = np.meshgrid(
            *[np.linspace(lo[i], hi[i], resolution, endpoint=False) + extent[i] / (2 * resolution) for i in other],
            indexing="ij",
        )
        for sign in (-1, 1):
            origins = np.zeros((resolution * resolution, 3))
            origins[:, other[0]] = u.ravel()
            origins[:, other[1]] = v.ravel()
            origins[:, axis] = (lo[axis] - 0.1 * span) if sign == 1 else (hi[axis] + 0.1 * span)
            dirs = np.zeros_like(origins)
            dirs[:, axis] = sign
            old = first_hit(source, origins, dirs)
            new = np.full(len(origins), np.inf)
            for p in parts:
                new = np.minimum(new, first_hit(p, origins, dirs))
            sealed = np.isinf(old) & np.isfinite(new)
            earlier = np.isfinite(old) & np.isfinite(new) & (new < old - 0.01 * span)
            hit = sealed | earlier
            raw.append(int(hit.sum()))
            counts.append(len(origins))
            if hit.any():
                flagged.append(origins[hit] + dirs[hit] * new[hit][:, None])
    newly = 0
    if flagged:
        points = np.concatenate(flagged)
        _, distance, _ = __import__("trimesh").proximity.closest_point(source, points)
        newly = int(np.sum(distance > 0.01 * span))
    total = sum(counts)
    return {
        "rays": total,
        "newly_blocked_or_advanced_rays": newly,
        "fraction": newly / total,
        "raw_flagged_rays": sum(raw),
        "raw_fraction": sum(raw) / total,
        "tolerance_source_extent_fraction": 0.01,
        "kind": "multi_direction_source_surface_clearance_not_semantic_GT",
    }


def snap_to_source(source, hulls):
    """CoACD remeshes non-manifold input on a voxel grid and returns hulls inflated by up to half
    a voxel. Project every hull vertex onto the supplied source surface and take the convex hull
    of the projections; a degenerate projection keeps the original hull. Deterministic, no tuning."""
    import trimesh

    if not hulls:
        return hulls, 0
    points = np.concatenate([np.asarray(h.vertices) for h in hulls])
    closest, _, _ = trimesh.proximity.closest_point(source, points)
    out = []
    offset = 0
    snapped = 0
    for h in hulls:
        n = len(h.vertices)
        proj = closest[offset : offset + n]
        offset += n
        try:
            c = trimesh.convex.convex_hull(proj)
            if c.is_watertight and float(c.volume) > 1e-12:
                if c.volume < 0:
                    c.invert()
                out.append(c)
                snapped += 1
                continue
        except Exception:
            pass
        out.append(h)
    return out, snapped


def clean_source(mesh):
    source = mesh.copy()
    source.merge_vertices(digits_vertex=12)
    # CAD exports can duplicate every surface with reversed winding. Removing
    # duplicate triangles changes neither the surface nor the operating space.
    _, face_ids = np.unique(np.sort(source.faces, axis=1), axis=0, return_index=True)
    source.update_faces(np.sort(face_ids))
    source.remove_unreferenced_vertices()
    source.fix_normals(multibody=True)
    return source


def _coacd_child(conn, vertices, faces, level):
    try:
        import coacd

        coacd.set_log_level("error")
        raw = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=level["threshold"],
            max_convex_hull=level["max_convex_hull"],
            preprocess_resolution=level["preprocess_resolution"],
            seed=20260916,
            preprocess_mode="auto",
            decimate=False,
            extrude=False,
        )
        conn.send({"parts": [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)) for v, f in raw]})
    except BaseException as exc:
        try:
            conn.send({"error": repr(exc)[:300]})
        except Exception:
            pass
    finally:
        conn.close()


def run_coacd(source, level, timeout=None):
    """CoACD runs in a forked child so a runaway level can be killed and its memory
    released. Level parameters are frozen; only the wall-clock cap is a runtime guard."""
    import trimesh, multiprocessing

    vertices = np.ascontiguousarray(source.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(source.faces, dtype=np.int32)
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_coacd_child, args=(child, vertices, faces, level), daemon=True)
    proc.start()
    child.close()
    payload = None
    timed_out = False
    try:
        if parent.poll(timeout):
            payload = parent.recv()
        else:
            timed_out = True
    except (EOFError, OSError):
        payload = None
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    parent.close()
    if payload is None:
        if timed_out:
            raise TimeoutError("decomposition_level_timeout")
        raise RuntimeError(f"decomposition_level_process_exit_{proc.exitcode}")
    if payload.get("error"):
        raise RuntimeError("decomposition_level_error:" + payload["error"])
    parts = [trimesh.Trimesh(vertices=v, faces=f, process=True) for v, f in payload["parts"]]
    for part in parts:
        if part.volume < 0:
            part.invert()
    return parts


def audit_parts(source, parts, mode="auto"):
    """Same two closure audits as the single-level decomposition; a level passes only if its audit passes.
    mode='rays' forces the first-hit audit, which stays valid for multi-element links whose
    closed components overlap (point containment by parity is not)."""
    if not parts or any(not h.is_watertight or abs(float(h.volume)) <= 1e-12 for h in parts):
        return {"valid": False, "pass": False, "reason": "collision_decomposition_invalid"}
    # Do not replace a concave part by a sealed global hull. For closed sources,
    # sample internal free space independently of task labels and preserve large voids.
    if mode == "auto" and source.is_watertight and not source.is_convex:
        lo, hi = source.bounds
        grid = np.meshgrid(
            *[np.linspace(lo[i], hi[i], 18, endpoint=False) + (hi[i] - lo[i]) / 36 for i in range(3)], indexing="ij"
        )
        points = np.stack([a.ravel() for a in grid], axis=1)
        original = source.contains(points)
        assembled = np.zeros(len(points), dtype=bool)
        for h in parts:
            assembled |= h.contains(points)
        void_added = float(np.mean(assembled & ~original))
        return {
            "valid": True,
            "pass": void_added <= AUDIT_TOLERANCE,
            "reason": None if void_added <= AUDIT_TOLERANCE else "decomposition_closes_source_free_space",
            "check": "free_space",
            "added_occupied_fraction": void_added,
        }
    ray_audit = accessible_space_audit(source, parts)
    return {
        "valid": True,
        "pass": ray_audit["fraction"] <= AUDIT_TOLERANCE,
        "reason": None if ray_audit["fraction"] <= AUDIT_TOLERANCE else "decomposition_closes_source_accessible_space",
        "check": "accessible_space",
        **ray_audit,
    }


def run_level(source, components, level, cap):
    """One cascade level on one cleaned piece; wall-clock capped, never raises."""
    tick = time.perf_counter()
    snapped = 0
    try:
        if level["split_components"]:
            parts = []
            for component in components:
                if len(component.faces) < 4:
                    continue
                left = cap - (time.perf_counter() - tick)
                if left < MIN_LEVEL_SECONDS / 2:
                    raise TimeoutError("decomposition_level_timeout")
                parts.extend(run_coacd(component, level, left))
        else:
            parts = run_coacd(source, level, cap)
        parts, snapped = snap_to_source(source, parts)
        audit = audit_parts(source, parts)
    except TimeoutError:
        parts = []
        audit = {"valid": False, "pass": False, "reason": "decomposition_level_timeout", "timeout": True}
    except Exception as exc:
        parts = []
        audit = {"valid": False, "pass": False, "reason": "decomposition_level_error:" + repr(exc)[:120]}
    record = {
        "level": level["level"],
        "seconds": time.perf_counter() - tick,
        "valid": bool(audit.get("valid")),
        "hull_count": len(parts) if audit.get("valid") else None,
        "snapped_hulls": snapped,
        **{k: v for k, v in audit.items() if k != "valid"},
    }
    return (parts if record["valid"] else []), record


def publish_cache_dir(cache, key, files, payload, name):
    """Atomic publication so concurrent workers never read a half-written entry."""
    path = Path(cache) / key
    path.mkdir(parents=True, exist_ok=True)
    if (path / name).exists():
        return path
    tmp = Path(cache) / (key + f".tmp-{os.getpid()}-{time.time_ns()}")
    tmp.mkdir(parents=True, exist_ok=False)
    for filename, writer in files:
        writer(tmp / filename)
    (tmp / name).write_text(json.dumps(payload, indent=2))
    for item in sorted(tmp.iterdir(), key=lambda p: p.name == name):
        target = path / item.name
        if not target.exists():
            try:
                os.replace(item, target)
            except OSError:
                pass
    for item in tmp.iterdir():
        item.unlink()
    tmp.rmdir()
    return path


def load_level(cache, key, level):
    import trimesh

    d = Path(cache) / key / f"level_{level}"
    r = d / "result.json"
    if not r.exists():
        return None
    data = json.loads(r.read_text())
    parts = [trimesh.load_mesh(d / x, process=False) for x in data.get("hulls", [])] if data.get("valid") else []
    return parts, {**data, "from_cache": True}


class Piece:
    """One supplied collision element: cleaned source, cache key and best audited hulls so far."""

    def __init__(self, mesh):
        self.source = clean_source(mesh)
        self.key = hashlib.sha256(
            np.asarray(self.source.vertices, dtype="<f8").tobytes()
            + np.asarray(self.source.faces, dtype="<i8").tobytes()
            + LEGACY_KEY_SALT
            + b";"
            + CASCADE_VERSION.encode()
        ).hexdigest()
        self.components = None
        self.parts = []
        self.closure = math.inf
        self.passed = False
        self.method = None
        self.trials = []
        if self.source.is_convex and self.source.is_watertight:
            self.parts = [self.source.convex_hull]
            self.closure = 0.0
            self.passed = True
            self.method = "convex_source"
            self.trials.append({"level": "convex_source", "pass": True})

    def step(self, level, cache, cap_fn):
        """Run (or load) one cascade level. cap_fn() yields the wall-clock cap and may raise
        when the construction budget is exhausted; it is only consulted for uncached work."""
        if self.passed:
            return 0.0
        faces = int(len(self.source.faces))
        if level["max_faces"] is not None and faces > level["max_faces"]:
            self.trials.append({"level": level["level"], "skipped": "source_faces_exceed_level_limit", "faces": faces})
            return 0.0
        if self.components is None:
            self.components = self.source.split(only_watertight=False)
        if level["split_components"] and len(self.components) < 2:
            self.trials.append({"level": level["level"], "skipped": "single_component"})
            return 0.0
        cached = load_level(cache, self.key, level["level"])
        if cached is not None:
            parts, record = cached
        else:
            parts, record = run_level(self.source, self.components, level, cap_fn())
            if not record.get("timeout"):
                # Timeouts depend on host load and are re-tried on resume; audited outcomes are stable.
                names = [f"hull_{i:03}.ply" for i in range(len(parts))]
                publish_cache_dir(
                    cache,
                    f"{self.key}/level_{level['level']}",
                    [(n, (lambda p, h=h: h.export(p))) for n, h in zip(names, parts)],
                    {**record, "hulls": names, "cache_key": self.key, "cascade": CASCADE_VERSION},
                    "result.json",
                )
        self.trials.append({k: v for k, v in record.items() if k not in ("hulls", "cache_key", "cascade")})
        if record["valid"]:
            closure = record.get("added_occupied_fraction", record.get("fraction", math.inf))
            if record["pass"] or closure < self.closure:
                self.parts = parts
                self.closure = closure
                self.method = f"source_collision_convex_decomposition;{CASCADE_VERSION};level_{level['level']}"
            if record["pass"]:
                self.passed = True
        return 0.0 if record.get("from_cache") else record["seconds"]

    def diagnostic(self):
        return {
            "cache_key": self.key,
            "method": self.method,
            "audit_pass": self.passed,
            "closure": None if not math.isfinite(self.closure) else self.closure,
            "hulls": len(self.parts),
            "trials": self.trials,
        }


def decompose_link(link_collision, piece_meshes, cache, deadline=None):
    """Cheapest first: code-004 level on every piece; accept if all pieces pass their own audit,
    or if the assembled hulls keep the whole link's accessible space open. Only then escalate
    failing pieces level by level, re-checking the link after each level. Returns (hulls, diagnostic)."""
    pieces = [Piece(m) for m in piece_meshes if m is not None]
    if not pieces:
        raise ValueError("collision_decomposition_invalid")
    link_source = clean_source(link_collision)

    def remaining():
        return None if deadline is None else deadline - time.perf_counter()

    def cap_for(level):
        r = remaining()
        if r is None:
            return level["cap_seconds"]
        if r < MIN_LEVEL_SECONDS:
            raise ValueError("decomposition_cascade_budget_exhausted")
        return min(level["cap_seconds"], r)

    def link_audit():
        if all(pc.passed for pc in pieces):
            return {"valid": True, "pass": True, "check": "all_pieces_passed"}
        if any(not pc.parts for pc in pieces):
            return {"valid": False, "pass": False, "reason": "collision_decomposition_invalid"}
        return audit_parts(link_source, [h for pc in pieces for h in pc.parts], mode="rays")

    for level in DECOMPOSITION_CASCADE:
        for pc in pieces:
            if not pc.passed:
                pc.step(level, cache, lambda level=level: cap_for(level))
        audit = link_audit()
        escalated = level["level"]
        if audit["pass"]:
            break
    else:
        raise ValueError(audit.get("reason") or "collision_decomposition_invalid")
    hulls = [h for pc in pieces for h in pc.parts]
    return hulls, {
        "method": "preserved_source_collision_components;" + CASCADE_VERSION,
        "source_components": len(pieces),
        "hulls": len(hulls),
        "highest_level_used": escalated,
        "pieces": [pc.diagnostic() for pc in pieces],
        "link_audit": {k: v for k, v in audit.items() if k != "valid"},
    }


def convex_parts(mesh, cache, deadline=None):
    """Single-piece convenience wrapper kept for tools/tests; the builder uses decompose_link."""
    hulls, d = decompose_link(mesh, [mesh], cache, deadline)
    return hulls, d


def same_geometry(a, b):
    return (
        a is not None
        and b is not None
        and np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
        and np.array_equal(np.asarray(a.faces), np.asarray(b.faces))
    )


def source_installation_evidence(project, candidate, parser_module):
    """Dataset-declared installation: the PartNet-Mobility semantics file tags the URDF root
    link 'static'. This is source metadata, never a model observation or category prior.
    Returns a verified evidence record (with artifact hash) or None."""
    urdf_ref = candidate.get("urdf")
    if not urdf_ref and candidate.get("structural", {}).get("joint_count", 0) > 0:
        urdf_ref = candidate.get("source", {}).get("urdf")
    semantics = candidate.get("semantics") or {}
    if not urdf_ref or not semantics.get("path") or len(semantics.get("sha256", "")) != 64:
        return None
    urdf = resolve(urdf_ref["path"], project)
    sem = resolve(semantics["path"], project)
    if not urdf.is_file() or not sem.is_file() or file_sha(sem) != semantics["sha256"]:
        return None
    if urdf_ref.get("sha256") and file_sha(urdf) != urdf_ref["sha256"]:
        return None
    source_dir = Path(candidate["source_dir"]) if candidate.get("source_dir") else urdf.parent
    try:
        specs, joints, _ = parser_module.parse_urdf(urdf, source_dir)
    except Exception:
        return None
    roots = set(specs) - {j.child for j in joints}
    if len(roots) != 1:
        return None
    structural = next(iter(roots))
    # The empty structural root is attached to exactly one child by a fixed joint.
    children = [j.child for j in joints if j.parent == structural and j.joint_type == "fixed"]
    if specs[structural].visuals or len(children) != 1:
        return None
    tags = {}
    for line in sem.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            tags[parts[0]] = (parts[1], parts[2])
    tag = tags.get(children[0])
    if not tag or tag[0] != "static":
        return None
    return {
        "kind": "source_installation_metadata",
        "assertion": "fixed_installation_required",
        "verified": True,
        "artifact": {"path": str(sem), "sha256": semantics["sha256"]},
        "root_link": children[0],
        "semantic_motion": tag[0],
        "semantic_part": tag[1],
        "rule": "partnet_mobility_root_link_declared_static",
    }


def emit_mesh(stage, path, mesh, collision=False):
    from pxr import UsdGeom, UsdPhysics, Gf

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh.vertices])
    m.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    m.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces, dtype=np.int32).ravel().tolist())
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr([Gf.Vec3f(*map(float, mesh.bounds[0])), Gf.Vec3f(*map(float, mesh.bounds[1]))])
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        m.CreateVisibilityAttr("invisible")
    return m


def construct_asset(project, candidate, grounding, task_metadata, out, cache, repair_mode=None, budget_seconds=None):
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = None if budget_seconds is None else started + float(budget_seconds)
    h, parser_ref = parser(project)
    support = SupportContract.from_task_metadata(task_metadata)
    refs = []

    def checked(ref):
        p = resolve(ref["path"], project)
        digest = file_sha(p)
        if ref.get("sha256") and digest != ref["sha256"]:
            raise RuntimeError("immutable_source_changed")
        refs.append({"path": str(p), "sha256": digest})
        return p

    visuals = {}
    collisions = {}
    source_parts = {}
    specs = {}
    joints = ()
    root = "body"
    source_frame = "Z"
    urdf_ref = candidate.get("urdf")
    if not urdf_ref and candidate.get("structural", {}).get("joint_count", 0) > 0:
        urdf_ref = candidate.get("source", {}).get("urdf")
    if urdf_ref:
        urdf = checked(urdf_ref)
        source_dir = Path(candidate["source_dir"]) if candidate.get("source_dir") else urdf.parent
        if candidate.get("source", {}).get("units") == "m_as_declared_by_source_urdf":
            unit = {"source": "registered_source_URDF_convention", "meters_per_unit_before": 1.0, "up_axis_before": "Z"}
        else:
            unit = h._dataset_unit_evidence(source_dir)
        specs, joints, _ = h.parse_urdf(urdf, source_dir)
        roots = set(specs) - {j.child for j in joints}
        if len(roots) != 1:
            raise ValueError("ambiguous_root")
        root = next(iter(roots))
        for n, spec in specs.items():
            visual = h._combine(spec.visuals, np, trimesh)
            collision = h._combine(spec.collisions, np, trimesh) if spec.collisions else visual
            visuals[n] = visual
            collisions[n] = collision
            source_parts[n] = [h._combine((element,), np, trimesh) for element in (spec.collisions or spec.visuals)]
        audit = h._source_files(source_dir, urdf)
    else:
        source = candidate.get("source", {})
        vp = checked(source["visual"])
        cp = checked(source["collision"])
        visual = trimesh.load(vp, force="mesh", process=False)
        collision = trimesh.load(cp, force="mesh", process=False)
        archive = source.get("archive", {}).get("path", "")
        meta = candidate.get("unit_metadata") or source.get("unit_metadata") or {}
        source_frame = str(meta.get("up_axis", "")).upper()
        if not source_frame and source.get("units") == "m_as_declared_by_source_urdf":
            source_frame = "Z"
        if not source_frame and str(archive).lower().endswith(".glb"):
            source_frame = "Y"
        if source_frame not in ("Y", "Z"):
            raise ValueError("source_up_axis_unverified")
        rigid_collision_parts = []
        for ref in source.get("collision_source", {}).get("sources", []):
            rigid_collision_parts.append(trimesh.load(checked(ref), force="mesh", process=False))
        if not rigid_collision_parts:
            rigid_collision_parts = [collision]
        if source_frame == "Y":
            tf = np.eye(4)
            tf[:3, :3] = rotation_matrix((math.pi / 2, 0, 0))
            visual.apply_transform(tf)
            collision.apply_transform(tf)
            for piece in rigid_collision_parts:
                if piece is not collision:
                    piece.apply_transform(tf)
        visuals[root] = visual
        collisions[root] = collision
        source_parts[root] = rigid_collision_parts
        specs[root] = h.LinkSpec(root, (), (), None, None, None)
        audit = refs
        unit = {"source_up_axis": source_frame, "format_units": "declared_glTF_or_catalog_units"}
    # Uniform query-scale hypothesis is chosen before physics, never best-of-scales.
    positions = forward_kinematics(specs, joints, root)
    cloud = np.concatenate(
        [
            np.asarray(v.vertices) @ positions[n][:3, :3].T + positions[n][:3, 3]
            for n, v in visuals.items()
            if v is not None
        ]
    )
    extent = np.ptp(cloud, axis=0)
    scale = 1.0
    size = grounding.get("size_m")
    confidence = grounding.get("size_confidence", 0)
    if (
        isinstance(size, list)
        and len(size) == 3
        and isinstance(confidence, (int, float))
        and confidence >= 0.65
        and all(isinstance(v, (int, float)) and math.isfinite(v) and 0.01 <= v <= 20 for v in size)
    ):
        scale = max(size) / float(max(extent))
    if not math.isfinite(scale) or not 1e-9 <= scale <= 1e9:
        raise ValueError("scale_hypothesis_invalid")
    scaled_joints = tuple(
        replace(
            j,
            origin_xyz=tuple(scale * x for x in j.origin_xyz),
            lower=j.lower * scale if j.joint_type == "prismatic" and j.lower is not None else j.lower,
            upper=j.upper * scale if j.joint_type == "prismatic" and j.upper is not None else j.upper,
        )
        for j in joints
    )
    initial_positions = {}
    for j in scaled_joints:
        if j.joint_type != "fixed":
            initial_positions[j.name] = (
                min(max(0.0, j.lower), j.upper) if j.lower is not None and j.upper is not None else 0.0
            )
    transforms = forward_kinematics(specs, scaled_joints, root, initial_positions)
    builds = {}
    decompositions = {}
    physics = {}
    geometry_audit = {
        "visual_equals_collision": all(
            same_geometry(visuals[n], collisions[n]) for n in specs if visuals[n] is not None
        ),
        "links": len(specs),
        "repair_mode": repair_mode,
    }
    (out / "geometry_audit.json").write_text(json.dumps(geometry_audit, indent=2), encoding="utf-8")
    if repair_mode == "rebuild_collision_from_source_visual" and geometry_audit["visual_equals_collision"]:
        # The supplied collision already is the visual surface: rebuilding from it cannot change the outcome.
        raise ValueError("repair_no_alternative_source_geometry")
    for name, spec in specs.items():
        v = visuals[name]
        c = collisions[name]
        if v is None:
            if name != root:
                raise ValueError("missing_nonroot_geometry")
            builds[name] = h.LinkBuild(spec, None, None, None, None, None)
            continue
        pieces = source_parts.get(name) or [c]
        if repair_mode == "rebuild_collision_from_source_visual":
            c = v.copy()
            pieces = [c]
        # Source-space decomposition is independent of a query's uniform scale.
        hulls, diagnostic = decompose_link(c, pieces, cache, deadline)
        source_check = v.copy()
        source_check.merge_vertices(digits_vertex=12)
        if source_check.is_watertight and not source_check.is_convex:
            lo_v, hi_v = source_check.bounds
            grid = np.meshgrid(
                *[np.linspace(lo_v[i], hi_v[i], 18, endpoint=False) + (hi_v[i] - lo_v[i]) / 36 for i in range(3)],
                indexing="ij",
            )
            pts = np.stack([a.ravel() for a in grid], axis=1)
            free = ~source_check.contains(pts)
            occupied = np.zeros(len(pts), dtype=bool)
            for part in hulls:
                occupied |= part.contains(pts)
            excess = float(np.mean(free & occupied))
            if excess > 0.02 and task_metadata.get("requires_interior_access") is True:
                raise ValueError("collision_closes_required_visual_free_space")
            diagnostic = {**diagnostic, "visual_free_space_added_occupancy": excess}
        visual_clearance = accessible_space_audit(v, hulls)
        if visual_clearance["fraction"] > 0.02 and task_metadata.get("requires_interior_access") is True:
            raise ValueError("collision_closes_required_visual_accessible_space")
        # Existing supplied collision-vs-appearance differences are diagnostic unless
        # a task explicitly requires that space. Newly introduced closure relative to
        # the supplied collision source is always rejected in convex_parts().
        diagnostic = {**diagnostic, "visual_surface_clearance": visual_clearance}
        hulls = [part.copy() for part in hulls]
        for part in hulls:
            part.apply_scale(scale)
        v = v.copy()
        v.apply_scale(scale)
        c = c.copy()
        c.apply_scale(scale)
        mp = mass_properties(hulls)
        physics[name] = mp
        decompositions[name] = {"parts": hulls, "diagnostic": diagnostic}
        builds[name] = h.LinkBuild(spec, v, c, trimesh.util.concatenate(hulls), mp, "source_convex_decomposition")
    asset_id = "Asset"
    usd = out / "asset.usda"
    usd.write_text(h.author_usda(asset_id, builds, scaled_joints, root, 500.0, np), encoding="utf-8")
    stage = Usd.Stage.Open(str(usd))
    asset_root = stage.GetPrimAtPath("/World/Asset")
    if not scaled_joints:
        asset_root.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))
    else:
        asset_root.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    cloud = np.concatenate(
        [
            np.asarray(b.visual.vertices) @ transforms[n][:3, :3].T + transforms[n][:3, 3]
            for n, b in builds.items()
            if b.visual is not None
        ]
    )
    lo, hi = cloud.min(0), cloud.max(0)
    collision_cloud = np.concatenate(
        [
            np.asarray(part.vertices) @ transforms[n][:3, :3].T + transforms[n][:3, 3]
            for n, d in decompositions.items()
            for part in d["parts"]
        ]
    )
    collision_low = float(collision_cloud[:, 2].min())
    offset = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.002 - min(lo[2], collision_low)])
    body_paths = {n: "/World/Asset/Bodies/" + h._safe(n) for n in builds}
    physical = {n for n, b in builds.items() if b.visual is not None}
    for name, b in builds.items():
        prim = stage.GetPrimAtPath(body_paths[name])
        tf = transforms[name].copy()
        tf[:3, 3] += offset
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(tf.T.tolist()))
        if b.visual is None:
            prim.CreateAttribute("affordcraft:emptyStaticFrame", Sdf.ValueTypeNames.Bool).Set(True)
            if support.role == "free":
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            continue
        stage.RemovePrim(body_paths[name] + "/collision_000")
        for i, part in enumerate(decompositions[name]["parts"]):
            emit_mesh(stage, body_paths[name] + f"/collision_{i:03}", part, True)
    expected_internal = 0
    axis_records = {}
    removed_external = []
    for j in scaled_joints:
        path = "/World/Asset/Joints/" + h._safe(j.name)
        prim = stage.GetPrimAtPath(path)
        api = UsdPhysics.Joint(prim)
        is_boundary = j.parent not in physical
        if is_boundary:
            if j.joint_type != "fixed":
                raise ValueError("unresolved_nonfixed_environment_boundary")
            if support.role == "free":
                stage.RemovePrim(path)
                removed_external.append(j.name)
                continue
            tf = transforms[j.child].copy()
            tf[:3, 3] += offset
            api.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, tf[:3, 3])))
            q = h._rpy_quaternion(j.origin_rpy)
            api.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        else:
            expected_internal += 1
        if j.axis is not None:
            # Both joint-frame quaternions map X to the source axis: the USD token must be X.
            prim.GetAttribute("physics:axis").Set("X")
            axis = transforms[j.parent][:3, :3] @ rotation_matrix(j.origin_rpy) @ np.asarray(j.axis)
            axis /= np.linalg.norm(axis)
            axis_records[path] = axis.tolist()
        if j.joint_type == "continuous":
            prim.CreateAttribute("affordcraft:continuous", Sdf.ValueTypeNames.Bool).Set(True)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.02))
    floor.AddScaleOp().Set(Gf.Vec3f(max(5, float(hi[0] - lo[0]) * 3), max(5, float(hi[1] - lo[1]) * 3), 0.04))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor.GetPrim().CreateAttribute("affordcraft:supportSurface", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    receipt = {
        "candidate_id": str(candidate["candidate_id"]),
        "exported": True,
        "usd": str(usd),
        "sha256": file_sha(usd),
        "support_role": support.role,
        "support_reason": support.reason,
        "task_metadata": task_metadata,
        "source_refs": refs,
        "source_mesh_audit": audit,
        "parser_ref": parser_ref,
        "diagnostics": geometry_audit,
        "decomposition_cascade": CASCADE_VERSION,
        "uniform_scale": scale,
        "source_unit_evidence": unit,
        "initial_translation_m": offset.tolist(),
        "initial_joint_positions": initial_positions,
        "initial_state_policy": "nearest_source_declared_legal_configuration_to_URDF_zero;not_selected_by_physics_outcomes",
        "geometry_source_verified": True,
        "expected_internal_joints": expected_internal,
        "requires_articulation": any(j.joint_type != "fixed" for j in scaled_joints),
        "expected_joint_axes_world": axis_records,
        "removed_external_boundary_joints": removed_external,
        "support_surface_z": 0.0,
        "repair_mode": repair_mode,
        "visual_geometry_changed_beyond_uniform_transform": False,
        "collision_diagnostics": {n: d["diagnostic"] for n, d in decompositions.items()},
        "physics_parameters": physics,
        "seconds": time.perf_counter() - started,
    }
    (out / "asset_manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
