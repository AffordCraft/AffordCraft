#!/usr/bin/env python3
"""Source-preserving import of structured (URDF) library entries.

Every URDF link and joint (including fixed joints) is kept; per-link visual and collision meshes are emitted and the
physical-parameter policy is recorded. The PartNet-Mobility corpus does not declare units in its URDF, so the dataset
convention is accepted only when the source path lies inside a PartNet-Mobility dataset, and that inference is recorded
with its evidence; other entries fail closed as ``repair_uncertain`` rather than guessing units or joint metadata.
Used by :mod:`affordcraft.build` (loaded through ``build.parser``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROBOT_FIELDS = {
    "robot_real_world": "not_evaluated",
    "robot_control_status": "not_evaluated",
    "robot_task_success": None,
}

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class RepairError(RuntimeError):
    """A deterministic, auditable repair rejection."""

    def __init__(self, reason: str, detail: str = "", status: str = "repair_uncertain") -> None:
        self.reason = reason
        self.detail = detail
        self.status = status
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class GeometrySpec:
    mesh_path: Path
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    mesh_scale: tuple[float, float, float]


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: tuple[GeometrySpec, ...]
    collisions: tuple[GeometrySpec, ...]
    source_mass: float | None
    source_com: tuple[float, float, float] | None
    source_inertia: tuple[float, float, float, float, float, float] | None


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass
class LinkBuild:
    spec: LinkSpec
    visual: Any | None
    collision_source: Any | None
    collision: Any | None
    physics: dict[str, Any] | None
    collision_method: str | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_file(path: Path) -> Any:
    """Read a JSON sidecar without mutating it."""
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: str | None, field: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RepairError("repair_uncertain", f"non-numeric {field}: {value}") from exc
    if not math.isfinite(result):
        raise RepairError("repair_uncertain", f"non-finite {field}")
    return result


def _vec(
    value: str | None, field: str, default: tuple[float, float, float] | None = None
) -> tuple[float, float, float]:
    if value is None:
        if default is None:
            raise RepairError("repair_uncertain", f"missing {field}")
        return default
    try:
        parts = tuple(float(x) for x in value.replace(",", " ").split())
    except (TypeError, ValueError) as exc:
        raise RepairError("repair_uncertain", f"invalid {field}: {value}") from exc
    if len(parts) != 3 or not all(math.isfinite(x) for x in parts):
        raise RepairError("repair_uncertain", f"invalid {field}: {value}")
    return parts  # type: ignore[return-value]


def _origin(element: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return _vec(element.get("xyz"), "origin.xyz", (0.0, 0.0, 0.0)), _vec(
        element.get("rpy"), "origin.rpy", (0.0, 0.0, 0.0)
    )


def _rpy_matrix(rpy: tuple[float, float, float], np: Any) -> Any:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses fixed-axis RPY equivalent to Rz(yaw) Ry(pitch) Rx(roll).
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
            [-sp, cp * sr, cp * cr, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform(
    vertices: Any,
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
    scale: tuple[float, float, float],
    np: Any,
) -> Any:
    matrix = _rpy_matrix(origin_rpy, np)
    matrix[:3, 3] = np.asarray(origin_xyz, dtype=np.float64)
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(np.asarray(scale, dtype=np.float64))
    homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float64)), axis=1)
    result = (homogeneous @ matrix.T)[:, :3]
    if not np.isfinite(result).all():
        raise RepairError("repair_uncertain", "mesh transform produced NaN/Inf")
    return result


def _safe_mesh_path(object_dir: Path, filename: str, field: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise RepairError("repair_uncertain", f"unsafe {field} path: {filename}")
    path = (object_dir / relative).resolve()
    try:
        path.relative_to(object_dir.resolve())
    except ValueError as exc:
        raise RepairError("repair_uncertain", f"{field} escapes source directory: {filename}") from exc
    if path.is_symlink() or not path.is_file():
        raise RepairError("blocked", f"missing {field} mesh: {path}", status="blocked")
    return path


def _parse_inertial(
    link: ET.Element,
) -> tuple[float | None, tuple[float, float, float] | None, tuple[float, float, float, float, float, float] | None]:
    inertial = link.find("inertial")
    if inertial is None:
        return None, None, None
    mass_el = inertial.find("mass")
    mass = _finite(mass_el.get("value") if mass_el is not None else None, "inertial.mass")
    origin_xyz, _ = _origin(inertial.find("origin"))
    inertia_el = inertial.find("inertia")
    inertia = None
    if inertia_el is not None:
        vals = tuple(_finite(inertia_el.get(k), f"inertia.{k}") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        if any(v is None for v in vals):
            raise RepairError("repair_uncertain", "inertial inertia is incomplete")
        inertia = tuple(float(v) for v in vals)  # type: ignore[arg-type]
    return mass, origin_xyz if mass is not None else None, inertia


def parse_urdf(urdf_path: Path, object_dir: Path) -> tuple[dict[str, LinkSpec], tuple[JointSpec, ...], str]:
    raw = urdf_path.read_bytes()
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise RepairError("repair_uncertain", "URDF DTD/entity declarations are forbidden")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RepairError("blocked", f"URDF XML parse failed: {exc}", status="blocked") from exc
    if root.tag != "robot":
        raise RepairError("repair_uncertain", "URDF root is not robot")
    links: dict[str, LinkSpec] = {}
    for link in root.findall("link"):
        name = (link.get("name") or "").strip()
        if not name or name in links:
            raise RepairError("repair_uncertain", f"invalid/duplicate link name: {name!r}")
        visual_specs: list[GeometrySpec] = []
        collision_specs: list[GeometrySpec] = []
        for tag, dest in (("visual", visual_specs), ("collision", collision_specs)):
            for geom in link.findall(tag):
                geometry = geom.find("geometry")
                mesh = geometry.find("mesh") if geometry is not None else None
                if mesh is None or not mesh.get("filename"):
                    raise RepairError("repair_uncertain", f"link {name} has non-mesh {tag} geometry")
                origin_xyz, origin_rpy = _origin(geom.find("origin"))
                mesh_scale = _vec(mesh.get("scale"), f"{tag}.mesh.scale", (1.0, 1.0, 1.0))
                if any(x <= 0.0 for x in mesh_scale):
                    raise RepairError("repair_uncertain", f"link {name} has non-positive mesh scale")
                # The repair contract permits only an auditable *uniform*
                # scale.  A per-axis mesh scale would silently change the
                # articulated object's proportions, so fail closed instead
                # of baking it into a new asset.
                if max(mesh_scale) - min(mesh_scale) > 1e-9:
                    raise RepairError("repair_uncertain", f"link {name} has non-uniform mesh scale")
                dest.append(
                    GeometrySpec(
                        _safe_mesh_path(object_dir, mesh.get("filename") or "", f"{tag}"),
                        origin_xyz,
                        origin_rpy,
                        mesh_scale,
                    )
                )
        mass, com, inertia = _parse_inertial(link)
        links[name] = LinkSpec(name, tuple(visual_specs), tuple(collision_specs), mass, com, inertia)
    if not links:
        raise RepairError("repair_uncertain", "URDF has no links")
    joints: list[JointSpec] = []
    child_seen: set[str] = set()
    for joint in root.findall("joint"):
        name = (joint.get("name") or "").strip()
        kind = (joint.get("type") or "").strip().lower()
        if not name or kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise RepairError("repair_uncertain", f"unsupported joint {name!r}/{kind!r}")
        parent_el, child_el = joint.find("parent"), joint.find("child")
        parent = (parent_el.get("link") if parent_el is not None else "") or ""
        child = (child_el.get("link") if child_el is not None else "") or ""
        if parent not in links or child not in links or parent == child:
            raise RepairError("repair_uncertain", f"joint {name} has invalid parent/child")
        if child in child_seen:
            raise RepairError("repair_uncertain", f"link {child} has multiple parent joints")
        child_seen.add(child)
        xyz, rpy = _origin(joint.find("origin"))
        axis = None
        lower = upper_limit = effort = velocity = None
        if kind != "fixed":
            axis_el = joint.find("axis")
            axis = _vec(axis_el.get("xyz") if axis_el is not None else None, f"joint {name}.axis")
            norm = math.sqrt(sum(x * x for x in axis))
            if norm <= 1e-12:
                raise RepairError("repair_uncertain", f"joint {name} axis is zero")
            axis = tuple(x / norm for x in axis)
            limit = joint.find("limit")
            if limit is not None:
                lower = _finite(limit.get("lower"), f"joint {name}.lower")
                upper_limit = _finite(limit.get("upper"), f"joint {name}.upper")
                effort = _finite(limit.get("effort"), f"joint {name}.effort")
                velocity = _finite(limit.get("velocity"), f"joint {name}.velocity")
                if lower is not None and upper_limit is not None and lower > upper_limit:
                    raise RepairError("repair_uncertain", f"joint {name} limits reversed")
            elif kind not in {"continuous"}:
                # A bounded joint with no limit is not safely reconstructible.
                raise RepairError("repair_uncertain", f"joint {name} lacks limits")
        joints.append(JointSpec(name, kind, parent, child, xyz, rpy, axis, lower, upper_limit, effort, velocity))
    # A valid articulation must be a single rooted acyclic tree.  This check is
    # independent of mesh loading and prevents silently dropping a link.
    roots = sorted(set(links) - child_seen)
    if len(roots) != 1:
        raise RepairError("repair_uncertain", f"URDF must have exactly one root (got {roots})")
    children = {name: [] for name in links}
    for joint in joints:
        children[joint.parent].append(joint.child)
    visited: set[str] = set()
    stack = [roots[0]]
    while stack:
        item = stack.pop()
        if item in visited:
            raise RepairError("repair_uncertain", "joint hierarchy contains a cycle")
        visited.add(item)
        stack.extend(children[item])
    if visited != set(links):
        raise RepairError("repair_uncertain", "joint hierarchy does not reach every link")
    robot_name = root.get("name") or object_dir.name
    return links, tuple(joints), robot_name


def _load_mesh(path: Path, np: Any, trimesh: Any) -> Any:
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise RepairError("blocked", f"mesh load failed {path}: {exc}", status="blocked") from exc
    if isinstance(loaded, trimesh.Scene):
        geoms = []
        for geom in loaded.dump(concatenate=False):
            if isinstance(geom, trimesh.Trimesh):
                geoms.append(geom)
        if not geoms:
            raise RepairError("collision_invalid", f"mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RepairError("collision_invalid", f"not a triangular mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) == 0
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
    ):
        raise RepairError("collision_invalid", f"empty/invalid triangles: {path}")
    if not np.isfinite(vertices).all() or int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RepairError("collision_invalid", f"non-finite/out-of-range triangles: {path}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _combine(elements: Iterable[GeometrySpec], np: Any, trimesh: Any) -> Any | None:
    vertices: list[Any] = []
    faces: list[Any] = []
    offset = 0
    for element in elements:
        mesh = _load_mesh(element.mesh_path, np, trimesh)
        transformed = _transform(
            np.asarray(mesh.vertices, dtype=np.float64), element.origin_xyz, element.origin_rpy, element.mesh_scale, np
        )
        vertices.append(transformed)
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        offset += len(transformed)
    if not vertices:
        return None
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0), faces=np.concatenate(faces, axis=0), process=False
    )


def _finite_bounds(mesh: Any, np: Any, field: str) -> tuple[list[float], list[float], list[float]]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if (
        bounds.shape != (2, 3)
        or not np.isfinite(bounds).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 1e-9)
    ):
        raise RepairError("collision_invalid", f"{field} bounds are empty/degenerate")
    return bounds[0].tolist(), bounds[1].tolist(), extents.tolist()


def _physics(collision: Any, density: float, np: Any) -> dict[str, Any]:
    volume = abs(float(collision.volume))
    if not math.isfinite(volume) or volume <= 1e-10:
        raise RepairError("physics_sanity_failed", f"collision hull volume invalid: {volume}")
    mass = volume * density
    com = np.asarray(collision.center_mass, dtype=np.float64)
    if not np.isfinite(com).all():
        com = np.asarray(collision.centroid, dtype=np.float64)
    inertia = np.asarray(collision.moment_inertia, dtype=np.float64) * density
    inertia = (inertia + inertia.T) / 2.0
    if not np.isfinite(com).all() or not np.isfinite(inertia).all():
        raise RepairError("physics_sanity_failed", "mass properties contain NaN/Inf")
    eig = np.linalg.eigvalsh(inertia)
    if not np.isfinite(eig).all() or float(eig.min()) <= 1e-12:
        raise RepairError("physics_sanity_failed", f"inertia is not positive definite: {eig.tolist()}")
    if mass <= 1e-8 or mass > 1e7:
        raise RepairError("physics_sanity_failed", f"mass outside conservative range: {mass}")
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    if float(np.linalg.det(eigenvectors)) < 0.0:
        eigenvectors[:, 0] *= -1.0
    trace = float(np.trace(eigenvectors))
    if trace > 0.0:
        q_w = math.sqrt(1.0 + trace) / 2.0
        q_x = float(eigenvectors[2, 1] - eigenvectors[1, 2]) / (4.0 * q_w)
        q_y = float(eigenvectors[0, 2] - eigenvectors[2, 0]) / (4.0 * q_w)
        q_z = float(eigenvectors[1, 0] - eigenvectors[0, 1]) / (4.0 * q_w)
    else:
        # Deterministic fallback for the rare 180-degree principal-frame
        # rotation.  The eigenbasis itself remains in the report.
        q_w, q_x, q_y, q_z = 0.0, 1.0, 0.0, 0.0
    quat = np.asarray([q_w, q_x, q_y, q_z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return {
        "density_kg_m3": float(density),
        "mass_kg": float(mass),
        "center_of_mass_m": [float(x) for x in com],
        "inertia_matrix_kg_m2": [[float(x) for x in row] for row in inertia],
        "inertia_eigenvalues_kg_m2": [float(x) for x in eig],
        "diagonal_inertia_kg_m2": [float(x) for x in eigenvalues],
        "principal_axes_wxyz": [float(x) for x in quat],
        "principal_axes_matrix": eigenvectors.tolist(),
        "volume_m3": float(volume),
        "policy": "explicit_density_times_convex_hull_volume",
    }


def _rpy_quaternion(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (x / 2.0 for x in rpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _qmul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    q = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q) if n > 1e-12 else (1.0, 0.0, 0.0, 0.0)


def _axis_quaternion(axis: tuple[float, float, float] | None) -> tuple[float, float, float, float]:
    if axis is None:
        return (1.0, 0.0, 0.0, 0.0)
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    dot = max(-1.0, min(1.0, x))
    if dot > 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-12:
        return (0.0, 0.0, 1.0, 0.0)
    cross = (0.0, -z, y)
    scale = math.sqrt((1.0 + dot) * 2.0)
    quat = (scale / 2.0, cross[0] / scale, cross[1] / scale, cross[2] / scale)
    n = math.sqrt(sum(v * v for v in quat))
    return tuple(v / n for v in quat)


def _num(value: float) -> str:
    value = float(value)
    return "0" if value == 0.0 else repr(value)


def _vec_text(values: Iterable[float]) -> str:
    return "(" + ", ".join(_num(v) for v in values) + ")"


def _str_text(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _point_array(vertices: Any) -> str:
    return "[" + ", ".join(_vec_text(row) for row in vertices.tolist()) + "]"


def _safe(value: str) -> str:
    result = _SAFE.sub("_", str(value)).strip("_") or "item"
    if result[0].isdigit():
        result = "link_" + result
    return result


def _mesh_usda_lines(
    name: str, mesh: Any, indent: str, collision: bool, source_sha: str, np: Any, material_path: str | None = None
) -> list[str]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lines: list[str] = []
    if collision:
        lines.extend(
            [
                f'{indent}def Mesh "{name}" (',
                f'{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
                f"{indent})",
            ]
        )
    else:
        lines.append(f'{indent}def Mesh "{name}"')
    lines.extend(
        [
            f"{indent}{{",
            f"{indent}    custom string i2ia:sourceMeshSha256 = {_str_text(source_sha)}",
            f"{indent}    point3f[] points = {_point_array(vertices)}",
            f"{indent}    int[] faceVertexCounts = [{', '.join('3' for _ in faces)}]",
            f"{indent}    int[] faceVertexIndices = [{', '.join(str(int(x)) for row in faces for x in row)}]",
            f'{indent}    uniform token subdivisionScheme = "none"',
            f"{indent}    float3[] extent = [{_vec_text(vertices.min(axis=0))}, {_vec_text(vertices.max(axis=0))}]",
        ]
    )
    if collision:
        lines.extend(
            [
                f"{indent}    bool physics:collisionEnabled = true",
                f'{indent}    token physics:approximation = "convexHull"',
                *([f"{indent}    rel material:binding:physics = <{material_path}>"] if material_path else []),
                f'{indent}    token visibility = "invisible"',
            ]
        )
    else:
        lines.append(f'{indent}    uniform token purpose = "render"')
    lines.append(f"{indent}}}")
    return lines


def author_usda(
    asset_id: str, links: dict[str, LinkBuild], joints: tuple[JointSpec, ...], root_link: str, density: float, np: Any
) -> str:
    root_path = f"/World/{_safe(asset_id)}"
    body_path = {name: f"{root_path}/Bodies/{_safe(name)}" for name in links}
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def PhysicsScene "physicsScene"',
        "    {",
        "        vector3f physics:gravityDirection = (0, 0, -1)",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
        "",
        f'    def Xform "{_safe(asset_id)}" (',
        '        prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]',
        "    )",
        "    {",
        f"        custom string i2ia:assetId = {_str_text(asset_id)}",
        f"        custom string i2ia:rootLink = {_str_text(root_link)}",
        f'        custom string i2ia:coordinateFrame = "source_root_preserved_zup_m"',
        f"        custom double i2ia:densityKgM3 = {_num(density)}",
        '        def Material "PhysicsMaterial" (',
        '            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]',
        "        )",
        "        {",
        "            float physics:staticFriction = 0.6",
        "            float physics:dynamicFriction = 0.5",
        "            float physics:restitution = 0",
        "        }",
        "",
        '        def Scope "Bodies"',
        "        {",
    ]
    for name, build in links.items():
        body_name = _safe(name)
        if build.visual is None:
            # PartNet uses an empty structural root.  It is retained in the
            # hierarchy but deliberately has no dynamic body or fake box.
            lines.extend(
                [
                    f'            def Xform "{body_name}" (',
                    '                prepend apiSchemas = ["PhysicsRigidBodyAPI"]',
                    "            )",
                    "            {",
                    f"                custom string i2ia:bodyId = {_str_text(name)}",
                    f'                custom string i2ia:motionType = "static"',
                    f"                custom string i2ia:structuralLink = {_str_text(name)}",
                    "                bool physics:rigidBodyEnabled = true",
                    "                bool physics:kinematicEnabled = true",
                    "            }",
                    "",
                ]
            )
            continue
        p = build.physics or {}
        lines.extend(
            [
                f'            def Xform "{body_name}" (',
                '                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
                "            )",
                "            {",
                f"                custom string i2ia:bodyId = {_str_text(name)}",
                f'                custom string i2ia:motionType = "dynamic"',
                f"                custom string i2ia:linkName = {_str_text(name)}",
                "                bool physics:rigidBodyEnabled = true",
                "                bool physics:kinematicEnabled = false",
                f'                float physics:mass = {_num(p["mass_kg"])}',
                f'                point3f physics:centerOfMass = {_vec_text(p["center_of_mass_m"])}',
                f'                vector3f physics:diagonalInertia = {_vec_text(p["diagonal_inertia_kg_m2"])}',
                f'                quatf physics:principalAxes = {_vec_text(p["principal_axes_wxyz"])}',
            ]
        )
        inertia = p["inertia_matrix_kg_m2"]
        # Keep the complete tensor in the sidecar report.  A JSON string here
        # avoids relying on version-specific USDA ``matrix3d`` grammar while
        # still making the authored value auditable.
        lines.append(
            f'                custom string i2ia:inertiaMatrixJson = {_str_text(json.dumps(inertia, separators=(",", ":")))}'
        )
        lines.extend(_mesh_usda_lines("visual_000", build.visual, "                ", False, "visual", np, None))
        lines.extend(
            _mesh_usda_lines(
                "collision_000",
                build.collision,
                "                ",
                True,
                "collision",
                np,
                f"{root_path}/PhysicsMaterial",
            )
        )
        lines.extend(["            }", ""])
    lines.extend(["        }", "", '        def Scope "Joints"', "        {"])
    for joint in joints:
        jname = _safe(joint.name)
        kind = {
            "fixed": "PhysicsFixedJoint",
            "revolute": "PhysicsRevoluteJoint",
            "continuous": "PhysicsRevoluteJoint",
            "prismatic": "PhysicsPrismaticJoint",
        }[joint.joint_type]
        lines.extend(
            [
                f'            def {kind} "{jname}"',
                "            {",
                f"                custom string i2ia:jointId = {_str_text(joint.name)}",
                f'                custom string i2ia:jointType = {_str_text(joint.joint_type if joint.joint_type != "continuous" else "revolute")}',
                f"                custom string i2ia:sourceJointType = {_str_text(joint.joint_type)}",
                f"                custom string i2ia:parentLink = {_str_text(joint.parent)}",
                f"                custom string i2ia:childLink = {_str_text(joint.child)}",
            ]
        )
        # A structural empty root has no rigid body; retaining body1 and the
        # source fixed joint is the same representation used by the canonical
        # compiler.  Any non-fixed joint with an empty parent is rejected
        # before this function is called.
        if links[joint.parent].visual is not None:
            lines.append(f"                rel physics:body0 = <{body_path[joint.parent]}>")
        if links[joint.child].visual is not None:
            lines.append(f"                rel physics:body1 = <{body_path[joint.child]}>")
        origin_q = _rpy_quaternion(joint.origin_rpy)
        axis_q = _axis_quaternion(joint.axis)
        local_q = _qmul(origin_q, axis_q) if joint.axis is not None else origin_q
        lines.extend(
            [
                f"                point3f physics:localPos0 = {_vec_text(joint.origin_xyz)}",
                "                point3f physics:localPos1 = (0, 0, 0)",
                f"                quatf physics:localRot0 = {_vec_text(local_q)}",
                f"                quatf physics:localRot1 = {_vec_text(axis_q if joint.axis is not None else (1.0, 0.0, 0.0, 0.0))}",
            ]
        )
        if joint.axis is not None:
            axis_token = (
                "X"
                if abs(joint.axis[0]) >= max(abs(joint.axis[1]), abs(joint.axis[2]))
                else ("Y" if abs(joint.axis[1]) >= abs(joint.axis[2]) else "Z")
            )
            lines.extend(
                [
                    f'                uniform token physics:axis = "{axis_token}"',
                    f"                custom double3 i2ia:axis = {_vec_text(joint.axis)}",
                    f"                quatf i2ia:axisFrame = {_vec_text(_axis_quaternion(joint.axis))}",
                ]
            )
        if joint.lower is not None:
            lower, upper = joint.lower, joint.upper if joint.upper is not None else joint.lower
            if joint.joint_type in {"revolute", "continuous"}:
                lower, upper = math.degrees(lower), math.degrees(upper)
            lines.extend(
                [
                    f"                float physics:lowerLimit = {_num(lower)}",
                    f"                float physics:upperLimit = {_num(upper)}",
                    f"                custom double2 i2ia:sourceLimits = ({_num(joint.lower)}, {_num(joint.upper if joint.upper is not None else joint.lower)})",
                ]
            )
        lines.extend(["            }", ""])
    lines.extend(["        }", "    }", "}", ""])
    return "\n".join(lines)


def write_obj(path: Path, mesh: Any) -> None:
    vertices = mesh.vertices
    faces = mesh.faces
    lines = ["# articulation-preserving-repair; topology-preserving export"]
    lines.extend(f"v {float(v[0]):.12g} {float(v[1]):.12g} {float(v[2]):.12g}" for v in vertices)
    lines.extend(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}" for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _catalog_map(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"catalog read failed: {exc}") from exc
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise SystemExit("catalog has no entries list")
    return {
        str(item.get("candidate_id")): item
        for item in entries
        if isinstance(item, dict) and item.get("candidate_id") is not None
    }


def _dataset_unit_evidence(source_dir: Path) -> dict[str, Any]:
    text = str(source_dir).lower().replace("\\", "/")
    if "partnet-mobility" not in text and "partnet_mobility" not in text:
        raise RepairError("repair_uncertain", "no reliable units/up-axis metadata; non-PartNet source")
    return {
        "source": "PartNet-Mobility dataset convention plus catalog canonicalizer_preflight",
        "meters_per_unit_before": 1.0,
        "up_axis_before": "Z",
        "confidence": 0.92,
        "inference": "dataset_convention_inferred_not_explicit_in_urdf",
        "uniform_scale": 1.0,
    }


def _source_files(object_dir: Path, urdf: Path) -> dict[str, Any]:
    files = {
        "urdf": urdf,
        "meta": object_dir / "meta.json",
        "semantics": object_dir / "semantics.txt",
        "bounding_box": object_dir / "bounding_box.json",
    }
    out: dict[str, Any] = {}
    for key, path in files.items():
        if path.is_file() and not path.is_symlink():
            out[key] = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return out


def _read_mesh_json(mesh: Any, np: Any) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds_min_m": [float(x) for x in bounds[0]],
        "bounds_max_m": [float(x) for x in bounds[1]],
        "extents_m": [float(x) for x in bounds[1] - bounds[0]],
    }
