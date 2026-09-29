#!/usr/bin/env python3
"""Run one real-contact Exp3 trajectory episode in Isaac/PhysX.

The only post-play commands in this runner are Franka arm joint targets and
gripper joint targets.  The target asset is observed through PhysX; no target
velocity, pose, joint drive, teleport, fallback, or post-play transform write
is permitted.  The primary MP4 is composed from raw Isaac camera RGB frames.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "i2ia.exp3_complete_robot_contact_episode.v0.88"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(list(command), capture_output=True, text=True, timeout=20, check=False)
        return (result.stdout or result.stderr).strip() or f"exit_code={result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The accepted protocol for this validation is sim-ready-only.  A robot
    # attempt must be an explicit opt-in in a separately versioned run so a
    # direct invocation cannot silently turn an out-of-scope control metric
    # into an evaluated (or failed) result.
    # Kept only for backwards-compatible command parsing.  This runner always
    # constructs and controls the same Franka; false is rejected below.
    parser.add_argument("--attempt-robot", choices=("true", "false"), default="true")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--settle-steps", type=int, default=180)
    parser.add_argument("--control-steps", type=int, default=150)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--rigid-contact-standoff", type=float, default=0.025)
    parser.add_argument("--articulated-workcell-yaw-deg", type=float, default=0.0)
    parser.add_argument("--position-only-ik", choices=("true", "false"), default="true")
    parser.add_argument("--save-observations", choices=("true", "false"), default="true")
    parser.add_argument("--articulated-placement", choices=("combined_center", "actuated_child_center"), default="actuated_child_center")
    parser.add_argument(
        "--articulated-orientation",
        choices=("identity", "upright_x90", "upright_y90", "z90", "zminus90", "z180"),
        default="identity",
        help="fixed pre-play orientation for articulated assets; never changed after play",
    )
    parser.add_argument("--support-width", type=float, default=3.0)
    parser.add_argument("--support-depth", type=float, default=3.0)
    parser.add_argument(
        "--support-top-z",
        type=float,
        default=0.90,
        help=(
            "pre-registered workcell support top height in metres; the "
            "asset is placed on this support before world.play"
        ),
    )
    parser.add_argument(
        "--articulated-pivot-x",
        type=float,
        default=-0.55,
        help="pre-registered workcell x slot for the authored first joint pivot",
    )
    parser.add_argument(
        "--articulated-pivot-y",
        type=float,
        default=0.0,
        help="pre-registered workcell y slot for the authored first joint pivot",
    )
    parser.add_argument(
        "--articulated-initial-joint-rad",
        type=float,
        default=0.0,
        help=(
            "setup-time initial value for the first authored revolute joint; "
            "applied before world.play only and never commanded during an episode"
        ),
    )
    parser.add_argument(
        "--controller",
        choices=("local_robot_policy",),
        default="local_robot_policy",
        help="local neural policy, robot joint actions only; no teacher branch is admissible",
    )
    parser.add_argument("--policy-socket", required=True)
    parser.add_argument("--policy-identity", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("untrained_vla","trained_vla"), required=True)
    parser.add_argument(
        "--franka-usd-path",
        default=os.environ.get("EXP3_FRANKA_USD_PATH", ""),
        help="local hash-locked Franka USD; avoids a runtime Nucleus dependency",
    )
    parser.add_argument(
        "--franka-base-x",
        type=float,
        default=-0.66,
        help="pre-play Franka base x position in the fixed workcell",
    )
    parser.add_argument(
        "--franka-base-y",
        type=float,
        default=0.0,
        help="pre-play Franka base y position in the fixed workcell",
    )
    parser.add_argument(
        "--franka-base-z",
        type=float,
        default=0.922,
        help="pre-play Franka base z position in the fixed workcell",
    )
    parser.add_argument(
        "--controller-orientation-wxyz",
        default="",
        help=(
            "optional fixed end-effector orientation as four comma-separated "
            "wxyz values.  It is a registered controller parameter and is "
            "never changed after world.play; empty preserves the historical "
            "UP-099 orientation."
        ),
    )
    parser.add_argument(
        "--finger-contact-compensation",
        choices=("true", "false"),
        default="false",
        help=(
            "optionally close the kinematic loop on the measured Franka "
            "finger midpoint for articulated contact; this only changes "
            "arm IK targets and never writes an object transform"
        ),
    )
    parser.add_argument(
        "--finger-contact-side",
        choices=("midpoint", "left", "right"),
        default="midpoint",
        help="registered finger link used by articulated contact compensation",
    )
    parser.add_argument(
        "--articulated-self-collision-policy",
        choices=("disabled_filtered", "enabled"),
        default="disabled_filtered",
        help=(
            "registered PhysX articulated self-collision policy; "
            "disabled_filtered is the historical adjacent-link filter, "
            "enabled preserves authored self-collision for diagnostics"
        ),
    )
    parser.add_argument(
        "--articulated-contact-radius-fraction",
        type=float,
        default=0.25,
        help="registered fraction of the actuated-link centerline radius used for contact",
    )
    parser.add_argument(
        "--asset-uniform-scale",
        type=float,
        default=1.0,
        help=(
            "pre-play, uniform physical-unit scale applied in the anonymous "
            "session layer; values other than 1.0 must be registered by the "
            "collection configuration and are never changed after play"
        ),
    )
    parser.add_argument(
        "--articulated-joint-index",
        type=int,
        default=0,
        help=(
            "zero-based index among authored revolute/prismatic joints used "
            "for the fixed contact task; selected before play"
        ),
    )
    parser.add_argument(
        "--articulated-tangent-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help=(
            "registered sign for the revolute tangential path; auto uses the "
            "reach-derived sign and never reads a task outcome"
        ),
    )
    parser.add_argument(
        "--articulated-pull-distance",
        type=float,
        default=0.16,
        help="fixed end-effector tangent displacement in metres",
    )
    parser.add_argument(
        "--articulated-approach-distance",
        type=float,
        default=0.12,
        help="registered tangential pre-contact distance for articulated IK",
    )
    parser.add_argument(
        "--articulated-approach-mode",
        choices=("tangent", "radial"),
        default="tangent",
        help="registered pre-contact direction; contact/pull remain geometry-derived",
    )
    parser.add_argument(
        "--articulated-motion-path",
        choices=("linear", "hinge_arc"),
        default="linear",
        help=(
            "registered articulated end-effector path; hinge_arc follows a "
            "precomputed pivot/axis arc and still emits only Franka actions"
        ),
    )
    parser.add_argument(
        "--articulated-arc-angle-rad",
        type=float,
        default=1.20,
        help="registered positive revolute arc angle for hinge_arc in radians",
    )
    parser.add_argument(
        "--articulated-approach-z-floor-offset",
        type=float,
        default=0.08,
        help=(
            "registered minimum approach height above the support plane; "
            "this only changes the Franka waypoint"
        ),
    )
    parser.add_argument(
        "--articulated-approach-contact-z-clearance",
        type=float,
        default=0.0,
        help=(
            "registered minimum approach height above the geometry-derived "
            "contact point; this only changes the Franka waypoint"
        ),
    )
    parser.add_argument(
        "--articulated-contact-z-offset",
        type=float,
        default=0.0,
        help=(
            "registered vertical offset for the contact waypoint; the "
            "target asset is never transformed"
        ),
    )
    parser.add_argument(
        "--articulated-hold-z-offset",
        type=float,
        default=0.0,
        help=(
            "registered vertical offset for the post-pull hold waypoint; "
            "the target asset is never transformed"
        ),
    )
    parser.add_argument(
        "--rigid-push-distance",
        type=float,
        default=0.0,
        help="optional registered rigid push distance in metres; zero derives it from bounds",
    )
    parser.add_argument("--articulated-ee-target-offset-x", type=float, default=0.0)
    parser.add_argument("--articulated-ee-target-offset-y", type=float, default=0.0)
    parser.add_argument("--articulated-ee-target-offset-z", type=float, default=0.0)
    return parser.parse_args()


def _repo_path(raw: str | Path, root: Path) -> Path:
    value = Path(raw)
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def _position(omni: Any, prim: Any, np: Any) -> Any:
    value = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _mesh_bounds(
    stage: Any,
    root: Any,
    omni: Any,
    Gf: Any,
    UsdGeom: Any,
    np: Any,
    mesh_prefix: str = "visual_",
) -> tuple[Any, Any, list[str]]:
    """Return world bounds for named meshes below root (without guessing)."""

    prefix = str(root.GetPath()) + "/"
    points_world: list[Any] = []
    mesh_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix) or not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.GetName().startswith(mesh_prefix):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not points:
            continue
        matrix = omni.usd.get_world_transform_matrix(prim)
        from i2ia.tasks.usd_affine_points import transform_points
        points_world.append(transform_points(points, matrix))
        mesh_paths.append(path)
    if not points_world:
        raise RuntimeError(f"no non-empty visual meshes below {root.GetPath()}")
    cloud = np.vstack(points_world)
    return cloud.min(axis=0), cloud.max(axis=0), mesh_paths


def _collision_meshes(stage: Any, root: Any, UsdGeom: Any) -> list[Any]:
    prefix = str(root.GetPath()) + "/"
    return [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix)
        and prim.IsA(UsdGeom.Mesh)
        and prim.GetName().startswith("collision_")
    ]


def _mesh_points_world(
    stage: Any,
    roots: list[Any],
    omni: Any,
    Gf: Any,
    UsdGeom: Any,
    np: Any,
    mesh_prefix: str,
) -> tuple[Any, list[str]]:
    """Collect authored mesh vertices in world coordinates for contact planning.

    This is read-only geometry inspection.  The returned points are used only
    to choose a surface-side waypoint; no mesh or object transform is edited.
    """
    points_world: list[Any] = []
    paths: list[str] = []
    for root in roots:
        prefix = str(root.GetPath()) + "/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(prefix) or not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.GetName().startswith(mesh_prefix):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            if not points:
                continue
            matrix = omni.usd.get_world_transform_matrix(prim)
            from i2ia.tasks.usd_affine_points import transform_points
            points_world.append(transform_points(points, matrix))
            paths.append(path)
    if not points_world:
        raise RuntimeError(f"no non-empty {mesh_prefix} meshes below geometry roots")
    return np.vstack(points_world), paths


def _mesh_bounds_for_roots(
    stage: Any,
    roots: list[Any],
    omni: Any,
    Gf: Any,
    UsdGeom: Any,
    np: Any,
    mesh_prefix: str,
) -> tuple[Any, Any, list[str]]:
    """Combine bounds below authored dynamic body roots only.

    Prepared validation stages may carry a static preview floor and a
    PhysicsMaterial beside the object.  Including that utility geometry in
    the object's extent makes every normal-sized object appear 4 m wide and
    incorrectly disables the controller workspace.  Dynamic body roots are
    structural evidence already present in the stage, so filtering to them
    does not alter the asset or introduce a fallback.
    """
    if not roots:
        raise RuntimeError("no dynamic body roots for geometry bounds")
    mins: list[Any] = []
    maxs: list[Any] = []
    paths: list[str] = []
    for root in roots:
        lo, hi, found = _mesh_bounds(stage, root, omni, Gf, UsdGeom, np, mesh_prefix)
        mins.append(np.asarray(lo, dtype=np.float64))
        maxs.append(np.asarray(hi, dtype=np.float64))
        paths.extend(found)
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0), paths


def _collision_meshes_for_roots(stage: Any, roots: list[Any], UsdGeom: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for root in roots:
        for prim in _collision_meshes(stage, root, UsdGeom):
            path = str(prim.GetPath())
            if path not in seen:
                result.append(prim)
                seen.add(path)
    return result


def _body_records(stage: Any, root: Any, UsdPhysics: Any) -> list[Any]:
    prefix = str(root.GetPath()) + "/"
    return [prim for prim in stage.Traverse() if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI)]


def _joint_records(stage: Any, root: Any, UsdPhysics: Any) -> list[Any]:
    prefix = str(root.GetPath()) + "/"
    return [prim for prim in stage.Traverse() if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdPhysics.Joint)]


def _attr(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None


def _numeric_values(value: Any) -> list[float] | None:
    """Convert a USD scalar/vector attribute to finite-checkable numbers.

    USD Python bindings expose Gf vectors as iterable objects, while a few
    schema values can arrive as strings.  Keeping this conversion local makes
    the runtime gate explicit without changing the authored asset.
    """

    if value is None:
        return None
    try:
        if isinstance(value, (str, bytes)):
            raw_values = re.findall(r"[-+0-9.eE]+", str(value))
            values = [float(raw) for raw in raw_values]
        else:
            try:
                values = [float(item) for item in value]
            except TypeError:
                values = [float(value)]
    except (TypeError, ValueError):
        return None
    return values if values and all(math.isfinite(item) for item in values) else None


def _look_at(position: Any, target: Any, np: Any, rot_utils: Any) -> Any:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        forward = np.asarray([0.0, 0.0, -1.0])
    else:
        forward /= norm
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    side = np.cross(up_hint, forward)
    if float(np.linalg.norm(side)) < 1e-8:
        side = np.cross(np.asarray([0.0, 1.0, 0.0]), forward)
    side /= np.linalg.norm(side)
    up = np.cross(forward, side)
    up /= np.linalg.norm(up)
    return rot_utils.rot_matrices_to_quats(np.column_stack((forward, side, up)))


def _finite_action_values(action: Any, np: Any) -> list[float] | None:
    """Extract a finite joint-position action without guessing missing values."""
    values = getattr(action, "joint_positions", None)
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float).tolist()


def _world_pose(omni: Any, prim: Any, np: Any) -> tuple[Any, Any]:
    """Read a prim world pose (translation + normalized wxyz quaternion)."""
    matrix = omni.usd.get_world_transform_matrix(prim)
    tr = matrix.ExtractTranslation()
    pos = np.asarray([float(tr[i]) for i in range(3)], dtype=np.float64)
    q = matrix.ExtractRotationQuat()
    im = q.GetImaginary()
    quat = np.asarray([float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-12:
        quat /= norm
    return pos, quat


def _franka_finger_world_positions(stage: Any, omni: Any, np: Any) -> dict[str, list[float]]:
    """Read the authored Franka finger link origins for diagnostics.

    This is observation only.  The collection controller still sends joint
    targets to the Franka articulation; no finger or object transform is
    authored here.  Keeping the two finger origins in the trace makes small
    handle-contact failures auditable without inferring a tool offset from a
    success result.
    """
    result: dict[str, list[float]] = {}
    for prim in stage.Traverse():
        name = str(prim.GetName()).lower()
        if name not in {"panda_leftfinger", "panda_rightfinger"}:
            continue
        try:
            p = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
            result[str(prim.GetName())] = [float(p[i]) for i in range(3)]
        except Exception:
            continue
    return result


def _franka_finger_midpoint(stage: Any, omni: Any, np: Any) -> Any | None:
    """Return the measured midpoint of the two finger link origins.

    This is an observation-only tool-frame estimate used by the optional
    articulated contact controller.  It never authors a transform or a
    velocity on the target asset.
    """
    positions = _franka_finger_world_positions(stage, omni, np)
    values = [np.asarray(positions[name], dtype=np.float64) for name in ("panda_leftfinger", "panda_rightfinger") if name in positions]
    if len(values) != 2:
        return None
    midpoint = (values[0] + values[1]) / 2.0
    return midpoint if np.all(np.isfinite(midpoint)) else None


def _franka_contact_finger_position(stage: Any, omni: Any, np: Any, side: str) -> Any | None:
    """Return a registered finger witness position for contact control."""
    positions = _franka_finger_world_positions(stage, omni, np)
    if side in {"left", "right"}:
        key = "panda_leftfinger" if side == "left" else "panda_rightfinger"
        value = positions.get(key)
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        return result if np.all(np.isfinite(result)) else None
    return _franka_finger_midpoint(stage, omni, np)


def _joint_axis_and_pivot(stage: Any, joint: Any, omni: Any, Gf: Any, UsdPhysics: Any, np: Any) -> tuple[Any, Any]:
    """Return an auditable world-space joint pivot and unit axis.

    The repair runner never guesses a joint frame.  It first uses the
    canonical ``i2ia:axis`` authored by the adapter, then falls back to the
    USD axis token transformed by the parent link's world rotation.  The
    pivot is always the authored ``physics:localPos0`` on body0.
    """
    rel0 = UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()
    parent = stage.GetPrimAtPath(rel0[0]) if rel0 else None
    if parent is None or not parent.IsValid():
        raise RuntimeError(f"joint {joint.GetPath()} has no valid body0")
    local_pos = _attr(joint, "physics:localPos0")
    if local_pos is None:
        raise RuntimeError(f"joint {joint.GetPath()} has no authored localPos0")
    parent_matrix = omni.usd.get_world_transform_matrix(parent)
    pivot = parent_matrix.Transform(Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])))
    token = _attr(joint, "physics:axis")
    rotation = _attr(joint, "physics:localRot0")
    if token not in ("X", "Y", "Z") or rotation is None:
        raise RuntimeError("missing_authored_joint_axis_or_frame")
    # USD axis lives in the JOINT frame, not body0 or custom source frame.
    basis = {"X": (1.,0.,0.), "Y": (0.,1.,0.), "Z": (0.,0.,1.)}[token]
    axis_local = Gf.Rotation(rotation).TransformDir(Gf.Vec3d(*basis))
    axis = np.asarray(parent_matrix.TransformDir(axis_local), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError(f"joint {joint.GetPath()} axis is degenerate")
    return np.asarray([float(pivot[i]) for i in range(3)], dtype=np.float64), axis / norm


def _child_for_joint(stage: Any, joint: Any, target_body: Any, UsdPhysics: Any) -> Any:
    rel = UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()
    if rel:
        child = stage.GetPrimAtPath(rel[0])
        if child and child.IsValid():
            return child
    # The authored custom childLink is a structural identifier, not a
    # category-specific fallback.  Resolve it only among existing body prims.
    child_name = _attr(joint, "i2ia:childLink")
    if child_name:
        for prim in stage.Traverse():
            if prim.GetName() == str(child_name) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
    return target_body


def _quaternion_angle(q0: Any, q1: Any, np: Any, math_module: Any) -> float:
    dot = min(1.0, max(-1.0, abs(float(np.dot(q0, q1)))))
    return float(2.0 * math_module.acos(dot))


def _font(ImageFont: Any, size: int, bold: bool = False) -> Any:
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _overlay(rgb: Any, *, phase: str, category: str, body_position: Any, metric: float, robot_success: bool | None, Image: Any, ImageDraw: Any, ImageFont: Any, np: Any) -> Any:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle((0, 0, width, 66), fill=(3, 10, 18, 230))
    draw.text((16, 12), "VLA TWIN", font=_font(ImageFont, 22, True), fill=(76, 232, 187, 255))
    draw.text((145, 15), f"EXP3 | {category} | ISAAC/PHYSX", font=_font(ImageFont, 16, True), fill=(239, 246, 250, 255))
    draw.rectangle((14, 82, min(width - 14, 430), 176), fill=(4, 12, 21, 218), outline=(64, 104, 126, 230), width=2)
    draw.text((28, 96), f"phase: {phase}", font=_font(ImageFont, 15, True), fill=(247, 250, 252, 255))
    draw.text((28, 126), f"body p=({body_position[0]:+.3f}, {body_position[1]:+.3f}, {body_position[2]:+.3f})", font=_font(ImageFont, 12), fill=(180, 218, 236, 255))
    robot_label = "PASS" if robot_success is True else "attempt" if robot_success is False else "not-evaluated"
    draw.text((28, 150), f"metric={metric:+.4f} | robot={robot_label}", font=_font(ImageFont, 12), fill=(205, 222, 230, 255))
    draw.rectangle((14, height - 54, width - 14, height - 12), fill=(3, 11, 19, 225), outline=(55, 91, 112, 220), width=2)
    draw.text((28, height - 42), "local learned robot policy; no teacher / fallback / object transform writes", font=_font(ImageFont, 10), fill=(193, 211, 220, 255))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "i2ia.interaction_first_exp1_isaac_environment.v0.1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.free", "--format=csv,noheader"]),
        "packages": {name: package_version(name) for name in ("isaacsim", "numpy", "Pillow", "imageio", "imageio-ffmpeg")},
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "argv": sys.argv,
        "renderer": args.renderer,
        "eula_confirmation": "ACCEPT_EULA=Y",
    }


def run_case(args: argparse.Namespace, app: Any) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np
    import omni
    import omni.kit.app
    from PIL import Image, ImageDraw, ImageFont
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from i2ia.tasks.complete_robot_precision import evaluate as evaluate_complete_task
    full_spec = json.loads(Path(os.environ["EXP3_COMPLETE_TASK_SPEC"]).read_text())
    if full_spec["task_type"] not in ("push_to_region","open","close") or full_spec["admission_status"] != "ready_for_pilot":
        raise RuntimeError("complete_task_blocked_no_articulated_fallback")

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    # The modern articulation view is used only to restore the authored
    # root frame during pre-play scene setup.  It is never commanded after
    # ``world.play()``; all episode actions are Franka actions.
    from isaacsim.core.prims import Articulation as CoreArticulation
    from isaacsim.core.utils.stage import is_stage_loading, open_stage
    from isaacsim.core.utils.types import ArticulationAction
    import isaacsim.core.utils.numpy.rotations as rot_utils
    # Isaac Sim 6 keeps the legacy Franka wrapper in an explicitly deprecated
    # extension and no longer adds the examples package to the namespace by
    # default.  Extend only the in-process namespace (never the asset or
    # history directories) so the v0.84 runner retains the same robot/action
    # contract as the frozen 5.x implementation.
    try:
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
    except ModuleNotFoundError:
        import isaacsim as _isaacsim
        import isaacsim.robot as _robot_ns
        import isaacsim.robot.manipulators as _manip_ns
        _deprecated_examples = Path(_isaacsim.__file__).resolve().parent / "extsDeprecated" / "isaacsim.robot.manipulators.examples"
        _robot_ns.__path__.append(str(_deprecated_examples / "isaacsim" / "robot"))
        _manip_ns.__path__.append(str(_deprecated_examples / "isaacsim" / "robot" / "manipulators"))
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
    from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
    from isaacsim.sensors.camera import Camera
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    root = args.repo_root.resolve()
    if args.controller != "local_robot_policy":raise RuntimeError("teacher_or_fallback_forbidden_in_vla_worker")
    policy_identity=json.loads(args.policy_identity.read_text())
    if policy_identity["model_variant"] != args.model_variant:raise RuntimeError("policy_variant_identity_mismatch")
    case_manifest_path = args.case_manifest.resolve()
    case = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = _runtime_environment(args)
    env["schema_version"] = "i2ia.exp3_success_trace_environment.v0.2"
    write_json(output / "environment.json", env)
    usd_raw = case.get("asset", {}).get("usd_path")
    if not isinstance(usd_raw, str) or not usd_raw:
        raise RuntimeError("case has no USD asset path")
    usd_path = _repo_path(usd_raw, root)
    if not usd_path.is_file() or usd_path.is_symlink():
        raise RuntimeError(f"USD asset is missing or symlinked: {usd_path}")
    expected_usd_sha = str(case.get("asset", {}).get("usd_sha256", ""))
    actual_usd_sha = sha256_file(usd_path)
    if expected_usd_sha and actual_usd_sha != expected_usd_sha:
        raise RuntimeError("USD hash does not match prepared case manifest")
    category = str(case.get("requested_category", ""))
    stratum = str(case.get("interaction_stratum", "rigid"))
    if not open_stage(str(usd_path)):
        raise RuntimeError(f"Isaac refused to open USD: {usd_path}")
    deadline = time.monotonic() + 180.0
    while is_stage_loading():
        if time.monotonic() > deadline:
            raise TimeoutError("stage loading timeout")
        app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac returned no stage")
    stage.SetEditTarget(stage.GetSessionLayer())
    # A prepared asset stage commonly contains a PhysicsMaterial (and may
    # contain other non-asset utility prims) alongside the actual asset root.
    # The old implementation counted every child of /World and therefore
    # rejected otherwise valid stages such as ObjectAsset + PhysicsMaterial.
    # Select roots by the presence of an authored body API, never by a
    # candidate-specific name or by adding a fallback asset.
    world_children = [prim for prim in stage.GetPrimAtPath("/World").GetChildren()
                      if prim.GetName() != "physicsScene"]
    roots = []
    for prim in world_children:
        try:
            if _body_records(stage, prim, UsdPhysics):
                roots.append(prim)
        except Exception:
            continue
    if len(roots) != 1:
        details = [(str(prim.GetPath()), prim.GetTypeName()) for prim in world_children]
        raise RuntimeError(f"expected one authored asset root under /World, found {len(roots)}; children={details}")
    object_root = roots[0]
    if stratum == "rigid" and "rigid_reset_rotation_xyz_deg" in full_spec:
        rotation = full_spec["rigid_reset_rotation_xyz_deg"]
        if len(rotation) != 3 or not all(math.isfinite(x) for x in rotation):
            raise RuntimeError("invalid_registered_rigid_reset_rotation")
        UsdGeom.XformCommonAPI(object_root).SetRotate(Gf.Vec3f(*rotation))
        write_json(output/"rigid_prephysics_orientation.json",dict(rotation_xyz_deg=rotation,
                   source="registered_source_collision_stable_pose",pre_episode_reset_only=True,
                   geometry_changed=False))
    # Utility floor is not a semantic object link. The canonical standalone
    # asset includes its own validation floor; the robot workcell has one
    # registered support already. Disable only the explicitly registered
    # duplicate utility prim in this anonymous session before reset.
    removed_supports=[]
    for utility_path in full_spec.get("duplicate_validation_support_paths",[]):
        utility=stage.GetPrimAtPath(utility_path)
        if not utility.IsValid():raise RuntimeError("registered_duplicate_support_missing")
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(utility)):raise RuntimeError("utility_support_contains_semantic_body")
        utility.SetActive(False);removed_supports.append(utility_path)
    write_json(output/"workcell_support_composition.json",{"disabled_utility_supports":removed_supports,"edit_target":"anonymous_session_layer_pre_reset","original_usd_unchanged":True,"registered_support":"/World/Exp1Table"})
    try:
        asset_uniform_scale = float(args.asset_uniform_scale)
    except (TypeError, ValueError):
        raise RuntimeError("asset_uniform_scale must be finite")
    if not math.isfinite(asset_uniform_scale) or asset_uniform_scale <= 0.0:
        raise RuntimeError("asset_uniform_scale must be finite and positive")
    try:
        articulated_pull_distance = float(args.articulated_pull_distance)
    except (TypeError, ValueError):
        raise RuntimeError("articulated_pull_distance must be finite")
    if not math.isfinite(articulated_pull_distance) or articulated_pull_distance <= 0.0:
        raise RuntimeError("articulated_pull_distance must be finite and positive")
    # Older diagnostic callers construct a SimpleNamespace directly.  Keep
    # those read-only callers compatible while exposing the new, explicitly
    # registered tool-envelope offset to the batch runner.
    ee_target_offset = np.asarray(
        [
            float(getattr(args, "articulated_ee_target_offset_x", 0.0)),
            float(getattr(args, "articulated_ee_target_offset_y", 0.0)),
            float(getattr(args, "articulated_ee_target_offset_z", 0.0)),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(ee_target_offset)) or float(np.linalg.norm(ee_target_offset)) > 0.35:
        raise RuntimeError("articulated EE target offset must be finite and <= 0.35 m")
    if int(args.articulated_joint_index) < 0:
        raise RuntimeError("articulated_joint_index must be non-negative")
    authored_scale_applied = False
    if abs(asset_uniform_scale - 1.0) > 1e-12:
        # This is a pre-play, session-layer unit normalization.  It scales
        # the complete authored articulation uniformly, preserving topology,
        # link hierarchy, joint axes and relative geometry.  The caller must
        # register the value; this runner never adapts it from a task outcome.
        UsdGeom.XformCommonAPI(object_root).SetScale(
            Gf.Vec3f(asset_uniform_scale, asset_uniform_scale, asset_uniform_scale)
        )
        authored_scale_applied = True
        for _ in range(3):
            app.update()
    bodies = _body_records(stage, object_root, UsdPhysics)
    joints = _joint_records(stage, object_root, UsdPhysics)
    if not bodies:
        raise RuntimeError("loaded asset has no rigid body API")
    # A uniform density policy keeps mass properties physically consistent with
    # a pre-play unit normalization.  This is authored only in the session
    # layer and is reported as an adaptation; the immutable source USD is
    # never edited.  Static bodies retain their authored values.
    mass_property_scale_policy = "preserve_authored_mass_properties"
    mass_property_scale_records: list[dict[str, Any]] = []
    if authored_scale_applied:
        mass_property_scale_policy = "uniform_density_mass_s3_inertia_s5"
        mass_factor = asset_uniform_scale ** 3
        inertia_factor = asset_uniform_scale ** 5
        for body in bodies:
            motion_type = _attr(body, "i2ia:motionType")
            if motion_type != "dynamic":
                continue
            mass_attr = body.GetAttribute("physics:mass")
            inertia_attr = body.GetAttribute("physics:diagonalInertia")
            old_mass_values = _numeric_values(mass_attr.Get() if mass_attr.IsValid() else None)
            old_inertia_values = _numeric_values(inertia_attr.Get() if inertia_attr.IsValid() else None)
            record: dict[str, Any] = {
                "body": str(body.GetPath()),
                "mass_before_kg": old_mass_values[0] if old_mass_values else None,
                "inertia_before_kg_m2": old_inertia_values,
            }
            if old_mass_values and mass_attr.IsValid():
                mass_attr.Set(float(old_mass_values[0] * mass_factor))
                record["mass_after_kg"] = float(old_mass_values[0] * mass_factor)
            else:
                record["mass_after_kg"] = None
            if len(old_inertia_values) == 3 and inertia_attr.IsValid():
                new_inertia = [float(value * inertia_factor) for value in old_inertia_values]
                old_inertia = inertia_attr.Get()
                try:
                    inertia_attr.Set(Gf.Vec3f(*new_inertia) if isinstance(old_inertia, Gf.Vec3f) else Gf.Vec3d(*new_inertia))
                except Exception:
                    inertia_attr.Set(tuple(new_inertia))
                record["inertia_after_kg_m2"] = new_inertia
            else:
                record["inertia_after_kg_m2"] = None
            mass_property_scale_records.append(record)
    dynamic_bodies = [body for body in bodies if _attr(body, "i2ia:motionType") == "dynamic"]
    # Whole-asset placement/framing must include the stationary housing.
    # A drawer bottom is not the cabinet's support surface. Geometric-free
    # root anchors are skipped, but no physical/semantic link is discarded.
    geometry_roots = [body for body in bodies if any(
        p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(body))]
    if not geometry_roots:
        raise RuntimeError("whole_asset_geometry_missing")
    write_json(output / "whole_asset_geometry_scope.json", {
        "body_paths": [str(p.GetPath()) for p in bodies],
        "geometry_paths": [str(p.GetPath()) for p in geometry_roots],
        "includes_static_housing": True, "source_assets_modified": False,
    })
    visual_min, visual_max, visual_paths = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "visual_")
    visual_extent = visual_max - visual_min
    if not np.all(np.isfinite(visual_extent)) or np.any(visual_extent <= 1e-8):
        raise RuntimeError("loaded asset has degenerate visual bounds")
    collision_prims = _collision_meshes_for_roots(stage, geometry_roots, UsdGeom)
    collision_min, collision_max, collision_paths = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
    collision_extent = collision_max - collision_min
    target_body = dynamic_bodies[0] if dynamic_bodies else bodies[-1]
    # For articulated tasks the legal robot-contact body is the registered
    # operation link, not whichever dynamic link happens to be first in USD
    # traversal order.  Choosing it here keeps contact legality and the task
    # evaluator aligned without changing the authored hierarchy or collision
    # policy.
    if stratum == "articulated" and full_spec.get("target_link"):
        requested_target = stage.GetPrimAtPath(str(full_spec["target_link"]))
        if requested_target.IsValid() and requested_target.HasAPI(UsdPhysics.RigidBodyAPI):
            target_body = requested_target
    target_body_id = str(_attr(target_body, "i2ia:bodyId") or target_body.GetName())
    # Retain all declared physical values before runtime; no values are
    # invented here.  The static contract is joined by the preparation step.
    body_physics = []
    for body in bodies:
        mass_value = _attr(body, "physics:mass")
        com_value = _attr(body, "physics:centerOfMass")
        inertia_value = _attr(body, "physics:diagonalInertia")
        body_physics.append({
            "prim_path": str(body.GetPath()),
            "body_id": str(_attr(body, "i2ia:bodyId") or body.GetName()),
            "motion_type": _attr(body, "i2ia:motionType"),
            "mass_kg": mass_value,
            "center_of_mass_m": str(com_value) if com_value is not None else None,
            "center_of_mass_values_m": _numeric_values(com_value),
            "diagonal_inertia_kg_m2": str(inertia_value) if inertia_value is not None else None,
            "diagonal_inertia_values_kg_m2": _numeric_values(inertia_value),
        })
    joint_physics = []
    for joint in joints:
        joint_type = "revolute" if joint.IsA(UsdPhysics.RevoluteJoint) else "prismatic" if joint.IsA(UsdPhysics.PrismaticJoint) else "fixed" if joint.IsA(UsdPhysics.FixedJoint) else "unknown"
        joint_physics.append({
            "prim_path": str(joint.GetPath()),
            "joint_id": str(_attr(joint, "i2ia:jointId") or joint.GetName()),
            "joint_type": joint_type,
            "axis": str(_attr(joint, "physics:axis")) if _attr(joint, "physics:axis") is not None else None,
            "lower_limit": _attr(joint, "physics:lowerLimit"),
            "upper_limit": _attr(joint, "physics:upperLimit"),
            "body0": [str(x) for x in UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()],
            "body1": [str(x) for x in UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()],
        })
    support_top_z = float(getattr(args, "support_top_z", 0.90))
    if not math.isfinite(support_top_z) or support_top_z <= 0.0:
        raise RuntimeError("support_top_z must be finite and positive")
    # Put the exact retrieved geometry on a standard validation table.  This
    # is a pre-play setup transform, retained in the session layer only.
    table_top = support_top_z
    extent = np.asarray(visual_extent, dtype=np.float64)
    center = (visual_min + visual_max) / 2.0
    # Place the lowest visual point just above the table.  The 2 mm clearance
    # is intentional (avoids an authored initial penetration) and gravity is
    # then allowed to settle the exact retrieved body; no post-play transform
    # is used as a substitute for contact.
    placement_clearance_m = 0.002
    # Keep the validation object in the central reachable part of the fixed
    # Franka workcell.  This is a workcell placement, not a candidate change;
    # the authored asset remains hash-locked and is only translated in the
    # anonymous session layer before play.
    # Collection workers may provide deterministic pre-registered x/y
    # perturbations.  They are applied before World.play and are recorded in
    # the result; no post-play transform is ever authored.
    try:
        requested_x = float(os.environ.get("EXP3_INITIAL_X_M", "-0.15"))
        requested_y = float(os.environ.get("EXP3_INITIAL_Y_M", "0.0"))
    except ValueError:
        requested_x, requested_y = -0.15, 0.0
    # For articulated assets, the body that a robot can actually contact may
    # be far from the combined link bounds (for example, a door leaf can be
    # offset from its hinge).  The contact protocol therefore uses the
    # authored first-joint pivot as the setup anchor.  This keeps the actual
    # joint frame inside the fixed Franka workspace without changing any
    # link mesh or joint metadata.  The root correction is setup-only,
    # before world.play; it is never an episode-time teleport.
    placement_child = None
    placement_child_center = None
    if stratum == "articulated" and args.articulated_placement == "actuated_child_center":
        movable_for_placement = [j for j in joints if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)]
        if movable_for_placement:
            try:
                placement_index = int(args.articulated_joint_index)
                if placement_index >= len(movable_for_placement):
                    raise RuntimeError(
                        f"articulated_joint_index={placement_index} outside movable joint count={len(movable_for_placement)}"
                    )
                placement_child = _child_for_joint(stage, movable_for_placement[placement_index], target_body, UsdPhysics)
                c_lo, c_hi, _ = _mesh_bounds(stage, placement_child, omni, Gf, UsdGeom, np)
                placement_child_center = (np.asarray(c_lo, dtype=np.float64) + np.asarray(c_hi, dtype=np.float64)) / 2.0
            except Exception:
                placement_child = None
                placement_child_center = None
    desired_center = np.asarray([requested_x, requested_y, table_top + float(extent[2]) / 2.0 + placement_clearance_m], dtype=np.float64)
    # Rigid assets use their visual center.  Articulations are anchored later
    # from the measured joint pivot after PhysX reconstructs the authored
    # hierarchy; this initial delta is retained only for diagnostic lineage.
    placement_delta = desired_center - center
    if stratum != "articulated":
        UsdGeom.XformCommonAPI(object_root).SetTranslate(Gf.Vec3d(*[float(x) for x in placement_delta]))
        # Flush the session-layer edit before reading world-space bounds.
        for _ in range(2):
            app.update()
    # Some authored multi-link assets keep an additional local frame on the
    # body prim.  Re-read the child after the root setup write and apply one
    # deterministic pre-play correction so the contacted link is actually at
    # the registered workcell slot.  This is still session-layer setup, never
    # an episode-time teleport.
    if placement_child is not None and stratum != "articulated":
        try:
            corrected_lo, corrected_hi, _ = _mesh_bounds(stage, placement_child, omni, Gf, UsdGeom, np)
            corrected_center = (np.asarray(corrected_lo, dtype=np.float64) + np.asarray(corrected_hi, dtype=np.float64)) / 2.0
            placement_delta = placement_delta + np.asarray([requested_x - corrected_center[0], requested_y - corrected_center[1], 0.0], dtype=np.float64)
            UsdGeom.XformCommonAPI(object_root).SetTranslate(Gf.Vec3d(*[float(x) for x in placement_delta]))
            for _ in range(2):
                app.update()
        except Exception:
            # Keep the first authored placement and retain the failure in the
            # structured target debug record rather than guessing a frame.
            pass
    placed_min, placed_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "visual_")
    placed_collision_min, placed_collision_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
    # Visual and collision meshes can have a small authored bottom offset.
    # Correct that offset once in the session layer before play so the frozen
    # support setup starts without penetration.  This is geometry-derived
    # placement, never an episode-time object transform write.
    if stratum != "articulated" and float(placed_collision_min[2]) < table_top + placement_clearance_m:
        setup_z_correction = float(table_top + placement_clearance_m - float(placed_collision_min[2]))
        placement_delta = placement_delta + np.asarray([0.0, 0.0, setup_z_correction], dtype=np.float64)
        UsdGeom.XformCommonAPI(object_root).SetTranslate(Gf.Vec3d(*[float(x) for x in placement_delta]))
        for _ in range(2):
            app.update()
        placed_min, placed_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "visual_")
        placed_collision_min, placed_collision_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
    body_bounds_debug: list[dict[str, Any]] = []
    for body in geometry_roots:
        try:
            body_visual_lo, body_visual_hi, _ = _mesh_bounds(stage, body, omni, Gf, UsdGeom, np, "visual_")
            body_collision_lo, body_collision_hi, _ = _mesh_bounds(stage, body, omni, Gf, UsdGeom, np, "collision_")
            body_bounds_debug.append({
                "path": str(body.GetPath()),
                "visual_min": np.asarray(body_visual_lo, dtype=np.float64).tolist(),
                "visual_max": np.asarray(body_visual_hi, dtype=np.float64).tolist(),
                "collision_min": np.asarray(body_collision_lo, dtype=np.float64).tolist(),
                "collision_max": np.asarray(body_collision_hi, dtype=np.float64).tolist(),
            })
        except Exception as exc:
            body_bounds_debug.append({"path": str(body.GetPath()), "error": f"{type(exc).__name__}:{exc}"})
    # Add neutral display materials in the anonymous session layer only.
    material = UsdShade.Material.Define(stage, "/World/Exp1ValidationMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Exp1ValidationMaterial/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15, 0.55, 0.82) if stratum == "articulated" else Gf.Vec3f(0.85, 0.28, 0.08))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(str(object_root.GetPath()) + "/") and prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith("visual_"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    dome = UsdLux.DomeLight.Define(stage, "/World/Exp1Dome")
    dome.CreateIntensityAttr(650.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Exp1Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(35.0, -25.0, 25.0))

    world = World(physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / args.fps, stage_units_in_meters=1.0)
    world.scene.add(FixedCuboid(prim_path="/World/Exp1Table", name="exp1_table", position=np.asarray([0.0, 0.0, table_top - 0.025]), scale=np.asarray([args.support_width, args.support_depth, 0.05]), color=np.asarray([0.25, 0.22, 0.18])))
    target_articulation = None
    if stratum == "articulated":
        # Keep the multi-link articulation registered with Isaac's scene so
        # reset restores the authored hierarchy and the session-layer setup
        # pose can be applied through the root view before play.
        target_articulation = world.scene.add(
            CoreArticulation(
                prim_paths_expr=str(object_root.GetPath()),
                name="exp3_target_articulation",
                reset_xform_properties=False,
            )
        )
    # Adjacent links in a PhysX articulation may have intentionally overlapping
    # authored collision envelopes at a hinge (for example a laptop display
    # and its base).  Filter only those parent/child pairs in the anonymous
    # session layer.  This preserves every link, mesh, joint and robot/object
    # contact while preventing an internal articulation self-collision from
    # invalidating the initial state.  The policy is auditable and fail-closed:
    # an inability to author the filter is retained in the result instead of
    # silently changing geometry or using a fallback body.
    articulation_self_collision_policy = "not_applicable"
    articulation_filtered_pairs: list[dict[str, Any]] = []
    articulation_self_collision_error = None
    if stratum == "articulated":
        try:
            articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(object_root)
            if str(getattr(args, "articulated_self_collision_policy", "disabled_filtered")) == "enabled":
                articulation_api.CreateEnabledSelfCollisionsAttr().Set(True)
                articulation_self_collision_policy = "enabled_authored_articulation"
            else:
                articulation_api.CreateEnabledSelfCollisionsAttr().Set(False)
                articulation_self_collision_policy = "disabled_for_authored_articulation"
        except Exception as exc:
            articulation_self_collision_error = f"articulation_self_collision_api_failed:{type(exc).__name__}:{exc}"
        for joint in joints:
            if str(getattr(args, "articulated_self_collision_policy", "disabled_filtered")) == "enabled":
                continue
            if not (joint.IsA(UsdPhysics.RevoluteJoint) or joint.IsA(UsdPhysics.PrismaticJoint) or joint.IsA(UsdPhysics.FixedJoint)):
                continue
            body0_targets = UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()
            body1_targets = UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()
            if not body0_targets or not body1_targets:
                continue
            body0 = stage.GetPrimAtPath(body0_targets[0])
            body1 = stage.GetPrimAtPath(body1_targets[0])
            if not body0.IsValid() or not body1.IsValid():
                continue
            try:
                UsdPhysics.FilteredPairsAPI.Apply(body0).CreateFilteredPairsRel().AddTarget(body1.GetPath())
                UsdPhysics.FilteredPairsAPI.Apply(body1).CreateFilteredPairsRel().AddTarget(body0.GetPath())
                body0_shapes = [
                    prim for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(str(body0.GetPath()) + "/")
                    and prim.IsA(UsdGeom.Mesh)
                    and prim.GetName().startswith("collision_")
                ]
                body1_shapes = [
                    prim for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(str(body1.GetPath()) + "/")
                    and prim.IsA(UsdGeom.Mesh)
                    and prim.GetName().startswith("collision_")
                ]
                for shape0 in body0_shapes:
                    for shape1 in body1_shapes:
                        UsdPhysics.FilteredPairsAPI.Apply(shape0).CreateFilteredPairsRel().AddTarget(shape1.GetPath())
                        UsdPhysics.FilteredPairsAPI.Apply(shape1).CreateFilteredPairsRel().AddTarget(shape0.GetPath())
                articulation_filtered_pairs.append({
                    "joint": str(joint.GetPath()),
                    "body0": str(body0.GetPath()),
                    "body1": str(body1.GetPath()),
                    "body0_collision_shape_count": len(body0_shapes),
                    "body1_collision_shape_count": len(body1_shapes),
                })
            except Exception as exc:
                articulation_filtered_pairs.append({
                    "joint": str(joint.GetPath()),
                    "body0": str(body0.GetPath()),
                    "body1": str(body1.GetPath()),
                    "error": f"filtered_pair_authoring_failed:{type(exc).__name__}:{exc}",
                })
        if articulation_self_collision_policy == "not_applicable":
            articulation_self_collision_policy = "filtered_adjacent_pairs_only"
    # This runner is exclusively the real-contact robot track.  Keeping a
    # single explicit arm path avoids silently falling back to object control.
    if args.attempt_robot != "true":
        raise RuntimeError("real-contact runner requires --attempt-robot true")
    robot_attempted = True
    # Supplemental collection uses the same pre-play placement contract but
    # does not silently reject a large authored mesh.  Reachability is decided
    # by the recorded IK/contact gates below; no scale or transform is changed
    # to make an object fit the robot workspace.
    workspace_ok = bool(float(np.min(extent)) >= 0.002 and float(np.linalg.norm(desired_center[:2])) <= 0.65 and float(desired_center[2]) <= 2.5)
    franka = None
    solver = None
    rmpflow_controller = None
    initial_joints = np.asarray([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, math.pi / 4.0, 0.04, 0.04], dtype=np.float64)
    if "robot_initial_joint_positions" in full_spec:
        initial_joints = np.asarray(full_spec["robot_initial_joint_positions"], dtype=np.float64)
        contract = full_spec.get("robot_initial_state_contract", {})
        if contract.get("phase") != "pre_episode_reset_only" or contract.get("object_contact_at_start") is not False:
            raise RuntimeError("registered_robot_reset_contract_invalid")
        if initial_joints.shape != (9,) or not np.isfinite(initial_joints).all() or np.any(initial_joints < np.asarray(full_spec["action_limits"]["min"])) or np.any(initial_joints > np.asarray(full_spec["action_limits"]["max"])):
            raise RuntimeError("registered_robot_reset_joint_limits_invalid")
        write_json(output / "registered_robot_initial_state.json", {"joint_positions":initial_joints.tolist(),"contract":contract,"phase":"before_episode","object_control_command":False})
    base_position = np.asarray(
        [float(args.franka_base_x), float(args.franka_base_y), float(args.franka_base_z)],
        dtype=np.float64,
    )
    mount = full_spec.get("robot_mount", {})
    if mount.get("required") is True:
        lower, upper = np.asarray(mount["minimum"]), np.asarray(mount["maximum"])
        if not np.isfinite(np.r_[lower,upper]).all() or not np.all(upper > lower):
            raise RuntimeError("invalid_registered_robot_mount")
        world.scene.add(FixedCuboid(prim_path="/World/Exp3RobotPedestal",name="exp3_robot_pedestal",
                        position=(lower+upper)/2,scale=upper-lower,color=np.asarray([.32,.35,.36])))
        write_json(output/"robot_mount.json",{**mount,"asset_geometry_modified":False,
                   "object_attachment_created":False,"visible_collision_fixture":True})
    if robot_attempted:
        manager = omni.kit.app.get_app().get_extension_manager()
        extension = "isaacsim.robot.manipulators.examples"
        if not manager.is_extension_enabled(extension):
            manager.set_extension_enabled_immediate(extension, True)
        if not manager.is_extension_enabled(extension):
            raise RuntimeError("Franka extension is unavailable")
        franka_kwargs = {
            "prim_path": "/World/Exp1Franka",
            "name": "exp1_franka",
            "position": base_position,
            "orientation": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }
        if args.franka_usd_path:
            local_franka_usd = Path(args.franka_usd_path).expanduser().resolve()
            if not local_franka_usd.is_file():
                raise FileNotFoundError(f"local Franka USD does not exist: {local_franka_usd}")
            franka_kwargs["usd_path"] = str(local_franka_usd)
        franka = world.scene.add(Franka(**franka_kwargs))
        franka.set_default_state(position=base_position,orientation=np.asarray([1.,0.,0.,0.]))
        franka.set_joints_default_state(positions=initial_joints)
        franka.gripper.set_default_state(initial_joints[-2:])
    camera = world.scene.add(Camera(prim_path="/World/Exp1Camera", name="exp1_camera", resolution=(args.width, args.height), frequency=args.fps))
    extent_norm = float(max(np.max(extent), 0.25))
    # Use one fixed wide workcell view for both models and every episode.  A
    # crop-only camera can hide the arm, which would make raw video unable to
    # establish that contact came from the Franka.  The target is derived from
    # the authored object placement and the fixed Franka base, never from a
    # task result.
    scene_target = (np.asarray(desired_center, dtype=np.float64) * 0.62) + (base_position * 0.38)
    scene_target[2] = max(0.78, float(scene_target[2]))
    camera_position = scene_target + np.asarray([2.15, -3.45, 1.95])
    camera_target = scene_target.copy()
    camera.set_world_pose(position=camera_position, orientation=_look_at(camera_position, camera_target, np, rot_utils), camera_axes="world")
    camera.set_focal_length(2.2)
    camera.set_clipping_range(0.03, 20.0)
    # Observe all robot rigid bodies, including forbidden robot/support and self contacts.
    robot_report_bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Exp1Franka")) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    # Register contact reporting BEFORE PhysX creates actors at reset.
    # Late schema authoring did not produce table contact data in v0.1.
    for report_prim in [target_body, *collision_prims, stage.GetPrimAtPath("/World/Exp1Table"), *robot_report_bodies]:
        if report_prim and report_prim.IsValid():
            PhysxSchema.PhysxContactReportAPI.Apply(report_prim).CreateThresholdAttr().Set(0.0)
    if full_spec["task_type"]=="push_to_region":
        goal = full_spec["target_region_world_m"]
        for edge in range(4):
            marker=UsdGeom.Cube.Define(stage,f"/World/CompleteTaskGoal/edge_{edge}")
            marker.CreateSizeAttr(1.)
            low=np.asarray(goal["min"]);high=np.asarray(goal["max"]);center=(low+high)*.5
            if edge<2:
                point=[center[0],low[1] if edge==0 else high[1],table_top+.001]
                size=[high[0]-low[0],.002,.001]
            else:
                point=[low[0] if edge==2 else high[0],center[1],table_top+.001]
                size=[.002,high[1]-low[1],.001]
            UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*point))
            UsdGeom.XformCommonAPI(marker).SetScale(Gf.Vec3f(*size))
            marker.CreateDisplayColorAttr().Set([Gf.Vec3f(.05,.9,.05)])
    evidence_camera = world.scene.add(Camera(prim_path="/World/CompleteTaskEvidenceCamera",name="complete_task_evidence",resolution=(args.width,args.height),frequency=args.fps))
    evidence_position = scene_target + np.asarray([1.7,2.3,1.4])
    evidence_camera.set_world_pose(position=evidence_position,orientation=_look_at(evidence_position,camera_target,np,rot_utils),camera_axes="world")
    evidence_camera.set_focal_length(2.2);evidence_camera.set_clipping_range(.03,20.)
    registered_cameras=full_spec.get("registered_cameras")
    if registered_cameras:
        camera_position=np.asarray(registered_cameras["policy_position_m"])
        evidence_position=np.asarray(registered_cameras["evidence_position_m"])
        camera_target=np.asarray(registered_cameras["target_m"])
        for cam,pos in [(camera,camera_position),(evidence_camera,evidence_position)]:
            cam.set_horizontal_aperture(registered_cameras["horizontal_aperture"])
            cam.set_vertical_aperture(registered_cameras["vertical_aperture"])
            cam.set_focal_length(registered_cameras["focal_length"])
            cam.set_clipping_range(*registered_cameras["clipping_range_m"])
            cam.set_world_pose(position=pos,orientation=_look_at(pos,camera_target,np,rot_utils),camera_axes="world")
    camera_jitter=full_spec.get("camera_reset_delta", {})
    if camera_jitter:
        from scipy.spatial.transform import Rotation
        angles=[float(camera_jitter["azimuth_deg"]),float(camera_jitter["elevation_deg"])]
        distance_factor=float(camera_jitter["distance_factor"])
        if max(map(abs,angles))>10. or not .9<=distance_factor<=1.1:
            raise RuntimeError("camera_variation_outside_registered_envelope")
        rotate=Rotation.from_euler("zy",angles,degrees=True)
        camera_position=camera_target+distance_factor*rotate.apply(camera_position-camera_target)
        evidence_position=camera_target+distance_factor*rotate.apply(evidence_position-camera_target)
        write_json(output/"registered_camera_variation.json",dict(**camera_jitter,
            policy_position_m=camera_position.tolist(),evidence_position_m=evidence_position.tolist(),
            target_m=camera_target.tolist(),pre_episode_only=True,evidence_not_model_input=True))
    camera_jitter=full_spec.get("camera_reset_delta", {})
    if camera_jitter:
        from scipy.spatial.transform import Rotation
        angles=[float(camera_jitter["azimuth_deg"]),float(camera_jitter["elevation_deg"])]
        distance_factor=float(camera_jitter["distance_factor"])
        if max(map(abs,angles))>10. or not .9<=distance_factor<=1.1:
            raise RuntimeError("camera_variation_outside_registered_envelope")
        rotate=Rotation.from_euler("zy",angles,degrees=True)
        camera_position=camera_target+distance_factor*rotate.apply(camera_position-camera_target)
        evidence_position=camera_target+distance_factor*rotate.apply(evidence_position-camera_target)
        write_json(output/"registered_camera_variation.json",dict(**camera_jitter,
            policy_position_m=camera_position.tolist(),evidence_position_m=evidence_position.tolist(),
            target_m=camera_target.tolist(),pre_episode_only=True,evidence_not_model_input=True))
    startup_events=[];startup_contacts=[]
    def startup_snapshot(phase):
        lo,hi,_=_mesh_bounds_for_roots(stage,geometry_roots,omni,Gf,UsdGeom,np,"visual_")
        startup_events.append({"phase":phase,"simulation_time_s":float(world.current_time),"bounds_min":lo.tolist(),"bounds_max":hi.tolist(),"contacts":startup_contacts[:]})
        startup_contacts.clear()
        write_json(output/"startup_evidence.json",{"events":startup_events,"observation_only":True})
    def startup_on_contact(headers,data):
        for header in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.actor0,header.actor1,header.collider0,header.collider1)]
            samples=[]
            for c in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]:
                samples.append({"position":list(c.position),"impulse":list(c.impulse),"separation":float(c.separation)})
            startup_contacts.append({"paths":paths,"samples":samples})
    startup_subscription=get_physx_simulation_interface().subscribe_contact_report_events(startup_on_contact)
    # Seed only ROBOT joints in the anonymous initialization layer before
    # PhysX's first reset step. Articulation.post_reset occurs too late.
    robot_joint_initialization=[]
    values={**{f"panda_joint{i+1}":float(initial_joints[i]) for i in range(7)},
            "panda_finger_joint1":float(initial_joints[7]),"panda_finger_joint2":float(initial_joints[8])}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Exp1Franka")):
        if prim.GetName() not in values or not prim.IsA(UsdPhysics.Joint):continue
        kind="angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else None
        if kind is None:raise RuntimeError("robot_joint_initialization_type_mismatch")
        q=values[prim.GetName()];authored=math.degrees(q) if kind=="angular" else q
        state_api=PhysxSchema.JointStateAPI.Apply(prim,kind)
        state_api.CreatePositionAttr().Set(authored);state_api.CreateVelocityAttr().Set(0.)
        drive=UsdPhysics.DriveAPI(prim,kind)
        if drive:drive.CreateTargetPositionAttr().Set(authored)
        robot_joint_initialization.append({"joint":str(prim.GetPath()),"type":kind,"canonical_initial_position":q,"authored_initial_position":authored})
    if len(robot_joint_initialization)!=9:raise RuntimeError("robot_joint_initialization_count_mismatch")
    write_json(output/"robot_prephysics_joint_initialization.json",{"joints":robot_joint_initialization,"robot_only":True,"pre_first_physics_step":True,"edit_target":"anonymous_session_layer","object_state_written":False})
    # Articulated task state is also part of the frozen reset contract.  The
    # catalog may author a non-zero JointState value which PhysX reapplies on
    # reset, overriding a late articulation-view write.  Author the requested
    # initial value in the anonymous session layer *before* the first reset so
    # the same state is reconstructed deterministically.  This is setup-only:
    # no object joint target is ever sent after world.play().
    object_joint_initialization=[]
    if stratum == "articulated":
        movable_for_prephysics=[j for j in joints if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)]
        if int(args.articulated_joint_index) >= len(movable_for_prephysics):
            raise RuntimeError("articulated_joint_index_out_of_range_prephysics")
        for movable_index, prim in enumerate(movable_for_prephysics):
            kind="angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
            # Preserve authored values for non-target joints.  The target
            # value is explicitly registered by the caller's TaskSpec.
            q_value=None
            if movable_index == int(args.articulated_joint_index):
                q_value=float(args.articulated_initial_joint_rad)
            if q_value is None:
                raw_attr=prim.GetAttribute(f"state:position:{kind}")
                raw=raw_attr.Get() if raw_attr.IsValid() else None
                q_value=math.radians(float(raw)) if kind == "angular" and raw is not None else float(raw or 0.0)
            authored_value=math.degrees(q_value) if kind == "angular" else q_value
            state_api=PhysxSchema.JointStateAPI.Apply(prim,kind)
            state_api.CreatePositionAttr().Set(float(authored_value))
            state_api.CreateVelocityAttr().Set(0.0)
            drive=UsdPhysics.DriveAPI(prim,kind)
            if drive:
                drive.CreateTargetPositionAttr().Set(float(authored_value))
            object_joint_initialization.append({"joint":str(prim.GetPath()),"type":kind,"index":movable_index,"canonical_initial_position":q_value,"authored_initial_position":authored_value})
    write_json(output/"object_prephysics_joint_initialization.json",{"joints":object_joint_initialization,"object_state_written":bool(object_joint_initialization),"pre_first_physics_step":True,"edit_target":"anonymous_session_layer"})
    startup_snapshot("before_reset")
    world.reset()
    startup_snapshot("immediately_after_reset")
    # Complete robot reset BEFORE any camera warmup physics steps; previously
    # the root/posture correction occurred only after camera warmup.
    franka.set_world_pose(position=base_position,orientation=np.asarray([1.,0.,0.,0.]))
    franka.set_joint_positions(initial_joints)
    franka.apply_action(ArticulationAction(joint_positions=initial_joints))
    write_json(output/"robot_reset_boundary.json",{"phase":"post_reset_before_warmup","robot_pose":list(base_position),"robot_joints":initial_joints.tolist(),"object_state_written":False})
    # PhysX reconstructs articulated link poses during reset.  Re-apply the
    # exact same pre-registered session-layer root placement after reset and
    # before play so every link starts in the recorded workcell frame.  This
    # is still setup-time authoring; the post-play path below never writes the
    # target object's transform, velocity, or joint state.
    # Capture the post-reset authored root frame before applying setup.  The
    # modern articulation view expects a world-space root pose.  Articulated
    # roots are solved from an authored joint pivot and the aggregate
    # collision bottom, never from a guessed visual replacement.
    root_world_after_reset = _position(omni, object_root, np)
    target_root_world = root_world_after_reset + np.asarray(placement_delta, dtype=np.float64)
    articulated_initial_joint_positions_setup = None
    articulated_initial_joint_error = None
    if stratum != "articulated":
        UsdGeom.XformCommonAPI(object_root).SetTranslate(Gf.Vec3d(*[float(x) for x in placement_delta]))
    if stratum == "articulated":
        # PhysX has now reconstructed each link from its authored joint frame.
        # Re-measure the actuated link in that post-reset state and compose a
        # single deterministic root-frame correction.  This is setup-only,
        # before ``world.play``; it is not an episode teleport and is never
        # repeated in the control loop.
        if target_articulation is not None:
            try:
                current_root = np.asarray(target_articulation.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
                movable_for_setup = [j for j in joints if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)]
                if int(args.articulated_joint_index) >= len(movable_for_setup):
                    raise RuntimeError(
                        f"articulated_joint_index={args.articulated_joint_index} "
                        f"outside movable joint count={len(movable_for_setup)}"
                    )
                setup_joint = movable_for_setup[int(args.articulated_joint_index)] if movable_for_setup else None
                # Several catalog articulated panels are authored lying flat
                # (thin Z extent) even though their revolute axis describes a
                # door/lid.  Apply the *registered* orientation first, then
                # solve translation from the resulting authored pivot and
                # collision bounds.  Both writes are before play.
                if args.articulated_orientation == "upright_x90":
                    setup_orientation = np.asarray(
                        [[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]],
                        dtype=np.float32,
                    )
                elif args.articulated_orientation == "upright_y90":
                    setup_orientation = np.asarray(
                        [[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]],
                        dtype=np.float32,
                    )
                elif args.articulated_orientation == "z90":
                    setup_orientation = np.asarray(
                        [[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]],
                        dtype=np.float32,
                    )
                elif args.articulated_orientation == "z180":
                    setup_orientation = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                elif args.articulated_orientation == "zminus90":
                    setup_orientation = np.asarray(
                        [[math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]],
                        dtype=np.float32,
                    )
                else:
                    setup_orientation = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                yaw_deg = float(args.articulated_workcell_yaw_deg)
                if not math.isfinite(yaw_deg) or not -180. <= yaw_deg <= 180.:
                    raise RuntimeError("invalid_registered_workcell_yaw")
                cy, sy = math.cos(math.radians(yaw_deg)/2), math.sin(math.radians(yaw_deg)/2)
                qw,qx,qy,qz = setup_orientation.reshape(-1)
                setup_orientation = np.asarray([[cy*qw-sy*qz,cy*qx-sy*qy,cy*qy+sy*qx,cy*qz+sy*qw]],dtype=np.float32)
                # The articulation root body is not necessarily the asset frame.
                # Preserve its SOURCE fixed-joint rotation instead of replacing it.
                source_anchors=[j for j in joints if j.IsA(UsdPhysics.FixedJoint) and not UsdPhysics.Joint(j).GetBody0Rel().GetTargets() and UsdPhysics.Joint(j).GetBody1Rel().GetTargets()]
                if len(source_anchors)!=1:raise RuntimeError("unique_authored_fixed_root_frame_missing")
                anchor=UsdPhysics.Joint(source_anchors[0])
                def values(q):return [float(q.GetReal()),*[float(v) for v in q.GetImaginary()]]
                asset_frame=omni.usd.get_world_transform_matrix(object_root).ExtractRotation().GetQuat()
                from i2ia.tasks.workcell_root_frame import compose
                workcell_delta=setup_orientation.reshape(-1).tolist()
                source_frame0=values(anchor.GetLocalRot0Attr().Get());source_frame1=values(anchor.GetLocalRot1Attr().Get())
                setup_orientation=np.asarray([compose(workcell_delta,values(asset_frame),source_frame0,source_frame1)],dtype=np.float32)
                write_json(output/"authored_root_frame_composition.json",{"root_joint":str(source_anchors[0].GetPath()),"workcell_delta_wxyz":workcell_delta,"asset_frame_wxyz":values(asset_frame),"source_localRot0_wxyz":source_frame0,"source_localRot1_wxyz":source_frame1,"composed_root_orientation_wxyz":setup_orientation.tolist(),"pre_episode_reset_only":True,"object_motion_during_episode_written":False})
                target_articulation.set_world_poses(
                    positions=np.asarray([target_root_world], dtype=np.float32),
                    orientations=setup_orientation,
                    usd=True,
                )
                for _ in range(3):
                    app.update()
                if setup_joint is not None:
                    pivot_after_orientation, _ = _joint_axis_and_pivot(stage, setup_joint, omni, Gf, UsdPhysics, np)
                    aggregate_lo, aggregate_hi, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
                    # X/Y anchor the actual authored pivot in the fixed
                    # Franka workcell. Z is derived from the collision bottom
                    # so gravity has a legitimate support plane.
                    pivot_slot_xy = np.asarray([float(args.articulated_pivot_x), float(args.articulated_pivot_y)], dtype=np.float64)
                    root_correction = np.asarray(
                        [pivot_slot_xy[0] - pivot_after_orientation[0],
                         pivot_slot_xy[1] - pivot_after_orientation[1],
                         (table_top + placement_clearance_m) - float(aggregate_lo[2])],
                        dtype=np.float64,
                    )
                    current_root = np.asarray(target_articulation.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
                    target_root_world = current_root + root_correction
                    target_articulation.set_world_poses(
                        positions=np.asarray([target_root_world], dtype=np.float32),
                        orientations=setup_orientation,
                        usd=True,
                    )
                    for _ in range(3):
                        app.update()
                    # A joint's initial state is part of the frozen reset
                    # configuration, not an episode-time control action.  It
                    # is applied once after reset and before ``world.play`` so
                    # an articulated panel can begin in a physically
                    # reachable open/closed pose.  The control loop below
                    # never writes this articulation's joint target.
                    try:
                        q_setup = np.asarray(target_articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
                        if q_setup.size:
                            if setup_joint.IsA(UsdPhysics.RevoluteJoint):
                                q_setup[int(args.articulated_joint_index)] = float(args.articulated_initial_joint_rad)
                            elif setup_joint.IsA(UsdPhysics.PrismaticJoint):
                                # The CLI value is an SI displacement for a
                                # prismatic joint; this runner's registered
                                # assets currently use revolute joints.
                                q_setup[int(args.articulated_joint_index)] = float(args.articulated_initial_joint_rad)
                            q_setup = np.asarray(q_setup, dtype=np.float32)
                            target_articulation.set_joint_positions(q_setup.reshape(1, -1))
                            target_articulation.set_joint_velocities(np.zeros_like(q_setup).reshape(1, -1))
                            for _ in range(3):
                                app.update()
                            articulated_initial_joint_positions_setup = q_setup.astype(np.float64).tolist()
                    except Exception as exc:
                        articulated_initial_joint_error = f"initial_joint_setup_failed:{type(exc).__name__}:{exc}"
                else:
                    root_correction = np.zeros(3, dtype=np.float64)
                articulated_setup_correction = root_correction.tolist()
            except Exception as exc:
                articulated_setup_error = f"articulation_root_setup_failed:{type(exc).__name__}:{exc}"
                articulated_setup_correction = None
            else:
                articulated_setup_error = None
        else:
            articulated_setup_error = None
            articulated_setup_correction = None
    else:
        articulated_setup_error = None
        articulated_setup_correction = None
    for _ in range(2):
        app.update()
    # The articulation root setup above changes every child link's world
    # bounds.  Re-read them before constructing the camera and controller so
    # reachability, support and task geometry are based on the actual
    # pre-play frame rather than the authored catalog frame.
    placed_min, placed_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "visual_")
    placed_collision_min, placed_collision_max, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
    if stratum == "articulated":
        desired_center = (np.asarray(placed_min, dtype=np.float64) + np.asarray(placed_max, dtype=np.float64)) / 2.0
        extent = np.asarray(placed_max, dtype=np.float64) - np.asarray(placed_min, dtype=np.float64)
        # Aggregate articulated bounds can be far from the first actuated
        # link (for example a laptop hinge plus a long panel).  Reachability
        # is therefore decided by the geometry-derived contact waypoint and
        # Lula's finite IK result below, not by the aggregate center.  This
        # gate only rejects degenerate or clearly unbounded assets.
        workspace_ok = bool(float(np.min(extent)) >= 0.002 and np.all(np.isfinite(desired_center)) and float(desired_center[2]) <= 2.5)
    # Reassert both registered camera poses after World.reset has reset scene
    # objects; otherwise Isaac 6 returns an identity-view camera buffer.
    camera.set_world_pose(position=camera_position, orientation=_look_at(camera_position, camera_target, np, rot_utils), camera_axes="world")
    evidence_camera.set_world_pose(position=evidence_position, orientation=_look_at(evidence_position, camera_target, np, rot_utils), camera_axes="world")
    camera.set_focal_length(2.2)
    evidence_camera.set_focal_length(2.2)
    # Isaac cameras must be initialized after reset and warmed with rendered
    # steps before the first RGB is accepted as evidence.
    try:
        camera.initialize()
        evidence_camera.initialize()
    except Exception as exc:
        raise RuntimeError(f"Isaac camera initialization failed: {exc}") from exc
    startup_snapshot("before_camera_warmup")
    for _ in range(8):
        world.step(render=True)
        startup_snapshot("camera_warmup_"+str(_))
    # A real RGB product is a hard admission gate for this trajectory track.
    # In CPU-only/container sessions Isaac can construct the Camera object but
    # Hydra has no GPU foundation; ``get_rgb`` then raises a low-level TypeError
    # (or returns None).  Fail with an explicit, auditable reason before any
    # episode action is sent rather than creating a state-trace-only artifact.
    try:
        camera_probe_rgb = camera.get_rgb()
    except Exception as exc:
        raise RuntimeError(f"camera_capture_unavailable_raw_isaac_rgb:{type(exc).__name__}:{exc}") from exc
    if camera_probe_rgb is None or getattr(camera_probe_rgb, "size", 0) == 0:
        raise RuntimeError("camera_capture_unavailable_raw_isaac_rgb:empty_frame")
    if franka is not None:
        franka.set_world_pose(position=base_position, orientation=np.asarray([1.0, 0.0, 0.0, 0.0]))
        franka.set_joint_positions(initial_joints)
        franka.apply_action(ArticulationAction(joint_positions=initial_joints))
        solver = KinematicsSolver(franka, end_effector_frame_name="right_gripper")
        solver.get_kinematics_solver().set_robot_base_pose(base_position, np.asarray([1.0, 0.0, 0.0, 0.0]))
        if args.controller == "rmpflow":
            rmpflow_controller = RMPFlowController(
                name="exp3_franka_real_contact_rmpflow",
                robot_articulation=franka,
                physics_dt=1.0 / args.fps,
            )
    while is_stage_loading():
        app.update()
    # Contact reporting is enabled before play and remains observation-only.
    # Apply the report API to both the body and its collision shapes (and the
    # validation support) so a resting contact is observable even when the
    # body goes to sleep immediately after gravity settle.
    # Retain both a bounded sample and counters so a resting object cannot fill
    # the sample buffer and hide a later robot contact.
    report_prims = [target_body, *collision_prims, stage.GetPrimAtPath("/World/Exp1Table"), *robot_report_bodies]
    for report_prim in report_prims:
        if report_prim and report_prim.IsValid() and not report_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(report_prim)
        if report_prim and report_prim.IsValid():
            PhysxSchema.PhysxContactReportAPI(report_prim).CreateThresholdAttr().Set(0.0)
    complete_contacts = []
    contact_events: list[dict[str, Any]] = []
    actuated_child_contact_event_samples: list[dict[str, Any]] = []
    contact_event_header_count_total = 0
    contact_event_count_total = 0
    robot_contact_event_count_total = 0
    actuated_child_contact_event_count_total = 0
    # This path is assigned from the authored first movable joint before the
    # Franka trajectory starts.  The callback reads it only as an observation
    # filter; it never authors a target link state.
    actuated_child_path: str | None = None
    maximum_interpenetration_m = 0.0
    maximum_settle_interpenetration_m = 0.0
    robot_prefix = "/World/Exp1Franka"
    current_phase = "settle"

    def on_contact(headers: Any, data: Any) -> None:
        nonlocal contact_event_count_total
        nonlocal contact_event_header_count_total
        nonlocal robot_contact_event_count_total
        nonlocal actuated_child_contact_event_count_total
        nonlocal maximum_interpenetration_m
        nonlocal maximum_settle_interpenetration_m
        for header in headers:
            paths = [
                str(PhysicsSchemaTools.intToSdfPath(value))
                for value in (header.actor0, header.actor1, header.collider0, header.collider1)
            ]
            if not any(str(object_root.GetPath()) in path or path.startswith(robot_prefix+"/") for path in paths):
                continue
            contact_event_header_count_total += 1
            start = int(header.contact_data_offset)
            end = start + int(header.num_contact_data)
            samples: list[dict[str, Any]] = []
            event_penetration = 0.0
            # A zero-data PhysX header is only a notification; without a
            # sample it is not enough evidence of physical contact.  The
            # geometric support witness below handles already-sleeping bodies.
            physical = False
            for index in range(start, end):
                try:
                    sample = data[index]
                    separation = float(sample.separation)
                except (IndexError, TypeError, ValueError, AttributeError):
                    continue
                event_penetration = max(event_penetration, max(0.0, -separation))
                physical = physical or separation <= 0.001
                if len(samples) < 8:
                    samples.append(
                        {
                            "position": [float(item) for item in sample.position],
                            "normal": [float(item) for item in sample.normal],
                            "impulse": [float(item) for item in sample.impulse],
                            "separation": separation,
                        }
                    )
            # Preserve raw contact positions/impulses at each physics tick.
            complete_contacts.append({"paths": paths, "samples": samples, "maximum_interpenetration_m": event_penetration, "event_type": getattr(header.type, "name", str(header.type).split(".")[-1])})
            if not physical or not any(str(object_root.GetPath()) in path for path in paths):
                continue
            contact_event_count_total += 1
            is_robot_contact = any(path.startswith(robot_prefix) for path in paths)
            if is_robot_contact and current_phase.startswith("robot_"):
                robot_contact_event_count_total += 1
            is_actuated_child_contact = bool(
                is_robot_contact
                and current_phase.startswith("robot_")
                and actuated_child_path
                and any(
                    path == actuated_child_path or path.startswith(actuated_child_path + "/")
                    for path in paths
                )
            )
            if is_actuated_child_contact:
                actuated_child_contact_event_count_total += 1
            maximum_interpenetration_m = max(maximum_interpenetration_m, event_penetration)
            if current_phase == "settle":
                maximum_settle_interpenetration_m = max(maximum_settle_interpenetration_m, event_penetration)
            if len(contact_events) < 2048:
                contact_events.append(
                    {
                        "phase": current_phase,
                        "paths": paths,
                        "num_contact_data": int(header.num_contact_data),
                        "is_robot_contact": is_robot_contact,
                        "is_actuated_child_contact": is_actuated_child_contact,
                        "maximum_interpenetration_m": event_penetration,
                        "samples": samples,
                    }
                )
            if is_actuated_child_contact and len(actuated_child_contact_event_samples) < 32:
                actuated_child_contact_event_samples.append(
                    {
                        "phase": current_phase,
                        "paths": paths,
                        "num_contact_data": int(header.num_contact_data),
                        "maximum_interpenetration_m": event_penetration,
                        "samples": samples,
                    }
                )

    # The primary artifact is a raw Isaac RGB stream.  It is opened before
    # play so reset/settle/control are part of one continuous recording.
    writer = imageio.get_writer(output / "rollout_raw.mp4", fps=args.fps, codec="libx264", quality=7, macro_block_size=None, ffmpeg_log_level="error", output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    evidence_writer=imageio.get_writer(output/"evidence_raw.mp4",fps=args.fps,codec="libx264",quality=7,macro_block_size=None,ffmpeg_log_level="error")
    synchronized_frame_times=[]
    from i2ia.tasks.raw_camera_framing import LiveFramingCapture
    framing_capture = LiveFramingCapture(stage, "/World/Exp1Franka", geometry_roots,
        [("policy", camera), ("evidence", evidence_camera)], Usd, UsdGeom, Gf, omni)
    write_json(output / "contact_instrumentation.json", {
        "scope": "all_robot_bodies_object_support",
        "robot_report_paths": [str(p.GetPath()) for p in robot_report_bodies],
        "target_body": str(target_body.GetPath()),
        "support_path": "/World/Exp1Table",
        "threshold_ns": 0.0, "observation_only": True,
        "runner_sha256": sha256_file(Path(__file__).resolve())})
    raw_frame_count = 0
    raw_frame_nonblack = 0

    def append_raw_frame(frame: Any) -> None:
        nonlocal raw_frame_count, raw_frame_nonblack
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        raw = np.asarray(frame, dtype=np.uint8)
        if raw.ndim != 3 or raw.shape[-1] < 3:
            return
        raw = raw[..., :3].copy()
        side=np.asarray(evidence_camera.get_rgb())
        if side.ndim!=3 or side.shape[-1]<3 or not side.size:raise RuntimeError("evidence_camera_frame_missing")
        framing_capture.capture(raw_frame_count, float(world.current_time))
        writer.append_data(raw)
        evidence_writer.append_data(side[...,:3].astype(np.uint8))
        synchronized_frame_times.append({"frame":raw_frame_count,"simulation_time_s":float(world.current_time),"camera_capture_type":"raw_isaac_rgb","same_physics_render_tick":True})
        raw_frame_count += 1
        if float(np.mean(raw)) > 1.0 and int(np.max(raw)) > 8:
            raw_frame_nonblack += 1

    world.play()
    if rmpflow_controller is not None:
        rmpflow_controller.reset()
        support_obstacle = world.scene.get_object("exp1_table")
        if support_obstacle is None:
            raise RuntimeError("registered_support_obstacle_missing_no_fallback")
        rmpflow_controller.add_obstacle(support_obstacle, static=True)
        write_json(output / "teacher_motion_policy_contract.json", {
            "teacher_id": "Franka_official_RMPflow_support_avoidance_v0_1",
            "prediction_dt_s": 1.0 / args.fps,
            "action_control_hz": args.fps,
            "physics_hz": args.physics_hz,
            "registered_obstacles": ["/World/Exp1Table"],
            "target_asset_not_obstacle": "explicit permitted robot-object contact, not a hidden fallback",
            "official_rmpflow_config": rmpflow_controller.rmp_flow_config,
            "teacher_is_not_vla": True,
            "object_commands": False
        })
    # Keep the handle explicit so exception paths cannot accidentally claim a
    # contact subscription was installed or tear down an uninitialised value.
    subscription = None
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    trace: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    observation_dir = output / "observations"
    if args.save_observations == "true":
        observation_dir.mkdir(parents=True, exist_ok=True)
    # Render warm-up and settle.  Track the target body's world position and
    # visual bottom to detect fall-through/instability.
    settle_positions: list[list[float]] = []
    settle_bottoms: list[float] = []
    settle_collision_bottoms: list[float] = []
    startup_snapshot("before_settle")
    for step in range(max(1, args.settle_steps)):
        world.step(render=True)
        startup_snapshot("settle_"+str(step))
        # Reset/warmup excluded: recording starts at the accepted episode state.
        body_pos = _position(omni, target_body, np)
        settle_positions.append(body_pos.tolist())
        try:
            lo, _, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "visual_")
            settle_bottoms.append(float(lo[2]))
            collision_lo, _, _ = _mesh_bounds_for_roots(stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_")
            settle_collision_bottoms.append(float(collision_lo[2]))
        except Exception:
            settle_bottoms.append(float("nan"))
            settle_collision_bottoms.append(float("nan"))
    startup_subscription=None
    if solver is not None:
        initial_ee, initial_rot = solver.compute_end_effector_pose()
        initial_ee = np.asarray(initial_ee, dtype=np.float64)
        measured_initial_orientation = rot_utils.rot_matrices_to_quats(initial_rot)
        # A fixed, downward-facing wrist orientation is part of the frozen
        # controller protocol (the same orientation used by the prior UP-099
        # Isaac contract).  It is not learned or selected per case.
        initial_orientation = np.asarray(
            [-0.02950371262943879, 7.352838082381605e-09, 0.9995646707147464, 3.36666919075283e-10],
            dtype=np.float64,
        )
        if str(args.controller_orientation_wxyz).strip():
            try:
                parsed_orientation = np.asarray(
                    [float(item.strip()) for item in str(args.controller_orientation_wxyz).split(",")],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid_controller_orientation_wxyz:{exc}") from exc
            if parsed_orientation.shape != (4,) or not np.all(np.isfinite(parsed_orientation)):
                raise RuntimeError("invalid_controller_orientation_wxyz:expected_four_finite_values")
            norm = float(np.linalg.norm(parsed_orientation))
            if norm <= 1e-12:
                raise RuntimeError("invalid_controller_orientation_wxyz:zero_norm")
            initial_orientation = parsed_orientation / norm
    else:
        initial_ee = np.zeros(3, dtype=np.float64)
        measured_initial_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        initial_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    initial_body_position = _position(omni, target_body, np)
    setup_rgb = camera.get_rgb()
    if setup_rgb is None or getattr(setup_rgb, "size", 0) == 0:
        raise RuntimeError("camera returned no setup RGB")
    Image.fromarray(np.asarray(setup_rgb, dtype=np.uint8), mode="RGB").save(output / "setup_frame.png")
    append_raw_frame(setup_rgb)
    # Capture a small, deterministic multi-view set from the same loaded
    # asset.  These are camera-only changes in the session layer; the object
    # is never repositioned after play.  Keeping these views beside the
    # rollout makes the geometry evidence inspectable without treating a
    # single render similarity score as a physical or control result.
    front_pose = np.asarray(camera_position, dtype=np.float64)
    view_poses = {
        "view_front": front_pose,
        "view_side": desired_center + np.asarray([-2.0 * extent_norm, -0.35 * extent_norm, 1.05 * extent_norm + 0.42]),
        "view_top": desired_center + np.asarray([0.05 * extent_norm, -0.10 * extent_norm, 2.65 * extent_norm + 0.55]),
    }
    view_artifacts: dict[str, str] = {}
    camera.set_world_pose(position=front_pose, orientation=_look_at(front_pose, camera_target, np, rot_utils), camera_axes="world")

    # The writer was opened before play; all subsequent frames are raw camera
    # products and are never replaced with a derived state-trace animation.
    # Episode starts after frozen reset and before the first teacher action.
    complete_states = []
    complete_action = np.asarray(initial_joints,dtype=np.float64).tolist()
    complete_tick = 0
    complete_episode_time_origin=float(world.current_time)
    complete_original_step = world.step
    complete_last_center = None
    complete_last_quat = None
    from i2ia.tasks.support_contact_lifecycle import SupportContactLifecycle
    support_tracker = SupportContactLifecycle(str(target_body.GetPath()), "/World/Exp1Table", full_spec["maximum_contact_separation_m"])
    def complete_snapshot():
        nonlocal complete_last_center, complete_last_quat
        lo,hi,_ = _mesh_bounds_for_roots(stage,geometry_roots,omni,Gf,UsdGeom,np,"visual_")
        pos,quat = _world_pose(omni,target_body,np)
        center=(np.asarray(lo)+np.asarray(hi))*.5
        linear=(center-complete_last_center)*args.physics_hz if complete_last_center is not None else np.zeros(3)
        angular=0. if complete_last_quat is None else 2*math.acos(min(1.,abs(float(np.dot(quat,complete_last_quat)))))*args.physics_hz
        w,x,y,z=[float(v) for v in quat]
        rot=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        body_world_matrix=omni.usd.get_world_transform_matrix(target_body)
        world_to_body=body_world_matrix.GetInverse()
        from i2ia.tasks.fixture_contact_guard_v0_1 import forbidden_fixture_contact
        fixture_violations=forbidden_fixture_contact(complete_contacts,full_spec)
        contacts=[];forbidden=bool(fixture_violations);penetration=0.;support=False
        for event in complete_contacts:
            paths=event["paths"];penetration=max(penetration,event["maximum_interpenetration_m"])
            if any("Exp1Table" in p for p in paths) and any(p==str(target_body.GetPath()) for p in paths):
                support=support or any(c["separation"]<=full_spec["maximum_contact_separation_m"] for c in event["samples"])
            robot_links=[p for p in paths[:2] if p.startswith("/World/Exp1Franka/")]
            if robot_links:
                rl=robot_links[0];actor_pair=paths[:2]
                actual_contact=any(c["separation"]<=full_spec["maximum_contact_separation_m"] and np.linalg.norm(c["impulse"])>=full_spec["minimum_contact_impulse_ns"] for c in event["samples"])
                legal_pair=rl in full_spec["allowed_robot_contact_links"] and str(target_body.GetPath()) in actor_pair and len(robot_links)==1
                forbidden=forbidden or (actual_contact and not legal_pair)
                # Never relabel a cabinet/table/self contact as target-handle contact.
                if str(target_body.GetPath()) in actor_pair:
                    for c in event["samples"]:
                        contacts.append({"object_link":str(target_body.GetPath()),"robot_link":rl,"point_world_m":c["position"],"point_link_local_m":list(world_to_body.Transform(Gf.Vec3d(*[float(v) for v in c["position"]]))),"impulse_world_ns":c["impulse"],"separation_m":c["separation"]})
        sleeping_readback = None
        if stratum == "rigid":
            sleeping_readback = bool(get_physx_simulation_interface().is_sleeping(omni.usd.get_context().get_stage_id(), PhysicsSchemaTools.sdfPathToInt(target_body.GetPath())))
        support_proof = support_tracker.observe(complete_contacts, pos.tolist(), quat.tolist(), sleeping_readback, float(world.current_time)-complete_episode_time_origin)
        support = support_proof["support_contact"]
        complete_states.append({"target_body_world_matrix_row_major": [list(row) for row in body_world_matrix], "contact_coordinate_convention": "usd_row_vector_world_to_link_inverse_including_uniform_scale", "support_lifecycle": support_proof, "timestamp_s":float(world.current_time)-complete_episode_time_origin,"simulation_time_s":float(world.current_time),"object_bounds_world_m":{"min":list(lo),"max":list(hi)},"object_position_world_m":pos.tolist(),"object_orientation_wxyz":quat.tolist(),"object_linear_velocity_m_s":linear.tolist(),"object_angular_velocity_rad_s":[angular,0.,0.],"robot_proprio":np.asarray(franka.get_joint_positions()).tolist(),"robot_action":complete_action[:],"contacts":contacts,"raw_contacts":complete_contacts[:],"max_penetration_m":penetration,"forbidden_collision":forbidden,"support_contact":support,"joint_positions":read_task_joints(),"teacher_phase":current_phase})
        complete_contacts.clear();complete_last_center=center;complete_last_quat=quat.copy()
    robot_geometry=[]
    for part in ("panda_hand","panda_leftfinger","panda_rightfinger","panda_link6","panda_link7"):
        prim=stage.GetPrimAtPath("/World/Exp1Franka/"+part);vertices=[]
        for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if child.IsA(UsdGeom.Mesh):
                points=UsdGeom.Mesh(child).GetPointsAttr().Get() or []
                transform=omni.usd.get_world_transform_matrix(child)
                vertices.extend([list(transform.Transform(Gf.Vec3d(p))) for p in points])
        if vertices:
            arr=np.asarray(vertices);robot_geometry.append({"link":part,"min":arr.min(0).tolist(),"max":arr.max(0).tolist()})
    write_json(output/"robot_tool_frame_geometry.json",{"robot_mesh_world_bounds":robot_geometry,"initial_ee":initial_ee.tolist(),"initial_orientation":initial_orientation.tolist(),"diagnostic_only":True})
    def read_task_joints():
        if target_articulation is None:return {}
        q=np.asarray(target_articulation.get_joint_positions()).reshape(-1)
        return {path:float(q[target_articulation.get_dof_index(Path(path).name)]) for path in full_spec["joint_limits"]}
    complete_snapshot()
    def complete_step(*args_step,**kwargs_step):
        nonlocal complete_tick
        from i2ia.tasks.complete_task_clock import advance_one_sample
        requested_render=kwargs_step.get("render",args_step[0] if args_step else True)
        value=advance_one_sample(world,complete_original_step,requested_render,1./args.physics_hz)
        complete_tick+=1;complete_snapshot()
        return value
    world.step=complete_step
    robot_contact_before = len(contact_events)
    robot_ik_frames = 0
    robot_ik_failures: list[str] = []
    robot_tracking_failures: list[str] = []
    robot_start_body = initial_body_position.copy()
    robot_end_body = initial_body_position.copy()
    robot_task_end_body = initial_body_position.copy()
    post_robot_control_body = initial_body_position.copy()
    robot_joint_start: float | None = None
    robot_joint_end: float | None = None
    robot_phase_count = 0
    push_feedback_records = []
    push_feedback_hold_target = None
    movable: list[Any] = []
    tracked_child: Any | None = None
    tracked_child_start: Any | None = None
    tracked_child_end: Any | None = None
    tracked_child_quat_start: Any | None = None
    tracked_child_quat_end: Any | None = None
    articulated_target_debug: dict[str, Any] = {}
    action_frame_index = 0
    try:
        # Fail before any policy action if the physically settled reset state
        # already moved outside its frozen start interval.
        sys.path.insert(0,str(root/"src"))
        from i2ia.tasks.complete_robot import inside,in_region
        first=complete_states[0]
        if full_spec["task_type"]=="push_to_region":
            initial_center=[(x+y)/2 for x,y in zip(first["object_bounds_world_m"]["min"],first["object_bounds_world_m"]["max"])]
            start_ok=inside(initial_center[:2],full_spec["initial_center_range_world_m"]["min"],full_spec["initial_center_range_world_m"]["max"]);already=in_region(first,full_spec)
            initial_record={"initial_center":initial_center,"initial_range":full_spec["initial_center_range_world_m"]}
        else:
            if articulated_setup_error or articulated_initial_joint_error:raise RuntimeError("articulated_reset_configuration_failed_no_fallback")
            q=first["joint_positions"][full_spec["target_joint"]];lo,hi=full_spec["initial_joint_range"]
            epsilon=full_spec["joint_numeric_comparison_epsilon_rad"];start_ok=lo-epsilon<=q<=hi+epsilon;lo,hi=full_spec["target_joint_range"];already=lo-epsilon<=q<=hi+epsilon
            initial_record={"joint_position_rad":q,"initial_range":full_spec["initial_joint_range"]}
        write_json(output/"initial_admission.json",{**initial_record,"initial_valid":start_ok,"already_in_goal":already,"teacher_action_count":0,"policy_action_count":0})
        if not start_ok or already:raise RuntimeError("initial_state_invalid_before_policy_no_fallback")
        # The sole enabled branch is the registered neural robot-action
        # interface. Legacy teacher code is not an allowed runtime controller.
        if args.controller == "local_robot_policy":
            if franka is None:raise RuntimeError("robot_controller_unavailable_no_fallback")
            from exp3_local_robot_policy_client_v0_134 import Client
            from exp3_training_instruction_contract_v0_134 import instruction,release_evidence
            from i2ia.tasks.robot_joint_reference_rate_v0_1 import limit_target
            from i2ia.tasks.full_task_episode_termination_v0_1 import completed
            wording=instruction(full_spec)
            write_json(output/"policy_input_contract.json",dict(**wording,model_variant=args.model_variant,policy_identity=policy_identity,policy_identity_sha256=sha256_file(args.policy_identity),model_inputs=["raw_isaac_rgb","measured_robot_joint_proprioception","instruction"],object_state_in_policy_input=False,teacher_used=False,execute_action_chunk_prefix=4))
            client=Client(args.policy_socket,args.model_variant,policy_identity["adapter_sha256"])
            action_chunk=None;response=None;chunk_index=0;responses=[]
            try:
                for step in range(min(args.control_steps,int(full_spec["time_budget_s"]*args.fps))):
                    envelope={"asset_id":full_spec["asset_id"],"candidate_id":full_spec["candidate_id"],"asset_sha256":full_spec["asset_sha256"],"model_variant":args.model_variant,"states":complete_states,"fallback_used":False,"teleport_used":False,"post_play_transform_writeback":False,"direct_object_command":False,"invisible_attachment_used":False}
                    done,terminal_evaluation=completed(full_spec,envelope)
                    if done and not release_evidence(full_spec,complete_states)['passed']:done=False
                    fatal={"forbidden_collision","excessive_penetration","object_dropped","joint_out_of_limits","robot_action_out_of_range","nonfinite_action_or_state"}.intersection(terminal_evaluation.get("failure_reasons",[]))
                    if done or fatal:
                        current_phase="full_task_terminal" if done else "safety_failure_terminal"
                        write_json(output/"policy_terminal_condition.json",dict(complete=done,safety_failure=sorted(fatal),step=step,evaluation=terminal_evaluation,predicate_not_model_input=True))
                        break
                    current_phase="local_policy_control"
                    before_rgb=camera.get_rgb();proprio=np.asarray(franka.get_joint_positions(),dtype=float)
                    if before_rgb is None or not getattr(before_rgb,"size",0):raise RuntimeError("missing_raw_policy_rgb_no_fallback")
                    if action_chunk is None or chunk_index>=4:
                        action_chunk,response=client.query(before_rgb,proprio,wording["training_and_paired_evaluation_instruction"]);responses.append(response);chunk_index=0
                    raw_target=action_chunk[chunk_index];target=np.clip(raw_target,np.asarray(full_spec["action_limits"]["min"]),np.asarray(full_spec["action_limits"]["max"]))
                    limited,rate_report=limit_target(proprio[:7],target[:7],dt_s=1./args.fps,maximum_velocity_rad_s=.6)
                    target[:7]=limited;complete_action=target.tolist()
                    observation_path=observation_dir/f"frame_{action_frame_index:05d}.png"
                    Image.fromarray(np.asarray(before_rgb,dtype=np.uint8),mode="RGB").save(observation_path)
                    obs=dict(timing="before_robot_action",simulation_time_s=float(world.current_time),robot_proprio=proprio.tolist(),camera_capture_type="raw_isaac_rgb",object_state_in_policy_input=False,path=str(observation_path),sha256=sha256_file(observation_path))
                    record=dict(step=action_frame_index,timestamp_s=action_frame_index/args.fps,phase=current_phase,action=target[:7].tolist(),gripper_action=target[7:].tolist(),action_type="franka_joint_position_target",source=args.model_variant,observation=obs,chunk_index=chunk_index,policy_input_sha256=response["input_sha256"],policy_response_sha256=response["response_sha256"],raw_policy_joint_target=raw_target.tolist(),robot_reference_rate_report=rate_report,robot_action_bounds_clamped=not np.allclose(raw_target,target),fallback_used=False)
                    action_trace.append(record);action_frame_index+=1;chunk_index+=1
                    franka.apply_action(ArticulationAction(joint_positions=target[:7],joint_indices=np.arange(7)));franka.gripper.apply_action(ArticulationAction(joint_positions=target[7:]))
                    for substep in range(max(1,args.physics_hz//args.fps)):world.step(render=substep==max(1,args.physics_hz//args.fps)-1)
                    robot_end_body=_position(omni,target_body,np)
                    trace.append(dict(phase=current_phase,timestamp_s=float(world.current_time)-complete_episode_time_origin,action=target[:7].tolist(),gripper_action=target[7:].tolist(),robot_proprioception={"joint_position":np.asarray(franka.get_joint_positions()).tolist()},body_position=robot_end_body.tolist(),object_transform_writes_after_play=0,fallback_used=False,replay_used=False))
                    rgb=camera.get_rgb()
                    if rgb is None or not getattr(rgb,"size",0):raise RuntimeError("missing_continuous_raw_policy_video")
                    append_raw_frame(rgb)
                write_json(output/"policy_responses.json",dict(model_variant=args.model_variant,responses=responses,scripted_teacher_used=False,object_commands_sent=False))
            finally:
                client.close()
            robot_phase_count=1
        elif franka is not None and solver is not None:
            if stratum == "rigid":
                # Derive a contact line from authored collision vertices.  We
                # choose the lowest stable band on the side facing the fixed
                # Franka base, then push through the object's horizontal
                # center.  This handles tall/irregular rigid meshes (chairs,
                # displays) without category-specific dimensions or a visual
                # proxy.  The vertices are read-only; only Franka joint
                # targets are sent after ``world.play``.
                object_center = (placed_collision_min + placed_collision_max) / 2.0
                base_xy = np.asarray(base_position[:2], dtype=np.float64)
                center_xy = np.asarray(object_center[:2], dtype=np.float64)
                inward_xy = center_xy - base_xy
                inward_norm = float(np.linalg.norm(inward_xy))
                if inward_norm < 1e-8:
                    inward_xy = np.asarray([1.0, 0.0], dtype=np.float64)
                    inward_norm = 1.0
                inward_xy /= inward_norm
                try:
                    collision_points, _ = _mesh_points_world(
                        stage, geometry_roots, omni, Gf, UsdGeom, np, "collision_"
                    )
                    point_projection = collision_points[:, :2] @ inward_xy
                    side_limit = float(np.min(point_projection))
                    # Use the actual extreme side rather than the median of a
                    # broad band.  For a chair/table-like mesh the median can
                    # land on an interior cross-member that is farther than
                    # the first reachable surface from the Franka base.
                    side_band = collision_points[point_projection <= side_limit + 0.012]
                    min_z = float(np.min(collision_points[:, 2]))
                    max_z = float(np.max(collision_points[:, 2]))
                    height = max(0.001, max_z - min_z)
                    low_band = side_band[side_band[:, 2] <= min_z + max(0.12, 0.36 * height)]
                    selected_points = low_band if len(low_band) else side_band
                    # Pre-registered centerline surface contact, rather than a
                    # vertex-density-biased corner which introduces yaw torque.
                    triangle_vertices=[];triangle_faces=[]
                    for body_root in geometry_roots:
                        for prim in Usd.PrimRange(body_root):
                            if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith("collision_"):continue
                            mesh=UsdGeom.Mesh(prim);transform=omni.usd.get_world_transform_matrix(prim);pts=mesh.GetPointsAttr().Get() or [];offset=len(triangle_vertices)
                            triangle_vertices.extend([list(transform.Transform(Gf.Vec3d(p))) for p in pts]);counts=mesh.GetFaceVertexCountsAttr().Get() or [];indices=mesh.GetFaceVertexIndicesAttr().Get() or [];cursor=0
                            for count in counts:
                                face=list(indices[cursor:cursor+count]);cursor+=count
                                for j in range(1,count-1):triangle_faces.append([offset+face[0],offset+face[j],offset+face[j+1]])
                    sys.path.insert(0,str(root/"src"))
                    from i2ia.tasks.contact_geometry import first_surface_intersection
                    from i2ia.tasks.tool_envelope_contact import support_clear_contact_height
                    clearance_plan = support_clear_contact_height(min_z, max_z, table_top, float(initial_ee[2]), robot_geometry, .020)
                    if full_spec.get("rigid_contact_height_policy") == "source_forbidden_wrist_envelope_v0_1":
                        from i2ia.tasks.wrist_clear_push_height_v0_1 import choose
                        wrist_plan=choose(min_z,max_z,float(initial_ee[2]),robot_geometry,clearance_plan["contact_ray_height_m"])
                        clearance_plan.update(wrist_plan)
                    write_json(output / "tool_support_contact_clearance.json", clearance_plan)
                    ray_start=np.asarray([center_xy[0],center_xy[1],clearance_plan["contact_ray_height_m"]])-np.asarray([inward_xy[0],inward_xy[1],0.])*max(2.,float(np.linalg.norm(placed_collision_max-placed_collision_min))*2.)
                    hit=first_surface_intersection(triangle_vertices,triangle_faces,ray_start,[inward_xy[0],inward_xy[1],0.])
                    surface_point=np.asarray(hit["point"])
                    geometry_contact_method=hit["method"]
                    write_json(output/"push_surface_intersection.json",hit)
                except Exception as exc:
                    # A non-empty collision mesh is already a required gate;
                    # if vertex extraction fails, retain an explicit failure
                    # record instead of silently selecting a category proxy.
                    raise RuntimeError("collision_vertex_read_failed_no_fallback") from exc
                # Contact the side band rather than a point above a thin
                # object.  The lower bound keeps the fingertip clear of the
                # support plane; the upper cap is derived from the authored
                # collision height and is especially important for flat
                # assets such as keyboards.
                # Keep the actual intersection; no unobserved box or height proxy.
                # Keep the gripper envelope outside the measured surface by a
                # registered 25 mm standoff.  This is a robot trajectory
                # correction (not a relaxed penetration gate) and remains
                # derived from the source collision geometry.
                if not math.isfinite(args.rigid_contact_standoff) or not 0.0 <= args.rigid_contact_standoff <= 0.1:
                    raise RuntimeError("invalid_registered_rigid_contact_standoff")
                contact_point = surface_point - np.asarray([inward_xy[0], inward_xy[1], 0.0]) * args.rigid_contact_standoff
                # Keep the registered push inside the verified Franka pose
                # envelope.  The distance is still derived from the authored
                # collision extent, but capped at the largest distance for
                # which the fixed Lula protocol has a reachable endpoint.
                derived_push_distance = float(min(max(0.035, np.linalg.norm(placed_collision_max[:2] - placed_collision_min[:2]) * 0.08), 0.07))
                registered_push_distance = float(getattr(args, "rigid_push_distance", 0.0) or 0.0)
                if not math.isfinite(registered_push_distance) or registered_push_distance < 0.0 or registered_push_distance > 0.20:
                    raise RuntimeError("rigid_push_distance must be zero or in [0, 0.20] m")
                push_distance = registered_push_distance if registered_push_distance > 0.0 else derived_push_distance
                push_point = surface_point + np.asarray([inward_xy[0], inward_xy[1], 0.0]) * push_distance
                ee_z = float(contact_point[2])
                targets = [
                    ("robot_approach", np.asarray([contact_point[0] - inward_xy[0] * 0.14, contact_point[1] - inward_xy[1] * 0.14, ee_z + 0.12])),
                    ("robot_contact", contact_point.copy()),
                    ("robot_push", push_point.copy()),
                    # Hold at the measured push endpoint.  Raising the wrist
                    # after contact is not part of a push predicate and can
                    # make a valid surface contact unreachable for tall or
                    # irregular assets.  The endpoint itself remains a
                    # Franka-only joint target and is observed by PhysX.
                    ("robot_retreat_and_hold", push_point - np.asarray([inward_xy[0], inward_xy[1], 0.]) * .08 + np.asarray([0.,0.,.08])),
                ]
                articulated_target_debug["rigid_contact_geometry"] = {
                    "method": geometry_contact_method,
                    "surface_point_world_m": np.asarray(surface_point, dtype=np.float64).tolist(),
                    "contact_point_world_m": np.asarray(contact_point, dtype=np.float64).tolist(),
                    "push_point_world_m": np.asarray(push_point, dtype=np.float64).tolist(),
                    "inward_direction_xy": inward_xy.tolist(),
                    "push_distance_m": push_distance,
                    "derived_push_distance_m": derived_push_distance,
                    "registered_push_distance_m": registered_push_distance,
                }
                fingers = [0.04, 0.0, 0.0, 0.0]
                approach_contract = full_spec.get("rigid_contact_approach", "diagonal_v0_1")
                if approach_contract == "vertical_then_horizontal_v0_1":
                    targets.insert(1, ("robot_lower_before_contact", contact_point-np.asarray([inward_xy[0],inward_xy[1],0.])*.14))
                    fingers.insert(1, .04)
                elif approach_contract != "diagonal_v0_1":
                    raise RuntimeError("unknown_registered_rigid_approach_protocol")
            else:
                # Source-matched handle mesh and authored joint, no proxy part.
                joint=stage.GetPrimAtPath(full_spec["target_joint"])
                if not (joint.IsA(UsdPhysics.RevoluteJoint) or joint.IsA(UsdPhysics.PrismaticJoint)):
                    raise RuntimeError("unsupported_full_open_joint_type_no_fallback")
                child=_child_for_joint(stage,joint,target_body,UsdPhysics)
                if str(child.GetPath())!=full_spec["target_link"]:raise RuntimeError("target_link_identity_mismatch")
                tracked_child=child;actuated_child_path=str(child.GetPath())
                tracked_child_start,tracked_child_quat_start=_world_pose(omni,child,np)
                pivot,axis=_joint_axis_and_pivot(stage,joint,omni,Gf,UsdPhysics,np)
                points=np.asarray(full_spec["handle_points_canonical_local_m"])
                matrix=omni.usd.get_world_transform_matrix(child)
                world_points=np.asarray([list(matrix.Transform(Gf.Vec3d(*[float(v) for v in p]))) for p in points])
                lo,hi,_=_mesh_bounds(stage,child,omni,Gf,UsdGeom,np,"visual_")
                from i2ia.tasks.handle_teacher_release_geometry_v0_1 import handle_open_waypoints
                if joint.IsA(UsdPhysics.PrismaticJoint):
                    from i2ia.tasks.prismatic_handle_teacher_v0_1 import handle_open_waypoints
                panel_world=np.asarray([list(matrix.Transform(Gf.Vec3d(*[float(v) for v in p]))) for p in full_spec["panel_points_canonical_local_m"]])
                plan=handle_open_waypoints(world_points,(np.asarray(lo)+np.asarray(hi))*.5,pivot,axis,first["joint_positions"][full_spec["target_joint"]],sum(full_spec["target_joint_range"])*.5,full_spec["teacher_handle_approach_m"],panel_points_world=panel_world,handle_triangles=full_spec.get("handle_triangles_canonical"))
                roll = float(full_spec.get("teacher_grasp_roll_deg", 0.))
                if roll not in (0., 180.):
                    raise RuntimeError("unregistered_gripper_symmetry_rotation")
                if roll:
                    from scipy.spatial.transform import Rotation
                    for waypoint in plan["waypoints"]:
                        w,x,y,z = waypoint["orientation_wxyz"]
                        q = (Rotation.from_quat([x,y,z,w])*Rotation.from_euler("z",roll,degrees=True)).as_quat()
                        waypoint["orientation_wxyz"] = [float(q[3]),*map(float,q[:3])]
                plan["teacher_grasp_roll_deg"] = roll
                write_json(output/"handle_robot_waypoints.json",plan)
                targets=[(p["phase"],np.asarray(p["position_m"])) for p in plan["waypoints"]]
                fingers=[p["finger_position_m"] for p in plan["waypoints"]]
                articulated_target_debug={"complete_handle_teacher":plan,"child_prim":str(child.GetPath()),"joint_prim":str(joint.GetPath()),"joint_positions_before_control":first["joint_positions"]}
            write_json(output/"registered_teacher_targets.json",{"targets":[{"phase":name,"position":np.asarray(point).tolist()} for name,point in targets],"debug":articulated_target_debug})
            full_task_terminal = False
            for target_index, (phase, target) in enumerate(targets):
                if stratum=="articulated":initial_orientation=np.asarray(plan["waypoints"][target_index]["orientation_wxyz"])
                target = np.asarray(target, dtype=np.float64)
                finger = float(fingers[min(target_index, len(fingers) - 1)])
                current_phase = phase
                start_ee, start_rotation = solver.compute_end_effector_pose()
                start_orientation = rot_utils.rot_matrices_to_quats(start_rotation)
                phase_orientation = initial_orientation.copy()
                start_ee = np.asarray(start_ee, dtype=np.float64)
                phase_steps = max(1, args.control_steps // len(targets))
                final_ik_target = start_ee.copy()
                finger_compensation_enabled = (
                    str(getattr(args, "finger_contact_compensation", "false")) == "true"
                    and stratum == "articulated"
                )
                phase_compensation_records: list[dict[str, Any]] = []
                for local in range(phase_steps):
                    # Episode termination is a frozen, common environment
                    # predicate, not a teacher fallback or a model input.
                    # Once the COMPLETE task and stable hold are achieved,
                    # do not send another motion command beyond the goal.
                    if full_spec.get("episode_termination_policy"):
                        from i2ia.tasks.full_task_episode_termination_v0_1 import completed
                        full_task_terminal, terminal_evaluation = completed(full_spec, {
                            "asset_id": full_spec["asset_id"],
                            "candidate_id": full_spec["candidate_id"],
                            "asset_sha256": full_spec["asset_sha256"],
                            "model_variant": "demonstration_teacher",
                            "states": complete_states,
                            "fallback_used": False, "teleport_used": False,
                            "post_play_transform_writeback": False,
                            "direct_object_command": False,
                            "invisible_attachment_used": False,
                        })
                        if full_task_terminal:
                            current_phase = "full_task_terminal"
                            write_json(output / "episode_terminal_evaluation.json", {
                                "policy": full_spec["episode_termination_policy"],
                                "evaluation": terminal_evaluation,
                                "next_teacher_action_not_sent": True,
                                "physics_and_media_independent_checks_still_required": True,
                            })
                            break
                    progress = (local + 1) / max(1, args.control_steps // len(targets))
                    progress = progress * progress * (3.0 - 2.0 * progress)
                    planned = start_ee + (target - start_ee) * progress
                    # Lula's end-effector frame is upstream of the two
                    # physical finger links.  When explicitly enabled for a
                    # registered articulated-contact run, close that known
                    # tool-envelope offset from the measured finger midpoint.
                    # The resulting command is still an arm joint target; no
                    # target-asset state is authored or inferred from physics.
                    command_target = planned.copy()
                    if stratum == "rigid" and full_spec.get("demonstration_push_feedback") is True:
                        if phase in ("robot_push", "robot_hold"):
                            from i2ia.tasks.bounded_push_feedback_v0_2 import waypoint
                            live_ee, _ = solver.compute_end_effector_pose()
                            observed = complete_states[-1]
                            feedback = waypoint(live_ee, observed["object_bounds_world_m"],
                                observed["object_linear_velocity_m_s"], full_spec["target_region_world_m"],
                                contact_point, push_point, previous_command=push_feedback_hold_target)
                            command_target = np.asarray(feedback["robot_ee_waypoint_m"])
                            push_feedback_hold_target = command_target.copy()
                            push_feedback_records.append({"physics_timestamp_s": observed["timestamp_s"], **feedback})
                    from i2ia.tasks.robot_pose_interpolation import slerp
                    command_orientation = slerp(start_orientation,phase_orientation,progress) if stratum=="articulated" else phase_orientation
                    compensation_offset = None
                    if finger_compensation_enabled:
                        try:
                            live_ee, _ = solver.compute_end_effector_pose()
                            live_ee = np.asarray(live_ee, dtype=np.float64)
                            live_fingers = _franka_contact_finger_position(
                                stage, omni, np, str(getattr(args, "finger_contact_side", "midpoint"))
                            )
                            if live_fingers is not None and np.all(np.isfinite(live_ee)):
                                offset = np.asarray(live_fingers, dtype=np.float64) - live_ee
                                # Refuse an implausible tool-frame estimate
                                # instead of silently turning it into a pose
                                # or category fallback.
                                if np.all(np.isfinite(offset)) and float(np.linalg.norm(offset)) <= 0.25:
                                    compensation_offset = offset
                                    command_target = planned - offset
                        except Exception as exc:
                            robot_tracking_failures.append(
                                f"{phase}:{local}:finger_compensation_read_error:{type(exc).__name__}"
                            )
                    phase_compensation_records.append(
                        {
                            "step": int(local),
                            "nominal_target_m": planned.tolist(),
                            "command_target_m": command_target.tolist(),
                            "command_orientation_wxyz": command_orientation.tolist(),
                            "offset_m": compensation_offset.tolist() if compensation_offset is not None else None,
                        }
                    )
                    try:
                        if rmpflow_controller is not None:
                            action = rmpflow_controller.forward(
                                target_end_effector_position=np.asarray(command_target, dtype=np.float64),
                                target_end_effector_orientation=np.asarray(command_orientation, dtype=np.float64),
                            )
                            ok = action is not None and getattr(action, "joint_positions", None) is not None
                        else:
                            action, ok = solver.compute_inverse_kinematics(
                                target_position=command_target,
                                target_orientation=None if args.position_only_ik == "true" else command_orientation,
                                position_tolerance=0.001,
                                orientation_tolerance=0.03,
                            )
                    except Exception as exc:
                        ok, action = False, None
                        robot_ik_failures.append(f"{phase}:{type(exc).__name__}:{exc}")
                    if not ok or action is None:
                        robot_ik_failures.append(f"{phase}:{local}:ik_failed")
                        raise RuntimeError("complete_task_teacher_ik_failed_no_fallback")
                    from i2ia.tasks.robot_action_limits_v0_1 import safe_position_targets
                    safe_arm, numerical_boundary = safe_position_targets(
                        action.joint_positions, full_spec["action_limits"]["min"][:7],
                        full_spec["action_limits"]["max"][:7])
                    from i2ia.tasks.robot_joint_reference_rate_v0_1 import limit_target
                    limited, rate_report = limit_target(
                        np.asarray(franka.get_joint_positions())[:7], safe_arm,
                        dt_s=1./args.fps, maximum_velocity_rad_s=.6)
                    action.joint_positions = limited
                    phase_compensation_records[-1]["robot_reference_rate_report"] = rate_report
                    phase_compensation_records[-1]["robot_action_boundary_adapter"] = numerical_boundary
                    final_ik_target = command_target.copy()
                    robot_ik_frames += 1
                    observation_metadata = {
                        "timing":"before_robot_action",
                        "simulation_time_s":float(world.current_time),
                        "robot_proprio":np.asarray(franka.get_joint_positions(),dtype=float).tolist(),
                        "camera_capture_type":"raw_isaac_rgb",
                        "object_state_in_policy_input":False,
                    }
                    if args.save_observations == "true":
                        before_rgb=camera.get_rgb()
                        if before_rgb is None or not getattr(before_rgb,"size",0):
                            raise RuntimeError("missing_pre_action_camera_rgb")
                        observation_path=observation_dir / f"frame_{action_frame_index:05d}.png"
                        Image.fromarray(np.asarray(before_rgb,dtype=np.uint8),mode="RGB").save(observation_path)
                        observation_metadata.update(path=str(observation_path),sha256=sha256_file(observation_path))
                    franka.apply_action(action)
                    franka.gripper.apply_action(ArticulationAction(joint_positions=np.asarray([finger, finger], dtype=np.float64)))
                    action_values = _finite_action_values(action, np)
                    complete_action = list(action_values or []) + [finger,finger]
                    if action_values is not None:
                        action_record_item = {
                            "step": action_frame_index,
                            "timestamp_s": float(action_frame_index) / float(max(1, args.fps)),
                            "phase": phase,
                            "action": action_values,
                            "gripper_action": [finger, finger],
                            "action_type": "franka_joint_position_target",
                            "source": "registered_" + args.controller + "_robot_demonstration_teacher",
                            "observation": observation_metadata,
                            "robot_tool_command_world_m":command_target.tolist(),
                            "robot_tool_orientation_wxyz":command_orientation.tolist(),
                        }
                        action_trace.append(action_record_item)
                        action_frame_index += 1
                    for substep in range(max(1, args.physics_hz // args.fps)):
                        world.step(render=substep == max(1, args.physics_hz // args.fps) - 1)
                    robot_end_body = _position(omni, target_body, np)
                    observed_ee, _ = solver.compute_end_effector_pose()
                    finger_positions = _franka_finger_world_positions(stage, omni, np)
                    finger_midpoint = (
                        _franka_contact_finger_position(
                            stage, omni, np, str(getattr(args, "finger_contact_side", "midpoint"))
                        )
                        if finger_compensation_enabled
                        else None
                    )
                    ee_metric = float(np.linalg.norm(np.asarray(observed_ee, dtype=np.float64) - target))
                    finger_metric = (
                        float(np.linalg.norm(np.asarray(finger_midpoint, dtype=np.float64) - target))
                        if finger_midpoint is not None
                        else None
                    )
                    metric = finger_metric if finger_metric is not None else ee_metric
                    trace_item = {"phase": phase, "frame": len(trace), "timestamp_s": float(len(trace)) / float(max(1, args.fps)), "body_position": robot_end_body.tolist(), "ee_position": np.asarray(observed_ee, dtype=np.float64).tolist(), "ik_target": target.tolist(), "ik_command_target": command_target.tolist(), "ik_error_m": metric, "ee_target_error_m": ee_metric, "finger_target_error_m": finger_metric, "finger_contact_compensation": bool(finger_compensation_enabled), "contact_event_count": len(contact_events), "object_transform_writes_after_play": 0, "fallback_used": False, "replay_used": False}
                    trace_item["franka_finger_world_positions_m"] = finger_positions
                    try:
                        trace_item["robot_proprioception"] = {
                            "joint_position": np.asarray(franka.get_joint_positions(), dtype=np.float64).reshape(-1).tolist(),
                            "joint_velocity": np.asarray(franka.get_joint_velocities(), dtype=np.float64).reshape(-1).tolist(),
                            "end_effector_position_m": np.asarray(observed_ee, dtype=np.float64).tolist(),
                        }
                    except Exception as exc:
                        trace_item["robot_proprioception_error"] = f"{type(exc).__name__}:{exc}"
                    if target_articulation is not None:
                        try:
                            trace_item["object_joint_state"] = {
                                "joint_position": np.asarray(target_articulation.get_joint_positions(), dtype=np.float64).reshape(-1).tolist(),
                                "joint_velocity": np.asarray(target_articulation.get_joint_velocities(), dtype=np.float64).reshape(-1).tolist(),
                            }
                        except Exception as exc:
                            trace_item["object_joint_state_error"] = f"{type(exc).__name__}:{exc}"
                    trace_item["contact_evidence"] = {
                        "robot_object_contact_event_count": robot_contact_event_count_total,
                        "robot_actuated_link_contact_event_count": actuated_child_contact_event_count_total,
                    }
                    if action_values is not None:
                        trace_item["action"] = action_values
                        trace_item["gripper_action"] = [finger, finger]
                    trace.append(trace_item)
                    rgb = camera.get_rgb()
                    if rgb is not None and getattr(rgb, "size", 0) > 0:
                        append_raw_frame(rgb)
                if full_task_terminal:
                    break
                # Do not count an IK solution as execution success unless the
                # measured EE actually reaches the commanded waypoint.  This
                # fail-closed tracking gate prevents a stalled controller
                # from being admitted merely because Lula returned targets.
                observed_phase_end, _ = solver.compute_end_effector_pose()
                if finger_compensation_enabled:
                    phase_finger_midpoint = _franka_contact_finger_position(
                        stage, omni, np, str(getattr(args, "finger_contact_side", "midpoint"))
                    )
                    tracking_error = (
                        float(np.linalg.norm(np.asarray(phase_finger_midpoint, dtype=np.float64) - np.asarray(target, dtype=np.float64)))
                        if phase_finger_midpoint is not None
                        else float("inf")
                    )
                else:
                    tracking_error = float(np.linalg.norm(np.asarray(observed_phase_end, dtype=np.float64) - np.asarray(target, dtype=np.float64)))
                if tracking_error > 0.055:
                    robot_tracking_failures.append(f"{phase}:tracking_error_m={tracking_error:.6f}")
                articulated_target_debug.setdefault("finger_contact_compensation", bool(finger_compensation_enabled))
                articulated_target_debug.setdefault(
                    "finger_contact_side", str(getattr(args, "finger_contact_side", "midpoint"))
                )
                if finger_compensation_enabled:
                    articulated_target_debug.setdefault("finger_compensation_records", []).extend(phase_compensation_records)
            robot_phase_count = len(targets)
            if tracked_child is not None:
                try:
                    tracked_child_end, tracked_child_quat_end = _world_pose(omni, tracked_child, np)
                except Exception:
                    tracked_child_end = None
                    tracked_child_quat_end = None
        else:
            current_phase = "robot_unavailable"
            for _ in range(max(1, args.fps)):
                world.step(render=True)
                rgb = camera.get_rgb()
                if rgb is not None and getattr(rgb, "size", 0) > 0:
                    append_raw_frame(rgb)
        # Freeze the endpoint of the Franka-only attempt.  There is no second
        # model-facing object-control interface in this runner.
        robot_task_end_body = _position(omni, target_body, np)
        control_before = robot_start_body.copy()
        control_after = robot_task_end_body.copy()
        control_displacement = control_after - control_before
        post_robot_control_body = _position(omni, target_body, np)
    finally:
        # Capture the final state from the canonical front camera before
        # stopping the world.  This is an observation artifact only.
        try:
            camera.set_world_pose(position=front_pose, orientation=_look_at(front_pose, camera_target, np, rot_utils), camera_axes="world")
            world.step(render=True)
            final_rgb = camera.get_rgb()
            if final_rgb is not None and getattr(final_rgb, "size", 0) > 0:
                append_raw_frame(final_rgb)
                Image.fromarray(np.asarray(final_rgb, dtype=np.uint8), mode="RGB").save(output / "final_frame.png")
        except Exception:
            # Preserve the primary physics/control result if a late camera
            # read fails; the missing artifact is recorded below.
            pass
        write_json(output / "demonstration_push_feedback.json", {"enabled": False, "records": [], "object_commands_sent": False, "teacher_used": False})
        writer.close()
        evidence_writer.close()
        write_json(output / "live_framing_trace.json", framing_capture.report())
        write_json(output/"synchronized_video_receipt.json",{"camera_capture_type":"raw_isaac_rgb","model_input_camera":"/World/Exp1Camera","evidence_camera_not_model_input":True,"frames":synchronized_frame_times,"motion_synthesized":False,"recording_starts_after_reset_before_first_teacher_action":True,"legacy_clock_key_applies_to_first_robot_action":True,"recording_starts_after_reset_before_first_policy_action":True})
        # Read all terminal state before stopping the simulation.  Stopping is
        # lifecycle teardown only and never participates in the task result.
        terminal_body_before_stop = _position(omni, target_body, np)
        world.step=complete_original_step
        write_json(output / "complete_task_raw_trace.json", {"asset_id":full_spec["asset_id"],"candidate_id":full_spec["candidate_id"],"asset_sha256":full_spec["asset_sha256"],"model_variant":args.model_variant,"states":complete_states,"fallback_used":False,"teleport_used":False,"post_play_transform_writeback":False,"direct_object_command":False,"invisible_attachment_used":False,"teacher_used":False})
        world.stop()
        if subscription is not None:
            del subscription
    # No USD/PhysX reads are performed after world.stop().
    final_body_position = terminal_body_before_stop
    settle_positions_np = np.asarray(settle_positions, dtype=np.float64)
    settle_drift = float(np.linalg.norm(settle_positions_np[-1] - settle_positions_np[max(0, len(settle_positions_np) - min(30, len(settle_positions_np)))])) if len(settle_positions_np) >= 2 else float("nan")
    min_bottom = float(np.nanmin(np.asarray(settle_bottoms, dtype=np.float64))) if settle_bottoms and np.isfinite(np.asarray(settle_bottoms, dtype=np.float64)).any() else float("nan")
    min_collision_bottom = float(np.nanmin(np.asarray(settle_collision_bottoms, dtype=np.float64))) if settle_collision_bottoms and np.isfinite(np.asarray(settle_collision_bottoms, dtype=np.float64)).any() else float("nan")
    robot_contact_events = [item for item in contact_events[robot_contact_before:] if any("Exp1Franka" in path for path in item.get("paths", []))]
    robot_displacement = robot_task_end_body - robot_start_body
    # Robot control is an optional, separately-scoped track.  The current
    # validation asks only for the Isaac/PhysX sim-ready gate, so a disabled
    # robot track is represented as not evaluated rather than as a failure.
    # ``sim_task_success`` is the Exp3 simulator predicate.  The required
    # top-level robot_task_success field is intentionally always null because
    # no physical robot was evaluated.
    sim_task_success: bool | None = None
    robot_failure = None
    if not robot_attempted:
        robot_failure = "not_evaluated_by_request"
    elif not workspace_ok:
        robot_failure = "mesh_extent_or_workspace_out_of_bounds"
    elif franka is None:
        robot_failure = "franka_not_constructed"
    elif robot_ik_failures:
        robot_failure = "ik_failures_without_fallback"
    elif robot_contact_event_count_total <= 0:
        robot_failure = "no_robot_object_contact_reported"
    elif stratum == "rigid":
        # The fixed task is a geometry-derived push.  Evaluate displacement
        # along the same measured inward direction used to place the contact
        # waypoint; a valid side push need not align with world X.
        push_debug = articulated_target_debug.get("rigid_contact_geometry", {})
        push_direction = np.asarray(push_debug.get("inward_direction_xy", [1.0, 0.0]), dtype=np.float64).reshape(-1)
        if push_direction.size != 2 or not np.all(np.isfinite(push_direction)) or float(np.linalg.norm(push_direction)) < 1e-8:
            push_direction = np.asarray([1.0, 0.0], dtype=np.float64)
        push_direction = push_direction / float(np.linalg.norm(push_direction))
        projected_displacement = float(np.dot(np.asarray(robot_displacement[:2], dtype=np.float64), push_direction))
        articulated_target_debug.setdefault("rigid_contact_geometry", {})["measured_projected_displacement_m"] = projected_displacement
        # The task contract registered for all rigid Exp3 assets is a
        # displacement of at least 15 mm after robot contact.  Do not derive
        # a larger threshold from object dimensions: that would change the
        # frozen task semantics for tall or wide assets.
        task_threshold_m = 0.015
        sim_task_success = bool(projected_displacement >= task_threshold_m)
        if not sim_task_success:
            robot_failure = "robot_waypoint_tracking_failed" if robot_tracking_failures else "contact_without_required_push_displacement"
    else:
        # A supplemental articulated success requires both simulator-reported
        # robot/object contact and measurable motion of the first actuated
        # child during the Franka phase.  The child transform is read-only
        # observation; it is never authored after play.
        child_displacement = None
        child_rotation = None
        if tracked_child_start is not None and tracked_child_end is not None:
            child_displacement = float(np.linalg.norm(tracked_child_end - tracked_child_start))
        if tracked_child_quat_start is not None and tracked_child_quat_end is not None:
            child_rotation = _quaternion_angle(tracked_child_quat_start, tracked_child_quat_end, np, math)
        if actuated_child_contact_event_count_total <= 0:
            sim_task_success = False
            robot_failure = "no_robot_actuated_link_contact_reported"
        elif robot_ik_failures:
            sim_task_success = False
            robot_failure = "ik_failures_without_fallback"
        elif (child_displacement is None and child_rotation is None) or ((child_displacement or 0.0) < 0.005 and (child_rotation or 0.0) < 0.035):
            sim_task_success = False
            robot_failure = "robot_waypoint_tracking_failed" if robot_tracking_failures else "contact_without_required_joint_motion"
        else:
            sim_task_success = True
            robot_failure = None
    body_finite = finite_tree(final_body_position.tolist()) and all(finite_tree(row) for row in trace)
    dynamic_records = [record for record in body_physics if record["motion_type"] == "dynamic"]
    dynamic_body_present = bool(dynamic_records)
    mass_ok = bool(dynamic_records) and all(
        record["mass_kg"] is not None
        and math.isfinite(float(record["mass_kg"]))
        and float(record["mass_kg"]) > 0.0
        for record in dynamic_records
    )
    center_of_mass_ok = bool(dynamic_records) and all(
        record["center_of_mass_values_m"] is not None
        and len(record["center_of_mass_values_m"]) == 3
        and all(math.isfinite(float(value)) for value in record["center_of_mass_values_m"])
        for record in dynamic_records
    )
    inertia_ok = bool(dynamic_records) and all(
        record["diagonal_inertia_values_kg_m2"] is not None
        and len(record["diagonal_inertia_values_kg_m2"]) == 3
        and all(math.isfinite(float(value)) and float(value) > 0.0 for value in record["diagonal_inertia_values_kg_m2"])
        for record in dynamic_records
    )
    collision_nonempty = bool(collision_prims) and all(
        bool(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []) for prim in collision_prims
    )
    collision_valid = bool(collision_prims) and collision_nonempty and all(
        bool(prim.HasAPI(UsdPhysics.CollisionAPI))
        and (_attr(prim, "physics:collisionEnabled") is not False)
        for prim in collision_prims
    )
    initial_penetration_ok = bool(
        math.isfinite(min_bottom)
        and min_bottom >= table_top - 0.06
        and maximum_settle_interpenetration_m <= 0.02
    )
    # Geometric support contact is an independent, simulator-agnostic witness
    # for the requested contact test.  It is deliberately not inferred from a
    # render score: the loaded collision mesh must finish within a small band
    # of the known support plane and overlap its finite footprint.
    support_xy_overlap = bool(
        placed_collision_max[0] >= -float(args.support_width) / 2.0
        and placed_collision_min[0] <= float(args.support_width) / 2.0
        and placed_collision_max[1] >= -float(args.support_depth) / 2.0
        and placed_collision_min[1] <= float(args.support_depth) / 2.0
    )
    geometric_support_contact = bool(
        math.isfinite(min_collision_bottom)
        and min_collision_bottom >= table_top - 0.06
        and min_collision_bottom <= table_top + 0.025
        and support_xy_overlap
    )
    contact_test_pass = bool(contact_event_count_total or geometric_support_contact)
    gravity_settle_ok = bool(math.isfinite(settle_drift) and settle_drift <= 0.08)
    physics_sanity = bool(
        dynamic_body_present
        and body_finite
        and mass_ok
        and center_of_mass_ok
        and inertia_ok
        and collision_valid
        and initial_penetration_ok
        and gravity_settle_ok
        and contact_test_pass
    )
    articulation_valid = (not joints and stratum == "rigid") or bool([j for j in joints if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)])
    controller_available = bool(case.get("candidate", {}).get("control_interface", {}).get("model_output_consumable") is True)
    format_load = {"usd": True, "urdf": False, "urdf_evaluation":"not_evaluated_metadata_reference_only", "mjcf": False, "isaac_stage_opened": True}
    physics_status = "passed" if physics_sanity else "failed"
    # Task success is a conjunction with the full simulator gate, not merely
    # motion/contact.  This prevents an unstable or penetrating asset from
    # entering the successful-trajectory dataset.
    if sim_task_success is True and not (physics_sanity and articulation_valid and format_load.get("isaac_stage_opened")):
        sim_task_success = False
        robot_failure = "task_motion_observed_but_sim_ready_gate_failed"
    robot_status = "simulated_robot_only"
    failure_reasons = []
    if not physics_sanity:
        failure_reasons.append("physics_sanity_failed")
        for check_name, passed in (
            ("dynamic_body_present", dynamic_body_present),
            ("mass_positive_finite", mass_ok),
            ("center_of_mass_finite", center_of_mass_ok),
            ("inertia_positive_finite", inertia_ok),
            ("collision_geometry_nonempty", collision_nonempty),
            ("initial_penetration", initial_penetration_ok),
            ("gravity_settle", gravity_settle_ok),
            ("contact_test", contact_test_pass),
        ):
            if not passed:
                failure_reasons.append(f"{check_name}_failed")
    if not articulation_valid:
        failure_reasons.append("articulation_schema_failed")
    if not controller_available:
        failure_reasons.append("controller_interface_missing")
    if robot_attempted and robot_failure:
        failure_reasons.append(robot_failure)
    # The simulator task predicate is evidence, not a substitute for the
    # physical-robot field.  It is retained under an Exp3-specific key.
    video_validation = {
        "path": str(output / "rollout_raw.mp4"),
        "camera_capture_type": "raw_isaac_rgb",
        "frame_count": raw_frame_count,
        "nonblack_frame_count": raw_frame_nonblack,
        "valid": bool(raw_frame_count > 1 and raw_frame_nonblack > 1),
    }
    try:
        reader = imageio.get_reader(output / "rollout_raw.mp4")
        decoded = 0
        decoded_nonblack = 0
        for frame in reader:
            decoded += 1
            if decoded <= 4 or decoded % 16 == 0:
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.size and float(np.mean(arr)) > 1.0 and int(np.max(arr)) > 8:
                    decoded_nonblack += 1
        reader.close()
        video_validation.update({"decoded_frame_count": decoded, "decoded_nonblack_sample_count": decoded_nonblack, "decode_ok": decoded > 1})
        video_validation["valid"] = bool(video_validation["valid"] and decoded > 1)
    except Exception as exc:
        video_validation.update({"decode_ok": False, "decode_error": f"{type(exc).__name__}:{exc}"})
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "case_id": case.get("case_id"),
        "subset_index": case.get("subset_index"),
        "candidate_id": case.get("candidate_id"),
        "requested_category": category,
        "candidate_category": case.get("candidate_category"),
        "interaction_stratum": stratum,
        "source_admitted": True,
        "asset_retrieved_or_generated": "retrieved_rank1_asset",
        "format_load": format_load,
        "collision_valid": collision_valid,
        "physical_sanity": physics_sanity,
        "articulation_valid": articulation_valid,
        "controller_available": controller_available,
        "robot_task_success": None,
        "sim_task_success": sim_task_success,
        "simulator_success_predicate": (
            "physics_gates_pass AND robot_contact_event_count>0 AND "
            "geometry_projected_displacement_m>=0.015"
            if stratum == "rigid"
            else "physics_gates_pass AND robot_actuated_link_contact_event_count>0 AND "
            "(actuated_child_displacement_m>=0.005 OR actuated_child_rotation_rad>=0.035)"
        ),
        "initial_state": {
            "target_body_position_m": initial_body_position.tolist(),
            "tracked_child_position_m": tracked_child_start.tolist() if tracked_child_start is not None else None,
            "tracked_child_orientation_wxyz": tracked_child_quat_start.tolist() if tracked_child_quat_start is not None else None,
        },
        "final_state": {
            "target_body_position_m": final_body_position.tolist(),
            "tracked_child_position_m": tracked_child_end.tolist() if tracked_child_end is not None else None,
            "tracked_child_orientation_wxyz": tracked_child_quat_end.tolist() if tracked_child_quat_end is not None else None,
        },
        "sim_robot_control_status": "simulated_only",
        "physics_validation_status": physics_status,
        "sim_ready_status": "sim_ready_only" if physics_sanity and articulation_valid and format_load.get("isaac_stage_opened") else "blocked",
        "robot_control_status": "simulated_robot_only",
        "robot_real_world": "not_evaluated",
        "robot_evaluation": "evaluated" if robot_attempted else "not_evaluated_by_request",
        "failure_reason": failure_reasons,
        "asset_inspection": {
            "usd_path": str(usd_path),
            "usd_sha256": actual_usd_sha,
            "asset_uniform_scale": asset_uniform_scale,
            "authored_scale_applied": authored_scale_applied,
            "mass_property_scale_policy": mass_property_scale_policy,
            "mass_property_scale_records": mass_property_scale_records,
            "root_prim": str(object_root.GetPath()),
            "target_body_prim": str(target_body.GetPath()),
            "visual_mesh_count": len(visual_paths),
            "collision_mesh_count": len(collision_prims),
            "visual_extent_m": extent.tolist(),
            "collision_extent_m": collision_extent.tolist(),
            "collision_mesh_paths": collision_paths,
            "placed_collision_bounds_m": {"min": placed_collision_min.tolist(), "max": placed_collision_max.tolist()},
            "preplay_placement_delta_m": placement_delta.tolist(),
            "body_physics": body_physics,
            "joint_physics": joint_physics,
            "dynamic_body_count": len(dynamic_records),
            "articulated_setup_error": articulated_setup_error,
            "articulated_setup_correction_m": articulated_setup_correction,
            "articulated_initial_joint_positions_setup": articulated_initial_joint_positions_setup,
            "articulated_initial_joint_error": articulated_initial_joint_error,
            "articulated_joint_index": int(args.articulated_joint_index),
            "articulated_tangent_sign": args.articulated_tangent_sign,
            "articulated_pull_distance_m": articulated_pull_distance,
            "articulated_ee_target_offset_m": ee_target_offset.tolist(),
            "articulated_target_root_world_m": target_root_world.tolist() if stratum == "articulated" else None,
            "articulated_orientation_policy": args.articulated_orientation if stratum == "articulated" else None,
            "articulation_self_collision_policy": articulation_self_collision_policy,
            "articulation_self_collision_error": articulation_self_collision_error,
            "articulation_filtered_pairs": articulation_filtered_pairs,
            "body_bounds_debug": body_bounds_debug,
            "target_articulation_view": bool(target_articulation is not None),
        },
        "physics_evidence": {
            "initial_target_body_position_m": initial_body_position.tolist(),
            "final_target_body_position_m": final_body_position.tolist(),
            "settle_drift_m": settle_drift,
            "minimum_visual_bottom_z_m": min_bottom,
            "minimum_collision_bottom_z_m": min_collision_bottom,
            "contact_event_count": len(contact_events),
            "contact_event_header_count_total": contact_event_header_count_total,
            "contact_event_count_total": contact_event_count_total,
            "robot_contact_event_count": robot_contact_event_count_total,
            "robot_actuated_link_contact_event_count": actuated_child_contact_event_count_total,
            "control_displacement_m": control_displacement.tolist(),
            "trace_state_count": len(trace),
            "action_trace_count": len(action_trace),
            "action_trace_finite": bool(action_trace) and finite_tree(action_trace),
            "tracked_child_path": str(tracked_child.GetPath()) if tracked_child is not None else None,
            "tracked_child_start_position_m": tracked_child_start.tolist() if tracked_child_start is not None else None,
            "tracked_child_end_position_m": tracked_child_end.tolist() if tracked_child_end is not None else None,
            "tracked_child_displacement_m": float(np.linalg.norm(tracked_child_end - tracked_child_start)) if tracked_child_start is not None and tracked_child_end is not None else None,
            "tracked_child_rotation_rad": _quaternion_angle(tracked_child_quat_start, tracked_child_quat_end, np, math) if tracked_child_quat_start is not None and tracked_child_quat_end is not None else None,
            "finite_trace": body_finite,
            "object_transform_writes_after_play": 0,
            "maximum_interpenetration_m": maximum_interpenetration_m,
            "maximum_settle_interpenetration_m": maximum_settle_interpenetration_m,
            "articulation_self_collision_policy": articulation_self_collision_policy,
            "articulation_filtered_pair_count": len(articulation_filtered_pairs),
            "initial_penetration_status": "passed" if initial_penetration_ok else "failed",
            "gravity_settle_status": "passed" if gravity_settle_ok else "failed",
            "stability_status": "passed" if gravity_settle_ok else "failed",
            "contact_status": "observed" if contact_event_count_total else "not_observed",
            "contact_test_status": "passed" if contact_test_pass else "not_observed",
            "geometric_support_contact": geometric_support_contact,
            "support_xy_overlap": support_xy_overlap,
            "contact_probe_status": "not_run_real_contact_track",
            "contact_probe_event_count": None,
            "contact_probe_start_position_m": None,
            "contact_probe_end_position_m": None,
            "physical_checks": {
                "dynamic_body_present": dynamic_body_present,
                "mass_positive_finite": mass_ok,
                "center_of_mass_finite": center_of_mass_ok,
                "inertia_positive_finite": inertia_ok,
                "collision_geometry_nonempty": collision_nonempty,
                "collision_api_valid": collision_valid,
                "initial_penetration": initial_penetration_ok,
                "gravity_settle": gravity_settle_ok,
                "contact_event_observed": bool(contact_event_count_total),
                "geometric_support_contact": geometric_support_contact,
                "contact_test": contact_test_pass,
            },
            "contact_event_samples": contact_events[:16],
            "actuated_child_contact_event_samples": actuated_child_contact_event_samples,
        },
        "robot_evidence": {
            "robot_real_world": "not_evaluated",
            "robot_model": "Franka registered RMPflow teacher joint targets" if franka is not None else "not_constructed",
            "attempted": robot_attempted,
            "workspace_ok": workspace_ok,
            "ik_success_frames": robot_ik_frames,
            "ik_failure_count": len(robot_ik_failures),
            "ik_failures": robot_ik_failures[:32],
            "waypoint_tracking_failure_count": len(robot_tracking_failures),
            "waypoint_tracking_failures": robot_tracking_failures[:16],
            "robot_displacement_m": robot_displacement.tolist(),
            "articulated_child_displacement_m": float(np.linalg.norm(tracked_child_end - tracked_child_start)) if tracked_child_start is not None and tracked_child_end is not None else None,
            "articulated_child_rotation_rad": _quaternion_angle(tracked_child_quat_start, tracked_child_quat_end, np, math) if tracked_child_quat_start is not None and tracked_child_quat_end is not None else None,
            "robot_actuated_link_contact_event_count": actuated_child_contact_event_count_total,
            "robot_task_initial_body_position_m": robot_start_body.tolist(),
            "robot_task_final_body_position_m": robot_task_end_body.tolist(),
            "post_robot_control_body_position_m": post_robot_control_body.tolist(),
            "post_robot_control_displacement_m": (post_robot_control_body - robot_start_body).tolist(),
            "measured_initial_orientation_wxyz": measured_initial_orientation.tolist(),
            "controller_orientation_wxyz": initial_orientation.tolist(),
            "controller_orientation_source": "cli_registered_orientation" if str(args.controller_orientation_wxyz).strip() else "frozen_UP-099_downward_wrist_protocol",
            "failure": robot_failure,
            "learned_vla": True,
            "fallback_used": False,
            "replay_used": False,
            "object_transform_writes_after_play": 0,
            "articulated_target_debug": articulated_target_debug,
            "franka_base_position_m": base_position.tolist(),
        },
        "action_interface_evidence": {
            "interface": "franka_joint_position_targets_plus_gripper",
            "controller": args.controller,
            "post_play_target_object_commands": 0,
            "post_play_object_transform_writes": 0,
            "post_play_object_velocity_commands": 0,
            "post_play_object_joint_drive_commands": 0,
            "post_play_object_joint_target_commands": 0,
            "action_source": "registered_local_neural_robot_policy",
            "franka_base_position_m": base_position.tolist(),
        },
        "video_validation": video_validation,
        "action_trace": action_trace,
        "artifact_paths": {
            "case_manifest": str(case_manifest_path),
            "environment": str(output / "environment.json"),
            "setup_frame": str(output / "setup_frame.png"),
            "view_front": str(output / "view_front.png") if (output / "view_front.png").is_file() else None,
            "view_side": str(output / "view_side.png") if (output / "view_side.png").is_file() else None,
            "view_top": str(output / "view_top.png") if (output / "view_top.png").is_file() else None,
            "final_frame": str(output / "final_frame.png") if (output / "final_frame.png").is_file() else None,
            "rollout_video": str(output / "rollout_raw.mp4"),
            "state_trace": str(output / "state_trace.json"),
            "usd": str(usd_path),
            "source_rgb": str(_repo_path(case["source"]["copy_path"], root)),
            "candidate_preview": str(_repo_path(case["artifact_paths"]["candidate_preview"], root)) if case.get("artifact_paths", {}).get("candidate_preview") else None,
        },
        "reproducibility": {
            "case_manifest_sha256": sha256_file(case_manifest_path),
            "usd_sha256": actual_usd_sha,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "environment_sha256": sha256_file(output / "environment.json"),
            "git_sha": run_text(["git", "rev-parse", "HEAD"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "pid": os.getpid(),
            "initial_center_request_m": [requested_x, requested_y],
            "collection_seed": os.environ.get("EXP3_COLLECTION_SEED"),
            "command": sys.argv,
            "articulated_placement": args.articulated_placement,
            "support_width_m": args.support_width,
            "support_depth_m": args.support_depth,
            "support_top_z_m": table_top,
            "articulated_pivot_x_m": args.articulated_pivot_x,
            "articulated_pivot_y_m": args.articulated_pivot_y,
            "articulated_initial_joint_rad": args.articulated_initial_joint_rad,
            "articulated_contact_radius_fraction": args.articulated_contact_radius_fraction,
            "articulated_joint_index": int(args.articulated_joint_index),
            "articulated_tangent_sign": args.articulated_tangent_sign,
            "articulated_pull_distance_m": articulated_pull_distance,
            "articulated_approach_distance_m": float(getattr(args, "articulated_approach_distance", 0.12)),
            "articulated_approach_mode": str(getattr(args, "articulated_approach_mode", "tangent")),
            "articulated_motion_path": str(getattr(args, "articulated_motion_path", "linear")),
            "articulated_arc_angle_rad": float(getattr(args, "articulated_arc_angle_rad", 1.20)),
            "finger_contact_compensation": str(getattr(args, "finger_contact_compensation", "false")),
            "finger_contact_side": str(getattr(args, "finger_contact_side", "midpoint")),
            "articulated_self_collision_policy": str(getattr(args, "articulated_self_collision_policy", "disabled_filtered")),
            "articulated_approach_z_floor_offset_m": float(getattr(args, "articulated_approach_z_floor_offset", 0.08)),
            "articulated_approach_contact_z_clearance_m": float(getattr(args, "articulated_approach_contact_z_clearance", 0.0)),
            "articulated_contact_z_offset_m": float(getattr(args, "articulated_contact_z_offset", 0.0)),
            "articulated_hold_z_offset_m": float(getattr(args, "articulated_hold_z_offset", 0.0)),
            "rigid_push_distance_m": float(getattr(args, "rigid_push_distance", 0.0) or 0.0),
            "asset_uniform_scale": asset_uniform_scale,
            "mass_property_scale_policy": mass_property_scale_policy,
            "franka_usd_path": str(Path(args.franka_usd_path).expanduser().resolve()) if args.franka_usd_path else None,
            "franka_usd_sha256": sha256_file(Path(args.franka_usd_path).expanduser().resolve()) if args.franka_usd_path and Path(args.franka_usd_path).expanduser().is_file() else None,
            "franka_base_position_m": base_position.tolist(),
        },
    }
    complete_raw=json.loads((output/"complete_task_raw_trace.json").read_text())
    from i2ia.tasks.support_contact_lifecycle import independently_verify as verify_support_lifecycle
    support_replay = verify_support_lifecycle(full_spec, complete_raw)
    write_json(output/"independent_support_lifecycle.json",support_replay)
    if not support_replay["passed"]:raise RuntimeError("support_lifecycle_independent_replay_failed")
    complete_eval=evaluate_complete_task(full_spec,complete_raw)
    from exp3_training_instruction_contract_v0_134 import release_evidence
    release_check=release_evidence(full_spec,complete_raw['states'])
    if not release_check['passed']:
        complete_eval={**complete_eval,"sim_task_success":False,"status":"failed","failure_reasons":sorted(set(complete_eval['failure_reasons']+['release_then_hold_clause_not_verified']))}
    result['release_evidence']=release_check
    result["small_motion_diagnostic_not_task_success"]=result.pop("sim_task_success")
    result["sim_task_success"]=bool(complete_eval["sim_task_success"] is True and result["sim_ready_status"] == "sim_ready_only")
    if complete_eval["sim_task_success"] and not result["sim_task_success"]:
        complete_eval = {**complete_eval, "sim_task_success": False, "status": "failed", "failure_reasons": ["full_task_reached_but_physics_gate_failed"]}
    result["complete_task_evaluation"]=complete_eval
    result["legacy_small_motion_predicate_diagnostic_only"]=result.pop("simulator_success_predicate")
    result["simulator_success_predicate"]="independent complete TaskSpec state machine AND physical validation; training requires separate admission verifier"
    result["full_task_spec_sha256"]=sha256_file(Path(os.environ["EXP3_COMPLETE_TASK_SPEC"]))
    result["full_task_trace_sha256"]=sha256_file(output/"complete_task_raw_trace.json")
    result["teacher_used"]=False
    result["model_variant"]=args.model_variant
    result["policy_identity"]=policy_identity
    result["failure_reason"]=complete_eval["failure_reasons"]
    write_json(output / "state_trace.json", {"schema_version": "i2ia.exp3_success_trace_state.v0.2", "states": trace, "actions": action_trace, "case_id": case.get("case_id"), "object_transform_writes_after_play": 0, "fallback_used": False, "replay_used": False, "control_hz": args.fps})
    result["artifact_paths"]["state_trace_sha256"] = sha256_file(output / "state_trace.json")
    result["artifact_paths"]["setup_frame_sha256"] = sha256_file(output / "setup_frame.png")
    for key in ("view_front", "view_side", "view_top", "final_frame"):
        raw = result["artifact_paths"].get(key)
        if raw and Path(raw).is_file():
            result["artifact_paths"][f"{key}_sha256"] = sha256_file(Path(raw))
    result["artifact_paths"]["rollout_video_sha256"] = sha256_file(output / "rollout_raw.mp4")
    result["artifact_paths"]["rollout_video"] = str(output / "rollout_raw.mp4")
    result["artifact_paths"]["rollout_video_frame_count"] = raw_frame_count
    video_metadata_path = output / "rollout_raw.metadata.json"
    write_json(
        video_metadata_path,
        {
            "asset_id": full_spec['asset_id'],
            "task_id": full_spec['task_id'],
            "seed": os.environ.get("EXP3_COLLECTION_SEED"),
            "model_variant": args.model_variant,
            "final_success": result["sim_task_success"],
            "camera_capture_type": "raw_isaac_rgb",
            "video_path": str(output / "rollout_raw.mp4"),
            "video_sha256": result["artifact_paths"]["rollout_video_sha256"],
            "frame_count": raw_frame_count,
        },
    )
    result["artifact_paths"]["rollout_video_metadata"] = str(video_metadata_path)
    result["artifact_paths"]["rollout_video_metadata_sha256"] = sha256_file(video_metadata_path)
    result["artifact_paths"]["action_trace"] = str(output / "state_trace.json")
    result["artifact_paths"]["observation_dir"] = str(observation_dir) if args.save_observations == "true" else None
    result["artifact_paths"]["observation_count"] = len(list(observation_dir.glob("frame_*.png"))) if observation_dir.is_dir() else 0
    result["observation_action_alignment"]="raw_isaac_rgb_and_measured_robot_proprio_before_action"
    write_json(output / "case_result.json", result)
    return result


def main() -> int:
    args = parse_args()
    if os.environ.get("ACCEPT_EULA", "").strip().upper() not in {"1", "TRUE", "YES", "Y"}:
        print("ACCEPT_EULA=Y is required", file=sys.stderr)
        return 2
    # Kit consumes argv at construction; do not leak runner arguments into the
    # application parser.
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "hide_ui": True, "width": args.width, "height": args.height, "renderer": args.renderer, "anti_aliasing": 0, "multi_gpu": False, "active_gpu": args.gpu_index, "physics_gpu": args.gpu_index, "fast_shutdown": True})
    finally:
        sys.argv = original_argv
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = run_case(args, app)
        print(json.dumps({"status": result["status"], "case_id": result["case_id"], "physics_validation_status": result["physics_validation_status"], "robot_task_success": result["robot_task_success"], "output": str(output / "case_result.json")}, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = {"schema_version": SCHEMA, "status": "runtime_exception", "case_manifest": str(args.case_manifest.resolve()), "failure_type": type(exc).__name__, "failure_detail": str(exc), "traceback": traceback.format_exc(), "robot_real_world": "not_evaluated", "robot_control_status": "simulated_robot_only", "robot_task_success": None, "sim_task_success": None, "model_variant": args.model_variant, "learned_vla": True, "teacher_used": False, "fallback_used": False, "replay_used": False, "pid": os.getpid(), "command": sys.argv}
        write_json(output / "runtime_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            app.close()
        except BaseException as exc:
            write_json(output / "runtime_close_warning.json", {"failure_type": type(exc).__name__, "failure_detail": str(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    raise SystemExit(main())
