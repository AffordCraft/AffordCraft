from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._validation import (
    ContractValidationError,
    check_schema_version,
    finite_float,
    json_copy,
    literal,
    mapping,
    non_empty_string,
    normalized_json_object,
    optional_string,
    positive_float,
    required,
    string_tuple,
    validate_schema_version,
    vector,
)
from .artifacts import ArtifactRef


_MOTION_TYPES = {"static", "dynamic", "kinematic"}
_JOINT_TYPES = {"fixed", "revolute", "prismatic"}
_INVALID_SPATIAL_FRAMES = {
    "n/a", "na", "none", "unknown", "unversioned", "unspecified"
}


@dataclass(frozen=True, slots=True)
class RigidBody:
    body_id: str
    motion_type: str
    geometry: tuple[ArtifactRef, ...] = ()
    collision_geometry: tuple[ArtifactRef, ...] = ()
    mass_kg: float | None = None
    center_of_mass_m: tuple[float, float, float] | None = None
    diagonal_inertia_kg_m2: tuple[float, float, float] | None = None
    principal_axes_wxyz: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.body_id, "RigidBody.body_id")
        literal(self.motion_type, _MOTION_TYPES, "RigidBody.motion_type")
        geometry = tuple(self.geometry)
        collision = tuple(self.collision_geometry)
        if not all(isinstance(item, ArtifactRef) for item in geometry):
            raise ContractValidationError(
                "RigidBody.geometry must contain only ArtifactRef values"
            )
        if not all(isinstance(item, ArtifactRef) for item in collision):
            raise ContractValidationError(
                "RigidBody.collision_geometry must contain only ArtifactRef values"
            )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "collision_geometry", collision)
        if self.mass_kg is not None:
            object.__setattr__(
                self,
                "mass_kg",
                positive_float(self.mass_kg, "RigidBody.mass_kg"),
            )
        if self.center_of_mass_m is not None:
            object.__setattr__(
                self,
                "center_of_mass_m",
                vector(
                    self.center_of_mass_m,
                    "RigidBody.center_of_mass_m",
                    length=3,
                ),
            )
        if (self.diagonal_inertia_kg_m2 is None) != (
            self.principal_axes_wxyz is None
        ):
            raise ContractValidationError(
                "RigidBody.diagonal_inertia_kg_m2 and "
                "RigidBody.principal_axes_wxyz must be declared together"
            )
        if self.diagonal_inertia_kg_m2 is not None and self.mass_kg is None:
            raise ContractValidationError(
                "RigidBody.diagonal_inertia_kg_m2 requires mass_kg"
            )
        if self.diagonal_inertia_kg_m2 is not None:
            inertia = vector(
                self.diagonal_inertia_kg_m2,
                "RigidBody.diagonal_inertia_kg_m2",
                length=3,
            )
            if any(value <= 0.0 for value in inertia):
                raise ContractValidationError(
                    "RigidBody.diagonal_inertia_kg_m2 values must be positive"
                )
            object.__setattr__(self, "diagonal_inertia_kg_m2", inertia)
            axes = vector(
                self.principal_axes_wxyz,
                "RigidBody.principal_axes_wxyz",
                length=4,
            )
            norm = math.sqrt(sum(value * value for value in axes))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ContractValidationError(
                    "RigidBody.principal_axes_wxyz must be a unit quaternion"
                )
            object.__setattr__(self, "principal_axes_wxyz", axes)
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(self.metadata, "RigidBody.metadata"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "RigidBody",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_id": self.body_id,
            "motion_type": self.motion_type,
            "geometry": [item.to_dict() for item in self.geometry],
            "collision_geometry": [
                item.to_dict() for item in self.collision_geometry
            ],
            "mass_kg": self.mass_kg,
            "center_of_mass_m": (
                list(self.center_of_mass_m)
                if self.center_of_mass_m is not None
                else None
            ),
            "diagonal_inertia_kg_m2": (
                list(self.diagonal_inertia_kg_m2)
                if self.diagonal_inertia_kg_m2 is not None
                else None
            ),
            "principal_axes_wxyz": (
                list(self.principal_axes_wxyz)
                if self.principal_axes_wxyz is not None
                else None
            ),
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RigidBody":
        data = mapping(value, "RigidBody")
        check_schema_version(data, cls.SCHEMA_VERSION, "RigidBody")
        raw_geometry = data.get("geometry", [])
        raw_collision = data.get("collision_geometry", [])
        if not isinstance(raw_geometry, (list, tuple)):
            raise ContractValidationError("RigidBody.geometry must be an array")
        if not isinstance(raw_collision, (list, tuple)):
            raise ContractValidationError(
                "RigidBody.collision_geometry must be an array"
            )
        raw_center = data.get("center_of_mass_m")
        raw_mass = data.get("mass_kg")
        raw_inertia = data.get("diagonal_inertia_kg_m2")
        raw_principal_axes = data.get("principal_axes_wxyz")
        return cls(
            body_id=non_empty_string(
                required(data, "body_id", "RigidBody"),
                "RigidBody.body_id",
            ),
            motion_type=literal(
                required(data, "motion_type", "RigidBody"),
                _MOTION_TYPES,
                "RigidBody.motion_type",
            ),
            geometry=tuple(
                ArtifactRef.from_dict(item) for item in raw_geometry
            ),
            collision_geometry=tuple(
                ArtifactRef.from_dict(item) for item in raw_collision
            ),
            mass_kg=(
                None
                if raw_mass is None
                else positive_float(raw_mass, "RigidBody.mass_kg")
            ),
            center_of_mass_m=(
                None
                if raw_center is None
                else vector(
                    raw_center,
                    "RigidBody.center_of_mass_m",
                    length=3,
                )
            ),
            diagonal_inertia_kg_m2=(
                None
                if raw_inertia is None
                else vector(
                    raw_inertia,
                    "RigidBody.diagonal_inertia_kg_m2",
                    length=3,
                )
            ),
            principal_axes_wxyz=(
                None
                if raw_principal_axes is None
                else vector(
                    raw_principal_axes,
                    "RigidBody.principal_axes_wxyz",
                    length=4,
                )
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "RigidBody.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class Joint:
    joint_id: str
    joint_type: str
    parent_body: str
    child_body: str
    coordinate_frame: str
    axis: tuple[float, float, float] | None = None
    origin_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits: tuple[float, float] | None = None
    limit_units: str | None = None
    effort_limit: float | None = None
    velocity_limit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.joint_id, "Joint.joint_id")
        literal(self.joint_type, _JOINT_TYPES, "Joint.joint_type")
        non_empty_string(self.parent_body, "Joint.parent_body")
        non_empty_string(self.child_body, "Joint.child_body")
        if self.parent_body == self.child_body:
            raise ContractValidationError(
                "Joint.parent_body and Joint.child_body must differ"
            )
        non_empty_string(self.coordinate_frame, "Joint.coordinate_frame")
        object.__setattr__(
            self,
            "origin_xyz_m",
            vector(self.origin_xyz_m, "Joint.origin_xyz_m", length=3),
        )

        if self.joint_type == "fixed":
            if self.axis is not None:
                raise ContractValidationError(
                    "Joint.axis must be omitted for a fixed joint"
                )
            if self.limits is not None:
                raise ContractValidationError(
                    "Joint.limits must be omitted for a fixed joint"
                )
            if self.limit_units is not None:
                raise ContractValidationError(
                    "Joint.limit_units must be omitted for a fixed joint"
                )
        else:
            if self.axis is None:
                raise ContractValidationError(
                    f"Joint.axis is required for a {self.joint_type} joint"
                )
            axis = vector(self.axis, "Joint.axis", length=3)
            if math.sqrt(sum(component * component for component in axis)) <= 1e-12:
                raise ContractValidationError("Joint.axis must be non-zero")
            object.__setattr__(self, "axis", axis)
            expected_units = (
                "rad" if self.joint_type == "revolute" else "m"
            )
            if self.limit_units is None:
                object.__setattr__(self, "limit_units", expected_units)
            elif self.limit_units != expected_units:
                raise ContractValidationError(
                    f"Joint.limit_units must be {expected_units!r} for a "
                    f"{self.joint_type} joint"
                )
            if self.limits is not None:
                limits = vector(self.limits, "Joint.limits", length=2)
                if limits[0] > limits[1]:
                    raise ContractValidationError(
                        "Joint.limits lower value must not exceed upper value"
                    )
                object.__setattr__(self, "limits", limits)

        if self.effort_limit is not None:
            object.__setattr__(
                self,
                "effort_limit",
                positive_float(self.effort_limit, "Joint.effort_limit"),
            )
        if self.velocity_limit is not None:
            object.__setattr__(
                self,
                "velocity_limit",
                positive_float(
                    self.velocity_limit,
                    "Joint.velocity_limit",
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(self.metadata, "Joint.metadata"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "Joint",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "joint_id": self.joint_id,
            "joint_type": self.joint_type,
            "parent_body": self.parent_body,
            "child_body": self.child_body,
            "coordinate_frame": self.coordinate_frame,
            "axis": list(self.axis) if self.axis is not None else None,
            "origin_xyz_m": list(self.origin_xyz_m),
            "limits": list(self.limits) if self.limits is not None else None,
            "limit_units": self.limit_units,
            "effort_limit": self.effort_limit,
            "velocity_limit": self.velocity_limit,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "Joint":
        data = mapping(value, "Joint")
        check_schema_version(data, cls.SCHEMA_VERSION, "Joint")
        raw_axis = data.get("axis")
        raw_limits = data.get("limits")
        raw_effort = data.get("effort_limit")
        raw_velocity = data.get("velocity_limit")
        return cls(
            joint_id=non_empty_string(
                required(data, "joint_id", "Joint"),
                "Joint.joint_id",
            ),
            joint_type=literal(
                required(data, "joint_type", "Joint"),
                _JOINT_TYPES,
                "Joint.joint_type",
            ),
            parent_body=non_empty_string(
                required(data, "parent_body", "Joint"),
                "Joint.parent_body",
            ),
            child_body=non_empty_string(
                required(data, "child_body", "Joint"),
                "Joint.child_body",
            ),
            coordinate_frame=non_empty_string(
                required(data, "coordinate_frame", "Joint"),
                "Joint.coordinate_frame",
            ),
            axis=(
                None
                if raw_axis is None
                else vector(raw_axis, "Joint.axis", length=3)
            ),
            origin_xyz_m=vector(
                data.get("origin_xyz_m", [0.0, 0.0, 0.0]),
                "Joint.origin_xyz_m",
                length=3,
            ),
            limits=(
                None
                if raw_limits is None
                else vector(raw_limits, "Joint.limits", length=2)
            ),
            limit_units=optional_string(
                data.get("limit_units"),
                "Joint.limit_units",
            ),
            effort_limit=(
                None
                if raw_effort is None
                else finite_float(raw_effort, "Joint.effort_limit")
            ),
            velocity_limit=(
                None
                if raw_velocity is None
                else finite_float(raw_velocity, "Joint.velocity_limit")
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "Joint.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PhysicalSceneGraph:
    scene_id: str
    rigid_bodies: tuple[RigidBody, ...]
    joints: tuple[Joint, ...]
    coordinate_frame: str
    units: str
    up_axis: str
    root_body_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.1"

    SCHEMA_VERSION: ClassVar[str] = "1.1"

    def __post_init__(self) -> None:
        non_empty_string(self.scene_id, "PhysicalSceneGraph.scene_id")
        bodies = tuple(self.rigid_bodies)
        joints = tuple(self.joints)
        if not bodies:
            raise ContractValidationError(
                "PhysicalSceneGraph.rigid_bodies must not be empty"
            )
        if not all(isinstance(item, RigidBody) for item in bodies):
            raise ContractValidationError(
                "PhysicalSceneGraph.rigid_bodies must contain only RigidBody values"
            )
        if not all(isinstance(item, Joint) for item in joints):
            raise ContractValidationError(
                "PhysicalSceneGraph.joints must contain only Joint values"
            )
        object.__setattr__(self, "rigid_bodies", bodies)
        object.__setattr__(self, "joints", joints)
        non_empty_string(
            self.coordinate_frame,
            "PhysicalSceneGraph.coordinate_frame",
        )
        if self.coordinate_frame.strip().lower() in _INVALID_SPATIAL_FRAMES:
            raise ContractValidationError(
                "PhysicalSceneGraph.coordinate_frame must identify a spatial frame"
            )
        non_empty_string(self.units, "PhysicalSceneGraph.units")
        literal(
            self.up_axis,
            {"X", "Y", "Z", "-X", "-Y", "-Z"},
            "PhysicalSceneGraph.up_axis",
        )

        body_ids = [body.body_id for body in bodies]
        if len(body_ids) != len(set(body_ids)):
            raise ContractValidationError(
                "PhysicalSceneGraph.rigid_bodies must have unique body_id values"
            )
        joint_ids = [joint.joint_id for joint in joints]
        if len(joint_ids) != len(set(joint_ids)):
            raise ContractValidationError(
                "PhysicalSceneGraph.joints must have unique joint_id values"
            )
        known_bodies = set(body_ids)
        child_to_joint: dict[str, str] = {}
        adjacency: dict[str, list[str]] = {
            body_id: [] for body_id in body_ids
        }
        for joint in joints:
            if joint.parent_body not in known_bodies:
                raise ContractValidationError(
                    f"PhysicalSceneGraph joint {joint.joint_id!r} references "
                    f"unknown parent body {joint.parent_body!r}"
                )
            if joint.child_body not in known_bodies:
                raise ContractValidationError(
                    f"PhysicalSceneGraph joint {joint.joint_id!r} references "
                    f"unknown child body {joint.child_body!r}"
                )
            if joint.child_body in child_to_joint:
                raise ContractValidationError(
                    f"PhysicalSceneGraph body {joint.child_body!r} has more than "
                    "one parent joint"
                )
            child_to_joint[joint.child_body] = joint.joint_id
            adjacency[joint.parent_body].append(joint.child_body)
        self._validate_acyclic(adjacency)

        roots = string_tuple(
            self.root_body_ids,
            "PhysicalSceneGraph.root_body_ids",
        )
        inferred_roots = tuple(
            body_id for body_id in body_ids if body_id not in child_to_joint
        )
        if not roots:
            roots = inferred_roots
        unknown_roots = set(roots) - known_bodies
        if unknown_roots:
            raise ContractValidationError(
                "PhysicalSceneGraph.root_body_ids contains unknown bodies: "
                + ", ".join(sorted(unknown_roots))
            )
        non_roots = set(roots) & set(child_to_joint)
        if non_roots:
            raise ContractValidationError(
                "PhysicalSceneGraph.root_body_ids contains bodies with parent "
                "joints: "
                + ", ".join(sorted(non_roots))
            )
        if set(roots) != set(inferred_roots):
            raise ContractValidationError(
                "PhysicalSceneGraph.root_body_ids must list every graph root"
            )
        object.__setattr__(self, "root_body_ids", roots)
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "PhysicalSceneGraph.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "PhysicalSceneGraph",
        )

    @staticmethod
    def _validate_acyclic(adjacency: dict[str, list[str]]) -> None:
        state: dict[str, int] = {body_id: 0 for body_id in adjacency}

        def visit(body_id: str) -> None:
            if state[body_id] == 1:
                raise ContractValidationError(
                    "PhysicalSceneGraph.joints must not contain a cycle"
                )
            if state[body_id] == 2:
                return
            state[body_id] = 1
            for child in adjacency[body_id]:
                visit(child)
            state[body_id] = 2

        for body_id in adjacency:
            visit(body_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "rigid_bodies": [
                body.to_dict() for body in self.rigid_bodies
            ],
            "joints": [joint.to_dict() for joint in self.joints],
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "up_axis": self.up_axis,
            "root_body_ids": list(self.root_body_ids),
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PhysicalSceneGraph":
        data = mapping(value, "PhysicalSceneGraph")
        actual_version = non_empty_string(
            required(data, "schema_version", "PhysicalSceneGraph"),
            "PhysicalSceneGraph.schema_version",
        )
        if actual_version not in {"1.0", cls.SCHEMA_VERSION}:
            raise ContractValidationError(
                "PhysicalSceneGraph.schema_version must be '1.0' or "
                f"{cls.SCHEMA_VERSION!r}, got {actual_version!r}"
            )
        raw_bodies = required(
            data,
            "rigid_bodies",
            "PhysicalSceneGraph",
        )
        raw_joints = required(data, "joints", "PhysicalSceneGraph")
        if not isinstance(raw_bodies, (list, tuple)):
            raise ContractValidationError(
                "PhysicalSceneGraph.rigid_bodies must be an array"
            )
        if not isinstance(raw_joints, (list, tuple)):
            raise ContractValidationError(
                "PhysicalSceneGraph.joints must be an array"
            )
        bodies = tuple(RigidBody.from_dict(item) for item in raw_bodies)
        if actual_version == "1.0":
            axes = {
                reference.up_axis
                for body in bodies
                for reference in (*body.geometry, *body.collision_geometry)
                if reference.up_axis is not None
            }
            if len(axes) != 1:
                raise ContractValidationError(
                    "PhysicalSceneGraph legacy schema 1.0 requires one "
                    "consistent artifact up_axis for migration"
                )
            up_axis = axes.pop()
        else:
            up_axis = non_empty_string(
                required(data, "up_axis", "PhysicalSceneGraph"),
                "PhysicalSceneGraph.up_axis",
            )
        return cls(
            scene_id=non_empty_string(
                required(data, "scene_id", "PhysicalSceneGraph"),
                "PhysicalSceneGraph.scene_id",
            ),
            rigid_bodies=bodies,
            joints=tuple(Joint.from_dict(item) for item in raw_joints),
            coordinate_frame=non_empty_string(
                required(
                    data,
                    "coordinate_frame",
                    "PhysicalSceneGraph",
                ),
                "PhysicalSceneGraph.coordinate_frame",
            ),
            units=non_empty_string(
                required(data, "units", "PhysicalSceneGraph"),
                "PhysicalSceneGraph.units",
            ),
            up_axis=up_axis,
            root_body_ids=string_tuple(
                data.get("root_body_ids", []),
                "PhysicalSceneGraph.root_body_ids",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "PhysicalSceneGraph.metadata",
            ),
            schema_version=cls.SCHEMA_VERSION,
        )
