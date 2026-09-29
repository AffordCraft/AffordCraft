from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._validation import (
    ContractValidationError,
    check_schema_version,
    json_copy,
    literal,
    mapping,
    non_empty_string,
    normalized_json_object,
    positive_int,
    required,
    string_tuple,
    validate_schema_version,
)
from .artifacts import ArtifactRef
from .scene import PhysicalSceneGraph


_PREDICATE_TYPES = {
    "displacement",
    "relative_pose",
    "joint_change",
    "contact",
    "region_contains",
    "duration",
    "scalar_compare",
    "all",
    "any",
    "not",
}
_SUCCESS_MODES = {"all", "any"}
_INVALID_SPATIAL_FRAMES = {
    "n/a", "na", "none", "unknown", "unversioned", "unspecified"
}


@dataclass(frozen=True, slots=True)
class TaskPredicate:
    predicate_id: str
    predicate_type: str
    parameters: dict[str, Any]
    hold_steps: int = 1
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(
            self.predicate_id,
            "TaskPredicate.predicate_id",
        )
        literal(
            self.predicate_type,
            _PREDICATE_TYPES,
            "TaskPredicate.predicate_type",
        )
        parameters = normalized_json_object(
            self.parameters,
            "TaskPredicate.parameters",
        )
        if not parameters:
            raise ContractValidationError(
                "TaskPredicate.parameters must not be empty"
            )
        object.__setattr__(self, "parameters", parameters)
        positive_int(self.hold_steps, "TaskPredicate.hold_steps")
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "TaskPredicate",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "predicate_id": self.predicate_id,
            "predicate_type": self.predicate_type,
            "parameters": json_copy(self.parameters),
            "hold_steps": self.hold_steps,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TaskPredicate":
        data = mapping(value, "TaskPredicate")
        check_schema_version(data, cls.SCHEMA_VERSION, "TaskPredicate")
        return cls(
            predicate_id=non_empty_string(
                required(data, "predicate_id", "TaskPredicate"),
                "TaskPredicate.predicate_id",
            ),
            predicate_type=literal(
                required(data, "predicate_type", "TaskPredicate"),
                _PREDICATE_TYPES,
                "TaskPredicate.predicate_type",
            ),
            parameters=normalized_json_object(
                required(data, "parameters", "TaskPredicate"),
                "TaskPredicate.parameters",
            ),
            hold_steps=positive_int(
                data.get("hold_steps", 1),
                "TaskPredicate.hold_steps",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    instruction: str
    predicates: tuple[TaskPredicate, ...]
    success_mode: str = "all"
    max_steps: int | None = None
    required_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.task_id, "TaskSpec.task_id")
        non_empty_string(self.instruction, "TaskSpec.instruction")
        predicates = tuple(self.predicates)
        if not predicates:
            raise ContractValidationError(
                "TaskSpec.predicates must not be empty"
            )
        if not all(
            isinstance(predicate, TaskPredicate)
            for predicate in predicates
        ):
            raise ContractValidationError(
                "TaskSpec.predicates must contain only TaskPredicate values"
            )
        predicate_ids = [
            predicate.predicate_id for predicate in predicates
        ]
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ContractValidationError(
                "TaskSpec.predicates must have unique predicate_id values"
            )
        object.__setattr__(self, "predicates", predicates)
        literal(
            self.success_mode,
            _SUCCESS_MODES,
            "TaskSpec.success_mode",
        )
        if self.max_steps is not None:
            positive_int(self.max_steps, "TaskSpec.max_steps")
        object.__setattr__(
            self,
            "required_capabilities",
            string_tuple(
                self.required_capabilities,
                "TaskSpec.required_capabilities",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(self.metadata, "TaskSpec.metadata"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "TaskSpec",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "predicates": [
                predicate.to_dict() for predicate in self.predicates
            ],
            "success_mode": self.success_mode,
            "max_steps": self.max_steps,
            "required_capabilities": list(
                self.required_capabilities
            ),
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TaskSpec":
        data = mapping(value, "TaskSpec")
        check_schema_version(data, cls.SCHEMA_VERSION, "TaskSpec")
        raw_predicates = required(data, "predicates", "TaskSpec")
        if not isinstance(raw_predicates, (list, tuple)):
            raise ContractValidationError(
                "TaskSpec.predicates must be an array"
            )
        raw_max_steps = data.get("max_steps")
        return cls(
            task_id=non_empty_string(
                required(data, "task_id", "TaskSpec"),
                "TaskSpec.task_id",
            ),
            instruction=non_empty_string(
                required(data, "instruction", "TaskSpec"),
                "TaskSpec.instruction",
            ),
            predicates=tuple(
                TaskPredicate.from_dict(item)
                for item in raw_predicates
            ),
            success_mode=literal(
                data.get("success_mode", "all"),
                _SUCCESS_MODES,
                "TaskSpec.success_mode",
            ),
            max_steps=(
                None
                if raw_max_steps is None
                else positive_int(
                    raw_max_steps,
                    "TaskSpec.max_steps",
                )
            ),
            required_capabilities=string_tuple(
                data.get("required_capabilities", []),
                "TaskSpec.required_capabilities",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "TaskSpec.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class SimulationAssetManifest:
    asset_id: str
    simulator: str
    scene_graph: PhysicalSceneGraph
    artifacts: tuple[ArtifactRef, ...]
    primary_artifact_id: str
    units: str
    up_axis: str
    coordinate_frame: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.1"

    SCHEMA_VERSION: ClassVar[str] = "1.1"

    def __post_init__(self) -> None:
        non_empty_string(
            self.asset_id,
            "SimulationAssetManifest.asset_id",
        )
        non_empty_string(
            self.simulator,
            "SimulationAssetManifest.simulator",
        )
        if not isinstance(self.scene_graph, PhysicalSceneGraph):
            raise ContractValidationError(
                "SimulationAssetManifest.scene_graph must be a PhysicalSceneGraph"
            )
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ContractValidationError(
                "SimulationAssetManifest.artifacts must not be empty"
            )
        if not all(
            isinstance(artifact, ArtifactRef)
            for artifact in artifacts
        ):
            raise ContractValidationError(
                "SimulationAssetManifest.artifacts must contain only ArtifactRef values"
            )
        artifact_ids = [
            artifact.artifact_id for artifact in artifacts
        ]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ContractValidationError(
                "SimulationAssetManifest.artifacts must have unique artifact_id values"
            )
        object.__setattr__(self, "artifacts", artifacts)
        non_empty_string(
            self.primary_artifact_id,
            "SimulationAssetManifest.primary_artifact_id",
        )
        if self.primary_artifact_id not in set(artifact_ids):
            raise ContractValidationError(
                "SimulationAssetManifest.primary_artifact_id must reference "
                "one of its artifacts"
            )
        primary_artifact = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_id == self.primary_artifact_id
        )
        non_empty_string(
            self.units,
            "SimulationAssetManifest.units",
        )
        literal(
            self.up_axis,
            {"X", "Y", "Z", "-X", "-Y", "-Z"},
            "SimulationAssetManifest.up_axis",
        )
        non_empty_string(
            self.coordinate_frame,
            "SimulationAssetManifest.coordinate_frame",
        )
        if self.coordinate_frame.strip().lower() in _INVALID_SPATIAL_FRAMES:
            raise ContractValidationError(
                "SimulationAssetManifest.coordinate_frame must identify a spatial frame"
            )
        if self.units != self.scene_graph.units:
            raise ContractValidationError(
                "SimulationAssetManifest.units must match scene_graph.units"
            )
        if self.up_axis != self.scene_graph.up_axis:
            raise ContractValidationError(
                "SimulationAssetManifest.up_axis must match scene_graph.up_axis"
            )
        for field_name, expected in (
            ("units", self.units),
            ("up_axis", self.up_axis),
            ("coordinate_frame", self.coordinate_frame),
        ):
            if getattr(primary_artifact, field_name) != expected:
                raise ContractValidationError(
                    "SimulationAssetManifest primary artifact "
                    f"{field_name} must match the manifest"
                )
        inconsistent_axes = sorted(
            {
                artifact.up_axis
                for artifact in artifacts
                if artifact.up_axis not in {None, "NONE", self.up_axis}
            }
        )
        if inconsistent_axes:
            raise ContractValidationError(
                "SimulationAssetManifest artifact up_axis values must match "
                "the manifest up_axis"
            )
        if (
            self.coordinate_frame
            != self.scene_graph.coordinate_frame
        ):
            raise ContractValidationError(
                "SimulationAssetManifest.coordinate_frame must match "
                "scene_graph.coordinate_frame"
            )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "SimulationAssetManifest.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "SimulationAssetManifest",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "simulator": self.simulator,
            "scene_graph": self.scene_graph.to_dict(),
            "artifacts": [
                artifact.to_dict() for artifact in self.artifacts
            ],
            "primary_artifact_id": self.primary_artifact_id,
            "units": self.units,
            "up_axis": self.up_axis,
            "coordinate_frame": self.coordinate_frame,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SimulationAssetManifest":
        data = mapping(value, "SimulationAssetManifest")
        actual_version = non_empty_string(
            required(data, "schema_version", "SimulationAssetManifest"),
            "SimulationAssetManifest.schema_version",
        )
        if actual_version not in {"1.0", cls.SCHEMA_VERSION}:
            raise ContractValidationError(
                "SimulationAssetManifest.schema_version must be '1.0' or "
                f"{cls.SCHEMA_VERSION!r}, got {actual_version!r}"
            )
        raw_artifacts = required(
            data,
            "artifacts",
            "SimulationAssetManifest",
        )
        if not isinstance(raw_artifacts, (list, tuple)):
            raise ContractValidationError(
                "SimulationAssetManifest.artifacts must be an array"
            )
        scene_graph = PhysicalSceneGraph.from_dict(
            required(data, "scene_graph", "SimulationAssetManifest")
        )
        if actual_version == "1.0":
            up_axis = scene_graph.up_axis
        else:
            up_axis = non_empty_string(
                required(data, "up_axis", "SimulationAssetManifest"),
                "SimulationAssetManifest.up_axis",
            )
        return cls(
            asset_id=non_empty_string(
                required(
                    data,
                    "asset_id",
                    "SimulationAssetManifest",
                ),
                "SimulationAssetManifest.asset_id",
            ),
            simulator=non_empty_string(
                required(
                    data,
                    "simulator",
                    "SimulationAssetManifest",
                ),
                "SimulationAssetManifest.simulator",
            ),
            scene_graph=scene_graph,
            artifacts=tuple(
                ArtifactRef.from_dict(item)
                for item in raw_artifacts
            ),
            primary_artifact_id=non_empty_string(
                required(
                    data,
                    "primary_artifact_id",
                    "SimulationAssetManifest",
                ),
                "SimulationAssetManifest.primary_artifact_id",
            ),
            units=non_empty_string(
                required(
                    data,
                    "units",
                    "SimulationAssetManifest",
                ),
                "SimulationAssetManifest.units",
            ),
            up_axis=up_axis,
            coordinate_frame=non_empty_string(
                required(
                    data,
                    "coordinate_frame",
                    "SimulationAssetManifest",
                ),
                "SimulationAssetManifest.coordinate_frame",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "SimulationAssetManifest.metadata",
            ),
            schema_version=cls.SCHEMA_VERSION,
        )
