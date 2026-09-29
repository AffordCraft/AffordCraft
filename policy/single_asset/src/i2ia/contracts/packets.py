from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ._validation import (
    ContractValidationError,
    check_schema_version,
    finite_float,
    json_copy,
    mapping,
    non_empty_string,
    non_negative_int,
    normalized_json_object,
    optional_string,
    positive_float,
    required,
    string_tuple,
    validate_schema_version,
    vector,
)
from .artifacts import ArtifactRef


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    embodiment: str
    required_channels: tuple[str, ...]
    optional_channels: tuple[str, ...]
    coordinate_frame: str
    instruction_required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.embodiment, "ObservationSpec.embodiment")
        required_channels = string_tuple(
            self.required_channels,
            "ObservationSpec.required_channels",
            allow_empty=False,
        )
        optional_channels = string_tuple(
            self.optional_channels,
            "ObservationSpec.optional_channels",
        )
        overlap = set(required_channels) & set(optional_channels)
        if overlap:
            raise ContractValidationError(
                "ObservationSpec required_channels and optional_channels "
                "must be disjoint: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(
            self,
            "required_channels",
            required_channels,
        )
        object.__setattr__(
            self,
            "optional_channels",
            optional_channels,
        )
        non_empty_string(
            self.coordinate_frame,
            "ObservationSpec.coordinate_frame",
        )
        if not isinstance(self.instruction_required, bool):
            raise ContractValidationError(
                "ObservationSpec.instruction_required must be boolean"
            )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "ObservationSpec.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ObservationSpec",
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return self.required_channels + self.optional_channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "embodiment": self.embodiment,
            "required_channels": list(self.required_channels),
            "optional_channels": list(self.optional_channels),
            "coordinate_frame": self.coordinate_frame,
            "instruction_required": self.instruction_required,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ObservationSpec":
        data = mapping(value, "ObservationSpec")
        check_schema_version(data, cls.SCHEMA_VERSION, "ObservationSpec")
        instruction_required = data.get("instruction_required", True)
        if not isinstance(instruction_required, bool):
            raise ContractValidationError(
                "ObservationSpec.instruction_required must be boolean"
            )
        return cls(
            embodiment=non_empty_string(
                required(data, "embodiment", "ObservationSpec"),
                "ObservationSpec.embodiment",
            ),
            required_channels=string_tuple(
                required(
                    data,
                    "required_channels",
                    "ObservationSpec",
                ),
                "ObservationSpec.required_channels",
                allow_empty=False,
            ),
            optional_channels=string_tuple(
                data.get("optional_channels", []),
                "ObservationSpec.optional_channels",
            ),
            coordinate_frame=non_empty_string(
                required(
                    data,
                    "coordinate_frame",
                    "ObservationSpec",
                ),
                "ObservationSpec.coordinate_frame",
            ),
            instruction_required=instruction_required,
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "ObservationSpec.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ObservationPacket:
    observation_id: str
    spec: ObservationSpec
    step_index: int
    values: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    instruction: str | None = None
    timestamp_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(
            self.observation_id,
            "ObservationPacket.observation_id",
        )
        if not isinstance(self.spec, ObservationSpec):
            raise ContractValidationError(
                "ObservationPacket.spec must be an ObservationSpec"
            )
        non_negative_int(self.step_index, "ObservationPacket.step_index")
        values = normalized_json_object(
            self.values,
            "ObservationPacket.values",
        )
        if not isinstance(self.artifacts, dict):
            raise ContractValidationError(
                "ObservationPacket.artifacts must be an object"
            )
        artifacts: dict[str, ArtifactRef] = {}
        for key, artifact in self.artifacts.items():
            channel = non_empty_string(
                key,
                "ObservationPacket.artifacts key",
            )
            if not isinstance(artifact, ArtifactRef):
                raise ContractValidationError(
                    f"ObservationPacket.artifacts.{channel} must be an ArtifactRef"
                )
            artifacts[channel] = artifact
        overlap = set(values) & set(artifacts)
        if overlap:
            raise ContractValidationError(
                "ObservationPacket channels cannot appear in both values and "
                "artifacts: "
                + ", ".join(sorted(overlap))
            )
        provided = set(values) | set(artifacts)
        allowed = set(self.spec.channels)
        unknown = provided - allowed
        if unknown:
            raise ContractValidationError(
                "ObservationPacket contains channels not declared by spec: "
                + ", ".join(sorted(unknown))
            )
        missing = set(self.spec.required_channels) - provided
        if missing:
            raise ContractValidationError(
                "ObservationPacket is missing required channels: "
                + ", ".join(sorted(missing))
            )
        instruction = optional_string(
            self.instruction,
            "ObservationPacket.instruction",
        )
        if self.spec.instruction_required and instruction is None:
            raise ContractValidationError(
                "ObservationPacket.instruction is required by its spec"
            )
        timestamp = self.timestamp_sec
        if timestamp is not None:
            timestamp = finite_float(
                timestamp,
                "ObservationPacket.timestamp_sec",
            )
            if timestamp < 0:
                raise ContractValidationError(
                    "ObservationPacket.timestamp_sec must be non-negative"
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "timestamp_sec", timestamp)
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "ObservationPacket.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ObservationPacket",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "spec": self.spec.to_dict(),
            "step_index": self.step_index,
            "values": json_copy(self.values),
            "artifacts": {
                key: value.to_dict()
                for key, value in self.artifacts.items()
            },
            "instruction": self.instruction,
            "timestamp_sec": self.timestamp_sec,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ObservationPacket":
        data = mapping(value, "ObservationPacket")
        check_schema_version(data, cls.SCHEMA_VERSION, "ObservationPacket")
        raw_artifacts = mapping(
            data.get("artifacts", {}),
            "ObservationPacket.artifacts",
        )
        return cls(
            observation_id=non_empty_string(
                required(
                    data,
                    "observation_id",
                    "ObservationPacket",
                ),
                "ObservationPacket.observation_id",
            ),
            spec=ObservationSpec.from_dict(
                required(data, "spec", "ObservationPacket")
            ),
            step_index=non_negative_int(
                required(data, "step_index", "ObservationPacket"),
                "ObservationPacket.step_index",
            ),
            values=normalized_json_object(
                data.get("values", {}),
                "ObservationPacket.values",
            ),
            artifacts={
                non_empty_string(
                    key,
                    "ObservationPacket.artifacts key",
                ): ArtifactRef.from_dict(item)
                for key, item in raw_artifacts.items()
            },
            instruction=optional_string(
                data.get("instruction"),
                "ObservationPacket.instruction",
            ),
            timestamp_sec=(
                None
                if data.get("timestamp_sec") is None
                else finite_float(
                    data["timestamp_sec"],
                    "ObservationPacket.timestamp_sec",
                )
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "ObservationPacket.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


class ActionType(str, Enum):
    EE_DELTA_POSE = "ee_delta_pose"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_POSITION = "joint_position"
    TRAJECTORY = "trajectory"


def _action_type(value: Any, path: str) -> ActionType:
    if isinstance(value, ActionType):
        return value
    try:
        return ActionType(non_empty_string(value, path))
    except ValueError as exc:
        choices = ", ".join(item.value for item in ActionType)
        raise ContractValidationError(
            f"{path} must be one of: {choices}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_type: ActionType
    frame: str
    units: str
    frequency_hz: float
    joint_names: tuple[str, ...]
    gripper_convention: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        action_type = _action_type(
            self.action_type,
            "ActionSpec.action_type",
        )
        object.__setattr__(self, "action_type", action_type)
        non_empty_string(self.frame, "ActionSpec.frame")
        non_empty_string(self.units, "ActionSpec.units")
        object.__setattr__(
            self,
            "frequency_hz",
            positive_float(
                self.frequency_hz,
                "ActionSpec.frequency_hz",
            ),
        )
        joint_names = string_tuple(
            self.joint_names,
            "ActionSpec.joint_names",
        )
        if action_type == ActionType.EE_DELTA_POSE and joint_names:
            raise ContractValidationError(
                "ActionSpec.joint_names must be empty for ee_delta_pose"
            )
        if (
            action_type
            in {
                ActionType.JOINT_VELOCITY,
                ActionType.JOINT_POSITION,
                ActionType.TRAJECTORY,
            }
            and not joint_names
        ):
            raise ContractValidationError(
                f"ActionSpec.joint_names is required for {action_type.value}"
            )
        object.__setattr__(self, "joint_names", joint_names)
        non_empty_string(
            self.gripper_convention,
            "ActionSpec.gripper_convention",
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "ActionSpec.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ActionSpec",
        )

    @property
    def value_dimension(self) -> int:
        if self.action_type == ActionType.EE_DELTA_POSE:
            return 6
        return len(self.joint_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_type": self.action_type.value,
            "frame": self.frame,
            "units": self.units,
            "frequency_hz": self.frequency_hz,
            "joint_names": list(self.joint_names),
            "gripper_convention": self.gripper_convention,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ActionSpec":
        data = mapping(value, "ActionSpec")
        check_schema_version(data, cls.SCHEMA_VERSION, "ActionSpec")
        return cls(
            action_type=_action_type(
                required(data, "action_type", "ActionSpec"),
                "ActionSpec.action_type",
            ),
            frame=non_empty_string(
                required(data, "frame", "ActionSpec"),
                "ActionSpec.frame",
            ),
            units=non_empty_string(
                required(data, "units", "ActionSpec"),
                "ActionSpec.units",
            ),
            frequency_hz=positive_float(
                required(data, "frequency_hz", "ActionSpec"),
                "ActionSpec.frequency_hz",
            ),
            joint_names=string_tuple(
                data.get("joint_names", []),
                "ActionSpec.joint_names",
            ),
            gripper_convention=non_empty_string(
                required(
                    data,
                    "gripper_convention",
                    "ActionSpec",
                ),
                "ActionSpec.gripper_convention",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "ActionSpec.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


ActionValues = (
    tuple[float, ...]
    | tuple[tuple[float, ...], ...]
)


@dataclass(frozen=True, slots=True)
class ActionPacket:
    action_id: str
    spec: ActionSpec
    values: ActionValues
    gripper: float | None
    timestamp_sec: float
    trajectory_time_offsets_sec: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.action_id, "ActionPacket.action_id")
        if not isinstance(self.spec, ActionSpec):
            raise ContractValidationError(
                "ActionPacket.spec must be an ActionSpec"
            )
        timestamp = finite_float(
            self.timestamp_sec,
            "ActionPacket.timestamp_sec",
        )
        if timestamp < 0:
            raise ContractValidationError(
                "ActionPacket.timestamp_sec must be non-negative"
            )
        object.__setattr__(self, "timestamp_sec", timestamp)

        if self.spec.action_type == ActionType.TRAJECTORY:
            if not isinstance(self.values, (list, tuple)):
                raise ContractValidationError(
                    "ActionPacket.values must be an array of trajectory waypoints"
                )
            waypoints = tuple(
                vector(
                    waypoint,
                    f"ActionPacket.values[{index}]",
                    length=self.spec.value_dimension,
                )
                for index, waypoint in enumerate(self.values)
            )
            if not waypoints:
                raise ContractValidationError(
                    "ActionPacket.values must contain at least one trajectory waypoint"
                )
            offsets = vector(
                self.trajectory_time_offsets_sec,
                "ActionPacket.trajectory_time_offsets_sec",
                allow_empty=True,
            )
            if not offsets:
                offsets = tuple(
                    index / self.spec.frequency_hz
                    for index in range(len(waypoints))
                )
            if len(offsets) != len(waypoints):
                raise ContractValidationError(
                    "ActionPacket.trajectory_time_offsets_sec must contain one "
                    "value per trajectory waypoint"
                )
            if any(offset < 0 for offset in offsets):
                raise ContractValidationError(
                    "ActionPacket.trajectory_time_offsets_sec must be non-negative"
                )
            if any(
                current <= previous
                for previous, current in zip(offsets, offsets[1:])
            ):
                raise ContractValidationError(
                    "ActionPacket.trajectory_time_offsets_sec must be strictly increasing"
                )
            object.__setattr__(self, "values", waypoints)
            object.__setattr__(
                self,
                "trajectory_time_offsets_sec",
                offsets,
            )
        else:
            flat_values = vector(
                self.values,
                "ActionPacket.values",
                length=self.spec.value_dimension,
            )
            if self.trajectory_time_offsets_sec:
                raise ContractValidationError(
                    "ActionPacket.trajectory_time_offsets_sec is only valid for trajectory actions"
                )
            object.__setattr__(self, "values", flat_values)
            object.__setattr__(
                self,
                "trajectory_time_offsets_sec",
                (),
            )

        if self.spec.gripper_convention == "none":
            if self.gripper is not None:
                raise ContractValidationError(
                    "ActionPacket.gripper must be omitted when gripper_convention is 'none'"
                )
        elif self.gripper is None:
            raise ContractValidationError(
                "ActionPacket.gripper is required by its gripper_convention"
            )
        else:
            object.__setattr__(
                self,
                "gripper",
                finite_float(self.gripper, "ActionPacket.gripper"),
            )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "ActionPacket.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ActionPacket",
        )

    def to_dict(self) -> dict[str, Any]:
        if self.spec.action_type == ActionType.TRAJECTORY:
            values: list[Any] = [
                list(waypoint) for waypoint in self.values
            ]
        else:
            values = list(self.values)
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "spec": self.spec.to_dict(),
            "values": values,
            "gripper": self.gripper,
            "timestamp_sec": self.timestamp_sec,
            "trajectory_time_offsets_sec": list(
                self.trajectory_time_offsets_sec
            ),
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ActionPacket":
        data = mapping(value, "ActionPacket")
        check_schema_version(data, cls.SCHEMA_VERSION, "ActionPacket")
        spec = ActionSpec.from_dict(
            required(data, "spec", "ActionPacket")
        )
        raw_values = required(data, "values", "ActionPacket")
        if spec.action_type == ActionType.TRAJECTORY:
            if not isinstance(raw_values, (list, tuple)):
                raise ContractValidationError(
                    "ActionPacket.values must be an array of trajectory waypoints"
                )
            parsed_values: ActionValues = tuple(
                vector(
                    waypoint,
                    f"ActionPacket.values[{index}]",
                    length=spec.value_dimension,
                )
                for index, waypoint in enumerate(raw_values)
            )
        else:
            parsed_values = vector(
                raw_values,
                "ActionPacket.values",
                length=spec.value_dimension,
            )
        raw_gripper = data.get("gripper")
        return cls(
            action_id=non_empty_string(
                required(data, "action_id", "ActionPacket"),
                "ActionPacket.action_id",
            ),
            spec=spec,
            values=parsed_values,
            gripper=(
                None
                if raw_gripper is None
                else finite_float(
                    raw_gripper,
                    "ActionPacket.gripper",
                )
            ),
            timestamp_sec=finite_float(
                required(data, "timestamp_sec", "ActionPacket"),
                "ActionPacket.timestamp_sec",
            ),
            trajectory_time_offsets_sec=vector(
                data.get("trajectory_time_offsets_sec", []),
                "ActionPacket.trajectory_time_offsets_sec",
                allow_empty=True,
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "ActionPacket.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )
