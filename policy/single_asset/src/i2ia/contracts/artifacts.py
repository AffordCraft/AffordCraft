from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from ._validation import (
    ContractValidationError,
    check_schema_version,
    json_copy,
    literal,
    mapping,
    non_empty_string,
    non_negative_int,
    normalized_json_object,
    optional_string,
    required,
    string_tuple,
    validate_schema_version,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KINDS = {"file", "directory"}
_SPATIAL_UP_AXES = {"X", "Y", "Z", "-X", "-Y", "-Z"}
_ARTIFACT_UP_AXES = {*_SPATIAL_UP_AXES, "NONE"}
_PERSISTENCE_PLACEHOLDERS = {
    "n/a",
    "na",
    "none",
    "unknown",
    "unversioned",
    "unspecified",
}


@dataclass(frozen=True, slots=True)
class ProducerIdentity:
    backend: str
    model: str
    version: str
    adapter_version: str | None = None
    capabilities: tuple[str, ...] = ()
    capability_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.backend, "ProducerIdentity.backend")
        non_empty_string(self.model, "ProducerIdentity.model")
        non_empty_string(self.version, "ProducerIdentity.version")
        optional_string(self.adapter_version, "ProducerIdentity.adapter_version")
        object.__setattr__(
            self,
            "capabilities",
            string_tuple(self.capabilities, "ProducerIdentity.capabilities"),
        )
        object.__setattr__(
            self,
            "capability_snapshot",
            normalized_json_object(
                self.capability_snapshot,
                "ProducerIdentity.capability_snapshot",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ProducerIdentity",
        )

    @classmethod
    def unknown(cls) -> "ProducerIdentity":
        return cls(backend="unknown", model="unknown", version="unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "model": self.model,
            "version": self.version,
            "adapter_version": self.adapter_version,
            "capabilities": list(self.capabilities),
            "capability_snapshot": json_copy(self.capability_snapshot),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ProducerIdentity":
        data = mapping(value, "ProducerIdentity")
        check_schema_version(data, cls.SCHEMA_VERSION, "ProducerIdentity")
        return cls(
            backend=non_empty_string(
                required(data, "backend", "ProducerIdentity"),
                "ProducerIdentity.backend",
            ),
            model=non_empty_string(
                required(data, "model", "ProducerIdentity"),
                "ProducerIdentity.model",
            ),
            version=non_empty_string(
                required(data, "version", "ProducerIdentity"),
                "ProducerIdentity.version",
            ),
            adapter_version=optional_string(
                data.get("adapter_version"),
                "ProducerIdentity.adapter_version",
            ),
            capabilities=string_tuple(
                data.get("capabilities", []),
                "ProducerIdentity.capabilities",
            ),
            capability_snapshot=normalized_json_object(
                data.get("capability_snapshot", {}),
                "ProducerIdentity.capability_snapshot",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    input_artifact_ids: tuple[str, ...] = ()
    run_id: str | None = None
    code_revision: str | None = None
    git_sha: str | None = None
    created_at: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_artifact_ids",
            string_tuple(
                self.input_artifact_ids,
                "Provenance.input_artifact_ids",
            ),
        )
        optional_string(self.run_id, "Provenance.run_id")
        optional_string(self.code_revision, "Provenance.code_revision")
        optional_string(self.git_sha, "Provenance.git_sha")
        optional_string(self.created_at, "Provenance.created_at")
        object.__setattr__(
            self,
            "parameters",
            normalized_json_object(self.parameters, "Provenance.parameters"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "Provenance",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_artifact_ids": list(self.input_artifact_ids),
            "run_id": self.run_id,
            "code_revision": self.code_revision,
            "git_sha": self.git_sha,
            "created_at": self.created_at,
            "parameters": json_copy(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "Provenance":
        data = mapping(value, "Provenance")
        check_schema_version(data, cls.SCHEMA_VERSION, "Provenance")
        return cls(
            input_artifact_ids=string_tuple(
                data.get("input_artifact_ids", []),
                "Provenance.input_artifact_ids",
            ),
            run_id=optional_string(data.get("run_id"), "Provenance.run_id"),
            code_revision=optional_string(
                data.get("code_revision"),
                "Provenance.code_revision",
            ),
            git_sha=optional_string(
                data.get("git_sha"),
                "Provenance.git_sha",
            ),
            created_at=optional_string(
                data.get("created_at"),
                "Provenance.created_at",
            ),
            parameters=normalized_json_object(
                data.get("parameters", {}),
                "Provenance.parameters",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    uri: str
    sha256: str
    size_bytes: int
    kind: str
    format: str
    path: str | None = None
    media_type: str | None = None
    producer: ProducerIdentity = field(default_factory=ProducerIdentity.unknown)
    provenance: Provenance = field(default_factory=Provenance)
    units: str | None = None
    up_axis: str | None = None
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        non_empty_string(self.artifact_id, "ArtifactRef.artifact_id")
        non_empty_string(self.uri, "ArtifactRef.uri")
        digest = non_empty_string(self.sha256, "ArtifactRef.sha256").lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ContractValidationError(
                "ArtifactRef.sha256 must be a 64-character lowercase hexadecimal digest"
            )
        object.__setattr__(self, "sha256", digest)
        non_negative_int(self.size_bytes, "ArtifactRef.size_bytes")
        literal(self.kind, _ARTIFACT_KINDS, "ArtifactRef.kind")
        non_empty_string(self.format, "ArtifactRef.format")
        optional_string(self.path, "ArtifactRef.path")
        optional_string(self.media_type, "ArtifactRef.media_type")
        if not isinstance(self.producer, ProducerIdentity):
            raise ContractValidationError(
                "ArtifactRef.producer must be a ProducerIdentity"
            )
        if not isinstance(self.provenance, Provenance):
            raise ContractValidationError(
                "ArtifactRef.provenance must be a Provenance"
            )
        optional_string(self.units, "ArtifactRef.units")
        if self.up_axis is not None:
            literal(self.up_axis, _ARTIFACT_UP_AXES, "ArtifactRef.up_axis")
        optional_string(self.coordinate_frame, "ArtifactRef.coordinate_frame")
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(self.metadata, "ArtifactRef.metadata"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "ArtifactRef",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "format": self.format,
            "media_type": self.media_type,
            "producer": self.producer.to_dict(),
            "provenance": self.provenance.to_dict(),
            "units": self.units,
            "up_axis": self.up_axis,
            "coordinate_frame": self.coordinate_frame,
            "metadata": json_copy(self.metadata),
        }

    def validate_persistence_complete(self) -> "ArtifactRef":
        """Reject metadata that is too incomplete for a persisted artifact."""

        if (
            self.path is None
            or self.path.strip().lower() in _PERSISTENCE_PLACEHOLDERS
        ):
            raise ContractValidationError(
                "ArtifactRef.path is required for persistence"
            )
        if self.format.strip().lower() in _PERSISTENCE_PLACEHOLDERS:
            raise ContractValidationError(
                "ArtifactRef.format must identify the persisted encoding"
            )
        for name, value in (
            ("producer.backend", self.producer.backend),
            ("producer.model", self.producer.model),
            ("producer.version", self.producer.version),
        ):
            if value.strip().lower() in _PERSISTENCE_PLACEHOLDERS:
                raise ContractValidationError(
                    f"ArtifactRef.{name} must identify the real producer"
                )
        if not self.producer.capability_snapshot:
            raise ContractValidationError(
                "ArtifactRef.producer.capability_snapshot must not be empty "
                "for persistence"
            )
        for name, value in (
            ("provenance.run_id", self.provenance.run_id),
            ("provenance.code_revision", self.provenance.code_revision),
            ("provenance.git_sha", self.provenance.git_sha),
            ("provenance.created_at", self.provenance.created_at),
        ):
            if value is None or value.strip().lower() in _PERSISTENCE_PLACEHOLDERS:
                raise ContractValidationError(
                    f"ArtifactRef.{name} is required for persistence"
                )
        assert self.provenance.git_sha is not None
        git_sha = self.provenance.git_sha.lower()
        if len(git_sha) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in git_sha
        ):
            raise ContractValidationError(
                "ArtifactRef.provenance.git_sha must be a 40- or "
                "64-character hexadecimal digest for persistence"
            )
        assert self.provenance.created_at is not None
        try:
            created_at = datetime.fromisoformat(
                self.provenance.created_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ContractValidationError(
                "ArtifactRef.provenance.created_at must be an ISO 8601 timestamp"
            ) from exc
        if created_at.tzinfo is None:
            raise ContractValidationError(
                "ArtifactRef.provenance.created_at must include a timezone"
            )
        for name, value in (
            ("units", self.units),
            ("up_axis", self.up_axis),
            ("coordinate_frame", self.coordinate_frame),
        ):
            if value is None:
                raise ContractValidationError(
                    f"ArtifactRef.{name} is required for persistence"
                )
        assert self.units is not None
        if self.units.strip().lower() in _PERSISTENCE_PLACEHOLDERS:
            raise ContractValidationError(
                "ArtifactRef.units must be explicit for persistence"
            )
        assert self.coordinate_frame is not None
        frame = self.coordinate_frame.strip().lower()
        if frame in _PERSISTENCE_PLACEHOLDERS and frame != "none":
            raise ContractValidationError(
                "ArtifactRef.coordinate_frame must be explicit for persistence"
            )
        if frame == "none" and self.up_axis != "NONE":
            raise ContractValidationError(
                "ArtifactRef.coordinate_frame 'none' requires up_axis 'NONE'"
            )
        return self

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactRef":
        data = mapping(value, "ArtifactRef")
        check_schema_version(data, cls.SCHEMA_VERSION, "ArtifactRef")
        return cls(
            artifact_id=non_empty_string(
                required(data, "artifact_id", "ArtifactRef"),
                "ArtifactRef.artifact_id",
            ),
            uri=non_empty_string(
                required(data, "uri", "ArtifactRef"),
                "ArtifactRef.uri",
            ),
            path=optional_string(data.get("path"), "ArtifactRef.path"),
            sha256=non_empty_string(
                required(data, "sha256", "ArtifactRef"),
                "ArtifactRef.sha256",
            ),
            size_bytes=non_negative_int(
                required(data, "size_bytes", "ArtifactRef"),
                "ArtifactRef.size_bytes",
            ),
            kind=literal(
                required(data, "kind", "ArtifactRef"),
                _ARTIFACT_KINDS,
                "ArtifactRef.kind",
            ),
            format=non_empty_string(
                required(data, "format", "ArtifactRef"),
                "ArtifactRef.format",
            ),
            media_type=optional_string(
                data.get("media_type"),
                "ArtifactRef.media_type",
            ),
            producer=ProducerIdentity.from_dict(
                required(data, "producer", "ArtifactRef")
            ),
            provenance=Provenance.from_dict(
                required(data, "provenance", "ArtifactRef")
            ),
            units=optional_string(data.get("units"), "ArtifactRef.units"),
            up_axis=optional_string(data.get("up_axis"), "ArtifactRef.up_axis"),
            coordinate_frame=optional_string(
                data.get("coordinate_frame"),
                "ArtifactRef.coordinate_frame",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "ArtifactRef.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class GeometryArtifact:
    mesh: ArtifactRef
    units: str
    up_axis: str
    coordinate_frame: str
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, ArtifactRef):
            raise ContractValidationError(
                "GeometryArtifact.mesh must be an ArtifactRef"
            )
        if self.mesh.kind != "file":
            raise ContractValidationError(
                "GeometryArtifact.mesh must reference a file"
            )
        non_empty_string(self.units, "GeometryArtifact.units")
        literal(self.up_axis, _SPATIAL_UP_AXES, "GeometryArtifact.up_axis")
        non_empty_string(
            self.coordinate_frame,
            "GeometryArtifact.coordinate_frame",
        )
        if self.coordinate_frame.strip().lower() in _PERSISTENCE_PLACEHOLDERS:
            raise ContractValidationError(
                "GeometryArtifact.coordinate_frame must identify a spatial frame"
            )
        if self.mesh.units is not None and self.mesh.units != self.units:
            raise ContractValidationError(
                "GeometryArtifact.units must match mesh.units when mesh.units is set"
            )
        if self.mesh.up_axis is not None and self.mesh.up_axis != self.up_axis:
            raise ContractValidationError(
                "GeometryArtifact.up_axis must match mesh.up_axis when mesh.up_axis is set"
            )
        if (
            self.mesh.coordinate_frame is not None
            and self.mesh.coordinate_frame != self.coordinate_frame
        ):
            raise ContractValidationError(
                "GeometryArtifact.coordinate_frame must match mesh.coordinate_frame "
                "when mesh.coordinate_frame is set"
            )
        object.__setattr__(
            self,
            "capabilities",
            string_tuple(self.capabilities, "GeometryArtifact.capabilities"),
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(self.metadata, "GeometryArtifact.metadata"),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "GeometryArtifact",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mesh": self.mesh.to_dict(),
            "units": self.units,
            "up_axis": self.up_axis,
            "coordinate_frame": self.coordinate_frame,
            "capabilities": list(self.capabilities),
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GeometryArtifact":
        data = mapping(value, "GeometryArtifact")
        check_schema_version(data, cls.SCHEMA_VERSION, "GeometryArtifact")
        return cls(
            mesh=ArtifactRef.from_dict(
                required(data, "mesh", "GeometryArtifact")
            ),
            units=non_empty_string(
                required(data, "units", "GeometryArtifact"),
                "GeometryArtifact.units",
            ),
            up_axis=literal(
                required(data, "up_axis", "GeometryArtifact"),
                _SPATIAL_UP_AXES,
                "GeometryArtifact.up_axis",
            ),
            coordinate_frame=non_empty_string(
                required(data, "coordinate_frame", "GeometryArtifact"),
                "GeometryArtifact.coordinate_frame",
            ),
            capabilities=string_tuple(
                data.get("capabilities", []),
                "GeometryArtifact.capabilities",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "GeometryArtifact.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )


_SEGMENTATION_REPRESENTATIONS = {
    "face_labels",
    "vertex_labels",
    "part_meshes",
    "masks",
}


@dataclass(frozen=True, slots=True)
class SegmentationArtifact:
    source_geometry: ArtifactRef
    representation: str
    artifacts: tuple[ArtifactRef, ...]
    segment_ids: tuple[str, ...]
    units: str
    coordinate_frame: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.source_geometry, ArtifactRef):
            raise ContractValidationError(
                "SegmentationArtifact.source_geometry must be an ArtifactRef"
            )
        literal(
            self.representation,
            _SEGMENTATION_REPRESENTATIONS,
            "SegmentationArtifact.representation",
        )
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ContractValidationError(
                "SegmentationArtifact.artifacts must not be empty"
            )
        if not all(isinstance(item, ArtifactRef) for item in artifacts):
            raise ContractValidationError(
                "SegmentationArtifact.artifacts must contain only ArtifactRef values"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "segment_ids",
            string_tuple(
                self.segment_ids,
                "SegmentationArtifact.segment_ids",
                allow_empty=False,
            ),
        )
        if (
            self.representation == "part_meshes"
            and len(self.artifacts) != len(self.segment_ids)
        ):
            raise ContractValidationError(
                "SegmentationArtifact.artifacts must have one entry per segment_id "
                "for part_meshes representation"
            )
        non_empty_string(self.units, "SegmentationArtifact.units")
        non_empty_string(
            self.coordinate_frame,
            "SegmentationArtifact.coordinate_frame",
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_json_object(
                self.metadata,
                "SegmentationArtifact.metadata",
            ),
        )
        validate_schema_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "SegmentationArtifact",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_geometry": self.source_geometry.to_dict(),
            "representation": self.representation,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "segment_ids": list(self.segment_ids),
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SegmentationArtifact":
        data = mapping(value, "SegmentationArtifact")
        check_schema_version(data, cls.SCHEMA_VERSION, "SegmentationArtifact")
        raw_artifacts = required(data, "artifacts", "SegmentationArtifact")
        if not isinstance(raw_artifacts, (list, tuple)):
            raise ContractValidationError(
                "SegmentationArtifact.artifacts must be an array"
            )
        return cls(
            source_geometry=ArtifactRef.from_dict(
                required(data, "source_geometry", "SegmentationArtifact")
            ),
            representation=literal(
                required(data, "representation", "SegmentationArtifact"),
                _SEGMENTATION_REPRESENTATIONS,
                "SegmentationArtifact.representation",
            ),
            artifacts=tuple(
                ArtifactRef.from_dict(item) for item in raw_artifacts
            ),
            segment_ids=string_tuple(
                required(data, "segment_ids", "SegmentationArtifact"),
                "SegmentationArtifact.segment_ids",
                allow_empty=False,
            ),
            units=non_empty_string(
                required(data, "units", "SegmentationArtifact"),
                "SegmentationArtifact.units",
            ),
            coordinate_frame=non_empty_string(
                required(data, "coordinate_frame", "SegmentationArtifact"),
                "SegmentationArtifact.coordinate_frame",
            ),
            metadata=normalized_json_object(
                data.get("metadata", {}),
                "SegmentationArtifact.metadata",
            ),
            schema_version=str(data["schema_version"]),
        )
