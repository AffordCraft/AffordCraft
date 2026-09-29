"""Versioned, backend-neutral contracts for the I2IA pipeline."""

from ._validation import ContractValidationError
from .artifacts import (
    ArtifactRef,
    GeometryArtifact,
    ProducerIdentity,
    Provenance,
    SegmentationArtifact,
)
from .packets import (
    ActionPacket,
    ActionSpec,
    ActionType,
    ObservationPacket,
    ObservationSpec,
)
from .scene import Joint, PhysicalSceneGraph, RigidBody
from .task import (
    SimulationAssetManifest,
    TaskPredicate,
    TaskSpec,
)

__all__ = [
    "ActionPacket",
    "ActionSpec",
    "ActionType",
    "ArtifactRef",
    "ContractValidationError",
    "GeometryArtifact",
    "Joint",
    "ObservationPacket",
    "ObservationSpec",
    "PhysicalSceneGraph",
    "ProducerIdentity",
    "Provenance",
    "RigidBody",
    "SegmentationArtifact",
    "SimulationAssetManifest",
    "TaskPredicate",
    "TaskSpec",
]
