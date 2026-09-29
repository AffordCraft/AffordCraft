from dataclasses import dataclass, asdict, field
from typing import Any, Literal
import hashlib, json, math


@dataclass(frozen=True)
class EvaluationPolicy:
    steps: int = 360
    dt: float = 1 / 120
    tail_steps: int = 60
    penetration_m: float = 0.001
    support_separation_m: float = 0.0001
    settling_translation_m: float = 0.05
    settling_rotation_deg: float = 5.0
    floor_tolerance_m: float = 0.012

    def __post_init__(self):
        if (self.steps, self.tail_steps) != (360, 60) or abs(self.dt - 1 / 120) > 1e-12:
            raise ValueError("Final evaluation timing is fixed, not optimized against outcomes")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SupportContract:
    role: Literal["free", "mounted"] = "free"
    root_motion_required: bool = False
    evidence: tuple[dict[str, Any], ...] = ()
    reason: str = "no_verified_mount_evidence"

    @classmethod
    def from_task_metadata(cls, metadata: dict[str, Any]):
        d = metadata.get("affordcraft_support", {})
        mobile = d.get("root_motion_required") is True
        evidence = tuple(d.get("evidence", []))
        # A model guess, category name or a kinematic flag cannot authorize a mount.
        approved = tuple(
            e
            for e in evidence
            if e.get("kind") in ("source_installation_metadata", "explicit_task_installation")
            and e.get("assertion") == "fixed_installation_required"
            and e.get("verified") is True
            and isinstance(e.get("artifact"), dict)
            and len(e["artifact"].get("sha256", "")) == 64
        )
        if mobile:
            return cls("free", True, evidence, "whole_object_motion_required")
        if d.get("role") == "mounted" and approved:
            return cls("mounted", False, approved, "verified_installation")
        return cls("free", False, evidence, "no_verified_mount_evidence")

    def task_metadata(self):
        return {"affordcraft_support": asdict(self)}


VARIANTS = (
    "full",
    "encoder_replacement",
    "without_task_condition",
    "without_multimodal_selection",
    "without_scale_adaptation",
    "without_articulation_adaptation",
    "without_physics_reselection",
    "without_repair",
    # the two later additions that answer the Window/Door/StorageFurniture failures
    "without_installation_evidence",
    "without_decomposition_cascade",
)


@dataclass(frozen=True)
class ConstructionPolicy:
    candidates: int = 20
    batch_size: int = 5
    repairs_per_candidate: int = 1
    minimum_confidence: float = 0.55
    minimum_geometry_score: float = 0.20
    seed: int = 20260916
    materialization_timeout_seconds: int = 600
    # One-factor paired study variant; 'full' is the frozen method. Every other value
    # removes exactly one component and is recorded in the frozen definition.
    variant: str = "full"

    def __post_init__(self):
        if not 1 <= self.candidates <= 20 or self.batch_size != 5 or self.repairs_per_candidate not in (0, 1):
            raise ValueError("Invalid construction budget")
        if self.variant not in VARIANTS:
            raise ValueError("Unknown study variant")
        if self.variant == "without_repair" and self.repairs_per_candidate != 0:
            raise ValueError("without_repair requires repairs_per_candidate=0")
        if self.variant != "without_repair" and self.repairs_per_candidate != 1:
            raise ValueError("Only without_repair changes the repair budget")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_definition(cls, d):
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def clean_model_input(case):
    """Positive allowlist: evaluation-only fields never enter perception/ranking."""
    clean = {k: case[k] for k in ("input_id", "instruction", "target_noun") if k in case}
    if "image" in case:
        clean["image"] = {k: case["image"][k] for k in ("path", "sha256") if k in case["image"]}
    return clean


def semantic_eligible(row, policy):
    vals = [row.get("confidence"), row.get("geometry_similarity")]
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
        return False
    return (
        row.get("category_compatible") is True
        and row.get("mechanism_compatible") is True
        and row.get("subtype_match") not in ("mismatch", None)
        and policy.minimum_confidence <= vals[0] <= 1
        and policy.minimum_geometry_score <= vals[1] <= 1
    )
