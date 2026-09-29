from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from i2ia.contracts import TaskPredicate, TaskSpec


class TaskEvaluationError(ValueError):
    """Raised when a task predicate cannot be interpreted."""


class _MissingObservation(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class PredicateEvidence:
    """Structured evidence for one predicate evaluation."""

    predicate_id: str
    predicate_type: str
    passed: bool
    observed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    children: tuple["PredicateEvidence", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "predicate_type": self.predicate_type,
            "passed": self.passed,
            "observed": _jsonable(self.observed),
            "expected": _jsonable(self.expected),
            "details": _jsonable(self.details),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    """Result and predicate-level evidence for a complete task."""

    task_id: str
    instruction: str
    passed: bool
    success_mode: str
    predicates: tuple[PredicateEvidence, ...]
    step: int | None = None
    max_steps: int | None = None
    missing_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "passed": self.passed,
            "success_mode": self.success_mode,
            "step": self.step,
            "max_steps": self.max_steps,
            "missing_capabilities": list(self.missing_capabilities),
            "predicates": [predicate.to_dict() for predicate in self.predicates],
        }


class TaskEvaluator:
    """Evaluate a :class:`TaskSpec` against simulator-neutral state snapshots.

    State is intentionally a JSON-like mapping rather than an Isaac type. The
    canonical keys consumed by the primitive predicates are:

    ``bodies``
        ``{body_id: {"position": [x, y, z]}}``
    ``joints``
        ``{joint_id: number}`` or ``{joint_id: {"position": number}}``
    ``contacts``
        An array of ``{"body_a": id, "body_b": id, "active": bool}``
    ``regions``
        ``{region_id: {"min": [x, y, z], "max": [x, y, z]}}``
    ``step`` / ``timestamp_s``
        Optional temporal coordinates.

    ``scalar_compare`` may address any nested value using a dotted ``path``.
    The ``metric`` alias resolves a top-level field first and then
    ``metrics.<metric>`` for compatibility with metric-producing simulators.
    Common aliases for positions, joint values, bounds, and step indices are
    accepted so simulator adapters can stay thin.

    ``history`` contains earlier snapshots. Supplying the current snapshot as
    its final item is also supported. ``initial_state`` is the reference for
    displacement and joint-change predicates; when omitted, the first timeline
    snapshot is used.
    """

    def evaluate(
        self,
        task: TaskSpec | Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        initial_state: Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        capabilities: Iterable[str] | None = None,
    ) -> TaskEvaluation:
        task_data = _as_mapping(task, "task")
        current = _as_mapping(state, "state")
        timeline = [_as_mapping(item, f"history[{index}]") for index, item in enumerate(history)]
        if not timeline or not _same_snapshot(timeline[-1], current):
            timeline.append(current)
        reference = (
            _as_mapping(initial_state, "initial_state")
            if initial_state is not None
            else timeline[0]
        )

        raw_predicates = task_data.get("predicates")
        if not isinstance(raw_predicates, Sequence) or isinstance(
            raw_predicates, (str, bytes, bytearray)
        ) or not raw_predicates:
            raise TaskEvaluationError("task.predicates must be a non-empty array")
        evidence = tuple(
            self._evaluate_predicate(
                _as_mapping(predicate, f"task.predicates[{index}]"),
                current=current,
                initial=reference,
                timeline=timeline,
                apply_hold=True,
            )
            for index, predicate in enumerate(raw_predicates)
        )

        success_mode = str(task_data.get("success_mode", "all"))
        if success_mode == "all":
            predicate_success = all(item.passed for item in evidence)
        elif success_mode == "any":
            predicate_success = any(item.passed for item in evidence)
        else:
            raise TaskEvaluationError(
                f"task.success_mode must be 'all' or 'any', got {success_mode!r}"
            )

        required = _string_sequence(
            task_data.get("required_capabilities", ()),
            "task.required_capabilities",
        )
        available = None if capabilities is None else {str(value) for value in capabilities}
        missing = (
            ()
            if available is None
            else tuple(capability for capability in required if capability not in available)
        )
        step = _optional_step(current)
        raw_max_steps = task_data.get("max_steps")
        max_steps = None if raw_max_steps is None else _positive_int(raw_max_steps, "task.max_steps")
        within_horizon = max_steps is None or step is None or step <= max_steps

        return TaskEvaluation(
            task_id=_required_string(task_data, "task_id", "task"),
            instruction=_required_string(task_data, "instruction", "task"),
            passed=predicate_success and not missing and within_horizon,
            success_mode=success_mode,
            predicates=evidence,
            step=step,
            max_steps=max_steps,
            missing_capabilities=missing,
        )

    def _evaluate_predicate(
        self,
        predicate: Mapping[str, Any],
        *,
        current: Mapping[str, Any],
        initial: Mapping[str, Any],
        timeline: Sequence[Mapping[str, Any]],
        apply_hold: bool,
    ) -> PredicateEvidence:
        predicate_id = _required_string(predicate, "predicate_id", "predicate")
        predicate_type = _required_string(predicate, "predicate_type", "predicate")
        parameters = _as_mapping(predicate.get("parameters"), f"{predicate_id}.parameters")
        hold_steps = _positive_int(predicate.get("hold_steps", 1), f"{predicate_id}.hold_steps")

        instantaneous = self._evaluate_once(
            predicate_id,
            predicate_type,
            parameters,
            current=current,
            initial=initial,
            timeline=timeline,
        )
        if not apply_hold or hold_steps == 1:
            return instantaneous

        sample_states = list(timeline[-hold_steps:])
        samples = tuple(
            self._evaluate_once(
                predicate_id,
                predicate_type,
                parameters,
                current=sample,
                initial=initial,
                timeline=timeline[: len(timeline) - len(sample_states) + index + 1],
            )
            for index, sample in enumerate(sample_states)
        )
        consecutive = 0
        for sample in reversed(samples):
            if not sample.passed:
                break
            consecutive += 1
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type=predicate_type,
            passed=len(sample_states) == hold_steps and consecutive == hold_steps,
            observed={
                "available_steps": len(sample_states),
                "consecutive_passing_steps": consecutive,
            },
            expected={"hold_steps": hold_steps},
            details={"instantaneous_passed": instantaneous.passed},
            children=samples,
        )

    def _evaluate_once(
        self,
        predicate_id: str,
        predicate_type: str,
        parameters: Mapping[str, Any],
        *,
        current: Mapping[str, Any],
        initial: Mapping[str, Any],
        timeline: Sequence[Mapping[str, Any]],
    ) -> PredicateEvidence:
        handlers = {
            "scalar_compare": self._scalar_compare,
            "displacement": self._displacement,
            "relative_pose": self._relative_pose,
            "joint_change": self._joint_change,
            "contact": self._contact,
            "region_contains": self._region_contains,
            "all": self._all,
            "any": self._any,
            "not": self._not,
            "duration": self._duration,
        }
        handler = handlers.get(predicate_type)
        if handler is None:
            raise TaskEvaluationError(
                f"Unsupported predicate_type {predicate_type!r} for {predicate_id!r}"
            )
        try:
            return handler(
                predicate_id,
                parameters,
                current=current,
                initial=initial,
                timeline=timeline,
            )
        except _MissingObservation as exc:
            return PredicateEvidence(
                predicate_id=predicate_id,
                predicate_type=predicate_type,
                passed=False,
                details={"error": str(exc), "failure_domain": "observation"},
            )

    def _scalar_compare(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        raw_path = parameters.get("path")
        raw_metric = parameters.get("metric")
        if raw_path is None and raw_metric is None:
            raise TaskEvaluationError(
                f"{predicate_id}.parameters requires path or metric"
            )
        source_key = "path" if raw_path is not None else "metric"
        path = _required_string(
            {source_key: raw_path if raw_path is not None else raw_metric},
            source_key,
            f"{predicate_id}.parameters",
        )
        candidate_paths = [path]
        if source_key == "metric" and "." not in path:
            candidate_paths.append(f"metrics.{path}")
        value_at_path: Any | None = None
        resolved_path: str | None = None
        missing: _MissingObservation | None = None
        for candidate in candidate_paths:
            try:
                value_at_path = _get_path(context["current"], candidate)
                resolved_path = candidate
                break
            except _MissingObservation as exc:
                missing = exc
        if resolved_path is None:
            assert missing is not None
            raise missing
        value = _finite_number(
            value_at_path,
            f"state.{resolved_path}",
        )
        return _comparison_evidence(
            predicate_id,
            "scalar_compare",
            value,
            parameters,
            observed={
                source_key: path,
                "resolved_path": resolved_path,
                "value": value,
            },
        )

    def _displacement(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        body_id = _required_string(parameters, "body_id", f"{predicate_id}.parameters")
        current_position = _body_position(context["current"], body_id)
        if "reference_position" in parameters:
            reference_position = _vector3(
                parameters["reference_position"],
                f"{predicate_id}.parameters.reference_position",
            )
        else:
            reference_position = _body_position(context["initial"], body_id)
        delta = tuple(
            current_position[index] - reference_position[index] for index in range(3)
        )
        axis = parameters.get("axis")
        if axis is None:
            measured = _norm(delta)
            axis_description: Any = "3d"
        else:
            axis_vector, axis_description = _axis_vector(axis, f"{predicate_id}.parameters.axis")
            measured = sum(delta[index] * axis_vector[index] for index in range(3))
            if bool(parameters.get("absolute", False)):
                measured = abs(measured)
        return _comparison_evidence(
            predicate_id,
            "displacement",
            measured,
            parameters,
            observed={
                "body_id": body_id,
                "reference_position": list(reference_position),
                "current_position": list(current_position),
                "delta": list(delta),
                "displacement": measured,
                "axis": axis_description,
            },
        )

    def _relative_pose(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        body_a = _required_string(parameters, "body_a", f"{predicate_id}.parameters")
        body_b = _required_string(parameters, "body_b", f"{predicate_id}.parameters")
        position_a = _body_position(context["current"], body_a)
        position_b = _body_position(context["current"], body_b)
        translation = tuple(position_a[index] - position_b[index] for index in range(3))
        target = parameters.get("target_translation")
        if target is None:
            error = translation
            target_translation = (0.0, 0.0, 0.0)
        else:
            target_translation = _vector3(
                target,
                f"{predicate_id}.parameters.target_translation",
            )
            error = tuple(
                translation[index] - target_translation[index] for index in range(3)
            )
        distance = _norm(error)
        return _comparison_evidence(
            predicate_id,
            "relative_pose",
            distance,
            parameters,
            observed={
                "body_a": body_a,
                "body_b": body_b,
                "relative_translation": list(translation),
                "target_translation": list(target_translation),
                "translation_distance": distance,
            },
        )

    def _joint_change(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        joint_id = _required_string(parameters, "joint_id", f"{predicate_id}.parameters")
        current_value = _joint_value(context["current"], joint_id)
        if "reference_value" in parameters:
            reference_value = _finite_number(
                parameters["reference_value"],
                f"{predicate_id}.parameters.reference_value",
            )
        else:
            reference_value = _joint_value(context["initial"], joint_id)
        delta = current_value - reference_value
        measured = abs(delta) if bool(parameters.get("absolute", True)) else delta
        return _comparison_evidence(
            predicate_id,
            "joint_change",
            measured,
            parameters,
            observed={
                "joint_id": joint_id,
                "reference_value": reference_value,
                "current_value": current_value,
                "delta": delta,
                "measured_change": measured,
            },
        )

    def _contact(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        body_a = _required_string(parameters, "body_a", f"{predicate_id}.parameters")
        body_b = _required_string(parameters, "body_b", f"{predicate_id}.parameters")
        expected = parameters.get("expected", True)
        if not isinstance(expected, bool):
            raise TaskEvaluationError(f"{predicate_id}.parameters.expected must be boolean")
        active = _contact_active(context["current"], body_a, body_b)
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="contact",
            passed=active is expected,
            observed={"body_a": body_a, "body_b": body_b, "active": active},
            expected={"active": expected},
        )

    def _region_contains(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        if "body_id" in parameters:
            body_id = _required_string(parameters, "body_id", f"{predicate_id}.parameters")
            point = _body_position(context["current"], body_id)
            point_source = {"body_id": body_id}
        elif "point_path" in parameters:
            path = _required_string(parameters, "point_path", f"{predicate_id}.parameters")
            point = _vector3(_get_path(context["current"], path), f"state.{path}")
            point_source = {"point_path": path}
        else:
            raise TaskEvaluationError(
                f"{predicate_id}.parameters requires body_id or point_path"
            )
        if "region_id" in parameters:
            region_id = _required_string(parameters, "region_id", f"{predicate_id}.parameters")
            lower, upper = _region_bounds(context["current"], region_id)
            region_source: dict[str, Any] = {"region_id": region_id}
        else:
            lower = _vector3(
                parameters.get("min"),
                f"{predicate_id}.parameters.min",
            )
            upper = _vector3(
                parameters.get("max"),
                f"{predicate_id}.parameters.max",
            )
            region_source = {"region_id": None}
        if any(lower[index] > upper[index] for index in range(3)):
            raise TaskEvaluationError(f"{predicate_id} region min must not exceed max")
        margin = _finite_number(parameters.get("margin", 0.0), f"{predicate_id}.parameters.margin")
        if margin < 0:
            raise TaskEvaluationError(f"{predicate_id}.parameters.margin must be non-negative")
        contained = all(
            lower[index] - margin <= point[index] <= upper[index] + margin
            for index in range(3)
        )
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="region_contains",
            passed=contained,
            observed={
                **point_source,
                **region_source,
                "point": list(point),
            },
            expected={
                "min": list(lower),
                "max": list(upper),
                "margin": margin,
            },
        )

    def _all(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        children = self._composite_children(predicate_id, parameters, context)
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="all",
            passed=all(child.passed for child in children),
            observed={"passing_children": sum(child.passed for child in children)},
            expected={"required_children": len(children)},
            children=children,
        )

    def _any(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        children = self._composite_children(predicate_id, parameters, context)
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="any",
            passed=any(child.passed for child in children),
            observed={"passing_children": sum(child.passed for child in children)},
            expected={"minimum_passing_children": 1},
            children=children,
        )

    def _not(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        child_value = parameters.get("predicate", parameters.get("child"))
        child = self._evaluate_predicate(
            _as_mapping(child_value, f"{predicate_id}.parameters.predicate"),
            current=context["current"],
            initial=context["initial"],
            timeline=context["timeline"],
            apply_hold=True,
        )
        observation_failure = _find_observation_failure(child)
        if observation_failure is not None:
            return PredicateEvidence(
                predicate_id=predicate_id,
                predicate_type="not",
                passed=False,
                observed={"child_passed": child.passed},
                expected={"child_passed": False},
                details={
                    "failure_domain": "observation",
                    "error": observation_failure,
                },
                children=(child,),
            )
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="not",
            passed=not child.passed,
            observed={"child_passed": child.passed},
            expected={"child_passed": False},
            children=(child,),
        )

    def _duration(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        **context: Any,
    ) -> PredicateEvidence:
        child_value = parameters.get("predicate", parameters.get("child"))
        child_predicate = _as_mapping(
            child_value,
            f"{predicate_id}.parameters.predicate",
        )
        timeline = context["timeline"]
        if "steps" in parameters:
            required_steps = _positive_int(
                parameters["steps"],
                f"{predicate_id}.parameters.steps",
            )
            samples = list(timeline[-required_steps:])
            evidence = tuple(
                self._evaluate_predicate(
                    child_predicate,
                    current=sample,
                    initial=context["initial"],
                    timeline=timeline[: len(timeline) - len(samples) + index + 1],
                    apply_hold=True,
                )
                for index, sample in enumerate(samples)
            )
            passed = len(samples) == required_steps and all(item.passed for item in evidence)
            return PredicateEvidence(
                predicate_id=predicate_id,
                predicate_type="duration",
                passed=passed,
                observed={
                    "available_steps": len(samples),
                    "passing_steps": sum(item.passed for item in evidence),
                },
                expected={"steps": required_steps},
                children=evidence,
            )
        if "duration_s" not in parameters:
            raise TaskEvaluationError(
                f"{predicate_id}.parameters requires steps or duration_s"
            )
        required_duration = _finite_number(
            parameters["duration_s"],
            f"{predicate_id}.parameters.duration_s",
        )
        if required_duration <= 0:
            raise TaskEvaluationError(
                f"{predicate_id}.parameters.duration_s must be greater than zero"
            )
        current_timestamp = _timestamp(context["current"])
        samples: list[Mapping[str, Any]] = []
        sample_evidence: list[PredicateEvidence] = []
        for reverse_index, sample in enumerate(reversed(timeline)):
            evidence = self._evaluate_predicate(
                child_predicate,
                current=sample,
                initial=context["initial"],
                timeline=timeline[: len(timeline) - reverse_index],
                apply_hold=True,
            )
            if not evidence.passed:
                break
            samples.append(sample)
            sample_evidence.append(evidence)
        samples.reverse()
        sample_evidence.reverse()
        elapsed = (
            0.0
            if not samples
            else current_timestamp - _timestamp(samples[0])
        )
        return PredicateEvidence(
            predicate_id=predicate_id,
            predicate_type="duration",
            passed=bool(samples) and elapsed >= required_duration,
            observed={
                "continuous_duration_s": elapsed,
                "passing_samples": len(samples),
            },
            expected={"duration_s": required_duration},
            children=tuple(sample_evidence),
        )

    def _composite_children(
        self,
        predicate_id: str,
        parameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> tuple[PredicateEvidence, ...]:
        raw_children = parameters.get("predicates", parameters.get("children"))
        if not isinstance(raw_children, Sequence) or isinstance(
            raw_children, (str, bytes, bytearray)
        ) or not raw_children:
            raise TaskEvaluationError(
                f"{predicate_id}.parameters.predicates must be a non-empty array"
            )
        return tuple(
            self._evaluate_predicate(
                _as_mapping(child, f"{predicate_id}.parameters.predicates[{index}]"),
                current=context["current"],
                initial=context["initial"],
                timeline=context["timeline"],
                apply_hold=True,
            )
            for index, child in enumerate(raw_children)
        )


def evaluate_task(
    task: TaskSpec | Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    initial_state: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] = (),
    capabilities: Iterable[str] | None = None,
) -> TaskEvaluation:
    """Convenience wrapper around :class:`TaskEvaluator`."""

    return TaskEvaluator().evaluate(
        task,
        state,
        initial_state=initial_state,
        history=history,
        capabilities=capabilities,
    )


def _comparison_evidence(
    predicate_id: str,
    predicate_type: str,
    measured: float,
    parameters: Mapping[str, Any],
    *,
    observed: Mapping[str, Any],
) -> PredicateEvidence:
    raw_expected = parameters.get("value", parameters.get("threshold"))
    if raw_expected is None:
        raise TaskEvaluationError(
            f"{predicate_id}.parameters requires value or threshold"
        )
    expected = _finite_number(raw_expected, f"{predicate_id}.parameters.value")
    operator = str(parameters.get("operator", parameters.get("comparison", ">=")))
    tolerance = _finite_number(
        parameters.get("tolerance", 0.0),
        f"{predicate_id}.parameters.tolerance",
    )
    if tolerance < 0:
        raise TaskEvaluationError(
            f"{predicate_id}.parameters.tolerance must be non-negative"
        )
    passed = _compare(measured, operator, expected, tolerance)
    return PredicateEvidence(
        predicate_id=predicate_id,
        predicate_type=predicate_type,
        passed=passed,
        observed=dict(observed),
        expected={
            "operator": _canonical_operator(operator),
            "value": expected,
            "tolerance": tolerance,
        },
    )


def _compare(observed: float, operator: str, expected: float, tolerance: float) -> bool:
    canonical = _canonical_operator(operator)
    if canonical == ">":
        return observed > expected
    if canonical == ">=":
        return observed >= expected
    if canonical == "<":
        return observed < expected
    if canonical == "<=":
        return observed <= expected
    equal = math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance)
    return equal if canonical == "==" else not equal


def _canonical_operator(operator: str) -> str:
    aliases = {
        ">": ">",
        "gt": ">",
        ">=": ">=",
        "ge": ">=",
        "gte": ">=",
        "<": "<",
        "lt": "<",
        "<=": "<=",
        "le": "<=",
        "lte": "<=",
        "==": "==",
        "eq": "==",
        "!=": "!=",
        "ne": "!=",
    }
    try:
        return aliases[operator.lower()]
    except KeyError as exc:
        raise TaskEvaluationError(f"Unsupported comparison operator: {operator!r}") from exc


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TaskEvaluationError(f"{path} must be a mapping or expose to_dict()")


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise TaskEvaluationError(f"{path}.{key} must be a non-empty string")
    return result


def _string_sequence(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TaskEvaluationError(f"{path} must be an array of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise TaskEvaluationError(f"{path} must contain non-empty strings")
    return result


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TaskEvaluationError(f"{path} must be a positive integer")
    return value


def _finite_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TaskEvaluationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TaskEvaluationError(f"{path} must be a finite number")
    return result


def _vector3(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TaskEvaluationError(f"{path} must be a three-element vector")
    if len(value) != 3:
        raise TaskEvaluationError(f"{path} must be a three-element vector")
    return tuple(_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _axis_vector(value: Any, path: str) -> tuple[tuple[float, float, float], Any]:
    if isinstance(value, str):
        axes = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "-x": (-1.0, 0.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "-z": (0.0, 0.0, -1.0),
        }
        try:
            return axes[value.lower()], value.lower()
        except KeyError as exc:
            raise TaskEvaluationError(f"{path} must name x, y, z, -x, -y, or -z") from exc
    if isinstance(value, int) and not isinstance(value, bool):
        if value not in (0, 1, 2):
            raise TaskEvaluationError(f"{path} integer axis must be 0, 1, or 2")
        axis = tuple(1.0 if index == value else 0.0 for index in range(3))
        return axis, value  # type: ignore[return-value]
    vector = _vector3(value, path)
    length = _norm(vector)
    if length <= 1e-12:
        raise TaskEvaluationError(f"{path} must be non-zero")
    normalized = tuple(component / length for component in vector)
    return normalized, list(vector)  # type: ignore[return-value]


def _body_position(state: Mapping[str, Any], body_id: str) -> tuple[float, float, float]:
    bodies = state.get("bodies", state.get("body_positions"))
    if not isinstance(bodies, Mapping) or body_id not in bodies:
        raise _MissingObservation(f"state has no body pose for {body_id!r}")
    body = bodies[body_id]
    if isinstance(body, Mapping):
        pose = body.get("pose")
        candidates = (
            body.get("position"),
            body.get("position_m"),
            body.get("translation"),
            body.get("translation_m"),
            body.get("world_position"),
            body.get("world_position_m"),
            pose.get("translation") if isinstance(pose, Mapping) else None,
            pose.get("position") if isinstance(pose, Mapping) else None,
        )
        value = next((candidate for candidate in candidates if candidate is not None), None)
    else:
        value = body
    if value is None:
        raise _MissingObservation(f"body {body_id!r} has no translation")
    try:
        return _vector3(value, f"state.bodies.{body_id}.position")
    except TaskEvaluationError as exc:
        raise _MissingObservation(str(exc)) from exc


def _joint_value(state: Mapping[str, Any], joint_id: str) -> float:
    joints = state.get("joints", state.get("joint_positions"))
    if not isinstance(joints, Mapping) or joint_id not in joints:
        raise _MissingObservation(f"state has no joint value for {joint_id!r}")
    joint = joints[joint_id]
    if isinstance(joint, Mapping):
        value = next(
            (
                joint[key]
                for key in ("value", "position", "position_rad", "position_m")
                if key in joint
            ),
            None,
        )
    else:
        value = joint
    if value is None:
        raise _MissingObservation(f"joint {joint_id!r} has no position value")
    try:
        return _finite_number(value, f"state.joints.{joint_id}")
    except TaskEvaluationError as exc:
        raise _MissingObservation(str(exc)) from exc


def _contact_active(state: Mapping[str, Any], body_a: str, body_b: str) -> bool:
    contacts = state.get("contacts", state.get("contact_pairs", ()))
    wanted = frozenset((body_a, body_b))
    if isinstance(contacts, Mapping):
        for key in (f"{body_a}|{body_b}", f"{body_b}|{body_a}"):
            if key in contacts:
                value = contacts[key]
                return bool(value.get("active", True)) if isinstance(value, Mapping) else bool(value)
        nested = contacts.get(body_a)
        if isinstance(nested, Mapping) and body_b in nested:
            return bool(nested[body_b])
        return False
    if not isinstance(contacts, Sequence) or isinstance(
        contacts, (str, bytes, bytearray)
    ):
        raise _MissingObservation("state.contacts must be an array or mapping")
    for item in contacts:
        if isinstance(item, Mapping):
            first = item.get("body_a", item.get("a"))
            second = item.get("body_b", item.get("b"))
            pair = item.get("pair")
            if pair is not None and isinstance(pair, Sequence) and len(pair) == 2:
                first, second = pair
            active = bool(item.get("active", True))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
            first, second = item
            active = True
        else:
            continue
        if frozenset((str(first), str(second))) == wanted and active:
            return True
    return False


def _region_bounds(
    state: Mapping[str, Any],
    region_id: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    regions = state.get("regions")
    if not isinstance(regions, Mapping) or region_id not in regions:
        raise _MissingObservation(f"state has no region {region_id!r}")
    region = regions[region_id]
    if not isinstance(region, Mapping):
        raise _MissingObservation(f"region {region_id!r} must be a mapping")
    lower = region.get("min", region.get("min_xyz", region.get("lower")))
    upper = region.get("max", region.get("max_xyz", region.get("upper")))
    try:
        return (
            _vector3(lower, f"state.regions.{region_id}.min"),
            _vector3(upper, f"state.regions.{region_id}.max"),
        )
    except TaskEvaluationError as exc:
        raise _MissingObservation(str(exc)) from exc


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            try:
                current = current[int(component)]
                continue
            except (ValueError, IndexError):
                pass
        raise _MissingObservation(f"state path {path!r} is missing at {component!r}")
    return current


def _timestamp(state: Mapping[str, Any]) -> float:
    for key in ("timestamp_s", "time_s", "sim_time_s"):
        if key in state:
            try:
                return _finite_number(state[key], f"state.{key}")
            except TaskEvaluationError as exc:
                raise _MissingObservation(str(exc)) from exc
    raise _MissingObservation("state has no timestamp_s, time_s, or sim_time_s")


def _optional_step(state: Mapping[str, Any]) -> int | None:
    for key in ("step", "step_idx"):
        if key in state:
            value = state[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TaskEvaluationError(f"state.{key} must be a non-negative integer")
            return value
    return None


def _same_snapshot(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("step", "step_idx", "timestamp_s", "time_s", "sim_time_s"):
        if key in left and key in right:
            return left[key] == right[key]
    return left is right or left == right


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _find_observation_failure(evidence: PredicateEvidence) -> str | None:
    if evidence.details.get("failure_domain") == "observation":
        return str(evidence.details.get("error", "required observation is missing"))
    for child in evidence.children:
        failure = _find_observation_failure(child)
        if failure is not None:
            return failure
    return None
