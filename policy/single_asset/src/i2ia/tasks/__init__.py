"""Backend-neutral task evaluation."""

from .evaluator import (
    PredicateEvidence,
    TaskEvaluation,
    TaskEvaluationError,
    TaskEvaluator,
    evaluate_task,
)

__all__ = [
    "PredicateEvidence",
    "TaskEvaluation",
    "TaskEvaluationError",
    "TaskEvaluator",
    "evaluate_task",
]
