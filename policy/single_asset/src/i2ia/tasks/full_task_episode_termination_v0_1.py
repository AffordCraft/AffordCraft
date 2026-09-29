"""Common teacher/VLA terminal condition; no action or object writes."""
from .complete_robot_precision import evaluate

POLICY = "independent_full_task_predicate_at_control_boundary_v0_1"


def completed(spec, trace):
    if spec.get("episode_termination_policy") != POLICY:
        raise ValueError("missing_preregistered_episode_termination_policy")
    result = evaluate(spec, trace)
    return result["sim_task_success"] is True, result
