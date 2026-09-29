"""One trace evaluator for all support roles. This module does not run a simulator."""

import math
from .contracts import EvaluationPolicy, SupportContract


def finite_vector(v, n):
    return (
        isinstance(v, (list, tuple))
        and len(v) == n
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    )


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def distance(a, b):
    return norm([x - y for x, y in zip(a, b)])


def qangle(a, b):
    na, nb = norm(a), norm(b)
    if min(na, nb) < 1e-12:
        return math.inf
    return math.degrees(2 * math.acos(min(1, max(0, abs(sum(x * y for x, y in zip(a, b))) / (na * nb)))))


def late_samples(trace, policy):
    if not trace:
        return []
    end = trace[-1].get("simulation_seconds", policy.steps * policy.dt)
    return [
        s
        for s in trace
        if s.get("simulation_seconds", -1) > end - policy.tail_steps * policy.dt + 1e-9
        and s.get("simulation_seconds", math.inf) <= end + 1e-8
    ]


def body_inertia_valid(body):
    empty_frame = (
        body.get("role") == "coordinate_frame"
        and body.get("has_geometry") is False
        and body.get("source_declared_empty_static_frame") is True
        and body.get("kinematic") is True
    )
    if empty_frame:
        return True
    v = body.get("inertia_diagonal")
    mass = body.get("mass")
    return (
        finite_vector(v, 3)
        and min(v) > 0
        and 2 * max(v) <= sum(v) + 1e-9
        and isinstance(mass, (int, float))
        and math.isfinite(mass)
        and mass > 0
        and finite_vector(body.get("center_of_mass"), 3)
    )


class ContactLifecycle:
    def __init__(self):
        self.active = set()
        self.real_support = False
        self.initial_separations = []
        self.self_separations = []

    def feed(self, event):
        pair = tuple(sorted(event.get("colliders", [])))
        kind = event.get("kind")
        state = event.get("event_type")
        samples = event.get("samples", [])
        if kind == "support":
            if state == "CONTACT_LOST":
                self.active.discard(pair)
            elif state in ("CONTACT_FOUND", "CONTACT_PERSIST") and len(pair) == 2:
                self.active.add(pair)
            if len(pair) == 2 and any(
                math.isfinite(s.get("separation", math.nan)) and s["separation"] <= 0.0001 for s in samples
            ):
                self.real_support = True
        if 0 <= event.get("step", 999) <= 1:
            self.initial_separations.extend(s.get("separation", math.nan) for s in samples)
        if kind == "self":
            self.self_separations.extend(s.get("separation", math.nan) for s in samples)

    def supported(self):
        return bool(self.active)


def evaluate_trace(trace, static, contacts, support, policy=EvaluationPolicy()):
    if not isinstance(support, SupportContract):
        raise TypeError("SupportContract required")
    required = (
        "stage_load",
        "visual_geometry",
        "collision_geometry",
        "collision_enabled",
        "units_frames_valid",
        "source_geometry_preserved",
        "internal_joint_topology_preserved",
        "joint_frames_valid",
        "joint_limits_valid",
        "required_mobility_preserved",
        "gravity_enabled",
        "runtime_bodies_registered",
        "constraint_solver_consistency",
        "no_post_play_transform_writeback",
    )
    gates = {k: static.get(k) is True for k in required}
    bodies = static.get("bodies", [])
    physical = [b for b in bodies if b.get("role") != "coordinate_frame"]
    gates["physical_bodies_present"] = bool(physical)
    gates["mass_com_inertia"] = bool(bodies) and all(body_inertia_valid(b) for b in bodies)
    ids = {b["id"] for b in bodies}
    complete = len(trace) == policy.steps
    if complete:
        # Isaac stores dt as float32: 360 * float32(1/120) differs by 0.157 us.
        complete = all(
            s.get("step") == i + 1
            and abs(s.get("simulation_seconds", -1) - (i + 1) * policy.dt) < 1e-6
            and set(s.get("bodies", {})) == ids
            for i, s in enumerate(trace)
        )
    gates["complete_per_step_trace"] = complete
    finite = bool(trace) and all(
        finite_vector(b.get("position"), 3)
        and finite_vector(b.get("orientation_wxyz"), 4)
        and norm(b["orientation_wxyz"]) > 1e-12
        for s in trace
        for b in s.get("bodies", {}).values()
    )
    gates["finite_state"] = finite
    tail = late_samples(trace, policy)
    translation = rotation = math.inf
    if finite and tail and ids:
        translation = rotation = 0.0
        for bid in ids:
            final = trace[-1]["bodies"].get(bid)
            if final is None:
                translation = rotation = math.inf
                break
            for s in tail:
                b = s.get("bodies", {}).get(bid)
                if b is None:
                    translation = rotation = math.inf
                    break
                translation = max(translation, distance(b["position"], final["position"]))
                rotation = max(rotation, qangle(b["orientation_wxyz"], final["orientation_wxyz"]))
    gates["settle_stability"] = (
        translation <= policy.settling_translation_m and rotation <= policy.settling_rotation_deg
    )
    life = ContactLifecycle()
    for event in contacts:
        life.feed(event)
    clearance = static.get("initial_floor_clearance_m")
    initial_floor_ok = (
        isinstance(clearance, (int, float)) and math.isfinite(clearance) and clearance >= -policy.penetration_m
    )
    if static.get("support_plane_present") is False and support.role == "mounted":
        initial_floor_ok = True
    gates["initial_penetration"] = initial_floor_ok and all(
        math.isfinite(s) and s >= -policy.penetration_m for s in life.initial_separations
    )
    gates["self_penetration"] = all(math.isfinite(s) and s >= -policy.penetration_m for s in life.self_separations)
    gates["contact_instrumentation_valid"] = static.get("contact_instrumentation_valid") is True
    if support.role == "free":
        gates["natural_root_mobility"] = static.get("external_anchor_present") is False and all(
            b.get("kinematic") is False for b in physical
        )
        gates["support_valid"] = (
            life.real_support
            and len(tail) == policy.tail_steps
            and all(s.get("active_support_constraint") is True for s in tail)
        )
        z = static.get("final_collision_min_z")
        floor = static.get("support_surface_z")
        gates["gravity_settle"] = (
            finite
            and isinstance(z, (int, float))
            and isinstance(floor, (int, float))
            and math.isfinite(z - floor)
            and z >= floor - policy.floor_tolerance_m
            and gates["settle_stability"]
        )
    else:
        gates["natural_root_mobility"] = not support.root_motion_required and support.reason == "verified_installation"
        gates["support_valid"] = (
            static.get("external_anchor_present") is True
            and static.get("anchor_chain_valid") is True
            and static.get("anchor_pose_stable") is True
        )
        gates["gravity_settle"] = finite and gates["settle_stability"] and gates["support_valid"]
    passed = all(gates.values())
    return {
        "physical_pass": passed,
        "support_role": support.role,
        "gates": gates,
        "failure_reasons": [k for k, v in gates.items() if not v],
        "measurements": {
            "settle_translation_m": translation if math.isfinite(translation) else None,
            "settle_rotation_deg": rotation if math.isfinite(rotation) else None,
            "tail_samples": len(tail),
        },
        "task_match": None,
        "robot_task_success": None,
    }
