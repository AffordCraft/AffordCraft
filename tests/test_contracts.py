"""CPU contract tests are not simulation results."""

import copy, json, math, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordcraft.contracts import *
from affordcraft.physics import *
from affordcraft.search import *


def fixture():
    b = {
        "id": "body",
        "role": "physical_body",
        "kinematic": False,
        "has_geometry": True,
        "mass": 1.0,
        "inertia_diagonal": [1.0, 1.0, 1.0],
        "center_of_mass": [0.0, 0.0, 0.0],
    }
    names = [
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
        "contact_instrumentation_valid",
    ]
    static = {k: True for k in names}
    static.update(
        bodies=[b],
        external_anchor_present=False,
        final_collision_min_z=0.0,
        support_surface_z=0.0,
        initial_floor_clearance_m=0.002,
        support_plane_present=True,
    )
    trace = [
        {
            "step": i,
            "simulation_seconds": i / 120,
            "bodies": {"body": {"position": [0.0, 0.0, 0.1], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}},
            "active_support_constraint": True,
        }
        for i in range(1, 361)
    ]
    contact = [
        {
            "step": 1,
            "kind": "support",
            "event_type": "CONTACT_FOUND",
            "colliders": ["body/collision", "floor/collision"],
            "samples": [{"separation": 0.0}],
        }
    ]
    return trace, static, contact


def verified_mount():
    return SupportContract.from_task_metadata(
        {
            "affordcraft_support": {
                "role": "mounted",
                "evidence": [
                    {
                        "kind": "source_installation_metadata",
                        "assertion": "fixed_installation_required",
                        "verified": True,
                        "artifact": {"path": "source.json", "sha256": "a" * 64},
                    }
                ],
            }
        }
    )


class GateTests(unittest.TestCase):
    def test_free_body_positive(self):
        t, s, c = fixture()
        self.assertTrue(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_kinematic_bottle_rejected(self):
        t, s, c = fixture()
        s["bodies"][0]["kinematic"] = True
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["gates"]["natural_root_mobility"])

    def test_extra_anchor_on_free_body_rejected(self):
        t, s, c = fixture()
        s["external_anchor_present"] = True
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_mounted_without_ground_contact(self):
        t, s, c = fixture()
        s.update(external_anchor_present=True, anchor_chain_valid=True, anchor_pose_stable=True)
        for row in t:
            row["active_support_constraint"] = False
        self.assertTrue(evaluate_trace(t, s, [], verified_mount())["physical_pass"])

    def test_mounted_broken_chain_rejected(self):
        t, s, c = fixture()
        s.update(external_anchor_present=True, anchor_chain_valid=False, anchor_pose_stable=True)
        self.assertFalse(evaluate_trace(t, s, c, verified_mount())["physical_pass"])

    def test_model_guess_does_not_authorize_mount(self):
        m = {
            "affordcraft_support": {
                "role": "mounted",
                "kinematic": True,
                "evidence": [
                    {
                        "kind": "model_observation",
                        "verified": True,
                        "assertion": "fixed_installation_required",
                        "artifact": {"sha256": "a" * 64},
                    }
                ],
            }
        }
        self.assertEqual(SupportContract.from_task_metadata(m).role, "free")

    def test_root_motion_overrides_mount(self):
        m = verified_mount().task_metadata()
        m["affordcraft_support"]["root_motion_required"] = True
        self.assertEqual(SupportContract.from_task_metadata(m).role, "free")

    def test_unverified_declaration_rejected(self):
        m = verified_mount().task_metadata()
        m["affordcraft_support"]["evidence"][0]["verified"] = False
        self.assertEqual(SupportContract.from_task_metadata(m).role, "free")

    def test_empty_coordinate_frame_exemption(self):
        self.assertTrue(
            body_inertia_valid(
                {
                    "role": "coordinate_frame",
                    "has_geometry": False,
                    "source_declared_empty_static_frame": True,
                    "kinematic": True,
                }
            )
        )

    def test_mesh_body_not_exempt(self):
        self.assertFalse(
            body_inertia_valid(
                {
                    "role": "coordinate_frame",
                    "has_geometry": True,
                    "source_declared_empty_static_frame": True,
                    "kinematic": True,
                }
            )
        )

    def test_bad_triangle_rejected(self):
        t, s, c = fixture()
        s["bodies"][0]["inertia_diagonal"] = [1, 1, 4]
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_locked_part_rejected(self):
        t, s, c = fixture()
        s["required_mobility_preserved"] = False
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_initial_penetration(self):
        t, s, c = fixture()
        c[0]["samples"][0]["separation"] = -0.0011
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["gates"]["initial_penetration"])

    def test_self_penetration(self):
        t, s, c = fixture()
        c.append({"step": 30, "kind": "self", "event_type": "CONTACT_FOUND", "samples": [{"separation": -0.0011}]})
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_pre_solver_overlap_cannot_disappear_from_evidence(self):
        t, s, c = fixture()
        s["initial_floor_clearance_m"] = -0.01
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["gates"]["initial_penetration"])

    def test_nested_answer_not_model_input(self):
        d = clean_model_input(
            {"image": {"path": "image.png", "sha256": "a" * 64, "bbox": [0, 0, 1, 1], "expected_asset": "answer"}}
        )
        self.assertEqual(set(d["image"]), {"path", "sha256"})

    def test_unregistered_body_is_not_physics_success(self):
        t, s, c = fixture()
        s["runtime_bodies_registered"] = False
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_broken_runtime_constraint_is_not_success(self):
        t, s, c = fixture()
        s["constraint_solver_consistency"] = False
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_nan_state(self):
        t, s, c = fixture()
        t[44]["bodies"]["body"]["position"][0] = math.nan
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_missing_sample(self):
        t, s, c = fixture()
        t.pop(7)
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_tail_uses_seconds_not_sparse_count(self):
        t, s, c = fixture()
        dense = late_samples(t, EvaluationPolicy())
        sparse = late_samples(t[3::4], EvaluationPolicy())
        self.assertEqual(len(dense), 60)
        self.assertEqual(len(sparse), 15)
        self.assertTrue(all(r["simulation_seconds"] > 2.5 for r in sparse))

    def test_actual_float32_simulation_clock(self):
        t, s, c = fixture()
        for row in t:
            row["simulation_seconds"] = row["step"] * 0.008333333767950535
        r = evaluate_trace(t, s, c, SupportContract())
        self.assertTrue(r["physical_pass"])
        self.assertEqual(r["measurements"]["tail_samples"], 60)

    def test_early_motion_not_counted_as_late_drift(self):
        t, s, c = fixture()
        for r in t[:120]:
            r["bodies"]["body"]["position"][0] = 1.0
        self.assertTrue(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_actual_late_motion_rejected(self):
        t, s, c = fixture()
        t[310]["bodies"]["body"]["position"][0] = 0.1
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_sleeping_contact_not_lost(self):
        life = ContactLifecycle()
        life.feed(fixture()[2][0])
        self.assertTrue(life.supported())
        life.feed(
            {
                "kind": "support",
                "event_type": "CONTACT_LOST",
                "colliders": ["body/collision", "floor/collision"],
                "samples": [],
            }
        )
        self.assertFalse(life.supported())

    def test_missing_required_gate_fails(self):
        t, s, c = fixture()
        s.pop("collision_enabled")
        self.assertFalse(evaluate_trace(t, s, c, SupportContract())["physical_pass"])

    def test_dt_is_not_tunable(self):
        with self.assertRaises(ValueError):
            EvaluationPolicy(dt=1 / 60)

    def test_model_input_allowlist(self):
        d = clean_model_input(
            {
                "input_id": "x",
                "image": {},
                "instruction": "pick",
                "target_noun": "bottle",
                "bbox": [0, 0, 1, 1],
                "expected_asset": "answer",
                "physical_pass": True,
            }
        )
        self.assertEqual(set(d), {"input_id", "image", "instruction", "target_noun"})


class FakeBackend:
    def __init__(self, success_at=None, final=True, blocked=False):
        self.success_at = success_at
        self.final = final
        self.blocked = blocked
        self.final_calls = 0
        self.screens = 0
        self.rank_sizes = []

    def ground(self, c):
        if self.blocked:
            raise InfrastructureBlocked("checkpoint unavailable")
        return {"localized": True}

    def retrieve(self, c, g, limit):
        return [{"candidate_id": str(i)} for i in range(30)] + [{"candidate_id": "0"}]

    def rank(self, c, g, b):
        self.rank_sizes.append(len(b))
        return [
            dict(
                candidate_id=x["candidate_id"],
                confidence=0.8,
                geometry_similarity=0.7,
                category_compatible=True,
                mechanism_compatible=True,
                subtype_match="match",
            )
            for x in b
        ]

    def build(self, c, g, a):
        return {"candidate_id": a["candidate_id"], "exported": True}

    def screen(self, a, g):
        self.screens += 1
        return {"physical_pass": int(a["candidate_id"]) == self.success_at}

    def repair(self, a, g, c):
        return {**a, "repaired": True}

    def validate_final(self, a, g):
        self.final_calls += 1
        return {"physical_pass": self.final}


class SearchTests(unittest.TestCase):
    def test_budget_and_once_repair(self):
        b = FakeBackend()
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertEqual(len(r["attempts"]), 20)
        self.assertEqual(b.rank_sizes, [5] * 4)
        self.assertEqual(b.screens, 40)
        self.assertEqual(b.final_calls, 0)

    def test_first_construction_pass_selected(self):
        b = FakeBackend(2)
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertTrue(r["physical_pass"])
        self.assertEqual(len(r["attempts"]), 3)

    def test_final_failure_never_selects_next(self):
        b = FakeBackend(0, False)
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertEqual(r["status"], "final_validation_failed")
        self.assertEqual(len(r["attempts"]), 1)
        self.assertEqual(b.final_calls, 1)

    def test_infrastructure_not_zero_success(self):
        r = BudgetedConstruction(FakeBackend(blocked=True)).run({"input_id": "a"})
        self.assertIsNone(r["physical_pass"])
        self.assertEqual(r["status"], "infrastructure_blocked")

    def test_invalid_selector_is_method_failure(self):
        b = FakeBackend()
        b.rank = lambda *x: []
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertEqual(r["status"], "construction_failed")
        self.assertFalse(r["physical_pass"])

    def test_low_confidence_not_bypassed(self):
        p = ConstructionPolicy()
        self.assertFalse(
            semantic_eligible(
                dict(
                    category_compatible=True,
                    mechanism_compatible=True,
                    confidence=0.1,
                    geometry_similarity=1,
                    subtype_match="match",
                ),
                p,
            )
        )

    def test_unsuccessful_repair_never_sent_to_physics(self):
        b = FakeBackend()
        b.repair = lambda *a: {"exported": False, "reason": "geometry_unsupported"}
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertEqual(b.screens, 20)
        self.assertEqual(r["status"], "construction_failed")

    def test_repair_dependency_failure_remains_blocked(self):
        b = FakeBackend()
        b.repair = lambda *a: {"exported": False, "status": "infrastructure_blocked", "reason": "dependency_missing"}
        r = BudgetedConstruction(b).run({"input_id": "a"})
        self.assertEqual(r["status"], "infrastructure_blocked")
        self.assertIsNone(r["physical_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
