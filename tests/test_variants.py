"""CPU contract tests for the one-factor study variants (formal-code-006). Not simulation results."""

import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordcraft.contracts import ConstructionPolicy, VARIANTS
from affordcraft.search import BudgetedConstruction


class Backend:
    def __init__(self):
        self.rank_calls = 0
        self.screens = 0
        self.finals = 0
        self.built = []

    def ground(self, c):
        return {"localized": True, "category_for_retrieval": "Door", "bbox_normalized": [0, 0, 1, 1]}

    def retrieve(self, c, g, limit):
        return [{"candidate_id": str(i)} for i in range(limit)]

    def rank(self, c, g, b):
        self.rank_calls += 1
        return [
            dict(
                candidate_id=x["candidate_id"],
                confidence=0.9,
                geometry_similarity=0.9,
                category_compatible=True,
                mechanism_compatible=True,
                subtype_match="match",
            )
            for x in b
        ]

    def build(self, c, g, a):
        self.built.append(a["candidate_id"])
        return {"candidate_id": a["candidate_id"], "exported": True}

    def screen(self, a, g):
        self.screens += 1
        return {"physical_pass": a["candidate_id"] == "2"}

    def repair(self, a, g, c):
        return None

    def validate_final(self, a, g):
        self.finals += 1
        return {"physical_pass": True}


class VariantPolicy(unittest.TestCase):
    def test_all_registered_variants_construct(self):
        for v in VARIANTS:
            ConstructionPolicy(variant=v, repairs_per_candidate=0 if v == "without_repair" else 1)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            ConstructionPolicy(variant="without_gravity")

    def test_repair_budget_tied_to_variant(self):
        with self.assertRaises(ValueError):
            ConstructionPolicy(variant="without_repair")
        with self.assertRaises(ValueError):
            ConstructionPolicy(variant="full", repairs_per_candidate=0)

    def test_from_definition_ignores_unknown_keys_and_keeps_variant(self):
        p = ConstructionPolicy.from_definition(
            {
                "candidates": 20,
                "batch_size": 5,
                "repairs_per_candidate": 1,
                "variant": "without_task_condition",
                "extra": 1,
            }
        )
        self.assertEqual(p.variant, "without_task_condition")


class VariantBehaviour(unittest.TestCase):
    def test_full_reselects_after_screen_failure(self):
        b = Backend()
        r = BudgetedConstruction(b, ConstructionPolicy()).run({"input_id": "x"})
        self.assertTrue(r["physical_pass"])
        self.assertEqual(b.built, ["0", "1", "2"])
        self.assertEqual(b.rank_calls, 1)

    def test_without_physics_reselection_stops_at_first_constructed_candidate(self):
        b = Backend()
        r = BudgetedConstruction(b, ConstructionPolicy(variant="without_physics_reselection")).run({"input_id": "x"})
        self.assertFalse(r["physical_pass"])
        self.assertEqual(r["failure_reason"], "first_constructed_candidate_failed_physics")
        self.assertEqual(b.built, ["0"])
        self.assertEqual(b.finals, 0)

    def test_without_multimodal_selection_never_calls_the_model(self):
        b = Backend()
        r = BudgetedConstruction(b, ConstructionPolicy(variant="without_multimodal_selection")).run({"input_id": "x"})
        self.assertTrue(r["physical_pass"])
        self.assertEqual(b.rank_calls, 0)
        self.assertEqual(r["attempts"][0]["selection"]["evidence"], "selector_disabled_retrieval_order")
        self.assertTrue(
            any(
                t["phase"] == "multimodal_ranking" and t["timing_status"] == "skipped_by_variant"
                for t in r["phase_timings"]
            )
        )

    def test_without_repair_never_repairs(self):
        b = Backend()
        calls = []
        b.repair = lambda *a: calls.append(a) or {"exported": True, "candidate_id": "0"}
        BudgetedConstruction(b, ConstructionPolicy(variant="without_repair", repairs_per_candidate=0)).run(
            {"input_id": "x"}
        )
        self.assertEqual(calls, [])

    def test_variant_recorded_in_result_policy(self):
        b = Backend()
        r = BudgetedConstruction(b, ConstructionPolicy(variant="without_scale_adaptation")).run({"input_id": "x"})
        self.assertEqual(r["policy"]["variant"], "without_scale_adaptation")


class BackendVariants(unittest.TestCase):
    def test_without_articulation_adaptation_rejects_structured_sources_before_build(self):
        from affordcraft.backend import RuntimeBackend

        b = object.__new__(RuntimeBackend)
        b.variant = "without_articulation_adaptation"
        b.task_metadata = {}
        b.project = Path("/nonexistent")
        b.requests = {}
        r = b.build(
            {"input_id": "x"},
            {},
            {"candidate_id": "9288", "urdf": {"path": "/x/mobility.urdf"}, "structural": {"joint_count": 2}},
        )
        self.assertFalse(r["exported"])
        self.assertEqual(r["reason"], "articulation_adapter_disabled")

    def test_without_scale_adaptation_strips_size_hypothesis(self):
        from affordcraft.backend import RuntimeBackend

        b = object.__new__(RuntimeBackend)
        b.variant = "without_scale_adaptation"
        b.task_metadata = {}
        b.project = Path("/nonexistent")
        seen = {}
        b._materialize = lambda req: seen.update(req) or {"exported": False}
        b.build({"input_id": "x"}, {"size_m": [1, 1, 1], "size_confidence": 0.9}, {"candidate_id": "c", "source": {}})
        self.assertIsNone(seen["grounding"]["size_m"])
        self.assertEqual(seen["grounding"]["size_confidence"], 0.0)
        self.assertEqual(seen["variant"], "without_scale_adaptation")

    def test_without_installation_evidence_never_consults_dataset_metadata(self):
        from affordcraft.backend import RuntimeBackend
        import affordcraft.build as B
        from unittest.mock import patch

        seen = {}
        for variant, expected_calls in (("full", 1), ("without_installation_evidence", 0)):
            b = object.__new__(RuntimeBackend)
            b.variant = variant
            b.task_metadata = {}
            b.project = Path("/nonexistent")
            b.requests = {}
            b._materialize = lambda req: seen.update(req) or {"exported": False}
            with patch.object(B, "source_installation_evidence", return_value=None) as ev, patch.object(
                B, "parser", return_value=(None, {})
            ):
                b.build({"input_id": "x"}, {}, {"candidate_id": "103032", "urdf": {"path": "/x/mobility.urdf"}})
            self.assertEqual(ev.call_count, expected_calls, variant)
            self.assertEqual(seen["task_metadata"]["affordcraft_support"]["role"], "free")
            self.assertEqual(seen["variant"], variant)

    def test_without_decomposition_cascade_keeps_only_level_zero(self):
        import importlib.util, tempfile, json, os, subprocess, sys

        code = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for variant, levels in (("full", 5), ("without_decomposition_cascade", 1)):
                req = Path(tmp) / f"{variant}.json"
                req.write_text(
                    json.dumps(
                        {
                            "project": tmp,
                            "candidate": {"candidate_id": "x", "source": {}},
                            "grounding": {},
                            "task_metadata": {},
                            "variant": variant,
                        }
                    )
                )
                probe = Path(tmp) / f"probe_{variant}.py"
                probe.write_text(
                    'import sys,json,runpy\nsys.argv=["materialize_asset.py","--request",sys.argv[1],"--output",sys.argv[2],"--cache",sys.argv[3]]\n'
                    "import affordcraft.build as B\norig=B.construct_asset\n"
                    'def fake(*a,**k):\n    print("LEVELS",len(B.DECOMPOSITION_CASCADE),B.DECOMPOSITION_CASCADE[0]["level"],B.CASCADE_VERSION);raise RuntimeError("stop")\n'
                    'B.construct_asset=fake\nrunpy.run_path(%r,run_name="__main__")\n'
                    % str(code / "scripts/materialize_asset.py"),
                    encoding="utf-8",
                )
                p = subprocess.run(
                    [sys.executable, str(probe), str(req), str(Path(tmp) / ("out_" + variant)), tmp],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONPATH": str(code)},
                )
                line = [l for l in p.stdout.splitlines() if l.startswith("LEVELS")]
                self.assertEqual(line, [f"LEVELS {levels} 0 decomposition-cascade-v2"], p.stdout + p.stderr)

    def test_without_task_condition_retrieves_category_blind(self):
        from affordcraft.catalog import CatalogIndex
        import numpy as np

        idx = object.__new__(CatalogIndex)
        idx.ids = ["a", "b", "c"]
        idx.entries = {
            k: {"candidate_id": k, "category": c} for k, c in (("a", "Door"), ("b", "Window"), ("c", "Door"))
        }
        idx.features = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
        idx.encoder = type("E", (), {"encode": lambda self, im: np.array([0, 1], dtype=np.float32)})()
        blind = [r["candidate_id"] for r in idx.retrieve(None, None, 3)]
        cond = [r["candidate_id"] for r in idx.retrieve(None, "Door", 3)]
        self.assertEqual(blind, ["b", "c", "a"])
        self.assertEqual(cond[:2], ["c", "a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
