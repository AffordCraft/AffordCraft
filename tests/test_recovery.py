import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordcraft.execution import known_coacd_abort, completion, supervised_completion
from affordcraft.search import BudgetedConstruction
from affordcraft.contracts import ConstructionPolicy

LOG = "std::logic_error unexpected code path was hit coacd/__init__.py in run_coacd"


class RecoveryTests(unittest.TestCase):
    def test_all_source_files_parse(self):
        import ast

        root = Path(__file__).resolve().parents[1]
        for folder in ("affordcraft", "scripts", "tests"):
            for path in (root / folder).rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_only_exact_abort_can_be_local(self):
        self.assertTrue(known_coacd_abort(-6, LOG))
        for code in (-9, -11, 0, 1, 137):
            self.assertFalse(known_coacd_abort(code, LOG))

    def test_unrelated_abort_blocked(self):
        self.assertFalse(known_coacd_abort(-6, "fatal abort"))
        self.assertFalse(known_coacd_abort(-6, "std::logic_error unexpected code path was hit"))

    def test_zero_exit_incomplete_is_not_complete(self):
        self.assertFalse(supervised_completion(0, {"complete": False}))
        self.assertFalse(supervised_completion(-11, {"complete": True}))
        self.assertFalse(supervised_completion(0, None))
        self.assertTrue(supervised_completion(0, {"complete": True}))

    def test_both_physics_processes_required(self):
        rows = [{"input_id": "x", "physical_pass": True}]
        ok = {p: {"exit_code": 0, "in_process_completed": True} for p in ("construction", "final")}
        self.assertTrue(completion(rows, 1, ok))
        self.assertFalse(completion(rows, 2000, ok))
        self.assertFalse(completion(rows * 2, 2, ok))
        self.assertFalse(completion(rows, 1, {"final": ok["final"]}))
        ok["final"]["exit_code"] = -11
        self.assertFalse(completion(rows, 1, ok))

    def test_candidate_crash_preserved_within_fixed_budget(self):
        class Backend:
            def ground(self, *a):
                return {"localized": True}

            def retrieve(self, *a, **k):
                return [{"candidate_id": str(i)} for i in range(5)]

            def rank(self, a, b, cs):
                return [
                    dict(
                        c,
                        category_compatible=True,
                        mechanism_compatible=True,
                        subtype_match="match",
                        confidence=0.9,
                        geometry_similarity=0.9,
                    )
                    for c in cs
                ]

            def build(self, a, b, c):
                return dict(
                    c,
                    exported=c["candidate_id"] != "0",
                    status="construction_failed" if c["candidate_id"] == "0" else "exported",
                    reason="collision_decomposition_native_abort",
                )

            def repair(self, *a):
                return None

            def screen(self, *a):
                return {"physical_pass": True}

            def validate_final(self, *a):
                return {"physical_pass": False}

        r = BudgetedConstruction(Backend(), ConstructionPolicy(candidates=5)).run({"input_id": "test", "image": {}})
        self.assertFalse(r["physical_pass"])
        self.assertEqual(r["status"], "final_validation_failed")
        self.assertEqual(len(r["attempts"]), 2)
        self.assertEqual(r["attempts"][0]["asset"]["reason"], "collision_decomposition_native_abort")
        self.assertFalse(r["final_validation_used_for_selection"])

    def test_frozen_modulo_shards_cover_exact_denominator(self):
        ids = list(range(2000))
        shards = [ids[i::40] for i in range(40)]
        self.assertEqual(sorted(sum(shards, [])), ids)
        self.assertTrue(all(len(x) == 50 for x in shards))


if __name__ == "__main__":
    unittest.main()
