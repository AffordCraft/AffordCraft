"""CPU tests for the second detection pass of the cluttered-image experiment (no model, no GPU)."""

import json, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from detect_scene_objects import prefix_objects, box_of, periodic_run, token_repeat, accept_cases, iou, strip_fence


def obj(cat, box, conf=0.9):
    return {
        "category": cat,
        "bbox_1000": box,
        "confidence": conf,
        "operated_part": None,
        "motion_family": "rigid",
        "size_m": None,
        "size_confidence": 0,
    }


SCENE = {"input_id": "scene_x", "image": {"path": "/nonexistent.jpg", "sha256": "0" * 64}}
CATS = ["Bottle", "Book", "Chair"]


class ParseTests(unittest.TestCase):
    def test_complete_reply_with_fence(self):
        raw = "```json\n" + json.dumps({"objects": [obj("Bottle", [1, 2, 30, 40])]}) + "\n```"
        objs, complete = prefix_objects(raw)
        self.assertTrue(complete)
        self.assertEqual(len(objs), 1)

    def test_complete_reply_without_object_list(self):
        objs, complete = prefix_objects('{"objects": 3}')
        self.assertTrue(complete)
        self.assertIsNone(objs)

    def test_truncated_reply_keeps_complete_prefix(self):
        full = json.dumps(
            {"objects": [obj("Bottle", [1, 2, 30, 40]), obj("Book", [5, 5, 50, 50]), obj("Chair", [0, 0, 100, 100])]},
            indent=2,
        )
        cut = full[: full.rfind('"bbox_1000"') + 20]  # inside the third object
        objs, complete = prefix_objects(cut)
        self.assertFalse(complete)
        self.assertEqual([o["category"] for o in objs], ["Bottle", "Book"])

    def test_garbage_reply_is_empty(self):
        self.assertEqual(prefix_objects("no json here"), ([], False))
        self.assertEqual(prefix_objects('{"objects": ['), ([], False))

    def test_strip_fence(self):
        self.assertEqual(strip_fence('```json\n{"a":1}\n```'), '{"a":1}')
        self.assertEqual(strip_fence('```json\n{"a":1'), '{"a":1')


class LoopTests(unittest.TestCase):
    def test_stride_loop_detected_with_seed_period(self):
        seq = [box_of(obj("Chair", [0, 0, 100, 100])), box_of(obj("Bottle", [500, 100, 600, 300]))] + [
            box_of(obj("Bottle", [268 - 3 * i, 332, 271 - 3 * i, 372])) for i in range(9)
        ]
        self.assertEqual(periodic_run(seq), (2, 1))

    def test_verbatim_repeat_detected(self):
        seq = [box_of(obj("Book", [10, 10, 50, 50]))] + [box_of(obj("Bottle", [15, 342, 262, 400]))] * 8
        self.assertEqual(periodic_run(seq), (1, 1))

    def test_period_two_loop(self):
        seq = [box_of(obj("Chair", [0, 0, 100, 100]))]
        for i in range(6):
            seq += [
                box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i, 100])),
                box_of(obj("Book", [100 + 30 * i, 200, 120 + 30 * i, 300])),
            ]
        self.assertEqual(periodic_run(seq), (1, 2))

    def test_legitimate_lists_are_silent(self):
        seq = [
            box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i, 100])) for i in range(7)
        ]  # below the minimum run length
        self.assertIsNone(periodic_run(seq))
        seq = [
            box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i + ((i * i) % 7) * 5, 100])) for i in range(12)
        ]  # sizes vary aperiodically
        self.assertIsNone(periodic_run(seq))
        seq = [
            box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i + (i % 3) * 5, 100])) for i in range(12)
        ]  # period-3 size cycle with constant stride IS a loop
        self.assertEqual(periodic_run(seq), (0, 3))
        seq = [
            box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i, 100])) for i in range(12)
        ]  # equal size, constant stride: loop
        self.assertEqual(periodic_run(seq), (0, 1))
        seq = [box_of(obj("Book", [100 + 30 * i, 0, 120 + 30 * i, 100])) for i in range(12)] + [
            box_of(obj("Chair", [0, 0, 900, 900]))
        ]  # run not trailing
        self.assertIsNone(periodic_run(seq))

    def test_token_repeat(self):
        ids = list(range(50)) + [7, 8, 9, 10, 11, 12, 13, 14] * 3
        self.assertEqual(token_repeat(ids), 8)
        self.assertIsNone(token_repeat(list(range(50)) + [7, 8, 9, 10, 11, 12, 13, 14] * 2))
        self.assertIsNone(token_repeat(list(range(400))))


class AcceptTests(unittest.TestCase):
    def test_indices_are_positions_in_the_reply(self):
        objs = [
            obj("Bottle", [1, 2, 30, 40]),
            obj("Unknown", [1, 2, 30, 40]),
            obj("Book", [5, 5, 50, 50], 0.3),
            obj("Chair", [0, 0, 100, 100]),
            obj("Chair", [1, 1, 100, 100]),
            obj("Bottle", [0, 0, 5, 50]),
            obj("Book", [900, 900, 850, 950]),
            obj("Book", [10, 10, 200, 200]),
        ]
        cases, skipped = accept_cases(objs, CATS, SCENE, 9, "src")
        self.assertEqual([c["input_id"] for c in cases], ["scene_x__det_000", "scene_x__det_003", "scene_x__det_007"])
        self.assertEqual(
            skipped, {"category_or_confidence": 2, "same_category_duplicate": 1, "degenerate_box": 1, "invalid_box": 1}
        )
        self.assertEqual(cases[0]["automatic_grounding"]["bbox_normalized"], [0.001, 0.002, 0.03, 0.04])
        self.assertEqual(cases[0]["automatic_grounding_provenance"]["model_call_index"], 9)
        self.assertFalse(cases[0]["automatic_grounding_provenance"]["reference_annotations_used"])

    def test_duplicate_needs_same_category(self):
        objs = [obj("Bottle", [0, 0, 100, 100]), obj("Book", [0, 0, 100, 100])]
        cases, skipped = accept_cases(objs, CATS, SCENE, 1, "src")
        self.assertEqual(len(cases), 2)
        self.assertEqual(skipped, {})

    def test_iou(self):
        self.assertAlmostEqual(iou(("a", 0, 0, 100, 100), ("a", 50, 0, 150, 100)), 1 / 3)
        self.assertEqual(iou(("a", 0, 0, 10, 10), ("a", 20, 20, 30, 30)), 0)


if __name__ == "__main__":
    unittest.main()
