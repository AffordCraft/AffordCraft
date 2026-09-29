"""CPU contract tests for the formal-code-005 revisions. Not simulation results."""

import json, os, sys, tempfile, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordcraft.contracts import SupportContract, ConstructionPolicy, semantic_eligible
from affordcraft.search import normalize_assessments, MethodFailure, BudgetedConstruction
from affordcraft import build as B

URDF = """<robot name="w"><link name="base"/><link name="link_0"><visual><geometry><mesh filename="a.obj"/></geometry></visual><collision><geometry><mesh filename="a.obj"/></geometry></collision></link>
<link name="link_1"><visual><geometry><mesh filename="b.obj"/></geometry></visual><collision><geometry><mesh filename="b.obj"/></geometry></collision></link>
<joint name="joint_0" type="prismatic"><parent link="link_1"/><child link="link_0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.5" effort="1" velocity="1"/></joint>
<joint name="joint_1" type="fixed"><parent link="base"/><child link="link_1"/></joint></robot>"""


def fake_candidate(tmp, semantics_text):
    d = Path(tmp)
    (d / "mobility.urdf").write_text(URDF)
    (d / "semantics.txt").write_text(semantics_text)
    for n in ("a.obj", "b.obj"):
        (d / n).write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
    from affordcraft.catalog import file_sha

    return {
        "candidate_id": "x",
        "urdf": {"path": str(d / "mobility.urdf"), "sha256": file_sha(d / "mobility.urdf")},
        "semantics": {"path": str(d / "semantics.txt"), "sha256": file_sha(d / "semantics.txt")},
        "source_dir": str(d),
        "structural": {"joint_count": 2},
    }


class SelectorNormalization(unittest.TestCase):
    def batch(self):
        return [{"candidate_id": str(i)} for i in range(5)]

    def test_duplicate_reply_keeps_first_and_every_listed_id_once(self):
        reply = [
            {"candidate_id": "2", "confidence": 0.9},
            {"candidate_id": "0", "confidence": 0.8},
            {"candidate_id": "2", "confidence": 0.1},
            {"candidate_id": "1"},
            {"candidate_id": "3"},
            {"candidate_id": "4"},
        ]
        rows, info = normalize_assessments(reply, self.batch())
        self.assertEqual([r["candidate_id"] for r in rows], ["2", "0", "1", "3", "4"])
        self.assertEqual(rows[0]["confidence"], 0.9)
        self.assertTrue(info["normalized"])
        self.assertEqual(info["dropped"][0]["reason"], "duplicate_id")

    def test_omitted_ids_are_abstentions_not_inventions(self):
        rows, info = normalize_assessments(
            [
                {
                    "candidate_id": "4",
                    "confidence": 0.9,
                    "geometry_similarity": 0.9,
                    "category_compatible": True,
                    "mechanism_compatible": True,
                    "subtype_match": "match",
                }
            ],
            self.batch(),
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(info["omitted"], ["0", "1", "2", "3"])
        self.assertTrue(all(r.get("selector_omitted") for r in rows[1:]))
        self.assertFalse(any(semantic_eligible(r, ConstructionPolicy()) for r in rows[1:]))

    def test_unlisted_ids_dropped(self):
        rows, info = normalize_assessments([{"candidate_id": "99"}, {"candidate_id": "1"}], self.batch())
        self.assertEqual([r["candidate_id"] for r in rows][:1], ["1"])
        self.assertEqual(info["dropped"][0]["reason"], "unlisted_id")

    def test_empty_reply_is_a_whole_batch_abstention_not_a_case_failure(self):
        for reply in ([], [{"candidate_id": "99"}]):
            rows, info = normalize_assessments(reply, self.batch())
            self.assertEqual([r["candidate_id"] for r in rows], ["0", "1", "2", "3", "4"])
            self.assertTrue(info["batch_abstention"])
            self.assertEqual(info["omitted"], ["0", "1", "2", "3", "4"])
            self.assertFalse(any(semantic_eligible(r, ConstructionPolicy()) for r in rows))
        with self.assertRaises(MethodFailure):
            normalize_assessments(None, self.batch())
        with self.assertRaises(MethodFailure):
            normalize_assessments({"candidate_id": "1"}, self.batch())

    def test_search_continues_to_next_batch_after_an_empty_reply(self):
        from affordcraft.search import BudgetedConstruction

        class Backend:
            def __init__(self):
                self.calls = 0
                self.built = []

            def ground(self, c):
                return {"localized": True, "category_for_retrieval": "Door", "bbox_normalized": [0, 0, 1, 1]}

            def retrieve(self, c, g, limit):
                return [{"candidate_id": str(i)} for i in range(limit)]

            def rank(self, c, g, b):
                self.calls += 1
                if self.calls == 1:
                    return []
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
                return {"physical_pass": True}

            def repair(self, a, g, c):
                return None

            def validate_final(self, a, g):
                return {"physical_pass": True}

        b = Backend()
        r = BudgetedConstruction(b, ConstructionPolicy()).run({"input_id": "x"})
        self.assertTrue(r["physical_pass"])
        self.assertEqual(b.built, ["5"])
        self.assertEqual(b.calls, 2)
        self.assertEqual(r["selector_normalizations"][0]["batch_offset"], 0)
        self.assertTrue(r["selector_normalizations"][0]["batch_abstention"])
        self.assertEqual([a["status"] for a in r["attempts"][:5]], ["semantic_rejection"] * 5)

    def test_integer_ids_map_back_to_listed_objects(self):
        rows, _ = normalize_assessments(
            [{"candidate_id": 3}, {"candidate_id": 0}, {"candidate_id": 1}, {"candidate_id": 2}, {"candidate_id": 4}],
            self.batch(),
        )
        self.assertEqual([r["candidate_id"] for r in rows], ["3", "0", "1", "2", "4"])


class SourceInstallation(unittest.TestCase):
    def test_static_root_gives_verified_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fake_candidate(tmp, "link_0 slider translation_window\nlink_1 static window_frame\n")
            h, _ = B.parser(tmp)
            e = B.source_installation_evidence(tmp, c, h)
            self.assertIsNotNone(e)
            self.assertEqual(e["root_link"], "link_1")
            self.assertEqual(len(e["artifact"]["sha256"]), 64)
            meta = {"affordcraft_support": {"role": "mounted", "evidence": [e]}}
            s = SupportContract.from_task_metadata(meta)
            self.assertEqual(s.role, "mounted")
            self.assertEqual(s.reason, "verified_installation")

    def test_heavy_or_free_root_gives_no_evidence(self):
        for tag in ("heavy", "free"):
            with tempfile.TemporaryDirectory() as tmp:
                c = fake_candidate(tmp, f"link_0 slider drawer\nlink_1 {tag} furniture_body\n")
                h, _ = B.parser(tmp)
                self.assertIsNone(B.source_installation_evidence(tmp, c, h))

    def test_tampered_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fake_candidate(tmp, "link_0 slider translation_window\nlink_1 static window_frame\n")
            h, _ = B.parser(tmp)
            Path(c["semantics"]["path"]).write_text("link_0 slider x\nlink_1 static y\n")
            self.assertIsNone(B.source_installation_evidence(tmp, c, h))

    def test_root_motion_still_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fake_candidate(tmp, "link_0 slider translation_window\nlink_1 static window_frame\n")
            h, _ = B.parser(tmp)
            e = B.source_installation_evidence(tmp, c, h)
            s = SupportContract.from_task_metadata(
                {"affordcraft_support": {"role": "mounted", "root_motion_required": True, "evidence": [e]}}
            )
            self.assertEqual(s.role, "free")

    def test_model_observation_never_counts(self):
        s = SupportContract.from_task_metadata(
            {
                "affordcraft_support": {
                    "role": "mounted",
                    "evidence": [
                        {
                            "kind": "model_observation",
                            "assertion": "fixed_installation_required",
                            "verified": True,
                            "artifact": {"path": "x", "sha256": "a" * 64},
                        }
                    ],
                }
            }
        )
        self.assertEqual(s.role, "free")


def square_ring(outer=0.5, inner=0.45, thickness=0.05):
    """Watertight square frame (window-frame-like ring) built without boolean backends."""
    import numpy as np, trimesh

    z = [-thickness / 2, thickness / 2]
    V = []
    for r in (outer, inner):
        for zz in z:
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                V.append([sx * r, sy * r, zz])

    # index: ring(0 outer,1 inner)*8 + level(0 bottom,1 top)*4 + corner
    def i(ring, level, corner):
        return ring * 8 + level * 4 + (corner % 4)

    F = []

    def quad(a, b, c, d):
        F.extend([[a, b, c], [a, c, d]])

    for k in range(4):
        quad(i(0, 0, k), i(0, 0, k + 1), i(0, 1, k + 1), i(0, 1, k))  # outer wall
        quad(i(1, 0, k + 1), i(1, 0, k), i(1, 1, k), i(1, 1, k + 1))  # inner wall
        quad(i(0, 1, k), i(0, 1, k + 1), i(1, 1, k + 1), i(1, 1, k))  # top annulus
        quad(i(0, 0, k + 1), i(0, 0, k), i(1, 0, k), i(1, 0, k + 1))  # bottom annulus
    m = trimesh.Trimesh(vertices=np.asarray(V, dtype=float), faces=np.asarray(F), process=True)
    m.fix_normals()
    return m


def frame_sides():
    import trimesh

    T = trimesh.transformations.translation_matrix
    return [trimesh.creation.box(extents=[1, 0.05, 0.05], transform=T([0, y, 0])) for y in (-0.475, 0.475)] + [
        trimesh.creation.box(extents=[0.05, 0.9, 0.05], transform=T([x, 0, 0])) for x in (-0.475, 0.475)
    ]


class Cascade(unittest.TestCase):
    def test_level0_reproduces_frozen_code004_parameters(self):
        l = B.DECOMPOSITION_CASCADE[0]
        self.assertEqual(
            (l["threshold"], l["max_convex_hull"], l["preprocess_resolution"], l["split_components"]),
            (0.05, 32, 50, False),
        )

    def test_levels_only_refine(self):
        for a, b in zip(B.DECOMPOSITION_CASCADE, B.DECOMPOSITION_CASCADE[1:]):
            self.assertLessEqual(b["threshold"], a["threshold"])
            self.assertGreaterEqual(b["max_convex_hull"], a["max_convex_hull"])

    def test_tolerance_unchanged(self):
        self.assertEqual(B.AUDIT_TOLERANCE, 0.02)

    def test_convex_source_shortcut_without_coacd(self):
        import trimesh
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            B, "run_coacd", side_effect=AssertionError("must not run")
        ):
            hulls, d = B.decompose_link(
                trimesh.creation.box(extents=[0.1, 0.2, 0.3]), [trimesh.creation.box(extents=[0.1, 0.2, 0.3])], tmp
            )
        self.assertEqual(len(hulls), 1)
        self.assertEqual(d["pieces"][0]["method"], "convex_source")
        self.assertTrue(d["link_audit"]["pass"])

    def test_ring_audit_rejects_sealed_hull_and_accepts_open_frame(self):
        ring = square_ring()
        self.assertTrue(ring.is_watertight)
        sides = frame_sides()
        sealed = B.audit_parts(B.clean_source(ring), [ring.convex_hull])
        self.assertFalse(sealed["pass"])
        self.assertEqual(sealed["reason"], "decomposition_closes_source_free_space")
        self.assertTrue(B.audit_parts(B.clean_source(ring), sides)["pass"])
        self.assertTrue(B.audit_parts(B.clean_source(ring), sides, mode="rays")["pass"])
        self.assertFalse(B.audit_parts(B.clean_source(ring), [ring.convex_hull], mode="rays")["pass"])

    def test_escalates_until_a_level_passes_and_caches_every_level(self):
        from unittest.mock import patch

        ring = square_ring()
        sides = frame_sides()
        calls = []

        def fake(source, level, timeout=None):
            calls.append(level["level"])
            return [source.convex_hull] if level["level"] < 2 else [s.copy() for s in sides]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(B, "run_coacd", side_effect=fake):
                hulls, d = B.decompose_link(ring, [ring], tmp)
            self.assertEqual(calls, [0, 2])
            self.assertEqual(d["highest_level_used"], 2)
            self.assertTrue(d["link_audit"]["pass"])
            pc = d["pieces"][0]
            self.assertTrue(pc["audit_pass"])
            self.assertTrue(pc["method"].endswith("level_2"))
            self.assertEqual([t.get("level") for t in pc["trials"]], [0, 1, 2])
            self.assertFalse(pc["trials"][0]["pass"])
            self.assertEqual(pc["trials"][1].get("skipped"), "single_component")
            self.assertTrue((Path(tmp) / pc["cache_key"] / "level_0" / "result.json").exists())
            self.assertTrue((Path(tmp) / pc["cache_key"] / "level_2" / "hull_000.ply").exists())
            # Second visit: everything served from the level cache, CoACD never runs.
            with patch.object(B, "run_coacd", side_effect=AssertionError("must not run")):
                hulls2, d2 = B.decompose_link(ring, [ring], tmp)
            self.assertEqual(len(hulls2), len(hulls))
            self.assertTrue(all(t.get("from_cache") for t in d2["pieces"][0]["trials"] if "pass" in t))

    def test_link_level_audit_rescues_pieces_that_fail_alone(self):
        """A small ring bracket whose own hull seals 21% of its bbox (piece audit fails) is
        accepted when the assembled link keeps >98% of the whole body's accessible space open:
        the same tolerance applied at the granularity of the rigid body, as in the code-004 repair path."""
        import trimesh
        from unittest.mock import patch

        ring = square_ring()
        plate = trimesh.creation.box(
            extents=[5, 5, 0.05], transform=trimesh.transformations.translation_matrix([0, 0, -0.5])
        )
        link = trimesh.util.concatenate([ring.copy(), plate.copy()])
        calls = []

        def fake(source, level, timeout=None):
            calls.append(level["level"])
            return [source.convex_hull]

        with tempfile.TemporaryDirectory() as tmp, patch.object(B, "run_coacd", side_effect=fake):
            hulls, d = B.decompose_link(link, [ring, plate], tmp)
        self.assertEqual(calls, [0])
        self.assertEqual(d["highest_level_used"], 0)
        self.assertTrue(d["link_audit"]["pass"])
        self.assertEqual(d["link_audit"]["check"], "accessible_space")
        self.assertFalse(d["pieces"][0]["audit_pass"])
        self.assertEqual(d["pieces"][1]["method"], "convex_source")

    def test_nothing_passes_raises_closure_reason(self):
        from unittest.mock import patch

        ring = square_ring()
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            B, "run_coacd", side_effect=lambda s, l, timeout=None: [s.convex_hull]
        ):
            with self.assertRaises(ValueError) as ctx:
                B.decompose_link(ring, [ring], tmp)
        self.assertIn("decomposition_closes_source", str(ctx.exception))

    def test_level_timeout_is_recorded_and_cascade_continues(self):
        from unittest.mock import patch

        ring = square_ring()
        sides = frame_sides()

        def fake(source, level, timeout=None):
            if level["level"] == 0:
                raise TimeoutError("decomposition_level_timeout")
            return [s.copy() for s in sides]

        with tempfile.TemporaryDirectory() as tmp, patch.object(B, "run_coacd", side_effect=fake):
            hulls, d = B.decompose_link(ring, [ring], tmp)
        pc = d["pieces"][0]
        self.assertTrue(pc["trials"][0].get("timeout"))
        self.assertTrue(pc["method"].endswith("level_2"))

    def test_budget_exhaustion_raises_and_keeps_level_cache_for_resume(self):
        from unittest.mock import patch

        ring = square_ring()
        sides = frame_sides()
        calls = []

        def fake(source, level, timeout=None):
            calls.append(level["level"])
            time.sleep(0.6)
            return [source.convex_hull]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(B, "run_coacd", side_effect=fake):
                with self.assertRaises(ValueError) as ctx:
                    B.decompose_link(ring, [ring], tmp, deadline=time.perf_counter() + B.MIN_LEVEL_SECONDS + 0.3)
            self.assertIn("budget_exhausted", str(ctx.exception))
            self.assertEqual(calls, [0])
            self.assertTrue(list(Path(tmp).glob("*/level_0/result.json")))
            self.assertFalse(list(Path(tmp).rglob("rejection.json")))

            # Resume without budget: level 0 comes from cache (no CoACD call), level 2 now passes.
            def fake2(source, level, timeout=None):
                calls.append(level["level"])
                return [s.copy() for s in sides]

            with patch.object(B, "run_coacd", side_effect=fake2):
                hulls, d = B.decompose_link(ring, [ring], tmp)
            self.assertEqual(calls, [0, 2])
            self.assertTrue(d["link_audit"]["pass"])
            self.assertTrue(d["pieces"][0]["trials"][0].get("from_cache"))

    def test_level_timeouts_are_not_cached(self):
        from unittest.mock import patch

        ring = square_ring()
        sides = frame_sides()
        calls = []

        def fake(source, level, timeout=None):
            calls.append(level["level"])
            if level["level"] == 0 and calls.count(0) == 1:
                raise TimeoutError("decomposition_level_timeout")
            return [s.copy() for s in sides]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(B, "run_coacd", side_effect=fake):
                B.decompose_link(ring, [ring], tmp)
                self.assertEqual(calls, [0, 2])
                self.assertFalse(list(Path(tmp).glob("*/level_0/result.json")))
                self.assertTrue(list(Path(tmp).glob("*/level_2/result.json")))
                B.decompose_link(ring, [ring], tmp)
                self.assertEqual(calls, [0, 2, 0])  # level 0 retried, level 2 cached

    def test_real_isolated_coacd_on_box_and_timeout_kill(self):
        import trimesh

        parts = B.run_coacd(trimesh.creation.box(extents=[0.13, 0.19, 0.23]), B.DECOMPOSITION_CASCADE[0], timeout=60)
        self.assertTrue(parts and all(p.is_watertight for p in parts))
        with self.assertRaises(TimeoutError):
            B.run_coacd(trimesh.creation.icosphere(subdivisions=4), B.DECOMPOSITION_CASCADE[4], timeout=0.05)

    def test_publish_cache_dir_is_idempotent_and_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = "k"
            B.publish_cache_dir(tmp, key, [("hull_000.ply", lambda p: p.write_text("x"))], {"a": 1}, "receipt.json")
            B.publish_cache_dir(tmp, key, [("hull_000.ply", lambda p: p.write_text("y"))], {"a": 2}, "receipt.json")
            self.assertEqual((Path(tmp) / key / "hull_000.ply").read_text(), "x")
            self.assertEqual(json.loads((Path(tmp) / key / "receipt.json").read_text())["a"], 1)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [key])


class AuditAndSnap(unittest.TestCase):
    def open_plate(self):
        """Single-sided open quad 1 x 0.6 in the XY plane: no edge faces, no thickness."""
        import trimesh, numpy as np

        v = np.array([[-0.5, -0.3, 0], [0.5, -0.3, 0], [0.5, 0.3, 0], [-0.5, 0.3, 0]], float)
        f = np.array([[0, 1, 2], [0, 2, 3]])
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        return m

    def test_open_plate_hull_is_not_a_closure(self):
        import trimesh, numpy as np

        plate = self.open_plate()
        hull = trimesh.creation.box(extents=[1, 0.6, 0.004])  # what a remesh of the plate produces
        a = B.accessible_space_audit(plate, [hull])
        self.assertGreater(a["raw_fraction"], 0.5)
        self.assertLess(a["fraction"], 0.02)
        self.assertTrue(B.audit_parts(plate, [hull], mode="rays")["pass"])

    def test_hull_across_a_real_opening_is_still_a_closure(self):
        import trimesh

        ring = square_ring()
        open_ring = ring.copy()
        open_ring.update_faces(open_ring.face_normals[:, 2] < 0.5)
        open_ring.remove_unreferenced_vertices()  # remove the top annulus -> open shell
        a = B.accessible_space_audit(open_ring, [ring.convex_hull])
        self.assertGreater(a["fraction"], 0.02)
        self.assertFalse(B.audit_parts(open_ring, [ring.convex_hull], mode="rays")["pass"])

    def test_snap_removes_voxel_inflation_and_keeps_degenerate_hulls(self):
        import trimesh, numpy as np

        box = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
        inflated = trimesh.creation.box(extents=[0.212, 0.312, 0.412])
        snapped, n = B.snap_to_source(box, [inflated])
        self.assertEqual(n, 1)
        self.assertTrue(np.allclose(snapped[0].bounding_box.extents, [0.2, 0.3, 0.4], atol=1e-6))
        plate = self.open_plate()
        thin = trimesh.creation.box(extents=[1, 0.6, 0.004])
        snapped, n = B.snap_to_source(plate, [thin])
        self.assertEqual(n, 0)
        self.assertIs(snapped[0], thin)

    def test_real_coacd_level0_on_zero_thickness_pane_keeps_hulls_and_passes_audit(self):
        """A double-sided open pane (two offset zero-thickness quads, no edge faces) is remeshed by
        CoACD with voxel inflation. Each sheet's projection is coplanar, so snapping must keep the
        original hulls rather than emit degenerate ones; the edge-aware audit still accepts them."""
        import trimesh, numpy as np

        top = self.open_plate()
        bot = self.open_plate()
        bot.apply_translation([0, 0, -0.02])
        bot.invert()
        pane = trimesh.util.concatenate([top, bot])
        self.assertFalse(pane.is_watertight)
        src = B.clean_source(pane)
        parts = B.run_coacd(src, B.DECOMPOSITION_CASCADE[0], timeout=120)
        z = np.concatenate([np.asarray(p.vertices)[:, 2] for p in parts])
        self.assertGreater(max(z.max(), -0.02 - z.min()), 1e-4)
        snapped, n = B.snap_to_source(src, parts)
        self.assertEqual(n, 0)
        self.assertEqual(len(snapped), len(parts))
        self.assertTrue(all(p.is_watertight and p.volume > 1e-12 for p in snapped))
        self.assertTrue(B.audit_parts(src, snapped)["pass"])

    def test_snap_on_thick_shell_recovers_source_faces(self):
        """A 2 cm thick closed slab whose CoACD-style hull is inflated by 5 mm on every side."""
        import trimesh, numpy as np

        slab = trimesh.creation.box(extents=[1, 0.6, 0.02])
        inflated = trimesh.creation.box(extents=[1.01, 0.61, 0.03])
        snapped, n = B.snap_to_source(slab, [inflated])
        self.assertEqual(n, 1)
        self.assertTrue(np.allclose(snapped[0].bounding_box.extents, [1, 0.6, 0.02], atol=1e-6))
        self.assertTrue(B.audit_parts(slab, snapped)["pass"])


class RepairShortCircuit(unittest.TestCase):
    def test_repair_skipped_when_collision_is_visual(self):
        from affordcraft.backend import RuntimeBackend

        b = object.__new__(RuntimeBackend)
        b.requests = {"r": {"repair_mode": None}}
        asset = {"request_file": "r", "exported": False, "diagnostics": {"visual_equals_collision": True}}
        self.assertIsNone(
            b.repair(asset, {}, {"failure_reasons": ["ValueError('decomposition_closes_source_accessible_space')"]})
        )

    def test_repair_still_offered_for_distinct_collision_source(self):
        from affordcraft.backend import RuntimeBackend

        b = object.__new__(RuntimeBackend)
        b.requests = {"r": {"repair_mode": None}}
        calls = []
        b._materialize = lambda req: calls.append(req) or {"exported": True}
        b.repair(
            {"request_file": "r", "diagnostics": {"visual_equals_collision": False}},
            {},
            {"failure_reasons": ["gravity_settle"]},
        )
        self.assertEqual(calls[0]["repair_mode"], "rebuild_collision_from_source_visual")


if __name__ == "__main__":
    unittest.main(verbosity=2)
