import unittest, sys, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affordcraft.build import forward_kinematics


class KinematicTests(unittest.TestCase):
    def test_fixed_source_transform_not_solver_snap(self):
        j = SimpleNamespace(
            name="j",
            parent="base",
            child="link",
            origin_xyz=(1, 0, 0),
            origin_rpy=(0, 0, math.pi / 2),
            joint_type="fixed",
            axis=None,
        )
        t = forward_kinematics(["base", "link"], [j], "base")
        self.assertTrue(np.allclose(t["link"][:3, 3], [1, 0, 0]))
        self.assertTrue(np.allclose(t["link"][:3, :3] @ [1, 0, 0], [0, 1, 0]))

    def test_prismatic_legal_nonzero_initial_configuration(self):
        j = SimpleNamespace(
            name="j",
            parent="base",
            child="link",
            origin_xyz=(0, 0, 0),
            origin_rpy=(0, 0, 0),
            joint_type="prismatic",
            axis=(1, 0, 0),
        )
        t = forward_kinematics(["base", "link"], [j], "base", {"j": 0.1})
        self.assertTrue(np.allclose(t["link"][:3, 3], [0.1, 0, 0]))

    def test_revolute_legal_nonzero_initial_configuration(self):
        j = SimpleNamespace(
            name="j",
            parent="base",
            child="link",
            origin_xyz=(0, 0, 0),
            origin_rpy=(0, 0, 0),
            joint_type="revolute",
            axis=(0, 0, 1),
        )
        t = forward_kinematics(["base", "link"], [j], "base", {"j": math.pi / 2})
        self.assertTrue(np.allclose(t["link"][:3, :3] @ [1, 0, 0], [0, 1, 0]))


if __name__ == "__main__":
    unittest.main()
