"""CPU-only known-positive extension health check, never a replacement asset."""

from pathlib import Path
import argparse
import importlib.metadata
import json
import resource
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import coacd
    import numpy as np
    import trimesh

    start = time.perf_counter()
    mesh = trimesh.creation.box(extents=[0.13, 0.19, 0.23])
    coacd.set_log_level("error")
    raw = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=0.05,
        max_convex_hull=32,
        seed=20260916,
        preprocess_mode="auto",
        decimate=False,
        extrude=False,
    )
    parts = [trimesh.Trimesh(vertices=v, faces=f, process=True) for v, f in raw]
    passed = bool(parts) and all(
        p.is_watertight and abs(float(p.volume)) > 1e-12 and np.isfinite(p.vertices).all() for p in parts
    )
    result = {
        "passed": bool(passed),
        "coacd": importlib.metadata.version("coacd"),
        "numpy": np.__version__,
        "trimesh": trimesh.__version__,
        "fixture": "procedural_box_0.13_0.19_0.23_m",
        "scope": "library_health_not_case_physics",
        "parts": len(parts),
        "seconds": time.perf_counter() - start,
    }
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
