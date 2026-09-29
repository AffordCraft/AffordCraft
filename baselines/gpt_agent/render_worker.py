"""Offscreen renderer run as a subprocess (EGL/pyrender, matplotlib fallback).
usage: python render_worker.py job.json
job = {"npz": path, "items": [{"v": key, "f": key, "transform": 4x4, "color": [r,g,b]}], "views": [{"azim": deg, "elev": deg}],
       "ground": bool, "size": [w,h], "out": [png paths, one per view], "label": str}
"""

from __future__ import annotations
import json, math, os, sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-12)
    s = np.cross(f, up)
    n = np.linalg.norm(s)
    if n < 1e-9:
        s = np.cross(f, (0, 1, 0))
        n = np.linalg.norm(s)
    s /= n
    u = np.cross(s, f)
    T = np.eye(4)
    T[:3, 0] = s
    T[:3, 1] = u
    T[:3, 2] = -f
    T[:3, 3] = eye
    return T


def load_items(job):
    data = np.load(job["npz"])
    items = []
    for it in job["items"]:
        v = data[it["v"]].astype(np.float64)
        f = data[it["f"]].astype(np.int64)
        T = np.asarray(it["transform"], float)
        v = v @ T[:3, :3].T + T[:3, 3]
        items.append((v, f, it.get("color", [0.7, 0.7, 0.7])))
    return items


def frame_camera(allv, azim, elev, yfov):
    lo, hi = allv.min(0), allv.max(0)
    c = (lo + hi) / 2
    r = max(np.linalg.norm(hi - lo) / 2, 1e-3)
    d = r / math.tan(yfov / 2) * 1.25
    a, e = math.radians(azim), math.radians(elev)
    eye = c + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return eye, c, r, lo, hi


def ground_geometry(lo, hi):
    ext = max(float(np.max(hi - lo)), 0.3)
    half = ext * 1.5
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    quad_v = np.array(
        [[cx - half, cy - half, 0], [cx + half, cy - half, 0], [cx + half, cy + half, 0], [cx - half, cy + half, 0]],
        float,
    )
    quad_f = np.array([[0, 1, 2], [0, 2, 3]])
    step = 0.1 if ext < 1.2 else (0.25 if ext < 3 else 1.0)
    lines = []
    x0 = math.floor((cx - half) / step) * step
    x1 = math.ceil((cx + half) / step) * step
    y0 = math.floor((cy - half) / step) * step
    y1 = math.ceil((cy + half) / step) * step
    xs = np.arange(x0, x1 + step / 2, step)
    ys = np.arange(y0, y1 + step / 2, step)
    if len(xs) > 60:
        xs = xs[:: int(math.ceil(len(xs) / 60))]
    if len(ys) > 60:
        ys = ys[:: int(math.ceil(len(ys) / 60))]
    for x in xs:
        lines.append([[x, y0, 0.0005], [x, y1, 0.0005]])
    for y in ys:
        lines.append([[x0, y, 0.0005], [x1, y, 0.0005]])
    return quad_v, quad_f, np.asarray(lines, float).reshape(-1, 2, 3), step


def render_pyrender(job, items):
    import pyrender, trimesh

    w, h = job.get("size", [640, 480])
    yfov = math.radians(45)
    allv = np.concatenate([v for v, _, _ in items])
    outs = []
    r = pyrender.OffscreenRenderer(w, h)
    try:
        for view, out in zip(job["views"], job["out"]):
            scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.45, 0.45, 0.45])
            for v, f, col in items:
                m = trimesh.Trimesh(vertices=v, faces=f, process=False)
                m.visual.face_colors = np.asarray([[*(int(255 * c) for c in col[:3]), 255]] * len(f), dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
            eye, c, rad, lo, hi = frame_camera(allv, view["azim"], view["elev"], yfov)
            if job.get("ground", True):
                qv, qf, lines, step = ground_geometry(lo, hi)
                g = trimesh.Trimesh(vertices=qv, faces=qf, process=False)
                g.visual.face_colors = np.asarray([[225, 225, 225, 255]] * 2, dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
                if len(lines):
                    prim = pyrender.Primitive(
                        positions=lines.reshape(-1, 3),
                        mode=1,
                        color_0=np.tile([[0.55, 0.55, 0.55, 1.0]], (lines.reshape(-1, 3).shape[0], 1)),
                    )
                    scene.add(pyrender.Mesh(primitives=[prim]))
                # axis triad at world origin (x red, y green, z blue), length = grid step
                tri = np.array([[[0, 0, 0], [step, 0, 0]], [[0, 0, 0], [0, step, 0]], [[0, 0, 0], [0, 0, step]]], float)
                cols = np.array(
                    [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0.6, 0, 1], [0, 0.6, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], float
                )
                scene.add(
                    pyrender.Mesh(primitives=[pyrender.Primitive(positions=tri.reshape(-1, 3), mode=1, color_0=cols)])
                )
            cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=w / h)
            pose = look_at(eye, c)
            scene.add(cam, pose=pose)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
            key = look_at(c + np.array([-rad * 2, rad * 2, rad * 3]), c)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=key)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            from PIL import Image

            rgba = np.asarray(color, dtype=np.float32)
            a = rgba[..., 3:4] / 255.0
            rgb = rgba[..., :3] * a + 255.0 * (
                1.0 - a
            )  # composite over white: the EGL clear colour is not applied reliably
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out)
            outs.append(out)
    finally:
        r.delete()
    return outs


def render_matplotlib(job, items):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    w, h = job.get("size", [640, 480])
    allv = np.concatenate([v for v, _, _ in items])
    lo, hi = allv.min(0), allv.max(0)
    c = (lo + hi) / 2
    rad = max(np.max(hi - lo) / 2, 1e-3)
    outs = []
    for view, out in zip(job["views"], job["out"]):
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        for v, f, col in items:
            if len(f) > 20000:
                f = f[np.random.RandomState(0).choice(len(f), 20000, replace=False)]
            ax.add_collection3d(Poly3DCollection(v[f], facecolors=[*col[:3], 1.0], edgecolors="none"))
        ax.set_xlim(c[0] - rad, c[0] + rad)
        ax.set_ylim(c[1] - rad, c[1] + rad)
        ax.set_zlim(max(0, c[2] - rad), c[2] + rad)
        ax.view_init(elev=view["elev"], azim=view["azim"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    job = json.loads(open(sys.argv[1]).read())
    items = load_items(job)
    try:
        outs = render_pyrender(job, items)
        backend = "pyrender_egl"
    except Exception as exc:
        outs = render_matplotlib(job, items)
        backend = "matplotlib_fallback:" + repr(exc)[:200]
    print(json.dumps({"ok": True, "backend": backend, "out": outs}))


if __name__ == "__main__":
    main()
