"""Point-based figure helpers (the paper's visual grammar): palette, fonts, a canvas measured in points, and a save
routine that writes PDF + PNG + a JSON receipt (size, minimum text size, text outside the canvas, input hashes, facts).
The figure scripts set OUT (their --out directory) before saving."""

from pathlib import Path
import hashlib, json, os
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT = Path(os.environ.get("AFFORDCRAFT_ANALYSIS_OUT") or (Path(__file__).resolve().parents[1] / "out"))
INK = "#193548"
TEAL = "#147D92"
INDIGO = "#7566A7"
AMBER = "#C58743"
RED = "#B25B69"
GRAY = "#77838E"
LIGHT = "#E0E6EB"
PALE = "#F3F6F8"
NOEXPORT = "#C9D3DA"
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": LIGHT,
        "axes.linewidth": 0.55,
    }
)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Canvas:
    def __init__(self, height, width=396):
        self.w = width
        self.h = height
        self.fig = plt.figure(figsize=(width / 72, height / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.axis("off")
        self.labels = []
        self.inputs = []

    def text(self, x, y, s, size=8, color=INK, weight="normal", ha="left", va="center", **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)
        self.labels.append(t)
        return t

    def rect(self, x, y, w, h, fill="white", edge=None, lw=0.6, radius=0, z=1, alpha=1):
        p = (
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                fc=fill,
                ec=edge or fill,
                lw=lw,
                zorder=z,
                alpha=alpha,
            )
            if radius
            else Rectangle((x, y), w, h, fc=fill, ec=edge or fill, lw=lw, zorder=z, alpha=alpha)
        )
        self.ax.add_patch(p)
        return p

    def line(self, x1, y1, x2, y2, color=LIGHT, lw=0.6, ls="-", z=1):
        return self.ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z)[0]

    def image(self, path, x, y, w, h, crop=None, bleed=0.25):
        """raster tile: the image is stretched to the axes (aspect 'auto', so integer-pixel crops never leave a hairline
        strip) and the axes bleeds `bleed` pt past the tile on every side, so abutting tiles show no seam in PDF viewers
        """
        im = Image.open(path).convert("RGB")
        if crop:
            im = im.crop(crop)
        a = self.fig.add_axes(
            [(x - bleed) / self.w, (y - bleed) / self.h, (w + 2 * bleed) / self.w, (h + 2 * bleed) / self.h]
        )
        a.imshow(im, interpolation="lanczos", aspect="auto")
        a.set_axis_off()
        self.inputs.append(str(path))
        return a

    def overlay_rect(self, x, y, w, h, fill="none", edge=None, lw=0.7, alpha=1.0, z=20):
        """figure-level rectangle: drawn above every image axes (main-axes patches are hidden under images)"""
        p = Rectangle(
            (x / self.w, y / self.h),
            w / self.w,
            h / self.h,
            transform=self.fig.transFigure,
            fc=fill,
            ec=edge or fill,
            lw=lw,
            alpha=alpha,
            zorder=z,
        )
        self.fig.add_artist(p)
        return p

    def overlay_text(self, x, y, s, size=8, color=INK, weight="normal", ha="left", va="center", z=21):
        t = self.fig.text(x / self.w, y / self.h, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, zorder=z)
        self.labels.append(t)
        return t

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h])

    def save(self, name, directory, facts):
        out = OUT / directory
        out.mkdir(parents=True, exist_ok=True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        bbox = self.fig.bbox
        outside = []
        for t in self.labels:
            b = t.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > bbox.x1 + 1 or b.y1 > bbox.y1 + 1:
                outside.append(t.get_text())
        for ext in ("pdf", "png"):
            self.fig.savefig(out / (name + "." + ext), dpi=400, facecolor="white")
        meta = {
            "name": name,
            "width_pt": self.w,
            "height_pt": self.h,
            "minimum_text_pt": min((t.get_fontsize() for t in self.labels), default=None),
            "out_of_canvas_text": outside,
            "inputs": {p: sha(p) for p in sorted(set(self.inputs))},
            "facts": facts,
        }
        (out / (name + ".json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        plt.close(self.fig)
        assert not outside, outside
        return meta
