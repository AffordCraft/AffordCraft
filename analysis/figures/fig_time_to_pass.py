"""Appendix figure: accepted assets as a function of the per-input time budget (the Table 1 column "Pass in 5 min" is one
point of these curves).

For every method, the share of its timed inputs whose asset passes the physical gate within a wall-clock budget t, as a
step function of t: the local routes over the 32 inputs of the clean timing run (one worker per exclusive RTX
5090; AffordCraft with the geometry of its library entries built once, the regime of Table 1), each with its own gate
verdicts in that run; the two API routes over the 200 inputs of their own runs. Native path where it applies, else the
adapted path (verdict and time of the adapted build). Reads recorded evidence only:
  clean_timing/clean_timing_summary_*.json   (newest; paths relative to --inputs)
  external/aggregate_all_*.json              (newest; subset200 rows of the API routes)
Writes figures/s05_time_to_pass.{pdf,png,json}."""

import glob, json, math
from pathlib import Path
import numpy as np
import argparse

_AP = argparse.ArgumentParser(description="time-to-pass figure")
_AP.add_argument(
    "--inputs",
    default=str(Path(__file__).resolve().parents[1] / "results"),
    help="recorded inputs (layout of analysis/results)",
)
_AP.add_argument(
    "--out",
    default=str(Path(__file__).resolve().parents[1] / "out"),
    help="output directory (tables/, figures/, facts/, sections/)",
)
_A = _AP.parse_args()
IN = Path(_A.inputs)
OUT = Path(_A.out)
import style

style.OUT = OUT
from style import Canvas, INK, TEAL, INDIGO, AMBER, RED, GRAY, LIGHT

CLEAN_P = sorted(glob.glob(str(IN / "clean_timing/clean_timing_summary_*.json")))[-1]
AGG_P = sorted(glob.glob(str(IN / "external/aggregate_all_*.json")))[-1]
CLEAN = json.load(open(CLEAN_P, encoding="utf-8"))
AGG = json.load(open(AGG_P, encoding="utf-8"))["methods"]
GREEN = "#4E8F5A"
BROWN = "#8C6D4F"
# (label, source, method id, colour, line style, width)
SERIES = [
    ("AffordCraft (ours)", "ours", None, TEAL, "-", 2.0),
    ("PhysX-Anything", "clean", "physx-anything-v0.1", INDIGO, "-", 1.0),
    ("PhysX-Omni", "clean", "physx-omni-v0.1", AMBER, "-", 1.0),
    ("PAct", "clean", "pact-v0.1", RED, "-", 1.0),
    ("PartCrafter", "clean", "partcrafter-v0.1", GRAY, (0, (4, 2)), 1.0),
    ("TRELLIS.2", "clean", "trellis2-4b-v0.1", INK, (0, (4, 2)), 1.0),
    ("Articulate-Anything", "api", "articulate-anything-v0.1", GREEN, (0, (1, 1.2)), 1.2),
    ("GPT-6 Astra agent", "api", "gpt6-astra-agent-v0.1", BROWN, (0, (1, 1.2)), 1.2),
]
BUDGET = 300.0


def times(src, mid):
    """(sorted accepted times, n timed inputs)"""
    if src == "ours":
        rows = CLEAN["ours"]["warm"]["rows"]
        return sorted(r["total_wall_seconds"] for r in rows if r["physical_pass"]), len(rows)
    rows = CLEAN["external"][mid]["rows"] if src == "clean" else AGG[mid]["subset200"]["rows"]
    nat = (CLEAN["external"][mid] if src == "clean" else AGG[mid]["subset200"])["timing"]["native_applies"]
    tk, pk = ("e2e_native_seconds", "native_pass") if nat else ("e2e_adapted_seconds", "adapted_pass")
    return sorted(r[tk] for r in rows if r.get(pk) and r.get(tk) is not None), len(rows)


X0, X1 = 10.0, 12000.0
W = 396.0
H = 168.0
c = Canvas(H, width=W)
ax = c.axes(34, 24, W - 34 - 6, H - 24 - 6)
facts = {"clean_timing": Path(CLEAN_P).name, "aggregate": Path(AGG_P).name, "budget_s_marked": BUDGET, "series": {}}
for label, src, mid, col, ls, lw in SERIES:
    t, n = times(src, mid)
    xs = [X0] + [x for x in t for _ in (0, 1)] + [X1]
    ys = [0.0]
    for i in range(len(t)):
        ys += [100.0 * i / n, 100.0 * (i + 1) / n]
    ys += [100.0 * len(t) / n]
    xs = [min(max(x, X0), X1) for x in xs]
    ax.plot(xs, ys, color=col, ls=ls, lw=lw, label=label, zorder=5 if src == "ours" else 3, solid_capstyle="butt")
    at = lambda b: round(100.0 * sum(1 for x in t if x <= b) / n, 1)
    facts["series"][label] = {
        "n": n,
        "accepted": len(t),
        "at_60s": at(60),
        "at_300s": at(300),
        "at_600s": at(600),
        "at_1800s": at(1800),
        "final": round(100.0 * len(t) / n, 1),
        "source": {"ours": "clean timing, warm regime", "clean": "clean timing", "api": "own run (200 inputs)"}[src],
    }
ax.set_xscale("log")
ax.set_xlim(X0, X1)
ax.set_ylim(0, 82)
ax.axvline(BUDGET, color=GRAY, lw=0.7, ls=(0, (2, 2)), zorder=1)
ticks = [10, 60, 300, 1800, 10800]
ax.set_xticks(ticks)
ax.set_xticklabels(["10 s", "1 min", "5 min", "30 min", "3 h"])
ax.minorticks_off()
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels(["0", "20", "40", "60", "80%"])
ax.tick_params(labelsize=8, length=2, width=0.5, colors=INK)
for s in ("left", "bottom"):
    ax.spines[s].set_color(LIGHT)
ax.grid(axis="y", color=LIGHT, lw=0.4)
ax.set_axisbelow(True)
ax.set_xlabel("Wall-clock budget per input (log scale)", fontsize=8, labelpad=1)
leg = ax.legend(
    loc="upper left",
    fontsize=8,
    frameon=True,
    ncol=2,
    handlelength=2.2,
    columnspacing=1.2,
    labelspacing=0.25,
    borderaxespad=0.2,
    borderpad=0.2,
)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(1.0)
leg.set_zorder(10)
c.labels += [ax.xaxis.label] + list(leg.get_texts())
meta = c.save("s05_time_to_pass", "figures", facts)
print(
    json.dumps({k: meta[k] for k in ("name", "width_pt", "height_pt", "minimum_text_pt", "out_of_canvas_text")}),
    json.dumps(facts["series"], indent=0)[:1500],
)
