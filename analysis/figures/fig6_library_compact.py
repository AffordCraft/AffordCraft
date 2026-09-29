"""Figure 5 (library growth), compact layout: the same five configurations and four data series as the earlier grouped-bar
figure f06_library_grouped_bars (whose receipt is development/f06_library_grouped_bars.json), drawn without the legend
block, the 'Entries' strip and the abbreviation key. Entry counts are the x tick labels, values sit on the bars, and the series names are inline
text inside each axes; the meaning of the five configurations is stated in the caption. Facts are asserted against the
recorded figure receipt so no number changes."""

import json
from pathlib import Path
import numpy as np
import argparse

_AP = argparse.ArgumentParser(description="Figure 5 (library growth)")
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
_A, _REST = _AP.parse_known_args()
IN = Path(_A.inputs)
OUT = Path(_A.out)
import style

style.OUT = OUT
from style import Canvas, INK, TEAL, INDIGO, AMBER, GRAY, LIGHT

V2 = IN / "development"
E = json.load(open(V2 / "evidence.json", encoding="utf-8"))
T = {x["condition"]: x for x in json.load(open(V2 / "catalog_reanalysis.json", encoding="utf-8"))}
BEFORE = json.load(open(V2 / "f06_library_grouped_bars.json", encoding="utf-8"))["facts"]
KEYS = ["base_library", "additional_instances", "additional_mechanisms", "additional_categories", "full_library"]
P95C = "#6D7783"


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=8, length=2.5, pad=2, colors=INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)


def main(name="f06_library_compact", height=95):
    """height: canvas height in pt; the axes are 19 pt shorter (14 pt below for the tick labels, 5 pt above). 95 pt is the
    tallest canvas that keeps the figure on page 7 with the text that cites it (at 96 pt it floats to page 8 and leaves
    two 40-50 pt gaps there)"""
    entries = [E["catalog"][k]["catalog_size"] for k in KEYS]
    coverage = [100 * E["catalog"][k]["category_coverage"]["rate"] for k in KEYS]
    agreement = [100 * E["catalog"][k]["category_top1_agreement"]["rate"] for k in KEYS]
    median = [1000 * T[k]["query_scope_median_seconds"] for k in KEYS]
    p95 = [1000 * T[k]["query_scope_p95_seconds"] for k in KEYS]
    for key, val in [
        ("entries", entries),
        ("coverage", coverage),
        ("agreement", agreement),
        ("median_ms", median),
        ("p95_ms", p95),
    ]:
        assert all(abs(a - b) < 1e-9 for a, b in zip(val, BEFORE[key])), key
    labels = [f"{e:,}" for e in entries]
    x = np.arange(5)
    # margins cut to the tick labels, no y-axis titles (the unit rides the top tick: 100%, 80 ms),
    # both axes 162 x 76 pt on a 95-pt canvas instead of 154/146 x 68 pt on 92 pt
    c = Canvas(height)
    ah = height - 19
    left = c.axes(30, 14, 162, ah)
    right = c.axes(230, 14, 162, ah)
    left.bar(x - 0.2, coverage, width=0.36, color=INDIGO)
    left.bar(x + 0.2, agreement, width=0.36, color=TEAL)
    left.set(ylim=(0, 118), xlim=(-0.62, 4.62), yticks=[0, 50, 100])
    left.set_yticklabels(["0", "50", "100%"])
    left.set_xticks(x, labels)
    style_axis(left)
    for xi, (cv, ag) in enumerate(zip(coverage, agreement)):
        left.text(xi - 0.2, cv + 2, f"{cv:.0f}", fontsize=8, color=INK, ha="center", va="bottom")
        left.text(xi + 0.2, ag + 2, f"{ag:.0f}", fontsize=8, color=INK, ha="center", va="bottom")
    right.bar(x - 0.2, median, width=0.36, color=AMBER)
    right.bar(x + 0.2, p95, width=0.36, color=P95C)
    right.set(ylim=(0, 100), xlim=(-0.62, 4.62), yticks=[0, 40, 80])
    right.set_yticklabels(["0", "40", "80 ms"])
    right.set_xticks(x, labels)
    style_axis(right)
    for xi, (md, pp) in enumerate(zip(median, p95)):
        right.text(xi - 0.2, md + 1.5, f"{md:.0f}", fontsize=8, color=INK, ha="center", va="bottom")
        right.text(xi + 0.2, pp + 1.5, f"{pp:.0f}", fontsize=8, color=INK, ha="center", va="bottom")
    # legends inside the axes, in the region no bar or value label reaches: two rows upper-left of the rate panel, one row upper-left of the latency panel
    from matplotlib.patches import Rectangle

    for i, (lab, col) in enumerate([("Category coverage", INDIGO), ("Top-1 agreement", TEAL)]):
        y = 107 - i * 16
        left.add_patch(Rectangle((-0.5, y - 6), 0.27, 12, fc=col, ec=col, lw=0, zorder=3))
        left.text(-0.1, y, lab, fontsize=8, color=INK, ha="left", va="center")
    for x0, (lab, col) in zip((-0.5, 1.05), [("Median", AMBER), ("P95", P95C)]):
        right.add_patch(Rectangle((x0, 93 - 5), 0.28, 10, fc=col, ec=col, lw=0, zorder=3))
        right.text(x0 + 0.4, 93, lab, fontsize=8, color=INK, ha="left", va="center")
    for a in (left, right):
        c.labels.extend(a.texts + a.get_xticklabels() + a.get_yticklabels())
    meta = c.save(
        name,
        "figures",
        dict(
            entries=entries,
            coverage=coverage,
            agreement=agreement,
            median_ms=median,
            p95_ms=p95,
            configurations=KEYS,
            bars=20,
            rate_y_limit=[0, 118],
            latency_y_limit=[0, 100],
            median_and_p95_are_quantiles_not_confidence_intervals=True,
            value_labels_rounded_to="integer",
            axes_pt={"left": [30, 14, 162, ah], "right": [230, 14, 162, ah]},
            y_axis_titles="none; the unit rides the top tick label (100%, 80 ms)",
            text_moved_to_caption=["Entries strip", "abbreviation key", "x-axis title", "y-axis titles"],
            legend_in_figure=["Category coverage", "Top-1 agreement", "Median", "P95"],
            source_figure_receipt=str(V2 / "f06_library_grouped_bars.json"),
        ),
    )
    print(json.dumps({k: meta[k] for k in ("name", "width_pt", "height_pt", "minimum_text_pt", "out_of_canvas_text")}))


if __name__ == "__main__":
    main(*([_REST[0], int(_REST[1])] if len(_REST) > 1 else []))  # optional positional arguments: name, height
