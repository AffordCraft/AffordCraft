"""Figures 4 and the appendix category figure, from the main campaign's evidence export (formal-code-009,
campaign/evidence_v2_20260920-181120.json). The category axes fill the canvas, and the region no
bar reaches (right of Phone, above the 100-input gridline) holds an inset panel with the stacked-bar key and two 100 %
composition bars, which replace the earlier lower row of thin bar lists. Panel headings, the total and the repair footnote
live in the caption.

f04_physical_counts_013: all 2,000 inputs by category, stacked physical pass / no structured export (the campaign scores
physical checks per candidate inside the construction loop, so no input ends in a 'failed checks' state); inset: terminal
reasons of the 297 inputs without an asset and the outcome of every candidate attempt. s01_all_categories_013: per-category
pass rate with Wilson 95% intervals for all 31 categories."""

import json, math
from pathlib import Path
import numpy as np
import argparse

_AP = argparse.ArgumentParser(description="Figure 4 and the appendix category figure")
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
from style import Canvas, INK, TEAL, GRAY, LIGHT, PALE, AMBER, RED, NOEXPORT, INDIGO

EV = IN / "campaign/evidence_v2_20260920-181120.json"
E = json.load(open(EV, encoding="utf-8"))
X1 = E["main_campaign"]["exp1"]
CAT = X1["by_category"]
SHORT = {
    "StorageFurniture": "Storage",
    "Refrigerator": "Fridge",
    "WashingMachine": "Washer",
    "CoffeeMachine": "Coffee maker",
    "TrashCan": "Trash can",
}


def wilson(k, n, z=1.9599639845):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


from matplotlib.patches import Rectangle


def _tint(hex_color, k):
    """the colour mixed towards white; k=1 is the colour itself, smaller k is lighter (ordinal steps of one hue)"""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(255 - (255 - v) * k) for v in (r, g, b))


def _shade(hex_color, k):
    """the colour mixed towards black; k=1 is the colour itself, smaller k is darker"""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(v * k) for v in (r, g, b))


def _lum(hex_color):
    def ch(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _text_on(fill):
    """white or ink for text inside a fill, whichever has the higher WCAG contrast ratio against it"""
    L = _lum(fill)
    return "white" if 1.05 / (L + 0.05) > (L + 0.05) / (_lum(INK) + 0.05) else INK


def _measure(c, a, runs, size=8):
    r = c.fig.canvas.get_renderer()
    tot = 0
    for s, wt in runs:
        t = a.text(0, 0, s, fontsize=size, weight=wt)
        tot += t.get_window_extent(r).width * 72 / c.fig.dpi
        t.remove()
    return tot


def _runs(c, a, x, y, runs, ha="left", size=8):
    """consecutive text runs [(text, weight, colour)] on one line (an axes whose data units are points); ha places the whole sequence"""
    r = c.fig.canvas.get_renderer()
    ts = [a.text(0, y, s, fontsize=size, weight=wt, color=col, ha="left", va="center", zorder=5) for s, wt, col in runs]
    ws = [t.get_window_extent(r).width * 72 / c.fig.dpi for t in ts]
    tot = sum(ws)
    x0 = x - tot / 2 if ha == "center" else (x - tot if ha == "right" else x)
    for t, w in zip(ts, ws):
        t.set_x(x0)
        x0 += w
        c.labels.append(t)
    return tot


def _key(c, a, x, y, fill, runs):
    """swatch + text runs; returns the x where the item ends"""
    a.add_patch(Rectangle((x, y - 3.2), 5, 6.4, fc=fill, ec="none", lw=0, zorder=4))
    return x + 7.5 + _runs(c, a, x + 7.5, y, runs)


def _composition(c, a, x, y, w, h, segs):
    """one 100 % stacked horizontal bar (parts of a whole) with 1-pt surface gaps between segments. A segment carries its
    name and count inside when they fit with padding, only the count when just that fits, nothing when neither fits; the
    returned key entries [(fill, runs)] name every segment whose name is not inside the bar (with its count if that is
    not inside either). Segment widths are exact shares of the total."""
    tot = sum(s[1] for s in segs)
    key = []
    cur = x
    for i, (nm, v, fill) in enumerate(segs):
        sw = w * v / tot
        x0 = cur + (0.5 if i else 0)
        x1 = cur + sw - (0.5 if i < len(segs) - 1 else 0)
        a.add_patch(Rectangle((x0, y - h / 2), max(x1 - x0, 0.25), h, fc=fill, ec="none", lw=0, zorder=3))
        tc = _text_on(fill)
        cnt = f"{v:,}"
        free = x1 - x0
        if _measure(c, a, [(nm + " ", "normal"), (cnt, "bold")]) + 6 <= free:
            _runs(c, a, (x0 + x1) / 2, y, [(nm + " ", "normal", tc), (cnt, "bold", tc)], ha="center")
        elif _measure(c, a, [(cnt, "bold")]) + 4 <= free:
            _runs(c, a, (x0 + x1) / 2, y, [(cnt, "bold", tc)], ha="center")
            key.append((fill, [(nm, "normal", INK)]))
        else:
            key.append((fill, [(nm + " ", "normal", INK), (cnt, "bold", INK)]))
        cur += sw
    return key


def _keyline(c, a, x, y, limit, items):
    """swatch-and-name entries on one line; the gap between entries shrinks from 8 pt to 5 pt before the line may overflow"""
    for gap in (8, 7, 6, 5):
        need = sum(7.5 + _measure(c, a, [(s, wt) for s, wt, _ in runs]) for _, runs in items) + gap * (len(items) - 1)
        if x + need <= limit:
            break
    assert x + need <= limit, (need, limit)
    for fill, runs in items:
        x = _key(c, a, x, y, fill, runs) + gap


def physical(name="f04_physical_counts_013"):
    cats = sorted(CAT, key=lambda k: -CAT[k]["n"])
    n = [CAT[k]["n"] for k in cats]
    p = [CAT[k]["physical_pass"] for k in cats]
    ne = [CAT[k]["no_structured_export"] for k in cats]
    assert (
        sum(n) == 2000
        and sum(p) == 1703
        and sum(ne) == 297
        and all(
            CAT[k]["physics_validation_failed"] == 0
            and CAT[k]["infrastructure_blocked"] == 0
            and CAT[k]["not_run"] == 0
            for k in cats
        )
    )
    H = 177
    c = Canvas(H)
    AX = (34, 47, 358, 124)
    XL = (-0.6, len(cats) - 0.4)
    YL = (0, 300)
    ax = c.axes(*AX)
    xs = np.arange(len(cats))
    ax.bar(xs, p, width=0.72, color=TEAL, linewidth=0, zorder=3)
    ax.bar(xs, ne, width=0.72, bottom=p, color=NOEXPORT, linewidth=0, zorder=3)
    ax.set(xlim=XL, ylim=YL, yticks=[0, 100, 200, 300])
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(k, k) for k in cats], rotation=60, ha="right", fontsize=8, rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=8, pad=2, length=2.5)
    ax.tick_params(axis="x", pad=1, length=0)
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Inputs", fontsize=8, labelpad=2)
    c.labels.extend(ax.get_xticklabels() + ax.get_yticklabels())
    # inset panel in the region no bar reaches: right of Phone (index 7, 96 inputs) and above 100 inputs (Cart, index 8, has 89).
    # Its bottom edge sits on the 100-input gridline. Inside: the stacked-bar key, then two composition bars.
    X0, Y0 = 7.5, 100

    def dx(v):
        return AX[0] + (v - XL[0]) / (XL[1] - XL[0]) * AX[2]

    def dy(v):
        return AX[1] + (v - YL[0]) / (YL[1] - YL[0]) * AX[3]

    ix, iy, iw, ih = dx(X0), dy(Y0), dx(XL[1]) - dx(X0), dy(YL[1]) - dy(Y0)
    assert all(CAT[k]["n"] < Y0 for k in cats[8:]) and CAT[cats[7]]["n"] < 300
    ins = c.axes(ix, iy, iw, ih)
    ins.set(xlim=(0, iw), ylim=(0, ih))
    ins.axis("off")
    ins.add_patch(Rectangle((0, 0), iw, ih, fc="white", ec="none", lw=0, zorder=0))
    pad = 3
    lh = 9.8
    bh = 9
    right = iw - pad
    x = pad
    y = ih - pad - lh / 2
    legend = [("Physical pass", TEAL), ("No structured export", NOEXPORT)]
    for i, (lab, col) in enumerate(legend):
        x = _key(c, ins, x + (8 if i else 0), y, col, [(lab, "normal", INK)])
    reasons = X1["failure_reasons_nonexclusive"]
    rk = [
        ("target_not_localized", "Target not localized"),
        ("candidate_budget_exhausted", "Candidate budget exhausted"),
        ("invalid_automatic_box", "Invalid automatic box"),
        ("model_output_not_object", "Model output not an object"),
    ]
    assert (
        sum(reasons[k] for k, _ in rk) == 297
        and reasons["invalid_automatic_box"] + reasons["model_output_not_object"] == 3
    )
    # one hue per bar, stepped dark to light in drawing order (largest share darkest); both ramps pass the dataviz palette
    # validator in --ordinal mode (monotone lightness, adjacent dL >= .06, light end >= 2:1 against the surface)
    segA = [
        ("Target not localized", reasons["target_not_localized"], _shade(RED, 0.78)),
        ("Candidate budget exhausted", reasons["candidate_budget_exhausted"], RED),
        (
            "Invalid box or non-object reply",
            reasons["invalid_automatic_box"] + reasons["model_output_not_object"],
            _tint(RED, 0.6),
        ),
    ]
    att = X1["attempt_status_counts"]
    ak = [
        ("selected", "Selected"),
        ("not_exported", "No export"),
        ("semantic_rejection", "Semantic rejection"),
        ("construction_physics_failed", "Failed physical gate"),
        ("repair_not_exported", "Repair, no export"),
    ]
    total = sum(att.values())
    assert att["selected"] == 1703 and total == 5874 and set(att) == {k for k, _ in ak}
    # selected = teal (the pass colour of the category bars); the four non-selected outcomes are one amber family
    fills = [TEAL, _shade(AMBER, 0.58), _shade(AMBER, 0.8), AMBER, _tint(AMBER, 0.7)]
    segB = [(lab, att[k], f) for (k, lab), f in zip(ak, fills)]
    keys = {}
    for title, segs, tag in [
        ("297 inputs without an asset, by terminal reason", segA, "reasons"),
        ("5,874 candidate attempts, by outcome", segB, "attempts"),
    ]:
        y -= lh + 1
        _runs(c, ins, pad, y, [(title, "normal", INK)])
        y -= lh / 2 + 1 + bh / 2
        keys[tag] = _composition(c, ins, pad, y, iw - 2 * pad, bh, segs)
        y -= bh / 2 + 1.5 + lh / 2
        _keyline(c, ins, pad, y, right, keys[tag])
        y -= 4
    assert y - lh / 2 + 4 >= 0, (y, ih)
    c.save(
        name,
        "figures",
        dict(
            categories=cats,
            n=n,
            physical_pass=p,
            no_structured_export=ne,
            terminal_reasons={k: reasons[k] for k, _ in rk},
            attempt_status_counts=att,
            repairs_attempted=X1["repairs_attempted"],
            passes_whose_selected_asset_was_repaired=X1["passes_whose_selected_asset_was_repaired"],
            source=str(EV.name),
            campaign="main campaign, formal-code-009",
            physics_failed_terminal_state_count=0,
            layout="category axes with an inset panel (bottom on the 100-input gridline, left of Phone) holding the stacked-bar key and two 100 % composition bars",
            composition_bars={"reasons": [(nm, v) for nm, v, _ in segA], "attempts": [(nm, v) for nm, v, _ in segB]},
            merged_segment="Invalid box or non-object reply = invalid_automatic_box 2 + model_output_not_object 1",
            segment_fills={"reasons": [f for _, _, f in segA], "attempts": fills},
            names_in_key_only={t: [r[0][0].strip() for _, r in ks] for t, ks in keys.items()},
            text_moved_to_caption=[
                "total",
                "panel headings",
                "repair footnote",
                "full names of the merged reason segment",
            ],
            legend_in_figure=[l for l, _ in legend],
        ),
    )


def all_categories(name="s01_all_categories_013"):
    cats = sorted(CAT, key=lambda k: (-CAT[k]["n"], k))
    rows = []
    pitch = 8.9
    h = round(len(cats) * pitch + 18)
    c = Canvas(h)
    for x, v in [(140, 0), (140 + 210 * 0.25, 25), (140 + 210 * 0.5, 50), (140 + 210 * 0.75, 75), (350, 100)]:
        c.line(x, 12, x, h - 3, LIGHT, 0.5)
        c.text(x, 6, f"{v}%", 8, GRAY, ha="center")
    for i, k in enumerate(cats):
        d = CAT[k]
        y = h - 7 - i * pitch
        r = d["physical_pass"] / d["n"]
        lo, hi = wilson(d["physical_pass"], d["n"])
        assert abs(lo - d["wilson95"]["lower"]) < 1e-9 and abs(hi - d["wilson95"]["upper"]) < 1e-9
        c.text(8, y, SHORT.get(k, k), 8)
        c.text(112, y, f"{d['physical_pass']}/{d['n']}", 8, INK, ha="right")
        c.rect(140, y - 2.6, 210 * r, 5.2, fill=TEAL, z=3)
        c.line(140 + 210 * lo, y, 140 + 210 * hi, y, INK, 0.8, z=4)
        c.line(140 + 210 * lo, y - 2, 140 + 210 * lo, y + 2, INK, 0.8, z=4)
        c.line(140 + 210 * hi, y - 2, 140 + 210 * hi, y + 2, INK, 0.8, z=4)
        c.text(388, y, f"{100*r:.1f}", 8, INK, ha="right")
        rows.append(
            dict(
                category=k,
                n=d["n"],
                physical_pass=d["physical_pass"],
                no_structured_export=d["no_structured_export"],
                rate=r,
                wilson95=[lo, hi],
            )
        )
    assert sum(x["n"] for x in rows) == 2000 and sum(x["physical_pass"] for x in rows) == 1703
    c.save(
        name,
        "figures",
        dict(
            rows=rows,
            n=2000,
            physical_pass=1703,
            source=str(EV.name),
            campaign="main campaign, formal-code-009",
            sorted_by="input count, descending",
            text_moved_to_caption=["column headings"],
        ),
    )


if __name__ == "__main__":
    physical()
    all_categories()
    print("ok")
