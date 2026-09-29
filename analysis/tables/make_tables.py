"""Tables of the development evidence export (development runs under the target-assisted protocol); no manual result
cells. The manuscript uses the library table (catalog_details.tex: the same entries, coverage, top-1, median and P95
numbers, in a compact layout without the mean column); the other tables of this generator were superseded by the other
generators of this folder and are written to <out>/development_tables/ so that they never overwrite them."""

from pathlib import Path
import json
import argparse

_AP = argparse.ArgumentParser(description="tables of the development evidence export")
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
E = json.loads((IN / "development/evidence.json").read_text(encoding="utf-8"))
T = OUT / "development_tables"
T.mkdir(parents=True, exist_ok=True)


def table(name, label, caption, columns, rows, spec):
    text = "\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
    text += "\\begin{tabularx}{\\linewidth}{" + spec + "}\n\\toprule\n"
    text += " & ".join(columns) + " \\\\\n\\midrule\n"
    for row in rows:
        text += " & ".join(map(str, row)) + " \\\\\n"
    text += "\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    (T / name).write_text(text, encoding="utf-8")


def pct(p):
    return f"{100*p:.2f}\\%"


def count(k, n):
    return f"{k:,}/{n:,}"


def main():
    names = {
        "baseline": "Single-candidate + adaptation",
        "vlm_v02": "+ scale estimation",
        "vlm_v03": "+ selected scale hypotheses",
        "top5": "Multimodal candidate selection",
        "top5_art": "AffordCraft (derived track)",
    }
    timing = {r["condition"]: r for r in json.loads((IN / "development/catalog_reanalysis.json").read_text())}
    rr = []
    for k, name in names.items():
        d = E["historical"][k]
        ci = d["ci"]
        rr.append(
            [
                name,
                count(d["success"], d["total"]),
                pct(d["rate"]),
                f"{ci['lower']*100:.2f}--{ci['upper']*100:.2f}",
                "--",
            ]
        )
    table(
        "main_results.tex",
        "tab:main-results",
        "Retained construction results under the legacy target-assisted protocol. Confidence intervals are percentages. E2E denotes complete image-to-physical-validation latency; no eligible matched latency is available. Full-RGB external comparisons are reported only when complete.",
        ["Configuration", "Pass / input", "Rate", "95\\% CI", "\\metrichead{E2E\\\\time}"],
        rr,
        "@{}p{0.38\\linewidth}YYYY@{}",
    )
    labels = {
        "crop_only": "Crop-only control",
        "object_centric": "Category-conditioned",
        "top5_vlm": "Multimodal selection",
    }
    rr = []
    for k, name in labels.items():
        d = E["exp2"][k]
        e = d["metrics"]["asset_emitted"]["raw"]
        p = d["metrics"]["sim_ready_only"]["raw"]
        s = d["all_objects_physics_only"]
        # The source stores an explicit image-level summary, not inferred task success.
        if "raw" in s:
            s = s["raw"]
        sk = s.get("success", s.get("k"))
        sn = s.get("total", s.get("n", s.get("denominator", 50)))
        rr.append([name, count(e["success"], 237), count(p["success"], 237), pct(p["rate"]), count(sk, sn)])
    table(
        "multi_object.tex",
        "tab:multi-object",
        "Object-centric construction on 50 images / 237 instances. All-object counts are image-level physical-only passes, not semantic or scene-relation success.",
        ["Configuration", "Exported", "Physical pass", "Rate", "\\metrichead{All objects\\\\(images)}"],
        rr,
        "@{}p{0.32\\linewidth}YYYY@{}",
    )
    rr = []
    for k, name in [("full", "AffordCraft"), ("without_articulation_adapter", "Without articulation adapter")]:
        d = E["paired"]["arms"][k]
        rr.append(
            [
                name,
                count(d["raw_physics_gate"]["k"], 2000),
                count(d["role_aware_review"]["k"], 2000),
                count(d["role_aware_with_visibility"]["k"], 2000),
            ]
        )
    rr.extend(
        [
            [name, "--", "--", "--"]
            for name in [
                "Encoder replacement",
                "Without task condition",
                "Without multimodal selection",
                "Without scale adaptation",
            ]
        ]
    )
    table(
        "ablations.tex",
        "tab:ablations",
        "Paired post-selection physical replay, with separate post-hoc role review. Selected candidates are fixed for the first two rows. Dashes denote unmeasured paired physical outcomes, not failed runs.",
        ["Configuration", "Raw gate", "Role review", "\\metrichead{Review +\\\\visibility}"],
        rr,
        "@{}p{0.43\\linewidth}YYY@{}",
    )
    rr = []
    for k, name in [("untrained_vla", "Untrained action head"), ("trained_vla", "Trained action head")]:
        d = E["vla"]["by_variant"][k]
        ci = d["wilson_95_ci"]
        rr.append(
            [
                name,
                count(d["successes"], d["denominator"]),
                pct(d["raw_success_rate"]),
                f"{ci[0]*100:.2f}--{ci[1]*100:.2f}",
            ]
        )
    table(
        "vla_results.tex",
        "tab:vla-results",
        "Finite v0.167 simulation evaluation. Both 200-episode denominators retain five blocked initial states. The task suite is reused during development; this is not an unseen-instance test.",
        ["Condition", "Success / episodes", "Rate", "95\\% CI"],
        rr,
        "@{}p{0.35\\linewidth}YYY@{}",
    )
    rr = []
    for k, label in [
        ("base_library", "Base"),
        ("additional_instances", "More instances"),
        ("additional_mechanisms", "More mechanisms"),
        ("additional_categories", "More categories"),
        ("full_library", "Full"),
    ]:
        d = E["catalog"][k]
        q = timing[k]
        rr.append(
            [
                label,
                f"{d['catalog_size']:,}",
                pct(d["category_coverage"]["rate"]),
                pct(d["category_top1_agreement"]["rate"]),
                f"{1000*d['query_time_seconds_mean']:.2f}",
                f"{1000*q['query_scope_median_seconds']:.2f}",
                f"{1000*q['query_scope_p95_seconds']:.2f}",
            ]
        )
    table(
        "catalog_details.tex",
        "tab:catalog",
        "Recorded library diagnostics on 2,000 fixed queries per setting. Coverage and top-1 agreement are category-label metrics; query time is not end-to-end construction latency.",
        [
            "Library",
            "Entries",
            "Coverage",
            "Top-1",
            "\\metrichead{Mean\\\\(ms)}",
            "\\metrichead{Median\\\\(ms)}",
            "\\metrichead{P95\\\\(ms)}",
        ],
        rr,
        "@{}p{0.24\\linewidth}YYYYYY@{}",
    )
    rr = []
    for aid, d in E["vla"]["by_asset"].items():
        label = aid.replace("articulated_", "").replace("rigid_compact_", "").replace("rigid_", "").replace("_", " ")
        rr.append([label, count(d["untrained_vla"]["successes"], 20), count(d["trained_vla"]["successes"], 20)])
    table(
        "vla_assets.tex",
        "tab:vla-assets",
        "Per-asset paired simulation results, 20 episodes per condition. Identifiers distinguish assets; they do not denote new object categories.",
        ["Asset", "Untrained", "Trained"],
        rr,
        "@{}p{0.56\\linewidth}YY@{}",
    )
    rr = [
        ["PhysX-Omni", "2,000", "Running; full physical result pending"],
        ["PhysX-Anything", "200", "Exclusive device unavailable"],
        ["PAct", "200", "Automatic part-mask producer required"],
        ["GPT-6 Astra agent (adapted)", "200", "API credentials required"],
        ["AffordCraft, full-RGB control", "2,000 / 200", "Matched run pending"],
    ]
    table(
        "external_registry.tex",
        "tab:external-registry",
        "Registered full-RGB comparisons. Native and commonly adapted outputs will be measured separately; planned denominators are not completed sample counts.",
        ["Method", "Planned inputs", "Current quantitative status"],
        rr,
        "@{}p{0.34\\linewidth}p{0.16\\linewidth}Z@{}",
    )
    (T / "paper_values.tex").write_text(
        "% Generated from data/evidence.json by tools/make_tables.py.\n\\newcommand{\\InputCount}{2,000}\n",
        encoding="utf-8",
    )
    print("Generated 4 main tables and 3 appendix tables.")


if __name__ == "__main__":
    main()
