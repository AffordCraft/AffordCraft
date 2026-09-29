"""Table 2 from the verified one-factor study (frozen method code formal-code-010), exported by export_paper_evidence.py
as campaign/evidence_v2_20260922-193012.json. Two tracks: the registered 200-input subset (all ten one-factor conditions,
paired against the study's own run of the frozen method) and the 548 Window/Door/StorageFurniture inputs (installation
evidence and the decomposition cascade only). Every number is read from the evidence file; the asserts pin the values
quoted in the text. Writes CRLF like tables_comparison.py and a facts file.
The gate scores physics, not whether the asset is the requested object, so a Correct column comes from
campaign/ablation_task_fidelity_20260925.json (analysis/ablation_task_fidelity.py over the study's per-case records): a
pass whose delivered entry carries the requested category. Rate, Delta, Gain/Loss and p refer to Correct; Pass stays as
the physical count, and the receipt's physical-pass counts and gain/loss must equal the verified study summary."""

import json, math
from pathlib import Path
import argparse

_AP = argparse.ArgumentParser(description="Table 2 (paired one-factor study)")
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
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
E = json.load(open(IN / "campaign/evidence_v2_20260922-193012.json", encoding="utf-8"))
S = E["study"]
V = S["verifier_summary"]
assert (
    S["complete"]
    and V["complete"]
    and V["one_factor_only"]
    and V["source_hashes_unchanged"]
    and S["method_version"] == "formal-code-010"
)
T200 = V["tracks"]["subset200"]
T548 = V["tracks"]["category548"]
assert T200["denominator"] == 200 and T548["denominator"] == 548
assert S["tracks"]["subset200"]["input_sha256"] == "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
TF = json.load(open(IN / "campaign/ablation_task_fidelity_20260925.json", encoding="utf-8"))["tracks"]


def write(name, lines):
    (TAB / name).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8", newline="")


def pval(p):
    if p >= 0.995:
        return "1.00"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    m = p / 10**e
    return f"{m:.1f}\\times 10^{{{e}}}"


def rows(track, names, n, tf):
    pv = track["per_variant"]
    pf = track["paired_vs_full"]
    out = []
    facts = []
    for key, title in names:
        d = pv[key]
        assert d["complete"] and d["n"] == n and d["physical_pass"] == d["status_counts"]["physical_pass"]
        c = tf[key]
        assert c["n"] == n and c["passes"] == d["physical_pass"], (key, c["passes"], d["physical_pass"])
        if key == "full":
            out.append(
                r"\rowcolor{acPale}"
                + f'{title} & {d["physical_pass"]} & {c["correct"]} & {100 * c["correct"] / n:.1f} & -- & -- & -- & -- '
                + r"\\"
            )
            facts.append(
                dict(
                    variant=key,
                    passes=d["physical_pass"],
                    correct=c["correct"],
                    n=n,
                    rate=d["physical_pass"] / n,
                    correct_rate=c["correct"] / n,
                    wilson95=d["wilson95"],
                )
            )
            continue
        p = pf[key]
        assert (
            p["undecided_pairs"] == 0
            and p["both_pass"] + p["full_only"] == pv["full"]["physical_pass"]
            and p["both_pass"] + p["variant_only"] == d["physical_pass"]
        )
        assert abs(p["absolute_change_pp"] - 100 * (d["physical_pass"] - pv["full"]["physical_pass"]) / n) < 1e-9
        assert (c["passed_gain"], c["passed_loss"]) == (p["variant_only"], p["full_only"]), (
            key,
            c["passed_gain"],
            c["passed_loss"],
            p["variant_only"],
            p["full_only"],
        )
        dpp = 100 * (c["correct"] - tf["full"]["correct"]) / n
        ds = ("$+$" if dpp > 0 else "$-$" if dpp < 0 else "") + f"{abs(dpp):.1f}"
        out.append(
            f'{title} & {d["physical_pass"]} & {c["correct"]} & {100 * c["correct"] / n:.1f} & {ds} & {c["correct_gain"]} & {c["correct_loss"]} & ${pval(c["correct_mcnemar_exact_two_sided"])}$ '
            + r"\\"
        )
        facts.append(
            dict(
                variant=key,
                passes=d["physical_pass"],
                correct=c["correct"],
                n=n,
                rate=d["physical_pass"] / n,
                correct_rate=c["correct"] / n,
                wilson95=d["wilson95"],
                delta_pp=p["absolute_change_pp"],
                variant_only=p["variant_only"],
                full_only=p["full_only"],
                both_pass=p["both_pass"],
                both_fail=p["both_fail"],
                mcnemar_exact_two_sided=p["mcnemar_exact_two_sided"],
                correct_delta_pp=dpp,
                correct_gain=c["correct_gain"],
                correct_loss=c["correct_loss"],
                correct_mcnemar_exact_two_sided=c["correct_mcnemar_exact_two_sided"],
                wrong_category=c["wrong_category"],
            )
        )
    return out, facts


N200 = [
    ("full", r"\textbf{\method\ (full method)}"),
    ("encoder_replacement", "Encoder replacement (CLIP for DINOv2)"),
    ("without_task_condition", "Without task condition"),
    ("without_multimodal_selection", "Without multimodal selection"),
    ("without_scale_adaptation", "Without scale adaptation"),
    ("without_articulation_adaptation", "Without articulation adapter"),
    ("without_physics_reselection", "Without physical-feedback reselection"),
    ("without_repair", "Without deterministic repair"),
    ("without_installation_evidence", "Without installation evidence"),
    ("without_decomposition_cascade", "Without decomposition cascade"),
]
N548 = [
    ("full", r"\textbf{\method\ (full method)}"),
    ("without_installation_evidence", "Without installation evidence"),
    ("without_decomposition_cascade", "Without decomposition cascade"),
]
r200, f200 = rows(T200, N200, 200, TF["subset200"])
r548, f548 = rows(T548, N548, 548, TF["category548"])
# values quoted in the text (Section 4, appendix, conclusion)
F = {f["variant"]: f for f in f200}
G = {f["variant"]: f for f in f548}
assert (
    F["full"]["passes"] == 169
    and F["without_articulation_adaptation"]["passes"] == 93
    and F["without_physics_reselection"]["passes"] == 147
    and F["without_decomposition_cascade"]["passes"] == 153
)
assert (
    F["encoder_replacement"]["passes"] == 176
    and F["without_multimodal_selection"]["passes"] == 175
    and F["without_task_condition"]["passes"] == 173
    and F["without_scale_adaptation"]["passes"] == 167
)
assert (
    F["without_repair"]["passes"] == 169
    and F["without_installation_evidence"]["passes"] == 169
    and G["full"]["passes"] == 411
    and G["without_installation_evidence"]["passes"] == 411
    and G["without_decomposition_cascade"]["passes"] == 408
)
assert (
    F["without_articulation_adaptation"]["full_only"],
    F["without_physics_reselection"]["full_only"],
    F["without_decomposition_cascade"]["full_only"],
) == (76, 22, 16)
assert (
    F["encoder_replacement"]["variant_only"],
    F["encoder_replacement"]["full_only"],
    F["without_multimodal_selection"]["variant_only"],
    F["without_multimodal_selection"]["full_only"],
) == (8, 1, 6, 0)
assert (
    F["without_installation_evidence"]["variant_only"] == 1
    and F["without_installation_evidence"]["full_only"] == 1
    and G["without_decomposition_cascade"]["full_only"] == 3
)
assert (
    F["without_scale_adaptation"]["full_only"] == 2
    and F["without_scale_adaptation"]["variant_only"] == 0
    and F["without_task_condition"]["variant_only"] == 6
    and F["without_task_condition"]["full_only"] == 2
)
# Correct counts quoted in the text
assert [
    F[k]["correct"]
    for k in (
        "full",
        "encoder_replacement",
        "without_task_condition",
        "without_multimodal_selection",
        "without_articulation_adaptation",
    )
] == [169, 173, 125, 173, 89]
assert (
    F["without_task_condition"]["correct_gain"],
    F["without_task_condition"]["correct_loss"],
    F["without_articulation_adaptation"]["correct_loss"],
) == (3, 47, 80)
assert (
    F["encoder_replacement"]["correct_gain"],
    F["encoder_replacement"]["correct_loss"],
    F["without_multimodal_selection"]["correct_gain"],
    F["without_multimodal_selection"]["correct_loss"],
) == (7, 3, 4, 0)
assert (
    F["without_physics_reselection"]["correct_loss"],
    F["without_decomposition_cascade"]["correct_loss"],
    F["without_scale_adaptation"]["correct_loss"],
) == (22, 16, 3)
GEN = "% Generated by analysis/tables/tables_ablation.py from campaign/evidence_v2_20260922-193012.json (one-factor study, formal-code-010) and campaign/ablation_task_fidelity_20260925.json (Correct column). Edit the generator, not numeric cells."
L = [
    GEN,
    r"\begin{table}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\renewcommand{\arraystretch}{1.0}",
]
L += [
    r"\caption{\textbf{Paired one-factor study} on the 200-input subset. Correct:",
    r"passes whose asset has the requested category; Rate, $\Delta$, Gain, and Loss",
    r"refer to it (Gain/Loss: inputs correct only with the variant or only with the",
    r"full method, shaded). $p$: exact two-sided McNemar test.}",
]
L.append(r"\label{tab:ablations}")
L.append(r"\begin{tabularx}{\linewidth}{@{}X r r r r r r r@{}}")
L.append(r"\toprule")
L.append(r"Configuration & Pass & Correct & {Rate (\%)} & {$\Delta$ (pp)} & Gain & Loss & {$p$} \\")
L.append(r"\midrule")
L += r200  # the 548-input track (r548) is quoted in the text and the appendix; it stays out of the table for the page budget
L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
write("ablations.tex", L)
(OUT / "facts").mkdir(parents=True, exist_ok=True)
(OUT / "facts/ablation_table_facts.json").write_text(
    json.dumps(
        dict(
            source="campaign/evidence_v2_20260922-193012.json",
            study_path=S["path"],
            method_version=S["method_version"],
            subset200=f200,
            category548=f548,
            subset200_input_sha256=S["tracks"]["subset200"]["input_sha256"],
            category548_input_sha256=S["tracks"]["category548"]["input_sha256"],
        ),
        indent=1,
    ),
    encoding="utf-8",
)
print("written tables/ablations.tex and facts/ablation_table_facts.json")
print("\n".join(r200 + r548))
