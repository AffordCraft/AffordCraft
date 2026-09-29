"""Publication export must not turn partial coverage, calibration or old data into a main rate."""

from pathlib import Path
import argparse, json, math
from collections import Counter


def wilson(k, n):
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (k / n + z * z / (2 * n)) / d
    e = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / d
    return [c - e, c + e]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--inputs", required=True)
    a.add_argument("--output", required=True)
    args = a.parse_args()
    root = Path(args.run)
    summary = json.loads((root / "summary.json").read_text())
    inputs = json.loads(Path(args.inputs).read_text())
    inputs = inputs.get("inputs", inputs) if isinstance(inputs, dict) else inputs
    rows = [json.loads(l) for l in (root / "per_input.jsonl").open() if l.strip()]
    if summary.get("scope") != "formal" or not summary.get("complete"):
        raise RuntimeError("No publishable full formal result")
    if len(rows) != len(inputs) or {r["input_id"] for r in rows} != {r["input_id"] for r in inputs}:
        raise RuntimeError("Denominator or input identity mismatch")
    if len({r["input_id"] for r in rows}) != len(rows) or any(type(r.get("physical_pass")) is not bool for r in rows):
        raise RuntimeError("Duplicate or unresolved result")
    n = len(rows)
    k = sum(r["physical_pass"] for r in rows)
    by_id = {r["input_id"]: r for r in inputs}
    cats = {}
    for c in sorted({r.get("target_noun", "unspecified") for r in inputs}):
        rr = [r for r in rows if by_id[r["input_id"]].get("target_noun", "unspecified") == c]
        cats[c] = {"n": len(rr), "k": sum(x["physical_pass"] for x in rr)}
    result = {
        "method": "AffordCraft",
        "input_count": n,
        "physical_passes": k,
        "physical_pass_rate": k / n,
        "wilson_95": wilson(k, n),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "categories": cats,
        "unique_selected_candidates": len(
            {r["selected_asset"]["candidate_id"] for r in rows if r.get("selected_asset")}
        ),
        "source_run": str(root),
        "old_results_used": False,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "final_results.json").write_text(json.dumps(result, indent=2))
    (out / "final_result_values.tex").write_text(
        "\\resultsreadytrue\n"
        + r"\newcommand{\FinalAbstractResults}{On "
        + str(n)
        + r" frozen queries, the final construction attains "
        + str(k)
        + "/"
        + str(n)
        + f" physical passes ({100*k/n:.2f}"
        + r"\%). These are physical acceptance results, not independently verified task-match or robot-success rates.}"
        + "\n"
    )
    (out / "final_results_body.tex").write_text(
        r"\subsection{Final construction results}"
        + "\n"
        + f"The frozen method produces {k}/{n} independently validated physical outputs. "
        + r"Table~\ref{tab:final-main} reports the unchanged full-input denominator."
        + "\n"
        + r"\begin{table}[t]\centering\caption{Final physical acceptance under the common support-aware contract.}\label{tab:final-main}\begin{tabular}{lrrr}\toprule Method & Passes & Inputs & Rate (\%)\\\midrule AffordCraft & "
        + str(k)
        + " & "
        + str(n)
        + f" & {100*k/n:.2f}"
        + r"\\\bottomrule\end{tabular}\end{table}"
        + "\n"
    )
    print(json.dumps({"n": n, "k": k, "output": str(out)}))


if __name__ == "__main__":
    main()
