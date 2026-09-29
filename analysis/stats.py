#!/usr/bin/env python3
"""Reference implementations of the statistics used by the paper's generators, collected in one place for readers. The
generators keep their own identical inline copies (the executed code); this module is not imported by them.

- Wilson score interval, 95 %: aggregate_external.py (fractions, clipped to [0, 1]; z = 1.959963984540054) and the table
  generators (percent; z = 1.9599639845).
- Exact two-sided McNemar test on the discordant pairs (b, c): ablation_task_fidelity.py (Table 2, p column).
- Quantiles: nearest rank (aggregate_external.py P95, tables_comparison.py P90) and linear interpolation
  (export_paper_evidence.py P95 of the campaign, numpy.percentile default).
No bootstrap is used.
usage: python analysis/stats.py [--inputs analysis/results]   (self-check against recorded values)"""
import argparse, json, math
from pathlib import Path


def wilson(k, n, z=1.959963984540054):
    """95 % Wilson score interval of k successes in n trials, as fractions [lower, upper]."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - e), min(1.0, c + e)]


def wilson_pct(k, n, z=1.9599639845):
    """the table generators' variant: (lower, upper) in percent, not clipped"""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    """exact two-sided McNemar p for b and c discordant pairs (binomial test with p = 1/2)"""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def quantile_nearest_rank(v, q):
    """nearest-rank quantile: the ceil(q * n)-th smallest value (at least the first)"""
    v = sorted(v)
    return v[max(math.ceil(q * len(v)), 1) - 1] if v else None


def quantile_linear(xs, q):
    """linear interpolation between order statistics (numpy.percentile default)"""
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(Path(__file__).resolve().parent / "results"))
    IN = Path(ap.parse_args().inputs)
    lo, hi = wilson_pct(1703, 2000)
    assert (round(lo, 2), round(hi, 2)) == (83.52, 86.64), (lo, hi)  # main campaign, 1,703 / 2,000
    rec = json.load(open(IN / "campaign/ablation_task_fidelity_20260925.json"))
    n = 0
    for track, rows in rec["tracks"].items():
        for v, r in rows.items():
            for key in ("passed", "correct"):
                assert mcnemar_exact(r[key + "_gain"], r[key + "_loss"]) == r[key + "_mcnemar_exact_two_sided"], (
                    track,
                    v,
                    key,
                )
                n += 1
    print("wilson(1703, 2000) = [%.2f, %.2f] %%; %d McNemar p-values reproduced" % (lo, hi, n))
