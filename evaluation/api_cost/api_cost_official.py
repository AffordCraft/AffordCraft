#!/usr/bin/env python3
"""API cost per input of the three API-based routes at the model providers' OFFICIAL list prices, from the per-call usage
each run recorded:
  - GPT-6 Astra agent (OpenAI Standard, developers.openai.com/api/docs/models/gpt-6-astra, 2026-09):
    input $10/M, cached input $1/M, cache writes $12.50/M (1.25x input; prompt caching is on by default for GPT-5.6 and
    later), output $50/M; prompts over 272K input tokens: 2x input/cache rates and 1.5x output for the whole request.
    Per call (transcript.jsonl 'usage'): cached = prompt_tokens_details.cached_tokens; the recorded usage does not
    include cache_write_tokens, so two bounds are reported: uncached input at the input rate (lower) or all of it billed
    as cache writes (upper, the default implicit-caching case); the paper uses the upper bound;
  - Articulate-Anything (Google, gemini-2.5-flash): input $0.30/M, output $2.50/M (thinking included), per VLM call from
    vlm_transcript.jsonl; cached tokens are reported and billed at the input rate here if present (they are 0 in these
    records -- checked below);
  - PartCrafter (Google, gemini-3.8-flash, introductory price through 2026-12-31): input $0.75/M, output $3.75/M; one
    part-count call per input (subset run and remaining-1,800 run); the run keeps the reply, not the usage, so the tokens
    per call come from the API endpoint's per-call usage records of 569 of its calls (PartCrafter was the only client of
    that model on the evaluation key), summed per model: {"per_model": {"gemini-3.8-flash": {"calls", "prompt_tokens",
    "completion_tokens"}}}.
The price table is also published as official_prices.json (checked equal below).
usage: api_cost_official.py --agent-run DIR --articulate-anything-run DIR --partcrafter-usage JSON [--out JSON]"""
import argparse, glob, json, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument(
    "--agent-run", required=True, help="GPT-6 Astra agent run (lane-*/cases/<id>/{result.json, transcript.jsonl})"
)
ap.add_argument(
    "--articulate-anything-run",
    required=True,
    help="Articulate-Anything run (lane-*/cases/<id>/{result.json, vlm_transcript.jsonl})",
)
ap.add_argument(
    "--partcrafter-usage", required=True, help="per-model usage totals of the part-count calls (JSON, see docstring)"
)
ap.add_argument("--out", default="api_cost_official.json")
A = ap.parse_args()
out = {
    "prices_usd_per_million": {
        "gpt-6-astra": {
            "input": 10,
            "cached_input": 1,
            "cache_write": 12.5,
            "output": 50,
            "long_prompt_threshold": 272000,
        },
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    }
}
assert (
    json.load(open(Path(__file__).with_name("official_prices.json")))["prices_usd_per_million"]
    == out["prices_usd_per_million"]
)


def astra(u, write_uncached):
    p = int(u.get("prompt_tokens") or 0)
    c = int(u.get("completion_tokens") or 0)
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    unc = max(p - cached, 0)
    k_in, k_out = (2.0, 1.5) if p > 272000 else (1.0, 1.0)
    return (
        (unc * (12.5 if write_uncached else 10.0) * k_in + cached * 1.0 * k_in + c * 50.0 * k_out) / 1e6,
        p,
        cached,
        c,
        p > 272000,
    )


lo, hi, tok = [], [], [0, 0, 0, 0]
for d in sorted(glob.glob(str(Path(A.agent_run) / "lane-*/cases/*"))):
    if not Path(d, "result.json").exists():
        continue
    a = b = 0.0
    for l in open(Path(d, "transcript.jsonl")) if Path(d, "transcript.jsonl").exists() else []:
        r = json.loads(l)
        u = r.get("usage")
        if not u:
            continue
        x, p, cached, c, long_ = astra(u, False)
        a += x
        b += astra(u, True)[0]
        tok[0] += p
        tok[1] += cached
        tok[2] += c
        tok[3] += int(long_)
    lo.append(a)
    hi.append(b)
out["gpt6-astra-agent-v0.1"] = {
    "inputs": len(hi),
    "prompt_tokens": tok[0],
    "cached_prompt_tokens": tok[1],
    "completion_tokens": tok[2],
    "calls_over_272k": tok[3],
    "usd_per_input_mean_upper": round(sum(hi) / len(hi), 3),
    "usd_per_input_median_upper": round(st.median(hi), 3),
    "usd_total_upper": round(sum(hi), 2),
    "usd_per_input_mean_lower": round(sum(lo) / len(lo), 3),
    "usd_total_lower": round(sum(lo), 2),
}
per, cached_total = [], 0
for d in sorted(glob.glob(str(Path(A.articulate_anything_run) / "lane-*/cases/*"))):
    if not Path(d, "result.json").exists():
        continue
    x = 0.0
    t = Path(d, "vlm_transcript.jsonl")
    for l in open(t) if t.exists() else []:
        u = json.loads(l).get("usage") or {}
        cached_total += int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        x += (int(u.get("prompt_tokens") or 0) * 0.30 + int(u.get("completion_tokens") or 0) * 2.50) / 1e6
    per.append(x)
out["articulate-anything-v0.1"] = {
    "inputs": len(per),
    "cached_prompt_tokens_reported": cached_total,
    "usd_per_input_mean": round(sum(per) / len(per), 4),
    "usd_per_input_median": round(st.median(per), 4),
    "usd_total": round(sum(per), 2),
}
b = json.load(open(A.partcrafter_usage))["per_model"]["gemini-3.8-flash"]
pp, qq = b["prompt_tokens"] / b["calls"], b["completion_tokens"] / b["calls"]
out["partcrafter-v0.1"] = {
    "calls_sampled": b["calls"],
    "prompt_tokens_per_call": round(pp, 1),
    "completion_tokens_per_call": round(qq, 1),
    "usd_per_input_mean": round((pp * 0.75 + qq * 3.75) / 1e6, 4),
}
json.dump(out, open(A.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "prices_usd_per_million"}, indent=1))
