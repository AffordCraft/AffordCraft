# api_cost/ -- API cost at official list prices

Purpose: the API cost per input of the three routes that call a hosted model (GPT-6 Astra agent, Articulate-Anything,
PartCrafter's part-count query), from the token counts each run recorded, priced at the model providers' official list
prices. Supports the "API cost (USD / input)" row of the resource table; the methods that run only local models cost 0.

## Entry point

```bash
python evaluation/api_cost/api_cost_official.py --agent-run <GPT-6 agent run> \
    --articulate-anything-run <Articulate-Anything run> --partcrafter-usage partcrafter_usage.json --out api_cost_official.json
```

Inputs: the two run directories (`lane-*/cases/<id>/{result.json, transcript.jsonl}` with one `usage` record per model
call for the agent; `vlm_transcript.jsonl` for Articulate-Anything); for PartCrafter, whose run keeps the model's reply
but not its usage, the per-call usage records of the API endpoint for the part-count model summed per model
(`{"per_model": {"gemini-3.8-flash": {"calls", "prompt_tokens", "completion_tokens"}}}`; 569 calls in the paper).
`official_prices.json` holds the price table with its rules and sources; the script checks that it equals the prices in
the code. Output: tokens, totals and USD per input (GPT-6 agent: lower and upper bound, the paper reports the upper
bound, i.e. every uncached prompt token billed as a cache write). The paper's output is
`analysis/results/external/api_cost_official_20260925.json`. Standard library only.

## Changes from the executed version

Run directories and the usage file became options (the executed version read fixed run and dispatch directories); the
output path became `--out`; `official_prices.json` was added (same numbers as the code) together with the equality check;
docstring without dates and without the name of the API endpoint. Not included: the script that read the endpoint's
usage log (it needs the endpoint's address and key) and a variant priced at the endpoint's own rates.

## Provenance

| Original file | Published path | sha256 of the original |
|---|---|---|
| api_cost_official_20260925.py | evaluation/api_cost/api_cost_official.py | 74ab27b0940c6da0597a543084e1e0fc38b26211ed2443cd0e3f6a1392ea5e06 |
| (prices of api_cost_official_20260925.py) | evaluation/api_cost/official_prices.json | -- |
