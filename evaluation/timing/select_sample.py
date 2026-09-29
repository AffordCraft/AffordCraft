#!/usr/bin/env python3
"""Clean-GPU timing sample. The first N inputs of the registered 200-input subset (subset_200_inputs.json, sha256
db043e72...) in sha256(input_id) order; same input records, same format. Prints the main campaign's statistics of those
inputs (its exp1 per-case records) next to the subset's, so the sample can be compared with the recorded run
like-for-like. usage: select_sample.py <subset> <exp1_per_case> <N> <out>   (the paper uses N = 32)"""
import sys, json, hashlib, statistics as st

sub, per, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
raw = json.load(open(sub))
cases = raw["inputs"]
pick = sorted(cases, key=lambda c: hashlib.sha256(c["input_id"].encode()).hexdigest())[:n]
ids = {c["input_id"] for c in pick}
pick = [c for c in cases if c["input_id"] in ids]  # registered subset order
json.dump(
    {
        "inputs": pick,
        "n": len(pick),
        "source_sha256": hashlib.sha256(open(sub, "rb").read()).hexdigest(),
        "selection": f"first {n} of the registered 200-input subset in sha256(input_id) order, kept in subset order",
        "no_reference_boxes_or_masks": True,
        "order": "registered_subset_order",
    },
    open(out, "w"),
    indent=1,
)
rows = {}
for l in open(per):
    r = json.loads(l)
    if r["input_id"] in {c["input_id"] for c in cases}:
        rows[r["input_id"]] = r


def e2e(r):
    return sum(p["runtime_seconds"] for p in r["phase_timings"])


def stats(keys):
    v = [e2e(rows[k]) for k in keys if k in rows]
    p = sum(rows[k]["physical_pass"] is True for k in keys if k in rows)
    return dict(
        n=len(v),
        passes=p,
        e2e_median=round(st.median(v), 1),
        e2e_mean=round(st.mean(v), 1),
        e2e_max=round(max(v), 1),
        gpu_min_per_pass=round(sum(v) / 60 / max(p, 1), 1),
    )


print(
    json.dumps(
        {
            "sample": stats(ids),
            "subset200": stats(set(rows)),
            "sample_ids": sorted(ids)[:3],
            "cats": sorted({c["target_noun"] for c in pick}),
        }
    )
)
