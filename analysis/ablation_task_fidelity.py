#!/usr/bin/env python3
"""Task fidelity of the one-factor study (read-only over the study's per-case records, frozen method code
formal-code-010): the paired one-factor study scored the physical gate only, which cannot tell whether a passing asset is the requested object.
For every variant and input this counts
  correct - physical pass AND the delivered library entry carries the requested category (its catalog label, as listed
            in the case's ordered candidate pool, equals requested_category of the frozen 2,000-input manifest; the same
            label test as the top-1 agreement of the library-scaling analysis),
and, against the study's own run of the frozen method, gain/loss (correct only under the variant / only under the
reference) and the exact two-sided McNemar p. Physical-pass gain/loss is recomputed from the same records as a check
against the verified study summary. Writes a receipt with source hashes and a per-case JSONL.
usage: ablation_task_fidelity.py <study_results dir> <input manifest> <out prefix>"""
import collections, hashlib, json, math, sys

S, M, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TRACKS = {
    "subset200": [
        "full",
        "encoder_replacement",
        "without_task_condition",
        "without_multimodal_selection",
        "without_scale_adaptation",
        "without_articulation_adaptation",
        "without_physics_reselection",
        "without_repair",
        "without_installation_evidence",
        "without_decomposition_cascade",
    ],
    "category548": ["full", "without_installation_evidence", "without_decomposition_cascade"],
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


req = {}
for l in open(M):
    if l.strip():
        r = json.loads(l)
        req[r["source_id"]] = r["requested_category"]
rec = {
    "written": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
    "study_results": S,
    "manifest": M,
    "manifest_sha256": sha(M),
    "definition": "correct = physical_pass and catalog label of the delivered entry (ordered_candidate_pool) == requested_category of the frozen manifest",
    "tracks": {},
    "sources": {},
}
cases = open(OUT + "_cases.jsonl", "w")
for track, variants in TRACKS.items():
    per = {}
    for v in variants:
        f = f"{S}/per_case_{track}_{v}.jsonl"
        rec["sources"][f] = sha(f)
        per[v] = {}
        for l in open(f):
            r = json.loads(l)
            sid = r["input_id"]
            ok = bool(r.get("physical_pass"))
            sel = (r.get("selected_asset") or {}).get("candidate_id")
            cat = (
                next(
                    (c.get("category") for c in r.get("ordered_candidate_pool") or [] if c.get("candidate_id") == sel),
                    None,
                )
                if sel
                else None
            )
            per[v][sid] = dict(
                passed=ok, correct=ok and cat == req[sid], delivered_category=cat if ok else None, requested=req[sid]
            )
            cases.write(json.dumps(dict(track=track, variant=v, input_id=sid, **per[v][sid])) + "\n")
    base = per["full"]
    out = {}
    for v in variants:
        d = per[v]
        assert set(d) == set(base)
        row = dict(
            n=len(d),
            passes=sum(x["passed"] for x in d.values()),
            correct=sum(x["correct"] for x in d.values()),
            wrong_category=collections.Counter(
                (x["requested"], x["delivered_category"]) for x in d.values() if x["passed"] and not x["correct"]
            ).most_common(),
        )
        for key in ("passed", "correct"):
            g = sum(1 for s in d if d[s][key] and not base[s][key])
            lo = sum(1 for s in d if base[s][key] and not d[s][key])
            row[key + "_gain"] = g
            row[key + "_loss"] = lo
            row[key + "_mcnemar_exact_two_sided"] = mcnemar(g, lo)
        row["wrong_category"] = [[list(k), n] for k, n in row["wrong_category"]]
        out[v] = row
        print(
            f"{track:11s} {v:32s} pass {row['passes']:3d} correct {row['correct']:3d} gain {row['correct_gain']:2d} loss {row['correct_loss']:2d} p {row['correct_mcnemar_exact_two_sided']:.3g}"
        )
    rec["tracks"][track] = out
cases.close()
rec["cases_jsonl_sha256"] = sha(OUT + "_cases.jsonl")
json.dump(rec, open(OUT + ".json", "w"), indent=1)
print("written", OUT + ".json")
