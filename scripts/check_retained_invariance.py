"""Prove that the second detection pass is the identity on the retained (successfully parsed) outputs of a prior detection run.

For every prior scene with status complete: (1) the accepted cases re-derived from the stored `response` by the new
acceptance rule (vocabulary, confidence, valid box, degenerate-box and same-category duplicate filters) are exactly the
stored cases (same input ids, categories and normalised boxes); (2) the periodic-run stop would not have fired on any
prefix of the object list; (3) with the decoded text re-tokenised, the token-repeat stop would not have fired at any
16-token check point (approximate: re-tokenisation can differ from the generated ids; the exact answer comes from the
verification pass that re-generates the 32 images). Writes a JSON report; exit 0 only when (1) and (2) hold everywhere.
"""

from pathlib import Path
import argparse, json, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detect_scene_objects import (
    prefix_objects,
    box_of,
    periodic_run,
    token_repeat,
    accept_cases,
    GENERATION,
    FILTERS,
    CODE,
    CHECKPOINT,
)

sys.path.insert(0, str(CODE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default=str(CHECKPOINT))
    ap.add_argument("--skip-tokens", action="store_true")
    a = ap.parse_args()
    prior = Path(a.prior)
    categories = sorted({x["category"] for x in json.loads(Path(a.catalog).read_text())["entries"]})
    tok = None
    if not a.skip_tokens:
        from transformers import AutoProcessor

        tok = AutoProcessor.from_pretrained(a.tokenizer, local_files_only=True).tokenizer
    rows = []
    ok_cases = ok_periodic = ok_tokens = 0
    n = 0
    for p in sorted((prior / "scenes").glob("*.json")):
        d = json.loads(p.read_text())
        rec = d["scene"]
        if rec.get("status") != "complete":
            continue
        n += 1
        idx = d["inputs"][0]["automatic_grounding_provenance"]["model_call_index"] if d["inputs"] else None
        call = json.loads((prior / "model_calls" / f"call_{idx:07}.json").read_text()) if idx else None
        row = {"scene_id": rec["scene_id"], "model_call_index": idx, "stored_cases": len(d["inputs"])}
        if call is None:
            row.update(cases_identical=None, note="no accepted case, so no call index recorded")
            rows.append(row)
            ok_cases += 1
            ok_periodic += 1
            ok_tokens += 1
            continue
        objs, complete = prefix_objects(call["response"])
        row["complete_json"] = complete
        row["objects_in_reply"] = len(objs or [])
        scene = {"input_id": rec["scene_id"], "image": d["inputs"][0]["image"]}
        cases, skipped = accept_cases(objs, categories, scene, idx, "x")
        same = [(c["input_id"], c["target_noun"], c["automatic_grounding"]["bbox_normalized"]) for c in cases] == [
            (c["input_id"], c["target_noun"], c["automatic_grounding"]["bbox_normalized"]) for c in d["inputs"]
        ]
        row.update(cases_identical=bool(same and complete), skipped_by_new_filters=skipped)
        ok_cases += bool(same and complete)
        seq = [box_of(o) for o in objs]
        fired = [
            k
            for k in range(1, len(seq) + 1)
            if periodic_run(
                seq[:k],
                GENERATION["loop_min_objects"],
                GENERATION["loop_max_period"],
                GENERATION["loop_tolerance_1000"],
            )
        ]
        row["periodic_rule_fires_on_prefixes"] = fired
        ok_periodic += not fired
        if tok is not None:
            ids = tok.encode(call["response"], add_special_tokens=False)
            hits = []
            for k in range(4 * GENERATION["check_every_tokens"], len(ids) + 1, GENERATION["check_every_tokens"]):
                L = token_repeat(ids[:k], tuple(GENERATION["token_repeat_block"]), GENERATION["token_repeats"])
                if L:
                    hits.append({"at_token": k, "block": L})
            row["token_rule_fires_retokenised"] = hits
            row["retokenised_length"] = len(ids)
            row["stored_output_tokens"] = call.get("output_tokens")
            ok_tokens += not hits
        rows.append(row)
    report = {
        "prior": str(prior),
        "retained_scenes": n,
        "cases_identical": ok_cases,
        "periodic_rule_silent": ok_periodic,
        "token_rule_silent_retokenised": (ok_tokens if tok is not None else None),
        "filters": FILTERS,
        "generation": GENERATION,
        "identity_on_retained_outputs": ok_cases == n and ok_periodic == n,
        "rows": rows,
    }
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    return 0 if report["identity_on_retained_outputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
