#!/usr/bin/env python
"""make_lane_manifests.py -- write <run root>/lane-<k>/input_manifest.jsonl for k in 0..lanes-1 from the frozen manifest
rows of the registered 200-input subset (subset order, subset_index % lanes == lane). Verifies every input image exists
and matches its sha256. Refuses to change an existing lane manifest (drift check)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FULL_MANIFEST, FULL_MANIFEST_SHA, P, REV, SUBSET, SUBSET_SHA, read_json, rows_of, sha256


def compute_assignment(lanes, write):
    assert sha256(FULL_MANIFEST) == FULL_MANIFEST_SHA, "frozen manifest hash drift"
    assert sha256(SUBSET) == SUBSET_SHA, "subset hash drift"
    rows = {r["source_id"]: r for r in rows_of(FULL_MANIFEST)}
    sub = read_json(SUBSET)
    ids = [x["input_id"] for x in sub["inputs"]]
    assert len(ids) == 200 and len(set(ids)) == 200
    for x in sub["inputs"]:
        r = rows[x["input_id"]]
        assert r["image"]["sha256"] == x["image"]["sha256"], x["input_id"]
        assert r["requested_category"] == x["target_noun"], x["input_id"]
        p = P / r["image"]["path"]
        assert p.is_file(), p
        assert sha256(p) == r["image"]["sha256"], ("image hash mismatch", x["input_id"])
    assignment = {}
    for k in range(lanes):
        out = [rows[i] for idx, i in enumerate(ids) if idx % lanes == k]
        lane_dir = REV / ("lane-%d" % k)
        dst = lane_dir / "input_manifest.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        if write:
            lane_dir.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                assert dst.read_text(encoding="utf-8") == text, "lane manifest drift: %s" % dst
            else:
                dst.write_text(text, encoding="utf-8")
        assignment["lane-%d" % k] = dict(
            rows=len(out),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            first=out[0]["source_id"],
            last=out[-1]["source_id"],
            categories=sorted({r["requested_category"] for r in out}),
        )
    return dict(lanes=lanes, assignment=assignment)


def main():
    lanes = int(sys.argv[1])
    write = "--dry-run" not in sys.argv
    print(json.dumps(compute_assignment(lanes, write), indent=1))


if __name__ == "__main__":
    main()
