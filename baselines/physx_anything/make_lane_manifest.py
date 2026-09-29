#!/usr/bin/env python
"""Write lane-<k>/input_manifest.jsonl: the registered 200-input subset (subset order) split round-robin (index % lanes == lane).
Rows are the frozen 2,000-input manifest rows (source_id, requested_category, image.path, image.sha256). Verifies the sha256 of
the frozen manifest, of the subset file, and of every input image of the lane; refuses to change an existing lane manifest.
"""
import sys, json, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    lane_dir, lane, lanes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert sha256(FULL_MANIFEST) == FULL_MANIFEST_SHA, "frozen manifest drift"
    assert sha256(SUBSET) == SUBSET_SHA, "registered subset drift"
    rows = {r["source_id"]: r for r in rows_of(FULL_MANIFEST)}
    assert len(rows) == 2000
    d = read_json(SUBSET)
    items = d["inputs"] if isinstance(d, dict) else d
    ids = [r["input_id"] for r in items]
    assert len(ids) == 200 and len(set(ids)) == 200, "subset must hold 200 unique inputs"
    missing = [i for i in ids if i not in rows]
    assert not missing, ("subset id missing from frozen manifest", missing[:3])
    for it in items:  # the subset carries its own sha256 of the same photograph; both must agree
        assert it["image"]["sha256"] == rows[it["input_id"]]["image"]["sha256"], (
            "subset/manifest sha disagreement",
            it["input_id"],
        )
    out = [dict(rows[i], subset_index=k) for k, i in enumerate(ids) if k % lanes == lane]
    for r in out:
        img = P / r["image"]["path"]
        assert img.is_file(), ("input image missing on this node", str(img))
        h = sha256(img)
        assert h == r["image"]["sha256"], ("input image sha256 mismatch", r["source_id"], h)
    dst = lane_dir / "input_manifest.jsonl"
    if dst.exists():
        old = [json.loads(x)["source_id"] for x in open(dst) if x.strip()]
        assert old == [r["source_id"] for r in out], "lane manifest drift"
        print("lane manifest unchanged rows=%d" % len(out))
    else:
        lane_dir.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("lane manifest written rows=%d" % len(out))
    print(
        json.dumps(
            {
                "lane": lane,
                "lanes": lanes,
                "rows": len(out),
                "subset_sha256": SUBSET_SHA,
                "lane_manifest_sha256": sha256(dst),
            }
        )
    )


if __name__ == "__main__":
    main()
