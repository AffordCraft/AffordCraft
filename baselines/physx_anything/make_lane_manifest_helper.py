#!/usr/bin/env python
"""Write lane-<k>/input_manifest.jsonl for the HELPER lanes of the PhysX-Anything remaining-1800 run:
the tail rows (position >= SPLIT_AT) of every lane manifest of the remaining-1800 run (PHYSX_ANYTHING_REMAINING_RUN),
ordered by (position descending, lane ascending), round-robin over HELPER_LANES lanes (position % HELPER_LANES == lane
over that ordering). The lanes of the remaining-1800 run stop their VLM stage once SPLIT_AT rows have a VLM result, so
every input is computed exactly once (a row can only be duplicated if such a lane had started it in the polling minute;
duplicates are identical computations and are listed by the aggregator). Verifies the frozen manifest, the subset file, the remaining-run lane manifests and every input image sha256;
refuses to change an existing lane manifest."""
import sys, json, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SPLIT_AT = 100
HELPER_LANES = 9
REV065 = Path(os.environ.get("PHYSX_ANYTHING_REMAINING_RUN", str(RUNS / "external/physx-anything/remaining1800")))


def helper_rows():
    assert sha256(FULL_MANIFEST) == FULL_MANIFEST_SHA and sha256(SUBSET) == SUBSET_SHA
    pool = []
    for k in range(10):
        rows = rows_of(REV065 / f"lane-{k}" / "input_manifest.jsonl")
        assert len(rows) == 180, k
        for pos, r in enumerate(rows):
            if pos >= SPLIT_AT:
                pool.append((pos, k, r))
    pool.sort(key=lambda t: (-t[0], t[1]))
    assert len(pool) == 10 * (180 - SPLIT_AT)
    return pool


def main():
    lane_dir, lane, lanes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert lanes == HELPER_LANES
    pool = helper_rows()
    out = [dict(r, origin_lane=k, origin_position=pos) for i, (pos, k, r) in enumerate(pool) if i % lanes == lane]
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
                "input_set": "remaining-1800 lane rows at position >= %d (tail-first order)" % SPLIT_AT,
                "lane_manifest_sha256": sha256(dst),
            }
        )
    )


if __name__ == "__main__":
    main()
