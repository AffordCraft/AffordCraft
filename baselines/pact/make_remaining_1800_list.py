#!/usr/bin/env python
"""Write the input list of the PAct remaining-1800 run: the 1,800 inputs of the frozen 2,000-input manifest outside the
registered 200-input subset, in frozen manifest order, in the subset schema the lane runner and the mask adapter read
(inputs[].input_id). Refuses to overwrite. Excerpt of the executed preparation script (its input-list part); the rest
of that script (deriving a per-run copy of the tools, packing a bundle for the compute nodes) is infrastructure.
Output: $AFFORDCRAFT_RUNS/inputs/remaining_1800_inputs.json."""
import json, os, hashlib
from pathlib import Path

P = Path(os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace"))
RUNS = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
FULL = RUNS / "inputs/input_manifest_2000.jsonl"
FULL_SHA = os.environ.get(
    "AFFORDCRAFT_MANIFEST_SHA256", "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"
)
SUBSET = RUNS / "inputs/subset_200_inputs.json"
SUBSET_SHA = os.environ.get(
    "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
)
LIST = RUNS / "inputs/remaining_1800_inputs.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


assert sha(FULL) == FULL_SHA and sha(SUBSET) == SUBSET_SHA, "frozen inputs drift"
rows = [json.loads(l) for l in open(FULL) if l.strip()]
assert len(rows) == 2000
sub = json.load(open(SUBSET))
sset = {x["input_id"] for x in sub["inputs"]}
assert len(sset) == 200
rem = [r for r in rows if r["source_id"] not in sset]
assert len(rem) == 1800
if not LIST.exists():
    LIST.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "input_id": r["source_id"],
                        "dataset": r.get("dataset"),
                        "requested_category": r["requested_category"],
                        "image": {"path": str(P / r["image"]["path"]), "sha256": r["image"]["sha256"]},
                        "remaining_index": k,
                    }
                    for k, r in enumerate(rem)
                ],
                "n": 1800,
                "source_sha256": FULL_SHA,
                "subset_sha256": SUBSET_SHA,
                "no_reference_boxes_or_masks": True,
                "order": "frozen manifest order over the 1800 inputs outside the registered 200-input subset (same rule as the remaining-1800 runs of PhysX-Anything and TRELLIS.2)",
            },
            indent=1,
        )
    )
print("list", LIST, "sha", sha(LIST)[:12])
