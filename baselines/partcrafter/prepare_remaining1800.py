#!/usr/bin/env python
"""Prepare a PartCrafter run on the REMAINING 1,800 inputs (= frozen 2,000 manifest minus the registered 200 subset,
frozen manifest order): verify inputs, write configuration.json first (open 'x'), then the fixed round-robin lane
manifests, and copy the runner code of the registered-subset run (PARTCRAFTER_SUBSET_RUN) into <root>/code/. Code,
checkpoints, environment, generation parameters and the part-count model (gemini-3.8-flash through the OpenAI-compatible
API, official prompt and parser) are those of the registered-subset run, whose 200 finished cases serve as the smoke
test; only the input set and the lane count differ. usage: prepare_remaining1800.py --root <run dir> --lanes 44
Environment: AFFORDCRAFT_PROJECT_ROOT, AFFORDCRAFT_RUNS, PARTCRAFTER_SUBSET_RUN."""
import argparse, datetime as _dt, hashlib, json, os, shutil, sys

P = os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace")
RUNS = os.environ.get("AFFORDCRAFT_RUNS", "runs")
MANIFEST = f"{RUNS}/inputs/input_manifest_2000.jsonl"
SUBSET = f"{RUNS}/inputs/subset_200_inputs.json"
OLD = os.environ.get("PARTCRAFTER_SUBSET_RUN", f"{RUNS}/external/partcrafter/subset200")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lanes", type=int, default=44)
    a = ap.parse_args()
    root = a.root
    os.makedirs(root, exist_ok=True)
    assert not os.path.exists(os.path.join(root, "configuration.json")), "configuration.json exists; append-only"
    man_sha, sub_sha = sha256(MANIFEST), sha256(SUBSET)
    assert man_sha == os.environ.get(
        "AFFORDCRAFT_MANIFEST_SHA256", "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"
    ), man_sha
    assert sub_sha == os.environ.get(
        "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
    ), sub_sha
    old = json.load(open(os.path.join(OLD, "configuration.json")))
    assert old["method_id"] == "partcrafter-v0.1" and old["inputs"]["manifest_sha256"] == man_sha
    sset = {s["input_id"] for s in json.load(open(SUBSET))["inputs"]}
    assert len(sset) == 200
    rows = []
    for l in open(MANIFEST):
        if not l.strip():
            continue
        m = json.loads(l)
        if m["source_id"] in sset:
            continue
        h = sha256(os.path.join(P, m["image"]["path"]))
        assert h == m["image"]["sha256"], (m["source_id"], h)
        rows.append(
            {
                "source_id": m["source_id"],
                "requested_category": m["requested_category"],
                "image": {"path": m["image"]["path"], "sha256": h},
            }
        )
    assert len(rows) == 1800
    lanes = a.lanes
    partition = {f"lane-{k}": [r["source_id"] for i, r in enumerate(rows) if i % lanes == k] for k in range(lanes)}
    code_dir = os.path.join(root, "code")
    os.makedirs(code_dir, exist_ok=True)
    code = {}
    for f in sorted(os.listdir(os.path.join(OLD, "code"))):
        if f.endswith((".py", ".sh")):
            shutil.copy2(os.path.join(OLD, "code", f), os.path.join(code_dir, f))
            code[f] = sha256(os.path.join(code_dir, f))
    shutil.copy2(os.path.abspath(__file__), os.path.join(code_dir, "prepare_remaining1800.py"))
    code["prepare_remaining1800.py"] = sha256(os.path.join(code_dir, "prepare_remaining1800.py"))
    cfg = dict(old)
    cfg.update(
        {
            "revision": os.path.basename(root),
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "task": "PartCrafter part-level mesh generation on the 1,800 inputs of the frozen 2,000-input manifest outside the registered 200-input subset (frozen manifest order)",
            "inputs": {
                **old["inputs"],
                "n_cases": len(rows),
                "order": "frozen manifest order, registered subset excluded",
                "smoke_subset": None,
                "input_set": "frozen_manifest_2000_minus_registered_subset_200",
            },
            "part_suggest": {
                **old["part_suggest"],
                "endpoint_note": "same OpenAI-compatible endpoint and model as the registered-subset run; the endpoint and key come from EVAL_API_BASE / EVAL_API_KEY and are never written to records",
            },
            "lanes": {
                "n_lanes": lanes,
                "rule": "fixed partition: remaining index i -> lane i mod n_lanes; each lane runs on exactly one node/GPU at a time (lane_assignment.jsonl appended at every launch)",
                "partition": partition,
                "lane_env": "code/pc_lane.sh (PC_ROOT, PC_LANES, PC_GPU, PC_TAG)",
                "assignment_log": "lane_assignment.jsonl",
            },
            "provenance": {
                "parent_revision": old["revision"],
                "parent_configuration_sha256": sha256(os.path.join(OLD, "configuration.json")),
                "note": "identical official code, checkpoints, environment, generation parameters and part-count model as the registered-subset run, whose 200 finished cases serve as the smoke test; only the input set and the lane count differ",
            },
            "code_sha256": code,
        }
    )
    with open(os.path.join(root, "configuration.json"), "x") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for k in range(lanes):
        d = os.path.join(root, f"lane-{k}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "input_manifest.jsonl"), "x") as f:
            for r in rows:
                if r["source_id"] in partition[f"lane-{k}"]:
                    f.write(json.dumps(r) + "\n")
    print(f"prepared {root}: {len(rows)} cases, {lanes} lanes")


if __name__ == "__main__":
    sys.exit(main())
