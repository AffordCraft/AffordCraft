#!/usr/bin/env python3
"""Build the runnable input manifests from the published input lists (data/inputs/) and local image copies.

The pipeline reads manifests with absolute image paths and image hashes. The repository ships the same inputs without
paths: Open Images V7 image ids for the 2,000 single-object photographs and COCO image ids for the 50 cluttered images.
This script resolves them against local dataset folders, checks every image hash, and writes manifests in exactly the
format of the frozen experiment definitions:

    single     2,000 single-object inputs, frozen manifest order                  -> single_inputs.json
    subset     registered 200-input subset (paired study, external methods)     -> subset_200.json
    category   every Window, Door and StorageFurniture input (548)              -> category_548.json
    cluttered  50 COCO images (model inputs) and their 237 reference objects     -> model_inputs.json, evaluation_only.json
    baselines  the two input files of the external-method wrappers (baselines/)   -> input_manifest_2000.jsonl,
                                                                                    subset_200_inputs.json

Examples:
    python scripts/prepare_inputs.py single --open-images /datasets/openimages_v7 --output runs/inputs/single_inputs.json
    python scripts/prepare_inputs.py subset --single runs/inputs/single_inputs.json --output runs/inputs/subset_200.json
    python scripts/prepare_inputs.py category --single runs/inputs/single_inputs.json --output runs/inputs/category_548.json
    python scripts/prepare_inputs.py cluttered --coco /datasets/coco/val2017 /datasets/coco/train2017 --output runs/inputs/scenes
    python scripts/prepare_inputs.py baselines --single runs/inputs/single_inputs.json
        --subset runs/inputs/subset_200.json --project-root $AFFORDCRAFT_PROJECT_ROOT --output $AFFORDCRAFT_RUNS/inputs
        (one command, wrapped here)

`--open-images` must hold the images as <split>/<image id>.jpg (split = train or validation), as written by the official
Open Images downloader. An image whose bytes differ from the recorded hash is refused unless --allow-hash-mismatch is
given, in which case the new hash is recorded and the mismatch is listed in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "inputs"
CATEGORIES = ("Window", "Door", "StorageFurniture")
CATEGORY_COUNTS = {"Window": 263, "Door": 145, "StorageFurniture": 140}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    print(json.dumps({"written": str(path), "sha256": sha(path)}))


def checked_image(path, expected, allow, mismatches, key):
    if not path.is_file():
        raise SystemExit(f"missing image for {key}: {path}")
    digest = sha(path)
    if digest != expected:
        if not allow:
            raise SystemExit(f"image hash differs for {key}: {path} (use --allow-hash-mismatch to record the new hash)")
        mismatches.append({"input_id": key, "expected_sha256": expected, "found_sha256": digest})
    return {"path": str(path.resolve()), "sha256": digest}


def single(args):
    listing = load("single_object_2000.json")
    root = Path(args.open_images)
    rows, mismatches = [], []
    for r in listing["inputs"]:
        image = root / r["open_images_split"] / (r["open_images_id"] + ".jpg")
        rows.append(
            {
                "input_id": r["input_id"],
                "dataset": "single_object",
                "image": checked_image(image, r["image_sha256"], args.allow_hash_mismatch, mismatches, r["input_id"]),
                "instruction": r["instruction"],
                "target_noun": r["target_noun"],
            }
        )
    out = {
        "inputs": rows,
        "n": len(rows),
        "source_sha256": sha(DATA / "single_object_2000.json"),
        "no_reference_boxes_or_masks": True,
    }
    if mismatches:
        out["image_hash_mismatches"] = mismatches
    write_new(args.output, out)


def subset(args):
    full = json.loads(Path(args.single).read_text(encoding="utf-8"))
    ids = load("subset_200.json")["input_ids"]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise SystemExit("subset must hold 200 distinct ids")
    by_id = {r["input_id"]: r for r in full["inputs"]}
    write_new(
        args.output,
        {
            "inputs": [by_id[i] for i in ids],
            "n": len(ids),
            "source_sha256": full["source_sha256"],
            "subset_sha256": sha(DATA / "subset_200.json"),
            "no_reference_boxes_or_masks": True,
            "order": "registered_subset_order",
        },
    )


def category(args):
    full = json.loads(Path(args.single).read_text(encoding="utf-8"))
    rows = [r for r in full["inputs"] if r["target_noun"] in CATEGORIES]
    counts = {c: sum(r["target_noun"] == c for r in rows) for c in CATEGORIES}
    if counts != CATEGORY_COUNTS:
        raise SystemExit("unexpected category counts: " + json.dumps(counts))
    write_new(
        args.output,
        {
            "inputs": rows,
            "n": len(rows),
            "source_sha256": full["source_sha256"],
            "categories": list(CATEGORIES),
            "counts": counts,
            "no_reference_boxes_or_masks": True,
            "order": "frozen_manifest_order",
        },
    )


def cluttered(args):
    scenes = load("cluttered_scenes_50.json")["inputs"]
    reference = load("cluttered_reference_objects.json")
    roots = [Path(p) for p in args.coco]
    inputs, mismatches = [], []
    for s in scenes:
        name = f"{s['coco_image_id']:012d}.jpg"
        found = next((r / name for r in roots if (r / name).is_file()), roots[0] / name)
        inputs.append(
            {
                "input_id": s["scene_id"],
                "image": checked_image(found, s["image_sha256"], args.allow_hash_mismatch, mismatches, s["scene_id"]),
                "instruction": s["instruction"],
            }
        )
    out = Path(args.output)
    model_inputs = {"inputs": inputs, "count": len(inputs)}
    if mismatches:
        model_inputs["image_hash_mismatches"] = mismatches
    write_new(out / "model_inputs.json", model_inputs)
    write_new(out / "evaluation_only.json", {"images": reference["images"], "objects": reference["objects"]})


def baselines(args):
    """input_manifest_2000.jsonl: one row per input in manifest order, image path relative to the project root when the
    image lies inside it (the wrappers resolve it against AFFORDCRAFT_PROJECT_ROOT); subset_200_inputs.json: the subset
    manifest. The wrappers pin the hashes of the paper's copies of both files; see baselines/README.md."""
    full = json.loads(Path(args.single).read_text(encoding="utf-8"))
    root = Path(args.project_root).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in full["inputs"]:
        image = Path(r["image"]["path"]).resolve()
        try:
            path = image.relative_to(root).as_posix()
        except ValueError:
            path = str(image)
        row = {
            "source_id": r["input_id"],
            "dataset": "exp1_2000",
            "requested_category": r["target_noun"],
            "image": {"path": path, "sha256": r["image"]["sha256"]},
        }
        lines.append(json.dumps(row) + "\n")
    with (out / "input_manifest_2000.jsonl").open("x", encoding="utf-8") as f:
        f.writelines(lines)
    print(
        json.dumps(
            {"written": str(out / "input_manifest_2000.jsonl"), "sha256": sha(out / "input_manifest_2000.jsonl")}
        )
    )
    subset_manifest = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    write_new(out / "subset_200_inputs.json", subset_manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("single")
    p.add_argument("--open-images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-hash-mismatch", action="store_true")
    for name in ("subset", "category"):
        p = sub.add_parser(name)
        p.add_argument("--single", required=True, help="manifest written by the 'single' command")
        p.add_argument("--output", required=True)
    p = sub.add_parser("cluttered")
    p.add_argument("--coco", nargs="+", required=True, help="folders holding COCO images named <12-digit id>.jpg")
    p.add_argument("--output", required=True, help="output folder")
    p.add_argument("--allow-hash-mismatch", action="store_true")
    p = sub.add_parser("baselines")
    p.add_argument("--single", required=True, help="manifest written by the 'single' command")
    p.add_argument("--subset", required=True, help="manifest written by the 'subset' command")
    p.add_argument("--project-root", required=True, help="root against which the wrappers resolve image paths")
    p.add_argument("--output", required=True, help="output folder (the wrappers read $AFFORDCRAFT_RUNS/inputs)")
    args = ap.parse_args()
    commands = {
        "single": single,
        "subset": subset,
        "category": category,
        "cluttered": cluttered,
        "baselines": baselines,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
