"""One-to-one matching is evaluation-only and never used by construction."""

from pathlib import Path
import argparse, json, numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    inter = float(np.prod(np.maximum(0, hi - lo)))
    area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
    return inter / max(1e-12, area(a) + area(b) - inter)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--reference", required=True)
    a.add_argument("--detections", required=True)
    a.add_argument("--results", required=True)
    a.add_argument("--output", required=True)
    args = a.parse_args()
    ref = json.loads(Path(args.reference).read_text())["images"]
    det = json.loads(Path(args.detections).read_text())["inputs"]
    results = {r["input_id"]: r for l in Path(args.results).open() if (r := json.loads(l))}
    scenes = []
    pass_count = matched_count = 0
    complete = all(p["input_id"] in results and type(results[p["input_id"]].get("physical_pass")) is bool for p in det)
    assert len(ref) == 50 and sum(len(s["objects"]) for s in ref) == 237
    for scene in ref:
        preds = [x for x in det if x["scene_id"] == scene["scene_id"]]
        gt = scene["objects"]
        matrix = np.zeros((len(gt), len(preds)))
        for i, g in enumerate(gt):
            for j, p in enumerate(preds):
                if p["target_noun"] == g["category"]:
                    matrix[i, j] = iou(g["bbox_normalized"], p["automatic_grounding"]["bbox_normalized"])
        match = {}
        if matrix.size:
            ii, jj = linear_sum_assignment(-matrix)
            match = {int(i): int(j) for i, j in zip(ii, jj) if matrix[i, j] >= 0.5}
        rows = []
        for i, g in enumerate(gt):
            if i not in match:
                rows.append({"object_id": g["object_id"], "localized": False, "physical_pass": False})
                continue
            p = preds[match[i]]
            r = results.get(p["input_id"])
            value = r.get("physical_pass") if r else None
            if value is None:
                complete = False
            rows.append(
                {
                    "object_id": g["object_id"],
                    "localized": True,
                    "matched_input": p["input_id"],
                    "iou": float(matrix[i, match[i]]),
                    "physical_pass": value,
                }
            )
        pass_count += sum(x["physical_pass"] is True for x in rows)
        matched_count += len(match)
        scenes.append(
            {
                "scene_id": scene["scene_id"],
                "objects": rows,
                "all_objects_physical_pass": all(x["physical_pass"] is True for x in rows),
                "unmatched_detections": len(preds) - len(match),
            }
        )
    report = {
        "complete": complete,
        "reference_images": len(ref),
        "reference_objects": sum(len(s["objects"]) for s in ref),
        "localized": matched_count,
        "physical_passes": pass_count,
        "all_object_images": sum(x["all_objects_physical_pass"] for x in scenes),
        "scene_relation_success": None,
        "matching": "same_category_hungarian_IoU_at_least_0.5",
        "scenes": scenes,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}))


if __name__ == "__main__":
    main()
