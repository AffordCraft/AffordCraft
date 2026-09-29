from pathlib import Path
import json, argparse, hashlib, sys

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from affordcraft import paths


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", required=True)
    a.add_argument("--manifest", required=True, help="calibration manifest: objects with source_rgb {path, sha256}")
    a.add_argument("--catalog", default=str(paths.catalog_path()))
    a.add_argument("--project", default=str(paths.PROJECT_ROOT), help="root for relative image paths")
    args = a.parse_args()
    P = Path(args.project)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.manifest)
    raw = json.loads(source.read_text())
    result = []
    catalog = {
        str(x["candidate_id"]): x
        for x in json.loads(Path(args.catalog).read_text())[
            "entries"
        ]
    }
    for i, r in enumerate(raw["objects"]):
        image = r["source_rgb"]
        path = Path(image["path"])
        path = path if path.is_absolute() else P / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == image["sha256"]
        noun = r.get("requested_category") or r.get("category")
        result.append(
            {
                "input_id": str(r.get("calibration_id", f"cal_{i:03}")),
                "image": {"path": str(path), "sha256": digest},
                "target_noun": noun,
                "instruction": "Construct an interactive simulation asset for the "
                + noun
                + " visible in this image. Preserve its mechanism and natural support conditions.",
                "calibration": True,
            }
        )
    (out / "inputs.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "calibration_cases": len(result),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "first_case": result[0],
            }
        )
    )


if __name__ == "__main__":
    main()
