from pathlib import Path
import argparse, json, os, sys, time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--inputs", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--limit", type=int, default=3)
    args = a.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from affordcraft.vision import LocalVisualBackend
    from affordcraft import paths

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(Path(args.inputs).read_text())[: args.limit]
    (out / "process.json").write_text(
        json.dumps(
            {"pid": os.getpid(), "gpu": args.gpu, "argv": sys.argv, "kind": "non_test_catalog_render_calibration"},
            indent=2,
        )
    )
    model = LocalVisualBackend(paths.QWEN3_VL, out / "model_calls")
    records = []
    for c in cases:
        try:
            r = model.ground(c)
        except Exception as e:
            r = {"localized": False, "error": repr(e)}
        records.append({"input": c, "grounding": r})
        print(json.dumps({"id": c["input_id"], "grounding": r}), flush=True)
    (out / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "cases": len(records),
                "localized": sum(x["grounding"].get("localized") is True for x in records),
            }
        )
    )


if __name__ == "__main__":
    main()
