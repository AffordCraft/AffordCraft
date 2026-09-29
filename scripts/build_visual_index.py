from pathlib import Path
import argparse, os, sys, json

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", required=True)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--encoder", choices=["dinov2", "clip"], default="dinov2")
    args = a.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from affordcraft.catalog import VisualEncoder, CLIPImageEncoder, build_index
    from affordcraft import paths

    base = paths.DINOV2_ROOT
    project = paths.PROJECT_ROOT
    if args.encoder == "clip":
        if "clip" not in Path(args.output).name:
            raise RuntimeError('CLIP index directory name must contain "clip"')
        encoder = CLIPImageEncoder(paths.CLIP_VIT_B32)
    else:
        encoder = VisualEncoder(base / "image_encoder_dinov2", base / "feature_extractor_dinov2")
    report = build_index(
        project,
        paths.catalog_path(project),
        args.output,
        encoder,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
