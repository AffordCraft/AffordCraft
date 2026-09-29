#!/usr/bin/env python3
"""Automatic part-mask input-contract adapter for the PAct baseline.

PAct's released loader (modules/pact/datasets/components.py::ImageConditioned_dataset)
needs an RGBA object image (<name>_processed.png) plus a per-part label map
(<name>_mask.exr, float32 RGB, identical channels, 0 = background) and reads the
part count from the label map.  Our full-RGB protocol gives only the photograph
and the category name.  This adapter produces those two files with LOCAL models
only and never reads any annotation (no bbox_normalized, no reference masks):

  1. object localization: Grounding-DINO tiny, prompt = category name
     (CamelCase split into lowercase words + '.'), highest-scoring box, box
     threshold 0.25 -> failure_reason automatic_localization_failed otherwise;
  2. object alpha: SAM ViT-B prompted with that box (multimask, highest predicted IoU);
  3. part labels: SAM automatic mask generation (transformers mask-generation
     pipeline, SamAutomaticMaskGenerator defaults) on the official 518x518
     white-background canvas, merged by the OFFICIAL PAct 2D labeling code
     (modules/label_2d_mask/label_parts.py: get_sam_mask -> get_sam_mask(existing)
     -> clean_segment_edges, exactly the app.py process_image + apply_merge path
     with an empty merge list), capped at PAct's max part count (32).

Everything is deterministic (fixed seeds, no sampling).  This is an input-contract
adapter, not a semantic prediction of ours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

ADAPTER_ID = "automatic_part_mask_adapter_v1"
CANVAS = 518
PART_CAP = 32  # PartBasedSparseStructureFlowModel max_num_parts=32 (part_pe has 33 rows)
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.25
CROP_MARGIN = 0.10
MIN_ALPHA_PIXELS = 500
SAM_AUTO = dict(
    points_per_batch=64,
    points_per_crop=32,
    crops_n_layers=0,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crops_nms_thresh=0.7,
)
OFFICIAL_SIZE_TH = 2000  # modules/label_2d_mask/label_parts.py size_th


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def category_prompt(cat: str) -> str:
    words = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", cat)
    return " ".join(w.lower() for w in words) + "."


def write_exr_labels(path: Path, labels: np.ndarray) -> None:
    import OpenEXR
    import Imath

    h, w = labels.shape
    hdr = OpenEXR.Header(w, h)
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    hdr["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in ("R", "G", "B")}
    out = OpenEXR.OutputFile(str(path), hdr)
    b = np.ascontiguousarray(labels.astype(np.float32)).tobytes()
    out.writePixels({"R": b, "G": b, "B": b})
    out.close()


def read_exr_rgb(path: Path) -> np.ndarray:
    import OpenEXR
    import Imath

    f = OpenEXR.InputFile(str(path))
    hdr = f.header()
    dw = hdr["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ch = {
        c: np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w) for c in "RGB"
    }
    return np.stack([ch["R"], ch["G"], ch["B"]], -1)


def install_official_label_parts(pact_src: Path):
    """Import the official PAct 2D labeling module with its two unavailable deps stubbed.

    segment_anything (SAM ViT-H generator) is replaced by our transformers-based
    generator object; detectron2-based Visualizer is only used for a debug image."""
    sa = types.ModuleType("segment_anything")
    sa.SamAutomaticMaskGenerator = object
    sa.build_sam = lambda *a, **k: None
    sys.modules.setdefault("segment_anything", sa)
    vis = types.ModuleType("modules.label_2d_mask.visualizer")

    class Visualizer:  # noqa: D401 - stub
        def __init__(self, img):
            self.img = img

    vis.Visualizer = Visualizer
    sys.modules.setdefault("modules.label_2d_mask.visualizer", vis)
    if str(pact_src) not in sys.path:
        sys.path.insert(0, str(pact_src))
    import importlib

    lp = importlib.import_module("modules.label_2d_mask.label_parts")
    return lp


class _GenStub:
    def __init__(self, masks):
        self.masks = masks

    def generate(self, image):
        return self.masks


class _VisStub:
    def __init__(self, img):
        self.img = img

    def draw_binary_mask(self, *a, **k):
        return self

    def draw_binary_mask_with_number(self, *a, **k):
        return self

    def get_image(self):
        return self.img


def cap_parts(labels: np.ndarray, cap: int):
    """Keep the `cap` largest parts; merge each dropped part into its largest touching
    kept neighbour (fallback: nearest kept centroid).  Returns (labels, n_merged)."""
    import cv2

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if len(ids) <= cap:
        return labels, 0
    order = np.argsort(-counts, kind="stable")
    keep = [int(i) for i in ids[order[:cap]]]
    drop = [int(i) for i in ids[order[cap:]]][::-1]  # smallest first
    kernel = np.ones((3, 3), np.uint8)
    cents = {k: np.argwhere(labels == k).mean(0) for k in keep}
    for d in drop:
        m = labels == d
        dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=2) > 0
        neigh = labels[dil & ~m]
        neigh = neigh[np.isin(neigh, keep)]
        if len(neigh):
            vals, cnt = np.unique(neigh, return_counts=True)
            target = int(vals[np.argmax(cnt)])
        else:
            c = np.argwhere(m).mean(0)
            target = min(keep, key=lambda k: float(np.sum((cents[k] - c) ** 2)))
        labels[m] = target
    return labels, len(drop)


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[labels > 0])
    out = np.zeros_like(labels)
    for new, old in enumerate(ids, 1):
        out[labels == old] = new
    return out


def colorize(labels: np.ndarray) -> np.ndarray:
    vis = np.full((*labels.shape, 3), 255, np.uint8)
    for i, uid in enumerate(np.unique(labels[labels > 0])):
        vis[labels == uid] = [(i * 50 + 80) % 256, (i * 120 + 40) % 256, (i * 180 + 20) % 256]
    return vis


class Adapter:
    def __init__(self, gdino_dir: Path, sam_dir: Path, pact_src: Path, device: str = "cuda"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor, pipeline

        torch.manual_seed(0)
        np.random.seed(0)
        self.torch = torch
        self.device = device
        t0 = time.perf_counter()
        self.gp = AutoProcessor.from_pretrained(str(gdino_dir))
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(str(gdino_dir)).to(device).eval()
        self.sp = SamProcessor.from_pretrained(str(sam_dir))
        self.sm = SamModel.from_pretrained(str(sam_dir)).to(device).eval()
        self.gen = pipeline(
            "mask-generation",
            model=self.sm,
            image_processor=self.sp.image_processor,
            device=0 if device == "cuda" else -1,
        )
        self.lp = install_official_label_parts(pact_src)
        assert self.lp.size_th == OFFICIAL_SIZE_TH
        self.load_seconds = time.perf_counter() - t0
        self.ds_cls = None

    # ---- stage 1: localization ------------------------------------------------
    def localize(self, img, prompt: str):
        torch = self.torch
        inp = self.gp(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gm(**inp)
        res = self.gp.post_process_grounded_object_detection(
            out,
            inp.input_ids,
            threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[(img.height, img.width)],
        )[0]
        boxes = res["boxes"].detach().cpu().numpy().astype(float)
        scores = res["scores"].detach().cpu().numpy().astype(float)
        labels = res.get("text_labels", None)
        labels = [str(x) for x in labels] if labels is not None else []
        return boxes, scores, labels

    # ---- stage 2: object alpha ------------------------------------------------
    def alpha_from_box(self, img, box):
        torch = self.torch
        si = self.sp(img, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            so = self.sm(**si, multimask_output=True)
        masks = self.sp.image_processor.post_process_masks(
            so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu()
        )[0][0]
        ious = so.iou_scores[0, 0].detach().cpu().numpy().astype(float)
        j = int(np.argmax(ious))
        return masks[j].numpy().astype(bool), ious, j

    # ---- stage 3: automatic masks --------------------------------------------
    def auto_masks(self, image_rgb: np.ndarray):
        from PIL import Image

        out = self.gen(Image.fromarray(image_rgb), **SAM_AUTO)
        masks = out["masks"]
        scores = out["scores"]
        scores = scores.detach().cpu().numpy().tolist() if hasattr(scores, "detach") else list(scores)
        res = []
        for m, s in zip(masks, scores):
            m = np.asarray(m).astype(bool)
            res.append({"segmentation": m, "area": int(m.sum()), "predicted_iou": float(s)})
        return res

    def loader_check(self, case_dir: Path):
        """Load the produced pair through the official dataset code path (CPU)."""
        import imageio.v3 as iio

        if self.ds_cls is None:
            _orig = iio.imread

            def _imread(uri, *a, **k):
                if not a and not k and str(uri).lower().endswith(".exr"):
                    return read_exr_rgb(Path(uri))
                return _orig(uri, *a, **k)

            iio.imread = _imread
            from modules.pact.datasets.components import ImageConditioned_dataset

            self.ds_cls = ImageConditioned_dataset
        ds = self.ds_cls(str(case_dir), is_depth_one_dir=True)
        assert len(ds) == 1, len(ds)
        it = ds[0]
        return {
            "num_parts_loader": int(it["num_parts"]),
            "unordered_max": int(it["unordered_masks"].max()),
            "cond_shape": list(it["cond"].shape),
            "masks_shape": list(it["masks"].shape),
            "part_idx": it["part_idx"].tolist(),
        }

    # ---- per case ------------------------------------------------------------
    def run_case(self, sid: str, category: str, image_path: Path, expected_sha: str, out_dir: Path):
        import cv2
        from PIL import Image, ImageDraw

        torch = self.torch
        out_dir.mkdir(parents=True, exist_ok=False)
        rec = {
            "adapter_id": ADAPTER_ID,
            "source_id": sid,
            "requested_category": category,
            "prompt_text": category_prompt(category),
            "image_path": str(image_path),
            "image_sha256": None,
            "image_sha256_verified": False,
            "status": "failed",
            "failure_reason": None,
            "stage_reached": "input",
            "timing_seconds": {},
            "thresholds": {
                "gdino_box_threshold": GDINO_BOX_THRESHOLD,
                "gdino_text_threshold": GDINO_TEXT_THRESHOLD,
                "crop_margin": CROP_MARGIN,
                "min_alpha_pixels": MIN_ALPHA_PIXELS,
                "canvas": CANVAS,
                "sam_auto": SAM_AUTO,
                "official_size_th": OFFICIAL_SIZE_TH,
                "part_cap": PART_CAP,
            },
            "annotation_access": "none (no bbox_normalized, no reference mask read)",
        }
        try:
            sha = sha256_file(image_path)
            rec["image_sha256"] = sha
            if sha != expected_sha:
                rec["failure_reason"] = "input_image_hash_mismatch"
                return rec
            rec["image_sha256_verified"] = True
            img = Image.open(image_path).convert("RGB")
            W, H = img.size
            rec["image_size"] = [W, H]

            # 1. localization
            t = time.perf_counter()
            boxes, scores, labels = self.localize(img, rec["prompt_text"])
            rec["timing_seconds"]["localization"] = time.perf_counter() - t
            rec["localization"] = {
                "model": "grounding-dino-tiny (local HF)",
                "n_candidates": int(len(scores)),
                "candidates": [
                    {"box_xyxy": b.tolist(), "score": float(s), "label": (labels[i] if i < len(labels) else None)}
                    for i, (b, s) in enumerate(zip(boxes, scores))
                ],
            }
            rec["stage_reached"] = "localization"
            if len(scores) == 0 or float(np.max(scores)) < GDINO_BOX_THRESHOLD:
                rec["failure_reason"] = "automatic_localization_failed"
                return rec
            i = int(np.argmax(scores))
            box = boxes[i].tolist()
            rec["localization"]["chosen"] = {"index": i, "box_xyxy": box, "score": float(scores[i])}

            # 2. alpha
            t = time.perf_counter()
            m, ious, j = self.alpha_from_box(img, box)
            bw, bh = box[2] - box[0], box[3] - box[1]
            ex = [
                max(0, box[0] - CROP_MARGIN * bw),
                max(0, box[1] - CROP_MARGIN * bh),
                min(W, box[2] + CROP_MARGIN * bw),
                min(H, box[3] + CROP_MARGIN * bh),
            ]
            keep = np.zeros_like(m)
            keep[int(np.floor(ex[1])) : int(np.ceil(ex[3])), int(np.floor(ex[0])) : int(np.ceil(ex[2]))] = True
            m &= keep
            rec["timing_seconds"]["alpha"] = time.perf_counter() - t
            rec["alpha"] = {
                "model": "sam-vit-base (local HF), box prompt",
                "multimask_iou": ious.tolist(),
                "chosen_index": j,
                "alpha_pixels": int(m.sum()),
                "expanded_box_xyxy": ex,
            }
            rec["stage_reached"] = "alpha"
            if int(m.sum()) < MIN_ALPHA_PIXELS:
                rec["failure_reason"] = "automatic_segmentation_failed"
                return rec
            ys, xs = np.where(m)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            mg = int(round(CROP_MARGIN * max(x1 - x0, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            rec["alpha"]["alpha_bbox_xyxy"] = [x0, y0, x1, y1]
            rec["alpha"]["crop_xyxy"] = [cx0, cy0, cx1, cy1]
            rgba = np.dstack([np.asarray(img), (m * 255).astype(np.uint8)])[cy0:cy1, cx0:cx1]
            crop = Image.fromarray(rgba, "RGBA")
            processed = self.lp.resize_and_pad_to_square(crop, CANVAS)  # official function
            assert processed.mode == "RGBA" and processed.size == (CANVAS, CANVAS)
            white = Image.new("RGBA", processed.size, (255, 255, 255, 255))
            image = np.array(Image.alpha_composite(white, processed).convert("RGB"))
            processed_path = out_dir / f"{sid}_processed.png"
            processed.save(processed_path)
            Image.fromarray(image).save(out_dir / f"{sid}_canvas_white_bg.png")

            # 3. automatic part masks + official merging
            t = time.perf_counter()
            raw = self.auto_masks(image)
            rec["timing_seconds"]["sam_auto_masks"] = time.perf_counter() - t
            t = time.perf_counter()
            gen = _GenStub(raw)
            vis = _VisStub(image)
            group_ids, _ = self.lp.get_sam_mask(
                image,
                gen,
                vis,
                merge_groups=None,
                rgba_image=processed,
                img_name=sid,
                save_dir=str(out_dir),
                size_threshold=OFFICIAL_SIZE_TH,
            )
            new_group_ids, _ = self.lp.get_sam_mask(
                image,
                gen,
                vis,
                merge_groups=None,
                existing_group_ids=group_ids,
                rgba_image=processed,
                skip_split=True,
                img_name=sid,
                save_dir=str(out_dir),
                size_threshold=OFFICIAL_SIZE_TH,
            )
            new_group_ids = self.lp.clean_segment_edges(new_group_ids)
            self.lp.get_mask(new_group_ids, image, ids=3, img_name=sid, save_dir=str(out_dir))
            labels = (np.asarray(new_group_ids) + 1).astype(np.int64)  # official: background -1 -> 0
            n_before = int(len(np.unique(labels[labels > 0])))
            labels, n_merged = cap_parts(labels, PART_CAP)
            labels = relabel_contiguous(labels)
            n_parts = int(labels.max())
            rec["timing_seconds"]["official_merge"] = time.perf_counter() - t
            areas = {int(k): int(v) for k, v in zip(*np.unique(labels[labels > 0], return_counts=True))}
            rec["part_labeling"] = {
                "n_raw_sam_masks": len(raw),
                "n_parts_after_official_merge": n_before,
                "n_parts_merged_by_cap": n_merged,
                "n_parts": n_parts,
                "part_areas_px": areas,
                "labels_outside_alpha_px": int(((labels > 0) & (np.asarray(processed)[..., 3] == 0)).sum()),
                "official_functions": [
                    "resize_and_pad_to_square",
                    "get_sam_mask",
                    "get_sam_mask(existing_group_ids)",
                    "clean_segment_edges",
                    "get_mask",
                ],
            }
            rec["stage_reached"] = "part_mask"
            if n_parts < 1:
                rec["failure_reason"] = "automatic_part_labeling_failed"
                return rec
            exr_path = out_dir / f"{sid}_mask.exr"
            write_exr_labels(exr_path, labels)
            back = read_exr_rgb(exr_path)
            assert back.shape == (CANVAS, CANVAS, 3) and np.array_equal(back[..., 0], labels.astype(np.float32))
            assert np.array_equal(back[..., 0], back[..., 1]) and np.array_equal(back[..., 0], back[..., 2])
            # preview: original+box | processed on white | labels
            prev = Image.new("RGB", (CANVAS * 3, CANVAS), (255, 255, 255))
            o = img.copy()
            d = ImageDraw.Draw(o)
            d.rectangle(box, outline=(255, 0, 0), width=max(2, int(0.004 * max(W, H))))
            o.thumbnail((CANVAS, CANVAS))
            prev.paste(o, (0, 0))
            prev.paste(Image.fromarray(image), (CANVAS, 0))
            lab_vis = Image.fromarray(colorize(labels))
            dd = ImageDraw.Draw(lab_vis)
            for k in range(1, n_parts + 1):
                yy, xx = np.where(labels == k)
                dd.text((int(xx.mean()), int(yy.mean())), str(k), fill=(0, 0, 0))
            prev.paste(lab_vis, (CANVAS * 2, 0))
            prev.save(out_dir / f"{sid}_mask_preview.png")
            np.save(out_dir / f"{sid}_alpha_fullres_bool.npy", m)
            # 4. official loader check
            t = time.perf_counter()
            chk = self.loader_check(out_dir)
            rec["timing_seconds"]["official_loader_check"] = time.perf_counter() - t
            chk["matches_adapter_n_parts"] = chk["unordered_max"] == n_parts
            rec["loader_check"] = chk
            rec["files"] = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
            rec["status"] = "ok"
            rec["stage_reached"] = "part_mask_written"
            return rec
        except Exception as e:  # noqa: BLE001
            rec["failure_reason"] = f"adapter_exception:{type(e).__name__}:{str(e)[:300]}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            return rec
        finally:
            rec["torch_peak_alloc_mib"] = (
                float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--pact-src", type=Path, required=True)
    ap.add_argument("--gdino", type=Path, required=True)
    ap.add_argument("--sam", type=Path, required=True)
    ap.add_argument("--masks-root", type=Path, required=True)
    ap.add_argument("--ids", type=str, default="", help="comma separated subset of input ids (default all)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subset = json.loads(args.subset.read_text())
    ids = [x["input_id"] for x in subset["inputs"]]
    if args.ids:
        want = set(args.ids.split(","))
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[: args.limit]
    man = {}
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            man[r["source_id"]] = r
    args.masks_root.mkdir(parents=True, exist_ok=True)
    log = open(args.masks_root / "adapter_log.jsonl", "a")
    ad = Adapter(args.gdino, args.sam, args.pact_src)
    print(f"[adapter] models loaded in {ad.load_seconds:.1f}s; {len(ids)} cases", flush=True)
    n_ok = 0
    for k, sid in enumerate(ids):
        out_dir = args.masks_root / sid
        if (out_dir / "adapter.json").exists():
            print(f"[adapter] {k} {sid} exists, skip", flush=True)
            continue
        if out_dir.exists():  # a previous adapter process died mid-case: keep the partial dir, never overwrite
            partial = args.masks_root / f"{sid}.partial-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            out_dir.rename(partial)
            print(f"[adapter] {k} {sid} partial dir moved to {partial.name}", flush=True)
        r = man[sid]
        t0 = time.perf_counter()
        rec = ad.run_case(
            sid, r["requested_category"], args.project_root / r["image"]["path"], r["image"]["sha256"], out_dir
        )
        rec["timing_seconds"]["total"] = time.perf_counter() - t0
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(out_dir / "adapter.json", "x") as f:
            json.dump(rec, f, indent=2)
        log.write(
            json.dumps(
                {
                    "source_id": sid,
                    "status": rec["status"],
                    "failure_reason": rec["failure_reason"],
                    "n_parts": rec.get("part_labeling", {}).get("n_parts"),
                    "seconds": rec["timing_seconds"]["total"],
                }
            )
            + "\n"
        )
        log.flush()
        n_ok += rec["status"] == "ok"
        print(
            f"[adapter] {k} {sid} {r['requested_category']} -> {rec['status']} {rec['failure_reason']} "
            f"parts={rec.get('part_labeling', {}).get('n_parts')} {rec['timing_seconds']['total']:.1f}s",
            flush=True,
        )
    print(f"[adapter] done ok={n_ok}/{len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
