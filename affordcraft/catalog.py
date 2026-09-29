"""Immutable catalog inputs with a fresh, encoder-matched feature index."""

from pathlib import Path
import hashlib, json, math, time
import numpy as np


def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve(path, project):
    p = Path(path)
    return p if p.is_absolute() else Path(project) / p


def catalog_entries(path):
    return json.loads(Path(path).read_text())["entries"]


def preflight_entry(entry, project):
    errors = []
    refs = []
    preview = entry.get("preview") or {}
    sources = []
    if preview.get("path"):
        sources.append(("preview", preview))
    else:
        errors.append("preview_missing")
    if entry.get("urdf"):
        sources.append(("urdf", entry["urdf"]))
        source_dir = entry.get("source_dir")
        if not source_dir or not Path(source_dir).is_dir():
            errors.append("structured_source_missing")
        if entry.get("missing_mesh_paths"):
            errors.append("source_mesh_missing")
    else:
        for key in ("visual", "collision"):
            ref = (entry.get("source") or {}).get(key) or {}
            if ref.get("path"):
                sources.append((key, ref))
            else:
                errors.append(key + "_missing")
    for kind, ref in sources:
        p = resolve(ref["path"], project)
        if not p.is_file():
            errors.append(kind + "_missing")
            continue
        digest = file_sha(p)
        if ref.get("sha256") and digest != ref["sha256"]:
            errors.append(kind + "_hash_drift")
        refs.append({"kind": kind, "path": str(p), "sha256": digest})
    return {
        "candidate_id": str(entry["candidate_id"]),
        "eligible": not errors,
        "errors": errors,
        "source_refs": refs,
        "category": entry.get("category"),
        "motion_types": entry.get("motion_types", []),
    }


class VisualEncoder:
    def __init__(self, model_path, processor_path, device="cuda:0"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.model_path = str(model_path)
        self.processor_path = str(processor_path)
        self.processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        )
        self.dimension = int(self.model.config.hidden_size)
        self.identity = {
            "model_config_sha256": file_sha(Path(model_path) / "config.json"),
            "processor_sha256": file_sha(Path(processor_path) / "preprocessor_config.json"),
            "dimension": self.dimension,
            "feature": "normalized_CLS",
            "model_path": str(model_path),
        }

    def encode(self, image):
        t = self.processor(images=image, return_tensors="pt")
        t = {k: v.to(self.device, dtype=self.torch.float16) for k, v in t.items()}
        with self.torch.inference_mode():
            v = self.model(**t).last_hidden_state[:, 0].float()[0]
        a = v.cpu().numpy()
        n = np.linalg.norm(a)
        if not np.isfinite(a).all() or n <= 0:
            raise RuntimeError("invalid_visual_feature")
        return a / n


class CLIPImageEncoder:
    """Alternative visual encoder for the encoder-replacement study: OpenAI CLIP ViT-B/32 image
    tower from the TorchScript checkpoint, normalized image embedding, same interface as VisualEncoder."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, checkpoint, device="cuda:0"):
        import torch

        self.torch = torch
        self.device = device
        self.checkpoint = str(checkpoint)
        self.model = torch.jit.load(self.checkpoint, map_location=device).eval()
        self.resolution = int(self.model.input_resolution.item()) if hasattr(self.model, "input_resolution") else 224
        self.dimension = 512
        self.identity = {
            "model_config_sha256": file_sha(self.checkpoint),
            "processor_sha256": "clip_builtin_preprocess_224_bicubic_center_crop",
            "dimension": self.dimension,
            "feature": "normalized_CLIP_image_embedding",
            "model_path": self.checkpoint,
        }

    def encode(self, image):
        from PIL import Image

        im = image.convert("RGB")
        w, h = im.size
        s = self.resolution / min(w, h)
        im = im.resize((max(self.resolution, round(w * s)), max(self.resolution, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l = (w - self.resolution) // 2
        t = (h - self.resolution) // 2
        im = im.crop((l, t, l + self.resolution, t + self.resolution))
        x = self.torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - self.torch.tensor(self.MEAN)[:, None, None]) / self.torch.tensor(self.STD)[:, None, None]
        with self.torch.inference_mode():
            v = self.model.encode_image(x[None].to(self.device).half()).float()[0]
        a = v.cpu().numpy()
        n = np.linalg.norm(a)
        if not np.isfinite(a).all() or n <= 0:
            raise RuntimeError("invalid_visual_feature")
        return a / n


def build_index(project, catalog_path, out, encoder):
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    entries = catalog_entries(catalog_path)
    audit = []
    ids = []
    features = []
    with (out / "per_candidate.jsonl").open("x") as log:
        for i, e in enumerate(entries):
            r = preflight_entry(e, project)
            tick = time.perf_counter()
            if r["eligible"]:
                try:
                    preview = next(x for x in r["source_refs"] if x["kind"] == "preview")
                    im = Image.open(preview["path"]).convert("RGB")
                    v = encoder.encode(im)
                    features.append(v)
                    ids.append(str(e["candidate_id"]))
                    r["feature_row"] = len(ids) - 1
                    r["feature_dimension"] = len(v)
                except Exception as exc:
                    r.update(eligible=False, errors=["feature_encoding_failed:" + repr(exc)])
            r["seconds"] = time.perf_counter() - tick
            audit.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
            if i % 200 == 0:
                print(json.dumps({"catalog_progress": i + 1, "eligible": len(ids), "total": len(entries)}), flush=True)
    np.save(out / "features.npy", np.asarray(features, dtype=np.float32))
    (out / "candidate_ids.json").write_text(json.dumps(ids))
    report = {
        "catalog_sha256": file_sha(catalog_path),
        "raw_entries": len(entries),
        "eligible_entries": len(ids),
        "encoder": encoder.identity,
        "offline_wall_seconds": time.perf_counter() - start,
        "feature_sha256": file_sha(out / "features.npy"),
        "scope": "static source eligibility and visual encoding; not physical or task success",
    }
    (out / "index.json").write_text(json.dumps(report, indent=2))
    return report


class CatalogIndex:
    def __init__(self, project, catalog_path, index_root, encoder):
        self.project = Path(project)
        self.entries = {str(x["candidate_id"]): x for x in catalog_entries(catalog_path)}
        r = Path(index_root)
        self.meta = json.loads((r / "index.json").read_text())
        self.ids = json.loads((r / "candidate_ids.json").read_text())
        self.features = np.load(r / "features.npy", mmap_mode="r")
        self.encoder = encoder
        if self.meta["catalog_sha256"] != file_sha(catalog_path) or self.meta["encoder"] != encoder.identity:
            raise RuntimeError("index_encoder_or_catalog_mismatch")
        if len(self.ids) != len(self.features) or self.features.shape[1] != encoder.dimension:
            raise RuntimeError("feature_shape_mismatch")

    def retrieve(self, image, category, limit=20, allowed_ids=None):
        q = self.encoder.encode(image)
        scores = self.features @ q
        ranked = [i for i, k in enumerate(self.ids) if allowed_ids is None or k in allowed_ids]
        ranked.sort(key=lambda i: (-float(scores[i]), self.ids[i]))
        # category=None: no task conditioning, purely visual order over the whole library.
        preferred = (
            []
            if category is None
            else [
                i
                for i in ranked
                if str(self.entries[self.ids[i]].get("category", "")).casefold() == str(category).casefold()
            ]
        )
        # Noisy catalog labels are a prior, not semantic ground truth: reserve 40%
        # of the fixed budget for category-blind visual alternatives.
        selected = preferred[:12]
        chosen = set(selected)
        selected += [i for i in ranked if i not in chosen][: 20 - len(selected)]
        return [dict(self.entries[self.ids[i]], retrieval_score=float(scores[i])) for i in selected[:limit]]
