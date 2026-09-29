"""Shared constants and helpers for the GPT-6 Astra general-agent baseline harness (AffordCraft).

Locations (see README.md): AFFORDCRAFT_PROJECT_ROOT (asset sources of the catalog; the image paths of the input manifest
are relative to it), AFFORDCRAFT_CATALOG (catalog.json), AFFORDCRAFT_INDEX (visual index directory), AFFORDCRAFT_DINOV2
(image_encoder_dinov2/ + feature_extractor_dinov2/ of the index encoder), AFFORDCRAFT_RUNS (inputs/), AGENT_HOME (harness
home; cache/ holds entry summaries and renders), AFFORDCRAFT_RUNTIME_PYTHON (interpreter of the render worker). The
endpoint and key of the model API come from EVAL_API_BASE / EVAL_API_KEY (agent_loop.py)."""

from __future__ import annotations
import base64, hashlib, io, json, math, os, sys, time, datetime
from pathlib import Path

PROJECT = Path(os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace"))
RUNS = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
MODELS = Path(os.environ.get("AFFORDCRAFT_MODELS", str(PROJECT / "models")))
REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = (
    REPO_ROOT / "affordcraft"
)  # frozen AffordCraft package: catalog.py (CatalogIndex, VisualEncoder), source_parser.py
CATALOG_JSON = Path(os.environ.get("AFFORDCRAFT_CATALOG", str(PROJECT / "catalog/catalog.json")))
INDEX_ROOT = Path(os.environ.get("AFFORDCRAFT_INDEX", str(PROJECT / "index/visual-large-001")))
MODEL_DIR = Path(os.environ.get("AFFORDCRAFT_DINOV2", str(MODELS / "dinov2")))
INPUT_MANIFEST = RUNS / "inputs/input_manifest_2000.jsonl"
SUBSET_JSON = RUNS / "inputs/subset_200_inputs.json"
CONTRACT_MD = REPO_ROOT / "evaluation/gate/manifest_contract.md"
HARNESS_HOME = Path(os.environ.get("AGENT_HOME", str(RUNS / "external/gpt6-astra-agent")))
CACHE_ROOT = HARNESS_HOME / "cache"
PYTHON = os.environ.get("AFFORDCRAFT_RUNTIME_PYTHON", sys.executable)

METHOD_ID = "gpt6-astra-agent-v0.1"
MODEL_ID = "gpt-6-astra"
PROTOCOL = {
    "max_rounds": 32,
    "max_completion_tokens_total": 128000,
    "wall_clock_seconds": 3 * 3600,
    "per_call_max_tokens": 16000,
    "per_call_timeout_seconds": 600,  # inactivity timeout of a streamed call (no bytes received for 600 s)
    "per_call_total_timeout_seconds": 1800,  # hard cap on one streamed call
    "max_retries": 6,
    "reasoning_effort": "max",
    "temperature": "provider default (not set)",
}
PREVIEW_MAX_SIDE = 448
RENDER_SIZE = (640, 480)
IMAGE_CONTEXT_KEEP_ROUNDS = 3  # tool images older than this many assistant rounds are replaced by text stubs


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def image_to_data_uri(im, max_side=PREVIEW_MAX_SIDE, fmt="JPEG", quality=85):
    """Downscale a PIL image and return (data_uri, byte_length, sha256)."""
    from PIL import Image

    im = im.convert("RGB")
    w, h = im.size
    s = max(w, h)
    if max_side and s > max_side:
        im = im.resize((max(1, round(w * max_side / s)), max(1, round(h * max_side / s))), Image.LANCZOS)
    b = io.BytesIO()
    if fmt == "JPEG":
        im.save(b, "JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        im.save(b, "PNG")
        mime = "image/png"
    data = b.getvalue()
    return f"data:{mime};base64," + base64.b64encode(data).decode(), len(data), sha256_bytes(data)


def file_to_data_uri(path):
    """Send an input photograph as its original bytes (hash verified by the caller)."""
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(data).decode(), len(data), sha256_bytes(data)


def rpy_matrix(rpy):
    import numpy as np

    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def pose_matrix(xyz=(0, 0, 0), rpy=(0, 0, 0), scale=1.0):
    import numpy as np

    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy) * float(scale)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def rigid_matrix(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return pose_matrix(xyz, rpy, 1.0)
