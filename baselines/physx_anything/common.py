"""Shared constants/helpers for the PhysX-Anything runs (official code + official weights).
Nothing in here touches the official source tree. Every location comes from an environment variable (see README.md):
  AFFORDCRAFT_PROJECT_ROOT      root the image paths of the input manifest are relative to
  AFFORDCRAFT_RUNS              run outputs; inputs/ holds the frozen 2,000-input manifest and the registered 200-input subset
  AFFORDCRAFT_MODELS            default parent directory of the auxiliary models below
  PHYSX_ANYTHING_HOME           official PhysX-Anything checkout
  PHYSX_ANYTHING_CKPT           official weights (vlm/, decoder/, trellis/)
  PHYSX_ANYTHING_PROCESSOR      Qwen/Qwen2.5-VL-7B-Instruct processor files
  TORCH_HOME, U2NET_HOME        torch.hub cache (DINOv2 repo + checkpoint) and rembg cache (u2net.onnx), used offline
  PHYSX_ANYTHING_RUN            run root (holds configuration.json, code/, lane-<k>/)
  PHYSX_ANYTHING_INPUT_SET      input-set label written into the lane receipts
  PHYSX_ANYTHING_VLM_PYTHON     interpreter of stage A (VLM); PHYSX_ANYTHING_GEOM_PYTHON: interpreter of stage B (sm_120 env)
"""

import hashlib, json, os, sys, time, datetime, subprocess
from pathlib import Path

P = Path(os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace"))
RUNS = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
MODELS = Path(os.environ.get("AFFORDCRAFT_MODELS", str(P / "models")))
SRC = Path(os.environ.get("PHYSX_ANYTHING_HOME", "PhysX-Anything"))
WEIGHTS = Path(os.environ.get("PHYSX_ANYTHING_CKPT", str(MODELS / "physx-anything")))
VLM_CKPT = WEIGHTS / "vlm"
DECODER = WEIGHTS / "decoder"
PROCESSOR = Path(os.environ.get("PHYSX_ANYTHING_PROCESSOR", str(MODELS / "qwen2.5-vl-7b-processor")))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(MODELS / "torch")))
DINO_CKPT = TORCH_HOME / "hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
DINO_REPO = TORCH_HOME / "hub/facebookresearch_dinov2_main"
U2NET_HOME = Path(os.environ.get("U2NET_HOME", str(MODELS / "rembg")))
U2NET = U2NET_HOME / "u2net.onnx"
FULL_MANIFEST = RUNS / "inputs/input_manifest_2000.jsonl"
FULL_MANIFEST_SHA = os.environ.get(
    "AFFORDCRAFT_MANIFEST_SHA256", "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"
)
SUBSET = RUNS / "inputs/subset_200_inputs.json"
SUBSET_SHA = os.environ.get(
    "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
)
REV = Path(os.environ.get("PHYSX_ANYTHING_RUN", str(RUNS / "external/physx-anything/subset200")))
INPUT_SET = os.environ.get("PHYSX_ANYTHING_INPUT_SET", "registered_subset_200_seed20260916")
VLA_PY = os.environ.get("PHYSX_ANYTHING_VLM_PYTHON", sys.executable)
GEOM_PY = os.environ.get("PHYSX_ANYTHING_GEOM_PYTHON", sys.executable)
METHOD_ID = "physx-anything-v0.1"


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def lock(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def stamp():
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_once(path, obj):
    """Append-only discipline: refuse to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def rows_of(manifest):
    with open(manifest, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def capture(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except Exception as e:  # noqa
        return "capture_failed: %r" % (e,)


def log(fh, msg):
    line = "%s %s" % (utc(), msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def move_to_interrupted(case_dir, interrupted_root, tag):
    """Never overwrite: an interrupted case directory is moved aside (kept) before recomputation."""
    interrupted_root = Path(interrupted_root)
    interrupted_root.mkdir(parents=True, exist_ok=True)
    dest = interrupted_root / ("%s-%s-%s" % (Path(case_dir).name, tag, stamp()))
    os.replace(str(case_dir), str(dest))
    return dest
