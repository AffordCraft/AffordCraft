"""Shared constants/helpers for the Articulate-Anything run (official code, VLM through an OpenAI-compatible API) on the
registered 200-input subset. Nothing here touches the official source tree. Locations (see README.md):
  AFFORDCRAFT_PROJECT_ROOT   root the image paths of the input manifest are relative to
  AFFORDCRAFT_RUNS           run outputs; inputs/ holds the frozen 2,000-input manifest and the registered subset
  ARTICULATE_ANYTHING_HOME   official checkout (never modified)
  ARTICULATE_ANYTHING_CKPT   PartNet-Mobility library as preprocessed for Articulate-Anything (partnet-mobility-v0/, holds dataset/)
  AA_WORK                    working directory: code/ (this folder), src/ (patched copy of the official tree), lanes/, patches/, logs/
  AA_REV                     run root (default $AFFORDCRAFT_RUNS/external/articulate-anything/subset200); its name is the run id
  AA_PYTHON                  interpreter of the run environment; AA_GPU: GPU index (CUDA_VISIBLE_DEVICES and the renderer)
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

P = Path(os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace"))
RUNS = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
SRC_OFFICIAL = Path(os.environ.get("ARTICULATE_ANYTHING_HOME", "articulate-anything"))
AA_RUN = Path(os.environ.get("AA_WORK", "aa_run")).absolute()
CODE = AA_RUN / "code"
PATCHED_SRC = AA_RUN / "src"
LANES_ROOT = AA_RUN / "lanes"
PATCH_DIFF = AA_RUN / "patches/official_tree.diff"
DATASET_ROOT = Path(os.environ.get("ARTICULATE_ANYTHING_CKPT", "partnet-mobility-v0")).absolute() / "dataset"
FULL_MANIFEST = RUNS / "inputs/input_manifest_2000.jsonl"
FULL_MANIFEST_SHA = os.environ.get(
    "AFFORDCRAFT_MANIFEST_SHA256", "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"
)
SUBSET = RUNS / "inputs/subset_200_inputs.json"
SUBSET_SHA = os.environ.get(
    "AFFORDCRAFT_SUBSET_SHA256", "db043e72f336ca751d291c09f9e12704d4dd6b6e0e229281250c1b126a1903bf"
)
REV = Path(os.environ.get("AA_REV") or str(RUNS / "external/articulate-anything/subset200"))
RUN_ID = REV.name
PY = os.environ.get("AA_PYTHON", sys.executable)
METHOD_ID = "articulate-anything-v0.1"
VLM_MODEL = "gemini-2.5-flash"
OFFICIAL_VLM_MODEL = "gemini-1.5-flash-latest"
GPU = int(os.environ.get("AA_GPU", "0"))
CASE_TIMEOUT_SECONDS = 3600
MAX_OFFICIAL_ITERATIONS = 3
PARTNET_EXCLUDE_DIRS = (
    "images",
    "parts_render",
    "parts_render_after_merging",
    "point_sample",
)  # PartNet extras unused by the method


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


def capture(cmd, timeout=600):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
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
