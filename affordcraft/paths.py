"""Filesystem locations and interpreters used by the pipeline.

Every value is read from an environment variable, so the code carries no machine-specific paths. The defaults
describe the expected layout (see docs/setup.md):

    $AFFORDCRAFT_PROJECT_ROOT/            asset sources referenced by the catalog (relative catalog paths resolve here)
        catalog/catalog.json              library catalog (schema: docs/data.md)
    $AFFORDCRAFT_MODELS/
        qwen3-vl-8b-instruct/             Qwen3-VL-8B-Instruct (grounding and multimodal selection)
        dinov2/image_encoder_dinov2/      DINOv2 image tower used for the visual index
        dinov2/feature_extractor_dinov2/  its image processor
        clip/ViT-B-32.pt                  OpenAI CLIP ViT-B/32 (encoder-replacement study only)
    $AFFORDCRAFT_RUNS/                    run outputs

The pipeline runs three interpreters: Isaac Sim 5.1 for the physics services, a runtime environment with trimesh,
CoACD and PyTorch for asset construction, and an environment with a Transformers release that serves Qwen3-VL.
"""

import os
import sys
from pathlib import Path


def _path(variable, default):
    return Path(os.path.expanduser(os.environ.get(variable, str(default))))


PROJECT_ROOT = _path("AFFORDCRAFT_PROJECT_ROOT", "workspace")
MODELS = _path("AFFORDCRAFT_MODELS", PROJECT_ROOT / "models")
QWEN3_VL = _path("AFFORDCRAFT_QWEN3_VL", MODELS / "qwen3-vl-8b-instruct")
DINOV2_ROOT = _path("AFFORDCRAFT_DINOV2", MODELS / "dinov2")
CLIP_VIT_B32 = _path("AFFORDCRAFT_CLIP", MODELS / "clip" / "ViT-B-32.pt")
RUNS = _path("AFFORDCRAFT_RUNS", "runs")

ISAAC_PYTHON = os.environ.get("AFFORDCRAFT_ISAAC_PYTHON", "python")
RUNTIME_PYTHON = os.environ.get("AFFORDCRAFT_RUNTIME_PYTHON", sys.executable)
SELECTOR_PYTHON = os.environ.get("AFFORDCRAFT_SELECTOR_PYTHON", sys.executable)


def catalog_path(project=None):
    """Library catalog; relative asset paths inside it resolve against the project root."""
    default = Path(project or PROJECT_ROOT) / "catalog" / "catalog.json"
    return _path("AFFORDCRAFT_CATALOG", default)


def repository_root():
    """Root of this repository (holds affordcraft/ and scripts/)."""
    return Path(__file__).resolve().parents[1]
