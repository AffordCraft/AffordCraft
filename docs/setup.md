# Setup

## Environments

The pipeline runs in three processes with different dependencies. The versions below are the ones used for the paper.

| Environment | Runs | Requirements |
|---|---|---|
| selector (Python 3.10) | `scripts/run_pipeline.py` (grounding, retrieval, selection, orchestration), `build_visual_index.py`, `detect_scene_objects.py`, `calibrate_vision.py` | `requirements/selector.txt` |
| runtime (Python 3.10) | `scripts/materialize_asset.py` (source-preserving import, adaptation, collision decomposition, USD export), `coacd_health.py`; the external-method gate (`evaluation/gate`) | `requirements/runtime.txt` |
| Isaac Sim 5.1 (Python 3.11) | `scripts/physics_worker.py` (construction check and final validation as two independent services), `make_physics_fixtures.py`, `run_physics_job.py` | `requirements/isaac.txt` |

`run_pipeline.py` starts the two physics services and one construction subprocess per candidate itself; it only needs
to know the interpreters:

```bash
export AFFORDCRAFT_ISAAC_PYTHON=/path/to/isaac-sim-5.1/python       # physics services
export AFFORDCRAFT_RUNTIME_PYTHON=/path/to/runtime-env/bin/python    # asset construction
export AFFORDCRAFT_BUILD_THREADS=8                                   # CPU threads of one construction (paper: 8)
```

Isaac Sim runs headless. The physics services receive the GPU index as an argument and must not inherit
`CUDA_VISIBLE_DEVICES` (the backend removes it for them). The construction subprocess runs on CPU only.

## Locations (`affordcraft/paths.py`)

| Variable | Default | Content |
|---|---|---|
| `AFFORDCRAFT_PROJECT_ROOT` | `workspace` | root of the asset sources; relative paths in the catalog resolve against it |
| `AFFORDCRAFT_CATALOG` | `$AFFORDCRAFT_PROJECT_ROOT/catalog/catalog.json` | library catalog (`docs/data.md`) |
| `AFFORDCRAFT_MODELS` | `$AFFORDCRAFT_PROJECT_ROOT/models` | model checkpoints |
| `AFFORDCRAFT_QWEN3_VL` | `$AFFORDCRAFT_MODELS/qwen3-vl-8b-instruct` | grounding, selection and object discovery |
| `AFFORDCRAFT_DINOV2` | `$AFFORDCRAFT_MODELS/dinov2` | folders `image_encoder_dinov2/` and `feature_extractor_dinov2/` |
| `AFFORDCRAFT_CLIP` | `$AFFORDCRAFT_MODELS/clip/ViT-B-32.pt` | encoder-replacement study only |
| `AFFORDCRAFT_RUNS` | `runs` | outputs |
| `AFFORDCRAFT_ISAAC_PYTHON`, `AFFORDCRAFT_RUNTIME_PYTHON`, `AFFORDCRAFT_SELECTOR_PYTHON` | `python`, current, current | interpreters |

## Models

| Model | Use | Source |
|---|---|---|
| Qwen3-VL-8B-Instruct | grounding, multimodal selection, object discovery in cluttered images; bfloat16, greedy decoding, images resized to at most 512 x 512 pixels in area | `Qwen/Qwen3-VL-8B-Instruct` on Hugging Face |
| DINOv2 ViT-L/14 image tower (1024-d CLS feature) with its image processor | visual index and retrieval | the `image_encoder_dinov2` and `feature_extractor_dinov2` folders distributed with the PartCrafter checkpoint (see `baselines/partcrafter`) |
| CLIP ViT-B/32 (OpenAI TorchScript checkpoint `ViT-B-32.pt`) | encoder-replacement condition of the one-factor study | the OpenAI CLIP release |

The visual index records the encoder's configuration hash and the catalog hash; retrieval refuses an index built with
another encoder or catalog.

## Checks

```bash
python -m unittest discover -s tests          # CPU tests of the method (104 tests; needs numpy, scipy, trimesh, CoACD)
$AFFORDCRAFT_RUNTIME_PYTHON scripts/coacd_health.py --output /tmp/coacd_health.json
```

The physics services are checked with seven known positive and negative controls (`docs/running.md`, step 3) before any
formal run.
