#!/usr/bin/env bash
# setup_env.sh: build the Articulate-Anything run environment as in the executed run: byte copy of the sm_120 geometry
# environment geom5090 (AA_BASE_ENV; a venv layered on a python 3.10 base with sapien 3.0.3) + the Articulate-Anything
# pure-python deps installed with --no-deps from downloaded wheels (versions pinned to the ones of the first build),
# CLIP (openai/CLIP git) and co-tracker @5951295e (imported only), CLIP ViT-B/32 weights (the loader checks the sha256
# in the URL). SAPIEN 2.2.2 is not installed: it segfaulted on driver 595 and was removed, so SAPIEN 3.0.3 of the base
# is used (AA_RENDER_SCRIPT port). environment: AA_BASE_ENV, AA_ENV (target env directory), AA_WORK (logs/, wheels/).
# Log $AA_WORK/logs/env_setup.log; pip freeze $AA_WORK/logs/aa5090_pip_freeze.txt (both hashed by register_configuration.py).
set -u
AA=${AA_WORK:-aa_run}; mkdir -p $AA; AA=$(cd $AA && pwd)
E=${AA_ENV:?set AA_ENV}; W=$AA/wheels; LOG=$AA/logs/env_setup.log
mkdir -p $AA/logs $W $HOME/.cache/clip
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -Is) host=$(hostname)"
[ -d "$E" ] || { echo "copying base env -> $E"; cp -a ${AA_BASE_ENV:?set AA_BASE_ENV} "$E" || { echo COPY_FAILED; exit 1; }; }
PY=$E/bin/python; $PY -c "import sys; print(sys.prefix, sys.version)"
G=$AA_BASE_ENV/bin/python
( unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  cd $W && $G -m pip download "hydra-core==1.3.7" "antlr4-python3-runtime==4.9.3" astor==0.8.1 markdown2 GPUtil==1.4.0 "seaborn==0.13.2" "pandas==2.3.3" \
     "pytz==2026.3.post1" "tzdata==2026.4" "python-dateutil==2.9.0.post0" "numpy==1.26.4" "opencv-python==4.11.0.86" flow_vis \
     --no-deps -d $W; echo PIPDL_EXIT=$? )
( $G -m pip download --no-deps --no-build-isolation "git+https://github.com/openai/CLIP.git" -d $W; echo CLIPDL_EXIT=$?
  $G -m pip download --no-deps --no-build-isolation "git+https://github.com/facebookresearch/co-tracker.git@5951295e0ac49068824f75a497ae6749379ec62b" -d $W; echo COTRDL_EXIT=$?
  [ -s $HOME/.cache/clip/ViT-B-32.pt ] || curl -L -sS -o $HOME/.cache/clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt; echo CLIPW_EXIT=$? )
sha256sum $HOME/.cache/clip/ViT-B-32.pt
ls $W
cd $W
inst() { echo ">>> pip install --no-deps $*"; $PY -m pip install --no-deps --no-index --find-links "$W" "$@" || echo "INSTALL_FAILED: $*"; }
inst "numpy==1.26.4"
inst "opencv-python==4.11.0.86"
$PY -c "import transforms3d, requests" || echo "MISSING transforms3d/requests"
inst hydra-core antlr4-python3-runtime
inst astor markdown2 GPUtil seaborn pandas pytz tzdata python-dateutil
inst ./clip-1.0.zip
inst ./cotracker-2.0.zip
inst flow_vis
echo "=== pip freeze (aa5090 layer + base)"; $PY -m pip freeze 2>/dev/null | sort > $AA/logs/aa5090_pip_freeze.txt; wc -l $AA/logs/aa5090_pip_freeze.txt
echo "=== import check"
$PY - <<'PYEOF'
import importlib, warnings
warnings.simplefilter("ignore")
for m in ["numpy", "cv2", "sapien", "pybullet", "pybullet_data", "hydra", "omegaconf", "astor", "markdown2", "GPUtil", "seaborn", "pandas", "clip",
          "cotracker.predictor", "transforms3d", "requests", "scipy", "sklearn", "trimesh", "PIL", "torch", "torchvision", "flow_vis"]:
    try:
        mod = importlib.import_module(m); print("OK ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("MISSING", m, type(e).__name__, str(e)[:80])
PYEOF
echo "ENV_DONE $(date -Is)"
