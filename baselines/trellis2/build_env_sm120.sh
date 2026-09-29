#!/bin/bash
# Build the TRELLIS.2 runtime inside a copy of the sm_120 geometry environment (geom5090: torch 2.7.0+cu128, xformers
# 0.0.31, nvdiffrast 0.4.0, sm_120) whose interpreter is TRELLIS2_PYTHON. Every install is logged to stdout and later
# recorded in configuration.json. TRELLIS2_HOME: official checkout (provides o-voxel/); TRELLIS2_EXT_BUILD: build dir.
# A second pass of the executed build only repeated the CuMesh clone (the first clone returned HTTP 503), the o-voxel
# rebuild and the nvdiffrec step with the same commands. Extensions are compiled for TORCH_CUDA_ARCH_LIST=12.0 (RTX 5090).
set -u
PY=${TRELLIS2_PYTHON:?set TRELLIS2_PYTHON}; PIP="$PY -m pip"
export TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda MAX_JOBS=32 FORCE_CUDA=1
IDX=""
X=${TRELLIS2_EXT_BUILD:-$PWD/ext}; mkdir -p $X; cd $X
log(){ echo "$(date -Is) $*"; }
log "python $($PY --version 2>&1)"
$PY -c 'import torch;print("torch", torch.__version__, "cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)'
log "STEP basic deps"
$PIP install $IDX zstandard pandas easydict imageio imageio-ffmpeg ninja tqdm 2>&1 | tail -2
log "STEP transformers>=4.56 (DINOv3ViTModel)"
$PIP install $IDX "transformers>=4.56,<5" 2>&1 | tail -3
$PY -c "import transformers; from transformers import DINOv3ViTModel; print('transformers', transformers.__version__, 'DINOv3ViTModel ok')"
log "STEP utils3d pinned commit 9a4eb15e"
$PIP install --no-deps --no-build-isolation "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" 2>&1 | tail -2
$PY -c "import utils3d; print('utils3d ok', hasattr(utils3d.torch, 'extrinsics_look_at'))"
log "STEP flash-attn prebuilt wheel (v2.7.4.post1, torch2.7 cu12 cp310)"
ABI=$($PY -c 'import torch;print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
W="flash_attn-2.7.4.post1+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
[ -s $X/$W ] || curl -sSL --retry 3 -o $X/$W "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$W"
ls -la $X/$W
$PIP install --no-deps $X/$W 2>&1 | tail -2
$PY - <<'PY' 2>&1 | tail -3
import torch
try:
    import flash_attn
    from flash_attn import flash_attn_func
    q=torch.randn(1,64,8,64,device='cuda',dtype=torch.float16); o=flash_attn_func(q,q,q); torch.cuda.synchronize()
    print('flash_attn', flash_attn.__version__, 'runs on', torch.cuda.get_device_name(), tuple(o.shape))
except Exception as e:
    print('FLASH_ATTN_FAIL', type(e).__name__, str(e)[:300])
PY
log "STEP FlexGEMM"
[ -d $X/FlexGEMM ] || git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git $X/FlexGEMM 2>&1 | tail -1
echo "flexgemm commit $(git -C $X/FlexGEMM rev-parse HEAD)"
$PIP install --no-deps --no-build-isolation $X/FlexGEMM 2>&1 | tail -2
$PY -c "import flex_gemm; from flex_gemm.ops.grid_sample import grid_sample_3d; print('flex_gemm ok')"
log "STEP CuMesh"
[ -d $X/CuMesh ] || git clone --recursive https://github.com/JeffreyXiang/CuMesh.git $X/CuMesh 2>&1 | tail -1
echo "cumesh commit $(git -C $X/CuMesh rev-parse HEAD)"
$PIP install --no-deps --no-build-isolation $X/CuMesh 2>&1 | tail -3
$PY -c "import cumesh; m=cumesh.CuMesh(); print('cumesh ok')"
log "STEP o-voxel"
rm -rf $X/o-voxel; cp -r ${TRELLIS2_HOME:?set TRELLIS2_HOME}/o-voxel $X/o-voxel
$PIP install --no-deps --no-build-isolation $X/o-voxel 2>&1 | tail -3
$PY -c "import o_voxel; from o_voxel import _C; print('o_voxel ok')"
log "STEP nvdiffrec (renderutils branch, PBR renders)"
[ -d $X/nvdiffrec ] || git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git $X/nvdiffrec 2>&1 | tail -1
echo "nvdiffrec commit $(git -C $X/nvdiffrec rev-parse HEAD)"
$PIP install --no-deps --no-build-isolation $X/nvdiffrec 2>&1 | tail -3
$PY -c "import nvdiffrec_render; print('nvdiffrec_render ok')" 2>&1 | tail -1
log "STEP import check"
cd $TRELLIS2_HOME
$PY - <<'PY' 2>&1 | tail -8
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import torch, cv2
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel, cumesh, flex_gemm
print('trellis2 imports ok; xformers:', __import__('xformers').__version__, 'nvdiffrast:', __import__('nvdiffrast').__version__)
PY
$PIP list 2>/dev/null > $X/t2_env_piplist.txt
log "BUILD_DONE"
