#!/bin/bash
# TRELLIS.2 env fix: Triton 3.3.0 (inherited from the sim310 base of the venv copy) aborts in TritonGPUAccelerateMatmul
# for compute capability 12.0 ("computeCapability not supported"), which breaks every flex_gemm sparse-conv kernel on the
# RTX 5090. Triton 3.3.1 (the version torch 2.7.1 pins) adds the sm_120 MMA path. Installed into the venv copy only
# (shadows the base-env triton; the base env is untouched). TRELLIS2_PYTHON: interpreter of the run environment.
set -u
PY=${TRELLIS2_PYTHON:?set TRELLIS2_PYTHON}
echo "$(date -Is) before: $($PY -c 'import triton;print(triton.__version__, triton.__file__)')"
$PY -m pip install --no-deps "triton==3.3.1" 2>&1 | tail -2
echo "$(date -Is) after: $($PY -c 'import triton;print(triton.__version__, triton.__file__)')"
$PY - <<'PY' 2>&1 | tail -5
import torch, triton, triton.language as tl
@triton.jit
def k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)
a = torch.randn(64, 32, device='cuda', dtype=torch.float16); b = torch.randn(32, 64, device='cuda', dtype=torch.float16); c = torch.empty(64, 64, device='cuda', dtype=torch.float32)
k[(1,)](a, b, c, 64, 64, 32)
print('tl.dot on', torch.cuda.get_device_name(), 'max abs err', (c - (a.float() @ b.float())).abs().max().item())
import flash_attn, cumesh, o_voxel, flex_gemm, xformers
print('imports ok; triton', triton.__version__, 'torch', torch.__version__)
PY
$PY -m pip list 2>/dev/null > ${TRELLIS2_EXT_BUILD:-.}/t2_env_piplist.txt
echo "$(date -Is) FIX_DONE"
