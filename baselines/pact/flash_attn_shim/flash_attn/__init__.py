"""Minimal local compatibility shim for the official PAct flash-attn calls.

This probe-only module delegates to torch scaled_dot_product_attention.  It
does not alter PAct weights, data, or output semantics; it exists solely to
make the official release runnable in an environment without flash-attn.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__version__ = "probe-sdpa-shim-0.1"


def _attn(q, k, v, causal=False):
    # Official flash-attn uses [B,L,H,C]; torch SDPA uses [B,H,L,C].
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=bool(causal),
    )
    return out.permute(0, 2, 1, 3)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False, **kwargs):
    q, k, v = qkv.unbind(dim=2)
    return _attn(q, k, v, causal=causal)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, causal=False, **kwargs):
    k, v = kv.unbind(dim=2)
    return _attn(q, k, v, causal=causal)


def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, **kwargs):
    return _attn(q, k, v, causal=causal)


def _segments(cu):
    return [int(x) for x in cu.detach().cpu().tolist()]


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=False, **kwargs):
    cu = _segments(cu_seqlens)
    outs = []
    for i in range(len(cu) - 1):
        part = qkv[cu[i]:cu[i + 1]].unsqueeze(0)
        outs.append(flash_attn_qkvpacked_func(part, dropout_p=dropout_p, causal=causal).squeeze(0))
    return torch.cat(outs, dim=0) if outs else qkv.new_empty((0, qkv.shape[-2], qkv.shape[-1]))


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, causal=False, **kwargs):
    cuq, cuk = _segments(cu_seqlens_q), _segments(cu_seqlens_k)
    outs = []
    for i in range(len(cuq) - 1):
        qi = q[cuq[i]:cuq[i + 1]].unsqueeze(0)
        kvi = kv[cuk[i]:cuk[i + 1]].unsqueeze(0)
        outs.append(flash_attn_kvpacked_func(qi, kvi, dropout_p=dropout_p, causal=causal).squeeze(0))
    return torch.cat(outs, dim=0) if outs else q.new_empty((0, q.shape[-2], q.shape[-1]))


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, causal=False, **kwargs):
    cuq, cuk = _segments(cu_seqlens_q), _segments(cu_seqlens_k)
    outs = []
    for i in range(len(cuq) - 1):
        qi = q[cuq[i]:cuq[i + 1]].unsqueeze(0)
        ki = k[cuk[i]:cuk[i + 1]].unsqueeze(0)
        vi = v[cuk[i]:cuk[i + 1]].unsqueeze(0)
        outs.append(flash_attn_func(qi, ki, vi, dropout_p=dropout_p, causal=causal).squeeze(0))
    return torch.cat(outs, dim=0) if outs else q.new_empty((0, q.shape[-2], v.shape[-1]))

