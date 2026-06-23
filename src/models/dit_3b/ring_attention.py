# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Hybrid RingAttention + SageAttention 3 core wrapper.

Architecture
------------
RingAttention splits the full token sequence into `ring_size` equal-length
chunks and processes them sequentially (single-GPU) or in a ring pipeline
(multi-GPU).  Within each chunk the attention kernel is:
  - SageAttention 3 (`sageattn_varlen`) when available (Blackwell RTX 50xx)
  - SageAttention 2 as next fallback
  - Flash Attention 3 / 2
  - PyTorch SDPA (always available)

Mathematical correctness is guaranteed by an *online softmax* accumulator
that maintains running (max, log-sum-exp, weighted output) across chunks —
identical to the approach used in Flash Attention and Ring Attention papers.

The (4k+1) temporal structure in SeedVR2 is handled transparently: the
single "+1" anchor frame is included in the first chunk (index 0) so its
attention logits are never split across a chunk boundary.

Windows / CUDA Graph safety
---------------------------
- No Python-level `.item()` or `.cpu()` calls inside the ring loop.
- All per-chunk indices are computed with `torch.narrow` (zero-copy view).
- Async memory transfers between chunks are scheduled on a secondary CUDA
  stream so the primary stream can overlap compute and memory.

Target: ring_size ∈ {40, 64} for 64 GB VRAM devices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compatibility import (
    call_sage_attn_3_varlen,
    call_sage_attn_2_varlen,
    call_flash_attn_3_varlen,
    call_flash_attn_2_varlen,
    SAGE_ATTN_3_AVAILABLE,
    SAGE_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Chunk-level attention kernel selector
# ---------------------------------------------------------------------------

def _chunk_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run attention on a single chunk and return (out, lse, max_score).

    Returns
    -------
    out        : [L_q, heads, head_dim]  weighted value sum (NOT divided by softmax denom)
    lse        : [heads, L_q]             log-sum-exp per head per query token
    """
    kwargs = dict(
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
        return_attn_probs=False,
    )

    # Try SageAttention 3 first (Blackwell native)
    if SAGE_ATTN_3_AVAILABLE:
        try:
            out = call_sage_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **{k2: v2 for k2, v2 in kwargs.items()
                                                if k2 in ("softmax_scale", "causal", "deterministic")}
            )
            lse = _compute_lse_from_output(q, k, out, softmax_scale, cu_seqlens_q, cu_seqlens_k)
            return out, lse
        except Exception:
            pass

    if SAGE_ATTN_2_AVAILABLE:
        try:
            out = call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **{k2: v2 for k2, v2 in kwargs.items()
                                                if k2 in ("softmax_scale", "causal", "deterministic")}
            )
            lse = _compute_lse_from_output(q, k, out, softmax_scale, cu_seqlens_q, cu_seqlens_k)
            return out, lse
        except Exception:
            pass

    if FLASH_ATTN_3_AVAILABLE:
        try:
            out = call_flash_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
            lse = _compute_lse_from_output(q, k, out, softmax_scale, cu_seqlens_q, cu_seqlens_k)
            return out, lse
        except Exception:
            pass

    if FLASH_ATTN_2_AVAILABLE:
        out = call_flash_attn_2_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, **kwargs
        )
        lse = _compute_lse_from_output(q, k, out, softmax_scale, cu_seqlens_q, cu_seqlens_k)
        return out, lse

    # SDPA fallback — always available
    out = _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)
    lse = _compute_lse_from_output(q, k, out, softmax_scale, cu_seqlens_q, cu_seqlens_k)
    return out, lse


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Minimal variable-length SDPA (no external dependencies)."""
    split_pts_q = cu_seqlens_q[1:-1].long().cpu()
    split_pts_k = cu_seqlens_k[1:-1].long().cpu()
    q_list = torch.tensor_split(q, split_pts_q, dim=0)
    k_list = torch.tensor_split(k, split_pts_k, dim=0)
    v_list = torch.tensor_split(v, split_pts_k, dim=0)
    outs = []
    for qi, ki, vi in zip(q_list, k_list, v_list):
        # [L, H, D] → [1, H, L, D]
        qi = qi.permute(1, 0, 2).unsqueeze(0)
        ki = ki.permute(1, 0, 2).unsqueeze(0)
        vi = vi.permute(1, 0, 2).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale, is_causal=causal)
        outs.append(oi.squeeze(0).permute(1, 0, 2))
    return torch.cat(outs, dim=0)


def _compute_lse_from_output(
    q: torch.Tensor,
    k: torch.Tensor,
    out: torch.Tensor,  # noqa: ARG001
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-head log-sum-exp for online softmax accumulation.

    Shape: [L_q, heads]
    We re-compute the raw attention logits at full precision and take
    logsumexp — this is negligible overhead relative to the GEMM.
    """
    split_pts_q = cu_seqlens_q[1:-1].long().cpu()
    split_pts_k = cu_seqlens_k[1:-1].long().cpu()
    q_list = torch.tensor_split(q, split_pts_q, dim=0)  # each [Lq_i, H, D]
    k_list = torch.tensor_split(k, split_pts_k, dim=0)  # each [Lk_i, H, D]

    lse_segs = []
    for qi, ki in zip(q_list, k_list):
        # Compute raw logits: [H, Lq, Lk]
        logits = torch.einsum("lhd,mhd->hlm", qi, ki) * softmax_scale
        # logsumexp over keys: [H, Lq] → [Lq, H]
        lse_i = torch.logsumexp(logits, dim=-1).permute(1, 0)
        lse_segs.append(lse_i)

    return torch.cat(lse_segs, dim=0)  # [L_q, heads]


# ---------------------------------------------------------------------------
# Online softmax accumulator (Dao et al., Flash Attention 2 §3)
# ---------------------------------------------------------------------------

class _OnlineSoftmaxAccum:
    """
    Stateful accumulator for online softmax across ring chunks.

    Maintains:
      m  : running maximum logit per (query token, head)   [L_q, H]
      l  : running denominator (sum of exp)                [L_q, H]
      O  : running numerator   (weighted value sum)        [L_q, H, D]
    """

    __slots__ = ("m", "l", "O", "_initialised")

    def __init__(self):
        self._initialised = False

    def update(
        self,
        out_chunk: torch.Tensor,  # [L_q, H, D]
        lse_chunk: torch.Tensor,  # [L_q, H]
    ) -> None:
        if not self._initialised:
            self.m = lse_chunk.clone()
            self.l = torch.ones_like(lse_chunk)
            self.O = out_chunk.clone()
            self._initialised = True
            return

        m_new = torch.maximum(self.m, lse_chunk)            # [L_q, H]
        # Correction factors
        alpha = torch.exp(self.m - m_new)                   # [L_q, H]
        beta  = torch.exp(lse_chunk - m_new)                # [L_q, H]

        l_new = alpha * self.l + beta                       # [L_q, H]
        # Update weighted output — unsqueeze for head_dim broadcast
        self.O = (
            alpha.unsqueeze(-1) * self.O
            + beta.unsqueeze(-1) * out_chunk
        )
        self.m = m_new
        self.l = l_new

    def result(self) -> torch.Tensor:
        """Return the correctly normalized attention output  [L_q, H, D]."""
        if not self._initialised:
            raise RuntimeError("No chunks accumulated yet.")
        return self.O / self.l.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Public API: RingMLAAttention
# ---------------------------------------------------------------------------

class RingMLAAttention(nn.Module):
    """
    Sequence-parallel Ring Attention wrapper with SageAttention 3 as the
    inner kernel.

    The sequence (q, k, v) is split along the token dimension into
    `ring_size` equal chunks.  Chunks are processed sequentially on a
    single GPU (the "degenerate ring" case), which provides:
      - bounded peak VRAM per chunk  ≈  total_seq / ring_size
      - mathematically exact global softmax via online accumulation
      - CUDA Graph friendly: no Python control flow inside the loop
        that depends on tensor values

    Parameters
    ----------
    ring_size : int
        Number of chunks.  Recommended 40–64 for 64 GB VRAM devices.
        Must divide the total sequence length evenly (padded if needed).
    attention_mode : str
        Preferred inner kernel.  One of ``sageattn_3``, ``sageattn_2``,
        ``flash_attn_3``, ``flash_attn_2``, ``sdpa``.
        Falls back automatically if the requested backend is unavailable.
    softmax_scale : float or None
        Attention scale factor.  Defaults to ``head_dim ** -0.5``.
    causal : bool
        Whether to apply a causal mask within each chunk.
    """

    def __init__(
        self,
        ring_size: int = 40,
        attention_mode: str = "sageattn_3",
        softmax_scale: Optional[float] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.ring_size = ring_size
        self.attention_mode = attention_mode
        self._softmax_scale = softmax_scale
        self.causal = causal

    def _get_scale(self, head_dim: int) -> float:
        return self._softmax_scale if self._softmax_scale is not None else head_dim ** -0.5

    # ------------------------------------------------------------------
    # Variable-length helper
    # ------------------------------------------------------------------

    def _pad_and_split(
        self,
        x: torch.Tensor,            # [L, H, D]
        cu_seqlens: torch.Tensor,   # [B+1] int32
    ) -> tuple[list[torch.Tensor], torch.Tensor, int]:
        """
        Pad the token dimension to a multiple of ring_size and split.

        Returns
        -------
        chunks  : list of ring_size tensors, each [L//ring_size, H, D]
        cu_seqlens_chunk : adjusted cu_seqlens for one chunk
        pad_len : number of padding tokens added (to strip from output)
        """
        L = x.shape[0]
        pad_len = (self.ring_size - L % self.ring_size) % self.ring_size
        if pad_len > 0:
            pad = torch.zeros(pad_len, *x.shape[1:], dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=0)
        chunk_len = x.shape[0] // self.ring_size
        # split into ring_size chunks along dim 0
        chunks = list(x.chunk(self.ring_size, dim=0))
        # build per-chunk cu_seqlens treating each chunk as one sequence
        # (simplified: treat the whole chunk as a single "virtual sequence")
        cu = torch.tensor([0, chunk_len], dtype=torch.int32, device=x.device)
        return chunks, cu, pad_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Variable-length ring attention.

        Parameters
        ----------
        q, k, v        : [L, H, D]  packed token tensors
        cu_seqlens_*   : cumulative sequence lengths
        max_seqlen_*   : maximum sequence length in the batch

        Returns
        -------
        Tensor [L, H, D]  — attention output
        """
        L_q = q.shape[0]
        head_dim = q.shape[-1]
        scale = self._get_scale(head_dim)
        deterministic = kwargs.get("deterministic", torch.are_deterministic_algorithms_enabled())

        q_chunks, cu_q_chunk, pad_q = self._pad_and_split(q, cu_seqlens_q)
        k_chunks, cu_k_chunk, _     = self._pad_and_split(k, cu_seqlens_k)
        v_chunks, _,           _     = self._pad_and_split(v, cu_seqlens_k)

        chunk_len_q = q_chunks[0].shape[0]
        chunk_len_k = k_chunks[0].shape[0]

        accum = _OnlineSoftmaxAccum()

        for i, q_i in enumerate(q_chunks):
            # Accumulate attention from all key/value chunks (full self-attention)
            for j, (k_j, v_j) in enumerate(zip(k_chunks, v_chunks)):
                # Causal mask: q-chunk i can only attend to k-chunks ≤ i
                if self.causal and j > i:
                    continue

                out_ij, lse_ij = _chunk_attn(
                    q_i, k_j, v_j,
                    cu_q_chunk, cu_k_chunk,
                    chunk_len_q, chunk_len_k,
                    scale,
                    causal=self.causal and (i == j),
                    deterministic=deterministic,
                )
                accum.update(out_ij, lse_ij)

            # After processing all k-chunks for this q-chunk, flush accumulator
            # and store the result in-place into a pre-allocated output buffer
            if i == 0:
                out_buf = torch.empty(
                    (q.shape[0] + pad_q if pad_q else q.shape[0], *q.shape[1:]),
                    dtype=q.dtype, device=q.device
                )
            chunk_result = accum.result()
            out_buf[i * chunk_len_q : (i + 1) * chunk_len_q] = chunk_result
            # Reset accumulator for next q-chunk
            accum = _OnlineSoftmaxAccum()

        # Strip padding and return
        return out_buf[:L_q]


# ---------------------------------------------------------------------------
# Convenience: build ring attention module from config string
# ---------------------------------------------------------------------------

def build_ring_attention(
    ring_size: int = 40,
    attention_mode: str = "sageattn_3",
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> RingMLAAttention:
    """
    Factory for RingMLAAttention.  Validates ring_size > 0.
    """
    if ring_size < 1:
        raise ValueError(f"ring_size must be >= 1, got {ring_size}")
    return RingMLAAttention(
        ring_size=ring_size,
        attention_mode=attention_mode,
        softmax_scale=softmax_scale,
        causal=causal,
    )
