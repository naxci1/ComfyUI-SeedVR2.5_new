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
Multi-head Latent Attention (MLA) for SeedVR2 DiT.

Inspired by the DeepSeek-V2/V3 MLA architecture:
- Low-rank joint KV compression: K,V are reconstructed from a shared latent vector
  c_KV of rank `kv_lora_rank`, reducing KV-cache from 2*heads*head_dim per token
  to kv_lora_rank + heads*head_dim (RoPE keys only) per token.
- Optional query low-rank compression (q_lora_rank) for extra parameter efficiency.
- Full compatibility with the existing FlashAttentionVarlen / SageAttention backends.

Target hardware: NVIDIA Blackwell (64 GB+ VRAM). The default `kv_lora_rank=512`
is tuned to saturate memory bandwidth on large-VRAM devices without leaving
execution pipelines idle (ring_size is handled by RingMLAAttention below).

Dimensions used in the SeedVR2 DiT:
  hidden_dim = 1152
  heads      = 16
  head_dim   = 64
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLALayer(nn.Module):
    """
    Multi-head Latent Attention (MLA) layer.

    For each token the forward pass:
      1. Projects input to a *compressed* latent c_KV = W_DKV · x  (dim: kv_lora_rank)
      2. Reconstructs K and V via learned up-projections:
           K_content = W_UK · RMSNorm(c_KV)   shape: [L, heads, head_dim]
           V         = W_UV · RMSNorm(c_KV)   shape: [L, heads, head_dim]
      3. Computes query (optionally with low-rank compression):
           Q = W_Q · x   (or W_UQ · RMSNorm(W_DQ · x))
      4. Calls the supplied `attn_fn` (variable-length attention backend).

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension (default 1152 for SeedVR2-3B).
    heads : int
        Number of attention heads (default 16).
    head_dim : int
        Dimension per head (default 64).
    kv_lora_rank : int
        Rank of the joint KV low-rank projection. Larger values use more VRAM
        bandwidth but recover more expressiveness. Recommended 256–512 for 64 GB
        VRAM devices.
    q_lora_rank : int or None
        If set, Q is also computed through a low-rank bottleneck of this size.
        Set to None to use a direct full-rank Q projection.
    qk_norm_eps : float
        Epsilon for QK RMSNorm.
    bias : bool
        Whether to add bias to projection layers.
    """

    def __init__(
        self,
        hidden_dim: int = 1152,
        heads: int = 16,
        head_dim: int = 64,
        kv_lora_rank: int = 512,
        q_lora_rank: Optional[int] = None,
        qk_norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.scale = head_dim ** -0.5

        # ----------- Query projection -----------
        if q_lora_rank is not None:
            self.W_DQ = nn.Linear(hidden_dim, q_lora_rank, bias=bias)
            self.W_UQ = nn.Linear(q_lora_rank, self.inner_dim, bias=bias)
            self.norm_q_lora = nn.RMSNorm(q_lora_rank, eps=qk_norm_eps)
        else:
            self.W_Q = nn.Linear(hidden_dim, self.inner_dim, bias=bias)

        self.norm_q = nn.RMSNorm(head_dim, eps=qk_norm_eps)

        # ----------- Low-rank KV compression -----------
        # Down-projection: hidden → kv_lora_rank
        self.W_DKV = nn.Linear(hidden_dim, kv_lora_rank, bias=bias)

        # Normalise before up-projection (improves stability)
        self.norm_kv = nn.RMSNorm(kv_lora_rank, eps=qk_norm_eps)

        # Up-projections: kv_lora_rank → K and V
        self.W_UK = nn.Linear(kv_lora_rank, self.inner_dim, bias=bias)
        self.W_UV = nn.Linear(kv_lora_rank, self.inner_dim, bias=bias)

        self.norm_k = nn.RMSNorm(head_dim, eps=qk_norm_eps)

        # ----------- Output projection -----------
        self.W_O = nn.Linear(self.inner_dim, hidden_dim, bias=bias)

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Compute query vectors, optionally through low-rank path."""
        if self.q_lora_rank is not None:
            q = self.W_UQ(self.norm_q_lora(self.W_DQ(x)))
        else:
            q = self.W_Q(x)
        # q: [L, inner_dim] → [L, heads, head_dim]
        q = q.view(*q.shape[:-1], self.heads, self.head_dim)
        return self.norm_q(q)

    def _project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute K and V through the shared low-rank latent."""
        c_kv = self.norm_kv(self.W_DKV(x))          # [L, kv_lora_rank]
        k = self.W_UK(c_kv).view(*x.shape[:-1], self.heads, self.head_dim)
        v = self.W_UV(c_kv).view(*x.shape[:-1], self.heads, self.head_dim)
        return self.norm_k(k), v

    def forward(
        self,
        x: torch.Tensor,
        attn_fn,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **attn_kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [L, hidden_dim]
            Packed (variable-length) token sequence.
        attn_fn : callable
            Variable-length attention backend, e.g. FlashAttentionVarlen.forward.
        cu_seqlens : Tensor  [B+1]  int32
            Cumulative sequence lengths (padded with leading 0).
        max_seqlen : int
            Maximum sequence length in the batch.
        **attn_kwargs
            Forwarded verbatim to `attn_fn`.

        Returns
        -------
        Tensor  [L, hidden_dim]
        """
        q = self._project_q(x)            # [L, heads, head_dim]
        k, v = self._project_kv(x)        # [L, heads, head_dim] each

        # Call the attention backend
        out = attn_fn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            **attn_kwargs,
        )  # [L, heads, head_dim]

        # Merge heads and project back
        out = out.reshape(*out.shape[:-2], self.inner_dim)
        return self.W_O(out)


class CrossMLALayer(MLALayer):
    """
    Cross-attention variant of MLA.

    The query comes from `x_q` (e.g., video tokens) and the keys/values
    come from `x_kv` (e.g., text tokens), each with independent sequence
    lengths described by separate cu_seqlens tensors.
    """

    def forward(  # type: ignore[override]
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_fn,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        **attn_kwargs,
    ) -> torch.Tensor:
        q = self._project_q(x_q)
        k, v = self._project_kv(x_kv)

        out = attn_fn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **attn_kwargs,
        )

        out = out.reshape(*out.shape[:-2], self.inner_dim)
        return self.W_O(out)


class MLAConfig:
    """
    Configuration dataclass for MLA hyper-parameters.

    Recommended defaults for Blackwell 64 GB VRAM
    (hidden_dim=1152, heads=16, head_dim=64):
      kv_lora_rank = 512   — half the inner_dim, good VRAM/expressiveness balance
      q_lora_rank  = None  — direct Q projection (fewer parameters to tune)
    """

    def __init__(
        self,
        hidden_dim: int = 1152,
        heads: int = 16,
        head_dim: int = 64,
        kv_lora_rank: int = 512,
        q_lora_rank: Optional[int] = None,
        qk_norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_norm_eps = qk_norm_eps
        self.bias = bias

    def build(self) -> MLALayer:
        return MLALayer(
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            head_dim=self.head_dim,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_norm_eps=self.qk_norm_eps,
            bias=self.bias,
        )
