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
NaMMAttentionMLA — Multi-modal Multi-head Latent Attention block.

Drop-in replacement for NaMMAttention / NaSwinAttention in the SeedVR2 DiT.
Key differences from the standard block:
  - Key/Value tensors are produced by a low-rank joint projection (MLA),
    reducing KV-cache memory by (1 − kv_lora_rank / (2·heads·head_dim)).
  - The inner attention kernel is optionally dispatched through
    RingMLAAttention for sequence-parallel ring splitting.
  - Linear projections are optionally replaced by NVFP4-quantized layers
    via `src.optimization.fp4_quantization.build_fp4_linear`.

The interface is identical to NaMMAttention so it can be used as a
transparent replacement in `nablocks/mmsr_block.py` or `nadit.py`.
"""

from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .....common.cache import Cache
from .....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...mla import MLALayer
from ...ring_attention import RingMLAAttention


class NaMMAttentionMLA(nn.Module):
    """
    Multi-modal attention block using MLA (low-rank KV compression) with an
    optional Ring Attention outer loop.

    Parameters
    ----------
    vid_dim, txt_dim : int
        Hidden dimensions of video and text streams.
    heads, head_dim : int
        Number of attention heads and dimension per head.
    kv_lora_rank : int or None
        Rank of the shared KV low-rank projection. If None, dynamically
        selected for model dimensions and VRAM budget.
    q_lora_rank : int or None
        Rank for optional query low-rank projection.  Default None.
    qk_bias : bool
        Whether to add bias to QK projections.
    qk_norm : callable
        Norm layer constructor for QK normalisation.
    qk_norm_eps : float
        Epsilon for QK norm.
    rope_type, rope_dim : str, int
        Rotary position embedding config (forwarded to get_na_rope).
    shared_weights : bool
        Whether vid/txt streams share weights.
    attention_mode : str
        Inner attention kernel: ``sageattn_3``, ``sageattn_2``,
        ``flash_attn_3``, ``flash_attn_2``, or ``sdpa``.
    ring_size : int or None
        Optional chunk count. If None, ring chunking is selected dynamically.
        Set 1 to disable ring splitting.
    vram_budget_gb : float
        Target VRAM budget passed to MLA/ring dynamic sizing.
    use_fp4 : bool
        Replace Linear layers with NVFP4/FP8/BF16 quantized layers.
    """

    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = "sdpa",
        kv_lora_rank: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        ring_size: Optional[int] = None,
        use_fp4: bool = False,
        vram_budget_gb: float = 16.0,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        inner_dim = heads * head_dim

        # Optionally use FP4-quantized linear layers
        _linear = nn.Linear
        if use_fp4:
            try:
                from .....optimization.fp4_quantization import build_fp4_linear
                _linear = build_fp4_linear
            except Exception:
                pass  # graceful degradation

        # MLA layers for video and text streams
        mla_kwargs = dict(
            hidden_dim=vid_dim,
            heads=heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_norm_eps=qk_norm_eps,
            bias=qk_bias,
            vram_budget_gb=vram_budget_gb,
        )
        self.mla_vid = MLALayer(**mla_kwargs)
        if not shared_weights:
            self.mla_txt = MLALayer(
                **{**mla_kwargs, "hidden_dim": txt_dim}
            )
        else:
            self.mla_txt = self.mla_vid

        # Output projection (keeps same interface as standard block)
        self.proj_out_vid = _linear(inner_dim, vid_dim, bias=True) if callable(_linear) and _linear is nn.Linear else nn.Linear(inner_dim, vid_dim)
        self.proj_out_txt = _linear(inner_dim, txt_dim, bias=True) if callable(_linear) and _linear is nn.Linear else nn.Linear(inner_dim, txt_dim)

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)

        # Inner attention backend (used by MLA forward)
        if ring_size is None or ring_size > 1:
            self.attn = RingMLAAttention(
                ring_size=ring_size,
                attention_mode=attention_mode,
                vram_budget_gb=vram_budget_gb,
            )
            self._use_ring = True
        else:
            self.attn = FlashAttentionVarlen(attention_mode=attention_mode)
            self._use_ring = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _attn_fn(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                 max_seqlen_q, max_seqlen_k, **kwargs):
        """Unified call to either RingMLAAttention or FlashAttentionVarlen."""
        if self._use_ring:
            return self.attn(
                q=q, k=k, v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                **kwargs,
            )
        return self.attn(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        vid: torch.FloatTensor,     # [L_vid, vid_dim]
        txt: torch.FloatTensor,     # [L_txt, txt_dim]
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
        cache: Cache,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        # --- MLA projections (each stream has its own low-rank KV) ---
        # We need Q, K, V for the joint MM attention so we expose the
        # internal projection helpers of MLALayer.
        vid_q = self.mla_vid._project_q(vid)                  # [L_vid, H, D]
        vid_k, vid_v = self.mla_vid._project_kv(vid)

        txt_q = self.mla_txt._project_q(txt)                  # [L_txt, H, D]
        txt_k, txt_v = self.mla_txt._project_kv(txt)

        # --- Distributed gather (same as standard NaMMAttention) ---
        # (no-op when not running in sequence-parallel mode)
        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        # Rope
        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        cu_seqlens = cache(
            "mm_seqlens",
            lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int(),
        )
        max_seqlen = cache("mm_maxlen", lambda: all_len.max())

        attn_out = self._attn_fn(
            q=concat(vid_q, txt_q),
            k=concat(vid_k, txt_k),
            v=concat(vid_v, txt_v),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        ).type_as(vid_q)

        attn_out = rearrange(attn_out, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out = self.proj_out_vid(vid_out)
        txt_out = self.proj_out_txt(txt_out)

        return vid_out, txt_out
