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

import torch
import torch.nn.functional as F
import math

# Import flash/sage attn with automatic fallback from compatibility layer
from ...optimization.compatibility import (
    call_flash_attn_2_varlen, call_flash_attn_3_varlen,
    call_sage_attn_2_varlen, call_sage_attn_3_varlen
)

from torch import nn


def pytorch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False):
    """
    A PyTorch-based implementation of variable-length attention to replace flash_attn_varlen_func.
    It processes each sequence in the batch individually.
    
    NOTE: max_seqlen_q and max_seqlen_k are accepted for API compatibility but not used.
    PyTorch's scaled_dot_product_attention automatically handles variable sequence lengths.
    
    COMPILE OPTIMIZATION: Uses torch.tensor_split to avoid .item() graph breaks
    """
    # Split q, k, v using cumulative sequence lengths
    # NOTE: torch.tensor_split requires int64 dtype and CPU device (PyTorch requirements)
    q_splits = list(torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0))
    k_splits = list(torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    v_splits = list(torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0))

    # Process each sequence
    output_splits = []
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
        # Reshape for torch's scaled_dot_product_attention which expects (batch, heads, seq, dim).
        # Here, we treat each sequence as a batch of 1.
        q_i = q_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_q, head_dim)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0) # (1, heads, seq_len_k, head_dim)

        # Use PyTorch's built-in scaled dot-product attention.
        output_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, 
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=causal
        )

        # Reshape the output back to the original format (seq_len, heads, head_dim)
        output_i = output_i.squeeze(0).permute(1, 0, 2)
        output_splits.append(output_i)
    
    # Concatenate all outputs
    return torch.cat(output_splits, dim=0)


def _resolve_local_window_size(local_window_size, video_tokens: int) -> int:
    if torch.is_tensor(local_window_size):
        local_window_size = int(local_window_size.item())
    elif isinstance(local_window_size, (list, tuple)):
        local_window_size = int(local_window_size[0]) if local_window_size else 0

    if local_window_size is None or local_window_size <= 0:
        local_window_size = max(8, min(video_tokens, int(math.sqrt(max(video_tokens, 1))) * 4))

    return max(1, min(int(local_window_size), max(video_tokens, 1)))


def pytorch_local_varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    deterministic=False,
    video_token_counts=None,
    text_token_counts=None,
    local_window_size=None,
):
    q_splits = list(torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0))
    k_splits = list(torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0))
    v_splits = list(torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0))

    output_splits = []
    for seq_idx, (q_i, k_i, v_i) in enumerate(zip(q_splits, k_splits, v_splits)):
        seq_q = q_i.shape[0]
        seq_k = k_i.shape[0]
        txt_q = int(text_token_counts[seq_idx].item()) if text_token_counts is not None else 0
        txt_k = int(text_token_counts[seq_idx].item()) if text_token_counts is not None else 0
        vid_q = int(video_token_counts[seq_idx].item()) if video_token_counts is not None else max(seq_q - txt_q, 0)
        vid_k = int(video_token_counts[seq_idx].item()) if video_token_counts is not None else max(seq_k - txt_k, 0)
        radius = _resolve_local_window_size(local_window_size, vid_k)

        q_sdpa = q_i.permute(1, 0, 2).unsqueeze(0)
        k_sdpa = k_i.permute(1, 0, 2).unsqueeze(0)
        v_sdpa = v_i.permute(1, 0, 2).unsqueeze(0)

        if vid_q > 0 and vid_k > 0:
            q_pos = torch.arange(vid_q, device=q_i.device)[:, None]
            k_pos = torch.arange(vid_k, device=q_i.device)[None, :]
            video_keep = (q_pos - k_pos).abs() <= radius
            mask = torch.full((seq_q, seq_k), float("-inf"), device=q_i.device, dtype=q_i.dtype)
            mask[:vid_q, :vid_k] = torch.where(video_keep, torch.zeros((), device=q_i.device, dtype=q_i.dtype), mask[:vid_q, :vid_k])
            if txt_k > 0:
                mask[:vid_q, vid_k:vid_k + txt_k] = 0
            if txt_q > 0:
                mask[vid_q:vid_q + txt_q, :] = 0
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None

        output_i = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask,
            dropout_p=dropout_p if not deterministic else 0.0,
            is_causal=causal and mask is None,
            scale=softmax_scale,
        )
        output_splits.append(output_i.squeeze(0).permute(1, 0, 2))

    return torch.cat(output_splits, dim=0)


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """
    Variable-length attention with configurable backend.
    
    Supported backends:
    - sdpa: PyTorch SDPA (fully compilable, always available)
    - flash_attn_2: Flash Attention 2 (Ampere+)
    - flash_attn_3: Flash Attention 3 (Hopper+)
    - sageattn_2: SageAttention 2
    - sageattn_3: SageAttention 3 (Blackwell/RTX 50xx)
    
    All non-SDPA backends use @torch._dynamo.disable wrapper (C++ extensions).
    """

    def __init__(self, attention_mode: str = 'sdpa', compute_dtype: torch.dtype = None):
        """
        Initialize with specified attention backend.
        
        Args:
            attention_mode: 'sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3', or 'local_block_sparse'
            compute_dtype: Compute dtype for attention (set by pipeline, defaults to None for auto-detection)
        """
        super().__init__()
        self.attention_mode = attention_mode
        self.compute_dtype = compute_dtype

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        
        # Convert to pipeline compute_dtype if configured (handles FP8 → fp16/bf16)
        if self.compute_dtype is not None and q.dtype != self.compute_dtype:
            q = q.to(self.compute_dtype)
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        
        if self.attention_mode == 'flash_attn_3':
            return call_flash_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'flash_attn_2':
            return call_flash_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_3':
            return call_sage_attn_3_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'sageattn_2':
            return call_sage_attn_2_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        elif self.attention_mode == 'local_block_sparse':
            return pytorch_local_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )
        else:
            # PyTorch SDPA
            return pytorch_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, **kwargs
            )