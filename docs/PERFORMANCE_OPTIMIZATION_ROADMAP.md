# Performance & Optimization Roadmap

## Target System Specifications

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) |
| CPU | Intel Core i7-14700KF |
| RAM | 96 GB DDR5 |
| Software | Python 3.12.10, PyTorch 2.7.1+cu128, CUDA 12.8, cuDNN 90701 |

---

## 1. Modern GPU Architecture: RTX 50-Series (Blackwell) Optimizations

### 1.1 Native FP8 Inference

**Current State:** The codebase already has substantial FP8 infrastructure:
- `CompatibleDiT` wrapper (`src/optimization/compatibility.py:720`) detects FP8 model weights (`torch.float8_e4m3fn`, `torch.float8_e5m2`) and handles input/output conversion.
- RoPE frequency buffers are converted from FP8 to `compute_dtype` for numerical stability.
- FP8 parameters are kept in FP8 for memory efficiency; only converted to `compute_dtype` (typically `bfloat16`) during arithmetic operations.

**Recommendations:**

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Use `torch.float8_e4m3fn` as the default weight dtype for RTX 50xx** — Blackwell supports native FP8 Tensor Core operations. This halves memory for DiT parameters (3B model: ~6GB → ~3GB, 7B model: ~14GB → ~7GB). | 50% VRAM reduction on model weights | LOW — already supported |
| HIGH | **Enable FP8 GEMM via `torch._scaled_mm`** — PyTorch 2.7 supports hardware-accelerated FP8 matrix multiplications on Blackwell. The current `CompatibleDiT._process_inputs()` converts FP8→bfloat16 before computation; instead, use per-tensor scaling to keep computation in FP8. | 2× throughput on linear layers | MEDIUM |
| MEDIUM | **FP8 KV-Cache for attention** — Store key/value tensors in FP8 during the attention pass. The `FlashAttentionVarlen` class (`src/models/dit_3b/attention.py:80`) already handles dtype conversion via `self.compute_dtype`; extend this to cache Q/K/V in FP8 between layers. | ~30% reduction in attention VRAM | MEDIUM |
| MEDIUM | **Leverage Transformer Engine (TE)** — NVIDIA's `transformer_engine` library provides drop-in replacements for `nn.Linear` with automatic FP8 quantization and per-tensor scaling. Replace the `Linear` layers inside attention blocks (`nablocks/attention/mmattn.py`) with `te.Linear`. | 2-3× speedup on linear ops with automatic scaling | MEDIUM — requires `transformer_engine` dependency |

### 1.2 SageAttention 3 (Blackwell-Native)

**Current State:** The attention backend system (`src/models/dit_3b/attention.py:80-148`) already supports SageAttention 3 as a selectable backend:
```
flash_attn_3 → flash_attn_2 → sageattn_3 → sageattn_2 → sdpa (fallback)
```
The wrapper is at `src/optimization/compatibility.py:448` (`call_sage_attn_3_varlen()`).

**Recommendations:**

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Default to `sageattn_3` on RTX 50xx** — SageAttention 3 is specifically optimized for Blackwell's new Tensor Cores. The `validate_attention_mode()` function (`compatibility.py:175`) already has auto-detection; add hardware generation detection to prefer SA3 on SM100+. | 30-50% faster attention vs SDPA | LOW — backend already implemented |
| MEDIUM | **INT8 quantized attention with SA3** — SageAttention 3 supports INT8 QK multiplication with FP8 PV accumulation. This doubles throughput for attention operations compared to FP16. | 2× attention throughput | LOW — SA3 handles this internally |

---

## 2. VAE Bottleneck: CPU-Offloading & Pinned Memory Strategy (No Tiling)

### 2.1 Problem Analysis

The VAE decoder is the primary VRAM bottleneck. During decode, the decoder upsamples latents through multiple spatial upsampling stages (`Upsample3D` in `video_vae.py:114`), which can spike VRAM to 4-8× the input latent size. For 1080p output, a single decode pass can require 8-12GB of intermediate activations.

**Current mitigation:** The `attn_video_vae.py` variant supports spatial tiling (`tiled_encode`/`tiled_decode`), but this introduces blending artifacts at tile boundaries.

### 2.2 Strategy: Pinned Memory + Activation Streaming

Rather than tiling, exploit the 96GB system RAM as an activation spillover buffer:

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Pinned memory pre-allocation for VAE activations** — Allocate pinned (page-locked) CPU tensors at startup via `torch.empty(..., pin_memory=True)` sized to the expected activation footprint. During `Decoder3D.forward()`, stream intermediate activations to pinned memory between `UpDecoderBlock3D` stages. Transfer back to GPU only for the next stage. With PCIe 4.0 x16 (~25 GB/s), a 2GB activation transfer takes ~80ms — acceptable for a decode that runs once per batch. | Reduce peak VRAM by 40-60% during decode | MEDIUM |
| HIGH | **CUDA Streams for overlapped transfers** — Use a dedicated `torch.cuda.Stream()` for CPU↔GPU transfers during VAE decode. While the decoder processes block N, asynchronously prefetch block N+1's activations and offload block N-1's results. This hides transfer latency behind computation. | Hide 80-90% of transfer overhead | MEDIUM |
| MEDIUM | **`torch.cuda.memory.CUDAPluggableAllocator`** — PyTorch 2.7 supports custom memory allocators. Implement a two-tier allocator: GPU for active computation, pinned CPU for overflow. This is transparent to model code and requires no architectural changes. | Automatic VRAM/RAM tiering | HIGH |
| LOW | **Memory-mapped tensors** — For very long videos (>100 frames), use `torch.UntypedStorage.from_file()` to memory-map intermediate latents to disk-backed storage. The 96GB RAM acts as a cache layer. | Handles arbitrarily long videos | HIGH |

### 2.3 Strategy: CUDA Graphs for Repeated Passes

**Current State:** The `SeedVR2TorchCompileSettings` node (`src/interfaces/torch_compile_settings.py`) supports `cudagraphs` as a backend, but only through `torch.compile`. Direct CUDA graph capture is not used.

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **CUDA Graph capture for DiT inference** — The DiT upscaling pass (`runner.inference()` in `generation_phases.py:720-730`) runs the same model with the same shapes for every batch. Capture a CUDA graph on the first batch and replay it for subsequent batches. This eliminates CPU-side kernel launch overhead (typically 5-15% of total time). | 10-20% speedup on DiT pass | MEDIUM — requires fixed shapes per graph capture |
| MEDIUM | **CUDA Graph capture for VAE decode** — Similarly, the VAE decode pass has fixed shapes per-resolution. Capture once, replay for all batches at that resolution. | 5-10% speedup on VAE pass | MEDIUM |
| LOW | **Warm-up graph pool** — Pre-capture CUDA graphs for common resolutions (720p, 1080p, 4K) during model loading. This amortizes the first-batch compilation penalty. | Eliminates first-batch latency | LOW |

---

## 3. PyTorch 2.7+ Exclusive Features

### 3.1 Advanced `torch.compile` Modes

**Current State:** The codebase is extensively prepared for `torch.compile`:
- `na.py` modules (both dit_3b and dit_7b) are documented as "optimized for torch.compile compatibility" with eliminated graph breaks.
- All C++ extension calls (Flash Attention, SageAttention, GGUF dequant) are wrapped with `@torch._dynamo.disable` to avoid tracing.
- `pytorch_varlen_attention()` uses `torch.tensor_split` (not `.item()`) to avoid graph breaks.

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Enable `torch.compile(mode="max-autotune")` by default** — The codebase is already compile-ready. `max-autotune` triggers Triton autotuning for optimal kernel configurations on the specific hardware. First-run cost is ~2-5 minutes, but amortized over video processing. | 20-40% DiT throughput increase | LOW — already supported via TorchCompileSettings node |
| HIGH | **Use `torch.compile(mode="reduce-overhead")` for VAE** — The VAE has simpler shapes and fewer dynamic paths. `reduce-overhead` mode uses CUDA graphs internally with minimal compilation cost. | 15-25% VAE throughput increase | LOW |
| MEDIUM | **`torch._inductor.config.coordinate_descent_tuning = True`** — PyTorch 2.7's inductor backend supports coordinate descent autotuning. Enable this globally for the best Triton kernel configurations per-layer. | 5-15% additional throughput on compile | LOW — single config line |
| MEDIUM | **`torch._inductor.config.triton.unique_kernel_names = True`** — Generates unique kernel names for profiling. Essential for identifying which layers are bottlenecks via `torch.profiler`. | Enables precise optimization | LOW |

### 3.2 SDPA Backend Selection

**Current State:** `F.scaled_dot_product_attention()` is the default fallback in both `TorchAttention` and `pytorch_varlen_attention()` (`attention.py:53-57`). PyTorch 2.7 selects the backend automatically.

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Force FlashAttention backend via `torch.nn.attention.sdpa_kernel`** — PyTorch 2.7 allows explicit backend selection: `with sdpa_kernel(SDPBackend.FLASH_ATTENTION):`. On RTX 50xx, this ensures the flash attention CUDA kernel is used instead of the math fallback. | Ensures optimal attention path | LOW |
| MEDIUM | **CuDNN Attention backend** — PyTorch 2.7 added CuDNN as an SDPA backend. For fixed sequence lengths (same-resolution batches), CuDNN attention can be 10-20% faster than FlashAttention. | 10-20% attention speedup for fixed shapes | LOW |
| LOW | **`torch.nn.attention.flex_attention`** — PyTorch 2.7's FlexAttention API enables custom attention patterns (e.g., windowed attention) with compiled kernels. The current `window.py` modules use manual windowing that breaks compilation; FlexAttention could replace these. | Eliminates window attention graph breaks | HIGH |

---

## 4. Memory Management: Tensor Lifecycle Analysis

### 4.1 Current Clone Analysis

The codebase is relatively clean. Only 3 `.clone()` calls exist in the entire `src/` directory:

| File | Line | Usage | Assessment |
|------|------|-------|------------|
| `color_fix.py:714` | `matched_s = content_s.clone()` | HSV saturation matching — needs independent mutation | **Necessary** |
| `alpha_upscaling.py:332` | `rgb_normalized = upscaled_rgb.clone()` | Alpha channel processing — modified in-place after | **Necessary** |
| `alpha_upscaling.py:415` | `alpha_final = alpha_upscaled.clone()` | Alpha refinement — modified in-place after | **Necessary** |

**Assessment:** No unnecessary clones detected. All 3 are protecting data that would be corrupted by subsequent in-place operations.

### 4.2 Transfer Overhead Analysis

| Pattern | Locations | Assessment |
|---------|-----------|------------|
| `non_blocking=False` in BlockSwap | `blockswap.py:377,385,506,512,585,591` | **Optimization opportunity** — Change to `non_blocking=True` with explicit `torch.cuda.synchronize()` before computation. Currently blocking on every swap. |
| `non_blocking=False` in model loading | `model_loader.py:202` | **Optimization opportunity** — Model loading can use non-blocking transfers. |
| `manage_tensor()` defaults | `memory_manager.py:589` | Already supports `non_blocking` parameter but defaults to `False`. |

### 4.3 Zero-Copy Recommendations

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **Enable `non_blocking=True` for BlockSwap transfers** — The block swap mechanism (`blockswap.py`) uses synchronous transfers for safety. On a high-bandwidth PCIe 4.0 bus, switching to async transfers with strategic synchronization would hide latency behind computation. The `_on_forward_pre_hook` / `_on_forward_post_hook` pattern is ideal for this since the synchronization point is naturally at the forward pass boundary. | 20-40% faster block swapping | LOW |
| HIGH | **Pre-allocate and reuse transfer buffers** — Instead of creating new tensors for each block swap, maintain a pair of pinned CPU buffers sized to the largest transformer block. Reuse them via `buffer.copy_(block_data, non_blocking=True)`. | Eliminates allocation overhead | MEDIUM |
| MEDIUM | **Tensor view sharing between phases** — In `generation_phases.py`, the `_prepare_video_batch()` function (line 92) already uses view/slice instead of copy. Extend this pattern to phase transitions: when moving from Phase 1→2→3, avoid materializing intermediate lists where possible. | Reduces peak memory between phases | MEDIUM |
| MEDIUM | **`torch.cuda.memory.set_per_process_memory_fraction(0.95)`** — Allow PyTorch to use 95% of GPU memory (vs default ~67%). On a dedicated inference machine with 16GB VRAM, this gives an extra ~4.5GB headroom. | ~4.5GB additional usable VRAM | LOW |

---

## 5. Architectural Overhaul: Library & Kernel Integration

### 5.1 Attention Libraries

| Library | Status | Recommendation |
|---------|--------|---------------|
| **FlashAttention 2** | ✅ Integrated (`call_flash_attn_2_varlen`) | Keep as fallback for Ampere GPUs |
| **FlashAttention 3** | ✅ Integrated (`call_flash_attn_3_varlen`) | Keep — optimal for Hopper/Blackwell |
| **SageAttention 2/3** | ✅ Integrated | SA3 should be default on RTX 50xx |
| **xFormers** | ⚠️ Stubbed (`ensure_xformers_flash_compat`) | **Not recommended** — redundant with FA2/3 and SA2/3. Remove the stub to simplify dependency tree. |
| **PyTorch SDPA** | ✅ Default fallback | Already optimal as universal fallback |

### 5.2 Quantization & Compression

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| HIGH | **GGUF Q4_K_M for DiT weights** — The codebase already supports GGUF loading (`src/optimization/gguf_ops.py`, `gguf_dequant.py`). Q4_K_M quantization reduces the 7B model from ~14GB to ~4GB, fitting comfortably in 16GB VRAM alongside the VAE. | 70% model weight reduction | LOW — already supported |
| MEDIUM | **Dynamic quantization for VAE decoder** — Apply `torch.ao.quantization.quantize_dynamic()` to VAE `Conv2d`/`Conv3d` layers during decode. This reduces VAE weight memory from ~500MB to ~250MB with negligible quality loss. | 50% VAE weight reduction | LOW |
| LOW | **KV-Cache quantization** — Quantize attention KV-cache to INT8 during inference. The `Cache` class (`src/common/cache.py`) could support this transparently. | 50% attention VRAM reduction | MEDIUM |

### 5.3 Custom CUDA Kernels

| Priority | Recommendation | Impact | Complexity |
|----------|---------------|--------|------------|
| MEDIUM | **Fused RoPE + QKV projection** — The current RoPE implementation (`rope.py`) runs as a separate kernel after Q/K projection. Fusing these into a single Triton kernel eliminates one memory read/write cycle. | 5-10% speedup on attention | HIGH |
| MEDIUM | **Fused GroupNorm + SiLU** — The VAE uses the pattern `norm → SiLU → conv` repeatedly (`ResnetBlock3D.custom_forward` in `video_vae.py:302-315`). Fuse `GroupNorm + SiLU` into a single kernel. PyTorch 2.7's Triton code generation can do this automatically with `torch.compile`. | 5-10% speedup on VAE | LOW if using torch.compile |
| LOW | **Custom temporal convolution kernel** — The `InflatedCausalConv3d` (`causal_inflation_lib.py`) handles temporal convolutions with custom slicing logic. A dedicated Triton kernel could handle the causal masking + convolution in a single pass. | 10-15% speedup on temporal ops | HIGH |

---

## Summary: Prioritized Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. Default to `sageattn_3` on SM100+ hardware
2. Enable `torch.compile(mode="max-autotune")` for DiT
3. Enable `torch.compile(mode="reduce-overhead")` for VAE
4. Switch BlockSwap transfers to `non_blocking=True`
5. Set `torch.cuda.memory.set_per_process_memory_fraction(0.95)`

### Phase 2: Medium Effort (1-2 weeks)
1. Implement pinned memory activation streaming for VAE decode
2. CUDA Graph capture for DiT inference loop
3. Pre-allocate reusable transfer buffers for BlockSwap
4. Integrate `transformer_engine` for FP8 linear layers

### Phase 3: Architectural (2-4 weeks)
1. Custom Triton kernel for fused RoPE + QKV
2. FlexAttention for windowed attention patterns
3. Two-tier memory allocator (GPU + pinned CPU)
4. KV-Cache quantization in attention layers

### Expected Combined Impact
| Metric | Current (est.) | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------------|---------------|---------------|---------------|
| DiT throughput | 1× | 1.3-1.5× | 1.8-2.5× | 2.5-3.5× |
| VAE decode VRAM | ~12GB peak | ~12GB | ~6-8GB | ~4-6GB |
| Block swap overhead | ~200ms/swap | ~120ms/swap | ~40ms/swap | ~20ms/swap |
| End-to-end (5 frames 1080p) | ~30s | ~22s | ~14s | ~10s |
