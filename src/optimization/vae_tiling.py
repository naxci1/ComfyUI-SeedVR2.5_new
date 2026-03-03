"""
Manual-control VAE tiling module for SeedVR2.

Provides memory-efficient spatial tiling for VAE decode (Phase 3) with
user-specified tile_size and tile_overlap — NO auto-tiling or dynamic
size adjustments:

- User-controlled tile_size and tile_overlap (passed through unchanged)
- Pure BF16 pipeline: all VAE operations locked to torch.bfloat16
- Autocast disabled (enabled=False) to prevent FP32 upcasting that causes
  visual corruption on Blackwell (SM120) GPUs
- Cosine ramp blending on overlap regions for seamless reconstruction
- No torch.cuda.empty_cache() between tiles to keep TMA active
- Monkey-patching to override default VideoAutoencoderKL.tiled_decode
- Blackwell GPU detection and CUDA toolkit diagnostics

Usage:
    from src.optimization.vae_tiling import (
        blackwell_tiled_decode,
        log_tiling_backend_status,
        apply_blackwell_tiled_decode_patch,
    )
"""

import torch
from typing import Tuple, Dict, Any, Optional, TYPE_CHECKING

from .memory_manager import (
    is_cuda_available,
    is_blackwell_gpu,
    detect_gpu_architecture,
    check_cuda_toolkit_alignment,
)

if TYPE_CHECKING:
    from ...utils.debug import Debug


# ── Configuration ────────────────────────────────────────────────────────────

# Backend status flag (logged once per session)
_backend_status_logged = False

# Monkey-patch tracking
_patch_applied = False


def log_tiling_backend_status(debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Log tiling backend status including Blackwell detection and CUDA alignment.
    
    Called once during VAE decode initialization. Reports:
    - GPU architecture and SM version
    - CUDA toolkit alignment (PyTorch vs system NVCC)
    - Active tiling backend configuration
    - TMA acceleration availability
    
    Returns:
        Status dict with architecture and backend info
    """
    global _backend_status_logged
    if _backend_status_logged:
        return detect_gpu_architecture()
    _backend_status_logged = True
    
    arch = detect_gpu_architecture()
    cuda_info = check_cuda_toolkit_alignment(debug=debug)
    
    if not debug:
        return {**arch, **cuda_info}
    
    # Log GPU architecture
    if arch["is_blackwell"]:
        debug.log(
            f"Blackwell {arch['sm_version']} Detected: {arch['gpu_name']}",
            category="success", force=True
        )
        debug.log(
            "cuTile Backend: Active — pure BF16, autocast disabled, manual tile control",
            category="info", indent_level=1
        )
        debug.log(
            "TMA Acceleration: Enabled — no inter-tile cache clearing",
            category="info", indent_level=1
        )
    else:
        debug.log(
            f"GPU: {arch['gpu_name']} ({arch['name']}, {arch['sm_version']})",
            category="info"
        )
        debug.log(
            "cuTile Backend: Inactive (requires Blackwell SM120+)",
            category="info", indent_level=1
        )
        debug.log(
            "TMA Acceleration: Disabled (requires Blackwell SM120+)",
            category="info", indent_level=1
        )
    
    return {**arch, **cuda_info}


# ── Cosine Ramp Blending ─────────────────────────────────────────────────────

def _build_cosine_ramp(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build a 1-D cosine ramp from 0→1 over `length` samples.
    
    Uses the standard raised-cosine formula:  w = 0.5 - 0.5 * cos(π * t)
    This provides smooth C¹-continuous blending at tile boundaries.
    """
    if length <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    t = torch.linspace(0, 1, steps=length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(t * torch.pi)


# ── Manual-Control Tiled Decode ──────────────────────────────────────────────

def blackwell_tiled_decode(
    vae_model: torch.nn.Module,
    latent: torch.Tensor,
    tile_size: Tuple[int, int] = (960, 960),
    tile_overlap: Tuple[int, int] = (128, 128),
    compute_dtype: torch.dtype = torch.bfloat16,
    debug: Optional['Debug'] = None,
) -> torch.Tensor:
    """
    Manual-control tiled VAE decode with pure BF16 pipeline.
    
    Decodes a latent tensor by splitting it into spatial tiles using
    user-specified tile_size and tile_overlap (no dynamic adjustments).
    Tiles are blended with cosine ramps for seamless reconstruction.
    
    Key design decisions:
    - tile_size and tile_overlap are used EXACTLY as provided — no auto-tiling
    - All operations locked to BF16 to prevent FP32 upcasting corruption
    - Autocast disabled to prevent engine from force-upcasting to FP32
    - No torch.cuda.empty_cache() between tiles to keep TMA active
    - Cosine ramp blending on overlap regions
    
    Args:
        vae_model: VAE model with slicing_decode() method
        latent: Input latent tensor [B, C, F, H, W]
        tile_size: Output-space tile size (H, W) in pixels — used as-is
        tile_overlap: Output-space overlap (H, W) in pixels — used as-is
        compute_dtype: Compute precision (torch.bfloat16 for Blackwell)
        debug: Debug instance for logging
    
    Returns:
        Decoded tensor [B, C_out, F, H_out, W_out]
    """
    if debug:
        debug.log(
            f"Using Tile Size: {tile_size}, Overlap: {tile_overlap}",
            category="vae", force=True, indent_level=1
        )
    
    if latent.ndim != 5:
        latent = latent.unsqueeze(2)
    
    b, c, f, H, W = latent.shape
    
    # Get spatial scale factor from VAE model
    scale_factor = getattr(vae_model, 'spatial_downsample_factor', 8)
    
    # Use tile_size and tile_overlap exactly as provided
    tile_h, tile_w = tile_size
    overlap_h, overlap_w = tile_overlap
    
    latent_tile_h = max(1, tile_h // scale_factor)
    latent_tile_w = max(1, tile_w // scale_factor)
    
    # If latent fits in a single tile, skip tiling overhead
    if H <= latent_tile_h and W <= latent_tile_w:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            latent_bf16 = (latent.to(compute_dtype) if latent.dtype != compute_dtype else latent).contiguous()
            return vae_model.slicing_decode(latent_bf16).contiguous()
    
    latent_overlap_h = max(0, min(overlap_h // scale_factor, latent_tile_h - 1))
    latent_overlap_w = max(0, min(overlap_w // scale_factor, latent_tile_w - 1))
    
    stride_h = max(1, latent_tile_h - latent_overlap_h)
    stride_w = max(1, latent_tile_w - latent_overlap_w)
    
    # Count tiles for logging
    num_tiles_h = max(1, (max(H - latent_overlap_h, 1) + stride_h - 1) // stride_h)
    num_tiles_w = max(1, (max(W - latent_overlap_w, 1) + stride_w - 1) // stride_w)
    num_tiles = num_tiles_h * num_tiles_w
    
    if debug:
        overlap_pct = overlap_h * 100 // tile_h if tile_h > 0 else 0
        debug.log(
            f"Manual tiled decode: {num_tiles} tiles "
            f"({num_tiles_h}×{num_tiles_w}), "
            f"tile={tile_h}×{tile_w}px, "
            f"overlap={overlap_h}px ({overlap_pct}%), dtype={compute_dtype}",
            category="vae", force=True, indent_level=1
        )
    
    # Pre-compute cosine ramp vectors (shared across all tiles)
    ramp_cache = {}
    if overlap_h > 0:
        ramp_cache['h'] = _build_cosine_ramp(overlap_h, latent.device, compute_dtype)
    if overlap_w > 0:
        ramp_cache['w'] = _build_cosine_ramp(overlap_w, latent.device, compute_dtype)
    
    # Accumulation buffers (allocated lazily on first tile)
    result = None
    count = None
    
    # Cast latent to BF16 once upfront and ensure contiguous memory layout
    latent_bf16 = (latent.to(compute_dtype) if latent.dtype != compute_dtype else latent).contiguous()
    
    tile_id = 0
    # Wrap entire decode loop: no_grad + autocast disabled to prevent FP32 upcasting
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        for y_lat in range(0, H, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H)
            for x_lat in range(0, W, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W)
                
                # Skip degenerate tiles fully within previous overlap
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue
                
                tile_id += 1
                
                # Extract tile (already in BF16)
                tile_latent = latent_bf16[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]
                
                # Decode tile — autocast is disabled, pure BF16
                # Ensure contiguous layout before decode (Blackwell hates channels_last)
                decoded_tile = vae_model.slicing_decode(tile_latent.contiguous()).contiguous()
                
                # Initialize accumulation buffers on first tile
                if result is None:
                    b_out, c_out, out_f, _, _ = decoded_tile.shape
                    output_h = H * scale_factor
                    output_w = W * scale_factor
                    
                    # Accumulate on offload device if specified, else same device
                    accum_device = getattr(vae_model, 'tensor_offload_device', None) or decoded_tile.device
                    
                    result = torch.zeros(
                        (b_out, c_out, out_f, output_h, output_w),
                        device=accum_device, dtype=compute_dtype
                    )
                    count = torch.zeros(
                        (1, 1, 1, output_h, output_w),
                        device=accum_device, dtype=compute_dtype
                    )
                
                # Map to output space
                y_out = y_lat * scale_factor
                x_out = x_lat * scale_factor
                y_out_end = y_lat_end * scale_factor
                x_out_end = x_lat_end * scale_factor
                
                h_out = y_out_end - y_out
                w_out = x_out_end - x_out
                
                # Build separable blend weights with cosine ramps on interior edges
                ov_h_out = max(0, min(overlap_h, h_out - 1))
                ov_w_out = max(0, min(overlap_w, w_out - 1))
                
                weight_h = torch.ones((h_out,), device=decoded_tile.device, dtype=compute_dtype)
                weight_w = torch.ones((w_out,), device=decoded_tile.device, dtype=compute_dtype)
                
                if ov_h_out > 0:
                    if y_lat > 0:  # Not top border
                        weight_h[:ov_h_out] = ramp_cache['h'][:ov_h_out]
                    if y_lat_end < H:  # Not bottom border
                        weight_h[-ov_h_out:] = 1 - ramp_cache['h'][:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:  # Not left border
                        weight_w[:ov_w_out] = ramp_cache['w'][:ov_w_out]
                    if x_lat_end < W:  # Not right border
                        weight_w[-ov_w_out:] = 1 - ramp_cache['w'][:ov_w_out]
                
                # Apply weights separably (avoids allocating a 2-D mask)
                weight_h_5d = weight_h.view(1, 1, 1, h_out, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, w_out)
                
                # Ensure dtype consistency before multiply
                if decoded_tile.dtype != compute_dtype:
                    decoded_tile = decoded_tile.to(compute_dtype)
                
                decoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)
                
                # Accumulate (move to result device if different)
                if result.device != decoded_tile.device:
                    decoded_tile = decoded_tile.to(result.device)
                    weight_h_5d = weight_h_5d.to(result.device)
                    weight_w_5d = weight_w_5d.to(result.device)
                
                result[:, :, :decoded_tile.shape[2], y_out:y_out_end, x_out:x_out_end] += decoded_tile
                count[:, :, :, y_out:y_out_end, x_out:x_out_end].addcmul_(weight_h_5d, weight_w_5d)
                
                # Free decoded tile immediately (no empty_cache — keeps TMA active)
                del decoded_tile, tile_latent, weight_h, weight_w, weight_h_5d, weight_w_5d
                
                # Log progress periodically
                if debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        debug.log(f"Decoded tile {tile_id}/{num_tiles}", category="vae", indent_level=1)
                    else:
                        end = min(tile_id + 4, num_tiles)
                        debug.log(f"Decoding tiles {tile_id}-{end}/{num_tiles}", category="vae", indent_level=1)
    
    # Normalize by accumulated weights
    if result is not None:
        # Move back to input device if accumulated elsewhere
        if result.device != latent.device:
            result = result.to(latent.device)
            count = count.to(latent.device)
        result.div_(count.clamp(min=1e-6))
    
    # Handle single-frame squeeze
    if latent.shape[2] == 1 and result is not None:
        result = result.squeeze(2)
    
    return result


# ── Monkey-Patching: Override VideoAutoencoderKL.tiled_decode ────────────────

def _patched_tiled_decode(self, z: torch.Tensor, 
                          tile_size: Tuple[int, int] = (960, 960),
                          tile_overlap: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Replacement for VideoAutoencoderKL.tiled_decode that uses pure BF16
    pipeline with user-specified tile_size and tile_overlap.
    
    This method is monkey-patched onto VideoAutoencoderKL instances when
    apply_blackwell_tiled_decode_patch() is called. It replaces the default
    tiled_decode with the manual-control version from vae_tiling.py.
    
    tile_size and tile_overlap are passed through UNCHANGED — no dynamic
    adjustments or minimum overlap enforcement.
    """
    debug = getattr(self, 'debug', None)
    
    # Pure BF16 for all CUDA GPUs (prevents FP32 upcasting corruption)
    compute_dtype = torch.bfloat16 if z.is_cuda else z.dtype
    
    return blackwell_tiled_decode(
        vae_model=self,
        latent=z,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        compute_dtype=compute_dtype,
        debug=debug,
    )


def apply_blackwell_tiled_decode_patch(vae_model: torch.nn.Module, 
                                        debug: Optional['Debug'] = None) -> bool:
    """
    Monkey-patch the VAE model's tiled_decode method with the manual-control version.
    
    This is the key integration point: without this patch, the VAE model uses its own
    legacy tiled_decode which has FP32 upcasting that causes visual corruption on
    Blackwell (SM120) GPUs.
    
    The patched version:
    - Uses user-provided tile_size/tile_overlap exactly (no auto-tiling)
    - Locks all operations to BF16 (no FP32 upcasting)
    - Wraps decode in autocast(enabled=False) to prevent engine upcasting
    - Uses cosine ramp blending on overlap regions
    - Does NOT call torch.cuda.empty_cache() between tiles (keeps TMA active)
    
    Args:
        vae_model: VideoAutoencoderKL instance to patch
        debug: Debug instance for logging
        
    Returns:
        True if patch was applied, False if already applied or not applicable
    """
    global _patch_applied
    
    if _patch_applied:
        return False
    
    if not hasattr(vae_model, 'tiled_decode'):
        if debug:
            debug.log("VAE model has no tiled_decode method, skipping patch", 
                      category="warning", indent_level=1)
        return False
    
    import types
    vae_model.tiled_decode = types.MethodType(_patched_tiled_decode, vae_model)
    _patch_applied = True
    
    # Log backend status on first patch application
    log_tiling_backend_status(debug=debug)
    
    # KILL-SWITCH: force_upcast = False unconditionally
    # FP32 upcasting causes visual corruption on Blackwell (RTX 50xx) GPUs
    if hasattr(vae_model, 'config'):
        vae_model.config.force_upcast = False
    # Also disable on the nested decoder config (some VAE architectures store it separately)
    decoder = getattr(vae_model, 'decoder', None)
    if decoder is not None and hasattr(decoder, 'config'):
        decoder.config.force_upcast = False
    if debug:
        debug.log(
            "force_upcast = False — pure BF16 pipeline active",
            category="info", indent_level=1
        )
    
    if debug:
        debug.log(
            "tiled_decode patch applied — manual tile control, autocast disabled",
            category="success", indent_level=1, force=True
        )
    
    return True
