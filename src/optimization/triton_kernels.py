"""
Fused Triton kernels for the SeedVR2 post-processing loop.

These kernels move the Phase 4 post-processing bottleneck (color correction,
resizing and normalization) off the CPU and onto the GPU. The frame-by-frame
``clamp(-1, 1) -> *0.5 -> +0.5`` normalization and the affine color transfer that
were previously dispatched as several small CUDA ops (each incurring a host-side
launch/sync) are fused into a single elementwise Triton kernel. This keeps the
GPU saturated and prevents the CPU-side stalling that occurs while waiting on
per-frame work.

Design constraints (stability-first):

* Every kernel runs on the *current* CUDA stream. No new Python threads, no
  background workers, and no extra processes are created, so nothing can race
  with the persistent FFMPEG stderr-drain pipe in ``inference_cli.py``.
* Triton is strictly optional. When Triton is unavailable, the tensor is not on
  CUDA, or the dtype/layout is unsupported, every public function transparently
  falls back to the exact PyTorch implementation it replaces. The numerical
  result is identical, so the GGUF inference path remains the master source of
  truth and behaviour never changes on machines without Triton.

The public API intentionally mirrors the in-place PyTorch operations it
replaces so call sites stay a one-line change.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Optional Triton import.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on the runtime environment
    import triton
    import triton.language as tl

    _TRITON_IMPORTED = True
except Exception:  # pragma: no cover - Triton is an optional dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORTED = False


def is_triton_available() -> bool:
    """Return True when fused Triton kernels can actually be dispatched.

    Requires Triton to be importable *and* a CUDA device to be present. On any
    other backend (CPU, MPS) the PyTorch fallbacks are used instead.
    """
    return _TRITON_IMPORTED and torch.cuda.is_available()


def _triton_usable(tensor: torch.Tensor) -> bool:
    """Return True when ``tensor`` is eligible for the fused Triton path."""
    return (
        is_triton_available()
        and tensor.is_cuda
        and tensor.is_floating_point()
        and tensor.is_contiguous()
        and tensor.numel() > 0
    )


# --------------------------------------------------------------------------- #
# Triton kernels (only defined when Triton is importable).
# --------------------------------------------------------------------------- #
if _TRITON_IMPORTED:  # pragma: no cover - exercised only on Triton-capable GPUs

    @triton.jit
    def _fused_affine_normalize_kernel(
        ptr,
        n_elements,
        scale,
        shift,
        clamp_lo,
        clamp_hi,
        DO_CLAMP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """In-place ``out = clamp(x, lo, hi) * scale + shift`` over a flat buffer.

        A single kernel fuses the clamp, multiply and add that the post-processing
        loop previously issued as three separate CUDA ops per frame.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(ptr + offsets, mask=mask)
        if DO_CLAMP:
            x = tl.minimum(tl.maximum(x, clamp_lo), clamp_hi)
        x = x * scale + shift
        tl.store(ptr + offsets, x, mask=mask)

    def _launch_affine_normalize(
        tensor: torch.Tensor,
        scale: float,
        shift: float,
        clamp_lo: Optional[float],
        clamp_hi: Optional[float],
    ) -> torch.Tensor:
        n_elements = tensor.numel()
        do_clamp = clamp_lo is not None and clamp_hi is not None
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _fused_affine_normalize_kernel[grid](
            tensor,
            n_elements,
            float(scale),
            float(shift),
            float(clamp_lo if clamp_lo is not None else 0.0),
            float(clamp_hi if clamp_hi is not None else 0.0),
            DO_CLAMP=do_clamp,
            BLOCK_SIZE=1024,
        )
        return tensor


# --------------------------------------------------------------------------- #
# Public fused operations (Triton when possible, PyTorch fallback otherwise).
# --------------------------------------------------------------------------- #
def fused_affine_(
    tensor: torch.Tensor,
    scale: float,
    shift: float,
    clamp_lo: Optional[float] = None,
    clamp_hi: Optional[float] = None,
) -> torch.Tensor:
    """In-place fused ``clamp(tensor, lo, hi) * scale + shift``.

    Used both for [-1, 1] -> [0, 1] normalization and for affine color
    correction (mean/std transfer). Falls back to the identical PyTorch
    sequence when the fused Triton path is not usable.
    """
    if _triton_usable(tensor):
        try:
            return _launch_affine_normalize(tensor, scale, shift, clamp_lo, clamp_hi)
        except Exception:
            # Any kernel/compile failure must never break inference: fall back.
            pass

    if clamp_lo is not None and clamp_hi is not None:
        tensor.clamp_(clamp_lo, clamp_hi)
    if scale != 1.0:
        tensor.mul_(scale)
    if shift != 0.0:
        tensor.add_(shift)
    return tensor


def fused_normalize_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place ``clamp(-1, 1) * 0.5 + 0.5`` mapping [-1, 1] video to [0, 1].

    Drop-in replacement for ``tensor.clamp_(-1, 1).mul_(0.5).add_(0.5)`` that
    fuses the three ops into one GPU kernel when Triton is available.
    """
    return fused_affine_(tensor, scale=0.5, shift=0.5, clamp_lo=-1.0, clamp_hi=1.0)


def fused_resize(
    tensor: torch.Tensor,
    size: Union[int, Sequence[int]],
    mode: str = "bicubic",
    align_corners: bool = False,
    antialias: bool = True,
) -> torch.Tensor:
    """GPU-resident spatial resize for the post-processing loop.

    ``F.interpolate`` already dispatches to fused CUDA kernels, so resizing is
    kept on the GPU stream rather than round-tripping through CPU/PIL. The
    wrapper exists so the post-processing loop has a single, consistent entry
    point alongside the fused normalize/affine kernels and so ``align_corners``
    is only forwarded for the interpolation modes that accept it.
    """
    interp_kwargs = {"size": size, "mode": mode}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        interp_kwargs["align_corners"] = align_corners
        interp_kwargs["antialias"] = antialias
    return F.interpolate(tensor, **interp_kwargs)
