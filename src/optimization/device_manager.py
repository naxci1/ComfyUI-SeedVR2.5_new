"""
Unified device management abstraction.

Provides a single API that works across CUDA (NVIDIA), ROCm (AMD, which uses the
``torch.cuda.*`` API surface), Intel XPU, Apple MPS, and CPU. Every helper is
defensive: torch is imported lazily and all backend probes are wrapped so that
calling code never has to special-case the active platform.
"""

from __future__ import annotations

import gc
from enum import Enum
from typing import Tuple


class Backend(Enum):
    """Supported compute backends."""

    CUDA = "cuda"   # Also covers AMD ROCm (uses the torch.cuda.* API)
    XPU = "xpu"     # Intel Arc / Data Center Max
    MPS = "mps"     # Apple Silicon
    CPU = "cpu"


def detect_backend() -> Backend:
    """Auto-detect the best available compute backend."""
    try:
        import torch

        # Check XPU first (distinct API surface from CUDA).
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return Backend.XPU
        # CUDA covers both NVIDIA and AMD ROCm.
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return Backend.CUDA
        # Apple MPS.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Backend.MPS
    except (ImportError, AttributeError):
        pass
    return Backend.CPU


def get_device_string(device_id: str = "0") -> str:
    """Convert a device ID to a platform-appropriate device string."""
    backend = detect_backend()
    if backend == Backend.XPU:
        return f"xpu:{device_id}"
    if backend == Backend.MPS:
        return "mps"
    if backend == Backend.CUDA:
        return f"cuda:{device_id}"
    return "cpu"


def get_device(device_id: str = "0"):
    """Return a ``torch.device`` for the current backend."""
    import torch

    return torch.device(get_device_string(device_id))


def empty_cache() -> None:
    """Platform-agnostic memory cache flush. Never raises."""
    backend = detect_backend()
    try:
        import torch

        if backend == Backend.CUDA:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif backend == Backend.XPU:
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
        elif backend == Backend.MPS:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass


def memory_info(device_id: int = 0) -> Tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the current GPU.

    Returns ``(0, 0)`` on CPU or when the backend does not expose memory info.
    """
    backend = detect_backend()
    try:
        import torch

        if backend == Backend.CUDA and torch.cuda.is_available():
            return torch.cuda.mem_get_info(device_id)
        if backend == Backend.XPU:
            # XPU has no mem_get_info; derive free from total - allocated.
            total = torch.xpu.get_device_properties(device_id).total_memory
            allocated = torch.xpu.memory_allocated(device_id)
            return (total - allocated, total)
    except Exception:
        pass
    return (0, 0)


def gc_and_flush() -> None:
    """Full Python garbage collection followed by a platform cache flush."""
    gc.collect()
    empty_cache()
