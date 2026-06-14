"""
Intel XPU (Arc GPU / Data Center Max) compatibility utilities.

Intel GPUs use the ``torch.xpu.*`` API, which is distinct from ``torch.cuda.*``.
These helpers wrap the XPU API surface defensively so the rest of the codebase
can detect and use Intel GPUs without importing oneAPI/IPEX directly. The
optional Intel Extension for PyTorch (IPEX) import is wrapped in try/except so it
remains a soft dependency.
"""

from __future__ import annotations

import os


def is_xpu_available() -> bool:
    """Check whether an Intel XPU (oneAPI/SYCL) backend is available."""
    try:
        import torch

        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False


def get_device_count() -> int:
    """Return the number of Intel XPU devices (0 if unavailable)."""
    try:
        import torch

        if is_xpu_available():
            return torch.xpu.device_count()
    except Exception:
        pass
    return 0


def get_device_name(device_id: int = 0) -> str:
    """Return the Intel XPU device name, or a placeholder on failure."""
    try:
        import torch

        return torch.xpu.get_device_name(device_id)
    except Exception:
        return "Unknown Intel XPU"


def empty_cache() -> None:
    """Clear the Intel XPU memory cache. Never raises."""
    try:
        import torch

        if is_xpu_available():
            torch.xpu.empty_cache()
    except Exception:
        pass


def memory_allocated(device_id: int = 0) -> int:
    """Return the current XPU memory allocation in bytes (0 on failure)."""
    try:
        import torch

        return torch.xpu.memory_allocated(device_id)
    except Exception:
        return 0


def apply_xpu_env() -> None:
    """Apply Intel XPU-specific environment variables (idempotent)."""
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1")


def optimize_model_for_xpu(model, dtype=None):
    """Apply Intel IPEX optimizations to a model for XPU inference.

    Returns the model unchanged when ``intel_extension_for_pytorch`` is not
    installed (IPEX is an optional dependency).
    """
    try:
        import intel_extension_for_pytorch as ipex

        if dtype is None:
            import torch

            dtype = torch.bfloat16
        return ipex.optimize(model, dtype=dtype)
    except ImportError:
        return model
