"""
AMD ROCm compatibility utilities.

ROCm's PyTorch backend exposes the ``torch.cuda.*`` API surface, but certain
operations are unavailable or behave differently (notably ``ipc_collect`` and
the ``expandable_segments`` allocator option). These helpers centralize the
ROCm-specific detection and environment setup so the rest of the codebase can
stay backend-agnostic.
"""

from __future__ import annotations

import os


# AMD GPU marketing-name → ROCm GFX target mapping.
_GFX_MAPPING = {
    # RDNA 4 (RX 9000 series)
    "9070 xt": "gfx1201",
    "9070 gre": "gfx1201",
    "9070": "gfx1201",
    "9060 xt": "gfx1201",
    "9060": "gfx1201",

    # RDNA 3 (RX 7000 series)
    "7900 xtx": "gfx1100",
    "7900 xt": "gfx1100",
    "7900 gre": "gfx1100",
    "7800 xt": "gfx1101",
    "7700 xt": "gfx1101",
    "7600": "gfx1102",

    # RDNA 2 (RX 6000 series)
    "6900 xt": "gfx1030",
    "6800 xt": "gfx1030",
    "6800": "gfx1030",
    "6700 xt": "gfx1031",
    "6600 xt": "gfx1031",
    "6600": "gfx1032",

    # CDNA (Instinct accelerators)
    "mi300x": "gfx942",
    "mi300a": "gfx942",
    "mi250": "gfx90a",
    "mi210": "gfx90a",
    "mi100": "gfx908",
}


def is_rocm_backend() -> bool:
    """Detect whether PyTorch is running on the ROCm/HIP backend."""
    try:
        import torch

        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except (ImportError, AttributeError):
        return False


def get_rocm_gfx_version(gpu_name: str) -> str:
    """Map an AMD GPU name to its ROCm GFX version string.

    Defaults to ``gfx1100`` (latest RDNA) when the name is unrecognized.
    """
    gpu_lower = (gpu_name or "").lower()
    for pattern, gfx in _GFX_MAPPING.items():
        if pattern in gpu_lower:
            return gfx
    return "gfx1100"


def apply_rocm_env(gpu_name: str = "") -> None:
    """Apply ROCm-specific environment variables (idempotent)."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", get_rocm_gfx_version(gpu_name))
    # ROCm may not support cudaMallocAsync/expandable_segments combinations
    # reliably; only set a safe default when the user hasn't configured it.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def rocm_safe_empty_cache() -> None:
    """Empty the CUDA/HIP cache with ROCm-safe guards. Never raises."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def rocm_safe_ipc_collect() -> None:
    """Run ``ipc_collect`` with a ROCm guard — it can raise ``RuntimeError`` on HIP."""
    try:
        import torch

        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except (RuntimeError, AttributeError, ImportError):
        pass
