"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension

# Run NVFP4 diagnostics on module load
try:
    from .src.utils.startup_diagnostics import print_nvfp4_status
    print_nvfp4_status()
except Exception:
    # Silently continue if diagnostics fail - not critical
    pass

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]