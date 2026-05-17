"""Hardware capability detection for NVFP4 support"""
import torch
from typing import Tuple, Dict, Any

def get_gpu_compute_capability() -> Tuple[int, int]:
    """Get CUDA compute capability (major, minor)"""
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()

def check_nvfp4_support() -> bool:
    """
    Check if GPU supports NVFP4 (Blackwell architecture)
    Returns True only for RTX 50 series (compute capability 9.0+)
    """
    capability = get_gpu_compute_capability()
    
    # Blackwell is compute capability 9.0+
    # Use tuple comparison to avoid floating-point precision issues
    return capability >= (9, 0)

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(),
        "compute_capability": get_gpu_compute_capability(),
        "nvfp4_supported": check_nvfp4_support(),
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
    }
