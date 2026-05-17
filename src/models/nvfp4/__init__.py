"""
NVFP4 inference support for Blackwell GPUs
Provides native NVFP4 dequantization and tensor handling

Native NVFP4 (E2M1) 4-bit floating point format:
- Block size: 16 values per FP8 scale
- Two-level scaling: FP8 micro-block + FP32 tensor
- Value range: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- Compression: ~4x vs FP16, ~2x vs FP8
- Performance: 2-3x faster on Blackwell with native ops
"""

# Import dequantization functions
from .dequantize import (
    decode_nvfp4_e2m1,
    dequantize_nvfp4,
    create_nvfp4_dequantize_method,
    detect_nvfp4_format,
    validate_nvfp4_tensors,
    is_native_nvfp4_available,
    NVFP4_E2M1_LUT,
)

# Import tensor wrapper
from .tensor import (
    NVFP4Tensor,
    wrap_nvfp4_parameters,
    unwrap_nvfp4_parameters,
)

import torch
from typing import Optional
import warnings

__all__ = [
    # Dequantization
    'decode_nvfp4_e2m1',
    'dequantize_nvfp4',
    'create_nvfp4_dequantize_method',
    'detect_nvfp4_format',
    'validate_nvfp4_tensors',
    'is_native_nvfp4_available',
    'NVFP4_E2M1_LUT',
    
    # Tensor wrapper
    'NVFP4Tensor',
    'wrap_nvfp4_parameters',
    'unwrap_nvfp4_parameters',
    
    # Legacy API
    'load_nvfp4_kernels',
    'NVFP4ModelLoader',
]

_nvfp4_kernels_loaded = False
_nvfp4_available = False


def load_nvfp4_kernels() -> bool:
    """
    Load NVFP4 CUDA kernels for Blackwell GPUs
    Returns True if native NVFP4 ops are available
    
    This checks for:
    1. NVIDIA Model Optimizer (modelopt) with NVFP4 support
    2. TensorRT-LLM with NVFP4 ops
    3. Custom NVFP4 CUDA extensions
    
    If none available, falls back to software implementation.
    """
    global _nvfp4_kernels_loaded, _nvfp4_available
    
    if _nvfp4_kernels_loaded:
        return _nvfp4_available
    
    # Check using dequantize module
    _nvfp4_available = is_native_nvfp4_available()
    _nvfp4_kernels_loaded = True
    
    return _nvfp4_available


class NVFP4ModelLoader:
    """
    Loader for NVFP4 quantized models with automatic backend selection
    
    Supports two loading modes:
    1. Native: Uses TensorRT-LLM/ModelOpt for hardware acceleration
    2. Software: Uses pure PyTorch implementation for compatibility
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.nvfp4_native = load_nvfp4_kernels()
        
    def load(self):
        """Load model with appropriate backend"""
        import safetensors.torch
        
        print(f"[NVFP4] Loading model: {self.model_path}")
        
        # Load safetensors file
        state_dict = safetensors.torch.load_file(self.model_path, device=self.device)
        
        # Detect NVFP4 format
        metadata = {}  # TODO: Extract metadata from safetensors
        is_nvfp4 = detect_nvfp4_format(state_dict, metadata)
        
        if not is_nvfp4:
            print("[NVFP4] ⚠️ Model doesn't appear to be NVFP4 format")
            return state_dict
        
        print(f"[NVFP4] ✅ NVFP4 format detected")
        
        # Validate NVFP4 structure
        valid, error = validate_nvfp4_tensors(state_dict)
        if not valid:
            warnings.warn(f"[NVFP4] ⚠️ NVFP4 validation failed: {error}")
        
        if self.nvfp4_native:
            print("[NVFP4] Using native NVFP4 acceleration (Blackwell)")
            return self._load_native(state_dict)
        else:
            print("[NVFP4] Using software NVFP4 implementation")
            return self._load_software(state_dict)
    
    def _load_native(self, state_dict):
        """Load with native NVFP4 ops (Blackwell hardware acceleration)"""
        # TODO: Implement native NVFP4 tensor creation with TensorRT-LLM
        # For now, use software implementation
        print("[NVFP4] ℹ️ Native NVFP4 ops detected but not yet integrated, using software")
        return self._load_software(state_dict)
    
    def _load_software(self, state_dict):
        """Load with software NVFP4 implementation (pure PyTorch)"""
        # Wrap NVFP4 parameters
        wrapped = wrap_nvfp4_parameters(state_dict, block_size=16)
        
        print(f"[NVFP4] ✅ Wrapped {len([k for k, v in wrapped.items() if isinstance(v, NVFP4Tensor)])} NVFP4 parameters")
        
        return wrapped
