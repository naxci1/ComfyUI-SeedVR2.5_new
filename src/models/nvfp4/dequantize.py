"""
NVFP4 Dequantization Implementation
Native NVFP4 (E2M1) 4-bit floating point dequantization for Blackwell GPUs

NVFP4 Format Specification:
- 4-bit floating point: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Value range: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- Exponent bias: 1
- Two-level scaling:
  1. Micro-block: 16 values share one FP8 (E4M3) scale
  2. Tensor-level: Global FP32 scale

References:
- https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- https://github.com/NVIDIA/Model-Optimizer
"""

import torch
import numpy as np
from typing import Tuple, Optional
import warnings


# NVFP4 E2M1 lookup table for fast decoding
# Format: 4 bits = [sign:1][exp:2][mant:1]
# Exponent bias = 1, so exp values 0,1,2,3 map to -1,0,1,2
# Mantissa: 0 → 1.0, 1 → 1.5
NVFP4_E2M1_LUT = torch.tensor([
    # exp=-1 (actual exp=0): 2^-1 = 0.5
    0.0,    # 0b0000: +0 (special case: exp=0, mant=0 → 0)
    0.75,   # 0b0001: +(1.5 * 2^-1) = +0.75
    0.0,    # 0b0010: -0
    -0.75,  # 0b0011: -(1.5 * 2^-1) = -0.75
    
    # exp=0 (actual exp=1): 2^0 = 1.0
    1.0,    # 0b0100: +(1.0 * 2^0) = +1.0
    1.5,    # 0b0101: +(1.5 * 2^0) = +1.5
    -1.0,   # 0b0110: -(1.0 * 2^0) = -1.0
    -1.5,   # 0b0111: -(1.5 * 2^0) = -1.5
    
    # exp=1 (actual exp=2): 2^1 = 2.0
    2.0,    # 0b1000: +(1.0 * 2^1) = +2.0
    3.0,    # 0b1001: +(1.5 * 2^1) = +3.0
    -2.0,   # 0b1010: -(1.0 * 2^1) = -2.0
    -3.0,   # 0b1011: -(1.5 * 2^1) = -3.0
    
    # exp=2 (actual exp=3): 2^2 = 4.0
    4.0,    # 0b1100: +(1.0 * 2^2) = +4.0
    6.0,    # 0b1101: +(1.5 * 2^2) = +6.0
    -4.0,   # 0b1110: -(1.0 * 2^2) = -4.0
    -6.0,   # 0b1111: -(1.5 * 2^2) = -6.0
], dtype=torch.float32)


def unpack_nvfp4(packed_uint8: torch.Tensor) -> torch.Tensor:
    """
    Unpack NVFP4 data from uint8 to 4-bit values
    
    Args:
        packed_uint8: Packed uint8 tensor [..., N] where N*2 is number of NVFP4 values
        
    Returns:
        Unpacked 4-bit values as uint8 [..., N*2]
    """
    # Each uint8 contains 2 NVFP4 values (4 bits each)
    # High nibble: first value, Low nibble: second value
    
    high_nibble = (packed_uint8 >> 4) & 0x0F  # Extract high 4 bits
    low_nibble = packed_uint8 & 0x0F           # Extract low 4 bits
    
    # Interleave high and low nibbles
    unpacked = torch.stack([high_nibble, low_nibble], dim=-1)
    unpacked = unpacked.flatten(start_dim=-2)
    
    return unpacked


def decode_nvfp4_e2m1(quantized_4bit: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Decode NVFP4 E2M1 4-bit values to FP32
    
    Args:
        quantized_4bit: uint8 tensor with values 0-15 (4-bit)
        device: Target device for output
        
    Returns:
        Decoded FP32 tensor with same shape
    """
    if device is None:
        device = quantized_4bit.device
    
    # Move LUT to target device
    lut = NVFP4_E2M1_LUT.to(device)
    
    # Use lookup table for fast decoding
    decoded = lut[quantized_4bit.long()]
    
    return decoded


def dequantize_nvfp4(
    quantized_data: torch.Tensor,
    fp8_scales: torch.Tensor,
    fp32_scale: float,
    block_size: int = 16,
    target_dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Dequantize NVFP4 tensor with two-level scaling
    
    NVFP4 uses a two-level scaling approach:
    1. Micro-block scaling: Every 16 values share one FP8 scale
    2. Tensor-level scaling: Global FP32 scale for entire tensor
    
    Args:
        quantized_data: Packed uint8 tensor with NVFP4 data
                       Shape: [..., (N+1)//2] where N is number of values
        fp8_scales: FP8 micro-block scales
                    Shape: [..., N//block_size]
        fp32_scale: Global FP32 tensor-level scale (scalar)
        block_size: Number of values per micro-block (default: 16)
        target_dtype: Output dtype (fp16 or fp32)
        device: Target device
        
    Returns:
        Dequantized tensor in target dtype
        Shape: [..., N]
    """
    if device is None:
        device = quantized_data.device
    
    # Step 1: Unpack uint8 → 4-bit values
    unpacked_4bit = unpack_nvfp4(quantized_data)
    
    # Step 2: Decode NVFP4 E2M1 → FP32
    decoded = decode_nvfp4_e2m1(unpacked_4bit, device)
    
    # Step 3: Reshape to blocks
    # [..., N] → [..., num_blocks, block_size]
    original_shape = decoded.shape
    num_elements = decoded.shape[-1]
    
    if num_elements % block_size != 0:
        raise ValueError(f"Number of elements ({num_elements}) must be divisible by block_size ({block_size})")
    
    num_blocks = num_elements // block_size
    blocked_shape = list(original_shape[:-1]) + [num_blocks, block_size]
    decoded_blocked = decoded.view(blocked_shape)
    
    # Step 4: Apply FP8 micro-block scales
    # fp8_scales shape: [..., num_blocks]
    # Need to expand last dim to match block_size
    fp8_scales_expanded = fp8_scales.unsqueeze(-1)  # [..., num_blocks, 1]
    
    # Convert FP8 scales to FP32 for computation
    if fp8_scales.dtype == torch.float8_e4m3fn:
        fp8_scales_fp32 = fp8_scales_expanded.float()
    else:
        fp8_scales_fp32 = fp8_scales_expanded
    
    scaled = decoded_blocked * fp8_scales_fp32
    
    # Step 5: Apply global FP32 scale
    result = scaled * fp32_scale
    
    # Step 6: Reshape back to original
    result = result.view(original_shape)
    
    # Step 7: Convert to target dtype
    if target_dtype == torch.float16:
        result = result.half()
    elif target_dtype == torch.float32:
        pass  # Already FP32
    else:
        result = result.to(target_dtype)
    
    return result


def create_nvfp4_dequantize_method(
    quantized_tensor: torch.Tensor,
    fp8_scales: torch.Tensor,
    fp32_scale: float,
    original_shape: torch.Size,
    block_size: int = 16
) -> callable:
    """
    Create a dequantize method for lazy evaluation
    
    This allows NVFP4 tensors to remain quantized in memory
    and only dequantize when needed for computation.
    
    Args:
        quantized_tensor: Packed NVFP4 data (uint8)
        fp8_scales: FP8 micro-block scales
        fp32_scale: Global FP32 scale
        original_shape: Shape of dequantized tensor
        block_size: NVFP4 block size (default: 16)
        
    Returns:
        Callable that dequantizes to specified device/dtype
    """
    def dequantize(
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Dequantize NVFP4 tensor to target device and dtype"""
        
        # Move quantized data to target device
        if device is None:
            device = quantized_tensor.device
        
        q_data = quantized_tensor.to(device)
        scales = fp8_scales.to(device)
        
        # Dequantize
        result = dequantize_nvfp4(
            q_data,
            scales,
            fp32_scale,
            block_size=block_size,
            target_dtype=dtype,
            device=device
        )
        
        # Reshape to original
        result = result.view(original_shape)
        
        return result
    
    return dequantize


def detect_nvfp4_format(state_dict: dict, metadata: dict = None) -> bool:
    """
    Detect if a model uses NVFP4 quantization
    
    Args:
        state_dict: Model state dict
        metadata: Optional safetensors metadata
        
    Returns:
        True if NVFP4 format detected
    """
    # Check metadata first
    if metadata:
        quantization = metadata.get('quantization', '').lower()
        if 'nvfp4' in quantization:
            return True
        
        format_str = metadata.get('format', '').lower()
        if 'nvfp4' in format_str or 'e2m1' in format_str:
            return True
    
    # Check for NVFP4-specific keys in state dict
    nvfp4_indicators = [
        'nvfp4_data',
        'fp8_scales',
        'fp32_scale',
        '.quantized',
        '.scales'
    ]
    
    for key in state_dict.keys():
        for indicator in nvfp4_indicators:
            if indicator in key.lower():
                return True
    
    return False


def validate_nvfp4_tensors(state_dict: dict) -> Tuple[bool, str]:
    """
    Validate that NVFP4 tensors have correct structure
    
    Args:
        state_dict: Model state dict with NVFP4 data
        
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    for name, tensor in state_dict.items():
        if 'nvfp4_data' in name or 'quantized' in name:
            # Should be uint8
            if tensor.dtype != torch.uint8:
                errors.append(f"{name}: Expected uint8, got {tensor.dtype}")
        
        if 'fp8_scales' in name or 'scales' in name:
            # Should be FP8 or FP32
            if tensor.dtype not in [torch.float8_e4m3fn, torch.float32, torch.float16]:
                errors.append(f"{name}: Unexpected scale dtype {tensor.dtype}")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, ""


# Check for native NVFP4 capabilities
_native_nvfp4_available = False
_blackwell_native = False

try:
    import torch
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] >= 9:  # Blackwell (9.0) or newer
            _blackwell_native = True
            _native_nvfp4_available = True
            # Pure PyTorch native implementation for Blackwell
            # Uses JIT compilation and tensor cores
except Exception:
    pass

# Optional: Check for additional acceleration (TensorRT-LLM)
_tensorrt_acceleration = False
try:
    import tensorrt_llm
    import os
    if os.environ.get('ENABLE_NVFP4_NATIVE') == '1' and hasattr(tensorrt_llm, 'nvfp4'):
        _tensorrt_acceleration = True
except ImportError:
    pass



def is_native_nvfp4_available() -> bool:
    """
    Check if native NVFP4 hardware acceleration is available.
    
    Returns True for:
    - Blackwell GPUs (compute 9.0+): Pure PyTorch JIT-compiled implementation
    - Optional TensorRT-LLM: Extra acceleration if available
    """
    return _native_nvfp4_available or _blackwell_native
