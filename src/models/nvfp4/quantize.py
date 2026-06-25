"""
NVFP4 Quantization Module
Converts FP16/FP32 models to native NVFP4 (E2M1) format with two-level scaling.

Based on NVIDIA's NVFP4 specification:
- 4-bit floating point: [sign:1][exponent:2][mantissa:1]
- Micro-block scaling: 16 values share one FP8 (E4M3) scale
- Tensor-level scaling: Global FP32 scale

Reference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


# E2M1 representable values (exponent bias = 1)
# Format: [sign:1][exp:2][mant:1]
# exp: 00→2^-1, 01→2^0, 10→2^1, 11→2^2
# mant: 0→1.0, 1→1.5
E2M1_VALUES = np.array([
    0.0,   # Special case: zero
    0.5,   # exp=00, mant=0: 2^-1 * 1.0
    0.75,  # exp=00, mant=1: 2^-1 * 1.5
    1.0,   # exp=01, mant=0: 2^0 * 1.0
    1.5,   # exp=01, mant=1: 2^0 * 1.5
    2.0,   # exp=10, mant=0: 2^1 * 1.0
    3.0,   # exp=10, mant=1: 2^1 * 1.5
    4.0,   # exp=11, mant=0: 2^2 * 1.0
    6.0,   # exp=11, mant=1: 2^2 * 1.5
], dtype=np.float32)

# E2M1 encoding lookup (value → 4-bit code)
# Code format: [sign:1][exp:2][mant:1]
E2M1_CODES = {
    0.0:  0b0000,  # +0
    0.5:  0b0000,  # exp=00, mant=0
    0.75: 0b0001,  # exp=00, mant=1
    1.0:  0b0010,  # exp=01, mant=0
    1.5:  0b0011,  # exp=01, mant=1
    2.0:  0b0100,  # exp=10, mant=0
    3.0:  0b0101,  # exp=10, mant=1
    4.0:  0b0110,  # exp=11, mant=0
    6.0:  0b0111,  # exp=11, mant=1
}


def find_nearest_e2m1_value(value: float) -> Tuple[float, int]:
    """
    Find nearest E2M1 representable value.
    
    Args:
        value: Input value (must be positive or zero)
        
    Returns:
        (nearest_value, code): Nearest E2M1 value and its 3-bit code (without sign)
    """
    if value == 0.0:
        return 0.0, 0b000
    
    # Find nearest value in E2M1_VALUES
    idx = np.argmin(np.abs(E2M1_VALUES - value))
    nearest = E2M1_VALUES[idx]
    
    # Get code (without sign bit)
    if nearest == 0.0:
        code = 0b000
    elif nearest == 0.5:
        code = 0b000  # exp=00, mant=0
    elif nearest == 0.75:
        code = 0b001  # exp=00, mant=1
    elif nearest == 1.0:
        code = 0b010  # exp=01, mant=0
    elif nearest == 1.5:
        code = 0b011  # exp=01, mant=1
    elif nearest == 2.0:
        code = 0b100  # exp=10, mant=0
    elif nearest == 3.0:
        code = 0b101  # exp=10, mant=1
    elif nearest == 4.0:
        code = 0b110  # exp=11, mant=0
    elif nearest == 6.0:
        code = 0b111  # exp=11, mant=1
    else:
        raise ValueError(f"Unexpected nearest value: {nearest}")
    
    return nearest, code


def encode_nvfp4_e2m1(values: np.ndarray) -> np.ndarray:
    """
    Encode normalized values to NVFP4 E2M1 format (4-bit).
    
    Args:
        values: Normalized values (should be scaled to fit E2M1 range)
        
    Returns:
        encoded: 4-bit encoded values as uint8 (0-15)
    """
    values = np.asarray(values, dtype=np.float32)
    encoded = np.zeros(values.shape, dtype=np.uint8)
    
    for i, val in enumerate(values.flat):
        # Extract sign
        if val < 0:
            sign = 1
            abs_val = -val
        else:
            sign = 0
            abs_val = val
        
        # Handle special cases
        if abs_val == 0.0 or np.isnan(val):
            encoded.flat[i] = 0b0000
            continue
        
        # Clip to E2M1 range
        if abs_val > 6.0:
            abs_val = 6.0
        
        # Find nearest E2M1 value
        _, code = find_nearest_e2m1_value(abs_val)
        
        # Combine sign and code: [sign:1][exp:2][mant:1]
        encoded.flat[i] = (sign << 3) | code
    
    return encoded


def pack_nvfp4_tensor(values: np.ndarray) -> torch.Tensor:
    """
    Pack 4-bit NVFP4 values into uint8 storage (2 values per byte).
    
    Args:
        values: 4-bit encoded values as uint8 [N]
        
    Returns:
        packed: Packed tensor [N//2] as uint8
    """
    values = values.flatten()
    
    # Pad to even length
    if len(values) % 2 != 0:
        values = np.concatenate([values, np.array([0], dtype=np.uint8)])
    
    # Pack: high nibble = values[0], low nibble = values[1]
    packed = np.zeros(len(values) // 2, dtype=np.uint8)
    packed = (values[::2] << 4) | (values[1::2] & 0x0F)
    
    return torch.from_numpy(packed)


def calculate_optimal_scales(tensor: torch.Tensor, 
                             block_size: int = 16) -> Tuple[torch.Tensor, float]:
    """
    Calculate optimal two-level scales for NVFP4 quantization.
    
    Args:
        tensor: Input tensor to quantize
        block_size: Micro-block size (default: 16)
        
    Returns:
        (fp8_scales, fp32_scale): Micro-block FP8 scales and global FP32 scale
    """
    # Ensure tensor is float32
    tensor = tensor.float()
    
    # Reshape to blocks
    original_shape = tensor.shape
    flat = tensor.flatten()
    
    # Pad to block boundary
    pad_size = (block_size - len(flat) % block_size) % block_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
    
    # Reshape to blocks [num_blocks, block_size]
    blocks = flat.view(-1, block_size)
    
    # Calculate per-block scales (max absolute value)
    block_maxes = blocks.abs().max(dim=1)[0]
    
    # Avoid division by zero
    block_maxes = torch.where(block_maxes == 0, torch.ones_like(block_maxes), block_maxes)
    
    # Normalize by maximum E2M1 value (6.0) to fit in range
    fp8_scales = block_maxes / 6.0
    
    # Calculate global scale (max of all block scales)
    fp32_scale = fp8_scales.max().item()
    
    # Avoid zero scale
    if fp32_scale == 0:
        fp32_scale = 1.0
    
    # Normalize fp8_scales by global scale
    fp8_scales = fp8_scales / fp32_scale
    
    # Convert to FP8 E4M3 (if available, otherwise keep FP16)
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_scales = fp8_scales.to(torch.float8_e4m3fn)
    else:
        # Fallback to FP16 if FP8 not available
        fp8_scales = fp8_scales.to(torch.float16)
    
    return fp8_scales, fp32_scale


def quantize_nvfp4(tensor: torch.Tensor, 
                   block_size: int = 16,
                   return_error: bool = False) -> Dict[str, torch.Tensor]:
    """
    Quantize tensor to NVFP4 format with two-level scaling.
    
    Args:
        tensor: Input tensor (FP16/FP32)
        block_size: Micro-block size (default: 16)
        return_error: If True, include quantization error metrics
        
    Returns:
        Dictionary with:
            - 'nvfp4_data': Packed 4-bit data as uint8
            - 'fp8_scales': Per-block FP8 scales
            - 'fp32_scale': Global FP32 scale
            - 'original_shape': Original tensor shape
            - 'error': Error metrics (if return_error=True)
    """
    original_shape = tensor.shape
    device = tensor.device
    
    # Move to CPU for quantization
    tensor_cpu = tensor.cpu().float()
    
    # Calculate optimal scales
    fp8_scales, fp32_scale = calculate_optimal_scales(tensor_cpu, block_size)
    
    # Flatten and pad
    flat = tensor_cpu.flatten()
    pad_size = (block_size - len(flat) % block_size) % block_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size)])
    
    # Reshape to blocks
    blocks = flat.view(-1, block_size)
    
    # Dequantize fp8_scales to float32 for computation
    if fp8_scales.dtype == torch.float16:
        fp8_scales_float = fp8_scales.float()
    elif hasattr(torch, 'float8_e4m3fn') and fp8_scales.dtype == torch.float8_e4m3fn:
        fp8_scales_float = fp8_scales.float()
    else:
        fp8_scales_float = fp8_scales
    
    # Normalize blocks by two-level scales
    scales_expanded = (fp8_scales_float * fp32_scale).unsqueeze(1)  # [num_blocks, 1]
    normalized = blocks / scales_expanded
    
    # Clip to E2M1 range [-6, 6]
    normalized = torch.clamp(normalized, -6.0, 6.0)
    
    # Encode to E2M1 (4-bit)
    normalized_np = normalized.numpy()
    encoded = encode_nvfp4_e2m1(normalized_np)
    
    # Pack to uint8
    packed = pack_nvfp4_tensor(encoded)
    
    result = {
        'nvfp4_data': packed,
        'fp8_scales': fp8_scales,
        'fp32_scale': fp32_scale,
        'original_shape': original_shape,
    }
    
    # Calculate error if requested
    if return_error:
        # Dequantize to check error
        from .dequantize import dequantize_nvfp4, unpack_nvfp4
        
        unpacked = unpack_nvfp4(packed.numpy())
        dequantized = dequantize_nvfp4(
            unpacked,
            fp8_scales_float.numpy(),
            fp32_scale
        )
        dequantized_tensor = torch.from_numpy(dequantized).view(-1)[:len(tensor_cpu.flatten())]
        dequantized_tensor = dequantized_tensor.view(original_shape)
        
        error = calculate_quantization_error(tensor_cpu, dequantized_tensor)
        result['error'] = error
    
    return result


def calculate_quantization_error(original: torch.Tensor, 
                                 quantized: torch.Tensor) -> Dict[str, float]:
    """
    Calculate quantization error metrics.
    
    Args:
        original: Original tensor
        quantized: Quantized and dequantized tensor
        
    Returns:
        Dictionary with MSE, PSNR, max error, relative error
    """
    original = original.float()
    quantized = quantized.float()
    
    # Mean Squared Error
    mse = torch.mean((original - quantized) ** 2).item()
    
    # Peak Signal-to-Noise Ratio
    if mse > 0:
        max_val = original.abs().max().item()
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Maximum absolute error
    max_error = torch.max(torch.abs(original - quantized)).item()
    
    # Relative error
    original_norm = torch.norm(original).item()
    if original_norm > 0:
        relative_error = torch.norm(original - quantized).item() / original_norm
    else:
        relative_error = 0.0
    
    return {
        'mse': mse,
        'psnr': psnr,
        'max_error': max_error,
        'relative_error': relative_error,
    }


def validate_nvfp4_tensor(packed_data: torch.Tensor,
                         fp8_scales: torch.Tensor,
                         fp32_scale: float,
                         original_shape: tuple,
                         block_size: int = 16) -> bool:
    """
    Validate NVFP4 quantized tensor integrity.
    
    Args:
        packed_data: Packed 4-bit data
        fp8_scales: FP8 micro-block scales
        fp32_scale: FP32 global scale
        original_shape: Original tensor shape
        block_size: Block size used for quantization
        
    Returns:
        True if validation passes
    """
    try:
        # Check shapes
        total_elements = np.prod(original_shape)
        padded_elements = ((total_elements + block_size - 1) // block_size) * block_size
        expected_packed_size = padded_elements // 2
        
        if len(packed_data) != expected_packed_size:
            warnings.warn(f"Packed data size mismatch: {len(packed_data)} != {expected_packed_size}")
            return False
        
        num_blocks = padded_elements // block_size
        if len(fp8_scales) != num_blocks:
            warnings.warn(f"FP8 scales size mismatch: {len(fp8_scales)} != {num_blocks}")
            return False
        
        # Check scale validity
        if fp32_scale <= 0 or not np.isfinite(fp32_scale):
            warnings.warn(f"Invalid FP32 scale: {fp32_scale}")
            return False
        
        # Check FP8 scales
        if hasattr(torch, 'float8_e4m3fn'):
            if fp8_scales.dtype not in [torch.float8_e4m3fn, torch.float16, torch.float32]:
                warnings.warn(f"Invalid FP8 scales dtype: {fp8_scales.dtype}")
                return False
        
        return True
        
    except Exception as e:
        warnings.warn(f"Validation failed: {e}")
        return False


def quantize_model_to_nvfp4(state_dict: Dict[str, torch.Tensor],
                            block_size: int = 16,
                            skip_small_tensors: bool = True,
                            min_elements: int = 128) -> Dict[str, torch.Tensor]:
    """
    Quantize entire model state dict to NVFP4 format.
    
    Args:
        state_dict: Model state dictionary
        block_size: Micro-block size
        skip_small_tensors: Skip tensors smaller than min_elements
        min_elements: Minimum tensor size to quantize
        
    Returns:
        NVFP4 quantized state dictionary
    """
    nvfp4_state = {}
    
    for name, tensor in state_dict.items():
        # Skip non-floating point tensors
        if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            nvfp4_state[name] = tensor
            continue
        
        # Skip small tensors if requested
        if skip_small_tensors and tensor.numel() < min_elements:
            nvfp4_state[name] = tensor
            continue
        
        # Quantize tensor
        print(f"Quantizing {name}: shape={tensor.shape}, elements={tensor.numel()}")
        result = quantize_nvfp4(tensor, block_size=block_size, return_error=True)
        
        # Store quantized data
        nvfp4_state[f"{name}.nvfp4_data"] = result['nvfp4_data']
        nvfp4_state[f"{name}.fp8_scales"] = result['fp8_scales']
        nvfp4_state[f"{name}.fp32_scale"] = torch.tensor(result['fp32_scale'])
        
        # Log error metrics
        if 'error' in result:
            error = result['error']
            print(f"  Error: MSE={error['mse']:.6f}, PSNR={error['psnr']:.2f}dB, "
                  f"MaxErr={error['max_error']:.6f}, RelErr={error['relative_error']:.4f}")
    
    # Add metadata
    nvfp4_state['_metadata'] = {
        'quantization': 'nvfp4',
        'format': 'e2m1',
        'block_size': block_size,
        'two_level_scaling': True,
        'fp8_scale_dtype': 'float8_e4m3fn',
        'fp32_scale_dtype': 'float32',
    }
    
    return nvfp4_state
