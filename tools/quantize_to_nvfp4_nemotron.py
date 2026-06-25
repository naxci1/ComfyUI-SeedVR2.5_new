#!/usr/bin/env python3
"""
NVFP4 Quantizer - Nemotron-Aligned
Converts FP16/FP32 models to Native NVFP4 format following NVIDIA Nemotron standards.

Features:
- E2M1 floating point (1 sign, 2 exponent, 1 mantissa bits)
- Microscaling (MX) with block size 16
- torch.float8_e4m3fn for scales (hardware-aligned)
- Nemotron naming: .weight + .weight_scale_inv
- Hardware-aligned packing for Blackwell GPUs
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from safetensors.torch import save_file, load_file


# E2M1 lookup table (1 sign, 2 exp, 1 mantissa)
# Exponent bias = 1, values = sign * (1 + mantissa * 0.5) * 2^(exp - 1)
E2M1_LOOKUP = np.array([
    0.0,   # 0000: +0
    0.5,   # 0001: +2^-1 * 1.0
    0.75,  # 0010: +2^-1 * 1.5
    1.0,   # 0011: +2^0 * 1.0
    1.5,   # 0100: +2^0 * 1.5
    2.0,   # 0101: +2^1 * 1.0
    3.0,   # 0110: +2^1 * 1.5
    4.0,   # 0111: +2^2 * 1.0
    6.0,   # 1000: +2^2 * 1.5
    -0.5,  # 1001: -2^-1 * 1.0
    -0.75, # 1010: -2^-1 * 1.5
    -1.0,  # 1011: -2^0 * 1.0
    -1.5,  # 1100: -2^0 * 1.5
    -2.0,  # 1101: -2^1 * 1.0
    -3.0,  # 1110: -2^1 * 1.5
    -4.0,  # 1111: -2^2 * 1.0
    # Note: -6.0 is not representable in 4-bit E2M1
], dtype=np.float32)


def encode_e2m1_nemotron(values: np.ndarray) -> np.ndarray:
    """
    Encode float32 values to E2M1 4-bit format (Nemotron-aligned).
    
    Args:
        values: Normalized float32 values (should be in range [-6, 6])
        
    Returns:
        uint8 array with 4-bit E2M1 encoded values
    """
    # Find nearest E2M1 value for each input
    abs_values = np.abs(values)
    signs = (values < 0).astype(np.uint8)
    
    # Find nearest positive value
    abs_lookup = E2M1_LOOKUP[:9]  # Positive values only
    diffs = np.abs(abs_values[:, np.newaxis] - abs_lookup[np.newaxis, :])
    indices = np.argmin(diffs, axis=1).astype(np.uint8)
    
    # Apply sign
    e2m1_codes = np.where(signs == 1, indices + 8, indices).astype(np.uint8)
    
    # Handle special case: 0.0 should map to code 0
    e2m1_codes[np.abs(values) < 1e-8] = 0
    
    return e2m1_codes


def pack_nvfp4_hardware_aligned(values: np.ndarray) -> np.ndarray:
    """
    Pack E2M1 4-bit values into uint8 (hardware-aligned for Blackwell).
    
    Packing: (val1 << 4) | (val0 & 0x0F)
    - val0 in lower 4 bits
    - val1 in upper 4 bits
    
    Args:
        values: uint8 array with 4-bit values (only lower 4 bits used)
        
    Returns:
        uint8 array with packed values (length = ceil(len(values) / 2))
    """
    # Ensure even length
    if len(values) % 2 != 0:
        values = np.pad(values, (0, 1), constant_values=0)
    
    # Pack pairs: (val1 << 4) | (val0 & 0x0F)
    val0 = values[0::2] & 0x0F
    val1 = values[1::2] & 0x0F
    packed = (val1 << 4) | val0
    
    return packed.astype(np.uint8)


def calculate_mx_scales(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Calculate MX (Microscaling) scales using FP8 E4M3.
    
    Args:
        tensor: Input tensor (any shape)
        block_size: Size of micro-blocks (default: 16)
        
    Returns:
        FP8 E4M3 inverse scales (1 / scale) for each block
    """
    original_shape = tensor.shape
    numel = tensor.numel()
    
    # Flatten and pad to block boundary
    flat = tensor.flatten()
    padded_len = ((numel + block_size - 1) // block_size) * block_size
    if padded_len > numel:
        flat = torch.cat([flat, torch.zeros(padded_len - numel, 
                                            dtype=tensor.dtype, 
                                            device=tensor.device)])
    
    # Reshape to blocks
    blocks = flat.view(-1, block_size)
    
    # Calculate scale per block (max absolute value)
    block_maxes = torch.max(torch.abs(blocks), dim=1)[0]
    
    # Avoid division by zero
    block_maxes = torch.clamp(block_maxes, min=1e-8)
    
    # Calculate inverse scales (for Nemotron style: weight_scale_inv)
    # Scale factor: max_val / 6.0 (E2M1 max representable value)
    scales = block_maxes / 6.0
    scale_inv = 1.0 / scales
    
    # Convert to FP8 E4M3 (hardware-aligned for Blackwell)
    scale_inv_fp8 = scale_inv.to(torch.float8_e4m3fn)
    
    return scale_inv_fp8


def quantize_tensor_nemotron(tensor: torch.Tensor, 
                             block_size: int = 16) -> Dict[str, torch.Tensor]:
    """
    Quantize tensor to NVFP4 format (Nemotron-aligned).
    
    Args:
        tensor: Input tensor (FP16/FP32)
        block_size: MX block size (default: 16)
        
    Returns:
        Dictionary with 'data' (packed uint8) and 'scale_inv' (FP8 E4M3)
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    
    # Work in FP32 for accuracy
    tensor_fp32 = tensor.float()
    
    # Calculate MX inverse scales
    scale_inv_fp8 = calculate_mx_scales(tensor_fp32, block_size)
    
    # Flatten and prepare for quantization
    flat = tensor_fp32.flatten()
    numel = flat.numel()
    padded_len = ((numel + block_size - 1) // block_size) * block_size
    
    if padded_len > numel:
        flat = torch.cat([flat, torch.zeros(padded_len - numel, device=flat.device)])
    
    # Reshape to blocks
    blocks = flat.view(-1, block_size)
    
    # Convert inverse scales back to regular scales for quantization
    scales = 1.0 / scale_inv_fp8.float()
    scales = scales.view(-1, 1)  # Shape: [num_blocks, 1]
    
    # Normalize blocks by scales
    normalized = blocks / scales
    
    # Encode to E2M1
    normalized_np = normalized.cpu().numpy().flatten()
    e2m1_codes = encode_e2m1_nemotron(normalized_np)
    
    # Pack to uint8 (hardware-aligned)
    packed_data = pack_nvfp4_hardware_aligned(e2m1_codes)
    
    # Convert to torch tensor
    packed_tensor = torch.from_numpy(packed_data)
    
    # Store original shape for reconstruction
    return {
        'data': packed_tensor,
        'scale_inv': scale_inv_fp8,
        'original_shape': original_shape,
        'original_dtype': str(original_dtype)
    }


def quantize_model_nemotron(input_path: str,
                            output_path: str,
                            block_size: int = 16,
                            skip_small: bool = True,
                            min_elements: int = 128) -> None:
    """
    Quantize entire model to Nemotron-aligned NVFP4 format.
    
    Args:
        input_path: Path to input FP16/FP32 safetensors file
        output_path: Path to output NVFP4 safetensors file
        block_size: MX block size (default: 16)
        skip_small: Skip quantizing small tensors
        min_elements: Minimum elements to quantize (if skip_small=True)
    """
    print("=" * 70)
    print("NVFP4 Quantizer - Nemotron-Aligned")
    print("=" * 70)
    print()
    
    # Load input model
    print(f"📂 Loading input model: {input_path}")
    start_time = time.time()
    state_dict = load_file(input_path)
    load_time = time.time() - start_time
    print(f"✅ Loaded in {load_time:.2f}s")
    print()
    
    # Model info
    total_params = sum(t.numel() for t in state_dict.values())
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 ** 2)
    
    print("=" * 70)
    print("Input Model Information")
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Number of tensors: {len(state_dict)}")
    print("=" * 70)
    print()
    
    # Quantize
    print("🔄 Quantizing to NVFP4 (Nemotron format)...")
    print(f"   Block size (MX): {block_size}")
    print(f"   Skip small tensors: {skip_small}")
    if skip_small:
        print(f"   Minimum elements: {min_elements}")
    print()
    
    nvfp4_state_dict = {}
    metadata = {
        "quantization": "nvfp4_nemotron",
        "format": "e2m1",
        "block_size": str(block_size),
        "packing": "hardware_aligned",
        "scale_format": "float8_e4m3fn",
        "naming": "nemotron",
    }
    
    quant_start = time.time()
    quantized_count = 0
    skipped_count = 0
    
    for name, tensor in state_dict.items():
        # Skip if too small
        if skip_small and tensor.numel() < min_elements:
            nvfp4_state_dict[name] = tensor
            skipped_count += 1
            continue
        
        # Quantize
        result = quantize_tensor_nemotron(tensor, block_size)
        
        # Store in Nemotron format
        nvfp4_state_dict[f"{name}"] = result['data']  # Packed NVFP4 data
        nvfp4_state_dict[f"{name}_scale_inv"] = result['scale_inv']  # FP8 inverse scales
        
        # Store metadata for this tensor
        metadata[f"{name}.original_shape"] = str(tuple(result['original_shape']))
        metadata[f"{name}.original_dtype"] = result['original_dtype']
        
        quantized_count += 1
        
        if quantized_count % 100 == 0:
            print(f"  Quantized {quantized_count} tensors...")
    
    quant_time = time.time() - quant_start
    print(f"✅ Quantization completed in {quant_time:.2f}s")
    print(f"   Quantized: {quantized_count} tensors")
    print(f"   Skipped: {skipped_count} tensors")
    print()
    
    # Add metadata
    nvfp4_state_dict["_metadata"] = metadata
    
    # Output info
    output_size_mb = sum(
        t.numel() * t.element_size() for t in nvfp4_state_dict.values()
        if isinstance(t, torch.Tensor)
    ) / (1024 ** 2)
    
    compression_ratio = total_size_mb / output_size_mb if output_size_mb > 0 else 0
    
    print("=" * 70)
    print("Output Model Information")
    print("=" * 70)
    print(f"Total size: {output_size_mb:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved: {total_size_mb - output_size_mb:.2f} MB")
    print("=" * 70)
    print()
    
    # Save
    print(f"💾 Saving NVFP4 model: {output_path}")
    save_start = time.time()
    save_file(nvfp4_state_dict, output_path)
    save_time = time.time() - save_start
    print(f"✅ Saved in {save_time:.2f}s")
    print()
    
    # Verify
    print("🔍 Verifying saved file...")
    saved_state = load_file(output_path)
    print(f"✅ Verification successful")
    print(f"   Saved tensors: {len(saved_state)}")
    print()
    
    print("=" * 70)
    print("✅ NVFP4 Quantization Complete (Nemotron-Aligned)!")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Format: Nemotron-style (.weight + .weight_scale_inv)")
    print(f"Compression: {compression_ratio:.2f}x")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 Quantizer - Nemotron-Aligned for Blackwell GPUs"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input model path (.safetensors, FP16/FP32)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output model path (.safetensors, NVFP4)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="MX block size (default: 16)"
    )
    parser.add_argument(
        "--no-skip-small",
        action="store_true",
        help="Quantize all tensors (don't skip small ones)"
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=128,
        help="Minimum elements to quantize (default: 128)"
    )
    
    args = parser.parse_args()
    
    quantize_model_nemotron(
        input_path=args.input,
        output_path=args.output,
        block_size=args.block_size,
        skip_small=not args.no_skip_small,
        min_elements=args.min_elements
    )


if __name__ == "__main__":
    main()
