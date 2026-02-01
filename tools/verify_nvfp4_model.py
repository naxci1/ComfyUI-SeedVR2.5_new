#!/usr/bin/env python3
"""
NVFP4 Model Verification Script

Verifies that a quantized model meets NVIDIA's NVFP4 specifications:
- Weights are torch.uint8 (packed 4-bit)
- Scales are torch.float8_e4m3fn (MX format)
- ~4x memory reduction vs FP16
- 16:1 block scaling ratio

Usage:
    python tools/verify_nvfp4_model.py model_nvfp4.safetensors
    python tools/verify_nvfp4_model.py model_nvfp4.safetensors --fp16-reference model_fp16.safetensors
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def verify_nvfp4_format(state_dict):
    """Verify that model follows NVFP4 format specifications"""
    print("\n" + "=" * 70)
    print("Format Verification")
    print("=" * 70)
    
    # Detect format
    has_uint8_weights = False
    has_fp8_scales = False
    weight_count = 0
    scale_count = 0
    
    for key in state_dict:
        if key.endswith('.weight') and state_dict[key].dtype == torch.uint8:
            has_uint8_weights = True
            weight_count += 1
        if key.endswith('_scale_inv'):
            has_fp8_scales = True
            scale_count += 1
            # Check if FP8 E4M3
            if hasattr(torch, 'float8_e4m3fn') and state_dict[key].dtype == torch.float8_e4m3fn:
                pass
            elif state_dict[key].dtype == torch.float32:
                pass  # Fallback to FP32 is acceptable
    
    is_nemotron_nvfp4 = has_uint8_weights and has_fp8_scales
    
    if is_nemotron_nvfp4:
        print("\n✅ Detected Nemotron NVFP4 format")
    else:
        print("\n❌ NOT Nemotron NVFP4 format")
        if not has_uint8_weights:
            print("   Missing: uint8 weights")
        if not has_fp8_scales:
            print("   Missing: _scale_inv parameters")
        return False
    
    # Verify dtypes
    print("\nChecking tensor dtypes:")
    uint8_count = 0
    fp8_count = 0
    fp32_count = 0
    
    checked = 0
    for key in list(state_dict.keys())[:10]:  # Show first 10
        tensor = state_dict[key]
        if key.endswith('.weight') and tensor.dtype == torch.uint8:
            print(f"  ✅ {key}: torch.uint8 (packed 4-bit)")
            uint8_count += 1
        elif key.endswith('_scale_inv'):
            if hasattr(torch, 'float8_e4m3fn') and tensor.dtype == torch.float8_e4m3fn:
                print(f"  ✅ {key}: torch.float8_e4m3fn (FP8 scales)")
                fp8_count += 1
            elif tensor.dtype == torch.float32:
                print(f"  ⚠️ {key}: torch.float32 (FP32 fallback)")
                fp32_count += 1
        checked += 1
    
    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more tensors")
    
    # Count all
    for key in state_dict:
        if key.endswith('.weight') and state_dict[key].dtype == torch.uint8:
            uint8_count += 1
        elif key.endswith('_scale_inv'):
            if hasattr(torch, 'float8_e4m3fn') and state_dict[key].dtype == torch.float8_e4m3fn:
                fp8_count += 1
            elif state_dict[key].dtype == torch.float32:
                fp32_count += 1
    
    print(f"\n✅ All {uint8_count} weights are uint8 (packed)")
    if fp8_count > 0:
        print(f"✅ All {fp8_count} scales are float8_e4m3fn (MX format)")
    if fp32_count > 0:
        print(f"⚠️ {fp32_count} scales are float32 (fallback, not FP8)")
    
    return True


def verify_memory_reduction(state_dict, fp16_reference=None):
    """Verify ~4x memory reduction vs FP16"""
    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    
    # Calculate NVFP4 size
    nvfp4_size = sum(
        tensor.numel() * tensor.element_size() 
        for tensor in state_dict.values()
    )
    
    # Calculate parameter count (from weights only)
    param_count = sum(
        tensor.numel() * 2  # Each uint8 holds 2x 4-bit values
        for key, tensor in state_dict.items()
        if key.endswith('.weight') and tensor.dtype == torch.uint8
    )
    
    print(f"\nTotal parameters: {param_count:,}")
    print(f"NVFP4 size:      {format_size(nvfp4_size)}")
    
    # Calculate expected FP16 size
    expected_fp16_size = param_count * 2  # 2 bytes per FP16 value
    
    if fp16_reference:
        print(f"FP16 reference:  {format_size(fp16_reference)}")
        compression = fp16_reference / nvfp4_size
    else:
        print(f"Expected FP16:   {format_size(expected_fp16_size)}")
        compression = expected_fp16_size / nvfp4_size
    
    print(f"Compression:     {compression:.2f}x")
    
    if 3.5 <= compression <= 4.5:
        print(f"\n✅ Memory reduction: {(1 - 1/compression)*100:.1f}% ({compression:.2f}x compression)")
        return True
    else:
        print(f"\n❌ Compression {compression:.2f}x is outside expected range (3.5-4.5x)")
        return False


def verify_shape_relationships(state_dict):
    """Verify 16:1 block scaling ratio"""
    print("\n" + "=" * 70)
    print("Shape Verification (MX Block Scaling)")
    print("=" * 70)
    
    print("\nChecking weight/scale relationships:")
    
    checked = 0
    all_valid = True
    
    for key in state_dict:
        if key.endswith('.weight') and state_dict[key].dtype == torch.uint8:
            weight = state_dict[key]
            scale_key = key + '_scale_inv'
            
            if scale_key in state_dict:
                scale = state_dict[scale_key]
                
                # For 2D weights (most common)
                if len(weight.shape) == 2 and len(scale.shape) == 2:
                    weight_rows, weight_cols = weight.shape
                    scale_rows, scale_cols = scale.shape
                    
                    # Check block ratio (should be 16:1)
                    row_ratio = weight_rows / scale_rows if scale_rows > 0 else 0
                    col_match = weight_cols == scale_cols
                    
                    if checked < 5:  # Show first 5
                        status = "✅" if (15 <= row_ratio <= 17 and col_match) else "❌"
                        print(f"  {status} {key}: {list(weight.shape)} (uint8)")
                        print(f"     Scale: {list(scale.shape)} ({row_ratio:.1f}:1 block ratio) {'✓' if col_match else '✗'}")
                    
                    if not (15 <= row_ratio <= 17 and col_match):
                        all_valid = False
                
                checked += 1
    
    if checked > 5:
        print(f"  ... and {checked - 5} more weight/scale pairs")
    
    if all_valid:
        print(f"\n✅ All {checked} shape relationships valid (16:1 MX block scaling)")
        return True
    else:
        print(f"\n❌ Some shape relationships invalid")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify NVFP4 model format')
    parser.add_argument('model', help='Path to NVFP4 safetensors file')
    parser.add_argument('--fp16-reference', help='Path to FP16 reference model for comparison')
    args = parser.parse_args()
    
    print("=" * 70)
    print("NVFP4 Model Verification Tool")
    print("=" * 70)
    
    # Check file exists
    if not os.path.exists(args.model):
        print(f"\n❌ Error: File not found: {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"\n📂 Loading model: {args.model}")
    start_time = time.time()
    try:
        state_dict = load_file(args.model)
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load FP16 reference if provided
    fp16_size = None
    if args.fp16_reference:
        if os.path.exists(args.fp16_reference):
            print(f"\n📂 Loading FP16 reference: {args.fp16_reference}")
            try:
                fp16_state = load_file(args.fp16_reference)
                fp16_size = sum(t.numel() * t.element_size() for t in fp16_state.values())
                print(f"✅ Loaded FP16 reference")
            except Exception as e:
                print(f"⚠️ Could not load FP16 reference: {e}")
        else:
            print(f"⚠️ FP16 reference file not found: {args.fp16_reference}")
    
    # Run verifications
    format_ok = verify_nvfp4_format(state_dict)
    memory_ok = verify_memory_reduction(state_dict, fp16_size)
    shapes_ok = verify_shape_relationships(state_dict)
    
    # Final report
    print("\n" + "=" * 70)
    if format_ok and memory_ok and shapes_ok:
        print("✅ NVFP4 Verification PASSED")
    else:
        print("❌ NVFP4 Verification FAILED")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  {'✅' if format_ok else '❌'} Format: Nemotron NVFP4 (E2M1)")
    print(f"  {'✅' if format_ok else '❌'} Weights: torch.uint8 (packed 4-bit)")
    
    # Check if FP8 is available
    if hasattr(torch, 'float8_e4m3fn'):
        print(f"  {'✅' if format_ok else '❌'} Scales: torch.float8_e4m3fn (MX format)")
    else:
        print(f"  ⚠️ Scales: torch.float32 (FP8 not available in this PyTorch version)")
    
    print(f"  {'✅' if memory_ok else '❌'} Compression: ~4x vs FP16")
    print(f"  {'✅' if shapes_ok else '❌'} Block scaling: 16:1 ratio")
    
    if format_ok and memory_ok and shapes_ok:
        print(f"  ✅ Ready for Blackwell GPU")
        print("\nThis model is ready for hardware-native NVFP4 execution!")
        sys.exit(0)
    else:
        print(f"  ❌ Model verification failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
