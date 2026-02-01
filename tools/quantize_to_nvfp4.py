#!/usr/bin/env python3
"""
NVFP4 Model Quantization Tool
Converts FP16/FP32 PyTorch models to native NVFP4 format.

Usage:
    python tools/quantize_to_nvfp4.py \
        --input seedvr2_ema_3b_fp16.safetensors \
        --output seedvr2_ema_3b_nvfp4.safetensors \
        --block-size 16

Requirements:
    - torch >= 2.0.0
    - safetensors >= 0.3.0
"""

import argparse
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import save_file, load_file
from src.models.nvfp4.quantize import quantize_model_to_nvfp4, calculate_quantization_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FP16/FP32 model to NVFP4 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert SeedVR2 FP16 model to NVFP4
  python tools/quantize_to_nvfp4.py \\
      --input models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors \\
      --output models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors
  
  # Convert with custom block size
  python tools/quantize_to_nvfp4.py \\
      --input model_fp16.safetensors \\
      --output model_nvfp4.safetensors \\
      --block-size 32
  
  # Quantize all tensors including small ones
  python tools/quantize_to_nvfp4.py \\
      --input model_fp16.safetensors \\
      --output model_nvfp4.safetensors \\
      --no-skip-small
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input model file (.safetensors)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output NVFP4 model file (.safetensors)"
    )
    
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Micro-block size for NVFP4 quantization (default: 16)"
    )
    
    parser.add_argument(
        "--no-skip-small",
        action="store_true",
        help="Quantize all tensors including small ones (<128 elements)"
    )
    
    parser.add_argument(
        "--min-elements",
        type=int,
        default=128,
        help="Minimum tensor size to quantize (default: 128)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for quantization (default: cpu)"
    )
    
    return parser.parse_args()


def print_model_info(state_dict: dict, name: str):
    """Print model information."""
    print(f"\n{'='*70}")
    print(f"{name} Model Information")
    print(f"{'='*70}")
    
    total_params = 0
    total_size_mb = 0
    dtype_counts = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            num_params = tensor.numel()
            size_mb = tensor.element_size() * num_params / (1024 * 1024)
            
            total_params += num_params
            total_size_mb += size_mb
            
            dtype = str(tensor.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Number of tensors: {len([k for k in state_dict.keys() if isinstance(state_dict[k], torch.Tensor)])}")
    print(f"Dtype distribution:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count} tensors")
    print(f"{'='*70}\n")


def calculate_compression_ratio(input_state: dict, output_state: dict) -> dict:
    """Calculate compression statistics."""
    
    # Calculate input size
    input_size = 0
    for tensor in input_state.values():
        if isinstance(tensor, torch.Tensor):
            input_size += tensor.element_size() * tensor.numel()
    
    # Calculate output size
    output_size = 0
    for key, tensor in output_state.items():
        if isinstance(tensor, torch.Tensor):
            output_size += tensor.element_size() * tensor.numel()
    
    compression_ratio = input_size / output_size if output_size > 0 else 0
    
    return {
        'input_size_mb': input_size / (1024 * 1024),
        'output_size_mb': output_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'space_saved_mb': (input_size - output_size) / (1024 * 1024),
        'space_saved_percent': ((input_size - output_size) / input_size * 100) if input_size > 0 else 0,
    }


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("NVFP4 Model Quantization Tool")
    print("="*70)
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Load input model
    print(f"\n📂 Loading input model: {args.input}")
    try:
        start_time = time.time()
        input_state = load_file(args.input)
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Error loading input model: {e}")
        return 1
    
    # Print input model info
    print_model_info(input_state, "Input")
    
    # Quantize to NVFP4
    print(f"\n🔄 Quantizing to NVFP4 format...")
    print(f"   Block size: {args.block_size}")
    print(f"   Skip small tensors: {not args.no_skip_small}")
    print(f"   Minimum elements: {args.min_elements}")
    print()
    
    try:
        start_time = time.time()
        nvfp4_state = quantize_model_to_nvfp4(
            input_state,
            block_size=args.block_size,
            skip_small_tensors=not args.no_skip_small,
            min_elements=args.min_elements
        )
        quant_time = time.time() - start_time
        print(f"\n✅ Quantization completed in {quant_time:.2f}s")
    except Exception as e:
        print(f"\n❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print output model info
    print_model_info(nvfp4_state, "Output (NVFP4)")
    
    # Calculate compression statistics
    stats = calculate_compression_ratio(input_state, nvfp4_state)
    print(f"\n{'='*70}")
    print("Compression Statistics")
    print(f"{'='*70}")
    print(f"Input size:         {stats['input_size_mb']:.2f} MB")
    print(f"Output size:        {stats['output_size_mb']:.2f} MB")
    print(f"Compression ratio:  {stats['compression_ratio']:.2f}x")
    print(f"Space saved:        {stats['space_saved_mb']:.2f} MB ({stats['space_saved_percent']:.1f}%)")
    print(f"{'='*70}\n")
    
    # Save output model
    print(f"💾 Saving NVFP4 model: {args.output}")
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        start_time = time.time()
        save_file(nvfp4_state, args.output)
        save_time = time.time() - start_time
        print(f"✅ Saved in {save_time:.2f}s")
    except Exception as e:
        print(f"❌ Error saving output model: {e}")
        return 1
    
    # Verify saved file
    print(f"\n🔍 Verifying saved file...")
    try:
        verified_state = load_file(args.output)
        print(f"✅ Verification successful")
        print(f"   Saved file size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ NVFP4 Quantization Complete!")
    print(f"{'='*70}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Compression: {stats['compression_ratio']:.2f}x ({stats['space_saved_percent']:.1f}% reduction)")
    print(f"Total time: {load_time + quant_time + save_time:.2f}s")
    print(f"{'='*70}\n")
    
    # Usage instructions
    print("Usage in ComfyUI:")
    print("1. Copy the output file to your ComfyUI/models/SEEDVR2/ directory")
    print("2. In the DiT Model Loader node:")
    print("   - Select the NVFP4 model from dropdown")
    print("   - Enable 'force_nvfp4' checkbox")
    print("3. Run your workflow normally")
    print()
    print("Expected performance on Blackwell GPUs (RTX 50 series):")
    print("  - Memory: 4x less than FP16")
    print("  - Speed: 2-3x faster than GGUF Q4_K_M")
    print("  - Quality: <1% loss vs FP16")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
