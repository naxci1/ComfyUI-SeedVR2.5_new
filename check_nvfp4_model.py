"""
Analyze the NVFP4 model structure issue.

The error shows shapes like [21120, 16] which is typical of GGUF Q4_K_M quantization
where data is stored in blocks. But the model is a .safetensors file.

This suggests either:
1. The file is actually a GGUF model but named .safetensors
2. The file contains quantized data in a different format (NVFP4)
3. There's a bug in how the model is being loaded
"""

# Let's analyze the shape pattern
errors = [
    ("vid_in.proj.weight", [21120, 16], [2560, 132]),
    ("vid_in.proj.bias", [160, 16], [2560]),
    ("txt_in.weight", [819200, 16], [2560, 5120]),
    ("txt_in.bias", [160, 16], [2560]),
]

print("Analyzing shape patterns:")
print("="*70)

for name, checkpoint_shape, model_shape in errors:
    print(f"\nParameter: {name}")
    print(f"  Checkpoint: {checkpoint_shape}")
    print(f"  Model:      {model_shape}")
    
    # Calculate products
    checkpoint_prod = 1
    for dim in checkpoint_shape:
        checkpoint_prod *= dim
    
    model_prod = 1
    for dim in model_shape:
        model_prod *= dim
    
    print(f"  Checkpoint elements: {checkpoint_prod}")
    print(f"  Model elements:      {model_prod}")
    print(f"  Match: {checkpoint_prod == model_prod}")
    
    # Check if it's a Q4_K_M pattern
    if len(checkpoint_shape) == 2 and checkpoint_shape[1] == 16:
        # Q4_K_M uses blocks of 256 elements, stored as type_size=144 bytes
        # For 2D weight, the quantized shape is different
        print(f"  Pattern: Looks like Q4_K_M quantization (dim2=16)")

print("\n" + "="*70)
print("\nConclusion:")
print("All shapes have the '...16' pattern typical of GGUF Q4_K_M quantization.")
print("This suggests the .safetensors file might actually contain GGUF-quantized")
print("data, or there's confusion in which file is being loaded.")
