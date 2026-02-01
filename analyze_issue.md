# Analysis of the NVFP4 / GGUF Shape Mismatch Issue

## The Problem

Error message shows:
```
Error(s) in loading state_dict for NaDiT:
  size mismatch for vid_in.proj.weight: 
    copying a param with shape torch.Size([21120, 16]) from checkpoint,
    the shape in current model is torch.Size([2560, 132])
```

## Key Observations

1. **Shape Pattern**: All checkpoint shapes end with `16` (e.g., `[21120, 16]`, `[160, 16]`)
   - This is the signature of GGUF Q4_K_M quantization
   - Q4_K_M stores data in blocks with a specific structure

2. **Element Count**: Total elements match perfectly
   - Checkpoint: 21120 × 16 = 337,920 elements
   - Model: 2560 × 132 = 337,920 elements
   - This confirms it's quantized data of the correct size

3. **File Extension**: Model is registered as `.safetensors`
   - But the data pattern is clearly GGUF Q4_K_M
   - Suggests file mismatch or incorrect file type

## Root Cause

The NVFP4 model listed in the registry probably doesn't exist yet:
- Registry entry: `seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors`
- Repository: `Nexus24/vaeGGUF` 
- The repository name hints it contains GGUF files, not NVFP4 safetensors

## What's Happening

When user tries to load the NVFP4 model:
1. File doesn't exist or user has wrong file
2. They're actually loading a GGUF Q4_K_M file
3. But the file has `.safetensors` extension (or is being treated as such)
4. The quantized GGUF shapes are being compared directly to model shapes
5. This causes the mismatch error

## Solution

We need to:
1. Detect when a `.safetensors` file contains GGUF data
2. Handle it appropriately by routing to GGUF loading path
3. Or detect the model file doesn't exist and provide clear error
4. Fix the NVFP4 model entry if the file doesn't actually exist
