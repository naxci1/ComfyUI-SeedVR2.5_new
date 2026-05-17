# NVFP4 De-quantization Fix

## Problem

Output video is pure black, indicating DiT model is producing zeros or NaNs.

## Root Cause

NVFP4 weights need proper de-quantization with scales and offsets, but current implementation:
1. May not detect quantization metadata correctly
2. May not apply scales/offsets
3. Weights might be loaded as-is without de-quantization

## NVFP4 Format Analysis

### Expected Format (Proper NVFP4)
```
layer.weight.nvfp4_data    → uint8 packed (4-bit E2M1)
layer.weight.fp8_scales    → FP8 micro-block scales (16 values per scale)
layer.weight.fp32_scale    → FP32 global scale
```

### Possible Alternative Formats
1. **Pre-dequantized but scaled**: Weights already in FP16/FP32 but with very small values
2. **Different naming**: Scales might be named differently (`.scale`, `.weight_scale`, `.zero_point`)
3. **Packed without metadata**: uint8 data without separate scale tensors

## De-quantization Formula

```python
# Two-level scaling for NVFP4:
decoded = decode_nvfp4_e2m1(unpacked_4bit)  # E2M1 lookup
scaled_micro = decoded * fp8_scales         # Per-block scaling
final = scaled_micro * fp32_scale           # Global scaling
```

## Solution Strategy

1. **Detection**: Check checkpoint for quantization indicators
2. **Metadata Extraction**: Find scale/offset tensors with flexible naming
3. **De-quantization**: Apply proper formula based on what's available
4. **Fallback**: If no metadata, check if weights need rescaling

## Implementation

### Files to Modify

1. **src/core/model_loader.py**:
   - Add `_detect_nvfp4_scales()` function
   - Add `_apply_nvfp4_dequantization()` function
   - Call during weight loading for NVFP4 models

2. **src/models/nvfp4/dequantize.py**:
   - Add flexible scale detection
   - Add weight value analysis for rescaling detection

## Testing

After fix:
1. Load NVFP4 checkpoint
2. Check weight statistics (should not be all near-zero)
3. Run inference
4. Check output video (should not be pure black)
