# force_nvfp4 Parameter Implementation

## Summary
Added `force_nvfp4` boolean parameter to DiT model loader node that bypasses GGUF detection when enabled.

## Parameter Flow

```
SeedVR2LoadDiTModel.execute(force_nvfp4)
  ↓ stores in config dict
prepare_runner(force_nvfp4)
  ↓
configure_runner(force_nvfp4)
  ↓
prepare_model_structure(force_nvfp4)
  ↓ stores in runner._dit_force_nvfp4
materialize_model()
  ↓ reads runner._dit_force_nvfp4
_load_model_weights(force_nvfp4)
  ↓
load_quantized_state_dict(force_nvfp4)
  ↓ skips _detect_gguf_in_safetensors() when true
```

## Files Modified

1. **src/interfaces/dit_model_loader.py**
   - Added `force_nvfp4` Boolean input parameter
   - Added comprehensive tooltip explaining experimental nature
   - Pass in config dict

2. **src/core/model_loader.py**
   - Updated `load_quantized_state_dict()` to accept `force_nvfp4`
   - Skip GGUF detection when `force_nvfp4=True`
   - Updated `prepare_model_structure()` to store `force_nvfp4` in runner
   - Updated `materialize_model()` to read `force_nvfp4` from runner
   - Updated `_load_model_weights()` to pass `force_nvfp4`

3. **src/core/model_configuration.py** (TODO)
   - Update `configure_runner()` signature
   - Pass `force_nvfp4` to `prepare_model_structure()`

4. **src/core/generation_utils.py** (TODO)
   - Update `prepare_runner()` signature
   - Pass `force_nvfp4` to `configure_runner()`

5. **src/interfaces/video_upscaler.py** (TODO)
   - Extract `force_nvfp4` from dit_config
   - Pass to `prepare_runner()`

## Behavior

### When force_nvfp4=False (default):
- Normal behavior
- GGUF detection active
- Rejects GGUF data in .safetensors files

### When force_nvfp4=True:
- Bypasses GGUF detection
- Loads .safetensors file as-is
- Treats as "native NVFP4" format
- User takes responsibility for format correctness

## Warning
This is experimental. Using it with actual GGUF files will cause shape mismatch errors.
Only use when you have genuine NVFP4 safetensors files.
