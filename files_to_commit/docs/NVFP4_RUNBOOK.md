# NVFP4 Runbook — building engines and validating on RTX 50 series

This runbook documents the recommended workflow to build and validate NVFP4 inference on Blackwell (RTX 50 series) hardware.

Important notes
- The repo currently runs models in PyTorch by default. The NVFP4 scaffolding adds:
  - a dedicated model route for the NVFP4-shaped checkpoint
  - a TensorRT engine-builder scaffold and a runtime wrapper
- A fully working NVFP4 pipeline requires:
  1. A matching TRT version that supports NVFP4 (check NVIDIA docs)
  2. An ONNX export path of your model or a direct builder that maps model pieces to TRT layers
  3. Possible weight packing / layout conversion to match TRT plugin expectations

Prerequisites (server)
- Ubuntu 24.04 (recommended)
- NVIDIA driver: your provided driver (590.48.01) — ensure compatibility with CUDA 13.1
- CUDA: 13.1
- TensorRT: install the TRT package that matches CUDA 13.1 for your platform
- PyTorch: 2.9.1+cu130 (or matching wheel)
- Python: 3.10+ recommended

High-level steps

1) Prepare a clean Python environment on the server
   python3 -m venv ~/nvfp4-env
   source ~/nvfp4-env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # If requirements.txt not present, install at least: pyjwt requests torch

2) Export model to ONNX (server)
   - Use tools/convert_nvfp4_to_trt.py to attempt an ONNX export.
   - This script expects your project loader to provide materialize_model(...) and may need adaptation to the project's API.
   - Example:
     python tools/convert_nvfp4_to_trt.py \
       --checkpoint /models/seedvr2_ema_3b_nvfp4_native.safetensors \
       --onnx-path /tmp/seedvr2_nvfp4.onnx \
       --plan-path /tmp/seedvr2_nvfp4.plan \
       --sample-shape "1,3,256,256"

   - Note: exporting large diffusion models to ONNX is complex; you may prefer to export a smaller "inference submodule" that TRT will run (e.g., the DiT forward only) or use custom plugins.

3) Build TensorRT engine
   - Once you have a stable ONNX representation, run the builder:
     python -c "from src.runtime.tensorrt.nvfp4_engine import build_engine_from_onnx; build_engine_from_onnx('/tmp/seedvr2_nvfp4.onnx', 'engines/nvfp4.plan', enable_nvfp4=True)"
   - If the TRT version doesn't expose NVFP4 flags, the builder will attempt FP16 fallback (and will log a warning).

4) Run smoke test
   - Place plan at engines/nvfp4.plan (or pass --engine to test script)
   - Activate venv and run:
     python tests/smoke_nvfp4.py --engine engines/nvfp4.plan --use-cuda

5) If TRT plan can't be built
   - Use PyTorch fallback (the repo already supports running via PyTorch; the new model loader route materializes NVFP4-shaped checkpoint into PyTorch tensors).
   - Validate functionality and performance differences.

Notes on large files and distribution
- The GitHub blob API used by the App push script has a 100MB per-blob limit.
- For model weights or TRT plans >100MB:
  - Use Git LFS for weights/trt plans, or
  - Upload plans as GitHub Release assets or host externally and commit a downloader script into the repo.

Troubleshooting
- ONNX export errors:
  - Operator not supported: you may need to replace or decompose operators before export.
  - Dynamic shapes: add optimization profiles in the TRTEngine builder.
- TRT builder errors:
  - Check builder logs (set trt.Logger to VERBOSE temporarily).
  - Ensure matching CUDA/TRT/PyTorch versions.

If you want me to:
- Adapt the ONNX export to the exact function names and signature your project uses, paste the exact model instantiation call (how you would materialize the model in Python) and I will adapt tools/convert_nvfp4_to_trt.py accordingly.
- Implement the TRT inference bindings (IO buffer binding & execution loop) for your exact model once you provide a successful ONNX export or the serialized plan.
