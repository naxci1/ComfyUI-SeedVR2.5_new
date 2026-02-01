# TensorRT-LLM Installation Guide for NVFP4 Native Kernels

## Prerequisites

Your hardware:
- **GPU**: RTX 5070 Ti (Blackwell architecture) ✅
- **Compute Capability**: 9.0+
- **VRAM**: 16GB

## Quick Start (For RTX 5070 Ti)

```bash
# 1. Install TensorRT-LLM
pip install tensorrt-llm==0.15.0 --extra-index-url https://pypi.nvidia.com

# 2. Set environment variable
export ENABLE_NVFP4_NATIVE=1
echo 'export ENABLE_NVFP4_NATIVE=1' >> ~/.bashrc

# 3. Restart ComfyUI
# Native kernels will now be active!
```

See full guide below for detailed instructions and troubleshooting.
