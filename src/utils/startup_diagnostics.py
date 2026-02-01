"""Startup diagnostics for NVFP4 support"""
import os

def print_nvfp4_status():
    """Print NVFP4 support status on startup"""
    from .hardware_detection import get_gpu_info, check_nvfp4_support
    
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        print("[SeedVR2] No CUDA GPU detected")
        return
    
    print(f"\n{'='*70}")
    print(f"[SeedVR2] GPU: {gpu_info['name']}")
    print(f"[SeedVR2] Compute Capability: {gpu_info['compute_capability']}")
    
    if check_nvfp4_support():
        print(f"[SeedVR2] ✅ NVFP4 Support: ENABLED (Blackwell architecture)")
        
        # Try to load kernels
        try:
            from ..models.nvfp4 import load_nvfp4_kernels
            kernels_loaded = load_nvfp4_kernels()
            
            if kernels_loaded:
                # Check TensorRT-LLM version
                try:
                    import tensorrt_llm
                    trt_version = tensorrt_llm.__version__
                    print(f"[SeedVR2] ✅ NVFP4 Native Kernels: ACTIVE")
                    print(f"[SeedVR2] ✅ TensorRT-LLM v{trt_version} loaded")
                    print(f"[SeedVR2] ✅ Native Blackwell acceleration enabled")
                    print(f"[SeedVR2] 🚀 Performance: 2-3x faster, 4x memory reduction")
                except ImportError:
                    print(f"[SeedVR2] ✅ NVFP4 Native Kernels: LOADED")
            else:
                print(f"[SeedVR2] ⚠️ NVFP4 Native Kernels: NOT FOUND (using emulation)")
                print(f"[SeedVR2]")
                print(f"[SeedVR2]    To enable native kernels (2-3x speedup):")
                print(f"[SeedVR2]    1. pip install tensorrt-llm>=0.14.0 --extra-index-url https://pypi.nvidia.com")
                print(f"[SeedVR2]    2. export ENABLE_NVFP4_NATIVE=1")
                print(f"[SeedVR2]    3. Restart ComfyUI")
                print(f"[SeedVR2]")
                print(f"[SeedVR2]    Or see: docs/INSTALL_TENSORRT_LLM.md")
                
                # Check if TensorRT-LLM is installed but not enabled
                try:
                    import tensorrt_llm
                    if os.environ.get('ENABLE_NVFP4_NATIVE') != '1':
                        print(f"[SeedVR2]")
                        print(f"[SeedVR2]    ℹ️ TensorRT-LLM found but not enabled!")
                        print(f"[SeedVR2]    Run: export ENABLE_NVFP4_NATIVE=1")
                except ImportError:
                    pass
                    
        except ImportError:
            # NVFP4 module might not be fully initialized yet, skip kernel check
            pass
    else:
        print(f"[SeedVR2] ℹ️ NVFP4 Support: DISABLED (requires RTX 50 series)")
        print(f"[SeedVR2]    Your GPU: {gpu_info['name']}")
        print(f"[SeedVR2]    NVFP4 requires: Blackwell GPUs (RTX 5070/5080/5090)")
    
    print(f"{'='*70}\n")
