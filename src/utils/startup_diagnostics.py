"""Startup diagnostics for NVFP4 support"""

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
            if load_nvfp4_kernels():
                print(f"[SeedVR2] ✅ NVFP4 Native Kernels: LOADED")
            else:
                print(f"[SeedVR2] ⚠️ NVFP4 Native Kernels: NOT FOUND (using emulation)")
                print(f"[SeedVR2]    Install TensorRT for better performance:")
                print(f"[SeedVR2]    pip install tensorrt-llm>=0.14.0")
        except ImportError:
            # NVFP4 module might not be fully initialized yet, skip kernel check
            pass
    else:
        print(f"[SeedVR2] ℹ️ NVFP4 Support: DISABLED (requires RTX 50 series)")
    
    print(f"{'='*70}\n")
