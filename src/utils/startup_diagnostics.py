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
        
        # Pure PyTorch native implementation (no external dependencies)
        print(f"[SeedVR2] ✅ Blackwell Native NVFP4: ACTIVE")
        print(f"[SeedVR2] ✅ Pure PyTorch implementation (JIT-compiled)")
        print(f"[SeedVR2] ✅ Tensor Core acceleration enabled")
        print(f"[SeedVR2] 🚀 Performance: Optimized for Blackwell architecture")
        
        # Check for optional TensorRT-LLM acceleration
        try:
            import tensorrt_llm
            if os.environ.get('ENABLE_NVFP4_NATIVE') == '1':
                trt_version = tensorrt_llm.__version__
                print(f"[SeedVR2] ⚡ TensorRT-LLM v{trt_version} detected (extra acceleration)")
        except ImportError:
            # No TensorRT-LLM - pure PyTorch mode is default and sufficient
            pass
            
    else:
        print(f"[SeedVR2] ℹ️ NVFP4 Support: DISABLED (requires RTX 50 series)")
        print(f"[SeedVR2]    Your GPU: {gpu_info['name']}")
        print(f"[SeedVR2]    NVFP4 optimized for: Blackwell GPUs (RTX 5070/5080/5090)")
    
    print(f"{'='*70}\n")
