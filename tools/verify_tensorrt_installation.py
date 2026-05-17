#!/usr/bin/env python3
"""
TensorRT-LLM Installation Verification Script
Verifies that TensorRT-LLM is properly installed for NVFP4 native kernels.
"""

import sys
import os

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            compute_version = f"{compute_cap[0]}.{compute_cap[1]}"
            
            print(f"✅ CUDA available: True")
            print(f"✅ CUDA version: {cuda_version}")
            print(f"✅ GPU: {device_name}")
            print(f"✅ Compute capability: {compute_version}", end="")
            
            if compute_cap[0] >= 9:
                print(" (Blackwell - NVFP4 supported)")
            elif compute_cap[0] >= 8:
                print(" (Ampere/Hopper - Limited support)")
            else:
                print(" (⚠️ NVFP4 requires compute 9.0+)")
            
            return True, compute_cap
        else:
            print("❌ CUDA not available")
            return False, None
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False, None
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False, None

def check_tensorrt_llm():
    """Check TensorRT-LLM installation"""
    try:
        import tensorrt_llm
        version = tensorrt_llm.__version__
        print(f"✅ TensorRT-LLM installed: True")
        print(f"✅ TensorRT-LLM version: {version}")
        
        # Try to import NVFP4 ops
        try:
            from tensorrt_llm import nvfp4
            print(f"✅ NVFP4 ops available: True")
            return True, version, True
        except (ImportError, AttributeError):
            print(f"⚠️ NVFP4 ops available: False (check TensorRT-LLM version)")
            return True, version, False
            
    except ImportError:
        print("❌ TensorRT-LLM not installed")
        print("   Install: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com")
        return False, None, False
    except Exception as e:
        print(f"❌ Error checking TensorRT-LLM: {e}")
        return False, None, False

def check_optional_deps():
    """Check optional dependencies"""
    results = {}
    
    # Check Triton
    try:
        import triton
        print(f"✅ Triton installed: True (v{triton.__version__})")
        results['triton'] = True
    except ImportError:
        print(f"⚠️ Triton not installed (optional for custom kernels)")
        results['triton'] = False
    
    # Check nvidia-modelopt
    try:
        import modelopt
        print(f"✅ nvidia-modelopt installed: True")
        results['modelopt'] = True
    except ImportError:
        print(f"⚠️ nvidia-modelopt not installed (optional for quantization)")
        results['modelopt'] = False
    
    return results

def check_environment():
    """Check environment variables"""
    print("\nEnvironment Variables:")
    
    enable_nvfp4 = os.environ.get('ENABLE_NVFP4_NATIVE', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    print(f"  ENABLE_NVFP4_NATIVE: {enable_nvfp4}")
    if enable_nvfp4 != '1':
        print(f"    ⚠️ Set to '1' to enable native kernels: export ENABLE_NVFP4_NATIVE=1")
    
    print(f"  CUDA_HOME: {cuda_home}")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    return enable_nvfp4 == '1'

def main():
    print_header("TensorRT-LLM Installation Verification")
    
    # Check CUDA
    cuda_ok, compute_cap = check_cuda()
    
    if not cuda_ok:
        print("\n❌ CUDA is required for NVFP4 native kernels")
        print("   Please install CUDA 12.4+ and PyTorch with CUDA support")
        return 1
    
    print()
    
    # Check TensorRT-LLM
    trt_installed, trt_version, nvfp4_ops = check_tensorrt_llm()
    
    print()
    
    # Check optional dependencies
    opt_deps = check_optional_deps()
    
    # Check environment
    env_ok = check_environment()
    
    # Final verdict
    print_header("Verification Summary")
    
    if cuda_ok and compute_cap and compute_cap[0] >= 9:
        print("✅ GPU: Blackwell architecture (NVFP4 supported)")
    elif cuda_ok and compute_cap:
        print(f"⚠️ GPU: Compute {compute_cap[0]}.{compute_cap[1]} (NVFP4 requires 9.0+)")
    
    if trt_installed and nvfp4_ops:
        print("✅ TensorRT-LLM: Installed with NVFP4 ops")
        
        if env_ok:
            print("✅ Environment: ENABLE_NVFP4_NATIVE=1")
            print_header("✅ Native NVFP4 Kernels: READY")
            print("\nYour system is ready for hardware-native NVFP4 execution!")
            print("Restart ComfyUI to activate native kernels.")
            return 0
        else:
            print("⚠️ Environment: ENABLE_NVFP4_NATIVE not set")
            print("\nTo enable native kernels:")
            print("  export ENABLE_NVFP4_NATIVE=1")
            print("  echo 'export ENABLE_NVFP4_NATIVE=1' >> ~/.bashrc")
            return 0
            
    elif trt_installed:
        print("⚠️ TensorRT-LLM: Installed but NVFP4 ops not found")
        print("\nTry updating to latest version:")
        print("  pip install --upgrade tensorrt-llm --extra-index-url https://pypi.nvidia.com")
        return 1
    else:
        print("❌ TensorRT-LLM: Not installed")
        print("\nInstallation command:")
        print("  pip install tensorrt-llm==0.15.0 --extra-index-url https://pypi.nvidia.com")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification cancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
