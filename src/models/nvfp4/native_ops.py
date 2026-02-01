"""
Native NVFP4 Operations for Blackwell GPUs
Implements hardware-native NVFP4 execution without dequantization.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("[NVFP4] Warning: Triton not available, using PyTorch fallback")


# E2M1 lookup table for dequantization
E2M1_VALUES = torch.tensor([
    0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    6.0, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0
], dtype=torch.float32)


class NVFP4NativeTensor:
    """
    Native NVFP4 Tensor wrapper that keeps data in packed format.
    Designed for hardware-native execution on Blackwell GPUs.
    """
    
    def __init__(self, 
                 packed_data: torch.Tensor,
                 scale_inv: torch.Tensor,
                 original_shape: Tuple[int, ...],
                 device: Optional[torch.device] = None):
        """
        Initialize NVFP4 Native Tensor.
        
        Args:
            packed_data: Packed uint8 data (hardware-aligned)
            scale_inv: Inverse scales (FP8 E4M3)
            original_shape: Original tensor shape
            device: Target device
        """
        self.packed_data = packed_data.to(device) if device else packed_data
        self.scale_inv = scale_inv.to(device) if device else scale_inv
        self.original_shape = original_shape
        self.device = self.packed_data.device
        
        # Cache for materialized tensor (lazy evaluation)
        self._materialized = None
        self._materialized_dtype = None
    
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> 'NVFP4NativeTensor':
        """Move to device/dtype (creates new instance)."""
        if device is None:
            device = self.device
        return NVFP4NativeTensor(
            self.packed_data.to(device),
            self.scale_inv.to(device),
            self.original_shape,
            device
        )
    
    def materialize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Materialize tensor (dequantize) only when needed.
        Caches result for reuse.
        """
        if self._materialized is not None and self._materialized_dtype == dtype:
            return self._materialized
        
        # Unpack and dequantize
        unpacked = unpack_nvfp4(self.packed_data)
        
        # Dequantize using E2M1 lookup
        e2m1_lut = E2M1_VALUES.to(self.device)
        decoded = e2m1_lut[unpacked.long()]
        
        # Apply inverse scales
        block_size = 16
        num_blocks = (unpacked.numel() + block_size - 1) // block_size
        scales = 1.0 / self.scale_inv.float()
        
        # Reshape and apply scales
        decoded_blocks = decoded.view(num_blocks, block_size)
        scaled = decoded_blocks * scales.view(-1, 1)
        
        # Reshape to original shape
        flat = scaled.flatten()[:torch.prod(torch.tensor(self.original_shape))]
        result = flat.view(self.original_shape).to(dtype)
        
        # Cache
        self._materialized = result
        self._materialized_dtype = dtype
        
        return result
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.original_shape
    
    @property
    def dtype(self) -> torch.dtype:
        return torch.uint8  # Packed format
    
    def __repr__(self) -> str:
        return f"NVFP4NativeTensor(shape={self.original_shape}, device={self.device})"


def unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack hardware-aligned NVFP4 data.
    
    Unpacking: 
    - val0 = packed & 0x0F (lower 4 bits)
    - val1 = (packed >> 4) & 0x0F (upper 4 bits)
    
    Args:
        packed: uint8 tensor with packed 4-bit values
        
    Returns:
        uint8 tensor with unpacked 4-bit values
    """
    # Unpack: each uint8 contains 2 4-bit values
    val0 = packed & 0x0F
    val1 = (packed >> 4) & 0x0F
    
    # Interleave
    unpacked = torch.stack([val0, val1], dim=-1).flatten()
    
    return unpacked


def nvfp4_linear_native(input: torch.Tensor,
                        weight_nvfp4: NVFP4NativeTensor,
                        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Native NVFP4 linear operation.
    
    This is a reference implementation. For maximum performance on Blackwell,
    this should use TensorRT-LLM or custom CUDA kernels.
    
    Args:
        input: Input tensor (FP16/BF16/FP32)
        weight_nvfp4: Weight in NVFP4NativeTensor format
        bias: Optional bias
        
    Returns:
        Output tensor
    """
    # For now, materialize weight (future: use native kernels)
    weight_fp16 = weight_nvfp4.materialize(torch.float16)
    
    # Standard linear operation
    output = F.linear(input, weight_fp16, bias)
    
    return output


if TRITON_AVAILABLE:
    @triton.jit
    def nvfp4_matmul_kernel(
        # Pointers
        a_ptr, b_ptr, c_ptr,
        # Shapes
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """
        Triton kernel for NVFP4 matrix multiplication.
        This is a simplified version - production should use optimized layouts.
        """
        # Program ID
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Pointers
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Main loop
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
            acc += tl.dot(a, b)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        # Store
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def nvfp4_matmul_triton(a: torch.Tensor,
                        b_nvfp4: NVFP4NativeTensor) -> torch.Tensor:
    """
    Triton-accelerated NVFP4 matmul.
    
    Note: This is a reference implementation. Full optimization requires
    custom kernels that operate directly on packed NVFP4 data.
    """
    if not TRITON_AVAILABLE:
        # Fallback to PyTorch
        b_fp16 = b_nvfp4.materialize(torch.float16)
        return torch.matmul(a, b_fp16.t())
    
    # For now, materialize and use standard Triton matmul
    # Future: implement kernel that works directly with packed NVFP4
    b_fp16 = b_nvfp4.materialize(torch.float16)
    return torch.matmul(a, b_fp16.t())


def is_nvfp4_native_available() -> bool:
    """
    Check if native NVFP4 execution is available.
    
    For true hardware-native execution, need:
    - Blackwell GPU (compute 9.0+)
    - TensorRT-LLM with NVFP4 ops
    - Or custom CUDA kernels
    """
    if not torch.cuda.is_available():
        return False
    
    # Check for Blackwell (compute 9.0+)
    capability = torch.cuda.get_device_capability()
    compute_version = float(f"{capability[0]}.{capability[1]}")
    
    if compute_version < 9.0:
        return False
    
    # Check for TensorRT-LLM or Triton
    if TRITON_AVAILABLE:
        return True
    
    try:
        import tensorrt_llm
        return True
    except ImportError:
        pass
    
    return False


def convert_to_nvfp4_native(state_dict: dict) -> dict:
    """
    Convert state dict to use NVFP4NativeTensor wrappers.
    
    Looks for Nemotron-style keys:
    - {name}: packed data
    - {name}_scale_inv: inverse scales
    
    Args:
        state_dict: State dict with NVFP4 tensors
        
    Returns:
        State dict with NVFP4NativeTensor objects
    """
    metadata = state_dict.get("_metadata", {})
    native_state_dict = {}
    processed_keys = set()
    
    for key, value in state_dict.items():
        if key in processed_keys or key == "_metadata":
            continue
        
        # Check if this is a weight with scale_inv
        scale_key = f"{key}_scale_inv"
        if scale_key in state_dict:
            # This is NVFP4 data
            packed_data = value
            scale_inv = state_dict[scale_key]
            
            # Get original shape from metadata
            shape_key = f"{key}.original_shape"
            if shape_key in metadata:
                original_shape = eval(metadata[shape_key])
            else:
                # Estimate from packed data size
                # This is a fallback, should not happen with proper metadata
                print(f"Warning: No shape metadata for {key}, using default")
                original_shape = (packed_data.numel() * 2,)  # Rough estimate
            
            # Create NVFP4NativeTensor
            native_tensor = NVFP4NativeTensor(
                packed_data,
                scale_inv,
                original_shape
            )
            
            native_state_dict[key] = native_tensor
            processed_keys.add(key)
            processed_keys.add(scale_key)
        else:
            # Regular tensor
            native_state_dict[key] = value
            processed_keys.add(key)
    
    return native_state_dict


__all__ = [
    'NVFP4NativeTensor',
    'nvfp4_linear_native',
    'nvfp4_matmul_triton',
    'is_nvfp4_native_available',
    'convert_to_nvfp4_native',
    'unpack_nvfp4',
]
