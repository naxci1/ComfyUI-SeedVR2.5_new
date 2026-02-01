"""
NVFP4Tensor Wrapper Class
Lazy-loading tensor wrapper for NVFP4 quantized weights

Keeps tensors in quantized format (4-bit) in memory and only
dequantizes when needed for computation, maximizing memory efficiency.
"""

import torch
from typing import Optional, Tuple
from .dequantize import dequantize_nvfp4, create_nvfp4_dequantize_method


class NVFP4Tensor:
    """
    Wrapper for NVFP4 quantized tensors
    
    Stores:
    - quantized_data: uint8 packed NVFP4 values
    - fp8_scales: FP8 micro-block scales (16 values per scale)
    - fp32_scale: Global FP32 tensor scale
    - original_shape: Shape of dequantized tensor
    
    Provides lazy dequantization on demand.
    """
    
    def __init__(
        self,
        quantized_data: torch.Tensor,
        fp8_scales: torch.Tensor,
        fp32_scale: float,
        original_shape: torch.Size,
        block_size: int = 16,
        parameter_name: str = None
    ):
        """
        Initialize NVFP4 tensor wrapper
        
        Args:
            quantized_data: Packed uint8 NVFP4 data
            fp8_scales: FP8 micro-block scales
            fp32_scale: Global FP32 scale
            original_shape: Shape after dequantization
            block_size: NVFP4 block size (default: 16)
            parameter_name: Optional name for debugging
        """
        self.quantized_data = quantized_data
        self.fp8_scales = fp8_scales
        self.fp32_scale = fp32_scale
        self.original_shape = original_shape
        self.block_size = block_size
        self.parameter_name = parameter_name or "unknown"
        
        # Create dequantize method
        self.dequantize = create_nvfp4_dequantize_method(
            quantized_data,
            fp8_scales,
            fp32_scale,
            original_shape,
            block_size
        )
        
        # Cache for dequantized tensor
        self._dequantized_cache = None
        self._cache_device = None
        self._cache_dtype = None
    
    @property
    def shape(self) -> torch.Size:
        """Return shape of dequantized tensor"""
        return self.original_shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Return dtype (always uint8 for quantized storage)"""
        return torch.uint8
    
    @property
    def device(self) -> torch.device:
        """Return device of quantized data"""
        return self.quantized_data.device
    
    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self.original_shape)
    
    @property
    def numel(self) -> int:
        """Number of elements in dequantized tensor"""
        return self.original_shape.numel()
    
    def size(self, dim: Optional[int] = None):
        """Get size of specific dimension or all dimensions"""
        if dim is None:
            return self.original_shape
        return self.original_shape[dim]
    
    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Dequantize and move to target device/dtype
        
        Args:
            device: Target device
            dtype: Target dtype (fp16 or fp32)
            
        Returns:
            Dequantized tensor
        """
        # Check cache
        if (self._dequantized_cache is not None and
            self._cache_device == device and
            self._cache_dtype == dtype):
            return self._dequantized_cache
        
        # Dequantize
        result = self.dequantize(device=device, dtype=dtype)
        
        # Update cache
        self._dequantized_cache = result
        self._cache_device = device
        self._cache_dtype = dtype
        
        return result
    
    def cuda(self, device: Optional[int] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize and move to CUDA device"""
        if device is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{device}')
        return self.to(device, dtype)
    
    def cpu(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize and move to CPU"""
        return self.to(torch.device('cpu'), dtype)
    
    def float(self) -> torch.Tensor:
        """Dequantize to FP32"""
        return self.to(self.device, torch.float32)
    
    def half(self) -> torch.Tensor:
        """Dequantize to FP16"""
        return self.to(self.device, torch.float16)
    
    def memory_usage(self) -> dict:
        """
        Calculate memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        quantized_bytes = self.quantized_data.numel() * self.quantized_data.element_size()
        scales_bytes = self.fp8_scales.numel() * self.fp8_scales.element_size()
        total_quantized = quantized_bytes + scales_bytes + 4  # +4 for fp32_scale
        
        # Estimated dequantized size (FP32)
        dequantized_bytes = self.numel * 4  # 4 bytes per FP32
        
        compression_ratio = dequantized_bytes / total_quantized if total_quantized > 0 else 0
        
        return {
            'quantized_data_bytes': quantized_bytes,
            'scales_bytes': scales_bytes,
            'total_quantized_bytes': total_quantized,
            'dequantized_bytes_fp32': dequantized_bytes,
            'compression_ratio': compression_ratio,
            'memory_saved_bytes': dequantized_bytes - total_quantized,
            'memory_saved_pct': (1 - total_quantized / dequantized_bytes) * 100 if dequantized_bytes > 0 else 0
        }
    
    def __repr__(self) -> str:
        mem = self.memory_usage()
        return (
            f"NVFP4Tensor(name={self.parameter_name}, "
            f"shape={self.original_shape}, "
            f"device={self.device}, "
            f"quantized_size={mem['total_quantized_bytes']//1024}KB, "
            f"compression={mem['compression_ratio']:.1f}x)"
        )
    
    def __str__(self) -> str:
        return self.__repr__()


def wrap_nvfp4_parameters(state_dict: dict, block_size: int = 16) -> dict:
    """
    Wrap NVFP4 parameters in NVFP4Tensor objects
    
    Expects state_dict with keys like:
    - 'layer.weight.nvfp4_data' → quantized data
    - 'layer.weight.fp8_scales' → micro-block scales
    - 'layer.weight.fp32_scale' → tensor scale
    
    Returns state_dict with:
    - 'layer.weight' → NVFP4Tensor
    
    Args:
        state_dict: Raw state dict with NVFP4 data
        block_size: NVFP4 block size (default: 16)
        
    Returns:
        Wrapped state dict with NVFP4Tensor objects
    """
    wrapped = {}
    processed_params = set()
    
    # Find all NVFP4 parameter groups
    for key in state_dict.keys():
        if key.endswith('.nvfp4_data'):
            # Extract base parameter name
            base_name = key[:-len('.nvfp4_data')]
            
            if base_name in processed_params:
                continue
            
            # Get components
            quantized_data = state_dict.get(f'{base_name}.nvfp4_data')
            fp8_scales = state_dict.get(f'{base_name}.fp8_scales')
            fp32_scale_tensor = state_dict.get(f'{base_name}.fp32_scale')
            original_shape_tensor = state_dict.get(f'{base_name}.shape')
            
            if quantized_data is None or fp8_scales is None or fp32_scale_tensor is None:
                print(f"[NVFP4] ⚠️ Incomplete NVFP4 data for {base_name}, skipping")
                continue
            
            # Extract scalar values
            fp32_scale = fp32_scale_tensor.item() if torch.is_tensor(fp32_scale_tensor) else float(fp32_scale_tensor)
            
            # Get original shape
            if original_shape_tensor is not None:
                if torch.is_tensor(original_shape_tensor):
                    original_shape = torch.Size(original_shape_tensor.tolist())
                else:
                    original_shape = torch.Size(original_shape_tensor)
            else:
                # Infer shape from scales
                num_blocks = fp8_scales.numel()
                num_elements = num_blocks * block_size
                # Assume 2D weight matrix
                original_shape = torch.Size([num_elements // block_size, block_size])
            
            # Create NVFP4Tensor
            nvfp4_tensor = NVFP4Tensor(
                quantized_data,
                fp8_scales,
                fp32_scale,
                original_shape,
                block_size,
                base_name
            )
            
            wrapped[base_name] = nvfp4_tensor
            processed_params.add(base_name)
            
            print(f"[NVFP4] Wrapped {base_name}: {original_shape} → {nvfp4_tensor.memory_usage()['compression_ratio']:.1f}x compression")
    
    # Copy non-NVFP4 parameters
    for key, value in state_dict.items():
        if not any(key.endswith(suffix) for suffix in ['.nvfp4_data', '.fp8_scales', '.fp32_scale', '.shape']):
            if key not in wrapped:
                wrapped[key] = value
    
    return wrapped


def unwrap_nvfp4_parameters(
    state_dict: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> dict:
    """
    Unwrap and dequantize all NVFP4Tensor objects
    
    Args:
        state_dict: State dict with NVFP4Tensor objects
        device: Target device
        dtype: Target dtype
        
    Returns:
        State dict with dequantized tensors
    """
    unwrapped = {}
    
    for key, value in state_dict.items():
        if isinstance(value, NVFP4Tensor):
            # Dequantize
            unwrapped[key] = value.to(device, dtype)
            print(f"[NVFP4] Dequantized {key}: {value.shape} to {dtype} on {device}")
        else:
            unwrapped[key] = value
    
    return unwrapped
