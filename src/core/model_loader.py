"""
Model Weight Loading for SeedVR2

This module handles all weight loading operations for DiT and VAE models:
- Loading state dictionaries from multiple formats (SafeTensors, PyTorch, GGUF)
- Materializing models from meta device to target device
- Applying weights with dtype conversion
- GGUF quantized model support with dequantization
- Meta buffer initialization for non-persistent buffers

Key Features:
- Multi-format support: .safetensors, .pth, .gguf files
- Memory-efficient loading with meta device initialization
- Native FP8 weight handling with optimal performance
- GGUF quantization support (Q4_K_M, Q8_0, etc.)
- Automatic dtype conversion for compatibility
- Meta buffer initialization post-materialization

Main Functions:
- load_quantized_state_dict: Load state dict from checkpoint file
- materialize_model: Move model from meta device and load weights
- prepare_model_structure: Create model structure on meta device

GGUF Support:
- apply_gguf_parameters: Apply GGUF weights to model (handles meta and materialized)
- _load_gguf_state: Load GGUF quantized weights from file
- _load_gguf_weights: Apply GGUF weights to model with validation
- _validate_gguf_architecture: Validate GGUF model architecture
- _create_dequantize_method: Create dequantization callable
- _create_gguf_parameter: Create parameter preserving quantization info
- _set_parameter_on_meta_model: Set parameter on meta device model
- _set_parameter_on_materialized_model: Set parameter on materialized model
- _navigate_to_parameter: Navigate to module containing parameter
- _get_tensor_shape: Get logical shape of tensor (handling GGUF)
- _is_quantized_tensor: Check if tensor is GGUF quantized
- _report_parameter_mismatches: Report parameter mismatches

Meta Buffer Initialization:
- initialize_meta_buffers: Initialize meta buffers with timing wrapper
- initialize_meta_buffers_impl: Initialize non-persistent buffers on target device

Standard Loading:
- _load_model_weights: Orchestrate weight loading process
- _load_standard_weights: Apply SafeTensors/PyTorch weights
- _convert_state_dtype: Convert weight dtypes
- _log_weight_stats: Log weight statistics

This module is used by model_configuration for weight loading during materialization.
"""

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable, List

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .infer import VideoDiffusionInfer
from ..common.config import create_object
from ..optimization.compatibility import (
    GGUF_AVAILABLE,
    GGMLQuantizationType,
    validate_gguf_availability
)

# GGUF-specific imports (only when available)
if GGUF_AVAILABLE:
    import gguf
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor
    from ..optimization.gguf_ops import replace_linear_with_quantized

from ..utils.constants import get_script_directory, suppress_tensor_warnings

# Get script directory for config paths
script_directory = get_script_directory()


def load_quantized_state_dict(checkpoint_path: str, device: torch.device = torch.device("cpu"),
                              debug: Optional['Debug'] = None, force_nvfp4: bool = False) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint with support for multiple formats.
    
    Handles .safetensors, .gguf, and .pth files. GGUF models support quantization
    for memory-efficient loading. Validates required libraries are installed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device for tensor placement (torch.device object, defaults to CPU)
        debug: Optional Debug instance for logging
        force_nvfp4: If True, bypasses GGUF detection and treats file as NVFP4 safetensors
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
        - force_nvfp4 mode skips automatic GGUF detection in .safetensors files
    """
    device_str = str(device)
    
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        
        # Try direct device loading first (optimal path)
        try:
            state = load_safetensors_file(checkpoint_path, device=device_str)
        except RuntimeError as e:
            # MPS allocator fallback: some PyTorch/macOS versions have issues with
            # direct MPS loading (allocation failures, watermark errors, etc.)
            error_msg = str(e).lower()
            is_mps_alloc_error = device.type == "mps" and any(
                keyword in error_msg for keyword in ["watermark", "allocat", "memory"]
            )
            
            if is_mps_alloc_error:
                # Transparent fallback - only log if debug enabled
                if debug:
                    debug.log("Using CPU intermediate loading for MPS compatibility", 
                            category="info", indent_level=1)
                state = load_safetensors_file(checkpoint_path, device="cpu")
                # Tensors will be moved to MPS during model.load_state_dict()
            else:
                # Re-raise if it's a different error (file corruption, etc.)
                raise
        
        # CRITICAL: Detect GGUF data masquerading as safetensors
        # GGUF Q4_K_M files have characteristic shape pattern: [..., 16]
        # If found, the file is GGUF with wrong extension
        if not force_nvfp4:
            _detect_gguf_in_safetensors(state, checkpoint_path, debug)
        
        # Check for NVFP4 format and wrap if detected
        state = _detect_and_wrap_nvfp4(state, checkpoint_path, debug, force_nvfp4)
        
        # If force_nvfp4 is enabled but no NVFP4 data found, check if it's actually GGUF
        if force_nvfp4 and not any(k.endswith('.nvfp4_data') for k in state.keys()):
            # Check if this is actually a GGUF file
            shapes_ending_16 = sum(
                1 for tensor in list(state.values())[:10]
                if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2 and tensor.shape[-1] == 16
            )
            
            if shapes_ending_16 >= 5:
                # This is GGUF data, not NVFP4!
                if debug:
                    debug.log("⚠️ force_nvfp4 enabled but file contains GGUF data (not NVFP4)", 
                             category="warning", indent_level=1)
                    debug.log("   Loading as GGUF instead to prevent meta device errors", 
                             category="info", indent_level=1)
                
                # Reload as GGUF
                validate_gguf_availability(f"load {os.path.basename(checkpoint_path)}", debug)
                state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
            else:
                if debug:
                    debug.log("⚠️ force_nvfp4 enabled but no NVFP4 format detected", 
                             category="warning", indent_level=1)
                    debug.log("   File may be standard FP16/FP8 - loading as-is", 
                             category="info", indent_level=1)
        
    elif checkpoint_path.endswith('.gguf'):
        validate_gguf_availability(f"load {os.path.basename(checkpoint_path)}", debug)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device_str, mmap=True, weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _detect_gguf_in_safetensors(state_dict: Dict[str, torch.Tensor], checkpoint_path: str,
                                debug: Optional['Debug'] = None) -> None:
    """
    Detect if a safetensors file actually contains GGUF quantized data.
    
    GGUF Q4_K_M models have a characteristic shape pattern where tensors end with dimension 16.
    If we find this pattern in a .safetensors file, it's actually GGUF data with the wrong extension.
    
    Args:
        state_dict: Loaded state dictionary
        checkpoint_path: Path to the file
        debug: Debug instance for logging
        
    Raises:
        ValueError: If GGUF data is detected in safetensors file
    """
    # Check first 10 tensors for GGUF pattern
    shapes_ending_16 = 0
    total_checked = 0
    sample_shapes = []
    
    for key, tensor in list(state_dict.items())[:10]:
        total_checked += 1
        shape = tensor.shape
        sample_shapes.append(f"{key}: {list(shape)}")
        
        # GGUF Q4_K_M tensors have shape [..., 16]
        if len(shape) >= 2 and shape[-1] == 16:
            shapes_ending_16 += 1
    
    # If more than half the tensors end in 16, it's likely GGUF
    if shapes_ending_16 >= 5:
        filename = os.path.basename(checkpoint_path)
        base_name = os.path.splitext(filename)[0]
        
        error_msg = (
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════╗\n"
            f"║  GGUF FILE WITH WRONG EXTENSION DETECTED                             ║\n"
            f"╠══════════════════════════════════════════════════════════════════════╣\n"
            f"║  File: {filename[:62]:<62} ║\n"
            f"║                                                                      ║\n"
            f"║  This file contains GGUF quantized data but has .safetensors         ║\n"
            f"║  extension. This causes shape mismatch errors during loading.        ║\n"
            f"║                                                                      ║\n"
            f"║  NOTE: If filename says 'nvfp4', that's just the NAME.              ║\n"
            f"║        The actual FORMAT is GGUF Q4_K_M (see evidence below).       ║\n"
            f"║        True NVFP4 doesn't exist in PyTorch yet.                     ║\n"
            f"║                                                                      ║\n"
            f"║  Evidence:                                                           ║\n"
            f"║    • {shapes_ending_16}/{total_checked} tensors have shape [..., 16] (GGUF Q4_K_M pattern) ║\n"
            f"║    • Example shapes:                                                 ║\n"
        )
        
        for shape_info in sample_shapes[:3]:
            if len(shape_info) > 64:
                shape_info = shape_info[:61] + "..."
            error_msg += f"║      {shape_info:<66} ║\n"
        
        error_msg += (
            f"║                                                                      ║\n"
            f"║  SOLUTION 1: Rename the file (Recommended)                          ║\n"
            f"║    Rename to: {base_name}.gguf                                       \n"
            f"║    Location: models/dit/                                             ║\n"
            f"║                                                                      ║\n"
            f"║  SOLUTION 2: Use proven alternative models                          ║\n"
            f"║    • seedvr2_ema_3b-Q4_K_M.gguf (GGUF, any GPU)                     ║\n"
            f"║      From: https://huggingface.co/cmeka/SeedVR2-GGUF                ║\n"
            f"║                                                                      ║\n"
            f"║    • seedvr2_ema_3b_fp8_e4m3fn.safetensors (FP8, RTX 40/50)         ║\n"
            f"║      Already available in default models                            ║\n"
            f"║                                                                      ║\n"
            f"║    • seedvr2_ema_3b_fp16.safetensors (FP16, any GPU)                ║\n"
            f"║      Already available in default models                            ║\n"
            f"║                                                                      ║\n"
            f"║  TECHNICAL NOTE:                                                    ║\n"
            f"║    GGUF files use block quantization with specific shape patterns.  ║\n"
            f"║    They must have .gguf extension to be loaded correctly.           ║\n"
            f"║    True NVFP4 format doesn't exist yet in PyTorch.                  ║\n"
            f"╚══════════════════════════════════════════════════════════════════════╝\n"
        )
        
        if debug:
            debug.log(error_msg, level="ERROR", category="dit", force=True)
        
        raise ValueError(
            f"GGUF data detected in .safetensors file: {filename}\n"
            f"Rename to {base_name}.gguf or use alternative models.\n"
            f"See error message above for details."
        )


def _detect_and_wrap_nvfp4(state_dict: Dict[str, torch.Tensor], checkpoint_path: str,
                           debug: Optional['Debug'] = None, force_nvfp4: bool = False) -> Dict[str, torch.Tensor]:
    """
    Detect if a safetensors file contains NVFP4 quantized data and wrap it appropriately.
    
    NVFP4 models use E2M1 4-bit floating point with two-level scaling:
    - Micro-block: 16 values per FP8 scale
    - Tensor-level: Global FP32 scale
    
    Expected keys in state_dict:
    - 'layer.weight.nvfp4_data' → uint8 packed NVFP4 values
    - 'layer.weight.fp8_scales' → FP8 micro-block scales
    - 'layer.weight.fp32_scale' → FP32 tensor scale
    - 'layer.weight.shape' → Original shape (optional)
    
    Args:
        state_dict: Loaded state dictionary
        checkpoint_path: Path to the file
        debug: Debug instance for logging
        force_nvfp4: If True, forces NVFP4 detection even without proper keys
        
    Returns:
        State dict with NVFP4Tensor wrappers if NVFP4 format detected,
        otherwise returns original state dict
    """
    # Import NVFP4 modules
    try:
        from ..models.nvfp4 import (
            detect_nvfp4_format,
            wrap_nvfp4_parameters,
            validate_nvfp4_tensors,
            is_native_nvfp4_available,
        )
    except ImportError:
        if debug:
            debug.log("⚠️ NVFP4 modules not available, skipping NVFP4 detection", 
                     category="warning", indent_level=1)
        return state_dict
    
    # Try to get metadata (safetensors files may have metadata)
    metadata = {}
    
    # Detect NVFP4 format
    is_nvfp4 = detect_nvfp4_format(state_dict, metadata)
    
    if force_nvfp4 and not is_nvfp4 and debug:
        debug.log("⚠️ force_nvfp4 enabled but NVFP4 format not detected", 
                 category="warning", indent_level=1)
    
    if not is_nvfp4:
        return state_dict  # Not NVFP4, return as-is
    
    # NVFP4 format detected!
    filename = os.path.basename(checkpoint_path)
    
    if debug:
        debug.log(f"✅ NVFP4 format detected in {filename}", 
                 category="info", indent_level=1)
        
        # Check for native acceleration
        native_available = is_native_nvfp4_available()
        if native_available:
            debug.log("✅ Native NVFP4 acceleration available (TensorRT/ModelOpt)", 
                     category="info", indent_level=2)
        else:
            debug.log("ℹ️ Using software NVFP4 implementation (pure PyTorch)", 
                     category="info", indent_level=2)
    
    # Validate NVFP4 structure
    valid, error = validate_nvfp4_tensors(state_dict)
    if not valid:
        error_msg = f"NVFP4 validation failed for {filename}: {error}"
        if debug:
            debug.log(f"⚠️ {error_msg}", category="warning", indent_level=1)
        # Continue anyway - might still work
    
    # Wrap NVFP4 parameters
    try:
        wrapped_state = wrap_nvfp4_parameters(state_dict, block_size=16)
        
        # Count wrapped parameters
        nvfp4_count = sum(1 for v in wrapped_state.values() 
                         if hasattr(v, '__class__') and v.__class__.__name__ == 'NVFP4Tensor')
        
        if debug and nvfp4_count > 0:
            debug.log(f"✅ Wrapped {nvfp4_count} NVFP4 parameters", 
                     category="info", indent_level=2)
            debug.log("ℹ️ Tensors will dequantize on demand (lazy evaluation)", 
                     category="info", indent_level=2)
        
        return wrapped_state
        
    except Exception as e:
        error_msg = f"Failed to wrap NVFP4 parameters: {str(e)}"
        if debug:
            debug.log(f"❌ {error_msg}", category="error", indent_level=1)
        # Return original state dict as fallback
        return state_dict



def _load_gguf_state(checkpoint_path: str, device: torch.device, debug: Optional['Debug'] = None,
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    
    Args:
        checkpoint_path: Path to GGUF file
        device: Target device (torch.device object)
        debug: Debug instance
        handle_prefix: Prefix to strip from tensor names
        
    Returns:
        State dictionary with loaded tensors
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    device_str = str(device)
    debug.log(f"Loading {total_tensors} tensors to {str(device_str)}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"Loaded {i+1}/{total_tensors} tensors...", category="dit", indent_level=1)

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device_str}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Shape: {self.shape}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True, indent_level=1)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"Function: {func}", level="WARNING", category="dit", force=True, indent_level=1)
                        debug.log(f"Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True, indent_level=1)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise

        # Handle conv2d/conv3d operations (critical for GGUF VAE models)
        # Conv3d layers (InflatedCausalConv3d) are not replaced by layer replacement
        if func in {torch.nn.functional.conv2d, torch.nn.functional.conv3d}:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is second arg
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in conv dequantization: {e}", level="WARNING", category="dit", force=True)
                    raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)


def _detect_model_parameters_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    debug: 'Debug'
) -> Dict[str, Any]:
    """
    Detect model parameters from checkpoint before creating model.
    Prevents architecture mismatch by reading actual checkpoint dimensions.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: "dit" or "vae"
        debug: Debug instance for logging
        
    Returns:
        Dict with detected parameters: vid_dim, num_layers, etc.
    """
    detected_params = {}
    
    if not SAFETENSORS_AVAILABLE or not checkpoint_path.endswith('.safetensors'):
        debug.log(f"Cannot detect parameters: SafeTensors not available or not .safetensors file", 
                 category=model_type, force=True)
        return detected_params
    
    try:
        debug.log(f"Detecting model parameters from checkpoint...", 
                 category=model_type, force=True)
        
        from safetensors import safe_open
        
        with safe_open(checkpoint_path, framework='pt') as f:
            keys = list(f.keys())
            
            # Detect vid_dim (hidden_size) from various possible keys
            # CRITICAL: Use BIAS tensors - they show TRUE dimension
            bias_candidates = [
                'vid_in.proj.bias',
                'txt_in.bias', 
                'emb_in.proj_in.bias'
            ]
            
            vid_dim = None
            for bias_key in bias_candidates:
                if bias_key in keys:
                    bias = f.get_tensor(bias_key)
                    vid_dim = bias.shape[0]
                    detected_params['vid_dim'] = vid_dim
                    debug.log(f"Detected vid_dim from {bias_key}: {vid_dim}", 
                             category=model_type, force=True)
                    break
            
            if vid_dim is None:
                # Fallback to weight if no bias
                for key in ['vid_in.proj.weight', 'txt_in.weight']:
                    if key in keys:
                        tensor = f.get_tensor(key)
                        if len(tensor.shape) == 2:
                            vid_dim = tensor.shape[0]
                            detected_params['vid_dim'] = vid_dim
                            debug.log(f"Detected vid_dim from {key}: {vid_dim}", 
                                     category=model_type, force=True)
                            break
            
            # Detect num_layers (depth) by counting blocks
            block_keys = [k for k in keys if k.startswith('blocks.')]
            if block_keys:
                # Extract block indices
                block_indices = set()
                for k in block_keys:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_indices.add(int(parts[1]))
                
                if block_indices:
                    num_layers = max(block_indices) + 1
                    detected_params['num_layers'] = num_layers
                    debug.log(f"Detected num_layers: {num_layers}", 
                             category=model_type, force=True)
            
            # Detect heads from attention tensors
            for key in ['blocks.0.attn.proj_qkv.vid.weight', 'blocks.0.attn.proj_qkv.txt.weight']:
                if key in keys:
                    tensor = f.get_tensor(key)
                    shape = tensor.shape
                    # proj_qkv has shape [3 * heads * head_dim, hidden_dim]
                    # or for multi-head: [heads * 3 * head_dim, hidden_dim]
                    if 'vid_dim' in detected_params:
                        qkv_dim = shape[0]
                        vid_dim = detected_params['vid_dim']
                        # qkv_dim = 3 * heads * head_dim
                        # For head_dim=128 (common): heads = qkv_dim / (3 * 128)
                        if qkv_dim % (3 * 128) == 0:
                            heads = qkv_dim // (3 * 128)
                            detected_params['heads'] = heads
                            debug.log(f"Detected heads from {key}: {heads}", 
                                     category=model_type, force=True)
                    break
        
        if detected_params:
            debug.log(f"Successfully detected parameters: {detected_params}", 
                     category=model_type, force=True)
        else:
            debug.log(f"No parameters detected from checkpoint", 
                     category=model_type, force=True)
                     
    except Exception as e:
        debug.log(f"Error detecting parameters: {e}", 
                 category=model_type, level="WARNING", force=True)
        import traceback
        traceback.print_exc()
    
    return detected_params


def prepare_model_structure(
    runner: VideoDiffusionInfer,
    model_type: str,
    checkpoint_path: str,
    config: OmegaConf,
    debug: 'Debug',
    block_swap_config: Optional[Dict[str, Any]] = None,
    force_nvfp4: bool = False
) -> VideoDiffusionInfer:
    """
    Prepare model structure on meta device without loading weights.
    This uses zero memory as meta device doesn't allocate real memory.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_type: "dit" or "vae"
        checkpoint_path: Path to checkpoint (stored for later loading)
        config: Model configuration
        debug: Debug instance for logging (required)
        block_swap_config: BlockSwap config (stored for DiT, optional)
        force_nvfp4: Force NVFP4 loading mode (bypasses GGUF detection)
        
    Returns:
        runner: Updated runner with model structure on meta device
    """
    if debug is None:
        raise ValueError(f"Debug instance required for prepare_model_structure")
    
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Detect model parameters from checkpoint BEFORE creating model
    if is_dit and os.path.exists(checkpoint_path):
        detected_params = _detect_model_parameters_from_checkpoint(
            checkpoint_path, model_type, debug
        )
        
        # Update config with detected parameters
        if detected_params:
            for param_name, param_value in detected_params.items():
                if hasattr(model_config, param_name):
                    current_value = getattr(model_config, param_name)
                    if current_value != param_value:
                        debug.log(f"Config {param_name} ({current_value}) != checkpoint ({param_value})", 
                                 category=model_type, force=True)
                        debug.log(f"✅ Using checkpoint value: {param_name} = {param_value}", 
                                 category=model_type, force=True)
                        setattr(model_config, param_name, param_value)
                        
                        # Update dependent parameters
                        if param_name == 'vid_dim':
                            # txt_dim usually equals vid_dim
                            if hasattr(model_config, 'txt_dim'):
                                setattr(model_config, 'txt_dim', param_value)
                                debug.log(f"Also updated txt_dim = {param_value}", 
                                         category=model_type, force=True)
                            # emb_dim usually equals 6 * vid_dim
                            if hasattr(model_config, 'emb_dim'):
                                emb_dim = 6 * param_value
                                setattr(model_config, 'emb_dim', emb_dim)
                                debug.log(f"Also updated emb_dim = {emb_dim}", 
                                         category=model_type, force=True)
    
    # Always create on meta device for zero memory usage
    debug.log(f"Creating {model_type_upper} model structure on meta device", 
             category=model_type, force=True)
    debug.start_timer(f"{model_type}_structure")
    
    with torch.device("meta"):
        model = create_object(model_config)
    
    debug.end_timer(f"{model_type}_structure", f"{model_type_upper} structure created")
    
    # Store model and config for later materialization
    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
        runner._dit_force_nvfp4 = force_nvfp4
    else:
        runner.vae = model  
        runner._vae_checkpoint = checkpoint_path
    
    return runner


def materialize_model(runner: VideoDiffusionInfer, model_type: str, device: torch.device, 
                     config: OmegaConf, debug: 'Debug') -> None:
    """
    Materialize model weights from checkpoint to memory.
    Call this right before the model is needed.
    
    Args:
        runner: Runner with model structure on meta device
        model_type: "dit" or "vae"
        device: Target device for inference (torch.device object)
        config: Full configuration
        debug: Debug instance
    """
    if debug is None:
        raise ValueError(f"Debug instance required for materialize_model")
        
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    
    # Get model and checkpoint path
    if is_dit:
        model = runner.dit
        checkpoint_path = runner._dit_checkpoint
        block_swap_config = runner._dit_block_swap_config
        override_dtype = getattr(runner, '_dit_dtype_override', None)
        force_nvfp4 = getattr(runner, '_dit_force_nvfp4', False)
    else:
        model = runner.vae
        checkpoint_path = runner._vae_checkpoint
        block_swap_config = None
        override_dtype = getattr(runner, '_vae_dtype_override', None)
        force_nvfp4 = False  # VAE doesn't support force_nvfp4
    
    # Check if already materialized
    if model is None:
        debug.log(f"No {model_type_upper} model structure found", level="WARNING", category=model_type, force=True)
        return
    param_device = next(model.parameters()).device
    if param_device.type != 'meta':
        debug.log(f"{model_type_upper} already materialized on {model.device}", category=model_type)
        return
    
    # Determine target device for materialization
    offload_device_str = None
    if hasattr(runner, f'_{model_type}_offload_device'):
        offload_device_str = getattr(runner, f'_{model_type}_offload_device')

    # If offload_device is set and not "none", materialize to offload device
    if offload_device_str and offload_device_str != "none":
        target_device = torch.device(offload_device_str)
        offload_reason = " (offload device)"
    else:
        # Otherwise materialize to inference device
        target_device = device
        offload_reason = ""
    
    # Start materialization
    debug.start_timer(f"{model_type}_materialize")
    
    # Load weights (this materializes from meta to target device)
    model = _load_model_weights(model, checkpoint_path, target_device, True,
                               model_type_upper, offload_reason, debug, override_dtype) 
   
    # Apply model-specific configurations (includes BlockSwap and torch.compile)
    # Import here to avoid circular dependency 
    from .model_configuration import apply_model_specific_config
    model = apply_model_specific_config(model, runner, config, is_dit, debug)
    
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")
    
    # Clean up checkpoint paths (no longer needed after weights are loaded)
    # Note: Config attributes (_dit_block_swap_config, _dit_compile_args, _dit_force_nvfp4) are preserved
    # for configuration change detection on subsequent runs
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
        # Note: _dit_force_nvfp4 is preserved for potential config change detection
    else:
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: torch.device, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Optional['Debug'] = None, override_dtype: Optional[torch.dtype] = None,
                        force_nvfp4: bool = False) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights (torch.device object)
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype: Optional dtype override for weights
        force_nvfp4: Force NVFP4 loading mode (bypasses GGUF detection)
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Log loading action
    action = "Materializing" if used_meta else "Loading"
    target_device_str = str(target_device).upper()
    debug.log(f"{action} {model_type} weights to {target_device_str}{cpu_reason}: {checkpoint_path}", 
             category=model_type_lower, force=True)
    
    # Load state dict from file
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device, debug, force_nvfp4)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")
    
    # Apply dtype conversion if requested
    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype, model_type, debug)
    
    # Log weight statistics
    _log_weight_stats(state, used_meta, model_type, debug)
    
    # Handle GGUF or standard loading
    if checkpoint_path.endswith('.gguf'):
        model = _load_gguf_weights(model, state, used_meta, model_type_lower, debug)
    else:
        model = _load_standard_weights(model, state, used_meta, model_type, model_type_lower, debug, force_nvfp4)
    
    # Clean up state dict
    del state
    
    # Initialize meta buffers if needed
    if used_meta:
        initialize_meta_buffers(model, target_device, debug)
        
        # Validate that no tensors remain on meta device
        _validate_no_meta_tensors(model, model_type, debug)
    
    return model


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype, 
                        model_type: str, debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors in state dict to target dtype."""
    debug.log(f"Converting {model_type} weights to {target_dtype} during loading", category="precision")
    debug.start_timer(f"{model_type.lower()}_dtype_convert")
    
    for key in state:
        if torch.is_tensor(state[key]) and state[key].is_floating_point():
            state[key] = state[key].to(target_dtype)
    
    debug.end_timer(f"{model_type.lower()}_dtype_convert", f"{model_type} weights converted to {target_dtype}")
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], used_meta: bool, model_type: str, debug: Optional['Debug'] = None) -> None:
    """Log statistics about loaded weights."""
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action = "Materializing" if used_meta else "Applying"
    debug.log(f"{action} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type.lower())


def apply_gguf_parameters(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                           model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Apply GGUF parameters to model, handling both meta and materialized models.
    
    Returns:
        Statistics dictionary with loaded count, quantized count, and parameter names
    """
    loaded_names = set()
    quantized_count = 0
    
    for name, param in state.items():
        if name not in model_state:
            continue
            
        model_param = model_state[name]
        param_shape = _get_tensor_shape(param)
        
        if param_shape != model_param.shape:
            debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", 
                     level="ERROR", category="dit", force=True)
            raise ValueError(f"Shape mismatch for parameter {name}")
        
        # Apply parameter based on device type
        with torch.no_grad():
            if model_param.device.type == 'meta':
                _set_parameter_on_meta_model(model, name, param, debug)
            else:
                _set_parameter_on_materialized_model(model, name, param, debug)
        
        loaded_names.add(name)
        if _is_quantized_tensor(param):
            quantized_count += 1
    
    return {
        'loaded': len(loaded_names), 
        'quantized': quantized_count, 
        'loaded_names': loaded_names
    }


def _set_parameter_on_meta_model(model: torch.nn.Module, param_name: str, 
                                 param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on meta device model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    new_param = _create_gguf_parameter(param_value, debug)
    setattr(module, attr_name, new_param)


def _set_parameter_on_materialized_model(model: torch.nn.Module, param_name: str, 
                                         param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on already materialized model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    
    if _is_quantized_tensor(param_value):
        # For quantized tensors, replace with wrapped parameter
        new_param = _create_gguf_parameter(param_value, debug)
        setattr(module, attr_name, new_param)
    else:
        # For regular tensors, just copy
        existing_param = getattr(module, attr_name)
        existing_param.copy_(param_value)


def _navigate_to_parameter(model: torch.nn.Module, param_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Navigate to the module containing a parameter.
    
    Args:
        model: Root model
        param_path: Dot-separated path to parameter
        
    Returns:
        Tuple of (parent module, parameter name)
    """
    path_parts = param_path.split('.')
    module = model
    
    # Navigate to parent module
    for part in path_parts[:-1]:
        module = getattr(module, part)
    
    return module, path_parts[-1]


def _create_gguf_parameter(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> torch.nn.Parameter:
    """
    Create a parameter from a GGUF tensor, preserving quantization info.
    
    Args:
        tensor: GGUF tensor (may be quantized)
        debug: Debug instance for logging
        
    Returns:
        Parameter with GGUF attributes and dequantize method if quantized
    """
    param = torch.nn.Parameter(tensor, requires_grad=False)
    
    # Preserve GGUF attributes if present
    if hasattr(tensor, 'tensor_type'):
        param.tensor_type = tensor.tensor_type
        param.tensor_shape = tensor.tensor_shape
        
        # Add dequantize method for runtime dequantization
        param.gguf_dequantize = _create_dequantize_method(tensor, debug)
    
    return param


def _get_tensor_shape(tensor: torch.Tensor) -> torch.Size:
    """Get the logical shape of a tensor (handling GGUF quantized tensors)."""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    return tensor.shape


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is GGUF quantized."""
    return hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape')


def _report_parameter_mismatches(state: Dict[str, torch.Tensor], 
                                 model_state: Dict[str, torch.Tensor], 
                                 loaded_names: set, debug: Optional['Debug'] = None) -> None:
    """Report any parameter mismatches between GGUF and model."""
    # Check for unmatched GGUF parameters
    unmatched = [name for name in state if name not in model_state]
    if unmatched:
        debug.log(f"Warning: {len(unmatched)} parameters from GGUF not found in model", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few unmatched: {unmatched[:5]}", level="WARNING", category="dit", force=True, indent_level=1)
    
    # Check for missing model parameters  
    missing = [name for name in model_state if name not in loaded_names]
    if missing:
        debug.log(f"Warning: {len(missing)} model parameters not loaded from GGUF", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few missing: {missing[:5]}", level="WARNING", category="dit", force=True, indent_level=1)


def initialize_meta_buffers(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = initialize_meta_buffers_impl(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def initialize_meta_buffers_impl(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization (torch.device object)
        debug: Debug instance for logging
        
    Returns:
        Number of buffers initialized
    """
    initialized_count = 0
    
    # Simply initialize all meta device buffers to zeros on target device
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model
            
            # Create a zero tensor of the same shape on target device
            # Use torch.zeros() instead of zeros_like() to avoid meta device issues
            # This is safe for all non-persistent buffers (caches, dummy tensors, etc.)
            initialized_buffer = torch.zeros(
                buffer.shape, 
                dtype=buffer.dtype if buffer.dtype != torch.float16 else torch.float32,
                device=target_device
            )
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
            initialized_count += 1
    
    return initialized_count


def _safe_setattr(obj, attr, value):
    """
    Safely set nested attribute using dot notation with error handling.
    
    Example: _safe_setattr(model, "blocks.0.attn.proj_qkv.vid.weight", new_param)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        parts = attr.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        return True
    except Exception as e:
        return False


def _detect_nemotron_nvfp4(state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Detect Nemotron-style NVFP4 format.
    
    Nemotron NVFP4 format has:
    - Weights with dtype=torch.uint8 (packed 4-bit)
    - Corresponding _scale_inv parameters with dtype=torch.float8_e4m3fn
    
    Returns:
        True if Nemotron NVFP4 format detected
    """
    has_uint8_weights = False
    has_scale_inv = False
    
    for key, tensor in state_dict.items():
        # Check for uint8 weights
        if key.endswith('.weight') and tensor.dtype == torch.uint8:
            has_uint8_weights = True
        # Check for scale_inv parameters
        if key.endswith('_scale_inv'):
            has_scale_inv = True
        
        # Early exit if both found
        if has_uint8_weights and has_scale_inv:
            return True
    
    return has_uint8_weights and has_scale_inv


def _patch_model_for_nemotron_nvfp4(model: torch.nn.Module, 
                                    state_dict: Dict[str, torch.Tensor],
                                    device: torch.device,
                                    debug: Optional['Debug'] = None) -> Tuple[bool, List[str]]:
    """
    Safely patch model architecture to match packed NVFP4 shapes.
    
    This function performs two critical operations with comprehensive error handling:
    
    Phase 1: Patch weight parameters to match uint8 packed shapes
      - For each parameter where checkpoint shape != model shape
      - Re-initialize parameter with packed shape and uint8 dtype
      - This fixes the size mismatch error
    
    Phase 2: Register scale_inv parameters for native Blackwell execution
      - For every weight, register {name}_scale_inv as FP8 parameter
      - Required for MX Microscaling on Blackwell GPUs
    
    Args:
        model: Model to patch
        state_dict: Checkpoint state dict with packed NVFP4 tensors
        device: Target device for parameters
        debug: Debug instance for logging
        
    Returns:
        Tuple of (success: bool, patched_params: List[str])
    """
    patched_params = []
    errors = []
    
    try:
        # Phase 1: Patch weight parameters to match packed shapes
        # Use simple iteration over named_parameters
        for name, param in model.named_parameters():
            try:
                if name in state_dict:
                    checkpoint_tensor = state_dict[name]
                    
                    # Check if shapes don't match (packed vs unpacked)
                    if checkpoint_tensor.shape != param.shape:
                        # Create new parameter with packed shape and uint8 dtype
                        new_param = torch.nn.Parameter(
                            torch.empty(checkpoint_tensor.shape, 
                                       dtype=torch.uint8, 
                                       device=device),
                            requires_grad=False
                        )
                        
                        # Replace the parameter using safe setattr
                        if _safe_setattr(model, name, new_param):
                            patched_params.append(
                                f"{name} [{list(param.shape)}→{list(checkpoint_tensor.shape)}]"
                            )
                        else:
                            errors.append(f"Failed to patch {name}")
            except Exception as e:
                errors.append(f"Error patching {name}: {str(e)}")
                continue
        
        # Phase 2: Register _scale_inv parameters for Blackwell MX scaling
        for key in state_dict.keys():
            try:
                if key.endswith('_scale_inv'):
                    # Extract parent parameter name (remove _scale_inv suffix)
                    parent_name = key.replace('_scale_inv', '')
                    
                    # Check if parent weight exists in model
                    param_names = [n for n, _ in model.named_parameters()]
                    if parent_name in param_names or any(p.startswith(parent_name) for p in param_names):
                        scale_tensor = state_dict[key]
                        
                        # Create FP8 parameter for scale_inv
                        # Use safe dtype check
                        try:
                            scale_dtype = torch.float8_e4m3fn
                        except AttributeError:
                            # Fallback if float8_e4m3fn not available
                            scale_dtype = torch.float32
                            if debug:
                                debug.log(f"[NVFP4] Warning: float8_e4m3fn not available, using float32 for scales")
                        
                        scale_param = torch.nn.Parameter(
                            torch.empty(scale_tensor.shape,
                                       dtype=scale_dtype,
                                       device=device),
                            requires_grad=False
                        )
                        
                        # Register the scale_inv parameter
                        if _safe_setattr(model, key, scale_param):
                            patched_params.append(f"{key} [NEW FP8 scale]")
                        else:
                            errors.append(f"Failed to register {key}")
            except Exception as e:
                errors.append(f"Error registering scale {key}: {str(e)}")
                continue
        
        # Report any errors
        if errors and debug:
            debug.log(f"[NVFP4] Encountered {len(errors)} errors during patching:", category="warning")
            for error in errors[:3]:  # Show first 3
                debug.log(f"[NVFP4]   {error}", category="warning")
        
        # Return success if we patched anything
        success = len(patched_params) > 0 and len(errors) == 0
        return success, patched_params
        
    except Exception as e:
        # Catch-all for unexpected errors
        if debug:
            debug.log(f"[NVFP4] Critical error during patching: {str(e)}", category="error")
        return False, []


def _load_standard_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                          used_meta: bool, model_type: str, model_type_lower: str,
                          debug: Optional['Debug'] = None, force_nvfp4: bool = False) -> torch.nn.Module:
    """
    Load standard (non-GGUF) weights into model, with Native NVFP4 support.
    
    When force_nvfp4=True and Nemotron NVFP4 format is detected:
      1. Detects packed uint8 weights + _scale_inv parameters
      2. Patches model architecture to match packed shapes
      3. Registers scale_inv parameters for Blackwell MX scaling
      4. Loads with strict=False to allow new parameters
      5. NO dequantization - keeps weights as uint8 for native execution
    
    Args:
        model: Target model
        state: State dict (may contain packed NVFP4 tensors)
        used_meta: Whether model was initialized on meta device
        model_type: Model type string for logging
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        force_nvfp4: Whether to force Native NVFP4 execution path
    
    Returns:
        Model with weights loaded
    """
    
    # NATIVE NVFP4 PATH: force_nvfp4=True
    if force_nvfp4:
        try:
            # Detect if this is Nemotron NVFP4 format (packed uint8 + scale_inv)
            is_nemotron_nvfp4 = _detect_nemotron_nvfp4(state)
            
            if is_nemotron_nvfp4:
                if debug:
                    debug.log("[NVFP4] ✅ Detected Nemotron NVFP4 format (packed uint8 + scale_inv)", 
                             category="nvfp4")
                    debug.log("[NVFP4] force_nvfp4=True: Activating Native Blackwell execution path", 
                             category="nvfp4")
                    debug.log("[NVFP4] Patching model architecture for packed NVFP4...", 
                             category="nvfp4")
                
                # Get target device safely
                try:
                    target_device = next(model.parameters()).device if not used_meta else torch.device('cuda')
                except Exception:
                    target_device = torch.device('cuda')  # Safe fallback
                
                # Dynamically patch model to match packed shapes (with error handling)
                success, patched = _patch_model_for_nemotron_nvfp4(model, state, target_device, debug)
                
                if success and patched and debug:
                    debug.log(f"[NVFP4] ✅ Patched {len(patched)} parameters for packed NVFP4:", 
                             category="nvfp4")
                    # Show first 5 patched parameters
                    for param_info in patched[:5]:
                        debug.log(f"[NVFP4]   {param_info}", category="nvfp4")
                    if len(patched) > 5:
                        debug.log(f"[NVFP4]   ... and {len(patched) - 5} more", category="nvfp4")
                elif not success:
                    if debug:
                        debug.log("[NVFP4] ⚠️ Patching failed, falling back to standard loading", 
                                 category="warning")
                    # Fall through to standard path
                    force_nvfp4 = False
                
                # Only proceed with NVFP4 loading if patching succeeded
                if success and patched:
                    # Load with strict=False to allow new scale_inv parameters
                    if debug:
                        debug.log("[NVFP4] Loading state_dict with strict=False (allows scale_inv parameters)", 
                                 category="nvfp4")
                    
                    try:
                        debug.start_timer(f"{model_type_lower}_state_apply")
                        missing, unexpected = model.load_state_dict(state, strict=False)
                        debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights loaded")
                        
                        if debug:
                            # Log loading results
                            if missing:
                                debug.log(f"[NVFP4] Missing keys: {len(missing)}", category="nvfp4")
                            if unexpected:
                                unexpected_shown = unexpected[:3]
                                debug.log(f"[NVFP4] Unexpected keys: {unexpected_shown}", category="nvfp4")
                                if len(unexpected) > 3:
                                    debug.log(f"[NVFP4]   ... and {len(unexpected) - 3} more", category="nvfp4")
                            
                            debug.log("[NVFP4] ✅ Native NVFP4 model loaded successfully", category="success")
                            debug.log("[NVFP4] Ready for hardware-native execution on Blackwell GPU", category="nvfp4")
                            debug.log("[NVFP4] Weights remain as uint8 (no dequantization)", category="nvfp4")
                        
                        return model
                    except Exception as e:
                        if debug:
                            debug.log(f"[NVFP4] ⚠️ Error during load_state_dict: {str(e)}", category="error")
                            debug.log("[NVFP4] Falling back to standard loading", category="warning")
                        # Fall through to standard path
                        force_nvfp4 = False
            else:
                # Not Nemotron format
                if debug:
                    debug.log("[NVFP4] ⚠️ force_nvfp4=True but file is not Nemotron NVFP4 format", 
                             category="warning")
                    debug.log("[NVFP4] Falling back to standard loading", category="nvfp4")
                force_nvfp4 = False
                
        except Exception as e:
            # Catch-all for any errors in NVFP4 path
            if debug:
                debug.log(f"[NVFP4] ⚠️ Critical error in NVFP4 path: {str(e)}", category="error")
                debug.log("[NVFP4] Falling back to standard loading", category="warning")
            force_nvfp4 = False
    
    # STANDARD PATH: Check if state contains NVFP4Tensor objects (original format)
    if not force_nvfp4:
        has_nvfp4 = any(
            hasattr(v, '__class__') and v.__class__.__name__ == 'NVFP4Tensor'
            for v in state.values()
        )
        
        if has_nvfp4:
            # Unwrap NVFP4 tensors to target device/dtype
            try:
                from ..models.nvfp4 import unwrap_nvfp4_parameters
                
                target_device = next(model.parameters()).device if not used_meta else torch.device('cuda')
                target_dtype = next(model.parameters()).dtype if not used_meta else torch.float32
                
                if debug:
                    debug.log(f"Dequantizing NVFP4 tensors to {target_dtype} on {target_device}", 
                             category=model_type_lower, indent_level=1)
                
                debug.start_timer(f"{model_type_lower}_nvfp4_dequant")
                state = unwrap_nvfp4_parameters(state, target_device, target_dtype)
                debug.end_timer(f"{model_type_lower}_nvfp4_dequant", "NVFP4 dequantization completed")
            except Exception as e:
                if debug:
                    debug.log(f"Warning: Failed to unwrap NVFP4 tensors: {e}", 
                             category=model_type_lower, level="WARNING")
        
        # Standard loading with assignment
        debug.start_timer(f"{model_type_lower}_state_apply")
        model.load_state_dict(state, strict=False, assign=True)
        debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights loaded")
    
    return model


def _load_gguf_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                      used_meta: bool, model_type_lower: str, debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Load GGUF quantized weights into model with architecture validation.
    
    Args:
        model: Target model
        state: GGUF state dict with quantized tensors
        used_meta: Whether model was initialized on meta device
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        
    Returns:
        Model with GGUF weights loaded
    """
    debug.log("Loading GGUF weights", category="dit")
    
    # Get model state dict for validation
    model_state = model.state_dict()
    
    # Validate architecture compatibility
    _validate_gguf_architecture(state, model_state, debug)
    
    # Load GGUF parameters
    stats = apply_gguf_parameters(model, state, model_state, debug)
    
    # Log results
    debug.log(f"GGUF loading complete: {stats['loaded']} parameters loaded", category="success")
    debug.log(f"Quantized parameters: {stats['quantized']}", category="info")
    
    # Report any mismatches
    _report_parameter_mismatches(state, model_state, stats['loaded_names'], debug)
    
    # Replace Linear/Conv2d layers with quantized versions for optimal precision handling
    if stats['quantized'] > 0:
        debug.log("Replacing layers with GGUF-optimized versions for precision handling", category="dit")
        
        replacements, quant_types = replace_linear_with_quantized(model, debug=debug)
        
        if replacements > 0:
            debug.log(f"Replaced {replacements} layers with GGUF-optimized versions", category="success")
            
            # Show actual quantization types found and precision strategy
            if quant_types:
                qtypes_str = ', '.join([f"{qtype}:{count}" for qtype, count in quant_types.items()])
                debug.log(
                    f"GGUF precision path: {qtypes_str} → FP16 (preserve) → BF16/FP32 (compute)", 
                    category="precision"
                )
            else:
                debug.log(
                    "GGUF precision: Dequantizing to FP16 first, then converting to compute dtype", 
                    category="precision"
                )
        else:
            debug.log("Warning: No layers were replaced despite having quantized parameters", 
                     level="WARNING", category="dit", force=True)
    
    return model


def _validate_gguf_architecture(state: Dict[str, torch.Tensor], 
                                model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> None:
    """
    Validate GGUF model architecture matches target model.
    
    Raises:
        ValueError: If architecture mismatch is detected
    """
    key_params = [
        "blocks.0.attn.proj_qkv.vid.weight",
        "blocks.0.attn.proj_qkv.txt.weight", 
        "blocks.0.mlp.vid.proj_in.weight"
    ]

    for key in key_params:
        if key in state and key in model_state:
            model_shape = model_state[key].shape
            gguf_shape = _get_tensor_shape(state[key])
            
            if model_shape != gguf_shape:
                # Check if it's just a quantization difference
                if hasattr(state[key], 'tensor_shape') and state[key].tensor_shape == model_shape:
                    continue
                    
                raise ValueError(
                    f"GGUF model architecture mismatch: This GGUF model is incompatible with the current architecture.\n\n"
                    f"Detected mismatch:\n"
                    f"  Parameter: {key}\n"
                    f"  Expected shape: {model_shape}\n"
                    f"  GGUF shape: {gguf_shape}\n\n"
                    f"Possible solutions:\n"
                    f"1. Use a GGUF model that matches the current architecture\n"
                    f"2. Try using a regular FP16 model instead\n"
                    f"3. Verify you're using the correct model variant (3B vs 7B)"
                )
    
    debug.log(f"Architecture check complete, no shape mismatch", category="success")


def _create_dequantize_method(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> callable:
    """
    Create a dequantization method for a GGUF tensor.
    
    Args:
        tensor: GGUF quantized tensor with tensor_type and tensor_shape attributes
        debug: Debug instance
        
    Returns:
        Callable dequantization method
    """
    def dequantize(device: Optional[torch.device] = None, 
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize GGUF tensor on demand."""
        if hasattr(tensor, 'dequantize'):
            return tensor.dequantize(device, dtype)
        
        try:
            # Fallback to manual dequantization using gguf library
            numpy_data = tensor.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, tensor.tensor_type)
            result = torch.from_numpy(dequantized).to(device or tensor.device, dtype)
            result.requires_grad_(False)
            return result.reshape(tensor.tensor_shape)
        except Exception as e:
            if debug:
                debug.log(f"Warning: Could not dequantize tensor: {e}", level="WARNING", category="dit", force=True)
            return tensor.to(device or tensor.device, dtype)
    
    return dequantize


def _validate_no_meta_tensors(model: torch.nn.Module, model_type: str, debug: Optional['Debug'] = None) -> None:
    """
    Validate that no parameters or buffers remain on meta device after materialization.
    
    Args:
        model: Model to validate
        model_type: Model type for error messages
        debug: Debug instance for logging
        
    Raises:
        RuntimeError: If any tensors are still on meta device
    """
    meta_params = []
    meta_buffers = []
    
    # Check parameters
    for name, param in model.named_parameters():
        if param is not None and param.device.type == 'meta':
            meta_params.append(name)
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            meta_buffers.append(name)
    
    if meta_params or meta_buffers:
        error_msg = f"{model_type} model has tensors still on meta device after materialization:\n"
        if meta_params:
            error_msg += f"  Parameters on meta device ({len(meta_params)}): {meta_params[:5]}\n"
        if meta_buffers:
            error_msg += f"  Buffers on meta device ({len(meta_buffers)}): {meta_buffers[:5]}\n"
        error_msg += "\nThis indicates an incomplete materialization. Please report this issue."
        
        if debug:
            debug.log(error_msg, level="ERROR", category="model", force=True)
        
        raise RuntimeError(error_msg)
    
    if debug:
        debug.log(f"{model_type} materialization validated - no meta tensors found", category="success")