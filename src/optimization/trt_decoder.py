"""
TensorRT VAE Decoder Integration for SeedVR2

Provides a TensorRT-accelerated replacement for the VAE decode path.
When a pre-built TensorRT engine is available, this module replaces
the PyTorch VAE decoder with a TRT engine for ~3-5× faster decoding.

Usage:
    # Build the engine first:
    python scripts/export_vae_trt.py --vae-path /path/to/ema_vae_fp16.safetensors --fp8

    # Then in code:
    from src.optimization.trt_decoder import patch_vae_with_trt
    patch_vae_with_trt(runner.vae, engine_path="trt_engines/seedvr2_vae_decoder_135x240_fp8.engine")
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TRTDecoderOutput:
    """Output wrapper matching CausalDecoderOutput interface."""
    sample: torch.Tensor


class TRTDecoder(nn.Module):
    """
    TensorRT-accelerated VAE decoder.
    
    Wraps a pre-built TensorRT engine to replace the PyTorch VAE decoder.
    Handles GPU memory allocation, input/output binding, and CUDA stream
    management for maximum throughput.
    
    The engine is loaded lazily on first decode call, and GPU buffers are
    pre-allocated to avoid allocation overhead on subsequent calls.
    
    Args:
        engine_path: Path to the pre-built .engine file
        device: CUDA device for execution (default: cuda:0)
    """
    
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        super().__init__()
        self.engine_path = engine_path
        self.device = torch.device(device)
        
        # Lazy initialization - loaded on first decode call
        self._engine = None
        self._context = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None
        self._input_shape = None
        self._output_shape = None
    
    def _ensure_initialized(self):
        """Load engine and allocate buffers on first use."""
        if self._engine is not None:
            return
        
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is required for TRT decoder. Install with:\n"
                "  pip install tensorrt"
            )
        
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"TensorRT engine not found: {self.engine_path}\n"
                "Build it with: python scripts/export_vae_trt.py --vae-path /path/to/ema_vae_fp16.safetensors"
            )
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from: {self.engine_path}")
        
        # Create execution context
        self._context = self._engine.create_execution_context()
        
        # Create CUDA stream for async execution
        self._stream = torch.cuda.Stream(self.device)
        
        # Get input/output shapes and allocate buffers
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            mode = self._engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self._input_shape = tuple(shape)
                self._input_buffer = torch.empty(
                    self._input_shape, dtype=torch.float16, device=self.device
                )
                self._context.set_tensor_address(name, self._input_buffer.data_ptr())
                logger.info(f"TRT input '{name}': {self._input_shape}")
            else:
                self._output_shape = tuple(shape)
                self._output_buffer = torch.empty(
                    self._output_shape, dtype=torch.float16, device=self.device
                )
                self._context.set_tensor_address(name, self._output_buffer.data_ptr())
                logger.info(f"TRT output '{name}': {self._output_shape}")
        
        logger.info(
            f"TRT decoder initialized: {os.path.basename(self.engine_path)} "
            f"({os.path.getsize(self.engine_path) / 1e6:.1f} MB)"
        )
    
    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor using TensorRT engine.
        
        Args:
            z: Latent tensor [B, C, H, W] (will be converted to FP16 if needed)
            
        Returns:
            Decoded pixel tensor [B, 3, H*8, W*8]
        """
        self._ensure_initialized()
        
        # Ensure input is FP16 and contiguous
        if z.dtype != torch.float16:
            z = z.half()
        z = z.contiguous()
        
        # Validate shape matches engine expectations
        if z.shape != self._input_shape:
            raise ValueError(
                f"Input shape mismatch: got {tuple(z.shape)}, "
                f"engine expects {self._input_shape}. "
                f"Rebuild the engine with matching latent dimensions."
            )
        
        # Copy input to pre-allocated buffer and execute on dedicated stream
        with torch.cuda.stream(self._stream):
            self._input_buffer.copy_(z, non_blocking=True)
            self._context.execute_async_v3(self._stream.cuda_stream)
        
        # Synchronize to ensure output is ready
        self._stream.synchronize()
        
        # Return a copy of the output (buffer will be reused on next call)
        return self._output_buffer.clone()
    
    def cleanup(self):
        """Release TensorRT resources."""
        self._context = None
        self._engine = None
        self._input_buffer = None
        self._output_buffer = None
        self._stream = None


def patch_vae_with_trt(
    vae_model: nn.Module,
    engine_path: str,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Patch a SeedVR2 VAE model to use TensorRT for decoding.
    
    Replaces the VAE's decode() method with a TRT-accelerated version.
    The encode() method is unchanged (encoder is not on the hot path).
    
    The original decode method is preserved as _original_decode() for fallback.
    
    Args:
        vae_model: VideoAutoencoderKLWrapper instance
        engine_path: Path to the pre-built TensorRT .engine file
        device: CUDA device string
        
    Returns:
        The patched VAE model (modified in-place)
    """
    if not os.path.exists(engine_path):
        logger.warning(
            f"TensorRT engine not found: {engine_path}\n"
            f"Falling back to PyTorch VAE decoder.\n"
            f"Build the engine with: python scripts/export_vae_trt.py --vae-path /path/to/ema_vae_fp16.safetensors"
        )
        return vae_model
    
    trt_decoder = TRTDecoder(engine_path, device=device)
    
    # Preserve original decode for fallback
    vae_model._original_decode = vae_model.decode
    vae_model._trt_decoder = trt_decoder
    
    def trt_decode(self, z, return_dict=True, tiled=False, tile_size=(512, 512), 
                   tile_overlap=(64, 64)):
        """
        TRT-accelerated decode that replaces the PyTorch decoder.
        
        Falls back to the original PyTorch decoder if:
        - Input shape doesn't match the engine's static shape
        - Tiled decoding is requested (TRT doesn't support tiling)
        """
        # Fall back to PyTorch for tiled decoding
        if tiled:
            return self._original_decode(
                z, return_dict=return_dict, tiled=tiled,
                tile_size=tile_size, tile_overlap=tile_overlap
            )
        
        # Prepare input: handle temporal dimension
        # The VAE uses 3D convolutions internally, so 5D input [B, C, T, H, W]
        # is expected. TRT engine operates on 4D [B, C, H, W] for single frames.
        if z.ndim == 4:
            z_input = z
        elif z.ndim == 5:
            # Only squeeze if temporal dim is singleton (T=1)
            if z.shape[2] != 1:
                # Multi-frame temporal input - TRT engine doesn't support this
                return self._original_decode(
                    z, return_dict=return_dict, tiled=tiled,
                    tile_size=tile_size, tile_overlap=tile_overlap
                )
            z_input = z.squeeze(2)
        else:
            # Unexpected shape, fall back
            return self._original_decode(
                z, return_dict=return_dict, tiled=tiled,
                tile_size=tile_size, tile_overlap=tile_overlap
            )
        
        # Try TRT decode
        try:
            decoded = self._trt_decoder(z_input)
        except (ValueError, RuntimeError) as e:
            # Shape mismatch or TRT error - fall back to PyTorch
            logger.warning(f"TRT decode failed ({e}), falling back to PyTorch")
            return self._original_decode(
                z, return_dict=return_dict, tiled=tiled,
                tile_size=tile_size, tile_overlap=tile_overlap
            )
        
        if not return_dict:
            return (decoded,)
        
        return TRTDecoderOutput(sample=decoded)
    
    # Bind the TRT decode method
    import types
    vae_model.decode = types.MethodType(trt_decode, vae_model)
    
    logger.info(f"VAE decoder patched with TensorRT engine: {os.path.basename(engine_path)}")
    return vae_model
