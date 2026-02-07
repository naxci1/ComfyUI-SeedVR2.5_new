#!/usr/bin/env python3
"""
SeedVR2 VAE Decoder → TensorRT Engine Export Script

Loads the SeedVR2 EMA VAE (ema_vae_fp16.safetensors), exports the decoder to ONNX,
and builds a TensorRT engine optimized for Blackwell GPUs with FP8 precision.

Requirements:
    pip install safetensors torch onnx tensorrt

Usage:
    python scripts/export_vae_trt.py \
        --vae-path /path/to/ema_vae_fp16.safetensors \
        --output-dir ./trt_engines \
        --latent-height 135 \
        --latent-width 240 \
        --fp8

The default latent shape [1, 4, 135, 240] corresponds to 1080p output (1080×1920)
with the SeedVR2 VAE spatial downsample factor of 8.

Note: This script exports the decoder portion of the VAE only. The encoder is not
needed for the inference hot path (Phase 3: VAE Decoding).
"""

import argparse
import os
import sys
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_vae_model(vae_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load the SeedVR2 VAE model from safetensors checkpoint.
    
    Instantiates the VideoAutoencoderKLWrapper with the standard SeedVR2 config,
    loads weights, and returns the model in eval mode.
    
    Args:
        vae_path: Path to ema_vae_fp16.safetensors
        device: Target device (default: cuda)
        
    Returns:
        Loaded VAE model on the specified device
    """
    from safetensors.torch import load_file
    
    # Add the repo root to sys.path so we can import the model
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    
    # SeedVR2 VAE config from s8_c16_t4_inflation_sd3.yaml
    vae_config = dict(
        act_fn="silu",
        block_out_channels=[128, 256, 512, 512],
        down_block_types=["DownEncoderBlock3D"] * 4,
        in_channels=3,
        latent_channels=16,
        layers_per_block=2,
        norm_num_groups=32,
        out_channels=3,
        slicing_sample_min_size=4,
        temporal_scale_num=2,
        inflation_mode="pad",
        up_block_types=["UpDecoderBlock3D"] * 4,
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        freeze_encoder=False,
    )
    
    logger.info(f"Instantiating VideoAutoencoderKLWrapper...")
    model = VideoAutoencoderKLWrapper(**vae_config)
    
    logger.info(f"Loading weights from: {vae_path}")
    state_dict = load_file(vae_path, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    logger.info(f"VAE model loaded on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    
    return model


class VAEDecoderWrapper(torch.nn.Module):
    """
    Thin wrapper that exposes only the decode path of the VAE for ONNX export.
    
    The full VAE model has encoder + decoder + distribution logic.
    For TensorRT, we only need the decoder (post_quant_conv → decoder → output).
    This wrapper provides a clean forward() that takes latent input and returns
    decoded pixels directly.
    """
    
    def __init__(self, vae_model: torch.nn.Module):
        super().__init__()
        # Extract decoder components from the full VAE
        self.decoder = vae_model.decoder
        self.post_quant_conv = vae_model.post_quant_conv
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to pixel space.
        
        Args:
            z: Latent tensor of shape [B, C, H, W] where C=16 (latent_channels)
            
        Returns:
            Decoded pixel tensor of shape [B, 3, H*8, W*8]
        """
        # The decoder uses 3D (causal) convolutions internally, so it expects
        # 5D input [B, C, T, H, W] even for single-frame decoding. We add a
        # singleton temporal dimension T=1 for the forward pass and remove it
        # from the output to keep the ONNX graph operating on 4D tensors.
        z = z.unsqueeze(2)  # [B, C, 1, H, W]
        
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        
        decoded = self.decoder(z)
        
        # Remove temporal dimension
        decoded = decoded.squeeze(2)  # [B, 3, H*8, W*8]
        
        return decoded


def _force_half_precision(model: torch.nn.Module) -> torch.nn.Module:
    """
    Force all parameters and biases to FP16.
    
    The safetensors checkpoint may have FP16 weights but FP32 biases in some layers
    (e.g., GroupNorm, Conv3d), causing 'Input type (c10::Half) and bias type (float)
    should be the same' errors during F.conv3d in causal_inflation_lib.py.
    
    This function ensures uniform FP16 precision across the entire model.
    
    Args:
        model: Model to cast to FP16
        
    Returns:
        Model with all parameters and biases in FP16
    """
    model = model.half()
    
    # Explicitly cast any biases that may not have been converted by .half()
    # (e.g., registered buffers or non-persistent parameters)
    for m in model.modules():
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data = m.bias.data.half()
    
    return model


def export_to_onnx(
    vae_model: torch.nn.Module,
    onnx_path: str,
    latent_shape: tuple = (1, 16, 135, 240),
    device: str = "cuda",
) -> str:
    """
    Export VAE decoder to ONNX format with static input shape.
    
    Args:
        vae_model: Full VAE model (VideoAutoencoderKLWrapper)
        onnx_path: Output path for the ONNX file
        latent_shape: Static input shape [B, C, H, W]
        device: Device for dummy input generation
        
    Returns:
        Path to the exported ONNX file
    """
    logger.info(f"Creating decoder wrapper for ONNX export...")
    decoder = VAEDecoderWrapper(vae_model).to(device).eval()
    
    # Force all parameters and biases to FP16 to prevent type mismatch errors
    # during F.conv3d in the Causal Conv3D layers (causal_inflation_lib.py)
    decoder = _force_half_precision(decoder)
    
    # Disable CuDNN benchmarking to force standard convolution ops.
    # Without this, PyTorch routes convolutions through aten.cudnn_convolution
    # which the ONNX exporter cannot decompose, causing:
    #   DispatchError: No ONNX function found for <OpOverload(op='aten.cudnn_convolution')>
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    dummy_input = torch.randn(*latent_shape, device=device, dtype=torch.float16)
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    logger.info(f"  Input shape: {latent_shape}")
    logger.info(f"  Expected output: [{latent_shape[0]}, 3, {latent_shape[2] * 8}, {latent_shape[3] * 8}]")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                dummy_input,
                onnx_path,
                input_names=["latent"],
                output_names=["decoded"],
                opset_version=17,
                # Embed weights directly in the ONNX graph. This uses the legacy
                # torch.onnx exporter path, avoiding torch.export which fails on Windows.
                export_params=True,
                do_constant_folding=True,
                # Use ATEN fallback for complex ops (Causal Conv3D, custom padding)
                # that the standard ONNX exporter cannot trace on Windows
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
    finally:
        # Restore CuDNN settings
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic
    
    logger.info(f"ONNX export complete: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    return onnx_path


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    enable_fp8: bool = True,
    enable_fp16: bool = True,
    workspace_gb: int = 4,
) -> str:
    """
    Build a TensorRT engine from ONNX model.
    
    Uses the TensorRT Python API to build an optimized engine with optional
    FP8 (e4m3fn) precision for Blackwell Tensor Cores.
    
    Args:
        onnx_path: Path to input ONNX file
        engine_path: Path for output .engine file
        enable_fp8: Enable FP8 precision (requires Blackwell/Hopper GPU)
        enable_fp16: Enable FP16 precision
        workspace_gb: GPU memory workspace limit in GB
        
    Returns:
        Path to the built engine file
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "TensorRT is not installed. Install with:\n"
            "  pip install tensorrt\n"
            "  # Or for specific version:\n"
            "  pip install nvidia-tensorrt"
        )
        raise
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    logger.info(f"Building TensorRT engine from: {onnx_path}")
    logger.info(f"  FP16: {enable_fp16}, FP8: {enable_fp8}")
    logger.info(f"  Workspace: {workspace_gb} GB")
    
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    logger.info(f"ONNX parsed: {network.num_inputs} inputs, {network.num_outputs} outputs")
    
    # Configure precision
    if enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("  Enabled FP16 precision")
    
    if enable_fp8:
        # FP8 support requires TensorRT 9.0+ and Blackwell/Hopper GPU
        if hasattr(trt.BuilderFlag, 'FP8'):
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("  Enabled FP8 (e4m3fn) precision for Blackwell Tensor Cores")
        else:
            logger.warning(
                "FP8 not available in this TensorRT version. "
                "Requires TensorRT 9.0+ with Blackwell/Hopper GPU. "
                "Falling back to FP16 only."
            )
    
    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    
    # Build engine
    logger.info("Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    logger.info(f"TensorRT engine saved: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")
    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Export SeedVR2 VAE Decoder to TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        required=True,
        help="Path to ema_vae_fp16.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trt_engines",
        help="Output directory for ONNX and engine files (default: ./trt_engines)",
    )
    parser.add_argument(
        "--latent-height",
        type=int,
        default=135,
        help="Latent height (default: 135 for 1080p)",
    )
    parser.add_argument(
        "--latent-width",
        type=int,
        default=240,
        help="Latent width (default: 240 for 1080p/1920px)",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=False,
        help="Enable FP8 (e4m3fn) precision for Blackwell/Hopper GPUs",
    )
    parser.add_argument(
        "--workspace-gb",
        type=int,
        default=4,
        help="GPU workspace limit in GB (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model loading (default: cuda)",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (reuse existing ONNX file)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.vae_path):
        logger.error(f"VAE checkpoint not found: {args.vae_path}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File paths
    # Note: latent_channels=16 (not 4) for SeedVR2 VAE
    latent_shape = (1, 16, args.latent_height, args.latent_width)
    shape_tag = f"{args.latent_height}x{args.latent_width}"
    precision_tag = "fp8" if args.fp8 else "fp16"
    
    onnx_path = os.path.join(args.output_dir, f"seedvr2_vae_decoder_{shape_tag}.onnx")
    engine_path = os.path.join(args.output_dir, f"seedvr2_vae_decoder_{shape_tag}_{precision_tag}.engine")
    
    # Step 1: Load VAE model
    logger.info("=" * 60)
    logger.info("Step 1: Loading VAE model")
    logger.info("=" * 60)
    vae_model = load_vae_model(args.vae_path, device=args.device)
    
    # Step 2: Export to ONNX
    if not args.skip_onnx:
        logger.info("=" * 60)
        logger.info("Step 2: Exporting decoder to ONNX")
        logger.info("=" * 60)
        export_to_onnx(vae_model, onnx_path, latent_shape=latent_shape, device=args.device)
    else:
        logger.info(f"Skipping ONNX export, using existing: {onnx_path}")
    
    # Free VAE model from GPU
    del vae_model
    torch.cuda.empty_cache()
    
    # Step 3: Build TensorRT engine
    logger.info("=" * 60)
    logger.info("Step 3: Building TensorRT engine")
    logger.info("=" * 60)
    build_trt_engine(
        onnx_path,
        engine_path,
        enable_fp8=args.fp8,
        enable_fp16=True,
        workspace_gb=args.workspace_gb,
    )
    
    logger.info("=" * 60)
    logger.info("DONE!")
    logger.info(f"  ONNX:   {onnx_path}")
    logger.info(f"  Engine: {engine_path}")
    logger.info("")
    logger.info("To use this engine in SeedVR2:")
    logger.info("  from src.optimization.trt_decoder import patch_vae_with_trt")
    logger.info(f"  patch_vae_with_trt(vae_model, engine_path='{engine_path}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
