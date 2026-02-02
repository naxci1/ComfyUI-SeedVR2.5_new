#!/usr/bin/env python3
"""
convert_nvfp4_to_trt.py

Helper script that attempts to:
  1) Materialize the model using the project's model loader (PyTorch)
  2) Export a minimal inference wrapper to ONNX
  3) Optionally call the TensorRT builder to create a .plan

This script is intentionally conservative: exporting a complex diffusion model to ONNX
is non-trivial and often requires per-model operator support or model surgery. Use this
as a starting point and adapt the export function to your model.

Usage example (server):
  python tools/convert_nvfp4_to_trt.py \
    --checkpoint /path/to/seedvr2_ema_3b_nvfp4_native.safetensors \
    --onnx-path /tmp/seedvr2_nvfp4.onnx \
    --plan-path /tmp/seedvr2_nvfp4.plan \
    --sample-shape "1,3,256,256"

Notes:
  - This script attempts to import the repo's model loader. If you keep the repo's root
    on PYTHONPATH or run it from repo root it should import correctly.
  - You will probably need to adapt `export_model_to_onnx` to match the model's forward()
    signature and acceptable dummy inputs.
"""
import argparse
import logging
import os
import sys
from typing import Tuple

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("convert_nvfp4")

try:
    import torch
except Exception:
    torch = None

# Try to import the project's model loader. Assumes running from repo root.
# If your loader path differs adjust import below.
try:
    # Example: from src.core.model_loader import materialize_model
    from src.core.model_loader import materialize_model  # type: ignore
    HAS_PROJECT_LOADER = True
except Exception as e:
    log.warning("Could not import project model loader: %s", e)
    HAS_PROJECT_LOADER = False


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split(","))


def export_model_to_onnx(checkpoint: str, onnx_path: str, sample_shape: Tuple[int, ...]):
    """
    1) Use the project's model loader (if available) to create a PyTorch nn.Module
    2) Prepare a dummy input with sample_shape and run torch.onnx.export

    This is a generic routine — you will usually need to adapt it for your
    model architecture (inputs, additional arguments, stateful modules).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export")

    if not HAS_PROJECT_LOADER:
        raise RuntimeError("Project model loader unavailable. Adapt this script to load your model.")

    # Example pseudocode - adapt to your project's materialize_model arguments:
    log.info("Materializing model via project loader (may load weights / dequantization)")
    model = materialize_model(checkpoint_path=checkpoint, device="cuda:0", force_nvfp4=True)  # adapt args
    model.eval()
    model = model.to("cuda")

    # Construct dummy input - adapt shape to the model's expected input (e.g., latents)
    dummy = torch.randn(sample_shape, device="cuda")

    # Tracing/export - set opset_version as required
    log.info("Exporting model to ONNX at %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    log.info("Export completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to NVFP4 checkpoint (safetensors or other)")
    parser.add_argument("--onnx-path", required=True, help="Path to write ONNX model")
    parser.add_argument("--plan-path", required=False, help="Optional: path to write TRT plan (calls nvfp4_engine)")
    parser.add_argument("--sample-shape", default="1,3,256,256", help="Comma-separated shape for dummy input")
    parser.add_argument("--force-nvfp4", action="store_true", help="Pass force_nvfp4 flag to loader if supported")
    args = parser.parse_args()

    sample_shape = parse_shape(args.sample_shape)

    export_model_to_onnx(args.checkpoint, args.onnx_path, sample_shape)

    if args.plan_path:
        # build via nvfp4_engine if available
        try:
            from src.runtime.tensorrt.nvfp4_engine import build_engine_from_onnx  # type: ignore
        except Exception as e:
            log.error("nvfp4_engine not importable: %s", e)
            sys.exit(1)

        log.info("Building TRT engine from ONNX")
        build_engine_from_onnx(args.onnx_path, args.plan_path, enable_nvfp4=True)


if __name__ == "__main__":
    main()
