# Conservative TensorRT NVFP4 engine builder and utilities
# Safe: will not crash if TensorRT is absent. Attempts to enable NVFP4 only if the
# installed TensorRT exposes a suitable BuilderFlag.
#
# Note: NVFP4 support is new and vendor-specific; this builder is a scaffold.
# You will likely need to adapt optimization profiles, input shapes, and
# model export hooks to match how your model is represented in ONNX.
#
# Usage:
#   from src.runtime.tensorrt.nvfp4_engine import build_engine_from_onnx, is_trt_available
#   if is_trt_available():
#       build_engine_from_onnx("model.onnx", "engines/nvfp4.plan")
#
import os
import logging
from typing import Optional

log = logging.getLogger("nvfp4_engine")
log.setLevel(logging.INFO)

try:
    import tensorrt as trt  # type: ignore
    TRT_AVAILABLE = True
except Exception:
    trt = None  # type: ignore
    TRT_AVAILABLE = False


def is_trt_available() -> bool:
    return TRT_AVAILABLE


def get_trt_version() -> Optional[str]:
    if not TRT_AVAILABLE:
        return None
    try:
        return trt.__version__  # type: ignore
    except Exception:
        return None


def check_nvfp4_builderflag() -> bool:
    """
    Returns True if TensorRT exposes a BuilderFlag for NVFP4 (best-effort).
    Newer TRT versions may provide trt.BuilderFlag.NVFP4 or similar.
    """
    if not TRT_AVAILABLE:
        return False
    try:
        # Best-effort: check if such a flag exists
        return hasattr(trt.BuilderFlag, "NVFP4")  # type: ignore
    except Exception:
        return False


def build_engine_from_onnx(
    onnx_path: str,
    plan_path: str,
    max_workspace_size: int = 4 << 30,
    max_batch_size: int = 1,
    enable_nvfp4: bool = True,
) -> None:
    """
    Build a TensorRT engine (.plan) from ONNX. This is a conservative builder:
    - Parses ONNX using trt.OnnxParser
    - Creates builder & config, sets workspace
    - Enables NVFP4 if available and requested
    - Serializes engine to plan_path

    Important:
    - The builder assumes a single optimization profile with implicit batch (or static shapes).
    - You must adapt this function if your model needs dynamic shapes or optimization profiles.

    Args:
      onnx_path: path to ONNX file
      plan_path: output path for serialized engine (.plan)
      enable_nvfp4: attempt to enable NVFP4 mode when supported by TRT
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available in this Python environment")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"onnx file not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)  # change to INFO/ERROR as needed

    with trt.Builder(logger) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = max_workspace_size
        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    log.error("TensorRT ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size

        # Attempt to enable NVFP4 if requested and supported
        if enable_nvfp4 and check_nvfp4_builderflag():
            try:
                nvflag = getattr(trt.BuilderFlag, "NVFP4")
                config.set_flag(nvflag)  # type: ignore[arg-type]
                log.info("Enabled TensorRT BuilderFlag.NVFP4")
            except Exception:
                log.warning("Unable to set NVFP4 flag despite detection; proceeding without NVFP4")
        else:
            # Fallback: if NVFP4 unavailable, try FP16
            if hasattr(trt.BuilderFlag, "FP16"):
                try:
                    config.set_flag(trt.BuilderFlag.FP16)
                    log.info("NVFP4 not available; enabled FP16 fallback")
                except Exception:
                    log.warning("Failed to enable FP16 fallback")

        # (Optional) Builder config flags for safe building
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # enable strict types if desired

        # Create optimization profile if dynamic dims are required; this scaffold assumes static shapes.
        # Build the engine
        log.info("Building TensorRT engine (this may take time)...")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT builder failed to create an engine")

        # Serialize engine
        serialized = engine.serialize()
        os.makedirs(os.path.dirname(plan_path) or ".", exist_ok=True)
        with open(plan_path, "wb") as f:
            f.write(serialized)
        log.info("Saved TensorRT engine to %s", plan_path)


def load_engine_from_file(plan_path: str):
    """
    Load a serialized engine (.plan) and return a runtime and engine object pair.
    Caller must create an execution context as needed.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available in this Python environment")

    if not os.path.exists(plan_path):
        raise FileNotFoundError(plan_path)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        return engine  # runtime not returned because runtime must live during engine use
