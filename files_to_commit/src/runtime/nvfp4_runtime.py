"""
High-level NVFP4 runtime wrapper.
- Tries to load TensorRT engine (.plan) and run inference.
- If engine not available or TRT not present, falls back to a PyTorch model callable.

Usage:
  from src.runtime.nvfp4_runtime import NVFP4Runtime
  rt = NVFP4Runtime(engine_path="engines/nvfp4.plan", pytorch_model=your_torch_module)
  out = rt.infer(input_tensor)
"""
import os
import logging
from typing import Optional

log = logging.getLogger("nvfp4_runtime")
log.setLevel(logging.INFO)

try:
    import torch
except Exception:
    torch = None

try:
    import tensorrt as trt  # type: ignore
    TRT_AVAILABLE = True
except Exception:
    trt = None  # type: ignore
    TRT_AVAILABLE = False

# Local imports guarded to avoid import-time crashes
ENGINE_MODULE_AVAILABLE = True
try:
    from src.runtime.tensorrt.nvfp4_engine import load_engine_from_file, is_trt_available  # type: ignore
except Exception:
    ENGINE_MODULE_AVAILABLE = False


class NVFP4Runtime:
    def __init__(self, engine_path: Optional[str] = None, pytorch_model: Optional[object] = None, device: str = "cuda:0"):
        """
        engine_path: serialized .plan path (preferred for fast TRT execution)
        pytorch_model: a PyTorch nn.Module instance to use as fallback (must be moved to device)
        """
        self.engine_path = engine_path
        self.device = device
        self.pytorch_model = pytorch_model
        self.engine = None
        self.context = None
        self.use_trt = False

        if engine_path and os.path.exists(engine_path) and TRT_AVAILABLE and ENGINE_MODULE_AVAILABLE:
            try:
                log.info("Attempting to load TensorRT engine: %s", engine_path)
                self.engine = load_engine_from_file(engine_path)
                # note: in many TRT workflows you create an execution context and manage CUDA buffers
                self.use_trt = True
                log.info("Loaded TRT engine (NVFP4 path).")
            except Exception as e:
                log.warning("Failed to load TRT engine; falling back to PyTorch: %s", e)
                self.use_trt = False
        else:
            if engine_path:
                log.info("Engine path not found or TRT not available; engine_path=%s TRT_AVAILABLE=%s", engine_path, TRT_AVAILABLE)
            self.use_trt = False

        if self.use_trt:
            # TODO: create execution context and map IO; left as scaffold since exact bindings vary
            # self.context = self.engine.create_execution_context()
            pass
        else:
            if self.pytorch_model is None:
                raise RuntimeError("No TRT engine and no PyTorch model provided as fallback")
            if torch is None:
                raise RuntimeError("PyTorch not available for fallback execution")
            self.pytorch_model.eval()
            try:
                self.pytorch_model.to(self.device)
            except Exception:
                log.warning("Could not move fallback model to device %s", self.device)

    def available(self) -> bool:
        return self.use_trt

    def infer(self, input_tensor):
        """
        Run inference.
        - If TRT engine available: runs via TRT (not implemented here fully; placeholder).
        - Else: runs via PyTorch fallback (synchronous).
        """
        if self.use_trt:
            # Placeholder: you must implement buffer allocation, bindings, and asynchronous CUDA execution
            raise NotImplementedError("TRT inference path is not implemented in scaffold. See NVFP4_RUNBOOK.md for guidance.")
        else:
            # PyTorch fallback
            if torch is None:
                raise RuntimeError("PyTorch is required for fallback inference")
            device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                inp = input_tensor.to(device)
                out = self.pytorch_model(inp)
            return out

# End nvfp4_runtime.py
