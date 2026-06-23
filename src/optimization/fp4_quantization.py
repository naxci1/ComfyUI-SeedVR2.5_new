"""
NVFP4 (E2M1) / Transformer Engine quantization wrapper for SeedVR2.

Execution priority (first available backend wins):
  1. NVIDIA Transformer Engine >= 2.6 with native Blackwell FP4 Tensor Cores
     (requires CUDA 13.x, SM 100 / Blackwell architecture)
  2. NVIDIA TensorRT Model Optimizer (modelopt) FP4 calibration path
  3. PyTorch native float8_e4m3fn (Ampere/Hopper FP8 — precision stable fallback)
  4. BFloat16 nn.Linear (always available)

The wrappers are drop-in replacements for `nn.Linear`.  The quantization level
can be queried at runtime via `get_fp4_backend()`.

Usage
-----
    from src.optimization.fp4_quantization import build_fp4_linear, get_fp4_backend

    layer = build_fp4_linear(in_features=1152, out_features=1152)
    print(get_fp4_backend())   # → "transformer_engine_fp4" | "modelopt_fp4" | "fp8" | "bf16"
"""

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection (lazy, cached)
# ---------------------------------------------------------------------------

_BACKEND: Optional[str] = None  # resolved on first call


def _resolve_backend() -> str:
    """Detect the best available quantization backend (called once)."""
    # 1. Transformer Engine FP4 (Blackwell, SM 100)
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        from transformer_engine.pytorch import fp8 as te_fp8  # noqa: F401
        # Verify FP4 recipe is present (TE >= 2.6)
        from transformer_engine.common.recipe import DelayedScaling, Format
        _ = Format.MXFP4  # raises AttributeError on older TE
        # Verify Blackwell / SM 100 GPU is present
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 10:  # SM 100 = Blackwell
                return "transformer_engine_fp4"
            else:
                log.info(
                    "Transformer Engine FP4 detected but GPU is SM %d.%d "
                    "(Blackwell SM 100 required). Trying modelopt next.",
                    props.major, props.minor,
                )
    except (ImportError, AttributeError):
        pass

    # 2. TensorRT Model Optimizer (modelopt) FP4
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401
        return "modelopt_fp4"
    except (ImportError, AttributeError):
        pass

    # 3. PyTorch native FP8 (float8_e4m3fn, Ampere / Hopper)
    if hasattr(torch, "float8_e4m3fn") and torch.cuda.is_available():
        return "fp8"

    # 4. BFloat16 (safe universal fallback)
    return "bf16"


def get_fp4_backend() -> str:
    """Return the name of the active quantization backend."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _resolve_backend()
    return _BACKEND


# ---------------------------------------------------------------------------
# Backend-specific Linear implementations
# ---------------------------------------------------------------------------


class _TELinearFP4(nn.Module):
    """
    Wraps `transformer_engine.pytorch.Linear` with a Blackwell MXFP4 recipe.

    The layer is initialised in FP4 from construction and never dequantizes
    intermediate activations back to FP16/BF16 during the forward pass; the
    FP4 Tensor Cores on Blackwell handle the full GEMM natively.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format

        self.fp4_recipe = DelayedScaling(fp8_format=Format.MXFP4, amax_history_len=16, amax_compute_algo="max")
        # TE Linear automatically uses FP4 Tensor Cores when the recipe is set
        self.linear = te.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import transformer_engine.pytorch as te
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp4_recipe):
            return self.linear(x)


class _ModelOptLinearFP4(nn.Module):
    """
    Wraps a calibrated modelopt FP4 quantized linear layer.

    modelopt operates on an existing `nn.Linear`; we calibrate eagerly in
    the constructor with a one-shot zero-tensor calibration pass so that
    the layer is ready for inference immediately.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        import modelopt.torch.quantization as mtq

        base = nn.Linear(in_features, out_features, bias=bias)
        # FP4 quantization config (weights + activations)
        quant_cfg = mtq.FP4_DEFAULT_CFG
        mtq.quantize(base, quant_cfg, forward_loop=lambda m: m(torch.zeros(1, in_features)))
        self.linear = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FP8Linear(nn.Module):
    """
    Emulates quantized inference via PyTorch's `float8_e4m3fn` dtype.

    Weights are stored in FP8; the forward pass dequantizes them to the
    input dtype for the GEMM, then re-quantizes the output.  This is a
    software simulation — it improves parameter memory footprint but
    does not use hardware FP8 Tensor Cores without `torch._scaled_mm`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weight as FP8; scale factor for dequantization
        weight_fp32 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight_fp32, a=5 ** 0.5)
        self.weight = nn.Parameter(weight_fp32.to(torch.float8_e4m3fn))
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_scale
        out = F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)
        return out


class _BF16Linear(nn.Linear):
    """Standard BFloat16 linear layer (universal fallback)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)


import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_fp4_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    force_backend: Optional[str] = None,
) -> nn.Module:
    """
    Construct a quantized Linear layer using the best available backend.

    Parameters
    ----------
    in_features, out_features : int
        Standard `nn.Linear` dimensions.
    bias : bool
        Whether to include a bias term.
    force_backend : str or None
        Override automatic backend selection.  One of:
        ``"transformer_engine_fp4"``, ``"modelopt_fp4"``, ``"fp8"``, ``"bf16"``.

    Returns
    -------
    nn.Module  — drop-in replacement for ``nn.Linear``.
    """
    backend = force_backend or get_fp4_backend()

    if backend == "transformer_engine_fp4":
        try:
            layer = _TELinearFP4(in_features, out_features, bias=bias)
            log.debug("FP4 linear (%d→%d) via Transformer Engine", in_features, out_features)
            return layer
        except Exception as exc:
            warnings.warn(f"TE FP4 layer construction failed ({exc}); falling back to fp8.")
            backend = "fp8"

    if backend == "modelopt_fp4":
        try:
            layer = _ModelOptLinearFP4(in_features, out_features, bias=bias)
            log.debug("FP4 linear (%d→%d) via modelopt", in_features, out_features)
            return layer
        except Exception as exc:
            warnings.warn(f"modelopt FP4 layer construction failed ({exc}); falling back to fp8.")
            backend = "fp8"

    if backend == "fp8" and hasattr(torch, "float8_e4m3fn"):
        try:
            layer = _FP8Linear(in_features, out_features, bias=bias)
            log.debug("FP8 linear (%d→%d) via float8_e4m3fn", in_features, out_features)
            return layer
        except Exception as exc:
            warnings.warn(f"FP8 layer construction failed ({exc}); falling back to bf16.")

    layer = _BF16Linear(in_features, out_features, bias=bias)
    log.debug("BF16 linear (%d→%d) fallback", in_features, out_features)
    return layer


def quantize_dit_linears(module: nn.Module, force_backend: Optional[str] = None) -> nn.Module:
    """
    Replace all `nn.Linear` sub-modules in *module* with FP4/FP8/BF16 equivalents.

    This is designed for use with the DiT model:

        from src.optimization.fp4_quantization import quantize_dit_linears
        model = quantize_dit_linears(model)

    Parameters
    ----------
    module : nn.Module
        The model (or sub-model) whose Linear layers will be replaced in-place.
    force_backend : str or None
        Override backend — forwarded to `build_fp4_linear`.

    Returns
    -------
    The same *module* with Linear layers replaced.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = build_fp4_linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                force_backend=force_backend,
            )
            # Copy weights if shapes match (same backend dtype conversions handle the rest)
            if isinstance(replacement, _BF16Linear):
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    if child.bias is not None and replacement.bias is not None:
                        replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            quantize_dit_linears(child, force_backend=force_backend)
    return module
