"""
CUDA Graph support for VAE decode in SeedVR2.

Provides opt-in CUDA Graph capture and replay for the VAE decode pass.
When tensor shapes and dtypes are stable across batches, replaying a
pre-captured graph avoids repeated CUDA kernel-launch overhead.

Limitations / requirements:
  • Only active on CUDA devices (no-op on CPU / MPS).
  • Incompatible with tiled decode (tiles can have variable shapes).
  • Requires two warmup runs before the graph is captured.
  • If the latent shape or dtype changes the graph is re-captured automatically.
  • Memory safety: static tensors inside the graph are kept alive for the
    lifetime of the cache object; the object is stored on the runner so it
    is released whenever the runner is released.
"""

import logging
import time
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class VaeDecodeGraphCache:
    """
    Manages a single CUDA Graph for one fixed-shape VAE decode call.

    Usage::

        cache = VaeDecodeGraphCache()
        # 'decode_fn' must be a callable: (latent: Tensor) -> Tensor
        output = cache.run(decode_fn, latent)   # captures on first call, replays after

    The cache stores the graph, static input tensor, and static output tensor.
    It is invalidated (and re-captured) whenever the latent shape, dtype, or
    device changes.
    """

    def __init__(self) -> None:
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        # Key: (shape-tuple, dtype, device-str)
        self._graph_key: Optional[Tuple] = None
        self._capture_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_available() -> bool:
        """Return True when CUDA Graphs can be used on this system."""
        return torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")

    def run(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
        debug=None,
    ) -> torch.Tensor:
        """
        Run *decode_fn* on *latent* using CUDA Graph replay when possible.

        On the **first call** (or whenever the latent shape/dtype/device
        changes) two warmup forward passes are performed and then the graph
        is captured.  Subsequent calls with matching shapes replay the
        graph directly.

        The caller is responsible for any autocast context; if autocast is
        needed it should be active both during capture and, consistently,
        during replay (though replay itself does not re-enter Python-level
        autocast, the captured kernels already encode the chosen precision).

        Args:
            decode_fn: ``(latent) -> sample`` callable (no tiling, CUDA only).
            latent: Preprocessed input tensor on a CUDA device.
            debug: Optional SeedVR2 debug logger for timing/info messages.

        Returns:
            Decoded sample tensor (a clone of the static graph output).
        """
        key = (tuple(latent.shape), latent.dtype, str(latent.device))

        if key != self._graph_key:
            self._invalidate(debug)
            self._capture(decode_fn, latent, key, debug)

        # ------ Replay ------ #
        t0 = time.perf_counter()
        self._static_input.copy_(latent)
        self._graph.replay()
        output = self._static_output.clone()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if debug is not None:
            debug.log(
                f"CUDA Graph replay: {elapsed_ms:.1f} ms "
                f"(shape={latent.shape}, dtype={latent.dtype})",
                category="perf",
                indent_level=2,
            )
        else:
            logger.debug(
                "CUDA Graph replay: %.1f ms (shape=%s, dtype=%s)",
                elapsed_ms,
                latent.shape,
                latent.dtype,
            )

        return output

    def reset(self) -> None:
        """Release the cached graph and static tensors (e.g. after model reload)."""
        self._invalidate()
        self._capture_count = 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _capture(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
        key: Tuple,
        debug=None,
    ) -> None:
        """Warmup then capture a new CUDA Graph for *latent*'s shape/dtype/device."""
        shape_str = str(latent.shape)

        if debug is not None:
            debug.log(
                f"CUDA Graph: warmup (2 runs) for shape {shape_str}, "
                f"dtype={latent.dtype}",
                category="perf",
                indent_level=2,
            )
        else:
            logger.info("CUDA Graph: warmup for shape %s", shape_str)

        # Warmup: run on a side stream so we don't pollute the default stream.
        warmup_stream = torch.cuda.Stream(device=latent.device)
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                _ = decode_fn(latent)
        torch.cuda.synchronize(latent.device)

        # Capture
        if debug is not None:
            debug.log(
                f"CUDA Graph: capturing for shape {shape_str}",
                category="perf",
                indent_level=2,
            )
        else:
            logger.info("CUDA Graph: capturing for shape %s", shape_str)

        t0 = time.perf_counter()
        self._static_input = latent.clone()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=torch.cuda.Stream(device=latent.device)):
            self._static_output = decode_fn(self._static_input)
        torch.cuda.synchronize(latent.device)
        self._graph_key = key
        self._capture_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        msg = (
            f"CUDA Graph: captured graph #{self._capture_count} in "
            f"{elapsed_ms:.1f} ms "
            f"(shape={latent.shape}, dtype={latent.dtype})"
        )
        if debug is not None:
            debug.log(msg, category="perf", indent_level=2)
        else:
            logger.info(msg)

    def _invalidate(self, debug=None) -> None:
        """Free the current graph and static tensors."""
        if self._graph is not None:
            old_shape = self._graph_key[0] if self._graph_key else None
            if debug is not None and old_shape is not None:
                debug.log(
                    f"CUDA Graph: invalidating cached graph "
                    f"(shape {old_shape} no longer matches)",
                    category="perf",
                    indent_level=2,
                )
            else:
                logger.debug("CUDA Graph: invalidating cached graph")

            del self._graph
            self._graph = None
        self._static_input = None
        self._static_output = None
        self._graph_key = None
