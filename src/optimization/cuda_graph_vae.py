"""
CUDA Graph helpers for SeedVR2 VAE and DiT execution paths.
"""

import logging
import time
from typing import Any, Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _CudaGraphCacheBase:
    def __init__(self, label: str) -> None:
        self.label = label
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_output = None
        self._graph_key: Optional[Tuple] = None
        self._capture_count: int = 0

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")

    def run(
        self,
        fn: Callable[..., torch.Tensor],
        inputs: Tuple[torch.Tensor, ...],
        *,
        key_extra: Tuple = (),
        debug: Optional[Any] = None,
    ):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        key = (
            tuple((tuple(t.shape), t.dtype, str(t.device)) for t in inputs),
            key_extra,
        )

        if key != self._graph_key:
            self._invalidate(debug)
            self._capture(fn, inputs, key, debug)

        t0 = time.perf_counter()
        for static_t, input_t in zip(self._static_inputs, inputs):
            static_t.copy_(input_t)
        self._graph.replay()
        output = self._clone_output(self._static_output)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if debug is not None:
            debug.log(
                f"CUDA Graph replay ({self.label}): {elapsed_ms:.1f} ms",
                category="perf",
                indent_level=2,
            )
        else:
            logger.debug("CUDA Graph replay (%s): %.1f ms", self.label, elapsed_ms)
        return output

    def reset(self) -> None:
        self._invalidate()
        self._capture_count = 0

    def _capture(
        self,
        fn: Callable[..., torch.Tensor],
        inputs: Tuple[torch.Tensor, ...],
        key: Tuple,
        debug: Optional[Any] = None,
    ) -> None:
        shape_str = ", ".join(str(tuple(x.shape)) for x in inputs)
        try:
            warmup_stream = torch.cuda.Stream(device=inputs[0].device)
            with torch.cuda.stream(warmup_stream):
                for _ in range(getattr(self, "warmup_steps", getattr(self, "_warmup_steps", 2))):
                    _ = fn(*inputs)
            torch.cuda.synchronize(inputs[0].device)

            t0 = time.perf_counter()
            self._static_inputs = tuple(x.clone() for x in inputs)
            self._graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream(device=inputs[0].device)
            with torch.cuda.graph(self._graph, stream=capture_stream):
                self._static_output = fn(*self._static_inputs)
            torch.cuda.synchronize(inputs[0].device)
            self._graph_key = key
            self._capture_count += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            msg = (
                f"CUDA Graph captured ({self.label}) #{self._capture_count} in "
                f"{elapsed_ms:.1f} ms for [{shape_str}]"
            )
            if debug is not None:
                debug.log(msg, category="perf", indent_level=2)
            else:
                logger.info(msg)
        except Exception as exc:
            self._invalidate()
            err = f"[SeedVR] CUDA Graph capture failed for {self.label}: {exc!r}"
            logger.warning(err)
            raise RuntimeError(err) from exc

    def _invalidate(self, debug: Optional[Any] = None) -> None:
        if self._graph is not None:
            if debug is not None:
                debug.log(
                    f"CUDA Graph invalidated ({self.label})",
                    category="perf",
                    indent_level=2,
                )
            del self._graph
            self._graph = None
        self._static_inputs = None
        self._static_output = None
        self._graph_key = None

    @staticmethod
    def _clone_output(output: Any) -> Any:
        if torch.is_tensor(output):
            return output.clone()
        if isinstance(output, tuple):
            return tuple(_CudaGraphCacheBase._clone_output(v) for v in output)
        if isinstance(output, list):
            return [_CudaGraphCacheBase._clone_output(v) for v in output]
        return output


class VaeDecodeGraphCache(_CudaGraphCacheBase):
    def __init__(self) -> None:
        super().__init__("vae_decode")

    def run(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
        debug=None,
        graph_group: str = "default",
    ) -> torch.Tensor:
        return super().run(
            decode_fn,
            (latent,),
            key_extra=(graph_group,),
            debug=debug,
        )


class VaeEncodeGraphCache(_CudaGraphCacheBase):
    def __init__(self) -> None:
        super().__init__("vae_encode")

    def run(
        self,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        sample: torch.Tensor,
        debug=None,
        graph_group: str = "default",
    ) -> torch.Tensor:
        return super().run(
            encode_fn,
            (sample,),
            key_extra=(graph_group,),
            debug=debug,
        )


class DitGraphCache(_CudaGraphCacheBase):
    def __init__(self) -> None:
        super().__init__("dit_forward")

    def run(
        self,
        dit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        vid: torch.Tensor,
        timestep: torch.Tensor,
        debug=None,
        graph_group: str = "default",
    ) -> torch.Tensor:
        return super().run(
            dit_fn,
            (vid, timestep),
            key_extra=(graph_group,),
            debug=debug,
        )
