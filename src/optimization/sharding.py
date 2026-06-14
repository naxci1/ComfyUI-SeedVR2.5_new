"""
Extensible multi-GPU sharding plan for SeedVR2 DiT inference.

This module adds the *logic* required to shard DiT inference across multiple
GPUs without committing the pipeline to a multi-process rewrite. It extends the
project's existing context-parallel (sequence-parallel) primitives in
``src/common/distributed`` rather than introducing a parallel mechanism, so a
second GPU can be brought online later by launching the existing entrypoint
under ``torchrun`` -- no structural changes required.

Behaviour today (single GPU):

* :func:`plan_dit_sharding` detects the available devices and, for a single
  GPU, returns a *disabled* plan whose shard/gather helpers are exact identity
  passthroughs. The hot path is therefore byte-for-byte unchanged and the GGUF
  inference remains the master source of truth.

Behaviour when a second GPU is added:

* Launched as ``torchrun --nproc_per_node=N``, ``WORLD_SIZE`` is > 1, the plan
  becomes *enabled*, and :func:`initialize_sharding` wires up a context-parallel
  group sized to the world via :func:`init_sequence_parallel`. The DiT attention
  blocks already consume these groups through ``gather_seq_scatter_heads_qkv`` /
  ``gather_heads_scatter_seq``, so inference is load-balanced across the cohort.

Stability constraints:

* No threads, processes or background workers are ever spawned here. Multi-GPU
  fan-out is delegated to ``torch.distributed`` collectives on the existing
  process group, so nothing can contend with the FFMPEG stderr-drain pipe.
* Every collective is guarded; if ``torch.distributed`` is not initialized the
  helpers degrade to single-device passthroughs.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..utils.debug import Debug


@dataclass
class DiTShardingPlan:
    """Resolved description of how DiT inference is distributed across GPUs.

    Attributes:
        world_size: Number of cooperating processes/GPUs (1 == single GPU).
        device_ids: Visible CUDA device indices considered for the plan.
        sequence_parallel_size: Context-parallel degree to request. Equal to
            ``world_size`` when sharding is active, otherwise 1.
        enabled: True only when more than one process participates AND
            ``torch.distributed`` is available. Single GPU -> always False.
        reason: Human-readable explanation of why sharding is on/off, for logs.
    """

    world_size: int = 1
    device_ids: List[int] = field(default_factory=lambda: [0])
    sequence_parallel_size: int = 1
    enabled: bool = False
    reason: str = "single GPU"

    @property
    def is_master_source(self) -> bool:
        """True on the rank that owns the canonical (GGUF) master output.

        Rank 0 always assembles and writes the final video; other ranks only
        contribute sharded compute. With sharding disabled this is always True.
        """
        if not self.enabled:
            return True
        return _safe_get_rank() == 0


def _safe_get_rank() -> int:
    """Return the distributed rank, or 0 when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _detect_world_size() -> int:
    """Resolve the participating world size from the distributed env / launcher."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))


def plan_dit_sharding(
    requested_gpus: Optional[int] = None,
    debug: Optional["Debug"] = None,
) -> DiTShardingPlan:
    """Build a sharding plan that is a no-op on one GPU and ready for more.

    Args:
        requested_gpus: Optional cap on the number of GPUs to use. ``None`` lets
            the launcher (``WORLD_SIZE``) decide. ``1`` forces single-GPU.
        debug: Optional Debug instance for logging the resolved plan.

    Returns:
        A :class:`DiTShardingPlan`. The plan is only ``enabled`` when more than
        one process participates and ``torch.distributed`` is initialized, which
        guarantees the single-GPU hot path is unchanged.
    """
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_ids = list(range(visible)) if visible > 0 else [0]

    world_size = _detect_world_size()
    if requested_gpus is not None and requested_gpus > 0:
        world_size = min(world_size, requested_gpus)

    dist_ready = False
    try:
        import torch.distributed as dist

        dist_ready = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_ready = False

    if world_size > 1 and dist_ready:
        plan = DiTShardingPlan(
            world_size=world_size,
            device_ids=device_ids[:world_size] if visible >= world_size else device_ids,
            sequence_parallel_size=world_size,
            enabled=True,
            reason=f"context-parallel sharding across {world_size} ranks",
        )
    else:
        if world_size > 1 and not dist_ready:
            reason = (
                "multiple ranks requested but torch.distributed is not "
                "initialized; running single-GPU"
            )
        else:
            reason = "single GPU"
        plan = DiTShardingPlan(
            world_size=1,
            device_ids=device_ids[:1],
            sequence_parallel_size=1,
            enabled=False,
            reason=reason,
        )

    if debug is not None:
        try:
            debug.log(
                f"DiT sharding plan: enabled={plan.enabled}, world_size={plan.world_size}, "
                f"sequence_parallel_size={plan.sequence_parallel_size} ({plan.reason})",
                category="device",
            )
        except Exception:
            pass

    return plan


def initialize_sharding(plan: DiTShardingPlan, debug: Optional["Debug"] = None) -> bool:
    """Wire up the context-parallel group described by ``plan``.

    For a disabled (single-GPU) plan this is a no-op that returns False. For an
    enabled plan it configures the sequence-parallel group that the DiT
    attention blocks already consume, by reusing :func:`init_sequence_parallel`.

    Returns:
        True if a context-parallel group was initialized, False otherwise.
    """
    if not plan.enabled:
        return False

    try:
        from ..common.distributed.advanced import (
            get_sequence_parallel_world_size,
            init_sequence_parallel,
        )

        # Idempotent: skip if a group of the desired size already exists.
        if get_sequence_parallel_world_size() == plan.sequence_parallel_size:
            return True

        init_sequence_parallel(plan.sequence_parallel_size)
        if debug is not None:
            try:
                debug.log(
                    f"Initialized context-parallel group of size "
                    f"{plan.sequence_parallel_size}",
                    category="device",
                )
            except Exception:
                pass
        return True
    except Exception as exc:  # pragma: no cover - requires multi-GPU runtime
        if debug is not None:
            try:
                debug.log(
                    f"Sharding initialization failed, falling back to single GPU: {exc}",
                    level="WARNING",
                    category="device",
                    force=True,
                )
            except Exception:
                pass
        return False


def shard_sequence(tensor: torch.Tensor, plan: DiTShardingPlan, seq_dim: int = 0) -> torch.Tensor:
    """Scatter a sequence-dim tensor across the context-parallel group.

    Identity passthrough when sharding is disabled. When enabled it reuses the
    project's existing context-parallel scatter so callers never reimplement the
    collective.
    """
    if not plan.enabled:
        return tensor
    try:
        from ..common.distributed.ops import slice_inputs

        return slice_inputs(tensor, dim=seq_dim)
    except Exception:  # pragma: no cover - requires multi-GPU runtime
        return tensor


def gather_sequence(tensor: torch.Tensor, plan: DiTShardingPlan, seq_dim: int = 0) -> torch.Tensor:
    """Gather a sequence-dim tensor back from the context-parallel group.

    Identity passthrough when sharding is disabled, mirroring
    :func:`shard_sequence`.
    """
    if not plan.enabled:
        return tensor
    try:
        from ..common.distributed.ops import gather_outputs

        return gather_outputs(tensor, gather_dim=seq_dim)
    except Exception:  # pragma: no cover - requires multi-GPU runtime
        return tensor
