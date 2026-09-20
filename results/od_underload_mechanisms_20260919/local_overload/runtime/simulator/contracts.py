"""Immutable interfaces between client controllers and the event engine.

Snapshots expose only the documented client-visible state. No event queue,
request generator, filesystem input, or strategy implementation is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional


@dataclass(frozen=True)
class ControlRequestView:
    request_id: int
    npu_id: int
    category: str
    per_layer_compute_ms: float
    compute_done_up_to: int
    remaining_layers: int
    next_layer_work_gb_by_ssu: tuple[float, ...]
    waiting_for_io: bool
    remaining_work_gb_by_ssu: tuple[float, ...] = ()
    remaining_compute_budget_ms: float = 0.0
    prefetch_only: bool = False
    # ``arrival_time_ms`` is an observed input timestamp.  The other two
    # fields are deliberately absent for a cross-request Layer-0 prefetch:
    # admission has not happened yet, so assigning it an admission time or an
    # absolute deadline would expose a fictional future fact.  The deadline is
    # specifically for this simulator's admission-to-completion TTFT; callers
    # must use arrival_time_ms separately for an arrival-to-completion SLO.
    admission_time_ms: Optional[float] = None
    hard_deadline_time_ms: Optional[float] = None
    arrival_time_ms: Optional[float] = None


@dataclass(frozen=True)
class CIRControlSnapshot:
    time_ms: float
    evaluation: int
    layer_jobs_since_previous: int
    num_npu: int
    num_ssu: int
    active_requests: tuple[ControlRequestView, ...]
    current_path_cirs_by_ssu: tuple[tuple[float, ...], ...]
    trigger_reasons: tuple[str, ...] = ()
    # One immutable row per SSU and one count per hardware QoS Path.  The
    # scheduler's public pressure API may return a TTL-cached row; the three
    # cumulative counters make that observation cost explicit.
    path_outstanding_io_counts_by_ssu: tuple[tuple[int, ...], ...] = ()
    pressure_queries_cumulative_by_ssu: tuple[int, ...] = ()
    pressure_reads_cumulative_by_ssu: tuple[int, ...] = ()
    pressure_cache_hits_cumulative_by_ssu: tuple[int, ...] = ()


@dataclass(frozen=True)
class CIRControlDecision:
    path_cirs_by_ssu: tuple[tuple[float, ...], ...]


CIRControlCallback = Callable[[CIRControlSnapshot], Optional[CIRControlDecision]]


@dataclass(frozen=True, init=False)
class CIRControlConfig:
    """Evaluate at batch membership changes and optionally on a period.

    One layer-equivalent is one completed batch-layer weighted by its member
    count.  Wall-clock ticks are anchored at the first evaluation.  A causal
    evaluation also runs whenever an active microbatch is admitted or released
    unless ``on_batch_boundary`` is disabled.

    The custom initializer keeps the original positional form
    ``CIRControlConfig(every_layers, callback)`` compatible while also allowing
    ``CIRControlConfig(callback=callback, interval_ms=10.0)``.

    When explicitly set, ``hard_ttft_ideal_multiplier`` configures the absolute
    deadline exported for admitted requests as
    ``admission + multiplier * n_layers * C``.  It defaults to ``None`` so an
    old controller is not silently assigned a new SLO.  The resulting deadline
    is an admission-to-completion budget, matching this simulator's current
    TTFT metric, and is not an arrival-to-completion production latency promise.
    """

    every_layers: Optional[int]
    callback: CIRControlCallback
    interval_ms: Optional[float]
    on_batch_boundary: bool
    min_interval_ms: float
    hard_ttft_ideal_multiplier: Optional[float]

    def __init__(
        self,
        every_layers=None,
        callback=None,
        *,
        interval_ms=None,
        on_batch_boundary=True,
        min_interval_ms=0.0,
        hard_ttft_ideal_multiplier=None,
    ):
        if callback is None:
            raise ValueError("a CIR control callback is required")
        layer_mode = every_layers is not None
        interval_mode = interval_ms is not None
        if layer_mode and interval_mode:
            raise ValueError("set at most one of every_layers and interval_ms")
        if not layer_mode and not interval_mode and not on_batch_boundary:
            raise ValueError("the CIR controller has no evaluation trigger")
        if layer_mode and int(every_layers) <= 0:
            raise ValueError("every_layers must be positive")
        if interval_mode and float(interval_ms) <= 0.0:
            raise ValueError("interval_ms must be positive")
        if float(min_interval_ms) < 0.0:
            raise ValueError("min_interval_ms cannot be negative")
        if hard_ttft_ideal_multiplier is not None:
            hard_ttft_ideal_multiplier = float(hard_ttft_ideal_multiplier)
            if (
                not math.isfinite(hard_ttft_ideal_multiplier)
                or hard_ttft_ideal_multiplier <= 0.0
            ):
                raise ValueError(
                    "hard_ttft_ideal_multiplier must be positive and finite"
                )
        object.__setattr__(
            self, "every_layers", int(every_layers) if layer_mode else None
        )
        object.__setattr__(self, "callback", callback)
        object.__setattr__(
            self, "interval_ms", float(interval_ms) if interval_mode else None
        )
        object.__setattr__(self, "on_batch_boundary", bool(on_batch_boundary))
        object.__setattr__(self, "min_interval_ms", float(min_interval_ms))
        object.__setattr__(
            self,
            "hard_ttft_ideal_multiplier",
            hard_ttft_ideal_multiplier,
        )


@dataclass(frozen=True)
class CausalLayerObservation:
    """Only facts available after one microbatch layer has completed I/O."""

    batch_id: int
    npu_id: int
    observed_layer: int
    observed_work_gb_by_ssu: tuple[float, ...]
    compute_budget_ms: float
    manifest_layer0: bool = False


@dataclass(frozen=True)
class CausalLayerSnapshot:
    """Deployment-visible input; deliberately contains no simulator clock."""

    num_npu: int
    num_ssu: int
    active_batches: tuple[CausalLayerObservation, ...]


CausalLayerControlCallback = Callable[
    [CausalLayerSnapshot], Optional[CIRControlDecision]
]


@dataclass(frozen=True)
class CausalLayerControlConfig:
    callback: CausalLayerControlCallback


def main():
    from dataclasses import FrozenInstanceError
    callback = lambda snapshot: None
    config = CIRControlConfig(callback=callback, interval_ms=5, min_interval_ms=100)
    assert config.interval_ms == 5 and config.min_interval_ms == 100
    assert CIRControlConfig(2, callback).every_layers == 2
    try:
        config.interval_ms = 6
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("control contracts must be immutable")
    try:
        CIRControlConfig(2, callback, interval_ms=5)
    except ValueError:
        pass
    else:
        raise AssertionError("layer and wall-clock triggers must not be combined")
    print("contracts: PASS (immutable records and compatible control configuration)")


if __name__ == "__main__":
    main()
