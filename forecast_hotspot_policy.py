"""Client-side frozen hotspot forecast for selective SSU CIR materialization.

The forecast consumes only request manifests known before the measurement
window.  It never reads Path pressure.  The returned mask is immutable for one
measurement run so policy quality cannot change the control classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Sequence


DEFAULT_HOT_FRACTION = 0.70
DEFAULT_FORECAST_REQUESTS_PER_NPU = 32


@dataclass(frozen=True)
class FrozenSSUHotspotForecast:
    """Auditable output of the client-side manifest forecast."""

    demand_gbps_by_ssu: tuple[float, ...]
    capacity_gbps_by_ssu: tuple[float, ...]
    load_fraction_by_ssu: tuple[float, ...]
    fleet_load_fraction: float
    hot_fraction: float
    materialized_ssu_mask: tuple[bool, ...]
    classification_by_ssu: tuple[str, ...]
    full_protection_fallback: bool
    forecast_requests_per_npu: int
    input_fingerprint: str


def _caps(spec: Real | Sequence[float], count: int) -> tuple[float, ...]:
    if isinstance(spec, Real):
        values = (float(spec),) * count
    else:
        values = tuple(float(value) for value in spec)
    if len(values) != count:
        raise ValueError("capacity must have one entry per SSU")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("SSU capacities must be finite and positive")
    return values


def forecast_frozen_ssu_hotspots(
    demand_gbps_by_ssu: Sequence[float],
    *,
    ssu_capacity_gbps: Real | Sequence[float] = 40.0,
    hot_fraction: float = DEFAULT_HOT_FRACTION,
    forecast_requests_per_npu: int = DEFAULT_FORECAST_REQUESTS_PER_NPU,
) -> FrozenSSUHotspotForecast:
    """Classify which SSUs need protected floors for one measurement run.

    If aggregate fleet demand already consumes at least ``hot_fraction`` of
    aggregate SSU capacity, all SSUs are protected.  This conservative fallback
    keeps a capacity-constrained topology (for example 32 NPU + SSU6) exactly
    on the original PFO path.  Otherwise only SSUs whose own forecast load
    fraction reaches the same threshold materialize a CIR floor; cold SSUs use
    dedicated zero-CIR Paths and work-conserving WRR residual service.
    """

    demand = tuple(float(value) for value in demand_gbps_by_ssu)
    if not demand:
        raise ValueError("forecast needs at least one SSU")
    if any(not math.isfinite(value) or value < 0.0 for value in demand):
        raise ValueError("forecast demand must be finite and non-negative")
    threshold = float(hot_fraction)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("hot_fraction must be in (0, 1]")
    prefix = int(forecast_requests_per_npu)
    if prefix <= 0 or prefix != forecast_requests_per_npu:
        raise ValueError("forecast_requests_per_npu must be a positive integer")
    capacities = _caps(ssu_capacity_gbps, len(demand))
    fractions = tuple(
        demand_value / capacity for demand_value, capacity in zip(demand, capacities)
    )
    fleet_fraction = math.fsum(demand) / math.fsum(capacities)
    fallback = fleet_fraction >= threshold
    mask = tuple(fallback or fraction >= threshold for fraction in fractions)
    classifications = tuple(
        "fleet_full_protection_fallback"
        if fallback
        else "hot_ssu"
        if selected
        else "cold_ssu_zero_cir"
        for selected in mask
    )
    canonical_input = {
        "capacity_gbps_by_ssu": capacities,
        "demand_gbps_by_ssu": demand,
        "forecast_requests_per_npu": prefix,
        "hot_fraction": threshold,
        "policy": "frozen_manifest_hotspot_v1",
    }
    fingerprint = hashlib.sha256(
        b"frozen-manifest-hotspot:v1\0"
        + json.dumps(
            canonical_input,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return FrozenSSUHotspotForecast(
        demand_gbps_by_ssu=demand,
        capacity_gbps_by_ssu=capacities,
        load_fraction_by_ssu=fractions,
        fleet_load_fraction=fleet_fraction,
        hot_fraction=threshold,
        materialized_ssu_mask=mask,
        classification_by_ssu=classifications,
        full_protection_fallback=fallback,
        forecast_requests_per_npu=prefix,
        input_fingerprint=fingerprint,
    )


__all__ = (
    "DEFAULT_FORECAST_REQUESTS_PER_NPU",
    "DEFAULT_HOT_FRACTION",
    "FrozenSSUHotspotForecast",
    "forecast_frozen_ssu_hotspots",
)
