"""Synthetic bandwidth-satisfaction experiment for the retained policies.

The experiment has two deliberately separate layers:

* ``audit`` feeds an explicit NPU x SSU demand matrix to the authoritative
  Scheme-B allocator and checks SSD40/NPU50 fluid feasibility.  The three
  routing policies have no allocator, so no fictitious grants are invented for
  them.
* ``data_plane`` converts the same matrix into a continuously backlogged,
  repeated 16-layer request template and runs baseline, layer-once, refresh8,
  and three Scheme-B variants on the shared SSD40 -> NPU50 discrete-event
  model.

For a layer compute budget C seconds, placement work is exactly D[n,s] * C GB.
Every positive flow is split into at least nine commands by default so that
refresh8 actually performs a second pressure read and does not collapse to the
layer-once policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    ContinuousBatchRequest,
    SLOAwareSchemeBController,
    SteadyStateConfig,
    SteadyStateInvariantError,
    continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from policy_logic import ManifestDemand, plan_scheme_b, plan_slo_aware_scheme_b
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from slo_admission_scheme_b import (
    SLOAdmissionSchemeBController,
    allocate_slo_admission_grants,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "synthetic_bandwidth_satisfaction" / "results.json"
DEFAULT_PURE_SSU_OUTPUT = (
    ROOT / "results" / "synthetic_bandwidth_satisfaction" / "pure_ssu_32npu_8ssu.json"
)
STRATEGY_NAMES = (
    "baseline",
    "layer_once",
    "refresh8",
    "scheme_b",
    "scheme_b_slo",
    "scheme_b_admission",
)
DEFAULT_UNIFORM_LOADS_GBPS = (5.0, 20.0, 39.0, 40.0, 41.0, 60.0, 80.0, 84.0, 90.0)
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
PURE_SSU_NPU_COUNT = 32
PURE_SSU_SSU_COUNT = 8
PURE_SSU_BYPASS_CAP_GBPS = 1_000_000.0
PURE_SSU_UNIFORM_LOADS_GBPS = (32.0, 36.0, 39.0, 40.0, 41.0, 42.0, 44.0)
PURE_SSU_SKEWED_LOAD_GBPS = 41.0
PURE_SSU_SKEW_SHARES = (0.4, 0.3, 0.2, 0.1)
SLO_AWARE_CONTROL_INTERVAL_MS = 10.0
ADMISSION_TARGET_RATIO = 0.52
ADMISSION_REQUIRED_RATIO = 0.5
ADMISSION_BACKGROUND_RESERVE = 0.05
_EPS = 1e-9


@dataclass(frozen=True)
class DemandCase:
    """One immutable synthetic NPU x SSU demand matrix in GB/s."""

    name: str
    family: str
    description: str
    demands_gbps: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        matrix = tuple(
            tuple(float(value) for value in row) for row in self.demands_gbps
        )
        if not matrix or not matrix[0]:
            raise ValueError("demand matrix must be non-empty")
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ValueError("demand matrix must be rectangular")
        if any(
            value < 0.0 or not math.isfinite(value) for row in matrix for value in row
        ):
            raise ValueError("demand matrix must contain finite non-negative values")
        if any(sum(row) <= 0.0 for row in matrix):
            raise ValueError("every synthetic NPU must have positive demand")
        if len(matrix) > 128:
            raise ValueError("Scheme B supports at most 128 dedicated NPU Paths")
        object.__setattr__(self, "demands_gbps", matrix)

    @property
    def num_npu(self) -> int:
        return len(self.demands_gbps)

    @property
    def num_ssu(self) -> int:
        return len(self.demands_gbps[0])

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "demands_gbps": self.demands_gbps,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _format_load(load_gbps: float) -> str:
    return f"{load_gbps:g}".replace(".", "p")


def uniform_case(
    per_ssu_load_gbps: float,
    *,
    num_npu: int = 4,
    num_ssu: int = 2,
    layout: str = "dense",
) -> DemandCase:
    """Create equal aggregate offered load on every SSU.

    ``dense`` spreads every NPU over every SSU.  ``striped`` assigns each NPU
    to exactly one SSU and is useful for larger DES runs with fewer commands.
    """
    load = float(per_ssu_load_gbps)
    if load <= 0.0 or num_npu <= 0 or num_ssu <= 0:
        raise ValueError("uniform dimensions and load must be positive")
    if layout == "dense":
        matrix = tuple(
            tuple(load / num_npu for _ in range(num_ssu)) for _ in range(num_npu)
        )
    elif layout == "striped":
        if num_npu < num_ssu:
            raise ValueError("striped layout requires at least one NPU per SSU")
        counts = [
            sum(npu % num_ssu == ssu for npu in range(num_npu))
            for ssu in range(num_ssu)
        ]
        matrix = tuple(
            tuple(
                load / counts[ssu] if npu % num_ssu == ssu else 0.0
                for ssu in range(num_ssu)
            )
            for npu in range(num_npu)
        )
    else:
        raise ValueError(f"unknown uniform layout: {layout}")
    return DemandCase(
        name=f"uniform_{layout}_ssu_{_format_load(load)}",
        family="uniform_ramp",
        description=(
            f"Every SSU has {load:g} GB/s offered load using a {layout} matrix"
        ),
        demands_gbps=matrix,
    )


def skewed_striped_case(
    per_ssu_load_gbps: float = PURE_SSU_SKEWED_LOAD_GBPS,
    *,
    num_npu: int = PURE_SSU_NPU_COUNT,
    num_ssu: int = PURE_SSU_SSU_COUNT,
    shares: Sequence[float] = PURE_SSU_SKEW_SHARES,
) -> DemandCase:
    """Create one-SSU-per-NPU load with identical skew on every SSU.

    NPU IDs are arranged in modulo-SSU stripes.  For the canonical 32x8
    profile, IDs ``s, 8+s, 16+s, 24+s`` contribute 40/30/20/10 percent of
    SSU ``s``'s offered load.  The construction removes NPU-link aggregation
    while retaining four independently scheduled flows per SSD.
    """
    load = float(per_ssu_load_gbps)
    normalized_shares = tuple(float(share) for share in shares)
    if load <= 0.0 or num_npu <= 0 or num_ssu <= 0:
        raise ValueError("skewed striped dimensions and load must be positive")
    if (
        not normalized_shares
        or any(share <= 0.0 or not math.isfinite(share) for share in normalized_shares)
        or not math.isclose(sum(normalized_shares), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            "skewed striped shares must be finite, positive, and sum to one"
        )
    if num_npu != num_ssu * len(normalized_shares):
        raise ValueError("skewed striped layout requires one NPU per SSU/share pair")

    matrix = [[0.0] * num_ssu for _ in range(num_npu)]
    for share_index, share in enumerate(normalized_shares):
        for ssu_id in range(num_ssu):
            npu_id = share_index * num_ssu + ssu_id
            matrix[npu_id][ssu_id] = load * share
    return DemandCase(
        name=f"skewed_striped_ssu_{_format_load(load)}",
        family="skewed_striped",
        description=(
            f"Every SSU has {load:g} GB/s offered load split by shares "
            f"{normalized_shares}; every NPU targets exactly one SSU"
        ),
        demands_gbps=tuple(tuple(row) for row in matrix),
    )


def pure_ssu_case_suite(
    uniform_loads: Iterable[float] = PURE_SSU_UNIFORM_LOADS_GBPS,
) -> tuple[DemandCase, ...]:
    """Canonical 32-NPU/8-SSU capacity-knee suite without NPU bottlenecks."""
    uniform = tuple(
        uniform_case(
            load,
            num_npu=PURE_SSU_NPU_COUNT,
            num_ssu=PURE_SSU_SSU_COUNT,
            layout="striped",
        )
        for load in uniform_loads
    )
    return uniform + (skewed_striped_case(),)


def diagnostic_cases() -> tuple[DemandCase, ...]:
    """Hand-checkable matrices that isolate input, policy, and NPU caps."""
    return (
        DemandCase(
            "heterogeneous_raw_feasible",
            "heterogeneous",
            "Both SSUs total exactly 40 GB/s; every NPU remains below 50 GB/s",
            ((5.0, 25.0), (15.0, 5.0), (20.0, 0.0), (0.0, 10.0)),
        ),
        DemandCase(
            "deadline_feasible_scheme_objective",
            "heterogeneous",
            "Raw 80 GB/s is infeasible, but the warm 2x target is feasible; equal max-min can starve the 60-GB/s flow",
            ((60.0,), (10.0,), (10.0,)),
        ),
        DemandCase(
            "hotspot_raw_infeasible",
            "hotspot",
            "Fleet demand 70 < fleet capacity 80, but fixed placement puts 60 GB/s on SSU0",
            ((20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (0.0, 10.0)),
        ),
        DemandCase(
            "hotspot_deadline_infeasible",
            "hotspot",
            "SSU0 has 90 GB/s, so even the warm 2x fluid target exceeds SSD40",
            ((30.0, 0.0), (30.0, 0.0), (30.0, 0.0), (0.0, 10.0)),
        ),
        DemandCase(
            "single_flow_60",
            "single_flow",
            "One 60-GB/s flow exceeds SSD40 but remains warm-2x feasible",
            ((60.0,),),
        ),
        DemandCase(
            "single_flow_90",
            "single_flow",
            "One 90-GB/s flow exceeds both raw and warm-2x SSD capacity",
            ((90.0,),),
        ),
        DemandCase(
            "npu50_control",
            "npu_cap",
            "Each SSU sees only 30 GB/s, but one NPU requests 60 GB/s in aggregate",
            ((30.0, 30.0),),
        ),
        DemandCase(
            "npu50_link_stress",
            "npu_cap",
            "Eight SSUs each see only 10 GB/s, while one NPU requests 80 GB/s in aggregate",
            ((10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),),
        ),
    )


def case_suite(
    uniform_loads: Iterable[float] = DEFAULT_UNIFORM_LOADS_GBPS,
    *,
    uniform_num_npu: int = 4,
    uniform_num_ssu: int = 2,
    uniform_layout: str = "dense",
    uniform_only: bool = False,
) -> tuple[DemandCase, ...]:
    cases = tuple(
        uniform_case(
            load,
            num_npu=uniform_num_npu,
            num_ssu=uniform_num_ssu,
            layout=uniform_layout,
        )
        for load in uniform_loads
    )
    return cases if uniform_only else cases + diagnostic_cases()


def _row_sums(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [float(sum(row)) for row in matrix]


def _column_sums(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(sum(row[column] for row in matrix)) for column in range(len(matrix[0]))
    ]


def _jain(values: Sequence[float]) -> float | None:
    values = tuple(float(value) for value in values)
    if not values:
        return None
    square_sum = sum(value * value for value in values)
    if square_sum <= 0.0:
        return 1.0
    return sum(values) ** 2 / (len(values) * square_sum)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scaled_feasible(
    case: DemandCase,
    factor: float,
    *,
    allocator_npu_cap_gbps: float,
) -> bool:
    return (
        max(_row_sums(case.demands_gbps)) * factor <= allocator_npu_cap_gbps + _EPS
        and max(_column_sums(case.demands_gbps)) * factor <= SSD_CAP_GBPS + _EPS
    )


def _allocator_plan_metrics(
    plan,
    demands: Sequence[Sequence[float]],
    *,
    raw_feasible: bool,
    warm_slo_required_ratio: float,
) -> dict:
    grants = plan.grants_gbps
    grant_rows = _row_sums(grants)
    grant_columns = _column_sums(grants)
    total_demand = sum(sum(row) for row in demands)
    total_grant = sum(grant_rows)
    active_flow_ratios = [
        min(1.0, grants[npu][ssu] / demands[npu][ssu])
        for npu in range(len(demands))
        for ssu in range(len(demands[0]))
        if demands[npu][ssu] > 0.0
    ]
    coflow_ratios = [
        min(
            min(1.0, grants[npu][ssu] / demands[npu][ssu])
            for ssu in range(len(demands[0]))
            if demands[npu][ssu] > 0.0
        )
        for npu in range(len(demands))
    ]
    planned_attainment = sum(
        ratio + _EPS >= warm_slo_required_ratio for ratio in coflow_ratios
    ) / len(coflow_ratios)
    return {
        "grants_gbps": [list(row) for row in grants],
        "per_npu_grant_gbps": grant_rows,
        "per_ssu_grant_gbps": grant_columns,
        "total_grant_gbps": total_grant,
        "demand_weighted_grant_satisfaction": total_grant / total_demand,
        "active_flow_grant_satisfaction_mean": (
            sum(active_flow_ratios) / len(active_flow_ratios)
        ),
        "active_flow_grant_satisfaction_p10": _percentile(active_flow_ratios, 10),
        "active_flow_grant_satisfaction_min": min(active_flow_ratios),
        "coflow_grant_satisfaction_by_npu": coflow_ratios,
        "coflow_grant_satisfaction_jain": _jain(coflow_ratios),
        "planned_warm_2x_slo_attainment": planned_attainment,
        "full_demand_exact_when_raw_feasible": (
            not raw_feasible
            or all(
                abs(grants[npu][ssu] - demands[npu][ssu]) <= _EPS
                for npu in range(len(demands))
                for ssu in range(len(demands[0]))
            )
        ),
        "target_hash": getattr(plan, "target_hash", None),
    }


def audit_case(
    case: DemandCase,
    *,
    n_layers: int = 16,
    slo_alpha: float = 2.0,
    allocator_npu_cap_gbps: float = NPU_CAP_GBPS,
) -> dict:
    """Audit input feasibility and Scheme-B's pure allocator output."""
    allocator_npu_cap_gbps = float(allocator_npu_cap_gbps)
    if (
        n_layers < 2
        or slo_alpha <= 0.0
        or allocator_npu_cap_gbps <= 0.0
        or not math.isfinite(allocator_npu_cap_gbps)
    ):
        raise ValueError("the warm SLO audit parameters are invalid")
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(case.num_npu))
    manifests = tuple(
        ManifestDemand(
            request_id=npu,
            npu_id=npu,
            compute_budget_s=1.0,
            work_by_ssu_gb=row,
        )
        for npu, row in enumerate(case.demands_gbps)
    )
    plan = plan_scheme_b(
        manifests,
        num_npu=case.num_npu,
        num_ssu=case.num_ssu,
        npu_cap_gbps=allocator_npu_cap_gbps,
        path_by_npu=paths,
    )
    slo_plan = plan_slo_aware_scheme_b(
        manifests,
        num_npu=case.num_npu,
        num_ssu=case.num_ssu,
        slo_alpha=slo_alpha,
        npu_cap_gbps=allocator_npu_cap_gbps,
        path_by_npu=paths,
    )
    admission_plan = allocate_slo_admission_grants(
        case.demands_gbps,
        target_ratio=ADMISSION_TARGET_RATIO,
        required_ratio=ADMISSION_REQUIRED_RATIO,
        background_reserve_fraction=ADMISSION_BACKGROUND_RESERVE,
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=allocator_npu_cap_gbps,
    )
    demands = case.demands_gbps
    demand_rows = _row_sums(demands)
    demand_columns = _column_sums(demands)
    # Cross-request Layer-0 prefetch starts only one compute interval before
    # admission.  Once I/O is slower than compute, Layer 0 therefore retains
    # the same residual stall as the other layers.  For a repeated template,
    # TTFT becomes n_layers * C / service_ratio, so alpha-x SLO requires
    # service_ratio >= 1 / alpha (exactly 0.5 for the default 2x SLO).
    warm_2x_required_ratio = 1.0 / slo_alpha
    raw_feasible = _scaled_feasible(
        case,
        1.0,
        allocator_npu_cap_gbps=allocator_npu_cap_gbps,
    )
    warm_2x_feasible = _scaled_feasible(
        case,
        warm_2x_required_ratio,
        allocator_npu_cap_gbps=allocator_npu_cap_gbps,
    )
    total_demand = sum(demand_rows)
    scheme_b_metrics = _allocator_plan_metrics(
        plan,
        demands,
        raw_feasible=raw_feasible,
        warm_slo_required_ratio=warm_2x_required_ratio,
    )
    scheme_b_slo_metrics = _allocator_plan_metrics(
        slo_plan,
        demands,
        raw_feasible=raw_feasible,
        warm_slo_required_ratio=warm_2x_required_ratio,
    )
    scheme_b_admission_metrics = _allocator_plan_metrics(
        admission_plan,
        demands,
        raw_feasible=raw_feasible,
        warm_slo_required_ratio=warm_2x_required_ratio,
    )
    scheme_b_admission_metrics.update(
        {
            "selected_npu_ids": [
                int(npu_id) for npu_id in admission_plan.selected_npu_ids
            ],
            "target_ratio": admission_plan.target_ratio,
            "required_ratio": admission_plan.required_ratio,
            "background_reserve_fraction": (admission_plan.background_reserve_fraction),
        }
    )
    planned_attainment = scheme_b_metrics["planned_warm_2x_slo_attainment"]
    if raw_feasible:
        diagnosis = "raw_full_hide_input_feasible"
    elif warm_2x_feasible and planned_attainment < 1.0 - _EPS:
        diagnosis = "capacity_allows_warm_2x_but_scheme_b_objective_can_miss_it"
    elif warm_2x_feasible:
        diagnosis = "raw_infeasible_but_warm_2x_capacity_feasible"
    else:
        diagnosis = "input_capacity_prevents_100pct_warm_2x"
    return {
        "case": case.name,
        "family": case.family,
        "description": case.description,
        "case_fingerprint": case.fingerprint,
        "num_npu": case.num_npu,
        "num_ssu": case.num_ssu,
        "demands_gbps": [list(row) for row in demands],
        "per_npu_demand_gbps": demand_rows,
        "per_ssu_demand_gbps": demand_columns,
        "total_demand_gbps": total_demand,
        "ssd_cap_gbps": SSD_CAP_GBPS,
        "npu_cap_gbps": allocator_npu_cap_gbps,
        "allocator_npu_cap_gbps": allocator_npu_cap_gbps,
        "raw_full_hide_feasible": raw_feasible,
        "warm_2x_required_service_ratio": warm_2x_required_ratio,
        "warm_2x_fluid_capacity_feasible": warm_2x_feasible,
        "overloaded_ssu_ids": [
            ssu
            for ssu, demand in enumerate(demand_columns)
            if demand > SSD_CAP_GBPS + _EPS
        ],
        "overloaded_npu_ids": [
            npu
            for npu, demand in enumerate(demand_rows)
            if demand > allocator_npu_cap_gbps + _EPS
        ],
        "scheme_b": scheme_b_metrics,
        "scheme_b_slo": scheme_b_slo_metrics,
        "scheme_b_admission": scheme_b_admission_metrics,
        "routing_allocator": {
            name: "not_applicable:routing_policy_without_grant_allocator"
            for name in ("baseline", "layer_once", "refresh8")
        },
        "diagnosis": diagnosis,
    }


def build_requests(
    case: DemandCase,
    *,
    n_layers: int,
    compute_ms: float,
    requests_per_npu: int,
    blocks_per_flow: int,
) -> tuple[ContinuousBatchRequest, ...]:
    """Convert D into repeated placement templates with W=D*C."""
    if n_layers <= 0 or compute_ms <= 0.0 or requests_per_npu <= 0:
        raise ValueError("request generation parameters must be positive")
    if blocks_per_flow <= 0:
        raise ValueError("blocks_per_flow must be positive")
    placements = []
    compute_s = compute_ms / 1000.0
    for row in case.demands_gbps:
        blocks = []
        for ssu_id, demand_gbps in enumerate(row):
            if demand_gbps <= 0.0:
                continue
            flow_work_gb = demand_gbps * compute_s
            block_gb = flow_work_gb / blocks_per_flow
            blocks.extend((ssu_id, block_gb) for _ in range(blocks_per_flow))
        placements.append(tuple(blocks))

    requests = []
    for sequence in range(requests_per_npu):
        for npu_id, placement in enumerate(placements):
            request_id = sequence * case.num_npu + npu_id
            requests.append(
                ContinuousBatchRequest(
                    request_id=request_id,
                    npu_id=npu_id,
                    arrival_time_ms=0.0,
                    load={
                        "request_id": request_id,
                        "npu_id": npu_id,
                        "stream_id": sequence,
                        "category": "SS",
                        "per_layer_us": compute_ms * 1000.0,
                        "initial": sequence == 0,
                        "synthetic_case": case.name,
                    },
                    placement=(placement,),
                )
            )
    return tuple(requests)


def reconstruct_template_demands(
    requests: Sequence[ContinuousBatchRequest],
    *,
    num_npu: int,
    num_ssu: int,
) -> tuple[tuple[float, ...], ...]:
    """Rebuild D from the first repeated request template for validation."""
    first_by_npu = {}
    for request in sorted(requests, key=lambda item: item.request_id):
        first_by_npu.setdefault(request.npu_id, request)
    if len(first_by_npu) != num_npu:
        raise ValueError("request set does not contain every NPU")
    rows = []
    for npu_id in range(num_npu):
        request = first_by_npu[npu_id]
        compute_s = float(request.load["per_layer_us"]) / 1e6
        work = [0.0] * num_ssu
        for ssu_id, size_gb in request.placement[0]:
            work[ssu_id] += size_gb
        rows.append(tuple(amount / compute_s for amount in work))
    return tuple(rows)


def _matrix_delivery_metrics(
    delivered_gbps: Sequence[Sequence[float]],
    demand_gbps: Sequence[Sequence[float]],
) -> dict:
    active_ratios = []
    coflow_ratios = []
    capped_delivered = 0.0
    raw_delivered = 0.0
    total_demand = sum(sum(row) for row in demand_gbps)
    for npu, demand_row in enumerate(demand_gbps):
        row_ratios = []
        for ssu, demand in enumerate(demand_row):
            delivered = float(delivered_gbps[npu][ssu])
            raw_delivered += delivered
            if demand <= 0.0:
                continue
            capped = min(delivered, demand)
            capped_delivered += capped
            ratio = capped / demand
            active_ratios.append(ratio)
            row_ratios.append(ratio)
        coflow_ratios.append(min(row_ratios))
    return {
        "raw_delivered_gbps": raw_delivered,
        "raw_delivered_over_target": raw_delivered / total_demand,
        "demand_weighted_satisfaction": capped_delivered / total_demand,
        "active_flow_satisfaction_mean": sum(active_ratios) / len(active_ratios),
        "active_flow_satisfaction_p10": _percentile(active_ratios, 10),
        "active_flow_satisfaction_min": min(active_ratios),
        "coflow_satisfaction_by_npu": coflow_ratios,
        "coflow_satisfaction_mean": sum(coflow_ratios) / len(coflow_ratios),
        "coflow_satisfaction_min": min(coflow_ratios),
        "coflow_satisfaction_jain": _jain(coflow_ratios),
    }


def _compact_data_plane_run(
    case: DemandCase,
    strategy: str,
    summary: dict,
    *,
    compute_ms: float,
    wall_time_s: float,
) -> dict:
    if not all(summary["invariants"].values()):
        raise AssertionError(summary["invariants"])
    ssd_matrix = summary["measurement_npu_ssu_ssd_served_gbps"]
    link_matrix = summary["measurement_npu_ssu_link_served_gbps"]
    ssd_metrics = _matrix_delivery_metrics(ssd_matrix, case.demands_gbps)
    link_metrics = _matrix_delivery_metrics(link_matrix, case.demands_gbps)
    ideal_ttft_ms = summary["n_layers"] * compute_ms
    return {
        "status": "ok",
        "case": case.name,
        "strategy": strategy,
        "input_fingerprint": summary["input_fingerprint"],
        "wall_time_s": wall_time_s,
        "ssd_service": ssd_metrics,
        "npu_link_service": link_metrics,
        "measurement_ssd_served_gbps_by_ssu": summary[
            "measurement_ssd_served_gbps_by_ssu"
        ],
        "measurement_ssd_utilizations": summary["measurement_ssd_utilizations"],
        "measurement_ssd_mean_utilization": summary["measurement_ssd_mean_utilization"],
        "measurement_npu_ssu_ssd_served_gbps": ssd_matrix,
        "measurement_npu_ssu_link_served_gbps": link_matrix,
        "measurement_npu_link_utilizations": summary[
            "measurement_npu_link_utilizations"
        ],
        "measurement_npu_link_mean_utilization": summary[
            "measurement_npu_link_mean_utilization"
        ],
        "mean_npu_compute_utilization": summary["mean_npu_utilization"],
        "ttft_slo_attainment": summary["ttft_slo_attainment"],
        "request_weighted_slo_attainment": summary["request_weighted_slo_attainment"],
        "mean_ttft_ms": summary["mean_ttft_ms"],
        "p99_ttft_ms": summary["p99_ttft_ms"],
        "mean_normalized_ttft": summary["mean_ttft_ms"] / ideal_ttft_ms,
        "measurement_request_count": summary["measurement_request_count"],
        "request_counts_by_npu": summary["request_counts_by_npu"],
        "measurement_blocks": summary["measurement_blocks"],
        "ssd_outstanding_blocks_at_start": summary[
            "measurement_ssd_outstanding_blocks_at_start"
        ],
        "ssd_outstanding_blocks_at_end": summary[
            "measurement_ssd_outstanding_blocks_at_end"
        ],
        "pressure_reports": summary["pressure_reports"],
        "control_evaluations": summary["control_evaluations"],
        "cir_commits": summary["cir_commits"],
        "cir_path_writes": summary["cir_path_writes"],
        "controller_contract": (
            {
                "kind": "full_manifest_slo_aware",
                "interval_ms": SLO_AWARE_CONTROL_INTERVAL_MS,
                "on_batch_boundary": False,
                "npu_dedicated_paths": True,
                "shared_layer0_path": False,
            }
            if strategy == "scheme_b_slo"
            else {
                "kind": "full_manifest_slo_admission",
                "min_interval_ms": SLO_AWARE_CONTROL_INTERVAL_MS,
                "on_batch_boundary": True,
                "target_ratio": ADMISSION_TARGET_RATIO,
                "required_ratio": ADMISSION_REQUIRED_RATIO,
                "background_reserve_fraction": ADMISSION_BACKGROUND_RESERVE,
                "npu_dedicated_paths": True,
                "shared_layer0_path": False,
            }
            if strategy == "scheme_b_admission"
            else {
                "kind": "previous_layer_causal",
                "interval_ms": None,
                "on_batch_boundary": None,
                "npu_dedicated_paths": True,
                "shared_layer0_path": True,
            }
            if strategy == "scheme_b"
            else None
        ),
        "events_processed": summary["events_processed"],
        "invariants": summary["invariants"],
    }


def run_data_plane_case(
    case: DemandCase,
    strategy: str,
    requests: Sequence[ContinuousBatchRequest],
    *,
    n_layers: int,
    compute_ms: float,
    steady_state: SteadyStateConfig,
    submit_order_seed: int,
    physical_npu_bw_gbps: float,
    allocator_npu_cap_gbps: float,
) -> dict:
    common = {
        "num_npu": case.num_npu,
        "num_ssu": case.num_ssu,
        "n_layers": n_layers,
        "batch_size": 1,
        "submit_order_seed": submit_order_seed,
        "cross_request_layer0_prefetch": True,
        "steady_state": steady_state,
        "disk_bw_gbps": SSD_CAP_GBPS,
        "npu_bw_gbps": physical_npu_bw_gbps,
    }
    started = time.perf_counter()
    if strategy in ("baseline", "layer_once", "refresh8"):
        routing = next(
            spec for spec in routing_strategy_specs() if spec.name == strategy
        )
        summary = simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=routing.client_config(),
            **common,
        )
    elif strategy == "scheme_b":
        paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(case.num_npu))
        controller = CausalMaxMinSchemeBController(
            paths,
            cold_path_id=0,
            cold_path_cir_gbps=static_qos_config().path_cirs[0],
            path_count=PATH_COUNT,
            npu_cap_gbps=allocator_npu_cap_gbps,
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * case.num_ssu
            ),
            npu_dedicated_paths=paths,
            layer0_path_id=0,
            client_io_config=scheme_b_client_config(case.name),
            causal_control=CausalLayerControlConfig(controller),
            **common,
        )
    elif strategy == "scheme_b_slo":
        paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(case.num_npu))
        controller = SLOAwareSchemeBController(
            paths,
            slo_alpha=steady_state.slo_alpha,
            npu_cap_gbps=allocator_npu_cap_gbps,
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * case.num_ssu
            ),
            npu_dedicated_paths=paths,
            client_io_config=scheme_b_client_config(f"{case.name}_scheme_b_slo"),
            control=CIRControlConfig(
                callback=controller,
                interval_ms=SLO_AWARE_CONTROL_INTERVAL_MS,
                on_batch_boundary=False,
            ),
            **common,
        )
    elif strategy == "scheme_b_admission":
        paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(case.num_npu))
        controller = SLOAdmissionSchemeBController(
            paths,
            target_ratio=ADMISSION_TARGET_RATIO,
            required_ratio=ADMISSION_REQUIRED_RATIO,
            background_reserve_fraction=ADMISSION_BACKGROUND_RESERVE,
            npu_cap_gbps=allocator_npu_cap_gbps,
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * case.num_ssu
            ),
            npu_dedicated_paths=paths,
            client_io_config=scheme_b_client_config(f"{case.name}_scheme_b_admission"),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=SLO_AWARE_CONTROL_INTERVAL_MS,
            ),
            **common,
        )
    else:
        raise ValueError(f"unknown data-plane strategy: {strategy}")
    wall_time_s = time.perf_counter() - started
    if strategy == "baseline" and summary["pressure_reports"] != 0:
        raise AssertionError("baseline must not read Path pressure")
    if strategy in ("baseline", "layer_once", "refresh8") and any(
        summary[field]
        for field in ("control_evaluations", "cir_commits", "cir_path_writes")
    ):
        raise AssertionError("routing policies must not update CIR")
    if strategy in ("scheme_b", "scheme_b_slo", "scheme_b_admission") and (
        summary["control_evaluations"] <= 0
    ):
        raise AssertionError(f"{strategy} must evaluate its controller")
    return _compact_data_plane_run(
        case,
        strategy,
        summary,
        compute_ms=compute_ms,
        wall_time_s=wall_time_s,
    )


def run_experiment(
    cases: Sequence[DemandCase],
    *,
    mode: str,
    strategies: Sequence[str],
    n_layers: int,
    compute_ms: float,
    requests_per_npu: int,
    blocks_per_flow: int,
    steady_state: SteadyStateConfig,
    submit_order_seed: int,
    physical_npu_bw_gbps: float = NPU_CAP_GBPS,
    allocator_npu_cap_gbps: float = NPU_CAP_GBPS,
    experiment_profile: str = "general",
) -> dict:
    physical_npu_bw_gbps = float(physical_npu_bw_gbps)
    allocator_npu_cap_gbps = float(allocator_npu_cap_gbps)
    if (
        physical_npu_bw_gbps <= 0.0
        or allocator_npu_cap_gbps <= 0.0
        or not math.isfinite(physical_npu_bw_gbps)
        or not math.isfinite(allocator_npu_cap_gbps)
    ):
        raise ValueError(
            "physical and allocator NPU capacities must be finite and positive"
        )
    rows = []
    for case_index, case in enumerate(cases, 1):
        print(f"[{case_index}/{len(cases)}] audit {case.name}", flush=True)
        case_row = {
            "case": case.name,
            "audit": audit_case(
                case,
                n_layers=n_layers,
                slo_alpha=steady_state.slo_alpha,
                allocator_npu_cap_gbps=allocator_npu_cap_gbps,
            ),
            "data_plane_runs": [],
        }
        if mode in ("data_plane", "both"):
            requests = build_requests(
                case,
                n_layers=n_layers,
                compute_ms=compute_ms,
                requests_per_npu=requests_per_npu,
                blocks_per_flow=blocks_per_flow,
            )
            reconstructed = reconstruct_template_demands(
                requests,
                num_npu=case.num_npu,
                num_ssu=case.num_ssu,
            )
            if any(
                abs(reconstructed[npu][ssu] - case.demands_gbps[npu][ssu]) > 1e-8
                for npu in range(case.num_npu)
                for ssu in range(case.num_ssu)
            ):
                raise AssertionError("synthetic placement does not reconstruct D")
            expected_input_fingerprint = continuous_batch_input_fingerprint(requests)
            case_row["input_fingerprint"] = expected_input_fingerprint
            for strategy in strategies:
                print(f"    data-plane {strategy}", flush=True)
                try:
                    run = run_data_plane_case(
                        case,
                        strategy,
                        requests,
                        n_layers=n_layers,
                        compute_ms=compute_ms,
                        steady_state=steady_state,
                        submit_order_seed=submit_order_seed,
                        physical_npu_bw_gbps=physical_npu_bw_gbps,
                        allocator_npu_cap_gbps=allocator_npu_cap_gbps,
                    )
                except SteadyStateInvariantError as error:
                    print(
                        f"    INVALID {strategy}: {type(error).__name__}: {error}",
                        flush=True,
                    )
                    run = {
                        "status": "invalid",
                        "case": case.name,
                        "strategy": strategy,
                        "input_fingerprint": expected_input_fingerprint,
                        "failure_type": type(error).__name__,
                        "failure": str(error),
                        "failed_invariants": error.invariants,
                        "failure_diagnostics": error.diagnostics,
                        "interpretation": (
                            "No metric is reported. The finite saturated prefix "
                            "or another steady-state invariant failed; this can "
                            "indicate extreme tail latency/starvation and must not "
                            "be converted into a numeric policy result."
                        ),
                    }
                case_row["data_plane_runs"].append(run)
                if run["input_fingerprint"] != expected_input_fingerprint:
                    raise AssertionError("strategy input fingerprint mismatch")
            valid_runs = [
                run for run in case_row["data_plane_runs"] if run["status"] == "ok"
            ]
            fingerprints = {
                run["input_fingerprint"] for run in case_row["data_plane_runs"]
            }
            if fingerprints != {expected_input_fingerprint}:
                raise AssertionError("strategies did not receive paired input")
            case_row["paired_inputs_verified"] = True
            run_by_strategy = {run["strategy"]: run for run in valid_runs}
            if "layer_once" in run_by_strategy and "refresh8" in run_by_strategy:
                case_row["refresh8_more_pressure_reads_than_layer_once"] = (
                    run_by_strategy["refresh8"]["pressure_reports"]
                    > run_by_strategy["layer_once"]["pressure_reports"]
                )
        rows.append(case_row)
    invalid_run_count = sum(
        run["status"] == "invalid" for row in rows for run in row["data_plane_runs"]
    )
    return {
        "schema_version": 1,
        "experiment": "synthetic_bandwidth_satisfaction",
        "experiment_profile": experiment_profile,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "invalid_data_plane_run_count": invalid_run_count,
        "complete": invalid_run_count == 0,
        "methodology": {
            "demand_definition": "D[n,s]=per-layer placement GB / compute seconds",
            "ssd_service": "exact SSD40 busy-time service overlap in the common window, attributed by NPU and SSU",
            "npu_link_service": "exact physical NPU-link service overlap in the common window, attributed by source SSU; partial in-flight command bytes are not yet compute-visible",
            "allocator_scope": "Scheme B only; routing policies have no grant allocator",
            "npu_constraint_scope": (
                "pure_ssu counterfactual: both the physical receive link and "
                "Scheme-B allocator NPU cap are raised to a finite bypass rate"
                if experiment_profile == "pure_ssu"
                else "physical receive-link and Scheme-B allocator NPU caps are explicit"
            ),
            "cir_warning": "CIR is a guaranteed arbitration share, not a hard rate cap; surplus is work-conserving",
            "steady_warning": "delivered/target is a long-run repeated-template rate; it is not a claim that every flow stayed backlogged for the whole window",
            "warm_slo_fluid_rule": "Layer0 has one compute interval of prefetch lead; repeated-template alpha-x SLO requires service/demand >= 1/alpha",
            "scheme_b_slo_contract": {
                "manifest_scope": "all not-yet-ready layers of each active request",
                "control_interval_ms": SLO_AWARE_CONTROL_INTERVAL_MS,
                "on_batch_boundary": False,
                "path_mode": "one dedicated Path per NPU for every layer",
                "shared_layer0_path": False,
            },
            "scheme_b_admission_contract": {
                "manifest_scope": "all not-yet-ready layers of each active request",
                "min_control_interval_ms": SLO_AWARE_CONTROL_INTERVAL_MS,
                "trigger": "batch-boundary event, rate-limited",
                "target_ratio": ADMISSION_TARGET_RATIO,
                "required_ratio": ADMISSION_REQUIRED_RATIO,
                "background_reserve_fraction": ADMISSION_BACKGROUND_RESERVE,
                "path_mode": "one dedicated Path per NPU for every layer",
                "shared_layer0_path": False,
            },
        },
        "config": {
            "n_layers": n_layers,
            "compute_ms": compute_ms,
            "requests_per_npu": requests_per_npu,
            "blocks_per_positive_npu_ssu_flow": blocks_per_flow,
            "strategies": list(strategies),
            "submit_order_seed": submit_order_seed,
            "steady_state": {
                "warmup_requests_per_npu": steady_state.warmup_requests_per_npu,
                "settle_ms": steady_state.settle_ms,
                "measurement_ms": steady_state.measurement_ms,
                "slo_alpha": steady_state.slo_alpha,
                "block_ms": steady_state.block_ms,
            },
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": allocator_npu_cap_gbps,
            "allocator_npu_cap_gbps": allocator_npu_cap_gbps,
            "physical_npu_link_bw_gbps": physical_npu_bw_gbps,
            "npu_constraints_effectively_disabled": (
                experiment_profile == "pure_ssu"
                and physical_npu_bw_gbps == PURE_SSU_BYPASS_CAP_GBPS
                and allocator_npu_cap_gbps == PURE_SSU_BYPASS_CAP_GBPS
            ),
        },
        "cases": rows,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Synthetic bandwidth-satisfaction results",
        "",
        "## Allocator/input audit",
        "",
        "| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B satisfaction | Scheme-B 2x SLO | SLO-aware satisfaction | SLO-aware 2x SLO | Admission satisfaction | Admission 2x SLO | Diagnosis |",
        "|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["cases"]:
        audit = row["audit"]
        columns = audit["per_ssu_demand_gbps"]
        scheme = audit["scheme_b"]
        slo_scheme = audit["scheme_b_slo"]
        admission = audit["scheme_b_admission"]
        lines.append(
            "| {case} | {low:.2f}-{high:.2f} | {npu:.2f} | {raw} | {warm} | {sat:.3f} | {slo:.3f} | {slo_sat:.3f} | {slo_slo:.3f} | {admission_sat:.3f} | {admission_slo:.3f} | {diagnosis} |".format(
                case=audit["case"],
                low=min(columns),
                high=max(columns),
                npu=max(audit["per_npu_demand_gbps"]),
                raw="yes" if audit["raw_full_hide_feasible"] else "no",
                warm="yes" if audit["warm_2x_fluid_capacity_feasible"] else "no",
                sat=scheme["demand_weighted_grant_satisfaction"],
                slo=scheme["planned_warm_2x_slo_attainment"],
                slo_sat=slo_scheme["demand_weighted_grant_satisfaction"],
                slo_slo=slo_scheme["planned_warm_2x_slo_attainment"],
                admission_sat=admission["demand_weighted_grant_satisfaction"],
                admission_slo=admission["planned_warm_2x_slo_attainment"],
                diagnosis=audit["diagnosis"],
            )
        )
    if any(row["data_plane_runs"] for row in result["cases"]):
        lines.extend(
            [
                "",
                "## Full data-plane steady window",
                "",
                "| Case | Strategy | SSD satisfaction | Link satisfaction | SSD util | Link util | NPU compute util | TTFT 2x SLO | Mean TTFT ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["cases"]:
            for run in row["data_plane_runs"]:
                if run["status"] == "invalid":
                    lines.append(
                        f"| {run['case']} | {run['strategy']} | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID |"
                    )
                else:
                    lines.append(
                        "| {case} | {strategy} | {ssd:.3f} | {link:.3f} | {ssd_util:.3f} | {link_util:.3f} | {npu:.3f} | {slo:.3f} | {ttft:.2f} |".format(
                            case=run["case"],
                            strategy=run["strategy"],
                            ssd=run["ssd_service"]["demand_weighted_satisfaction"],
                            link=run["npu_link_service"][
                                "demand_weighted_satisfaction"
                            ],
                            ssd_util=run["measurement_ssd_mean_utilization"],
                            link_util=run["measurement_npu_link_mean_utilization"],
                            npu=run["mean_npu_compute_utilization"],
                            slo=run["ttft_slo_attainment"],
                            ttft=run["mean_ttft_ms"],
                        )
                    )
        invalid_runs = [
            run
            for row in result["cases"]
            for run in row["data_plane_runs"]
            if run["status"] == "invalid"
        ]
        if invalid_runs:
            lines.extend(["", "## Invalid data-plane runs", ""])
            for run in invalid_runs:
                lines.append(
                    f"- `{run['case']} / {run['strategy']}`: "
                    f"`{run['failure_type']}`; see the JSON `failure` field "
                    "for invariant and per-NPU drain diagnostics."
                )
    lines.extend(
        [
            "",
            "SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link service/target demand. Both use exact overlap at the common measurement boundaries; partial service from an in-flight link command is not compute-visible until that command completes.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "loads must be positive comma-separated numbers"
        )
    return parsed


def _parse_csv_strategies(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(STRATEGY_NAMES)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(
            f"strategies must be a subset of {','.join(STRATEGY_NAMES)}"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("audit", "data_plane", "both"), default="both"
    )
    parser.add_argument(
        "--pure-ssu",
        action="store_true",
        help=(
            "run the canonical 32-NPU/8-SSU striped capacity-knee suite; "
            "both physical and allocator NPU capacities are fixed at 1e6 GB/s"
        ),
    )
    parser.add_argument(
        "--uniform-loads",
        type=_parse_csv_floats,
        default=None,
        help="per-SSU GB/s sweep, comma separated",
    )
    parser.add_argument("--uniform-num-npu", type=int, default=None)
    parser.add_argument("--uniform-num-ssu", type=int, default=None)
    parser.add_argument(
        "--uniform-layout",
        choices=("dense", "striped"),
        default=None,
    )
    parser.add_argument("--uniform-only", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only this exact case name; may be repeated",
    )
    parser.add_argument(
        "--strategies",
        type=_parse_csv_strategies,
        default=STRATEGY_NAMES,
    )
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--compute-ms", type=float, default=10.0)
    parser.add_argument("--requests-per-npu", type=int, default=32)
    parser.add_argument("--blocks-per-flow", type=int, default=9)
    parser.add_argument("--warmup-requests-per-npu", type=int, default=1)
    parser.add_argument("--settle-ms", type=float, default=50.0)
    parser.add_argument("--measurement-ms", type=float, default=1_000.0)
    parser.add_argument("--block-ms", type=float, default=250.0)
    parser.add_argument("--slo-alpha", type=float, default=2.0)
    parser.add_argument("--submit-order-seed", type=int, default=42)
    parser.add_argument(
        "--physical-npu-bw-gbps",
        type=float,
        default=None,
        help="physical NPU receive-link rate",
    )
    parser.add_argument(
        "--allocator-npu-cap-gbps",
        type=float,
        default=None,
        help="per-NPU capacity used by the Scheme-B max-min allocator",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="write an incomplete matrix and return success when a steady invariant fails",
    )
    args = parser.parse_args(argv)

    if args.n_layers < 2 or args.compute_ms <= 0.0:
        parser.error("n-layers must be >=2 and compute-ms must be positive")
    if args.blocks_per_flow < 9 and args.mode in ("data_plane", "both"):
        parser.error("blocks-per-flow must be >=9 so refresh8 differs from layer-once")

    if args.pure_ssu:
        if args.uniform_only:
            parser.error("--uniform-only is incompatible with --pure-ssu's skewed case")
        if args.uniform_num_npu not in (None, PURE_SSU_NPU_COUNT):
            parser.error(f"--pure-ssu requires {PURE_SSU_NPU_COUNT} NPUs")
        if args.uniform_num_ssu not in (None, PURE_SSU_SSU_COUNT):
            parser.error(f"--pure-ssu requires {PURE_SSU_SSU_COUNT} SSUs")
        if args.uniform_layout not in (None, "striped"):
            parser.error("--pure-ssu requires striped placement")
        if args.physical_npu_bw_gbps not in (None, PURE_SSU_BYPASS_CAP_GBPS):
            parser.error(
                f"--pure-ssu fixes physical NPU bandwidth at "
                f"{PURE_SSU_BYPASS_CAP_GBPS:g} GB/s"
            )
        if args.allocator_npu_cap_gbps not in (None, PURE_SSU_BYPASS_CAP_GBPS):
            parser.error(
                f"--pure-ssu fixes allocator NPU capacity at "
                f"{PURE_SSU_BYPASS_CAP_GBPS:g} GB/s"
            )
        uniform_loads = args.uniform_loads or PURE_SSU_UNIFORM_LOADS_GBPS
        cases = pure_ssu_case_suite(uniform_loads)
        physical_npu_bw_gbps = PURE_SSU_BYPASS_CAP_GBPS
        allocator_npu_cap_gbps = PURE_SSU_BYPASS_CAP_GBPS
        experiment_profile = "pure_ssu"
        output_path = args.output or DEFAULT_PURE_SSU_OUTPUT
    else:
        uniform_loads = args.uniform_loads or DEFAULT_UNIFORM_LOADS_GBPS
        cases = case_suite(
            uniform_loads,
            uniform_num_npu=(
                4 if args.uniform_num_npu is None else args.uniform_num_npu
            ),
            uniform_num_ssu=(
                2 if args.uniform_num_ssu is None else args.uniform_num_ssu
            ),
            uniform_layout=args.uniform_layout or "dense",
            uniform_only=args.uniform_only,
        )
        physical_npu_bw_gbps = (
            NPU_CAP_GBPS
            if args.physical_npu_bw_gbps is None
            else args.physical_npu_bw_gbps
        )
        allocator_npu_cap_gbps = (
            NPU_CAP_GBPS
            if args.allocator_npu_cap_gbps is None
            else args.allocator_npu_cap_gbps
        )
        experiment_profile = "general"
        output_path = args.output or DEFAULT_OUTPUT
    if (
        physical_npu_bw_gbps <= 0.0
        or allocator_npu_cap_gbps <= 0.0
        or not math.isfinite(physical_npu_bw_gbps)
        or not math.isfinite(allocator_npu_cap_gbps)
    ):
        parser.error(
            "physical and allocator NPU capacities must be finite and positive"
        )
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
        missing = selected - {case.name for case in cases}
        if missing:
            parser.error(f"unknown case name(s): {', '.join(sorted(missing))}")
    steady_state = SteadyStateConfig(
        warmup_requests_per_npu=args.warmup_requests_per_npu,
        settle_ms=args.settle_ms,
        measurement_ms=args.measurement_ms,
        slo_alpha=args.slo_alpha,
        block_ms=args.block_ms,
    )
    result = run_experiment(
        cases,
        mode=args.mode,
        strategies=args.strategies,
        n_layers=args.n_layers,
        compute_ms=args.compute_ms,
        requests_per_npu=args.requests_per_npu,
        blocks_per_flow=args.blocks_per_flow,
        steady_state=steady_state,
        submit_order_seed=args.submit_order_seed,
        physical_npu_bw_gbps=physical_npu_bw_gbps,
        allocator_npu_cap_gbps=allocator_npu_cap_gbps,
        experiment_profile=experiment_profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    report_path = output_path.with_suffix(".md")
    report_path.write_text(render_markdown(result))
    print(f"wrote {output_path}")
    print(f"wrote {report_path}")
    if result["invalid_data_plane_run_count"]:
        print(
            f"warning: {result['invalid_data_plane_run_count']} data-plane run(s) "
            "were recorded as invalid; see JSON failure fields"
        )
        if not args.allow_invalid:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
