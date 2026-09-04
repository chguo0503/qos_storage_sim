#!/usr/bin/env python3
"""Small V/B admission policy built from existing simulator interfaces.

The policy has only two pools.  A request's compute window ``C`` and layer KV
work ``D`` define its masking bandwidth ``B = D / C`` and its bandwidth value
``V = C / B``.  Requests are considered by descending V.  A request enters the
protected LL pool only when its guarded per-SSU demand and total NPU receive
demand fit.  All other requests enter the background LS pool.

No simulator behavior is changed here: the output is only cloned request
manifests plus one ordinary :class:`sim.StaticQoSConfig` per SSU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from continuous_batch_sim import ContinuousBatchRequest
from policy_logic import GROUP_COUNT, PATH_COUNT
import sim


PROTECTED_CATEGORY = "LL"
BACKGROUND_CATEGORY = "LS"
CATEGORY_LABELS = ("SS", "SL", "LS", "LL")
CATEGORY_PATHS_PER_GROUP = (12, 4, 12, 4)


@dataclass(frozen=True)
class RequestVB:
    """One request's V/B values and deterministic pool decision."""

    request_id: int
    npu_id: int
    rank: int
    compute_window_s: float
    layer_kv_gib: float
    required_b_gibps: float
    value_v_s2_per_gib: float
    raw_demand_by_ssu_gibps: tuple[float, ...]
    guarded_demand_by_ssu_gibps: tuple[float, ...]
    guarded_npu_demand_gibps: float
    selected: bool
    assigned_category: str
    decision_reason: str
    limiting_ssu_ids: tuple[int, ...]

    def audit_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "rank": self.rank,
            "request_id": self.request_id,
            "npu_id": self.npu_id,
            "compute_window_ms": self.compute_window_s * 1000.0,
            "layer_kv_gib": self.layer_kv_gib,
            "B_required_gibps": self.required_b_gibps,
            "V_seconds2_per_gib": self.value_v_s2_per_gib,
            "guarded_npu_demand_gibps": self.guarded_npu_demand_gibps,
            "selected_protected": self.selected,
            "assigned_category": self.assigned_category,
            "decision_reason": self.decision_reason,
            "limiting_ssu_ids": ";".join(map(str, self.limiting_ssu_ids)),
        }
        for ssu_id, value in enumerate(self.raw_demand_by_ssu_gibps):
            row[f"raw_ssu{ssu_id}_gibps"] = value
        for ssu_id, value in enumerate(self.guarded_demand_by_ssu_gibps):
            row[f"guarded_ssu{ssu_id}_gibps"] = value
        return row


@dataclass(frozen=True)
class VBPoolPlan:
    """Complete two-pool admission and static-QoS plan."""

    alpha: float
    disk_bandwidth_gibps: float
    npu_bandwidth_gibps: float
    decisions: tuple[RequestVB, ...]
    protected_request_ids: tuple[int, ...]
    background_request_ids: tuple[int, ...]
    protected_cir_by_ssu_gibps: tuple[float, ...]
    background_cir_by_ssu_gibps: tuple[float, ...]
    qos_configs_by_ssu: tuple[sim.StaticQoSConfig, ...]

    def bandwidth_rows(self) -> list[dict[str, object]]:
        rows = []
        for ssu_id, (protected, background) in enumerate(
            zip(
                self.protected_cir_by_ssu_gibps,
                self.background_cir_by_ssu_gibps,
            )
        ):
            config = self.qos_configs_by_ssu[ssu_id]
            rows.append(
                {
                    "ssu_id": ssu_id,
                    "capacity_gibps": self.disk_bandwidth_gibps,
                    "protected_LL_total_cir_gibps": protected,
                    "background_LS_total_cir_gibps": background,
                    "configured_cir_sum_gibps": math.fsum(config.path_cirs),
                    "protected_LL_path_count": (
                        GROUP_COUNT * CATEGORY_PATHS_PER_GROUP[3]
                    ),
                    "background_LS_path_count": (
                        GROUP_COUNT * CATEGORY_PATHS_PER_GROUP[2]
                    ),
                    "protected_LL_cir_per_path_gibps": (
                        protected
                        / (GROUP_COUNT * CATEGORY_PATHS_PER_GROUP[3])
                    ),
                    "background_LS_cir_per_path_gibps": (
                        background
                        / (GROUP_COUNT * CATEGORY_PATHS_PER_GROUP[2])
                    ),
                }
            )
        return rows


def _layer_work_by_ssu(
    request: ContinuousBatchRequest,
    num_ssu: int,
) -> tuple[float, ...]:
    """Return the representative per-layer bytes used by the admission pass."""
    # A one-layer placement is intentionally reused for every layer by the
    # fixed experiment.  For a multi-layer manifest, use the largest local
    # work over all layers so the admission test remains conservative.
    maxima = [0.0] * num_ssu
    for layer in request.placement:
        work = [0.0] * num_ssu
        for ssu_id, size_gib in layer:
            work[int(ssu_id)] += float(size_gib)
        maxima = [max(old, new) for old, new in zip(maxima, work)]
    return tuple(maxima)


def _request_values(
    request: ContinuousBatchRequest,
    num_ssu: int,
    alpha: float,
) -> tuple[float, float, float, float, tuple[float, ...], tuple[float, ...]]:
    compute_s = float(request.load["per_layer_us"]) / 1e6
    layer_kv_gib = float(request.load["per_layer_kv_gb"])
    if compute_s <= 0.0 or layer_kv_gib < 0.0:
        raise ValueError("V/B policy requires positive compute and nonnegative KV work")
    required_b = layer_kv_gib / compute_s
    value_v = math.inf if required_b == 0.0 else compute_s / required_b
    work_by_ssu = _layer_work_by_ssu(request, num_ssu)
    raw_demand = tuple(work / compute_s for work in work_by_ssu)
    guarded_demand = tuple(alpha * value for value in raw_demand)
    return (
        compute_s,
        layer_kv_gib,
        required_b,
        value_v,
        raw_demand,
        guarded_demand,
    )


def request_v_b(request: ContinuousBatchRequest) -> tuple[float, float]:
    """Return ``(V, B)`` from the request profile, independent of placement."""
    compute_s = float(request.load["per_layer_us"]) / 1e6
    layer_kv_gib = float(request.load["per_layer_kv_gb"])
    if compute_s <= 0.0 or layer_kv_gib < 0.0:
        raise ValueError("V/B policy requires positive compute and nonnegative KV work")
    required_b = layer_kv_gib / compute_s
    value_v = math.inf if required_b == 0.0 else compute_s / required_b
    return value_v, required_b


def _qos_config(protected_cir: float, background_cir: float) -> sim.StaticQoSConfig:
    budgets = {
        "SS": 0.0,
        "SL": 0.0,
        "LS": float(background_cir),
        "LL": float(protected_cir),
    }
    path_cirs = []
    for _ in range(GROUP_COUNT):
        for label, count in zip(CATEGORY_LABELS, CATEGORY_PATHS_PER_GROUP):
            path_cirs.extend([budgets[label] / (GROUP_COUNT * count)] * count)
    if len(path_cirs) != PATH_COUNT:
        raise AssertionError("V/B pool layout must contain exactly 256 Paths")
    return sim.StaticQoSConfig(
        path_cirs=tuple(path_cirs),
        path_pirs=(float("inf"),) * PATH_COUNT,
        path_weights=(1.0,) * PATH_COUNT,
        group_weights=(1.0,) * GROUP_COUNT,
        category_paths_per_group=CATEGORY_PATHS_PER_GROUP,
        category_labels=CATEGORY_LABELS,
    )


def qos_configs_for_pool_cirs(
    protected_cir_by_ssu_gibps: Sequence[float],
    *,
    disk_bandwidth_gibps: float = 40.0,
) -> tuple[sim.StaticQoSConfig, ...]:
    """Build ordinary two-pool QoS tables from explicit per-SSU budgets."""
    configs = []
    for protected in protected_cir_by_ssu_gibps:
        protected = float(protected)
        if protected < 0.0 or protected > disk_bandwidth_gibps + 1e-12:
            raise ValueError("protected pool CIR must fit the SSU capacity")
        configs.append(
            _qos_config(protected, max(0.0, disk_bandwidth_gibps - protected))
        )
    return tuple(configs)


def relabel_requests_by_v_cutoff(
    requests: Iterable[ContinuousBatchRequest],
    *,
    v_cutoff_s2_per_gib: float,
) -> tuple[ContinuousBatchRequest, ...]:
    """Map each request independently to LL/LS using one explicit V cutoff."""
    clones = []
    for request in requests:
        value_v, _ = request_v_b(request)
        load = dict(request.load)
        load["category"] = (
            PROTECTED_CATEGORY
            if value_v > float(v_cutoff_s2_per_gib)
            else BACKGROUND_CATEGORY
        )
        clones.append(
            ContinuousBatchRequest(
                request_id=request.request_id,
                npu_id=request.npu_id,
                arrival_time_ms=request.arrival_time_ms,
                load=load,
                placement=request.placement,
            )
        )
    return tuple(clones)


def build_vb_pool_plan(
    requests: Sequence[ContinuousBatchRequest],
    *,
    num_ssu: int,
    alpha: float = 1.02,
    disk_bandwidth_gibps: float = 40.0,
    npu_bandwidth_gibps: float = 50.0,
) -> VBPoolPlan:
    """Rank by ``(-V, request_id)`` and greedily fill the protected pool."""
    if alpha <= 0.0 or num_ssu <= 0:
        raise ValueError("alpha and num_ssu must be positive")

    candidates = []
    for request in requests:
        values = _request_values(request, num_ssu, alpha)
        candidates.append((request, values))
    candidates.sort(key=lambda item: (-item[1][3], item[0].request_id))

    protected_sum = [0.0] * num_ssu
    decisions = []
    for rank, (request, values) in enumerate(candidates, start=1):
        compute_s, layer_kv_gib, required_b, value_v, raw, guarded = values
        guarded_total = math.fsum(guarded)
        limiting_ssus = tuple(
            ssu_id
            for ssu_id, value in enumerate(guarded)
            if protected_sum[ssu_id] + value > disk_bandwidth_gibps + 1e-12
        )
        link_fits = guarded_total <= npu_bandwidth_gibps + 1e-12
        selected = link_fits and not limiting_ssus
        if selected:
            protected_sum = [
                old + value for old, value in zip(protected_sum, guarded)
            ]
            reason = "selected_by_descending_V"
        elif not link_fits:
            reason = "rejected_by_npu_link_capacity"
        else:
            reason = "rejected_by_ssu_capacity"
        decisions.append(
            RequestVB(
                request_id=int(request.request_id),
                npu_id=int(request.npu_id),
                rank=rank,
                compute_window_s=compute_s,
                layer_kv_gib=layer_kv_gib,
                required_b_gibps=required_b,
                value_v_s2_per_gib=value_v,
                raw_demand_by_ssu_gibps=raw,
                guarded_demand_by_ssu_gibps=guarded,
                guarded_npu_demand_gibps=guarded_total,
                selected=selected,
                assigned_category=(
                    PROTECTED_CATEGORY if selected else BACKGROUND_CATEGORY
                ),
                decision_reason=reason,
                limiting_ssu_ids=limiting_ssus,
            )
        )

    protected_ids = tuple(
        decision.request_id for decision in decisions if decision.selected
    )
    background_ids = tuple(
        decision.request_id for decision in decisions if not decision.selected
    )
    protected_cir = tuple(protected_sum)
    background_cir = tuple(
        disk_bandwidth_gibps - value for value in protected_cir
    )
    configs = tuple(
        _qos_config(protected, background)
        for protected, background in zip(protected_cir, background_cir)
    )
    return VBPoolPlan(
        alpha=float(alpha),
        disk_bandwidth_gibps=float(disk_bandwidth_gibps),
        npu_bandwidth_gibps=float(npu_bandwidth_gibps),
        decisions=tuple(decisions),
        protected_request_ids=protected_ids,
        background_request_ids=background_ids,
        protected_cir_by_ssu_gibps=protected_cir,
        background_cir_by_ssu_gibps=background_cir,
        qos_configs_by_ssu=configs,
    )


def relabel_requests_for_vb_pools(
    requests: Iterable[ContinuousBatchRequest],
    plan: VBPoolPlan,
) -> tuple[ContinuousBatchRequest, ...]:
    """Clone manifests, changing only the logical QoS routing category."""
    protected = set(plan.protected_request_ids)
    clones = []
    for request in requests:
        load = dict(request.load)
        load["category"] = (
            PROTECTED_CATEGORY
            if request.request_id in protected
            else BACKGROUND_CATEGORY
        )
        clones.append(
            ContinuousBatchRequest(
                request_id=request.request_id,
                npu_id=request.npu_id,
                arrival_time_ms=request.arrival_time_ms,
                load=load,
                placement=request.placement,
            )
        )
    return tuple(clones)


def physical_fields_preserved(
    original: Sequence[ContinuousBatchRequest],
    relabeled: Sequence[ContinuousBatchRequest],
) -> bool:
    """Check that pool relabeling changed no physical workload input."""
    if len(original) != len(relabeled):
        return False
    for left, right in zip(original, relabeled):
        if (
            left.request_id != right.request_id
            or left.npu_id != right.npu_id
            or left.arrival_time_ms != right.arrival_time_ms
            or left.placement != right.placement
        ):
            return False
        left_load = {key: value for key, value in left.load.items() if key != "category"}
        right_load = {
            key: value for key, value in right.load.items() if key != "category"
        }
        if left_load != right_load:
            return False
    return True
