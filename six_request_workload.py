"""Balanced six-request streams for the fastest-finish cutoff experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics

import sim
from closed_loop_workload import _request_with_profile
from continuous_prefill_workload import (
    ContinuousPrefillWorkload,
    _fingerprints,
    _hash_json,
    _rng,
)


NUM_NPU = 128
REQUESTS_PER_NPU = 6
SLO_REQUESTS_PER_NPU = 5
CUTOFF_SEQUENCE = REQUESTS_PER_NPU - 1
LAYER_LIST = (16, 24, 56, 80)
SSU_LIST = (16, 24, 56)
SEED = 42
INITIAL_JITTER_MS = (0.0, 5.0)

# Four calibrated profiles make the fleet demand/capacity knee about 67 SSUs.
BALANCED_PROFILES = {
    "SS": (80, 64),
    "SL": (80, 512),
    "LS": (144, 256),
    "LL": (144, 512),
}
GUARD_GROUPS = ("A_SS", "A_LL", "B_SL", "B_LS")
TEMPLATE_CATEGORIES = {
    "A": ("SS", "SL", "LS", "LL", "SS", "LL"),
    "B": ("SS", "SL", "LS", "LL", "SL", "LS"),
}


def representative_profiles(table):
    """Return the four calibrated SS/SL/LS/LL stress profiles."""
    return tuple(BALANCED_PROFILES[category] for category in sim.WORKLOAD_CATEGORIES)


def _balanced_profile_orders(*, num_npu, seed):
    """Assign four balanced guard groups and rotate each five-request prefix."""
    group_ids = [index % len(GUARD_GROUPS) for index in range(num_npu)]
    _rng(seed, num_npu, 6, 0, b"six-request:guard-groups:v2\0").shuffle(group_ids)
    member_rank = defaultdict(int)
    orders = []
    groups = []
    for group_id in group_ids:
        group = GUARD_GROUPS[group_id]
        template_name, guard_category = group.split("_")
        categories = list(TEMPLATE_CATEGORIES[template_name])
        categories.remove(guard_category)
        base_order = _rng(
            seed,
            group_id,
            len(categories),
            0,
            b"six-request:group-prefix-order:v2\0",
        ).permutation(len(categories))
        measured = tuple(categories[int(index)] for index in base_order)
        offset = member_rank[group] % len(measured)
        member_rank[group] += 1
        ordered_categories = tuple(
            measured[(sequence + offset) % len(measured)]
            for sequence in range(len(measured))
        ) + (guard_category,)
        orders.append(tuple(BALANCED_PROFILES[category] for category in ordered_categories))
        groups.append(group)
    return tuple(orders), tuple(groups)


def prepare_six_request_workload(
    table,
    *,
    num_npu=NUM_NPU,
    n_layers=16,
    num_ssu=28,
    seed=SEED,
):
    """Create paired batch-1 streams with equal total work on every NPU."""
    profiles = representative_profiles(table)
    profile_orders, guard_groups = _balanced_profile_orders(
        num_npu=num_npu, seed=seed
    )
    initial_start_ms = tuple(
        float(
            _rng(seed, npu_id, 0, 0, b"six-request:initial-jitter:v1\0").uniform(
                *INITIAL_JITTER_MS
            )
        )
        for npu_id in range(num_npu)
    )

    requests = []
    placements = {}
    for npu_id, order in enumerate(profile_orders):
        for sequence, profile_key in enumerate(order):
            request_id = npu_id * REQUESTS_PER_NPU + sequence
            request, placement = _request_with_profile(
                table,
                request_id=request_id,
                npu_id=npu_id,
                sequence=sequence,
                profile_key=profile_key,
                arrival_ms=initial_start_ms[npu_id],
                num_ssu=num_ssu,
            )
            requests.append(request)
            placements[request_id] = placement

    requests = tuple(requests)
    workload_hash, placement_hash, trace_hash = _fingerprints(requests, placements)
    assignment_rows = [
        (
            request.request_id,
            request.npu_id,
            request.stream_id,
            request.profile_key,
            request.category,
            request.per_layer_us,
            request.per_layer_kv_gb,
        )
        for request in requests
    ]
    sequence_counts = {
        str(sequence): dict(
            Counter(
                f"{request.profile_key[0]},{request.profile_key[1]}"
                for request in requests
                if request.stream_id == sequence
            )
        )
        for sequence in range(REQUESTS_PER_NPU)
    }
    per_npu_compute_ms = [
        sum(table[key][1] for key in order) / 1000.0 for order in profile_orders
    ]
    per_npu_kv_gb = [sum(table[key][3] for key in order) for order in profile_orders]
    per_npu_demand_gbps = [
        kv_gb / (compute_ms / 1000.0)
        for kv_gb, compute_ms in zip(per_npu_kv_gb, per_npu_compute_ms)
    ]
    return ContinuousPrefillWorkload(
        requests=requests,
        placement_by_request=placements,
        num_npu=num_npu,
        batch_size=1,
        new_requests_per_npu=REQUESTS_PER_NPU - 1,
        n_layers=n_layers,
        num_ssu=num_ssu,
        seed=seed,
        t_layer_ms=float(
            statistics.median(request.per_layer_us / 1000.0 for request in requests)
        ),
        initial_npu_jitter_ms=INITIAL_JITTER_MS,
        arrival_layer_window=(0.0, 0.0),
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        trace_hash=trace_hash,
        statistics={
            "assignment_hash": _hash_json(assignment_rows),
            "request_count": len(requests),
            "requests_per_npu": REQUESTS_PER_NPU,
            "slo_requests_per_npu": SLO_REQUESTS_PER_NPU,
            "representative_profiles": [list(key) for key in profiles],
            "template_categories": {
                name: list(categories)
                for name, categories in TEMPLATE_CATEGORIES.items()
            },
            "guard_group_counts": dict(Counter(guard_groups)),
            "fleet_category_counts": {
                category: sum(request.category == category for request in requests)
                for category in sim.WORKLOAD_CATEGORIES
            },
            "per_npu_per_layer_compute_ms_min": min(per_npu_compute_ms),
            "per_npu_per_layer_compute_ms_max": max(per_npu_compute_ms),
            "per_npu_per_layer_kv_gb_min": min(per_npu_kv_gb),
            "per_npu_per_layer_kv_gb_max": max(per_npu_kv_gb),
            "per_npu_demand_gbps_min": min(per_npu_demand_gbps),
            "per_npu_demand_gbps_max": max(per_npu_demand_gbps),
            "fleet_demand_gbps": sum(per_npu_demand_gbps),
            "capacity_knee_ssu": sum(per_npu_demand_gbps) / sim.DISK_BW,
            "sequence_profile_counts": sequence_counts,
            "profile_balance": (
                "64 near-identical A-template and 64 B-template NPUs; per-NPU "
                "six-request compute/KV spread below 0.1%; four fleet categories "
                "are exactly balanced; first-five prefixes are group-rotated"
            ),
            "initial_npu_start_ms": list(initial_start_ms),
            "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
            "placement_reuse_layers": n_layers,
        },
    )
