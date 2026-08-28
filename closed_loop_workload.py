"""Deterministic batch-1 request streams for the closed-loop QoS experiment."""

from __future__ import annotations

import statistics

import sim
from continuous_prefill_workload import (
    ContinuousPrefillRequest,
    ContinuousPrefillWorkload,
    _fingerprints,
    _hash_json,
    _profile_catalog,
    _rng,
    _sticky_placement,
    _work_by_ssu,
)


NUM_NPU = 128
REQUESTS_PER_NPU = 64
WARMUP_REQUESTS = 8
MEASURED_REQUESTS = 48
N_LAYERS = 16
SSU_LIST = (8, 16, 28, 40, 56, 80, 112)
SEED = 42
INITIAL_JITTER_MS = (0.0, 5.0)
FORMAL_WINDOWS = (
    ("warmup", 0, WARMUP_REQUESTS),
    ("measured", WARMUP_REQUESTS, MEASURED_REQUESTS),
    (
        "cooldown",
        WARMUP_REQUESTS + MEASURED_REQUESTS,
        REQUESTS_PER_NPU - WARMUP_REQUESTS - MEASURED_REQUESTS,
    ),
)


def _windows(requests_per_npu):
    if requests_per_npu == REQUESTS_PER_NPU:
        return FORMAL_WINDOWS
    return (("all", 0, requests_per_npu),)


def _window_profiles(catalog, *, seed, window_name, count):
    """Choose one common balanced profile multiset for every NPU."""
    categories = sim.WORKLOAD_CATEGORIES
    per_category, remainder = divmod(count, len(categories))
    category_order = _rng(
        seed,
        0,
        count,
        0,
        f"closed-loop:{window_name}:category-remainder:v2\0".encode(),
    ).permutation(len(categories))
    category_counts = [per_category] * len(categories)
    for index in category_order[:remainder]:
        category_counts[int(index)] += 1
    profiles = []
    for category_index, category in enumerate(categories):
        choices = catalog[category]
        order = _rng(
            seed,
            category_index,
            count,
            0,
            f"closed-loop:{window_name}:profile-set:v2\0".encode(),
        ).permutation(len(choices))
        profiles.extend(
            choices[int(index)]
            for index in order[: category_counts[category_index]]
        )
    return tuple(profiles)


def _request_with_profile(
    table,
    *,
    request_id,
    npu_id,
    sequence,
    profile_key,
    arrival_ms,
    num_ssu,
):
    required_bw, per_layer_us, _, per_layer_kv_gb = table[profile_key]
    placement = _sticky_placement(
        request_id,
        profile_key[0],
        profile_key[1],
        float(per_layer_kv_gb),
        num_ssu,
    )
    request = ContinuousPrefillRequest(
        request_id=request_id,
        npu_id=npu_id,
        stream_id=sequence,
        generation=0,
        profile_key=profile_key,
        seq_len_k=profile_key[0],
        nql=profile_key[1],
        category=sim.classify_request(*profile_key),
        required_bw_input_gbps=float(required_bw),
        per_layer_us=float(per_layer_us),
        per_layer_kv_gb=float(per_layer_kv_gb),
        arrival_ms=float(arrival_ms),
        initial=sequence == 0,
        work_by_ssu_gb=_work_by_ssu(placement, num_ssu),
    )
    return request, placement


def prepare_closed_loop_workload(
    table,
    *,
    num_npu=NUM_NPU,
    requests_per_npu=REQUESTS_PER_NPU,
    n_layers=N_LAYERS,
    num_ssu=28,
    seed=SEED,
):
    """Pre-generate one random profile sequence per NPU for every policy."""
    catalog = _profile_catalog(table)
    requests = []
    placements = {}
    initial_start_ms = tuple(
        float(
            _rng(seed, npu_id, 0, 0, b"closed-loop:initial-jitter:v1\0").uniform(
                *INITIAL_JITTER_MS
            )
        )
        for npu_id in range(num_npu)
    )
    windows = _windows(requests_per_npu)
    window_profiles = {
        window_name: _window_profiles(
            catalog,
            seed=seed,
            window_name=window_name,
            count=count,
        )
        for window_name, _, count in windows
    }
    for npu_id in range(num_npu):
        for window_name, start, count in windows:
            profiles = window_profiles[window_name]
            order = _rng(
                seed,
                npu_id,
                start,
                count,
                f"closed-loop:{window_name}:request-order:v2\0".encode(),
            ).permutation(count)
            for offset, profile_index in enumerate(order):
                sequence = start + offset
                request_id = npu_id * requests_per_npu + sequence
                request, placement = _request_with_profile(
                    table,
                    request_id=request_id,
                    npu_id=npu_id,
                    sequence=sequence,
                    profile_key=profiles[int(profile_index)],
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
    per_layer_ms = tuple(request.per_layer_us / 1000.0 for request in requests)
    return ContinuousPrefillWorkload(
        requests=requests,
        placement_by_request=placements,
        num_npu=num_npu,
        batch_size=1,
        new_requests_per_npu=requests_per_npu - 1,
        n_layers=n_layers,
        num_ssu=num_ssu,
        seed=seed,
        t_layer_ms=float(statistics.median(per_layer_ms)),
        initial_npu_jitter_ms=INITIAL_JITTER_MS,
        arrival_layer_window=(0.0, 0.0),
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        trace_hash=trace_hash,
        statistics={
            "assignment_hash": _hash_json(assignment_rows),
            "request_count": len(requests),
            "requests_per_npu": requests_per_npu,
            "warmup_requests_per_npu": (
                WARMUP_REQUESTS if requests_per_npu == REQUESTS_PER_NPU else 0
            ),
            "measured_requests_per_npu": (
                MEASURED_REQUESTS
                if requests_per_npu == REQUESTS_PER_NPU
                else requests_per_npu
            ),
            "cooldown_requests_per_npu": (
                requests_per_npu - WARMUP_REQUESTS - MEASURED_REQUESTS
                if requests_per_npu == REQUESTS_PER_NPU
                else 0
            ),
            "category_counts": {
                category: sum(request.category == category for request in requests)
                for category in sim.WORKLOAD_CATEGORIES
            },
            "profile_balance": "same multiset per NPU within warmup/measured/cooldown; independent order",
            "initial_npu_start_ms": list(initial_start_ms),
            "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
            "placement_reuse_layers": n_layers,
        },
    )
