"""Balanced saturated request streams for steady-state batch-1 experiments."""

from __future__ import annotations

from collections import Counter
import statistics

import sim
from continuous_prefill_workload import (
    ContinuousPrefillWorkload,
    _fingerprints,
    _hash_json,
    _rng,
)
from six_request_workload import (
    BALANCED_PROFILES,
    INITIAL_JITTER_MS,
    NUM_NPU,
    SEED,
    _request_with_profile,
    representative_profiles,
)


REQUESTS_PER_NPU = 32
PROFILE_CYCLE = tuple(sim.WORKLOAD_CATEGORIES)


def _rotations(num_npu, seed):
    order = _rng(
        seed,
        num_npu,
        len(PROFILE_CYCLE),
        0,
        b"steady-state:npu-profile-rotations:v1\0",
    ).permutation(num_npu)
    rotations = [0] * num_npu
    for rank, npu_id in enumerate(order):
        rotations[int(npu_id)] = rank % len(PROFILE_CYCLE)
    return tuple(rotations)


def prepare_steady_state_workload(
    table,
    *,
    num_npu=NUM_NPU,
    n_layers=16,
    num_ssu=16,
    requests_per_npu=REQUESTS_PER_NPU,
    seed=SEED,
):
    """Create a prefix-stable, category-balanced saturated stream."""
    rotations = _rotations(num_npu, seed)
    initial_start_ms = tuple(
        float(
            _rng(seed, npu_id, 0, 0, b"steady-state:initial-jitter:v1\0").uniform(
                *INITIAL_JITTER_MS
            )
        )
        for npu_id in range(num_npu)
    )
    requests = []
    placements = {}
    for sequence in range(requests_per_npu):
        for npu_id in range(num_npu):
            category = PROFILE_CYCLE[
                (sequence + rotations[npu_id]) % len(PROFILE_CYCLE)
            ]
            request_id = sequence * num_npu + npu_id
            request, placement = _request_with_profile(
                table,
                request_id=request_id,
                npu_id=npu_id,
                sequence=sequence,
                profile_key=BALANCED_PROFILES[category],
                arrival_ms=initial_start_ms[npu_id],
                num_ssu=num_ssu,
            )
            requests.append(request)
            placements[request_id] = placement

    requests = tuple(requests)
    workload_hash, placement_hash, trace_hash = _fingerprints(requests, placements)
    cycle_compute_ms = sum(
        table[BALANCED_PROFILES[category]][1] / 1000.0
        for category in PROFILE_CYCLE
    )
    cycle_kv_gb = sum(
        table[BALANCED_PROFILES[category]][3]
        for category in PROFILE_CYCLE
    )
    sequence_category_counts = {
        str(sequence): dict(
            Counter(
                request.category
                for request in requests
                if request.stream_id == sequence
            )
        )
        for sequence in range(len(PROFILE_CYCLE))
    }
    assignment_hash = _hash_json(
        [
            (
                request.request_id,
                request.npu_id,
                request.stream_id,
                request.profile_key,
                request.category,
            )
            for request in requests
        ]
    )
    return ContinuousPrefillWorkload(
        requests=requests,
        placement_by_request=placements,
        num_npu=num_npu,
        batch_size=1,
        new_requests_per_npu=requests_per_npu - 1,
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
            "assignment_hash": assignment_hash,
            "request_count": len(requests),
            "requests_per_npu": requests_per_npu,
            "profile_cycle": list(PROFILE_CYCLE),
            "rotation_counts": dict(Counter(rotations)),
            "sequence_category_counts": sequence_category_counts,
            "representative_profiles": [
                list(key) for key in representative_profiles(table)
            ],
            "cycle_compute_ms_per_npu": cycle_compute_ms,
            "cycle_kv_gb_per_npu": cycle_kv_gb,
            "per_npu_demand_gbps": cycle_kv_gb / (cycle_compute_ms / 1000.0),
            "fleet_demand_gbps": num_npu
            * cycle_kv_gb
            / (cycle_compute_ms / 1000.0),
            "capacity_knee_ssu": num_npu
            * cycle_kv_gb
            / (cycle_compute_ms / 1000.0)
            / sim.DISK_BW,
            "initial_npu_start_ms": list(initial_start_ms),
            "request_id_formula": "sequence * num_npu + npu_id",
            "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
            "placement_reuse_layers": n_layers,
        },
    )
