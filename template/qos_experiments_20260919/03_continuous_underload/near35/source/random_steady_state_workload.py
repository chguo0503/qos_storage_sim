"""Reproducible heterogeneous request streams built from the ``data`` catalog."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import statistics

import sim
from continuous_prefill_workload import (
    ContinuousPrefillWorkload,
    _fingerprints,
    _rng,
)
from six_request_workload import INITIAL_JITTER_MS, _request_with_profile


SCHEMA_VERSION = 1
STRATIFIED_RANDOM_CATALOG_V1 = "stratified_random_catalog_v1"
IID_UNIFORM_CATEGORY_CATALOG_V1 = "iid_uniform_category_catalog_v1"
IID_UNIFORM_PROFILE_CATALOG_V1 = "iid_uniform_profile_catalog_v1"
SAMPLING_MODES = (
    STRATIFIED_RANDOM_CATALOG_V1,
    IID_UNIFORM_CATEGORY_CATALOG_V1,
    IID_UNIFORM_PROFILE_CATALOG_V1,
)
CATEGORIES = tuple(sim.WORKLOAD_CATEGORIES)
PREFIX_HASH_REQUESTS_PER_NPU = 32


def _canonical_hash(value, namespace=b""):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _catalog(table):
    catalog = {
        category: tuple(
            key for key in sorted(table) if sim.classify_request(*key) == category
        )
        for category in CATEGORIES
    }
    missing = tuple(category for category, keys in catalog.items() if not keys)
    if missing:
        raise ValueError(f"data 中缺少请求类别: {missing}")
    return catalog


def catalog_hash(table):
    return _canonical_hash(
        [
            [list(key), [float(value) for value in table[key]]]
            for key in sorted(table)
        ],
        b"random-steady-state:data-catalog:v1\0",
    )


@dataclass(frozen=True)
class SteadyStateProfileSchedule:
    schema_version: int
    mode: str
    seed: int
    num_npu: int
    requests_per_npu: int
    catalog_hash: str
    recipe_hash: str
    assignment_hash: str
    schedule_hash: str
    assignments: tuple[tuple[int, int, int, str, tuple[int, int]], ...]

    def as_fingerprint_dict(self):
        return {
            "catalog": self.catalog_hash,
            "recipe": self.recipe_hash,
            "schedule": self.schedule_hash,
            "assignment": self.assignment_hash,
        }

    @property
    def prefix_32_assignment_hash(self):
        """Hash the first 32 requests/NPU in the full-assignment hash domain.

        This value for an extended schedule therefore equals the
        ``assignment_hash`` of the independently generated 32-request schedule.
        ``None`` means this finite backing schedule is too short.
        """
        if self.requests_per_npu < PREFIX_HASH_REQUESTS_PER_NPU:
            return None
        prefix = tuple(
            assignment
            for assignment in self.assignments
            if assignment[2] < PREFIX_HASH_REQUESTS_PER_NPU
        )
        return _assignment_hash(prefix)

    @property
    def full_assignment_hash(self):
        """Explicit alias for reports that show prefix and full hashes."""
        return self.assignment_hash


def _assignment_rows(assignments):
    return [
        [request_id, npu_id, sequence, category, list(profile_key)]
        for request_id, npu_id, sequence, category, profile_key in assignments
    ]


def _assignment_hash(assignments):
    return _canonical_hash(
        _assignment_rows(assignments),
        b"random-steady-state:assignments:v1\0",
    )


def _balanced_rotations(*, seed, block_index, num_npu):
    if num_npu % len(CATEGORIES):
        raise ValueError("stratified 模式要求 NPU 数能被类别数整除")
    order = _rng(
        seed,
        block_index,
        num_npu,
        0,
        b"random-steady-state:balanced-npu-rotations:v1\0",
    ).permutation(num_npu)
    rotations = [0] * num_npu
    for rank, npu_id in enumerate(order):
        rotations[int(npu_id)] = rank % len(CATEGORIES)
    return tuple(rotations)


def _stratified_categories(*, seed, block_index, num_npu):
    category_order = _rng(
        seed,
        block_index,
        len(CATEGORIES),
        0,
        b"random-steady-state:category-order:v1\0",
    ).permutation(len(CATEGORIES))
    rotations = _balanced_rotations(
        seed=seed,
        block_index=block_index,
        num_npu=num_npu,
    )
    return tuple(
        tuple(
            CATEGORIES[
                int(category_order[(position + rotations[npu_id]) % len(CATEGORIES)])
            ]
            for npu_id in range(num_npu)
        )
        for position in range(len(CATEGORIES))
    )


def _iid_category(*, seed, npu_id, sequence):
    rng = _rng(
        seed,
        npu_id,
        sequence,
        0,
        b"random-steady-state:iid-uniform-category:v1\0",
    )
    return CATEGORIES[int(rng.randint(len(CATEGORIES)))]


def _iid_uniform_profile(profile_keys, *, seed, npu_id, sequence):
    """Select one catalog profile IID with replacement for one NPU request.

    The random stream is addressed only by ``(seed, npu_id, sequence)``. It is
    independent across NPUs and prefix-stable when a finite backing schedule is
    extended from 32 requests/NPU to a longer sequence.
    """
    rng = _rng(
        seed,
        npu_id,
        sequence,
        0,
        b"random-steady-state:iid-uniform-profile:v1\0",
    )
    return profile_keys[int(rng.randint(len(profile_keys)))]


def _profile_from_shuffle_bag(
    catalog,
    *,
    category,
    occurrence,
    seed,
    npu_id,
):
    choices = catalog[category]
    epoch, offset = divmod(occurrence, len(choices))
    category_index = CATEGORIES.index(category)
    order = _rng(
        seed,
        npu_id,
        category_index,
        epoch,
        b"random-steady-state:profile-shuffle-bag:v1\0",
    ).permutation(len(choices))
    return choices[int(order[offset])]


def build_steady_state_profile_schedule(
    table,
    *,
    mode=STRATIFIED_RANDOM_CATALOG_V1,
    seed=42,
    num_npu=128,
    requests_per_npu=64,
):
    """Pre-generate one immutable profile schedule shared by every strategy."""
    if mode not in SAMPLING_MODES:
        raise ValueError(f"mode 必须是 {SAMPLING_MODES} 之一")
    if num_npu <= 0 or requests_per_npu <= 0:
        raise ValueError("num_npu 和 requests_per_npu 必须为正数")
    catalog = _catalog(table)
    profile_keys = tuple(sorted(table))
    data_hash = catalog_hash(table)
    if mode == STRATIFIED_RANDOM_CATALOG_V1:
        category_policy = (
            "one of each category per NPU per four-request block; "
            "each fleet sequence has equal category counts"
        )
        profile_policy = "per-NPU per-category deterministic shuffle bag"
    elif mode == IID_UNIFORM_CATEGORY_CATALOG_V1:
        category_policy = "IID uniform category per (NPU, sequence)"
        profile_policy = "per-NPU per-category deterministic shuffle bag"
    else:
        category_policy = (
            "category derived from an IID uniformly selected catalog profile"
        )
        profile_policy = (
            "IID uniform over the sorted data catalog with replacement per "
            "(seed, NPU, sequence); prefix-stable when extended"
        )
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "seed": int(seed),
        "num_npu": int(num_npu),
        "requests_per_npu": int(requests_per_npu),
        "categories": list(CATEGORIES),
        "category_policy": category_policy,
        "profile_policy": profile_policy,
        "request_order": "sequence-major",
        "request_id": "sequence * num_npu + npu_id",
        "initial_jitter": "existing deterministic 0-5 ms rule",
        "catalog_hash": data_hash,
    }
    recipe_hash = _canonical_hash(
        recipe,
        b"random-steady-state:recipe:v1\0",
    )

    category_occurrences = defaultdict(int)
    assignments = []
    stratified_blocks = {}
    for sequence in range(requests_per_npu):
        if mode == IID_UNIFORM_PROFILE_CATALOG_V1:
            selected_profiles = tuple(
                _iid_uniform_profile(
                    profile_keys,
                    seed=seed,
                    npu_id=npu_id,
                    sequence=sequence,
                )
                for npu_id in range(num_npu)
            )
            categories = tuple(
                sim.classify_request(*profile_key)
                for profile_key in selected_profiles
            )
        elif mode == STRATIFIED_RANDOM_CATALOG_V1:
            block_index, position = divmod(sequence, len(CATEGORIES))
            if block_index not in stratified_blocks:
                stratified_blocks[block_index] = _stratified_categories(
                    seed=seed,
                    block_index=block_index,
                    num_npu=num_npu,
                )
            categories = stratified_blocks[block_index][position]
        else:
            categories = tuple(
                _iid_category(seed=seed, npu_id=npu_id, sequence=sequence)
                for npu_id in range(num_npu)
            )
        for npu_id, category in enumerate(categories):
            if mode == IID_UNIFORM_PROFILE_CATALOG_V1:
                profile_key = selected_profiles[npu_id]
            else:
                occurrence_key = (npu_id, category)
                occurrence = category_occurrences[occurrence_key]
                category_occurrences[occurrence_key] += 1
                profile_key = _profile_from_shuffle_bag(
                    catalog,
                    category=category,
                    occurrence=occurrence,
                    seed=seed,
                    npu_id=npu_id,
                )
            request_id = sequence * num_npu + npu_id
            assignments.append(
                (request_id, npu_id, sequence, category, profile_key)
            )

    assignment_rows = _assignment_rows(assignments)
    assignment_hash = _assignment_hash(assignments)
    schedule_hash = _canonical_hash(
        {"recipe_hash": recipe_hash, "assignments": assignment_rows},
        b"random-steady-state:schedule:v1\0",
    )
    return SteadyStateProfileSchedule(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        seed=int(seed),
        num_npu=int(num_npu),
        requests_per_npu=int(requests_per_npu),
        catalog_hash=data_hash,
        recipe_hash=recipe_hash,
        assignment_hash=assignment_hash,
        schedule_hash=schedule_hash,
        assignments=tuple(assignments),
    )


def _schedule_statistics(table, schedule, requests, num_ssu):
    requests_by_npu = defaultdict(list)
    for request in requests:
        requests_by_npu[request.npu_id].append(request)
    per_npu_compute_s = tuple(
        sum(request.per_layer_us for request in requests_by_npu[npu_id]) / 1e6
        for npu_id in range(schedule.num_npu)
    )
    per_npu_kv_gb = tuple(
        sum(request.per_layer_kv_gb for request in requests_by_npu[npu_id])
        for npu_id in range(schedule.num_npu)
    )
    per_npu_demand = tuple(
        kv_gb / compute_s
        for kv_gb, compute_s in zip(per_npu_kv_gb, per_npu_compute_s)
    )
    per_npu_ms_per_gb = tuple(
        1000.0 * compute_s / kv_gb
        for compute_s, kv_gb in zip(per_npu_compute_s, per_npu_kv_gb)
    )
    demand_mean = statistics.mean(per_npu_demand)
    demand_cv = (
        statistics.pstdev(per_npu_demand) / demand_mean
        if demand_mean > 0.0
        else 0.0
    )
    ms_per_gb_mean = statistics.mean(per_npu_ms_per_gb)
    ms_per_gb_spread = max(per_npu_ms_per_gb) - min(per_npu_ms_per_gb)
    demand_by_ssu = [0.0] * num_ssu
    for npu_id, compute_s in enumerate(per_npu_compute_s):
        for request in requests_by_npu[npu_id]:
            for ssu_id, work_gb in enumerate(request.work_by_ssu_gb):
                demand_by_ssu[ssu_id] += work_gb / compute_s
    category_counts = Counter(request.category for request in requests)
    profile_counts = Counter(request.profile_key for request in requests)
    all_profile_keys = tuple(sorted(table))
    per_npu_category_ranges = {}
    for category in CATEGORIES:
        counts = tuple(
            sum(request.category == category for request in requests_by_npu[npu_id])
            for npu_id in range(schedule.num_npu)
        )
        per_npu_category_ranges[category] = {
            "min": min(counts),
            "mean": statistics.mean(counts),
            "max": max(counts),
        }
    required_bw = [float(table[request.profile_key][0]) for request in requests]
    return {
        "fleet_category_counts": dict(category_counts),
        "fleet_category_counts_all": {
            category: category_counts.get(category, 0)
            for category in CATEGORIES
        },
        "fleet_profile_counts": {
            f"{key[0]},{key[1]}": count
            for key, count in sorted(profile_counts.items())
        },
        "fleet_profile_counts_all": {
            f"{key[0]},{key[1]}": profile_counts.get(key, 0)
            for key in all_profile_keys
        },
        "catalog_profile_count": len(all_profile_keys),
        "profiles_used": len(profile_counts),
        "per_npu_category_count_ranges": per_npu_category_ranges,
        "per_npu_compute_s_range": [
            min(per_npu_compute_s),
            max(per_npu_compute_s),
        ],
        "per_npu_kv_gb_range": [min(per_npu_kv_gb), max(per_npu_kv_gb)],
        "per_npu_demand_gbps_range": [
            min(per_npu_demand),
            max(per_npu_demand),
        ],
        "per_npu_demand_gbps_mean": demand_mean,
        "per_npu_demand_gbps_cv": demand_cv,
        "per_npu_raw_demand_gbps": {
            "min": min(per_npu_demand),
            "max": max(per_npu_demand),
            "mean": demand_mean,
            "coefficient_of_variation": demand_cv,
        },
        "per_npu_ms_per_gb": {
            "min": min(per_npu_ms_per_gb),
            "max": max(per_npu_ms_per_gb),
            "mean": ms_per_gb_mean,
            "spread": ms_per_gb_spread,
            "spread_pct_of_mean": (
                100.0 * ms_per_gb_spread / ms_per_gb_mean
                if ms_per_gb_mean > 0.0
                else 0.0
            ),
        },
        "fleet_demand_gbps": sum(per_npu_demand),
        "capacity_knee_ssu": sum(per_npu_demand) / sim.DISK_BW,
        "demand_gbps_by_ssu": demand_by_ssu,
        "max_ssu_demand_gbps": max(demand_by_ssu),
        "ssu_over_40_count": sum(value > sim.DISK_BW for value in demand_by_ssu),
        "required_bw_profile_bins": {
            "le_50": sum(value <= sim.NPU_BW_LIMIT for value in required_bw),
            "gt_50_le_100": sum(
                sim.NPU_BW_LIMIT < value <= 2.0 * sim.NPU_BW_LIMIT
                for value in required_bw
            ),
            "gt_100": sum(value > 2.0 * sim.NPU_BW_LIMIT for value in required_bw),
        },
    }


def prepare_random_steady_state_workload(
    table,
    *,
    schedule,
    num_ssu,
    n_layers=16,
):
    """Materialize placement for one topology without changing the schedule."""
    if catalog_hash(table) != schedule.catalog_hash:
        raise ValueError("data catalog 与 schedule 指纹不一致")
    if num_ssu <= 0 or n_layers <= 0:
        raise ValueError("num_ssu 和 n_layers 必须为正数")
    initial_start_ms = tuple(
        float(
            _rng(
                schedule.seed,
                npu_id,
                0,
                0,
                b"steady-state:initial-jitter:v1\0",
            ).uniform(*INITIAL_JITTER_MS)
        )
        for npu_id in range(schedule.num_npu)
    )
    requests = []
    placements = {}
    for request_id, npu_id, sequence, category, profile_key in schedule.assignments:
        if sim.classify_request(*profile_key) != category:
            raise AssertionError("schedule 类别与 data 画像不一致")
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
    statistics_payload = _schedule_statistics(table, schedule, requests, num_ssu)
    statistics_payload.update(
        {
            **schedule.as_fingerprint_dict(),
            "prefix_32_assignment_hash": schedule.prefix_32_assignment_hash,
            "full_assignment_hash": schedule.full_assignment_hash,
            "prefix_hash_requests_per_npu": PREFIX_HASH_REQUESTS_PER_NPU,
            "profile_sampling": schedule.mode,
            "profile_sampling_uniform_over_catalog": (
                schedule.mode == IID_UNIFORM_PROFILE_CATALOG_V1
            ),
            "profile_sampling_with_replacement": (
                schedule.mode == IID_UNIFORM_PROFILE_CATALOG_V1
            ),
            "profile_sequence_prefix_stable": True,
            "request_count": len(requests),
            "requests_per_npu": schedule.requests_per_npu,
            "seed": schedule.seed,
            "request_id_formula": "sequence * num_npu + npu_id",
            "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
            "placement_reuse_layers": n_layers,
            "initial_npu_start_ms": list(initial_start_ms),
        }
    )
    return ContinuousPrefillWorkload(
        requests=requests,
        placement_by_request=placements,
        num_npu=schedule.num_npu,
        batch_size=1,
        new_requests_per_npu=schedule.requests_per_npu - 1,
        n_layers=n_layers,
        num_ssu=num_ssu,
        seed=schedule.seed,
        t_layer_ms=float(
            statistics.median(request.per_layer_us / 1000.0 for request in requests)
        ),
        initial_npu_jitter_ms=INITIAL_JITTER_MS,
        arrival_layer_window=(0.0, 0.0),
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        trace_hash=trace_hash,
        statistics=statistics_payload,
    )
