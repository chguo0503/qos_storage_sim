"""Deterministic continuous-prefill request trace and sticky placement.

The formal trace starts eight requests on every NPU after a deterministic
per-NPU 0--5 ms launch jitter.  Each NPU also owns one queued replacement whose
arrival is sampled in a fixed four-to-twelve layer-equivalent window.  A token
block is ring-hashed once and the resulting single-layer placement is reused
by all sixteen model layers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import statistics
import struct

import numpy as np

import sim


DEFAULT_NUM_NPU = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_NEW_REQUESTS_PER_NPU = 1
DEFAULT_N_LAYERS = 16
DEFAULT_NUM_SSU = 28
DEFAULT_SEED = 42
DEFAULT_INITIAL_NPU_JITTER_MS = (0.0, 5.0)
DEFAULT_ARRIVAL_LAYER_WINDOW = (4.0, 12.0)

_RNG_KEY = struct.Struct("!QIII")


def _hash_json(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_json_rows(rows) -> str:
    """Hash an iterable as the same compact JSON array without materializing it."""
    digest = hashlib.sha256()
    digest.update(b"[")
    separator = b""
    for row in rows:
        digest.update(separator)
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        separator = b","
    digest.update(b"]")
    return digest.hexdigest()


def _rng(seed: int, npu_id: int, stream_id: int, generation: int, namespace: bytes):
    digest = hashlib.sha256(namespace)
    digest.update(_RNG_KEY.pack(seed, npu_id, stream_id, generation))
    return np.random.RandomState(int.from_bytes(digest.digest()[:4], "big"))


def _profile_catalog(table):
    return {
        category: tuple(
            key
            for key in sorted(table)
            if sim.classify_request(*key) == category
        )
        for category in sim.WORKLOAD_CATEGORIES
    }


def _sticky_placement(request_id, seq_len_k, nql, per_layer_kv_gb, num_ssu):
    _, _, ssd_tokens = sim.calculate_token_partition(seq_len_k, nql)
    if ssd_tokens <= 0 or per_layer_kv_gb <= 0.0:
        return ()
    gb_per_token = per_layer_kv_gb / ssd_tokens
    block_count = int(np.ceil(ssd_tokens / sim.BLOCK_SIZE))
    return tuple(
        (
            sim.block_ring_hash_disk_id(request_id, block_index, num_ssu),
            float(
                min(
                    sim.BLOCK_SIZE,
                    ssd_tokens - block_index * sim.BLOCK_SIZE,
                )
                * gb_per_token
            ),
        )
        for block_index in range(block_count)
    )


def _work_by_ssu(placement, num_ssu):
    work = [0.0] * num_ssu
    for ssu_id, block_gb in placement:
        work[ssu_id] += block_gb
    return tuple(work)


@dataclass(frozen=True)
class ContinuousPrefillRequest:
    request_id: int
    npu_id: int
    stream_id: int
    generation: int
    profile_key: tuple[int, int]
    seq_len_k: int
    nql: int
    category: str
    required_bw_input_gbps: float
    per_layer_us: float
    per_layer_kv_gb: float
    arrival_ms: float
    initial: bool
    work_by_ssu_gb: tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "npu_id": self.npu_id,
            "stream_id": self.stream_id,
            "generation": self.generation,
            "profile_key": self.profile_key,
            "seq_len_k": self.seq_len_k,
            "nql": self.nql,
            "category": self.category,
            "required_bw_input_gbps": self.required_bw_input_gbps,
            "per_layer_us": self.per_layer_us,
            "per_layer_kv_gb": self.per_layer_kv_gb,
            "arrival_ms": self.arrival_ms,
            "arrival_time": self.arrival_ms,
            "initial": self.initial,
            "work_by_ssu_gb": self.work_by_ssu_gb,
        }


@dataclass(frozen=True)
class ContinuousPrefillWorkload:
    requests: tuple[ContinuousPrefillRequest, ...]
    placement_by_request: dict[int, tuple[tuple[int, float], ...]]
    num_npu: int
    batch_size: int
    new_requests_per_npu: int
    n_layers: int
    num_ssu: int
    seed: int
    t_layer_ms: float
    initial_npu_jitter_ms: tuple[float, float]
    arrival_layer_window: tuple[float, float]
    workload_hash: str
    placement_hash: str
    trace_hash: str
    statistics: dict

    @property
    def initial_requests(self):
        return tuple(request for request in self.requests if request.initial)

    @property
    def new_requests(self):
        return tuple(request for request in self.requests if not request.initial)

    def request_dicts(self):
        return tuple(request.as_dict() for request in self.requests)

    def expanded_placement(self):
        """Return the legacy ``request -> layer -> blocks`` representation."""
        return {
            request_id: {
                layer: placement for layer in range(self.n_layers)
            }
            for request_id, placement in self.placement_by_request.items()
        }


def _request(
    table,
    catalog,
    *,
    seed,
    request_id,
    npu_id,
    stream_id,
    generation,
    category,
    arrival_ms,
    initial,
    num_ssu,
):
    choices = catalog[category]
    profile_rng = _rng(
        seed,
        npu_id,
        stream_id,
        generation,
        b"continuous-prefill:profile:v1\0",
    )
    key = choices[int(profile_rng.randint(len(choices)))]
    required_bw, per_layer_us, _, per_layer_kv_gb = table[key]
    placement = _sticky_placement(
        request_id,
        key[0],
        key[1],
        float(per_layer_kv_gb),
        num_ssu,
    )
    spec = ContinuousPrefillRequest(
        request_id=request_id,
        npu_id=npu_id,
        stream_id=stream_id,
        generation=generation,
        profile_key=key,
        seq_len_k=key[0],
        nql=key[1],
        category=category,
        required_bw_input_gbps=float(required_bw),
        per_layer_us=float(per_layer_us),
        per_layer_kv_gb=float(per_layer_kv_gb),
        arrival_ms=float(arrival_ms),
        initial=bool(initial),
        work_by_ssu_gb=_work_by_ssu(placement, num_ssu),
    )
    return spec, placement


def _initial_demand(requests, num_npu, num_ssu):
    work = np.zeros((num_npu, num_ssu), dtype=np.float64)
    compute_s = np.zeros(num_npu, dtype=np.float64)
    for request in requests:
        work[request.npu_id] += request.work_by_ssu_gb
        compute_s[request.npu_id] += request.per_layer_us / 1e6
    demand = np.divide(
        work,
        compute_s[:, None],
        out=np.zeros_like(work),
        where=compute_s[:, None] > 0.0,
    )
    return demand


def _fingerprints(requests, placements):
    workload_rows = [request.as_dict() for request in requests]
    workload_hash = _hash_json(workload_rows)
    placement_rows = (
        (request_id, block_index, ssu_id, block_gb)
        for request_id in sorted(placements)
        for block_index, (ssu_id, block_gb) in enumerate(
            placements[request_id]
        )
    )
    placement_hash = _hash_json_rows(placement_rows)
    return (
        workload_hash,
        placement_hash,
        _hash_json(
            {
                "workload": workload_hash,
                "placement": placement_hash,
                "placement_reuse": "all_layers",
            }
        ),
    )


def prepare_continuous_prefill_workload(
    table,
    *,
    num_npu=DEFAULT_NUM_NPU,
    batch_size=DEFAULT_BATCH_SIZE,
    new_requests_per_npu=DEFAULT_NEW_REQUESTS_PER_NPU,
    n_layers=DEFAULT_N_LAYERS,
    num_ssu=DEFAULT_NUM_SSU,
    seed=DEFAULT_SEED,
    initial_npu_jitter_ms=DEFAULT_INITIAL_NPU_JITTER_MS,
    arrival_layer_window=DEFAULT_ARRIVAL_LAYER_WINDOW,
) -> ContinuousPrefillWorkload:
    """Build the paired finite continuous-prefill workload used by all policies."""
    num_npu = int(num_npu)
    batch_size = int(batch_size)
    new_requests_per_npu = int(new_requests_per_npu)
    n_layers = int(n_layers)
    num_ssu = int(num_ssu)
    seed = int(seed)
    catalog = _profile_catalog(table)
    if any(not catalog[category] for category in sim.WORKLOAD_CATEGORIES):
        raise ValueError("the profile table must contain all four workload categories")

    requests = []
    placements = {}
    request_id = 0
    initial_by_npu = [[] for _ in range(num_npu)]
    jitter_low, jitter_high = map(float, initial_npu_jitter_ms)
    initial_start_ms = tuple(
        float(
            _rng(
                seed,
                npu_id,
                0,
                0,
                b"continuous-prefill:initial-npu-jitter:v1\0",
            ).uniform(jitter_low, jitter_high)
        )
        for npu_id in range(num_npu)
    )
    for npu_id in range(num_npu):
        for slot_id in range(batch_size):
            category = sim.WORKLOAD_CATEGORIES[
                (npu_id + slot_id) % len(sim.WORKLOAD_CATEGORIES)
            ]
            spec, placement = _request(
                table,
                catalog,
                seed=seed,
                request_id=request_id,
                npu_id=npu_id,
                stream_id=slot_id,
                generation=0,
                category=category,
                arrival_ms=initial_start_ms[npu_id],
                initial=True,
                num_ssu=num_ssu,
            )
            requests.append(spec)
            placements[request_id] = placement
            initial_by_npu[npu_id].append(spec)
            request_id += 1

    per_npu_layer_ms = tuple(
        sum(request.per_layer_us for request in batch) / 1000.0
        for batch in initial_by_npu
    )
    t_layer_ms = float(statistics.median(per_npu_layer_ms))
    arrival_low, arrival_high = map(float, arrival_layer_window)
    for npu_id in range(num_npu):
        for new_index in range(new_requests_per_npu):
            stream_id = batch_size + new_index
            generation = 1
            category = sim.WORKLOAD_CATEGORIES[
                (npu_id + stream_id) % len(sim.WORKLOAD_CATEGORIES)
            ]
            arrival_rng = _rng(
                seed,
                npu_id,
                stream_id,
                generation,
                b"continuous-prefill:arrival:v1\0",
            )
            arrival_ms = float(
                arrival_rng.uniform(
                    arrival_low * t_layer_ms,
                    arrival_high * t_layer_ms,
                )
            )
            spec, placement = _request(
                table,
                catalog,
                seed=seed,
                request_id=request_id,
                npu_id=npu_id,
                stream_id=stream_id,
                generation=generation,
                category=category,
                arrival_ms=arrival_ms,
                initial=False,
                num_ssu=num_ssu,
            )
            requests.append(spec)
            placements[request_id] = placement
            request_id += 1

    requests = tuple(requests)
    initial = tuple(request for request in requests if request.initial)
    demand = _initial_demand(initial, num_npu, num_ssu)
    workload_hash, placement_hash, trace_hash = _fingerprints(
        requests, placements
    )
    initial_categories = Counter(request.category for request in initial)
    new_categories = Counter(
        request.category for request in requests if not request.initial
    )
    per_ssu_demand = np.sum(demand, axis=0)
    statistics_payload = {
        "initial_request_count": len(initial),
        "new_request_count": len(requests) - len(initial),
        "total_request_count": len(requests),
        "initial_category_counts": {
            category: initial_categories[category]
            for category in sim.WORKLOAD_CATEGORIES
        },
        "new_category_counts": {
            category: new_categories[category]
            for category in sim.WORKLOAD_CATEGORIES
        },
        "per_npu_initial_layer_compute_ms": list(per_npu_layer_ms),
        "t_layer_ms": t_layer_ms,
        "initial_total_demand_gbps": float(np.sum(demand)),
        "initial_per_ssu_demand_gbps": per_ssu_demand.tolist(),
        "ssd_capacity_gbps": num_ssu * sim.DISK_BW,
        "initial_demand_to_ssd_capacity": float(
            np.sum(demand) / (num_ssu * sim.DISK_BW)
        ),
        "initial_npu_jitter_window_ms": [jitter_low, jitter_high],
        "initial_npu_start_ms": list(initial_start_ms),
        "arrival_window_ms": [
            arrival_low * t_layer_ms,
            arrival_high * t_layer_ms,
        ],
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "placement_reuse_layers": n_layers,
    }
    return ContinuousPrefillWorkload(
        requests=requests,
        placement_by_request=placements,
        num_npu=num_npu,
        batch_size=batch_size,
        new_requests_per_npu=new_requests_per_npu,
        n_layers=n_layers,
        num_ssu=num_ssu,
        seed=seed,
        t_layer_ms=t_layer_ms,
        initial_npu_jitter_ms=(jitter_low, jitter_high),
        arrival_layer_window=(arrival_low, arrival_high),
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        trace_hash=trace_hash,
        statistics=statistics_payload,
    )
