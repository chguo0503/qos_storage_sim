"""Run the Scheme-B continuous-batch control-plane trace.

This trace measures CIR-control feasibility, not end-to-end NPU utilization.
Each ``(NPU, slot)`` owns an independent deterministic request stream, and a
batch of size B always uses slots ``[0:B)``.  Ring-hash placement derives from
stable request IDs.  The batch compute window is the sum of active requests'
per-layer compute times: a serial-equivalent finite-NPU-throughput model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

from continuous_batch_control import (
    allocate_grants,
    batch_cir_diff,
    plan_cir_commit,
    plan_global_cir_commit,
)
import sim


SCHEMA_VERSION = 3
DEFAULT_OUTPUT = Path("results/continuous_batch_control/results.json")
DEFAULT_QUICK_OUTPUT = Path(
    "results/continuous_batch_control/quick_results.json"
)
NUM_NPU = 128
SSU_LIST = (8, 16, 28, 40, 56, 80, 112)
SEEDS = (42, 43)
BATCH_SIZE = 8
BATCH_SIZE_SWEEP = (1, 4, 8, 16)
EPOCHS = 256
WARMUP_EPOCHS = 32
REPLACEMENT_PROBABILITY = 0.025
GROWTH_MODES = ("static_kv", "decode_grow_1token")

POLICIES = {
    "fixed_initial": {"mode": "fixed_initial"},
    "fresh_every_epoch": {"mode": "fresh_every_epoch"},
    "per_path10_max8": {
        "mode": "per_path",
        "relative_threshold": 0.10,
        "max_epoch_interval": 8,
    },
    "global2_max8": {
        "mode": "global",
        "global_deficit_threshold": 0.02,
        "max_epoch_interval": 8,
    },
    "global5_max8": {
        "mode": "global",
        "global_deficit_threshold": 0.05,
        "max_epoch_interval": 8,
    },
    "global5_guard80_max8": {
        "mode": "global",
        "global_deficit_threshold": 0.05,
        "minimum_npu_coverage": 0.80,
        "minimum_ssu_coverage": 0.80,
        "max_epoch_interval": 8,
    },
    "global5_guard90_max8": {
        "mode": "global",
        "global_deficit_threshold": 0.05,
        "minimum_npu_coverage": 0.90,
        "minimum_ssu_coverage": 0.90,
        "max_epoch_interval": 8,
    },
    "global10_max8": {
        "mode": "global",
        "global_deficit_threshold": 0.10,
        "max_epoch_interval": 8,
    },
    "periodic8": {
        "mode": "periodic",
        "max_epoch_interval": 8,
    },
}

_CATEGORY_CODE = {
    category: index for index, category in enumerate(sim.WORKLOAD_CATEGORIES)
}
_TRACE_RECORD = struct.Struct("!IHHIQIIBBddd")
_GROWTH_RECORD = struct.Struct("!IHHQIHB")
_RNG_KEY = struct.Struct("!QII")


@dataclass
class ActiveRequest:
    request_id: int
    generation: int
    profile_key: tuple[int, int]
    category: str
    required_bw_input_gbps: float
    work_by_ssu_gb: np.ndarray
    per_layer_us: float
    per_layer_kv_gb: float
    ssd_tokens: int
    gb_per_token: float


def _rng_seed(seed, npu_id, slot_id, namespace):
    digest = hashlib.sha256()
    digest.update(namespace)
    digest.update(_RNG_KEY.pack(int(seed), int(npu_id), int(slot_id)))
    return int.from_bytes(digest.digest()[:4], "big")


def _request_id(seed, npu_id, slot_id, generation):
    return (
        (int(seed) << 32)
        | (int(npu_id) << 20)
        | (int(slot_id) << 12)
        | int(generation)
    )


def _source_fingerprint(table):
    payload = [
        [list(key), list(table[key])]
        for key in sorted(table)
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RequestStream:
    """One deterministic stream whose profile and replacement RNGs are split."""

    def __init__(self, table, keys_by_category, seed, npu_id, slot_id, num_ssu):
        self.table = table
        self.keys_by_category = keys_by_category
        self.seed = int(seed)
        self.npu_id = int(npu_id)
        self.slot_id = int(slot_id)
        self.num_ssu = int(num_ssu)
        self.profile_rng = np.random.RandomState(
            _rng_seed(seed, npu_id, slot_id, b"continuous-batch:profile:v1")
        )
        self.replacement_rng = np.random.RandomState(
            _rng_seed(seed, npu_id, slot_id, b"continuous-batch:replace:v1")
        )
        self.category_phase = (self.npu_id + self.slot_id) % len(
            sim.WORKLOAD_CATEGORIES
        )
        self.generation = 0
        self.current = self._request(self.generation)

    def _request(self, generation):
        category = sim.WORKLOAD_CATEGORIES[
            (self.category_phase + generation) % len(sim.WORKLOAD_CATEGORIES)
        ]
        keys = self.keys_by_category[category]
        key = keys[int(self.profile_rng.randint(len(keys)))]
        required_bw, per_layer_us, _, per_layer_kv_gb = self.table[key]
        request_id = _request_id(
            self.seed,
            self.npu_id,
            self.slot_id,
            generation,
        )
        work = np.zeros(self.num_ssu, dtype=np.float64)
        _, _, ssd_tokens = sim.calculate_token_partition(*key)
        gb_per_token = per_layer_kv_gb / ssd_tokens if ssd_tokens else 0.0
        if ssd_tokens > 0 and per_layer_kv_gb > 0.0:
            block_count = int(np.ceil(ssd_tokens / sim.BLOCK_SIZE))
            for block_index in range(block_count):
                block_tokens = min(
                    sim.BLOCK_SIZE,
                    ssd_tokens - block_index * sim.BLOCK_SIZE,
                )
                disk_id = sim.block_ring_hash_disk_id(
                    request_id,
                    block_index,
                    self.num_ssu,
                )
                work[disk_id] += block_tokens * gb_per_token
        return ActiveRequest(
            request_id=request_id,
            generation=generation,
            profile_key=key,
            category=category,
            required_bw_input_gbps=float(required_bw),
            work_by_ssu_gb=work,
            per_layer_us=float(per_layer_us),
            per_layer_kv_gb=float(per_layer_kv_gb),
            ssd_tokens=ssd_tokens,
            gb_per_token=float(gb_per_token),
        )

    def end_epoch(self, growth_mode):
        replaced = bool(
            self.replacement_rng.random_sample() < REPLACEMENT_PROBABILITY
        )
        if replaced:
            self.generation += 1
            self.current = self._request(self.generation)
            return replaced, 0.0, False, False, -1, -1
        if growth_mode == "static_kv":
            return replaced, 0.0, False, False, -1, -1

        request = self.current
        block_index = request.ssd_tokens // sim.BLOCK_SIZE
        disk_id = sim.block_ring_hash_disk_id(
            request.request_id,
            block_index,
            self.num_ssu,
        )
        new_block = request.ssd_tokens % sim.BLOCK_SIZE == 0
        activates_flow = request.work_by_ssu_gb[disk_id] <= 1e-12
        request.work_by_ssu_gb[disk_id] += request.gb_per_token
        request.ssd_tokens += 1
        return (
            replaced,
            request.gb_per_token,
            new_block,
            activates_flow,
            block_index,
            disk_id,
        )


def _profile_catalog(table):
    return {
        category: tuple(
            key
            for key in sorted(table)
            if sim.classify_request(*key) == category
        )
        for category in sim.WORKLOAD_CATEGORIES
    }


def _zero_matrix(rows, columns):
    return tuple((0.0,) * columns for _ in range(rows))


def _demand_matrix(batches, num_ssu):
    demand = np.zeros((len(batches), num_ssu), dtype=np.float64)
    compute_ms = np.zeros(len(batches), dtype=np.float64)
    for npu_id, batch in enumerate(batches):
        compute_ms[npu_id] = (
            sum(request.per_layer_us for request in batch) / 1000.0
        )
        for request in batch:
            demand[npu_id] += request.work_by_ssu_gb
        demand[npu_id] /= compute_ms[npu_id] / 1000.0
    return demand, compute_ms


def _new_policy_state(num_npu, num_ssu):
    return {
        name: {
            "current": _zero_matrix(num_npu, num_ssu),
            "last_commit": -1,
            "pending_since": None,
            "pending_age": 0,
            "commits": 0,
            "path_writes": 0,
            "touched_ssus": 0,
            "config_bytes": 0,
            "reasons": Counter(),
            "coverage": [],
            "fidelity": [],
            "under_target": [],
            "over_demand_gbps": [],
            "min_npu_coverage": [],
            "min_ssu_coverage": [],
            "starved_active_flow_fraction": [],
            "max_pending_age": 0,
            "max_ssu_cir_sum": 0.0,
            "max_npu_cir_sum": 0.0,
        }
        for name in POLICIES
    }


def _commit(policy, state, target, epoch):
    changes = batch_cir_diff(state["current"], target)
    state["pending_age"] = 0
    if epoch == 0:
        state["pending_since"] = None
        return True, "initial", changes
    if not changes:
        state["pending_since"] = None
        return False, "unchanged", ()

    config = POLICIES[policy]
    mode = config["mode"]
    if mode == "fixed_initial":
        return False, "fixed", ()
    if mode == "fresh_every_epoch":
        return True, "fresh_every_epoch", changes

    if state["pending_since"] is None:
        state["pending_since"] = epoch
    state["pending_age"] = epoch - state["pending_since"]
    if mode == "per_path":
        plan = plan_cir_commit(
            state["current"],
            target,
            epoch,
            state["pending_since"],
            relative_threshold=config["relative_threshold"],
            max_epoch_interval=config["max_epoch_interval"],
        )
    else:
        plan = plan_global_cir_commit(
            state["current"],
            target,
            epoch,
            state["pending_since"],
            global_deficit_threshold=(
                config["global_deficit_threshold"]
                if mode == "global"
                else float("inf")
            ),
            max_epoch_interval=config["max_epoch_interval"],
            minimum_npu_coverage=config.get("minimum_npu_coverage"),
            minimum_ssu_coverage=config.get("minimum_ssu_coverage"),
        )
    if plan.commit:
        state["pending_since"] = None
    return plan.commit, plan.reason, plan.changes


def _resource_min_coverage(current, target, axis):
    target_sum = np.sum(target, axis=axis)
    covered_sum = np.sum(np.minimum(current, target), axis=axis)
    active = target_sum > 1e-12
    if not np.any(active):
        return 1.0
    return float(np.min(covered_sum[active] / target_sum[active]))


def _record_policy(state, target_array, demand, measured):
    current = np.asarray(state["current"], dtype=np.float64)
    state["max_ssu_cir_sum"] = max(
        state["max_ssu_cir_sum"], float(np.max(np.sum(current, axis=0)))
    )
    state["max_npu_cir_sum"] = max(
        state["max_npu_cir_sum"], float(np.max(np.sum(current, axis=1)))
    )
    if not measured:
        return
    target_sum = float(np.sum(target_array))
    denominator = float(np.sum(current) + target_sum)
    state["coverage"].append(
        float(np.minimum(current, target_array).sum() / target_sum)
        if target_sum else 1.0
    )
    state["fidelity"].append(
        1.0 - float(np.abs(current - target_array).sum() / denominator)
        if denominator else 1.0
    )
    state["under_target"].append(
        float(np.maximum(target_array - current, 0.0).sum() / target_sum)
        if target_sum else 0.0
    )
    state["over_demand_gbps"].append(
        float(np.maximum(current - demand, 0.0).sum())
    )
    state["min_npu_coverage"].append(
        _resource_min_coverage(current, target_array, axis=1)
    )
    state["min_ssu_coverage"].append(
        _resource_min_coverage(current, target_array, axis=0)
    )
    active = demand > 1e-12
    state["starved_active_flow_fraction"].append(
        float(np.count_nonzero(active & (current <= 1e-12)) / np.count_nonzero(active))
        if np.any(active)
        else 0.0
    )
    state["max_pending_age"] = max(
        state["max_pending_age"], state["pending_age"]
    )


def _update_trace(trace_digest, workload_digest, epoch, npu_id, slot_id, request, replaced):
    record = _TRACE_RECORD.pack(
        epoch,
        npu_id,
        slot_id,
        request.generation,
        request.request_id,
        request.profile_key[0],
        request.profile_key[1],
        _CATEGORY_CODE[request.category],
        int(replaced),
        request.per_layer_us,
        request.per_layer_kv_gb,
        request.required_bw_input_gbps,
    )
    trace_digest.update(record)
    workload_digest.update(record[:29] + b"\0" + record[30:])


def _prefix_fingerprints(slot_digests, batch_size):
    prefixes = tuple(
        size
        for size in dict.fromkeys((*BATCH_SIZE_SWEEP, batch_size))
        if size <= batch_size
    )
    return {
        str(prefix): hashlib.sha256(
            b"continuous-batch:prefix:v1\0"
            + b"".join(
                slot_digests[npu_id][slot_id].digest()
                for npu_id in range(len(slot_digests))
                for slot_id in range(prefix)
            )
        ).hexdigest()
        for prefix in prefixes
    }


def run_case(
    table,
    *,
    seed,
    num_ssu,
    batch_size,
    growth_mode,
    num_npu,
    epochs,
    warmup,
):
    started = time.perf_counter()
    keys_by_category = _profile_catalog(table)
    streams = [
        [
            RequestStream(
                table,
                keys_by_category,
                seed,
                npu_id,
                slot_id,
                num_ssu,
            )
            for slot_id in range(batch_size)
        ]
        for npu_id in range(num_npu)
    ]
    trace_digests = [
        [hashlib.sha256(b"continuous-batch:trace-slot:v1\0") for _ in row]
        for row in streams
    ]
    workload_digests = [
        [hashlib.sha256(b"continuous-batch:workload-slot:v1\0") for _ in row]
        for row in streams
    ]
    growth_digest = hashlib.sha256(growth_mode.encode())
    states = _new_policy_state(num_npu, num_ssu)
    previous_target = None
    target_churn = []
    target_grant_fraction = []
    active_flows = []
    compute_windows_ms = []
    category_counts = Counter()
    replacements = 0
    measured_replacements = 0
    appended_tokens = 0
    new_block_events = 0
    partial_block_extensions = 0
    aggregate_flow_activations = 0
    growth_gb = 0.0

    for epoch in range(epochs):
        batches = []
        for npu_id, npu_streams in enumerate(streams):
            batch = []
            for slot_id, stream in enumerate(npu_streams):
                request = stream.current
                batch.append(request)
                category_counts[request.category] += 1
            batches.append(batch)

        demand, compute_ms = _demand_matrix(batches, num_ssu)
        target = allocate_grants(demand, sim.DISK_BW, sim.NPU_BW_LIMIT)
        target_array = np.asarray(target, dtype=np.float64)
        assert np.all(target_array >= -1e-10)
        assert np.all(target_array <= demand + 1e-8)
        assert np.max(np.sum(target_array, axis=0)) <= sim.DISK_BW + 1e-8
        assert np.max(np.sum(target_array, axis=1)) <= sim.NPU_BW_LIMIT + 1e-8
        if previous_target is not None and epoch >= warmup:
            denominator = float(np.sum(previous_target) + np.sum(target_array))
            target_churn.append(
                float(np.abs(target_array - previous_target).sum() / denominator)
                if denominator else 0.0
            )
        previous_target = target_array
        if epoch >= warmup:
            demand_sum = float(np.sum(demand))
            target_grant_fraction.append(
                float(np.sum(target_array) / demand_sum) if demand_sum else 1.0
            )
            active_flows.append(int(np.count_nonzero(demand > 1e-12)))
            compute_windows_ms.extend(compute_ms.tolist())

        for policy, state in states.items():
            commit, reason, changes = _commit(policy, state, target, epoch)
            if commit:
                state["current"] = target
                state["last_commit"] = epoch
                if epoch >= warmup:
                    state["commits"] += 1
                    state["path_writes"] += len(changes)
                    state["touched_ssus"] += len(
                        {change.ssu for change in changes}
                    )
                    state["config_bytes"] += 16 + 16 * len(changes)
                    state["reasons"][reason] += 1
            _record_policy(state, target_array, demand, epoch >= warmup)

        for npu_id, npu_streams in enumerate(streams):
            for slot_id, stream in enumerate(npu_streams):
                request = stream.current
                replaced = False
                if epoch + 1 < epochs:
                    (
                        replaced,
                        added_gb,
                        new_block,
                        activates_flow,
                        block_index,
                        disk_id,
                    ) = stream.end_epoch(growth_mode)
                    replacements += int(replaced)
                    measured_transition = epoch + 1 >= warmup
                    if measured_transition:
                        measured_replacements += int(replaced)
                        appended_tokens += int(added_gb > 0.0)
                        new_block_events += int(new_block)
                        partial_block_extensions += int(
                            added_gb > 0.0 and not new_block
                        )
                        aggregate_flow_activations += int(activates_flow)
                        growth_gb += added_gb
                    if added_gb > 0.0:
                        growth_digest.update(
                            _GROWTH_RECORD.pack(
                                epoch + 1,
                                npu_id,
                                slot_id,
                                request.request_id,
                                block_index,
                                disk_id,
                                int(new_block),
                            )
                        )
                _update_trace(
                    trace_digests[npu_id][slot_id],
                    workload_digests[npu_id][slot_id],
                    epoch,
                    npu_id,
                    slot_id,
                    request,
                    replaced,
                )

    measured_epochs = epochs - warmup
    policy_summaries = {}
    for policy, state in states.items():
        policy_summaries[policy] = {
            "mean_target_cir_coverage": float(np.mean(state["coverage"])),
            "min_target_cir_coverage": float(np.min(state["coverage"])),
            "mean_cir_fidelity": float(np.mean(state["fidelity"])),
            "mean_under_target_fraction": float(np.mean(state["under_target"])),
            "mean_over_demand_gbps": float(np.mean(state["over_demand_gbps"])),
            "mean_min_npu_coverage": float(np.mean(state["min_npu_coverage"])),
            "worst_npu_coverage": float(np.min(state["min_npu_coverage"])),
            "mean_min_ssu_coverage": float(np.mean(state["min_ssu_coverage"])),
            "worst_ssu_coverage": float(np.min(state["min_ssu_coverage"])),
            "mean_starved_active_flow_fraction": float(
                np.mean(state["starved_active_flow_fraction"])
            ),
            "max_pending_age": state["max_pending_age"],
            "mean_touched_ssus_per_commit": (
                state["touched_ssus"] / state["commits"]
                if state["commits"]
                else 0.0
            ),
            "commit_epochs": state["commits"],
            "commit_epoch_fraction": state["commits"] / measured_epochs,
            "path_writes": state["path_writes"],
            "path_writes_per_epoch": state["path_writes"] / measured_epochs,
            "config_bytes_per_epoch": state["config_bytes"] / measured_epochs,
            "commit_reasons": dict(state["reasons"]),
            "max_ssu_cir_sum_gbps": state["max_ssu_cir_sum"],
            "max_npu_cir_sum_gbps": state["max_npu_cir_sum"],
        }

    trace_prefixes = _prefix_fingerprints(trace_digests, batch_size)
    workload_prefixes = _prefix_fingerprints(workload_digests, batch_size)
    replacement_trials = (epochs - 1) * num_npu * batch_size
    measured_transitions = epochs - warmup
    measured_replacement_trials = measured_transitions * num_npu * batch_size
    observed_profiles = sum(category_counts.values())
    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "num_npu": num_npu,
        "batch_size": batch_size,
        "growth_mode": growth_mode,
        "epochs": epochs,
        "warmup_epochs": warmup,
        "measured_controller_epochs": measured_epochs,
        "workload_steady_from_epoch": 0,
        "warmup_scope": "controller_metrics_only",
        "replacement_probability": REPLACEMENT_PROBABILITY,
        "replacement_timing": "between observed epochs",
        "replacement_trials": replacement_trials,
        "request_replacements": replacements,
        "replacement_rate": replacements / replacement_trials,
        "request_replacements_per_transition": replacements / (epochs - 1),
        "measured_request_replacements": measured_replacements,
        "measured_replacement_rate": (
            measured_replacements / measured_replacement_trials
        ),
        "measured_request_replacements_per_epoch": (
            measured_replacements / measured_transitions
        ),
        "batch_compute_model": "sum_active_per_layer_us_serial_equivalent",
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "request_id_scheme": "seed32|npu12|slot8|generation12",
        "profile_rng": "sha256-derived independent per-(seed,npu,slot)",
        "replacement_rng": "separate sha256-derived per-(seed,npu,slot)",
        "category_schedule": "WORKLOAD_CATEGORIES[(npu+slot+generation)%4]",
        "profile_category_counts": dict(category_counts),
        "profile_category_fractions": {
            category: category_counts[category] / observed_profiles
            for category in sim.WORKLOAD_CATEGORIES
        },
        "source_fingerprint": _source_fingerprint(table),
        "workload_fingerprint": workload_prefixes[str(batch_size)],
        "trace_fingerprint": trace_prefixes[str(batch_size)],
        "growth_placement_fingerprint": growth_digest.hexdigest(),
        "workload_prefix_fingerprints": workload_prefixes,
        "trace_prefix_fingerprints": trace_prefixes,
        "trace_fingerprint_scope": "epoch replacement event, profile, request_id",
        "mean_target_churn": float(np.mean(target_churn)),
        "max_target_churn": float(np.max(target_churn)),
        "mean_target_grant_fraction": float(np.mean(target_grant_fraction)),
        "mean_active_npu_ssu_flows": float(np.mean(active_flows)),
        "mean_batch_compute_window_ms": float(np.mean(compute_windows_ms)),
        "appended_tokens": appended_tokens,
        "new_block_events": new_block_events,
        "partial_block_extensions": partial_block_extensions,
        "aggregate_flow_activations": aggregate_flow_activations,
        "growth_gb_per_measured_epoch": growth_gb / measured_transitions,
        "policies": policy_summaries,
        "wall_time_s": time.perf_counter() - started,
    }


def _tasks(num_npu, epochs, warmup):
    main = [
        {
            "seed": seed,
            "num_ssu": num_ssu,
            "batch_size": BATCH_SIZE,
            "growth_mode": growth_mode,
            "num_npu": num_npu,
            "epochs": epochs,
            "warmup": warmup,
        }
        for seed in SEEDS
        for num_ssu in SSU_LIST
        for growth_mode in GROWTH_MODES
    ]
    sensitivity = [
        {
            "seed": seed,
            "num_ssu": 40,
            "batch_size": batch_size,
            "growth_mode": growth_mode,
            "num_npu": num_npu,
            "epochs": epochs,
            "warmup": warmup,
        }
        for seed in SEEDS
        for batch_size in BATCH_SIZE_SWEEP
        if batch_size != BATCH_SIZE
        for growth_mode in GROWTH_MODES
    ]
    return main + sensitivity


def _fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("sim.py", "continuous_batch_control.py", Path(__file__).name):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


_WORKER_TABLE = None


def _init_worker(table):
    global _WORKER_TABLE
    _WORKER_TABLE = table


def _worker(task):
    return run_case(_WORKER_TABLE, **task)


def _validate_trace_pairing(rows):
    by_key = {
        (
            row["seed"],
            row["num_ssu"],
            row["batch_size"],
            row["growth_mode"],
        ): row
        for row in rows
    }
    for seed in SEEDS:
        main = [
            by_key[(seed, ssu, BATCH_SIZE, growth_mode)]
            for ssu in SSU_LIST
            for growth_mode in GROWTH_MODES
            if (seed, ssu, BATCH_SIZE, growth_mode) in by_key
        ]
        if main:
            assert len({row["trace_fingerprint"] for row in main}) == 1
            assert len({row["workload_fingerprint"] for row in main}) == 1
        for growth_mode in GROWTH_MODES:
            at_40 = {
                batch_size: by_key[(seed, 40, batch_size, growth_mode)]
                for batch_size in BATCH_SIZE_SWEEP
                if (seed, 40, batch_size, growth_mode) in by_key
            }
            for prefix, reference in at_40.items():
                for larger, row in at_40.items():
                    if larger >= prefix:
                        assert row["trace_prefix_fingerprints"][str(prefix)] == (
                            reference["trace_fingerprint"]
                        )
                        assert row["workload_prefix_fingerprints"][str(prefix)] == (
                            reference["workload_fingerprint"]
                        )


def run_matrix(output, *, workers, rerun, num_npu, epochs, warmup):
    table = sim.load_bw_table_cache(num_npu=num_npu)
    tasks = _tasks(num_npu, epochs, warmup)
    experiment = {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _fingerprint(),
        "source_fingerprint": _source_fingerprint(table),
        "control_plane_only": True,
        "end_to_end_npu_utilization": False,
        "num_npu": num_npu,
        "ssu_list": list(SSU_LIST),
        "seeds": list(SEEDS),
        "batch_size": BATCH_SIZE,
        "batch_size_sweep": list(BATCH_SIZE_SWEEP),
        "growth_modes": list(GROWTH_MODES),
        "epochs": epochs,
        "warmup_epochs": warmup,
        "workload_steady_from_epoch": 0,
        "warmup_scope": "controller_metrics_only",
        "replacement_probability": REPLACEMENT_PROBABILITY,
        "ssd_capacity_gbps": sim.DISK_BW,
        "npu_capacity_gbps": sim.NPU_BW_LIMIT,
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "batch_compute_model": "sum_active_per_layer_us_serial_equivalent",
        "request_stream_pairing": "independent (npu,slot); batch B uses slots[0:B]",
        "profile_rng": "independent from replacement RNG for every (seed,npu,slot)",
        "replacement_rng": "independent from profile RNG for every (seed,npu,slot)",
        "category_balance": "category cycles on each slot replacement",
        "replacement_timing": "between observed epochs",
        "request_id_scheme": "seed32|npu12|slot8|generation12",
        "policies": POLICIES,
    }
    rows = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                (
                    row["seed"],
                    row["num_ssu"],
                    row["batch_size"],
                    row["growth_mode"],
                ): row
                for row in cached["results"]
            }
    pending = [
        task
        for task in tasks
        if (
            task["seed"],
            task["num_ssu"],
            task["batch_size"],
            task["growth_mode"],
        ) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[
                (
                    task["seed"],
                    task["num_ssu"],
                    task["batch_size"],
                    task["growth_mode"],
                )
            ]
            for task in tasks
            if (
                task["seed"],
                task["num_ssu"],
                task["batch_size"],
                task["growth_mode"],
            ) in rows
        ]
        _validate_trace_pairing(ordered)
        _write(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered) == len(tasks),
                "experiment": experiment,
                "results": ordered,
            },
        )

    if pending and workers == 1:
        for task in pending:
            row = run_case(table, **task)
            rows[
                (
                    row["seed"],
                    row["num_ssu"],
                    row["batch_size"],
                    row["growth_mode"],
                )
            ] = row
            checkpoint()
            _print(row)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            initializer=_init_worker,
            initargs=(table,),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in pending}
            for future in as_completed(futures):
                row = future.result()
                rows[
                    (
                        row["seed"],
                        row["num_ssu"],
                        row["batch_size"],
                        row["growth_mode"],
                    )
                ] = row
                checkpoint()
                _print(row)
    checkpoint()
    return json.loads(output.read_text())


def _print(row):
    recommended = row["policies"]["global5_max8"]
    print(
        "seed=%d SSU=%3d batch=%2d growth=%-20s coverage=%6.2f%% commits=%5.1f%% writes=%8.1f/epoch wall=%6.1fs"
        % (
            row["seed"],
            row["num_ssu"],
            row["batch_size"],
            row["growth_mode"],
            100.0 * recommended["mean_target_cir_coverage"],
            100.0 * recommended["commit_epoch_fraction"],
            recommended["path_writes_per_epoch"],
            row["wall_time_s"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_QUICK_OUTPUT if args.quick else DEFAULT_OUTPUT
    )
    if args.quick and output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("--quick cannot overwrite the formal output")
    num_npu = 16 if args.quick else NUM_NPU
    epochs = 24 if args.quick else EPOCHS
    warmup = 4 if args.quick else WARMUP_EPOCHS
    result = run_matrix(
        output,
        workers=args.workers,
        rerun=args.rerun,
        num_npu=num_npu,
        epochs=epochs,
        warmup=warmup,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))
    print("output:", output)


if __name__ == "__main__":
    main()
