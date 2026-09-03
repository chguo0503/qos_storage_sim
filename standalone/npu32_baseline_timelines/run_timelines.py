#!/usr/bin/env python3
"""Run the three fixed 32-NPU Baseline micro-timeline experiments.

The orchestration and plots are intentionally local to this file.  The actual
data plane is not reimplemented: the runner imports the authoritative frozen
``sim.py`` and ``continuous_batch_sim.py`` beside it and observes their
physical events with reversible wrappers.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import continuous_batch_sim as batch_sim
import continuous_batch_control
import policy_logic
import sim


for _module in (sim, batch_sim, policy_logic, continuous_batch_control):
    _module_path = Path(_module.__file__).resolve()
    if _module_path.parent != HERE:
        raise RuntimeError(
            f"non-local simulator module loaded: {_module.__name__}={_module_path}"
        )


NUM_NPU = 32
SEED = 42
DISK_GBPS = 40.0
NPU_LINK_GBPS = 50.0
TIME_TOLERANCE_MS = 1e-8
HOMOGENEOUS_PROFILE = (48, 512)
F_PROFILE = (200, 256)
V_PROFILE = (32, 2048)
VICTIM_NPUS = frozenset((24, 26))
DEFAULT_OUTPUT_ROOT = HERE / "results"

IO_COLOR = "#2C7FB8"
OVERRUN_COLOR = "#D62728"
COMPUTE_COLOR = "#72B7B2"
DEADLINE_COLOR = "#4A4A4A"


@dataclass(frozen=True)
class CaseSpec:
    cli_name: str
    output_name: str
    num_ssu: int
    n_layers: int
    mixed: bool
    expected_summary_sha256: str
    expected_physical_blocks: int
    expected_makespan_ms: float
    expected_fleet_npu_utilization: float
    expected_active_npu_utilization: float
    expected_ssd_utilization: float
    expected_incremental_stall_npu_ms: tuple[float, ...]

    @property
    def profiles(self) -> tuple[tuple[int, int], ...]:
        if not self.mixed:
            return (HOMOGENEOUS_PROFILE,) * NUM_NPU
        return tuple(
            V_PROFILE if npu_id in VICTIM_NPUS else F_PROFILE
            for npu_id in range(NUM_NPU)
        )

    @property
    def layer_tag(self) -> str:
        return "".join(str(layer) for layer in range(self.n_layers))


CASES = {
    case.cli_name: case
    for case in (
        CaseSpec(
            cli_name="homogeneous-layer0",
            output_name="baseline_layer0_32npu_single_request_timeline",
            num_ssu=11,
            n_layers=1,
            mixed=False,
            expected_summary_sha256=(
                "4a9e83696b87c9f42eeb549a0b6bd3d600bf7292a7fa814b05e9ad2bcadda74e"
            ),
            expected_physical_blocks=12_160,
            expected_makespan_ms=10.12582897470067,
            expected_fleet_npu_utilization=0.49816266827765204,
            expected_active_npu_utilization=0.5079391494098344,
            expected_ssd_utilization=0.458102925359464,
            expected_incremental_stall_npu_ms=(156.3719567626953,),
        ),
        CaseSpec(
            cli_name="homogeneous-layer012",
            output_name="baseline_layer012_32npu_single_request_timeline",
            num_ssu=11,
            n_layers=3,
            mixed=False,
            expected_summary_sha256=(
                "aa67cbd91aa75ae59f9f358e8ac315292dfa7622724e5c75bcb28711d25f127c"
            ),
            expected_physical_blocks=36_480,
            expected_makespan_ms=20.28055309579442,
            expected_fleet_npu_utilization=0.7461793507405992,
            expected_active_npu_utilization=0.7558294701607702,
            expected_ssd_utilization=0.6861753503106267,
            expected_incremental_stall_npu_ms=(
                156.3719567626953,
                0.03305207998682924,
                0.03305207998682924,
            ),
        ),
        CaseSpec(
            cli_name="mixed-layer012",
            output_name=(
                "candidate_R_baseline_cold_F_burst_layer012_mechanism_probe"
            ),
            num_ssu=4,
            n_layers=3,
            mixed=True,
            expected_summary_sha256=(
                "eeeaddd58f9b491aac99112ae981d6916755514c0486c1003243a6646104b0d7"
            ),
            expected_physical_blocks=145_260,
            expected_makespan_ms=167.05225709287382,
            expected_fleet_npu_utilization=0.16839164948497842,
            expected_active_npu_utilization=0.1999496483411022,
            expected_ssd_utilization=0.912192429920718,
            expected_incremental_stall_npu_ms=(
                1586.012471069336,
                877.8604017100711,
                1137.9267901831015,
            ),
        ),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _savefig(fig, path: Path, description: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        metadata={"Title": path.stem, "Description": description},
    )
    plt.close(fig)
    temporary.replace(path)


def _canonical_table(table) -> list[list[object]]:
    return [
        [int(key[0]), int(key[1]), [float(value) for value in table[key]]]
        for key in sorted(table)
    ]


def _load_authenticated_table() -> tuple[dict, dict]:
    data_path = ROOT / "data"
    raw = ast.literal_eval(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("data must contain a non-empty profile dictionary")
    table = {}
    for raw_key, raw_value in raw.items():
        key = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(f"invalid profile key: {raw_key!r}")
        profile = (int(key[0]), int(key[1]))
        values = tuple(raw_value)
        if len(values) == 3:
            required_bw, per_layer_us, ttft_ms = values
            values = (
                required_bw,
                per_layer_us,
                ttft_ms,
                required_bw * per_layer_us / 1e6,
            )
        converted = tuple(float(value) for value in values)
        if (
            profile in table
            or len(converted) != 4
            or any(not math.isfinite(value) or value <= 0.0 for value in converted)
        ):
            raise ValueError(f"invalid or duplicate profile: {profile!r}")
        table[profile] = converted

    cache_path = ROOT / "results" / f"bw_table_cache_v2_{NUM_NPU}npu.npz"
    if cache_path.exists():
        cached = sim.load_bw_table_cache(
            results_dir=str(ROOT / "results"), num_npu=NUM_NPU
        )
        if _canonical_table(cached) != _canonical_table(table):
            raise RuntimeError(f"cache differs from authenticated data: {cache_path}")

    canonical = _canonical_table(table)
    table_bytes = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    catalog_canonical = [
        [list(key), [float(value) for value in table[key]]]
        for key in sorted(table)
    ]
    catalog_bytes = json.dumps(
        catalog_canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return table, {
        "source": "data",
        "source_sha256": _sha256(data_path),
        "catalog_hash": hashlib.sha256(
            b"random-steady-state:data-catalog:v1\0" + catalog_bytes
        ).hexdigest(),
        "table_fingerprint": hashlib.sha256(
            b"authenticated-bw-table:v1\0" + table_bytes
        ).hexdigest(),
        "profile_count": len(table),
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_present": cache_path.exists(),
        "cache_verified_equal": cache_path.exists(),
    }


def _sticky_placement(
    request_id: int,
    seq_len_k: int,
    nql: int,
    per_layer_kv_gb: float,
    num_ssu: int,
) -> tuple[tuple[int, float], ...]:
    _, _, ssd_tokens = sim.calculate_token_partition(seq_len_k, nql)
    if ssd_tokens <= 0 or per_layer_kv_gb <= 0.0:
        return ()
    gb_per_token = per_layer_kv_gb / ssd_tokens
    block_count = int(np.ceil(ssd_tokens / sim.BLOCK_SIZE))
    return tuple(
        (
            sim.block_ring_hash_disk_id(request_id, block_idx, num_ssu),
            float(
                min(sim.BLOCK_SIZE, ssd_tokens - block_idx * sim.BLOCK_SIZE)
                * gb_per_token
            ),
        )
        for block_idx in range(block_count)
    )


def _static_qos_config() -> sim.StaticQoSConfig:
    path_cirs = []
    for _ in range(8):
        for budget, count in zip((20.0, 6.0, 8.0, 6.0), (12, 4, 12, 4)):
            path_cirs.extend([budget / (8 * count)] * count)
    return sim.StaticQoSConfig(
        path_cirs=tuple(path_cirs),
        path_pirs=(float("inf"),) * 256,
        path_weights=(1.0,) * 256,
        group_weights=(1.0,) * 8,
        category_paths_per_group=(12, 4, 12, 4),
    )


def _baseline_client_config() -> sim.ClientIOConfig:
    return sim.ClientIOConfig(
        name="continuous_prefill_baseline",
        pressure_window_io=None,
        submit_batch_size=1,
        issue_interval_us=0.1,
        path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
    )


def _build_requests(case: CaseSpec, table: dict):
    requests = []
    for npu_id, profile in enumerate(case.profiles):
        required_gbps, per_layer_us, ttft_ms, per_layer_kv_gb = table[profile]
        placement = _sticky_placement(
            npu_id, profile[0], profile[1], per_layer_kv_gb, case.num_ssu
        )
        load = {
            "request_id": npu_id,
            "npu_id": npu_id,
            "stream_id": 0 if not case.mixed else npu_id,
            "generation": 0,
            "profile_key": profile,
            "seq_len_k": profile[0],
            "nql": profile[1],
            "category": sim.classify_request(*profile),
            "required_bw_input_gbps": required_gbps,
            "per_layer_us": per_layer_us,
            "per_layer_kv_gb": per_layer_kv_gb,
            "arrival_ms": 0.0,
            "arrival_time": 0.0,
            "initial": True,
        }
        if case.mixed:
            load.update(
                {
                    "profile_id": f"seq{profile[0]}_nql{profile[1]}",
                    "profile_name": (
                        f"seq_len_k={profile[0]},nql={profile[1]}"
                    ),
                    "raw_demand_gbps": required_gbps,
                    "source_ttft_ms": ttft_ms,
                }
            )
        requests.append(
            batch_sim.ContinuousBatchRequest(
                request_id=npu_id,
                npu_id=npu_id,
                arrival_time_ms=0.0,
                load=load,
                placement=(placement,),
            )
        )
    return tuple(requests)


def _run_simulation(case: CaseSpec, requests):
    return batch_sim.simulate_continuous_batch(
        requests,
        num_npu=NUM_NPU,
        num_ssu=case.num_ssu,
        n_layers=case.n_layers,
        batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR,
        qos_config=_static_qos_config(),
        cross_request_layer0_prefetch=False,
        client_io_config=_baseline_client_config(),
        disk_bw_gbps=DISK_GBPS,
        npu_bw_gbps=NPU_LINK_GBPS,
        submit_order_seed=SEED,
    )


class PhysicalTrace:
    """Read-only wrappers for every physical block transition."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.by_key: dict[tuple[int, int, int, int], dict] = {}
        self.enqueue_count = defaultdict(int)
        self.service_count = defaultdict(int)
        self._original_enqueue_many = sim.DiskIOScheduler.enqueue_many
        self._original_activate_flow = sim.DiskIOScheduler._activate_flow
        self._original_complete_ready = sim.DiskIOScheduler.complete_ready_flows
        self._original_link_completion = batch_sim._handle_link_completion

    @staticmethod
    def _key(flow: sim.BlockIOFlow) -> tuple[int, int, int, int]:
        return (
            int(flow.request_id),
            int(flow.layer),
            int(flow.block_idx),
            int(flow.disk_id),
        )

    def __enter__(self):
        collector = self

        def enqueue_many(scheduler, flows, current_time_ms):
            flows = tuple(flows)
            for flow in flows:
                key = collector._key(flow)
                if key in collector.by_key:
                    raise AssertionError(f"duplicate physical block enqueue: {key}")
                ssu_id = int(flow.disk_id)
                row = {
                    "global_enqueue_order": len(collector.rows),
                    "ssu_id": ssu_id,
                    "ssu_enqueue_order": collector.enqueue_count[ssu_id],
                    "ssd_service_order": None,
                    "npu_id": int(flow.npu_id),
                    "request_id": int(flow.request_id),
                    "layer": int(flow.layer),
                    "block_idx": int(flow.block_idx),
                    "size_gb": float(flow.total_gb),
                    "path_id": int(flow.queue_id),
                    "enqueue_time_ms": float(current_time_ms),
                    "ssd_start_time_ms": None,
                    "ssd_end_time_ms": None,
                    "ssd_queue_wait_ms": None,
                    "link_enqueue_time_ms": None,
                    "link_start_time_ms": None,
                    "link_end_time_ms": None,
                }
                collector.enqueue_count[ssu_id] += 1
                collector.rows.append(row)
                collector.by_key[key] = row
            return collector._original_enqueue_many(
                scheduler, flows, current_time_ms
            )

        def activate_flow(scheduler, flow, current_time_ms):
            result = collector._original_activate_flow(
                scheduler, flow, current_time_ms
            )
            row = collector.by_key[collector._key(flow)]
            ssu_id = int(flow.disk_id)
            row["ssd_service_order"] = collector.service_count[ssu_id]
            collector.service_count[ssu_id] += 1
            row["ssd_start_time_ms"] = float(current_time_ms)
            row["ssd_queue_wait_ms"] = float(flow.ssd_queue_wait_ms)
            return result

        def complete_ready_flows(scheduler, current_time_ms):
            completed = collector._original_complete_ready(
                scheduler, current_time_ms
            )
            for flow in completed:
                row = collector.by_key[collector._key(flow)]
                row["ssd_end_time_ms"] = float(current_time_ms)
                row["link_enqueue_time_ms"] = float(current_time_ms)
            return completed

        def handle_link_completion(context, npu_id, generation, current_time_ms):
            npu = context.npus[npu_id]
            flow = npu.link_active_flow
            valid = bool(
                flow is not None
                and generation == npu.link_generation
                and current_time_ms + 1e-12 >= flow.link_end_time
            )
            result = collector._original_link_completion(
                context, npu_id, generation, current_time_ms
            )
            if valid:
                row = collector.by_key[collector._key(flow)]
                row["link_start_time_ms"] = float(flow.link_start_time)
                row["link_end_time_ms"] = float(current_time_ms)
            return result

        sim.DiskIOScheduler.enqueue_many = enqueue_many
        sim.DiskIOScheduler._activate_flow = activate_flow
        sim.DiskIOScheduler.complete_ready_flows = complete_ready_flows
        batch_sim._handle_link_completion = handle_link_completion
        return self

    def __exit__(self, exc_type, exc, traceback):
        sim.DiskIOScheduler.enqueue_many = self._original_enqueue_many
        sim.DiskIOScheduler._activate_flow = self._original_activate_flow
        sim.DiskIOScheduler.complete_ready_flows = self._original_complete_ready
        batch_sim._handle_link_completion = self._original_link_completion
        return False


def _extract_layer_rows(case: CaseSpec, summary: dict) -> list[dict]:
    rows = []
    for batch in summary["microbatch_metrics"]:
        if batch["batch_size"] != 1 or len(batch["member_request_ids"]) != 1:
            raise AssertionError("timeline requires one request per microbatch")
        request_id = int(batch["member_request_ids"][0])
        npu_id = int(batch["npu_id"])
        profile = case.profiles[npu_id]
        metrics = batch["layer_metrics"]
        if len(metrics) != case.n_layers:
            raise AssertionError("unexpected layer count")
        for layer, metric in enumerate(metrics):
            release = float(metric["io_start_time_ms"])
            ready = float(metric["io_ready_time_ms"])
            compute_ms = float(metric["compute_duration_ms"])
            deadline = (
                release + compute_ms
                if layer == 0
                else float(metrics[layer - 1]["compute_end_ms"])
            )
            overrun = max(0.0, ready - deadline)
            row = {
                "npu_id": npu_id,
                "request_id": request_id,
            }
            if case.mixed:
                row.update(
                    {
                        "profile": "V" if npu_id in VICTIM_NPUS else "F",
                        "profile_seq_len_k": profile[0],
                        "profile_nql": profile[1],
                    }
                )
            row.update(
                {
                    "layer": layer,
                    "io_release_time_ms": release,
                    "io_ready_time_ms": ready,
                    "io_duration_ms": ready - release,
                    "comparison_deadline_ms": deadline,
                    "read_exceeds_compute_window": overrun > TIME_TOLERANCE_MS,
                    "overrun_ms": overrun,
                    "previous_compute_end_ms": (
                        0.0
                        if layer == 0
                        else float(metrics[layer - 1]["compute_end_ms"])
                    ),
                    "compute_start_time_ms": float(metric["compute_start_ms"]),
                    "compute_end_time_ms": float(metric["compute_end_ms"]),
                    "compute_duration_ms": compute_ms,
                    "actual_io_barrier_wait_ms": float(
                        metric["io_barrier_wait_ms"]
                    ),
                }
            )
            rows.append(row)
    return sorted(rows, key=lambda row: (row["npu_id"], row["layer"]))


def _layer_metrics(case: CaseSpec, layer_rows: list[dict]) -> dict[str, dict]:
    result = {}
    cumulative_stall = 0.0
    cumulative_compute = 0.0
    warm_stall = 0.0
    for layer in range(case.n_layers):
        rows = [row for row in layer_rows if row["layer"] == layer]
        stall = math.fsum(row["actual_io_barrier_wait_ms"] for row in rows)
        compute = math.fsum(row["compute_duration_ms"] for row in rows)
        cumulative_stall += stall
        cumulative_compute += compute
        if layer >= 1:
            warm_stall += stall
        overruns = np.asarray([row["overrun_ms"] for row in rows])
        metric = {
            "incremental_io_barrier_npu_ms": stall,
            "incremental_compute_npu_ms": compute,
            "cumulative_io_barrier_npu_ms": cumulative_stall,
            "cumulative_compute_npu_ms": cumulative_compute,
            "cumulative_npu_utilization": cumulative_compute
            / (cumulative_compute + cumulative_stall),
            "over_budget_count": int(np.sum(overruns > TIME_TOLERANCE_MS)),
            "over_budget_npu_ids": [
                row["npu_id"]
                for row in rows
                if row["read_exceeds_compute_window"]
            ],
            "max_overrun_ms": float(overruns.max()),
        }
        if layer >= 1:
            metric.update(
                {
                    "warm_io_barrier_npu_ms": warm_stall,
                    "warm_compute_npu_ms": cumulative_compute,
                    "warm_cumulative_npu_utilization": cumulative_compute
                    / (cumulative_compute + warm_stall),
                }
            )
        result[f"layer{layer}"] = metric
    return result


def _enqueue_timing(
    case: CaseSpec, block_rows: list[dict], layer_rows: list[dict]
) -> dict[str, dict]:
    result = {}
    for layer in range(case.n_layers):
        blocks = [row for row in block_rows if row["layer"] == layer]
        layers = [row for row in layer_rows if row["layer"] == layer]
        release = np.asarray([row["io_release_time_ms"] for row in layers])
        enqueue = np.asarray([row["enqueue_time_ms"] for row in blocks])
        rounded = np.round(enqueue, decimals=12)
        _, occupancy = np.unique(rounded, return_counts=True)
        histogram = Counter(int(value) for value in occupancy)
        per_ssu = []
        for ssu_id in range(case.num_ssu):
            values = np.sort(
                np.round(
                    np.asarray(
                        [
                            row["enqueue_time_ms"]
                            for row in blocks
                            if row["ssu_id"] == ssu_id
                        ]
                    ),
                    decimals=12,
                )
            )
            unique, local_occupancy = np.unique(values, return_counts=True)
            gaps = np.diff(unique) * 1000.0
            per_ssu.append(
                {
                    "ssu_id": ssu_id,
                    "physical_block_count": int(values.size),
                    "first_enqueue_time_ms": float(values.min()),
                    "last_enqueue_time_ms": float(values.max()),
                    "enqueue_span_us": 1000.0 * float(values.max() - values.min()),
                    "unique_enqueue_timestamp_count": int(unique.size),
                    "max_blocks_same_timestamp": int(local_occupancy.max()),
                    "positive_interarrival_gap_us": {
                        "p50": float(np.percentile(gaps, 50)),
                        "p95": float(np.percentile(gaps, 95)),
                        "max": float(gaps.max()),
                    },
                }
            )
        result[f"layer{layer}"] = {
            "logical_release": {
                "first_time_ms": float(release.min()),
                "last_time_ms": float(release.max()),
                "span_us": 1000.0 * float(release.max() - release.min()),
                "unique_timestamp_count": int(
                    np.unique(np.round(release, decimals=12)).size
                ),
            },
            "physical_enqueue": {
                "first_time_ms": float(enqueue.min()),
                "last_time_ms": float(enqueue.max()),
                "span_us": 1000.0 * float(enqueue.max() - enqueue.min()),
                "physical_block_count": int(enqueue.size),
                "unique_timestamp_count": int(occupancy.size),
                "max_blocks_same_timestamp": int(occupancy.max()),
                "max_blocks_same_timestamp_one_ssu": max(
                    item["max_blocks_same_timestamp"] for item in per_ssu
                ),
                "timestamp_occupancy_histogram": {
                    str(value): int(count)
                    for value, count in sorted(histogram.items())
                },
                "per_ssu": per_ssu,
            },
        }
    return result


def _close(left, right, tolerance=1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _validate(
    case: CaseSpec,
    summary: dict,
    summary_hash: str,
    observer_match: bool,
    requests,
    layer_rows: list[dict],
    block_rows: list[dict],
    metrics: dict[str, dict],
    *,
    verify_reference: bool,
) -> dict[str, bool]:
    expected = {}
    for request in requests:
        for layer in range(case.n_layers):
            placement = request.placement[0]
            for block_idx, (ssu_id, size_gb) in enumerate(placement):
                expected[(request.request_id, layer, block_idx)] = (
                    request.npu_id,
                    ssu_id,
                    size_gb,
                )
    observed = {
        (row["request_id"], row["layer"], row["block_idx"]): (
            row["npu_id"],
            row["ssu_id"],
            row["size_gb"],
        )
        for row in block_rows
    }
    by_npu_layer = {
        (row["npu_id"], row["layer"]): row for row in layer_rows
    }
    ready_by_request_layer = defaultdict(list)
    for row in block_rows:
        ready_by_request_layer[(row["request_id"], row["layer"])].append(
            row["link_end_time_ms"]
        )

    def nonoverlap(rows, start_field: str, end_field: str) -> bool:
        ordered = sorted(rows, key=lambda row: (row[start_field], row[end_field]))
        return all(
            right[start_field] + TIME_TOLERANCE_MS >= left[end_field]
            for left, right in zip(ordered, ordered[1:])
        )

    required = (
        "ssd_service_order",
        "ssd_start_time_ms",
        "ssd_end_time_ms",
        "ssd_queue_wait_ms",
        "link_enqueue_time_ms",
        "link_start_time_ms",
        "link_end_time_ms",
    )
    reference_metrics_match = all(
        (
            _close(summary["makespan_ms"], case.expected_makespan_ms, 1e-9),
            _close(
                summary["fleet_npu_compute_utilization"],
                case.expected_fleet_npu_utilization,
                1e-12,
            ),
            _close(
                summary["active_window_npu_compute_utilization"],
                case.expected_active_npu_utilization,
                1e-12,
            ),
            _close(
                summary["ssd_mean_utilization"],
                case.expected_ssd_utilization,
                1e-12,
            ),
            *(
                _close(
                    metrics[f"layer{layer}"][
                        "incremental_io_barrier_npu_ms"
                    ],
                    expected_stall,
                    1e-8,
                )
                for layer, expected_stall in enumerate(
                    case.expected_incremental_stall_npu_ms
                )
            ),
        )
    )
    checks = {
        "simulator_invariants": all(summary["invariants"].values()),
        "observer_noninterference": observer_match,
        "frozen_reference_summary_hash": (
            summary_hash == case.expected_summary_sha256
        ),
        "frozen_reference_metrics": reference_metrics_match,
        "request_layer_count": len(layer_rows) == NUM_NPU * case.n_layers,
        "physical_block_count": len(block_rows) == case.expected_physical_blocks,
        "expected_block_count": len(expected) == case.expected_physical_blocks,
        "block_identity_and_placement": observed.keys() == expected.keys()
        and all(
            observed[key][:2] == expected[key][:2]
            and _close(observed[key][2], expected[key][2], 1e-15)
            for key in expected
        ),
        "all_transitions_recorded": all(
            row[field] is not None for row in block_rows for field in required
        ),
        "every_block_uses_path0": all(row["path_id"] == 0 for row in block_rows),
        "global_enqueue_order_contiguous": [
            row["global_enqueue_order"] for row in block_rows
        ]
        == list(range(len(block_rows))),
        "per_ssu_enqueue_order_contiguous": all(
            sorted(
                row["ssu_enqueue_order"]
                for row in block_rows
                if row["ssu_id"] == ssu_id
            )
            == list(range(sum(row["ssu_id"] == ssu_id for row in block_rows)))
            for ssu_id in range(case.num_ssu)
        ),
        "per_ssu_service_order_contiguous": all(
            sorted(
                row["ssd_service_order"]
                for row in block_rows
                if row["ssu_id"] == ssu_id
            )
            == list(range(sum(row["ssu_id"] == ssu_id for row in block_rows)))
            for ssu_id in range(case.num_ssu)
        ),
        "path0_service_order_equals_enqueue_order": all(
            row["ssd_service_order"] == row["ssu_enqueue_order"]
            for row in block_rows
        ),
        "transition_time_causality": all(
            row["enqueue_time_ms"]
            <= row["ssd_start_time_ms"] + TIME_TOLERANCE_MS
            and row["ssd_start_time_ms"]
            <= row["ssd_end_time_ms"] + TIME_TOLERANCE_MS
            and _close(row["ssd_end_time_ms"], row["link_enqueue_time_ms"])
            and row["link_enqueue_time_ms"]
            <= row["link_start_time_ms"] + TIME_TOLERANCE_MS
            and row["link_start_time_ms"]
            <= row["link_end_time_ms"] + TIME_TOLERANCE_MS
            for row in block_rows
        ),
        "ssd_queue_wait_accounting": all(
            _close(
                row["ssd_queue_wait_ms"],
                row["ssd_start_time_ms"] - row["enqueue_time_ms"],
            )
            for row in block_rows
        ),
        "ssd_service_duration_40GBps": all(
            _close(
                row["ssd_end_time_ms"] - row["ssd_start_time_ms"],
                1000.0 * row["size_gb"] / DISK_GBPS,
            )
            for row in block_rows
        ),
        "link_service_duration_50GBps": all(
            _close(
                row["link_end_time_ms"] - row["link_start_time_ms"],
                1000.0 * row["size_gb"] / NPU_LINK_GBPS,
            )
            for row in block_rows
        ),
        "per_ssu_service_nonoverlap": all(
            nonoverlap(
                (row for row in block_rows if row["ssu_id"] == ssu_id),
                "ssd_start_time_ms",
                "ssd_end_time_ms",
            )
            for ssu_id in range(case.num_ssu)
        ),
        "per_npu_link_nonoverlap": all(
            nonoverlap(
                (row for row in block_rows if row["npu_id"] == npu_id),
                "link_start_time_ms",
                "link_end_time_ms",
            )
            for npu_id in range(NUM_NPU)
        ),
        "io_ready_equals_last_link_completion": all(
            _close(
                row["io_ready_time_ms"],
                max(ready_by_request_layer[(row["request_id"], row["layer"])]),
            )
            for row in layer_rows
        ),
        "layer0_release_zero": all(
            _close(row["io_release_time_ms"], 0.0, 1e-12)
            for row in layer_rows
            if row["layer"] == 0
        ),
        "layer0_compute_starts_at_io_ready": all(
            _close(row["compute_start_time_ms"], row["io_ready_time_ms"])
            for row in layer_rows
            if row["layer"] == 0
        ),
        "later_release_at_previous_compute_start": all(
            _close(
                by_npu_layer[(npu_id, layer)]["io_release_time_ms"],
                by_npu_layer[(npu_id, layer - 1)]["compute_start_time_ms"],
            )
            for npu_id in range(NUM_NPU)
            for layer in range(1, case.n_layers)
        ),
        "later_deadline_at_previous_compute_end": all(
            _close(
                by_npu_layer[(npu_id, layer)]["comparison_deadline_ms"],
                by_npu_layer[(npu_id, layer - 1)]["compute_end_time_ms"],
            )
            for npu_id in range(NUM_NPU)
            for layer in range(1, case.n_layers)
        ),
        "later_compute_waits_for_io_and_previous_compute": all(
            _close(
                by_npu_layer[(npu_id, layer)]["compute_start_time_ms"],
                max(
                    by_npu_layer[(npu_id, layer)]["io_ready_time_ms"],
                    by_npu_layer[(npu_id, layer - 1)]["compute_end_time_ms"],
                ),
            )
            for npu_id in range(NUM_NPU)
            for layer in range(1, case.n_layers)
        ),
        "later_overrun_equals_actual_barrier": all(
            _close(row["overrun_ms"], row["actual_io_barrier_wait_ms"])
            for row in layer_rows
            if row["layer"] >= 1
        ),
        "late_flags_match_numeric_criterion": all(
            row["read_exceeds_compute_window"]
            == (row["overrun_ms"] > TIME_TOLERANCE_MS)
            for row in layer_rows
        ),
    }
    if not case.mixed:
        expected_compute = 5.044309980560046
        checks["homogeneous_compute_duration"] = all(
            _close(row["compute_duration_ms"], expected_compute)
            for row in layer_rows
        )
    else:
        expected_by_profile = {F_PROFILE: 9.043910535256628, V_PROFILE: 14.369102622125}
        profile_counts = Counter(case.profiles)
        checks["mixed_input_exactly_30F_2V_no_S"] = (
            profile_counts == Counter({F_PROFILE: 30, V_PROFILE: 2})
            and all(
                case.profiles[npu_id]
                == (V_PROFILE if npu_id in VICTIM_NPUS else F_PROFILE)
                for npu_id in range(NUM_NPU)
            )
            and all(
                tuple(request.load["profile_key"])
                == (V_PROFILE if request.npu_id in VICTIM_NPUS else F_PROFILE)
                for request in requests
            )
        )
        checks["mixed_compute_duration"] = all(
            _close(
                row["compute_duration_ms"],
                expected_by_profile[case.profiles[row["npu_id"]]],
            )
            for row in layer_rows
        )
    if case.cli_name == "homogeneous-layer0":
        checks["layer0_deadline_met_count_25"] = (
            sum(not row["read_exceeds_compute_window"] for row in layer_rows) == 25
        )
    required_checks = (
        checks
        if verify_reference
        else {
            key: value
            for key, value in checks.items()
            if key not in {"frozen_reference_summary_hash", "frozen_reference_metrics"}
        }
    )
    if not all(required_checks.values()):
        raise AssertionError(
            "validation failed: "
            + repr(
                {key: value for key, value in required_checks.items() if not value}
            )
        )
    return checks


def _npu_colors() -> list:
    return list(plt.get_cmap("tab20").colors) + list(
        plt.get_cmap("tab20b").colors[:12]
    )


def _case_scope(case: CaseSpec) -> str:
    if case.mixed:
        return (
            "COLD SYNCHRONIZED F-BURST MICROSCOPE — NOT WARM FORMAL R; "
            "30×F + 2×V, no S phase"
        )
    return (
        "Homogeneous cold Baseline microscope — profile (48,512), "
        f"32 NPU, 11 SSU, {case.n_layers} layer(s)"
    )


def _plot_npu_timeline(
    case: CaseSpec,
    layer_rows: list[dict],
    metrics: dict[str, dict],
    path: Path,
    *,
    visible_layers: Sequence[int] | None = None,
) -> None:
    visible_layers = tuple(
        range(case.n_layers) if visible_layers is None else visible_layers
    )
    visible_rows = [row for row in layer_rows if row["layer"] in visible_layers]
    by_key = {(row["npu_id"], row["layer"]): row for row in visible_rows}
    fig, ax = plt.subplots(figsize=(18.0, 12.7 + 0.2 * len(visible_layers)))
    io_y_offset = -0.37
    compute_y_offset = 0.07
    track_height = 0.29

    for npu_id in range(NUM_NPU):
        for layer in visible_layers:
            row = by_key[(npu_id, layer)]
            start = row["io_release_time_ms"]
            ready = row["io_ready_time_ms"]
            deadline = row["comparison_deadline_ms"]
            io_y = npu_id + io_y_offset
            compute_y = npu_id + compute_y_offset

            # The colors partition one I/O interval.  Blue and red never
            # overlap: red is exactly max(release, deadline) -> I/O Ready.
            before_end = min(ready, deadline)
            if before_end > start:
                ax.broken_barh(
                    [(start, before_end - start)],
                    (io_y, track_height),
                    facecolors=IO_COLOR,
                    edgecolors="white",
                    linewidth=0.25,
                    zorder=2,
                )
            overrun_start = max(start, deadline)
            if ready > overrun_start:
                ax.broken_barh(
                    [(overrun_start, ready - overrun_start)],
                    (io_y, track_height),
                    facecolors=OVERRUN_COLOR,
                    edgecolors="#8B0000",
                    linewidth=0.35,
                    hatch="////",
                    zorder=3,
                )

            ax.vlines(
                deadline,
                npu_id - 0.43,
                npu_id + 0.40,
                color=DEADLINE_COLOR,
                linestyle=(0, (2.2, 1.4)),
                linewidth=1.02,
                zorder=5,
            )
            ax.plot(
                ready,
                io_y + track_height / 2,
                marker="|",
                color="#111111",
                markersize=5.5,
                markeredgewidth=0.8,
                zorder=6,
            )
            ax.broken_barh(
                [(row["compute_start_time_ms"], row["compute_duration_ms"])],
                (compute_y, track_height),
                facecolors=COMPUTE_COLOR,
                edgecolors="white",
                linewidth=0.25,
                zorder=2,
            )

            label_npus = (0, 24) if case.mixed else (0,)
            if npu_id in label_npus:
                ax.annotate(
                    f"L{layer} deadline",
                    (deadline, npu_id - 0.43),
                    xytext=(2, -1),
                    textcoords="offset points",
                    rotation=90,
                    ha="left",
                    va="top",
                    fontsize=6.7,
                    color="#333333",
                    zorder=7,
                )
            if npu_id == 0:
                ax.text(
                    (start + ready) / 2.0,
                    io_y + track_height / 2,
                    f"L{layer}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="white",
                    zorder=7,
                )
                ax.text(
                    row["compute_start_time_ms"]
                    + row["compute_duration_ms"] / 2.0,
                    compute_y + track_height / 2,
                    f"L{layer}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="white",
                    zorder=7,
                )

    for boundary in np.arange(0.5, NUM_NPU, 1.0):
        ax.axhline(boundary, color="#D3D3D3", linewidth=0.5, zorder=0)
    labels = []
    for npu_id in range(NUM_NPU):
        suffix = ""
        if case.mixed:
            suffix = f" ({'V' if npu_id in VICTIM_NPUS else 'F'})"
        labels.append(f"NPU {npu_id:02d}{suffix}")
    ax.set_yticks(range(NUM_NPU))
    ax.set_yticklabels(labels)
    ax.set_ylim(NUM_NPU - 0.48, -0.68)
    ax.set_xlim(0.0, 1.015 * max(row["compute_end_time_ms"] for row in visible_rows))
    ax.set_xlabel("Time since simultaneous request release (ms)")
    ax.set_ylabel("NPU (upper subtrack: I/O; lower subtrack: compute)")
    late_counts = [
        sum(
            row["read_exceeds_compute_window"]
            for row in visible_rows
            if row["layer"] == layer
        )
        for layer in visible_layers
    ]
    view_note = (
        "same 3-layer run, displaying only L0/L1"
        if tuple(visible_layers) != tuple(range(case.n_layers))
        else f"Layers {visible_layers[0]}–{visible_layers[-1]}"
    )
    ax.set_title(
        f"{_case_scope(case)}\nBaseline Path0 · {view_note}; "
        f"beyond-deadline NPU count = {late_counts}"
    )
    ax.grid(axis="x", color="#A0A0A0", alpha=0.32, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=IO_COLOR, label="I/O before deadline"),
            Patch(
                facecolor=OVERRUN_COLOR,
                edgecolor="#8B0000",
                hatch="////",
                label="I/O time beyond deadline",
            ),
            Patch(facecolor=COMPUTE_COLOR, label="Compute"),
            Line2D(
                [0],
                [0],
                color=DEADLINE_COLOR,
                linestyle="--",
                linewidth=1.05,
                label="Per-(NPU, layer) deadline",
            ),
            Line2D(
                [0],
                [0],
                color="#111111",
                marker="|",
                linestyle="None",
                label="I/O Ready",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=5,
        framealpha=0.96,
    )

    usage_lines = []
    for layer in visible_layers:
        metric = metrics[f"layer{layer}"]
        line = (
            f"Layer {layer}   Incremental I/O stall = "
            f"{metric['incremental_io_barrier_npu_ms']:.6f} NPU·ms   "
            f"Cumulative stall = {metric['cumulative_io_barrier_npu_ms']:.6f} NPU·ms   "
            f"Cumulative utilization = {metric['cumulative_npu_utilization']:.2%}"
        )
        if layer >= 1:
            line += (
                "   Warm cumulative utilization = "
                f"{metric['warm_cumulative_npu_utilization']:.4%}"
            )
        usage_lines.append(line)
    fig.text(
        0.5,
        0.052,
        "\n".join(usage_lines),
        ha="center",
        va="center",
        fontsize=9.0,
        family="monospace",
        linespacing=1.32,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#F7F7F7",
            "edgecolor": "#BDBDBD",
            "linewidth": 0.7,
        },
    )
    fig.text(
        0.5,
        0.008,
        "L0 deadline = release + that NPU's one-layer compute time: comparison only; "
        "the full cold L0 read is real barrier. For L≥1, deadline = previous compute end, "
        "so only the red interval equals actual I/O barrier wait.",
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#333333",
    )
    fig.subplots_adjust(bottom=0.205)
    _savefig(
        fig,
        path,
        "Per-NPU deadline-partitioned I/O and compute intervals; blue is before "
        "deadline, red hatching is beyond deadline, and green is compute.",
    )


def _plot_enqueue_order(case: CaseSpec, block_rows: list[dict], path: Path) -> None:
    lane_count = case.num_ssu * case.n_layers
    total_counts = [
        sum(row["ssu_id"] == ssu_id for row in block_rows)
        for ssu_id in range(case.num_ssu)
    ]
    maximum = max(total_counts)
    raster = np.full((lane_count, maximum), np.nan)
    lane_counts = Counter((row["ssu_id"], row["layer"]) for row in block_rows)
    for row in block_rows:
        lane = row["ssu_id"] * case.n_layers + row["layer"]
        raster[lane, row["ssu_enqueue_order"]] = row["npu_id"]

    colors = _npu_colors()
    cmap = ListedColormap(colors, name="npu32")
    cmap.set_bad("#F2F2F2")
    norm = BoundaryNorm(np.arange(-0.5, NUM_NPU + 0.5, 1.0), NUM_NPU)
    height = 7.8 if case.n_layers == 1 else 6.7 + 0.38 * lane_count
    fig, ax = plt.subplots(figsize=(18.0, height))
    image = ax.imshow(
        np.ma.masked_invalid(raster),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, maximum - 0.5, lane_count - 0.5, -0.5),
    )
    for ssu_id in range(case.num_ssu):
        for layer in range(1, case.n_layers):
            ax.axhline(
                ssu_id * case.n_layers + layer - 0.5,
                color="white",
                linewidth=0.65,
            )
        if ssu_id < case.num_ssu - 1:
            ax.axhline(
                (ssu_id + 1) * case.n_layers - 0.5,
                color="white",
                linewidth=2.0,
            )
    labels = []
    for ssu_id in range(case.num_ssu):
        for layer in range(case.n_layers):
            if case.n_layers == 1:
                labels.append(
                    f"SSU {ssu_id:02d} ({lane_counts[(ssu_id, layer)]:,} blocks)"
                )
            else:
                labels.append(
                    f"SSU {ssu_id:02d} · L{layer} "
                    f"({lane_counts[(ssu_id, layer)]:,})"
                )
    ax.set_yticks(range(lane_count))
    ax.set_yticklabels(labels, fontsize=7.7)
    ax.set_xlabel("Exact combined Path0 FCFS enqueue position within each SSU")
    ax.set_ylabel("Physical SSU and layer (layer lanes are visual only)")
    ax.set_title(
        "Every physical KV-block enqueue, colored by NPU\n"
        f"{len(block_rows):,} blocks; x is queue rank, not time"
    )
    ax.grid(axis="x", color="#202020", alpha=0.15, linewidth=0.55)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=range(NUM_NPU),
        boundaries=np.arange(-0.5, NUM_NPU + 0.5, 1.0),
        fraction=0.027,
        pad=0.018,
    )
    colorbar.set_label("NPU ID")
    colorbar.ax.tick_params(labelsize=7, length=2)
    fig.text(
        0.5,
        0.012,
        "All layer lanes on one SSU share the same non-preemptive Path0 queue; "
        "the full combined queue rank is retained.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#333333",
    )
    fig.subplots_adjust(bottom=0.075)
    _savefig(fig, path, "Exact per-SSU Path0 enqueue rank for every block.")


def _plot_service(
    case: CaseSpec,
    block_rows: list[dict],
    layer_rows: list[dict],
    path: Path,
) -> None:
    colors = _npu_colors()
    cmap = ListedColormap(colors, name="npu32")
    norm = BoundaryNorm(np.arange(-0.5, NUM_NPU + 0.5, 1.0), NUM_NPU)
    segments = []
    segment_colors = []
    for row in sorted(
        block_rows, key=lambda value: (value["ssu_id"], value["ssd_service_order"])
    ):
        lane = row["ssu_id"] * case.n_layers + row["layer"]
        segments.append(
            ((row["ssd_start_time_ms"], lane), (row["ssd_end_time_ms"], lane))
        )
        segment_colors.append(colors[row["npu_id"]])
    lane_count = case.num_ssu * case.n_layers
    height = 7.8 if case.n_layers == 1 else 6.7 + 0.38 * lane_count
    fig, ax = plt.subplots(figsize=(18.0, height))
    ax.add_collection(
        LineCollection(
            segments,
            colors=segment_colors,
            linewidths=7.0 if case.n_layers == 1 else 5.2,
            capstyle="butt",
            antialiased=False,
        )
    )
    l0_compute = next(
        row["compute_duration_ms"] for row in layer_rows if row["layer"] == 0
    )
    if not case.mixed:
        ax.axvline(
            l0_compute,
            color=DEADLINE_COLOR,
            linestyle="--",
            linewidth=1.25,
            label="cold L0 comparison budget",
        )
    for layer in range(1, case.n_layers):
        deadlines = [
            row["comparison_deadline_ms"]
            for row in layer_rows
            if row["layer"] == layer
        ]
        ax.axvspan(
            min(deadlines),
            max(deadlines),
            color="#F2CF5B",
            alpha=0.13,
            label=f"range of Layer{layer} deadlines",
        )
    ax.set_xlim(0.0, 1.01 * max(row["link_end_time_ms"] for row in block_rows))
    ax.set_ylim(lane_count - 0.48, -0.55)
    labels = [
        (
            f"SSU {ssu_id:02d}"
            if case.n_layers == 1
            else f"SSU {ssu_id:02d} · L{layer}"
        )
        for ssu_id in range(case.num_ssu)
        for layer in range(case.n_layers)
    ]
    ax.set_yticks(range(lane_count))
    ax.set_yticklabels(labels, fontsize=7.7)
    for ssu_id in range(case.num_ssu):
        for layer in range(1, case.n_layers):
            ax.axhline(
                ssu_id * case.n_layers + layer - 0.5,
                color="#D3D3D3",
                linewidth=0.55,
            )
        if ssu_id < case.num_ssu - 1:
            ax.axhline(
                (ssu_id + 1) * case.n_layers - 0.5,
                color="#AFAFAF",
                linewidth=1.2,
            )
    ax.set_xlabel("Absolute SSD service time since request release (ms)")
    ax.set_ylabel("Physical SSU and layer (layer lanes are visual only)")
    ax.set_title(
        "Physical SSD command service timeline\n"
        "Every segment is one non-preemptive Path0 block; color identifies NPU"
    )
    ax.grid(axis="x", color="#A0A0A0", alpha=0.32, linewidth=0.75)
    ax.set_axisbelow(True)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        ticks=range(NUM_NPU),
        boundaries=np.arange(-0.5, NUM_NPU + 0.5, 1.0),
        fraction=0.027,
        pad=0.018,
    )
    colorbar.set_label("NPU ID")
    colorbar.ax.tick_params(labelsize=7, length=2)
    if case.n_layers > 1 or not case.mixed:
        ax.legend(loc="lower right", framealpha=0.96, fontsize=8.8)
    fig.text(
        0.5,
        0.012,
        "Layer lanes do not create capacity: each SSU serves exactly one command "
        "at a time across all layers.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#333333",
    )
    fig.subplots_adjust(bottom=0.075)
    _savefig(fig, path, "Exact physical non-preemptive SSD service intervals.")


def _plot_enqueue_time(
    case: CaseSpec,
    block_rows: list[dict],
    layer_rows: list[dict],
    timing: dict[str, dict],
    path: Path,
) -> None:
    colors = _npu_colors()
    cmap = ListedColormap(colors, name="npu32")
    norm = BoundaryNorm(np.arange(-0.5, NUM_NPU + 0.5, 1.0), NUM_NPU)
    fig, axes = plt.subplots(
        case.n_layers,
        1,
        figsize=(18.0, 3.1 + 2.25 * case.n_layers),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    maximum_span_us = max(
        timing[f"layer{layer}"]["physical_enqueue"]["span_us"]
        for layer in range(case.n_layers)
    )
    for layer, ax in enumerate(axes):
        blocks = [row for row in block_rows if row["layer"] == layer]
        layers = [row for row in layer_rows if row["layer"] == layer]
        metric = timing[f"layer{layer}"]
        origin = metric["logical_release"]["first_time_ms"]
        npu_ids = np.asarray([row["npu_id"] for row in blocks], dtype=np.int16)
        x_us = 1000.0 * (
            np.asarray([row["enqueue_time_ms"] for row in blocks]) - origin
        )
        ssu_ids = np.asarray([row["ssu_id"] for row in blocks], dtype=np.float64)
        y = ssu_ids + 0.48 * (npu_ids - (NUM_NPU - 1) / 2.0) / (NUM_NPU - 1)
        ax.scatter(
            x_us,
            y,
            c=npu_ids,
            cmap=cmap,
            norm=norm,
            marker="|",
            s=9.0,
            linewidths=0.55,
            alpha=0.88,
            rasterized=True,
            zorder=3,
        )
        release_npus = np.asarray(
            [row["npu_id"] for row in layers], dtype=np.int16
        )
        release_x = 1000.0 * (
            np.asarray([row["io_release_time_ms"] for row in layers]) - origin
        )
        release_y = -1.0 + 0.42 * (
            release_npus - (NUM_NPU - 1) / 2.0
        ) / (NUM_NPU - 1)
        ax.scatter(
            release_x,
            release_y,
            c=release_npus,
            cmap=cmap,
            norm=norm,
            marker="v",
            s=18.0,
            linewidths=0.0,
            alpha=0.95,
            zorder=5,
        )
        ax.axvline(
            metric["logical_release"]["span_us"],
            color="#4D4D4D",
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )
        for boundary in np.arange(-0.5, case.num_ssu, 1.0):
            ax.axhline(boundary, color="#D7D7D7", linewidth=0.48, zorder=0)
        ax.set_yticks([-1, *range(case.num_ssu)])
        ax.set_yticklabels(
            ["release", *[f"SSU {ssu_id:02d}" for ssu_id in range(case.num_ssu)]],
            fontsize=7.3,
        )
        ax.set_ylim(case.num_ssu - 0.45, -1.48)
        ax.set_ylabel(f"Layer {layer}", fontweight="bold")
        ax.grid(axis="x", color="#A0A0A0", alpha=0.30, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(
            0.995,
            0.94,
            (
                f"absolute t0 = {origin:.9f} ms | release span "
                f"{metric['logical_release']['span_us']:.3f} µs "
                f"({metric['logical_release']['unique_timestamp_count']} timestamp(s))\n"
                f"enqueue span {metric['physical_enqueue']['span_us']:.3f} µs | "
                "max same-time blocks (fleet / one SSU): "
                f"{metric['physical_enqueue']['max_blocks_same_timestamp']} / "
                f"{metric['physical_enqueue']['max_blocks_same_timestamp_one_ssu']}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#C8C8C8",
                "alpha": 0.94,
                "linewidth": 0.6,
            },
            zorder=6,
        )
    axes[-1].set_xlim(-7.0, 1.015 * maximum_span_us)
    axes[-1].set_xlabel(
        "Microseconds since earliest logical release in each layer (shared scale)"
    )
    fig.suptitle(
        "Physical client enqueue timing by layer and SSU\n"
        "Every vertical tick is one block; NPU is color and within-SSU y offset",
        y=0.993,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markerfacecolor="#555555",
                markeredgecolor="none",
                markersize=6,
                label="Logical NPU-layer release",
            ),
            Line2D(
                [0],
                [0],
                marker="|",
                color="#555555",
                linestyle="None",
                markersize=9,
                label="Physical block enqueue",
            ),
            Line2D(
                [0],
                [0],
                color="#4D4D4D",
                linestyle="--",
                linewidth=0.9,
                label="Latest logical release in layer",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=3,
        framealpha=0.96,
        fontsize=8.8,
    )
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_axis = fig.add_axes([0.947, 0.095, 0.014, 0.79])
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_axis,
        ticks=range(NUM_NPU),
        boundaries=np.arange(-0.5, NUM_NPU + 0.5, 1.0),
    )
    colorbar.set_label("NPU ID")
    colorbar.ax.tick_params(labelsize=6.8, length=2)
    fig.text(
        0.5,
        0.014,
        "Each panel independently resets x=0 to its earliest logical release; "
        "this is not a global cross-layer timeline. Enqueue is not SSD service.",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#333333",
    )
    fig.subplots_adjust(left=0.082, right=0.928, top=0.915, bottom=0.090, hspace=0.20)
    _savefig(fig, path, "Exact physical enqueue times by layer and SSU.")


def _request_csv_rows(case: CaseSpec, layer_rows: list[dict]):
    if case.n_layers != 1:
        return layer_rows
    return [
        {
            "npu_id": row["npu_id"],
            "request_id": row["request_id"],
            "io_start_time_ms": row["io_release_time_ms"],
            "io_ready_time_ms": row["io_ready_time_ms"],
            "io_duration_ms": row["io_duration_ms"],
            "compute_budget_ms": row["compute_duration_ms"],
            "deadline_missed": row["read_exceeds_compute_window"],
            "lateness_ms": row["overrun_ms"],
            "compute_start_time_ms": row["compute_start_time_ms"],
            "compute_end_time_ms": row["compute_end_time_ms"],
        }
        for row in layer_rows
    ]


def _output_names(case: CaseSpec) -> dict[str, str]:
    if case.n_layers == 1:
        return {
            "timeline": "01_npu_layer0_io_compute_timeline.png",
            "enqueue": "02_ssu_path0_enqueue_order.png",
            "service": "03_ssu_path0_service_timeline.png",
            "requests": "request_timeline.csv",
        }
    names = {
        "timeline": f"01_npu_layer{case.layer_tag}_io_compute_timeline.png",
        "enqueue": f"02_ssu_path0_enqueue_order_layer{case.layer_tag}.png",
        "service": f"03_ssu_path0_service_timeline_layer{case.layer_tag}.png",
        "enqueue_time": f"04_ssu_path0_enqueue_time_layer{case.layer_tag}.png",
        "requests": "request_layer_timeline.csv",
    }
    if case.mixed:
        names["timeline_simplified"] = "01b_npu_layer01_io_compute_timeline_simplified.png"
    return names


def _profile_summary(case: CaseSpec, table: dict) -> dict:
    if not case.mixed:
        required, per_layer_us, _, kv_gb = table[HOMOGENEOUS_PROFILE]
        return {
            "profile": list(HOMOGENEOUS_PROFILE),
            "per_npu_required_gbps": required,
            "per_layer_compute_ms": per_layer_us / 1000.0,
            "per_layer_kv_gb": kv_gb,
            "fleet_nominal_demand_gbps": NUM_NPU * required,
            "fleet_capacity_gbps": case.num_ssu * DISK_GBPS,
        }
    return {
        "background": {
            "profile": list(F_PROFILE),
            "npu_count": NUM_NPU - len(VICTIM_NPUS),
            "required_gbps": table[F_PROFILE][0],
            "per_layer_compute_ms": table[F_PROFILE][1] / 1000.0,
            "per_layer_kv_gb": table[F_PROFILE][3],
        },
        "victim": {
            "profile": list(V_PROFILE),
            "npu_ids": sorted(VICTIM_NPUS),
            "required_gbps": table[V_PROFILE][0],
            "per_layer_compute_ms": table[V_PROFILE][1] / 1000.0,
            "per_layer_kv_gb": table[V_PROFILE][3],
        },
        "fleet_capacity_gbps": case.num_ssu * DISK_GBPS,
    }


def _render_report(
    case: CaseSpec,
    summary: dict,
    metrics: dict[str, dict],
    names: dict[str, str],
    observer_match: bool,
    checks: dict[str, bool],
    write_physical_trace: bool,
) -> str:
    title = (
        "Baseline 单请求 Layer0 时间线"
        if case.n_layers == 1
        else (
            "Candidate R：Baseline 三层同步 F-burst 冷启动机制探针"
            if case.mixed
            else "Baseline 单请求 Layer0 + Layer1 + Layer2 时间线"
        )
    )
    lines = [f"# {title}", ""]
    if case.mixed:
        lines.extend(
            [
                "> **同步 cold F-burst 显微探针：30×F + 2×V；不含 S，非 warm formal R。**",
                "",
                "本实验不能替代或证明 formal-R 的长窗口利用率结果。",
                "",
            ]
        )
    profile_text = (
        "全部 `(48,512)`"
        if not case.mixed
        else "30 个 F=`(200,256)`；NPU 24/26 为 V=`(32,2048)`"
    )
    lines.extend(
        [
            f"- 配置：32 NPU、{case.num_ssu} SSU、{case.n_layers} 层、{profile_text}。",
            "- 每 NPU 一个请求，全部在 `t=0` 到达；无 ingress network；无跨请求 Layer0 prefetch。",
            f"- makespan：`{summary['makespan_ms']:.9f} ms`；物理 block：`{case.expected_physical_blocks:,}`。",
            f"- 仿真全程 NPU compute utilization：`{summary['fleet_npu_compute_utilization']:.6%}`；SSD mean utilization：`{summary['ssd_mean_utilization']:.6%}`。",
        ]
    )
    for layer in range(case.n_layers):
        metric = metrics[f"layer{layer}"]
        warm = (
            ""
            if layer == 0
            else "; warm 累计 NPU 利用率 "
            f"`{metric['warm_cumulative_npu_utilization']:.6%}`"
        )
        lines.append(
            f"- Layer{layer}：新增 I/O Stall "
            f"`{metric['incremental_io_barrier_npu_ms']:.6f} NPU·ms`；"
            f"累计利用率 `{metric['cumulative_npu_utilization']:.6%}`{warm}；"
            f"超 deadline NPU `{metric['over_budget_npu_ids']}`。"
        )
    lines.extend(
        [
            "",
            "## 图片",
            "",
            f"1. [NPU I/O/compute timeline]({names['timeline']})",
        ]
    )
    next_index = 2
    if "timeline_simplified" in names:
        lines.append(
            f"{next_index}. [Same-run Layer0/1 simplified timeline]({names['timeline_simplified']})"
        )
        next_index += 1
    lines.extend(
        [
            f"{next_index}. [Per-SSU Path0 enqueue order]({names['enqueue']})",
            f"{next_index + 1}. [Per-SSU physical service timeline]({names['service']})",
        ]
    )
    if "enqueue_time" in names:
        lines.append(
            f"{next_index + 2}. [Physical enqueue time by layer]({names['enqueue_time']})"
        )
    lines.extend(
        [
            "",
            "蓝色仅表示 deadline 前 I/O，红色斜线仅表示 deadline 后 I/O，绿色表示 compute。",
            "Layer0 deadline 只是比较预算；Layer1/2 红段才等于真实本层 barrier。",
            "",
            "## 数据与验证",
            "",
            f"- [Small per-request timeline]({names['requests']})",
            "- [Machine-readable summary](summary.json)",
            (
                "- [Every physical block](physical_block_trace.csv)"
                if write_physical_trace
                else "- 完整 `physical_block_trace.csv` 默认省略；使用 `--write-physical-trace` 生成。"
            ),
            f"- 校验：`{sum(checks.values())}/{len(checks)}` 为真。",
            f"- 观察器非干扰：插桩/无插桩完整 summary SHA-256 一致：`{observer_match}`。",
            "",
        ]
    )
    return "\n".join(lines)


PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def run_case(
    case: CaseSpec,
    table: dict,
    provenance: dict,
    output_root: Path,
    *,
    write_physical_trace: bool,
    verify_reference: bool,
) -> dict:
    output = (output_root / case.output_name).resolve()
    physical_path = output / "physical_block_trace.csv"
    if physical_path.exists() and not write_physical_trace:
        raise RuntimeError(
            f"refusing to leave stale {physical_path}; use a fresh --output-dir "
            "or rerun with --write-physical-trace"
        )

    requests = _build_requests(case, table)
    with PhysicalTrace() as trace:
        summary = _run_simulation(case, requests)
    traced_hash = _sha256_json(summary)

    verification = _run_simulation(case, _build_requests(case, table))
    verification_hash = _sha256_json(verification)
    observer_match = traced_hash == verification_hash
    block_rows = sorted(trace.rows, key=lambda row: row["global_enqueue_order"])
    layer_rows = _extract_layer_rows(case, summary)
    metrics = _layer_metrics(case, layer_rows)
    timing = _enqueue_timing(case, block_rows, layer_rows)
    checks = _validate(
        case,
        summary,
        traced_hash,
        observer_match,
        requests,
        layer_rows,
        block_rows,
        metrics,
        verify_reference=verify_reference,
    )

    output.mkdir(parents=True, exist_ok=True)
    names = _output_names(case)
    request_rows = _request_csv_rows(case, layer_rows)
    request_path = output / names["requests"]
    _atomic_csv(request_path, list(request_rows[0]), request_rows)

    artifact_paths = [request_path]
    if write_physical_trace:
        _atomic_csv(physical_path, list(block_rows[0]), block_rows)
        artifact_paths.append(physical_path)

    with plt.rc_context(PLOT_STYLE):
        timeline_path = output / names["timeline"]
        _plot_npu_timeline(case, layer_rows, metrics, timeline_path)
        artifact_paths.append(timeline_path)
        if "timeline_simplified" in names:
            simplified_path = output / names["timeline_simplified"]
            _plot_npu_timeline(
                case,
                layer_rows,
                metrics,
                simplified_path,
                visible_layers=(0, 1),
            )
            artifact_paths.append(simplified_path)
        enqueue_path = output / names["enqueue"]
        _plot_enqueue_order(case, block_rows, enqueue_path)
        artifact_paths.append(enqueue_path)
        service_path = output / names["service"]
        _plot_service(case, block_rows, layer_rows, service_path)
        artifact_paths.append(service_path)
        if "enqueue_time" in names:
            enqueue_time_path = output / names["enqueue_time"]
            _plot_enqueue_time(case, block_rows, layer_rows, timing, enqueue_time_path)
            artifact_paths.append(enqueue_time_path)

    report_path = output / "report.md"
    _atomic_text(
        report_path,
        _render_report(
            case,
            summary,
            metrics,
            names,
            observer_match,
            checks,
            write_physical_trace,
        ),
    )
    artifact_paths.append(report_path)

    result = {
        "schema_version": 1,
        "experiment": case.cli_name,
        "scope": {
            "description": _case_scope(case),
            "is_warm_formal_R_replay": False,
            "contains_S_phase": False if case.mixed else None,
            "simplified_layer01_is_projection_of_three_layer_run": case.mixed,
        },
        "config": {
            "num_npu": NUM_NPU,
            "num_ssu": case.num_ssu,
            "n_layers": case.n_layers,
            "requests_per_npu": 1,
            "arrival_ms": 0.0,
            "profiles_by_npu": [list(profile) for profile in case.profiles],
            "victim_npu_ids": sorted(VICTIM_NPUS) if case.mixed else [],
            "policy": "Baseline static CIR / Path0",
            "submit_order_seed": SEED,
            "disk_gbps_each": DISK_GBPS,
            "npu_link_gbps_each": NPU_LINK_GBPS,
            "cross_request_layer0_prefetch": False,
            "ingress_network": False,
            "client_submit_batch_size": 1,
            "client_issue_interval_us": 0.1,
        },
        "profile": _profile_summary(case, table),
        "metrics": {
            "makespan_ms": float(summary["makespan_ms"]),
            "fleet_npu_compute_utilization": float(
                summary["fleet_npu_compute_utilization"]
            ),
            "active_window_npu_compute_utilization": float(
                summary["active_window_npu_compute_utilization"]
            ),
            "ssd_mean_utilization": float(summary["ssd_mean_utilization"]),
            "request_count": NUM_NPU,
            "request_layer_count": len(layer_rows),
            "physical_block_count": len(block_rows),
            "layers": metrics,
            "enqueue_timing_by_layer": timing,
        },
        "observer_equivalence": {
            "instrumented_summary_sha256": traced_hash,
            "unobserved_summary_sha256": verification_hash,
            "exact_match": observer_match,
        },
        "reference_gate": {
            "enabled": verify_reference,
            "expected_summary_sha256": case.expected_summary_sha256,
            "exact_summary_match": traced_hash == case.expected_summary_sha256,
        },
        "validation": checks,
        "provenance": provenance,
        "physical_trace": {
            "row_count": len(block_rows),
            "written": write_physical_trace,
            "filename": "physical_block_trace.csv" if write_physical_trace else None,
        },
        "source": {
            path.name: {"sha256": _sha256(path)}
            for path in (
                Path(__file__),
                ROOT / "sim.py",
                ROOT / "continuous_batch_sim.py",
                ROOT / "policy_logic.py",
                ROOT / "continuous_batch_control.py",
                ROOT / "data",
            )
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    _atomic_text(
        output / "summary.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "case": case.cli_name,
        "output": str(output),
        "summary_sha256": traced_hash,
        "reference_match": traced_hash == case.expected_summary_sha256,
        "validation": f"{sum(checks.values())}/{len(checks)}",
        "makespan_ms": summary["makespan_ms"],
        "physical_block_count": len(block_rows),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(*CASES, "all"),
        default="all",
        help="fixed experiment case (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root under which one case directory is created",
    )
    parser.add_argument(
        "--write-physical-trace",
        action="store_true",
        help="also write the large physical_block_trace.csv",
    )
    parser.add_argument(
        "--verify-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the frozen simulator hash and metrics (default: enabled)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    table, provenance = _load_authenticated_table()
    selected = list(CASES.values()) if args.case == "all" else [CASES[args.case]]
    results = []
    for case in selected:
        result = run_case(
            case,
            table,
            provenance,
            args.output_dir,
            write_physical_trace=args.write_physical_trace,
            verify_reference=args.verify_reference,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(
        json.dumps(
            {
                "cases": len(results),
                "all_reference_matches": all(
                    result["reference_match"] for result in results
                ),
                "output_root": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
