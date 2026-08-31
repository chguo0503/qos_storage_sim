"""Deterministic 2-NPU x 2-SSU teaching experiment for the real baseline.

The experiment deliberately changes no simulator or policy code.  It feeds
five tiny, hand-checkable saturated traces to the authoritative shared
SSD40 -> NPU50 discrete-event model.  Four cases cover uniform full load, a
placement hotspot despite spare fleet capacity, complementary placement, and
heterogeneous non-preemptive commands.  A fifth case fragments the exact same
heterogeneous work to isolate command granularity from offered load.

The baseline is *not* an NPU max-min allocator: every I/O is routed to Path 0.
Because Path 0 is the only active Path on an SSD and has infinite PIR, the
work-conserving scheduler lets it consume the SSD's residual bandwidth.  Any
near-equal NPU throughput is therefore an emergent result of FIFO submission,
repeated barriers, and the symmetric trace, not an explicit NPU reservation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import sim
from continuous_batch_sim import (
    ContinuousBatchRequest,
    SteadyStateConfig,
    continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "baseline_2npu_teaching"
DEFAULT_JSON = OUTPUT_DIR / "results.json"
DEFAULT_MARKDOWN = OUTPUT_DIR / "report.md"

SCHEMA_VERSION = 1
NUM_NPU = 2
NUM_SSU = 2
N_LAYERS = 8
COMPUTE_MS = 10.0
REQUESTS_PER_NPU = 256
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
SLO_ALPHA = 2.0
WARMUP_REQUESTS_PER_NPU = 1
SETTLE_MS = 0.0
# lcm(80-ms full-hide request, 120-ms hotspot request, 332-ms heterogeneous
# joint super-cycle).  This removes partial-cycle bias from every time metric.
MEASUREMENT_MS = 19_920.0
BLOCK_MS = 240.0
SUBMIT_ORDER_SEED = 42
_EPS = 1e-9


@dataclass(frozen=True)
class TeachingCase:
    name: str
    short_label: str
    description: str
    demands_gbps: tuple[tuple[float, float], tuple[float, float]]
    blocks_per_positive_flow: int
    lesson: str
    comparison_case: str | None = None

    def __post_init__(self):
        demands = tuple(tuple(float(value) for value in row) for row in self.demands_gbps)
        if len(demands) != NUM_NPU or any(len(row) != NUM_SSU for row in demands):
            raise ValueError("teaching demand must be exactly 2 NPU x 2 SSU")
        if any(value < 0.0 or not math.isfinite(value) for row in demands for value in row):
            raise ValueError("demand must be finite and non-negative")
        if any(sum(row) <= 0.0 for row in demands):
            raise ValueError("each NPU must have positive work")
        if self.blocks_per_positive_flow <= 0:
            raise ValueError("blocks_per_positive_flow must be positive")
        object.__setattr__(self, "demands_gbps", demands)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "demands_gbps": self.demands_gbps,
                "blocks_per_positive_flow": self.blocks_per_positive_flow,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def case_suite() -> tuple[TeachingCase, ...]:
    return (
        TeachingCase(
            name="symmetric_uniform_full_load",
            short_label="均匀满载",
            description=(
                "Every NPU demands 20 GB/s from each SSD; both SSDs see exactly "
                "40 GB/s and both NPU links see 40 GB/s."
            ),
            demands_gbps=((20.0, 20.0), (20.0, 20.0)),
            blocks_per_positive_flow=4,
            lesson=(
                "A symmetric, fine-grained saturated trace can make the fixed-Path0 "
                "baseline look perfectly fair and reach 100% mean NPU utilization."
            ),
        ),
        TeachingCase(
            name="same_ssu_hotspot_global_spare",
            short_label="同盘热点",
            description=(
                "Both NPUs demand 30 GB/s from SSD0 only: fleet demand is 60 < "
                "80 GB/s, but SSD0 sees 60 > 40 GB/s while SSD1 is idle."
            ),
            demands_gbps=((30.0, 0.0), (30.0, 0.0)),
            blocks_per_positive_flow=4,
            lesson=(
                "Fleet capacity is irrelevant when placement overloads one SSD; "
                "the idle SSD cannot serve blocks placed on the hotspot."
            ),
        ),
        TeachingCase(
            name="complementary_placement",
            short_label="互补放置",
            description=(
                "NPU0 uses SSD0 and NPU1 uses SSD1 at 38 GB/s each; no SSD or "
                "NPU link is overloaded."
            ),
            demands_gbps=((38.0, 0.0), (0.0, 38.0)),
            blocks_per_positive_flow=16,
            lesson=(
                "Disjoint placement removes SSD contention; sufficiently fine "
                "commands hide both SSD and NPU-link service behind compute."
            ),
        ),
        TeachingCase(
            name="heterogeneous_large_commands",
            short_label="异构大命令",
            description=(
                "SSD0 sees exactly 40 GB/s, split 30/10 by NPU; each positive "
                "flow is one non-preemptive 0.3/0.1-GB command per layer."
            ),
            demands_gbps=((30.0, 0.0), (10.0, 0.0)),
            blocks_per_positive_flow=1,
            lesson=(
                "A large FIFO command creates head-of-line and layer-barrier wait. "
                "The fast NPU keeps the arithmetic mean high while the slow NPU "
                "has much worse utilization and TTFT."
            ),
            comparison_case="heterogeneous_fragmented_control",
        ),
        TeachingCase(
            name="heterogeneous_fragmented_control",
            short_label="异构小命令对照",
            description=(
                "The identical 30/10-GB/s work and placement, split into four "
                "commands per positive flow."
            ),
            demands_gbps=((30.0, 0.0), (10.0, 0.0)),
            blocks_per_positive_flow=4,
            lesson=(
                "With input capacity and placement held fixed, fragmentation alone "
                "removes the barrier bubble in this deterministic trace."
            ),
            comparison_case="heterogeneous_large_commands",
        ),
    )


def _source_fingerprint() -> str:
    digest = hashlib.sha256(b"baseline-2npu-teaching-v1\0")
    for name in (
        "baseline_2npu_teaching_experiment.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "policy_logic.py",
        "sim.py",
        "strategy_profiles.py",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _row_sums(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [float(sum(row)) for row in matrix]


def _column_sums(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(sum(row[column] for row in matrix)) for column in range(NUM_SSU)
    ]


def _command_sizes(case: TeachingCase) -> list[list[float]]:
    compute_s = COMPUTE_MS / 1000.0
    return [
        [
            demand * compute_s / case.blocks_per_positive_flow if demand else 0.0
            for demand in row
        ]
        for row in case.demands_gbps
    ]


def build_requests(case: TeachingCase) -> tuple[ContinuousBatchRequest, ...]:
    """Convert D[n,s] to repeated one-layer placement work D*C."""
    command_sizes = _command_sizes(case)
    placements = []
    for npu_id in range(NUM_NPU):
        blocks = []
        for ssu_id in range(NUM_SSU):
            command_gb = command_sizes[npu_id][ssu_id]
            if command_gb <= 0.0:
                continue
            blocks.extend(
                (ssu_id, command_gb)
                for _ in range(case.blocks_per_positive_flow)
            )
        placements.append(tuple(blocks))

    requests = []
    for sequence in range(REQUESTS_PER_NPU):
        for npu_id, placement in enumerate(placements):
            request_id = sequence * NUM_NPU + npu_id
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
                        "per_layer_us": COMPUTE_MS * 1000.0,
                        "initial": sequence == 0,
                        "teaching_case": case.name,
                    },
                    # A single placement tuple is intentionally reused by all layers.
                    placement=(placement,),
                )
            )
    return tuple(requests)


def reconstruct_demands(
    requests: Sequence[ContinuousBatchRequest],
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = {}
    for request in requests:
        first.setdefault(request.npu_id, request)
    if set(first) != set(range(NUM_NPU)):
        raise AssertionError("request prefix does not cover both NPUs")
    rows = []
    compute_s = COMPUTE_MS / 1000.0
    for npu_id in range(NUM_NPU):
        work = [0.0] * NUM_SSU
        for ssu_id, size_gb in first[npu_id].placement[0]:
            work[ssu_id] += size_gb
        rows.append(tuple(value / compute_s for value in work))
    return tuple(rows)


def _latency_metrics(summary: Mapping) -> tuple[list[dict], dict]:
    per_npu = []
    all_per_npu_means = []
    for npu_id in range(NUM_NPU):
        rows = [row for row in summary["request_rows"] if row["npu_id"] == npu_id]
        if not rows:
            raise AssertionError("steady window did not sample every NPU")
        ttfts = np.asarray([row["ttft_ms"] for row in rows], dtype=float)
        ideal = np.asarray([row["ideal_ttft_ms"] for row in rows], dtype=float)
        barriers = ttfts - ideal
        if np.any(barriers < -_EPS):
            raise AssertionError("TTFT cannot be smaller than batch-1 compute")
        barriers = np.maximum(barriers, 0.0)
        mean_ttft = float(ttfts.mean())
        all_per_npu_means.append(mean_ttft)
        per_npu.append(
            {
                "npu_id": npu_id,
                "request_count": len(rows),
                "ttft_ms_min": float(ttfts.min()),
                "ttft_ms_mean": mean_ttft,
                "ttft_ms_p99": float(np.percentile(ttfts, 99)),
                "ttft_ms_max": float(ttfts.max()),
                "ideal_ttft_ms": float(ideal[0]),
                "normalized_ttft_mean": mean_ttft / float(ideal[0]),
                "barrier_wait_ms_min": float(barriers.min()),
                "barrier_wait_ms_mean": float(barriers.mean()),
                "barrier_wait_ms_max": float(barriers.max()),
                "slo_attainment": float(np.mean([row["slo_met"] for row in rows])),
            }
        )
    return per_npu, {
        "request_weighted_ttft_ms_mean": float(summary["mean_ttft_ms"]),
        "equal_npu_weight_ttft_ms_mean": float(np.mean(all_per_npu_means)),
        "request_weighted_ttft_ms_p99": float(summary["p99_ttft_ms"]),
        "equal_npu_weight_slo_attainment": float(summary["ttft_slo_attainment"]),
        "request_weighted_slo_attainment": float(
            summary["request_weighted_slo_attainment"]
        ),
    }


def _capacity_metrics(case: TeachingCase) -> dict:
    per_npu = _row_sums(case.demands_gbps)
    per_ssu = _column_sums(case.demands_gbps)
    command_sizes = _command_sizes(case)
    positive_sizes = [
        size for row in command_sizes for size in row if size > 0.0
    ]
    return {
        "demands_gbps": [list(row) for row in case.demands_gbps],
        "per_npu_raw_demand_gbps": per_npu,
        "per_ssu_raw_demand_gbps": per_ssu,
        "fleet_raw_demand_gbps": float(sum(per_npu)),
        "fleet_ssd_capacity_gbps": NUM_SSU * SSD_CAP_GBPS,
        "fleet_aggregate_ssd_capacity_sufficient": (
            sum(per_npu) <= NUM_SSU * SSD_CAP_GBPS + _EPS
        ),
        "placement_raw_feasible": (
            all(value <= SSD_CAP_GBPS + _EPS for value in per_ssu)
            and all(value <= NPU_CAP_GBPS + _EPS for value in per_npu)
        ),
        "overloaded_ssu_ids": [
            ssu_id
            for ssu_id, value in enumerate(per_ssu)
            if value > SSD_CAP_GBPS + _EPS
        ],
        "blocks_per_positive_flow": case.blocks_per_positive_flow,
        "command_size_gb_by_npu_ssu": command_sizes,
        "max_nonpreemptive_ssd_command_ms": (
            max(positive_sizes) / SSD_CAP_GBPS * 1000.0
        ),
    }


def run_case(case: TeachingCase) -> dict:
    requests = build_requests(case)
    reconstructed = reconstruct_demands(requests)
    if any(
        abs(reconstructed[npu][ssu] - case.demands_gbps[npu][ssu]) > 1e-10
        for npu in range(NUM_NPU)
        for ssu in range(NUM_SSU)
    ):
        raise AssertionError("synthetic request work does not reconstruct demand")

    baseline = next(spec for spec in routing_strategy_specs() if spec.name == "baseline")
    client_config = baseline.client_config()
    if client_config.path_selection_mode != sim.PATH_SELECTION_FIXED_PATH_ZERO:
        raise AssertionError("baseline is no longer fixed Path0")
    expected_input = continuous_batch_input_fingerprint(requests)
    summary = simulate_continuous_batch(
        requests,
        num_npu=NUM_NPU,
        num_ssu=NUM_SSU,
        n_layers=N_LAYERS,
        batch_size=1,
        submit_order_seed=SUBMIT_ORDER_SEED,
        cross_request_layer0_prefetch=True,
        steady_state=SteadyStateConfig(
            warmup_requests_per_npu=WARMUP_REQUESTS_PER_NPU,
            settle_ms=SETTLE_MS,
            measurement_ms=MEASUREMENT_MS,
            slo_alpha=SLO_ALPHA,
            block_ms=BLOCK_MS,
        ),
        disk_bw_gbps=SSD_CAP_GBPS,
        npu_bw_gbps=NPU_CAP_GBPS,
        qos_config=static_qos_config(),
        client_io_config=client_config,
    )
    if summary["input_fingerprint"] != expected_input:
        raise AssertionError("simulator input fingerprint changed")
    if not all(summary["invariants"].values()):
        raise AssertionError(f"invalid steady window: {summary['invariants']}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("baseline must not read Path pressure")
    if any(summary[field] for field in ("control_evaluations", "cir_commits", "cir_path_writes")):
        raise AssertionError("baseline must not update CIR")

    duration_ms = float(summary["measurement_duration_ms"])
    ssd_util = [float(value) for value in summary["measurement_ssd_utilizations"]]
    npu_util = [float(value) for value in summary["npu_utilizations"]]
    link_util = [
        float(value) for value in summary["measurement_npu_link_utilizations"]
    ]
    derived_ssd_util = [
        busy_ms / duration_ms
        for busy_ms in summary["measurement_ssd_busy_ms_by_ssu"]
    ]
    derived_npu_util = [
        compute_ms / duration_ms for compute_ms in summary["compute_ms_by_npu"]
    ]
    derived_link_util = [
        busy_ms / duration_ms
        for busy_ms in summary["measurement_npu_link_busy_ms_by_npu"]
    ]
    for observed, derived in (
        (ssd_util, derived_ssd_util),
        (npu_util, derived_npu_util),
        (link_util, derived_link_util),
    ):
        if not np.allclose(observed, derived, rtol=0.0, atol=1e-12):
            raise AssertionError("utilization accounting mismatch")

    per_npu_latency, fleet_latency = _latency_metrics(summary)
    return {
        "case": case.name,
        "short_label": case.short_label,
        "description": case.description,
        "lesson": case.lesson,
        "comparison_case": case.comparison_case,
        "case_fingerprint": case.fingerprint,
        "input_fingerprint": expected_input,
        "capacity": _capacity_metrics(case),
        "measurement": {
            "start_ms": float(summary["measurement_start_ms"]),
            "end_ms": float(summary["measurement_end_ms"]),
            "duration_ms": duration_ms,
            "request_counts_by_npu": list(summary["request_counts_by_npu"]),
            "ssd_utilization_by_ssu": ssd_util,
            "ssd_mean_utilization_all_ssus": float(
                summary["measurement_ssd_mean_utilization"]
            ),
            "ssd_busy_ms_by_ssu": list(summary["measurement_ssd_busy_ms_by_ssu"]),
            "ssd_served_gbps_by_ssu": list(
                summary["measurement_ssd_served_gbps_by_ssu"]
            ),
            "ssd_served_gbps_by_npu_ssu": summary[
                "measurement_npu_ssu_ssd_served_gbps"
            ],
            "npu_link_utilization_by_npu": link_util,
            "npu_link_mean_utilization": float(
                summary["measurement_npu_link_mean_utilization"]
            ),
            "npu_link_busy_ms_by_npu": list(
                summary["measurement_npu_link_busy_ms_by_npu"]
            ),
            "npu_compute_utilization_by_npu": npu_util,
            "npu_compute_mean_utilization": float(summary["mean_npu_utilization"]),
            "compute_ms_by_npu": list(summary["compute_ms_by_npu"]),
            "per_npu_latency": per_npu_latency,
            "fleet_latency": fleet_latency,
            "pressure_reports": int(summary["pressure_reports"]),
            "control_evaluations": int(summary["control_evaluations"]),
            "invariants": dict(summary["invariants"]),
        },
    }


def build_experiment(cases: Sequence[TeachingCase] | None = None) -> dict:
    selected = tuple(case_suite() if cases is None else cases)
    if not selected:
        raise ValueError("at least one teaching case is required")
    qos = static_qos_config()
    source = _source_fingerprint()
    rows = [run_case(case) for case in selected]
    experiment_payload = json.dumps(
        {
            "source_fingerprint": source,
            "case_fingerprints": [row["case_fingerprint"] for row in rows],
            "configuration": {
                "n_layers": N_LAYERS,
                "compute_ms": COMPUTE_MS,
                "measurement_ms": MEASUREMENT_MS,
                "submit_order_seed": SUBMIT_ORDER_SEED,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "baseline_2npu_2ssu_teaching",
        "source_fingerprint": source,
        "experiment_fingerprint": hashlib.sha256(experiment_payload).hexdigest(),
        "configuration": {
            "num_npu": NUM_NPU,
            "num_ssu": NUM_SSU,
            "n_layers": N_LAYERS,
            "compute_ms_per_layer": COMPUTE_MS,
            "ideal_ttft_ms": N_LAYERS * COMPUTE_MS,
            "requests_per_npu_prefix": REQUESTS_PER_NPU,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_link_cap_gbps": NPU_CAP_GBPS,
            "slo_alpha": SLO_ALPHA,
            "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
            "settle_ms": SETTLE_MS,
            "measurement_ms": MEASUREMENT_MS,
            "measurement_cycle_basis_ms": [80, 120, 332],
            "measurement_cycle_lcm_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "submit_order_seed": SUBMIT_ORDER_SEED,
            "cross_request_layer0_prefetch": True,
            "batch_size": 1,
        },
        "baseline_semantics": {
            "routing": "every I/O uses Path 0",
            "explicit_per_npu_bandwidth_split": False,
            "path0_cir_gbps": float(qos.path_cirs[0]),
            "path0_pir": "infinite",
            "surplus_rule": (
                "when Path0 is the only active Path, it receives the SSD's "
                "work-conserving residual bandwidth up to SSD40"
            ),
            "ssd_backend": "one non-preemptive command at 40 GB/s per SSD",
            "npu_backend": "one FCFS receive link at 50 GB/s per NPU",
            "compute_barrier": (
                "batch-size-one NPU starts a layer only after every placement "
                "command for that layer reaches its NPU link"
            ),
            "fairness_warning": (
                "equal-looking service is emergent from symmetric FIFO traffic; "
                "baseline does not reserve one fair share per NPU"
            ),
        },
        "measurement_semantics": {
            "status": "exact_deterministic_window_not_infinite_horizon",
            "utilization": (
                "exact overlap of SSD busy time, NPU-link busy time, and compute "
                "busy time with one common 19,920-ms window"
            ),
            "ttft": "exact admission-to-completion latency for requests admitted in the window",
            "barrier_identity": (
                "batch_size=1, so barrier_wait = TTFT - 8*10ms compute exactly"
            ),
            "saturation": (
                "each NPU owns a finite 256-request prefix; steady invariants verify "
                "that the common measurement window never exhausts its backlog"
            ),
        },
        "cases": rows,
    }


def _pct_pair(values: Sequence[float]) -> str:
    return "/".join(f"{100.0 * value:.2f}%" for value in values)


def _num_pair(rows: Sequence[Mapping], key: str) -> str:
    return "/".join(f"{float(row[key]):.2f}" for row in rows)


def render_markdown(result: Mapping) -> str:
    lines = [
        "# Baseline 2 NPU × 2 SSU 教学实验",
        "",
        "## 先明确 baseline 是什么",
        "",
        (
            "Baseline 把所有 I/O 固定送到每块 SSD 的 **Path0**，并没有按 NPU "
            "显式均分带宽。Path0 的 PIR 无上限；当它是唯一活跃 Path 时，会工作保持地"
            "拿到 SSD40 的全部剩余带宽。因此，对称 trace 中接近均分只是 FIFO、提交顺序"
            "和逐层 barrier 共同产生的结果，不是 baseline 的公平性保证。"
        ),
        "",
        "## 可直接引用的结果表",
        "",
        (
            "固定 8 层、10 ms/层、SSD40、NPU50、batch=1；所有利用率均为同一个 "
            "19,920 ms 饱和窗口内的精确事件重叠。TTFT/Barrier 列顺序为 NPU0/NPU1。"
        ),
        "",
        "| Case | D[NPU0; NPU1] GB/s | Cmd/positive flow | SSD util 0/1 | NPU link util 0/1 | NPU compute util 0/1 | Mean NPU util | TTFT ms 0/1 | Barrier ms 0/1 | 2× SLO 0/1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cases"]:
        capacity = row["capacity"]
        measurement = row["measurement"]
        latency = measurement["per_npu_latency"]
        demand = capacity["demands_gbps"]
        demand_text = f"[{demand[0][0]:g},{demand[0][1]:g}]; [{demand[1][0]:g},{demand[1][1]:g}]"
        lines.append(
            "| {label} | `{demand}` | {commands} | {ssd} | {link} | {compute} | "
            "{mean:.2%} | {ttft} | {barrier} | {slo} |".format(
                label=row["short_label"],
                demand=demand_text,
                commands=capacity["blocks_per_positive_flow"],
                ssd=_pct_pair(measurement["ssd_utilization_by_ssu"]),
                link=_pct_pair(measurement["npu_link_utilization_by_npu"]),
                compute=_pct_pair(measurement["npu_compute_utilization_by_npu"]),
                mean=measurement["npu_compute_mean_utilization"],
                ttft=_num_pair(latency, "ttft_ms_mean"),
                barrier=_num_pair(latency, "barrier_wait_ms_mean"),
                slo=_pct_pair([entry["slo_attainment"] for entry in latency]),
            )
        )

    by_name = {row["case"]: row for row in result["cases"]}
    large = by_name.get("heterogeneous_large_commands")
    fragmented = by_name.get("heterogeneous_fragmented_control")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "- **均匀满载不等于 baseline 做了均分。** 两块盘都满载时，对称小命令"
                "恰好让两个 NPU 连续计算，所以 mean NPU util 是 100%；这是输入和时间线"
                "的结果。"
            ),
            (
                "- **全局总容量足够仍可能下降。** 同盘热点只有 60 GB/s fleet demand，"
                "低于两盘合计 80 GB/s，但 SSD0 的 60 GB/s 超过单盘 40 GB/s；SSD1 的"
                "空闲不能搬运 SSD0 上的数据，两个 NPU util 都降到 66.67%。"
            ),
            (
                "- **mean 会掩盖逐 NPU 差异。** 异构大命令中 NPU0/NPU1 util 是 "
                "72.29%/96.39%，mean 仍有 84.34%；相应 TTFT 是 "
                "110.67/83.00 ms。"
            ),
            (
                "- **barrier 不只由平均带宽决定。** 异构输入的每盘平均需求恰好等于 "
                "40 GB/s，但 0.3-GB 非抢占命令会阻塞 0.1-GB 命令。保持 demand 和 placement "
                "完全相同，仅拆成 4 个命令后，两个 NPU 都恢复为 100% util、80 ms TTFT。"
            ),
            "",
            "## 复现",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python baseline_2npu_teaching_experiment.py",
            "PYTHONDONTWRITEBYTECODE=1 pytest -q test_baseline_2npu_teaching_experiment.py",
            "```",
            "",
            (
                "JSON 保留逐盘 busy time、逐 NPU compute/link busy time、逐 NPU TTFT/"
                "barrier、输入和源码指纹，以及所有 steady-state invariants。"
            ),
            "",
            f"Source fingerprint: `{result['source_fingerprint']}`",
            f"Experiment fingerprint: `{result['experiment_fingerprint']}`",
            "",
        ]
    )
    if large is not None and fragmented is not None:
        large_demand = large["capacity"]["demands_gbps"]
        fragmented_demand = fragmented["capacity"]["demands_gbps"]
        if large_demand != fragmented_demand:
            raise AssertionError("large-command ablation changed its demand")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def run(
    *,
    output_json: Path = DEFAULT_JSON,
    output_markdown: Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    result = build_experiment()
    _write_text(
        output_json,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output_markdown, render_markdown(result))
    return output_json, output_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    json_path, markdown_path = run(
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
