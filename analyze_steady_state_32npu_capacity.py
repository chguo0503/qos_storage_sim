"""Reproducible capacity/input audit for the 32-NPU warm workload.

This analyzer is deliberately independent of every scheduling strategy.  It
rebuilds the real trace and ring placement, then separates four questions:

* per-profile raw demand and the NPU50-only utilization ceiling;
* long-run, compute-time-weighted fleet and per-SSU demand;
* exact feasibility of each synchronized 32-NPU sequence cohort; and
* an explicitly approximate, independently dephased phase model.

The dephased model is not a DES result.  It samples each NPU's category with
probability proportional to compute time and uniformly samples one of that
NPU/category's eight deterministic placements.  The sampled phase indices
are common to every SSU count and fingerprinted in the output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import sim
from six_request_workload import BALANCED_PROFILES, SEED
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_normalized_slo"
DEFAULT_JSON = OUTPUT_DIR / "capacity_analysis.json"
DEFAULT_MARKDOWN = OUTPUT_DIR / "capacity_analysis.md"

SCHEMA_VERSION = 1
NUM_NPU = 32
N_LAYERS = 16
SSU_LIST = (6, 10, 18)
SLO_ALPHA = 2.0
SSD_CAP_GBPS = float(sim.DISK_BW)
NPU_CAP_GBPS = float(sim.NPU_BW_LIMIT)
DEPHASED_SAMPLES = 20_000
DEPHASED_SEED = 20_260_830
_EPS = 1e-9


def _source_fingerprint() -> str:
    digest = hashlib.sha256(b"steady-state-32npu-capacity-analysis-v1\0")
    for name in (
        "sim.py",
        "continuous_prefill_workload.py",
        "six_request_workload.py",
        "steady_state_workload.py",
        "analyze_steady_state_32npu_capacity.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"dephased-phase-indices-v1\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _draw_dephased_phase_indices(
    *,
    sample_count: int,
    num_npu: int,
    category_probabilities: Sequence[float],
    placements_per_category: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Draw common category/placement phases for every SSU count."""
    if sample_count <= 0 or num_npu <= 0 or placements_per_category <= 0:
        raise ValueError("dephased dimensions must be positive")
    probabilities = np.asarray(category_probabilities, dtype=float)
    if (
        probabilities.ndim != 1
        or np.any(probabilities < 0.0)
        or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("category probabilities must be a probability vector")
    rng = np.random.RandomState(int(seed))
    categories = rng.choice(
        len(probabilities),
        size=(sample_count, num_npu),
        p=probabilities,
    ).astype(np.uint8)
    placements = rng.randint(
        placements_per_category,
        size=(sample_count, num_npu),
    ).astype(np.uint8)
    return categories, placements, _hash_arrays(categories, placements)


def _profile_metrics(table: Mapping) -> tuple[dict, ...]:
    rows = []
    for category in sim.WORKLOAD_CATEGORIES:
        profile = BALANCED_PROFILES[category]
        _, per_layer_us, _, per_layer_work_gb = table[profile]
        compute_s = float(per_layer_us) / 1e6
        compute_ms = float(per_layer_us) / 1000.0
        work_gb = float(per_layer_work_gb)
        raw_demand = work_gb / compute_s
        link_ms = 1000.0 * work_gb / NPU_CAP_GBPS
        rows.append(
            {
                "category": category,
                "profile": list(profile),
                "per_layer_compute_ms": compute_ms,
                "per_layer_work_gb": work_gb,
                "raw_demand_gbps": raw_demand,
                "warm_2x_target_gbps": raw_demand / SLO_ALPHA,
                "exceeds_npu50_raw": raw_demand > NPU_CAP_GBPS + _EPS,
                "exceeds_npu50_at_2x": (
                    raw_demand / SLO_ALPHA > NPU_CAP_GBPS + _EPS
                ),
                "npu50_layer_service_ms": link_ms,
                "npu50_profile_utilization_upper_bound": min(
                    1.0, compute_ms / link_ms
                ),
                "npu50_normalized_ttft_lower_bound": max(
                    1.0, link_ms / compute_ms
                ),
            }
        )
    return tuple(rows)


def _npu50_infinite_ssd_upper_bound(
    profile_rows: Sequence[Mapping[str, float]],
    *,
    n_layers: int = N_LAYERS,
) -> dict:
    compute_ms = sum(
        n_layers * float(row["per_layer_compute_ms"]) for row in profile_rows
    )
    ideal_wall_ms = sum(
        n_layers
        * max(
            float(row["per_layer_compute_ms"]),
            float(row["npu50_layer_service_ms"]),
        )
        for row in profile_rows
    )
    return {
        "model": (
            "infinite SSD, one-layer warm prefetch, ideal SSD/link pipeline; "
            "one equally frequent request from each category"
        ),
        "cycle_compute_ms": compute_ms,
        "cycle_wall_lower_bound_ms": ideal_wall_ms,
        "mean_npu_utilization_upper_bound": compute_ms / ideal_wall_ms,
        "headroom_below_100_percent_pp": 100.0 * (1.0 - compute_ms / ideal_wall_ms),
    }


def _requests_by_npu(workload) -> dict[int, list]:
    grouped = defaultdict(list)
    for request in workload.requests:
        grouped[int(request.npu_id)].append(request)
    if set(grouped) != set(range(NUM_NPU)):
        raise AssertionError("workload does not cover every NPU")
    for requests in grouped.values():
        requests.sort(key=lambda request: request.stream_id)
        if len(requests) != REQUESTS_PER_NPU:
            raise AssertionError("unexpected requests-per-NPU prefix")
    return dict(grouped)


def _fleet_average_demand(workload) -> dict:
    per_npu = []
    for npu_id, requests in sorted(_requests_by_npu(workload).items()):
        compute_s = sum(request.per_layer_us / 1e6 for request in requests)
        work_gb = sum(request.per_layer_kv_gb for request in requests)
        per_npu.append(
            {
                "npu_id": npu_id,
                "raw_demand_gbps": work_gb / compute_s,
            }
        )
    values = [row["raw_demand_gbps"] for row in per_npu]
    fleet = float(sum(values))
    return {
        "per_npu_raw_demand_gbps_min": min(values),
        "per_npu_raw_demand_gbps_max": max(values),
        "fleet_raw_demand_gbps": fleet,
        "raw_capacity_knee_ssu": fleet / SSD_CAP_GBPS,
        "warm_2x_capacity_knee_ssu": fleet / (SLO_ALPHA * SSD_CAP_GBPS),
    }


def _long_run_placement(workload) -> tuple[np.ndarray, dict]:
    columns = np.zeros(workload.num_ssu, dtype=float)
    for requests in _requests_by_npu(workload).values():
        compute_s = sum(request.per_layer_us / 1e6 for request in requests)
        work = np.sum(
            np.asarray([request.work_by_ssu_gb for request in requests]),
            axis=0,
        )
        columns += work / compute_s
    half = columns / SLO_ALPHA
    return columns, {
        "raw_gbps_by_ssu": [float(value) for value in columns],
        "warm_2x_target_gbps_by_ssu": [float(value) for value in half],
        "raw_mean_gbps": float(columns.mean()),
        "raw_min_gbps": float(columns.min()),
        "raw_max_gbps": float(columns.max()),
        "raw_max_to_mean_ratio": float(columns.max() / columns.mean()),
        "warm_2x_max_gbps": float(half.max()),
        "raw_overloaded_ssu_count": int(np.count_nonzero(columns > SSD_CAP_GBPS + _EPS)),
        "warm_2x_overloaded_ssu_count": int(
            np.count_nonzero(half > SSD_CAP_GBPS + _EPS)
        ),
        "raw_feasible": bool(np.all(columns <= SSD_CAP_GBPS + _EPS)),
        "warm_2x_feasible": bool(np.all(half <= SSD_CAP_GBPS + _EPS)),
    }


def _sequence_demand_matrix(workload, sequence: int) -> np.ndarray:
    requests = sorted(
        (request for request in workload.requests if request.stream_id == sequence),
        key=lambda request: request.npu_id,
    )
    if len(requests) != NUM_NPU or [request.npu_id for request in requests] != list(
        range(NUM_NPU)
    ):
        raise AssertionError("synchronized sequence does not cover every NPU")
    return np.asarray(
        [
            np.asarray(request.work_by_ssu_gb, dtype=float)
            / (request.per_layer_us / 1e6)
            for request in requests
        ]
    )


def _synchronized_sequence_audit(workload) -> dict:
    rows = []
    for sequence in range(REQUESTS_PER_NPU):
        demand = _sequence_demand_matrix(workload, sequence)
        ssd_target = demand.sum(axis=0) / SLO_ALPHA
        npu_target = demand.sum(axis=1) / SLO_ALPHA
        ssd_feasible = bool(np.all(ssd_target <= SSD_CAP_GBPS + _EPS))
        npu_feasible = bool(np.all(npu_target <= NPU_CAP_GBPS + _EPS))
        rows.append(
            {
                "sequence": sequence,
                "fleet_raw_demand_gbps": float(demand.sum()),
                "max_ssu_raw_demand_gbps": float(demand.sum(axis=0).max()),
                "max_ssu_warm_2x_target_gbps": float(ssd_target.max()),
                "overloaded_ssu_count_at_2x": int(
                    np.count_nonzero(ssd_target > SSD_CAP_GBPS + _EPS)
                ),
                "max_npu_warm_2x_target_gbps": float(npu_target.max()),
                "ssd_feasible_at_2x": ssd_feasible,
                "npu_feasible_at_2x": npu_feasible,
                "joint_feasible_at_2x": ssd_feasible and npu_feasible,
            }
        )
    return {
        "sequence_count": len(rows),
        "ssd_feasible_sequence_count_at_2x": sum(
            row["ssd_feasible_at_2x"] for row in rows
        ),
        "npu_feasible_sequence_count_at_2x": sum(
            row["npu_feasible_at_2x"] for row in rows
        ),
        "joint_feasible_sequence_count_at_2x": sum(
            row["joint_feasible_at_2x"] for row in rows
        ),
        "max_ssu_warm_2x_target_gbps_across_sequences": max(
            row["max_ssu_warm_2x_target_gbps"] for row in rows
        ),
        "per_sequence": rows,
    }


def _dephased_demand_pool(workload) -> np.ndarray:
    categories = tuple(sim.WORKLOAD_CATEGORIES)
    category_id = {category: index for index, category in enumerate(categories)}
    grouped = defaultdict(list)
    for request in workload.requests:
        demand = np.asarray(request.work_by_ssu_gb, dtype=float) / (
            request.per_layer_us / 1e6
        )
        grouped[(int(request.npu_id), category_id[request.category])].append(demand)
    counts = {len(values) for values in grouped.values()}
    if counts != {REQUESTS_PER_NPU // len(categories)}:
        raise AssertionError("category placements are not balanced per NPU")
    repetitions = counts.pop()
    pool = np.zeros(
        (NUM_NPU, len(categories), repetitions, workload.num_ssu), dtype=float
    )
    for (npu_id, category), values in grouped.items():
        pool[npu_id, category] = np.asarray(values)
    return pool


def _dephased_phase_audit(
    workload,
    category_indices: np.ndarray,
    placement_indices: np.ndarray,
) -> dict:
    pool = _dephased_demand_pool(workload)
    if category_indices.shape != placement_indices.shape:
        raise ValueError("dephased index matrices must have the same shape")
    if category_indices.shape[1] != NUM_NPU:
        raise ValueError("dephased indices have the wrong NPU dimension")
    npu_ids = np.arange(NUM_NPU)[None, :]
    selected = pool[npu_ids, category_indices, placement_indices]
    ssd_targets = selected.sum(axis=1) / SLO_ALPHA
    npu_targets = selected.sum(axis=2) / SLO_ALPHA
    ssd_overloaded = ssd_targets > SSD_CAP_GBPS + _EPS
    npu_overloaded = npu_targets > NPU_CAP_GBPS + _EPS
    infeasible = ssd_overloaded.any(axis=1) | npu_overloaded.any(axis=1)
    max_ssu = ssd_targets.max(axis=1)
    fleet_target = ssd_targets.sum(axis=1)
    return {
        "sample_count": int(category_indices.shape[0]),
        "infeasible_sample_count_at_2x": int(np.count_nonzero(infeasible)),
        "infeasible_sample_fraction_at_2x": float(np.mean(infeasible)),
        "ssd_infeasible_sample_fraction_at_2x": float(
            np.mean(ssd_overloaded.any(axis=1))
        ),
        "npu_infeasible_sample_fraction_at_2x": float(
            np.mean(npu_overloaded.any(axis=1))
        ),
        "mean_overloaded_ssu_count_at_2x": float(ssd_overloaded.sum(axis=1).mean()),
        "max_ssu_target_gbps_mean": float(max_ssu.mean()),
        "max_ssu_target_gbps_p50": float(np.percentile(max_ssu, 50)),
        "max_ssu_target_gbps_p95": float(np.percentile(max_ssu, 95)),
        "max_ssu_target_gbps_p99": float(np.percentile(max_ssu, 99)),
        "fleet_target_gbps_mean": float(fleet_target.mean()),
        "fleet_target_gbps_p95": float(np.percentile(fleet_target, 95)),
    }


def build_capacity_analysis(
    *,
    dephased_samples: int = DEPHASED_SAMPLES,
    dephased_seed: int = DEPHASED_SEED,
) -> dict:
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    profiles = _profile_metrics(table)
    compute_times = np.asarray(
        [row["per_layer_compute_ms"] for row in profiles], dtype=float
    )
    category_probabilities = compute_times / compute_times.sum()
    placements_per_category = REQUESTS_PER_NPU // len(sim.WORKLOAD_CATEGORIES)
    category_indices, placement_indices, phase_hash = _draw_dephased_phase_indices(
        sample_count=dephased_samples,
        num_npu=NUM_NPU,
        category_probabilities=category_probabilities,
        placements_per_category=placements_per_category,
        seed=dephased_seed,
    )

    workloads = {}
    for num_ssu in SSU_LIST:
        workloads[num_ssu] = prepare_steady_state_workload(
            table,
            num_npu=NUM_NPU,
            n_layers=N_LAYERS,
            num_ssu=num_ssu,
            requests_per_npu=REQUESTS_PER_NPU,
            seed=SEED,
        )
    fleet_rows = [_fleet_average_demand(workload) for workload in workloads.values()]
    fleet = fleet_rows[0]
    for row in fleet_rows[1:]:
        if not math.isclose(
            row["fleet_raw_demand_gbps"],
            fleet["fleet_raw_demand_gbps"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError("fleet work changed with SSU count")

    ssu_rows = []
    for num_ssu, workload in workloads.items():
        columns, long_run = _long_run_placement(workload)
        if not math.isclose(
            float(columns.sum()),
            fleet["fleet_raw_demand_gbps"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise AssertionError("placement did not conserve fleet demand")
        ssu_rows.append(
            {
                "num_ssu": num_ssu,
                "assignment_hash": workload.statistics["assignment_hash"],
                "workload_hash": workload.workload_hash,
                "placement_hash": workload.placement_hash,
                "trace_hash": workload.trace_hash,
                "long_run_placement": long_run,
                "synchronized_sequences": _synchronized_sequence_audit(workload),
                "dephased_phase_model": _dephased_phase_audit(
                    workload, category_indices, placement_indices
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "configuration": {
            "num_npu": NUM_NPU,
            "n_layers": N_LAYERS,
            "requests_per_npu": REQUESTS_PER_NPU,
            "ssu_list": list(SSU_LIST),
            "workload_seed": SEED,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_CAP_GBPS,
            "slo_alpha": SLO_ALPHA,
        },
        "profile_raw_demand": list(profiles),
        "npu50_infinite_ssd_upper_bound": _npu50_infinite_ssd_upper_bound(profiles),
        "fleet_average_capacity": fleet,
        "dephased_phase_model": {
            "status": "approximation_not_des",
            "model": (
                "independent NPU phases; category occupancy proportional to "
                "compute-only duration; one uniformly sampled deterministic "
                "placement for that NPU/category"
            ),
            "sample_count": dephased_samples,
            "seed": dephased_seed,
            "category_order": list(sim.WORKLOAD_CATEGORIES),
            "category_probabilities": [
                float(value) for value in category_probabilities
            ],
            "placements_per_npu_category": placements_per_category,
            "common_phase_indices_hash": phase_hash,
        },
        "ssu_capacity_audits": ssu_rows,
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(analysis: Mapping) -> str:
    lines = [
        "# 32-NPU steady-state capacity/input audit",
        "",
        "本报告只分析真实 trace、ring placement、SSD40 和 NPU50 容量；不运行或比较任何调度策略。",
        "",
        "## 四类请求与 NPU50",
        "",
        "| Category | Raw demand | 2× target | Compute/layer | Work/layer | NPU50 raw bottleneck | NPU50 util ceiling |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["profile_raw_demand"]:
        lines.append(
            "| {category} | {raw} GB/s | {target} GB/s | {compute} ms | "
            "{work} GB | {bottleneck} | {util}% |".format(
                category=row["category"],
                raw=_fmt(row["raw_demand_gbps"]),
                target=_fmt(row["warm_2x_target_gbps"]),
                compute=_fmt(row["per_layer_compute_ms"]),
                work=_fmt(row["per_layer_work_gb"], 6),
                bottleneck="yes" if row["exceeds_npu50_raw"] else "no",
                util=_fmt(100.0 * row["npu50_profile_utilization_upper_bound"], 2),
            )
        )
    link = analysis["npu50_infinite_ssd_upper_bound"]
    fleet = analysis["fleet_average_capacity"]
    lines.extend(
        [
            "",
            "在无限 SSD、理想 warm pipeline 下，NPU50 给出的 fleet mean NPU utilization 上限为 "
            f"`{100.0 * link['mean_npu_utilization_upper_bound']:.3f}%`。",
            "",
            "## Fleet average 和 placement",
            "",
            f"长期 fleet raw demand 为 `{fleet['fleet_raw_demand_gbps']:.3f} GB/s`；"
            f"raw knee=`{fleet['raw_capacity_knee_ssu']:.3f}` SSU，"
            f"2× knee=`{fleet['warm_2x_capacity_knee_ssu']:.3f}` SSU。",
            "",
            "| SSU | Long-run max raw/SSU | Long-run max 2× target | Raw feasible | 2× feasible | Synchronized feasible sequences | Dephased 2× infeasible |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["ssu_capacity_audits"]:
        long_run = row["long_run_placement"]
        synchronized = row["synchronized_sequences"]
        dephased = row["dephased_phase_model"]
        lines.append(
            "| {ssu} | {raw:.3f} GB/s | {half:.3f} GB/s | {raw_ok} | "
            "{half_ok} | {sync}/{total} | {dephased:.2%} |".format(
                ssu=row["num_ssu"],
                raw=long_run["raw_max_gbps"],
                half=long_run["warm_2x_max_gbps"],
                raw_ok="yes" if long_run["raw_feasible"] else "no",
                half_ok="yes" if long_run["warm_2x_feasible"] else "no",
                sync=synchronized["joint_feasible_sequence_count_at_2x"],
                total=synchronized["sequence_count"],
                dephased=dephased["infeasible_sample_fraction_at_2x"],
            )
        )
    phase = analysis["dephased_phase_model"]
    lines.extend(
        [
            "",
            "## 口径和限制",
            "",
            "- 长期 placement 负载对每个 NPU 使用完整32请求周期的 `Σwork/Σcompute`，再跨 NPU 求和。",
            "- synchronized 行逐一检查32个 sequence 的每盘 `ΣD/2 <= 40` 和每 NPU `ΣD/2 <= 50`。",
            f"- dephased 行是固定 seed `{phase['seed']}` 的 `{phase['sample_count']}` 次独立 phase 抽样，"
            "不是 DES 稳态结果；真实 residence time 还会受到 SSD/NPU stall 和策略影响。",
            f"- 所有 SSU 共用同一组 phase indices，fingerprint=`{phase['common_phase_indices_hash']}`。",
            "- `2×` 指 processing TTFT 的流体带宽条件，不包含外部 arrival-to-admission 排队。",
            "",
            f"Source fingerprint: `{analysis['source_fingerprint']}`",
            "",
        ]
    )
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
    dephased_samples: int = DEPHASED_SAMPLES,
    dephased_seed: int = DEPHASED_SEED,
) -> tuple[Path, Path]:
    analysis = build_capacity_analysis(
        dephased_samples=dephased_samples,
        dephased_seed=dephased_seed,
    )
    _write_text(
        output_json,
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output_markdown, render_markdown(analysis))
    return output_json, output_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--dephased-samples", type=int, default=DEPHASED_SAMPLES)
    parser.add_argument("--dephased-seed", type=int, default=DEPHASED_SEED)
    args = parser.parse_args(argv)
    if args.dephased_samples <= 0:
        parser.error("--dephased-samples must be positive")
    json_path, markdown_path = run(
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        dephased_samples=args.dephased_samples,
        dephased_seed=args.dephased_seed,
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
