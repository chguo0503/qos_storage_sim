"""Six balanced batch-1 requests per NPU with a fastest-finish cutoff."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import sim
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    legacy_qos_config,
    legacy_strategy_specs,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import (
    CUTOFF_SEQUENCE,
    LAYER_LIST,
    NUM_NPU,
    REQUESTS_PER_NPU,
    SEED,
    SLO_REQUESTS_PER_NPU,
    SSU_LIST,
    prepare_six_request_workload,
    representative_profiles,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "six_request_fastest_cutoff" / "results.json"
SCHEMA_VERSION = 1
SLO_ALPHAS = (1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str


CASES = (
    Case("baseline", "legacy"),
    Case("scheme_b", "causal_scheme_b"),
    Case("full_info_edf", "full_info"),
)
CASE_BY_NAME = {case.name: case for case in CASES}
RUNTIME_WEIGHTS = {
    "baseline": 1.0,
    "scheme_b": 1.22,
    "full_info_edf": 0.87,
}
_WORKER_TABLE = None


def _alpha_key(alpha):
    return f"{alpha:g}"


def _source_fingerprint():
    digest = hashlib.sha256(b"six-request-fastest-cutoff-v1\0")
    for name in (
        "sim.py",
        "strategy_profiles.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "closed_loop_workload.py",
        "closed_loop_scheme_b.py",
        "six_request_workload.py",
        "six_request_experiment.py",
        "scheme_b_prefill.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def cutoff_metrics(workload, summary, *, alphas=SLO_ALPHAS):
    """Clip utilization and classify the fixed first-five SLO cohort at T."""
    specs = {request.request_id: request for request in workload.requests}
    raw_requests = {
        row["request_id"]: row for row in summary["request_metrics"]
    }
    final_request_rows = [
        raw_requests[request.request_id]
        for request in workload.requests
        if request.stream_id == CUTOFF_SEQUENCE
    ]
    cutoff_ms = min(row["completion_time_ms"] for row in final_request_rows)
    cutoff_npus = sorted(
        row["npu_id"]
        for row in final_request_rows
        if abs(row["completion_time_ms"] - cutoff_ms) <= 1e-9
    )

    batches_by_npu = defaultdict(list)
    batches_by_id = {}
    for batch in summary["microbatch_metrics"]:
        batches_by_npu[batch["npu_id"]].append(batch)
        batches_by_id[batch["batch_id"]] = batch

    npu_rows = []
    for npu_id in range(workload.num_npu):
        first_request_id = next(
            request.request_id
            for request in workload.requests
            if request.npu_id == npu_id and request.stream_id == 0
        )
        start_ms = raw_requests[first_request_id]["admission_time_ms"]
        compute_ms = 0.0
        for batch in batches_by_npu[npu_id]:
            for layer in batch["layer_metrics"]:
                compute_ms += max(
                    0.0,
                    min(layer["compute_end_ms"], cutoff_ms)
                    - max(layer["compute_start_ms"], start_ms),
                )
        active_window_ms = cutoff_ms - start_ms
        completed = sum(
            raw_requests[request.request_id]["completion_time_ms"]
            <= cutoff_ms + 1e-9
            for request in workload.requests
            if request.npu_id == npu_id
        )
        npu_rows.append(
            {
                "npu_id": npu_id,
                "start_ms": start_ms,
                "cutoff_ms": cutoff_ms,
                "compute_ms": compute_ms,
                "active_window_ms": active_window_ms,
                "utilization": compute_ms / active_window_ms,
                "completed_requests_at_cutoff": completed,
            }
        )

    request_rows = []
    for request in workload.requests:
        if request.stream_id >= SLO_REQUESTS_PER_NPU:
            continue
        row = raw_requests[request.request_id]
        batch = batches_by_id[row["batch_id"]]
        admitted = row["admission_time_ms"] <= cutoff_ms + 1e-9
        completed = row["completion_time_ms"] <= cutoff_ms + 1e-9
        ideal_ms = workload.n_layers * request.per_layer_us / 1000.0
        statuses = {}
        for alpha in alphas:
            threshold_ms = alpha * ideal_ms
            if completed:
                status = (
                    "pass"
                    if row["processing_latency_ms"] <= threshold_ms + 1e-9
                    else "fail"
                )
            elif admitted and cutoff_ms - row["admission_time_ms"] >= threshold_ms:
                status = "fail"
            else:
                status = "censored"
            statuses[_alpha_key(alpha)] = status
        request_rows.append(
            {
                "request_id": request.request_id,
                "npu_id": request.npu_id,
                "sequence": request.stream_id,
                "profile_key": list(request.profile_key),
                "category": request.category,
                "admitted_before_cutoff": admitted,
                "completed_before_cutoff": completed,
                "admission_time_ms": row["admission_time_ms"],
                "completion_time_ms": row["completion_time_ms"] if completed else None,
                "eventual_completion_time_ms": row["completion_time_ms"],
                "ttft_ms": row["processing_latency_ms"] if completed else None,
                "eventual_ttft_ms": row["processing_latency_ms"],
                "ideal_ttft_ms": ideal_ms,
                "eventual_slowdown": row["processing_latency_ms"] / ideal_ms,
                "compute_ms": row["batch_compute_ms"],
                "io_barrier_ms": row["batch_io_barrier_wait_ms"],
                "layer0_barrier_ms": batch["layer_metrics"][0][
                    "io_barrier_wait_ms"
                ],
                "later_layer_barrier_ms": sum(
                    layer["io_barrier_wait_ms"]
                    for layer in batch["layer_metrics"][1:]
                ),
                "avg_ssd_queue_wait_ms": row["avg_ssd_queue_wait_ms"],
                "avg_npu_link_queue_wait_ms": row[
                    "avg_npu_link_queue_wait_ms"
                ],
                "slo_status": statuses,
            }
        )

    slo = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        counts = {
            status: sum(row["slo_status"][key] == status for row in request_rows)
            for status in ("pass", "fail", "censored")
        }
        total = len(request_rows)
        lower = counts["pass"] / total
        upper = (counts["pass"] + counts["censored"]) / total
        slo[key] = {
            **counts,
            "total": total,
            "lower_bound": lower,
            "upper_bound": upper,
            "attainment": None,
        }

    cohort_completed = sum(row["completed_before_cutoff"] for row in request_rows)
    cohort_complete = cohort_completed == len(request_rows)
    if cohort_complete:
        for value in slo.values():
            value["attainment"] = value["lower_bound"]
    max_cohort_completion_ms = max(
        raw_requests[request.request_id]["completion_time_ms"]
        for request in workload.requests
        if request.stream_id < SLO_REQUESTS_PER_NPU
    )
    common_start_ms = min(row["start_ms"] for row in npu_rows)
    return {
        "cutoff_ms": cutoff_ms,
        "cutoff_npu_ids": cutoff_npus,
        "mean_npu_utilization": float(
            np.mean([row["utilization"] for row in npu_rows])
        ),
        "aggregate_active_window_utilization": sum(
            row["compute_ms"] for row in npu_rows
        )
        / sum(row["active_window_ms"] for row in npu_rows),
        "common_horizon_fleet_utilization": sum(
            row["compute_ms"] for row in npu_rows
        )
        / (workload.num_npu * (cutoff_ms - common_start_ms)),
        "npu_rows": npu_rows,
        "request_rows": request_rows,
        "slo": slo,
        "slo_cohort_completed_at_cutoff": cohort_completed,
        "slo_cohort_size": len(request_rows),
        "slo_cohort_complete": cohort_complete,
        "max_slo_cohort_completion_ms": max_cohort_completion_ms,
        "slo_guard_margin_ms": cutoff_ms - max_cohort_completion_ms,
    }


def _compact(case, num_ssu, n_layers, workload, summary, wall_time_s):
    metrics = cutoff_metrics(workload, summary)
    return {
        "strategy": case.name,
        "kind": case.kind,
        "num_ssu": num_ssu,
        "n_layers": n_layers,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary["input_fingerprint"],
        "representative_profiles": workload.statistics["representative_profiles"],
        "fleet_category_counts": workload.statistics["fleet_category_counts"],
        **metrics,
        "diagnostics": {
            "ssd_mean_utilization_full_run": summary["ssd_mean_utilization"],
            "npu_link_mean_utilization_full_run": summary[
                "npu_link_mean_utilization"
            ],
            "pressure_reports": summary["pressure_reports"],
            "control_evaluations": summary["control_evaluations"],
            "cir_commits": summary["cir_commits"],
            "cir_path_writes": summary["cir_path_writes"],
            "events_processed": summary["events_processed"],
            "submitted_blocks": summary["submitted_blocks"],
            "expected_read_gb": summary["expected_read_gb"],
            "request_count": summary["request_count"],
            "full_run_makespan_ms": summary["makespan_ms"],
            "invariants": summary["invariants"],
        },
    }


def _run_case(task):
    case, num_ssu, n_layers = task
    started = time.perf_counter()
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    workload = prepare_six_request_workload(
        table, num_ssu=num_ssu, n_layers=n_layers
    )
    requests = requests_from_continuous_prefill_workload(workload)
    kwargs = {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": n_layers,
        "batch_size": 1,
        "submit_order_seed": SEED,
    }
    if case.kind == "legacy":
        spec = next(item for item in legacy_strategy_specs() if item.name == case.name)
        summary = simulate_continuous_batch(
            requests,
            qos_config=legacy_qos_config(),
            client_io_config=spec.client_config(),
            **kwargs,
        )
    elif case.kind == "causal_scheme_b":
        path_by_npu = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
        controller = CausalMaxMinSchemeBController(
            path_by_npu,
            cold_path_id=0,
            cold_path_cir_gbps=legacy_qos_config().path_cirs[0],
            path_count=PATH_COUNT,
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * num_ssu
            ),
            npu_dedicated_paths=path_by_npu,
            layer0_path_id=0,
            client_io_config=scheme_b_client_config(case.name),
            causal_control=CausalLayerControlConfig(controller),
            **kwargs,
        )
    else:
        summary = simulate_continuous_batch(
            requests,
            policy=sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
            client_io_config=scheme_b_client_config(case.name),
            **kwargs,
        )
    if not all(summary["invariants"].values()):
        raise AssertionError(summary["invariants"])
    return _compact(
        case,
        num_ssu,
        n_layers,
        workload,
        summary,
        time.perf_counter() - started,
    )


def _experiment_spec():
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    profiles = representative_profiles(table)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": NUM_NPU,
        "requests_per_npu": REQUESTS_PER_NPU,
        "slo_requests_per_npu": SLO_REQUESTS_PER_NPU,
        "cutoff_sequence": CUTOFF_SEQUENCE,
        "layer_list": list(LAYER_LIST),
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "seed": SEED,
        "strategies": [case.name for case in CASES],
        "representative_profiles": [list(key) for key in profiles],
        "slo_alphas": list(SLO_ALPHAS),
        "ttft_definition": "completion minus admission for each batch-1 request",
        "cutoff_definition": "earliest sixth-request completion across all NPUs",
        "npu_utilization_definition": (
            "per-NPU compute interval overlap / (cutoff - first admission), "
            "then equal-weight mean across all NPUs"
        ),
        "slo_definition": "TTFT <= alpha * layer-count * compute-only layer time",
        "physical_data_plane": "SSD40 single-command -> NPU50 FCFS",
        "full_info_issue_model": "same submit_batch_size=1 and 0.1us issue interval",
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _key(row):
    return row["strategy"], row["num_ssu"], row["n_layers"]


def run(
    output=OUTPUT,
    *,
    workers=10,
    strategies=(),
    ssu_list=(),
    layer_list=(),
    rerun=False,
):
    experiment = _experiment_spec()
    wanted_strategies = tuple(strategies) or tuple(case.name for case in CASES)
    wanted_ssus = tuple(ssu_list) or SSU_LIST
    wanted_layers = tuple(layer_list) or LAYER_LIST
    rows = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if cached.get("experiment") == experiment:
            rows = {_key(row): row for row in cached["results"]}
    tasks = [
        (CASE_BY_NAME[strategy], num_ssu, n_layers)
        for n_layers in wanted_layers
        for num_ssu in wanted_ssus
        for strategy in wanted_strategies
        if (strategy, num_ssu, n_layers) not in rows
    ]
    tasks.sort(
        key=lambda task: task[2] * RUNTIME_WEIGHTS[task[0].name], reverse=True
    )

    def checkpoint():
        ordered = [
            rows[(case.name, num_ssu, n_layers)]
            for n_layers in LAYER_LIST
            for num_ssu in SSU_LIST
            for case in CASES
            if (case.name, num_ssu, n_layers) in rows
        ]
        _write(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": all(
                    (case.name, num_ssu, n_layers) in rows
                    for case in CASES
                    for num_ssu in SSU_LIST
                    for n_layers in LAYER_LIST
                ),
                "selected_complete": all(
                    (strategy, num_ssu, n_layers) in rows
                    for strategy in wanted_strategies
                    for num_ssu in wanted_ssus
                    for n_layers in wanted_layers
                ),
                "experiment": experiment,
                "results": ordered,
            },
        )

    if workers == 1:
        _init_worker()
        for task in tasks:
            row = _run_case(task)
            rows[_key(row)] = row
            checkpoint()
            _print(row)
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker
        ) as pool:
            futures = {pool.submit(_run_case, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows[_key(row)] = row
                checkpoint()
                _print(row)
    checkpoint()
    return output


def _print(row):
    slo = row["slo"]["2"]
    slo_text = (
        f"{slo['attainment']:.2%}"
        if slo["attainment"] is not None
        else f"[{slo['lower_bound']:.2%},{slo['upper_bound']:.2%}]"
    )
    print(
        f"{row['strategy']} layers={row['n_layers']} ssu={row['num_ssu']}: "
        f"util={row['mean_npu_utilization']:.2%}, SLO@2x={slo_text}, "
        f"cohort={row['slo_cohort_completed_at_cutoff']}/"
        f"{row['slo_cohort_size']}, wall={row['wall_time_s']:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--ssu", action="append", type=int, choices=SSU_LIST)
    parser.add_argument("--layer", action="append", type=int, choices=LAYER_LIST)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        workers=args.workers,
        strategies=tuple(args.case or ()),
        ssu_list=tuple(args.ssu or ()),
        layer_list=tuple(args.layer or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
