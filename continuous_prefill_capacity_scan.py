"""Focused baseline/Scheme-B SSU-count scan for causal diagnosis."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    MaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    legacy_qos_config,
    legacy_strategy_specs,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from continuous_prefill_experiment import DEFAULT_OUTPUT as MAIN_RESULTS
from continuous_prefill_experiment import _experiment_spec
from continuous_prefill_workload import prepare_continuous_prefill_workload
from scheme_b_prefill import PATH_COUNT, dedicated_path_id


SSU_POINTS = (24, 26, 28, 30, 32, 40)
STRATEGIES = ("baseline", "scheme_b_once")
OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "continuous_prefill_capacity_scan"
    / "results.json"
)


def _scan_spec():
    parent = _experiment_spec()
    return {
        "source_fingerprint": parent["source_fingerprint"],
        "num_npu": parent["num_npu"],
        "batch_size": parent["batch_size"],
        "n_layers": parent["n_layers"],
        "seed": parent["seed"],
        "ssu_points": list(SSU_POINTS),
        "strategies": list(STRATEGIES),
        "physical_data_plane": parent["physical_data_plane"],
    }


def _critical_path_metrics(summary):
    by_npu = defaultdict(list)
    for request in summary["request_metrics"]:
        by_npu[request["npu_id"]].append(request)
    lower_bounds = []
    for npu_id, requests in by_npu.items():
        compute_only_finish = 0.0
        for request in sorted(requests, key=lambda row: row["arrival_time_ms"]):
            compute_only_finish = max(
                compute_only_finish, request["arrival_time_ms"]
            ) + request["own_compute_ms"]
        compute_ms = sum(request["own_compute_ms"] for request in requests)
        lower_bounds.append((compute_only_finish, npu_id, compute_ms))
    lower_bound_ms, critical_npu, critical_compute_ms = max(lower_bounds)
    return {
        "compute_only_lower_bound_ms": lower_bound_ms,
        "critical_npu": critical_npu,
        "critical_npu_compute_ms": critical_compute_ms,
        "makespan_gap_above_compute_bound_ms": (
            summary["makespan_ms"] - lower_bound_ms
        ),
    }


def _run_case(task):
    num_ssu, strategy = task
    started = time.perf_counter()
    table = sim.load_bw_table_cache(num_npu=128)
    workload = prepare_continuous_prefill_workload(table, num_ssu=num_ssu)
    requests = requests_from_continuous_prefill_workload(workload)
    kwargs = {
        "num_npu": workload.num_npu,
        "num_ssu": workload.num_ssu,
        "n_layers": workload.n_layers,
        "batch_size": workload.batch_size,
        "submit_order_seed": workload.seed,
    }
    if strategy == "baseline":
        summary = simulate_continuous_batch(
            requests,
            qos_config=legacy_qos_config(),
            client_io_config=legacy_strategy_specs()[0].client_config(),
            **kwargs,
        )
    else:
        configs = qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * workload.num_ssu
        )
        path_by_npu = tuple(
            dedicated_path_id(npu_id) for npu_id in range(workload.num_npu)
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=configs,
            npu_dedicated_paths=path_by_npu,
            client_io_config=scheme_b_client_config(strategy),
            control=CIRControlConfig(
                callback=MaxMinSchemeBController(path_by_npu)
            ),
            **kwargs,
        )
    assert all(summary["invariants"].values())
    return {
        "num_ssu": num_ssu,
        "strategy": strategy,
        "trace_hash": workload.trace_hash,
        "workload_statistics": workload.statistics,
        "wall_time_s": time.perf_counter() - started,
        "critical_path": _critical_path_metrics(summary),
        "summary": summary,
    }


def _existing_28_rows():
    payload = json.loads(MAIN_RESULTS.read_text())
    assert payload["complete"] is True
    assert payload["experiment"] == _experiment_spec()
    rows = []
    for row in payload["results"]:
        if row["strategy"] not in STRATEGIES:
            continue
        rows.append(
            {
                "num_ssu": 28,
                "strategy": row["strategy"],
                "trace_hash": row["trace_hash"],
                "workload_statistics": row["workload_statistics"],
                "wall_time_s": row["wall_time_s"],
                "critical_path": _critical_path_metrics(row["summary"]),
                "summary": row["summary"],
            }
        )
    return rows


def _write(path, rows, complete):
    ordered = [
        next(
            row
            for row in rows.values()
            if row["num_ssu"] == num_ssu and row["strategy"] == strategy
        )
        for num_ssu in SSU_POINTS
        for strategy in STRATEGIES
        if (num_ssu, strategy) in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "complete": complete,
                "experiment": _scan_spec(),
                "ssu_points": SSU_POINTS,
                "strategies": STRATEGIES,
                "results": ordered,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


def run(output=OUTPUT, workers=6, rerun=False):
    rows = {
        (row["num_ssu"], row["strategy"]): row for row in _existing_28_rows()
    }
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if cached.get("experiment") == _scan_spec():
            rows.update(
                {
                    (row["num_ssu"], row["strategy"]): row
                    for row in cached["results"]
                }
            )
    for row in rows.values():
        row["critical_path"] = _critical_path_metrics(row["summary"])
    pending = [
        (num_ssu, strategy)
        for num_ssu in SSU_POINTS
        for strategy in STRATEGIES
        if (num_ssu, strategy) not in rows
    ]
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(_run_case, task): task for task in pending}
            for future in as_completed(futures):
                row = future.result()
                rows[(row["num_ssu"], row["strategy"])] = row
                _write(output, rows, False)
                summary = row["summary"]
                print(
                    f"SSU={row['num_ssu']} {row['strategy']}: "
                    f"fleet={summary['fleet_npu_compute_utilization']:.4%}, "
                    f"gap={row['critical_path']['makespan_gap_above_compute_bound_ms']:.3f} ms, "
                    f"wall={row['wall_time_s']:.1f} s",
                    flush=True,
                )
    _write(output, rows, len(rows) == len(SSU_POINTS) * len(STRATEGIES))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, args.rerun)


if __name__ == "__main__":
    main()
