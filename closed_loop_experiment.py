"""Batch-1 closed-loop QoS experiment with 64 fixed requests per NPU."""

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
from closed_loop_workload import (
    MEASURED_REQUESTS,
    N_LAYERS,
    NUM_NPU,
    REQUESTS_PER_NPU,
    SEED,
    SSU_LIST,
    WARMUP_REQUESTS,
    prepare_closed_loop_workload,
)
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    CIRControlConfig,
    MaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from closed_loop_scheme_b import CoflowSchemeBController
from continuous_prefill_client import (
    legacy_qos_config,
    legacy_strategy_specs,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id, dedicated_path_id


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "closed_loop_batch1" / "results.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Case:
    name: str
    kind: str


CASES = (
    Case("baseline", "legacy"),
    Case("layer_once", "legacy"),
    Case("scheme_b", "causal_scheme_b"),
    Case("scheme_b_manifest", "manifest_scheme_b"),
    Case("scheme_b_coflow", "coflow_scheme_b"),
    Case("full_info_edf", "full_info"),
)
CASE_BY_NAME = {case.name: case for case in CASES}
_WORKER_TABLE = None


def _source_fingerprint():
    digest = hashlib.sha256(b"closed-loop-batch1-v1\0")
    for name in (
        "sim.py",
        "strategy_profiles.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "closed_loop_workload.py",
        "closed_loop_scheme_b.py",
        "closed_loop_experiment.py",
        "scheme_b_prefill.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _compact(case, num_ssu, workload, summary, wall_time_s):
    specs = {request.request_id: request for request in workload.requests}
    batches = {batch["batch_id"]: batch for batch in summary["microbatch_metrics"]}
    raw_requests = {
        row["request_id"]: row for row in summary["request_metrics"]
    }
    measured_stop = WARMUP_REQUESTS + MEASURED_REQUESTS
    request_rows = []
    by_npu = defaultdict(list)
    for row in summary["request_metrics"]:
        spec = specs[row["request_id"]]
        sequence = spec.stream_id
        if not WARMUP_REQUESTS <= sequence < measured_stop:
            continue
        batch = batches[row["batch_id"]]
        layer0_ms = batch["layer_metrics"][0]["io_barrier_wait_ms"]
        later_ms = sum(
            layer["io_barrier_wait_ms"] for layer in batch["layer_metrics"][1:]
        )
        ideal_ms = N_LAYERS * spec.per_layer_us / 1000.0
        compact = {
            "request_id": spec.request_id,
            "npu_id": spec.npu_id,
            "sequence": sequence,
            "profile_key": list(spec.profile_key),
            "category": spec.category,
            "ttft_ms": row["processing_latency_ms"],
            "ideal_ttft_ms": ideal_ms,
            "slowdown": row["processing_latency_ms"] / ideal_ms,
            "compute_ms": row["batch_compute_ms"],
            "io_barrier_ms": row["batch_io_barrier_wait_ms"],
            "layer0_barrier_ms": layer0_ms,
            "layer1_15_barrier_ms": later_ms,
            "avg_ssd_queue_wait_ms": row["avg_ssd_queue_wait_ms"],
            "avg_npu_link_queue_wait_ms": row["avg_npu_link_queue_wait_ms"],
        }
        request_rows.append(compact)
        by_npu[spec.npu_id].append((row, compact))

    npu_rows = []
    for npu_id in range(NUM_NPU):
        rows = sorted(by_npu[npu_id], key=lambda item: item[1]["sequence"])
        first = rows[0][0]["admission_time_ms"]
        last = rows[-1][0]["completion_time_ms"]
        compute_ms = sum(item[1]["compute_ms"] for item in rows)
        npu_rows.append(
            {
                "npu_id": npu_id,
                "compute_ms": compute_ms,
                "active_window_ms": last - first,
                "utilization": compute_ms / (last - first),
            }
        )

    last_measured_sequence = measured_stop - 1
    last_sequence = REQUESTS_PER_NPU - 1
    last_measured_completion_ms = max(
        raw_requests[spec.request_id]["completion_time_ms"]
        for spec in workload.requests
        if spec.stream_id == last_measured_sequence
    )
    final_completion_by_npu_ms = [
        raw_requests[npu_id * REQUESTS_PER_NPU + last_sequence][
            "completion_time_ms"
        ]
        for npu_id in range(NUM_NPU)
    ]
    npus_present_through_measurement = sum(
        completion_ms + 1e-9 >= last_measured_completion_ms
        for completion_ms in final_completion_by_npu_ms
    )

    return {
        "strategy": case.name,
        "kind": case.kind,
        "num_ssu": num_ssu,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary["input_fingerprint"],
        "category_counts": workload.statistics["category_counts"],
        "mean_npu_utilization": float(
            np.mean([row["utilization"] for row in npu_rows])
        ),
        "request_rows": request_rows,
        "npu_rows": npu_rows,
        "diagnostics": {
            "mean_ttft_ms": float(np.mean([row["ttft_ms"] for row in request_rows])),
            "mean_layer0_barrier_ms": float(
                np.mean([row["layer0_barrier_ms"] for row in request_rows])
            ),
            "mean_layer1_15_barrier_ms": float(
                np.mean([row["layer1_15_barrier_ms"] for row in request_rows])
            ),
            "ssd_mean_utilization": summary["ssd_mean_utilization"],
            "npu_link_mean_utilization": summary["npu_link_mean_utilization"],
            "pressure_reports": summary["pressure_reports"],
            "control_evaluations": summary["control_evaluations"],
            "cir_commits": summary["cir_commits"],
            "cir_path_writes": summary["cir_path_writes"],
            "events_processed": summary["events_processed"],
            "submitted_blocks": summary["submitted_blocks"],
            "expected_read_gb": summary["expected_read_gb"],
            "request_count": summary["request_count"],
            "last_measured_completion_ms": last_measured_completion_ms,
            "first_stream_end_ms": min(final_completion_by_npu_ms),
            "guard_margin_ms": min(final_completion_by_npu_ms)
            - last_measured_completion_ms,
            "npus_present_through_measurement": npus_present_through_measurement,
            "full_load_through_measurement": (
                npus_present_through_measurement == NUM_NPU
            ),
            "invariants": summary["invariants"],
        },
    }


def _run_case(task):
    case, num_ssu = task
    started = time.perf_counter()
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    workload = prepare_closed_loop_workload(table, num_ssu=num_ssu)
    requests = requests_from_continuous_prefill_workload(workload)
    kwargs = {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
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
    elif case.kind in ("manifest_scheme_b", "coflow_scheme_b"):
        path_by_npu = tuple(dedicated_path_id(npu) for npu in range(NUM_NPU))
        controller = (
            MaxMinSchemeBController(path_by_npu, horizon_layers=1)
            if case.kind == "manifest_scheme_b"
            else CoflowSchemeBController(path_by_npu, PATH_COUNT)
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * num_ssu
            ),
            npu_dedicated_paths=path_by_npu,
            client_io_config=scheme_b_client_config(case.name),
            control=CIRControlConfig(callback=controller),
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
        workload,
        summary,
        time.perf_counter() - started,
    )


def _experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": NUM_NPU,
        "requests_per_npu": REQUESTS_PER_NPU,
        "warmup_requests_per_npu": WARMUP_REQUESTS,
        "measured_requests_per_npu": MEASURED_REQUESTS,
        "cooldown_requests_per_npu": REQUESTS_PER_NPU
        - WARMUP_REQUESTS
        - MEASURED_REQUESTS,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "seed": SEED,
        "strategies": [case.name for case in CASES],
        "ttft_definition": "completion minus admission (closed-loop processing TTFT)",
        "physical_data_plane": "SSD40 single-command -> NPU50 FCFS",
        "full_info_issue_model": "same submit_batch_size=1 and 0.1us issue interval",
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run(output=OUTPUT, *, workers=10, strategies=(), ssu_list=(), rerun=False):
    experiment = _experiment_spec()
    wanted_strategies = tuple(strategies) or tuple(case.name for case in CASES)
    wanted_ssus = tuple(ssu_list) or SSU_LIST
    rows = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                (row["strategy"], row["num_ssu"]): row
                for row in cached["results"]
            }
    tasks = [
        (CASE_BY_NAME[strategy], num_ssu)
        for num_ssu in wanted_ssus
        for strategy in wanted_strategies
        if (strategy, num_ssu) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[(case.name, num_ssu)]
            for num_ssu in SSU_LIST
            for case in CASES
            if (case.name, num_ssu) in rows
        ]
        _write(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": all(
                    (case.name, num_ssu) in rows
                    for case in CASES
                    for num_ssu in SSU_LIST
                ),
                "selected_complete": all(
                    (strategy, num_ssu) in rows
                    for strategy in wanted_strategies
                    for num_ssu in wanted_ssus
                ),
                "experiment": experiment,
                "results": ordered,
            },
        )

    if workers == 1:
        _init_worker()
        for task in tasks:
            row = _run_case(task)
            rows[(row["strategy"], row["num_ssu"])] = row
            checkpoint()
            _print(row)
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker
        ) as pool:
            futures = {pool.submit(_run_case, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows[(row["strategy"], row["num_ssu"])] = row
                checkpoint()
                _print(row)
    checkpoint()
    return output


def _print(row):
    print(
        f"{row['strategy']} ssu={row['num_ssu']}: "
        f"util={row['mean_npu_utilization']:.2%}, "
        f"ttft={row['diagnostics']['mean_ttft_ms']:.1f} ms, "
        f"wall={row['wall_time_s']:.1f} s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--ssu", action="append", type=int, choices=SSU_LIST)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        workers=args.workers,
        strategies=tuple(args.case or ()),
        ssu_list=tuple(args.ssu or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
