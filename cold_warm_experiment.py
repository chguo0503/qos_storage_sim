"""Modified baseline/Scheme-B cold-versus-warm six-request experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import sim
from cold_warm_metrics import DEFAULT_SLO_ALPHAS, cold_warm_metrics
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    routing_strategy_specs,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
    static_qos_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import (
    LAYER_LIST,
    NUM_NPU,
    REQUESTS_PER_NPU,
    SEED,
    prepare_six_request_workload,
    representative_profiles,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "cold_warm_modified" / "results.json"
SCHEMA_VERSION = 1
SSU_LIST = (40, 56, 70)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str


CASES = (
    Case("modified_baseline", "routing"),
    Case("modified_scheme_b", "causal_scheme_b"),
)
CASE_BY_NAME = {case.name: case for case in CASES}
RUNTIME_WEIGHTS = {
    "modified_baseline": 1.0,
    "modified_scheme_b": 1.22,
}
_WORKER_TABLE = None


def _source_fingerprint():
    digest = hashlib.sha256(b"six-request-cold-warm-modified-v1\0")
    for name in (
        "sim.py",
        "policy_logic.py",
        "strategy_profiles.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "six_request_workload.py",
        "cold_warm_metrics.py",
        "cold_warm_experiment.py",
        "scheme_b_prefill.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _simulate(case, requests, *, num_ssu, n_layers):
    kwargs = {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": n_layers,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
    }
    if case.kind == "routing":
        baseline = next(
            spec for spec in routing_strategy_specs() if spec.name == "baseline"
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=baseline.client_config(),
            **kwargs,
        )

    path_by_npu = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
    controller = CausalMaxMinSchemeBController(
        path_by_npu,
        cold_path_id=0,
        cold_path_cir_gbps=static_qos_config().path_cirs[0],
        path_count=PATH_COUNT,
    )
    return simulate_continuous_batch(
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


def _compact(case, num_ssu, n_layers, workload, summary, wall_time_s):
    expected_prefetches = NUM_NPU * (REQUESTS_PER_NPU - 1)
    if summary["cross_request_layer0_prefetches"] != expected_prefetches:
        raise AssertionError(
            "every post-cold request must receive cross-request Layer-0 prefetch"
        )
    expected_manifest_prefetches = (
        expected_prefetches if case.kind == "causal_scheme_b" else 0
    )
    if summary["manifest_layer0_prefetches"] != expected_manifest_prefetches:
        raise AssertionError(
            "Scheme B must configure every warm Layer-0 prefetch from its manifest"
        )
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
        **cold_warm_metrics(workload, summary),
        "diagnostics": {
            "cross_request_layer0_prefetch": summary[
                "cross_request_layer0_prefetch"
            ],
            "prefetched_request_count": summary[
                "cross_request_layer0_prefetches"
            ],
            "manifest_prefetched_request_count": summary[
                "manifest_layer0_prefetches"
            ],
            "expected_prefetched_request_count": expected_prefetches,
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
            "full_run_makespan_ms": summary["makespan_ms"],
            "invariants": summary["invariants"],
        },
    }


def _run_case(task):
    case, num_ssu, n_layers = task
    started = time.perf_counter()
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    workload = prepare_six_request_workload(
        table,
        num_ssu=num_ssu,
        n_layers=n_layers,
    )
    requests = requests_from_continuous_prefill_workload(workload)
    summary = _simulate(case, requests, num_ssu=num_ssu, n_layers=n_layers)
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


def experiment_spec():
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": NUM_NPU,
        "requests_per_npu": REQUESTS_PER_NPU,
        "layer_list": list(LAYER_LIST),
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "seed": SEED,
        "strategies": [case.name for case in CASES],
        "representative_profiles": [
            list(key) for key in representative_profiles(table)
        ],
        "slo_alphas": list(DEFAULT_SLO_ALPHAS),
        "primary_slo_alpha": 2.0,
        "modified_data_plane": "cross-request Layer-0 prefetch enabled",
        "cold_definition": (
            "per NPU: request 0 admission through request 5 completion; "
            "six-request TTFT cohort and six-request compute busy time"
        ),
        "warm_definition": (
            "per NPU: request 1 admission through request 5 completion; "
            "request 1--5 TTFT cohort and request 1--5 compute busy time"
        ),
        "npu_utilization_definition": (
            "compute busy / per-NPU cohort window, then equal-weight mean "
            "across all 128 NPUs"
        ),
        "ttft_definition": (
            "completion minus admission; external arrival queue wait excluded"
        ),
        "slo_definition": (
            "TTFT <= alpha * layer-count * request compute-only layer time"
        ),
        "survivorship_policy": (
            "simulate all six requests to completion and use fixed cohorts"
        ),
        "physical_data_plane": "SSD40 single-command -> NPU50 FCFS",
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
    workers=8,
    strategies=(),
    ssu_list=(),
    layer_list=(),
    rerun=False,
):
    experiment = experiment_spec()
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
        key=lambda task: task[2] * RUNTIME_WEIGHTS[task[0].name],
        reverse=True,
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
    cold = row["cohorts"]["cold"]
    warm = row["cohorts"]["warm"]
    print(
        f"{row['strategy']} layers={row['n_layers']} ssu={row['num_ssu']}: "
        f"cold util={cold['mean_npu_utilization']:.2%}, "
        f"SLO@2x={cold['slo']['2']['attainment']:.2%}; "
        f"warm util={warm['mean_npu_utilization']:.2%}, "
        f"SLO@2x={warm['slo']['2']['attainment']:.2%}; "
        f"wall={row['wall_time_s']:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
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
