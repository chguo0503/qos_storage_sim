"""Steady-state, continuously backlogged batch-1 comparison.

Every NPU owns the same deterministic balanced request stream.  The simulator
first warms every NPU, waits for a short settling interval, then measures one
common wall-clock window while the request streams remain backlogged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import sim
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    SteadyStateConfig,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import NUM_NPU, SEED, representative_profiles
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "steady_state_full_load_layer16"
OUTPUT = OUTPUT_DIR / "results.json"
SCHEMA_VERSION = 1
N_LAYERS = 16
SSU_LIST = (16, 24, 40, 70)
STEADY_CONFIG = SteadyStateConfig(
    warmup_requests_per_npu=4,
    settle_ms=500.0,
    measurement_ms=2_000.0,
    slo_alpha=2.0,
    block_ms=500.0,
)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str


CASES = (
    Case("baseline", "routing"),
    Case("layer_once", "routing"),
    Case("scheme_b", "causal_scheme_b"),
)
CASE_BY_NAME = {case.name: case for case in CASES}
RUNTIME_WEIGHTS = {"baseline": 1.0, "layer_once": 3.0, "scheme_b": 1.3}
_WORKER_TABLE = None


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-full-load-layer16-v1\0")
    for name in (
        "sim.py",
        "policy_logic.py",
        "strategy_profiles.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "six_request_workload.py",
        "steady_state_workload.py",
        "steady_state_experiment.py",
        "scheme_b_prefill.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _simulate(case, requests, *, num_ssu):
    common = {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
        "steady_state": STEADY_CONFIG,
    }
    if case.kind == "routing":
        routing = next(
            spec for spec in routing_strategy_specs() if spec.name == case.name
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=routing.client_config(),
            **common,
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
        **common,
    )


def _compact(case, num_ssu, workload, summary, wall_time_s):
    if not all(summary["invariants"].values()):
        raise AssertionError(summary["invariants"])
    if summary["mode"] != "steady_state_full_load":
        raise AssertionError("steady-state simulator returned the wrong summary")
    pressure_expected = case.name == "layer_once"
    if (summary["pressure_reports"] > 0) != pressure_expected:
        raise AssertionError("only Read once per layer may read Path pressure")
    if case.kind == "routing" and any(
        summary[field]
        for field in ("control_evaluations", "cir_commits", "cir_path_writes")
    ):
        raise AssertionError("static routing must not update CIR")
    if case.kind == "causal_scheme_b" and summary["control_evaluations"] <= 0:
        raise AssertionError("Scheme B must run its causal CIR controller")

    return {
        "strategy": case.name,
        "kind": case.kind,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary.get("input_fingerprint"),
        "mean_npu_utilization": summary["mean_npu_utilization"],
        "ttft_slo_attainment": summary["ttft_slo_attainment"],
        "request_weighted_slo_attainment": summary[
            "request_weighted_slo_attainment"
        ],
        "mean_ttft_ms": summary["mean_ttft_ms"],
        "p99_ttft_ms": summary["p99_ttft_ms"],
        "measurement_request_count": summary["measurement_request_count"],
        "request_counts_by_npu": summary["request_counts_by_npu"],
        "npu_utilizations": summary["npu_utilizations"],
        "measurement_blocks": summary["measurement_blocks"],
        "warmup_reached_ms": summary["warmup_reached_ms"],
        "measurement_start_ms": summary["measurement_start_ms"],
        "measurement_end_ms": summary["measurement_end_ms"],
        "tail_drain_ms": summary["tail_drain_ms"],
        "completed_by_npu_at_stop": summary["completed_by_npu_at_stop"],
        "pressure_reports": summary["pressure_reports"],
        "control_evaluations": summary["control_evaluations"],
        "cir_commits": summary["cir_commits"],
        "cir_path_writes": summary["cir_path_writes"],
        "events_processed": summary["events_processed"],
        "invariants": summary["invariants"],
    }


def _run_case(task):
    case, num_ssu = task
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.name} ssu={num_ssu}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    try:
        workload = prepare_steady_state_workload(
            table,
            num_npu=NUM_NPU,
            n_layers=N_LAYERS,
            num_ssu=num_ssu,
            requests_per_npu=REQUESTS_PER_NPU,
            seed=SEED,
        )
        requests = requests_from_continuous_prefill_workload(workload)
        summary = _simulate(case, requests, num_ssu=num_ssu)
        return _compact(
            case,
            num_ssu,
            workload,
            summary,
            time.perf_counter() - started,
        )
    finally:
        finished.set()


def experiment_spec():
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": NUM_NPU,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "seed": SEED,
        "strategies": [case.name for case in CASES],
        "steady_state": asdict(STEADY_CONFIG),
        "representative_profiles": [
            list(key) for key in representative_profiles(table)
        ],
        "placement": "token-block ring hash, reused by all layers",
        "full_load_policy": (
            "all requests are available from the deterministic per-NPU stream; "
            "every completion immediately admits the next request"
        ),
        "measurement_cohort": (
            "requests admitted in the half-open common measurement window; "
            "all tagged requests are drained to completion"
        ),
        "npu_utilization_definition": (
            "exact compute-interval overlap with the common measurement window "
            "/ window duration, equal-weight mean over all 128 NPUs"
        ),
        "ttft_definition": "completion minus admission for tagged requests",
        "slo_definition": (
            "per-request TTFT <= 2 * 16 * that request's compute-only layer time; "
            "attainment averaged per NPU and then equally across NPUs"
        ),
        "physical_data_plane": "SSD40 single-command -> NPU50 FCFS",
        "excluded_strategies": ["refresh8", "best_feasible"],
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _key(row):
    return row["strategy"], row["num_ssu"]


def _print(row):
    blocks = row["measurement_blocks"]
    util_range = max(block["npu_utilization"] for block in blocks) - min(
        block["npu_utilization"] for block in blocks
    )
    print(
        f"{row['strategy']} ssu={row['num_ssu']}: "
        f"util={row['mean_npu_utilization']:.2%}, "
        f"SLO@2x={row['ttft_slo_attainment']:.2%}, "
        f"requests={row['measurement_request_count']}, "
        f"block-util-range={util_range:.2%}, "
        f"wall={row['wall_time_s']:.1f}s",
        flush=True,
    )


def run(
    output=OUTPUT,
    *,
    workers=4,
    strategies=(),
    ssu_list=(),
    rerun=False,
):
    experiment = experiment_spec()
    wanted_strategies = tuple(strategies) or tuple(case.name for case in CASES)
    wanted_ssus = tuple(ssu_list) or SSU_LIST
    rows = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if cached.get("experiment") == experiment:
            rows = {_key(row): row for row in cached["results"]}

    tasks = [
        (CASE_BY_NAME[strategy], num_ssu)
        for num_ssu in wanted_ssus
        for strategy in wanted_strategies
        if (strategy, num_ssu) not in rows
    ]
    tasks.sort(
        key=lambda task: RUNTIME_WEIGHTS[task[0].name],
        reverse=True,
    )

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
                    for num_ssu in SSU_LIST
                    for case in CASES
                ),
                "selected_complete": all(
                    (strategy, num_ssu) in rows
                    for num_ssu in wanted_ssus
                    for strategy in wanted_strategies
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
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker
        )
        try:
            futures = {pool.submit(_run_case, task): task for task in tasks}
            for future in as_completed(futures):
                case, num_ssu = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    print(
                        f"FAILED {case.name} ssu={num_ssu}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                rows[_key(row)] = row
                checkpoint()
                _print(row)
        except BaseException:
            processes = tuple(pool._processes.values())
            for process in processes:
                process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    checkpoint()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
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
