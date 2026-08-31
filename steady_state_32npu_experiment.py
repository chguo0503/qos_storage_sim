"""32-NPU real-profile warm/full-load controller comparison.

This is intentionally separate from :mod:`steady_state_experiment`: the
official 128-NPU result grid and its checkpoint must never be overwritten by
the smaller controller-development experiment.

The primary matrix contains ``baseline``, the retained causal
``current_scheme_b``, and the full-manifest normalized/SLO ``new_scheme_b``.
Two optional ablations keep the same manifest-aware data plane while changing
only the allocation objective or the controller's minimum update interval.
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
    CIRControlConfig,
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    MaxMinSchemeBController,
    SLOAwareSchemeBController,
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
from six_request_workload import SEED, representative_profiles
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_normalized_slo"
OUTPUT = OUTPUT_DIR / "results.json"
SCHEMA_VERSION = 1
NUM_NPU = 32
N_LAYERS = 16
SSU_LIST = (6, 10, 18)
NEW_SCHEME_MIN_UPDATE_MS = 10.0
SLOW_CONTROL_MIN_UPDATE_MS = 100.0
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
    Case("baseline", "baseline"),
    Case("current_scheme_b", "causal_scheme_b"),
    Case("manifest_absolute", "manifest_absolute_10ms"),
    Case("new_scheme_b", "normalized_slo_10ms"),
    Case("new_scheme_b_100ms", "normalized_slo_100ms"),
)
CASE_BY_NAME = {case.name: case for case in CASES}

DEFAULT_STRATEGIES = ("baseline", "current_scheme_b", "new_scheme_b")
RUNTIME_WEIGHTS = {
    "baseline": 1.0,
    "current_scheme_b": 1.3,
    "manifest_absolute": 1.4,
    "new_scheme_b": 1.4,
    "new_scheme_b_100ms": 1.4,
}
_WORKER_TABLE = None


def _source_fingerprint() -> str:
    digest = hashlib.sha256(b"steady-state-32npu-normalized-slo-v1\0")
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
        "steady_state_32npu_experiment.py",
        "scheme_b_prefill.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _common_simulation_args(num_ssu: int) -> dict:
    return {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
        "steady_state": STEADY_CONFIG,
    }


def _simulate_baseline(requests, *, num_ssu: int):
    routing = next(
        spec for spec in routing_strategy_specs() if spec.name == "baseline"
    )
    return simulate_continuous_batch(
        requests,
        qos_config=static_qos_config(),
        client_io_config=routing.client_config(),
        **_common_simulation_args(num_ssu),
    )


def _scheme_b_paths() -> tuple[int, ...]:
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _simulate_current_scheme_b(requests, *, num_ssu: int):
    paths = _scheme_b_paths()
    controller = CausalMaxMinSchemeBController(
        paths,
        cold_path_id=0,
        cold_path_cir_gbps=static_qos_config().path_cirs[0],
        path_count=PATH_COUNT,
    )
    return simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * num_ssu
        ),
        npu_dedicated_paths=paths,
        layer0_path_id=0,
        client_io_config=scheme_b_client_config("current_scheme_b"),
        causal_control=CausalLayerControlConfig(controller),
        **_common_simulation_args(num_ssu),
    )


def _simulate_manifest_scheme_b(
    requests,
    *,
    num_ssu: int,
    normalized_slo: bool,
    min_update_ms: float,
):
    paths = _scheme_b_paths()
    controller = (
        SLOAwareSchemeBController(paths, slo_alpha=STEADY_CONFIG.slo_alpha)
        if normalized_slo
        else MaxMinSchemeBController(paths, horizon_layers=N_LAYERS)
    )
    return simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * num_ssu
        ),
        npu_dedicated_paths=paths,
        # The warm Layer-0 command uses the same NPU-dedicated Path.  Its queued
        # manifest is included in the control snapshot before admission.
        layer0_path_id=None,
        client_io_config=scheme_b_client_config(
            "normalized_slo" if normalized_slo else "manifest_absolute"
        ),
        control=CIRControlConfig(
            callback=controller,
            on_batch_boundary=True,
            min_interval_ms=min_update_ms,
        ),
        **_common_simulation_args(num_ssu),
    )


def _simulate(case: Case, requests, *, num_ssu: int):
    if case.kind == "baseline":
        return _simulate_baseline(requests, num_ssu=num_ssu)
    if case.kind == "causal_scheme_b":
        return _simulate_current_scheme_b(requests, num_ssu=num_ssu)
    if case.kind == "manifest_absolute_10ms":
        return _simulate_manifest_scheme_b(
            requests,
            num_ssu=num_ssu,
            normalized_slo=False,
            min_update_ms=NEW_SCHEME_MIN_UPDATE_MS,
        )
    if case.kind in ("normalized_slo_10ms", "normalized_slo_100ms"):
        return _simulate_manifest_scheme_b(
            requests,
            num_ssu=num_ssu,
            normalized_slo=True,
            min_update_ms=(
                NEW_SCHEME_MIN_UPDATE_MS
                if case.kind == "normalized_slo_10ms"
                else SLOW_CONTROL_MIN_UPDATE_MS
            ),
        )
    raise ValueError(f"unknown 32-NPU case kind: {case.kind}")


def _validate_strategy_counters(case: Case, summary: dict):
    if summary["pressure_reports"] != 0:
        raise AssertionError("32-NPU controller comparison must not read pressure")
    if case.kind == "baseline":
        if any(
            summary[field]
            for field in ("control_evaluations", "cir_commits", "cir_path_writes")
        ):
            raise AssertionError("baseline must not evaluate or update CIR")
        return
    if summary["control_evaluations"] <= 0:
        raise AssertionError(f"{case.name} did not evaluate its controller")
    if summary["cir_commits"] <= 0 or summary["cir_path_writes"] <= 0:
        raise AssertionError(f"{case.name} did not commit a CIR target")


def _compact(case: Case, num_ssu: int, workload, summary: dict, wall_time_s: float):
    if summary["mode"] != "steady_state_full_load":
        raise AssertionError("steady-state simulator returned the wrong summary")
    if not all(summary["invariants"].values()):
        raise AssertionError(summary["invariants"])
    _validate_strategy_counters(case, summary)
    return {
        "strategy": case.name,
        "kind": case.kind,
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary["input_fingerprint"],
        "workload_statistics": workload.statistics,
        # Preserve every exact-window metric (including SSD/link matrices,
        # request rows, block stability, drain diagnostics, and invariants).
        "steady_summary": summary,
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


def experiment_spec() -> dict:
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    probe = prepare_steady_state_workload(
        table,
        num_npu=NUM_NPU,
        n_layers=N_LAYERS,
        num_ssu=SSU_LIST[1],
        requests_per_npu=len(sim.WORKLOAD_CATEGORIES),
        seed=SEED,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": NUM_NPU,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "seed": SEED,
        "strategy_slots": [case.name for case in CASES],
        "default_runnable_strategies": list(DEFAULT_STRATEGIES),
        "new_scheme_min_update_ms": NEW_SCHEME_MIN_UPDATE_MS,
        "slow_control_min_update_ms": SLOW_CONTROL_MIN_UPDATE_MS,
        "new_scheme_trigger": (
            "full-manifest request/prefetch membership event, rate-limited"
        ),
        "new_scheme_status": "connected",
        "steady_state": asdict(STEADY_CONFIG),
        "representative_profiles": [
            list(key) for key in representative_profiles(table)
        ],
        "per_npu_cycle_demand_gbps": probe.statistics["per_npu_demand_gbps"],
        "fleet_cycle_demand_gbps": probe.statistics["fleet_demand_gbps"],
        "raw_capacity_knee_ssu": probe.statistics["capacity_knee_ssu"],
        "warm_2x_fluid_capacity_knee_ssu": (
            probe.statistics["capacity_knee_ssu"] / STEADY_CONFIG.slo_alpha
        ),
        "ssu_selection": (
            "6 is below the warm-2x capacity knee; 10 is warm-2x feasible but "
            "below the raw knee; 18 is above the raw knee"
        ),
        "placement": "token-block ring hash, reused by all 16 layers",
        "full_load_policy": (
            "all 32 deterministic requests per NPU arrive into its saturated "
            "queue; completion immediately admits the next request"
        ),
        "measurement_cohort": (
            "requests admitted in the half-open common measurement window; all "
            "tagged requests are drained while every NPU remains backlogged"
        ),
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
    }


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _key(row: dict) -> tuple[str, int]:
    return row["strategy"], int(row["num_ssu"])


def _validate_paired_rows(rows: dict[tuple[str, int], dict], num_ssu: int):
    group = [row for (strategy, ssu), row in rows.items() if ssu == num_ssu]
    if len(group) < 2:
        return
    for field in (
        "assignment_hash",
        "workload_hash",
        "placement_hash",
        "trace_hash",
        "simulator_input_fingerprint",
    ):
        values = {row[field] for row in group}
        if len(values) != 1:
            raise AssertionError(f"unpaired {field} at SSU={num_ssu}")


def _print(row: dict):
    summary = row["steady_summary"]
    blocks = summary["measurement_blocks"]
    util_range = max(block["npu_utilization"] for block in blocks) - min(
        block["npu_utilization"] for block in blocks
    )
    print(
        f"{row['strategy']} ssu={row['num_ssu']}: "
        f"util={summary['mean_npu_utilization']:.2%}, "
        f"SLO@2x={summary['ttft_slo_attainment']:.2%}, "
        f"requests={summary['measurement_request_count']}, "
        f"block-util-range={util_range:.2%}, "
        f"wall={row['wall_time_s']:.1f}s",
        flush=True,
    )


def run(
    output: Path = OUTPUT,
    *,
    workers: int = 3,
    strategies=(),
    ssu_list=(),
    rerun: bool = False,
):
    experiment = experiment_spec()
    wanted_strategies = tuple(strategies) or DEFAULT_STRATEGIES
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
    tasks.sort(key=lambda task: RUNTIME_WEIGHTS[task[0].name], reverse=True)

    def checkpoint():
        for num_ssu in wanted_ssus:
            _validate_paired_rows(rows, num_ssu)
        ordered = [
            rows[(case.name, num_ssu)]
            for num_ssu in SSU_LIST
            for case in CASES
            if (case.name, num_ssu) in rows
        ]
        selected_complete = all(
            (strategy, num_ssu) in rows
            for num_ssu in wanted_ssus
            for strategy in wanted_strategies
        )
        _write(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": selected_complete,
                "primary_three_strategy_complete": all(
                    (strategy, num_ssu) in rows
                    for num_ssu in SSU_LIST
                    for strategy in DEFAULT_STRATEGIES
                ),
                "full_ablation_matrix_complete": all(
                    (case.name, num_ssu) in rows
                    for num_ssu in SSU_LIST
                    for case in CASES
                ),
                "selected_strategies": list(wanted_strategies),
                "selected_ssu_list": list(wanted_ssus),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--ssu", action="append", type=int, choices=SSU_LIST)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output,
        workers=args.workers,
        strategies=tuple(args.case or ()),
        ssu_list=tuple(args.ssu or ()),
        rerun=args.rerun,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
