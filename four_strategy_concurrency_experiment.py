"""Run the focused four-strategy finite-issue experiment.

The matrix is intentionally fixed so the resulting curves compare only:

* the NPU round-robin baseline;
* static QoS before CIR tuning (20/4/12/4 GB/s);
* static QoS after CIR tuning (20/6/8/6 GB/s); and
* the infeasible fluid no-inter-NPU-contention upper bound.

All realizable strategies use one-I/O client submission with a 0.1 us issue
interval.  Both static policies use the same refresh-every-eight-I/Os routing,
the same Path allocation, and the same paired workload/placement inputs.  Thus
their only intended policy difference is the four category CIR budgets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import sim
from experiment import _summary
from strategy_profiles import CURRENT_STATIC, STATIC_PROFILES
from upper_bounds import isolated_no_contention_bound


SCHEMA_VERSION = 1
NUM_NPU = 128
N_LAYERS = 16
SSU_LIST = (40, 56, 80)
SEEDS = (42, 43)
LS_RATIO = 0.5
ARRIVAL_DELAY_MAX_MS = 5.0
ISSUE_INTERVAL_US = 0.1
PRESSURE_WINDOW_IO = 8
SUBMIT_BATCH_SIZE = 1
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
DEFAULT_OUTPUT_DIR = Path("results/four_strategy_concurrency")

BEFORE_PROFILE = CURRENT_STATIC
AFTER_PROFILE = STATIC_PROFILES[
    "low_protect_cir_20_6_8_6_current_paths"
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    policy: str | None = None
    profile_name: str | None = None
    description: str = ""

    def client_config(self):
        if self.kind == "upper_bound":
            return None
        return sim.ClientIOConfig(
            name=(
                "refresh8_batch1_issue0p1us"
                if self.policy == sim.POLICY_QOS_STATIC_CIR
                else "baseline_batch1_issue0p1us"
            ),
            pressure_window_io=(
                PRESSURE_WINDOW_IO
                if self.policy == sim.POLICY_QOS_STATIC_CIR
                else None
            ),
            submit_batch_size=SUBMIT_BATCH_SIZE,
            issue_interval_us=ISSUE_INTERVAL_US,
        )

    def config(self):
        result = asdict(self)
        client = self.client_config()
        if client is not None:
            result["client_io"] = asdict(client)
        if self.profile_name is not None:
            profile = STATIC_PROFILES[self.profile_name]
            result["category_cir_gbps"] = list(profile.category_cir_gbps)
            result["category_paths_per_group"] = list(
                profile.category_paths_per_group
            )
            result["path_pir"] = "uncapped"
        return result


def strategy_specs():
    """Return the only four strategies admitted by this experiment."""
    return (
        StrategySpec(
            "baseline",
            "simulation",
            sim.POLICY_BASELINE_BYPASS,
            description="NPU round-robin baseline on the shared data plane",
        ),
        StrategySpec(
            "static_cir_before",
            "simulation",
            sim.POLICY_QOS_STATIC_CIR,
            BEFORE_PROFILE.name,
            "Static Path QoS with CIR 20/4/12/4 GB/s",
        ),
        StrategySpec(
            "static_cir_after",
            "simulation",
            sim.POLICY_QOS_STATIC_CIR,
            AFTER_PROFILE.name,
            "Static Path QoS with CIR 20/6/8/6 GB/s",
        ),
        StrategySpec(
            "fluid_no_contention_bound",
            "upper_bound",
            description=(
                "Fluid upper bound after removing all inter-NPU contention"
            ),
        ),
    )


def seed_bundle(seed):
    return {
        "workload": int(seed),
        "placement": int(seed) + PLACEMENT_SEED_OFFSET,
        "submit_order": int(seed) + SUBMIT_ORDER_SEED_OFFSET,
        "arrival_delay": int(seed) + ARRIVAL_DELAY_SEED_OFFSET,
    }


def runtime_config():
    return {
        "mode": "formal",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "seeds": list(SEEDS),
        "seed_bundles": {str(seed): seed_bundle(seed) for seed in SEEDS},
        "ls_ratio": LS_RATIO,
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": sim.DISK_BW,
        "npu_bw_limit_gbps": sim.NPU_BW_LIMIT,
        "client_submit_batch_size": SUBMIT_BATCH_SIZE,
        "client_issue_interval_us": ISSUE_INTERVAL_US,
        "static_pressure_window_io": PRESSURE_WINDOW_IO,
    }


def _table_fingerprint(table):
    rows = [[list(key), list(table[key])] for key in sorted(table)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


def _code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "experiment.py",
        "strategy_profiles.py",
        "upper_bounds.py",
        "four_strategy_concurrency_experiment.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def experiment_spec(table, runtime):
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "data_fingerprint": _table_fingerprint(table),
        "runtime": runtime,
        "formal_contract": {
            "num_npu": NUM_NPU,
            "n_layers": N_LAYERS,
            "ssu_list": list(SSU_LIST),
            "seeds": list(SEEDS),
        },
        "controlled_comparison": {
            "simulation_submit_batch_size": SUBMIT_BATCH_SIZE,
            "simulation_issue_interval_us": ISSUE_INTERVAL_US,
            "static_routing": "refresh_path_pressure_every_8_ios",
            "static_path_allocation_per_group": list(
                BEFORE_PROFILE.category_paths_per_group
            ),
            "isolated_static_difference": "category_cir_gbps",
        },
        "backend": {
            "model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
            "ssd_service": "io_size_gb / 40 GB/s",
            "ssd_max_active_io": 1,
            "npu_service": "io_size_gb / 50 GB/s",
            "npu_max_active_io": 1,
            "block_visible_after": "npu_link_completion",
        },
        "available_strategies": [spec.config() for spec in strategy_specs()],
    }


def prepare(table, runtime, seed, num_ssu):
    seeds = runtime["seed_bundles"][str(seed)]
    return sim.prepare_simulation_inputs(
        table,
        total_requests=runtime["num_npu"],
        n_layers=runtime["n_layers"],
        num_disk=num_ssu,
        ls_ratio=runtime["ls_ratio"],
        workload_seed=seeds["workload"],
        placement_seed=seeds["placement"],
        arrival_delay_seed=seeds["arrival_delay"],
        arrival_delay_max_ms=runtime["arrival_delay_ms"][1],
    )


def _bound_summary(bound):
    return {
        "name": bound["name"],
        "interpretation": "optimistic_infeasible_relaxation_upper_bound",
        "relaxations": bound["relaxations"],
        "avg_request_compute_fraction_upper_bound": bound[
            "avg_request_compute_fraction_upper_bound"
        ],
    }


def run_strategy_case(table, runtime, seed, num_ssu, strategy, prepared=None):
    started = time.perf_counter()
    if prepared is None:
        prepared = prepare(table, runtime, seed, num_ssu)
    seeds = runtime["seed_bundles"][str(seed)]
    base = {
        "seed": seed,
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "kind": strategy.kind,
        "config": strategy.config(),
        "seeds": seeds,
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
    }
    if strategy.kind == "upper_bound":
        bound = isolated_no_contention_bound(prepared)
        base.update(
            {
                "summary": _bound_summary(bound),
                "request_metrics": bound["request_bounds"],
                "wall_time_s": time.perf_counter() - started,
            }
        )
        return base

    qos_config = (
        STATIC_PROFILES[strategy.profile_name].hardware_config()
        if strategy.profile_name is not None
        else None
    )
    _, full = sim.simulate_continuous(
        table,
        policy=strategy.policy,
        num_npu=runtime["num_npu"],
        num_disk=num_ssu,
        n_layers=runtime["n_layers"],
        ls_ratio=runtime["ls_ratio"],
        qos_config=qos_config,
        client_io_config=strategy.client_config(),
        submit_order_seed=seeds["submit_order"],
        prepared_inputs=prepared,
    )
    compact = _summary(full)
    request_metrics = compact.pop("request_metrics")
    compact["client_submit_batch_size"] = full["client_submit_batch_size"]
    compact["client_submit_interval_us"] = full["client_submit_interval_us"]
    compact["pressure_read_interval"] = full["pressure_read_interval"]
    compact["client_submission"] = full["client_submission"]

    assert all(compact["invariants"].values())
    assert compact["workload_fingerprint"] == prepared.workload_hash
    assert compact["placement_hash"] == prepared.placement_hash
    assert compact["client_submit_batch_size"] == SUBMIT_BATCH_SIZE
    assert compact["client_submit_interval_us"] == ISSUE_INTERVAL_US
    if strategy.policy == sim.POLICY_QOS_STATIC_CIR:
        assert compact["pressure_read_interval"] == PRESSURE_WINDOW_IO
    else:
        assert compact["pressure_read_interval"] is None
        assert compact["pressure_reports"] == 0

    base.update(
        {
            "summary": compact,
            "request_metrics": request_metrics,
            "wall_time_s": time.perf_counter() - started,
        }
    )
    return base


_WORKER_TABLE = None
_WORKER_RUNTIME = None
_WORKER_STRATEGIES = None
_WORKER_PREPARED = None


def _init_worker(table, runtime, selected):
    global _WORKER_TABLE, _WORKER_RUNTIME, _WORKER_STRATEGIES, _WORKER_PREPARED
    _WORKER_TABLE = table
    _WORKER_RUNTIME = runtime
    _WORKER_STRATEGIES = {strategy.name: strategy for strategy in selected}
    _WORKER_PREPARED = {}


def _worker(task):
    seed, num_ssu, strategy_name = task
    key = (seed, num_ssu)
    if key not in _WORKER_PREPARED:
        _WORKER_PREPARED[key] = prepare(
            _WORKER_TABLE, _WORKER_RUNTIME, seed, num_ssu
        )
    return run_strategy_case(
        _WORKER_TABLE,
        _WORKER_RUNTIME,
        seed,
        num_ssu,
        _WORKER_STRATEGIES[strategy_name],
        _WORKER_PREPARED[key],
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _result_key(seed, num_ssu, strategy_name):
    return f"{seed}:{num_ssu}:{strategy_name}"


def _ordered_results(rows, runtime, selected):
    return [
        rows[_result_key(seed, num_ssu, strategy.name)]
        for seed in runtime["seeds"]
        for num_ssu in runtime["ssu_list"]
        for strategy in selected
        if _result_key(seed, num_ssu, strategy.name) in rows
    ]


def _validate_paired_inputs(results, runtime, selected):
    expected_strategies = {strategy.name for strategy in selected}
    for seed in runtime["seeds"]:
        for num_ssu in runtime["ssu_list"]:
            group = [
                row
                for row in results
                if row["seed"] == seed and row["num_ssu"] == num_ssu
            ]
            assert {row["strategy"] for row in group} == expected_strategies
            assert len(
                {
                    (row["workload_fingerprint"], row["placement_hash"])
                    for row in group
                }
            ) == 1


def run_matrix(result_path, *, workers, rerun=False):
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    runtime = runtime_config()
    selected = strategy_specs()
    experiment = experiment_spec(table, runtime)
    rows = {}
    if result_path.exists() and not rerun:
        cached = json.loads(result_path.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                _result_key(row["seed"], row["num_ssu"], row["strategy"]): row
                for row in cached["results"]
            }

    def checkpoint():
        ordered = _ordered_results(rows, runtime, selected)
        _write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered)
                == len(SEEDS) * len(SSU_LIST) * len(selected),
                "experiment": experiment,
                "selected_strategies": [row.name for row in selected],
                "results": ordered,
            },
        )

    pending = [
        (seed, num_ssu, strategy.name)
        for seed in runtime["seeds"]
        for num_ssu in runtime["ssu_list"]
        for strategy in selected
        if _result_key(seed, num_ssu, strategy.name) not in rows
    ]

    if pending and workers == 1:
        prepared_cache = {}
        for seed, num_ssu, strategy_name in pending:
            key = (seed, num_ssu)
            if key not in prepared_cache:
                prepared_cache[key] = prepare(table, runtime, seed, num_ssu)
            strategy = next(
                row for row in selected if row.name == strategy_name
            )
            result = run_strategy_case(
                table,
                runtime,
                seed,
                num_ssu,
                strategy,
                prepared_cache[key],
            )
            rows[_result_key(seed, num_ssu, strategy_name)] = result
            checkpoint()
            _print_completed(result)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, workers), len(pending)),
            initializer=_init_worker,
            initargs=(table, runtime, selected),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                rows[
                    _result_key(
                        result["seed"], result["num_ssu"], result["strategy"]
                    )
                ] = result
                checkpoint()
                _print_completed(result)

    ordered = _ordered_results(rows, runtime, selected)
    _validate_paired_inputs(ordered, runtime, selected)
    checkpoint()
    return json.loads(result_path.read_text())


def _print_completed(row):
    summary = row["summary"]
    fraction = summary.get(
        "avg_request_compute_fraction",
        summary.get("avg_request_compute_fraction_upper_bound"),
    )
    print(
        "seed=%d SSU=%d %-27s request=%7.3f%% wall=%6.1fs"
        % (
            row["seed"],
            row["num_ssu"],
            row["strategy"],
            100.0 * fraction,
            row["wall_time_s"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers", type=int, default=min(12, os.cpu_count() or 1)
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    data = run_matrix(
        args.output_dir / "results.json",
        workers=args.workers,
        rerun=args.rerun,
    )
    print("complete:", data["complete"])
    print("rows:", len(data["results"]))


if __name__ == "__main__":
    main()
