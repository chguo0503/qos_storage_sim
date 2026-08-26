"""Run the final finite-issue baseline and Path-routing comparison.

Every realizable strategy uses the same optimized static QoS hardware and
differs only in the Path IDs chosen by the NPU:

* all I/Os use Path 0 (the final baseline);
* category-legal Paths are selected by stateless round-robin;
* once for each ``(request, layer, SSU)`` submission state;
* once every eight I/Os; or
* once for every I/O.

All simulations submit one command every 0.1 us so other NPUs and device
events can interleave between consecutive commands.
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
from strategy_profiles import FINAL_STATIC


SCHEMA_VERSION = 3
NUM_NPU = 128
N_LAYERS = 16
LS_RATIO = 0.5
SEEDS = (42, 43)
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
ARRIVAL_DELAY_MAX_MS = 5.0
SSU_LIST = (8, 16, 28, 40, 56, 80, 112)
ISSUE_INTERVAL_US = 0.1
SUBMIT_BATCH_SIZE = 1
DEFAULT_OUTPUT_DIR = Path("results/routing_refresh_concurrency")
STATIC_PROFILE = FINAL_STATIC


def seed_bundle(seed):
    return {
        "workload": seed,
        "placement": seed + PLACEMENT_SEED_OFFSET,
        "submit_order": seed + SUBMIT_ORDER_SEED_OFFSET,
        "arrival_delay": seed + ARRIVAL_DELAY_SEED_OFFSET,
    }


def _table_fingerprint(table):
    payload = [
        [list(key), list(table[key])]
        for key in sorted(table)
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    path_selection_mode: str
    pressure_window_io: int | None = None

    @property
    def policy(self):
        return sim.POLICY_QOS_STATIC_CIR

    def client_config(self):
        return sim.ClientIOConfig(
            name=f"{self.name}_batch1_issue0p1us",
            pressure_window_io=self.pressure_window_io,
            submit_batch_size=SUBMIT_BATCH_SIZE,
            issue_interval_us=ISSUE_INTERVAL_US,
            path_selection_mode=self.path_selection_mode,
        )

    def config(self):
        result = {
            "name": self.name,
            "kind": "simulation",
            "policy": self.policy,
            "description": self.description,
        }
        client_config = self.client_config()
        result["client_io"] = asdict(client_config)
        result.update(
            {
                "profile_name": STATIC_PROFILE.name,
                "category_cir_gbps": list(
                    STATIC_PROFILE.category_cir_gbps
                ),
                "category_paths_per_group": list(
                    STATIC_PROFILE.category_paths_per_group
                ),
                "path_pir": "uncapped",
            }
        )
        return result


def strategy_specs():
    return (
        StrategySpec(
            "baseline",
            "Final baseline: every NPU sends every I/O to Path 0",
            sim.PATH_SELECTION_FIXED_PATH_ZERO,
        ),
        StrategySpec(
            "path_rr_baseline",
            "Zero-telemetry category-legal Path round-robin baseline",
            sim.PATH_SELECTION_STATELESS_RR,
        ),
        StrategySpec(
            "refresh1",
            "Read live Path pressure before every planned I/O",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            1,
        ),
        StrategySpec(
            "refresh8",
            "Read Path pressure once every eight planned I/Os",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            8,
        ),
        StrategySpec(
            "layer_once",
            "Read once per request-layer-SSU submission state",
            sim.PATH_SELECTION_PRESSURE_AWARE,
        ),
    )


def runtime_config():
    return {
        "mode": "formal",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "seeds": list(SEEDS),
        "seed_bundles": {str(seed): seed_bundle(seed) for seed in SEEDS},
        "seed_offsets": {
            "placement": PLACEMENT_SEED_OFFSET,
            "submit_order": SUBMIT_ORDER_SEED_OFFSET,
            "arrival_delay": ARRIVAL_DELAY_SEED_OFFSET,
        },
        "ls_ratio": LS_RATIO,
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": sim.DISK_BW,
        "npu_bw_limit_gbps": sim.NPU_BW_LIMIT,
        "client_submit_batch_size": SUBMIT_BATCH_SIZE,
        "client_issue_interval_us": ISSUE_INTERVAL_US,
    }


def _code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "advanced_policies.py",
        "experiment.py",
        "strategy_profiles.py",
        "routing_refresh_concurrency_experiment.py",
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
            "static_profile": STATIC_PROFILE.name,
            "static_category_cir_gbps": list(
                STATIC_PROFILE.category_cir_gbps
            ),
            "static_path_allocation_per_group": list(
                STATIC_PROFILE.category_paths_per_group
            ),
            "shared_simulation_policy": sim.POLICY_QOS_STATIC_CIR,
            "isolated_difference": "NPU Path-ID selection",
            "final_baseline_path": 0,
        },
        "backend": {
            "model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
            "ssd_service": "io_size_gb / 40 GB/s",
            "ssd_max_active_io": 1,
            "npu_service": "io_size_gb / 50 GB/s",
            "npu_max_active_io": 1,
            "block_visible_after": "npu_link_completion",
        },
        "available_strategies": [
            strategy.config() for strategy in strategy_specs()
        ],
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


def run_strategy_case(table, runtime, seed, num_ssu, strategy, prepared=None):
    started = time.perf_counter()
    if prepared is None:
        prepared = prepare(table, runtime, seed, num_ssu)
    seeds = runtime["seed_bundles"][str(seed)]
    _, full = sim.simulate_continuous(
        table,
        policy=strategy.policy,
        num_npu=runtime["num_npu"],
        num_disk=num_ssu,
        n_layers=runtime["n_layers"],
        ls_ratio=runtime["ls_ratio"],
        qos_config=STATIC_PROFILE.hardware_config(),
        client_io_config=strategy.client_config(),
        submit_order_seed=seeds["submit_order"],
        prepared_inputs=prepared,
    )
    compact = _summary(full)
    request_metrics = compact.pop("request_metrics")
    compact["client_submit_batch_size"] = full["client_submit_batch_size"]
    compact["client_submit_interval_us"] = full["client_submit_interval_us"]
    compact["pressure_read_interval"] = full["pressure_read_interval"]
    compact["path_selection"] = full["path_selection"]
    compact["qos_client_routing"] = full["qos_client_routing"]
    compact["client_submission"] = full["client_submission"]

    assert all(compact["invariants"].values())
    assert compact["workload_fingerprint"] == prepared.workload_hash
    assert compact["placement_hash"] == prepared.placement_hash
    assert compact["client_submit_batch_size"] == SUBMIT_BATCH_SIZE
    assert compact["client_submit_interval_us"] == ISSUE_INTERVAL_US
    expected_reported_path_selection = (
        "per_io"
        if strategy.path_selection_mode == sim.PATH_SELECTION_PRESSURE_AWARE
        else strategy.path_selection_mode
    )
    assert compact["path_selection"] == expected_reported_path_selection
    if strategy.path_selection_mode == sim.PATH_SELECTION_PRESSURE_AWARE:
        assert compact["pressure_read_interval"] == strategy.pressure_window_io
        assert compact["pressure_reports"] > 0
    else:
        assert compact["pressure_read_interval"] is None
        assert compact["pressure_reports"] == 0
    if strategy.path_selection_mode == sim.PATH_SELECTION_FIXED_PATH_ZERO:
        assert compact["enqueued_path_ids"] == [0]

    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "kind": "simulation",
        "config": strategy.config(),
        "seeds": seeds,
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": compact,
        "request_metrics": request_metrics,
        "wall_time_s": time.perf_counter() - started,
    }


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
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
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
    expected = {strategy.name for strategy in selected}
    for seed in runtime["seeds"]:
        for num_ssu in runtime["ssu_list"]:
            group = [
                row
                for row in results
                if row["seed"] == seed and row["num_ssu"] == num_ssu
            ]
            assert {row["strategy"] for row in group} == expected
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
                "selected_strategies": [
                    strategy.name for strategy in selected
                ],
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
                item for item in selected if item.name == strategy_name
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
                        result["seed"],
                        result["num_ssu"],
                        result["strategy"],
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
    print(
        "seed=%d SSU=%3d %-26s request=%7.3f%% fleet=%7.3f%% reads=%7d wall=%6.1fs"
        % (
            row["seed"],
            row["num_ssu"],
            row["strategy"],
            100.0 * summary["avg_request_compute_fraction"],
            100.0 * summary["fleet_npu_compute_utilization"],
            summary["pressure_reports"],
            row["wall_time_s"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 1)
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run_matrix(
        args.output_dir / "results.json",
        workers=args.workers,
        rerun=args.rerun,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
