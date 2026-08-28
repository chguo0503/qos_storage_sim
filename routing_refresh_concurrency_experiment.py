"""Run the five retained Path-routing strategies on ring-hash placement."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

from experiment import compact_summary
import sim
from strategy_profiles import FINAL_STATIC

SCHEMA_VERSION = 5
NUM_NPU = 128
N_LAYERS = 16
LS_RATIO = 0.5
SEEDS = (42, 43)
SSU_LIST = (8, 16, 28, 40, 56, 80, 112)
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
ARRIVAL_DELAY_MAX_MS = 5.0
ISSUE_INTERVAL_US = 0.1
SUBMIT_BATCH_SIZE = 1
DEFAULT_OUTPUT_DIR = Path("results/routing_refresh_concurrency")
STATIC_PROFILE = FINAL_STATIC


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    path_selection_mode: str
    pressure_window_io: int | None

    def client_config(self):
        return sim.ClientIOConfig(
            name=f"{self.name}_batch1_issue0p1us",
            pressure_window_io=self.pressure_window_io,
            submit_batch_size=SUBMIT_BATCH_SIZE,
            issue_interval_us=ISSUE_INTERVAL_US,
            path_selection_mode=self.path_selection_mode,
        )

    def metadata(self):
        return {
            "name": self.name,
            "description": self.description,
            "policy": sim.POLICY_QOS_STATIC_CIR,
            "path_selection_mode": self.path_selection_mode,
            "pressure_window_io": self.pressure_window_io,
            "submit_batch_size": SUBMIT_BATCH_SIZE,
            "issue_interval_us": ISSUE_INTERVAL_US,
        }


def strategy_specs():
    return (
        StrategySpec(
            "baseline",
            "All I/Os use Path 0; no pressure reads",
            sim.PATH_SELECTION_FIXED_PATH_ZERO,
            None,
        ),
        StrategySpec(
            "path_rr",
            "Category-legal deterministic Path round-robin; no pressure reads",
            sim.PATH_SELECTION_STATELESS_RR,
            None,
        ),
        StrategySpec(
            "layer_once",
            "Read pressure once per request-layer-SSU submission state",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            None,
        ),
        StrategySpec(
            "refresh8",
            "Read pressure once every eight planned I/Os",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            8,
        ),
        StrategySpec(
            "refresh1",
            "Read live pressure before every planned I/O",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            1,
        ),
    )


def seed_bundle(seed):
    return {
        "workload": seed,
        "placement": seed + PLACEMENT_SEED_OFFSET,
        "submit_order": seed + SUBMIT_ORDER_SEED_OFFSET,
        "arrival_delay": seed + ARRIVAL_DELAY_SEED_OFFSET,
    }


def runtime_config():
    return {
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "seeds": list(SEEDS),
        "seed_bundles": {str(seed): seed_bundle(seed) for seed in SEEDS},
        "ls_ratio": LS_RATIO,
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "ring_virtual_nodes": sim.BLOCK_RING_VIRTUAL_NODES,
        "disk_bw_gbps": sim.DISK_BW,
        "npu_bw_limit_gbps": sim.NPU_BW_LIMIT,
        "client_submit_batch_size": SUBMIT_BATCH_SIZE,
        "client_issue_interval_us": ISSUE_INTERVAL_US,
    }


def _table_fingerprint(table):
    payload = [[list(key), list(table[key])] for key in sorted(table)]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def _code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "experiment.py",
        "strategy_profiles.py",
        "routing_refresh_concurrency_experiment.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def experiment_spec(table, runtime, selected):
    profile = STATIC_PROFILE
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "data_fingerprint": _table_fingerprint(table),
        "runtime": runtime,
        "placement": {
            "mode": sim.PLACEMENT_BLOCK_RING_HASH,
            "block_key": ["request_id", "block_index"],
            "virtual_node_key": ["disk_id", "virtual_node"],
            "virtual_nodes_per_ssu": sim.BLOCK_RING_VIRTUAL_NODES,
            "hash_version": "sha256_u64_pair_v1",
            "cross_layer_ssu_reuse": True,
        },
        "static_qos": {
            "profile": profile.name,
            "category_labels": list(sim.QOS_ROUTING_CATEGORIES),
            "category_cir_gbps": list(profile.category_cir_gbps),
            "category_paths_per_group": list(profile.category_paths_per_group),
        },
        "backend": {
            "ssd": "one nonpreemptive command at 40 GB/s",
            "npu_link": "one FCFS command at 50 GB/s per NPU",
            "block_visible_after": "npu_link_completion",
        },
        "strategies": [strategy.metadata() for strategy in selected],
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
        placement_mode=sim.PLACEMENT_BLOCK_RING_HASH,
    )


def run_strategy_case(table, runtime, seed, num_ssu, strategy, prepared=None):
    started = time.perf_counter()
    prepared = prepared or prepare(table, runtime, seed, num_ssu)
    seeds = runtime["seed_bundles"][str(seed)]
    _, full = sim.simulate_continuous(
        table,
        policy=sim.POLICY_QOS_STATIC_CIR,
        num_npu=runtime["num_npu"],
        num_disk=num_ssu,
        n_layers=runtime["n_layers"],
        ls_ratio=runtime["ls_ratio"],
        qos_config=STATIC_PROFILE.hardware_config(),
        client_io_config=strategy.client_config(),
        submit_order_seed=seeds["submit_order"],
        prepared_inputs=prepared,
    )
    summary = compact_summary(full)
    request_metrics = summary.pop("request_metrics")
    summary.update(
        {
            "placement_mode": full["placement_mode"],
            "placement_ring_virtual_nodes": full[
                "placement_ring_virtual_nodes"
            ],
            "placement_ring_hash_version": full[
                "placement_ring_hash_version"
            ],
            "client_submit_batch_size": full["client_submit_batch_size"],
            "client_submit_interval_us": full["client_submit_interval_us"],
            "pressure_read_interval": full["pressure_read_interval"],
            "path_selection": full["path_selection"],
            "qos_client_routing": full["qos_client_routing"],
        }
    )
    assert all(summary["invariants"].values())
    assert summary["workload_fingerprint"] == prepared.workload_hash
    assert summary["placement_hash"] == prepared.placement_hash
    assert summary["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
    if strategy.path_selection_mode == sim.PATH_SELECTION_PRESSURE_AWARE:
        assert summary["pressure_reports"] > 0
        assert summary["pressure_read_interval"] == strategy.pressure_window_io
    else:
        assert summary["pressure_reports"] == 0
        assert summary["pressure_read_interval"] is None
    if strategy.path_selection_mode == sim.PATH_SELECTION_FIXED_PATH_ZERO:
        assert summary["enqueued_path_ids"] == [0]
    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "kind": "simulation",
        "config": strategy.metadata(),
        "seeds": seeds,
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": summary,
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
    pair = (seed, num_ssu)
    if pair not in _WORKER_PREPARED:
        _WORKER_PREPARED[pair] = prepare(
            _WORKER_TABLE, _WORKER_RUNTIME, seed, num_ssu
        )
    return run_strategy_case(
        _WORKER_TABLE,
        _WORKER_RUNTIME,
        seed,
        num_ssu,
        _WORKER_STRATEGIES[strategy_name],
        _WORKER_PREPARED[pair],
    )


def _key(seed, num_ssu, strategy):
    return f"{seed}:{num_ssu}:{strategy}"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _ordered(rows, runtime, selected):
    return [
        rows[_key(seed, ssu, strategy.name)]
        for seed in runtime["seeds"]
        for ssu in runtime["ssu_list"]
        for strategy in selected
        if _key(seed, ssu, strategy.name) in rows
    ]


def _validate_pairing(rows, runtime, selected):
    expected = {strategy.name for strategy in selected}
    for seed in runtime["seeds"]:
        for ssu in runtime["ssu_list"]:
            group = [
                row for row in rows
                if row["seed"] == seed and row["num_ssu"] == ssu
            ]
            assert {row["strategy"] for row in group} == expected
            assert len(
                {
                    (row["workload_fingerprint"], row["placement_hash"])
                    for row in group
                }
            ) == 1


def run_matrix(result_path, *, workers, rerun=False, strategy_names=None):
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    runtime = runtime_config()
    all_strategies = strategy_specs()
    if strategy_names is None:
        selected = all_strategies
    else:
        by_name = {strategy.name: strategy for strategy in all_strategies}
        selected = tuple(by_name[name] for name in strategy_names)
    experiment = experiment_spec(table, runtime, selected)
    rows = {}
    if result_path.exists() and not rerun:
        cached = json.loads(result_path.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                _key(row["seed"], row["num_ssu"], row["strategy"]): row
                for row in cached["results"]
            }

    pending = [
        (seed, ssu, strategy.name)
        for seed in runtime["seeds"]
        for ssu in runtime["ssu_list"]
        for strategy in selected
        if _key(seed, ssu, strategy.name) not in rows
    ]

    def checkpoint():
        ordered = _ordered(rows, runtime, selected)
        _write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered)
                == len(runtime["seeds"])
                * len(runtime["ssu_list"])
                * len(selected),
                "experiment": experiment,
                "selected_strategies": [s.name for s in selected],
                "results": ordered,
            },
        )

    if pending and workers == 1:
        prepared_cache = {}
        by_name = {strategy.name: strategy for strategy in selected}
        for seed, ssu, name in pending:
            prepared_cache.setdefault(
                (seed, ssu), prepare(table, runtime, seed, ssu)
            )
            row = run_strategy_case(
                table, runtime, seed, ssu, by_name[name], prepared_cache[(seed, ssu)]
            )
            rows[_key(seed, ssu, name)] = row
            checkpoint()
            _print_row(row)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, workers), len(pending)),
            initializer=_init_worker,
            initargs=(table, runtime, selected),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in pending}
            for future in as_completed(futures):
                row = future.result()
                rows[_key(row["seed"], row["num_ssu"], row["strategy"])] = row
                checkpoint()
                _print_row(row)

    ordered = _ordered(rows, runtime, selected)
    _validate_pairing(ordered, runtime, selected)
    checkpoint()
    return json.loads(result_path.read_text())


def _print_row(row):
    summary = row["summary"]
    print(
        "seed=%d SSU=%3d %-10s request=%7.3f%% fleet=%7.3f%% reads=%7d wall=%6.1fs"
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[strategy.name for strategy in strategy_specs()],
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run_matrix(
        args.output or args.output_dir / "results.json",
        workers=args.workers,
        rerun=args.rerun,
        strategy_names=args.strategies,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
