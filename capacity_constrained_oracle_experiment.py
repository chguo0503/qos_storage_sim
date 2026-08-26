"""Run the physical-capacity-preserving oracle candidate matrix.

This experiment deliberately does not claim an exact optimum.  It runs a
clairvoyant, per-SSD demand-weighted shortest-layer-work scheduler on the same
discrete SSD40 -> NPU50 event model as the routing experiment.  The analysis
combines this candidate with every already measured runnable strategy and
plots their pointwise best feasible envelope.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import sim
from experiment import _summary
import routing_refresh_concurrency_experiment as routing


SCHEMA_VERSION = 1
STRATEGY = "demand_weighted_sjf_oracle_candidate"
PRIORITY_DEMAND_EXPONENT = 0.25
DEFAULT_OUTPUT_PATH = (
    routing.DEFAULT_OUTPUT_DIR / "capacity_constrained_oracle_results.json"
)


def oracle_priority_key(flow):
    """Prefer short visible layer work, with a mild demand urgency weight."""
    demand = max(flow.demand_gbps, 1e-12)
    weighted_work = flow.layer_work_gb / demand**PRIORITY_DEMAND_EXPONENT
    return (
        weighted_work,
        flow.deadline_time,
        flow.layer_work_gb,
        flow.enqueue_time,
        flow.request_id,
        flow.layer,
        flow.block_idx,
        flow.disk_id,
    )


def oracle_client_config():
    return sim.ClientIOConfig(
        name="capacity_oracle_batch1_issue0p1us",
        pressure_window_io=None,
        submit_batch_size=routing.SUBMIT_BATCH_SIZE,
        issue_interval_us=routing.ISSUE_INTERVAL_US,
    )


def experiment_spec(table, runtime):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "capacity_constrained_oracle_experiment.py",
        "routing_refresh_concurrency_experiment.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": digest.hexdigest(),
        "data_fingerprint": routing._table_fingerprint(table),
        "runtime": runtime,
        "strategy": STRATEGY,
        "objective": "candidate_for_mean_per_request_compute_fraction",
        "priority": {
            "name": "demand_weighted_shortest_visible_layer_work",
            "demand_exponent": PRIORITY_DEMAND_EXPONENT,
        },
        "information": (
            "all metadata of currently released layer work; future layers "
            "remain unavailable until the original prefetch release event"
        ),
        "physical_constraints": {
            "placement": "original immutable block_to_ssd mapping",
            "ssd": "one nonpreemptive command at 40 GB/s per SSD",
            "npu_link": "one FCFS command at 50 GB/s per NPU",
            "visibility": "after NPU-link completion",
            "arrivals": "original 0-5 ms vector",
            "layer_release": "original one-layer-ahead prefetch dependency",
            "client_issue": "batch 1, 0.1 us between commands",
        },
        "optimality": {
            "exact_optimum_proven": False,
            "interpretation": "feasible_candidate_for_oracle_envelope",
        },
    }


def run_case(table, runtime, seed, num_ssu, prepared=None):
    started = time.perf_counter()
    if prepared is None:
        prepared = routing.prepare(table, runtime, seed, num_ssu)
    seeds = runtime["seed_bundles"][str(seed)]
    original_priority = sim.omniscient_edf_key
    sim.omniscient_edf_key = oracle_priority_key
    try:
        _, full = sim.simulate_continuous(
            table,
            policy=sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
            num_npu=runtime["num_npu"],
            num_disk=num_ssu,
            n_layers=runtime["n_layers"],
            ls_ratio=runtime["ls_ratio"],
            client_io_config=oracle_client_config(),
            submit_order_seed=seeds["submit_order"],
            prepared_inputs=prepared,
        )
    finally:
        sim.omniscient_edf_key = original_priority

    compact = _summary(full)
    request_metrics = compact.pop("request_metrics")
    compact["client_submit_batch_size"] = full["client_submit_batch_size"]
    compact["client_submit_interval_us"] = full["client_submit_interval_us"]
    compact["oracle_candidate"] = STRATEGY
    compact["exact_optimum_proven"] = False
    assert all(compact["invariants"].values())
    assert compact["workload_fingerprint"] == prepared.workload_hash
    assert compact["placement_hash"] == prepared.placement_hash
    assert compact["block_conservation"]["placement_targets_preserved"]
    assert compact["client_submit_batch_size"] == routing.SUBMIT_BATCH_SIZE
    assert compact["client_submit_interval_us"] == routing.ISSUE_INTERVAL_US
    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "strategy": STRATEGY,
        "kind": "feasible_oracle_candidate",
        "seeds": seeds,
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": compact,
        "request_metrics": request_metrics,
        "wall_time_s": time.perf_counter() - started,
    }


_WORKER_TABLE = None
_WORKER_RUNTIME = None


def _init_worker(table, runtime):
    global _WORKER_TABLE, _WORKER_RUNTIME
    _WORKER_TABLE = table
    _WORKER_RUNTIME = runtime


def _worker(task):
    seed, num_ssu = task
    prepared = routing.prepare(_WORKER_TABLE, _WORKER_RUNTIME, seed, num_ssu)
    return run_case(
        _WORKER_TABLE,
        _WORKER_RUNTIME,
        seed,
        num_ssu,
        prepared,
    )


def _key(seed, num_ssu):
    return f"{seed}:{num_ssu}"


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


def run_matrix(output_path, *, workers):
    table = sim.load_bw_table_cache(num_npu=routing.NUM_NPU)
    runtime = routing.runtime_config()
    experiment = experiment_spec(table, runtime)
    rows = {}
    if output_path.exists():
        cached = json.loads(output_path.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                _key(row["seed"], row["num_ssu"]): row
                for row in cached["results"]
            }

    tasks = [
        (seed, num_ssu)
        for seed in runtime["seeds"]
        for num_ssu in runtime["ssu_list"]
        if _key(seed, num_ssu) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[_key(seed, num_ssu)]
            for seed in runtime["seeds"]
            for num_ssu in runtime["ssu_list"]
            if _key(seed, num_ssu) in rows
        ]
        _write_json(
            output_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered)
                == len(runtime["seeds"]) * len(runtime["ssu_list"]),
                "experiment": experiment,
                "results": ordered,
            },
        )

    if tasks and workers == 1:
        for seed, num_ssu in tasks:
            prepared = routing.prepare(table, runtime, seed, num_ssu)
            row = run_case(table, runtime, seed, num_ssu, prepared)
            rows[_key(seed, num_ssu)] = row
            checkpoint()
            print_completed(row)
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=min(max(1, workers), len(tasks)),
            initializer=_init_worker,
            initargs=(table, runtime),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows[_key(row["seed"], row["num_ssu"])] = row
                checkpoint()
                print_completed(row)
    checkpoint()
    return json.loads(output_path.read_text())


def print_completed(row):
    summary = row["summary"]
    print(
        "seed=%d SSU=%3d oracle_candidate request=%7.3f%% fleet=%7.3f%% wall=%6.1fs"
        % (
            row["seed"],
            row["num_ssu"],
            100.0 * summary["avg_request_compute_fraction"],
            100.0 * summary["fleet_npu_compute_utilization"],
            row["wall_time_s"],
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    result = run_matrix(args.output, workers=args.workers)
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
