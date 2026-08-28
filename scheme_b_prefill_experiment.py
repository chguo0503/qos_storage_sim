"""Run one-shot Scheme B on the retained prefill experiment matrix.

Scheme B knows the admitted batch and immutable ring placement before the
prefill starts.  It computes one demand-capped max-min ``NPU x SSU`` grant,
writes that grant into one dedicated Path per NPU on every SSU, and reuses the
same configuration for all 16 layers.  The SSD40 -> NPU50 data plane is the
same discrete-event implementation used by the retained routing strategies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from experiment import compact_summary
import routing_refresh_concurrency_experiment as routing
import scheme_b_prefill
import sim


SCHEMA_VERSION = 1
STRATEGY = "scheme_b_prefill"
DEFAULT_OUTPUT_PATH = routing.DEFAULT_OUTPUT_DIR / "scheme_b_prefill_results.json"


def client_config():
    return sim.ClientIOConfig(
        name="scheme_b_prefill_config_once",
        pressure_window_io=None,
        submit_batch_size=routing.SUBMIT_BATCH_SIZE,
        issue_interval_us=routing.ISSUE_INTERVAL_US,
        path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
    )


def experiment_spec(table, runtime):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "experiment.py",
        "continuous_batch_control.py",
        "scheme_b_prefill.py",
        "scheme_b_prefill_experiment.py",
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
        "admission": "all 128 one-shot prefill requests known before time zero",
        "control": {
            "demand": "one-layer ring-placement demand vector per NPU",
            "allocation": "demand-capped max-min NPU_x_SSU grant",
            "configuration_writes": 1,
            "configuration_lifetime": "all 16 prefill layers",
            "path_mapping": "one dedicated hardware Path per NPU on every SSU",
            "pressure_reads": 0,
        },
        "physical_constraints": {
            "placement": "immutable block ring hash, reused by every layer",
            "ssd": "one nonpreemptive command at 40 GB/s per SSD",
            "npu_link": "one FCFS command at 50 GB/s per NPU",
            "visibility": "after NPU-link completion",
            "arrivals": "original paired 0-5 ms vector",
            "client_issue": "batch 1, 0.1 us between commands",
        },
    }


def _grant_metadata(plan):
    configs = plan.qos_configs
    paths = plan.path_by_npu
    per_ssu = [sum(config.path_cirs) for config in configs]
    per_npu = [
        sum(config.path_cirs[path_id] for config in configs)
        for path_id in paths
    ]
    assert len(paths) == plan.num_npu
    assert len(set(paths)) == len(paths)
    assert all(0 <= path_id < 256 for path_id in paths)
    assert all(value <= sim.DISK_BW + 1e-9 for value in per_ssu)
    assert all(value <= sim.NPU_BW_LIMIT + 1e-9 for value in per_npu)
    metadata = plan.summary()
    assert metadata["all_constraints_hold"]
    metadata.update(
        {
            "active_grants": sum(
                config.path_cirs[path_id] > 0.0
                for config in configs
                for path_id in paths
            ),
            "max_ssu_grant_gbps": max(per_ssu, default=0.0),
            "max_npu_grant_gbps": max(per_npu, default=0.0),
            "path_id_by_npu": list(paths),
        }
    )
    return metadata


def run_case(table, runtime, seed, num_ssu, prepared=None):
    started = time.perf_counter()
    prepared = prepared or routing.prepare(table, runtime, seed, num_ssu)
    seeds = runtime["seed_bundles"][str(seed)]
    plan = scheme_b_prefill.build_scheme_b_prefill_plan(
        prepared,
        ssd_cap_gbps=sim.DISK_BW,
        npu_cap_gbps=sim.NPU_BW_LIMIT,
    )
    assert plan.num_npu == runtime["num_npu"]
    assert plan.num_ssu == num_ssu
    grant_metadata = _grant_metadata(plan)
    _, full = sim.simulate_continuous(
        table,
        policy=sim.POLICY_QOS_STATIC_CIR,
        num_npu=runtime["num_npu"],
        num_disk=num_ssu,
        n_layers=runtime["n_layers"],
        ls_ratio=runtime["ls_ratio"],
        qos_configs_by_disk=plan.qos_configs,
        npu_dedicated_paths=plan.path_by_npu,
        client_io_config=client_config(),
        submit_order_seed=seeds["submit_order"],
        prepared_inputs=prepared,
    )
    summary = compact_summary(full)
    request_metrics = summary.pop("request_metrics")
    summary.update(
        {
            "placement_mode": full["placement_mode"],
            "placement_ring_virtual_nodes": full["placement_ring_virtual_nodes"],
            "placement_ring_hash_version": full["placement_ring_hash_version"],
            "client_submit_batch_size": full["client_submit_batch_size"],
            "client_submit_interval_us": full["client_submit_interval_us"],
            "pressure_read_interval": full["pressure_read_interval"],
            "path_selection": full["path_selection"],
            "path_pool_mode": full["path_pool_mode"],
            "qos_client_routing": full["qos_client_routing"],
            "scheme_b_grant": grant_metadata,
        }
    )
    assert all(summary["invariants"].values())
    assert summary["workload_fingerprint"] == prepared.workload_hash
    assert summary["placement_hash"] == prepared.placement_hash
    assert summary["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
    assert summary["pressure_reports"] == 0
    assert summary["pressure_read_interval"] is None
    assert summary["path_pool_mode"] == "npu_dedicated"
    assert set(summary["enqueued_path_ids"]) <= set(plan.path_by_npu)
    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "strategy": STRATEGY,
        "kind": "simulation",
        "seeds": seeds,
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "scheme_b_plan_fingerprint": plan.fingerprint,
        "summary": summary,
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
    return run_case(_WORKER_TABLE, _WORKER_RUNTIME, seed, num_ssu, prepared)


def _key(seed, num_ssu):
    return f"{seed}:{num_ssu}"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def run_matrix(output_path, *, workers, rerun=False):
    table = sim.load_bw_table_cache(num_npu=routing.NUM_NPU)
    runtime = routing.runtime_config()
    experiment = experiment_spec(table, runtime)
    rows = {}
    if output_path.exists() and not rerun:
        cached = json.loads(output_path.read_text())
        if cached.get("experiment") == experiment:
            rows = {
                _key(row["seed"], row["num_ssu"]): row
                for row in cached["results"]
            }
    pending = [
        (seed, ssu)
        for seed in runtime["seeds"]
        for ssu in runtime["ssu_list"]
        if _key(seed, ssu) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[_key(seed, ssu)]
            for seed in runtime["seeds"]
            for ssu in runtime["ssu_list"]
            if _key(seed, ssu) in rows
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

    if pending and workers == 1:
        for seed, ssu in pending:
            prepared = routing.prepare(table, runtime, seed, ssu)
            row = run_case(table, runtime, seed, ssu, prepared)
            rows[_key(seed, ssu)] = row
            checkpoint()
            print_completed(row)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(max(1, workers), len(pending)),
            initializer=_init_worker,
            initargs=(table, runtime),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in pending}
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
        "seed=%d SSU=%3d scheme_b request=%7.3f%% fleet=%7.3f%% wall=%6.1fs"
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
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run_matrix(args.output, workers=args.workers, rerun=args.rerun)
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
