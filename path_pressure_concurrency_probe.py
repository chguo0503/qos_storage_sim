"""Paired audit of client submission atomicity and Path-pressure telemetry.

The formal matrix keeps the requested 128 NPU / 16 layer / 40,56,80 SSU
workload.  It changes only two client-side dimensions:

* ``submit_batch_size`` is 8 (the historical model) or 1.
* ``issue_interval_us`` is 0 (the historical instantaneous burst) or 0.1 us.
  A positive interval lets independently arriving NPUs and SSD events interleave
  between commands; batch-size 1 alone cannot do that when ready times are unique.
* routing reads Path pressure every 8 I/Os, every I/O, or never.  The no-read
  policy round-robins over the same category-legal Path pool.
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
from strategy_profiles import CURRENT_STATIC


SCHEMA_VERSION = 1
NUM_NPU = 128
N_LAYERS = 16
SSU_LIST = (40, 56, 80)
LS_RATIO = 0.5
ARRIVAL_DELAY_MAX_MS = 5.0
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
DEFAULT_OUTPUT_DIR = Path("results/path_pressure_concurrency")


@dataclass(frozen=True)
class ProbeStrategy:
    name: str
    policy: str
    pressure_mode: str
    pressure_window_io: int | None
    submit_batch_size: int
    issue_interval_us: float

    def client_config(self):
        return sim.ClientIOConfig(
            self.name,
            self.pressure_window_io,
            self.submit_batch_size,
            issue_interval_us=self.issue_interval_us,
        )


def strategies():
    return (
        ProbeStrategy(
            "baseline_atomic8", sim.POLICY_BASELINE_BYPASS, "none", None, 8, 0.0
        ),
        ProbeStrategy(
            "no_pressure_rr_atomic8",
            sim.POLICY_QOS_STATIC_CIR,
            "no_pressure_round_robin",
            None,
            8,
            0.0,
        ),
        ProbeStrategy(
            "refresh8_atomic8",
            sim.POLICY_QOS_STATIC_CIR,
            "refresh8_shadow",
            8,
            8,
            0.0,
        ),
        ProbeStrategy(
            "per_io_atomic8",
            sim.POLICY_QOS_STATIC_CIR,
            "per_io_live",
            1,
            8,
            0.0,
        ),
        # This pair isolates batch size.  With unique ready times and zero issue
        # cost it is not expected to create realistic cross-NPU overlap.
        ProbeStrategy(
            "refresh8_batch1_zero",
            sim.POLICY_QOS_STATIC_CIR,
            "refresh8_shadow",
            8,
            1,
            0.0,
        ),
        ProbeStrategy(
            "per_io_batch1_zero",
            sim.POLICY_QOS_STATIC_CIR,
            "per_io_live",
            1,
            1,
            0.0,
        ),
        ProbeStrategy(
            "baseline_issue01us",
            sim.POLICY_BASELINE_BYPASS,
            "none",
            None,
            1,
            0.1,
        ),
        ProbeStrategy(
            "no_pressure_rr_issue01us",
            sim.POLICY_QOS_STATIC_CIR,
            "no_pressure_round_robin",
            None,
            1,
            0.1,
        ),
        ProbeStrategy(
            "refresh8_issue01us",
            sim.POLICY_QOS_STATIC_CIR,
            "refresh8_shadow",
            8,
            1,
            0.1,
        ),
        ProbeStrategy(
            "per_io_issue01us",
            sim.POLICY_QOS_STATIC_CIR,
            "per_io_live",
            1,
            1,
            0.1,
        ),
    )


def seed_bundle(seed):
    return {
        "workload": int(seed),
        "placement": int(seed) + PLACEMENT_SEED_OFFSET,
        "submit_order": int(seed) + SUBMIT_ORDER_SEED_OFFSET,
        "arrival_delay": int(seed) + ARRIVAL_DELAY_SEED_OFFSET,
    }


def runtime(seed):
    return {
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "ls_ratio": LS_RATIO,
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "seeds": seed_bundle(seed),
    }


def _code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "sim.py",
        "experiment.py",
        "strategy_profiles.py",
        "path_pressure_concurrency_probe.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def experiment_spec(run_config, selected):
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "runtime": run_config,
        "static_qos_profile": {
            "name": CURRENT_STATIC.name,
            "category_cir_gbps": list(CURRENT_STATIC.category_cir_gbps),
            "category_paths_per_group": list(
                CURRENT_STATIC.category_paths_per_group
            ),
        },
        "backend": {
            "ssd": "one nonpreemptive command, size/40GBps",
            "npu": "one FCFS receive command per NPU, size/50GBps",
            "visible_after": "NPU receive completion",
        },
        "selected_strategies": [asdict(strategy) for strategy in selected],
    }


def prepare(table, run_config, num_ssu):
    seeds = run_config["seeds"]
    return sim.prepare_simulation_inputs(
        table,
        total_requests=run_config["num_npu"],
        n_layers=run_config["n_layers"],
        num_disk=num_ssu,
        ls_ratio=run_config["ls_ratio"],
        workload_seed=seeds["workload"],
        placement_seed=seeds["placement"],
        arrival_delay_seed=seeds["arrival_delay"],
        arrival_delay_max_ms=run_config["arrival_delay_ms"][1],
    )


def _plan_no_pressure_round_robin(context, state, current_time):
    del context, current_time
    window_start = len(state.planned_path_ids)
    allowed = state.allowed_path_ids
    state.planned_path_ids.extend(
        allowed[(state.start_offset + index) % len(allowed)]
        for index in range(window_start, len(state.blocks))
    )


def simulate_case(table, run_config, num_ssu, strategy):
    prepared = prepare(table, run_config, num_ssu)
    original_planner = sim._plan_qos_pressure_window
    if strategy.pressure_mode == "no_pressure_round_robin":
        sim._plan_qos_pressure_window = _plan_no_pressure_round_robin
    try:
        _, full = sim.simulate_continuous(
            table,
            policy=strategy.policy,
            num_npu=run_config["num_npu"],
            num_disk=num_ssu,
            n_layers=run_config["n_layers"],
            ls_ratio=run_config["ls_ratio"],
            qos_config=(
                CURRENT_STATIC.hardware_config()
                if strategy.policy == sim.POLICY_QOS_STATIC_CIR
                else None
            ),
            client_io_config=strategy.client_config(),
            submit_order_seed=run_config["seeds"]["submit_order"],
            prepared_inputs=prepared,
        )
    finally:
        sim._plan_qos_pressure_window = original_planner

    compact = _summary(full)
    request_metrics = compact.pop("request_metrics")
    compact["client_submission"] = full["client_submission"]
    assert all(compact["invariants"].values())
    if strategy.pressure_mode in ("none", "no_pressure_round_robin"):
        assert compact["pressure_reports"] == 0
    return prepared, compact, request_metrics


def run_case(table, run_config, num_ssu, strategy):
    started = time.perf_counter()
    prepared, compact, request_metrics = simulate_case(
        table, run_config, num_ssu, strategy
    )
    return {
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "config": asdict(strategy),
        "seeds": run_config["seeds"],
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": compact,
        "request_metrics": request_metrics,
        "wall_time_s": time.perf_counter() - started,
    }


_TABLE = None
_RUNTIME = None
_STRATEGIES = None


def _init_worker(table, run_config, selected):
    global _TABLE, _RUNTIME, _STRATEGIES
    _TABLE = table
    _RUNTIME = run_config
    _STRATEGIES = {strategy.name: strategy for strategy in selected}


def _worker(task):
    num_ssu, name = task
    return run_case(_TABLE, _RUNTIME, num_ssu, _STRATEGIES[name])


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run(output_path, *, seed, workers, selection="all", rerun=False):
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    available = strategies()
    names = (
        {name.strip() for name in selection.split(",")}
        if selection != "all"
        else {strategy.name for strategy in available}
    )
    selected = tuple(strategy for strategy in available if strategy.name in names)
    run_config = runtime(seed)
    spec = experiment_spec(run_config, selected)
    rows = {}
    if output_path.exists() and not rerun:
        cached = json.loads(output_path.read_text())
        if cached.get("experiment") == spec:
            rows = {
                (row["num_ssu"], row["strategy"]): row
                for row in cached["results"]
            }
    tasks = [
        (num_ssu, strategy.name)
        for num_ssu in SSU_LIST
        for strategy in selected
        if (num_ssu, strategy.name) not in rows
    ]

    def checkpoint():
        ordered = [
            rows[(num_ssu, strategy.name)]
            for num_ssu in SSU_LIST
            for strategy in selected
            if (num_ssu, strategy.name) in rows
        ]
        _write_json(
            output_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": len(ordered) == len(SSU_LIST) * len(selected),
                "experiment": spec,
                "selected_strategies": [strategy.name for strategy in selected],
                "results": ordered,
            },
        )

    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(max(1, workers), len(tasks)),
            initializer=_init_worker,
            initargs=(table, run_config, selected),
        ) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows[(row["num_ssu"], row["strategy"])] = row
                checkpoint()
                print(
                    "SSU=%d %-24s request=%7.3f%% fleet=%7.3f%% reads=%d wall=%6.1fs"
                    % (
                        row["num_ssu"],
                        row["strategy"],
                        100.0 * row["summary"]["avg_request_compute_fraction"],
                        100.0 * row["summary"]["fleet_npu_compute_utilization"],
                        row["summary"]["pressure_reports"],
                        row["wall_time_s"],
                    ),
                    flush=True,
                )
    checkpoint()
    return json.loads(output_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--select", default="all")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run(
        args.output_dir / f"results_seed{args.seed}.json",
        seed=args.seed,
        workers=args.workers,
        selection=args.select,
        rerun=args.rerun,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
