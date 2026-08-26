"""Run the paired joint Path-routing and dynamic-CIR experiment matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Optional

from experiment import _summary
from sim import (
    ARRIVAL_DELAY_MAX_MS,
    DISK_BW,
    DYNAMIC_CIR_DEMAND_PROPORTIONAL,
    DYNAMIC_CIR_SLACK_LINK_GUARDED,
    DYNAMIC_ROUTE_FIXED,
    DYNAMIC_ROUTE_LEAST_WORK,
    DynamicCIRPolicyConfig,
    NPU_BW_LIMIT,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_DEMAND_MAXMIN,
    POLICY_QOS_DYNAMIC_JOINT_CIR,
    POLICY_QOS_STATIC_CIR,
    load_bw_table_cache,
    prepare_simulation_inputs,
    simulate_continuous,
)
from strategy_profiles import CLIENT_VARIANTS, CURRENT_STATIC, STATIC_PROFILES


SCHEMA_VERSION = 1
NUM_NPU = 128
N_LAYERS = 16
SSU_LIST = (40, 56, 80)
LS_RATIO = 0.5
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
DEFAULT_OUTPUT_DIR = Path("results/joint_dynamic_cir")


@dataclass(frozen=True)
class Strategy:
    name: str
    policy: str
    description: str
    profile_name: Optional[str] = None
    client_variant: str = "refresh8_batch8"
    dynamic_mode: Optional[str] = None
    routing_mode: Optional[str] = None

    def config(self):
        value = asdict(self)
        if self.profile_name is not None:
            profile = STATIC_PROFILES[self.profile_name]
            value["category_cir_gbps"] = list(profile.category_cir_gbps)
            value["category_paths_per_group"] = list(
                profile.category_paths_per_group
            )
        if self.dynamic_mode is not None:
            value["dynamic_cir"] = {
                "decision_owner": "npu_collective_control_plane",
                "configuration_latency_ms": 0.0,
                "mode": self.dynamic_mode,
                "routing_mode": self.routing_mode,
                "paths_per_npu_per_ssu": 2,
                "per_npu_cap_gbps": NPU_BW_LIMIT,
                "per_ssu_cir_sum_cap_gbps": DISK_BW,
                "apply_boundary": "before_next_nonpreemptive_ssd_command",
            }
        return value


BEST_FIXED_PROFILE = "low_protect_cir_20_6_8_6_current_paths"


def strategies():
    return (
        Strategy(
            "baseline",
            POLICY_BASELINE_BYPASS,
            "NPU round-robin baseline",
        ),
        Strategy(
            "current_static",
            POLICY_QOS_STATIC_CIR,
            "Current static category Path and CIR",
            CURRENT_STATIC.name,
        ),
        Strategy(
            "best_fixed_static",
            POLICY_QOS_STATIC_CIR,
            "Best fixed CIR 20/6/8/6 with current Path layout",
            BEST_FIXED_PROFILE,
        ),
        Strategy(
            "ticket_static",
            POLICY_QOS_STATIC_CIR,
            "Static demand-ticket Path routing and static CIR",
            "equal_cir_demand_ticket_paths",
            "ticket_refresh8",
        ),
        Strategy(
            "demand_maxmin",
            POLICY_QOS_DEMAND_MAXMIN,
            "Dynamic per-NPU demand scheduler without hardware Paths",
        ),
        Strategy(
            "dynamic_demand_fixed_path",
            POLICY_QOS_DYNAMIC_JOINT_CIR,
            "NPU dynamic demand CIR with one fixed owned Path per layer/SSD",
            dynamic_mode=DYNAMIC_CIR_DEMAND_PROPORTIONAL,
            routing_mode=DYNAMIC_ROUTE_FIXED,
        ),
        Strategy(
            "joint_demand_path_cir",
            POLICY_QOS_DYNAMIC_JOINT_CIR,
            "NPU jointly selects its owned Path and demand-proportional CIR",
            dynamic_mode=DYNAMIC_CIR_DEMAND_PROPORTIONAL,
            routing_mode=DYNAMIC_ROUTE_LEAST_WORK,
        ),
        Strategy(
            "dynamic_slack_fixed_path",
            POLICY_QOS_DYNAMIC_JOINT_CIR,
            "NPU dynamic slack/link-guarded CIR with a fixed owned Path",
            dynamic_mode=DYNAMIC_CIR_SLACK_LINK_GUARDED,
            routing_mode=DYNAMIC_ROUTE_FIXED,
        ),
        Strategy(
            "joint_slack_path_cir",
            POLICY_QOS_DYNAMIC_JOINT_CIR,
            "NPU jointly selects Path and slack/link-guarded CIR",
            dynamic_mode=DYNAMIC_CIR_SLACK_LINK_GUARDED,
            routing_mode=DYNAMIC_ROUTE_LEAST_WORK,
        ),
    )


def seed_bundle(seed):
    return {
        "workload": int(seed),
        "placement": int(seed + PLACEMENT_SEED_OFFSET),
        "submit_order": int(seed + SUBMIT_ORDER_SEED_OFFSET),
        "arrival_delay": int(seed + ARRIVAL_DELAY_SEED_OFFSET),
    }


def runtime(seed):
    return {
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "ssu_list": list(SSU_LIST),
        "ls_ratio": LS_RATIO,
        "seeds": seed_bundle(seed),
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": DISK_BW,
        "npu_bw_limit_gbps": NPU_BW_LIMIT,
    }


def code_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "sim.py",
        "advanced_policies.py",
        "strategy_profiles.py",
        "experiment.py",
        "joint_dynamic_experiment.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def data_fingerprint(table):
    rows = [[list(key), list(table[key])] for key in sorted(table)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


def experiment_spec(table, run_config, selected):
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": code_fingerprint(),
        "data_fingerprint": data_fingerprint(table),
        "runtime": run_config,
        "backend": {
            "ssd": "one nonpreemptive command, size/40GBps",
            "npu": "one FCFS receive command per NPU, size/50GBps",
            "visible_after": "NPU receive completion",
        },
        "selected_strategies": [strategy.config() for strategy in selected],
    }


def prepare(table, run_config, num_ssu):
    seeds = run_config["seeds"]
    return prepare_simulation_inputs(
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


def run_case(table, run_config, num_ssu, strategy):
    started = time.perf_counter()
    prepared = prepare(table, run_config, num_ssu)
    kwargs = {
        "num_npu": run_config["num_npu"],
        "num_disk": num_ssu,
        "n_layers": run_config["n_layers"],
        "ls_ratio": run_config["ls_ratio"],
        "submit_order_seed": run_config["seeds"]["submit_order"],
        "prepared_inputs": prepared,
        "client_io_config": CLIENT_VARIANTS[strategy.client_variant],
    }
    if strategy.profile_name is not None:
        kwargs["qos_config"] = STATIC_PROFILES[strategy.profile_name].hardware_config()
    if strategy.dynamic_mode is not None:
        kwargs["dynamic_cir_config"] = DynamicCIRPolicyConfig(
            mode=strategy.dynamic_mode,
            routing_mode=strategy.routing_mode,
        )
    _, full = simulate_continuous(table, policy=strategy.policy, **kwargs)
    summary = _summary(full)
    request_metrics = summary.pop("request_metrics")
    summary["dynamic_cir_control"] = full["dynamic_cir_control"]
    assert all(summary["invariants"].values())
    return {
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "config": strategy.config(),
        "seeds": run_config["seeds"],
        "workload_fingerprint": prepared.workload_hash,
        "placement_hash": prepared.placement_hash,
        "summary": summary,
        "request_metrics": request_metrics,
        "wall_time_s": time.perf_counter() - started,
    }


_TABLE = None
_RUNTIME = None
_STRATEGIES = None


def init_worker(table, run_config, selected):
    global _TABLE, _RUNTIME, _STRATEGIES
    _TABLE = table
    _RUNTIME = run_config
    _STRATEGIES = {strategy.name: strategy for strategy in selected}


def worker(task):
    num_ssu, name = task
    return run_case(_TABLE, _RUNTIME, num_ssu, _STRATEGIES[name])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run(output_path, *, seed, workers, selection="all", rerun=False):
    table = load_bw_table_cache(num_npu=NUM_NPU)
    available = strategies()
    names = (
        {name.strip() for name in selection.split(",")}
        if selection != "all"
        else {strategy.name for strategy in available}
    )
    selected = tuple(strategy for strategy in available if strategy.name in names)
    run_config = runtime(seed)
    spec = experiment_spec(table, run_config, selected)
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
        write_json(
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
        worker_count = min(max(1, workers), len(tasks))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(table, run_config, selected),
        ) as pool:
            futures = {pool.submit(worker, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows[(row["num_ssu"], row["strategy"])] = row
                checkpoint()
                print(
                    "SSU=%d %-30s request=%7.3f%% fleet=%7.3f%% wall=%6.1fs"
                    % (
                        row["num_ssu"],
                        row["strategy"],
                        100.0 * row["summary"]["avg_request_compute_fraction"],
                        100.0 * row["summary"]["fleet_npu_compute_utilization"],
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
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--select", default="all")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run(
        args.output_dir / ("results_seed%d.json" % args.seed),
        seed=args.seed,
        workers=args.workers,
        selection=args.select,
        rerun=args.rerun,
    )
    print("complete:", result["complete"])
    print("rows:", len(result["results"]))


if __name__ == "__main__":
    main()
