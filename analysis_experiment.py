"""Run the paired, modular strategy matrix used by the full QoS analysis.

The default run is the formal experiment: 128 NPUs, 16 layers, and 40/56/80
SSUs.  ``--quick`` is a deliberately smaller validation-only run for checking
the runner itself; its output is kept separate from formal results.
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
from typing import Optional

from experiment import _summary
from sim import (
    ARRIVAL_DELAY_MAX_MS,
    DISK_BW,
    NPU_BW_LIMIT,
    POLICY_BASELINE_BYPASS,
    POLICY_GLOBAL_LINK_AWARE,
    POLICY_PER_SSD_FULL_VISIBLE_EDF,
    POLICY_QOS_DEMAND_MAXMIN,
    POLICY_QOS_STATIC_CIR,
    QOS_ROUTING_CATEGORIES,
    load_bw_table_cache,
    prepare_simulation_inputs,
    simulate_continuous,
)
from strategy_profiles import (
    CLIENT_VARIANTS,
    CURRENT_STATIC,
    EQUAL_TICKET_STATIC,
    PRIMARY_STATIC_CANDIDATES,
    REFINEMENT_STATIC_CANDIDATES,
    STATIC_PROFILES,
)
from upper_bounds import isolated_no_contention_bound


SCHEMA_VERSION = 2
FORMAL_NUM_NPU = 128
FORMAL_LAYERS = 16
FORMAL_SSU_LIST = (40, 56, 80)
QUICK_NUM_NPU = 16
QUICK_LAYERS = 2
QUICK_SSU_LIST = (8,)
LS_RATIO = 0.5
WORKLOAD_SEED = 42
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003
DEFAULT_OUTPUT_DIR = Path("results/full_analysis")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    policy: Optional[str] = None
    profile_name: Optional[str] = None
    client_variant: Optional[str] = None
    family: str = ""
    description: str = ""

    def config(self):
        result = asdict(self)
        if self.profile_name is not None:
            profile = STATIC_PROFILES[self.profile_name]
            result["category_cir_gbps"] = list(profile.category_cir_gbps)
            result["category_paths_per_group"] = list(
                profile.category_paths_per_group
            )
            result["path_cir_gbps"] = list(profile.path_cirs())
            result["path_pir"] = "uncapped"
        if self.client_variant is not None:
            client = CLIENT_VARIANTS[self.client_variant]
            result["client_io"] = {
                "name": client.name,
                "pressure_window_io": client.pressure_window_io,
                "submit_batch_size": client.submit_batch_size,
                "path_pool_mode": client.path_pool_mode,
            }
        return result


def strategy_specs():
    """Return the stable ordered strategy registry for tuning and analysis."""
    specs = [
        StrategySpec(
            "baseline",
            "simulation",
            POLICY_BASELINE_BYPASS,
            family="core",
            description="NPU round-robin baseline on the shared data plane",
        ),
        StrategySpec(
            "current_layer_snapshot",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            CURRENT_STATIC.name,
            "layer_snapshot_batch8",
            "cadence",
            "Current static CIR, one pressure snapshot per request/layer/SSD",
        ),
        StrategySpec(
            "current_refresh8",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            CURRENT_STATIC.name,
            "refresh8_batch8",
            "cadence",
            "Current static CIR, refresh pressure every eight I/Os",
        ),
        StrategySpec(
            "current_per_io",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            CURRENT_STATIC.name,
            "per_io_live_batch8",
            "cadence",
            "Current static CIR, read live pressure for every I/O",
        ),
        StrategySpec(
            "current_refresh8_batch16",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            CURRENT_STATIC.name,
            "refresh8_batch16",
            "batch",
            "Current static CIR with 16-I/O client submission batches",
        ),
        StrategySpec(
            "current_refresh8_batch32",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            CURRENT_STATIC.name,
            "refresh8_batch32",
            "batch",
            "Current static CIR with 32-I/O client submission batches",
        ),
        StrategySpec(
            "ticket_layer_snapshot",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            EQUAL_TICKET_STATIC.name,
            "ticket_layer_snapshot",
            "ticket",
            "Demand-proportional per-NPU path tickets, layer snapshot",
        ),
        StrategySpec(
            "ticket_refresh8",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            EQUAL_TICKET_STATIC.name,
            "ticket_refresh8",
            "ticket",
            "Demand-proportional per-NPU path tickets, refresh every eight",
        ),
        StrategySpec(
            "ticket_per_io",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            EQUAL_TICKET_STATIC.name,
            "ticket_per_io",
            "ticket",
            "Demand-proportional per-NPU path tickets, live per-I/O pressure",
        ),
    ]

    tuned_profile_names = [
        profile.name
        for profile in PRIMARY_STATIC_CANDIDATES[1:]
        + REFINEMENT_STATIC_CANDIDATES
    ]
    specs.extend(
        StrategySpec(
            f"tune__{profile_name}",
            "simulation",
            POLICY_QOS_STATIC_CIR,
            profile_name,
            "refresh8_batch8",
            "tuning",
            f"Static CIR/path candidate {profile_name}",
        )
        for profile_name in tuned_profile_names
    )
    specs.extend(
        (
            StrategySpec(
                "demand_maxmin",
                "simulation",
                POLICY_QOS_DEMAND_MAXMIN,
                client_variant="refresh8_batch8",
                family="advanced",
                description=(
                    "Work-conserving demand-aware max-min packet scheduler"
                ),
            ),
            StrategySpec(
                "per_ssd_full_visible_edf",
                "simulation",
                POLICY_PER_SSD_FULL_VISIBLE_EDF,
                client_variant="oracle_layer_submit",
                family="advanced",
                description=(
                    "Per-SSD full-visible-layer EDF heuristic"
                ),
            ),
            StrategySpec(
                "global_link_aware_online",
                "simulation",
                POLICY_GLOBAL_LINK_AWARE,
                client_variant="oracle_layer_submit",
                family="advanced",
                description=(
                    "Online cross-SSD coordinator aware of committed NPU-link work"
                ),
            ),
            StrategySpec(
                "isolated_no_contention_bound",
                "upper_bound",
                family="bound",
                description=(
                    "Optimistic relaxation removing all inter-NPU contention"
                ),
            ),
        )
    )
    return tuple(specs)


def seed_bundle(seed):
    return {
        "workload": int(seed),
        "placement": int(seed + PLACEMENT_SEED_OFFSET),
        "submit_order": int(seed + SUBMIT_ORDER_SEED_OFFSET),
        "arrival_delay": int(seed + ARRIVAL_DELAY_SEED_OFFSET),
    }


def runtime_config(quick, seed):
    if quick:
        num_npu = QUICK_NUM_NPU
        n_layers = QUICK_LAYERS
        ssu_list = QUICK_SSU_LIST
        mode = "quick_validation_only"
    else:
        num_npu = FORMAL_NUM_NPU
        n_layers = FORMAL_LAYERS
        ssu_list = FORMAL_SSU_LIST
        mode = "formal"
    return {
        "mode": mode,
        "num_npu": num_npu,
        "n_layers": n_layers,
        "ssu_list": list(ssu_list),
        "ls_ratio": LS_RATIO,
        "seeds": seed_bundle(seed),
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": DISK_BW,
        "npu_bw_limit_gbps": NPU_BW_LIMIT,
    }


def select_strategies(selection):
    specs = strategy_specs()
    if selection == "all":
        return specs
    groups = {
        "core": ("baseline", "current_refresh8", "demand_maxmin"),
        "cadence": tuple(spec.name for spec in specs if spec.family == "cadence"),
        "batch": tuple(spec.name for spec in specs if spec.family == "batch"),
        "ticket": tuple(spec.name for spec in specs if spec.family == "ticket"),
        "tuning": tuple(spec.name for spec in specs if spec.family == "tuning"),
        "advanced": tuple(
            spec.name for spec in specs if spec.family == "advanced"
        ),
        "bound": tuple(spec.name for spec in specs if spec.family == "bound"),
    }
    requested = []
    for token in selection.split(","):
        requested.extend(groups.get(token, (token,)))
    requested_set = set(requested)
    return tuple(spec for spec in specs if spec.name in requested_set)


def _table_fingerprint(table):
    rows = [[list(key), list(table[key])] for key in sorted(table)]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


def _code_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "sim.py",
        "advanced_policies.py",
        "strategy_profiles.py",
        "upper_bounds.py",
        "experiment.py",
        "analysis_experiment.py",
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
            "num_npu": FORMAL_NUM_NPU,
            "n_layers": FORMAL_LAYERS,
            "ssu_list": list(FORMAL_SSU_LIST),
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


def _prepare(table, runtime, num_ssu):
    seeds = runtime["seeds"]
    return prepare_simulation_inputs(
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
    categories = {}
    for category in QOS_ROUTING_CATEGORIES:
        values = [
            row["compute_fraction_upper_bound"]
            for row in bound["request_bounds"]
            if row["category"] == category
        ]
        categories[category] = {
            "count": len(values),
            "avg_request_compute_fraction_upper_bound": sum(values) / len(values),
            "min_request_compute_fraction_upper_bound": min(values),
            "max_request_compute_fraction_upper_bound": max(values),
        }
    return {
        "name": bound["name"],
        "interpretation": "optimistic_infeasible_relaxation_upper_bound",
        "relaxations": bound["relaxations"],
        "avg_request_compute_fraction_upper_bound": bound[
            "avg_request_compute_fraction_upper_bound"
        ],
        "category_metrics": categories,
    }


def run_strategy_case(table, runtime, num_ssu, strategy):
    started = time.perf_counter()
    prepared = _prepare(table, runtime, num_ssu)
    base = {
        "num_ssu": num_ssu,
        "strategy": strategy.name,
        "family": strategy.family,
        "kind": strategy.kind,
        "config": strategy.config(),
        "seeds": runtime["seeds"],
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

    common = {
        "num_npu": runtime["num_npu"],
        "num_disk": num_ssu,
        "n_layers": runtime["n_layers"],
        "ls_ratio": runtime["ls_ratio"],
        "submit_order_seed": runtime["seeds"]["submit_order"],
        "prepared_inputs": prepared,
    }
    if strategy.client_variant is not None:
        common["client_io_config"] = CLIENT_VARIANTS[strategy.client_variant]
    if strategy.profile_name is not None:
        common["qos_config"] = STATIC_PROFILES[
            strategy.profile_name
        ].hardware_config()
    _, full = simulate_continuous(
        table,
        policy=strategy.policy,
        **common,
    )
    summary = _summary(full)
    request_metrics = summary.pop("request_metrics")
    assert all(summary["invariants"].values())
    assert summary["workload_fingerprint"] == prepared.workload_hash
    assert summary["placement_hash"] == prepared.placement_hash
    base.update(
        {
            "summary": summary,
            "request_metrics": request_metrics,
            "wall_time_s": time.perf_counter() - started,
        }
    )
    return base


_WORKER_TABLE = None
_WORKER_RUNTIME = None
_WORKER_STRATEGIES = None


def _init_worker(table, runtime, strategies):
    global _WORKER_TABLE, _WORKER_RUNTIME, _WORKER_STRATEGIES
    _WORKER_TABLE = table
    _WORKER_RUNTIME = runtime
    _WORKER_STRATEGIES = {spec.name: spec for spec in strategies}


def _run_worker_task(task):
    num_ssu, strategy_name = task
    return run_strategy_case(
        _WORKER_TABLE,
        _WORKER_RUNTIME,
        num_ssu,
        _WORKER_STRATEGIES[strategy_name],
    )


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _result_key(num_ssu, strategy_name):
    return f"{num_ssu}:{strategy_name}"


def _ordered_results(rows, runtime, selected):
    return [
        rows[_result_key(num_ssu, strategy.name)]
        for num_ssu in runtime["ssu_list"]
        for strategy in selected
        if _result_key(num_ssu, strategy.name) in rows
    ]


def _validate_paired_inputs(results):
    by_ssu = {}
    for row in results:
        by_ssu.setdefault(row["num_ssu"], set()).add(
            (row["workload_fingerprint"], row["placement_hash"])
        )
    assert all(len(fingerprints) == 1 for fingerprints in by_ssu.values())


def run_matrix(
    result_path,
    *,
    selected,
    runtime,
    workers,
    rerun=False,
):
    table = load_bw_table_cache(num_npu=runtime["num_npu"])
    spec = experiment_spec(table, runtime)
    rows = {}
    if result_path.exists() and not rerun:
        cached = json.loads(result_path.read_text())
        if cached.get("experiment") == spec:
            rows = {
                _result_key(row["num_ssu"], row["strategy"]): row
                for row in cached["results"]
            }

    selected_names = [strategy.name for strategy in selected]

    def checkpoint():
        selected_rows = _ordered_results(rows, runtime, selected)
        _write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": spec,
                "selected_strategies": selected_names,
                "complete": all(
                    _result_key(num_ssu, strategy.name) in rows
                    for num_ssu in runtime["ssu_list"]
                    for strategy in selected
                ),
                "results": selected_rows,
            },
        )

    pending = [
        (num_ssu, strategy.name)
        for num_ssu in runtime["ssu_list"]
        for strategy in selected
        if _result_key(num_ssu, strategy.name) not in rows
    ]
    if not pending:
        pass
    elif workers == 1:
        for num_ssu, strategy_name in pending:
            print(f"运行 SSU={num_ssu} strategy={strategy_name} ...", flush=True)
            strategy = next(
                spec for spec in selected if spec.name == strategy_name
            )
            row = run_strategy_case(table, runtime, num_ssu, strategy)
            rows[_result_key(num_ssu, strategy_name)] = row
            checkpoint()
            print(
                f"完成 SSU={num_ssu} strategy={strategy_name} "
                f"({row['wall_time_s']:.2f}s)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            initializer=_init_worker,
            initargs=(table, runtime, selected),
        ) as pool:
            futures = {
                pool.submit(_run_worker_task, task): task for task in pending
            }
            for future in as_completed(futures):
                num_ssu, strategy_name = futures[future]
                row = future.result()
                rows[_result_key(num_ssu, strategy_name)] = row
                checkpoint()
                print(
                    f"完成 SSU={num_ssu} strategy={strategy_name} "
                    f"({row['wall_time_s']:.2f}s)",
                    flush=True,
                )

    results = _ordered_results(rows, runtime, selected)
    _validate_paired_inputs(results)
    checkpoint()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": spec,
        "selected_strategies": selected_names,
        "complete": True,
        "results": results,
    }


def print_results(data):
    print("\nSSU  Strategy                                      Compute/Bound")
    for row in data["results"]:
        summary = row["summary"]
        value = summary.get(
            "avg_request_compute_fraction",
            summary.get("avg_request_compute_fraction_upper_bound"),
        )
        print(f"{row['num_ssu']:>3}  {row['strategy']:<44} {100 * value:>8.3f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(10, os.cpu_count() or 1),
        help="processes used for independent (SSU, strategy) cases",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--select",
        default="all",
        help=(
            "comma-separated names or groups: core,cadence,batch,ticket,"
            "tuning,advanced,bound"
        ),
    )
    parser.add_argument("--seed", type=int, default=WORKLOAD_SEED)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--list-strategies", action="store_true")
    args = parser.parse_args()

    if args.list_strategies:
        for strategy in strategy_specs():
            print(f"{strategy.name:<52} {strategy.family:<10} {strategy.description}")
        return

    selected = select_strategies(args.select)
    runtime = runtime_config(args.quick, args.seed)
    result_name = "quick_results.json" if args.quick else "results.json"
    data = run_matrix(
        args.output_dir / result_name,
        selected=selected,
        runtime=runtime,
        workers=args.workers,
        rerun=args.rerun,
    )
    print_results(data)


if __name__ == "__main__":
    main()
