"""Retained Full-prefill comparison on one paired 28-SSU trace."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import sim
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    CIRControlConfig,
    MaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    best_feasible_client_config,
    best_feasible_priority_key,
    routing_strategy_specs,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
    static_qos_config,
)
from continuous_prefill_workload import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_LAYERS,
    DEFAULT_NUM_NPU,
    DEFAULT_NUM_SSU,
    DEFAULT_SEED,
    prepare_continuous_prefill_workload,
)
from scheme_b_prefill import (
    PATH_COUNT,
    cold_start_hybrid_path_id,
    dedicated_path_id,
)


SCHEMA_VERSION = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "full_prefill_microbatch"
DEFAULT_OUTPUT = OUTPUT_DIR / "results.json"


@dataclass(frozen=True)
class Case:
    name: str
    kind: str


CASES = (
    *(Case(spec.name, "routing") for spec in routing_strategy_specs()),
    Case("scheme_b_once", "scheme_b"),
    Case("scheme_b_after_l0", "scheme_b_hybrid"),
    Case("best_feasible", "full_info"),
)
CASE_BY_NAME = {case.name: case for case in CASES}


def _source_fingerprint():
    digest = hashlib.sha256(b"full-prefill-microbatch-experiment-v3\0")
    root = Path(__file__).resolve().parent
    for name in (
        "sim.py",
        "policy_logic.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "continuous_prefill_experiment.py",
        "scheme_b_prefill.py",
        "strategy_profiles.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _run_case(case: Case):
    started = time.perf_counter()
    table = sim.load_bw_table_cache(num_npu=DEFAULT_NUM_NPU)
    workload = prepare_continuous_prefill_workload(table)
    requests = requests_from_continuous_prefill_workload(workload)
    kwargs = {
        "num_npu": workload.num_npu,
        "num_ssu": workload.num_ssu,
        "n_layers": workload.n_layers,
        "batch_size": workload.batch_size,
        "submit_order_seed": workload.seed,
    }
    control_metadata = None

    if case.kind == "routing":
        spec = next(spec for spec in routing_strategy_specs() if spec.name == case.name)
        summary = simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=spec.client_config(),
            **kwargs,
        )
    elif case.kind in ("scheme_b", "scheme_b_hybrid"):
        configs = qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * workload.num_ssu
        )
        path_by_npu = tuple(
            (
                cold_start_hybrid_path_id(npu_id)
                if case.kind == "scheme_b_hybrid"
                else dedicated_path_id(npu_id)
            )
            for npu_id in range(workload.num_npu)
        )
        causal = case.kind == "scheme_b_hybrid"
        control = None if causal else CIRControlConfig(
            on_batch_boundary=True,
            callback=MaxMinSchemeBController(path_by_npu, horizon_layers=1),
        )
        causal_control = (
            CausalLayerControlConfig(
                CausalMaxMinSchemeBController(
                    path_by_npu,
                    cold_path_id=0,
                    cold_path_cir_gbps=static_qos_config().path_cirs[0],
                    path_count=PATH_COUNT,
                )
            )
            if causal
            else None
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=configs,
            npu_dedicated_paths=path_by_npu,
            layer0_path_id=(0 if causal else None),
            client_io_config=scheme_b_client_config(case.name),
            control=control,
            causal_control=causal_control,
            **kwargs,
        )
        control_metadata = {
            "update_trigger": (
                "microbatch_membership_only"
                if not causal
                else "previous_layer_observation_change"
            ),
            "initial_target_hash": None,
            "initial_path_writes": 0,
            "layer0_policy": (
                "path0_then_previous_layer_observed_scheme_b"
                if causal
                else "scheme_b_dedicated_path"
            ),
            "policy_clock_input": False,
            "future_placement_input": not causal,
        }
    else:
        summary = simulate_continuous_batch(
            requests,
            policy=sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
            client_io_config=best_feasible_client_config(),
            oracle_priority_key=best_feasible_priority_key,
            **kwargs,
        )

    assert all(summary["invariants"].values())
    return {
        "strategy": case.name,
        "kind": case.kind,
        "wall_time_s": time.perf_counter() - started,
        "trace_hash": workload.trace_hash,
        "workload_statistics": workload.statistics,
        "control": control_metadata,
        "summary": summary,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "num_npu": DEFAULT_NUM_NPU,
        "num_ssu": DEFAULT_NUM_SSU,
        "batch_size": DEFAULT_BATCH_SIZE,
        "new_requests_per_npu": 1,
        "n_layers": DEFAULT_N_LAYERS,
        "seed": DEFAULT_SEED,
        "strategies": [case.name for case in CASES],
        "execution_model": "full_prefill_layer_synchronous_microbatch_v1",
        "batch_compute_model": "sum_member_singleton_layer_ms",
        "partial_batch_policy": "wait_for_full_then_drain_final_partial",
        "scheme_b_frequency_basis": "completed_microbatch_layer_equivalents",
        "scheme_b_membership_updates": True,
        "physical_data_plane": "SSD40 single-command -> NPU50 FCFS",
    }


def run(output_path=DEFAULT_OUTPUT, *, workers=6, selected=(), rerun=False):
    experiment = _experiment_spec()
    wanted = tuple(selected) if selected else tuple(case.name for case in CASES)
    rows = {}
    if output_path.exists() and not rerun:
        cached = json.loads(output_path.read_text())
        if cached.get("experiment") == experiment:
            rows = {row["strategy"]: row for row in cached["results"]}

    pending = [CASE_BY_NAME[name] for name in wanted if name not in rows]

    def checkpoint():
        ordered = [rows[case.name] for case in CASES if case.name in rows]
        _write_json(
            output_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": all(name in rows for name in wanted),
                "experiment": experiment,
                "results": ordered,
            },
        )

    if workers == 1:
        for case in pending:
            row = _run_case(case)
            rows[case.name] = row
            checkpoint()
            _print_row(row)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(_run_case, case): case for case in pending}
            for future in as_completed(futures):
                row = future.result()
                rows[row["strategy"]] = row
                checkpoint()
                _print_row(row)
    checkpoint()
    return output_path


def _print_row(row):
    summary = row["summary"]
    print(
        f"{row['strategy']}: fleet={summary['fleet_npu_compute_utilization']:.4%}, "
        f"p99={summary['p99_request_latency_ms']:.1f} ms, "
        f"wall={row['wall_time_s']:.1f} s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        workers=args.workers,
        selected=tuple(args.case or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
