"""Focused oracle and NPU-link counterfactuals for the 32-NPU warm trace."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    SLOAwareSchemeBController,
    SteadyStateInvariantError,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    best_feasible_client_config,
    best_feasible_priority_key,
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import SEED
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload
from steady_state_32npu_experiment import (
    NEW_SCHEME_MIN_UPDATE_MS,
    N_LAYERS,
    NUM_NPU,
    ROOT,
    STEADY_CONFIG,
)
from slo_admission_scheme_b import SLOAdmissionSchemeBController


OUTPUT = ROOT / "results" / "steady_state_32npu_normalized_slo" / "diagnostics.json"
NPU_BYPASS_GBPS = 1_000_000.0


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    kind: str
    num_ssu: int


CASES = (
    DiagnosticCase("scheme_b_slo_admission", "admission", 6),
    DiagnosticCase("scheme_b_slo_admission", "admission", 10),
    DiagnosticCase("scheme_b_slo_admission", "admission", 18),
    DiagnosticCase("released_io_oracle", "oracle", 6),
    DiagnosticCase("released_io_oracle", "oracle", 10),
    DiagnosticCase("released_io_oracle", "oracle", 18),
    DiagnosticCase("baseline_npu_bypass", "baseline_bypass", 18),
    DiagnosticCase("new_scheme_b_npu_bypass", "new_bypass", 18),
)
_WORKER_TABLE = None


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-32npu-diagnostics-v1\0")
    for name in (
        "sim.py",
        "policy_logic.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "six_request_workload.py",
        "steady_state_workload.py",
        "steady_state_32npu_experiment.py",
        "steady_state_32npu_diagnostics.py",
        "slo_admission_scheme_b.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _common(num_ssu, *, npu_bw_gbps=sim.NPU_BW_LIMIT):
    return {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
        "steady_state": STEADY_CONFIG,
        "npu_bw_gbps": npu_bw_gbps,
    }


def _simulate(case, requests):
    if case.kind == "admission":
        paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
        controller = SLOAdmissionSchemeBController(paths)
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * case.num_ssu
            ),
            npu_dedicated_paths=paths,
            client_io_config=scheme_b_client_config("slo_admission"),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=NEW_SCHEME_MIN_UPDATE_MS,
            ),
            **_common(case.num_ssu),
        )
    if case.kind == "oracle":
        return simulate_continuous_batch(
            requests,
            policy=sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
            client_io_config=best_feasible_client_config(),
            oracle_priority_key=best_feasible_priority_key,
            **_common(case.num_ssu),
        )
    if case.kind == "baseline_bypass":
        baseline = next(
            spec for spec in routing_strategy_specs() if spec.name == "baseline"
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=baseline.client_config(),
            **_common(case.num_ssu, npu_bw_gbps=NPU_BYPASS_GBPS),
        )
    if case.kind == "new_bypass":
        paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
        controller = SLOAwareSchemeBController(
            paths,
            slo_alpha=STEADY_CONFIG.slo_alpha,
            npu_cap_gbps=NPU_BYPASS_GBPS,
        )
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * case.num_ssu
            ),
            npu_dedicated_paths=paths,
            client_io_config=scheme_b_client_config("npu_bypass"),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=NEW_SCHEME_MIN_UPDATE_MS,
            ),
            **_common(case.num_ssu, npu_bw_gbps=NPU_BYPASS_GBPS),
        )
    raise ValueError(case.kind)


def _run(case, source_fingerprint):
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    workload = prepare_steady_state_workload(
        table,
        num_npu=NUM_NPU,
        num_ssu=case.num_ssu,
        n_layers=N_LAYERS,
        requests_per_npu=REQUESTS_PER_NPU,
        seed=SEED,
    )
    requests = requests_from_continuous_prefill_workload(workload)
    started = time.perf_counter()
    try:
        summary = _simulate(case, requests)
    except SteadyStateInvariantError as error:
        return {
            "status": "invalid",
            "source_fingerprint": source_fingerprint,
            "case_spec": asdict(case),
            "strategy": case.name,
            "kind": case.kind,
            "num_ssu": case.num_ssu,
            "wall_time_s": time.perf_counter() - started,
            "assignment_hash": workload.statistics["assignment_hash"],
            "workload_hash": workload.workload_hash,
            "placement_hash": workload.placement_hash,
            "trace_hash": workload.trace_hash,
            "failed_invariants": error.invariants,
            "failure_diagnostics": error.diagnostics,
        }
    if not all(summary["invariants"].values()):
        raise AssertionError(summary["invariants"])
    if case.kind in ("admission", "new_bypass"):
        if summary["control_evaluations"] <= 0 or summary["cir_commits"] <= 0:
            raise AssertionError(f"{case.kind} controller did not commit CIR")
    elif any(
        summary[field]
        for field in ("control_evaluations", "cir_commits", "cir_path_writes")
    ):
        raise AssertionError("non-controller diagnostic unexpectedly updated CIR")
    return {
        "status": "ok",
        "source_fingerprint": source_fingerprint,
        "case_spec": asdict(case),
        "strategy": case.name,
        "kind": case.kind,
        "num_ssu": case.num_ssu,
        "wall_time_s": time.perf_counter() - started,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary["input_fingerprint"],
        "physical_npu_bw_gbps": (
            NPU_BYPASS_GBPS
            if case.kind in ("baseline_bypass", "new_bypass")
            else sim.NPU_BW_LIMIT
        ),
        "allocator_npu_cap_gbps": (
            NPU_BYPASS_GBPS
            if case.kind == "new_bypass"
            else sim.NPU_BW_LIMIT
            if case.kind == "admission"
            else None
        ),
        "steady_summary": summary,
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def run(
    output=OUTPUT,
    *,
    workers=3,
    rerun=False,
    rerun_strategies=(),
):
    run_fingerprint = _source_fingerprint()
    experiment_spec = {
        "cases": [asdict(case) for case in CASES],
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "steady_state": STEADY_CONFIG.__dict__,
        "npu_bypass_gbps": NPU_BYPASS_GBPS,
        "new_scheme_min_update_ms": NEW_SCHEME_MIN_UPDATE_MS,
    }
    rows_by_key = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        if (
            cached.get("source_fingerprint") == run_fingerprint
            and cached.get("experiment_spec") == experiment_spec
            and all(
                row.get("source_fingerprint") == run_fingerprint
                for row in cached.get("results", ())
            )
        ):
            rows_by_key = {
                (row["strategy"], int(row["num_ssu"])): row
                for row in cached.get("results", ())
                if row["strategy"] not in rerun_strategies
            }
    pending_cases = [
        case for case in CASES if (case.name, case.num_ssu) not in rows_by_key
    ]

    def checkpoint():
        rows = list(rows_by_key.values())
        ending_fingerprint = _source_fingerprint()
        source_stable = ending_fingerprint == run_fingerprint
        _write(
            output,
            {
                "schema_version": 1,
                "complete": source_stable
                and len(rows) == len(CASES)
                and all(row.get("status", "ok") == "ok" for row in rows),
                "all_cases_finished": len(rows) == len(CASES),
                "source_stable_during_run": source_stable,
                "source_fingerprint": run_fingerprint,
                "ending_source_fingerprint": ending_fingerprint,
                "experiment_spec": experiment_spec,
                "num_npu": NUM_NPU,
                "n_layers": N_LAYERS,
                "steady_state": STEADY_CONFIG.__dict__,
                "npu_bypass_gbps": NPU_BYPASS_GBPS,
                "results": sorted(
                    rows, key=lambda item: (item["num_ssu"], item["strategy"])
                ),
            },
        )

    if pending_cases:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending_cases)), initializer=_init_worker
        ) as pool:
            futures = {
                pool.submit(_run, case, run_fingerprint): case for case in pending_cases
            }
            for future in as_completed(futures):
                case = futures[future]
                row = future.result()
                rows_by_key[(case.name, case.num_ssu)] = row
                if row["status"] == "ok":
                    summary = row["steady_summary"]
                    print(
                        f"{case.name} ssu={case.num_ssu}: "
                        f"util={summary['mean_npu_utilization']:.2%}, "
                        f"SLO={summary['ttft_slo_attainment']:.2%}, "
                        f"wall={row['wall_time_s']:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"INVALID {case.name} ssu={case.num_ssu}: "
                        f"{row['failed_invariants']}",
                        flush=True,
                    )
                checkpoint()
    else:
        checkpoint()
    if _source_fingerprint() != run_fingerprint:
        raise RuntimeError("diagnostic source changed during the run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--rerun-strategy",
        action="append",
        default=[],
        choices=sorted({case.name for case in CASES}),
        help="discard cached rows for one strategy; may be repeated",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output,
        workers=args.workers,
        rerun=args.rerun,
        rerun_strategies=tuple(args.rerun_strategy),
    )


if __name__ == "__main__":
    main()
