"""Isolated 128-NPU real-data matrix for Adaptive Admission V2.1.

Runs the frozen 25-ms adaptive controller at SSU24/40/70.  Results, complete
simulator summaries, fingerprints, and resumable checkpoints are private to
this runner; existing V1/V2 experiments and their SOURCE_FILES are untouched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import sim
import steady_state_128npu_admission_experiment as v1_experiment
from adaptive_admission_scheme_b_v2_1 import (
    AdaptiveAdmissionSchemeBControllerV2_1,
)
from continuous_batch_sim import (
    CIRControlConfig,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from steady_state_32npu_adaptive_v2_1_experiment import (
    Case,
    _category_slo,
    _pct,
    _percentile,
    _write,
)
from steady_state_workload import prepare_steady_state_workload


ROOT = v1_experiment.ROOT
OUTPUT_DIR = ROOT / "results" / "steady_state_128npu_adaptive_v2_1"
OUTPUT = OUTPUT_DIR / "results.json"
REPORT = OUTPUT_DIR / "report.md"
SCHEMA_VERSION = 1
NUM_NPU = v1_experiment.NUM_NPU
N_LAYERS = v1_experiment.N_LAYERS
SSU_LIST = (24, 40, 70)
MIN_INTERVAL_MS = 25.0
EXPLICIT_SPILL_THRESHOLD = 0.75
TARGET_RATIO = v1_experiment.TARGET_RATIO
REQUIRED_RATIO = v1_experiment.REQUIRED_RATIO
BACKGROUND_RESERVE_FRACTION = v1_experiment.BACKGROUND_RESERVE_FRACTION
SSD_CAP_GBPS = v1_experiment.SSD_CAP_GBPS
NPU_CAP_GBPS = v1_experiment.NPU_CAP_GBPS
REQUESTS_PER_NPU = v1_experiment.REQUESTS_PER_NPU
SEED = v1_experiment.SEED
STEADY_CONFIG = v1_experiment.STEADY_CONFIG
CATEGORIES = v1_experiment.CATEGORIES
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *v1_experiment.SOURCE_FILES,
            "slo_admission_scheme_b_v2.py",
            "adaptive_admission_scheme_b_v2_1.py",
            "steady_state_32npu_adaptive_v2_1_experiment.py",
            Path(__file__).name,
        )
    )
)
_WORKER_TABLE = None


CASES = tuple(
    Case("adaptive_v2_1_25ms", "adaptive_admission_v2_1", num_ssu, 25.0)
    for num_ssu in SSU_LIST
)
CASE_BY_KEY = {(case.name, case.num_ssu): case for case in CASES}


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-128npu-adaptive-v2-1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "isolated_128npu_adaptive_admission_v2_1",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "strategies": [asdict(case) for case in CASES],
        "seed": SEED,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "steady_state": asdict(STEADY_CONFIG),
        "controller": {
            "name": "AdaptiveAdmissionSchemeBControllerV2_1",
            "explicit_spill_threshold": EXPLICIT_SPILL_THRESHOLD,
            "decision_signal": "current_active_manifest_selected_fraction",
            "comparison": "strictly_less_than_threshold_uses_explicit_spill",
            "target_ratio": TARGET_RATIO,
            "required_ratio": REQUIRED_RATIO,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_CAP_GBPS,
        },
        "control": {
            "trigger": "batch_membership_event",
            "on_batch_boundary": True,
            "interval_ms": None,
            "min_interval_ms": MIN_INTERVAL_MS,
            "semantics": "event-gated minimum spacing, not periodic ticks",
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash, reused by all 16 layers",
        "real_data": True,
        "measurement_cohort": (
            "requests admitted in the half-open common measurement window; "
            "all tagged requests drain while all NPUs remain backlogged"
        ),
        "parent_experiments_unchanged": True,
        "result_directory": str(OUTPUT_DIR.relative_to(ROOT)),
    }


def _config_fingerprint(spec=None):
    return _canonical_hash(
        spec or experiment_spec(), b"128npu-adaptive-v2-1-config-v1\0"
    )


def _case_fingerprint(case, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"128npu-adaptive-v2-1-case-v1\0",
    )


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _dedicated_paths():
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _simulate(case, requests):
    if case.kind != "adaptive_admission_v2_1":
        raise ValueError(f"unknown adaptive case kind: {case.kind}")
    paths = _dedicated_paths()
    controller = AdaptiveAdmissionSchemeBControllerV2_1(
        paths,
        explicit_spill_threshold=EXPLICIT_SPILL_THRESHOLD,
        target_ratio=TARGET_RATIO,
        required_ratio=REQUIRED_RATIO,
        background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
        ssd_cap_gbps=SSD_CAP_GBPS,
        npu_cap_gbps=NPU_CAP_GBPS,
    )
    summary = simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * case.num_ssu
        ),
        npu_dedicated_paths=paths,
        layer0_path_id=None,
        client_io_config=scheme_b_client_config(case.name),
        control=CIRControlConfig(
            callback=controller,
            on_batch_boundary=True,
            min_interval_ms=case.min_interval_ms,
        ),
        num_npu=NUM_NPU,
        num_ssu=case.num_ssu,
        n_layers=N_LAYERS,
        batch_size=1,
        submit_order_seed=SEED,
        cross_request_layer0_prefetch=True,
        steady_state=STEADY_CONFIG,
    )
    enriched = dict(summary)
    enriched["adaptive_residual_mode_evaluations"] = dict(
        controller.residual_mode_evaluations
    )
    enriched["adaptive_last_selected_fraction"] = (
        controller.last_allocation.selected_fraction
        if controller.last_allocation is not None
        else None
    )
    enriched["adaptive_explicit_spill_threshold"] = EXPLICIT_SPILL_THRESHOLD
    return enriched


def _validate_summary(case, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("adaptive runner returned the wrong simulator mode")
    if (
        int(summary["num_npu"]) != NUM_NPU
        or int(summary["num_ssu"]) != case.num_ssu
    ):
        raise AssertionError("adaptive runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("adaptive controller must not read Path pressure")
    if any(
        summary[name] <= 0
        for name in ("control_evaluations", "cir_commits", "cir_path_writes")
    ):
        raise AssertionError("adaptive controller did not evaluate and commit CIR")
    if not math.isclose(
        float(summary["control_min_interval_ms"]),
        case.min_interval_ms,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("adaptive controller used the wrong interval")
    counts = summary.get("adaptive_residual_mode_evaluations", {})
    if sum(counts.values()) != summary["control_evaluations"]:
        raise AssertionError("adaptive mode counts do not cover every evaluation")
    if not math.isclose(
        float(summary["adaptive_explicit_spill_threshold"]),
        EXPLICIT_SPILL_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("adaptive result used the wrong threshold")


def _run_case(task):
    case, source_fingerprint, config_fingerprint = task
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.name} ssu={case.num_ssu}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
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
        summary = _simulate(case, requests)
        _validate_summary(case, summary)
        inputs = {
            "assignment": workload.statistics["assignment_hash"],
            "workload": workload.workload_hash,
            "placement": workload.placement_hash,
            "trace": workload.trace_hash,
            "simulator": summary["input_fingerprint"],
        }
        return {
            "status": "ok",
            "strategy": case.name,
            "kind": case.kind,
            "num_ssu": case.num_ssu,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, source_fingerprint, config_fingerprint
            ),
            "input_fingerprints": inputs,
            "assignment_hash": inputs["assignment"],
            "workload_hash": inputs["workload"],
            "placement_hash": inputs["placement"],
            "trace_hash": inputs["trace"],
            "simulator_input_fingerprint": inputs["simulator"],
            "wall_time_s": time.perf_counter() - started,
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _key(row):
    return str(row["strategy"]), int(row["num_ssu"])


def _validate_cached_rows(cached, source_fingerprint, spec, config_fingerprint):
    if cached.get("source_fingerprint") != source_fingerprint:
        return {}
    if cached.get("ending_source_fingerprint") != source_fingerprint:
        return {}
    if not cached.get("source_stable_during_run"):
        return {}
    if cached.get("experiment_spec") != spec:
        return {}
    if cached.get("config_fingerprint") != config_fingerprint:
        return {}
    if cached.get("ending_config_fingerprint") != config_fingerprint:
        return {}
    if not cached.get("config_stable_during_run"):
        return {}
    rows = {}
    required_inputs = {"assignment", "workload", "placement", "trace", "simulator"}
    for row in cached.get("results", ()):
        try:
            key = _key(row)
            case = CASE_BY_KEY[key]
        except (KeyError, TypeError, ValueError):
            return {}
        if key in rows:
            return {}
        if row.get("status") != "ok" or row.get("case_spec") != asdict(case):
            return {}
        if row.get("source_fingerprint") != source_fingerprint:
            return {}
        if row.get("config_fingerprint") != config_fingerprint:
            return {}
        if row.get("case_fingerprint") != _case_fingerprint(
            case, source_fingerprint, config_fingerprint
        ):
            return {}
        if set(row.get("input_fingerprints", ())) != required_inputs:
            return {}
        _validate_summary(case, row["steady_summary"])
        rows[key] = row
    return rows


def render_markdown(payload):
    lines = [
        "# 128-NPU Adaptive Admission V2.1 real-data matrix",
        "",
        "The current active-manifest selected fraction is causal. Fractions "
        "strictly below 0.75 use explicit V2 spill; all others retain V1 "
        "coflow residual. The cutoff is a calibrated heuristic, not a universal "
        "optimum.",
        "",
        f"Complete: `{str(payload['complete']).lower()}`; selected complete: "
        f"`{str(payload['selected_complete']).lower()}`; source stable: "
        f"`{str(payload['source_stable_during_run']).lower()}`; config stable: "
        f"`{str(payload['config_stable_during_run']).lower()}`.",
        "",
        f"Source fingerprint: `{payload['source_fingerprint']}`",
        "",
        f"Config fingerprint: `{payload['config_fingerprint']}`",
        "",
        "| SSU | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | "
        "SS | SL | LS | LL | Explicit/coflow evals | Last selected | Requests | "
        "Path writes | Wall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        summary = row["steady_summary"]
        utils = summary["npu_utilizations"]
        category = _category_slo(summary["request_rows"])
        modes = summary["adaptive_residual_mode_evaluations"]
        lines.append(
            "| {ssu} | {minimum} | {p10} | {mean} | {equal} | {request} | "
            "{SS} | {SL} | {LS} | {LL} | {explicit}/{coflow} | {last:.4f} | "
            "{count} | {writes} | {wall:.1f}s |".format(
                ssu=row["num_ssu"],
                minimum=_pct(min(utils)),
                p10=_pct(_percentile(utils, 10)),
                mean=_pct(summary["mean_npu_utilization"]),
                equal=_pct(summary["ttft_slo_attainment"]),
                request=_pct(summary["request_weighted_slo_attainment"]),
                explicit=modes["v2_explicit_selected_spill"],
                coflow=modes["v1_coflow_residual"],
                last=summary["adaptive_last_selected_fraction"],
                count=summary["measurement_request_count"],
                writes=summary["cir_path_writes"],
                wall=row["wall_time_s"],
                **{name: _pct(category[name]) for name in CATEGORIES},
            )
        )
    return "\n".join(lines) + "\n"


def run(
    output=OUTPUT,
    report=REPORT,
    *,
    workers=3,
    ssu_list=(),
    rerun=False,
):
    run_source = _source_fingerprint()
    spec = experiment_spec()
    config = _config_fingerprint(spec)
    wanted_ssus = tuple(int(value) for value in ssu_list) or SSU_LIST
    rows = {}
    if output.exists() and not rerun:
        rows = _validate_cached_rows(
            json.loads(output.read_text()), run_source, spec, config
        )
    tasks = [
        (CASE_BY_KEY[("adaptive_v2_1_25ms", num_ssu)], run_source, config)
        for num_ssu in wanted_ssus
        if ("adaptive_v2_1_25ms", num_ssu) not in rows
    ]
    tasks.sort(key=lambda task: task[0].num_ssu, reverse=True)
    source_stable = True
    config_stable = True

    def checkpoint():
        nonlocal source_stable, config_stable
        ending_source = _source_fingerprint()
        ending_config = _config_fingerprint(experiment_spec())
        source_stable = source_stable and ending_source == run_source
        config_stable = config_stable and ending_config == config
        ordered = [
            rows[(case.name, case.num_ssu)]
            for case in CASES
            if (case.name, case.num_ssu) in rows
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": source_stable
            and config_stable
            and len(ordered) == len(CASES),
            "selected_complete": source_stable
            and config_stable
            and all(
                ("adaptive_v2_1_25ms", num_ssu) in rows
                for num_ssu in wanted_ssus
            ),
            "source_stable_during_run": source_stable,
            "config_stable_during_run": config_stable,
            "source_fingerprint": run_source,
            "ending_source_fingerprint": ending_source,
            "config_fingerprint": config,
            "ending_config_fingerprint": ending_config,
            "experiment_spec": spec,
            "selected_ssu_list": list(wanted_ssus),
            "results": ordered,
        }
        _write(
            output,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        )
        _write(report, render_markdown(payload))

    if workers == 1:
        _init_worker()
        for task in tasks:
            row = _run_case(task)
            rows[_key(row)] = row
            checkpoint()
    elif tasks:
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker
        )
        try:
            futures = {pool.submit(_run_case, task): task[0] for task in tasks}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    print(
                        f"FAILED {case.name} ssu={case.num_ssu}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                rows[_key(row)] = row
                checkpoint()
                summary = row["steady_summary"]
                print(
                    f"{case.name} ssu={case.num_ssu}: "
                    f"util={summary['mean_npu_utilization']:.2%}, "
                    f"SLO={summary['ttft_slo_attainment']:.2%}, "
                    f"wall={row['wall_time_s']:.1f}s",
                    flush=True,
                )
        except BaseException:
            processes = tuple(pool._processes.values())
            for process in processes:
                process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    checkpoint()
    if not source_stable:
        raise RuntimeError("adaptive experiment source changed during run")
    if not config_stable:
        raise RuntimeError("adaptive experiment config changed during run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--ssu", action="append", type=int, choices=SSU_LIST)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output,
        args.report,
        workers=args.workers,
        ssu_list=tuple(args.ssu or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()

