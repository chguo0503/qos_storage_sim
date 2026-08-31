"""Paired 128-NPU warm/full-load admission-controller experiment.

This runner is intentionally separate from ``steady_state_experiment.py`` and
its historical result directory.  The frozen matrix compares baseline, the
retained causal Scheme B, and the optimized SLO-admission controller with
25/50-ms event-gated minimum update spacing at SSU24/40/70.

Runs may be accumulated in phases through repeated CLI calls.  Cached rows are
reused only when the complete experiment spec plus source/config/case
fingerprints match.  Source/config fingerprints are frozen before worker launch
and checked again at every checkpoint and at run end.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    CausalLayerControlConfig,
    CausalMaxMinSchemeBController,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import NUM_NPU, SEED
from slo_admission_scheme_b import SLOAdmissionSchemeBController
from steady_state_experiment import N_LAYERS, ROOT, STEADY_CONFIG
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


OUTPUT_DIR = ROOT / "results" / "steady_state_128npu_admission_v1"
OUTPUT = OUTPUT_DIR / "results.json"
REPORT = OUTPUT_DIR / "report.md"
SCHEMA_VERSION = 1
SSU_LIST = (24, 40, 70)
TARGET_RATIO = 0.52
REQUIRED_RATIO = 1.0 / STEADY_CONFIG.slo_alpha
BACKGROUND_RESERVE_FRACTION = 0.05
SSD_CAP_GBPS = sim.DISK_BW
NPU_CAP_GBPS = sim.NPU_BW_LIMIT
CATEGORIES = ("SS", "SL", "LS", "LL")
_WORKER_TABLE = None

SOURCE_FILES = (
    "sim.py",
    "policy_logic.py",
    "strategy_profiles.py",
    "continuous_batch_control.py",
    "continuous_batch_sim.py",
    "continuous_prefill_client.py",
    "continuous_prefill_workload.py",
    "six_request_workload.py",
    "steady_state_workload.py",
    "steady_state_experiment.py",
    "scheme_b_prefill.py",
    "slo_admission_scheme_b.py",
    "steady_state_128npu_admission_experiment.py",
    "data",
)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    min_interval_ms: float | None = None


CASES = (
    Case("baseline", "baseline"),
    Case("current_scheme_b", "causal_scheme_b"),
    Case("admission_25ms", "slo_admission", 25.0),
    Case("admission_50ms", "slo_admission", 50.0),
)
CASE_BY_NAME = {case.name: case for case in CASES}
RUNTIME_WEIGHTS = {
    "baseline": 1.0,
    "current_scheme_b": 1.6,
    "admission_25ms": 1.5,
    "admission_50ms": 1.5,
}


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-128npu-admission-v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "strategies": [asdict(case) for case in CASES],
        "seed": SEED,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "steady_state": asdict(STEADY_CONFIG),
        "controller": {
            "name": "SLOAdmissionSchemeBController",
            "target_ratio": TARGET_RATIO,
            "required_ratio": REQUIRED_RATIO,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_CAP_GBPS,
        },
        "admission_control": {
            "trigger": "batch_membership_event",
            "on_batch_boundary": True,
            "interval_ms": None,
            "min_intervals_ms": [25.0, 50.0],
            "semantics": "event-gated minimum spacing, not periodic ticks",
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash, reused by all 16 layers",
        "measurement_cohort": (
            "requests admitted in the half-open common measurement window; "
            "all tagged requests drain while all NPUs remain backlogged"
        ),
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
    }


def _config_fingerprint(spec=None):
    return _canonical_hash(spec or experiment_spec(), b"128npu-config-v1\0")


def _case_fingerprint(case, num_ssu, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "num_ssu": int(num_ssu),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"128npu-case-v1\0",
    )


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _dedicated_paths():
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _common_args(num_ssu):
    return {
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
        "steady_state": STEADY_CONFIG,
    }


def _simulate(case, requests, *, num_ssu):
    common = _common_args(num_ssu)
    if case.kind == "baseline":
        routing = next(
            spec for spec in routing_strategy_specs() if spec.name == "baseline"
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=routing.client_config(),
            **common,
        )

    paths = _dedicated_paths()
    empty_qos = qos_configs_from_path_cirs(
        ((0.0,) * PATH_COUNT,) * num_ssu
    )
    if case.kind == "causal_scheme_b":
        controller = CausalMaxMinSchemeBController(
            paths,
            cold_path_id=0,
            cold_path_cir_gbps=static_qos_config().path_cirs[0],
            path_count=PATH_COUNT,
        )
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=empty_qos,
            npu_dedicated_paths=paths,
            layer0_path_id=0,
            client_io_config=scheme_b_client_config(case.name),
            causal_control=CausalLayerControlConfig(controller),
            **common,
        )

    if case.kind == "slo_admission":
        controller = SLOAdmissionSchemeBController(
            paths,
            target_ratio=TARGET_RATIO,
            required_ratio=REQUIRED_RATIO,
            background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
            ssd_cap_gbps=SSD_CAP_GBPS,
            npu_cap_gbps=NPU_CAP_GBPS,
        )
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=empty_qos,
            npu_dedicated_paths=paths,
            layer0_path_id=None,
            client_io_config=scheme_b_client_config(case.name),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=case.min_interval_ms,
            ),
            **common,
        )
    raise ValueError(f"unknown 128-NPU case kind: {case.kind}")


def _validate_summary(case, num_ssu, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("128-NPU runner returned the wrong simulator mode")
    if int(summary["num_npu"]) != NUM_NPU or int(summary["num_ssu"]) != num_ssu:
        raise AssertionError("128-NPU runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("128-NPU paired strategies must not read pressure")

    counters = (
        summary["control_evaluations"],
        summary["cir_commits"],
        summary["cir_path_writes"],
    )
    if case.kind == "baseline":
        if any(counters):
            raise AssertionError("baseline must not evaluate or update CIR")
        if summary["control_min_interval_ms"] is not None:
            raise AssertionError("baseline unexpectedly has a control interval")
    else:
        if any(value <= 0 for value in counters):
            raise AssertionError(f"{case.name} did not evaluate and commit CIR")
        if case.kind == "slo_admission":
            if not math.isclose(
                float(summary["control_min_interval_ms"]),
                float(case.min_interval_ms),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError("admission used the wrong minimum interval")
        elif summary["control_min_interval_ms"] is not None:
            raise AssertionError("causal Scheme B unexpectedly has a wall control")


def _run_case(task):
    case, num_ssu, source_fingerprint, config_fingerprint = task
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.name} ssu={num_ssu}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
        workload = prepare_steady_state_workload(
            table,
            num_npu=NUM_NPU,
            num_ssu=num_ssu,
            n_layers=N_LAYERS,
            requests_per_npu=REQUESTS_PER_NPU,
            seed=SEED,
        )
        requests = requests_from_continuous_prefill_workload(workload)
        summary = _simulate(case, requests, num_ssu=num_ssu)
        _validate_summary(case, num_ssu, summary)
        return {
            "status": "ok",
            "strategy": case.name,
            "kind": case.kind,
            "num_ssu": num_ssu,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, num_ssu, source_fingerprint, config_fingerprint
            ),
            "wall_time_s": time.perf_counter() - started,
            "assignment_hash": workload.statistics["assignment_hash"],
            "workload_hash": workload.workload_hash,
            "placement_hash": workload.placement_hash,
            "trace_hash": workload.trace_hash,
            "simulator_input_fingerprint": summary["input_fingerprint"],
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _key(row):
    return str(row["strategy"]), int(row["num_ssu"])


def _validate_cached_rows(
    cached,
    source_fingerprint,
    spec,
    config_fingerprint,
):
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
    for row in cached.get("results", ()):
        try:
            strategy, num_ssu = _key(row)
            case = CASE_BY_NAME[strategy]
        except (KeyError, TypeError, ValueError):
            return {}
        if num_ssu not in SSU_LIST or (strategy, num_ssu) in rows:
            return {}
        if row.get("status") != "ok" or row.get("case_spec") != asdict(case):
            return {}
        if row.get("source_fingerprint") != source_fingerprint:
            return {}
        if row.get("config_fingerprint") != config_fingerprint:
            return {}
        if row.get("case_fingerprint") != _case_fingerprint(
            case, num_ssu, source_fingerprint, config_fingerprint
        ):
            return {}
        _validate_summary(case, num_ssu, row["steady_summary"])
        rows[(strategy, num_ssu)] = row
    return rows


def _validate_paired_rows(rows, num_ssu):
    group = [row for (_, ssu), row in rows.items() if ssu == num_ssu]
    if len(group) < 2:
        return
    for field in (
        "assignment_hash",
        "workload_hash",
        "placement_hash",
        "trace_hash",
        "simulator_input_fingerprint",
    ):
        if len({row[field] for row in group}) != 1:
            raise AssertionError(f"unpaired {field} at SSU={num_ssu}")


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _category_slo(request_rows):
    result = {}
    for category in CATEGORIES:
        rows = [row for row in request_rows if row["category"] == category]
        result[category] = (
            statistics.fmean(bool(row["slo_met"]) for row in rows)
            if rows
            else None
        )
    return result


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def render_markdown(payload):
    lines = [
        "# 128-NPU warm/full-load admission experiment",
        "",
        "All rows use seed 42, 16 layers, batch 1, four warmup completions per "
        "NPU, 500-ms settle, and the same 2,000-ms measurement window. Admission "
        "25/50 ms denotes event-gated minimum spacing, not a periodic timer.",
        "",
        f"Complete: `{str(payload['complete']).lower()}`; selected complete: "
        f"`{str(payload['selected_complete']).lower()}`; source stable: "
        f"`{str(payload['source_stable_during_run']).lower()}`; config stable: "
        f"`{str(payload['config_stable_during_run']).lower()}`.",
        "",
        "| SSU | Strategy | NPU util min | p10 | mean | Equal-NPU SLO | "
        "Request SLO | SS | SL | LS | LL | Requests | Evals | Commits | "
        "Path writes | Wall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        summary = row["steady_summary"]
        utils = summary["npu_utilizations"]
        category = _category_slo(summary["request_rows"])
        lines.append(
            "| {ssu} | {strategy} | {minimum} | {p10} | {mean} | {equal} | "
            "{request} | {SS} | {SL} | {LS} | {LL} | {count} | {evals} | "
            "{commits} | {writes} | {wall:.1f}s |".format(
                ssu=row["num_ssu"],
                strategy=row["strategy"],
                minimum=_pct(min(utils)),
                p10=_pct(_percentile(utils, 10)),
                mean=_pct(summary["mean_npu_utilization"]),
                equal=_pct(summary["ttft_slo_attainment"]),
                request=_pct(summary["request_weighted_slo_attainment"]),
                count=summary["measurement_request_count"],
                evals=summary["control_evaluations"],
                commits=summary["cir_commits"],
                writes=summary["cir_path_writes"],
                wall=row["wall_time_s"],
                **{name: _pct(category[name]) for name in CATEGORIES},
            )
        )
    return "\n".join(lines) + "\n"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _write_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_markdown(payload))
    temporary.replace(path)


def run(
    output=OUTPUT,
    report=REPORT,
    *,
    workers=4,
    strategies=(),
    ssu_list=(),
    rerun=False,
):
    run_fingerprint = _source_fingerprint()
    spec = experiment_spec()
    config_fingerprint = _config_fingerprint(spec)
    wanted_strategies = tuple(strategies) or tuple(case.name for case in CASES)
    wanted_ssus = tuple(int(value) for value in ssu_list) or SSU_LIST
    rows = {}
    if output.exists() and not rerun:
        rows = _validate_cached_rows(
            json.loads(output.read_text()),
            run_fingerprint,
            spec,
            config_fingerprint,
        )

    tasks = [
        (CASE_BY_NAME[strategy], num_ssu, run_fingerprint, config_fingerprint)
        for num_ssu in wanted_ssus
        for strategy in wanted_strategies
        if (strategy, num_ssu) not in rows
    ]
    tasks.sort(
        key=lambda task: task[1] * RUNTIME_WEIGHTS[task[0].name], reverse=True
    )
    source_stable = True
    config_stable = True

    def checkpoint():
        nonlocal source_stable, config_stable
        for num_ssu in SSU_LIST:
            _validate_paired_rows(rows, num_ssu)
        ending_source = _source_fingerprint()
        ending_config = _config_fingerprint(experiment_spec())
        source_stable = source_stable and ending_source == run_fingerprint
        config_stable = config_stable and ending_config == config_fingerprint
        ordered = [
            rows[(case.name, num_ssu)]
            for num_ssu in SSU_LIST
            for case in CASES
            if (case.name, num_ssu) in rows
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": source_stable
            and config_stable
            and len(ordered) == len(CASES) * len(SSU_LIST),
            "selected_complete": source_stable
            and config_stable
            and all(
                (strategy, num_ssu) in rows
                for num_ssu in wanted_ssus
                for strategy in wanted_strategies
            ),
            "all_cases_finished": len(ordered) == len(CASES) * len(SSU_LIST),
            "source_stable_during_run": source_stable,
            "config_stable_during_run": config_stable,
            "source_fingerprint": run_fingerprint,
            "ending_source_fingerprint": ending_source,
            "config_fingerprint": config_fingerprint,
            "ending_config_fingerprint": ending_config,
            "experiment_spec": spec,
            "selected_strategies": list(wanted_strategies),
            "selected_ssu_list": list(wanted_ssus),
            "results": ordered,
        }
        _write_json(output, payload)
        _write_report(report, payload)

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
            futures = {pool.submit(_run_case, task): task[:2] for task in tasks}
            for future in as_completed(futures):
                case, num_ssu = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    print(
                        f"FAILED {case.name} ssu={num_ssu}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                rows[_key(row)] = row
                checkpoint()
                summary = row["steady_summary"]
                print(
                    f"{case.name} ssu={num_ssu}: "
                    f"util={summary['mean_npu_utilization']:.2%}, "
                    f"SLO={summary['ttft_slo_attainment']:.2%}, "
                    f"requests={summary['measurement_request_count']}, "
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
        raise RuntimeError("128-NPU experiment source changed during the run")
    if not config_stable:
        raise RuntimeError("128-NPU experiment configuration changed during the run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--ssu", action="append", type=int, choices=SSU_LIST)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output,
        args.report,
        workers=args.workers,
        strategies=tuple(args.case or ()),
        ssu_list=tuple(args.ssu or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
