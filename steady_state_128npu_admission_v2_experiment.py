"""Isolated 128-NPU real-data experiment for explicit-tail admission V2.

The default matrix runs only ``admission_v2_25ms`` at SSU24/40.  SSU70 is an
optional CLI point.  The runner has its own result directory, source/config/case
fingerprints, resumable checkpoints, and retains each complete simulator
summary.  It does not modify or write the frozen V1 experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import sim
import steady_state_128npu_admission_experiment as v1_experiment
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
from slo_admission_scheme_b_v2 import SLOAdmissionSchemeBControllerV2
from steady_state_workload import prepare_steady_state_workload


ROOT = v1_experiment.ROOT
OUTPUT_DIR = ROOT / "results" / "steady_state_128npu_admission_v2"
OUTPUT = OUTPUT_DIR / "results.json"
REPORT = OUTPUT_DIR / "report.md"
SCHEMA_VERSION = 1
DEFAULT_SSU_LIST = (24, 40)
OPTIONAL_SSU_LIST = (70,)
SSU_LIST = DEFAULT_SSU_LIST + OPTIONAL_SSU_LIST
MIN_INTERVAL_MS = 25.0
CASE = v1_experiment.Case(
    "admission_v2_25ms", "slo_admission_v2", MIN_INTERVAL_MS
)
CASES = (CASE,)
CASE_BY_NAME = {case.name: case for case in CASES}
CATEGORIES = v1_experiment.CATEGORIES
NUM_NPU = v1_experiment.NUM_NPU
N_LAYERS = v1_experiment.N_LAYERS
REQUESTS_PER_NPU = v1_experiment.REQUESTS_PER_NPU
SEED = v1_experiment.SEED
STEADY_CONFIG = v1_experiment.STEADY_CONFIG
TARGET_RATIO = v1_experiment.TARGET_RATIO
REQUIRED_RATIO = v1_experiment.REQUIRED_RATIO
BACKGROUND_RESERVE_FRACTION = v1_experiment.BACKGROUND_RESERVE_FRACTION
SSD_CAP_GBPS = v1_experiment.SSD_CAP_GBPS
NPU_CAP_GBPS = v1_experiment.NPU_CAP_GBPS
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *v1_experiment.SOURCE_FILES,
            "slo_admission_scheme_b_v2.py",
            Path(__file__).name,
        )
    )
)
_WORKER_TABLE = None


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-128npu-admission-v2\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "isolated_explicit_tail_admission_v2_real_data",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "default_ssu_list": list(DEFAULT_SSU_LIST),
        "optional_ssu_list": list(OPTIONAL_SSU_LIST),
        "strategies": [asdict(case) for case in CASES],
        "seed": SEED,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "steady_state": asdict(STEADY_CONFIG),
        "controller": {
            "name": "SLOAdmissionSchemeBControllerV2",
            "target_ratio": TARGET_RATIO,
            "required_ratio": REQUIRED_RATIO,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_CAP_GBPS,
            "allocation_stages": [
                "selected_floor",
                "rejected_background_coflow",
                "selected_absolute_tail",
                "all_request_absolute_spill",
            ],
        },
        "admission_control": {
            "trigger": "batch_membership_event",
            "on_batch_boundary": True,
            "interval_ms": None,
            "min_intervals_ms": [MIN_INTERVAL_MS],
            "semantics": "event-gated minimum spacing, not periodic ticks",
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash, reused by all 16 layers",
        "measurement_cohort": (
            "requests admitted in the half-open common measurement window; "
            "all tagged requests drain while all NPUs remain backlogged"
        ),
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
        "real_data": True,
        "parent_v1_matrix_unchanged": True,
        "result_directory": str(OUTPUT_DIR.relative_to(ROOT)),
    }


def _config_fingerprint(spec=None):
    return _canonical_hash(
        spec or experiment_spec(), b"128npu-admission-v2-config-v1\0"
    )


def _case_fingerprint(case, num_ssu, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "num_ssu": int(num_ssu),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"128npu-admission-v2-case-v1\0",
    )


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _dedicated_paths():
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _simulate(case, requests, *, num_ssu):
    if case.kind != "slo_admission_v2":
        raise ValueError(f"unknown V2 case kind: {case.kind}")
    paths = _dedicated_paths()
    controller = SLOAdmissionSchemeBControllerV2(
        paths,
        target_ratio=TARGET_RATIO,
        required_ratio=REQUIRED_RATIO,
        background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
        ssd_cap_gbps=SSD_CAP_GBPS,
        npu_cap_gbps=NPU_CAP_GBPS,
    )
    return simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * num_ssu
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
        num_ssu=num_ssu,
        n_layers=N_LAYERS,
        batch_size=1,
        submit_order_seed=SEED,
        cross_request_layer0_prefetch=True,
        steady_state=STEADY_CONFIG,
    )


def _validate_summary(case, num_ssu, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("V2 runner returned the wrong simulator mode")
    if int(summary["num_npu"]) != NUM_NPU or int(summary["num_ssu"]) != num_ssu:
        raise AssertionError("V2 runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("V2 admission must not read Path pressure")
    if any(
        summary[name] <= 0
        for name in ("control_evaluations", "cir_commits", "cir_path_writes")
    ):
        raise AssertionError("V2 did not evaluate and commit CIR")
    if not math.isclose(
        float(summary["control_min_interval_ms"]),
        float(case.min_interval_ms),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("V2 used the wrong minimum interval")


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
        input_fingerprints = {
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
            "num_ssu": num_ssu,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, num_ssu, source_fingerprint, config_fingerprint
            ),
            "input_fingerprints": input_fingerprints,
            "assignment_hash": input_fingerprints["assignment"],
            "workload_hash": input_fingerprints["workload"],
            "placement_hash": input_fingerprints["placement"],
            "trace_hash": input_fingerprints["trace"],
            "simulator_input_fingerprint": input_fingerprints["simulator"],
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
        if set(row.get("input_fingerprints", ())) != required_inputs:
            return {}
        _validate_summary(case, num_ssu, row["steady_summary"])
        rows[(strategy, num_ssu)] = row
    return rows


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
        "# 128-NPU real-data explicit-tail admission V2",
        "",
        "This isolated run uses the frozen seed-42 real-data trace, 16 layers, "
        "batch 1, four warmup completions per NPU, 500-ms settle, and the same "
        "2,000-ms measurement window. The 25-ms controller is event-gated "
        "minimum spacing, not a periodic timer.",
        "",
        f"Default complete: `{str(payload['complete']).lower()}`; optional SSU70 "
        f"complete: `{str(payload['optional_complete']).lower()}`; selected "
        f"complete: `{str(payload['selected_complete']).lower()}`; source stable: "
        f"`{str(payload['source_stable_during_run']).lower()}`; config stable: "
        f"`{str(payload['config_stable_during_run']).lower()}`.",
        "",
        f"Source fingerprint: `{payload['source_fingerprint']}`",
        "",
        f"Config fingerprint: `{payload['config_fingerprint']}`",
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
    if payload["results"]:
        lines.extend(["", "## Input fingerprints by SSU", ""])
        for row in payload["results"]:
            fields = ", ".join(
                f"{name}=`{value}`"
                for name, value in row["input_fingerprints"].items()
            )
            lines.append(f"- SSU{row['num_ssu']}: {fields}")
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
    workers=2,
    strategies=(),
    ssu_list=(),
    rerun=False,
):
    run_fingerprint = _source_fingerprint()
    spec = experiment_spec()
    config_fingerprint = _config_fingerprint(spec)
    wanted_strategies = tuple(strategies) or tuple(case.name for case in CASES)
    wanted_ssus = tuple(int(value) for value in ssu_list) or DEFAULT_SSU_LIST
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
    tasks.sort(key=lambda task: task[1], reverse=True)
    source_stable = True
    config_stable = True

    def checkpoint():
        nonlocal source_stable, config_stable
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
            and all(
                (case.name, num_ssu) in rows
                for num_ssu in DEFAULT_SSU_LIST
                for case in CASES
            ),
            "optional_complete": source_stable
            and config_stable
            and all(
                (case.name, num_ssu) in rows
                for num_ssu in OPTIONAL_SSU_LIST
                for case in CASES
            ),
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
        raise RuntimeError("128-NPU V2 experiment source changed during the run")
    if not config_stable:
        raise RuntimeError("128-NPU V2 experiment configuration changed during the run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=2)
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

