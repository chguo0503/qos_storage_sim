"""Independent paired SSU16 run for the 128-NPU real-data experiment.

This narrow runner intentionally does not alter the frozen SSU24/40/70 matrix
or its result directory.  It compares baseline with the optimized 25-ms
event-gated SLO-admission policy at SSU16, while reusing the exact same
real-data workload builder and simulator wiring as the main 128-NPU runner.

Every checkpoint contains the complete simulator summaries plus independent
source, configuration, case, workload, placement, trace, assignment, and
simulator-input fingerprints.  A cached row is reused only when all frozen
experiment provenance still matches.
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
import steady_state_128npu_admission_experiment as main_experiment


ROOT = main_experiment.ROOT
OUTPUT_DIR = ROOT / "results" / "steady_state_128npu_admission_ssu16"
OUTPUT = OUTPUT_DIR / "results.json"
REPORT = OUTPUT_DIR / "report.md"
SCHEMA_VERSION = 1
NUM_SSU = 16
CASES = (
    main_experiment.CASE_BY_NAME["baseline"],
    main_experiment.CASE_BY_NAME["admission_25ms"],
)
CASE_BY_NAME = {case.name: case for case in CASES}
SOURCE_FILES = tuple(
    dict.fromkeys(
        (*main_experiment.SOURCE_FILES, Path(__file__).name)
    )
)


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-128npu-admission-ssu16-v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    spec = dict(main_experiment.experiment_spec())
    spec.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": "independent_ssu16_real_data_pair",
            "ssu_list": [NUM_SSU],
            "strategies": [asdict(case) for case in CASES],
            "parent_matrix_unchanged": True,
            "result_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        }
    )
    admission = dict(spec["admission_control"])
    admission["min_intervals_ms"] = [25.0]
    spec["admission_control"] = admission
    return spec


def _config_fingerprint(spec=None):
    return _canonical_hash(
        spec or experiment_spec(), b"128npu-admission-ssu16-config-v1\0"
    )


def _case_fingerprint(case, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "num_ssu": NUM_SSU,
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"128npu-admission-ssu16-case-v1\0",
    )


def _init_worker():
    main_experiment._init_worker()


def _simulate(case, requests):
    return main_experiment._simulate(case, requests, num_ssu=NUM_SSU)


def _validate_summary(case, summary):
    main_experiment._validate_summary(case, NUM_SSU, summary)


def _run_case(task):
    case, source_fingerprint, config_fingerprint = task
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.name} ssu={NUM_SSU}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        table = main_experiment._WORKER_TABLE or sim.load_bw_table_cache(
            num_npu=main_experiment.NUM_NPU
        )
        workload = main_experiment.prepare_steady_state_workload(
            table,
            num_npu=main_experiment.NUM_NPU,
            num_ssu=NUM_SSU,
            n_layers=main_experiment.N_LAYERS,
            requests_per_npu=main_experiment.REQUESTS_PER_NPU,
            seed=main_experiment.SEED,
        )
        requests = main_experiment.requests_from_continuous_prefill_workload(
            workload
        )
        summary = _simulate(case, requests)
        _validate_summary(case, summary)
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
            "num_ssu": NUM_SSU,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, source_fingerprint, config_fingerprint
            ),
            "input_fingerprints": input_fingerprints,
            # Retain the flat names used by the main matrix for easy joins.
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


def _validate_paired_rows(rows):
    if len(rows) < 2:
        return
    for name in (
        "assignment",
        "workload",
        "placement",
        "trace",
        "simulator",
    ):
        if len({row["input_fingerprints"][name] for row in rows.values()}) != 1:
            raise AssertionError(f"unpaired {name} input fingerprint at SSU16")


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
    required_input_names = {
        "assignment", "workload", "placement", "trace", "simulator"
    }
    for row in cached.get("results", ()):
        try:
            strategy = str(row["strategy"])
            case = CASE_BY_NAME[strategy]
            num_ssu = int(row["num_ssu"])
        except (KeyError, TypeError, ValueError):
            return {}
        if num_ssu != NUM_SSU or strategy in rows:
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
        if set(row.get("input_fingerprints", ())) != required_input_names:
            return {}
        _validate_summary(case, row["steady_summary"])
        rows[strategy] = row
    _validate_paired_rows(rows)
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
    for category in main_experiment.CATEGORIES:
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
        "# 128-NPU real-data paired experiment: SSU16",
        "",
        "This independent run compares baseline with admission_25ms using "
        "the same seed-42 real-data trace, 16-layer workload, placement, warmup, "
        "settle period, and 2,000-ms measurement window. The 25-ms setting is "
        "event-gated minimum spacing, not a periodic timer.",
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
        "| Strategy | NPU util min | p10 | mean | Equal-NPU SLO | Request "
        "SLO | SS | SL | LS | LL | Requests | Evals | Commits | Path writes | Wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        summary = row["steady_summary"]
        utils = summary["npu_utilizations"]
        category = _category_slo(summary["request_rows"])
        lines.append(
            "| {strategy} | {minimum} | {p10} | {mean} | {equal} | "
            "{request} | {SS} | {SL} | {LS} | {LL} | {count} | {evals} | "
            "{commits} | {writes} | {wall:.1f}s |".format(
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
                **{
                    name: _pct(category[name])
                    for name in main_experiment.CATEGORIES
                },
            )
        )
    if payload["results"]:
        fingerprints = payload["results"][0]["input_fingerprints"]
        lines.extend(["", "## Paired input fingerprints", ""])
        for name, value in fingerprints.items():
            lines.append(f"- {name}: `{value}`")
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
    rerun=False,
):
    run_fingerprint = _source_fingerprint()
    spec = experiment_spec()
    config_fingerprint = _config_fingerprint(spec)
    wanted_strategies = tuple(strategies) or tuple(
        case.name for case in CASES
    )
    rows = {}
    if output.exists() and not rerun:
        rows = _validate_cached_rows(
            json.loads(output.read_text()),
            run_fingerprint,
            spec,
            config_fingerprint,
        )

    tasks = [
        (CASE_BY_NAME[strategy], run_fingerprint, config_fingerprint)
        for strategy in wanted_strategies
        if strategy not in rows
    ]
    source_stable = True
    config_stable = True

    def checkpoint():
        nonlocal source_stable, config_stable
        _validate_paired_rows(rows)
        ending_source = _source_fingerprint()
        ending_config = _config_fingerprint(experiment_spec())
        source_stable = source_stable and ending_source == run_fingerprint
        config_stable = config_stable and ending_config == config_fingerprint
        ordered = [case.name for case in CASES if case.name in rows]
        result_rows = [rows[name] for name in ordered]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": source_stable
            and config_stable
            and len(result_rows) == len(CASES),
            "selected_complete": source_stable
            and config_stable
            and all(name in rows for name in wanted_strategies),
            "all_cases_finished": len(result_rows) == len(CASES),
            "source_stable_during_run": source_stable,
            "config_stable_during_run": config_stable,
            "source_fingerprint": run_fingerprint,
            "ending_source_fingerprint": ending_source,
            "config_fingerprint": config_fingerprint,
            "ending_config_fingerprint": ending_config,
            "experiment_spec": spec,
            "selected_strategies": list(wanted_strategies),
            "input_fingerprints": (
                result_rows[0]["input_fingerprints"] if result_rows else None
            ),
            "results": result_rows,
        }
        _write_json(output, payload)
        _write_report(report, payload)

    if workers == 1:
        _init_worker()
        for task in tasks:
            row = _run_case(task)
            rows[row["strategy"]] = row
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
                        f"FAILED {case.name} ssu={NUM_SSU}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    raise
                rows[row["strategy"]] = row
                checkpoint()
                summary = row["steady_summary"]
                print(
                    f"{case.name} ssu={NUM_SSU}: "
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
        raise RuntimeError("SSU16 experiment source changed during the run")
    if not config_stable:
        raise RuntimeError("SSU16 experiment configuration changed during the run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", action="append", choices=tuple(CASE_BY_NAME))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output,
        args.report,
        workers=args.workers,
        strategies=tuple(args.case or ()),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
