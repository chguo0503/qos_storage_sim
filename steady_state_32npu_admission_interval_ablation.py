"""Admission-controller minimum-interval ablation on the 32-NPU warm trace.

The matrix is intentionally independent from both the primary experiment and
``steady_state_32npu_diagnostics.py``.  It changes only the event-gated
``min_interval_ms`` of the final SLO-admission controller: 25/50/100 ms at
SSU10 and SSU18.  Source/config fingerprints are frozen before workers launch,
stored on every row, checked before cache reuse, and checked again at run end.
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
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import SEED
from slo_admission_scheme_b import SLOAdmissionSchemeBController
from steady_state_32npu_experiment import (
    N_LAYERS,
    NUM_NPU,
    ROOT,
    STEADY_CONFIG,
)
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_normalized_slo"
OUTPUT = OUTPUT_DIR / "admission_interval_ablation.json"
REPORT = OUTPUT_DIR / "admission_interval_ablation.md"
SCHEMA_VERSION = 1
SSU_LIST = (10, 18)
MIN_INTERVALS_MS = (25.0, 50.0, 100.0)
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
    "steady_state_32npu_experiment.py",
    "scheme_b_prefill.py",
    "slo_admission_scheme_b.py",
    "steady_state_32npu_admission_interval_ablation.py",
    "data",
)


@dataclass(frozen=True)
class IntervalCase:
    num_ssu: int
    min_interval_ms: float

    @property
    def name(self):
        return f"admission_{self.min_interval_ms:g}ms"


CASES = tuple(
    IntervalCase(num_ssu, interval)
    for num_ssu in SSU_LIST
    for interval in MIN_INTERVALS_MS
)
CASE_BY_KEY = {
    (case.num_ssu, case.min_interval_ms): case for case in CASES
}


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-32npu-admission-interval-v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix": [asdict(case) for case in CASES],
        "num_npu": NUM_NPU,
        "num_ssu": list(SSU_LIST),
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "seed": SEED,
        "steady_state": asdict(STEADY_CONFIG),
        "controller": {
            "name": "SLOAdmissionSchemeBController",
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
            "min_intervals_ms": list(MIN_INTERVALS_MS),
            "semantics": "event-gated minimum spacing, not periodic ticks",
        },
        "cross_request_layer0_prefetch": True,
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
    }


def _config_fingerprint(spec=None):
    return _canonical_hash(
        spec or experiment_spec(), b"admission-interval-config-v1\0"
    )


def _case_fingerprint(case, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case": asdict(case),
        },
        b"admission-interval-case-v1\0",
    )


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _paths():
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _simulate(case, requests):
    paths = _paths()
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
        num_npu=NUM_NPU,
        num_ssu=case.num_ssu,
        n_layers=N_LAYERS,
        batch_size=1,
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
        submit_order_seed=SEED,
        cross_request_layer0_prefetch=True,
        steady_state=STEADY_CONFIG,
    )


def _validate_summary(case, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("interval ablation returned the wrong simulator mode")
    if int(summary["num_npu"]) != NUM_NPU or int(summary["num_ssu"]) != case.num_ssu:
        raise AssertionError("interval ablation returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("admission interval ablation must not read pressure")
    if summary["control_evaluations"] <= 0:
        raise AssertionError("admission controller was never evaluated")
    if summary["cir_commits"] <= 0 or summary["cir_path_writes"] <= 0:
        raise AssertionError("admission controller never committed a CIR target")
    if not math.isclose(
        float(summary["control_min_interval_ms"]),
        case.min_interval_ms,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("simulator used the wrong minimum control interval")


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
        return {
            "status": "ok",
            "strategy": case.name,
            "num_ssu": case.num_ssu,
            "min_interval_ms": case.min_interval_ms,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, source_fingerprint, config_fingerprint
            ),
            "wall_time_s": time.perf_counter() - started,
            "assignment_hash": workload.statistics["assignment_hash"],
            "workload_hash": workload.workload_hash,
            "placement_hash": workload.placement_hash,
            "trace_hash": workload.trace_hash,
            "simulator_input_fingerprint": summary["input_fingerprint"],
            "controller_config": experiment_spec()["controller"],
            "control_config": {
                "on_batch_boundary": True,
                "interval_ms": None,
                "min_interval_ms": case.min_interval_ms,
            },
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _key_from_row(row):
    return int(row["num_ssu"]), float(row["min_interval_ms"])


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
            key = _key_from_row(row)
            case = CASE_BY_KEY[key]
        except (KeyError, TypeError, ValueError):
            return {}
        if key in rows or row.get("status") != "ok":
            return {}
        if row.get("case_spec") != asdict(case):
            return {}
        if row.get("source_fingerprint") != source_fingerprint:
            return {}
        if row.get("config_fingerprint") != config_fingerprint:
            return {}
        if row.get("case_fingerprint") != _case_fingerprint(
            case, source_fingerprint, config_fingerprint
        ):
            return {}
        _validate_summary(case, row["steady_summary"])
        rows[key] = row
    return rows


def _validate_paired_rows(rows, num_ssu):
    group = [row for (ssu, _), row in rows.items() if ssu == num_ssu]
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
    return "—" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(payload):
    spec = payload["experiment_spec"]
    lines = [
        "# 32-NPU SLO-admission minimum-interval ablation",
        "",
        "Only `min_interval_ms` changes across rows. Control remains "
        "batch-membership-event driven with `on_batch_boundary=True`; these are "
        "minimum spacings, not periodic timers.",
        "",
        f"Complete: `{str(payload['complete']).lower()}`; source stable: "
        f"`{str(payload['source_stable_during_run']).lower()}`; config stable: "
        f"`{str(payload['config_stable_during_run']).lower()}`.",
        "",
        "Controller: target={target:.2f}, required={required:.2f}, background "
        "reserve={reserve:.2%}, SSD cap={ssd:g} GB/s, NPU cap={npu:g} "
        "GB/s.".format(
            target=spec["controller"]["target_ratio"],
            required=spec["controller"]["required_ratio"],
            reserve=spec["controller"]["background_reserve_fraction"],
            ssd=spec["controller"]["ssd_cap_gbps"],
            npu=spec["controller"]["npu_cap_gbps"],
        ),
        "",
        "| SSU | Min interval | NPU util min | p10 | mean | Equal-NPU SLO | "
        "Request SLO | SS | SL | LS | LL | Requests | Evaluations | Commits | "
        "Path writes | Wall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        summary = row["steady_summary"]
        utils = summary["npu_utilizations"]
        category = _category_slo(summary["request_rows"])
        lines.append(
            "| {ssu} | {interval:g} ms | {minimum} | {p10} | {mean} | "
            "{equal} | {request} | {SS} | {SL} | {LS} | {LL} | {count} | "
            "{evaluations} | {commits} | {writes} | {wall:.1f}s |".format(
                ssu=row["num_ssu"],
                interval=row["min_interval_ms"],
                minimum=_pct(min(utils)),
                p10=_pct(_percentile(utils, 10)),
                mean=_pct(summary["mean_npu_utilization"]),
                equal=_pct(summary["ttft_slo_attainment"]),
                request=_pct(summary["request_weighted_slo_attainment"]),
                count=summary["measurement_request_count"],
                evaluations=summary["control_evaluations"],
                commits=summary["cir_commits"],
                writes=summary["cir_path_writes"],
                wall=row["wall_time_s"],
                **{name: _pct(category[name]) for name in CATEGORIES},
            )
        )

    if payload["results"]:
        lines.extend(
            [
                "",
                "## Delta from 25-ms row",
                "",
                "Percentage points; only SSUs with a completed 25-ms reference "
                "are shown.",
                "",
                "| SSU | Interval | Mean NPU util | Equal-NPU SLO | Request SLO |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        by_key = {_key_from_row(row): row for row in payload["results"]}
        for num_ssu in SSU_LIST:
            reference = by_key.get((num_ssu, 25.0))
            if reference is None:
                continue
            ref_summary = reference["steady_summary"]
            for interval in MIN_INTERVALS_MS:
                row = by_key.get((num_ssu, interval))
                if row is None:
                    continue
                summary = row["steady_summary"]
                lines.append(
                    "| {ssu} | {interval:g} ms | {util:+.2f} pp | "
                    "{equal:+.2f} pp | {request:+.2f} pp |".format(
                        ssu=num_ssu,
                        interval=interval,
                        util=100.0
                        * (
                            summary["mean_npu_utilization"]
                            - ref_summary["mean_npu_utilization"]
                        ),
                        equal=100.0
                        * (
                            summary["ttft_slo_attainment"]
                            - ref_summary["ttft_slo_attainment"]
                        ),
                        request=100.0
                        * (
                            summary["request_weighted_slo_attainment"]
                            - ref_summary["request_weighted_slo_attainment"]
                        ),
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


def run(output=OUTPUT, report=REPORT, *, workers=3, rerun=False):
    run_fingerprint = _source_fingerprint()
    spec = experiment_spec()
    config_fingerprint = _config_fingerprint(spec)
    rows = {}
    if output.exists() and not rerun:
        cached = json.loads(output.read_text())
        rows = _validate_cached_rows(
            cached, run_fingerprint, spec, config_fingerprint
        )

    tasks = [
        (case, run_fingerprint, config_fingerprint)
        for case in sorted(CASES, key=lambda value: value.num_ssu, reverse=True)
        if (case.num_ssu, case.min_interval_ms) not in rows
    ]
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
            rows[(case.num_ssu, case.min_interval_ms)]
            for case in CASES
            if (case.num_ssu, case.min_interval_ms) in rows
        ]
        complete = (
            source_stable
            and config_stable
            and len(ordered) == len(CASES)
            and all(row["status"] == "ok" for row in ordered)
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": complete,
            "all_cases_finished": len(ordered) == len(CASES),
            "source_stable_during_run": source_stable,
            "config_stable_during_run": config_stable,
            "source_fingerprint": run_fingerprint,
            "ending_source_fingerprint": ending_source,
            "config_fingerprint": config_fingerprint,
            "ending_config_fingerprint": ending_config,
            "experiment_spec": spec,
            "results": ordered,
        }
        _write_json(output, payload)
        _write_report(report, payload)

    if workers == 1:
        _init_worker()
        for task in tasks:
            row = _run_case(task)
            rows[_key_from_row(row)] = row
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
                rows[_key_from_row(row)] = row
                checkpoint()
                summary = row["steady_summary"]
                print(
                    f"{case.name} ssu={case.num_ssu}: "
                    f"util={summary['mean_npu_utilization']:.2%}, "
                    f"SLO={summary['ttft_slo_attainment']:.2%}, "
                    f"evals={summary['control_evaluations']}, "
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
        raise RuntimeError("ablation source changed during the run")
    if not config_stable:
        raise RuntimeError("ablation configuration changed during the run")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(args.output, args.report, workers=args.workers, rerun=args.rerun)


if __name__ == "__main__":
    main()
