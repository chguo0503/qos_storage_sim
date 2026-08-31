"""32-NPU real-trace sensitivity experiment for deadline Scheme B V3.

This runner is intentionally isolated from every frozen V1/V2/128-NPU
experiment.  It reuses the exact 32-NPU workload and steady window so its rows
pair directly with the existing baseline by workload/input fingerprints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    SteadyStateInvariantError,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import qos_configs_from_path_cirs, scheme_b_client_config
from deadline_barrier_scheme_b_v3 import DeadlineBarrierSchemeBControllerV3
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from six_request_workload import SEED
from steady_state_32npu_experiment import N_LAYERS, NUM_NPU, ROOT, STEADY_CONFIG
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_deadline_v3"
OUTPUT = OUTPUT_DIR / "results.json"
REPORT = OUTPUT_DIR / "report.md"
SCHEMA_VERSION = 1
TARGET_SSUS = (10, 18)
BACKGROUND_RESERVE_FRACTION = 0.05
PREFETCH_TARGET_RATIO = 0.65
HYSTERESIS_GBPS = 0.25
_WORKER_TABLE = None

REQUIRED_STEADY_KEYS = frozenset(
    {
        "mode",
        "num_npu",
        "num_ssu",
        "invariants",
        "pressure_reports",
        "control_evaluations",
        "control_min_interval_ms",
        "cir_commits",
        "cir_path_writes",
        "mean_npu_utilization",
        "ttft_slo_attainment",
        "request_weighted_slo_attainment",
        "measurement_request_count",
        "request_rows",
        "input_fingerprint",
    }
)


@dataclass(frozen=True)
class V3Case:
    name: str
    num_ssu: int
    interval_ms: float
    safety_margin: float
    waiting_boost_ratio: float


CASES = (
    V3Case("deadline_v3_25ms", 10, 25.0, 1.15, 0.65),
    V3Case("deadline_v3_wait55_25ms", 10, 25.0, 1.15, 0.55),
    V3Case("deadline_v3_50ms", 10, 50.0, 1.15, 0.65),
    V3Case("deadline_v3_25ms", 18, 25.0, 1.15, 0.65),
)
CASE_BY_KEY = {(case.name, case.num_ssu): case for case in CASES}

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
    "deadline_barrier_scheme_b_v3.py",
    "steady_state_32npu_deadline_v3_experiment.py",
    "data",
)


def _canonical_hash(payload, seed=b""):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(seed + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"steady-state-32npu-deadline-v3-v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix": [asdict(case) for case in CASES],
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "seed": SEED,
        "steady_state": asdict(STEADY_CONFIG),
        "controller": {
            "name": "DeadlineBarrierSchemeBControllerV3",
            "slo_alpha": STEADY_CONFIG.slo_alpha,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "prefetch_target_ratio": PREFETCH_TARGET_RATIO,
            "hysteresis_gbps": HYSTERESIS_GBPS,
            "ssd_cap_gbps": sim.DISK_BW,
            "npu_cap_gbps": sim.NPU_BW_LIMIT,
        },
        "control": {
            "trigger": "periodic_only",
            "on_batch_boundary": False,
            "interval_equals_minimum": True,
            "minimum_interval_ms": min(case.interval_ms for case in CASES),
        },
        "cross_request_layer0_prefetch": True,
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
    }


def _config_fingerprint(spec=None):
    return _canonical_hash(spec or experiment_spec(), b"deadline-v3-config-v1\0")


def _case_fingerprint(case, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case": asdict(case),
        },
        b"deadline-v3-case-v1\0",
    )


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _paths():
    return tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))


def _simulate(case, requests):
    paths = _paths()
    controller = DeadlineBarrierSchemeBControllerV3(
        paths,
        slo_alpha=STEADY_CONFIG.slo_alpha,
        safety_margin=case.safety_margin,
        waiting_boost_ratio=case.waiting_boost_ratio,
        prefetch_target_ratio=PREFETCH_TARGET_RATIO,
        background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
        min_decision_interval_ms=case.interval_ms,
        hysteresis_gbps=HYSTERESIS_GBPS,
        ssd_cap_gbps=sim.DISK_BW,
        npu_cap_gbps=sim.NPU_BW_LIMIT,
    )
    summary = simulate_continuous_batch(
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
            interval_ms=case.interval_ms,
            on_batch_boundary=False,
            min_interval_ms=case.interval_ms,
        ),
        submit_order_seed=SEED,
        cross_request_layer0_prefetch=True,
        steady_state=STEADY_CONFIG,
    )
    return summary, {
        "evaluations": controller.evaluations,
        "decisions": controller.decisions,
        "rate_limit_skips": controller.rate_limit_skips,
        "hysteresis_skips": controller.hysteresis_skips,
    }


def _validate_summary(case, summary):
    missing = REQUIRED_STEADY_KEYS - set(summary)
    if missing:
        raise AssertionError(f"V3 steady summary is missing keys: {sorted(missing)}")
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("V3 runner returned the wrong simulator mode")
    if int(summary["num_npu"]) != NUM_NPU or int(summary["num_ssu"]) != case.num_ssu:
        raise AssertionError("V3 runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    if summary["pressure_reports"] != 0:
        raise AssertionError("V3 must not read Path pressure")
    if summary["control_evaluations"] <= 0:
        raise AssertionError("V3 was never evaluated")
    if summary["cir_commits"] <= 0 or summary["cir_path_writes"] <= 0:
        raise AssertionError("V3 never committed a CIR target")
    if not math.isclose(
        float(summary["control_min_interval_ms"]),
        case.interval_ms,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("V3 used the wrong minimum interval")


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
        try:
            summary, controller_counters = _simulate(case, requests)
            _validate_summary(case, summary)
        except SteadyStateInvariantError as error:
            return {
                "status": "scientific_invalid",
                "strategy": case.name,
                "num_ssu": case.num_ssu,
                "case_spec": asdict(case),
                "source_fingerprint": source_fingerprint,
                "config_fingerprint": config_fingerprint,
                "case_fingerprint": _case_fingerprint(
                    case, source_fingerprint, config_fingerprint
                ),
                "wall_time_s": time.perf_counter() - started,
                "invariants": error.invariants,
                "diagnostics": error.diagnostics,
            }
        return {
            "status": "ok",
            "strategy": case.name,
            "num_ssu": case.num_ssu,
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
            "controller_counters": controller_counters,
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _baseline_rows():
    path = ROOT / "results" / "steady_state_32npu_normalized_slo" / "results.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        int(row["num_ssu"]): row
        for row in payload.get("results", ())
        if row.get("strategy") == "baseline"
        and int(row.get("num_ssu", -1)) in TARGET_SSUS
    }


def _report(payload):
    baselines = _baseline_rows()
    lines = [
        "# 32-NPU deadline/barrier Scheme B V3",
        "",
        "真实 `data`、batch=1、16 层复用固定 placement；所有 V3 行使用周期控制，且设置最小间隔与周期相同。",
        "",
        "| SSU | Case | Interval | Margin | Wait floor | NPU util | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Controls | Commits | Writes | Wall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        payload["results"], key=lambda item: (item["num_ssu"], item["strategy"])
    ):
        if row["status"] != "ok":
            lines.append(
                f"| {row['num_ssu']} | {row['strategy']} | — | — | — | INVALID | — | — | — | — | — | — | — | — | — | {row['wall_time_s']:.1f}s |"
            )
            continue
        summary = row["steady_summary"]
        category = {}
        for name in ("SS", "SL", "LS", "LL"):
            samples = [
                request
                for request in summary["request_rows"]
                if request["category"] == name
            ]
            category[name] = (
                100.0 * sum(bool(request["slo_met"]) for request in samples) / len(samples)
                if samples
                else 0.0
            )
        spec = row["case_spec"]
        lines.append(
            f"| {row['num_ssu']} | {row['strategy']} | {spec['interval_ms']:g} ms | "
            f"{spec['safety_margin']:.2f} | {spec['waiting_boost_ratio']:.2f} | "
            f"{100*summary['mean_npu_utilization']:.2f}% | "
            f"{100*summary['ttft_slo_attainment']:.2f}% | "
            f"{100*summary['request_weighted_slo_attainment']:.2f}% | "
            f"{category['SS']:.2f}% | {category['SL']:.2f}% | "
            f"{category['LS']:.2f}% | {category['LL']:.2f}% | "
            f"{summary['control_evaluations']} | {summary['cir_commits']} | "
            f"{summary['cir_path_writes']} | {row['wall_time_s']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## 与同输入 baseline 的差值",
            "",
            "| SSU | Case | Paired | NPU util | Equal-NPU SLO | Request SLO |",
            "|---:|---|:---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        (row for row in payload["results"] if row["status"] == "ok"),
        key=lambda item: (item["num_ssu"], item["strategy"]),
    ):
        baseline = baselines.get(row["num_ssu"])
        if baseline is None:
            continue
        paired = all(
            row[field] == baseline[field]
            for field in (
                "assignment_hash",
                "workload_hash",
                "placement_hash",
                "trace_hash",
                "simulator_input_fingerprint",
            )
        )
        summary = row["steady_summary"]
        base = baseline["steady_summary"]
        lines.append(
            f"| {row['num_ssu']} | {row['strategy']} | {'yes' if paired else 'NO'} | "
            f"{100*(summary['mean_npu_utilization']-base['mean_npu_utilization']):+.2f} pp | "
            f"{100*(summary['ttft_slo_attainment']-base['ttft_slo_attainment']):+.2f} pp | "
            f"{100*(summary['request_weighted_slo_attainment']-base['request_weighted_slo_attainment']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "`Paired=yes` 要求 assignment/workload/placement/trace/simulator input 五个指纹全部相同。",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))


def _checkpoint(rows, source_fingerprint, spec, config_fingerprint):
    ending_source = _source_fingerprint()
    ending_config = _config_fingerprint()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": len(rows) == len(CASES),
        "source_fingerprint": source_fingerprint,
        "ending_source_fingerprint": ending_source,
        "source_stable_during_run": ending_source == source_fingerprint,
        "config_fingerprint": config_fingerprint,
        "ending_config_fingerprint": ending_config,
        "config_stable_during_run": ending_config == config_fingerprint,
        "experiment_spec": spec,
        "results": sorted(
            rows.values(), key=lambda row: (row["num_ssu"], row["strategy"])
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")))
    _report(payload)
    return payload


def run(*, workers=4, selected_names=()):
    source_fingerprint = _source_fingerprint()
    spec = experiment_spec()
    config_fingerprint = _config_fingerprint(spec)
    selected = tuple(
        case for case in CASES if not selected_names or case.name in selected_names
    )
    rows = {}
    tasks = tuple((case, source_fingerprint, config_fingerprint) for case in selected)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_run_case, task): task[0] for task in tasks}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[(case.name, case.num_ssu)] = row
            _checkpoint(rows, source_fingerprint, spec, config_fingerprint)
            if row["status"] == "ok":
                summary = row["steady_summary"]
                print(
                    f"DONE {case.name} ssu={case.num_ssu}: "
                    f"util={100*summary['mean_npu_utilization']:.2f}% "
                    f"slo={100*summary['ttft_slo_attainment']:.2f}%",
                    flush=True,
                )
            else:
                print(f"INVALID {case.name} ssu={case.num_ssu}", flush=True)
    payload = _checkpoint(rows, source_fingerprint, spec, config_fingerprint)
    if not payload["source_stable_during_run"]:
        raise RuntimeError("V3 source fingerprint changed during execution")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    run(workers=args.workers, selected_names=tuple(args.case))


if __name__ == "__main__":
    main()
