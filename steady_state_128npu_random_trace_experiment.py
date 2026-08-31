"""Paired Baseline/layer-once/Adaptive experiment on a random data catalog trace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import sim
from adaptive_admission_scheme_b_v2_1 import (
    AdaptiveAdmissionSchemeBControllerV2_1,
)
from continuous_batch_sim import (
    CIRControlConfig,
    SteadyStateConfig,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from random_steady_state_workload import (
    STRATIFIED_RANDOM_CATALOG_V1,
    SteadyStateProfileSchedule,
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
NUM_NPU = 128
N_LAYERS = 16
SSU_LIST = (16, 20, 24, 28)
REQUESTS_PER_NPU = 64
SEED = 42
MIN_INTERVAL_MS = 25.0
EXPLICIT_SPILL_THRESHOLD = 0.75
TARGET_RATIO = 0.52
REQUIRED_RATIO = 0.50
BACKGROUND_RESERVE_FRACTION = 0.05
STEADY_CONFIG = SteadyStateConfig(
    warmup_requests_per_npu=4,
    settle_ms=500.0,
    measurement_ms=5_000.0,
    slo_alpha=2.0,
    block_ms=500.0,
)
SOURCE_FILES = (
    "sim.py",
    "policy_logic.py",
    "strategy_profiles.py",
    "continuous_batch_control.py",
    "continuous_batch_sim.py",
    "continuous_prefill_client.py",
    "continuous_prefill_workload.py",
    "six_request_workload.py",
    "random_steady_state_workload.py",
    "scheme_b_prefill.py",
    "slo_admission_scheme_b.py",
    "slo_admission_scheme_b_v2.py",
    "adaptive_admission_scheme_b_v2_1.py",
    Path(__file__).name,
    "data",
)


@dataclass(frozen=True)
class Case:
    strategy: str
    kind: str
    num_ssu: int


CASES = tuple(
    Case(strategy, kind, num_ssu)
    for num_ssu in SSU_LIST
    for strategy, kind in (
        ("baseline", "baseline"),
        ("layer_once", "layer_once"),
        ("adaptive_v2_1_25ms", "adaptive"),
    )
)
CASE_BY_KEY = {(case.strategy, case.num_ssu): case for case in CASES}

_WORKER_TABLE = None
_WORKER_SCHEDULE = None
_WORKER_REQUESTS = {}
_WORKER_WORKLOADS = {}


def _canonical_hash(value, namespace=b""):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _source_fingerprint():
    digest = hashlib.sha256(b"128npu-random-trace-three-strategy:v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def experiment_spec(schedule):
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "128npu_random_data_three_strategy_v1",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "ssu_list": list(SSU_LIST),
        "strategies": [asdict(case) for case in CASES],
        "workload": {
            "mode": schedule.mode,
            "seed": schedule.seed,
            "requests_per_npu_prefix": schedule.requests_per_npu,
            **schedule.as_fingerprint_dict(),
            "source": "all 84 profiles loaded from data",
            "category_policy": (
                "random order with exactly one SS/SL/LS/LL per NPU per four "
                "requests and exactly 32 of each category per fleet sequence"
            ),
            "profile_policy": "per-NPU per-category deterministic shuffle bag",
        },
        "steady_state": asdict(STEADY_CONFIG),
        "adaptive": {
            "name": "AdaptiveAdmissionSchemeBControllerV2_1",
            "min_interval_ms": MIN_INTERVAL_MS,
            "explicit_spill_threshold": EXPLICIT_SPILL_THRESHOLD,
            "target_ratio": TARGET_RATIO,
            "required_ratio": REQUIRED_RATIO,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "ssd_cap_gbps": sim.DISK_BW,
            "npu_cap_gbps": sim.NPU_BW_LIMIT,
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash reused across all 16 layers",
        "physical_data_plane": "SSD40 single-command -> per-NPU NPU50 FCFS",
        "measurement_cohort": (
            "requests admitted in the half-open common wall-time window; tagged "
            "requests drain while every NPU remains backlogged"
        ),
    }


def _config_fingerprint(spec):
    return _canonical_hash(spec, b"128npu-random-trace-config:v1\0")


def _case_fingerprint(case, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"128npu-random-trace-case:v1\0",
    )


def _init_worker(table, schedule):
    global _WORKER_TABLE, _WORKER_SCHEDULE, _WORKER_REQUESTS, _WORKER_WORKLOADS
    _WORKER_TABLE = table
    _WORKER_SCHEDULE = schedule
    _WORKER_REQUESTS = {}
    _WORKER_WORKLOADS = {}


def _workload_and_requests(num_ssu):
    if num_ssu not in _WORKER_REQUESTS:
        workload = prepare_random_steady_state_workload(
            _WORKER_TABLE,
            schedule=_WORKER_SCHEDULE,
            num_ssu=num_ssu,
            n_layers=N_LAYERS,
        )
        _WORKER_WORKLOADS[num_ssu] = workload
        _WORKER_REQUESTS[num_ssu] = requests_from_continuous_prefill_workload(
            workload
        )
    return _WORKER_WORKLOADS[num_ssu], _WORKER_REQUESTS[num_ssu]


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


def _simulate(case, requests):
    common = _common_args(case.num_ssu)
    if case.kind in ("baseline", "layer_once"):
        routing = next(
            spec
            for spec in routing_strategy_specs()
            if spec.name == case.strategy
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=routing.client_config(),
            **common,
        )
    if case.kind != "adaptive":
        raise ValueError(f"unknown strategy kind: {case.kind}")
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
    controller = AdaptiveAdmissionSchemeBControllerV2_1(
        paths,
        explicit_spill_threshold=EXPLICIT_SPILL_THRESHOLD,
        target_ratio=TARGET_RATIO,
        required_ratio=REQUIRED_RATIO,
        background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
        ssd_cap_gbps=sim.DISK_BW,
        npu_cap_gbps=sim.NPU_BW_LIMIT,
    )
    summary = simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * case.num_ssu
        ),
        npu_dedicated_paths=paths,
        layer0_path_id=None,
        client_io_config=scheme_b_client_config(case.strategy),
        control=CIRControlConfig(
            callback=controller,
            on_batch_boundary=True,
            min_interval_ms=MIN_INTERVAL_MS,
        ),
        **common,
    )
    result = dict(summary)
    result["adaptive_residual_mode_evaluations"] = dict(
        controller.residual_mode_evaluations
    )
    result["adaptive_last_selected_fraction"] = (
        controller.last_allocation.selected_fraction
        if controller.last_allocation is not None
        else None
    )
    return result


def _validate_summary(case, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("runner returned the wrong simulator mode")
    if (
        int(summary["num_npu"]) != NUM_NPU
        or int(summary["num_ssu"]) != case.num_ssu
    ):
        raise AssertionError("runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"steady-state invariant failure: {invariants}")
    pressure = int(summary["pressure_reports"])
    controls = (
        int(summary["control_evaluations"]),
        int(summary["cir_commits"]),
        int(summary["cir_path_writes"]),
    )
    if case.kind == "baseline":
        if pressure or any(controls):
            raise AssertionError("baseline unexpectedly used pressure/control")
    elif case.kind == "layer_once":
        if pressure <= 0 or any(controls):
            raise AssertionError("layer_once counters have the wrong semantics")
    else:
        if pressure or any(value <= 0 for value in controls):
            raise AssertionError("Adaptive counters have the wrong semantics")
        if not math.isclose(
            float(summary["control_min_interval_ms"]),
            MIN_INTERVAL_MS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("Adaptive used the wrong minimum interval")


def _cohort_profile_metrics(summary, schedule, table):
    profile_by_request = {
        request_id: profile_key
        for request_id, _, _, _, profile_key in schedule.assignments
    }
    category_rows = defaultdict(list)
    bandwidth_rows = defaultdict(list)
    profile_counts = Counter()
    for row in summary["request_rows"]:
        request_id = int(row["request_id"])
        profile_key = profile_by_request[request_id]
        profile_counts[profile_key] += 1
        category_rows[row["category"]].append(bool(row["slo_met"]))
        required_bw = float(table[profile_key][0])
        bandwidth_bin = (
            "le_50"
            if required_bw <= sim.NPU_BW_LIMIT
            else "gt_50_le_100"
            if required_bw <= 2.0 * sim.NPU_BW_LIMIT
            else "gt_100"
        )
        bandwidth_rows[bandwidth_bin].append(bool(row["slo_met"]))

    def summarize(rows):
        return {
            key: {
                "count": len(values),
                "slo_attainment": statistics.mean(values) if values else None,
            }
            for key, values in rows.items()
        }

    category = summarize(category_rows)
    return {
        "category": category,
        "macro_category_slo_attainment": statistics.mean(
            value["slo_attainment"] for value in category.values()
        ),
        "required_bw_bins": summarize(bandwidth_rows),
        "distinct_profiles": len(profile_counts),
        "profile_counts": {
            f"{key[0]},{key[1]}": count
            for key, count in sorted(profile_counts.items())
        },
    }


def _run_case(task):
    case, source_fingerprint, config_fingerprint = task
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.strategy} ssu={case.num_ssu}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        workload, requests = _workload_and_requests(case.num_ssu)
        summary = _simulate(case, requests)
        _validate_summary(case, summary)
        inputs = {
            **_WORKER_SCHEDULE.as_fingerprint_dict(),
            "workload": workload.workload_hash,
            "placement": workload.placement_hash,
            "trace": workload.trace_hash,
            "simulator": summary["input_fingerprint"],
        }
        return {
            "status": "ok",
            "strategy": case.strategy,
            "kind": case.kind,
            "num_ssu": case.num_ssu,
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "case_fingerprint": _case_fingerprint(
                case, source_fingerprint, config_fingerprint
            ),
            "input_fingerprints": inputs,
            "workload_statistics": workload.statistics,
            "cohort_profile_metrics": _cohort_profile_metrics(
                summary, _WORKER_SCHEDULE, _WORKER_TABLE
            ),
            "wall_time_s": time.perf_counter() - started,
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _row_key(row):
    return str(row["strategy"]), int(row["num_ssu"])


def _validate_cached(payload, source_fingerprint, spec, config_fingerprint):
    if not payload:
        return {}
    if (
        payload.get("source_fingerprint") != source_fingerprint
        or payload.get("ending_source_fingerprint") != source_fingerprint
        or payload.get("experiment_spec") != spec
        or payload.get("config_fingerprint") != config_fingerprint
        or payload.get("ending_config_fingerprint") != config_fingerprint
    ):
        return {}
    rows = {}
    for row in payload.get("results", []):
        try:
            key = _row_key(row)
            case = CASE_BY_KEY[key]
        except (KeyError, TypeError, ValueError):
            return {}
        if (
            row.get("status") != "ok"
            or row.get("case_spec") != asdict(case)
            or row.get("source_fingerprint") != source_fingerprint
            or row.get("config_fingerprint") != config_fingerprint
            or row.get("case_fingerprint")
            != _case_fingerprint(case, source_fingerprint, config_fingerprint)
        ):
            return {}
        _validate_summary(case, row["steady_summary"])
        rows[key] = row
    return rows


def _pairing_audit(rows):
    audit = {}
    for num_ssu in SSU_LIST:
        selected = [row for row in rows.values() if row["num_ssu"] == num_ssu]
        fields = (
            "catalog",
            "recipe",
            "schedule",
            "assignment",
            "workload",
            "placement",
            "trace",
            "simulator",
        )
        audit[str(num_ssu)] = {
            "strategies": sorted(row["strategy"] for row in selected),
            "all_available_rows_paired": bool(selected)
            and all(
                len({row["input_fingerprints"][field] for row in selected}) == 1
                for field in fields
            ),
        }
    return audit


def _payload(
    *,
    rows,
    schedule,
    spec,
    source_fingerprint,
    config_fingerprint,
    selected_keys,
):
    ending_source = _source_fingerprint()
    ending_config = _config_fingerprint(spec)
    ordered = [rows[key] for key in sorted(rows, key=lambda key: (key[1], key[0]))]
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": len(rows) == len(CASES),
        "selected_complete": all(key in rows for key in selected_keys),
        "source_stable_during_run": ending_source == source_fingerprint,
        "config_stable_during_run": ending_config == config_fingerprint,
        "source_fingerprint": source_fingerprint,
        "ending_source_fingerprint": ending_source,
        "config_fingerprint": config_fingerprint,
        "ending_config_fingerprint": ending_config,
        "experiment_spec": spec,
        "schedule_metadata": {
            **schedule.as_fingerprint_dict(),
            "mode": schedule.mode,
            "seed": schedule.seed,
            "num_npu": schedule.num_npu,
            "requests_per_npu": schedule.requests_per_npu,
        },
        "pairing_audit": _pairing_audit(rows),
        "selected_keys": [list(key) for key in sorted(selected_keys)],
        "results": ordered,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _preflight(table, schedule, ssus):
    rows = {}
    for num_ssu in ssus:
        workload = prepare_random_steady_state_workload(
            table,
            schedule=schedule,
            num_ssu=num_ssu,
            n_layers=N_LAYERS,
        )
        stats = workload.statistics
        rows[str(num_ssu)] = {
            "workload_hash": workload.workload_hash,
            "placement_hash": workload.placement_hash,
            "trace_hash": workload.trace_hash,
            "fleet_category_counts": stats["fleet_category_counts"],
            "profiles_used": stats["profiles_used"],
            "per_npu_demand_gbps_range": stats["per_npu_demand_gbps_range"],
            "fleet_demand_gbps": stats["fleet_demand_gbps"],
            "capacity_knee_ssu": stats["capacity_knee_ssu"],
            "max_ssu_demand_gbps": stats["max_ssu_demand_gbps"],
            "ssu_over_40_count": stats["ssu_over_40_count"],
            "required_bw_profile_bins": stats["required_bw_profile_bins"],
        }
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("baseline", "layer_once", "adaptive_v2_1_25ms"),
        default=("baseline", "layer_once", "adaptive_v2_1_25ms"),
    )
    parser.add_argument(
        "--ssus", nargs="+", type=int, choices=SSU_LIST, default=SSU_LIST
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("max-workers 必须为正数")
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=STRATIFIED_RANDOM_CATALOG_V1,
        seed=SEED,
        num_npu=NUM_NPU,
        requests_per_npu=REQUESTS_PER_NPU,
    )
    spec = experiment_spec(schedule)
    source_fingerprint = _source_fingerprint()
    config_fingerprint = _config_fingerprint(spec)
    selected_keys = {
        (strategy, num_ssu)
        for num_ssu in args.ssus
        for strategy in args.strategies
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "source_fingerprint": source_fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "schedule": schedule.as_fingerprint_dict(),
                    "preflight": _preflight(table, schedule, args.ssus),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    cached = None
    if args.output.exists():
        try:
            cached = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    rows = _validate_cached(cached, source_fingerprint, spec, config_fingerprint)
    pending = [
        CASE_BY_KEY[key]
        for key in sorted(selected_keys, key=lambda key: (key[1], key[0]))
        if key not in rows
    ]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(pending)),
            initializer=_init_worker,
            initargs=(table, schedule),
        ) as executor:
            for row in executor.map(
                _run_case,
                (
                    (case, source_fingerprint, config_fingerprint)
                    for case in pending
                ),
                chunksize=1,
            ):
                rows[_row_key(row)] = row
                payload = _payload(
                    rows=rows,
                    schedule=schedule,
                    spec=spec,
                    source_fingerprint=source_fingerprint,
                    config_fingerprint=config_fingerprint,
                    selected_keys=selected_keys,
                )
                _write_json(args.output, payload)
                print(
                    f"DONE {row['strategy']} ssu={row['num_ssu']} "
                    f"wall={row['wall_time_s']:.1f}s",
                    flush=True,
                )
    payload = _payload(
        rows=rows,
        schedule=schedule,
        spec=spec,
        source_fingerprint=source_fingerprint,
        config_fingerprint=config_fingerprint,
        selected_keys=selected_keys,
    )
    _write_json(args.output, payload)
    if not payload["source_stable_during_run"]:
        raise RuntimeError("source changed during experiment")
    if not payload["config_stable_during_run"]:
        raise RuntimeError("config changed during experiment")
    if not payload["selected_complete"]:
        raise RuntimeError("selected experiment cases are incomplete")
    print(json.dumps({
        "output": str(args.output),
        "complete": payload["complete"],
        "selected_complete": payload["selected_complete"],
        "pairing_audit": payload["pairing_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
