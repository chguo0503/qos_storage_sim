"""SSU70 lower/upper bracket using the 32-NPU steady/full-load trace.

This runner is intentionally independent from the main 32-NPU experiment and
its diagnostics.  It runs only the missing lower bracket, 32 NPU x 17 SSU,
with the static baseline and the final SLO-admission Scheme B controller.

Scaling both resources by four makes 32x17 capacity-ratio-equivalent to
128x68.  The existing 32x18 point is equivalent to 128x72, so the two points
bracket the requested 128-NPU SSU70 operating point.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    SteadyStateConfig,
    SteadyStateInvariantError,
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
from six_request_workload import SEED
from slo_admission_scheme_b import SLOAdmissionSchemeBController
from steady_state_workload import REQUESTS_PER_NPU, prepare_steady_state_workload


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_normalized_slo"
OUTPUT_JSON = OUTPUT_DIR / "ratio70_bracket.json"
OUTPUT_MARKDOWN = OUTPUT_DIR / "ratio70_bracket.md"
SCHEMA_VERSION = 1

NUM_NPU = 32
NUM_SSU = 17
N_LAYERS = 16
FORMAL_WORKERS = 2
SCALE_TO_128 = 128 // NUM_NPU
LOWER_EQUIVALENT_128_SSU = NUM_SSU * SCALE_TO_128
UPPER_32NPU_SSU = 18
UPPER_EQUIVALENT_128_SSU = UPPER_32NPU_SSU * SCALE_TO_128
TARGET_128_SSU = 70

CONTROL_MIN_INTERVAL_MS = 10.0
TARGET_RATIO = 0.52
BACKGROUND_RESERVE_FRACTION = 0.05
STEADY_CONFIG = SteadyStateConfig(
    warmup_requests_per_npu=4,
    settle_ms=500.0,
    measurement_ms=2_000.0,
    slo_alpha=2.0,
    block_ms=500.0,
)
REQUIRED_RATIO = 1.0 / STEADY_CONFIG.slo_alpha


@dataclass(frozen=True)
class BracketCase:
    name: str
    kind: str


CASES = (
    BracketCase("baseline", "baseline"),
    BracketCase("scheme_b_slo_admission", "slo_admission"),
)
CASE_BY_NAME = {case.name: case for case in CASES}
PAIRED_INPUT_FIELDS = (
    "assignment_hash",
    "workload_hash",
    "placement_hash",
    "trace_hash",
    "simulator_input_fingerprint",
)
_WORKER_TABLE = None


def _hash_json(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256(b"steady-state-32npu-ratio70-bracket-v1\0")
    for name in (
        "sim.py",
        "policy_logic.py",
        "strategy_profiles.py",
        "continuous_batch_control.py",
        "continuous_batch_sim.py",
        "continuous_prefill_client.py",
        "continuous_prefill_workload.py",
        "six_request_workload.py",
        "steady_state_workload.py",
        "scheme_b_prefill.py",
        "slo_admission_scheme_b.py",
        "steady_state_32npu_ratio70_bracket.py",
        "data",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _config_payload() -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cases": [asdict(case) for case in CASES],
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "requests_per_npu_prefix": REQUESTS_PER_NPU,
        "seed": SEED,
        "steady_state": asdict(STEADY_CONFIG),
        "physical_ssd_bw_gbps": sim.DISK_BW,
        "physical_npu_bw_gbps": sim.NPU_BW_LIMIT,
        "placement": "token-block ring hash, reused by all 16 layers",
        "cross_request_layer0_prefetch": True,
        "admission_controller": {
            "class": "SLOAdmissionSchemeBController",
            "target_ratio": TARGET_RATIO,
            "required_ratio": REQUIRED_RATIO,
            "background_reserve_fraction": BACKGROUND_RESERVE_FRACTION,
            "ssd_cap_gbps": sim.DISK_BW,
            "npu_cap_gbps": sim.NPU_BW_LIMIT,
            "dedicated_path_per_npu": True,
            "shared_layer0_path": False,
            "full_remaining_manifest": True,
            "control_trigger": "batch_boundary_event_gated",
            "on_batch_boundary": True,
            "interval_ms": None,
            "min_interval_ms": CONTROL_MIN_INTERVAL_MS,
        },
        "ratio70_bracket": {
            "lower_32npu_ssu": NUM_SSU,
            "lower_equivalent_128npu_ssu": LOWER_EQUIVALENT_128_SSU,
            "target_128npu_ssu": TARGET_128_SSU,
            "upper_32npu_ssu": UPPER_32NPU_SSU,
            "upper_equivalent_128npu_ssu": UPPER_EQUIVALENT_128_SSU,
            "scale_factor": SCALE_TO_128,
        },
    }
    return payload


def experiment_spec(source_fingerprint: str | None = None) -> dict:
    config = _config_payload()
    return {
        **config,
        "config_fingerprint": _hash_json(config),
        "source_fingerprint": source_fingerprint or _source_fingerprint(),
    }


def _init_worker():
    global _WORKER_TABLE
    _WORKER_TABLE = sim.load_bw_table_cache(num_npu=NUM_NPU)


def _common_simulation_args() -> dict:
    return {
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "batch_size": 1,
        "submit_order_seed": SEED,
        "cross_request_layer0_prefetch": True,
        "steady_state": STEADY_CONFIG,
        "disk_bw_gbps": sim.DISK_BW,
        "npu_bw_gbps": sim.NPU_BW_LIMIT,
    }


def _scheme_b_paths() -> tuple[int, ...]:
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))
    if len(paths) != NUM_NPU or len(set(paths)) != NUM_NPU:
        raise AssertionError("SLO admission requires one unique Path per NPU")
    if min(paths) < 0 or max(paths) >= PATH_COUNT:
        raise AssertionError("dedicated Path mapping is outside the QoS table")
    return paths


def _simulate(case: BracketCase, requests):
    if case.kind == "baseline":
        baseline = next(
            spec for spec in routing_strategy_specs() if spec.name == "baseline"
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=baseline.client_config(),
            **_common_simulation_args(),
        )
    if case.kind == "slo_admission":
        paths = _scheme_b_paths()
        controller = SLOAdmissionSchemeBController(
            paths,
            target_ratio=TARGET_RATIO,
            required_ratio=REQUIRED_RATIO,
            background_reserve_fraction=BACKGROUND_RESERVE_FRACTION,
            ssd_cap_gbps=sim.DISK_BW,
            npu_cap_gbps=sim.NPU_BW_LIMIT,
        )
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * NUM_SSU
            ),
            npu_dedicated_paths=paths,
            layer0_path_id=None,
            client_io_config=scheme_b_client_config("slo_admission_ratio70_bracket"),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=CONTROL_MIN_INTERVAL_MS,
            ),
            **_common_simulation_args(),
        )
    raise ValueError(f"unknown ratio70 bracket case kind: {case.kind}")


def _runner_invariants(case: BracketCase, summary: dict) -> dict[str, bool]:
    counters = (
        summary["control_evaluations"],
        summary["cir_commits"],
        summary["cir_path_writes"],
    )
    return {
        "simulator_mode_is_steady_state": summary["mode"]
        == "steady_state_full_load",
        "simulator_invariants_all_hold": all(summary["invariants"].values()),
        "dimensions_match_case": summary["num_npu"] == NUM_NPU
        and summary["num_ssu"] == NUM_SSU
        and summary["n_layers"] == N_LAYERS,
        "no_pressure_side_channel": summary["pressure_reports"] == 0,
        "baseline_has_no_controller_activity": case.kind != "baseline"
        or not any(counters),
        "admission_controller_evaluated": case.kind != "slo_admission"
        or summary["control_evaluations"] > 0,
        "admission_controller_committed_cir": case.kind != "slo_admission"
        or (
            summary["cir_commits"] > 0
            and summary["cir_path_writes"] > 0
        ),
        "admission_min_interval_recorded": case.kind != "slo_admission"
        or summary["control_min_interval_ms"] == CONTROL_MIN_INTERVAL_MS,
    }


def _invalid_row(
    case: BracketCase,
    workload,
    error: SteadyStateInvariantError,
    *,
    source_fingerprint: str,
    config_fingerprint: str,
    wall_time_s: float,
) -> dict:
    return {
        "status": "invalid",
        "strategy": case.name,
        "kind": case.kind,
        "case_spec": asdict(case),
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": config_fingerprint,
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_invariants": error.invariants,
        "failure_diagnostics": error.diagnostics,
    }


def _compact_row(
    case: BracketCase,
    workload,
    summary: dict,
    *,
    source_fingerprint: str,
    config_fingerprint: str,
    wall_time_s: float,
) -> dict:
    invariants = _runner_invariants(case, summary)
    if not all(invariants.values()):
        raise AssertionError(f"runner invariant failed for {case.name}: {invariants}")
    return {
        "status": "ok",
        "strategy": case.name,
        "kind": case.kind,
        "case_spec": asdict(case),
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": config_fingerprint,
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "wall_time_s": wall_time_s,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "simulator_input_fingerprint": summary["input_fingerprint"],
        "physical_ssd_bw_gbps": sim.DISK_BW,
        "physical_npu_bw_gbps": sim.NPU_BW_LIMIT,
        "allocator_npu_cap_gbps": (
            sim.NPU_BW_LIMIT if case.kind == "slo_admission" else None
        ),
        "runner_invariants": invariants,
        "workload_statistics": workload.statistics,
        "steady_summary": summary,
    }


def _run_case(task) -> dict:
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
    table = _WORKER_TABLE or sim.load_bw_table_cache(num_npu=NUM_NPU)
    try:
        workload = prepare_steady_state_workload(
            table,
            num_npu=NUM_NPU,
            num_ssu=NUM_SSU,
            n_layers=N_LAYERS,
            requests_per_npu=REQUESTS_PER_NPU,
            seed=SEED,
        )
        requests = requests_from_continuous_prefill_workload(workload)
        try:
            summary = _simulate(case, requests)
        except SteadyStateInvariantError as error:
            return _invalid_row(
                case,
                workload,
                error,
                source_fingerprint=source_fingerprint,
                config_fingerprint=config_fingerprint,
                wall_time_s=time.perf_counter() - started,
            )
        return _compact_row(
            case,
            workload,
            summary,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
            wall_time_s=time.perf_counter() - started,
        )
    finally:
        finished.set()


def _validate_paired_rows(rows: dict[str, dict]) -> dict:
    if set(rows) != set(CASE_BY_NAME):
        raise AssertionError("paired bracket requires exactly baseline and admission")
    if any(row.get("status") != "ok" for row in rows.values()):
        raise AssertionError("paired bracket contains an invalid case")
    shared = {}
    for field in PAIRED_INPUT_FIELDS:
        values = {row[field] for row in rows.values()}
        if len(values) != 1:
            raise AssertionError(f"unpaired {field}")
        shared[field] = values.pop()
    return {
        "all_fields_match": True,
        "fields": list(PAIRED_INPUT_FIELDS),
        "shared_values": shared,
    }


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _write_json(path: Path, payload: dict):
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
    )


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _number(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _render_report(payload: dict) -> str:
    rows = {row["strategy"]: row for row in payload.get("results", ())}
    lines = [
        "# SSU70 steady-state bracket (32 NPU)",
        "",
        f"Status: `{'complete' if payload.get('complete') else 'incomplete'}`. ",
        "",
        "Under the same per-NPU demand and SSD40/NPU50 data plane, scaling both "
        "resource counts by four preserves the NPU:SSU capacity ratio:",
        "",
        "| Role | 32-NPU point | Equivalent 128-NPU point |",
        "|---|---:|---:|",
        f"| Lower bracket (this run) | 32 NPU / {NUM_SSU} SSU | "
        f"128 NPU / {LOWER_EQUIVALENT_128_SSU} SSU |",
        f"| Requested target | — | 128 NPU / {TARGET_128_SSU} SSU |",
        f"| Upper bracket (existing point) | 32 NPU / {UPPER_32NPU_SSU} SSU | "
        f"128 NPU / {UPPER_EQUIVALENT_128_SSU} SSU |",
        "",
        f"Thus `{LOWER_EQUIVALENT_128_SSU} < {TARGET_128_SSU} < "
        f"{UPPER_EQUIVALENT_128_SSU}`: 32x17 and the existing 32x18 point bracket "
        "the 128-NPU SSU70 operating point. This is a capacity-ratio bracket, not "
        "a claim that finite-fleet scheduling variance is identical.",
        "",
        "The admission controller uses the full remaining manifest and a "
        "batch-boundary event-gated controller with a 10-ms minimum interval "
        "(`interval_ms=None`); it is not a fixed 10-ms periodic controller.",
        "",
        "## Exact-window results",
        "",
        "| Strategy | Status | NPU util | Equal-NPU TTFT SLO | Request-weighted "
        "SLO | Mean TTFT (ms) | P99 TTFT (ms) | SSD util | NPU-link util | "
        "Requests | Control evals |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        row = rows.get(case.name)
        if row is None or row.get("status") != "ok":
            lines.append(
                f"| {case.name} | {row.get('status', 'missing') if row else 'missing'} "
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        summary = row["steady_summary"]
        lines.append(
            f"| {case.name} | ok | {_pct(summary['mean_npu_utilization'])} | "
            f"{_pct(summary['ttft_slo_attainment'])} | "
            f"{_pct(summary['request_weighted_slo_attainment'])} | "
            f"{_number(summary['mean_ttft_ms'])} | "
            f"{_number(summary['p99_ttft_ms'])} | "
            f"{_pct(summary['measurement_ssd_mean_utilization'])} | "
            f"{_pct(summary['measurement_npu_link_mean_utilization'])} | "
            f"{summary['measurement_request_count']} | "
            f"{summary['control_evaluations']} |"
        )

    baseline = rows.get("baseline")
    admission = rows.get("scheme_b_slo_admission")
    if (
        baseline is not None
        and admission is not None
        and baseline.get("status") == admission.get("status") == "ok"
    ):
        base_summary = baseline["steady_summary"]
        admission_summary = admission["steady_summary"]
        lines.extend(
            [
                "",
                "## Admission minus baseline",
                "",
                "| Metric | Delta |",
                "|---|---:|",
                f"| NPU utilization | "
                f"{100.0 * (admission_summary['mean_npu_utilization'] - base_summary['mean_npu_utilization']):+.2f} pp |",
                f"| Equal-NPU TTFT SLO | "
                f"{100.0 * (admission_summary['ttft_slo_attainment'] - base_summary['ttft_slo_attainment']):+.2f} pp |",
                f"| Request-weighted TTFT SLO | "
                f"{100.0 * (admission_summary['request_weighted_slo_attainment'] - base_summary['request_weighted_slo_attainment']):+.2f} pp |",
                f"| Mean TTFT | "
                f"{admission_summary['mean_ttft_ms'] - base_summary['mean_ttft_ms']:+.3f} ms |",
                f"| P99 TTFT | "
                f"{admission_summary['p99_ttft_ms'] - base_summary['p99_ttft_ms']:+.3f} ms |",
            ]
        )

    lines.extend(
        [
            "",
            "## Validity and provenance",
            "",
            f"- Source stable during run: `{payload.get('source_stable_during_run')}`",
            f"- Source fingerprint: `{payload.get('source_fingerprint')}`",
            f"- Ending source fingerprint: `{payload.get('ending_source_fingerprint')}`",
            f"- Config fingerprint: `{payload.get('config_fingerprint')}`",
            f"- All simulator/runner invariants: `{payload.get('all_invariants_hold')}`",
            f"- Paired input fields all match: "
            f"`{payload.get('paired_input', {}).get('all_fields_match', False)}`",
        ]
    )
    shared = payload.get("paired_input", {}).get("shared_values", {})
    for field in PAIRED_INPUT_FIELDS:
        if field in shared:
            lines.append(f"- {field}: `{shared[field]}`")

    lines.extend(["", "## Measurement-block stability", ""])
    lines.extend(
        [
            "| Strategy | Block NPU-util range | SSD outstanding blocks start -> end |",
            "|---|---:|---:|",
        ]
    )
    for case in CASES:
        row = rows.get(case.name)
        if row is None or row.get("status") != "ok":
            lines.append(f"| {case.name} | n/a | n/a |")
            continue
        summary = row["steady_summary"]
        block_utils = [block["npu_utilization"] for block in summary["measurement_blocks"]]
        outstanding_start = sum(summary["measurement_ssd_outstanding_blocks_at_start"])
        outstanding_end = sum(summary["measurement_ssd_outstanding_blocks_at_end"])
        lines.append(
            f"| {case.name} | {100.0 * (max(block_utils) - min(block_utils)):.2f} pp | "
            f"{outstanding_start} -> {outstanding_end} |"
        )
    lines.append("")
    return "\n".join(lines)


def _ordered_rows(rows: dict[str, dict]) -> list[dict]:
    return [rows[case.name] for case in CASES if case.name in rows]


def run(
    output_json: Path = OUTPUT_JSON,
    output_markdown: Path = OUTPUT_MARKDOWN,
    *,
    workers: int = FORMAL_WORKERS,
    rerun: bool = False,
) -> Path:
    if workers <= 0:
        raise ValueError("workers must be positive")
    run_fingerprint = _source_fingerprint()
    spec = experiment_spec(run_fingerprint)
    config_fingerprint = spec["config_fingerprint"]
    rows: dict[str, dict] = {}

    if output_json.exists() and not rerun:
        cached = json.loads(output_json.read_text())
        cached_rows = cached.get("results", ())
        if (
            cached.get("source_fingerprint") == run_fingerprint
            and cached.get("config_fingerprint") == config_fingerprint
            and cached.get("experiment_spec") == spec
            and all(
                row.get("source_fingerprint") == run_fingerprint
                and row.get("config_fingerprint") == config_fingerprint
                for row in cached_rows
            )
        ):
            rows = {
                row["strategy"]: row
                for row in cached_rows
                if row["strategy"] in CASE_BY_NAME
            }

    def checkpoint():
        ending_fingerprint = _source_fingerprint()
        source_stable = ending_fingerprint == run_fingerprint
        ordered = _ordered_rows(rows)
        all_invariants_hold = bool(ordered) and all(
            row.get("status") == "ok"
            and all(row.get("runner_invariants", {}).values())
            and all(row.get("steady_summary", {}).get("invariants", {}).values())
            for row in ordered
        )
        paired_input = {"all_fields_match": False}
        if len(rows) == len(CASES) and all(
            row.get("status") == "ok" for row in rows.values()
        ):
            paired_input = _validate_paired_rows(rows)
        complete = (
            source_stable
            and len(rows) == len(CASES)
            and all_invariants_hold
            and paired_input["all_fields_match"]
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "complete": complete,
            "all_cases_finished": len(rows) == len(CASES),
            "source_stable_during_run": source_stable,
            "source_fingerprint": run_fingerprint,
            "ending_source_fingerprint": ending_fingerprint,
            "config_fingerprint": config_fingerprint,
            "experiment_spec": spec,
            "all_invariants_hold": all_invariants_hold,
            "paired_input": paired_input,
            "formal_workers": FORMAL_WORKERS,
            "requested_workers": workers,
            "results": ordered,
        }
        _write_json(output_json, payload)
        _write_text(output_markdown, _render_report(payload))
        return payload

    tasks = [
        (case, run_fingerprint, config_fingerprint)
        for case in CASES
        if case.name not in rows
    ]
    if tasks:
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker
        )
        try:
            futures = {pool.submit(_run_case, task): task[0] for task in tasks}
            for future in as_completed(futures):
                case = futures[future]
                row = future.result()
                rows[case.name] = row
                checkpoint()
                if row["status"] == "ok":
                    summary = row["steady_summary"]
                    print(
                        f"{case.name} ssu={NUM_SSU}: "
                        f"util={summary['mean_npu_utilization']:.2%}, "
                        f"SLO={summary['ttft_slo_attainment']:.2%}, "
                        f"wall={row['wall_time_s']:.1f}s",
                        flush=True,
                    )
                else:
                    failed = [
                        name
                        for name, holds in row["simulator_invariants"].items()
                        if not holds
                    ]
                    print(
                        f"INVALID {case.name} ssu={NUM_SSU}: {failed}",
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

    payload = checkpoint()
    if payload["ending_source_fingerprint"] != run_fingerprint:
        raise RuntimeError("ratio70 bracket source changed during the run")
    if not payload["complete"]:
        raise RuntimeError("ratio70 bracket did not produce two valid paired cases")
    return output_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    run(
        args.output_json,
        args.output_markdown,
        workers=args.workers,
        rerun=args.rerun,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
