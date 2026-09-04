#!/usr/bin/env python3
"""Fast, matched 28-high-V/4-low-V steady-state comparison.

The immutable IID catalog draw and block placement are generated once.  The
draw is then partitioned by the frozen V cutoff and round-robin assigned to
four low-V lanes (NPU 0..3) and 28 high-V lanes (NPU 4..31).  Every strategy
uses this exact transformed trace; only routing/category and the static CIR
table differ.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import (
    ContinuousBatchRequest,
    SteadyStateConfig,
    continuous_batch_input_fingerprint,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from random_steady_state_workload import (
    IID_UNIFORM_PROFILE_CATALOG_V1,
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)
import sim
from vb_pool_policy import qos_configs_for_pool_cirs, request_v_b


ROOT = Path(__file__).resolve().parent
NUM_NPU = 32
NUM_SSU = 5
N_LAYERS = 16
LOW_NPUS = tuple(range(4))
HIGH_NPUS = tuple(range(4, 32))
V_CUTOFF = 0.00031
DISK_BW_GIBPS = 40.0
NPU_BW_GIBPS = 50.0
CATALOG_LL_GIBPS = 24.56757751428572
SPLIT_LL_GIBPS = 37.6191030688
DURATION_AWARE_LL_GIBPS = 21.99828652397474
CASES = (
    "baseline",
    "route_only",
    "vb_catalog",
    "vb_duration_aware",
    "vb_ll36",
    "vb_split_aware",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _routing(name):
    return next(spec for spec in routing_strategy_specs() if spec.name == name)


def _load_and_split(seed, requests_per_npu):
    with redirect_stdout(io.StringIO()):
        table, authentication = load_authenticated_bw_table(NUM_NPU)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=IID_UNIFORM_PROFILE_CATALOG_V1,
        seed=seed,
        num_npu=NUM_NPU,
        requests_per_npu=requests_per_npu,
    )
    workload = prepare_random_steady_state_workload(
        table, schedule=schedule, num_ssu=NUM_SSU, n_layers=N_LAYERS
    )
    raw = requests_from_continuous_prefill_workload(workload)
    groups = {"low": [], "high": []}
    for request in raw:
        v_value, _ = request_v_b(request)
        groups["high" if v_value > V_CUTOFF else "low"].append(request)

    # Reuse the original deterministic per-NPU 0..5 ms jitter, while making
    # all queued requests on a target lane available at that lane's start.
    jitter = {
        npu_id: min(r.arrival_time_ms for r in raw if r.npu_id == npu_id)
        for npu_id in range(NUM_NPU)
    }
    transformed = []
    lane_counts = [0] * NUM_NPU
    for group_name, lanes in (("low", LOW_NPUS), ("high", HIGH_NPUS)):
        for index, request in enumerate(sorted(groups[group_name], key=lambda r: r.request_id)):
            npu_id = lanes[index % len(lanes)]
            sequence = lane_counts[npu_id]
            lane_counts[npu_id] += 1
            load = dict(request.load)
            load.update(
                {
                    "npu_id": npu_id,
                    "stream_id": sequence,
                    "generation": 0,
                    "arrival_ms": jitter[npu_id],
                    "arrival_time": jitter[npu_id],
                    "initial": sequence == 0,
                    "vb_input_group": group_name,
                }
            )
            transformed.append(
                ContinuousBatchRequest(
                    request_id=request.request_id,
                    npu_id=npu_id,
                    arrival_time_ms=jitter[npu_id],
                    load=load,
                    placement=request.placement,
                )
            )
    transformed.sort(key=lambda r: r.request_id)
    if min(lane_counts) < 9:
        raise ValueError(f"finite input cannot cover warm-up: lane_counts={lane_counts}")
    metadata = {
        "authentication": authentication,
        "schedule_fingerprints": schedule.as_fingerprint_dict(),
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "raw_input_fingerprint": continuous_batch_input_fingerprint(raw),
        "fixed_split_input_fingerprint": continuous_batch_input_fingerprint(transformed),
        "raw_request_count": len(raw),
        "high_request_count": len(groups["high"]),
        "low_request_count": len(groups["low"]),
        "lane_counts": lane_counts,
        "high_lane_count": len(HIGH_NPUS),
        "low_lane_count": len(LOW_NPUS),
    }
    return table, tuple(transformed), metadata


def _materialize_case(requests, case):
    if case in ("baseline", "route_only"):
        return requests, None
    protected = {
        "vb_catalog": CATALOG_LL_GIBPS,
        "vb_duration_aware": DURATION_AWARE_LL_GIBPS,
        "vb_ll36": 36.0,
        "vb_split_aware": SPLIT_LL_GIBPS,
    }[case]
    converted = []
    for request in requests:
        value_v, _ = request_v_b(request)
        load = dict(request.load)
        load["category"] = "LL" if value_v > V_CUTOFF else "LS"
        converted.append(
            ContinuousBatchRequest(
                request_id=request.request_id,
                npu_id=request.npu_id,
                arrival_time_ms=request.arrival_time_ms,
                load=load,
                placement=request.placement,
            )
        )
    qos_configs = qos_configs_for_pool_cirs(
        (protected,) * NUM_SSU, disk_bandwidth_gibps=DISK_BW_GIBPS
    )
    return tuple(converted), {
        "ll_cir_per_ssu_gibps": protected,
        "ls_cir_per_ssu_gibps": DISK_BW_GIBPS - protected,
    }, qos_configs


def _window_metrics(summary):
    duration_ms = float(summary["measurement_duration_ms"])
    ratios = [
        float(row["ttft_ms"]) / float(row["ideal_ttft_ms"])
        for row in summary["request_rows"]
    ]
    high_utils = [float(summary["npu_utilizations"][i]) for i in HIGH_NPUS]
    low_utils = [float(summary["npu_utilizations"][i]) for i in LOW_NPUS]
    high_compute = sum(float(summary["compute_ms_by_npu"][i]) for i in HIGH_NPUS)
    low_compute = sum(float(summary["compute_ms_by_npu"][i]) for i in LOW_NPUS)
    return {
        "duration_ms": duration_ms,
        "mean_npu_utilization": float(summary["mean_npu_utilization"]),
        "compute_area_npu_ms": high_compute + low_compute,
        "available_area_npu_ms": NUM_NPU * duration_ms,
        "high_v_28_mean_npu_utilization": statistics.fmean(high_utils),
        "low_v_4_mean_npu_utilization": statistics.fmean(low_utils),
        "high_v_compute_area_npu_ms": high_compute,
        "low_v_compute_area_npu_ms": low_compute,
        "mean_ssd_utilization": float(summary["measurement_ssd_mean_utilization"]),
        "per_ssu_ssd_utilization": list(summary["measurement_ssd_utilizations"]),
        "request_count": len(ratios),
        "ttft_over_ideal_mean": statistics.fmean(ratios) if ratios else None,
        "ttft_over_ideal_max": max(ratios) if ratios else None,
        "ttft_over_ideal_gt8_count": sum(value > 8.0 + 1e-12 for value in ratios),
        "all_32_npus_have_slo_samples": all(
            int(value) > 0 for value in summary["request_counts_by_npu"]
        ),
        "no_backlog_exhaustion": bool(summary["invariants"]["no_backlog_exhaustion"]),
        "all_invariants_pass": all(summary["invariants"].values()),
        "block_npu_utilizations": [
            float(row["npu_utilization"]) for row in summary["measurement_blocks"]
        ],
    }


def run(case, seed, requests_per_npu, measurement_ms, block_ms):
    started = time.monotonic()
    table, requests, input_metadata = _load_and_split(seed, requests_per_npu)
    materialized = _materialize_case(requests, case)
    if case in ("baseline", "route_only"):
        simulated_requests, pool = materialized
        qos_kwargs = {"qos_config": static_qos_config()}
        route = "baseline" if case == "baseline" else "layer_once"
    else:
        simulated_requests, pool, qos_configs = materialized
        qos_kwargs = {"qos_configs_by_ssu": qos_configs}
        route = "layer_once"
    summary = simulate_continuous_batch(
        simulated_requests,
        num_npu=NUM_NPU,
        num_ssu=NUM_SSU,
        n_layers=N_LAYERS,
        batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR,
        client_io_config=_routing(route).client_config(),
        cross_request_layer0_prefetch=True,
        steady_state=SteadyStateConfig(
            warmup_requests_per_npu=8,
            settle_ms=500.0,
            measurement_ms=measurement_ms,
            slo_alpha=8.0,
            block_ms=block_ms,
            timeline_diagnostics=False,
        ),
        disk_bw_gbps=DISK_BW_GIBPS,
        npu_bw_gbps=NPU_BW_GIBPS,
        pressure_ttl_ms=0.0,
        cir_write_threshold_gbps=0.0,
        submit_order_seed=seed,
        **qos_kwargs,
    )
    actual_fingerprint = continuous_batch_input_fingerprint(simulated_requests)
    if summary["input_fingerprint"] != actual_fingerprint:
        raise AssertionError("simulator input fingerprint drift")
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "case": case,
        "experiment": {
            "num_npu": NUM_NPU,
            "num_ssu": NUM_SSU,
            "n_layers": N_LAYERS,
            "high_v_npus": list(HIGH_NPUS),
            "low_v_npus": list(LOW_NPUS),
            "v_cutoff_s2_per_gib": V_CUTOFF,
            "seed": seed,
            "requests_per_original_npu": requests_per_npu,
            "warmup_requests_per_npu": 8,
            "settle_ms": 500.0,
            "measurement_ms": measurement_ms,
            "block_ms": block_ms,
            "routing": route,
            "pool": pool,
            "ttft_over_ideal_guard": 8.0,
        },
        "input": input_metadata,
        "simulated_input_fingerprint": actual_fingerprint,
        "metrics": _window_metrics(summary),
        "elapsed_wall_s": time.monotonic() - started,
        "source_sha256": {
            name: _sha256(ROOT / name)
            for name in (
                "continuous_batch_sim.py",
                "sim.py",
                "vb_pool_policy.py",
                Path(__file__).name,
                "data",
            )
        },
        "steady_summary": summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "vb_fixed_split_npu32_ssu5_28high_4low_4s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests-per-npu", type=int, default=64)
    parser.add_argument("--measurement-ms", type=float, default=4000.0)
    parser.add_argument("--block-ms", type=float, default=500.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _, requests, metadata = _load_and_split(args.seed, args.requests_per_npu)
    if args.dry_run:
        print(json.dumps({"case": args.case, "input": metadata}, indent=2, sort_keys=True))
        return
    result = run(
        args.case,
        args.seed,
        args.requests_per_npu,
        args.measurement_ms,
        args.block_ms,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.case}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": result["metrics"], "elapsed_wall_s": result["elapsed_wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
