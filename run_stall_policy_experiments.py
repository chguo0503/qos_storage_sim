#!/usr/bin/env python3
"""Same-input comparisons on the unmodified full-prefill simulator.

One logical I/O is exactly 176 KiB (128 GLM KV tokens for one layer).
The new controller sees only CIRControlSnapshot: arrived manifests and public
count-only telemetry, never a simulator FIFO, future arrival, or service event.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

from continuous_batch_sim import (
    CIRControlConfig, CIRControlDecision, ContinuousBatchRequest,
    SteadyStateConfig, continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs, routing_strategy_specs, static_qos_config,
)
import sim
from npu_stall_predictor import required_rate_gib_s

IO_BYTES = 176 * 1024
IO_GIB = IO_BYTES / 2**30
PATHS = (0, 32, 64, 96)
CAP_GIB_S = 40.0


@dataclass(frozen=True)
class Profile:
    blocks: int
    compute_ms: float
    nql: int = 128
    provenance: str = "mechanism_synthetic_not_data_calibrated"

    @property
    def demand_gib_s(self):
        return self.blocks * IO_GIB * 1000.0 / self.compute_ms


def old_rounded_profiles():
    """New full-block input, not byte-identical to the older partial-block run."""
    return tuple(Profile(k, c, q, "old_data_interpolation_compute_full_block_rounding")
                 for k, c, q in zip(
        (7, 1534, 1529, 1529),
        (0.582149491632, 12.3901648021, 29.8565628164, 29.8565628164),
        (169, 368, 896, 896),
    ))


def direct_data_profiles(keys=((32, 256), (192, 1024), (192, 1024), (192, 4096))):
    from authenticated_workload_inputs import load_authenticated_bw_table
    with redirect_stdout(io.StringIO()):
        table, _ = load_authenticated_bw_table(4)
    return tuple(Profile((seq * 1024 - nql) // 128,
                         float(table[seq, nql][1]) / 1000.0, nql,
                         f"direct_data_seq{seq}K_nql{nql}") for seq, nql in keys)


def calibrated_two_ls_profiles():
    """Data-calibrated C, aligned full blocks and original routing categories.

    NQL 384 interpolates the 256/512 measurements. The 1K compute extends
    the 32K/48K line, as the existing low-utilization tutorial does; this
    extrapolation is a model hypothesis, not an observed hardware datum.
    """
    from authenticated_workload_inputs import load_authenticated_bw_table
    with redirect_stdout(io.StringIO()):
        table, _ = load_authenticated_bw_table(4)
    profiles = []
    for seq, nql in ((1, 512), (1, 512), (192, 384), (96, 384)):
        if seq == 1:
            at32, at48 = float(table[32, nql][1]), float(table[48, nql][1])
            compute_us = at32 + (at48 - at32) / 16.0 * (seq - 32)
            source = f"seq{seq}K_nql{nql}_linear_seq_extrapolation_32K_48K"
        else:
            compute_us = (float(table[seq, 256][1]) + float(table[seq, 512][1])) / 2
            source = f"seq{seq}K_nql{nql}_linear_nql_interpolation_256_512"
        profiles.append(Profile((seq * 1024 - nql) // 128, compute_us / 1000,
                                nql, source))
    return tuple(profiles)


def profiles_for_demand(blocks, demands):
    return tuple(Profile(k, k * IO_GIB * 1000.0 / d)
                 for k, d in zip(blocks, demands))


def build_requests(profiles, *, phases_ms=(0.0,) * 4, horizon_ms=2500.0,
                   count_per_npu=None):
    """A finite, already-arrived saturated backlog with immutable NPU mapping."""
    requests = []
    for npu_id, profile in enumerate(profiles):
        count = (int(count_per_npu) if count_per_npu is not None else
                 math.ceil(horizon_ms / (8 * profile.compute_ms)) + 8)
        placement = (tuple((0, IO_GIB) for _ in range(profile.blocks)),)
        for generation in range(count):
            request_id = npu_id * 1_000_000 + generation
            total_tokens = profile.blocks * 128 + profile.nql
            load = {
                "request_id": request_id, "npu_id": npu_id,
                "stream_id": generation, "generation": generation,
                "seq_len_k": total_tokens / 1024, "nql": profile.nql,
                "category": sim.classify_request(total_tokens / 1024, profile.nql),
                "required_bw_input_gbps": profile.demand_gib_s,
                "per_layer_us": profile.compute_ms * 1000.0,
                "per_layer_kv_gb": profile.blocks * IO_GIB,
                "source_ttft_ms": profile.compute_ms * 78,
                "arrival_ms": phases_ms[npu_id],
                "arrival_time": phases_ms[npu_id], "initial": generation == 0,
                "constructed_profile": not profile.provenance.startswith("direct_data_"),
            }
            requests.append(ContinuousBatchRequest(
                request_id, npu_id, phases_ms[npu_id], load, placement))
    return tuple(requests)


def request_weighted_nominal_demand(requests):
    """Ideal saturated demand is sum(bytes)/sum(compute), not mean(V/C)."""
    work = [0.0] * 4
    compute_ms = [0.0] * 4
    for request in requests:
        work[request.npu_id] += 8 * sum(size for _, size in request.placement[0])
        compute_ms[request.npu_id] += 8 * float(request.load["per_layer_us"]) / 1000.0
    per_npu = [1000 * v / c if c else 0.0 for v, c in zip(work, compute_ms)]
    return {"per_npu_gib_s": per_npu, "sum_gib_s": sum(per_npu),
            "per_npu_total_read_gib": work, "per_npu_total_compute_ms": compute_ms}


def build_variable_requests(*, seed=20260906, arrival_gib_s=39.2,
                            duration_ms=2000.0, initial_backlog_per_npu=6,
                            burst_size=1, mix="long"):
    """Raw-data per-request variability with causally visible arrivals.

    The long-run per-lane mix is 17:14:1 of the exact 192K NQL512/1024/2048
    records. Each lane independently shuffles complete quota cycles. The
    initial backlog is explicitly a startup burst, not charged to the stated
    subsequent byte-paced arrival rate. ``burst_size`` batches subsequent
    byte-paced arrivals without changing requests or their total bytes.
    """
    from authenticated_workload_inputs import load_authenticated_bw_table
    with redirect_stdout(io.StringIO()):
        table, table_metadata = load_authenticated_bw_table(4)
    if mix == "broad":
        keys = ((32, 128), (32, 256), (64, 512), (96, 512), (192, 512), (192, 1024))
        counts = (8, 4, 4, 4, 8, 16)
    else:
        keys = ((192, 512), (192, 1024), (192, 2048))
        counts = (17, 14, 1)
    pool = tuple(Profile((seq * 1024 - q) // 128, float(table[seq, q][1]) / 1000,
                         q, f"direct_data_seq{seq}K_nql{q}") for seq, q in keys)
    rngs = [random.Random(seed + lane * 100003) for lane in range(4)]
    decks = [[] for _ in range(4)]
    generations = [0] * 4
    requests = []

    def add(lane, arrival):
        if not decks[lane]:
            decks[lane] = [i for i, count in enumerate(counts) for _ in range(count)]
            rngs[lane].shuffle(decks[lane])
        index = decks[lane].pop()
        profile = pool[index]
        seq, _ = keys[index]
        generation = generations[lane]
        generations[lane] += 1
        rid = lane * 1_000_000 + generation
        load = {
            "request_id": rid, "npu_id": lane, "stream_id": generation,
            "generation": generation, "seq_len_k": seq, "nql": profile.nql,
            "category": sim.classify_request(seq, profile.nql),
            "required_bw_input_gbps": profile.demand_gib_s,
            "per_layer_us": profile.compute_ms * 1000,
            "per_layer_kv_gb": profile.blocks * IO_GIB,
            "source_ttft_ms": profile.compute_ms * 78,
            "arrival_time": arrival, "arrival_ms": arrival,
            "initial": generation == 0, "constructed_profile": False,
        }
        requests.append(ContinuousBatchRequest(
            rid, lane, arrival, load,
            (tuple((0, IO_GIB) for _ in range(profile.blocks)),)))
        return 8 * profile.blocks * IO_GIB

    initial_bytes = 0.0
    for _ in range(initial_backlog_per_npu):
        for lane in range(4):
            initial_bytes += add(lane, 0.0)
    paced_bytes, paced_until_ms, count = 0.0, 0.0, 0
    while paced_until_ms < duration_ms:
        # Each burst is made visible only after its previous bytes' budget;
        # no scheduling policy receives the not-yet-arrived deck entries.
        arrival = paced_until_ms
        burst_bytes = 0.0
        for _ in range(burst_size):
            burst_bytes += add(count % 4, arrival)
            count += 1
        paced_bytes += burst_bytes
        paced_until_ms += burst_bytes / arrival_gib_s * 1000
    metadata = {
        "seed": seed, "source": "authenticated exact 4-NPU data rows",
        "authenticated_data": table_metadata,
        "mix": mix,
        "source_keys": [list(key) for key in keys],
        "per_lane_long_run_cycle_counts": list(counts),
        "long_run_cycle_fleet_demand_gib_s":
            4 * sum(c * p.blocks * IO_GIB for c, p in zip(counts, pool)) * 1000 /
            sum(c * p.compute_ms for c, p in zip(counts, pool)),
        "initial_backlog_per_npu": initial_backlog_per_npu,
        "initial_backlog_read_gib": initial_bytes,
        "subsequent_arrival_gib_s": arrival_gib_s,
        "subsequent_pacing_budget_end_ms": paced_until_ms,
        "subsequent_arrival_read_gib": paced_bytes,
        "subsequent_bytes_over_first_last_arrival_span_gib_s":
            paced_bytes * 1000 / arrival,
        "last_arrival_ms": arrival,
        "all_input_bytes_over_pacing_budget_gib_s":
            (initial_bytes + paced_bytes) * 1000 / paced_until_ms,
        "burst_size": burst_size,
        "finite_manifest_ideal_demand": request_weighted_nominal_demand(requests),
        "interpretation": "startup backlog plus near-capacity arrivals, not an instantaneous <=40 demand promise",
    }
    return tuple(requests), pool, metadata


def summarize_complete_run(result, *, start_ms=1000.0, end_ms=2000.0):
    """Common absolute window plus the identical complete request population."""
    summary = result["summary"]
    duration = end_ms - start_ms
    compute, active = [0.0] * 4, [0.0] * 4
    for batch in summary["microbatch_metrics"]:
        lane = batch["npu_id"]
        active[lane] += max(0.0, min(end_ms, batch["completion_time_ms"])
                            - max(start_ms, batch["admission_time_ms"]))
        for layer in batch["layer_metrics"]:
            compute[lane] += max(0.0, min(end_ms, layer["compute_end_ms"])
                                 - max(start_ms, layer["compute_start_ms"]))
    rows = summary["request_metrics"]
    latencies = sorted(row["latency_ms"] for row in rows)
    index = 0.99 * (len(latencies) - 1)
    low = math.floor(index)
    p99 = latencies[low] + (latencies[min(low + 1, len(latencies) - 1)]
                            - latencies[low]) * (index - low)
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "npu_utilizations": [value / duration for value in compute],
        "mean_npu_utilization": sum(compute) / (4 * duration),
        "active_ms_by_npu": active,
        "all_npus_active_whole_window": all(abs(v - duration) < 1e-7 for v in active),
        "io_stall_ms_by_npu": [a - c for a, c in zip(active, compute)],
        "complete_request_population": len(rows),
        "makespan_ms": max(row["completion_time_ms"] for row in rows),
        "mean_arrival_to_completion_ms": sum(latencies) / len(rows),
        "p99_arrival_to_completion_ms": p99,
        "mean_request_io_stall_ms": sum(row["io_stall_ms"] for row in rows) / len(rows),
        "arrival_slo_attainment_alpha2": sum(row["latency_ms"] <= 2 * row["own_compute_ms"]
                                               for row in rows) / len(rows),
        "admission_slo_attainment_alpha2": sum(row["processing_latency_ms"] <= 2 * row["own_compute_ms"]
                                                 for row in rows) / len(rows),
    }


class ManifestCIRController:
    """Per-NPU reservation from arrived current and prefetch-only manifests.

    Per-lane maximum avoids double-counting the last-layer current request and
    its already-arrived successor. Proportional capacity scaling is a heuristic
    for over-capacity snapshots, not a no-stall guarantee. Optional headroom
    consumes available capacity while preserving the demand ratios.
    """
    def __init__(self, mode="demand", headroom=1.0):
        self.mode = mode
        self.headroom = headroom
        self.decisions = []

    def __call__(self, snapshot):
        demands = [0.0] * 4
        for request in snapshot.active_requests:
            demand = required_rate_gib_s(
                request.next_layer_work_gb_by_ssu[0] / IO_GIB,
                request.per_layer_compute_ms)
            demands[request.npu_id] = max(demands[request.npu_id], demand)
        if self.mode == "equal":
            demands = [10.0 if demand else 0.0 for demand in demands]
        else:
            demands = [demand * self.headroom for demand in demands]
        scale = min(1.0, CAP_GIB_S / sum(demands)) if sum(demands) else 1.0
        cirs = [0.0] * 256
        for path, demand in zip(PATHS, demands):
            cirs[path] = demand * scale
        self.decisions.append({
            "time_ms": snapshot.time_ms,
            "active_request_ids": [r.request_id for r in snapshot.active_requests],
            "cirs": [cirs[p] for p in PATHS],
            "queue_counts": [snapshot.path_outstanding_io_counts_by_ssu[0][p]
                             for p in PATHS],
        })
        return CIRControlDecision((tuple(cirs),))


def run_case(profiles, strategy, *, seed=42, phases_ms=(0.0,) * 4,
             measurement_ms=1000.0, settle_ms=100.0, warmup_requests=2,
             finite_count=None, requests=None, headroom=1.0, complete_all=False):
    implementation_hashes = {
        filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for filename in ("sim.py", "continuous_batch_sim.py", "policy_logic.py",
                         "continuous_prefill_client.py", "npu_stall_predictor.py",
                         "run_stall_policy_experiments.py")}
    horizon = (max(p.compute_ms for p in profiles) * 8 * (warmup_requests + 2)
               + settle_ms + measurement_ms + 500)
    if requests is None:
        requests = build_requests(profiles, phases_ms=phases_ms,
                                  horizon_ms=horizon, count_per_npu=finite_count)
    spec = next(s for s in routing_strategy_specs()
                if s.name == (strategy if strategy in ("baseline", "layer_once")
                               else "baseline"))
    kwargs = {}
    controller = None
    if strategy.startswith("dedicated_"):
        controller = ManifestCIRController(strategy.removeprefix("dedicated_"), headroom)
        kwargs.update(npu_dedicated_paths=PATHS,
                      qos_config=qos_configs_from_path_cirs(((0.0,) * 256,))[0],
                      control=CIRControlConfig(callback=controller,
                                               on_batch_boundary=True))
    else:
        kwargs["qos_config"] = static_qos_config()
    if finite_count is None and not complete_all:
        kwargs["steady_state"] = SteadyStateConfig(
            warmup_requests_per_npu=warmup_requests, settle_ms=settle_ms,
            measurement_ms=measurement_ms, block_ms=min(100.0, measurement_ms),
            slo_alpha=2.0, timeline_diagnostics=False)
    started = time.perf_counter()
    summary = simulate_continuous_batch(
        requests, num_npu=4, num_ssu=1, n_layers=8, batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR, cross_request_layer0_prefetch=True,
        client_io_config=spec.client_config(), disk_bw_gbps=40.0,
        npu_bw_gbps=50.0, pressure_ttl_ms=0.0, cir_write_threshold_gbps=0.0,
        submit_order_seed=seed, **kwargs)
    wall_seconds = time.perf_counter() - started
    for row in summary.get("request_rows", []):
        request = next(r for r in requests if r.request_id == row["request_id"])
        row["arrival_time_ms"] = request.arrival_time_ms
        row["arrival_to_completion_ms"] = row["completion_time_ms"] - request.arrival_time_ms
        row["io_stall_ms"] = row["ttft_ms"] - row["ideal_ttft_ms"]
    return {
        "strategy": strategy, "seed": seed, "phases_ms": list(phases_ms),
        "profiles": [asdict(profile) for profile in profiles],
        "nominal_demand_gib_s": request_weighted_nominal_demand(requests)["sum_gib_s"],
        "manifest_ideal_demand": request_weighted_nominal_demand(requests),
        "input_fingerprint": continuous_batch_input_fingerprint(requests),
        "io_bytes": IO_BYTES, "n_layers": 8, "cross_request_prefetch": True,
        "measurement_config": {"warmup_requests": warmup_requests,
                               "settle_ms": settle_ms, "measurement_ms": measurement_ms,
                               "complete_all": complete_all},
        "wall_seconds": wall_seconds,
        "controller_decisions": controller.decisions if controller else [],
        "controller_information": "only arrived request manifests and public queue counts",
        "controller_model": "npu_stall_predictor.required_rate_gib_s; planning estimate, not service guarantee",
        "implementation_sha256": implementation_hashes,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", default=["baseline", "layer_once", "dedicated_demand"])
    parser.add_argument("--case", default="rounded_old")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement-ms", type=float, default=1000.0)
    parser.add_argument("--settle-ms", type=float, default=100.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--phases-ms", type=float, nargs=4, default=(0.0,) * 4)
    parser.add_argument("--finite-count", type=int)
    parser.add_argument("--complete-all", action="store_true")
    parser.add_argument("--arrival-gib-s", type=float, default=39.2)
    parser.add_argument("--duration-ms", type=float, default=2000.0)
    parser.add_argument("--initial-backlog", type=int, default=6)
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument("--mix", choices=("long", "broad"), default="long")
    parser.add_argument("--headroom", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/stall_prediction_experiments/data"))
    args = parser.parse_args()
    cases = {
        "rounded_old": old_rounded_profiles(),
        "asymmetric_095": profiles_for_demand((8, 1536, 1536, 1536), (0.5, 21.5, 8, 8)),
        "asymmetric_098": profiles_for_demand((8, 1536, 1536, 1536), (0.5, 22.7, 8, 8)),
        "two_short_098": profiles_for_demand((8, 32, 1536, 1536), (0.6, 1.6, 26, 11)),
        "tiny_victims_098": profiles_for_demand((1, 2, 1536, 1536), (1.2, 2, 25, 11)),
        "equal_098": profiles_for_demand((256,) * 4, (9.8,) * 4),
        "same_blocks_skew_098": profiles_for_demand((1536,) * 4, (27.2, 4, 4, 4)),
        "small_victim_skew_098": profiles_for_demand((8, 1536, 1536, 1536), (0.5, 28.7, 5, 5)),
    }
    if args.case == "direct_data":
        cases["direct_data"] = direct_data_profiles()
    if args.case == "raw_sticky":
        cases["raw_sticky"] = direct_data_profiles(((32, 512), (32, 512), (192, 512), (192, 4096)))
    if args.case == "calibrated_two_ls":
        cases["calibrated_two_ls"] = calibrated_two_ls_profiles()
    requests, variable_metadata = None, None
    if args.case == "variable_raw":
        requests, profiles, variable_metadata = build_variable_requests(
            seed=args.seed, arrival_gib_s=args.arrival_gib_s,
            duration_ms=args.duration_ms, initial_backlog_per_npu=args.initial_backlog,
            burst_size=args.burst_size, mix=args.mix)
    else:
        profiles = cases[args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    for strategy in args.strategies:
        result = run_case(profiles, strategy, seed=args.seed,
                          measurement_ms=args.measurement_ms,
                          settle_ms=args.settle_ms, warmup_requests=args.warmup_requests,
                          phases_ms=tuple(args.phases_ms),
                          finite_count=args.finite_count, headroom=args.headroom,
                          complete_all=args.complete_all or requests is not None,
                          requests=requests)
        if args.complete_all or requests is not None:
            result["common_absolute_window"] = summarize_complete_run(result)
        if variable_metadata is not None:
            result["variable_workload_metadata"] = variable_metadata
        name = f"{args.case}_{strategy}_seed{args.seed}_{args.measurement_ms:g}ms"
        if args.finite_count is not None:
            name += f"_finite{args.finite_count}"
        if args.complete_all:
            name += "_complete_all"
        if requests is not None:
            name += f"_variable_t{args.duration_ms:g}_backlog{args.initial_backlog}_burst{args.burst_size}"
            if args.mix != "long":
                name += f"_mix{args.mix}"
        if args.headroom != 1.0:
            name += f"_headroom{args.headroom:g}"
        if any(args.phases_ms):
            name += "_phases" + "_".join(f"{p:g}" for p in args.phases_ms)
        if args.settle_ms != 100.0 or args.warmup_requests != 2:
            name += f"_settle{args.settle_ms:g}_warm{args.warmup_requests}"
        (args.output / f"{name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = result.get("common_absolute_window", result["summary"])
        print(json.dumps({"case": args.case, "strategy": strategy,
                          "demand": result["nominal_demand_gib_s"],
                          "mean_util": summary.get("mean_npu_utilization"),
                          "npu_utils": summary.get("npu_utilizations"),
                          "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
