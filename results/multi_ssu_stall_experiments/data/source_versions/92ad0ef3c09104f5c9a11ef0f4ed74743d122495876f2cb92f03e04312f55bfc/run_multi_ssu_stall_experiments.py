"""Same-input 32-NPU / 6-or-7-SSU experiments on the unchanged simulator.

One I/O is exactly one 176 KiB GLM block. Raw data profile values and native
consistent-hash placement are retained. A fixed absolute 1-second window and
the complete finite request population are both measured.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import sim
from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import (
    CIRControlConfig, ContinuousBatchRequest, continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import qos_configs_from_path_cirs, routing_strategy_specs, static_qos_config
from policy_logic import dedicated_path_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/multi_ssu_stall_experiments"
IO_BYTES = 176 * 1024
IO_GIB = IO_BYTES / 2**30
KEYS = ((32, 128), (32, 256), (64, 512), (96, 512), (192, 512), (192, 1024), (192, 2048))
CORE_FILES = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")


def input_demand(requests, num_npu, num_ssu, layers=8):
    compute = [0.0] * num_npu
    work = [[0.0] * num_ssu for _ in range(num_npu)]
    for request in requests:
        n = request.npu_id
        compute[n] += layers * request.load["per_layer_us"] / 1000
        for placement in request.placement:
            multiplier = layers if len(request.placement) == 1 else 1
            for s, volume in placement:
                work[n][s] += multiplier * volume
    matrix = [[1000 * v / c if c else 0.0 for v in row] for row, c in zip(work, compute)]
    per_ssu = [sum(row[s] for row in matrix) for s in range(num_ssu)]
    per_npu = list(map(sum, matrix))
    return {
        "matrix_gib_s": matrix, "per_ssu_gib_s": per_ssu, "per_npu_gib_s": per_npu,
        "total_gib_s": sum(per_npu), "aggregate_load_ratio": sum(per_npu) / (num_ssu * 40),
        "hottest_ssu_load_ratio": max(per_ssu) / 40,
        "largest_npu_receive_load_ratio": max(per_npu) / 50,
        "per_npu_ideal_compute_ms": compute, "total_read_gib": sum(map(sum, work)),
    }


def build_workload(*, num_npu=32, num_ssu=6, seed=20260906,
                   regime="near", initial_backlog=4, quota_cycles=1):
    """Shuffle the SAME profile multiset independently on each original NPU.

    replicated: original broad mix; intentionally overloaded with 6/7 SSUs.
    near_total: roughly 96--97% of aggregate bandwidth, may overload a hot SSU.
    near: move the fewest 192K/1024 profiles to raw 192K/2048 until every SSU's
    finite input ideal demand is <=39.96 GiB/s. This is input construction,
    never an online policy's access to future requests. All policies share it.

    Each lane starts with initial_backlog arrived requests. Subsequent arrivals
    follow its ideal compute cadence shifted earlier by that starting backlog.
    This explicitly models a queued service, not an empty-system rate promise.
    """
    table, provenance = load_authenticated_bw_table(num_npu)
    start_replacements = 0 if regime == "replicated" else (4 if num_ssu == 6 else 2)
    for replacements in range(start_replacements, 9):
        counts = [4, 2, 2, 2, 4, 8 - replacements, replacements]
        requests = []
        schedules = []
        for n in range(num_npu):
            rng = random.Random(seed + n * 100003)
            deck = []
            for _ in range(quota_cycles):
                cycle = [p for p, count in enumerate(counts) for _ in range(count)]
                rng.shuffle(cycle)
                deck.extend(cycle)
            startup_ms = sum(8 * table[KEYS[p]][1] / 1000 for p in deck[:initial_backlog - 1])
            elapsed = 0.0
            schedules.append(deck)
            for generation, p in enumerate(deck):
                seq, nql = KEYS[p]
                bw, per_layer_us, ttft, volume = table[seq, nql]
                k = (seq * 1024 - nql) // 128
                assert abs(volume - k * IO_GIB) < 1e-12
                rid = n * 1_000_000 + generation
                arrival = max(0.0, elapsed - startup_ms)
                load = {
                    "request_id": rid, "npu_id": n, "seq_len_k": seq, "nql": nql,
                    "category": sim.classify_request(seq, nql), "per_layer_us": per_layer_us,
                    "per_layer_kv_gb": k * IO_GIB, "required_bw_input_gbps": bw,
                    "arrival_time": arrival, "arrival_ms": arrival, "generation": generation,
                    "source_ttft_ms": ttft, "constructed_profile": False,
                }
                placement = (tuple((sim.block_ring_hash_disk_id(rid, j, num_ssu), IO_GIB)
                                   for j in range(k)),)
                requests.append(ContinuousBatchRequest(rid, n, arrival, load, placement))
                elapsed += 8 * per_layer_us / 1000
        requests = tuple(requests)
        demand = input_demand(requests, num_npu, num_ssu)
        if regime != "near" or demand["hottest_ssu_load_ratio"] <= 0.999:
            break
    profiles = [{"seq_len_k": seq, "nql": nql, "kv_blocks": (seq * 1024 - nql) // 128,
                 "compute_ms": table[seq, nql][1] / 1000, "quota": count}
                for (seq, nql), count in zip(KEYS, counts) if count]
    initial_bytes = sum(8 * r.load["per_layer_kv_gb"] for r in requests if r.arrival_time_ms == 0)
    metadata = {
        "num_npu": num_npu, "num_ssu": num_ssu, "seed": seed, "regime": regime,
        "profiles": profiles, "source": provenance, "io_bytes": IO_BYTES, "n_layers": 8,
        "placement": "native block_ring_hash, unchanged by NPU assignment",
        "quota_cycles": quota_cycles, "initial_backlog": initial_backlog,
        "request_count": len(requests), "last_arrival_ms": max(r.arrival_time_ms for r in requests),
        "initial_arrived_read_gib": initial_bytes,
        "input_demand": demand,
        "sampling_caveat": "raw data parameters; profile frequencies and arrivals are synthetic, not an observed user trace",
    }
    return requests, metadata


def summarize(summary, start_ms=1000.0, end_ms=2000.0):
    n = summary["num_npu"]
    compute, active = [0.0] * n, [0.0] * n
    for batch in summary["microbatch_metrics"]:
        i = batch["npu_id"]
        active[i] += max(0.0, min(end_ms, batch["completion_time_ms"])
                         - max(start_ms, batch["admission_time_ms"]))
        for layer in batch["layer_metrics"]:
            compute[i] += max(0.0, min(end_ms, layer["compute_end_ms"])
                              - max(start_ms, layer["compute_start_ms"]))
    duration = end_ms - start_ms
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "mean_npu_utilization": sum(compute) / (n * duration),
        "npu_utilizations": [c / duration for c in compute],
        "active_ms_by_npu": active,
        "all_npus_active_whole_window": all(abs(a - duration) < 1e-7 for a in active),
        "io_stall_ms_by_npu": [a - c for a, c in zip(active, compute)],
        "idle_ms_by_npu": [duration - a for a in active],
        "full_run_mean_npu_utilization": summary["fleet_npu_compute_utilization"],
        "makespan_ms": summary["makespan_ms"],
        "mean_arrival_to_completion_ms": summary["avg_request_latency_ms"],
        "p99_arrival_to_completion_ms": summary["p99_request_latency_ms"],
        "mean_admission_wait_ms": summary["avg_admission_wait_ms"],
        "mean_io_stall_ms": summary["avg_io_stall_ms"],
        "complete_request_count": summary["request_count"],
    }


def run_case(requests, metadata, *, strategy="baseline", assignment="none", min_interval_ms=0.0):
    n, s = metadata["num_npu"], metadata["num_ssu"]
    source_names = list(CORE_FILES) + ["run_multi_ssu_stall_experiments.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_names}
    route = strategy if strategy in ("baseline", "layer_once") else "baseline"
    client = next(spec for spec in routing_strategy_specs() if spec.name == route).client_config()
    kwargs, controller, hook = {}, None, nullcontext()
    if strategy not in ("baseline", "layer_once"):
        from multi_ssu_qos_controller import MultiSSUController, multi_ssu_control_events
        controller = MultiSSUController(mode=strategy)
        hook = multi_ssu_control_events(controller)
        kwargs = dict(
            npu_dedicated_paths=tuple(dedicated_path_id(i) for i in range(n)),
            qos_configs_by_ssu=qos_configs_from_path_cirs(tuple((0.0,) * 256 for _ in range(s))),
            control=CIRControlConfig(callback=controller, on_batch_boundary=True,
                                     min_interval_ms=min_interval_ms))
        hashes["multi_ssu_qos_controller.py"] = hashlib.sha256((ROOT / "multi_ssu_qos_controller.py").read_bytes()).hexdigest()
    else:
        kwargs["qos_config"] = static_qos_config()
    assign = nullcontext([])
    if assignment != "none":
        from multi_ssu_npu_assignment import assign_requests_on_arrival
        assign = assign_requests_on_arrival(policy=assignment)
        hashes["multi_ssu_npu_assignment.py"] = hashlib.sha256((ROOT / "multi_ssu_npu_assignment.py").read_bytes()).hexdigest()
    started = time.perf_counter()
    with hook, assign as assignment_log:
        summary = simulate_continuous_batch(
            requests, num_npu=n, num_ssu=s, n_layers=8, batch_size=1,
            policy=sim.POLICY_QOS_STATIC_CIR, cross_request_layer0_prefetch=True,
            client_io_config=client, disk_bw_gbps=40, npu_bw_gbps=50,
            pressure_ttl_ms=0, cir_write_threshold_gbps=0,
            submit_order_seed=metadata["seed"], **kwargs)
    assert all(summary["invariants"].values())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == hashes[name] for name in CORE_FILES)
    window = summarize(summary)
    return {
        "strategy": strategy, "assignment": assignment, "metadata": metadata,
        "input_fingerprint": continuous_batch_input_fingerprint(requests),
        "submit_seed": metadata["seed"], "core_and_policy_sha256": hashes,
        "wall_seconds": time.perf_counter() - started, "common_window": window,
        "control_log": controller.decisions if controller is not None else [],
        "control_statistics": controller.statistics() if controller is not None else {},
        "assignment_log": assignment_log, "summary": summary,
        "modeled_control_latency_ms": 0, "control_min_interval_ms": min_interval_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-npu", type=int, default=32)
    parser.add_argument("--num-ssu", type=int, choices=(6, 7), default=6)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--regime", choices=("replicated", "near_total", "near"), default="near")
    parser.add_argument("--strategy", choices=("baseline", "layer_once", "demand", "deadline",
                                               "deadline_reserve", "least_slack"), default="baseline")
    parser.add_argument("--assignment", choices=("none", "compute", "fluid"), default="none")
    parser.add_argument("--quota-cycles", type=int, default=1)
    parser.add_argument("--initial-backlog", type=int, default=4)
    parser.add_argument("--min-interval-ms", type=float, default=0)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT / "data")
    args = parser.parse_args()
    requests, meta = build_workload(num_npu=args.num_npu, num_ssu=args.num_ssu,
                                   seed=args.seed, regime=args.regime,
                                   initial_backlog=args.initial_backlog, quota_cycles=args.quota_cycles)
    if args.describe_only:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return
    result = run_case(requests, meta, strategy=args.strategy,
                      assignment=args.assignment, min_interval_ms=args.min_interval_ms)
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"npu{args.num_npu}_ssu{args.num_ssu}_{args.regime}_seed{args.seed}_{args.strategy}_{args.assignment}"
    if args.initial_backlog != 4 or args.quota_cycles != 1:
        name += f"_backlog{args.initial_backlog}_cycles{args.quota_cycles}"
    if args.min_interval_ms:
        name += f"_interval{args.min_interval_ms:g}"
    (args.output / (name + ".json")).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"case": name, "wall_seconds": result["wall_seconds"],
                      "window": result["common_window"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
