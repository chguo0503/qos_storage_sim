"""Matched shared-Path experiments: 32 NPUs, 5/6/7 SSUs, 5 ms telemetry.

All five clients use the same static FINAL_STATIC CIR table and 176 KiB I/O.
Policy-specific routing/arrival assignment is supplied by the separately
audited shared_path_sim_adapter. This runner owns inputs, accounting and
provenance; it does not reuse the earlier dedicated-Path A/B controllers.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import sim
from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import (
    ContinuousBatchRequest, continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/shared_path_5ms_experiments/data"
STRATEGIES = ("baseline", "once", "new_once", "strategy1", "strategy2")
KEYS = ((32, 128), (32, 256), (64, 512), (96, 512),
        (192, 512), (192, 1024), (192, 2048))
LAYERS = 8
IO_BYTES = 176 * 1024
IO_GIB = IO_BYTES / 2**30
DISK_GIB_S = 40.0
NPU_GIB_S = 50.0
CORE_FILES = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py",
              "policy_logic.py", "continuous_batch_control.py", "strategy_profiles.py",
              "authenticated_workload_inputs.py", "random_steady_state_workload.py",
              "continuous_prefill_workload.py", "six_request_workload.py", "data")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def input_demand(requests, num_npu, num_ssu):
    """Ideal compute-weighted demand; not an external arrival-rate claim."""
    compute = [0.0] * num_npu
    work = [[0.0] * num_ssu for _ in range(num_npu)]
    for request in requests:
        n = request.npu_id
        compute[n] += LAYERS * request.load["per_layer_us"] / 1000
        for placement in request.placement:
            multiplier = LAYERS if len(request.placement) == 1 else 1
            for s, volume in placement:
                work[n][s] += multiplier * volume
    matrix = [[1000 * v / c if c else 0.0 for v in row] for row, c in zip(work, compute)]
    per_ssu = [sum(row[s] for row in matrix) for s in range(num_ssu)]
    per_npu = list(map(sum, matrix))
    return {
        "matrix_gib_s": matrix, "per_ssu_gib_s": per_ssu, "per_npu_gib_s": per_npu,
        "total_gib_s": sum(per_npu), "aggregate_load_ratio": sum(per_npu) / (num_ssu * DISK_GIB_S),
        "hottest_ssu_load_ratio": max(per_ssu) / DISK_GIB_S,
        "largest_npu_receive_load_ratio": max(per_npu) / NPU_GIB_S,
        "per_npu_ideal_compute_ms": compute, "total_read_gib": sum(map(sum, work)),
    }


def logical_input_fingerprint(requests):
    """Exclude physical SSD placement to identify a true cross-topology trace."""
    return _hash([{
        "request_id": r.request_id, "npu_id": r.npu_id,
        "arrival_time_ms": r.arrival_time_ms, "load": dict(r.load),
    } for r in requests])


def build_workload(*, num_npu=32, num_ssu=6, seed=20260906,
                   regime="near", initial_backlog=4, quota_cycles=1, small=False):
    """Create an immutable varying-request trace with direct raw-data profiles.

    near: replace the fewest 192K/1024 entries with 192K/2048 entries until
    every SSU's ideal demand is <=39.96 GiB/s. The finite catalog recipe has
    22 requests/NPU/cycle and reaches 5-SSU capacity without extrapolating C.
    same_trace: fix five replacements for every topology; request parameters
    and arrivals are identical across 5/6/7 SSUs, placement alone is rehashed.
    small: two shortest raw profiles per NPU, keeping requested topology;
    this is a smoke test, not a valid steady 1-second utilization experiment.
    """
    if regime not in ("near", "same_trace") or num_ssu not in (5, 6, 7):
        raise ValueError("use near/same_trace and 5, 6 or 7 SSUs")
    table, provenance = load_authenticated_bw_table(num_npu)
    placement_cache = {}
    largest_blocks = max((seq * 1024 - nql) // 128 for seq, nql in KEYS)

    def placement_for(rid, blocks):
        if rid not in placement_cache:
            placement_cache[rid] = tuple(
                (sim.block_ring_hash_disk_id(rid, k, num_ssu), IO_GIB)
                for k in range(largest_blocks if not small else blocks)
            )
        return placement_cache[rid][:blocks]

    candidates = (5,) if regime == "same_trace" else range(9)
    if small:
        candidates = (0,)
    for replacements in candidates:
        counts = (1, 1, 0, 0, 0, 0, 0) if small else (4, 2, 2, 2, 4, 8 - replacements, replacements)
        requests = []
        cycles = 1 if small else quota_cycles
        backlog = min(initial_backlog, sum(counts) * cycles)
        for n in range(num_npu):
            rng = random.Random(seed + n * 100003)
            deck = []
            for _ in range(cycles):
                cycle = [p for p, count in enumerate(counts) for _ in range(count)]
                rng.shuffle(cycle)
                deck.extend(cycle)
            startup = sum(LAYERS * table[KEYS[p]][1] / 1000 for p in deck[:backlog - 1])
            elapsed = 0.0
            for generation, p in enumerate(deck):
                seq, nql = KEYS[p]
                bw, per_layer_us, source_ttft_ms, volume = table[seq, nql]
                blocks = (seq * 1024 - nql) // 128
                assert math.isclose(volume, blocks * IO_GIB, rel_tol=0, abs_tol=1e-12)
                rid = n * 1_000_000 + generation
                arrival = max(0.0, elapsed - startup)
                load = {
                    "request_id": rid, "npu_id": n, "seq_len_k": seq, "nql": nql,
                    "category": sim.classify_request(seq, nql), "per_layer_us": per_layer_us,
                    "per_layer_kv_gb": blocks * IO_GIB, "required_bw_input_gbps": bw,
                    "arrival_time": arrival, "arrival_ms": arrival, "generation": generation,
                    "source_ttft_ms": source_ttft_ms, "constructed_profile": False,
                }
                requests.append(ContinuousBatchRequest.from_normalized(
                    rid, n, arrival, load, (placement_for(rid, blocks),)))
                elapsed += LAYERS * per_layer_us / 1000
        requests = tuple(requests)
        demand = input_demand(requests, num_npu, num_ssu)
        if small or regime == "same_trace" or demand["hottest_ssu_load_ratio"] <= 0.999:
            break
    if not small and regime == "near" and demand["hottest_ssu_load_ratio"] > 0.999:
        raise ValueError("this finite raw-profile recipe cannot satisfy the hottest-SSU target")
    profiles = [{
        "seq_len_k": seq, "nql": nql, "kv_blocks": (seq * 1024 - nql) // 128,
        "compute_ms": table[seq, nql][1] / 1000, "quota": count,
        "ideal_8layer_ms": LAYERS * table[seq, nql][1] / 1000,
        "slo_1p5_ms": 1.5 * LAYERS * table[seq, nql][1] / 1000,
        "source_ttft_ms_not_used_as_8layer_slo": table[seq, nql][2],
    } for (seq, nql), count in zip(KEYS, counts) if count]
    metadata = {
        "num_npu": num_npu, "num_ssu": num_ssu, "seed": seed, "regime": regime,
        "small": small, "profiles": profiles, "source": provenance,
        "io_bytes": IO_BYTES, "n_layers": LAYERS,
        "ssd_bandwidth_gib_s": DISK_GIB_S, "npu_link_bandwidth_gib_s": NPU_GIB_S,
        "placement": "native block_ring_hash, fixed within topology across all strategies",
        "quota_cycles": cycles, "initial_backlog": backlog, "replacements": replacements,
        "request_count": len(requests), "requests_per_original_npu": sum(counts) * cycles,
        "last_arrival_ms": max(r.arrival_time_ms for r in requests),
        "initial_arrived_request_count": sum(r.arrival_time_ms == 0 for r in requests),
        "initial_arrived_read_gib": sum(LAYERS * r.load["per_layer_kv_gb"]
                                        for r in requests if r.arrival_time_ms == 0),
        "input_demand": demand,
        "logical_input_fingerprint": logical_input_fingerprint(requests),
        "sampling_caveat": "raw data values; profile weights and arrival trace are constructed, not observed production frequencies",
        "near_topology_comparison_caveat": "near may change profile quotas across SSU counts; same_trace does not",
    }
    return requests, metadata


def summarize_window(summary, start_ms=1000.0, end_ms=2000.0):
    n = summary["num_npu"]
    compute, active = [0.0] * n, [0.0] * n
    def overlap(a, b):
        return max(0.0, min(end_ms, b) - max(start_ms, a))
    for batch in summary["microbatch_metrics"]:
        i = batch["npu_id"]
        active[i] += overlap(batch["admission_time_ms"], batch["completion_time_ms"])
        for layer in batch["layer_metrics"]:
            compute[i] += overlap(layer["compute_start_ms"], layer["compute_end_ms"])
    duration = end_ms - start_ms
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "mean_npu_utilization": sum(compute) / (n * duration),
        "npu_utilizations": [c / duration for c in compute],
        "compute_ms_by_npu": compute, "active_ms_by_npu": active,
        "all_npus_active_whole_window": all(abs(a - duration) < 1e-7 for a in active),
        "io_stall_ms_by_npu": [a - c for a, c in zip(active, compute)],
        "idle_ms_by_npu": [duration - a for a in active],
        "full_run_mean_npu_utilization": summary["fleet_npu_compute_utilization"],
        "makespan_ms": summary["makespan_ms"],
        "mean_arrival_to_completion_ms": summary["avg_request_latency_ms"],
        "p99_arrival_to_completion_ms": summary["p99_request_latency_ms"],
        "mean_admission_to_completion_ms": summary["avg_processing_latency_ms"],
        "mean_admission_wait_ms": summary["avg_admission_wait_ms"],
        "mean_io_stall_ms": summary["avg_io_stall_ms"],
        "complete_request_count": summary["request_count"],
    }


def summarize_slo(summary, requests, *, alpha=1.5, start_ms=1000.0, end_ms=2000.0):
    """Recompute from every final request record; never censor at window end."""
    rows = summary["request_metrics"]
    expected_ids = {r.request_id for r in requests}
    assert len(rows) == len(expected_ids) == len(requests)
    assert {r["request_id"] for r in rows} == expected_ids
    assert all(math.isfinite(r["completion_time_ms"]) for r in rows)

    def cohort(sample):
        def metric(start):
            passed = sum(r["completion_time_ms"] - r[start]
                         <= alpha * r["own_compute_ms"] + 1e-9 for r in sample)
            return {"passed": passed, "count": len(sample),
                    "rate": passed / len(sample) if sample else None}
        return {"admission": metric("admission_time_ms"),
                "arrival": metric("arrival_time_ms"),
                "request_ids": [r["request_id"] for r in sample],
                "completion_after_window_end_count": sum(r["completion_time_ms"] > end_ms for r in sample)}

    profile_by_id = {r.request_id: (r.load["seq_len_k"], r.load["nql"]) for r in requests}
    groups = defaultdict(list)
    for row in rows:
        groups[profile_by_id[row["request_id"]]].append(row)
    by_profile = [{"seq_len_k": key[0], "nql": key[1], **cohort(group)}
                  for key, group in sorted(groups.items())]
    return {
        "alpha": alpha, "ideal": "n_layers * per_layer_compute_ms (own_compute_ms)",
        "threshold_excludes_source_data_78layer_ttft": True,
        "primary": "admission_to_completion; excludes client admission queue",
        "secondary": "arrival_to_completion; includes client admission queue",
        "all_input_requests_completed": summary["invariants"]["all_requests_completed"],
        "all_requests": cohort(rows),
        "window_start_ms": start_ms, "window_end_ms": end_ms,
        "window_admissions": cohort([r for r in rows if start_ms <= r["admission_time_ms"] < end_ms]),
        "window_arrivals": cohort([r for r in rows if start_ms <= r["arrival_time_ms"] < end_ms]),
        "window_cohort_caveat": "admission cohorts can differ across policies; all_requests is the matched complete population",
        "per_profile": by_profile,
    }


def source_files():
    return tuple(sorted(set(CORE_FILES + (Path(__file__).name, "shared_ssu_state.py")
                            + tuple(p.name for p in ROOT.glob("shared_path_*.py")))))


def run_case(requests, metadata, *, strategy="baseline", collector_interval_ms=5.0,
             cir_min_interval_ms=100.0):
    from shared_path_sim_adapter import shared_path_adapter
    if strategy not in STRATEGIES:
        raise ValueError("unknown shared-Path strategy")
    if collector_interval_ms != 5.0 or cir_min_interval_ms < 100.0:
        raise ValueError("this experiment requires 5 ms collection and CIR interval >=100 ms")
    sources = {name: (ROOT / name).read_text() for name in source_files()}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    fingerprint = continuous_batch_input_fingerprint(requests)
    route = "baseline" if strategy == "baseline" else "layer_once"
    client = next(s for s in routing_strategy_specs() if s.name == route).client_config()
    started = time.perf_counter()
    with shared_path_adapter(strategy=strategy, collector_interval_ms=collector_interval_ms,
                             cir_min_interval_ms=cir_min_interval_ms) as adapter:
        summary = simulate_continuous_batch(
            requests, num_npu=metadata["num_npu"], num_ssu=metadata["num_ssu"],
            n_layers=LAYERS, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=static_qos_config(), client_io_config=client,
            cross_request_layer0_prefetch=True, pressure_ttl_ms=collector_interval_ms,
            disk_bw_gbps=DISK_GIB_S, npu_bw_gbps=NPU_GIB_S,
            submit_order_seed=metadata["seed"], control=None,
        )
        adapter_statistics = adapter.statistics()
        assignment_log = list(adapter.assignment_log)
    assert all(summary["invariants"].values())
    assert summary["invariants"]["all_requests_completed"]
    assert continuous_batch_input_fingerprint(requests) == fingerprint
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()), "source changed during experiment"
    qos = static_qos_config()
    return {
        "schema_version": 1, "strategy": strategy, "metadata": metadata,
        "input_fingerprint": fingerprint, "logical_input_fingerprint": metadata["logical_input_fingerprint"],
        "submit_seed": metadata["seed"], "core_and_policy_sha256": hashes,
        "wall_seconds": time.perf_counter() - started,
        "collector_interval_ms": collector_interval_ms, "cir_min_interval_ms": cir_min_interval_ms,
        "cir_policy": "shared FINAL_STATIC; no runtime CIR changes",
        "static_path_cirs_gib_s": list(qos.path_cirs),
        "common_window": summarize_window(summary), "slo": summarize_slo(summary, requests),
        "adapter_statistics": adapter_statistics, "assignment_log": assignment_log,
        "summary": summary, "modeled_control_latency_ms": 0,
        "_source_texts": sources,
    }


def _atomic_json(path, value):
    """Generated artifact writer; same-fingerprint inputs may be built in parallel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as temporary:
        json.dump(value, temporary, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary.write("\n")
        temporary_path = temporary.name
    os.replace(temporary_path, path)


def case_name(metadata, strategy):
    name = (f"npu{metadata['num_npu']}_ssu{metadata['num_ssu']}_{metadata['regime']}"
            f"_seed{metadata['seed']}_{strategy}")
    if metadata["small"]:
        name += "_small"
    elif metadata["initial_backlog"] != 4 or metadata["quota_cycles"] != 1:
        name += f"_backlog{metadata['initial_backlog']}_cycles{metadata['quota_cycles']}"
    return name


def save_result(result, requests, output):
    output = Path(output)
    fingerprint = result["input_fingerprint"]
    input_path = output / "inputs" / f"{fingerprint}.json"
    if not input_path.exists():
        _atomic_json(input_path, {
            "input_fingerprint": fingerprint,
            "logical_input_fingerprint": result["logical_input_fingerprint"],
            "metadata": result["metadata"],
            "requests": [{"request_id": r.request_id, "npu_id": r.npu_id,
                          "arrival_time_ms": r.arrival_time_ms, "load": dict(r.load),
                          "placement": r.placement} for r in requests],
        })
    result["input_artifact"] = {
        "path_relative_to_data": str(input_path.relative_to(output)),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    sources = result.pop("_source_texts", {})
    paths = {}
    for name, source in sources.items():
        digest = result["core_and_policy_sha256"][name]
        path = output / "source_versions" / digest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=name + ".", delete=False) as temporary:
                temporary.write(source)
                temporary_path = temporary.name
            os.replace(temporary_path, path)
        paths[name] = str(path.relative_to(output))
    result["source_artifacts"] = paths
    destination = output / (case_name(result["metadata"], result["strategy"]) + ".json")
    _atomic_json(destination, result)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-npu", type=int, default=32)
    parser.add_argument("--num-ssu", type=int, choices=(5, 6, 7), default=6)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--regime", choices=("near", "same_trace"), default="near")
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--initial-backlog", type=int, default=4)
    parser.add_argument("--quota-cycles", type=int, default=1)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    requests, metadata = build_workload(num_npu=args.num_npu, num_ssu=args.num_ssu,
        seed=args.seed, regime=args.regime, initial_backlog=args.initial_backlog,
        quota_cycles=args.quota_cycles, small=args.small)
    if args.describe_only:
        print(json.dumps({**metadata, "input_fingerprint": continuous_batch_input_fingerprint(requests)},
                         ensure_ascii=False, indent=2))
        return
    destination = args.output / (case_name(metadata, args.strategy) + ".json")
    if destination.exists():
        raise FileExistsError(f"preserving existing result: {destination}")
    result = run_case(requests, metadata, strategy=args.strategy)
    destination = save_result(result, requests, args.output)
    print(json.dumps({"case": destination.stem, "output": str(destination),
                      "wall_seconds": result["wall_seconds"], "window": result["common_window"],
                      "slo_all": result["slo"]["all_requests"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
