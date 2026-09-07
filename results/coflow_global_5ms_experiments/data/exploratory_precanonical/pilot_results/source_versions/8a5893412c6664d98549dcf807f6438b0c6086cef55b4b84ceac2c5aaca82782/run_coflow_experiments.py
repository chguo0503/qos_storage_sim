"""Global-coflow comparison with an explicit 218.315646777 GiB/s arrival rate.

All topologies use the previous 6-SSU same-trace request profiles and order.
Initial arrived backlog is reported separately; each later request arrives at
the cumulative eight-layer bytes divided by the target rate. This is an
external request-work arrival rate, not compute-weighted ideal SSD demand.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import sim
from continuous_batch_sim import (
    ContinuousBatchRequest, continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from run_shared_path_experiments import (
    ROOT, LAYERS, IO_BYTES, IO_GIB, DISK_GIB_S, NPU_GIB_S,
    _atomic_json, _hash, build_workload as build_reference_workload,
    case_name, input_demand, logical_input_fingerprint, save_result,
    source_files as shared_source_files, summarize_slo, summarize_window,
)


OUT = ROOT / "results/coflow_global_5ms_experiments/data"
STRATEGIES = ("baseline", "once", "new_once", "strategy1", "strategy2", "strategy3")
REFERENCE_STRATEGIES = STRATEGIES[:3]
TARGET_ARRIVAL_GIB_S = 218.315646777
REGIME = "arrival218"


def request_read_bytes(request):
    """Full eight-layer work; the workload stores one repeated layer manifest."""
    return LAYERS * len(request.placement[0]) * IO_BYTES


def retime_arrivals(requests, target_gib_s=TARGET_ARRIVAL_GIB_S):
    """Preserve the reference global arrival/rid order and the original tuple.

    The cumulative-byte clock starts at zero *after* excluding t=0 backlog.
    Thus the first noninitial request consumes a positive interval, and the
    last timestamp includes the last request's bytes. Zero-time initial work
    is never divided by an artificial observation interval.
    """
    postinitial = sorted((r for r in requests if r.arrival_time_ms > 0),
                         key=lambda r: (r.arrival_time_ms, r.request_id))
    cumulative_bytes = 0
    arrival_by_id = {}
    for request in postinitial:
        cumulative_bytes += request_read_bytes(request)
        arrival_by_id[request.request_id] = 1000 * (cumulative_bytes / 2**30) / target_gib_s
    result = []
    for request in requests:
        arrival = arrival_by_id.get(request.request_id, 0.0)
        load = dict(request.load)
        load.update(reference_arrival_ms=request.arrival_time_ms,
                    arrival_time=arrival, arrival_ms=arrival)
        result.append(ContinuousBatchRequest.from_normalized(
            request.request_id, request.npu_id, arrival, load, request.placement))
    return tuple(result)


def arrival_workload(requests, num_ssu):
    """Actual request-work rates from manifests, not their ideal V/C values."""
    post = [r for r in requests if r.arrival_time_ms > 0]
    initial = [r for r in requests if r.arrival_time_ms == 0]
    end_ms = max((r.arrival_time_ms for r in post), default=0.0)
    post_bytes = sum(request_read_bytes(r) for r in post)
    by_ssu_bytes = [0] * num_ssu
    initial_by_ssu_bytes = [0] * num_ssu
    for rows, by_ssu in ((post, by_ssu_bytes), (initial, initial_by_ssu_bytes)):
        for request in rows:
            for ssu, _ in request.placement[0]:
                by_ssu[ssu] += LAYERS * IO_BYTES
    rates = [value / 2**30 * 1000 / end_ms if end_ms else 0.0 for value in by_ssu_bytes]
    return {
        "target_postinitial_gib_s": TARGET_ARRIVAL_GIB_S,
        "actual_postinitial_gib_s": post_bytes / 2**30 * 1000 / end_ms if end_ms else None,
        "postinitial_request_count": len(post), "postinitial_read_bytes": post_bytes,
        "postinitial_read_gib": post_bytes / 2**30,
        "first_postinitial_arrival_ms": min((r.arrival_time_ms for r in post), default=None),
        "postinitial_last_arrival_ms": end_ms,
        "postinitial_read_bytes_by_ssu": by_ssu_bytes,
        "postinitial_read_gib_by_ssu": [value / 2**30 for value in by_ssu_bytes],
        "postinitial_gib_s_by_ssu": rates,
        "hottest_ssu_postinitial_gib_s": max(rates),
        "aggregate_arrival_load_ratio": sum(rates) / (num_ssu * DISK_GIB_S),
        "hottest_ssu_arrival_load_ratio": max(rates) / DISK_GIB_S,
        "initial_arrived_request_count": len(initial),
        "initial_arrived_read_bytes": sum(request_read_bytes(r) for r in initial),
        "initial_arrived_read_gib": sum(request_read_bytes(r) for r in initial) / 2**30,
        "initial_arrived_read_bytes_by_ssu": initial_by_ssu_bytes,
        "initial_work_excluded_from_rate": True,
        "rate_definition": "all 8-layer bytes of requests arriving after t=0 / last arrival time; t=0 backlog excluded",
        "retiming_rule": "sort reference positive arrivals by (arrival_ms, request_id); arrival_ms = 1000*cumulative_8layer_bytes/(2**30*target_gib_s)",
        "finite_window_caveat": "discrete request-work arrivals, not constant instantaneous SSD traffic or a per-window completion requirement",
    }


def build_workload(*, num_npu=32, num_ssu=6, seed=20260906, small=False):
    reference, metadata = build_reference_workload(
        num_npu=num_npu, num_ssu=num_ssu, seed=seed, regime="same_trace",
        initial_backlog=1 if small else 4, quota_cycles=1, small=small)
    requests = retime_arrivals(reference)
    arrivals = arrival_workload(requests, num_ssu)
    metadata = dict(metadata)
    metadata.update({
        "experiment_id": "coflow_global_5ms_v1",
        "regime": REGIME,
        "reference_regime": "previous fixed 6-SSU same_trace quota, five replacements",
        "reference_logical_input_fingerprint": logical_input_fingerprint(reference),
        "reference_last_arrival_ms": metadata["last_arrival_ms"],
        "last_arrival_ms": arrivals["postinitial_last_arrival_ms"],
        "initial_arrived_request_count": arrivals["initial_arrived_request_count"],
        "initial_arrived_read_gib": arrivals["initial_arrived_read_gib"],
        "input_demand": input_demand(requests, num_npu, num_ssu),
        "arrival_workload": arrivals,
        "logical_input_fingerprint": logical_input_fingerprint(requests),
        "near_topology_comparison_caveat": "not a near retuning: identical profile quotas, arrivals and original NPU bindings across topologies; only SSD placement changes",
        "small_caveat": "two raw profiles per original NPU with one initially arrived request; exercises positive arrivals but is not a steady one-second benchmark" if small else None,
    })
    return requests, metadata


def placement_fingerprint(requests):
    """Ignore execution NPU, retain every request's ordered physical manifest."""
    return _hash([{"request_id": r.request_id, "placement": r.placement}
                  for r in sorted(requests, key=lambda r: r.request_id)])


def source_files():
    extra = (Path(__file__).name, "sweep_coflow_experiments.py")
    return tuple(sorted(set(shared_source_files() + extra
                            + tuple(p.name for p in ROOT.glob("coflow_*.py")))))


def run_case(requests, metadata, *, strategy="baseline", collector_interval_ms=5.0,
             cir_min_interval_ms=100.0):
    if strategy not in STRATEGIES:
        raise ValueError("unknown global-coflow strategy")
    if collector_interval_ms != 5.0 or cir_min_interval_ms < 100.0:
        raise ValueError("requires 5 ms telemetry and CIR interval >=100 ms")
    # Describe-only and input tests intentionally do not import the new adapter.
    if strategy in REFERENCE_STRATEGIES:
        from shared_path_sim_adapter import shared_path_adapter
        manager = shared_path_adapter(strategy=strategy, collector_interval_ms=collector_interval_ms,
                                      cir_min_interval_ms=cir_min_interval_ms)
    else:
        from coflow_sim_adapter import coflow_adapter
        manager = coflow_adapter(policy=strategy, collector_interval_ms=collector_interval_ms,
                                 cir_min_interval_ms=cir_min_interval_ms)
    sources = {name: (ROOT / name).read_text() for name in source_files()}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    fingerprint = continuous_batch_input_fingerprint(requests)
    original_placement = placement_fingerprint(requests)
    route = "baseline" if strategy == "baseline" else "layer_once"
    client = next(s for s in routing_strategy_specs() if s.name == route).client_config()
    started = time.perf_counter()
    with manager as adapter:
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
        executed_placement = placement_fingerprint(
            runtime.manifest for runtime in adapter.context.requests.values())
    assert all(summary["invariants"].values())
    assert continuous_batch_input_fingerprint(requests) == fingerprint
    assert executed_placement == original_placement, "execution changed physical request placement"
    assert not adapter_statistics["cir_write_events"]
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()), "source changed during experiment"
    return {
        "schema_version": 1, "experiment": "coflow_global_5ms_arrival218",
        "strategy": strategy, "metadata": metadata,
        "input_fingerprint": fingerprint, "logical_input_fingerprint": metadata["logical_input_fingerprint"],
        "input_placement_fingerprint": original_placement,
        "execution_placement_fingerprint": executed_placement,
        "submit_seed": metadata["seed"], "core_and_policy_sha256": hashes,
        "wall_seconds": time.perf_counter() - started,
        "collector_interval_ms": collector_interval_ms, "cir_min_interval_ms": cir_min_interval_ms,
        "cir_policy": "shared FINAL_STATIC; no runtime CIR changes",
        "static_path_cirs_gib_s": list(static_qos_config().path_cirs),
        "common_window": summarize_window(summary), "slo": summarize_slo(summary, requests),
        "adapter_statistics": adapter_statistics, "assignment_log": assignment_log,
        "summary": summary, "modeled_control_latency_ms": 0, "_source_texts": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-npu", type=int, default=32)
    parser.add_argument("--num-ssu", type=int, choices=(5, 6, 7), default=6)
    parser.add_argument("--seed", type=int, choices=(20260906, 20260907), default=20260906)
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    requests, metadata = build_workload(num_npu=args.num_npu, num_ssu=args.num_ssu,
                                        seed=args.seed, small=args.small)
    if args.describe_only:
        print(json.dumps({**metadata, "input_fingerprint": continuous_batch_input_fingerprint(requests)},
                         ensure_ascii=False, indent=2))
        return
    destination = args.output / (case_name(metadata, args.strategy) + ".json")
    if destination.exists():
        raise FileExistsError(f"preserving existing result: {destination}")
    try:
        result = run_case(requests, metadata, strategy=args.strategy)
        destination = save_result(result, requests, args.output)
    except Exception as exc:
        _atomic_json(destination.with_suffix(".failure.json"), {
            "strategy": args.strategy, "metadata": metadata, "status": "failed",
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps({"case": destination.stem, "output": str(destination),
                      "wall_seconds": result["wall_seconds"], "window": result["common_window"],
                      "admission_slo": result["slo"]["all_requests"]["admission"],
                      "arrival_slo": result["slo"]["all_requests"]["arrival"],
                      "makespan_ms": result["summary"]["makespan_ms"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
