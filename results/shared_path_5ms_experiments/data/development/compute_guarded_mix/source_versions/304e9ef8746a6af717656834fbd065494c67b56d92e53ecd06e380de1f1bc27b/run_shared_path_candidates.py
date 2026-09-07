"""Development-only assignment/JIT candidates, without changing frozen clients.

Selection uses only the 20260906 development seed. Results are written below
development/<candidate>/ and labeled candidate_<name>, not as main-table
strategy1 results. The production-input builder, static CIR, 5 ms collector,
and complete-population SLO accounting are reused without modification.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import continuous_batch_sim as native
import shared_path_strategy1
from run_shared_path_experiments import (
    OUT, ROOT, _atomic_json, build_workload, case_name, run_case, save_result,
)


CANDIDATES = ("compute", "fair_pipeline", "compute_guarded_mix",
              "fair_pipeline_jit5", "fair_pipeline_jit10")
DEVELOPMENT_SEED = 20260906


@contextmanager
def jit_prefetch_events(release_function, guard_ms=None):
    """Delay only newly created client submission states, never queued I/O.

    The adapter inside run_case replaces pressure reads with cached 5 ms
    samples. This hook therefore cannot obtain a fresh SSD snapshot. Original
    io_start_time remains the activation timestamp; the actual delayed issue
    and any subsequent NPU wait remain part of the native timeline/accounting.
    """
    stats = {"enabled": guard_ms is not None, "guard_ms": guard_ms,
             "activation_count": 0, "delay_count": 0,
             "total_delay_ms": 0.0, "max_delay_ms": 0.0,
             "release_examples": [], "log_limit": 128,
             "modifies_already_queued_io": False,
             "telemetry": "only adapter-patched periodic-cache pressure reads"}
    if guard_ms is None:
        yield stats
        return
    original = native._start_layer_io

    def activate(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs):
        if layer < 0 or layer >= context.n_layers or request.io_started[layer]:
            return original(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs)
        placement = request.manifest.placement[0 if len(request.manifest.placement) == 1 else layer]
        counts = [0] * context.num_ssu
        for ssu, _size in placement:
            counts[ssu] += 1
        # During this call shared_path_adapter owns the patched report API.
        pressure = tuple(tuple(d.scheduler.report_path_pressure_analysis(now_ms).counts)
                         for d in context.disks)
        release_ms = release_function(
            now_ms, deadline_ms, tuple(counts), pressure,
            guard_ms=guard_ms, disk_bw_gib_s=context.disk_bw_gbps,
            link_bw_gib_s=context.npu_bw_gbps,
        )
        before = context.next_submission_id
        result = original(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs)
        for state_id in range(before, context.next_submission_id):
            state = context.submission_states[state_id]
            state.ready_time_ms = max(state.ready_time_ms, release_ms)
        delay = max(0.0, release_ms - now_ms)
        stats["activation_count"] += 1
        if delay > 1e-12:
            stats["delay_count"] += 1
            stats["total_delay_ms"] += delay
            stats["max_delay_ms"] = max(stats["max_delay_ms"], delay)
            if len(stats["release_examples"]) < stats["log_limit"]:
                stats["release_examples"].append({
                    "request_id": request.manifest.request_id,
                    "npu_id": request.manifest.npu_id, "layer": layer,
                    "activation_ms": now_ms, "deadline_ms": deadline_ms,
                    "release_ms": release_ms, "delay_ms": delay,
                    "layer_io_by_ssu": counts,
                    "cached_ssu_queue_io": [sum(row) for row in pressure],
                })
        return result

    with patch.object(native, "_start_layer_io", activate):
        yield stats


def run_candidate(requests, metadata, candidate):
    import experimental_shared_assignment as experimental
    if candidate not in CANDIDATES:
        raise ValueError("unknown development candidate")
    if metadata["seed"] != DEVELOPMENT_SEED:
        raise ValueError("candidate selection is restricted to development seed 20260906")
    mode = "fair_pipeline" if candidate.startswith("fair_pipeline_jit") else candidate
    guard = (float(candidate.rsplit("jit", 1)[1]) if "_jit" in candidate else None)
    extra_sources = {name: (ROOT / name).read_text() for name in
                     ("experimental_shared_assignment.py", Path(__file__).name)}
    extra_hashes = {name: hashlib.sha256(source.encode()).hexdigest()
                    for name, source in extra_sources.items()}
    choose = partial(experimental.choose_candidate, mode=mode)
    release = getattr(experimental, "prefetch_release_ms", None)
    if guard is not None and release is None:
        raise ValueError("JIT candidates require experimental.prefetch_release_ms")
    with patch.object(shared_path_strategy1, "choose_npu", choose), \
         jit_prefetch_events(release, guard) as jit_stats:
        result = run_case(requests, metadata, strategy="strategy1")
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in extra_hashes.items()), "candidate source changed during run"
    result.update({
        "strategy": f"candidate_{candidate}", "base_strategy": "strategy1",
        "development_only": True, "candidate_mode": candidate,
        "candidate_assignment_mode": mode, "development_seed": DEVELOPMENT_SEED,
        "candidate_stats": {"jit_prefetch": jit_stats},
        "selection_caveat": "development-seed candidate, not an independently validated main-table result",
    })
    result["core_and_policy_sha256"].update(extra_hashes)
    result["_source_texts"].update(extra_sources)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--num-ssu", type=int, choices=(5, 6, 7), default=6)
    parser.add_argument("--num-npu", type=int, default=32)
    parser.add_argument("--seed", type=int, choices=(DEVELOPMENT_SEED,), default=DEVELOPMENT_SEED)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT / "development")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    output = args.output / args.candidate
    started = time.perf_counter()
    metadata = None
    name = (f"npu{args.num_npu}_ssu{args.num_ssu}_near_seed{args.seed}"
            f"_candidate_{args.candidate}" + ("_small" if args.small else ""))
    try:
        requests, metadata = build_workload(num_npu=args.num_npu, num_ssu=args.num_ssu,
                                          seed=args.seed, regime="near", small=args.small)
        if args.describe_only:
            print(json.dumps({"candidate": args.candidate, "metadata": metadata,
                              "output": str(output)}, ensure_ascii=False, indent=2))
            return
        destination = output / (case_name(metadata, f"candidate_{args.candidate}") + ".json")
        if destination.exists():
            raise FileExistsError(f"preserving existing candidate result: {destination}")
        result = run_candidate(requests, metadata, args.candidate)
        destination = save_result(result, requests, output)
        print(json.dumps({"candidate": args.candidate, "status": "completed",
                          "output": str(destination), "wall_seconds": result["wall_seconds"],
                          "window": result["common_window"],
                          "admission_slo": result["slo"]["all_requests"]["admission"],
                          "arrival_slo": result["slo"]["all_requests"]["arrival"],
                          "makespan_ms": result["summary"]["makespan_ms"],
                          "candidate_stats": result["candidate_stats"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = {"candidate": args.candidate, "status": "failed", "development_only": True,
                   "num_npu": args.num_npu, "num_ssu": args.num_ssu, "seed": args.seed,
                   "small": args.small, "metadata": metadata,
                   "wall_seconds": time.perf_counter() - started,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
        _atomic_json(output / f"{name}.failure.json", failure)
        print(json.dumps({"candidate": args.candidate, "status": "failed", "error": str(exc)}), flush=True)
        raise


if __name__ == "__main__":
    main()
