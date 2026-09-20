#!/usr/bin/env python3
"""Small distinct profile perturbations with 19 long and 13 short fixed lanes.

Only input profiles vary: source-data compute is bilinearly interpolated in
total length and NQL, with no extrapolation. Hit-prefix lengths and the native
ring-hash placement stay unchanged from raw case 0. This is fixed role
separation, not a workload that mixes long and short requests on every NPU.
"""
import argparse
import ast
import bisect
import contextlib
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import sim
import continuous_batch_sim as native
import explore_fifo_underload as raw
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import run_case, save_manifest, write_json
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT = Path(__file__).resolve().parent
LAYERS, NPU_COUNT, SSU_COUNT, LONG_COUNT = 8, 32, 3, 19
IO = 176 * 1024 / 2**30
CASE = dict(name="tiny_jitter_s3_L19_S13", ssu=3, nlong=19, mode="separated")
DOMAINS = {
    "L": dict(prefix=200 * 1024 - 2048, nql_min=2032, nql_max=2048),
    "S": dict(prefix=32 * 1024 - 2048, nql_min=2048, nql_max=2111),
}


def bracket(grid, value):
    """Return adjacent measured coordinates; refuse every extrapolation."""
    if not grid[0] <= value <= grid[-1]:
        raise ValueError(f"Coordinate {value} falls outside measured [{grid[0]}, {grid[-1]}]")
    i = bisect.bisect_left(grid, value)
    if i < len(grid) and grid[i] == value:
        return grid[i], grid[i], 0.0
    lo, hi = grid[i - 1], grid[i]
    return lo, hi, (value - lo) / (hi - lo)


def profile(table, prefix, nql):
    tokens = prefix + nql
    seq = tokens / 1024
    if not 20 <= seq <= 200 or prefix <= 0 or prefix % 128:
        raise ValueError(f"Invalid profile: {prefix=}, {nql=}, {seq=}")
    xs = sorted({k[0] for k in table})
    ys = sorted({k[1] for k in table})
    x0, x1, wx = bracket(xs, seq)
    y0, y1, wy = bracket(ys, nql)
    weights = {}
    for x, vx in ((x0, 1 - wx), (x1, wx)):
        for y, vy in ((y0, 1 - wy), (y1, wy)):
            if vx * vy:
                weights[x, y] = weights.get((x, y), 0.0) + vx * vy
    assert all(v >= 0 for v in weights.values())
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    compute_us = math.fsum(table[k][1] * weight for k, weight in weights.items())
    kv_gib = prefix * 1408 / 2**30
    direct = (seq, nql) in table
    if direct:
        assert compute_us == table[seq, nql][1]
        assert math.isclose(kv_gib, table[seq, nql][3], abs_tol=1e-12)
    if not compute_us > 0:
        raise ValueError("Interpolated compute must be positive")
    return dict(total_tokens=tokens, total_length_k=seq, nql=nql,
        ssd_prefix_tokens=prefix, compute_us=compute_us, read_gib=kv_gib,
        B_gib_s=kv_gib / (compute_us / 1e6), constructed_profile=not direct,
        profile_construction=dict(method="direct_data_row" if direct else "bilinear_interpolation_length_nql",
            source="data", anchors=[dict(seq_len_k=k[0], nql=k[1], weight=v,
                compute_us=table[k][1]) for k, v in weights.items()],
            extrapolated=False, compute_scale=1.0,
            kv_formula="fixed_hit_prefix_tokens * 1408 / 2**30",
            units=dict(compute="microseconds", kv="GiB")))


def build(horizon=1000.0, seed=7):
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    table = ast.literal_eval((ROOT / "data").read_text())
    pools, counts, specifications = {}, {}, {}
    for role, domain in DOMAINS.items():
        pool = [profile(table, domain["prefix"], nql)
                for nql in range(domain["nql_min"], domain["nql_max"] + 1)]
        pools[role] = pool
        min_request_ms = LAYERS * min(p["compute_us"] for p in pool) / 1000
        counts[role] = math.ceil(horizon / min_request_ms) + 1
        if counts[role] > len(pool):
            raise ValueError(f"{role}: {counts[role]} unique profiles required for horizon={horizon} ms, "
                             f"but only {len(pool)} are available; profiles will not be repeated")
        specifications[role] = dict(**domain, domain_size=len(pool), requests_per_npu=counts[role],
            blocks_per_layer=domain["prefix"] // 128,
            total_length_k_range=[p["total_length_k"] for p in (pool[0], pool[-1])],
            compute_ms_range=[min(p["compute_us"] for p in pool) / 1000,
                              max(p["compute_us"] for p in pool) / 1000],
            B_gib_s_range=[min(p["B_gib_s"] for p in pool), max(p["B_gib_s"] for p in pool)])
    requests, vectors, lane_checks = [], {}, []
    maxima = [[0.] * SSU_COUNT for _ in range(NPU_COUNT)]
    for npu in range(NPU_COUNT):
        role = "L" if npu < LONG_COUNT else "S"
        deck = random.Random(seed + 100003 * npu).sample(pools[role], counts[role])
        unique = set()
        for generation, p in enumerate(deck):
            key = (p["total_tokens"], p["nql"])
            assert key not in unique
            unique.add(key)
            rid = npu * 1000000 + generation
            blocks = p["ssd_prefix_tokens"] // 128
            layer = tuple((sim.block_ring_hash_disk_id(rid, j, SSU_COUNT), IO) for j in range(blocks))
            assert math.isclose(blocks * IO, p["read_gib"], abs_tol=1e-12)
            load = dict(request_id=rid, npu_id=npu, generation=generation, original_request_id=rid,
                seq_len_k=p["total_length_k"], total_tokens=p["total_tokens"], nql=p["nql"], role=role,
                ssd_prefix_tokens=p["ssd_prefix_tokens"], category=sim.classify_request(p["total_length_k"], p["nql"]),
                per_layer_us=p["compute_us"], per_layer_kv_gb=p["read_gib"],
                required_bw_input_gbps=p["B_gib_s"], arrival_time=0., arrival_ms=0., initial=True,
                constructed_profile=p["constructed_profile"], profile_construction=p["profile_construction"],
                original_compute_us=p["compute_us"], padding_gib_per_layer=0.)
            request = ContinuousBatchRequest.from_normalized(rid, npu, 0., load, (layer,))
            assert all(native._manifest_layer(request, k) is layer for k in range(LAYERS))
            requests.append(request)
            disk_counts = Counter(ssu for ssu, _ in layer)
            vector = [disk_counts[s] * IO / (p["compute_us"] / 1e6) for s in range(SSU_COUNT)]
            vectors[rid] = vector
            maxima[npu] = [max(a, b) for a, b in zip(maxima[npu], vector)]
        ideal_ms = sum(LAYERS * p["compute_us"] / 1000 for p in deck)
        assert ideal_ms > horizon and len(unique) == len(deck)
        lane_checks.append(dict(npu_id=npu, role=role, request_count=len(deck),
            unique_length_nql_count=len(unique), ideal_compute_ms=ideal_ms,
            shuffle_seed=seed + 100003 * npu, nql_in_execution_order=[p["nql"] for p in deck]))
    requests = tuple(requests)
    static = [sum(row[s] for row in maxima) for s in range(SSU_COUNT)]
    metadata = dict(experiment="fifo_underload_tiny_jitter_20260914", case=CASE,
        profiles=specifications, num_npu=NPU_COUNT, num_ssu=SSU_COUNT, n_layers=LAYERS,
        seed=seed, horizon_ms=horizon, order="random_distinct_profiles_with_fixed_role_per_npu",
        npu_roles=["L"] * LONG_COUNT + ["S"] * (NPU_COUNT - LONG_COUNT), lane_checks=lane_checks,
        equal_176kib_blocks=True, layout="block_ring_hash", placement_virtual_nodes_per_ssu=256,
        placement_key="(request_id, block_index); excludes layer; raw case 0 block counts and placement preserved",
        input_fingerprint=continuous_batch_input_fingerprint(requests),
        logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests, NPU_COUNT, SSU_COUNT),
        per_ssu_static_upper_bound_gib_s=static, static_underload_all_request_combinations=max(static) <= 40,
        static_upper_definition="For each SSU sum each NPU's maximum D/C across its frozen deck",
        all_length_nql_unique_within_each_npu=True, all_compute_profiles_within_measured_grid=True,
        data_sha256=hashlib.sha256((ROOT / "data").read_bytes()).hexdigest(),
        workload_scope="Fixed NPU0-18 long, NPU19-31 short; not all-lane long/short mixing. "
            "NQL sampled without replacement separately on each NPU. Hit-prefix lengths stay fixed. "
            "Compute is bilinearly interpolated from original data with no extrapolation and no compute scale. "
            "Derived profiles are not additional measurements. All requests arrive at t=0; batch size 1.")
    return requests, metadata, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-ms", type=float, default=1000.)
    parser.add_argument("--window", type=float, nargs=2, default=[200., 800.])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage", default="screen")
    parser.add_argument("--strategy", choices=["baseline", "once"], default="baseline")
    parser.add_argument("--policy", choices=["fifo", "short_first"], default="fifo")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.window[0] < args.window[1] <= args.horizon_ms:
        parser.error("Require 0 <= window start < window end <= horizon")
    if args.strategy == "once" and args.policy != "fifo":
        parser.error("short_first is a separate baseline Path0 probe; use --strategy baseline")
    started = time.perf_counter()
    requests, metadata, vectors = build(args.horizon_ms, args.seed)
    metadata["queue_order"] = "FIFO" if args.policy == "fifo" else "ascending request per-layer bytes; FIFO ties"
    metadata["probe_policy"] = args.policy
    dest = raw.OUT / args.stage / f"{CASE['name']}_seed{args.seed}_{args.strategy}_{args.policy}"
    if (dest / "result.json.gz").exists():
        raise FileExistsError(dest)
    if (dest / "metadata.json").exists():
        existing = json.loads((dest / "metadata.json").read_text())
        if existing["input_fingerprint"] != metadata["input_fingerprint"]:
            raise ValueError(f"{dest} holds different input; choose another --stage")
    dest.mkdir(parents=True, exist_ok=True)
    save_manifest(dest / "manifest.json.gz", requests, metadata)
    write_json(dest / "metadata.json", metadata)
    print(json.dumps(dict(event="prepared" if args.prepare_only else "start", name=CASE["name"],
        strategy=args.strategy, policy=args.policy, request_count=len(requests), output=str(dest),
        static_max=metadata["per_ssu_static_upper_bound_gib_s"],
        average_ssu=metadata["input_demand"]["per_ssu_gib_s"], profiles=metadata["profiles"])), flush=True)
    if args.prepare_only:
        return
    with contextlib.ExitStack() as stack:
        if args.policy == "short_first":
            from run_fifo_proof import SizePriorityPending
            volumes = {r.request_id: r.load["per_layer_kv_gb"] for r in requests}
            original_init = sim.PathQueue.__init__
            def initialize(queue, *init_args, **kwargs):
                original_init(queue, *init_args, **kwargs)
                if queue.path_id == 0:
                    queue.pending = SizePriorityPending(volumes)
            stack.enter_context(patch.object(sim.PathQueue, "__init__", initialize))
        result = run_case(requests, metadata, strategy=args.strategy, assignment="fixed", windows=[tuple(args.window)])
    audit = raw.demand_audit(result["summary"], vectors, SSU_COUNT, *args.window)
    result["demand_audit"] = audit
    result["exploration_case"] = CASE
    result["probe_policy"] = args.policy
    write_json(dest / "result.json.gz", result)
    w = result["windows"][0]
    slo = result["slo"]["window_admissions"]["admission"]
    row = dict(case_index=0, name=CASE["name"], seed=args.seed, strategy=args.strategy, policy=args.policy,
        mode=CASE["mode"], num_ssu=SSU_COUNT, window_ms=args.window,
        U_percent=100 * w["mean_npu_utilization"],
        short_U_percent=100 * w["by_role"]["S"]["active_compute_fraction"],
        long_U_percent=100 * w["by_role"]["L"]["active_compute_fraction"],
        short_active_share=w["by_role"]["S"]["active_ms"] / (NPU_COUNT * (args.window[1] - args.window[0])),
        short_stall_card_ms=w["by_role"]["S"]["exposed_stall_ms"],
        slo_1p5_percent=100 * slo["rate"], slo_count=slo["count"], slo_passed=slo["passed"],
        static_upper_gib_s=metadata["per_ssu_static_upper_bound_gib_s"],
        input_average_gib_s=metadata["input_demand"]["per_ssu_gib_s"], demand_audit=audit,
        all_active=w["all_npus_active_whole_window"],
        all_invariants_passed=all(result["summary"]["invariants"].values()),
        all_length_nql_unique_within_each_npu=True, wall_seconds=time.perf_counter() - started)
    write_json(dest / "metrics.json", row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
