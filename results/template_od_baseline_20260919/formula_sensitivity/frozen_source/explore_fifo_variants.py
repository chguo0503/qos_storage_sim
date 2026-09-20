#!/usr/bin/env python3
"""Random, distinct length/NQL profiles for fixed-role FIFO HOL exploration.

Only placement and workload generation are new. The existing true per-IO
runner, accounting and TTFT SLO 1.5 implementation are reused unchanged.
20--32K compute times are explicitly linear extrapolations from 32/48K data;
other non-grid lengths use adjacent data rows. These are derived profiles,
not additional measurements from the original data file.
"""
import argparse
import ast
import bisect
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import sim
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from explore_fifo_underload import OUT, demand_audit
from run_baseline_npu32_stress import run_case, save_manifest, write_json
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT = Path(__file__).resolve().parent
IO = 176 * 1024 / 2**30
LAYERS = 8
CASES = [
    dict(name="s3_L19_S13", ssu=3, nlong=19, long_nql=2048, short_nql=2048, mode="separated"),
    dict(name="s3_L16_S16", ssu=3, nlong=16, long_nql=2048, short_nql=2048, mode="separated"),
    dict(name="s4_L20_S12", ssu=4, nlong=20, long_nql=2048, short_nql=1024, mode="separated"),
    dict(name="s4_L24_S8", ssu=4, nlong=24, long_nql=2048, short_nql=1024, mode="separated"),
]


def profile(table, tokens, nql):
    """Interpolate compute only; derive exact bytes from hit-token semantics."""
    seq = tokens / 1024
    key = (seq, nql)
    lengths = sorted(k[0] for k in table if k[1] == nql)
    if key in table:
        c_us = float(table[key][1])
        method, anchors = "direct_data_row", [list(key)]
    else:
        i = bisect.bisect_left(lengths, seq)
        if i == 0:
            lo, hi = lengths[:2]
            method = "linear_extrapolation_below_32K"
        elif i == len(lengths):
            raise ValueError(f"Sequence {seq}K exceeds measured upper bound")
        else:
            lo, hi = lengths[i - 1:i + 1]
            method = "adjacent_length_linear_interpolation"
        weight = (seq - lo) / (hi - lo)
        c_us = table[lo, nql][1] + weight * (table[hi, nql][1] - table[lo, nql][1])
        anchors = [[lo, nql], [hi, nql]]
    if tokens <= nql or (tokens - nql) % 128 or c_us <= 0:
        raise ValueError(f"Invalid aligned profile {tokens=}, {nql=}, {c_us=}")
    kv_gib = (tokens - nql) * 1408 / 2**30
    if key in table:
        assert math.isclose(kv_gib, table[key][3], abs_tol=1e-12)
    return dict(total_length_k=seq, total_tokens=tokens, nql=nql,
                compute_us=c_us, read_gib=kv_gib, B_gib_s=kv_gib / (c_us / 1e6),
                constructed_profile=method != "direct_data_row",
                profile_construction=dict(method=method, source="data", anchors=anchors,
                    compute_scale=1.0, kv_formula="(total_tokens - nql) * 1408 / 2**30",
                    units=dict(compute="microseconds", kv="GiB")))


def length_domain(bounds):
    lo, hi = bounds
    if not 20 <= lo <= hi <= 200:
        raise ValueError("Length bounds must satisfy 20 <= lower <= upper <= 200 (K=1024 tokens)")
    begin, end = round(lo * 1024), round(hi * 1024)
    if begin % 128 or end % 128 or begin != lo * 1024 or end != hi * 1024:
        raise ValueError("Range endpoints must lie on a 128-token (0.125K) grid")
    return list(range(begin, end + 1, 128))


def build(case, horizon, seed, long_range=(160, 200), short_range=(20, 40)):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    table = ast.literal_eval((ROOT / "data").read_text())
    pools = {
        role: [profile(table, tokens, case[f"{label}_nql"]) for tokens in length_domain(bounds)]
        for role, label, bounds in (("L", "long", long_range), ("S", "short", short_range))
    }
    counts = {}
    specifications = {}
    for role, pool in pools.items():
        # Guarantee all lanes have at least horizon ms of ideal compute even
        # when the shortest allowed profiles happen to be drawn first.
        minimum_request_ms = LAYERS * min(p["compute_us"] for p in pool) / 1000
        counts[role] = math.ceil(horizon / minimum_request_ms) + 1
        if counts[role] > len(pool):
            raise ValueError(f"{role}: {counts[role]} unique profiles needed for {horizon} ms, "
                             f"but range provides only {len(pool)}. Widen the length range or reduce horizon; "
                             "profiles will never be silently repeated.")
        specifications[role] = dict(total_length_k_range=[pool[0]["total_length_k"], pool[-1]["total_length_k"]],
            nql=pool[0]["nql"], domain_size=len(pool), requests_per_npu=counts[role],
            compute_ms_range=[min(p["compute_us"] for p in pool) / 1000, max(p["compute_us"] for p in pool) / 1000],
            B_gib_s_range=[min(p["B_gib_s"] for p in pool), max(p["B_gib_s"] for p in pool)])
    npu_roles = ["L"] * case["nlong"] + ["S"] * (32 - case["nlong"])
    random.Random(seed).shuffle(npu_roles)
    requests = []
    vectors = {}
    maxima = [[0.] * case["ssu"] for _ in range(32)]
    lane_checks = []
    for n, role in enumerate(npu_roles):
        rng = random.Random(seed + n * 100003)
        deck = rng.sample(pools[role], counts[role])
        unique_keys = set()
        for g, p in enumerate(deck):
            key = (p["total_tokens"], p["nql"])
            assert key not in unique_keys
            unique_keys.add(key)
            rid = n * 1000000 + g
            blocks = (p["total_tokens"] - p["nql"]) // 128
            layer = tuple((sim.block_ring_hash_disk_id(rid, j, case["ssu"]), IO) for j in range(blocks))
            assert math.isclose(blocks * IO, p["read_gib"], abs_tol=1e-12)
            load = dict(request_id=rid, npu_id=n, generation=g, original_request_id=rid,
                seq_len_k=p["total_length_k"], total_tokens=p["total_tokens"], nql=p["nql"], role=role,
                category=sim.classify_request(p["total_length_k"], p["nql"]),
                per_layer_us=p["compute_us"], per_layer_kv_gb=p["read_gib"],
                required_bw_input_gbps=p["B_gib_s"], arrival_time=0., arrival_ms=0., initial=True,
                constructed_profile=p["constructed_profile"], profile_construction=p["profile_construction"],
                original_compute_us=p["compute_us"], padding_gib_per_layer=0.)
            requests.append(ContinuousBatchRequest.from_normalized(rid, n, 0., load, (layer,)))
            disk_counts = Counter(s for s, _ in layer)
            vec = [disk_counts[s] * IO / (p["compute_us"] / 1e6) for s in range(case["ssu"])]
            vectors[rid] = vec
            maxima[n] = [max(a, b) for a, b in zip(maxima[n], vec)]
        lane_checks.append(dict(npu_id=n, role=role, request_count=len(deck),
            unique_length_nql_count=len(unique_keys),
            ideal_compute_ms=sum(LAYERS * p["compute_us"] / 1000 for p in deck)))
    requests = tuple(requests)
    static = [sum(row[s] for row in maxima) for s in range(case["ssu"])]
    meta = dict(experiment="fifo_underload_random_distinct_profiles_20260914", case=case,
        profiles=specifications, num_npu=32, num_ssu=case["ssu"], n_layers=LAYERS,
        seed=seed, horizon_ms=horizon, order="random_profiles_with_fixed_role_per_npu",
        npu_roles=npu_roles, lane_checks=lane_checks,
        equal_176kib_blocks=True, layout="block_ring_hash", placement_virtual_nodes_per_ssu=256,
        placement_key="(request_id, block_index); layer is excluded; same placement tuple reused for all 8 layers",
        input_fingerprint=continuous_batch_input_fingerprint(requests),
        logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests, 32, case["ssu"]),
        per_ssu_static_upper_bound_gib_s=static,
        static_underload_all_request_combinations=max(static) <= 40,
        static_upper_definition="For each SSU sum each NPU's maximum per-layer bytes/compute over its entire frozen deck",
        all_length_nql_unique_within_each_npu=True,
        workload_scope="Distinct random length/NQL profiles per NPU, fixed L/S roles. "
            "Compute at non-grid lengths is interpolated from data; below 32K it is extrapolated from 32/48K. "
            "These profiles are derived, not new raw measurements. All requests arrive at t=0, batch size 1.",
        data_sha256=hashlib.sha256((ROOT / "data").read_bytes()).hexdigest())
    assert min(q["ideal_compute_ms"] for q in lane_checks) > horizon
    return requests, meta, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon-ms", type=float, default=1000.)
    parser.add_argument("--window", type=float, nargs=2, default=[200., 800.])
    parser.add_argument("--long-range", type=float, nargs=2, default=[160., 200.], metavar=("LOW_K", "HIGH_K"))
    parser.add_argument("--short-range", type=float, nargs=2, default=[20., 40.], metavar=("LOW_K", "HIGH_K"))
    parser.add_argument("--strategy", choices=["baseline", "once"], default="baseline")
    parser.add_argument("--stage", default="screen")
    parser.add_argument("--prepare-only", action="store_true", help="Build, validate and save input; do not simulate")
    args = parser.parse_args()
    if not 0 <= args.window[0] < args.window[1] <= args.horizon_ms:
        parser.error("Require 0 <= window start < window end <= horizon")
    case = CASES[args.case]
    dest = OUT / args.stage / f"variant_{case['name']}_seed{args.seed}_{args.strategy}"
    if (dest / "result.json.gz").exists():
        raise FileExistsError(dest)
    started = time.perf_counter()
    requests, meta, vectors = build(case, args.horizon_ms, args.seed, args.long_range, args.short_range)
    if (dest / "metadata.json").exists():
        previous = json.loads((dest / "metadata.json").read_text())
        if previous["input_fingerprint"] != meta["input_fingerprint"]:
            raise ValueError(f"{dest} already holds a different input; choose a new --stage")
    dest.mkdir(parents=True, exist_ok=True)
    save_manifest(dest / "manifest.json.gz", requests, meta)
    write_json(dest / "metadata.json", meta)
    print(json.dumps(dict(event="prepared" if args.prepare_only else "start", name=case["name"],
        strategy=args.strategy, request_count=len(requests), output=str(dest),
        static_max=meta["per_ssu_static_upper_bound_gib_s"],
        static_underload=meta["static_underload_all_request_combinations"],
        average_ssu=meta["input_demand"]["per_ssu_gib_s"], profiles=meta["profiles"])), flush=True)
    if args.prepare_only:
        return
    result = run_case(requests, meta, strategy=args.strategy, assignment="fixed", windows=[tuple(args.window)])
    audit = demand_audit(result["summary"], vectors, case["ssu"], *args.window)
    result["demand_audit"] = audit
    result["exploration_case"] = case
    write_json(dest / "result.json.gz", result)
    w = result["windows"][0]
    slo = result["slo"]["window_admissions"]["admission"]
    row = dict(case_index=args.case, name=f"variant_{case['name']}", seed=args.seed,
        strategy=args.strategy, mode=case["mode"], num_ssu=case["ssu"], window_ms=args.window,
        U_percent=100 * w["mean_npu_utilization"],
        short_U_percent=100 * w["by_role"]["S"]["active_compute_fraction"],
        long_U_percent=100 * w["by_role"]["L"]["active_compute_fraction"],
        short_active_share=w["by_role"]["S"]["active_ms"] / (32 * (args.window[1] - args.window[0])),
        short_stall_card_ms=w["by_role"]["S"]["exposed_stall_ms"],
        slo_1p5_percent=100 * slo["rate"], slo_count=slo["count"], slo_passed=slo["passed"],
        static_upper_gib_s=meta["per_ssu_static_upper_bound_gib_s"],
        static_underload_all_request_combinations=meta["static_underload_all_request_combinations"],
        input_average_gib_s=meta["input_demand"]["per_ssu_gib_s"], demand_audit=audit,
        all_active=w["all_npus_active_whole_window"],
        all_invariants_passed=all(result["summary"]["invariants"].values()),
        all_length_nql_unique_within_each_npu=True, wall_seconds=time.perf_counter() - started)
    write_json(dest / "metrics.json", row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
