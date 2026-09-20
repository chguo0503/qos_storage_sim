#!/usr/bin/env python3
"""Controlled repeated-profile screening: independent random L/S decks per NPU.

L is measured 200K/NQL2048. S is 20K or 24K/NQL1024, plus two 20K/NQL1536
cases. Compute is explicitly extrapolated from original 32K/48K data rows;
NQL1536 also linearly interpolates NQL1024/2048. KV bytes are exactly
(total_tokens-NQL)*1408. Repeated length/NQL profiles are only for mechanism
screening; they are NOT a distinct-profile random trace or new measurements.
"""
import argparse
import ast
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
from explore_fifo_variants import profile
from run_baseline_npu32_stress import run_case, save_manifest, write_json
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT = Path(__file__).resolve().parent
IO = 176 * 1024 / 2**30
LAYERS = 8
CASES = [
    dict(name=f"mixed{short_k}_s4_L1_S{ratio}", ssu=4, long=[200, 2048],
         short=[short_k, 1024], short_per_long=ratio, mode="mixed")
    for short_k in (20, 24) for ratio in (4, 8, 16, 24)
]+[
    dict(name=f"mixed20n1536_s3_L1_S{ratio}", ssu=3, long=[200, 2048],
         short=[20, 1536], short_per_long=ratio, mode="mixed")
    for ratio in (4, 8)
]


def mixed_profile(table, tokens, nql):
    if nql != 1536:
        return profile(table, tokens, nql)
    # Bilinear construction: interpolate NQL at 32K and 48K, then extrapolate
    # those two computed points in sequence length. This is algebraically
    # identical to averaging the two extrapolated profile compute times.
    endpoints = [profile(table, tokens, q) for q in (1024, 2048)]
    c_us = sum(p["compute_us"] for p in endpoints) / 2
    kv_gib = (tokens - nql) * 1408 / 2**30
    assert (tokens - nql) % 128 == 0 and c_us > 0
    return dict(total_length_k=tokens / 1024, total_tokens=tokens, nql=nql,
        compute_us=c_us, read_gib=kv_gib, B_gib_s=kv_gib / (c_us / 1e6),
        constructed_profile=True,
        profile_construction=dict(method="linear_NQL_interpolation_then_length_extrapolation",
            source="data", nql_anchors=[1024, 2048], nql_weight=0.5,
            length_anchors_k=[32, 48], length_weight=(tokens / 1024 - 32) / 16,
            anchors=[[32, 1024], [32, 2048], [48, 1024], [48, 2048]],
            compute_scale=1.0, kv_formula="(total_tokens - nql) * 1408 / 2**30",
            units=dict(compute="microseconds", kv="GiB")))


def build(case, horizon, seed):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    table = ast.literal_eval((ROOT / "data").read_text())
    profiles = {role: mixed_profile(table, length * 1024, nql)
                for role, (length, nql) in (("L", case["long"]), ("S", case["short"]))}
    base_deck = ["L"] + ["S"] * case["short_per_long"]
    cycle_compute_ms = sum(LAYERS * profiles[role]["compute_us"] / 1000 for role in base_deck)
    cycles = math.ceil(horizon / cycle_compute_ms) + 1
    ideal_lane_compute_ms = cycles * cycle_compute_ms
    requests, decks = [], []
    vectors = {}
    maxima = [[0.] * case["ssu"] for _ in range(32)]
    for n in range(32):
        # Shuffle the entire population, not individual L/S cycles. Each NPU
        # receives a different deterministic RNG stream but identical counts.
        roles = base_deck * cycles
        random.Random(seed + n * 100003).shuffle(roles)
        decks.append(roles)
        assert Counter(roles) == Counter(L=cycles, S=cycles * case["short_per_long"])
        for g, role in enumerate(roles):
            p = profiles[role]
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
    requests = tuple(requests)
    static = [sum(row[s] for row in maxima) for s in range(case["ssu"])]
    meta = dict(experiment="fifo_underload_mixed20_screening_20260914", case=case, profiles=profiles,
        num_npu=32, num_ssu=case["ssu"], n_layers=LAYERS, seed=seed, horizon_ms=horizon,
        order="independent_full_deck_shuffle_per_npu", per_npu_role_order=decks,
        per_npu_role_counts=dict(L=cycles, S=cycles * case["short_per_long"]),
        per_npu_ideal_compute_ms=ideal_lane_compute_ms,
        equal_176kib_blocks=True, layout="block_ring_hash", placement_virtual_nodes_per_ssu=256,
        placement_key="(request_id, block_index); layer excluded; all 8 layers reuse one block placement",
        input_fingerprint=continuous_batch_input_fingerprint(requests),
        logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests, 32, case["ssu"]),
        per_ssu_static_upper_bound_gib_s=static,
        static_underload_all_request_combinations=max(static) <= 40,
        static_upper_definition="For each SSU sum each NPU's maximum per-layer bytes/compute over its frozen deck",
        unique_request_ids=True, unique_length_nql_per_npu=False,
        workload_scope="Controlled repeated L/S profiles for mechanism screening only. "
            "Full per-NPU population is independently shuffled. L200K/2048 comes directly from data; "
            "S20K or S24K compute is extrapolated from measured 32/48K rows; "
            "NQL1536 additionally interpolates the measured NQL1024/2048 rows. "
            "All requests arrive at t=0, batch size 1. No compute scaling, no KV padding.",
        data_sha256=hashlib.sha256((ROOT / "data").read_bytes()).hexdigest())
    assert ideal_lane_compute_ms > horizon
    return requests, meta, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon-ms", type=float, default=1000.)
    parser.add_argument("--window", type=float, nargs=2, default=[200., 800.])
    parser.add_argument("--strategy", choices=["baseline", "once"], default="baseline")
    parser.add_argument("--stage", default="mixed20_screen")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.window[0] < args.window[1] <= args.horizon_ms:
        parser.error("Require 0 <= window start < window end <= horizon")
    case = CASES[args.case]
    dest = OUT / args.stage / f"{case['name']}_seed{args.seed}_{args.strategy}"
    if (dest / "result.json.gz").exists():
        raise FileExistsError(dest)
    started = time.perf_counter()
    requests, meta, vectors = build(case, args.horizon_ms, args.seed)
    if (dest / "metadata.json").exists():
        previous = json.loads((dest / "metadata.json").read_text())
        if previous["input_fingerprint"] != meta["input_fingerprint"]:
            raise ValueError(f"{dest} already holds different inputs; use another --stage")
    dest.mkdir(parents=True, exist_ok=True)
    save_manifest(dest / "manifest.json.gz", requests, meta)
    write_json(dest / "metadata.json", meta)
    print(json.dumps(dict(event="prepared" if args.prepare_only else "start", case=args.case,
        name=case["name"], strategy=args.strategy, request_count=len(requests), output=str(dest),
        static_max=meta["per_ssu_static_upper_bound_gib_s"],
        input_average=meta["input_demand"]["per_ssu_gib_s"],
        roles_per_npu=meta["per_npu_role_counts"], ideal_compute_ms=meta["per_npu_ideal_compute_ms"],
        short_compute_ms=meta["profiles"]["S"]["compute_us"] / 1000,
        short_read_gib=meta["profiles"]["S"]["read_gib"],
        short_B_gib_s=meta["profiles"]["S"]["B_gib_s"])), flush=True)
    if args.prepare_only:
        return
    result = run_case(requests, meta, strategy=args.strategy, assignment="fixed", windows=[tuple(args.window)])
    audit = demand_audit(result["summary"], vectors, case["ssu"], *args.window)
    result["demand_audit"] = audit
    result["exploration_case"] = case
    write_json(dest / "result.json.gz", result)
    w = result["windows"][0]
    slo = result["slo"]["window_admissions"]["admission"]
    def role_fraction(role):
        fraction = w["by_role"].get(role, {}).get("active_compute_fraction")
        return None if fraction is None else 100 * fraction
    row = dict(case_index=args.case, name=case["name"], seed=args.seed, strategy=args.strategy,
        mode=case["mode"], num_ssu=case["ssu"], window_ms=args.window,
        U_percent=100 * w["mean_npu_utilization"], short_U_percent=role_fraction("S"),
        long_U_percent=role_fraction("L"),
        short_active_share=w["by_role"].get("S", {}).get("active_ms", 0.) / (32 * (args.window[1] - args.window[0])),
        short_stall_card_ms=w["by_role"].get("S", {}).get("exposed_stall_ms", 0.),
        slo_1p5_percent=100 * slo["rate"], slo_count=slo["count"], slo_passed=slo["passed"],
        static_upper_gib_s=meta["per_ssu_static_upper_bound_gib_s"],
        input_average_gib_s=meta["input_demand"]["per_ssu_gib_s"], demand_audit=audit,
        all_active=w["all_npus_active_whole_window"],
        all_invariants_passed=all(result["summary"]["invariants"].values()),
        unique_length_nql_per_npu=False, constructed_short_profile=True,
        wall_seconds=time.perf_counter() - started)
    write_json(dest / "metrics.json", row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
