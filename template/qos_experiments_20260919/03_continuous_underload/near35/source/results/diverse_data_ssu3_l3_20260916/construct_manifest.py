#!/usr/bin/env python3
"""Freeze diverse raw-data, fixed-NPU, independently shuffled populations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import profiles_for, save_manifest, load_manifest
from run_shared_path_experiments import logical_input_fingerprint
import sim

LENGTHS = (32, 64, 80, 128, 160, 200)
MISSES = (256, 1024, 2048, 4096)
DEFAULT_WEIGHTS = {"semi": (1, 1, 1, 2), "full": (3, 2, 1, 1)}
BLOCK_GIB = 176 * 1024 / 2**30
NUM_NPU, NUM_SSU, LAYERS = 32, 3, 8


def build_workload(scenario="semi", seed=7, horizon_ms=6500.0, weights=None,
                   guaranteed_full=False):
    if scenario not in DEFAULT_WEIGHTS:
        raise ValueError("scenario must be semi or full")
    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError("horizon_ms must be finite and positive")
    misses = (256, 512, 1024) if guaranteed_full else MISSES
    if guaranteed_full and scenario != "full":
        raise ValueError("guaranteed-full catalog requires scenario full")
    counts = tuple(weights or ((1, 1, 1) if guaranteed_full else DEFAULT_WEIGHTS[scenario]))
    if len(counts) != len(misses) or any(int(v) != v or v <= 0 for v in counts):
        raise ValueError("provide a positive integer count for each selected miss")
    keys = ",".join(f"{n}:{m}" for n in LENGTHS for m in misses)
    profiles, provenance = profiles_for("raw", keys)
    count_by_miss = dict(zip(misses, counts))
    base = []
    for index, p in enumerate(profiles):
        p["role"] = p["name"] = f"L{p['seq_len_k']}_M{p['nql']}"
        p["category"] = sim.classify_request(p["seq_len_k"], p["nql"])
        assert p["construction"]["method"] == "direct_data_row"
        assert p["ssd_prefix_tokens"] % 128 == 0
        base.extend([index] * count_by_miss[p["nql"]])
    base_compute_ms = math.fsum(LAYERS * profiles[i]["per_layer_compute_us"] / 1000 for i in base)
    repeats = math.ceil(horizon_ms / base_compute_ms)
    canonical = base * repeats
    requests, per_npu = [], []
    rates = []
    pure_compute_ms = repeats * base_compute_ms
    for npu in range(NUM_NPU):
        placements, npu_rates = [], []
        for p in profiles:
            layer = tuple(((j + npu) % NUM_SSU, BLOCK_GIB)
                          for j in range(p["ssd_prefix_tokens"] // 128))
            assert math.isclose(math.fsum(v for _, v in layer), p["per_layer_kv_gib"], rel_tol=0, abs_tol=1e-12)
            placements.append((layer,))
            npu_rates.append([math.fsum(v for d,v in layer if d == disk) /
                              (p["per_layer_compute_us"] / 1e6) for disk in range(NUM_SSU)])
        rates.append(npu_rates)
        identities = list(range(len(canonical)))
        random.Random(seed + 100003 * npu).shuffle(identities)
        for position, original in enumerate(identities):
            index = canonical[original]
            p = profiles[index]
            rid = npu * 1000000 + position
            load = dict(request_id=rid, npu_id=npu, generation=position,
                        original_request_id=npu * 1000000 + original,
                        profile_index=index, role=p["role"], seq_len_k=p["seq_len_k"],
                        nql=p["nql"], total_tokens=p["total_tokens"],
                        ssd_prefix_tokens=p["ssd_prefix_tokens"], category=p["category"],
                        per_layer_us=p["per_layer_compute_us"], per_layer_kv_gb=p["per_layer_kv_gib"],
                        required_bw_input_gbps=p["required_bandwidth_gibps"],
                        source_ttft_ms=p["source_equivalent_ttft_78_layers_ms"],
                        original_compute_us=p["per_layer_compute_us"], constructed_profile=False,
                        profile_construction=p["construction"], padding_gib_per_layer=0.0,
                        arrival_time=0.0, arrival_ms=0.0, initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, placements[index]))
        per_npu.append(dict(npu_id=npu, shuffle_seed=seed + 100003*npu,
                            requests=len(canonical), pure_compute_ms=pure_compute_ms,
                            unique_profiles=len(profiles),
                            profile_counts=dict(Counter(profiles[i]["role"] for i in canonical)),
                            first_24_profiles=[profiles[canonical[i]]["role"] for i in identities[:24]]))
    requests = tuple(requests)
    profile_counts = Counter(canonical)
    ideal_probs = [profile_counts[i] * LAYERS * p["per_layer_compute_us"] / 1000 / pure_compute_ms
                   for i,p in enumerate(profiles)]
    mean_by_disk = [math.fsum(rates[n][i][s] * ideal_probs[i] for n in range(NUM_NPU)
                              for i in range(len(profiles))) for s in range(NUM_SSU)]
    lower_by_disk = [math.fsum(min(rates[n][i][s] for i in range(len(profiles))) for n in range(NUM_NPU))
                     for s in range(NUM_SSU)]
    upper_by_disk = [math.fsum(max(rates[n][i][s] for i in range(len(profiles))) for n in range(NUM_NPU))
                     for s in range(NUM_SSU)]
    fp = continuous_batch_input_fingerprint(requests)
    label = f"{scenario}_{'guaranteed_' if guaranteed_full else ''}w{'-'.join(map(str, counts))}_h{horizon_ms:g}_seed{seed}"
    metadata = dict(experiment="diverse_data_ssu3_l3_20260916", label=label,
                    case_id=label + "_" + fp[:12], scenario_candidate=scenario,
                    num_npu=NUM_NPU, num_ssu=NUM_SSU, n_layers=LAYERS, seed=seed,
                    disk_bw_gib_s=40.0, npu_bw_gib_s=50.0, family="raw",
                    profiles=profiles, source=provenance, profile_keys=keys,
                    source_data_sha256=hashlib.sha256((ROOT / "data").read_bytes()).hexdigest(),
                    equal_176kib_blocks=True, blocks="exact", constructed_profile=False,
                    compute_scale_actual=1.0, input_fingerprint=fp,
                    logical_input_fingerprint=logical_input_fingerprint(requests),
                    request_count=len(requests), order="random", order_mode="random",
                    last_arrival_ms=0.0, horizon_pure_compute_ms=horizon_ms,
                    actual_pure_compute_ms_per_npu=pure_compute_ms, quota_repeats=repeats,
                    per_length_miss_counts={str(m): int(c) for m,c in zip(misses,counts)},
                    per_npu_assignment=per_npu, measurement_window_ms=[2000,4000],
                    additional_measurement_windows_ms=[[2000,6000]],
                    layout="stripe_npu_mod_ssu",
                    placement_rule="(block_index+npu_id)%3; exact176KiB; data row retained without padding",
                    population_rule="Stratified finite population: each NPU has identical weighted counts of all selected raw profiles; repeat count chosen to cover a lower bound on pure compute; all requests arrive at0 and stay bound to their NPU",
                    random_rule="One independent full-population shuffle per NPU with Random(seed+100003*npu); repeated counts do not repeat the randomized deck",
                    identity_rule="request_id=npu*1000000+position; original_request_id retains pre-shuffle identity",
                    input_length_semantics="Total input length includes miss tokens; raw data unchanged",
                    nominal_definition="Current admitted request per-layer bytes on each disk / its pure layer compute time; not instantaneous IO arrival rate, not sum of all queued future requests; no double-counted next-request L0",
                    nominal_capacity_status="PENDING event-based per-disk audit in each measured window",
                    static_per_ssu_lower_bound_gib_s=lower_by_disk,
                    static_per_ssu_upper_bound_gib_s=upper_by_disk,
                    static_full_overload_guarantee=all(v > 40 for v in lower_by_disk),
                    time_weighted_per_ssu_nominal_gib_s=mean_by_disk,
                    time_weighted_fleet_nominal_gib_s=math.fsum(mean_by_disk),
                    fluid_rho=math.fsum(mean_by_disk)/120,
                    caveat="Designed coverage of measured profiles, not an empirical production arrival distribution. Pure-compute time weighting ignores stalls; actual nominal demand is endogenous to request residence. Scenario label is provisional until event audit.")
    if guaranteed_full:
        assert metadata["static_full_overload_guarantee"]
    assert all(row["pure_compute_ms"] >= horizon_ms for row in per_npu)
    return requests, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(DEFAULT_WEIGHTS), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon-ms", type=float, default=6500)
    parser.add_argument("--weights", help="Positive integer per-length counts, comma separated; default semi1,1,1,2/full3,2,1,1")
    parser.add_argument("--guaranteed-full", action="store_true", help="18 profiles with misses256/512/1024 and a strict static lower bound")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = tuple(map(int,args.weights.split(","))) if args.weights else None
    requests, metadata = build_workload(args.scenario,args.seed,args.horizon_ms,weights,args.guaranteed_full)
    if args.output.exists():
        _, old = load_manifest(args.output)
        if old != metadata:
            raise FileExistsError(f"Preserving existing manifest with different metadata: {args.output}")
    save_manifest(args.output, requests, metadata)
    restored, restored_meta = load_manifest(args.output)
    assert continuous_batch_input_fingerprint(restored) == metadata["input_fingerprint"]
    assert restored_meta == metadata
    print(json.dumps({k: metadata[k] for k in ("label","request_count","actual_pure_compute_ms_per_npu",
                      "time_weighted_fleet_nominal_gib_s","fluid_rho","static_full_overload_guarantee","input_fingerprint")}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

