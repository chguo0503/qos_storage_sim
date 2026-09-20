#!/usr/bin/env python3
"""Frozen-input stress probes for Baseline head-of-line stalls at 4/32 NPUs.

Examples::

  python run_baseline_npu32_stress.py --family legacy --num-npu 32 --num-ssu 8 \
      --layout local --blocks exact --describe-only --manifest-out inputs/local.json.gz
  python run_baseline_npu32_stress.py --manifest inputs/local.json.gz \
      --strategy baseline_native --output results/baseline_npu32_investigation/pilot

All requests arrive at t=0 and provide at least horizon_ms of ideal compute
on every original NPU. These are saturated queue mechanism probes, not a claim
about a production arrival distribution. Shared/coflow policies require equal
176-KiB commands; exact historical tails are supported only by native policies.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import replace
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import sim
from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_multi_ssu_stall_experiments import input_demand
from run_shared_path_experiments import logical_input_fingerprint, summarize_slo, summarize_window


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/baseline_npu32_investigation/pilot"
IO_GIB = 176 * 1024 / 2**30
LAYERS = 8
COFLOW_STRATEGIES = ("baseline", "once", "new_once", "strategy1", "strategy2", "strategy3")
NATIVE_STRATEGIES = ("baseline_native", "once_native", "demand", "deadline",
                     "deadline_reserve", "least_slack", "stall_interchange")
STRATEGIES = COFLOW_STRATEGIES + NATIVE_STRATEGIES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def save_manifest(path, requests, metadata):
    """Deduplicate identical placements while preserving exact binary floats."""
    placements, indices, rows = [], {}, []
    for request in requests:
        placement = request.placement
        if placement not in indices:
            indices[placement] = len(placements)
            placements.append(placement)
        rows.append({"request_id": request.request_id, "npu_id": request.npu_id,
                     "arrival_time_ms": request.arrival_time_ms, "load": dict(request.load),
                     "placement_index": indices[placement]})
    payload = {"schema_version": 1, "metadata": metadata, "placements": placements,
               "requests": rows, "input_fingerprint": continuous_batch_input_fingerprint(requests)}
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing["input_fingerprint"] != payload["input_fingerprint"]:
            raise FileExistsError(f"preserving different manifest: {path}")
        return
    write_json(path, payload)


def load_manifest(path):
    payload = read_json(path)
    placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p)
                  for p in payload["placements"]]
    requests = tuple(ContinuousBatchRequest.from_normalized(
        row["request_id"], row["npu_id"], row["arrival_time_ms"], row["load"],
        placements[row["placement_index"]]) for row in payload["requests"])
    if continuous_batch_input_fingerprint(requests) != payload["input_fingerprint"]:
        raise AssertionError("frozen manifest fingerprint mismatch")
    return requests, payload["metadata"]


def profiles_for(family, keys, profile_keys=None):
    with redirect_stdout(io.StringIO()):
        table, provenance = load_authenticated_bw_table(32)
    if family in ("legacy", "aligned"):
        from run_baseline_4npu_ssu1_low_utilization import _profile
        chosen = ((1, 169), (192, 368), (192, 896), (192, 896))
        if family == "aligned":
            chosen = ((1, 128), (192, 384), (192, 896), (192, 896))
        if profile_keys:
            chosen = tuple(tuple(map(int, key.split(":"))) for key in profile_keys.split(","))
        profiles = [_profile(table, i, key, f"role{i}") for i, key in enumerate(chosen)]
    else:
        chosen = tuple(tuple(map(int, key.split(":"))) for key in (profile_keys or keys).split(","))
        profiles = []
        for i, key in enumerate(chosen):
            if key not in table:
                raise ValueError(f"raw profile is absent from data: {key}")
            bw, compute_us, ttft, volume = table[key]
            profiles.append({"npu_id": i, "role": f"role{i}", "seq_len_k": key[0],
                "total_tokens": key[0] * 1024, "nql": key[1],
                "ssd_prefix_tokens": key[0] * 1024 - key[1], "per_layer_kv_gib": volume,
                "per_layer_compute_us": compute_us, "required_bandwidth_gibps": bw,
                "source_equivalent_ttft_78_layers_ms": ttft,
                "construction": {"method": "direct_data_row"}})
    return profiles, provenance


def build_workload(*, family="legacy", num_npu=32, num_ssu=8, layout="local",
                   blocks="exact", raw_keys="32:256,192:1024,192:2048,192:2048",
                   profile_keys=None,
                   horizon_ms=4000.0, compute_scale=1.0, target_rho=None,
                   seed=20260908, order="fixed", hotspot_fraction=0.75):
    if min(num_npu, num_ssu, horizon_ms, compute_scale) <= 0:
        raise ValueError("positive dimensions, horizon, and compute_scale required")
    if not 0 <= hotspot_fraction <= 1:
        raise ValueError("hotspot fraction must be in [0, 1]")
    profiles, provenance = profiles_for(family, raw_keys, profile_keys)
    group_size = len(profiles)
    if num_npu % group_size:
        raise ValueError("num_npu must be divisible by number of role profiles")
    if target_rho is not None and not 0 < target_rho <= 2:
        raise ValueError("target_rho must be in (0, 2]")

    def layer_for(profile, rid, n):
        tokens = profile["ssd_prefix_tokens"]
        sizes = [min(128, tokens - j * 128) * 1408 / 2**30
                 for j in range(math.ceil(tokens / 128))]
        if blocks == "padded":
            sizes = [IO_GIB] * len(sizes)
        group = n // group_size
        values = []
        for j, size in enumerate(sizes):
            if layout == "local":
                ssu = group % num_ssu
            elif layout == "stripe":
                ssu = (j + group) % num_ssu
            elif layout == "hash":
                ssu = sim.block_ring_hash_disk_id(rid, j, num_ssu)
            elif layout == "hotspot":
                # Deterministic exact-length hot prefix, remaining commands striped.
                ssu = 0 if j < math.ceil(len(sizes) * hotspot_fraction) else (j % max(1, num_ssu - 1)) + (num_ssu > 1)
            else:
                raise ValueError(layout)
            values.append((int(ssu), size))
        return tuple(values)

    def construct(scale):
        requests = []
        # A shuffled cyclic deck changes profiles per request. Synchronized decks
        # use one common permutation; staggered decks use independent per-NPU RNG.
        for n in range(num_npu):
            deck = list(range(group_size))
            if order != "fixed":
                random.Random(seed if order == "synchronized" else seed + n * 100003).shuffle(deck)
            ideal_ms = 0.0
            generation = 0
            required_ms = horizon_ms + 2 * LAYERS * max(p["per_layer_compute_us"] for p in profiles) * scale / 1000
            cycle_ms = LAYERS * sum(p["per_layer_compute_us"] for p in profiles) * scale / 1000
            cycle_requests = math.ceil(required_ms / cycle_ms) * group_size
            while (ideal_ms < required_ms if order == "fixed" else generation < cycle_requests):
                p = profiles[n % group_size if order == "fixed" else deck[generation % group_size]]
                rid = n * 1_000_000 + generation
                layer = layer_for(p, rid, n)
                volume = sum(v for _, v in layer)
                compute_us = p["per_layer_compute_us"] * scale
                load = {"request_id": rid, "npu_id": n, "generation": generation,
                    "seq_len_k": p["seq_len_k"], "nql": p["nql"],
                    "category": sim.classify_request(p["seq_len_k"], p["nql"]),
                    "per_layer_us": compute_us, "per_layer_kv_gb": volume,
                    "required_bw_input_gbps": volume * 1e6 / compute_us,
                    "source_ttft_ms": p["source_equivalent_ttft_78_layers_ms"],
                    "arrival_time": 0.0, "arrival_ms": 0.0, "initial": True,
                    "role": p["role"], "original_compute_us": p["per_layer_compute_us"],
                    "constructed_profile": p["construction"]["method"] != "direct_data_row" or scale != 1.0,
                    "profile_construction": p["construction"],
                    "padding_gib_per_layer": volume - p["per_layer_kv_gib"]}
                requests.append(ContinuousBatchRequest.from_normalized(rid, n, 0.0, load, (layer,)))
                ideal_ms += LAYERS * compute_us / 1000
                generation += 1
        return tuple(requests)

    scale = compute_scale
    requests = construct(scale)
    demand = input_demand(requests, num_npu, num_ssu)
    if target_rho is not None:
        # Hash placements and finite mixed decks can change slightly with count.
        # Never decrease C here; scale upward until the actual manifest passes.
        for _ in range(8):
            limiting = max(demand["hottest_ssu_load_ratio"], demand["largest_npu_receive_load_ratio"])
            if limiting <= target_rho + 1e-10:
                break
            scale *= limiting / target_rho * (1 + 1e-9)
            requests = construct(scale)
            demand = input_demand(requests, num_npu, num_ssu)
        if max(demand["hottest_ssu_load_ratio"], demand["largest_npu_receive_load_ratio"]) > target_rho + 1e-8:
            raise AssertionError("finite input failed requested load bound")
    equal_blocks = all(v == IO_GIB for r in requests for layer in r.placement for _, v in layer)
    metadata = {"experiment": "baseline_npu32_stress_v1", "family": family,
        "num_npu": num_npu, "num_ssu": num_ssu, "n_layers": LAYERS, "seed": seed,
        "regime": "saturated_finite_backlog", "layout": layout, "blocks": blocks,
        "raw_keys": raw_keys, "profile_keys": profile_keys, "order": order, "profiles": profiles, "source": provenance,
        "compute_scale_requested": compute_scale, "compute_scale_actual": scale,
        "target_rho": target_rho, "horizon_ms": horizon_ms,
        "request_count": len(requests), "last_arrival_ms": 0,
        "equal_176kib_blocks": equal_blocks, "hotspot_fraction": hotspot_fraction,
        "input_demand": demand, "logical_input_fingerprint": logical_input_fingerprint(requests),
        "input_fingerprint": continuous_batch_input_fingerprint(requests),
        "sampling_caveat": "Constructed saturated finite backlog; all requests arrive at t=0. No external byte-rate claim.",
        "load_caveat": "sum per-lane total bytes / total ideal compute is a necessary long-run capacity check, not a sufficient per-layer deadline proof. Mixed-order finite averages can conceal bursts.",
        "count_rule": "fixed: complete requests until each lane reaches horizon + 2 longest-request compute times; varying order: equal integer cycle counts across lanes and order treatments",
        "scale_caveat": "target_rho only increases all compute times. Request counts are recomputed to retain the ideal-compute horizon, so scaled and unscaled populations may differ; strategies within a frozen manifest use identical requests.",
        "padding_total_gib": LAYERS * sum(r.load["padding_gib_per_layer"] for r in requests),
        "load_within_disk_and_link_capacity": demand["hottest_ssu_load_ratio"] <= 1 + 1e-10 and demand["largest_npu_receive_load_ratio"] <= 1 + 1e-10}
    metadata["case_id"] = (f"{family}_{order}_{layout}_{blocks}_npu{num_npu}_ssu{num_ssu}"
                           f"_seed{seed}_{metadata['input_fingerprint'][:12]}")
    return requests, metadata


def window_evidence(summary, requests, start_ms, end_ms):
    window = summarize_window(summary, start_ms, end_ms)
    roles = {r.request_id: r.load.get("role", "unknown") for r in requests}
    role_metrics = defaultdict(lambda: {"compute_ms": 0.0, "active_ms": 0.0,
                                       "completed_requests": 0, "admitted_requests": 0,
                                       "exposed_stall_ms": 0.0, "warm_layer_count": 0,
                                       "warm_late_layer_count": 0, "warm_read_latencies_ms": [],
                                       "warm_deadline_slacks_ms": []})
    latencies, slacks = [], []
    overlap = lambda a, b: max(0.0, min(end_ms, b) - max(start_ms, a))
    for batch in summary["microbatch_metrics"]:
        role = roles[batch["member_request_ids"][0]]
        row = role_metrics[role]
        row["active_ms"] += overlap(batch["admission_time_ms"], batch["completion_time_ms"])
        row["completed_requests"] += int(start_ms <= batch["completion_time_ms"] < end_ms)
        row["admitted_requests"] += int(start_ms <= batch["admission_time_ms"] < end_ms)
        for layer in batch["layer_metrics"]:
            row["compute_ms"] += overlap(layer["compute_start_ms"], layer["compute_end_ms"])
            if layer["layer"] and start_ms <= layer["io_start_time_ms"] < end_ms:
                latency = layer["io_ready_time_ms"] - layer["io_start_time_ms"]
                budget = batch["layer_metrics"][layer["layer"] - 1]["compute_duration_ms"]
                slack = budget - latency
                row["warm_layer_count"] += 1
                row["warm_late_layer_count"] += int(slack < -1e-8)
                row["warm_read_latencies_ms"].append(latency)
                row["warm_deadline_slacks_ms"].append(slack)
                latencies.append(latency)
                slacks.append(slack)
    def percentiles(values):
        if not values:
            return None
        import numpy as np
        return {f"p{p}": float(np.percentile(values, p)) for p in (0, 50, 95, 99, 100)}
    for row in role_metrics.values():
        row["exposed_stall_ms"] = row["active_ms"] - row["compute_ms"]
        row["active_compute_fraction"] = row["compute_ms"] / row["active_ms"] if row["active_ms"] else None
        row["warm_read_latency_ms"] = percentiles(row.pop("warm_read_latencies_ms"))
        row["warm_deadline_slack_ms"] = percentiles(row.pop("warm_deadline_slacks_ms"))
    window["by_role"] = dict(role_metrics)
    window["warm_read_latency_ms"] = percentiles(latencies)
    window["warm_deadline_slack_ms"] = percentiles(slacks)
    return window


def run_case(requests, metadata, *, strategy="baseline_native", assignment="pipeline",
             queue_window_ms=1.0, joint_rule="urgent_short", windows=((1000, 2000), (2000, 3000))):
    before = continuous_batch_input_fingerprint(requests)
    started = time.perf_counter()
    if strategy in COFLOW_STRATEGIES:
        if not metadata["equal_176kib_blocks"]:
            raise ValueError("coflow/shared policies require 176-KiB blocks; use --blocks padded, aligned, or raw profiles")
        from run_coflow_experiments import run_case as run_coflow
        result = run_coflow(requests, metadata, strategy=strategy, assignment=assignment,
                            queue_window_ms=queue_window_ms, joint_rule=joint_rule)
        result.pop("_source_texts", None)
    else:
        from run_multi_ssu_stall_experiments import run_case as run_native
        native = {"baseline_native": "baseline", "once_native": "layer_once"}.get(strategy, strategy)
        result = run_native(requests, metadata, strategy=native, assignment="none")
    summary = result["summary"]
    assert all(summary["invariants"].values())
    assert before == continuous_batch_input_fingerprint(requests)
    result["strategy"] = strategy
    result["experiment"] = "baseline_npu32_stress_v1"
    result["windows"] = [window_evidence(summary, requests, start, end) for start, end in windows]
    result["common_window"] = result["windows"][0]
    result["slo"] = summarize_slo(summary, requests, start_ms=windows[0][0], end_ms=windows[0][1])
    result["python_version"] = sys.version
    result["stress_runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["wall_seconds_total"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("legacy", "aligned", "raw"), default="legacy")
    parser.add_argument("--num-npu", type=int, default=32)
    parser.add_argument("--num-ssu", type=int, default=8)
    parser.add_argument("--layout", choices=("local", "stripe", "hash", "hotspot"), default="local")
    parser.add_argument("--blocks", choices=("exact", "padded"), default="exact")
    parser.add_argument("--raw-keys", default="32:256,192:1024,192:2048,192:2048")
    parser.add_argument("--profile-keys", help="Override role sequence, e.g. 1:169,1:169,1:169,192:224")
    parser.add_argument("--horizon-ms", type=float, default=4000)
    parser.add_argument("--compute-scale", type=float, default=1.0)
    parser.add_argument("--target-rho", type=float)
    parser.add_argument("--hotspot-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--order", choices=("fixed", "synchronized", "shuffled"), default="fixed")
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline_native")
    parser.add_argument("--assignment", choices=("pipeline", "compute", "fixed"), default="pipeline")
    parser.add_argument("--queue-window-ms", type=float, default=1.0)
    parser.add_argument("--joint-rule", choices=("urgent_short", "least_slack"), default="urgent_short")
    parser.add_argument("--window", action="append", help="START_MS:END_MS, repeatable")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    windows = tuple(tuple(map(float, x.split(":"))) for x in (args.window or ["1000:2000", "2000:3000"]))
    if any(len(w) != 2 or w[0] < 0 or w[1] <= w[0] for w in windows):
        parser.error("windows must have 0 <= START < END")
    if args.manifest:
        requests, metadata = load_manifest(args.manifest)
    else:
        requests, metadata = build_workload(**{key: getattr(args, key) for key in (
            "family", "num_npu", "num_ssu", "layout", "blocks", "raw_keys", "profile_keys", "horizon_ms",
            "compute_scale", "target_rho", "seed", "order", "hotspot_fraction")})
    if args.manifest_out:
        save_manifest(args.manifest_out, requests, metadata)
    if args.describe_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    config = {"strategy": args.strategy, "assignment": args.assignment,
              "queue_window_ms": args.queue_window_ms, "joint_rule": args.joint_rule, "windows": windows}
    destination = args.output / f"{metadata['case_id']}_{args.strategy}_{digest(config)[:10]}.json.gz"
    if destination.exists():
        raise FileExistsError(f"preserving result: {destination}")
    manifest_path = args.manifest or (args.output / "inputs" / f"{metadata['case_id']}.json.gz")
    if not args.manifest:
        save_manifest(manifest_path, requests, metadata)
    try:
        result = run_case(requests, metadata, **config)
        result["manifest_path"] = str(manifest_path)
        write_json(destination, result)
    except Exception as exc:
        write_json(destination.with_name(destination.name + ".failure.json"), {
            "status": "failed", "metadata": metadata, "config": config,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    print(json.dumps({"output": str(destination), "strategy": args.strategy,
        "input_fingerprint": result["input_fingerprint"],
        "wall_seconds": result["wall_seconds_total"],
        "windows": [{"start_ms": w["start_ms"], "end_ms": w["end_ms"],
                     "utilization": w["mean_npu_utilization"],
                     "all_active": w["all_npus_active_whole_window"], "by_role": w["by_role"]}
                    for w in result["windows"]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
