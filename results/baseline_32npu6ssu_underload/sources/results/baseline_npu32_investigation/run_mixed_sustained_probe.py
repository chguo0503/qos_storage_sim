#!/usr/bin/env python3
"""Frozen full-deck mixed-profile probes, with independent per-NPU shuffles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import sim
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import profiles_for, save_manifest, read_json, write_json
from run_multi_ssu_stall_experiments import input_demand
from run_shared_path_experiments import logical_input_fingerprint

OUT = ROOT / "results/baseline_npu32_investigation/mixed_varied_short"
STRATEGIES = ("baseline", "once", "new_once")
LAYERS, NUM_NPU, NUM_SSU = 8, 32, 8
IO_GIB = 176 * 1024 / 2**30


def build_input(family, seed, *, profile_keys=None, quotas=None, short_profile_count=None,
                label=None):
    keys, short_quota = (("1:128,192:256", 50) if family == "aligned"
                         else ("32:512,192:1024", 9))
    keys = profile_keys or keys
    quotas = tuple(quotas or (short_quota, 1))
    short_profile_count = short_profile_count or 1
    profiles, provenance = profiles_for(family, keys, keys)
    assert len(quotas) == len(profiles) and all(q > 0 for q in quotas)
    cycle_compute_ms = LAYERS * sum(q * p["per_layer_compute_us"]
                                  for q, p in zip(quotas, profiles)) / 1000
    required_compute_ms = 4000 + 2 * LAYERS * max(p["per_layer_compute_us"] for p in profiles) / 1000
    cycles = math.ceil(required_compute_ms / cycle_compute_ms)
    full_deck = [i for i, quota in enumerate(quotas) for _ in range(quota)] * cycles
    requests, deck_hashes = [], []
    for n in range(NUM_NPU):
        deck = full_deck.copy()
        random.Random(seed + n * 100003).shuffle(deck)
        deck_hashes.append(hashlib.sha256(json.dumps(deck).encode()).hexdigest())
        group = n // 4
        placements = []
        for profile in profiles:
            tokens = profile["ssd_prefix_tokens"]
            assert tokens % 128 == 0
            placements.append((tuple(((j + group) % NUM_SSU, IO_GIB)
                                     for j in range(tokens // 128)),))
        for generation, which in enumerate(deck):
            p = profiles[which]
            rid = n * 1_000_000 + generation
            load = {"request_id": rid, "npu_id": n, "generation": generation,
                "seq_len_k": p["seq_len_k"], "nql": p["nql"],
                "category": sim.classify_request(p["seq_len_k"], p["nql"]),
                "per_layer_us": p["per_layer_compute_us"],
                "per_layer_kv_gb": p["per_layer_kv_gib"],
                "required_bw_input_gbps": p["required_bandwidth_gibps"],
                "source_ttft_ms": p["source_equivalent_ttft_78_layers_ms"],
                "arrival_time": 0.0, "arrival_ms": 0.0, "initial": True,
                "role": "short" if which < short_profile_count else "long",
                "original_compute_us": p["per_layer_compute_us"],
                "constructed_profile": p["construction"]["method"] != "direct_data_row",
                "profile_construction": p["construction"], "padding_gib_per_layer": 0.0}
            requests.append(ContinuousBatchRequest.from_normalized(rid, n, 0.0, load, placements[which]))
    requests = tuple(requests)
    demand = input_demand(requests, NUM_NPU, NUM_SSU)
    short_share = sum(quotas[i] * profiles[i]["per_layer_compute_us"]
                      for i in range(short_profile_count)) / sum(
                          q * p["per_layer_compute_us"] for q, p in zip(quotas, profiles))
    for n in range(NUM_NPU):
        lane = [r for r in requests if r.npu_id == n]
        assert sum(r.load["role"] == "short" for r in lane) == sum(quotas[:short_profile_count]) * cycles
        assert sum(r.load["role"] == "long" for r in lane) == sum(quotas[short_profile_count:]) * cycles
        assert len({(r.load["seq_len_k"], r.load["nql"]) for r in lane}) == len(profiles)
        for p, quota in zip(profiles, quotas):
            assert sum((r.load["seq_len_k"], r.load["nql"]) == (p["seq_len_k"], p["nql"])
                       for r in lane) == quota * cycles
        actual_short = sum(r.load["per_layer_us"] for r in lane if r.load["role"] == "short")
        actual_all = sum(r.load["per_layer_us"] for r in lane)
        assert math.isclose(actual_short / actual_all, short_share, abs_tol=1e-12)
        assert LAYERS * actual_all / 1000 >= required_compute_ms - 1e-8
    fingerprint = continuous_batch_input_fingerprint(requests)
    label = f"{label or (family + '_short' + str(short_quota))}_seed{seed}"
    metadata = {"experiment": "mixed_varied_short_full_deck_v2", "label": label,
        "family": family, "num_npu": NUM_NPU, "num_ssu": NUM_SSU,
        "n_layers": LAYERS, "seed": seed, "regime": "saturated_finite_backlog",
        "layout": "stripe", "stripe_group_rule": "group = original_npu_id // 4; ssu=(block_index+group)%8",
        "blocks": "exact", "equal_176kib_blocks": True, "order": "independently_shuffled_complete_population",
        "profile_keys": keys, "profiles": profiles, "source": provenance,
        "profile_quotas_per_unit": list(quotas), "profile_count_per_npu": len(profiles),
        "short_profile_count_per_npu": short_profile_count,
        "profile_counts_per_npu": [q * cycles for q in quotas],
        "quota_units_per_npu": cycles, "short_requests_per_npu": sum(quotas[:short_profile_count]) * cycles,
        "long_requests_per_npu": sum(quotas[short_profile_count:]) * cycles, "requests_per_npu": len(full_deck),
        "request_count": len(requests), "horizon_ms": 4000,
        "required_minimum_compute_ms_per_npu": required_compute_ms,
        "short_compute_fraction_per_npu": short_share, "compute_scale_actual": 1.0,
        "last_arrival_ms": 0.0, "input_demand": demand,
        "input_fingerprint": fingerprint,
        "logical_input_fingerprint": logical_input_fingerprint(requests),
        "case_id": f"{label}_{fingerprint[:12]}", "per_npu_deck_sha256": deck_hashes,
        "shuffle_rule": "one random.Random(seed+npu_id*100003).shuffle of the entire complete request population; no fixed deck is repeated",
        "sampling_caveat": "New constructed populations, not the previous 52%-utilization input. Every NPU runs all specified profiles with identical complete quotas; all requests arrive at t=0. Frequencies and ordering are synthetic, not observed production arrivals.",
        "load_caveat": "Full-input compute-weighted per-SSU/link demand is a necessary long-run capacity check, not a per-layer deadline guarantee.",
        "load_within_disk_and_link_capacity": max(demand["hottest_ssu_load_ratio"], demand["largest_npu_receive_load_ratio"]) <= 1 + 1e-10}
    return requests, metadata


def prepare(output, specification=None):
    inputs = []
    specifications = (specification["inputs"] if specification else {
        "aligned_short50": {"family": "aligned"}, "raw_short9": {"family": "raw"}})
    for base_label, spec in specifications.items():
        spec = dict(spec)
        seeds = spec.pop("seeds", (7, 123))
        for seed in seeds:
            requests, metadata = build_input(seed=seed, label=base_label, **spec)
            path = output / "inputs" / f"{metadata['label']}.json.gz"
            save_manifest(path, requests, metadata)
            write_json(path.with_name(path.stem + ".description.json"), metadata)
            inputs.append({"label": metadata["label"], "manifest": str(path.resolve()),
                           "input_fingerprint": metadata["input_fingerprint"],
                           "request_count": metadata["request_count"],
                           "profile_quotas_per_unit": metadata["profile_quotas_per_unit"],
                           "profile_count_per_npu": metadata["profile_count_per_npu"],
                           "short_profile_count_per_npu": metadata["short_profile_count_per_npu"],
                           "minimum_ideal_compute_ms_per_npu": min(metadata["input_demand"]["per_npu_ideal_compute_ms"]),
                           "short_compute_fraction_per_npu": metadata["short_compute_fraction_per_npu"],
                           "hottest_ssu_load_ratio": metadata["input_demand"]["hottest_ssu_load_ratio"],
                           "largest_npu_receive_load_ratio": metadata["input_demand"]["largest_npu_receive_load_ratio"]})
    plan = {"inputs": inputs, "strategies": list(STRATEGIES), "job_count": len(inputs) * len(STRATEGIES),
            "windows_ms": [[1000, 2000], [2000, 3000]],
            "fixed_npu_binding": True, "collector_interval_ms": 5.0,
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "run_baseline_npu32_stress.py": hashlib.sha256((ROOT / "run_baseline_npu32_stress.py").read_bytes()).hexdigest()}}
    write_json(output / "plan.json", plan)
    return plan


def run_job(item, strategy, output):
    directory = output / "runs" / item["label"] / strategy
    directory.mkdir(parents=True, exist_ok=True)
    completed = list(directory.glob("*.json.gz"))
    if completed:
        result = read_json(completed[0])
        assert result["input_fingerprint"] == item["input_fingerprint"]
        assert result["strategy"] == strategy and result["collector_interval_ms"] == 5.0
        assert result["stress_runner_sha256"] == hashlib.sha256((ROOT / "run_baseline_npu32_stress.py").read_bytes()).hexdigest()
        assert all(row["assigned_npu_id"] == row["original_npu_id"]
                   for row in result.get("assignment_log", []))
        return {"label": item["label"], "strategy": strategy, "status": "existing", "output": str(completed[0])}
    command = [sys.executable, "-B", str(ROOT / "run_baseline_npu32_stress.py"),
               "--manifest", item["manifest"], "--strategy", strategy, "--output", str(directory)]
    record = {"command": command, "cwd": str(ROOT), "start_utc": datetime.now(timezone.utc).isoformat(),
              "python_executable": sys.executable, "python_version": sys.version,
              "numpy_version": np.__version__, "platform": platform.platform(),
              "input_fingerprint": item["input_fingerprint"], "label": item["label"], "strategy": strategy}
    write_json(directory / "command.json", record)
    started = time.perf_counter()
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        write_json(directory / "command.json", record)
        code = process.wait()
    record.update(returncode=code, end_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
    write_json(directory / "command.json", record)
    return {"label": item["label"], "strategy": strategy,
            "status": "complete" if code == 0 else "failed", "returncode": code,
            "wall_seconds": record["wall_seconds"], "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--spec", type=Path, help="JSON with inputs mapping labels to family/profile_keys/quotas/short_profile_count/seeds")
    args = parser.parse_args()
    if not args.run_only and args.spec is None:
        parser.error("provide --spec for an explicit multi-profile population, or --run-only for an existing frozen plan")
    plan = read_json(args.output / "plan.json") if args.run_only else prepare(
        args.output, read_json(args.spec) if args.spec else None)
    print(json.dumps({"phase": "prepared", **plan}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_job, item, strategy, args.output)
                   for item in plan["inputs"] for strategy in STRATEGIES}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(args.output / "run_status.json", {"completed": len(rows), "total": plan["job_count"], "rows": rows})
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if not done:
                print(json.dumps({"completed": len(rows), "total": plan["job_count"], "pending": len(pending),
                                  "note": "Pending includes queued jobs; process CPU activity is not simulated-time progress."}), flush=True)
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
