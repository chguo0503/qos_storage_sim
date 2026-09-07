#!/usr/bin/env python3
"""Reproduce the tutorial's five bounded sensitivity checks.

Uses the unchanged Baseline simulator; writes a separate result, never result.json.
These are deterministic probes, not a random workload distribution or jitter test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from baseline_path0_layout import ROOT, DATA

CASES = (
    {"name": "baseline_reproduction", "seed": 42,
     "offsets_ms": [0, 0, 0, 0], "npu3_nql": 896},
    {"name": "submit_seed_1", "seed": 1,
     "offsets_ms": [0, 0, 0, 0], "npu3_nql": 896},
    {"name": "staggered_start", "seed": 42,
     "offsets_ms": [0, 7.5, 15, 22.5], "npu3_nql": 896},
    {"name": "npu3_nql_900", "seed": 42,
     "offsets_ms": [0, 0, 0, 0], "npu3_nql": 900},
    {"name": "npu3_nql_928", "seed": 42,
     "offsets_ms": [0, 0, 0, 0], "npu3_nql": 928},
)
SOURCE_NAMES = (
    "data", "authenticated_workload_inputs.py", "sim.py",
    "continuous_batch_sim.py", "continuous_prefill_client.py",
    "policy_logic.py", "strategy_profiles.py",
    "baseline_path0_layout.py",
    "run_baseline_4npu_ssu1_low_utilization.py",
    "run_baseline_path0_tutorial_sensitivity.py",
)


def run_case(case: dict) -> dict:
    import run_baseline_4npu_ssu1_low_utilization as runner

    runner.SUBMIT_ORDER_SEED = case["seed"]
    table, _ = runner._load_table()
    keys = ((1, 169), (192, 368), (192, 896), (192, case["npu3_nql"]))
    profiles = runner._profiles(table, keys=keys)
    nominal = sum(p["required_bandwidth_gibps"] for p in profiles)
    if nominal > 40.0:
        raise AssertionError(f"Bandwidth constraint violated: {case['name']}")
    requests = tuple(
        replace(
            request,
            arrival_time_ms=case["offsets_ms"][request.npu_id],
            load={
                **request.load,
                "arrival_ms": case["offsets_ms"][request.npu_id],
                "arrival_time": case["offsets_ms"][request.npu_id],
            },
        )
        for request in runner._build_requests(profiles)
    )
    result = runner._simulate(requests)
    if not all(result["invariants"].values()):
        raise AssertionError(result["invariants"])
    return {
        **case,
        "nominal_gibps": nominal,
        "npu3_compute_ms": profiles[3]["per_layer_compute_us"] / 1000,
        "mean_utilization": result["mean_npu_utilization"],
        "per_npu_utilization": result["npu_utilizations"],
        "ssd_utilization": result["measurement_ssd_mean_utilization"],
        "measurement_start_ms": result["measurement_start_ms"],
        "measurement_end_ms": result["measurement_end_ms"],
        "all_invariants_pass": all(result["invariants"].values()),
        "subwindow_min": min(b["npu_utilization"] for b in result["measurement_blocks"]),
        "subwindow_max": max(b["npu_utilization"] for b in result["measurement_blocks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=DATA / "tutorial_sensitivity_results.json")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(CASES))) as pool:
        results = list(pool.map(run_case, CASES))
    payload = {
        "schema_version": 1,
        "scope": "Five deterministic Baseline probes; no host-DRAM cache stage",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for case in results:
        print(f"{case['name']}: mean={100 * case['mean_utilization']:.4f}%, "
              f"NPU0={100 * case['per_npu_utilization'][0]:.4f}%, "
              f"demand={case['nominal_gibps']:.6f} GiB/s")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
