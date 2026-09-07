#!/usr/bin/env python3
"""Audit tutorial arithmetic directly from data and saved event traces.

Does not change simulator code or the original experiment artifacts.
The optional --output writes a derived, independently reproducible audit report.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

from baseline_path0_layout import ROOT, DATA, DOCS


LAYOUT_MIGRATION_SNAPSHOTS = {
    "run_baseline_4npu_ssu1_low_utilization.py",
    "run_baseline_path0_tutorial_sensitivity.py",
}


def audit_sources(expected_hashes: dict[str, str]) -> dict[str, str]:
    """Verify historical provenance without rewriting hashes after relocation.

    The two output-path refactors retain exact pre-migration source snapshots.
    A snapshot verifies the source of saved evidence, not the current runner.
    """
    verified = {}
    for name, expected in expected_hashes.items():
        source = ROOT / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            if name not in LAYOUT_MIGRATION_SNAPSHOTS:
                raise AssertionError(f"Original experiment source changed: {name}")
            source = DOCS / "source_snapshots" / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Historical source snapshot changed: {name}")
        verified[name] = str(source.relative_to(ROOT))
    return verified



def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise AssertionError(f"{label}: {actual} != {expected}")


def read_csv(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coverage(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals
                     if min(b, end) > max(a, start))
    total = 0.0
    previous_end = start
    for left, right in clipped:
        total += max(0.0, right - max(left, previous_end))
        previous_end = max(previous_end, right)
    return total


def audit() -> dict:
    result = json.loads((DATA / "result.json").read_text(encoding="utf-8"))
    table = ast.literal_eval((ROOT / "data").read_text(encoding="utf-8"))
    source_verification = audit_sources(result["source_sha256"])
    if not all(result["validation"].values()):
        raise AssertionError(result["validation"])
    profile_rows = result["input"]["profiles"]
    profile_audit = []
    for p in profile_rows:
        total, nql = p["total_tokens"], p["nql"]
        prefix = total - nql
        volume_bytes = prefix * 1408
        volume_gib = volume_bytes / 2**30
        low, high = p["construction"]["nql_bounds"]

        def grid_time(n: int) -> float:
            if p["seq_len_k"] == 1:
                return table[(32, n)][1] + (1 - 32) * (
                    table[(48, n)][1] - table[(32, n)][1]) / 16
            return table[(p["seq_len_k"], n)][1]

        fraction = (nql - low) / (high - low)
        compute_us = grid_time(low) + fraction * (grid_time(high) - grid_time(low))
        close(compute_us, p["per_layer_compute_us"], "compute interpolation")
        close(volume_gib, p["per_layer_kv_gib"], "KV volume")
        close(volume_gib / (compute_us / 1e6), p["required_bandwidth_gibps"], "bandwidth")
        profile_audit.append({
            "npu_id": p["npu_id"], "prefix_tokens": prefix,
            "volume_bytes": volume_bytes, "volume_gib": volume_gib,
            "command_count": math.ceil(prefix / 128),
            "full_commands": prefix // 128, "tail_tokens": prefix % 128,
            "interpolation_endpoints_us": [grid_time(low), grid_time(high)],
            "interpolation_fraction": fraction, "compute_ms": compute_us / 1000,
            "service_alone_ms": volume_gib / 40 * 1000,
            "nominal_gibps": volume_gib / (compute_us / 1e6),
        })

    baseline = result["baseline"]
    start, end = baseline["measurement_start_ms"], baseline["measurement_end_ms"]
    close(end - start, 1000.0, "window duration")
    layers = read_csv("request_layer_timeline.csv")
    blocks = read_csv("physical_block_trace.csv")
    per_npu = []
    for npu in range(4):
        rows = [r for r in layers if int(r["npu_id"]) == npu]
        compute = coverage([(float(r["compute_start_time_ms"]), float(r["compute_end_time_ms"]))
                            for r in rows], start, end)
        barrier = coverage([(float(r["comparison_deadline_ms"]), float(r["compute_start_time_ms"]))
                            for r in rows], start, end)
        close(compute, baseline["compute_ms_by_npu"][npu], "compute coverage")
        close(compute + barrier, 1000.0, "continuous active work")
        cohort = [r for r in rows if start <= float(r["comparison_deadline_ms"]) < end]
        missed = [r for r in cohort if float(r["io_ready_time_ms"]) >
                  float(r["comparison_deadline_ms"]) + 1e-8]
        latency = [float(r["io_ready_time_ms"]) - float(r["io_release_time_ms"])
                   for r in cohort]
        per_npu.append({
            "npu_id": npu, "compute_ms": compute, "barrier_ms": barrier,
            "utilization": compute / 1000, "deadlines_in_window": len(cohort),
            "missed_deadlines": len(missed), "deadline_miss_fraction": len(missed) / len(cohort),
            "max_layer_release_to_ready_ms": max(latency),
        })
    close(sum(r["utilization"] for r in per_npu) / 4,
          baseline["mean_npu_utilization"], "mean utilization")
    busy = coverage([(float(r["ssd_start_time_ms"]), float(r["ssd_end_time_ms"]))
                     for r in blocks], start, end)
    close(busy / 1000, baseline["measurement_ssd_mean_utilization"], "SSD busy")

    layer = next(r for r in layers if r["request_id"] == "100020" and r["layer"] == "3")
    commands = sorted((r for r in blocks if r["request_id"] == "100020" and r["layer"] == "3"),
                      key=lambda r: float(r["ssd_start_time_ms"]))
    assert len(commands) == 1534
    first = float(commands[0]["ssd_start_time_ms"])
    last = float(commands[-1]["ssd_end_time_ms"])
    service = sum(float(r["ssd_end_time_ms"]) - float(r["ssd_start_time_ms"]) for r in commands)
    close(last - first, service, "uninterrupted own service")
    for before, after in zip(commands, commands[1:]):
        close(float(before["ssd_end_time_ms"]), float(after["ssd_start_time_ms"]), "service continuity")
    close(service, profile_audit[1]["service_alone_ms"], "own-service byte accounting")
    release = float(layer["io_release_time_ms"])
    ready = float(layer["io_ready_time_ms"])
    deadline = float(layer["comparison_deadline_ms"])
    example = {
        "request_id": 100020, "layer": 3, "npu_id": 1,
        "release_relative_ms": release - start, "first_service_relative_ms": first - start,
        "last_ssd_completion_relative_ms": last - start, "ready_relative_ms": ready - start,
        "deadline_relative_ms": deadline - start,
        "queue_before_first_service_ms": first - release,
        "own_service_ms": service, "link_tail_ms": ready - last,
        "ready_latency_ms": ready - release, "budget_ms": deadline - release,
        "barrier_ms": ready - deadline,
    }
    close(example["barrier_ms"], float(layer["io_barrier_wait_ms"]), "example barrier")
    c0 = profile_audit[0]["compute_ms"]
    sensitivity = json.loads((DATA / "tutorial_sensitivity_results.json").read_text(encoding="utf-8"))
    sensitivity_source_verification = audit_sources(sensitivity["source_sha256"])
    cases = sensitivity["cases"]
    assert len(cases) == 5
    for case in cases:
        assert case["all_invariants_pass"] and case["nominal_gibps"] <= 40
        close(case["measurement_end_ms"] - case["measurement_start_ms"], 1000, "probe window")
        close(sum(case["per_npu_utilization"]) / 4, case["mean_utilization"], "probe mean")
        assert case["mean_utilization"] < 0.8
    close(cases[0]["mean_utilization"], baseline["mean_npu_utilization"], "baseline reproduction")
    return {
        "schema_version": 1, "all_checks_pass": True,
        "source_verification": {
            "scope": "Historical evidence provenance; snapshots do not certify the current runners",
            "experiment": source_verification,
            "sensitivity": sensitivity_source_verification,
        },
        "measurement_window_ms": [start, end], "profiles": profile_audit,
        "npu_window_audit": per_npu, "ssd_busy_ms": busy,
        "npu1_causal_example": example,
        "npu0_queue_budget_mib_ignoring_link": (40 * c0 / 1000 - profile_audit[0]["volume_gib"]) * 1024,
        "npu0_compute_windows_for_16_7ms": math.ceil(16.7 / c0),
        "npu0_two_compute_windows_ms": 2 * c0,
        "sensitivity_cases": cases,
        "evidence_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                            for name in ("result.json", "request_layer_timeline.csv",
                                         "physical_block_trace.csv", "tutorial_sensitivity_results.json")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: profile derivation, historical source hashes, 1-second coverage, causal timeline, five probes")
    verified = report["source_verification"]
    snapshots = sorted({path for group in ("experiment", "sensitivity")
                        for path in verified[group].values() if "/source_snapshots/" in path})
    for path in snapshots:
        print(f"Historical source snapshot used (not current runner): {path}")
    print(json.dumps(report["npu1_causal_example"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
