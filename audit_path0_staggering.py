#!/usr/bin/env python3
"""Read-only comparison of commit 9b7e332 and the current four-NPU traces.

Only writes path0_staggering_audit.json. Does not check out an old worktree,
rerun the simulator, or overwrite any source measurement or tutorial.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import subprocess

from baseline_path0_layout import ROOT, DATA

COMMIT = "9b7e332"
OLD = ("standalone/npu32_baseline_timelines/results/"
       "baseline_layer012_32npu_single_request_timeline/")


def git_bytes(name: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=ROOT)


def close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError((actual, expected))


def main() -> None:
    old_summary_bytes = git_bytes(OLD + "summary.json")
    old_csv_bytes = git_bytes(OLD + "request_layer_timeline.csv")
    old_summary = json.loads(old_summary_bytes)
    old_rows = list(csv.DictReader(io.StringIO(old_csv_bytes.decode())))
    old_compute = old_summary["profile"]["per_layer_compute_ms"]
    old_layers = []
    for layer in range(3):
        rows = [r for r in old_rows if int(r["layer"]) == layer]
        releases = [float(r["io_release_time_ms"]) for r in rows]
        old_layers.append({
            "layer": layer,
            "release_min_ms": min(releases),
            "release_max_ms": max(releases),
            "release_span_ms": max(releases) - min(releases),
            "mean_release_to_hbm_ready_ms": statistics.mean(
                float(r["io_ready_time_ms"]) - float(r["io_release_time_ms"])
                for r in rows),
            "max_same_timestamp_physical_commands": old_summary["metrics"][
                "enqueue_timing_by_layer"][f"layer{layer}"][
                "physical_enqueue"]["max_blocks_same_timestamp"],
        })
    old_examples = []
    for npu, layer in ((0, 1), (1, 1), (1, 2), (17, 1), (17, 2)):
        row = next(r for r in old_rows
                   if int(r["npu_id"]) == npu and int(r["layer"]) == layer)
        old_examples.append({
            "npu": npu, "layer": layer,
            "release_ms": float(row["io_release_time_ms"]),
            "hbm_ready_ms": float(row["io_ready_time_ms"]),
            "read_latency_ms": float(row["io_duration_ms"]),
            "compute_budget_ms": float(row["compute_duration_ms"]),
            "stall_ms": float(row["actual_io_barrier_wait_ms"]),
        })
    ssu_counts = [r["physical_block_count"] for r in old_summary["metrics"][
        "enqueue_timing_by_layer"]["layer0"]["physical_enqueue"]["per_ssu"]]
    io_size = 128 * 1408
    busiest_service = max(ssu_counts) * io_size / (40 * 2**30) * 1000
    for row in old_examples:
        if row["npu"] == 17:
            close(row["stall_ms"], busiest_service - old_compute)
    close(old_layers[1]["release_span_ms"], 0.44079753417968703)

    current = json.loads((DATA / "result.json").read_text())
    baseline = current["baseline"]
    origin = baseline["measurement_start_ms"]
    keys = ((0, 111, 6), (0, 111, 7), (1, 100020, 3),
            (2, 200010, 2), (2, 200010, 3),
            (3, 300010, 2), (3, 300010, 3))
    with (DATA / "request_layer_timeline.csv").open() as handle:
        layer_rows = {(int(r["npu_id"]), int(r["request_id"]), int(r["layer"])): r
                      for r in csv.DictReader(handle)}
    blocks = {key: [] for key in keys}
    with (DATA / "physical_block_trace.csv").open() as handle:
        for r in csv.DictReader(handle):
            key = (int(r["npu_id"]), int(r["request_id"]), int(r["layer"]))
            if key in blocks:
                if int(r["path_id"]) != 0 or int(r["ssu_id"]) != 0:
                    raise AssertionError("not the one-SSU Path0 experiment")
                blocks[key].append(r)
    examples = []
    for key in keys:
        row, ios = layer_rows[key], blocks[key]
        release = float(row["io_release_time_ms"])
        final_ssd = max(float(r["ssd_end_time_ms"]) for r in ios)
        first_ssd = min(float(r["ssd_start_time_ms"]) for r in ios)
        service = sum(float(r["ssd_end_time_ms"]) - float(r["ssd_start_time_ms"])
                      for r in ios)
        total_bytes = sum(float(r["size_gib"]) * 2**30 for r in ios)
        profile = current["input"]["profiles"][key[0]]
        close(total_bytes, profile["per_layer_kv_gib"] * 2**30, 1e-5)
        close(service, total_bytes / (40 * 2**30) * 1000)
        examples.append({
            "npu": key[0], "request_id": key[1], "layer": key[2],
            "release_ms": release - origin,
            "first_ssd_ms": first_ssd - origin,
            "last_ssd_ms": final_ssd - origin,
            "hbm_ready_ms": float(row["io_ready_time_ms"]) - origin,
            "deadline_ms": float(row["comparison_deadline_ms"]) - origin,
            "compute_budget_ms": float(row["compute_duration_ms"]),
            "stall_ms": float(row["io_barrier_wait_ms"]),
            "ssd_latency_ms": final_ssd - release,
            "hbm_latency_ms": float(row["io_ready_time_ms"]) - release,
            "wait_before_first_service_ms": first_ssd - release,
            "own_ssd_service_ms": service,
            "total_bytes": total_bytes, "io_count": len(ios),
        })
    by_key = {(r["npu"], r["layer"]): r for r in examples}
    periods = {}
    for npu in (2, 3):
        interval = by_key[npu, 3]["release_ms"] - by_key[npu, 2]["release_ms"]
        close(interval, by_key[npu, 2]["compute_budget_ms"])
        close(by_key[npu, 2]["stall_ms"], 0.0)
        periods[str(npu)] = interval
    npu0 = current["input"]["profiles"][0]
    tolerated_bytes = (40 * 2**30 * npu0["per_layer_compute_us"] / 1e6
                       - npu0["per_layer_kv_gib"] * 2**30)
    paths = ("result.json", "request_layer_timeline.csv", "physical_block_trace.csv")
    payload = {
        "scope": "Frozen trace comparison, not a controlled change of NPU/SSU count",
        "historical_commit": subprocess.check_output(
            ["git", "rev-parse", COMMIT], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            COMMIT + ":" + OLD + "summary.json": hashlib.sha256(old_summary_bytes).hexdigest(),
            COMMIT + ":" + OLD + "request_layer_timeline.csv": hashlib.sha256(old_csv_bytes).hexdigest(),
            **{name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in paths},
        },
        "old": {
            "config": old_summary["config"], "profile": old_summary["profile"],
            "layer_release_statistics": old_layers, "examples_hbm_endpoint": old_examples,
            "blocks_per_ssu_per_layer": ssu_counts,
            "busiest_ssu_service_per_layer_ms": busiest_service,
            "busiest_service_minus_compute_ms": busiest_service - old_compute,
        },
        "current": {
            "time_origin_absolute_ms": origin,
            "measurement_end_absolute_ms": baseline["measurement_end_ms"],
            "mean_npu_utilization": baseline["mean_npu_utilization"],
            "ssd_utilization": baseline["measurement_ssd_mean_utilization"],
            "profiles": current["input"]["profiles"],
            "examples": examples, "no_stall_periods_ms": periods,
            "npu0_ssd_only_tolerated_ahead_mib": tolerated_bytes / 2**20,
            "npu0_full_176kib_commands_tolerated": int(tolerated_bytes // io_size),
        },
        "checks": "PASS: complete selected I/O sets, service-byte identity, old hotspot stall, P=C",
    }
    output = DATA / "path0_staggering_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(payload["checks"])
    print(output)


if __name__ == "__main__":
    main()
