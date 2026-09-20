#!/usr/bin/env python3
"""Read-only frozen-input replay to measure exact IO service for figure labels.

No simulator source or frozen experiment is modified. Hooks only observe existing
SSD/link completion events. The optional short-first queue matches the original
diagnostic control. Complete simulator summaries, warm-window metrics and SLOs
must remain bit-for-bit equivalent after canonical JSON serialization.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import continuous_batch_sim as native
import sim
from run_baseline_npu32_stress import load_manifest, read_json, run_case, write_json
from run_fifo_proof import SizePriorityPending

ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "results/fifo_underload_exploration_20260914"
OUTPUT_ROOT = ROOT / "results/fifo_figure_receipts_20260914"
LEFT, RIGHT, BIN_MS = 2000.0, 4000.0, 2.0
NPU_RATE, SSD_RATE = 50.0, 40.0
TOL_MS = 1e-8


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_service(row, start, end, rate):
    """Accumulate clipped service bytes into 2-ms bins; queueing earns no bytes."""
    start, end = max(start, LEFT), min(end, RIGHT)
    if start >= end:
        return 0.0
    total = (end - start) * rate / 1000.0
    index = min(len(row) - 1, int((start - LEFT) / BIN_MS))
    while start < end:
        stop = min(end, LEFT + (index + 1) * BIN_MS)
        row[index] += (stop - start) * rate / 1000.0
        start = stop
        index += 1
    return total


def replay(case_path):
    case = Path(case_path)
    dest = OUTPUT_ROOT / case.name
    dest.mkdir(parents=True, exist_ok=True)
    output_path = dest / "receipts.json"
    if output_path.exists():
        saved = read_json(output_path)
        assert saved["source_result_sha256"] == file_hash(case / "result.json.gz")
        assert saved["source_manifest_sha256"] == file_hash(case / "manifest.json.gz")
        assert all(saved["replay_checks"].values())
        return {"case": case.name, "event": "already_complete", "output": str(output_path)}
    original = read_json(case / "result.json.gz")
    requests, metadata = load_manifest(case / "manifest.json.gz")
    policy = original.get("probe_policy", metadata.get("probe_policy", "fifo"))
    assert policy in ("fifo", "short_first"), policy
    num_npu, num_ssu = metadata["num_npu"], metadata["num_ssu"]
    n_layers = metadata["n_layers"]
    core_hashes = {name: file_hash(ROOT / name)
                   for name in original["core_and_policy_sha256"]}
    assert core_hashes == original["core_and_policy_sha256"]
    protected_hashes = {name: file_hash(case / name)
                        for name in ("manifest.json.gz", "result.json.gz", "metadata.json")}
    volumes = {q.request_id: q.load["per_layer_kv_gb"] for q in requests}
    bins = int((RIGHT - LEFT) / BIN_MS)
    npu_bins = [[0.0] * bins for _ in range(num_npu)]
    ssu_bins = [[0.0] * bins for _ in range(num_ssu)]
    npu_received = [0.0] * num_npu
    ssu_read = [0.0] * num_ssu
    npu_full_bytes = [0.0] * num_npu
    ssu_full_bytes = [0.0] * num_ssu
    last_link_end = [-math.inf] * num_npu
    last_ssd_end = [-math.inf] * num_ssu
    layers = {}
    for request in requests:
        for layer in range(n_layers):
            layers[(request.request_id, layer)] = {
                "request_id": request.request_id, "npu_id": request.npu_id,
                "layer": layer, "completed_blocks": 0, "bytes_gib": 0.0,
                "first_link_start_ms": math.inf, "last_link_end_ms": -math.inf,
                "first_ssd_start_ms": math.inf, "last_ssd_end_ms": -math.inf,
                "window_received_gib": 0.0, "window_ssd_read_gib": 0.0,
            }
    checks = {"link_overlap_count": 0, "ssd_overlap_count": 0,
              "invalid_link_service_duration_count": 0,
              "invalid_ssd_service_duration_count": 0, "nonzero_path_io": 0,
              "ssd_completions_observed": 0, "link_completions_observed": 0}
    original_initialize = sim.PathQueue.__init__
    original_complete = native._register_complete
    original_enqueue_link = native._enqueue_link_io
    started = last_progress = time.perf_counter()

    def initialize(queue, *args, **kwargs):
        original_initialize(queue, *args, **kwargs)
        if queue.path_id == 0:
            queue.pending = SizePriorityPending(volumes)

    def observe_ssd(context, flow, current_time_ms):
        # This hook is SSD-completion ordered; NPU link completion order can differ.
        disk = flow.disk_id
        start, end = flow.ssd_activation_time, current_time_ms
        checks["ssd_completions_observed"] += 1
        checks["ssd_overlap_count"] += start < last_ssd_end[disk] - TOL_MS
        checks["invalid_ssd_service_duration_count"] += abs(
            (end - start) - 1000.0 * flow.total_gb / SSD_RATE) > TOL_MS
        last_ssd_end[disk] = end
        return original_enqueue_link(context, flow, current_time_ms)

    def observe_link(context, flow):
        nonlocal last_progress
        npu, disk = flow.npu_id, flow.disk_id
        start, end = flow.link_start_time, flow.link_end_time
        checks["link_completions_observed"] += 1
        checks["nonzero_path_io"] += flow.queue_id != 0
        checks["link_overlap_count"] += start < last_link_end[npu] - TOL_MS
        checks["invalid_link_service_duration_count"] += abs(
            (end - start) - 1000.0 * flow.total_gb / NPU_RATE) > TOL_MS
        last_link_end[npu] = end
        received = add_service(npu_bins[npu], start, end, NPU_RATE)
        read = add_service(ssu_bins[disk], flow.ssd_activation_time,
                           flow.link_enqueue_time, SSD_RATE)
        npu_received[npu] += received
        ssu_read[disk] += read
        npu_full_bytes[npu] += flow.total_gb
        ssu_full_bytes[disk] += flow.total_gb
        row = layers[(flow.request_id, flow.layer)]
        row["completed_blocks"] += flow.block_count
        row["bytes_gib"] += flow.total_gb
        row["first_link_start_ms"] = min(row["first_link_start_ms"], start)
        row["last_link_end_ms"] = max(row["last_link_end_ms"], end)
        row["first_ssd_start_ms"] = min(row["first_ssd_start_ms"], flow.ssd_activation_time)
        row["last_ssd_end_ms"] = max(row["last_ssd_end_ms"], flow.link_enqueue_time)
        row["window_received_gib"] += received
        row["window_ssd_read_gib"] += read
        if checks["link_completions_observed"] % 100000 == 0 and time.perf_counter() - last_progress >= 25:
            print(json.dumps({"event": "progress", "case": case.name,
                              "simulation_ms": context.current_time_ms,
                              "completed_requests": context.completed_requests,
                              "requests": len(requests)}), flush=True)
            last_progress = time.perf_counter()
        return original_complete(context, flow)

    print(json.dumps({"event": "start", "case": case.name, "policy": policy,
                      "requests": len(requests)}), flush=True)
    with contextlib.ExitStack() as stack:
        if policy == "short_first":
            stack.enter_context(patch.object(sim.PathQueue, "__init__", initialize))
        stack.enter_context(patch.object(native, "_register_complete", observe_link))
        stack.enter_context(patch.object(native, "_enqueue_link_io", observe_ssd))
        result = run_case(requests, metadata, strategy="baseline", assignment="fixed",
                          windows=[(LEFT, RIGHT)])
    matching_sections = ["summary", "windows", "slo"]
    comparison_hashes = {section: {"original": digest(original[section]),
                                   "replay": digest(result[section])}
                         for section in matching_sections}
    replay_checks = {f"{section}_exact_match": comparison_hashes[section]["original"]
                     == comparison_hashes[section]["replay"] for section in matching_sections}
    for section in ("input_fingerprint", "logical_input_fingerprint", "input_placement_fingerprint",
                    "execution_placement_fingerprint"):
        replay_checks[f"{section}_match"] = original.get(section) == result.get(section)
    replay_checks["source_core_unchanged"] = all(file_hash(ROOT / name) == value
                                                for name, value in core_hashes.items())
    replay_checks["frozen_inputs_results_unchanged"] = all(file_hash(case / name) == value
                                                          for name, value in protected_hashes.items())
    replay_checks["all_simulator_invariants_passed"] = all(result["summary"]["invariants"].values())
    replay_checks["all_expected_blocks_observed"] = checks["link_completions_observed"] == (
        n_layers * sum(len(q.placement[0]) for q in requests)) == checks["ssd_completions_observed"]
    replay_checks["no_service_overlap_or_duration_errors"] = all(checks[k] == 0 for k in (
        "link_overlap_count", "ssd_overlap_count", "invalid_link_service_duration_count",
        "invalid_ssd_service_duration_count", "nonzero_path_io"))
    replay_checks["all_layer_bytes_and_block_counts_match_manifest"] = all(
        layers[(q.request_id, layer)]["completed_blocks"] == len(q.placement[0])
        and layers[(q.request_id, layer)]["bytes_gib"] == volumes[q.request_id]
        for q in requests for layer in range(n_layers))
    replay_checks["all_window_bins_reconcile"] = all(
        math.isclose(math.fsum(row), total, abs_tol=1e-7, rel_tol=1e-10)
        for row, total in list(zip(npu_bins, npu_received)) + list(zip(ssu_bins, ssu_read)))
    replay_checks["full_run_byte_conservation"] = math.isclose(
        math.fsum(npu_full_bytes), result["summary"]["expected_read_gb"], abs_tol=1e-9)
    batch_by_request = {b["member_request_ids"][0]: b for b in result["summary"]["microbatch_metrics"]}
    replay_checks["all_layer_receipt_ends_match_io_ready"] = all(
        math.isclose(layers[(rid, layer["layer"])]["last_link_end_ms"],
                     layer["io_ready_time_ms"], abs_tol=TOL_MS, rel_tol=0)
        for rid, batch in batch_by_request.items() for layer in batch["layer_metrics"])
    payload = {
        "schema_version": 1, "case": case.name, "probe_policy": policy,
        "source_case_relative_path": str(case.relative_to(ROOT)),
        "source_result_sha256": protected_hashes["result.json.gz"],
        "source_manifest_sha256": protected_hashes["manifest.json.gz"],
        "observer_script_sha256": file_hash(__file__), "core_and_policy_sha256": core_hashes,
        "window_ms": [LEFT, RIGHT], "num_npu": num_npu, "num_ssu": num_ssu,
        "npu_link_capacity_gib_s": NPU_RATE, "ssu_capacity_gib_s": SSD_RATE,
        "measurement": "Exact intersection of each actual service interval with the window; wait intervals contribute zero bytes.",
        "window_npu_received_gib": npu_received, "window_ssu_read_gib": ssu_read,
        "window_npu_receive_bandwidth_gib_s": [v * 1000 / (RIGHT - LEFT) for v in npu_received],
        "window_ssu_read_bandwidth_gib_s": [v * 1000 / (RIGHT - LEFT) for v in ssu_read],
        "full_run_npu_received_gib": npu_full_bytes, "full_run_ssu_read_gib": ssu_full_bytes,
        "bin_width_ms": BIN_MS, "bin_edges_ms": [LEFT + BIN_MS * i for i in range(bins + 1)],
        "npu_bin_received_gib": npu_bins, "ssu_bin_read_gib": ssu_bins,
        "layers": [layers[key] for key in sorted(layers)],
        "observer_counters": checks, "replay_checks": replay_checks,
        "comparison_hashes": comparison_hashes,
        "window_U_percent": 100 * result["windows"][0]["mean_npu_utilization"],
        "wall_seconds": time.perf_counter() - started,
    }
    if not all(replay_checks.values()):
        write_json(dest / "replay_failure.json", payload)
        raise AssertionError({key: val for key, val in replay_checks.items() if not val})
    write_json(output_path, payload)
    return {"event": "complete", "case": case.name, "output": str(output_path),
            "U_percent": payload["window_U_percent"], "all_replay_checks": True,
            "wall_seconds": payload["wall_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", action="append", help="Optional exact result directory basename.")
    args = parser.parse_args()
    cases = [p.parent for stage in ("formal", "tiny_formal", "mixed_formal")
             for p in (INPUT_ROOT / stage).glob("*/result.json.gz")]
    order = {"s3_L19_S13_seed7_fifo": 0, "s4_L20_S12_seed7_fifo": 1,
             "s3_L19_S13_seed7_short_first": 2, "s4_L20_S12_seed7_short_first": 3}
    cases.sort(key=lambda p: (order.get(p.name, 10), p.parent.name, p.name))
    if args.case:
        cases = [p for p in cases if p.name in args.case]
        assert len(cases) == len(set(args.case)), "unknown case"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(replay, str(case)): case.name for case in cases}
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
