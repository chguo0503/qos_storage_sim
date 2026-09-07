#!/usr/bin/env python3
"""Three bounded NPU3 startup-offset probes with unchanged Baseline inputs.

Startup offsets are interventions, not assumed steady-state phases. Each case
is replayed with a layer-only read observer, and full summary hashes must match.
The original result.json and all existing traces are read-only references.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_baseline_path0_tutorial_sensitivity import SOURCE_NAMES, run_case
from baseline_path0_layout import ROOT, DATA


PERIOD_MS = 29.8565628164477
CASES = tuple({
    "name": f"npu3_start_offset_{numerator}_quarter_period",
    "seed": 42,
    "offsets_ms": [0.0, 0.0, 0.0, PERIOD_MS * numerator / 4],
    "npu3_nql": 896,
} for numerator in (1, 2, 3))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    values = list(values)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
    }


def phase_report(layer_rows, window_start, window_end):
    """Measure actual same-window release phases; exclude all layer-0 events."""
    lanes = {npu: sorted(({
        "npu_id": int(row["npu_id"]),
        "request_id": int(row["request_id"]),
        "layer": int(row["layer"]),
        "release_ms": float(row["io_release_time_ms"]) - window_start,
        "io_ready_ms": float(row["io_ready_time_ms"]) - window_start,
        "compute_start_ms": float(row["compute_start_time_ms"]) - window_start,
        "compute_end_ms": float(row["compute_end_time_ms"]) - window_start,
        "io_barrier_wait_ms": float(row.get("io_barrier_wait_ms", 0.0)),
    } for row in layer_rows
        if int(row["npu_id"]) == npu
        and 1 <= int(row["layer"]) <= 7
        and window_start <= float(row["io_release_time_ms"]) < window_end),
        key=lambda row: row["release_ms"]) for npu in (2, 3)}
    if not all(lanes.values()):
        raise AssertionError("phase analysis needs both NPU2 and NPU3 releases")
    pairs = []
    for event in lanes[2]:
        nearest = min(lanes[3], key=lambda row: abs(row["release_ms"] - event["release_ms"]))
        delta = nearest["release_ms"] - event["release_ms"]
        circular = (delta + PERIOD_MS / 2) % PERIOD_MS - PERIOD_MS / 2
        pairs.append({
            "npu2_request_id": event["request_id"], "npu2_layer": event["layer"],
            "npu2_release_ms": event["release_ms"],
            "npu3_request_id": nearest["request_id"], "npu3_layer": nearest["layer"],
            "npu3_release_ms": nearest["release_ms"],
            "nearest_actual_delta_ms": delta,
            "signed_modulo_period_ms": circular,
            "circular_separation_ms": abs(circular),
        })
    periods = {}
    for npu, events in lanes.items():
        gaps = [b["release_ms"] - a["release_ms"] for a, b in zip(events, events[1:])]
        # Excluding L0 leaves an expected 2*C gap between L7 and the next L1.
        residuals = [gap - max(1, round(gap / PERIOD_MS)) * PERIOD_MS for gap in gaps]
        periods[npu] = {
            "raw_adjacent_gaps_ms": gaps,
            "residual_from_nearest_positive_multiple_of_compute_ms": summarize(residuals),
            "observed_layer_barrier_ms": summarize(row["io_barrier_wait_ms"] for row in events),
        }
    return {
        "period_reference_ms": PERIOD_MS,
        "method": "For each NPU2 L1..L7 release inside the measured half-open second, "
                  "choose the temporally nearest NPU3 L1..L7 release in that same window. "
                  "Reduce (t3-t2) to [-C/2,C/2). This is measured modulo C, not proof "
                  "that the complete queue state is C-periodic; request IDs continue advancing.",
        "signed_phase_ms": summarize(pair["signed_modulo_period_ms"] for pair in pairs),
        "circular_separation_ms": summarize(pair["circular_separation_ms"] for pair in pairs),
        "per_npu_release_gap_checks": periods,
        "release_pairs": pairs,
        "layer_events": lanes,
    }


def run_probe(case):
    import run_baseline_4npu_ssu1_low_utilization as runner

    class LayerOnlyTrace(runner.WindowTimelineTrace):
        def __enter__(self):
            super().__enter__()
            # Retain only the existing read-only compute/layer observer. Physical
            # IO remains entirely unwrapped and no block trace is accumulated.
            runner.sim.DiskIOScheduler.enqueue_many = self._enqueue
            runner.sim.DiskIOScheduler._activate_flow = self._activate
            runner.sim.DiskIOScheduler.complete_ready_flows = self._complete
            return self

    summary_hashes = []
    original_simulate = runner._simulate

    def observe_summary(*args, **kwargs):
        summary = original_simulate(*args, **kwargs)
        summary_hashes.append(runner._json_hash(summary))
        return summary

    runner._simulate = observe_summary
    started = time.monotonic()
    try:
        result = run_case(case)
        print(f"{case['name']}: first pass mean={100*result['mean_utilization']:.6f}%; "
              "replaying layer-only observer", flush=True)
        with LayerOnlyTrace(result["measurement_start_ms"], result["measurement_end_ms"]) as trace:
            replay = run_case(case)
        if result != replay or len(summary_hashes) != 2 or summary_hashes[0] != summary_hashes[1]:
            raise AssertionError("layer-only replay changed the complete simulation summary")
        if trace.block_rows or trace.pending:
            raise AssertionError("layer-only observer unexpectedly recorded physical IO")
        result.update({
            "uninstrumented_summary_sha256": summary_hashes[0],
            "layer_observer_summary_sha256": summary_hashes[1],
            "observer_noninterference": True,
            "phase": phase_report(trace.layer_rows, result["measurement_start_ms"],
                                  result["measurement_end_ms"]),
            "wall_seconds_two_passes": time.monotonic() - started,
        })
        return result
    finally:
        runner._simulate = original_simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DATA / "path0_phase_probe.json")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    baseline_path = DATA / "result.json"
    trace_path = DATA / "request_layer_timeline.csv"
    reference_hashes = {str(path.relative_to(ROOT)): sha256(path)
                        for path in (baseline_path, trace_path)}
    source_hashes = {name: sha256(ROOT / name)
                     for name in (*SOURCE_NAMES, Path(__file__).name)}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["baseline"]
    with trace_path.open(encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    reference = {
        "name": "existing_baseline_zero_offsets",
        "offsets_ms": [0, 0, 0, 0],
        "mean_utilization": baseline["mean_npu_utilization"],
        "per_npu_utilization": baseline["npu_utilizations"],
        "ssd_utilization": baseline["measurement_ssd_mean_utilization"],
        "measurement_start_ms": baseline["measurement_start_ms"],
        "measurement_end_ms": baseline["measurement_end_ms"],
        "phase": phase_report(baseline_rows, baseline["measurement_start_ms"],
                              baseline["measurement_end_ms"]),
    }
    completed = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(CASES))) as pool:
        futures = {pool.submit(run_probe, case): case for case in CASES}
        for future in as_completed(futures):
            result = future.result()
            result["mean_utilization_delta_percentage_points_vs_reference"] = (
                100 * (result["mean_utilization"] - reference["mean_utilization"]))
            completed.append(result)
            print(f"{result['name']}: mean={100*result['mean_utilization']:.6f}%, "
                  f"measured circular phase median="
                  f"{result['phase']['circular_separation_ms']['median']:.6f} ms", flush=True)
    for name, digest in reference_hashes.items():
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"original evidence changed: {name}")
    for name, digest in source_hashes.items():
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"simulator source changed during probe: {name}")
    payload = {
        "schema_version": 1,
        "scope": "Three deterministic NPU3-only arrival-offset interventions; unchanged "
                 "4 NPU / 1 SSU / Path0 / 8 layers / cross-request prefetch / 40 GiB/s. "
                 "Every utilization uses its own same-rule middle one-second window. "
                 "Not random jitter, a global minimization, or a claim of universal phase locking.",
        "startup_period_reference_ms": PERIOD_MS,
        "source_sha256": source_hashes,
        "original_evidence_sha256": reference_hashes,
        "original_evidence_unchanged": True,
        "baseline_reference": reference,
        "cases": sorted(completed, key=lambda case: case["offsets_ms"][3]),
        "all_invariants_and_observer_checks_pass": all(
            case["all_invariants_pass"] and case["observer_noninterference"] for case in completed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
