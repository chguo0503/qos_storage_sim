#!/usr/bin/env python3
"""Run the existing multi-SSU deadline controller plus causal NPU assignment.

Inputs are frozen stress manifests. This wrapper does not change simulator,
controller, assignment, or stress-runner implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from continuous_batch_sim import continuous_batch_input_fingerprint
from run_baseline_npu32_stress import digest, load_manifest, window_evidence, write_json
from run_multi_ssu_stall_experiments import run_case as run_native
from run_shared_path_experiments import summarize_slo


SOURCE_NAMES = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py",
                "policy_logic.py", "multi_ssu_qos_controller.py",
                "multi_ssu_npu_assignment.py", "run_multi_ssu_stall_experiments.py",
                "run_baseline_npu32_stress.py", "run_shared_path_experiments.py")


def run_probe(requests, metadata, *, strategy="deadline", assignment="fluid",
              windows=((1000.0, 2000.0), (2000.0, 3000.0))):
    started = time.perf_counter()
    input_hash = continuous_batch_input_fingerprint(requests)
    metadata_hash = digest(metadata)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in SOURCE_NAMES}
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = run_native(requests, metadata, strategy=strategy, assignment=assignment)
    summary = result["summary"]
    assert all(summary["invariants"].values())
    assert continuous_batch_input_fingerprint(requests) == input_hash
    assert digest(metadata) == metadata_hash
    assert result["input_fingerprint"] == input_hash
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
               for name, value in sources.items())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    if assignment != "none":
        assigned = {row["request_id"]: row["assigned_npu_id"]
                    for row in result["assignment_log"]}
        assert len(assigned) == len(requests)
        assert all(assigned[row["request_id"]] == row["npu_id"]
                   for row in summary["request_metrics"])
    result.update({
        "strategy": f"{strategy}_{assignment}", "native_strategy": strategy,
        "experiment": "baseline_npu32_stress_native_assignment_v1",
        "metadata_sha256": metadata_hash, "input_fingerprint": input_hash,
        "logical_input_fingerprint": metadata["logical_input_fingerprint"],
        "probe_source_sha256": sources, "runner_sha256": own_hash,
        "stress_runner_sha256": sources["run_baseline_npu32_stress.py"],
        "python_version": sys.version,
        "windows": [window_evidence(summary, requests, start, end)
                    for start, end in windows],
        "slo": summarize_slo(summary, requests, start_ms=windows[0][0],
                             end_ms=windows[0][1]),
    })
    result["common_window"] = result["windows"][0]
    result["wall_seconds_total"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assignment", choices=("none", "fluid", "compute"), default="fluid")
    parser.add_argument("--strategy", choices=("deadline",), default="deadline")
    parser.add_argument("--window", action="append", help="START_MS:END_MS, repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = tuple(tuple(map(float, x.split(":")))
                    for x in (args.window or ("1000:2000", "2000:3000")))
    if any(len(w) != 2 or w[0] < 0 or w[1] <= w[0] for w in windows):
        parser.error("windows must have 0 <= START < END")
    requests, metadata = load_manifest(args.manifest)
    config = {"strategy": args.strategy, "assignment": args.assignment, "windows": windows}
    label = f"{args.strategy}_{args.assignment}"
    destination = args.output / f"{metadata['case_id']}_{label}_{digest(config)[:10]}.json.gz"
    if destination.exists():
        raise FileExistsError(f"preserving result: {destination}")
    try:
        result = run_probe(requests, metadata, **config)
        result["manifest_path"] = str(args.manifest.resolve())
        result["manifest_file_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        write_json(destination, result)
    except Exception as exc:
        write_json(destination.with_name(destination.name + ".failure.json"), {
            "status": "failed", "metadata": metadata, "config": config,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc()})
        raise
    print(json.dumps({"output": str(destination), "strategy": label,
        "input_fingerprint": result["input_fingerprint"],
        "wall_seconds": result["wall_seconds_total"],
        "windows": [{"start_ms": w["start_ms"], "end_ms": w["end_ms"],
                     "utilization": w["mean_npu_utilization"],
                     "all_active": w["all_npus_active_whole_window"]}
                    for w in result["windows"]],
        "admission_slo": result["slo"]["all_requests"]["admission"],
        "arrival_slo": result["slo"]["all_requests"]["arrival"]},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
