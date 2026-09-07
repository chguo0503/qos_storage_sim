#!/usr/bin/env python3
"""Audit collected mixed jobs without trusting a live orchestrator's status file."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mixed_varied", "mixed_strict_ss"), required=True)
    args = parser.parse_args()
    base = STUDY / args.stage
    plan = read(base / "plan.json")
    manifests = {item["label"]: read(Path(item["manifest"])) for item in plan["inputs"]}
    frozen_source_checks = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
                            for name, value in plan["source_sha256"].items()}
    hash_cache, rows = {}, []
    batches = ("local", "remote", "extra_local", "extra_remote") if args.stage == "mixed_varied" else ("local", "remote")
    for batch in batches:
        subplan = read(base / batch / "plan.json")
        for item in subplan["inputs"]:
            manifest = manifests[item["label"]]
            original_npu = {r["request_id"]: r["npu_id"] for r in manifest["requests"]}
            for strategy in subplan["strategies"]:
                directory = base / batch / "runs" / item["label"] / strategy
                outputs = list(directory.glob("*.json.gz"))
                command_path = directory / "command.json"
                command = read(command_path) if command_path.exists() else {}
                row = {"batch": batch, "label": item["label"], "strategy": strategy,
                       "command": str(command_path.relative_to(ROOT)),
                       "returncode": command.get("returncode"), "wall_seconds": command.get("wall_seconds")}
                if not outputs:
                    row["status"] = "pending" if command.get("returncode") is None else "failed"
                    rows.append(row)
                    continue
                assert len(outputs) == 1, directory
                result = read(outputs[0])
                summary = result["summary"]
                for name in result["core_and_policy_sha256"]:
                    if name not in hash_cache:
                        hash_cache[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                checks = {
                    "successful_command": command.get("returncode") == 0,
                    "input_fingerprint": result["input_fingerprint"] == manifest["input_fingerprint"] == item["input_fingerprint"] == command["input_fingerprint"],
                    "summary_input_fingerprint": summary["input_fingerprint"] == result["input_fingerprint"],
                    "strategy": result["strategy"] == strategy == command["strategy"],
                    "collector_5ms": result["collector_interval_ms"] == 5.0,
                    "stress_runner_hash": result["stress_runner_sha256"] == plan["source_sha256"]["run_baseline_npu32_stress.py"],
                    "all_core_policy_hashes": all(hash_cache[name] == value for name, value in result["core_and_policy_sha256"].items()),
                    "all_simulator_invariants": all(summary["invariants"].values()),
                    "all_requests_completed": result["slo"]["all_input_requests_completed"] and len(summary["request_metrics"]) == len(original_npu),
                    "same_request_ids": {r["request_id"] for r in summary["request_metrics"]} == set(original_npu),
                    "fixed_npu_binding": all(r["npu_id"] == original_npu[r["request_id"]] for r in summary["request_metrics"]),
                    "all_window_npus_active": all(w["all_npus_active_whole_window"] for w in result["windows"]),
                }
                row.update(status="complete" if all(checks.values()) else "failed", checks=checks,
                           result=str(outputs[0].relative_to(ROOT)),
                           result_file_sha256=hashlib.sha256(outputs[0].read_bytes()).hexdigest(),
                           input_fingerprint=result["input_fingerprint"],
                           window_utilizations=[w["mean_npu_utilization"] for w in result["windows"]])
                rows.append(row)
    output = {"stage": args.stage, "collected_utc": datetime.now(timezone.utc).isoformat(),
              "total": len(rows), "complete": sum(r["status"] == "complete" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "pending": sum(r["status"] == "pending" for r in rows),
              "frozen_source_checks": frozen_source_checks,
              "all_complete_and_verified": all(frozen_source_checks.values()) and len(rows) == plan["job_count"] and all(r["status"] == "complete" for r in rows),
              "note": ("Per-job records and artifacts are authoritative here. The mixed_varied remote orchestrator status stopped at 2 after SSH transport failure; no in-flight job was restarted."
                       if args.stage == "mixed_varied" else
                       "Per-job records and artifacts are authoritative here. Both strict-SS batches used detached process groups and persistent batch logs."),
              "rows": rows}
    (base / "collection_status.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))
    assert not output["failed"] and all(frozen_source_checks.values())


if __name__ == "__main__":
    main()
