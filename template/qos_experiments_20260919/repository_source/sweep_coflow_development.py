"""Bounded development-only comparisons; never inspect the held-out seed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from run_coflow_experiments import ROOT, _atomic_json


def candidates():
    return [(p, p, .25, "least_slack") for p in ("baseline", "once", "new_once")] + [
        ("s1_window010", "strategy1", .1, "least_slack"),
        ("s1_window025", "strategy1", .25, "least_slack"),
        ("s1_window050", "strategy1", .5, "least_slack"),
        ("s1_window100", "strategy1", 1.0, "least_slack"),
        ("s3_least_slack", "strategy3", .25, "least_slack"),
        ("s3_urgent_short", "strategy3", .25, "urgent_short"),
    ]


def execute(candidate, output):
    label, strategy, window, rule = candidate
    destination = output / label
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", str(ROOT / "run_coflow_experiments.py"),
               "--num-ssu", "6", "--seed", "20260906", "--strategy", strategy,
               "--queue-window-ms", str(window), "--joint-rule", rule,
               "--output", str(destination)]
    started = time.perf_counter()
    print(json.dumps({"start": label}), flush=True)
    with (destination / "run.log").open("w") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    row = {"label": label, "strategy": strategy, "queue_window_ms": window,
           "joint_rule": rule, "returncode": run.returncode,
           "wall_seconds": time.perf_counter() - started, "command": command}
    if run.returncode == 0:
        result = json.loads(next(destination.glob("npu32*.json")).read_text())
        row.update(utilization=result["common_window"]["mean_npu_utilization"],
                   admission_slo=result["slo"]["all_requests"]["admission"],
                   arrival_slo=result["slo"]["all_requests"]["arrival"],
                   makespan_ms=result["summary"]["makespan_ms"],
                   input_fingerprint=result["input_fingerprint"])
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    output = args.output.resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda candidate: execute(candidate, output), candidates()))
    _atomic_json(output / "development_candidates.json", rows)
    if any(row["returncode"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
