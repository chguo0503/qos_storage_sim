"""Process-isolated matched sweeps for the shared-Path 5 ms experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from run_shared_path_experiments import OUT, ROOT, STRATEGIES


def jobs(phase):
    """Primary: 30 near runs. Same-trace: 15 fixed-input capacity checks."""
    if phase == "primary":
        regimes, seeds = ("near",), (20260906, 20260907)
    elif phase == "same_trace":
        regimes, seeds = ("same_trace",), (20260906,)
    elif phase == "smoke":
        regimes, seeds = ("near",), (20260906,)
    elif phase == "pilot":
        return [{"num_ssu": 6, "regime": "near", "seed": 20260906,
                 "strategy": strategy, "small": False, "num_npu": 32}
                for strategy in STRATEGIES]
    else:
        raise ValueError("phase must be primary, same_trace, pilot or smoke")
    return [{"num_ssu": ssu, "regime": regime, "seed": seed, "strategy": strategy,
             "small": phase == "smoke", "num_npu": 4 if phase == "smoke" else 32}
            for ssu in (5, 6, 7) for regime in regimes for seed in seeds for strategy in STRATEGIES]


def job_name(job):
    name = (f"npu{job['num_npu']}_ssu{job['num_ssu']}_{job['regime']}"
            f"_seed{job['seed']}_{job['strategy']}")
    return name + ("_small" if job["small"] else "")


def execute(job, output, python):
    output = Path(output)
    name = job_name(job)
    destination = output / f"{name}.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        assert existing["summary"]["invariants"]["all_requests_completed"]
        return {"case": name, "status": "exists", "output": str(destination)}
    command = [python, "-B", str(ROOT / "run_shared_path_experiments.py"),
               "--num-npu", str(job["num_npu"]), "--num-ssu", str(job["num_ssu"]),
               "--seed", str(job["seed"]), "--regime", job["regime"],
               "--strategy", job["strategy"], "--output", str(output)]
    if job["small"]:
        command.append("--small")
    print(json.dumps({"start": name}), flush=True)
    started = time.perf_counter()
    log_path = output / "logs" / f"{name}.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    row = {"case": name, "returncode": completed.returncode,
           "wall_seconds": time.perf_counter() - started, "log": str(log_path)}
    if completed.returncode == 0:
        result = json.loads(destination.read_text())
        row.update({
            "input_fingerprint": result["input_fingerprint"],
            "logical_input_fingerprint": result["logical_input_fingerprint"],
            "utilization": result["common_window"]["mean_npu_utilization"],
            "all_npus_active_whole_window": result["common_window"]["all_npus_active_whole_window"],
            "admission_slo": result["slo"]["all_requests"]["admission"],
            "arrival_slo": result["slo"]["all_requests"]["arrival"],
            "makespan_ms": result["summary"]["makespan_ms"],
        })
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("primary", "same_trace", "pilot", "smoke"), default="primary")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    selected = jobs(args.phase)
    if args.describe_only:
        print(json.dumps(selected, indent=2))
        return
    args.output = args.output.resolve()
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job, args.output, args.python) for job in selected]
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda row: row["case"])
    (args.output / f"sweep_{args.phase}.json").write_text(json.dumps(rows, indent=2))
    if any(row.get("returncode", 0) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
