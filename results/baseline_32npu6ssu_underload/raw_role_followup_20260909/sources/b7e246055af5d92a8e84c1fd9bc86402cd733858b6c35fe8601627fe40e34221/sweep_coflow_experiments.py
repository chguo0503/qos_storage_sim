"""Process-isolated global-coflow sweeps; six policies share immutable inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from run_coflow_experiments import OUT, ROOT, REGIME, STRATEGIES, _atomic_json


def jobs(phase):
    if phase == "primary":
        disks, seeds, small, npus = (5, 6, 7), (20260906, 20260907), False, 32
    elif phase == "pilot":
        disks, seeds, small, npus = (6,), (20260906,), False, 32
    elif phase == "smoke":
        disks, seeds, small, npus = (5, 6, 7), (20260906,), True, 4
    else:
        raise ValueError("phase must be primary, pilot or smoke")
    return [{"num_ssu": ssu, "seed": seed, "strategy": strategy,
             "small": small, "num_npu": npus, "regime": REGIME}
            for ssu in disks for seed in seeds for strategy in STRATEGIES]


def job_name(job):
    name = (f"npu{job['num_npu']}_ssu{job['num_ssu']}_{REGIME}"
            f"_seed{job['seed']}_{job['strategy']}")
    return name + ("_small" if job["small"] else "")


def execute(job, output, python, policy_config=None):
    output = Path(output)
    name = job_name(job)
    destination = output / f"{name}.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        assert existing["summary"]["invariants"]["all_requests_completed"]
        return {"case": name, "status": "exists", "output": str(destination)}
    command = [python, "-B", str(ROOT / "run_coflow_experiments.py"),
               "--num-npu", str(job["num_npu"]), "--num-ssu", str(job["num_ssu"]),
               "--seed", str(job["seed"]), "--strategy", job["strategy"], "--output", str(output)]
    if job["small"]:
        command.append("--small")
    if policy_config:
        command.extend(["--queue-window-ms", str(policy_config["queue_window_ms"]),
                        "--assignment", policy_config["assignment"],
                        "--joint-rule", policy_config["joint_rule"]])
    print(json.dumps({"start": name}), flush=True)
    started = time.perf_counter()
    log_path = output / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--phase", choices=("primary", "pilot", "smoke"), default="primary")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--queue-window-ms", type=float, default=0.25)
    parser.add_argument("--assignment", choices=("compute", "fixed", "pipeline"), default="compute")
    parser.add_argument("--joint-rule", choices=("least_slack", "urgent_short"), default="least_slack")
    args = parser.parse_args()
    selected = [job for job in jobs(args.phase) if job["strategy"] in args.strategies]
    if args.describe_only:
        print(json.dumps(selected, indent=2))
        return
    args.output = args.output.resolve()
    config = {"queue_window_ms": args.queue_window_ms, "assignment": args.assignment,
              "joint_rule": args.joint_rule}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job, args.output, args.python, config) for job in selected]
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda row: row["case"])
    suffix = "" if tuple(args.strategies) == STRATEGIES else "_" + "-".join(args.strategies)
    _atomic_json(args.output / f"sweep_{args.phase}{suffix}.json", rows)
    if any(row.get("returncode", 0) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
