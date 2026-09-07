"""Run independent simulator processes; never share monkeypatches across runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/multi_ssu_stall_experiments/data"


def jobs(phase):
    policies = [("baseline", "none"), ("layer_once", "none"),
                ("deadline", "none"), ("deadline", "fluid")]
    if phase == "primary":
        return [(s, regime, 20260906, p, a, 0) for s in (6, 7)
                for regime in ("near", "replicated") for p, a in policies]
    if phase == "holdout":
        return [(s, "near", 20260907, p, a, 0) for s in (6, 7) for p, a in policies]
    if phase == "ablation":
        return [(s, "near", 20260906, p, a, interval) for s in (6, 7)
                for p, a, interval in [("demand", "none", 0),
                                        ("deadline_reserve", "none", 0),
                                        ("deadline", "compute", 0),
                                        ("deadline", "none", .1)]]
    if phase == "hotspot":
        return [(s, "near_total", 20260906, p, a, 0)
                for s in (6, 7) for p, a in policies]
    if phase == "interchange":
        return [(s, "near", seed, "stall_interchange", "none", 0)
                for s in (6, 7) for seed in (20260906, 20260907)]
    raise ValueError(phase)


def execute(job):
    s, regime, seed, policy, assignment, interval = job
    name = f"npu32_ssu{s}_{regime}_seed{seed}_{policy}_{assignment}"
    if interval:
        name += f"_interval{interval:g}"
    destination = OUT / (name + ".json")
    if destination.exists():
        return {"case": name, "status": "exists"}
    print(json.dumps({"start": name}), flush=True)
    begin = time.perf_counter()
    command = [sys.executable, "-B", "run_multi_ssu_stall_experiments.py",
               "--num-ssu", str(s), "--seed", str(seed), "--regime", regime,
               "--strategy", policy, "--assignment", assignment,
               "--min-interval-ms", str(interval)]
    with (OUT / "logs" / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    row = {"case": name, "returncode": result.returncode,
           "wall_seconds": time.perf_counter() - begin}
    if result.returncode == 0:
        data = json.loads(destination.read_text())
        window = data["common_window"]
        row.update(utilization=window["mean_npu_utilization"],
                   active=window["all_npus_active_whole_window"],
                   makespan_ms=window["makespan_ms"],
                   p99_ms=window["p99_arrival_to_completion_ms"])
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("primary", "holdout", "ablation", "hotspot", "interchange"), default="primary")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = [f.result() for f in as_completed([pool.submit(execute, j) for j in jobs(args.phase)])]
    (OUT / f"sweep_{args.phase}.json").write_text(json.dumps(rows, indent=2))
    if any(row.get("returncode", 0) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
