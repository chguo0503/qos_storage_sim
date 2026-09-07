#!/usr/bin/env python3
"""Run a reviewable JSON job list against frozen stress manifests in subprocesses."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_baseline_npu32_stress import build_workload, load_manifest, save_manifest, read_json, write_json

ROOT = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def execute(job, output, timeout):
    label, strategy = job["input"], job["strategy"]
    variant = job.get("variant", strategy)
    directory = output / "runs" / label / variant
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("*.json.gz"))
    if existing:
        if len(existing) != 1:
            raise ValueError(f"ambiguous result directory: {directory}")
        result = read_json(existing[0])
        assert all(result["summary"]["invariants"].values())
        assert read_json(directory / "command.json")["job"] == job, "cached job configuration changed"
        frozen = read_json(output / "inputs" / f"{label}.json.gz")
        assert result["input_fingerprint"] == frozen["input_fingerprint"], "cached input changed"
        assert result["strategy"] == strategy
        assert result["stress_runner_sha256"] == hashlib.sha256(
            (ROOT / "run_baseline_npu32_stress.py").read_bytes()).hexdigest(), "cached runner changed"
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
                   for name, value in result["core_and_policy_sha256"].items()), "cached source changed"
        return compact(job, result, existing[0], "existing")
    command = [sys.executable, "-B", str(ROOT / "run_baseline_npu32_stress.py"),
               "--manifest", str(output / "inputs" / f"{label}.json.gz"),
               "--strategy", strategy, "--output", str(directory)]
    for flag, value in job.get("config", {}).items():
        if flag == "windows":
            for start, end in value:
                command.extend(["--window", f"{start}:{end}"])
        else:
            command.extend(["--" + flag.replace("_", "-"), str(value)])
    write_json(directory / "command.json", {"argv": command, "started_utc": utc(), "job": job})
    started = time.monotonic()
    print(json.dumps({"started": label, "strategy": variant, "at": utc()}), flush=True)
    with (directory / "process.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout)
            status = "complete" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    results = list(directory.glob("*.json.gz"))
    if status == "complete" and len(results) == 1:
        return compact(job, read_json(results[0]), results[0], status)
    return {"input": label, "variant": variant, "strategy": strategy, "status": status,
            "wall_seconds": time.monotonic() - started, "log": str(directory / "process.log")}


def compact(job, result, path, status):
    return {"input": job["input"], "strategy": job["strategy"],
            "variant": job.get("variant", job["strategy"]), "status": status,
            "result": str(path), "input_fingerprint": result["input_fingerprint"],
            "wall_seconds": result["wall_seconds_total"],
            "windows": [{k: w[k] for k in ("start_ms", "end_ms", "mean_npu_utilization",
                          "all_npus_active_whole_window", "io_stall_ms_by_npu", "idle_ms_by_npu")}
                        for w in result["windows"]],
            "makespan_ms": result["summary"]["makespan_ms"],
            "full_run_utilization": result["summary"]["fleet_npu_compute_utilization"],
            "admission_slo": result["slo"]["all_requests"]["admission"],
            "arrival_slo": result["slo"]["all_requests"]["arrival"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--label", default="sweep")
    args = parser.parse_args()
    args.output = args.output.resolve()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"{args.label}_plan.json", plan)
    for label, kwargs in plan["inputs"].items():
        path = args.output / "inputs" / f"{label}.json.gz"
        spec_path = args.output / "inputs" / f"{label}.spec.json"
        if path.exists():
            if read_json(spec_path) != kwargs:
                raise AssertionError(f"frozen input specification changed: {label}")
            requests, metadata = load_manifest(path)
        else:
            requests, metadata = build_workload(**kwargs)
            save_manifest(path, requests, metadata)
            write_json(spec_path, kwargs)
        print(json.dumps({"prepared": label, "requests": len(requests),
            "fingerprint": metadata["input_fingerprint"],
            "rho_max": metadata["input_demand"]["hottest_ssu_load_ratio"],
            "compute_scale": metadata["compute_scale_actual"]}), flush=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(ROOT.glob("*.py"))}
    import numpy
    write_json(args.output / f"{args.label}_environment.json", {
        "at_utc": utc(), "python": sys.version, "executable": sys.executable,
        "numpy": numpy.__version__, "platform": platform.platform(),
        "workers": args.workers, "argv": sys.argv, "source_sha256": source_hashes})
    if args.prepare_only:
        return
    rows, start = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute, job, args.output, args.timeout): job for job in plan["jobs"]}
        while pending:
            finished, _ = wait(pending, timeout=45, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"input": job["input"], "strategy": job["strategy"],
                           "status": "orchestrator_error", "error": repr(exc)}
                rows.append(row)
                print(json.dumps({"finished": row}), flush=True)
            state = {"at_utc": utc(), "elapsed_seconds": time.monotonic() - start,
                     "completed": len(rows), "total": len(plan["jobs"]),
                     "failures": sum(r["status"] not in ("complete", "existing") for r in rows),
                     "rows": rows}
            write_json(args.output / f"{args.label}_status.json", state)
            if not finished:
                print(json.dumps({k: state[k] for k in ("at_utc", "elapsed_seconds", "completed", "total", "failures")}), flush=True)
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
               for name, value in source_hashes.items()), "sources changed during sweep"
    if any(r["status"] not in ("complete", "existing") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
