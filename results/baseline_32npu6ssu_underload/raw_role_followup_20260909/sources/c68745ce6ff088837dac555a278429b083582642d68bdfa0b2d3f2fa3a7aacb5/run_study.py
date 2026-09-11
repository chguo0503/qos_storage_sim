#!/usr/bin/env python3
"""Frozen mixed-input probes with a policy-independent active-profile rate bound."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "results/baseline_npu32_investigation")]

import run_mixed_sustained_probe as helper
from run_baseline_npu32_stress import save_manifest, read_json, write_json
from run_coflow_experiments import source_files

WINDOW = (2000, 4000)
SPECS = {
    "raw_s2_l1": dict(family="raw", profile_keys="32:1024,160:1024", quotas=(2, 1), short_profile_count=1),
    "raw_s4_l1": dict(family="raw", profile_keys="32:1024,160:1024", quotas=(4, 1), short_profile_count=1),
    "raw_s8_l1": dict(family="raw", profile_keys="32:1024,160:1024", quotas=(8, 1), short_profile_count=1),
    "raw_varied": dict(family="raw", profile_keys="32:1024,48:1024,64:1024,160:1024", quotas=(4, 3, 2, 2), short_profile_count=3),
    "synthetic_s60_l1": dict(family="aligned", profile_keys="1:128,1:256,1:384,160:1024", quotas=(20, 20, 20, 1), short_profile_count=3),
    "synthetic_s150_l1": dict(family="aligned", profile_keys="1:128,1:256,1:384,160:1024", quotas=(50, 50, 50, 1), short_profile_count=3),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate(requests):
    # This bounds V_s/C of the current admitted request for every possible
    # combination of profiles and every policy. It does not bound burst arrivals
    # or prove feasibility of short-to-long L0 prefetch deadlines.
    maxima = [[0.0] * 6 for _ in range(32)]
    total_maxima = [0.0] * 32
    for request in requests:
        c_ms = request.load["per_layer_us"] / 1000
        for layer in request.placement:
            by_disk = [math.fsum(v for s, v in layer if s == disk) for disk in range(6)]
            rate = [1000 * v / c_ms for v in by_disk]
            n = request.npu_id
            maxima[n] = [max(x, y) for x, y in zip(maxima[n], rate)]
            total_maxima[n] = max(total_maxima[n], math.fsum(rate))
    disk_bounds = [math.fsum(row[s] for row in maxima) for s in range(6)]
    return {
        "definition": "sum_n max_profile V[n,s]/C[n]; current admitted request only",
        "per_ssu_upper_bound_gib_s": disk_bounds,
        "total_upper_bound_gib_s": math.fsum(total_maxima),
        "per_npu_receive_upper_bound_gib_s": total_maxima,
        "passes": max(disk_bounds) <= 40 and math.fsum(total_maxima) <= 240 and max(total_maxima) <= 50,
        "caveat": "Not an arrival-rate envelope, outstanding-work bound, or proof that every prefetch deadline is feasible.",
    }


def prepare(seeds):
    helper.NUM_SSU = 6
    source_paths = {ROOT / name for name in source_files()}
    source_paths.update((ROOT / "data", ROOT / "run_baseline_npu32_stress.py", Path(helper.__file__), Path(__file__)))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(source_paths)}
    for path in sorted(source_paths):
        target = HERE / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(path), f"source snapshot changed: {path}"
        else:
            shutil.copyfile(path, target)
    inputs = []
    for seed in seeds:
        for label, spec in SPECS.items():
            requests, metadata = helper.build_input(seed=seed, label=label, **spec)
            proof = certificate(requests)
            assert proof["passes"], (label, proof)
            metadata.update(
                experiment="32npu_6ssu_active_profile_underload_v1",
                stripe_group_rule="group=original_npu_id//4; ssu=(block_index+group)%6",
                active_profile_rate_certificate=proof,
                measurement_window_ms=list(WINDOW),
                warmup_requirement="Every NPU completes at least four requests by 1500 ms; then at least 500 ms settling before [2000,4000).",
                mixed_window_requirement="Every NPU has positive short and long compute in the fixed window, and is active for all 2000 ms.",
                construction_runner_sha256=sha(__file__),
            )
            path = HERE / "inputs" / f"{metadata['label']}.json.gz"
            save_manifest(path, requests, metadata)
            write_json(path.with_name(path.stem + ".description.json"), metadata)
            inputs.append({"label": metadata["label"], "manifest": str(path),
                           "input_fingerprint": metadata["input_fingerprint"],
                           "request_count": len(requests),
                           "minimum_compute_ms": min(metadata["input_demand"]["per_npu_ideal_compute_ms"]),
                           "rate_certificate": proof})
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "num_npu": 32, "num_ssu": 6, "n_layers": 8, "window_ms": list(WINDOW),
            "source_sha256": hashes, "inputs": inputs,
            "python_executable": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "fixed_npu_binding": True,
            "planned_strategies": ["baseline", "new_once"],
            "rate_scope": "All-time current admitted profile V/C, with a per-SSU universal upper bound; bursts and L0 transitions analyzed separately."}
    write_json(HERE / "plan.json", plan)
    print(json.dumps({"prepared": [{"label": x["label"], "N": x["request_count"],
                                  "max_disk_gib_s": max(x["rate_certificate"]["per_ssu_upper_bound_gib_s"])} for x in inputs]}, ensure_ascii=False), flush=True)
    return plan


def run_job(item, strategy, timeout):
    directory = HERE / "runs" / item["label"] / strategy
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [p for p in directory.glob("*.json.gz") if ".failure." not in p.name]
    if outputs:
        assert len(outputs) == 1
        result = read_json(outputs[0])
        assert result["input_fingerprint"] == item["input_fingerprint"]
        return {"label": item["label"], "strategy": strategy, "status": "existing", "output": str(outputs[0])}
    command = [sys.executable, "-B", str(ROOT / "run_baseline_npu32_stress.py"),
               "--manifest", item["manifest"], "--strategy", strategy,
               "--assignment", "fixed", "--window", "2000:4000", "--output", str(directory)]
    started = time.perf_counter()
    record = {"label": item["label"], "strategy": strategy, "command": command,
              "cwd": str(ROOT), "start_utc": datetime.now(timezone.utc).isoformat(),
              "input_fingerprint": item["input_fingerprint"]}
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        write_json(directory / "command.json", record)
        try:
            code = process.wait(timeout=timeout)
            status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code, status = process.returncode, "timeout"
    record.update(returncode=code, status=status, wall_seconds=time.perf_counter() - started,
                  end_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "command.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--strategies", nargs="+", default=["baseline"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    plan = prepare(args.seed or [7]) if args.prepare else read_json(HERE / "plan.json")
    if not args.run:
        return
    assert all(sha(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    inputs = [item for item in plan["inputs"] if not args.labels or item["label"] in args.labels]
    assert inputs, "no selected inputs"
    status_path = HERE / ("status_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_job, item, strategy, args.timeout)
                   for item in inputs for strategy in args.strategies}
        total = len(pending)
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(status_path, {"total": total, "rows": rows})
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if not done:
                print(json.dumps({"finished": len(rows), "total": total, "pending_including_queued": len(pending)}), flush=True)
    assert all(sha(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    if any(row["status"] in ("failed", "timeout") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
