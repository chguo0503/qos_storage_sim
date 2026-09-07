#!/usr/bin/env python3
"""Read-only paired/source/host checks for the four six-SSU results."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else path.open()) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(directory):
    paths = list(directory.glob("*.json.gz"))
    assert len(paths) == 1, (directory, paths)
    return paths[0]


def time_accounting(result):
    summary = result["summary"]
    n, end = summary["num_npu"], summary["makespan_ms"]
    active, compute, last = [0.] * n, [0.] * n, [0.] * n
    for b in summary["microbatch_metrics"]:
        i = b["npu_id"]
        active[i] += b["completion_time_ms"] - b["admission_time_ms"]
        last[i] = max(last[i], b["completion_time_ms"])
        compute[i] += sum(layer["compute_end_ms"] - layer["compute_start_ms"]
                          for layer in b["layer_metrics"])
    return {"makespan_ms": end, "per_npu": [
        {"npu_id": i, "compute_ms": compute[i], "active_ms": active[i],
         "stall_ms": active[i] - compute[i], "idle_ms": end - active[i],
         "tail_idle_ms": end - last[i]} for i in range(n)],
        "mean_compute_ms": sum(compute)/n, "mean_active_ms": sum(active)/n,
        "mean_stall_ms": (sum(active)-sum(compute))/n,
        "mean_idle_ms": end-sum(active)/n, "mean_tail_idle_ms": end-sum(last)/n}


def main():
    subprocess.run([sys.executable, "-B", str(BASE / "notes/aggregate_mixed_profiles.py"),
                    "--study", str(HERE), "--output", str(HERE / "analysis"), "--no-plots"],
                   cwd=ROOT, check=True)
    plan = read(HERE / "plan.json")
    source_check = {name: sha(ROOT / name) == value
                    for name, value in plan["source_sha256"].items()}
    assert all(source_check.values())
    paired, topology = [], []
    signature_keys = ("core_and_policy_sha256", "stress_runner_sha256", "python_version",
                      "collector_interval_ms", "submit_seed", "policy_config")
    environment_keys = ("python_executable", "python_version", "numpy_version", "platform")
    for item in plan["inputs"]:
        results, commands, paths = {}, {}, {}
        for strategy in plan["strategies"]:
            directory = HERE / "runs" / item["label"] / strategy
            paths[strategy] = one(directory)
            results[strategy] = read(paths[strategy])
            commands[strategy] = read(directory / "command.json")
        a, b = results["baseline"], results["new_once"]
        checks = {"same_"+k: a[k] == b[k] for k in signature_keys}
        checks.update({"same_input_fingerprint": a["input_fingerprint"] == b["input_fingerprint"] == item["input_fingerprint"],
                       "same_manifest_metadata": a["metadata"] == b["metadata"],
                       "all_invariants": all(all(x["summary"]["invariants"].values()) for x in results.values()),
                       "fixed_original_npu_bindings": all(not x["assignment_log"] for x in results.values()),
                       "same_host_environment": all(commands["baseline"][k] == commands["new_once"][k] for k in environment_keys),
                       "both_returncode_zero": all(c["returncode"] == 0 for c in commands.values())})
        assert all(checks.values()), checks
        paired.append({"label": item["label"], "checks": checks, "all_checks_passed": True,
                       "results": {k: {"path": str(paths[k]), "sha256": sha(paths[k]),
                                       "full_run_time_accounting": time_accounting(x)} for k,x in results.items()}})
        if item["label"].startswith("aligned_"):
            for strategy, new in results.items():
                directory = BASE / "mixed_varied/local/runs/aligned_short080_seed7" / strategy
                old_path = one(directory)
                old = read(old_path)
                old_command = read(directory / "command.json")
                checks = {"same_"+k: old[k] == new[k] for k in signature_keys}
                checks.update({"same_host_environment": all(old_command[k] == commands[strategy][k] for k in environment_keys),
                               "same_logical_input": old["metadata"]["logical_input_fingerprint"] == new["metadata"]["logical_input_fingerprint"],
                               "physical_input_changed": old["input_fingerprint"] != new["input_fingerprint"]})
                assert all(checks.values()), checks
                topology.append({"strategy": strategy, "checks": checks, "all_checks_passed": True,
                    "old_8_ssu_result": str(old_path), "new_6_ssu_result": str(paths[strategy]),
                    "old_8_ssu_sha256": sha(old_path), "new_6_ssu_sha256": sha(paths[strategy]),
                    "windows": [{"start_ms": ow["start_ms"], "end_ms": ow["end_ms"],
                        "old_8_ssu_utilization": ow["mean_npu_utilization"],
                        "new_6_ssu_utilization": nw["mean_npu_utilization"],
                        "delta_utilization_pp": 100*(nw["mean_npu_utilization"]-ow["mean_npu_utilization"])}
                        for ow,nw in zip(old["windows"],new["windows"])],
                    "old_makespan_ms": old["summary"]["makespan_ms"],
                    "new_makespan_ms": new["summary"]["makespan_ms"]})
    artifact = {"analysis_script_sha256": sha(Path(__file__)), "all_frozen_sources_unchanged": all(source_check.values()),
                "source_checks": source_check, "paired_strategy_audits": paired, "topology_comparisons": topology}
    (HERE / "analysis/pair_and_topology_audit.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"paired_inputs": len(paired), "topology_policy_comparisons": len(topology),
                      "all_checks_passed": True}))


if __name__ == "__main__":
    main()
