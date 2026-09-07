"""Aggregate the 18-input / 36-run native 4-NPU grid, without simulations."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    directory = args.directory
    plan = read(directory / "grid_plan.json")
    assert len(plan["inputs"]) == 18 and len(plan["jobs"]) == 36
    input_rows, run_rows, npu_rows, artifacts = [], [], [], []
    paired = {}
    for label, spec in plan["inputs"].items():
        manifest_path = directory / "inputs" / f"{label}.json.gz"
        manifest = read(manifest_path)
        metadata = manifest["metadata"]
        roles = metadata["profiles"]
        short_ids = [p["npu_id"] for p in roles if p["seq_len_k"] == 1]
        long_profiles = [p for p in roles if p["seq_len_k"] == 192]
        short_count = len(short_ids)
        long_nql = long_profiles[0]["nql"]
        assert 1 <= short_count <= 3
        assert all(p["nql"] == long_nql for p in long_profiles)
        assert all(r["arrival_time_ms"] == 0 for r in manifest["requests"])
        assert metadata["compute_scale_actual"] == 1
        assert metadata["blocks"] == "exact" and metadata["padding_total_gib"] == 0
        assert metadata["equal_176kib_blocks"]
        demand = metadata["input_demand"]
        short_profile = roles[short_ids[0]]
        short_bw = short_profile["per_layer_kv_gib"] * 1e6 / short_profile["per_layer_compute_us"]
        long_bw = long_profiles[0]["per_layer_kv_gib"] * 1e6 / long_profiles[0]["per_layer_compute_us"]
        # Fixed roles, Bs < Bl: the work-conservation relaxation first funds
        # cheap short-card compute. This is asymptotic, not a finite-window bound.
        assert short_bw < long_bw
        long_run_u_relaxation = (short_count + min(4 - short_count,
                                      (40 - short_count * short_bw) / long_bw)) / 4
        label_row = {"input": label, "short_npu_count": short_count, "long_nql": long_nql,
                     "rho_ssu": demand["hottest_ssu_load_ratio"],
                     "capacity_feasible": metadata["load_within_disk_and_link_capacity"]}
        input_rows.append({**label_row, "nominal_gib_s": demand["total_gib_s"],
                           "rho_link_max": demand["largest_npu_receive_load_ratio"],
                           "short_compute_ms": roles[short_ids[0]]["per_layer_compute_us"] / 1000,
                           "long_compute_ms": long_profiles[0]["per_layer_compute_us"] / 1000,
                           "short_nominal_gib_s": short_bw, "long_nominal_gib_s": long_bw,
                           "short_construction": short_profile["construction"]["method"],
                           "long_construction": long_profiles[0]["construction"]["method"],
                           "long_run_compute_utilization_relaxation": long_run_u_relaxation,
                           "long_ssd_service_ms": long_profiles[0]["per_layer_kv_gib"] / 40 * 1000,
                           "request_count": metadata["request_count"], "exact_blocks": True,
                           "partial_tail_commands_present": False,
                           "compute_scale": metadata["compute_scale_actual"],
                           "minimum_lane_ideal_compute_ms": min(demand["per_npu_ideal_compute_ms"]),
                           "input_fingerprint": manifest["input_fingerprint"],
                           "manifest_sha256": sha(manifest_path)})
        for strategy in ("baseline_native", "once_native"):
            run_dir = directory / "runs" / label / strategy
            paths = list(run_dir.glob("*.json.gz"))
            assert len(paths) == 1, f"expected one completed result: {run_dir}"
            path = paths[0]
            result = read(path)
            summary = result["summary"]
            assert result["strategy"] == strategy
            assert result["input_fingerprint"] == manifest["input_fingerprint"]
            assert all(summary["invariants"].values())
            assert summary["request_count"] == metadata["request_count"]
            artifacts.append({"input": label, "strategy": strategy,
                              "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                              "result": str(path), "result_sha256": sha(path),
                              "input_fingerprint": result["input_fingerprint"],
                              "stress_runner_sha256": result["stress_runner_sha256"],
                              "core_and_policy_sha256": result["core_and_policy_sha256"]})
            for window in result["windows"]:
                start, end = window["start_ms"], window["end_ms"]
                assert (start, end) in ((1000, 2000), (2000, 3000))
                compute, active = [0.] * 4, [0.] * 4
                for batch in summary["microbatch_metrics"]:
                    n = batch["npu_id"]
                    active[n] += max(0., min(end, batch["completion_time_ms"])
                                     - max(start, batch["admission_time_ms"]))
                    compute[n] += sum(max(0., min(end, layer["compute_end_ms"])
                                          - max(start, layer["compute_start_ms"]))
                                      for layer in batch["layer_metrics"])
                u = sum(compute) / (4 * (end - start))
                assert abs(u - window["mean_npu_utilization"]) < 1e-10
                assert all(abs(a - (end - start)) < 1e-7 for a in active)
                short_compute = sum(compute[n] for n in short_ids)
                short_u = short_compute / (short_count * (end - start))
                long_u = (sum(compute) - short_compute) / ((4 - short_count) * (end - start))
                row = {**label_row, "strategy": strategy, "start_ms": start, "end_ms": end,
                       "fleet_utilization": u, "short_utilization": short_u,
                       "long_utilization": long_u,
                       "minimum_npu_utilization": min(compute) / (end - start),
                       "all_npus_active": True,
                       "mean_npu_io_stall_ms": sum(a - c for a, c in zip(active, compute)) / 4,
                       "mean_short_io_stall_ms": (end - start) * (1 - short_u),
                       "mean_long_io_stall_ms": (end - start) * (1 - long_u),
                       "full_run_utilization": summary["fleet_npu_compute_utilization"],
                       "makespan_ms": summary["makespan_ms"],
                       "admission_slo_passed": result["slo"]["all_requests"]["admission"]["passed"],
                       "total_request_count": summary["request_count"],
                       "wall_seconds": result["wall_seconds_total"],
                       "input_fingerprint": result["input_fingerprint"]}
                run_rows.append(row)
                paired[label, strategy, start] = row
                for n in range(4):
                    npu_rows.append({**label_row, "strategy": strategy,
                                     "start_ms": start, "end_ms": end, "npu_id": n,
                                     "role": "short" if n in short_ids else "long",
                                     "utilization": compute[n] / (end - start),
                                     "compute_ms": compute[n], "active_ms": active[n],
                                     "stall_ms": active[n] - compute[n],
                                     "idle_ms": (end - start) - active[n]})
    pair_rows = []
    for label, spec in plan["inputs"].items():
        for start in (1000, 2000):
            a, b = paired[label, "baseline_native", start], paired[label, "once_native", start]
            assert a["input_fingerprint"] == b["input_fingerprint"]
            pair_rows.append({k: a[k] for k in ("input", "short_npu_count", "long_nql", "rho_ssu",
                                              "capacity_feasible", "start_ms", "end_ms")}
                             | {"baseline_fleet_u": a["fleet_utilization"],
                                "once_fleet_u": b["fleet_utilization"],
                                "fleet_gain_pp": 100 * (b["fleet_utilization"] - a["fleet_utilization"]),
                                "baseline_short_u": a["short_utilization"],
                                "once_short_u": b["short_utilization"],
                                "baseline_long_u": a["long_utilization"],
                                "once_long_u": b["long_utilization"],
                                "all_npus_active": a["all_npus_active"] and b["all_npus_active"],
                                "input_fingerprint": a["input_fingerprint"]})
    for filename, rows in (("input_metrics.csv", input_rows), ("run_window_metrics.csv", run_rows),
                           ("paired_metrics.csv", pair_rows), ("per_npu_metrics.csv", npu_rows)):
        csv_write(directory / filename, rows)
    out = {"schema_version": 1, "input_count": len(input_rows), "run_count": len(artifacts),
           "all_windows_active": all(r["all_npus_active"] for r in run_rows),
           "all_sources": artifacts, "input_metrics": input_rows,
           "run_window_metrics": run_rows, "paired_metrics": pair_rows,
           "interpretation": "Only compare policies within a frozen input. Across m/NQL the ideal demand and finite request counts change. The short 1K profile is extrapolated, not an empirical data row."}
    (directory / "grid_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"inputs": len(input_rows), "completed_runs": len(artifacts),
                      "window_rows": len(run_rows), "paired_rows": len(pair_rows),
                      "all_windows_active": out["all_windows_active"]}))


if __name__ == "__main__":
    main()
