"""Recheck arrival-only perturbations and aggregate frozen 32-NPU phase runs."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    directory = args.directory
    plan = read(directory / "phase_plan.json")
    rows, artifacts, audits = [], [], []
    for label, spec in plan["inputs"].items():
        base_path = Path(spec["base_manifest"])
        base = read(base_path)
        manifest_path = directory / "inputs" / f"{label}.json.gz"
        manifest = read(manifest_path)
        metadata = manifest["metadata"]
        assert sha(base_path) == spec["base_manifest_sha256"]
        assert len(base["requests"]) == len(manifest["requests"]) == 24024
        assert metadata["input_demand"] == base["metadata"]["input_demand"]
        assert metadata["seed"] == base["metadata"]["seed"] == 42
        changed = 0
        short_npus = set()
        for a, b in zip(base["requests"], manifest["requests"]):
            assert a["request_id"] == b["request_id"] and a["npu_id"] == b["npu_id"]
            assert base["placements"][a["placement_index"]] == manifest["placements"][b["placement_index"]]
            assert {k: v for k, v in a["load"].items() if k not in ("arrival_ms", "arrival_time")} == {
                k: v for k, v in b["load"].items() if k not in ("arrival_ms", "arrival_time")}
            expected = metadata["startup_offset_ms_by_npu"][b["npu_id"]]
            assert b["arrival_time_ms"] == b["load"]["arrival_ms"] == b["load"]["arrival_time"] == expected
            changed += a["arrival_time_ms"] != b["arrival_time_ms"]
            if b["load"]["seq_len_k"] == 1:
                short_npus.add(b["npu_id"])
        assert changed == 24024 and len(short_npus) == 24
        audits.append({"input": label, "changed_arrival_request_count": changed,
                       "preserved_population_profiles_placement_and_submit_seed": True,
                       "base_manifest_sha256": sha(base_path), "manifest_sha256": sha(manifest_path),
                       "base_input_fingerprint": base["input_fingerprint"],
                       "input_fingerprint": manifest["input_fingerprint"],
                       "jitter_rng_seed": metadata["startup_offset_rng_seed"],
                       "submit_seed": metadata["seed"],
                       "offset_ms_by_npu": metadata["startup_offset_ms_by_npu"]})
        for strategy in ("baseline", "once"):
            files = list((directory / "runs" / label / strategy).glob("*.json.gz"))
            assert len(files) == 1
            path = files[0]
            result = read(path)
            summary = result["summary"]
            assert result["input_fingerprint"] == manifest["input_fingerprint"]
            assert result["collector_interval_ms"] == 5
            assert result["submit_seed"] == 42
            assert all(summary["invariants"].values())
            assert summary["request_count"] == 24024
            artifacts.append({"input": label, "strategy": strategy,
                              "result": str(path), "result_sha256": sha(path),
                              "core_and_policy_sha256": result["core_and_policy_sha256"],
                              "stress_runner_sha256": result["stress_runner_sha256"]})
            for window in result["windows"]:
                start, end = window["start_ms"], window["end_ms"]
                compute, active = [0.] * 32, [0.] * 32
                for batch in summary["microbatch_metrics"]:
                    n = batch["npu_id"]
                    active[n] += max(0., min(end, batch["completion_time_ms"])
                                     - max(start, batch["admission_time_ms"]))
                    compute[n] += sum(max(0., min(end, layer["compute_end_ms"])
                                          - max(start, layer["compute_start_ms"]))
                                      for layer in batch["layer_metrics"])
                assert all(abs(a - (end - start)) < 1e-7 for a in active)
                u = sum(compute) / (32 * (end - start))
                assert abs(u - window["mean_npu_utilization"]) < 1e-10
                short_compute = sum(compute[n] for n in short_npus)
                rows.append({"input": label, "jitter_rng_seed": metadata["startup_offset_rng_seed"],
                             "native_submit_seed": 42, "strategy": strategy,
                             "start_ms": start, "end_ms": end,
                             "rho_ssu": metadata["input_demand"]["hottest_ssu_load_ratio"],
                             "fleet_utilization": u,
                             "short_utilization": short_compute / (24 * (end - start)),
                             "long_utilization": (sum(compute) - short_compute) / (8 * (end - start)),
                             "min_npu_utilization": min(compute) / (end - start),
                             "all_npus_active": True,
                             "makespan_ms": summary["makespan_ms"],
                             "full_run_utilization": summary["fleet_npu_compute_utilization"],
                             "admission_slo_passed": result["slo"]["all_requests"]["admission"]["passed"],
                             "request_count": 24024, "wall_seconds": result["wall_seconds_total"],
                             "input_fingerprint": result["input_fingerprint"]})
    with (directory / "phase_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    references = []
    reference_directory = directory.parent / "screen/runs/strong32_stripe"
    for strategy in ("baseline", "once"):
        paths = list((reference_directory / strategy).glob("*.json.gz"))
        if len(paths) != 1:
            continue
        result = read(paths[0])
        assert all(result["input_fingerprint"] == a["base_input_fingerprint"] for a in audits)
        assert result["collector_interval_ms"] == 5 and result["submit_seed"] == 42
        references.append({"strategy": strategy, "result": str(paths[0]),
                           "result_sha256": sha(paths[0]), "input_fingerprint": result["input_fingerprint"],
                           "core_and_policy_sha256": result["core_and_policy_sha256"],
                           "windows": [{k: w[k] for k in ("start_ms", "end_ms", "mean_npu_utilization",
                                                         "all_npus_active_whole_window")}
                                       for w in result["windows"]]})
    (directory / "phase_metrics.json").write_text(json.dumps({
        "schema_version": 1, "manifest_audits": audits, "source_artifacts": artifacts,
        "unshifted_reference_results": references,
        "rows": rows, "all_windows_active": True,
        "caveat": "This is startup-phase robustness for two offset seeds, not a random production workload or a change in native submission RNG."}, indent=2) + "\n")
    print(json.dumps({"completed_runs": len(artifacts), "window_rows": len(rows),
                      "all_windows_active": True}))


if __name__ == "__main__":
    main()
