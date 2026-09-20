#!/usr/bin/env python3
"""Audit the document-population experiment and derive exact timing tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from metrics import summarize
from inputs.runners.run_baseline_npu32_stress import load_manifest
from simulator.policies.baselines import od_npu_path_ids

ORDERS = ("random", "ordered")
POLICIES = ("asu_baseline", "od_baseline")
SEEDS = (7, 19, 43)
WINDOWS = ("warm_2_4s", "long_2_6s", "full_population")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if Path(path).suffix == ".gz" else data)


def write_csv(name, rows, fallback=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def equivalent(a, b, label=""):
    if isinstance(a, dict):
        assert set(a) == set(b), (label, set(a) ^ set(b))
        for key in a:
            equivalent(a[key], b[key], label + "." + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, f"{label}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-10), (label, a, b)
    else:
        assert a == b, (label, a, b)


def flatten_stats(stats):
    result = dict(cohort_count=stats["count"])
    for clock in ("admission", "arrival"):
        for metric, value in stats[clock]["latency_ms"].items():
            result[f"{clock}_latency_{metric}_ms"] = value
        for metric, value in stats[clock]["normalized_latency"].items():
            result[f"{clock}_normalized_{metric}"] = value
        for factor, code in (("1", "1"), ("1.5", "1p5")):
            for metric, value in stats[clock]["slo"][factor].items():
                result[f"{clock}_slo{code}_{metric}"] = value
    return result


def macro(rows, identities):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in identities)].append(row)
    result = []
    for key, part in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in part)
        assert len(seeds) == len(set(seeds)), (key, "duplicate seed")
        output = dict(zip(identities, key))
        output.update(seed_count=len(seeds), seeds=json.dumps(seeds), complete_seed_set=seeds == list(SEEDS),
                      seed_interpretation="queue_shuffle_and_simulator_submission_seed" if output.get("order") == "random"
                      else "fixed_queue_varying_simulator_submission_seed")
        for field in part[0]:
            if field in identities or field in ("seed", "case", "manifest_sha256", "result_sha256"):
                continue
            values = [row[field] for row in part if isinstance(row[field], (int, float)) and not isinstance(row[field], bool)]
            if values:
                output["mean_" + field] = math.fsum(values) / len(values)
                output["min_" + field] = min(values)
                output["max_" + field] = max(values)
                if len(values) != len(part):
                    output["valid_seed_count_" + field] = len(values)
        for field in ("all_npus_active", "strict_underload_all_disks"):
            if field in part[0]:
                output["all_seeds_" + field] = all(row[field] for row in part)
        result.append(output)
    return result


def ownership_audit(command, raw):
    expected = [0] * 32 if command["policy"] == "asu_baseline" else list(od_npu_path_ids(32))
    assert raw["baseline_ownership_audit"]["passed"]
    assert raw["baseline_ownership_audit"]["npu_path_ids"] == expected
    for disk in raw["observed_path_ids_by_ssu_npu"]:
        assert all(ids == [expected[npu]] for npu, ids in enumerate(disk))
    if command["policy"] == "od_baseline":
        assert len(set(expected)) == 32
        for qos in raw["actual_qos_by_ssu"]:
            assert sum(value > 0 for value in qos["path_cirs_gib_s"]) == 32
            for path_id in expected:
                equivalent(qos["path_cirs_gib_s"][path_id], 1.25, "OD CIR")
                assert qos["path_pirs_gib_s"][path_id] == "unlimited"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    sources = {}

    def source(path, expected=None):
        key, digest = str(path.relative_to(ROOT)), sha(path)
        if key in sources:
            assert sources[key] == digest, ("source changed", key)
        if expected is not None:
            assert digest == expected, ("source hash mismatch", key)
        sources[key] = digest
        return digest

    source(HERE / "input_audit.json")
    audit = read(HERE / "input_audit.json")
    source(ROOT / "data", audit["source_data_sha256"])
    source(ROOT / "docs/continuous_underload_reproduction.md", audit["source_document_sha256"])
    inputs = {(case["order"], case["seed"]): case for case in audit["cases"]}
    assert set(inputs) == {(o, s) for o in ORDERS for s in SEEDS}
    assert len({case["canonical_population_and_placement_sha256"] for case in inputs.values()}) == 1
    cached = {}
    for key, case in inputs.items():
        path = ROOT / case["manifest"]
        source(path, case["manifest_sha256"])
        source(ROOT / case["csv"], case["csv_sha256"])
        requests, meta = load_manifest(path)
        assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"], meta["layout"]) == (32, 3, 8, "block_ring_hash")
        assert len(requests) == 640 and all(q.arrival_time_ms == 0 for q in requests)
        cached[key] = (requests, meta)
    expected = {(order, policy, seed) for order in ORDERS for policy in POLICIES for seed in SEEDS}
    commands = sorted((HERE / "runs").glob("*/command.json"))
    plan = HERE / "formal_plan.json"
    if plan.exists():
        source(plan)
        jobs = read(plan)["jobs"]
        labels = {job["label"] for job in jobs}
        assert len(labels) == len(jobs) == len(expected)
        commands = [path for path in commands if path.parent.name in labels]
    compared, grouped, npus, disks, intervals, violations, samples, cases = [], [], [], [], [], [], [], []
    seen, finished = set(), set()
    for path in commands:
        command = read(path)
        if command.get("smoke") or command.get("pilot"):
            continue
        key = (command["order"], command["policy"], command["seed"])
        assert key in expected and key not in seen, ("unexpected/duplicate formal case", key)
        seen.add(key)
        if command["status"] == "running":
            continue
        assert command["status"] == "complete", (path, command["status"], command.get("error"))
        assert command["completed_simulation"] and all(command["checks"].values())
        assert all(command[k] for k in ("core_unchanged", "extension_unchanged", "original_manifest_unchanged"))
        source(path)
        case = inputs[command["order"], command["seed"]]
        source(path.parent / "manifest.json.gz", case["manifest_sha256"])
        assert command["manifest_sha256"] == case["manifest_sha256"]
        source(path.parent / "result.json.gz", command["result_sha256"])
        assert command["output_sha256"] == command["result_sha256"]
        for name, digest in {**command["core_source_sha256"], **command["extension_source_sha256"]}.items():
            source(ROOT / name, digest)
        requests, meta = cached[command["order"], command["seed"]]
        byid = {q.request_id: q for q in requests}
        raw = read(path.parent / "result.json.gz")
        summary, rows = raw["summary"], raw["summary"]["request_metrics"]
        assert len(rows) == len({r["request_id"] for r in rows}) == command["completed_requests"] == 640
        assert command["observed_blocks"] == command["expected_blocks"] == case["blocks"]
        assert all(summary["invariants"].values())
        assert raw["input_fingerprint"] == meta["input_fingerprint"]
        assert raw["input_placement_fingerprint"] == raw["execution_placement_fingerprint"]
        assert raw["adapter_statistics"]["reorder_calls"] == 0 and raw["adapter_statistics"]["cir_write_events"] == []
        assert raw["prefetch_audit"]["all_releases_at_predecessor_last_compute_start"]
        assert raw["prefetch_audit"]["observed_cross_request_prefetches"] == 608
        ownership_audit(command, raw)
        base = dict(order=key[0], policy=key[1], seed=key[2])
        for row in rows:
            q = byid[row["request_id"]]
            ideal = 8 * q.load["per_layer_us"] / 1000
            equivalent(row["own_compute_ms"], ideal, "eight-layer compute")
            equivalent(q.load["source_ttft_ms"], 78 * q.load["per_layer_us"] / 1000, "source TTFT")
            sample = dict(**base, request_id=q.request_id, original_request_id=q.load["original_request_id"],
                          npu_id=q.npu_id, category=q.load["category"], length_K=q.load["seq_len_k"],
                          miss_tokens=q.load["nql"], profile=q.load["role"], own_compute_ms=ideal,
                          arrival_ms=row["arrival_time_ms"], admission_ms=row["admission_time_ms"],
                          completion_ms=row["completion_time_ms"],
                          in_warm_2_4s=2000 <= row["admission_time_ms"] < 4000,
                          in_long_2_6s=2000 <= row["admission_time_ms"] < 6000)
            for clock in ("admission", "arrival"):
                duration = row["completion_time_ms"] - row[clock + "_time_ms"]
                assert math.isfinite(duration) and duration >= ideal - 1e-7
                sample[clock + "_latency_ms"] = duration
                sample[clock + "_latency_ratio"] = duration / ideal
                for factor, code in ((1.0, "1"), (1.5, "1p5")):
                    sample[clock + "_slo" + code + "_pass"] = duration <= factor * ideal + 1e-9
            samples.append(sample)
        assert len(raw["analysis"]) == 3
        for i, (name, window) in enumerate(zip(WINDOWS, raw["analysis"])):
            left, right = window["start_ms"], window["end_ms"]
            assert (left, right) == ((2000, 4000) if i == 0 else (2000, 6000) if i == 1 else (0, summary["makespan_ms"]))
            measured = summarize(summary, requests, left, right, full=i == 2)
            equivalent(measured, {field: window[field] for field in measured}, path.parent.name + "." + name)
            line = dict(**base, window=name, start_ms=left, end_ms=right, U_percent=measured["U_percent"],
                        all_npus_active=measured["all_npus_active"],
                        admitted_in_window=measured["admitted_in_window"], completed_in_window=measured["completed_in_window"],
                        completed_after_window=measured["completed_after_window"],
                        timely_completions_per_second=measured["timely_completions_per_second"],
                        total_completions_per_second=measured["total_completions_per_second"],
                        strict_underload_all_disks=measured["demand"]["strict_underload_all_disks"],
                        any_disk_overload_percent=measured["demand"]["any_disk_overload_percent"],
                        any_disk_at_or_above_capacity_percent=measured["demand"]["any_disk_at_or_above_capacity_percent"],
                        makespan_ms=summary["makespan_ms"], **flatten_stats(measured["request_statistics"]))
            for d in range(3):
                actual = window["SSD_GiB_s"][d]
                assert 0 <= actual <= 40 + 1e-7
                equivalent(actual / 40 * 100, window["SSD_busy_percent"][d], "disk busy integral")
                if i == 0:
                    equivalent(sum(raw["warm_ssd_10ms_GiB_s"][d]) / 200, actual, "disk service bins")
                    equivalent(sum(raw["warm_ssd_GiB_s_by_ssu_npu"][d]), actual, "disk per-card service")
                if i == 2:
                    volume = math.fsum(v * 8 for q in requests for disk, v in q.placement[0] if disk == d)
                    equivalent(volume / (right / 1000), actual, "full bytes conservation")
                demand = measured["demand"]
                disk = dict(**base, window=name, ssu_id=d, capacity_GiB_s=40,
                            mean_demand_GiB_s=demand["per_disk_mean_GiB_s"][d],
                            maximum_demand_GiB_s=demand["per_disk_max_GiB_s"][d],
                            minimum_demand_GiB_s=demand["per_disk_min_GiB_s"][d],
                            overload_percent=demand["per_disk_overload_percent"][d],
                            at_or_above_capacity_percent=demand["per_disk_at_or_above_capacity_percent"][d],
                            actual_GiB_s=actual, busy_percent=window["SSD_busy_percent"][d])
                disks.append(disk)
                for metric in ("mean_demand_GiB_s", "maximum_demand_GiB_s", "overload_percent", "actual_GiB_s"):
                    line[f"SSU{d}_{metric}"] = disk[metric]
            line.update(case=path.parent.name, manifest_sha256=command["manifest_sha256"], result_sha256=command["result_sha256"])
            compared.append(line)
            for dimension in ("length", "category", "profile"):
                for value, stats in measured["by_" + dimension].items():
                    grouped.append(dict(**base, window=name, dimension=dimension, value=value,
                                        active_U_percent=stats["active_U_percent"], compute_ms=stats["compute_ms"],
                                        active_ms=stats["active_ms"], **flatten_stats(stats)))
            for npu, value in enumerate(measured["per_npu_U_percent"]):
                npus.append(dict(**base, window=name, npu_id=npu, U_percent=value,
                                 active_ms=measured["per_npu_active_ms"][npu],
                                 compute_ms=value / 100 * (right - left)))
            for segment, active in zip(measured["demand"]["segments"], measured["demand"]["active_npu_count_segments"]):
                a, z = segment[:2]
                assert active[:2] == [a, z]
                for d, value in enumerate(segment[2:]):
                    interval = dict(**base, window=name, ssu_id=d, start_ms=a, end_ms=z,
                                    duration_ms=z-a, demand_GiB_s=value, capacity_GiB_s=40,
                                    margin_GiB_s=40-value, active_npu_count=active[2],
                                    strictly_underload=value < 40, at_capacity=value == 40, overloaded=value > 40)
                    intervals.append(interval)
                    if value >= 40:
                        violations.append(interval)
        finished.add(key)
        cases.append(dict(**base, case=path.parent.name, manifest_sha256=command["manifest_sha256"],
                          result_sha256=command["result_sha256"]))
    macros = macro(compared, ("order", "policy", "window"))
    index = {(row["order"], row["policy"], row["seed"], row["window"]): row for row in compared}
    paired = []
    for order in ORDERS:
        for seed in SEEDS:
            for window in WINDOWS:
                asu, od = (index.get((order, p, seed, window)) for p in POLICIES)
                if not asu or not od:
                    continue
                assert asu["manifest_sha256"] == od["manifest_sha256"]
                diff = dict(order=order, seed=seed, window=window, comparison="OD-minus-ASU")
                for field in ("U_percent", "admission_slo1_percent", "admission_slo1p5_percent", "arrival_slo1p5_percent",
                              "admission_latency_mean_ms", "admission_latency_p95_ms", "makespan_ms"):
                    diff["delta_" + field] = od[field] - asu[field] if od[field] is not None and asu[field] is not None else None
                paired.append(diff)
    defaults = ("order", "policy", "seed", "window")
    for name, content in (("comparison.csv", compared), ("macro_summary.csv", macros),
                          ("group_metrics.csv", grouped), ("group_macro.csv", macro(grouped, ("order", "policy", "window", "dimension", "value"))),
                          ("per_npu_metrics.csv", npus), ("per_npu_macro.csv", macro(npus, ("order", "policy", "window", "npu_id"))),
                          ("disk_metrics.csv", disks), ("disk_macro.csv", macro(disks, ("order", "policy", "window", "ssu_id"))),
                          ("request_samples.csv", samples), ("demand_intervals.csv", intervals),
                          ("capacity_violations.csv", violations),
                          ("overload_intervals.csv", [row for row in violations if row["overloaded"]]),
                          ("paired_deltas.csv", paired)):
        write_csv(name, content, defaults)
    missing = [dict(order=o, policy=p, seed=s) for o, p, s in sorted(expected - finished)]
    status = "complete" if not missing else "preliminary"
    assert all(sha(ROOT / name) == digest for name, digest in sources.items()), "source changed during summary"
    checks = dict(status=status, complete_cases=len(finished), planned_cases=len(expected), missing_cases=missing,
                  cases=cases, raw_metrics_recalculated=True, full_service_bytes_conserved=True,
                  source_sha256=sources, source_files_unchanged=True, source_script_sha256=sha(Path(__file__)),
                  fixed_windows_ms=[[2000, 4000], [2000, 6000]], uncensored_admission_cohorts=True,
                  overload_not_filtered_or_used_to_modify_input=True,
                  ordered_seed_changes_submission_order_not_queues=True,
                  aggregation="equal mean of per-seed statistics; quantiles are nearest-rank within each seed",
                  documented_original_population=True, simulation_n_layers=8, source_TTFT_equivalent_layers=78)
    (HERE / "summary_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: checks[key] for key in ("status", "complete_cases", "planned_cases")}, ensure_ascii=False))
    if args.require_complete and missing:
        raise SystemExit("Formal grid incomplete; output explicitly marked preliminary.")


if __name__ == "__main__":
    main()
