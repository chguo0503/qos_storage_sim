#!/usr/bin/env python3
"""Audit completed Ring-hash cases and derive tables from raw timings.

No simulation is started here. Incomplete work is explicitly marked preliminary;
--require-complete additionally fails if the full 36-case grid is unavailable.
"""
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

POLICIES = ("asu_baseline", "od_baseline", "once", "static", "mild", "aggressive")
SCENARIOS = ("semi", "full")
SEEDS = (7, 19, 43)
WINDOWS = ("warm_2_4s", "long_2_6s", "full_population")
CLASSES = ("SS", "SL", "LS", "LL")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if Path(path).suffix == ".gz" else data)


def csv_write(name, rows, fallback=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, context):
    assert math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-10), (context, a, b)


def latency_tail(rows, left, right, *, full=False):
    """Uninterpolated nearest-rank quantiles of an uncensored cohort."""
    cohort = rows if full else [row for row in rows if left <= row["admission_time_ms"] < right]
    ratios = sorted((row["completion_time_ms"] - row["admission_time_ms"]) / row["own_compute_ms"]
                    for row in cohort)
    assert ratios and all(math.isfinite(value) and value > 0 for value in ratios)
    n = len(ratios)
    ranks = {95: (95 * n + 99) // 100, 99: (99 * n + 99) // 100}
    return dict(latency_ratio_count=n,
                latency_ratio_p95_rank=ranks[95], latency_ratio_p99_rank=ranks[99],
                latency_ratio_p95_nearest_rank=ratios[ranks[95] - 1],
                latency_ratio_p99_nearest_rank=ratios[ranks[99] - 1],
                latency_ratio_max=ratios[-1])


def equivalent(a, b, context=""):
    """Compare all derived metrics, with tolerance only for floating sums."""
    if isinstance(a, dict):
        assert set(a) == set(b), (context, set(a) ^ set(b))
        for key in a:
            equivalent(a[key], b[key], f"{context}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), context
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, f"{context}[{i}]")
    elif isinstance(a, float):
        close(a, b, context)
    else:
        assert a == b, (context, a, b)


def flatten(command, measured, window):
    row = dict(scenario=command["scenario"], seed=command["seed"], policy=command["policy"],
               window=window, start_ms=measured["start_ms"], end_ms=measured["end_ms"],
               U_percent=measured["U_percent"], slo_percent=measured["slo"]["percent"],
               slo_count=measured["slo"]["count"], slo_passed=measured["slo"]["passed"],
               arrival_slo_percent=measured["arrival_slo"]["percent"],
               completed_after_window=measured["completed_after_window"],
               timely_completions_per_second=measured["timely_completions_per_second"],
               total_completions_per_second=measured["total_completions_per_second"],
               all_npus_active=measured["all_npus_active"],
               all_disks_overload_percent=measured["demand"]["all_disks_overload_percent"],
               any_disk_overload_percent=measured["demand"]["any_disk_overload_percent"])
    for d in range(3):
        row[f"SSU{d}_mean_demand_GiB_s"] = measured["demand"]["per_disk_mean_GiB_s"][d]
        row[f"SSU{d}_overload_percent"] = measured["demand"]["per_disk_overload_percent"][d]
        row[f"SSU{d}_actual_GiB_s"] = measured["SSD_GiB_s"][d]
        row[f"SSU{d}_busy_percent"] = measured["SSD_busy_percent"][d]
    for category in CLASSES:
        part = measured["slo_by_category"][category]
        for key in ("percent", "count", "passed"):
            row[f"{category}_slo_{key}"] = part[key]
        row[f"{category}_active_U_percent"] = measured["category_active_U_percent"].get(category)
    return row


def audit_ownership(command, raw):
    policy = command["policy"]
    actual = raw["actual_qos_by_ssu"]
    assert len(actual) == 3
    paths = raw["observed_path_ids_by_ssu_npu"]
    assert len(paths) == 3 and all(len(disk) == 32 for disk in paths)
    for qos in actual:
        assert qos["path_cirs_gib_s"] == raw["static_path_cirs_gib_s"]
    if policy not in ("asu_baseline", "od_baseline"):
        return
    from simulator.policies.baselines import od_npu_path_ids
    expected = (0,) * 32 if policy == "asu_baseline" else od_npu_path_ids(32)
    for disk in paths:
        assert all(ids == [expected[npu]] for npu, ids in enumerate(disk))
    audit = raw["baseline_ownership_audit"]
    assert audit["passed"] and audit["all_blocks_on_owner_path"]
    assert audit["configured_paths_per_ssu"] == (1 if policy == "asu_baseline" else 32)
    if policy == "od_baseline":
        assert len(set(expected)) == 32
        for qos in actual:
            close(sum(qos["path_cirs_gib_s"]), 40, "OD total CIR")
            assert sum(value > 0 for value in qos["path_cirs_gib_s"]) == 32
            for path_id in expected:
                close(qos["path_cirs_gib_s"][path_id], 1.25, "OD per-card CIR")
                assert qos["path_pirs_gib_s"][path_id] == "unlimited"
            assert len({qos["path_weights"][p] for p in expected}) == 1
            assert len(set(qos["group_weights"])) == 1


def macro_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"], row["policy"], row["window"]].append(row)
    output = []
    for (scenario, policy, window), part in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in part)
        result = dict(scenario=scenario, policy=policy, window=window, seed_count=len(seeds),
                      seeds=json.dumps(seeds), complete_seed_set=seeds == list(SEEDS))
        for field in part[0]:
            if (field.endswith(("_percent", "_per_second", "_GiB_s")) or field == "makespan_ms" or
                    field in ("latency_ratio_p95_nearest_rank", "latency_ratio_p99_nearest_rank", "latency_ratio_max")):
                values = [row[field] for row in part if row[field] is not None]
                result["mean_" + field] = math.fsum(values) / len(values) if values else None
                result["min_" + field] = min(values) if values else None
                result["max_" + field] = max(values) if values else None
        result["pooled_slo_count"] = sum(row["slo_count"] for row in part)
        result["pooled_slo_passed"] = sum(row["slo_passed"] for row in part)
        output.append(result)
    return output


def paired_rows(rows):
    index = {(r["scenario"], r["policy"], r["window"], r["seed"]): r for r in rows}
    output = []
    pairs = (("od_baseline", "asu_baseline"), ("once", "od_baseline"),
             ("static", "od_baseline"), ("static", "once"),
             ("mild", "od_baseline"), ("aggressive", "od_baseline"))
    for scenario in SCENARIOS:
        for window in WINDOWS:
            for policy, reference in pairs:
                for seed in SEEDS:
                    current = index.get((scenario, policy, window, seed))
                    other = index.get((scenario, reference, window, seed))
                    if current is None or other is None:
                        continue
                    assert current["manifest_sha256"] == other["manifest_sha256"]
                    result = dict(scenario=scenario, window=window, seed=seed,
                                  policy=policy, reference=reference)
                    for field in ("U_percent", "slo_percent", "timely_completions_per_second",
                                  "total_completions_per_second", "makespan_ms"):
                        result["delta_" + field] = current[field] - other[field]
                    for category in CLASSES:
                        field = category + "_slo_percent"
                        result["delta_" + field] = (current[field] - other[field]
                                                    if current[field] is not None and other[field] is not None else None)
                    output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    source_hashes = {}

    def source(path):
        name = str(path.relative_to(ROOT))
        digest = sha(path)
        if name in source_hashes:
            assert source_hashes[name] == digest, ("source changed", name)
        source_hashes[name] = digest
        return digest

    source(HERE / "input_audit.json")
    input_audit = read(HERE / "input_audit.json")
    inputs = {(case["scenario"], case["seed"]): case for case in input_audit["cases"]}
    assert set(inputs) == {(s, k) for s in SCENARIOS for k in SEEDS}
    expected = {(s, p, k) for s in SCENARIOS for p in POLICIES for k in SEEDS}
    cached_inputs = {}
    for key, info in inputs.items():
        path = ROOT / info["input"]
        assert source(path) == info["manifest_sha256"]
        requests, meta = load_manifest(path)
        assert meta["layout"] == "block_ring_hash"
        assert meta["order"] == "random"
        assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (32, 3, 8)
        assert meta["source_data_sha256"] == source(ROOT / "data")
        assert len(requests) == info["request_count"]
        cached_inputs[key] = (requests, meta)
    commands = sorted((HERE / "runs").glob("*/command.json"))
    plan_path = HERE / "formal_plan.json"
    if plan_path.exists():
        source(plan_path)
        plan = read(plan_path)
        labels = {job["label"] for job in plan["jobs"]}
        assert len(labels) == len(expected)
        commands = [p for p in commands if p.parent.name in labels]
    complete, previews, categories, profiles, npus, disks, samples, tails = [], [], [], [], [], [], [], []
    finished = {}
    observed = set()
    cases = []
    for command_path in commands:
        command = read(command_path)
        if command.get("pilot") or command.get("smoke"):
            continue
        key = (command["scenario"], command["policy"], command["seed"])
        assert key in expected, ("unexpected formal case", key)
        assert key not in observed, ("duplicate case", key)
        observed.add(key)
        folder = command_path.parent
        input_info = inputs[command["scenario"], command["seed"]]
        assert command["manifest_sha256"] == input_info["manifest_sha256"]
        assert sha(folder / "manifest.json.gz") == input_info["manifest_sha256"]
        if command["status"] == "running":
            preview_path = folder / "warm_preview.json"
            if preview_path.exists():
                preview = read(preview_path)
                preview["SSD_busy_percent"] = [x / 40 * 100 for x in preview["SSD_GiB_s"]]
                previews.append(flatten(command, preview, "warm_2_4s"))
            continue
        assert command["status"] == "complete", (folder.name, command["status"], command.get("error"))
        assert command["completed_simulation"]
        assert command["core_unchanged"] and command["extension_unchanged"]
        assert command["original_manifest_unchanged"] and all(command["checks"].values())
        source(command_path)
        source(folder / "manifest.json.gz")
        assert source(folder / "result.json.gz") == command["result_sha256"] == command["output_sha256"]
        for name, digest in {**command["core_source_sha256"], **command["extension_source_sha256"]}.items():
            assert source(ROOT / name) == digest, ("run source differs from current", name)
        requests, meta = cached_inputs[command["scenario"], command["seed"]]
        byid = {q.request_id: q for q in requests}
        raw = read(folder / "result.json.gz")
        rows = raw["summary"]["request_metrics"]
        assert len(rows) == len(requests) == command["completed_requests"] == command["expected_requests"]
        assert {r["request_id"] for r in rows} == set(byid)
        assert command["observed_blocks"] == command["expected_blocks"] == input_info["blocks"]
        assert all(raw["summary"]["invariants"].values())
        assert raw["adapter_statistics"]["reorder_calls"] == 0
        assert raw["adapter_statistics"]["cir_write_events"] == []
        assert raw["adapter_statistics"]["assignment_count"] == 0
        assert raw["input_fingerprint"] == meta["input_fingerprint"] == command["input_fingerprint"]
        assert raw["input_placement_fingerprint"] == raw["execution_placement_fingerprint"]
        audit_ownership(command, raw)
        assert len(raw["analysis"]) == 3
        bymetric = {r["request_id"]: r for r in rows}
        for npu in range(32):
            queue = [q.request_id for q in requests if q.npu_id == npu]
            ordered = sorted(queue, key=lambda rid: bymetric[rid]["admission_time_ms"])
            assert queue == ordered, ("NPU queue order changed", folder.name, npu)
        for row in rows:
            q = byid[row["request_id"]]
            ideal = 8 * q.load["per_layer_us"] / 1000
            duration = row["completion_time_ms"] - row["admission_time_ms"]
            assert math.isfinite(duration) and duration >= ideal - 1e-7
            close(ideal, row["own_compute_ms"], "request compute budget")
            ratio = duration / ideal
            passed = duration <= 1.5 * ideal + 1e-9
            assert passed == (ratio <= 1.5), ("SLO tolerance changes CDF threshold", folder.name, row)
            samples.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                                request_id=q.request_id, npu_id=q.npu_id, category=q.load["category"],
                                profile=q.load["role"], admission_ms=row["admission_time_ms"],
                                completion_ms=row["completion_time_ms"], latency_ms=duration,
                                pure_compute_ms=ideal, latency_ratio=ratio, slo_1p5_pass=passed,
                                in_warm_2_4s=2000 <= row["admission_time_ms"] < 4000,
                                in_long_2_6s=2000 <= row["admission_time_ms"] < 6000))
        for i, (window, measured) in enumerate(zip(WINDOWS, raw["analysis"])):
            left, right = measured["start_ms"], measured["end_ms"]
            assert (left, right) == ((2000, 4000) if i == 0 else (2000, 6000) if i == 1 else
                                     (0, raw["summary"]["makespan_ms"]))
            recomputed = summarize(raw["summary"], requests, left, right, full=i == 2)
            equivalent(recomputed, {key: measured[key] for key in recomputed}, f"{folder.name}.{window}")
            if i < 2:
                assert measured["all_npus_active"], (folder.name, window)
                close(measured["U_percent"], 100 * raw["windows"][i]["mean_npu_utilization"], "window U")
            else:
                close(measured["U_percent"], 100 * raw["summary"]["fleet_npu_compute_utilization"], "full U")
            flat = flatten(command, measured, window)
            tail = latency_tail(rows, left, right, full=i == 2)
            assert tail["latency_ratio_count"] == measured["slo"]["count"]
            flat.update(tail)
            flat.update(makespan_ms=raw["summary"]["makespan_ms"], case=folder.name,
                        manifest_sha256=command["manifest_sha256"], result_sha256=command["result_sha256"])
            complete.append(flat)
            tails.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                              window=window, slo_percent=measured["slo"]["percent"], **tail))
            for category in CLASSES:
                part = measured["slo_by_category"][category]
                categories.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                                       window=window, category=category, **part,
                                       active_U_percent=measured["category_active_U_percent"].get(category)))
            for profile, part in measured["slo_by_profile"].items():
                profiles.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                                     window=window, profile=profile, **part))
            for npu, value in enumerate(measured["per_npu_U_percent"]):
                npus.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                                 window=window, npu_id=npu, U_percent=value))
            for disk in range(3):
                actual = measured["SSD_GiB_s"][disk]
                assert -1e-8 <= actual <= 40 + 1e-7, (folder.name, disk, actual)
                close(actual / 40 * 100, measured["SSD_busy_percent"][disk], "SSD busy integral")
                if i == 0:
                    close(sum(raw["warm_ssd_10ms_GiB_s"][disk]) / 200, actual, "SSD bins integral")
                    close(sum(raw["warm_ssd_GiB_s_by_ssu_npu"][disk]), actual, "SSD per-card integral")
                if i == 2:
                    volume = math.fsum(size * 8 for q in requests for d, size in q.placement[0] if d == disk)
                    close(volume / (right / 1000), actual, "full SSD bytes conservation")
                disks.append(dict(scenario=command["scenario"], policy=command["policy"], seed=command["seed"],
                                  window=window, ssu_id=disk, capacity_GiB_s=40,
                                  mean_demand_GiB_s=measured["demand"]["per_disk_mean_GiB_s"][disk],
                                  min_demand_GiB_s=measured["demand"]["per_disk_min_GiB_s"][disk],
                                  max_demand_GiB_s=measured["demand"]["per_disk_max_GiB_s"][disk],
                                  overload_percent=measured["demand"]["per_disk_overload_percent"][disk],
                                  actual_GiB_s=actual, busy_percent=measured["SSD_busy_percent"][disk]))
        finished[key] = folder.name
        cases.append(dict(scenario=key[0], policy=key[1], seed=key[2], case=folder.name,
                          manifest_sha256=command["manifest_sha256"], result_sha256=command["result_sha256"],
                          requests=len(requests), blocks=command["observed_blocks"]))
    macros = macro_rows(complete)
    paired = paired_rows(complete)
    missing = [dict(scenario=s, policy=p, seed=k) for s, p, k in sorted(expected - set(finished))]
    state = "complete" if not missing else "preliminary"
    csv_write("comparison.csv", complete, ("scenario", "seed", "policy", "window"))
    csv_write("macro_summary.csv", macros, ("scenario", "policy", "window", "seed_count"))
    csv_write("paired_deltas.csv", paired, ("scenario", "policy", "reference", "window", "seed"))
    csv_write("warm_previews.csv", previews, ("scenario", "seed", "policy", "window"))
    csv_write("category_metrics.csv", categories, ("scenario", "seed", "policy", "window", "category"))
    csv_write("profile_slo.csv", profiles, ("scenario", "seed", "policy", "window", "profile"))
    csv_write("per_npu_metrics.csv", npus, ("scenario", "seed", "policy", "window", "npu_id"))
    csv_write("disk_metrics.csv", disks, ("scenario", "seed", "policy", "window", "ssu_id"))
    csv_write("request_samples.csv", samples, ("scenario", "seed", "policy", "request_id"))
    csv_write("latency_tail.csv", tails, ("scenario", "seed", "policy", "window"))
    tail_macros = [{key: value for key, value in row.items()
                   if key in ("scenario", "policy", "window", "seed_count", "seeds", "complete_seed_set", "mean_slo_percent") or
                   key.startswith(("mean_latency_ratio_", "min_latency_ratio_", "max_latency_ratio_"))}
                  for row in macros]
    csv_write("latency_tail_macro.csv", tail_macros, ("scenario", "policy", "window", "seed_count"))
    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items()), "source changed during report"
    checks = dict(status=state, complete_cases=len(finished), planned_cases=len(expected), missing_cases=missing,
                  warm_previews=len(previews), preview_rows_excluded_from_final_tables=True,
                  raw_metrics_recalculated=True, uncensored_admission_cohorts=True,
                  latency_tail_definition="nearest-rank: sorted_ratios[ceil(p*n)-1], no interpolation",
                  latency_tail_macro_definition="equal mean of per-seed quantiles/maxima, never pooled quantiles",
                  equal_seed_macro_mean=True, full_service_bytes_checked=True,
                  source_files_unchanged=True, source_sha256=source_hashes,
                  source_script_sha256=sha(Path(__file__)), cases=cases,
                  policies=list(POLICIES), scenarios=list(SCENARIOS), seeds=list(SEEDS),
                  placement="block_ring_hash", historical_stripe_results_not_used_as_control=True)
    (HERE / "summary_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    (HERE / "comparison.json").write_text(json.dumps(dict(status=state, rows=complete,
                                                          complete_cases=len(finished), planned_cases=len(expected)),
                                                     ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: checks[key] for key in ("status", "complete_cases", "planned_cases", "warm_previews")},
                     ensure_ascii=False))
    if args.require_complete and missing:
        raise SystemExit("Incomplete formal grid; tables are explicitly preliminary. No completed result claimed.")


if __name__ == "__main__":
    main()
