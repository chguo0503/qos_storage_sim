"""Independently audit and report the five-policy, 5-ms telemetry experiments.

The primary SLO is admission-to-completion <= 1.5 * ideal compute. Arrival
latency is a separate metric. Both use the same complete request population,
never each policy's differently selected window admissions. No simulator or
policy implementation is imported to calculate utilization or SLO results.
"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/shared_path_5ms_experiments"
CORE_FILES = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")
STRATEGIES = ("baseline", "once", "new_once", "strategy1", "strategy2")
DEVELOPMENT_CANDIDATES = ("compute", "fair_pipeline", "compute_guarded_mix",
                          "fair_pipeline_jit5", "fair_pipeline_jit10")
LABELS = {"baseline": "Baseline", "once": "Once/layer", "new_once": "New once",
          "strategy1": "Strategy 1", "strategy2": "Strategy 2 (ideal)"}
EPS_MS = 1e-7
SLO_EPS_MS = 1e-9


def overlap_ms(start, end, left, right):
    return max(0.0, min(end, right) - max(start, left))


def linear_percentile(values, probability):
    """Empirical percentile with linear interpolation at (n - 1) * p."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _same_numeric(actual, cached, label):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(cached), label
        for a, b in zip(actual, cached):
            _same_numeric(a, b, label)
    elif isinstance(actual, bool):
        assert cached is actual, label
    else:
        assert math.isfinite(float(actual)) and math.isfinite(float(cached)), label
        assert math.isclose(actual, cached, rel_tol=1e-10, abs_tol=EPS_MS), label


def recompute_metrics(summary, *, start_ms=1000.0, end_ms=2000.0, slo_alpha=1.5):
    """Compute from native event intervals, including startup/tail and all IDs."""
    npu_count, layers = summary["num_npu"], summary["n_layers"]
    requests = summary["request_metrics"]
    assert len(requests) == summary["request_count"], "incomplete request population"
    ids = [r["request_id"] for r in requests]
    assert len(ids) == len(set(ids)), "duplicate request ID"
    by_id = {r["request_id"]: r for r in requests}
    active, compute = [0.0] * npu_count, [0.0] * npu_count
    admitted_intervals = [[] for _ in range(npu_count)]
    compute_by_request, batch_ids = {}, []
    for batch in summary["microbatch_metrics"]:
        assert batch["batch_size"] == 1 and len(batch["member_request_ids"]) == 1, "batch_size must be one"
        rid, npu = batch["member_request_ids"][0], batch["npu_id"]
        request = by_id[rid]
        assert request["npu_id"] == npu, "request/batch NPU mismatch"
        begin, end = batch["admission_time_ms"], batch["completion_time_ms"]
        assert math.isfinite(begin) and math.isfinite(end) and end >= begin, "invalid batch interval"
        _same_numeric(begin, request["admission_time_ms"], "admission mismatch")
        _same_numeric(end, request["completion_time_ms"], "completion mismatch")
        admitted_intervals[npu].append((begin, end))
        active[npu] += overlap_ms(begin, end, start_ms, end_ms)
        assert len(batch["layer_metrics"]) == layers, "incomplete layer metrics"
        previous_end, own_compute = begin, 0.0
        for expected_layer, layer in enumerate(batch["layer_metrics"]):
            assert layer["layer"] == expected_layer, "layer identity mismatch"
            left, right = layer["compute_start_ms"], layer["compute_end_ms"]
            assert (math.isfinite(left) and math.isfinite(right)
                    and left >= previous_end - EPS_MS and right >= left
                    and right <= end + EPS_MS), "overlapping or invalid compute interval"
            _same_numeric(right - left, layer["compute_duration_ms"], "compute duration mismatch")
            _same_numeric(left - previous_end, layer["io_barrier_wait_ms"], "barrier accounting mismatch")
            compute[npu] += overlap_ms(left, right, start_ms, end_ms)
            own_compute += right - left
            previous_end = right
        _same_numeric(previous_end, end, "batch does not finish at last compute end")
        _same_numeric(own_compute, request["own_compute_ms"], "request ideal compute mismatch")
        compute_by_request[rid] = own_compute
        batch_ids.append(rid)
    assert sorted(batch_ids) == sorted(ids), "batch/request population mismatch"
    for intervals in admitted_intervals:
        intervals.sort()
        assert all(previous[1] <= following[0] + EPS_MS
                   for previous, following in zip(intervals, intervals[1:])), "overlapping admitted requests"
    duration = end_ms - start_ms
    assert duration > 0, "empty measurement window"
    assert all(-EPS_MS <= c <= a + EPS_MS and a <= duration + EPS_MS
               for a, c in zip(active, compute)), "window accounting exceeds physical time"
    records = []
    for request in sorted(requests, key=lambda r: r["request_id"]):
        rid = request["request_id"]
        ideal = request["own_compute_ms"]
        processing = request["completion_time_ms"] - request["admission_time_ms"]
        arrival_latency = request["completion_time_ms"] - request["arrival_time_ms"]
        assert ideal > 0 and arrival_latency >= processing - EPS_MS, "invalid request timing"
        threshold = slo_alpha * ideal
        records.append({"request_id": rid, "npu_id": request["npu_id"],
                        "arrival_ms": request["arrival_time_ms"], "admission_ms": request["admission_time_ms"],
                        "completion_ms": request["completion_time_ms"], "ideal_compute_ms": ideal,
                        "io_count": request["io_count"], "slo_threshold_ms": threshold,
                        "admission_latency_ms": processing, "arrival_latency_ms": arrival_latency,
                        "admission_passed": processing <= threshold + SLO_EPS_MS,
                        "arrival_passed": arrival_latency <= threshold + SLO_EPS_MS})
    count = len(records)
    admission_passed = sum(r["admission_passed"] for r in records)
    arrival_passed = sum(r["arrival_passed"] for r in records)
    makespan = summary["makespan_ms"]
    _same_numeric(makespan, max(r["completion_ms"] for r in records), "makespan mismatch")
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "mean_npu_utilization": sum(compute) / (npu_count * duration),
        "npu_utilizations": [value / duration for value in compute],
        "compute_ms_by_npu": compute, "active_ms_by_npu": active,
        "io_stall_ms_by_npu": [max(0.0, a - c) for a, c in zip(active, compute)],
        "idle_ms_by_npu": [max(0.0, duration - a) for a in active],
        "all_npus_active_whole_window": all(abs(a - duration) <= EPS_MS for a in active),
        "active_npu_count": sum(abs(a - duration) <= EPS_MS for a in active),
        "slo_alpha": slo_alpha, "request_count": count,
        "admission_slo_passed": admission_passed, "admission_slo_rate": admission_passed / count,
        "arrival_slo_passed": arrival_passed, "arrival_slo_rate": arrival_passed / count,
        "mean_admission_latency_ms": sum(r["admission_latency_ms"] for r in records) / count,
        "mean_arrival_latency_ms": sum(r["arrival_latency_ms"] for r in records) / count,
        "p99_arrival_latency_ms": linear_percentile([r["arrival_latency_ms"] for r in records], .99),
        "makespan_ms": makespan,
        "full_run_mean_npu_utilization": sum(compute_by_request.values()) / (npu_count * makespan),
        "records": records,
    }


def audit_adapter(statistics, summary, *, period_ms=5.0, min_cir_interval_ms=100.0):
    """Verify the measurable telemetry/CIR contract; do not infer hidden facts."""
    times = statistics["collector_times_ms"]
    assert times and abs(times[0]) <= EPS_MS, "collector must initialize at t=0"
    assert all(abs(b - a - period_ms) <= EPS_MS for a, b in zip(times, times[1:])), "collector is not periodic at 5 ms"
    assert times[-1] <= summary["makespan_ms"] + EPS_MS, "collector read after run completion"
    assert summary["makespan_ms"] - times[-1] <= period_ms + EPS_MS, "collector stopped early"
    fresh = statistics["fresh_reads_by_ssu"]
    cached = statistics["cache_reads_by_ssu"]
    assert len(fresh) == len(cached) == summary["num_ssu"], "telemetry topology mismatch"
    assert all(count == len(times) for count in fresh), "every tick must read every SSU once"
    _same_numeric(fresh, summary["pressure_reports_by_ssu"], "SSU fresh reads outside collector")
    assert 0 <= statistics["max_snapshot_age_ms"] <= period_ms + EPS_MS, "snapshot too old or from the future"
    writes = defaultdict(set)
    for event in statistics["cir_write_events"]:
        writes[event["ssu_id"]].add(event["time_ms"])
    for ssu, values in writes.items():
        ordered = sorted(values)
        assert all(b - a >= min_cir_interval_ms - EPS_MS for a, b in zip(ordered, ordered[1:])), ("CIR writes too frequent", ssu)
        assert all(abs(t) <= EPS_MS for t in ordered), "first version must retain static CIR at runtime"
    assert summary["cir_path_writes"] == 0, "unexpected runtime CIR writes in static experiment"
    if "reserved_blocks" in statistics:
        assert statistics["reserved_blocks"] == statistics["acknowledged_blocks"] == summary["completed_blocks"], "global ledger lost/doubled I/O"
        assert statistics["min_ledger_count"] >= 0, "negative global ledger"
        assert all(value == 0 for value in statistics["ledger_end_counts_by_ssu"]), "undrained final ledger"
    return {"collector_tick_count": len(times), "collector_period_ms": period_ms,
            "fresh_read_count": sum(fresh), "cached_read_count": sum(cached),
            "max_snapshot_age_ms": statistics["max_snapshot_age_ms"],
            "runtime_cir_path_writes": summary["cir_path_writes"],
            "contract_passed": True}


def manifest_fingerprint(requests):
    """Reproduce the documented native input-v2 digest without importing it."""
    digest = hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    for request in sorted(requests, key=lambda r: r["request_id"]):
        placement = tuple(tuple((int(ssu), float(volume)) for ssu, volume in layer)
                          for layer in request["placement"])
        digest.update(repr((request["request_id"], request["npu_id"], request["arrival_time_ms"],
                            request["load"]["category"], request["load"]["per_layer_us"], placement)).encode())
    return digest.hexdigest()


def audit_input_artifact(saved, data_dir, metrics):
    artifact = saved["input_artifact"]
    input_path = Path(data_dir) / artifact["path_relative_to_data"]
    raw = input_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == artifact["sha256"], "input artifact SHA mismatch"
    manifest = json.loads(raw)
    assert manifest_fingerprint(manifest["requests"]) == saved["input_fingerprint"] == manifest["input_fingerprint"], "input manifest fingerprint mismatch"
    logical = [{key: request[key] for key in ("request_id", "npu_id", "arrival_time_ms", "load")}
               for request in manifest["requests"]]
    logical_hash = hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert logical_hash == manifest["logical_input_fingerprint"] == saved["logical_input_fingerprint"], "logical trace fingerprint mismatch"
    rows = {row["request_id"]: row for row in manifest["requests"]}
    assert len(rows) == len(manifest["requests"]) == metrics["request_count"], "input artifact population mismatch"
    for actual in metrics["records"]:
        request = rows[actual["request_id"]]
        _same_numeric(request["arrival_time_ms"], actual["arrival_ms"], "input arrival changed")
        _same_numeric(request["load"]["per_layer_us"] * 8 / 1000, actual["ideal_compute_ms"], "input compute changed")
        expected_blocks = (8 * len(request["placement"][0]) if len(request["placement"]) == 1
                           else sum(len(layer) for layer in request["placement"]))
        assert actual["io_count"] == expected_blocks, "input I/O count changed"
        assert all(volume == 176 * 1024 / 2**30 for layer in request["placement"] for _, volume in layer), "input I/O is not one 176 KiB block"
        if saved["strategy"] not in ("strategy1", "strategy2"):
            assert request["npu_id"] == actual["npu_id"], "fixed-NPU input binding changed"
    for name, relative in saved["source_artifacts"].items():
        digest = hashlib.sha256((Path(data_dir) / relative).read_bytes()).hexdigest()
        assert digest == saved["core_and_policy_sha256"][name], ("archived source SHA mismatch", name)
    assert set(saved["source_artifacts"]) == set(saved["core_and_policy_sha256"]), "source archive closure incomplete"
    assert saved["core_and_policy_sha256"]["data"] == saved["metadata"]["source"]["source_sha256"], "data source hash does not match archived data"
    return {"input_artifact_sha256": artifact["sha256"],
            "input_manifest_fingerprint_recomputed": True,
            "archived_source_count": len(saved["source_artifacts"]),
            "archived_sources_verified": True}


def audit_assignment(saved, metrics):
    """Check observed assignment permissions and causal snapshot timestamps."""
    decisions = saved["assignment_log"]
    if saved["strategy"] not in ("strategy1", "strategy2"):
        assert not decisions, "fixed-NPU policy made assignment decisions"
        return {"assignment_decisions_verified": 0}
    actual = {r["request_id"]: r for r in metrics["records"]}
    assert len(decisions) == len(actual), "arrival assignment population mismatch"
    assert len({d["request_id"] for d in decisions}) == len(actual), "duplicate arrival assignment"
    for decision in decisions:
        record = actual[decision["request_id"]]
        _same_numeric(decision["arrival_time_ms"], record["arrival_ms"], "assignment was not at arrival")
        assert decision["assigned_npu_id"] == record["npu_id"], "binding changed after assignment"
        age = decision["arrival_time_ms"] - decision["snapshot_time_ms"]
        assert -EPS_MS <= age <= 5 + EPS_MS, "assignment used future/stale snapshot"
    return {"assignment_decisions_verified": len(decisions)}


def audit_jit(statistics, summary, strategy):
    """Verify final S1/S2 JIT logs, without treating sparse logs as full traces."""
    jit = statistics.get("jit_prefetch")
    if strategy not in ("strategy1", "strategy2"):
        assert not jit or not jit["enabled"], "fixed-emission policy unexpectedly enabled JIT"
        return {"jit_activation_count": 0, "jit_delayed_layer_count": 0,
                "jit_sparse_examples_verified": 0}
    assert jit and jit["enabled"], "final Strategy 1/2 must include JIT"
    assert jit["guard_ms"] == 10, "final JIT guard differs from frozen 10 ms"
    assert jit["activation_count"] == summary["request_count"] * summary["n_layers"], "JIT activation population mismatch"
    assert not jit["modifies_already_queued_io"], "JIT must not modify queued I/O"
    assert 0 <= jit["delay_count"] <= jit["activation_count"], "invalid JIT count"
    assert 0 <= jit["max_delay_ms"] <= jit["total_delay_ms"] + EPS_MS, "invalid JIT total delay"
    assert math.isfinite(jit["decision_wall_us"]) and jit["decision_wall_us"] >= 0, "invalid JIT control cost"
    assert len(jit["release_examples"]) == min(128, jit["delay_count"]), "JIT example population mismatch"
    for example in jit["release_examples"]:
        activation, release = example["activation_ms"], example["release_ms"]
        assert 0 <= activation - example["snapshot_time_ms"] <= 5 + EPS_MS, "JIT used stale/future snapshot"
        _same_numeric(release - activation, example["delay_ms"], "JIT release/delay mismatch")
        counts, pending = example["layer_io_by_ssu"], example["cached_ssu_queue_io"]
        assert len(counts) == len(pending) == summary["num_ssu"], "JIT topology mismatch"
        estimate = 1000 * (176 * 1024 / 2**30) * max(
            max((q + k) / 40 for q, k in zip(pending, counts)), sum(counts) / 50)
        predicted_release = max(activation, example["deadline_ms"] - estimate - 10)
        _same_numeric(release, predicted_release, "JIT formula mismatch")
    return {"jit_activation_count": jit["activation_count"],
            "jit_delayed_layer_count": jit["delay_count"],
            "jit_sparse_examples_verified": len(jit["release_examples"])}


def _input_identity(metrics):
    # NPU binding is intentionally excluded: only strategy1/2 may change it.
    return tuple((r["request_id"], r["arrival_ms"], r["ideal_compute_ms"], r["io_count"])
                 for r in metrics["records"])


def collect_results(data_dir, *, core_root=ROOT, expected_npu=32, expected_requests=704):
    selected, rows, groups = {}, [], defaultdict(list)
    core_hash = {name: hashlib.sha256((core_root / name).read_bytes()).hexdigest() for name in CORE_FILES}
    versions = defaultdict(set)
    for path in sorted(Path(data_dir).glob("*.json")):
        saved = json.loads(path.read_text())
        if not isinstance(saved, dict) or "strategy" not in saved or "summary" not in saved:
            continue
        strategy = saved["strategy"]
        assert strategy in STRATEGIES, (path, "unknown strategy", strategy)
        meta, summary = saved["metadata"], saved["summary"]
        assert all(summary["invariants"].values()), (path, "native invariant failure")
        assert meta["num_npu"] == summary["num_npu"] == expected_npu, (path, "NPU count mismatch")
        assert meta["num_ssu"] == summary["num_ssu"] in (5, 6, 7), (path, "SSU count mismatch")
        assert meta["request_count"] == summary["request_count"] == expected_requests, (path, "population mismatch")
        assert summary["n_layers"] == 8 and summary["batch_size"] == 1, (path, "inference model mismatch")
        assert saved["submit_seed"] == meta["seed"], (path, "submission seed mismatch")
        for name, digest in core_hash.items():
            assert saved["core_and_policy_sha256"][name] == digest, (path, "core source mismatch", name)
        for name, digest in saved["core_and_policy_sha256"].items():
            versions[name].add(digest)
        metrics = recompute_metrics(summary)
        cached = saved["common_window"]
        for field in ("start_ms", "end_ms", "mean_npu_utilization", "npu_utilizations",
                      "active_ms_by_npu", "io_stall_ms_by_npu", "idle_ms_by_npu",
                      "all_npus_active_whole_window"):
            _same_numeric(metrics[field], cached[field], (path, field))
        if "slo" in saved:
            _same_numeric(1.5, saved["slo"]["alpha"], (path, "SLO alpha"))
            for prefix in ("admission", "arrival"):
                reported = saved["slo"]["all_requests"][prefix]
                _same_numeric(metrics["request_count"], reported["count"], (path, prefix, "count"))
                _same_numeric(metrics[prefix + "_slo_passed"], reported["passed"], (path, prefix, "passed"))
                _same_numeric(metrics[prefix + "_slo_rate"], reported["rate"], (path, prefix, "rate"))
        telemetry = audit_adapter(saved["adapter_statistics"], summary)
        telemetry.update(audit_jit(saved["adapter_statistics"], summary, strategy))
        provenance = audit_input_artifact(saved, data_dir, metrics)
        provenance.update(audit_assignment(saved, metrics))
        key = meta["num_ssu"], meta.get("regime", "near"), meta["seed"], strategy
        assert key not in selected, (path, "duplicate experimental case", key)
        selected[key] = {"saved": saved, "metrics": metrics, "telemetry": telemetry,
                         "provenance": provenance, "file": path}
        groups[key[:3]].append(selected[key])
        rows.append({"file": path.name, "num_ssu": key[0], "regime": key[1], "seed": key[2],
                     "strategy": strategy, "input_fingerprint": saved["input_fingerprint"],
                     **{k: v for k, v in metrics.items() if k != "records"}, **telemetry, **provenance})
    for key, cases in groups.items():
        assert len({c["saved"]["input_fingerprint"] for c in cases}) == 1, (key, "different input fingerprints")
        assert len({c["saved"]["metadata"]["source"]["source_sha256"] for c in cases}) == 1, (key, "different data source hashes")
        assert len({c["provenance"]["input_artifact_sha256"] for c in cases}) == 1, (key, "different archived input manifests")
        assert len({tuple(c["saved"]["static_path_cirs_gib_s"]) for c in cases}) == 1, (key, "different static CIR tables")
        original = cases[0]["metrics"]
        identity = _input_identity(original)
        for case in cases[1:]:
            current = _input_identity(case["metrics"])
            assert len(identity) == len(current), (key, "different request populations")
            for left, right in zip(identity, current):
                assert left[0] == right[0] and left[3] == right[3], (key, "request ID/I/O identity changed")
                _same_numeric(left[1:3], right[1:3], (key, "arrival/compute identity changed"))
        fixed = [c for c in cases if c["saved"]["strategy"] in ("baseline", "once", "new_once")]
        if fixed:
            binding = [(r["request_id"], r["npu_id"]) for r in fixed[0]["metrics"]["records"]]
            assert all([(r["request_id"], r["npu_id"]) for r in c["metrics"]["records"]] == binding
                       for c in fixed[1:]), (key, "fixed-NPU strategy changed placement")
    same_trace_groups = defaultdict(set)
    for key, case in selected.items():
        if key[1] == "same_trace":
            same_trace_groups[key[2]].add(case["saved"]["logical_input_fingerprint"])
    assert all(len(hashes) == 1 for hashes in same_trace_groups.values()), "same_trace changed logical requests across SSU topologies"
    implementation_versions = []
    for name, digests in sorted(versions.items()):
        current_file = core_root / name
        current_hash = hashlib.sha256(current_file.read_bytes()).hexdigest() if current_file.is_file() else None
        implementation_versions.append({"file": name, "saved_sha256": sorted(digests),
                                        "current_sha256": current_hash,
                                        "all_match_current": len(digests) == 1 and current_hash in digests})
    audit = {"run_count": len(rows), "window_ms": [1000, 2000], "slo_alpha": 1.5,
             "unique_physical_input_fingerprints": len({c["saved"]["input_fingerprint"] for c in selected.values()}),
             "primary_slo_clock": "admission_to_completion", "secondary_slo_clock": "arrival_to_completion",
             "same_complete_request_population_verified": True,
             "same_input_fingerprint_source_and_seed_verified": True,
             "complete_input_artifacts_and_source_archives_verified": True,
             "cached_window_recomputed": True, "core_sha256": core_hash,
             "implementation_versions": implementation_versions,
             "all_npus_active_every_primary_window": all(r["all_npus_active_whole_window"] for r in rows),
             "records": rows}
    return rows, selected, audit


def collect_development(data_dir, selected):
    """Audit the five seed-06 probes separately; never promote them to runs."""
    rows = []
    for path in sorted((Path(data_dir) / "development").glob("*/*.json")):
        saved = json.loads(path.read_text())
        if "summary" not in saved:
            continue
        candidate = saved["candidate_mode"]
        assert candidate in DEVELOPMENT_CANDIDATES and saved["development_only"], "unknown/unlabeled development candidate"
        assert saved["strategy"] == "candidate_" + candidate and saved["base_strategy"] == "strategy1", "development result mislabeled as final policy"
        meta = saved["metadata"]
        assert (meta["num_npu"], meta["num_ssu"], meta["seed"], meta["regime"]) == (32, 6, 20260906, "near"), "development selection scope changed"
        assert saved["development_seed"] == 20260906, "holdout seed used for candidate selection"
        metrics = recompute_metrics(saved["summary"])
        assert all(saved["summary"]["invariants"].values()), "development native invariant failure"
        assert metrics["request_count"] == 704, "development request population mismatch"
        _same_numeric(metrics["mean_npu_utilization"], saved["common_window"]["mean_npu_utilization"], "development utilization mismatch")
        telemetry = audit_adapter(saved["adapter_statistics"], saved["summary"])
        normalized = {**saved, "strategy": "strategy1"}
        provenance = audit_input_artifact(normalized, path.parent, metrics)
        provenance.update(audit_assignment(normalized, metrics))
        reference = selected.get((6, "near", 20260906, "baseline"))
        if reference:
            assert saved["input_fingerprint"] == reference["saved"]["input_fingerprint"], "candidate changed development input"
            assert all(saved["core_and_policy_sha256"][name] == reference["saved"]["core_and_policy_sha256"][name]
                       for name in CORE_FILES), "candidate changed native core"
            for a, b in zip(_input_identity(metrics), _input_identity(reference["metrics"])):
                assert a[0] == b[0] and a[3] == b[3], "candidate changed request identities"
                _same_numeric(a[1:3], b[1:3], "candidate changed arrivals/compute")
        for clock in ("admission", "arrival"):
            _same_numeric(metrics[clock + "_slo_passed"], saved["slo"]["all_requests"][clock]["passed"], "candidate SLO mismatch")
        final = selected.get((6, "near", 20260906, "strategy1"))
        if candidate == "fair_pipeline_jit10" and final:
            for field in ("mean_npu_utilization", "npu_utilizations", "admission_slo_passed",
                          "arrival_slo_passed", "makespan_ms"):
                _same_numeric(metrics[field], final["metrics"][field], ("frozen Strategy 1 differs from selected development candidate", field))
        jit = saved["candidate_stats"]["jit_prefetch"]
        assert not jit["modifies_already_queued_io"], "candidate illegally modifies queued I/O"
        for example in jit["release_examples"]:
            assert example["release_ms"] >= example["activation_ms"] - EPS_MS, "JIT release precedes activation"
            _same_numeric(example["release_ms"] - example["activation_ms"], example["delay_ms"], "JIT delay accounting mismatch")
        rows.append({"candidate": candidate, "file": str(path), "input_fingerprint": saved["input_fingerprint"],
                     **{key: value for key, value in metrics.items() if key != "records"},
                     "jit_delay_count": jit["delay_count"], "jit_guard_ms": jit["guard_ms"],
                     "provenance": provenance, "telemetry": telemetry})
    assert len({row["candidate"] for row in rows}) == len(rows), "duplicate development candidate"
    return rows


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                      + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def make_figure2(selected, figure_dir, *, seed=20260906, regime="near"):
    """Three panels, five fixed policy rows, 32 unsorted physical card columns."""
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrices = [[[100 * u for u in selected[(s, regime, seed, policy)]["metrics"]["npu_utilizations"]]
                 for policy in STRATEGIES] for s in (5, 6, 7)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.35), constrained_layout=True)
    for axis, ssu, matrix in zip(axes, (5, 6, 7), matrices):
        shown = axis.imshow(matrix, vmin=0, vmax=100, cmap="cividis", aspect="auto")
        for row, values in enumerate(matrix):
            for npu, value in enumerate(values):
                axis.text(npu, row, f"{value:.0f}", ha="center", va="center", fontsize=8,
                          color="white" if value < 55 else "black")
        axis.set_xticks(range(32), range(32), fontsize=8)
        axis.set_yticks(range(5), [LABELS[p] for p in STRATEGIES], fontsize=8)
        axis.set_title(f"32 NPU / {ssu} SSU; fixed [1000, 2000] ms; per-card compute %", fontsize=10)
        axis.set_xlabel("Physical NPU ID; same order in every row", fontsize=8)
    fig.colorbar(shown, ax=axes, shrink=.75, label="Compute utilization (%)")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "figure2_per_npu_utilization.pdf")
    fig.savefig(figure_dir / "figure2_per_npu_utilization.png", dpi=200)
    plt.close(fig)


def make_policy_figure(figure_dir):
    """Code-native vector diagram of the common observation and policy family."""
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, axis = plt.subplots(figsize=(11, 3.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3.4)
    axis.axis("off")
    def box(x, y, width, height, text, color):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.04",
                                     facecolor=color, edgecolor="#222222", linewidth=.8))
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11.5)
    box(2.8, 2.65, 4.4, .5, "SSU collector: t = 0, 5, 10, ... ms", "#DCEAF4")
    box(2.8, 1.8, 4.4, .5, "Shared immutable telemetry copy; no read-on-miss", "#E7EDF0")
    axis.annotate("", (5, 2.3), (5, 2.65), arrowprops={"arrowstyle": "->"})
    names = ("Baseline\nPath0", "Once/layer\noriginal routing", "New once\nshared host ledger",
             "Strategy 1\n+ placement + JIT", "Strategy 2\n+ same-Path reorder")
    for i, name in enumerate(names):
        x = .15 + 2 * i
        box(x, .45, 1.7, .8, name, "#E7EDF0" if i < 4 else "#F5DDBB")
        axis.annotate("", (x + .85, 1.25), (5, 1.8),
                      arrowprops={"arrowstyle": "->", "color": "#666666", "lw": .65})
    axis.text(5, .12, "All: static CIR; one 50 GiB/s receive link per NPU. Strategy 2 additionally needs SSD-side support.",
              ha="center", va="center", fontsize=10)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "figure1_policy_information.pdf")
    fig.savefig(figure_dir / "figure1_policy_information.png", dpi=180)
    plt.close(fig)


def build_document(selected, audit, *, seed=20260906, regime="near", development=()):
    primary, latency, costs, profiles, timings, decisions, jit_rows, demand_rows = [], [], [], [], [], [], [], []
    for ssu in (5, 6, 7):
        baseline = selected[(ssu, regime, seed, "baseline")]["saved"]
        demand = baseline["metadata"]["input_demand"]
        demand_rows.append([ssu, f'{demand["total_gib_s"]:.3f}', ssu * 40,
                            f'{max(demand["per_ssu_gib_s"]):.3f}',
                            f'{100*demand["aggregate_load_ratio"]:.2f}'])
        for profile in baseline["metadata"]["profiles"]:
            profiles.append([ssu, profile["seq_len_k"], profile["nql"], profile["kv_blocks"],
                             f'{profile["compute_ms"]:.6f}', profile["quota"]])
        for policy in STRATEGIES:
            case = selected[(ssu, regime, seed, policy)]
            m, statistics = case["metrics"], case["saved"]["adapter_statistics"]
            primary.append([ssu, LABELS[policy], f'{100*m["mean_npu_utilization"]:.3f}',
                            f'{m["admission_slo_passed"]}/{m["request_count"]}',
                            f'{100*m["admission_slo_rate"]:.2f}', f'{m["active_npu_count"]}/32'])
            latency.append([ssu, LABELS[policy], f'{m["mean_admission_latency_ms"]:.2f}',
                            f'{m["mean_arrival_latency_ms"]:.2f}', f'{m["p99_arrival_latency_ms"]:.2f}',
                            f'{100*m["arrival_slo_rate"]:.2f}',
                            f'{m["makespan_ms"]:.2f}'])
            costs.append([ssu, LABELS[policy], case["telemetry"]["collector_tick_count"],
                          case["telemetry"]["fresh_read_count"], case["telemetry"]["cached_read_count"],
                          f'{case["telemetry"]["max_snapshot_age_ms"]:.3f}',
                          case["telemetry"]["runtime_cir_path_writes"]])
            timings.append([ssu, LABELS[policy],
                            f'{statistics["collector_wall_us"]/1e6:.3f}',
                            f'{statistics["routing_wall_us"]/1e6:.3f}',
                            f'{statistics["assignment_wall_us"]/1e6:.3f}',
                            f'{statistics["reorder_wall_us"]/1e6:.3f}',
                            f'{statistics["ack_ledger_wall_us"]/1e6:.3f}'])
            mean_route = statistics["routing_wall_us"] / max(1, statistics["routing_calls"]) / 1000
            mean_assignment = statistics["assignment_wall_us"] / max(1, statistics["assignment_count"]) / 1000
            mean_reorder = statistics["reorder_wall_us"] / max(1, statistics["reorder_calls"])
            decisions.append([ssu, LABELS[policy], statistics["routing_calls"],
                              f'{mean_route:.3f}',
                              f'{mean_assignment:.3f}' if statistics["assignment_count"] else "—",
                              f'{mean_reorder:.3f}' if statistics["reorder_calls"] else "—",
                              f'{m["makespan_ms"]/1000:.3f}'])
            if policy in ("strategy1", "strategy2"):
                jit = statistics["jit_prefetch"]
                jit_rows.append([ssu, LABELS[policy], jit["activation_count"], jit["delay_count"],
                                 f'{jit["max_delay_ms"]:.3f}',
                                 f'{jit["decision_wall_us"]/1e6:.3f}',
                                 f'{jit["decision_wall_us"]/jit["activation_count"]:.3f}'])
    warnings = [row for row in audit["implementation_versions"] if not row["all_match_current"]]
    version_note = ("所有已保存实现哈希与当前文件一致。" if not warnings else
                    "存在历史实现版本与当前文件不一致；具体SHA保存在audit，需对应版本源码复现，不能只运行当前文件就宣称逐位一致。")
    development_table = markdown_table(["开发候选", "U/%", "接纳SLO/704", "用户SLO/704", "延迟发射层数"],
        [[row["candidate"].replace("_", " "), f'{100*row["mean_npu_utilization"]:.3f}',
          row["admission_slo_passed"], row["arrival_slo_passed"], row["jit_delay_count"]]
         for row in development]) if development else "尚未加载开发结果；不填造数字。"
    generalization_rows = []
    for (ssu, group, group_seed, policy), case in sorted(selected.items()):
        if (group, group_seed) == (regime, seed):
            continue
        m = case["metrics"]
        generalization_rows.append([ssu, group.replace("_", " "), str(group_seed)[-2:], LABELS[policy],
                                    f'{100*m["mean_npu_utilization"]:.3f}',
                                    f'{m["admission_slo_passed"]}/704', f'{m["arrival_slo_passed"]}/704'])
    generalization = markdown_table(["SSU", "输入组", "种子尾数", "策略", "U/%", "接纳SLO", "用户SLO"], generalization_rows)
    failure = ""
    failed_once = selected.get((5, "same_trace", 20260906, "once"))
    failed_new = selected.get((5, "same_trace", 20260906, "new_once"))
    if failed_once and failed_new:
        old, new = failed_once["metrics"], failed_new["metrics"]
        failure = (f'一个明确反例是5盘same_trace：Once接纳SLO通过{old["admission_slo_passed"]}/704，'
                   f'New once为{new["admission_slo_passed"]}/704；U却从{100*old["mean_npu_utilization"]:.3f}%'
                   f'变为{100*new["mean_npu_utilization"]:.3f}%。'
                   '因此“全局账本更准确”不推出“每个指标都更好”，U和单请求deadline还存在目标差异。')
    main_cases = [selected[(ssu, regime, seed, policy)]["metrics"] for ssu in (5, 6, 7) for policy in STRATEGIES]
    arrival_warning = (f'主矩阵完整用户到达SLO只有{min(m["arrival_slo_passed"] for m in main_cases)}'
                       f'至{max(m["arrival_slo_passed"] for m in main_cases)}/704通过。'
                       '接纳后高通过率不能解读为端到端用户体验达标；初始及后续接纳排队没有被主SLO计算进去。')
    near_baselines = [100 * case["metrics"]["mean_npu_utilization"]
                      for key, case in selected.items() if key[1] == "near" and key[3] == "baseline"]
    scope_outcome = (f'首先承认目标差距：本矩阵near组Baseline平均U已经是{min(near_baselines):.2f}%'
                     f'至{max(near_baselines):.2f}%，不能称为“baseline很糟糕”。'
                     '这组实验尚未兑现“极大平均提升”的原始愿望；应据实比较百分点改善、单请求SLO和完整批次代价。')
    heldout_pairs = [(selected[(s, "near", 20260907, "strategy1")]["metrics"],
                     selected[(s, "near", 20260907, "once")]["metrics"])
                    for s in (5, 6, 7) if (s, "near", 20260907, "strategy1") in selected
                    and (s, "near", 20260907, "once") in selected]
    heldout_outcome = ""
    if heldout_pairs:
        degraded = sum(new["mean_npu_utilization"] < old["mean_npu_utilization"] - EPS_MS for new, old in heldout_pairs)
        heldout_outcome = (f'冻结后的S1在未参与选参的seed07上，已完成的{len(heldout_pairs)}个拓扑中有{degraded}个'
                           '固定秒U低于Once。开发组的收益没有形成稳定泛化，当前S1不能称为Once的可靠升级。'
                           if degraded else '冻结后的seed07结果另列；有限输入上的成功不构成普遍保证。')
    development_tradeoff = ""
    chosen = next((row for row in development if row["candidate"] == "fair_pipeline_jit10"), None)
    developer_once = selected.get((6, "near", 20260906, "once"))
    if chosen and developer_once:
        old = developer_once["metrics"]
        development_tradeoff = (f'选中开发候选相对Once：固定秒U从{100*old["mean_npu_utilization"]:.3f}%'
                                f'变为{100*chosen["mean_npu_utilization"]:.3f}%，'
                                f'但整批完工从{old["makespan_ms"]:.3f} ms变为{chosen["makespan_ms"]:.3f} ms，'
                                f'整批U从{100*old["full_run_mean_npu_utilization"]:.3f}%'
                                f'变为{100*chosen["full_run_mean_npu_utilization"]:.3f}%；'
                                f'用户到达SLO从{old["arrival_slo_passed"]}/704变为{chosen["arrival_slo_passed"]}/704。'
                                '所以固定一秒改善并不意味着整批更快或用户SLO更好。')
    overload = ""
    five_same = selected.get((5, "same_trace", 20260906, "baseline"))
    if five_same:
        demand = five_same["saved"]["metadata"]["input_demand"]
        overload = (f'5盘same_trace另有容量限制：完整配比名义总需求{demand["total_gib_s"]:.3f} GiB/s'
                    f'超过200 GiB/s，最热盘{max(demand["per_ssu_gib_s"]):.3f} GiB/s超过40。'
                    '它不能称为平均容量可行的near场景；此组保留为过载反例，不能用它证明只靠排队优化就能全部满算。')
    baseline_six = selected.get((6, "near", 20260906, "baseline"))
    worked_example, weighting = "", ""
    if baseline_six:
        six_profiles = baseline_six["saved"]["metadata"]["profiles"]
        short = [p for p in six_profiles if p["seq_len_k"] == 32]
        short_count = 32 * sum(p["quota"] for p in short)
        short_compute_share = sum(p["quota"] * p["compute_ms"] for p in short) / sum(p["quota"] * p["compute_ms"] for p in six_profiles)
        weighting = (f'以6盘near06为例，32K的两个短画像占{short_count}/704'
                     f'（{100*short_count/704:.2f}%）请求，却仅占纯计算毫秒的{100*short_compute_share:.4f}%。'
                     'SLO按请求等权计数，U按计算时间加权；两者的改善幅度不能直接对应。')
        batch = next(b for b in baseline_six["saved"]["summary"]["microbatch_metrics"]
                     if b["member_request_ids"] == [24000007])
        layer_rows = batch["layer_metrics"]
        worked_example = ("再看6盘near06的实际Baseline：NPU24的request_id=24000007，32K/NQL256，"
                          "每层254块（43.65625 MiB），C=1.997480 ms。L1至L7的已保存记录如下；"
                          "deadline取前一层compute结束，ready表示所有盘数据已经经接收链路到HBM，不是某盘刚完成服务。\n\n" +
                          markdown_table(["层", "预取激活/ms", "数据deadline/ms", "到HBM齐/ms", "stall/ms"],
                              [[row["layer"], f'{row["io_start_time_ms"]:.6f}',
                                f'{layer_rows[row["layer"]-1]["compute_end_ms"]:.6f}',
                                f'{row["io_ready_time_ms"]:.6f}', f'{row["io_barrier_wait_ms"]:.6f}']
                               for row in layer_rows[1:]]) +
                          "\n\nL1从1526.636445激活到1545.587637到齐，耗时18.951193 ms，"
                          "不能藏在1.997480 ms计算内，所以等待16.953713 ms。随后发起时刻已经被各次stall推迟，"
                          "七个后续层仍全部迟到。错开改变相位，但没有给这个短计算请求预留服务；"
                          "它并不是证明无限层永远不能错开。仅凭这份层级trace也不能点名某一个确切的FIFO前驱。")
    return r'''---
title: "共享 Path、5 ms 遥测：32 NPU 与 5/6/7 SSU"
subtitle: "五种策略、固定一秒利用率、完整同批请求的 1.5× SLO"
date: "2026-09-06"
documentclass: article
fontsize: 10pt
geometry: a4paper,margin=18mm
CJKmainfont: "Noto Sans CJK JP"
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
toc: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{pdflscape}
  - \setlength{\emergencystretch}{3em}
---

# 先明确比较口径

**@SCOPE_OUTCOME@**

**@HELDOUT_OUTCOME@**

这是新的一组受限控制实验，不覆盖旧版“即时遥测、独占 Path、频繁 CIR 更新”的报告。
Baseline、Once、New once、Strategy 1、Strategy 2 都由同一个全局 collector 在0 ms初始化、之后每5 ms采集每盘一次。
路由决策只能读采集副本；读副本不等于又读了一次SSU。首版保持静态CIR，运行期CIR写入为0；这满足“不高于每100 ms一次写入”的限制，但不是已经实验了100 ms动态控制。

主SLO严格定义为：

$$t_{\rm complete}-t_{\rm admission}\le1.5\times T_{\rm ideal},\qquad T_{\rm ideal}=8C.$$

即“从NPU正式接纳开始，到八层全部完成”不得超过纯计算时间的1.5倍。
另行报告从用户到达开始的延迟；它还包含接纳前排队。不能将两个SLO口径互换。
全部策略用同一完整704请求集作为分母，按相同request_id逐一核验，而不是每种策略各自取窗口内接纳的不同请求。

**@ARRIVAL_WARNING@**

![共同的5 ms遥测副本与五种策略。Baseline仍采集但不使用压力选Path；New once额外使用主机实时共享账本；Strategy 2另需盘内对已入队命令的局部可见性。额外信息权限必须分开说明。](../figures/figure1_policy_information.pdf){width=100%}

# 主结果：同一秒与相同 SLO 样本

@PRIMARY@

U是所有32张卡在固定 $[1000,2000]$ ms 内的计算时间比例：

$$U=\frac{\sum_i |\text{该卡计算区间}\cap[1000,2000]|}{32\times1000\ {\rm ms}}.$$

active列表示整秒都有已接纳工作、没有入口空闲的卡数。active不足32时，利用率差值不能全部解释为IO stall。
数据由逐层compute区间独立重算，并校验active=compute+IO等待、1000=active+idle；没有移动窗口选择最优读数。
S1/S2可改变新请求分卡，所以热力图同一物理NPU列不代表相同请求逐一对照；严格比较请求时仍须按request_id配对。

```{=latex}
\clearpage
\begin{landscape}
```

![三个拓扑面板，每面板五种策略，横轴都是原物理NPU编号0–31，不逐行排序。颜色固定0–100%，格内为计算占比的四舍五入整数；精确值保存在audit JSON。它不等同某一个请求的计算占比。](../figures/figure2_per_npu_utilization.pdf){width=255mm}

```{=latex}
\end{landscape}
\clearpage
```

# 到底输入了什么？

每条I/O是一块GLM每层KV：128 token对应176 KiB，即180224 byte。
每盘后端40 GiB/s，每张NPU只有一个50 GiB/s接收链路，多盘返回在这条链路汇聚；数据载荷SSD→HBM，不经DRAM。
每盘256条共享Path采用同一静态配置：四类CIR总预算为20/6/8/6 GiB/s，跨8组分配；每组四类Path数量为12/4/12/4。PIR均不设额外上限，但整盘仍受40 GiB/s限制。
CIR是保底而非硬带宽上限：Baseline只有Path0有流量时仍可使用整盘能力，并不是被限在Path0很小的CIR上。Once系列保留原请求分类允许的Path集合，不随意跨类别借用编号。
每请求八层，batch=1，启用已有请求的跨请求L0预取。后继只有在前驱末层开始时已经到达才能预取，不补造迟到预取。
request_id只是请求的唯一编号，不是NPU编号、token数量或层号；同一请求的layer编号为0至7，NPU编号为0至31。arrival是用户到达，admission是该NPU正式开始处理，二者之间可能排队。

@PROFILES@

以上参数取自项目data；配额、每卡请求顺序及到达规律是实验构造，不是线上用户分布的实测频率。
同一拓扑内所有策略必须保持相同请求ID、arrival、每请求计算时长、I/O总量及原SSD放置；Strategy 1/2只允许在实际到达时改变NPU绑定。

完整输入的平均需求须按时间加权，不能简单平均每请求的V/C。对于固定原始NPU分配，

$$d_{is}=1000\frac{\sum_r V_{r,is}}{\sum_r8C_r},\quad d_s=\sum_i d_{is},$$

其中V包含该请求全部八层的GiB，C以ms计。若要求各卡连续满算并以该完整配比长期重复运行，必要平均约束为每盘 $d_s\le40$、每卡接收 $\sum_s d_{is}\le50$。
这不是存在接纳空闲时“单个请求零stall”的必要条件，也不是某一有限秒内计算利用率的硬上界，更不是选卡之后的实时需求事实。

@DEMAND@

near按最热盘不过载调整配额，因此5/6/7盘之间未必是同一逻辑请求trace；不能将跨拓扑差值全归因于增加SSD。same_trace组才保持跨拓扑相同逻辑请求，只重新计算落盘位置。两种组内，五策略始终共享同一输入。

# 五种策略究竟改变什么？

| 策略 | 允许的改变 | 没有授权的改变 |
|---|---|---|
| Baseline | 固定Path0 | 不选卡/重排 |
| Once | 原Once路由 + 5 ms副本 | 不实时读盘/选卡 |
| New once | 原路由 + 全局Path承诺账本 | 不改CIR/选卡 |
| Strategy 1 | New once + 到达选卡 + JIT | 不迁移旧请求/改到达 |
| Strategy 2 | S1 + 同Path pending重排 | 不跨Path/抢占/看未来 |

Strategy 2需要设备配合或修改盘内调度，是受限理想参照，不是仅靠客户端即可直接部署的策略。
Strategy 1/2选卡函数相同，但已经执行的状态可能不同，所以后续实际绑定仍可能不同；不能将最终利用率差异全部归因于某一条命令在队内移动了几位。

New once使用明确的封闭前提：这些SSU的**全部流量均由同一个协同客户端系统管理**。对每盘s、每Path p维护：

$$L_{sp}=\text{已经承诺到该Path的IO总数}-\text{已到HBM并确认完成的IO总数}.$$

L包括“已规划但未下发”及“已下发但未到HBM”两部分，5 ms采集不会清空它。
这里“已规划”特指已经执行Path规划、承诺到某条Path：尚未调用规划函数的未来层，以及被JIT暂扣但尚未规划的coflow，都不进入L。不能把它等同客户端全部已知未来IO。
在所有流量受管理且账本正确同步时，$L_{sp}$不少于当前SSD该Path的pending+active数量；但它不是精确SSD队列，因为还包含未发射及离盘但未到HBM的IO。
每次Once规划返回的所有Path立即原子记账，下一个NPU的规划可以看见这些承诺，避免多个客户端对同一个旧空队列副本同时下注。

**当前New once直接用L重建压力并继承原Once逐IO选择引擎；没有把5 ms快照数量再加上去。** 否则旧快照里的已完成IO会被重复保留。
因此收益来自新的全局客户端信息与协调，不是5 ms旧副本本身变得实时；若有未受管理的外部流量，L不再是全部盘负载的上界，需要另行扩展模型。

Strategy 1的数据面Path路由仍为New once，CIR不变。它还同时改变两件事：新请求到达时选NPU，以及尚未提交整层的发射时机。不能把它与New once的差值全部归因于选卡。
最终选卡公式为fair pipeline。假设将新请求放到候选卡j后，$R_j$为该卡所有已到达请求的剩余计算ms，$K_{js}$为这些已知请求在盘s的未完成IO数（含尚未发射的后续层），$A_s$为在盘s仍有这类工作量的NPU数量，则

$$\operatorname{score}(j)=now+\max\left(R_j,\ 1000\frac{b_G\sum_sK_{js}}{50},\ 1000\max_s\frac{b_G A_s K_{js}}{40}\right).$$

其中$b_G=176/1048576\ {\rm GiB}$是一个IO的大小。选择score最小的卡；平局依次比较未完成请求数、已有剩余计算时间和物理NPU编号。
三项分别表示计算积压、单卡接收链路耗时、各盘公平分摊估计。$A_s$数的是有已知未完成工作的卡，不是当前实际发射的卡；静态分类CIR并不承诺这份均分，所以score不是实际完工时刻，也不是严格下界或deadline保证。
公式不使用未来未到达的请求，选卡也不需要盘内FIFO顺序。5 ms副本的整盘队列量不作为所有候选共同的score下限，以免抹平候选间差异。

JIT意为“有余量的层稍晚发射”：对刚激活、完全未提交的一个层coflow，令$K_s$为该层在盘s的块数，$Q_s$为5 ms副本中盘s所有Path的排队/活跃块数，D为这层需要数据到齐的时间点，先估计

$$\widehat T=1000b_G\max\left(\max_s\frac{Q_s+K_s}{40},\frac{\sum_s K_s}{50}\right),$$

$$t_{\rm release}=\max\left(now,\ D-\widehat T-10\ {\rm ms}\right).$$

例如D=40 ms、now=0、估计IO=8 ms，则发射门限为22 ms，而不是立即入盘队列；保留10 ms安全余量给未建模争用。D=15 ms时则立即放行。
这只是启发式：副本陈旧、QoS仲裁、其他卡新提交和链路排队都可能使真实耗时超过估计，10 ms也不是形式化的成功保证。
延迟只写入客户端新建submission状态；已经进入SSU队列的IO绝不撤回或重排。实际发射可能晚于这个门限，所有盘分量共用该层门限。
native的io_start仍记录“层激活”，不是被延迟后的实际发射；必须区分 $t_{\rm activation}\le t_{\rm release}\le t_{\rm actual\ issue}$，而原生compute/stall仍按真实数据到HBM计时。

为什么这可能改善？长计算请求不必尽早把不紧急的下一层塞满队列，短计算请求可能因此更早服务；但门限估计偏激进，也可能让自身错过deadline。它是有代价的调度权衡，不是无成本新增带宽。

Strategy 2则只按已提交IO携带的层deadline排序，平局倾向该盘较小的整层分量；这个大小是提交时的分量总量，不是预知未来剩余量。
固定后端到达序列、等长176 KiB时，局部交换不改变跨Path的既有仲裁规则；闭环中后续层释放可能改变，不能宣称整场Path选中时间序列也保持不变。

# 为什么低带宽平均值仍可能 miss deadline？

预取下一层从当前层计算开始；当前层还可计算的时间不是固定永远有C，而是 $D_i-now$。
在baseline FIFO中，即便自身读量很小，只要先前已入队大读取不可绕过，自己的数据仍可能赶不上deadline。
等待会推迟下一层发起，但不会凭空延长它自己的计算窗口；跨请求切换还会改变C和读量，使之前形成的相位不再合适。

多盘必须看跨盘barrier以及唯一接收链路，不能把6个40 GiB/s简单当作每卡可用240 GiB/s。
只知道总排队数量通常无法唯一确定每层到齐时刻；更好的路由可以减少争用，不能保证所有突发deadline均可满足。
本报告比较实际仿真结果，不把流体估计或理想队内重排包装成硬件保证。

这里仍有一个切实的数学不可行性检查：固定当前时刻now与数据deadline D，令$W_s(now,D)$为所有deadline不晚于D、**尚须盘s实际服务**的剩余GiB。若

$$W_s(now,D)>40\frac{D-now}{1000},$$

则无论如何重排，这些读取也不可能全部按时。每张NPU接收链路另做同样检查，将40改为50，并使用尚未送到HBM的链路剩余字节。
盘内active IO只计未读部分；已经离盘但尚在链路上的数据不能再算作SSD剩余工作。若拿不到active残量，可以只计完全未开始服务的IO，得到更保守但仍有效的不可行性证据。
普通总队列计数没有deadline归属；全局L还包含在途及未发射工作，所以不能直接把L塞进这个式子就宣称“数学上必stall”，也不能把已知未来八层全压到当前层的同一个D。

例如一条176 KiB读取在40 GiB/s下服务时间约0.004196 ms。对data里的32K/NQL128请求，C约1.178026 ms；如果从当前计算开始到这次deadline，在某一盘必须读完的完整、尚未开读IO至少281条，那么仅盘服务已需约1.179123 ms，必有读取迟到，其他链路和仲裁开销尚未计算。
这项证书只裁决已经给定的当前deadline，不证明所有可能的层间相位永远无解。反之没有违反容量不等式也不保证成功：Baseline队首阻塞、QoS分配、跨盘最后一块和接收链路都可能增加等待。
较短C如1.178/1.997 ms的请求对这些尾部更敏感；它们的SLO计数可能显著改善，但长计算请求贡献了较多计算毫秒，因此全卡平均U的变化仍可能很小。

@WEIGHTING@

@WORKED_EXAMPLE@

# 开发选择与失败例子：不是所有改动都变好

只用6盘near、种子20260906做候选开发；以下记录不冒充最终主表，完整请求输入与该开发组Baseline逐项一致。

@DEVELOPMENT@

compute只平衡计算积压，是消融项；fair pipeline兼顾跨盘与接收链路；compute guarded mix是在计算积压门限内再比较争用估计。
仅加入JIT不保证变好：5 ms余量比10 ms更激进，开发组利用率更差；10 ms相对5 ms的接纳SLO也少通过1个请求。
最终按首要目标“固定窗口平均利用率”选择fair pipeline + 10 ms；不是宣称它在所有指标上支配其他候选。
@DEVELOPMENT_TRADEOFF@

种子20260907不参与参数选择；同一逻辑trace跨盘组也在冻结后验证。其中6盘same_trace与6盘near开发输入完全相同，只是复跑，不算新增独立测试。开发只试了五个简单候选，不构成全策略空间的最优性证明。

@GENERALIZATION@

@FAILURE@

@OVERLOAD@

# 完整请求延迟：不要只看接纳后的 SLO

@LATENCY@

均处理是admission到completion，均用户延迟及P99均从arrival到completion计时；P99使用排序后索引 $(n-1)\times0.99$ 线性插值。用户SLO列仍使用同一 $1.5\times8C$ 阈值，但从arrival计时，故一般更严格。
两者都覆盖完整704个请求。接纳后SLO改善可能同时伴随接纳前排队变化；完成整批的makespan也必须一起看。

# 控制代价与可部署信息

**实时可用性还没有得到证明。** 5 ms更新压力副本，不等于一次Path规划只需5 ms以内、更不等于规划免费。
下面“均路由”是每次request-layer-SSU完整规划的平均墙钟耗时；一个层访问多盘就需要多次规划。
如果整场累计路由墙钟秒数远大于模拟完成秒数，当前Python原型不能据此宣称已满足单线程在线预算。
它可能通过优化数据结构、编译实现和跨客户端并行加速，但这些尚未测量；不能将并行实验负载下的wall time换称为CPU时间或硬件通信延迟。

@DECISIONS@

@COST@

下表为分项累计墙钟秒，不是模拟时间：

@TIMINGS@

JIT另单独计时，不合并到选卡或Path路由中；以下“最大延迟”是激活到发射门限的推迟，不是实际stall。10 ms是提前余量，不是固定延迟，也不是延迟上限。

@JIT@

tick和SSU读次数覆盖完整运行，耗时更长的策略会经历更多采集周期；固定的是5 ms采集频率，不是所有完整运行必须读相同总次数。
副本读取次数是主机查询开销，不是设备采集次数。运行期CIR写0不代表初始化配置没有成本。
最大副本年龄只统计实际访问；若Baseline没有使用副本，计数器的0不表示它一直拥有零时延的新数据。
一次算法的Python墙钟耗时不等同纯CPU时间，也不是硬件固有通信延迟；采集、路由、选卡、盘内重排应分项计时，不能只报告其中一个callback。
所有策略共同记录账本/ACK以做守恒审计，即使Baseline/Once并不利用账本选Path；因此其测量包含公共实验记账，不代表最小化Baseline驱动必然需要这些操作。
这些本地计时还不包含真实跨主机账本消息、锁竞争、回执传输和CIR设备写入等成本；全局信息不是免费取得的。
如果New once只有不到1个百分点的利用率收益，真实的协调开销完全可能抵消它；本地callback计时不能证明分布式部署仍有净收益。

仿真目前不把这些控制耗时加入数据面时序；尤其Strategy 2的理想重排不是免费硬件能力。
实际部署还需考虑副本年龄、多个客户端共享账本的同步、完成回执时延、时钟一致性、SSU内部调度接口和软件执行开销。

# 审计、文件与复现

本报告独立核验同组input fingerprint、原data SHA、相同request_id/arrival/compute/IO总量；利用率和两个SLO都从逐层/逐请求原生记录重算。
采集时间必须0、5、10…ms，每盘实际fresh读次数必须与collector一致，副本年龄不得超过5 ms；运行期不得出现CIR写。
@VERSIONS@
版本差异有明确来源：Baseline/Once/New once的27条reference来自隔离的source_v2；正式S1/S2来自冻结fair-pipeline与JIT后的source_v3。后者新增S1/JIT分支，没有改变前3种策略分支；原生仿真core和原Once引擎SHA保持一致，逐结果均归档对应源码。这不是旧版独占Path或动态CIR策略混入。
6盘near06的正式S1与选中的jit10开发记录，逐卡U、两种SLO和makespan必须重新比对一致；callback墙钟计时不要求相同。

主目录只放最终PDF；数据及audit在data，图在figures，本文Markdown在docs。完整请求SLO记录另存CSV，方便按request_id核查。

```bash
python -B build_shared_path_report.py --audit-only
python -B build_shared_path_report.py
```

未经测量的结果不填0、也不补推测数字；五策略、三拓扑完整矩阵尚未齐备时，脚本不会生成看似完整的最终PDF。

# 独立函数怎么快速验证？

这五个Python文件提供的是纯决策函数，没有偷偷读取仿真器实时状态。请在项目根目录运行；Once复用项目原引擎，仍需要本项目及其依赖，不是五个完全无依赖的单文件替代仿真器。

- `shared_path_baseline.py`：`baseline_path_ids(io_count)`输入等长IO数量，输出同样数量的Path0编号。
- `shared_path_once.py`：`once_path_ids`输入块数、5 ms压力副本、该请求合法Path集合、静态QoS及起点偏移，输出每个IO的Path编号。一次调用规划一个request-layer-SSU，不是整层只选一条Path。
- `shared_path_new_once.py`：`new_once_path_ids`再接收全局已承诺/未HBM确认账本。函数只返回计划，不修改账本；调用方必须原子预留并在实际HBM确认时扣减。
- `shared_path_strategy1.py`：`choose_npu`接收已经到达的每卡计算/IO积压、新请求逐盘块数、每层C、层数、盘和链路带宽，返回NPU编号、候选评分及解释项。`prefetch_release_ms`另接收now、数据deadline、当前层逐盘块数和5 ms队列副本，返回最早允许发射的时间门限。
- `shared_path_strategy2.py`：`reorder_pending`只接收某一已选Path内已到达的pending IO及now，每项携带IO ID、NPU/request/layer ID、SSU/Path、提交时间、层deadline及该盘整层块数；输出这些IO ID的新顺序，不包含正在服务的IO。

完整S1必须分事件组合：到达时`choose_npu`，层激活时`prefetch_release_ms`，允许发射后用`new_once_path_ids`选Path并记账，完成时扣减账本。**只调用choose_npu并没有执行JIT或Path路由。** S2再在设备已经选中Path时调用`reorder_pending`。
这些输出是调度决策/启发式评分，不是“精确完工时间”或保证无stall的布尔证明。

以下五个演示已实际运行：

```bash
python -B shared_path_baseline.py
python -B shared_path_once.py
python -B shared_path_new_once.py
python -B shared_path_strategy1.py
python -B shared_path_strategy2.py
```

New once演示中两个客户端共享同一个旧空快照，第二个仍会看见第一个的Path预留；这正是全局账本增加的信息。S1演示会打印NPU编号、score和独立的release时间，S2演示将较早deadline的IO置前。

```bash
python -B -m pytest -q test_shared_path_policies.py
python -B -m pytest -q test_shared_path_strategy1.py
python -B -m pytest -q test_shared_path_sim_adapter.py
python -B -m pytest -q test_shared_path_reorder_integration.py
python -B -m pytest -q test_shared_path_jit_causality.py
python -B -m pytest -q test_shared_path_report.py
```

单测证明的是各自检查范围内的守恒、权限、公式与事件接入；不证明所有输入都能提高利用率，更不证明真实设备在线实时开销足够小。
'''.replace("@PRIMARY@", markdown_table(["SSU", "策略", "U/%", "接纳SLO通过", "通过率/%", "active"], primary)) \
        .replace("@LATENCY@", markdown_table(["SSU", "策略", "均处理/ms", "均用户/ms", "P99用户/ms", "用户SLO/%", "完工/ms"], latency)) \
        .replace("@COST@", markdown_table(["SSU", "策略", "采集tick", "SSU读", "副本读", "最大年龄/ms", "CIR写"], costs)) \
        .replace("@PROFILES@", markdown_table(["SSU", "长度K", "NQL", "块/层", "C/ms", "每卡配额"], profiles)) \
        .replace("@DEMAND@", markdown_table(["SSU", "总名义GiB/s", "总容量", "最热盘GiB/s", "总量/容量%"], demand_rows)) \
        .replace("@TIMINGS@", markdown_table(["SSU", "策略", "采集/s", "路由/s", "选卡/s", "重排/s", "ACK记账/s"], timings)) \
        .replace("@DECISIONS@", markdown_table(["SSU", "策略", "路由次数", "均路由/ms", "均选卡/ms", "均重排/us", "模拟全程/s"], decisions)) \
        .replace("@JIT@", markdown_table(["SSU", "策略", "激活层数", "延迟发射层数", "最大延迟/ms", "累计JIT/s", "均JIT/us"], jit_rows)) \
        .replace("@VERSIONS@", version_note) \
        .replace("@DEVELOPMENT@", development_table) \
        .replace("@GENERALIZATION@", generalization) \
        .replace("@FAILURE@", failure) \
        .replace("@ARRIVAL_WARNING@", arrival_warning) \
        .replace("@SCOPE_OUTCOME@", scope_outcome) \
        .replace("@HELDOUT_OUTCOME@", heldout_outcome) \
        .replace("@DEVELOPMENT_TRADEOFF@", development_tradeoff) \
        .replace("@OVERLOAD@", overload) \
        .replace("@WEIGHTING@", weighting) \
        .replace("@WORKED_EXAMPLE@", worked_example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUT / "data")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--regime", default="near")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows, selected, audit = collect_results(args.data)
    development = collect_development(args.data, selected)
    audit["development_candidates"] = development
    assert rows, "No completed experimental results; no numbers or PDF fabricated"
    if args.audit_only:
        print(json.dumps({k: v for k, v in audit.items() if k != "records"}, ensure_ascii=False, indent=2))
        return
    for ssu in (5, 6, 7):
        for policy in STRATEGIES:
            assert (ssu, args.regime, args.seed, policy) in selected, ("missing primary case", ssu, policy)
    for folder in ("docs", "data", "figures"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    make_figure2(selected, args.output / "figures", seed=args.seed, regime=args.regime)
    make_policy_figure(args.output / "figures")
    document = build_document(selected, audit, seed=args.seed, regime=args.regime, development=development)
    source = args.output / "docs/shared_path_5ms_report.md"
    source.write_text(document, encoding="utf-8")
    (args.output / "data/report_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "data/experiment_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "data/full_population_slo.csv").open("w", newline="") as handle:
        first = next(iter(selected.values()))["metrics"]["records"][0]
        writer = csv.DictWriter(handle, fieldnames=["num_ssu", "regime", "seed", "strategy", *first])
        writer.writeheader()
        for key, case in sorted(selected.items()):
            for record in case["metrics"]["records"]:
                writer.writerow(dict(zip(("num_ssu", "regime", "seed", "strategy"), key), **record))
    subprocess.run(["pandoc", str(source), "--standalone", "--number-sections", "--pdf-engine=xelatex",
                    "--resource-path", str(source.parent), "--output", str(args.output / "shared_path_5ms_report.pdf")], check=True)
    print(json.dumps({"runs": len(rows), "pdf": str(args.output / "shared_path_5ms_report.pdf")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
