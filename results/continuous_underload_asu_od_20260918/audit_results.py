#!/usr/bin/env python3
"""Independent audit of the complete, exact twelve-case formal grid.

Reads frozen manifests/raw results; imports neither simulator nor metrics.
Demand is rebuilt by scanning active requests at each interval midpoint,
independently of the production metric's event-increment active-set sweep.
Writes only audit/independent_results.json, after all twelve jobs complete.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NPU, SSU, LAYERS = 32, 3, 8
IO_BYTES, IO_GIB, CAPACITY = 176 * 1024, 176 * 1024 / 2**30, 40.
SEEDS = (7, 19, 43)
ORDERS = ("random", "ordered")
POLICIES = ("asu_baseline", "od_baseline")
WINDOWS = ((2000., 4000.), (2000., 6000.))
SLO_EPS_MS = 1e-9


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(actual, expected, *, tolerance=1e-7, label="value"):
    assert math.isfinite(actual) and math.isfinite(expected), (label, actual, expected)
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=tolerance), (label, actual, expected)


def check_vector(actual, expected, label):
    assert len(actual) == len(expected), label
    for i, (a, e) in enumerate(zip(actual, expected)):
        check_close(a, e, label=f"{label}[{i}]")


def clip_duration(begin, end, left, right):
    lo, hi = max(begin, left), min(end, right)
    return hi - lo if hi > lo else 0.


def request_slos(rows, source):
    """Own compute comes from raw manifests, not recorded request own_compute."""
    answer = {}
    for name, origin in (("admission", "admission_time_ms"), ("arrival", "arrival_time_ms")):
        latencies = [r["completion_time_ms"] - r[origin] for r in rows]
        own = [LAYERS * source[r["request_id"]]["load"]["per_layer_us"] / 1000. for r in rows]
        ratios = [latency / compute for latency, compute in zip(latencies, own)]
        item = {}
        for factor in (1., 1.5):
            passed = sum(latency <= factor * compute + SLO_EPS_MS for latency, compute in zip(latencies, own))
            item[f"{factor:g}"] = dict(count=len(rows), passed=passed,
                                      percent=100. * passed / len(rows) if rows else None)
        answer[name] = dict(slo=item, mean_ms=math.fsum(latencies) / len(rows) if rows else None,
                            mean_ratio=math.fsum(ratios) / len(rows) if rows else None)
    return answer


def check_slos(actual, stored, label):
    for clock in ("admission", "arrival"):
        for factor in ("1", "1.5"):
            a, e = actual[clock]["slo"][factor], stored[clock]["slo"][factor]
            assert (a["count"], a["passed"]) == (e["count"], e["passed"]), (label, clock, factor, a, e)
            if a["percent"] is None:
                assert e["percent"] is None
            else:
                check_close(a["percent"], e["percent"], label=f"{label}/{clock}/{factor}")
        if actual[clock]["mean_ms"] is not None:
            check_close(actual[clock]["mean_ms"], stored[clock]["latency_ms"]["mean"], label=f"{label}/{clock}/latency_mean")
            check_close(actual[clock]["mean_ratio"], stored[clock]["normalized_latency"]["mean"], label=f"{label}/{clock}/ratio_mean")


def midpoint_demand(rows, rates, source, left, right):
    """Re-find every active request afresh, without maintaining an active set."""
    edges = {left, right}
    relevant = []
    for row in rows:
        start, end = row["admission_time_ms"], row["completion_time_ms"]
        if start < right and end > left:
            relevant.append(row)
            edges.add(max(start, left))
            edges.add(min(end, right))
    edges = sorted(edges)
    segments, active_counts = [], []
    midpoint_fallbacks = 0
    for begin, end in zip(edges, edges[1:]):
        assert end > begin
        midpoint = begin + (end - begin) / 2.
        if not begin <= midpoint < end:
            # Adjacent representable floats may have no representable strict
            # interior midpoint. Right-continuous membership at begin agrees.
            midpoint = begin
            midpoint_fallbacks += 1
        active = [r["request_id"] for r in relevant
                  if r["admission_time_ms"] <= midpoint < r["completion_time_ms"]]
        npus = [source[rid]["npu_id"] for rid in active]
        assert len(set(npus)) == len(npus) <= NPU
        values = [math.fsum(rates[rid][disk] for rid in active) for disk in range(SSU)]
        segments.append([begin, end, *values])
        active_counts.append([begin, end, len(active)])
    duration = right - left
    minima = [min(row[2 + s] for row in segments) for s in range(SSU)]
    maxima = [max(row[2 + s] for row in segments) for s in range(SSU)]
    means = [math.fsum((row[1] - row[0]) * row[2+s] for row in segments) / duration for s in range(SSU)]
    over = [100. * math.fsum(row[1] - row[0] for row in segments if row[2+s] > CAPACITY) / duration for s in range(SSU)]
    ge = [100. * math.fsum(row[1] - row[0] for row in segments if row[2+s] >= CAPACITY) / duration for s in range(SSU)]
    intervals = [[dict(start_ms=row[0], end_ms=row[1], demand_GiB_s=row[2+s])
                  for row in segments if row[2+s] >= CAPACITY] for s in range(SSU)]
    return dict(per_disk_min_GiB_s=minima, per_disk_max_GiB_s=maxima,
                per_disk_mean_GiB_s=means, per_disk_overload_percent=over,
                per_disk_at_or_above_capacity_percent=ge,
                minimum_capacity_margin_GiB_s=[CAPACITY - value for value in maxima],
                all_disks_overload_percent=100. * math.fsum(row[1]-row[0] for row in segments if all(v > CAPACITY for v in row[2:])) / duration,
                any_disk_overload_percent=100. * math.fsum(row[1]-row[0] for row in segments if any(v > CAPACITY for v in row[2:])) / duration,
                any_disk_at_or_above_capacity_percent=100. * math.fsum(row[1]-row[0] for row in segments if any(v >= CAPACITY for v in row[2:])) / duration,
                strict_underload_all_disks=all(value < CAPACITY for value in maxima),
                segments=segments, active_npu_count_segments=active_counts,
                at_or_above_capacity_intervals_by_ssu=intervals,
                midpoint_rounding_fallback_count=midpoint_fallbacks)


def audit_window(result, source, rates, left, right, stored, *, full=False):
    rows = result["summary"]["request_metrics"]
    batches = result["summary"]["microbatch_metrics"]
    duration = right - left
    compute_pieces, active_pieces = [[] for _ in range(NPU)], [[] for _ in range(NPU)]
    for batch in batches:
        npu = batch["npu_id"]
        for layer in batch["layer_metrics"]:
            compute_pieces[npu].append(clip_duration(layer["compute_start_ms"], layer["compute_end_ms"], left, right))
    for r in rows:
        active_pieces[source[r["request_id"]]["npu_id"]].append(clip_duration(r["admission_time_ms"], r["completion_time_ms"], left, right))
    compute = list(map(math.fsum, compute_pieces))
    active = list(map(math.fsum, active_pieces))
    assert all(-1e-8 <= c <= a + 1e-7 <= duration + 2e-7 for c, a in zip(compute, active))
    utilization = 100. * math.fsum(compute) / (NPU * duration)
    per_npu = [100. * value / duration for value in compute]
    all_active = all(abs(value - duration) < 1e-7 for value in active)
    check_close(stored["start_ms"], left)
    check_close(stored["end_ms"], right)
    assert stored["full_population"] is full
    check_close(utilization, stored["U_percent"], label="U")
    check_vector(per_npu, stored["per_npu_U_percent"], "per_npu_U")
    check_vector(active, stored["per_npu_active_ms"], "per_npu_active_ms")
    assert all_active == stored["all_npus_active"]
    if not full:
        assert all_active, "Not all 32 NPUs remained active in a required warm window"
    cohort = list(rows) if full else [r for r in rows if left <= r["admission_time_ms"] < right]
    assert sorted(r["request_id"] for r in cohort) == stored["cohort_request_ids"]
    assert sum(r["completion_time_ms"] > right for r in cohort) == stored["completed_after_window"]
    assert sum(left <= r["admission_time_ms"] < right for r in rows) == stored["admitted_in_window"]
    finished = [r for r in rows if left <= r["completion_time_ms"] < right or (full and r["completion_time_ms"] == right)]
    assert len(finished) == stored["completed_in_window"]
    slos = request_slos(cohort, source)
    check_slos(slos, stored["request_statistics"], "all")
    assert slos["admission"]["slo"]["1.5"] == stored["slo"]
    assert slos["arrival"]["slo"]["1.5"] == stored["arrival_slo"]
    group_keys = {
        "length": lambda q: str(q["load"]["seq_len_k"]),
        "category": lambda q: q["load"]["category"],
        "profile": lambda q: f"{q['load']['seq_len_k']}:{q['load']['nql']}",
    }
    for dimension, key in group_keys.items():
        grouped = stored[f"by_{dimension}"]
        assert set(grouped) == {key(q) for q in source.values()}
        for value in grouped:
            sample = [r for r in cohort if key(source[r["request_id"]]) == value]
            check_slos(request_slos(sample, source), grouped[value], f"{dimension}/{value}")
    demand = midpoint_demand(rows, rates, source, left, right)
    expected_demand = stored["demand"]
    for field in ("per_disk_min_GiB_s", "per_disk_max_GiB_s", "per_disk_mean_GiB_s",
                  "per_disk_overload_percent", "per_disk_at_or_above_capacity_percent", "minimum_capacity_margin_GiB_s"):
        check_vector(demand[field], expected_demand[field], field)
    for field in ("all_disks_overload_percent", "any_disk_overload_percent", "any_disk_at_or_above_capacity_percent"):
        check_close(demand[field], expected_demand[field], label=field)
    assert demand["strict_underload_all_disks"] == expected_demand["strict_underload_all_disks"]
    assert len(demand["segments"]) == len(expected_demand["segments"])
    for actual, expected in zip(demand["segments"], expected_demand["segments"]):
        check_vector(actual, expected, "demand_segment")
    assert demand["active_npu_count_segments"] == expected_demand["active_npu_count_segments"]
    for actual, expected in zip(demand["at_or_above_capacity_intervals_by_ssu"], expected_demand["at_or_above_capacity_intervals_by_ssu"]):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            for field in ("start_ms", "end_ms", "demand_GiB_s"):
                check_close(a[field], e[field], label=f"capacity_interval/{field}")
    return dict(start_ms=left, end_ms=right, full_population=full, U_percent=utilization,
                per_npu_U_percent=per_npu, all_npus_active=all_active,
                cohort_count=len(cohort), completed_after_window=stored["completed_after_window"],
                slos=slos, demand={key:value for key,value in demand.items()
                                  if key not in ("segments", "active_npu_count_segments", "at_or_above_capacity_intervals_by_ssu")},
                demand_interval_count=len(demand["segments"]),
                active_npu_count_min=min(row[2] for row in demand["active_npu_count_segments"]),
                active_npu_count_max=max(row[2] for row in demand["active_npu_count_segments"]),
                checks="U, 32 per-NPU U/active durations, both SLO clocks/factors, category/length/profile SLO, all exact demand intervals passed")


def audit_case(job, plan):
    folder = HERE / "runs" / job["label"]
    paths = [folder / name for name in ("command.json", "manifest.json.gz", "result.json.gz")]
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    command, manifest, result = map(read, paths)
    assert command["status"] == "complete" and command["completed_simulation"] is True
    assert command["smoke"] is False and command["pilot"] is False
    for field in ("core_unchanged", "extension_unchanged", "original_manifest_unchanged"):
        assert command[field] is True
    assert all(command["checks"].values())
    assert (command["order"], command["policy"], command["seed"]) == (job["order"], job["policy"], job["seed"])
    assert sha(paths[1]) == job["manifest_sha256"] == command["manifest_sha256"] == command["original_manifest_sha256"]
    assert sha(paths[2]) == command["result_sha256"] == command["output_sha256"]
    for group in (command["core_source_sha256"], command["extension_source_sha256"], result["core_and_policy_sha256"]):
        for name, digest in group.items():
            assert plan["source_sha256"][name] == digest, (name, "frozen source mismatch")
    assert result["order"] == job["order"] and result["strategy"] == job["policy"]
    assert result["metadata"]["seed"] == job["seed"]
    assert result["input_fingerprint"] == manifest["input_fingerprint"] == command["input_fingerprint"]
    assert result["input_placement_fingerprint"] == result["execution_placement_fingerprint"]
    source = {q["request_id"]: q for q in manifest["requests"]}
    assert len(source) == len(manifest["requests"]) == 640
    summary = result["summary"]
    assert (summary["num_npu"], summary["num_ssu"], summary["n_layers"], summary["batch_size"]) == (NPU, SSU, LAYERS, 1)
    assert all(summary["invariants"].values())
    rows, batches = summary["request_metrics"], summary["microbatch_metrics"]
    assert len(rows) == len(batches) == summary["request_count"] == command["completed_requests"] == 640
    assert {r["request_id"] for r in rows} == set(source)
    assert {b["member_request_ids"][0] for b in batches} == set(source)
    runtime_by_id = {r["request_id"]: r for r in rows}
    rates, expected_blocks, inventory = {}, [[0] * NPU for _ in range(SSU)], {}
    for rid, q in source.items():
        npu, load = q["npu_id"], q["load"]
        assert 0 <= npu < NPU and q["arrival_time_ms"] == 0.
        layer_list = manifest["placements"][q["placement_index"]]
        assert len(layer_list) == 1
        layer = layer_list[0]
        counts = Counter(disk for disk, _ in layer)
        assert all(disk in range(SSU) and size == IO_GIB for disk, size in layer)
        rates[rid] = tuple(counts[s] * IO_GIB * 1e6 / load["per_layer_us"] for s in range(SSU))
        for s in range(SSU):
            expected_blocks[s][npu] += LAYERS * counts[s]
        r = runtime_by_id[rid]
        assert r["npu_id"] == npu
        assert all(math.isfinite(r[field]) for field in ("arrival_time_ms", "admission_time_ms", "completion_time_ms"))
        assert 0. == r["arrival_time_ms"] <= r["admission_time_ms"] <= r["completion_time_ms"]
        check_close(r["own_compute_ms"], LAYERS * load["per_layer_us"] / 1000., tolerance=1e-8, label="own_compute")
        stable_id = load["original_request_id"]
        assert stable_id not in inventory
        inventory[stable_id] = [npu, load["seq_len_k"], load["nql"], load["per_layer_us"], layer]
    layer_total = 0.
    batch_by_id = {}
    for batch in batches:
        assert batch["batch_size"] == len(batch["member_request_ids"]) == 1
        rid = batch["member_request_ids"][0]
        q, r = source[rid], runtime_by_id[rid]
        assert batch["npu_id"] == q["npu_id"]
        check_close(batch["admission_time_ms"], r["admission_time_ms"])
        check_close(batch["completion_time_ms"], r["completion_time_ms"])
        layers = batch["layer_metrics"]
        assert len(layers) == LAYERS and [m["layer"] for m in layers] == list(range(LAYERS))
        last_end = r["admission_time_ms"]
        for m in layers:
            begin, end = m["compute_start_ms"], m["compute_end_ms"]
            assert math.isfinite(begin) and math.isfinite(end) and begin >= last_end - 1e-8
            assert begin >= m["io_ready_time_ms"] - 1e-8
            check_close(end - begin, q["load"]["per_layer_us"] / 1000., tolerance=1e-8, label="layer C")
            layer_total += end - begin
            last_end = end
        check_close(last_end, r["completion_time_ms"])
        batch_by_id[rid] = batch
    for npu in range(NPU):
        executed = sorted((r for r in rows if source[r["request_id"]]["npu_id"] == npu), key=lambda r:r["admission_time_ms"])
        expected_order = [q["request_id"] for q in manifest["requests"] if q["npu_id"] == npu]
        assert [r["request_id"] for r in executed] == expected_order
        for previous, following in zip(executed, executed[1:]):
            assert following["admission_time_ms"] >= previous["completion_time_ms"] - 1e-8
            release = batch_by_id[previous["request_id"]]["layer_metrics"][-1]["compute_start_ms"]
            check_close(following["layer0_io_start_time_ms"], release, label="cross-request L0 release")
    makespan = max(row["completion_time_ms"] for row in rows)
    check_close(makespan, summary["makespan_ms"])
    check_close(layer_total, math.fsum(LAYERS * q["load"]["per_layer_us"] / 1000. for q in source.values()))
    expected_count = sum(map(sum, expected_blocks))
    assert summary["submitted_blocks"] == summary["completed_blocks"] == command["observed_blocks"] == command["expected_blocks"] == expected_count
    assert result["completed_blocks_by_ssu_npu"] == expected_blocks
    check_close(summary["completed_read_gb"], expected_count * IO_GIB)
    expected_paths = [(n % 8) * 32 + n // 8 if job["policy"] == "od_baseline" else 0 for n in range(NPU)]
    observed_paths = result["observed_path_ids_by_ssu_npu"]
    assert len(observed_paths) == SSU
    for disk in observed_paths:
        assert len(disk) == NPU
        assert disk == [[path] for path in expected_paths]
    adapter = result["adapter_statistics"]
    assert adapter["reserved_blocks"] == adapter["acknowledged_blocks"] == expected_count
    assert adapter["ledger_end_counts_by_ssu"] == [0] * SSU
    assert adapter["reorder_calls"] == adapter["assignment_count"] == 0
    assert adapter["cir_write_events"] == []
    planned_counts = {(row["ssu_id"], row["npu_id"], row["path_id"]): row["blocks"]
                      for row in adapter["routed_blocks_by_ssu_npu_path"]}
    assert planned_counts == {(s, n, expected_paths[n]): expected_blocks[s][n] for s in range(SSU) for n in range(NPU)}
    qos = result["actual_qos_by_ssu"]
    assert len(qos) == SSU
    if job["policy"] == "od_baseline":
        assert len(set(expected_paths)) == NPU
        cirs = [1.25 if path in expected_paths else 0. for path in range(256)]
        weights = [1. if path in expected_paths else 0. for path in range(256)]
    else:
        cirs = [budget/(8*count) for _ in range(8) for budget,count in ((20.,12),(6.,4),(8.,12),(6.,4)) for _ in range(count)]
        weights = [1.] * 256
    for disk in qos:
        check_vector(disk["path_cirs_gib_s"], cirs, "actual CIR")
        assert disk["path_weights"] == weights
        assert disk["group_weights"] == [1.] * 8
        assert disk["path_pirs_gib_s"] == ["unlimited"] * 256
    check_vector(result["static_path_cirs_gib_s"], cirs, "reported CIR")
    assert len(result["analysis"]) == 3
    recomputed = [audit_window(result, source, rates, a, z, result["analysis"][index])
                  for index, (a, z) in enumerate(WINDOWS)]
    recomputed.append(audit_window(result, source, rates, 0., makespan, result["analysis"][2], full=True))
    check_close(recomputed[2]["U_percent"], 100. * summary["fleet_npu_compute_utilization"])
    for s, disk in enumerate(summary["disk_stats"]):
        assert disk["ssu_id"] == s and disk["max_backend_active_io"] == 1
        assert disk["backend_dispatches"] == sum(expected_blocks[s])
        volume = sum(expected_blocks[s]) * IO_GIB
        check_close(disk["completed_gb"], volume)
        check_close(result["analysis"][2]["SSD_GiB_s"][s], volume * 1000. / makespan)
        check_close(math.fsum(result["warm_ssd_GiB_s_by_ssu_npu"][s]), result["analysis"][0]["SSD_GiB_s"][s])
        check_close(math.fsum(result["warm_ssd_10ms_GiB_s"][s]) / 200., result["analysis"][0]["SSD_GiB_s"][s])
    assert all(sha(ROOT / name) == digest for name, digest in before.items())
    inventory_fingerprint = hashlib.sha256(json.dumps(sorted(inventory.items()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(label=job["label"], order=job["order"], policy=job["policy"], seed=job["seed"],
                files_sha256=before, physical_inventory_sha256=inventory_fingerprint,
                original_ids_count=len(inventory), request_count=len(rows), block_count=expected_count,
                path_ownership_observed_and_configured=True, npu_path_ids=expected_paths,
                independent_layer_compute_total_ms=layer_total, makespan_ms=makespan,
                windows=recomputed, status="passed")


def main():
    plan_path = HERE / "formal_plan.json"
    plan = read(plan_path)
    jobs = plan["jobs"]
    expected = set(itertools.product(ORDERS, POLICIES, SEEDS))
    actual = [(job["order"], job["policy"], job["seed"]) for job in jobs]
    assert len(jobs) == len(set(actual)) == 12 and set(actual) == expected
    assert len({job["label"] for job in jobs}) == 12
    assert plan["windows_ms"] == [[a,z] for a,z in WINDOWS]
    # Never publish a successful partial-grid audit or include smoke runs.
    pending = []
    for job in jobs:
        folder = HERE / "runs" / job["label"]
        if not (folder / "command.json").exists():
            pending.append(job["label"])
        else:
            command = read(folder / "command.json")
            if command.get("status") != "complete" or not command.get("completed_simulation") or not (folder / "result.json.gz").exists():
                pending.append(job["label"])
    if pending:
        raise RuntimeError(f"All 12 formal cases must finish before audit; pending: {pending}")
    report = dict(status="running", plan_sha256=sha(plan_path), auditor_sha256=sha(Path(__file__)),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  independence="No simulator/metrics imports. U from raw layer intervals; SLO from request times + manifest C; demand from midpoint active-request scans.",
                  tolerances=dict(comparison_abs=1e-7, comparison_rel=1e-11, slo_ms=SLO_EPS_MS, demand_threshold="literal >40 or >=40, no tolerance"),
                  limitation="Path audit checks recorded runtime observed IDs/block counts against independent manifest budgets; it is not a second replay of every I/O event.", cases=[])
    output = HERE / "audit" / "independent_results.json"
    try:
        queue_checks = []
        queued_labels = []
        for host in ("local", "remote"):
            queue_plan_path = HERE / f"{host}_plan.json"
            queue_status_path = HERE / f"{host}_plan_status.json"
            queue_plan, queue_status = read(queue_plan_path), read(queue_status_path)
            assert queue_status["complete"] is True
            assert queue_status["sources_verified_before"] is True
            assert queue_status["sources_verified_after"] is True
            assert queue_status["plan_sha256"] == sha(queue_plan_path)
            assert queue_status["queue_source_sha256"] == sha(HERE / "run_queue.py")
            assert queue_plan["source_sha256"] == plan["source_sha256"]
            labels = {job["label"] for job in queue_plan["jobs"]}
            assert labels == set(queue_status["jobs"])
            for label, status in queue_status["jobs"].items():
                assert status["status"] == "complete" and status["returncode"] == 0, label
            queued_labels.extend(labels)
            queue_checks.append(dict(host=host, jobs=len(labels), complete=True,
                                     sources_verified_before=True, sources_verified_after=True,
                                     queue_source_sha256=queue_status["queue_source_sha256"],
                                     status_sha256=sha(queue_status_path)))
        assert len(queued_labels) == len(set(queued_labels)) == 12
        assert set(queued_labels) == {job["label"] for job in jobs}
        report["queue_checks"] = queue_checks
        for name, digest in plan["source_sha256"].items():
            assert sha(ROOT / name) == digest, f"Frozen source/input changed: {name}"
        for job in jobs:
            report["cases"].append(audit_case(job, plan))
        # All seeds/orders/policies keep the same canonical physical requests.
        assert len({case["physical_inventory_sha256"] for case in report["cases"]}) == 1
        for order, seed in itertools.product(ORDERS, SEEDS):
            matches = [job for job in jobs if job["order"] == order and job["seed"] == seed]
            assert len(matches) == 2 and len({j["manifest_sha256"] for j in matches}) == 1
        for name, digest in plan["source_sha256"].items():
            assert sha(ROOT / name) == digest
        assert sha(plan_path) == report["plan_sha256"]
        report.update(status="passed", formal_cases=12, windows_recomputed=36,
                      same_physical_requests_across_all_cases=True,
                      same_frozen_manifest_across_policies=True,
                      all_32_npus_active_in_both_warm_windows=True,
                      inputs_raw_results_and_frozen_sources_unchanged=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], formal_cases=12, windows=36,
                          output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
