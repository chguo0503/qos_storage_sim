#!/usr/bin/env python3
"""Independent audit of all 63 formula32 runs; no simulator/runner imports.

Read manifests, recompute placement/fingerprints and all statistics from native
intervals, then compare published metrics. Requires the main 6x3x3 grid plus
nine XY12_16 four-disk capacity controls. Five XY/X16 six-disk groups must be
strictly under nominal per-disk V/C capacity. R32_A200_B10 and four-disk runs
are measured without imposing that conclusion. AB warm coverage is
reported as a finding, not assumed. This program never runs a simulation.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
GROUPS = ("XY12_32", "XY12_24", "XY12_20", "XY12_16", "X16", "R32_A200_B10")
SEEDS = (7, 19, 43)
STRATEGIES = ("asu_baseline", "od_baseline", "once")
TIME_TOL = 1e-7
SLO_EPS_MS = 1e-9
BW_TOL = 1e-7


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


@lru_cache(maxsize=None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, absolute=TIME_TOL, relative=1e-10):
    check(math.isclose(actual, expected, abs_tol=absolute, rel_tol=relative),
          f"{message}: actual={actual!r}, expected={expected!r}")


def overlap(a, b, lo, hi):
    return max(0.0, min(b, hi) - max(a, lo))


def digest_position(namespace, first, second):
    return int.from_bytes(hashlib.sha256(namespace + struct.pack("!QQ", first, second)).digest(), "big")


@lru_cache(maxsize=None)
def ring(disks):
    nodes = sorted((digest_position(b"qos_storage_sim:block_ring_hash:vnode:v1\0", d, v), d)
                   for d in range(disks) for v in range(256))
    return [x[0] for x in nodes], [x[1] for x in nodes]


def validate_input(manifest, config):
    rows = manifest["requests"]
    n, disks = config["num_npu"], config["ssu"]
    check(n == 32 and config["mode"] == "random" and config["input_counts"] == [1, 12], "Formal input design changed")
    check(manifest["metadata"]["config"] == config, "Config differs from embedded manifest config")
    check(manifest["metadata"]["queue_depth_limit"] is None, "Unexpected queue depth cap")
    positions, disk_ids = ring(disks)
    by_id, by_npu = {}, defaultdict(list)
    blocks_by_npu_disk = Counter()
    volume_by_disk = [0.0] * disks
    data, fingerprint = {}, hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    tails = 0
    for row in sorted(rows, key=lambda r: r["request_id"]):
        rid, card, load = row["request_id"], row["npu_id"], row["load"]
        check(rid not in by_id and 0 <= card < n, "Duplicate request ID or invalid NPU")
        check(row["arrival_time_ms"] == load["arrival_ms"] == 0.0, "Arrival changed")
        check(load["npu_id"] == card and load["original_request_id"] == rid, "Identity/binding changed")
        check(rid == card * 1_000_000 + load["generation"], "Physical ID rule changed")
        role = load["profile_group"]
        p = config["profile_" + role]
        check(load["per_layer_us"] == p["compute_us"], "Manifest profile compute differs from config")
        check(load["ssd_prefix_tokens"] == p["ssd_prefix_tokens"] == p["total_tokens"] - p["nql"], "Prefix mismatch")
        check(load["total_tokens"] == p["total_tokens"] and load["nql"] == p["nql"], "Token profile mismatch")
        placement = manifest["placements"][row["placement_index"]]
        check(len(placement) == 1, "Expected one reused placement across eight layers")
        prefix, layer = p["ssd_prefix_tokens"], placement[0]
        check(len(layer) == (prefix + 127) // 128, "Command count/padding mismatch")
        per_disk_bytes = [0] * disks
        for index, (disk, gib) in enumerate(layer):
            token_count = min(128, prefix - index * 128)
            byte_count = token_count * 1408
            check(gib == byte_count / 2**30, "Exact tail bytes differ")
            position = digest_position(b"qos_storage_sim:block_ring_hash:block:v1\0", rid, index)
            expected_disk = disk_ids[bisect.bisect_left(positions, position) % len(positions)]
            check(disk == expected_disk, "RingHash placement mismatch")
            per_disk_bytes[disk] += byte_count
            blocks_by_npu_disk[(card, disk)] += 8
            tails += int(token_count < 128)
        check(sum(per_disk_bytes) == prefix * 1408, "Total per-layer bytes mismatch")
        for disk, byte_count in enumerate(per_disk_bytes):
            close(load["disk_gib"][disk], byte_count / 2**30, "Per-disk profile volume", absolute=1e-12)
            volume_by_disk[disk] += 8 * byte_count
        c_ms = p["compute_us"] / 1000
        data[rid] = dict(npu=card, role=role, compute_layer_ms=c_ms,
                        disk_bytes=per_disk_bytes, blocks_per_layer=len(layer),
                        rates=[v / 1e6 / c_ms for v in per_disk_bytes])
        exact_placement = tuple(tuple((int(d), float(v)) for d, v in l) for l in placement)
        fingerprint.update(repr((rid, card, float(row["arrival_time_ms"]), load["category"],
                                 load["per_layer_us"], exact_placement)).encode())
        by_id[rid] = row
        by_npu[card].append(row)
    check(set(by_npu) == set(range(n)), "Missing NPU inputs")
    na, nb = config["input_counts"]
    cycle_ms = 8 * (na * config["profile_A"]["compute_us"] + nb * config["profile_B"]["compute_us"]) / 1000
    cycles = math.ceil(config["horizon_ms"] / cycle_ms) + 1
    for card, card_rows in by_npu.items():
        expected = ["A"] * (na * cycles) + ["B"] * (nb * cycles)
        random.Random(config["seed"] + 100003 * card).shuffle(expected)
        observed = [r["load"]["profile_group"] for r in card_rows]
        check(observed == expected == manifest["metadata"]["queues"][card], "Per-card shuffled queue changed")
    check(fingerprint.hexdigest() == manifest["input_fingerprint"], "Independent input fingerprint mismatch")
    return data, dict(request_count=len(rows), request_id_sha256=object_sha(sorted(by_id)),
        independent_fingerprint=fingerprint.hexdigest(), blocks_total=sum(blocks_by_npu_disk.values()),
        bytes_total=sum(volume_by_disk), bytes_by_disk=volume_by_disk, partial_tail_count_per_reused_layer=tails,
        per_npu_request_counts={str(n): len(r) for n, r in by_npu.items()}), blocks_by_npu_disk


def sample_slo(rows, data):
    def one(selected):
        ratios, exact, tolerant, near = [], 0, 0, []
        for r in selected:
            base = 8 * data[r["request_id"]]["compute_layer_ms"]
            latency = r["completion_time_ms"] - r["admission_time_ms"]
            difference = latency - 1.5 * base
            ratios.append(latency / base)
            exact += difference <= 0
            tolerant += difference <= SLO_EPS_MS
            if abs(difference) < TIME_TOL:
                near.append(r["request_id"])
        return dict(count=len(selected), passed=tolerant, exact_no_tolerance_passed=exact,
                    rate=tolerant / len(selected) if selected else None,
                    numerical_boundary_request_ids=near,
                    max_ratio=max(ratios) if ratios else None)
    return {**one(rows), "by_group": {g: one([r for r in rows if data[r["request_id"]]["role"] == g]) for g in ("A", "B")}}


def validate_timing(summary, data):
    rows = summary["request_metrics"]
    request_by_id = {r["request_id"]: r for r in rows}
    check(len(request_by_id) == len(rows) == len(data) and set(request_by_id) == set(data), "Completed population differs")
    check(summary["n_layers"] == 8 and summary["batch_size"] == 1, "Layer/batch settings changed")
    batch_by_id, per_card = {}, defaultdict(list)
    max_c_error, max_ready_lag = 0.0, 0.0
    for b in summary["microbatch_metrics"]:
        check(len(b["member_request_ids"]) == 1, "Unexpected microbatch member count")
        rid = b["member_request_ids"][0]
        check(rid not in batch_by_id, "Repeated request microbatch")
        r, p = request_by_id[rid], data[rid]
        check(r["npu_id"] == b["npu_id"] == p["npu"], "Execution NPU changed")
        close(b["admission_time_ms"], r["admission_time_ms"], "Admission metrics differ")
        close(b["completion_time_ms"], r["completion_time_ms"], "Completion metrics differ")
        close(r["own_compute_ms"], 8 * p["compute_layer_ms"], "SLO own-compute denominator changed")
        check(r["io_count"] == p["blocks_per_layer"] * 8, "Per-request completed block count differs")
        layers = b["layer_metrics"]
        check([l["layer"] for l in layers] == list(range(8)), "Missing/duplicate layers")
        previous_end, previous_start = b["admission_time_ms"], None
        for layer in layers:
            start, end = layer["compute_start_ms"], layer["compute_end_ms"]
            check(all(math.isfinite(x) for x in (start, end, layer["io_start_time_ms"], layer["io_ready_time_ms"])), "Nonfinite event time")
            duration = end - start
            max_c_error = max(max_c_error, abs(duration - p["compute_layer_ms"]))
            close(duration, p["compute_layer_ms"], "Layer elapsed compute differs from profile")
            close(layer["compute_duration_ms"], p["compute_layer_ms"], "Layer stored compute differs from profile")
            check(start >= previous_end - TIME_TOL, "Overlapping per-card layer computes")
            check(layer["io_start_time_ms"] <= layer["io_ready_time_ms"] + TIME_TOL, "IO timestamps reversed")
            max_ready_lag = max(max_ready_lag, layer["io_ready_time_ms"] - start)
            check(layer["io_ready_time_ms"] <= start + TIME_TOL, "Compute starts before I/O ready")
            close(start - previous_end, layer["io_barrier_wait_ms"], "Unexplained delay outside compute/IO stall")
            if previous_start is not None:
                close(layer["io_start_time_ms"], previous_start, "Layerwise prefetch timing changed")
            previous_end, previous_start = end, start
        close(b["completion_time_ms"], previous_end, "Request does not finish with layer7 compute")
        batch_by_id[rid] = b
        per_card[p["npu"]].append(r)
    check(set(batch_by_id) == set(data), "Microbatch population differs")
    max_gap = 0.0
    for card, rr in per_card.items():
        previous_completion = 0.0
        for r in sorted(rr, key=lambda x: x["request_id"]):
            max_gap = max(max_gap, abs(r["admission_time_ms"] - previous_completion))
            close(r["admission_time_ms"], previous_completion, "FIFO per-card admission gap/reordering")
            previous_completion = r["completion_time_ms"]
    end = max(r["completion_time_ms"] for r in rows)
    close(end, summary["makespan_ms"], "Makespan does not equal last completion")
    return batch_by_id, dict(max_layer_compute_error_ms=max_c_error,
        max_io_ready_after_compute_start_ms=max_ready_lag, max_inter_request_admission_gap_ms=max_gap)


def window_stats(lo, hi, data, request_rows, batches, n):
    compute, active = [0.0] * n, [0.0] * n
    group_compute = {g: [0.0] * n for g in ("A", "B")}
    group_active = {g: [0.0] * n for g in ("A", "B")}
    for rid, b in batches.items():
        p, card = data[rid], data[rid]["npu"]
        c = math.fsum(overlap(l["compute_start_ms"], l["compute_end_ms"], lo, hi) for l in b["layer_metrics"])
        a = overlap(b["admission_time_ms"], b["completion_time_ms"], lo, hi)
        compute[card] += c
        active[card] += a
        group_compute[p["role"]][card] += c
        group_active[p["role"]][card] += a
    duration = hi - lo
    group_result = {}
    for g in ("A", "B"):
        c, a = math.fsum(group_compute[g]), math.fsum(group_active[g])
        group_result[g] = dict(compute_card_ms=c, active_card_ms=a,
            compute_active_utilization=c / a if a else None)
    per_npu = [dict(npu_id=card, utilization=compute[card] / duration,
        active_card_ms=active[card], compute_card_ms=compute[card],
        compute_ms_by_group={g: group_compute[g][card] for g in ("A", "B")},
        active_ms_by_group={g: group_active[g][card] for g in ("A", "B")}) for card in range(n)]
    inactive = [card for card, a in enumerate(active) if abs(a - duration) > 1e-6]
    no_mixed_compute = [card for card in range(n) if any(group_compute[g][card] <= 1e-9 for g in ("A", "B"))]
    return dict(start_ms=lo, end_ms=hi, fleet_utilization=math.fsum(compute) / (n * duration),
        compute_card_ms=math.fsum(compute), active_card_ms=math.fsum(active), per_npu=per_npu,
        by_group=group_result, all_npus_active_whole_window=not inactive,
        not_active_whole_window_npus=inactive, all_npus_compute_both_groups=not no_mixed_compute,
        missing_compute_group_npus=no_mixed_compute,
        slo_admitted=sample_slo([r for r in request_rows if lo <= r["admission_time_ms"] < hi], data))


def demand_intervals(request_rows, data, disks, makespan):
    # Unlike the producer's incremental floating sum, rebuild each interval's
    # demand directly from its current active request set after grouped edges.
    events = defaultdict(lambda: [[], []])
    events[0.0]
    events[makespan]
    for r in request_rows:
        events[r["admission_time_ms"]][1].append(r["request_id"])
        events[r["completion_time_ms"]][0].append(r["request_id"])
    points, active, result = sorted(events), {}, []
    for index, t in enumerate(points):
        for rid in events[t][0]:
            card = data[rid]["npu"]
            check(active.pop(card) == rid, "Bad request completion/active set")
        for rid in events[t][1]:
            card = data[rid]["npu"]
            check(card not in active, "Two active requests on same NPU")
            active[card] = rid
        if index + 1 < len(points):
            rates = [math.fsum(data[rid]["rates"][d] for rid in active.values()) for d in range(disks)]
            result.append((t, points[index + 1], rates))
    check(not active, "Active requests remain after last completion")
    return result


def demand_window(intervals, lo, hi, disks):
    peak, minimum, integral, over = [0.0] * disks, [math.inf] * disks, [0.0] * disks, [0.0] * disks
    any_over, total_over, peak_total = 0.0, 0.0, 0.0
    for a, b, values in intervals:
        duration = overlap(a, b, lo, hi)
        if not duration:
            continue
        flags = [v > 40 + 1e-9 for v in values]
        any_over += duration * any(flags)
        total_over += duration * (math.fsum(values) > disks * 40 + 1e-9)
        peak_total = max(peak_total, math.fsum(values))
        for disk, v in enumerate(values):
            peak[disk] = max(peak[disk], v)
            minimum[disk] = min(minimum[disk], v)
            integral[disk] += v * duration
            over[disk] += duration * flags[disk]
    return dict(peak_gb_s_by_ssu=peak, min_gb_s_by_ssu=minimum,
        mean_gb_s_by_ssu=[v / (hi - lo) for v in integral], overload_ms_by_ssu=over,
        overload_fraction_by_ssu=[v / (hi - lo) for v in over], any_ssu_overload_ms=any_over,
        any_ssu_overload_fraction=any_over / (hi - lo), aggregate_overload_ms=total_over,
        peak_total_gb_s=peak_total, strictly_under_capacity=all(v < 40 - 1e-9 for v in peak))


def compare_window(calculated, published, label):
    for key in ("fleet_utilization", "compute_card_ms", "active_card_ms"):
        close(calculated[key], published[key], f"{label}.{key}")
    check(calculated["all_npus_active_whole_window"] == published["all_npus_active_whole_window"], f"{label} active coverage mismatch")
    check(calculated["all_npus_compute_both_groups"] == published["all_npus_compute_both_groups"], f"{label} actual compute mixture mismatch")
    for own, theirs in zip(calculated["per_npu"], published["per_npu"]):
        check(own["npu_id"] == theirs["npu_id"], "Per-NPU metrics order mismatch")
        for key in ("utilization", "compute_card_ms", "active_card_ms"):
            close(own[key], theirs[key], f"{label}.NPU{own['npu_id']}.{key}")
        for g in ("A", "B"):
            close(own["compute_ms_by_group"][g], theirs["by_group"][g]["compute_card_ms"], f"{label} per-card group compute")
            close(own["active_ms_by_group"][g], theirs["by_group"][g]["active_card_ms"], f"{label} per-card group active")
    for g in ("A", "B"):
        for key, value in calculated["by_group"][g].items():
            expected = published["by_group"][g][key]
            if value is None:
                check(expected is None, f"{label}.{g}.{key} empty mismatch")
            else:
                close(value, expected, f"{label}.{g}.{key}")
    compare_slo(calculated["slo_admitted"], published["slo_admitted"], label + ".slo")
    for key, value in calculated["ordinary_demand"].items():
        if key == "min_gb_s_by_ssu":
            continue  # Added independent diagnostic, absent from old producer.
        expected = published["ordinary_demand"][key]
        if isinstance(value, list):
            for a, b in zip(value, expected):
                close(a, b, label + ".demand." + key, absolute=BW_TOL)
        elif isinstance(value, bool):
            check(value == expected, label + ".demand." + key)
        else:
            close(value, expected, label + ".demand." + key, absolute=BW_TOL)


def compare_slo(own, published, label):
    for a, b, name in ((own, published, "all"), *((own["by_group"][g], published["by_group"][g], g) for g in ("A", "B"))):
        check(a["count"] == b["count"] and a["passed"] == b["passed"], f"{label}.{name}: SLO numerator/population mismatch")


def audit_case(directory, config, deployed):
    metrics, config_saved = read(directory / "metrics.json"), read(directory / "config.json")
    manifest, summary = read(directory / "manifest.json.gz"), read(directory / "native_summary.json.gz")
    stats, command = read(directory / "adapter_statistics.json"), read(directory / "command.json")
    normalized_config = dict(config)
    normalized_config.setdefault("disk_gb_s", 40.0)
    normalized_config.setdefault("npu_gb_s", 50.0)
    check(config_saved == normalized_config, "Saved/input config changed beyond declared unit defaults")
    data, input_result, expected_disk_blocks = validate_input(manifest, normalized_config)
    input_fp, manifest_sha = input_result["independent_fingerprint"], sha(directory / "manifest.json.gz")
    check(input_fp == summary["input_fingerprint"] == metrics["input_fingerprint"], "Summary/metrics fingerprint mismatch")
    check(manifest_sha == metrics["manifest_sha256"] == command["manifest_sha256"], "Manifest SHA mismatch")
    check(metrics["source_sha256"] == command["source_sha256"] == read(directory / "source_sha256.json"), "Saved source tables differ")
    for name, value in metrics["source_sha256"].items():
        check(sha(PROJECT / name) == value, f"Current source SHA mismatch: {name}")
        check(deployed[name] == value, f"Local/remote deployment source mismatch: {name}")
    check(metrics["source_unchanged"] and all(summary["invariants"].values()), "Producer invariant/source check failed")
    blocks = input_result["blocks_total"]
    check(summary["request_count"] == input_result["request_count"], "Request count mismatch")
    check(summary["submitted_blocks"] == summary["completed_blocks"] == blocks, "Global blocks not conserved")
    check(stats["reserved_blocks"] == stats["acknowledged_blocks"] == blocks, "Adapter blocks not conserved")
    check(stats["ledger_end_counts_by_ssu"] == [0] * config["ssu"] and stats["min_ledger_count"] == 0, "Unfinished/negative routing ledger")
    check(not stats["cir_write_events"] and summary["cir_path_writes"] == 0, "Unexpected dynamic CIR changes")
    close(summary["completed_read_gb"] * 2**30, input_result["bytes_total"], "Completed bytes", absolute=1e-2, relative=1e-8)
    for disk in summary["disk_stats"]:
        d = disk["ssu_id"]
        close(disk["completed_gb"] * 2**30, input_result["bytes_by_disk"][d], "Per-disk completed bytes", absolute=1e-2, relative=1e-8)
        check(disk["max_backend_active_io"] == 1, "More than one SSD backend command")
    observed = Counter()
    layer0_observed = Counter()
    strategy = metrics["strategy"]
    for rows, counter in ((metrics["observed_paths"], observed), (metrics["observed_layer0_paths"], layer0_observed)):
        for r in rows:
            n, d, p = r["npu"], r["ssu"], r["path"]
            counter[(n, d)] += r["blocks"]
            if strategy == "asu_baseline":
                check(p == 0, "ASU used a nonzero path")
            elif strategy == "od_baseline":
                check(p == (n % 8) * 32 + n // 8, "OD path belongs to another NPU")
    check(observed == expected_disk_blocks, "Observed per-NPU/disk total blocks differ")
    check(layer0_observed == Counter({k: v // 8 for k, v in expected_disk_blocks.items()}), "Layer0 ownership/count mismatch")
    check(metrics["queue_depth_limit"] is None and command["od_queue_depth_limit"] is None, "Finite depth introduced")
    if strategy == "od_baseline":
        qos = metrics["static_qos"]
        paths = {(n % 8) * 32 + n // 8 for n in range(32)}
        for path, value in enumerate(qos["path_cirs"]):
            close(value, 40e9 / 2**30 / 32 if path in paths else 0.0, "OD equal-CIR table", absolute=1e-12)
            check(qos["path_weights"][path] == int(path in paths), "Unused OD path weight")
        check(all(math.isinf(v) for v in qos["path_pirs"]), "OD PIR capped")
    else:
        qos = metrics["static_qos"]
        for path, value in enumerate(qos["path_cirs"]):
            slot = path % 32
            decimal_cir = 20 / 96 if slot < 12 else 6 / 32 if slot < 16 else 8 / 96 if slot < 28 else 6 / 32
            close(value, decimal_cir * 1e9 / 2**30, "Archived category-pool CIR changed", absolute=1e-12)
        check(all(math.isinf(v) for v in qos["path_pirs"]), "Archived category-pool PIR changed")
    for value in summary["actual_cir_sum_gbps_by_ssu_at_stop"]:
        close(value, 40e9 / 2**30, "Actual per-disk CIR total changed", absolute=1e-10)
    batches, timing_result = validate_timing(summary, data)
    rows, end = summary["request_metrics"], summary["makespan_ms"]
    lo, hi = config["window_ms"]
    check(0 <= lo < hi <= end, "Warm window outside complete trace")
    events = demand_intervals(rows, data, config["ssu"], end)
    windows = {}
    for label, a, b in (("warm", lo, hi), ("full", 0.0, end)):
        windows[label] = window_stats(a, b, data, rows, batches, 32)
        windows[label]["ordinary_demand"] = demand_window(events, a, b, config["ssu"])
        compare_window(windows[label], metrics[label], label)
    full_slo = sample_slo(rows, data)
    compare_slo(full_slo, metrics["slo_all_requests"], "all_input_slo")
    close(100 * windows["warm"]["fleet_utilization"], metrics["warm_U_percent"], "Warm U alias")
    close(100 * windows["full"]["fleet_utilization"], metrics["full_U_percent"], "Full U alias")
    close(windows["full"]["fleet_utilization"], summary["fleet_npu_compute_utilization"], "Native full U scalar", absolute=1e-12)
    close(full_slo["rate"] * 100, metrics["full_SLO_1p5_percent"], "Full SLO alias")
    if windows["warm"]["slo_admitted"]["rate"] is not None:
        close(windows["warm"]["slo_admitted"]["rate"] * 100, metrics["warm_SLO_1p5_percent"], "Warm SLO alias")
    published_events = read(directory / "nominal_demand_events.json.gz")["intervals"]
    check(len(events) == len(published_events), "Exported demand event count mismatch")
    max_demand_error = 0.0
    for (a, b, values), exported in zip(events, published_events):
        check(a == exported["start_ms"] and b == exported["end_ms"], "Exported demand event boundary changed")
        for own, theirs in zip(values, exported["demand_decimal_gb_s_by_ssu"]):
            max_demand_error = max(max_demand_error, abs(own - theirs))
            close(own, theirs, "Exported nominal demand", absolute=BW_TOL)
    if config["ssu"] == 6 and config["id"] != "R32_A200_B10":
        check(windows["full"]["ordinary_demand"]["strictly_under_capacity"], "Main five-group run violates strict per-disk nominal underload")
    check(windows["warm"]["all_npus_active_whole_window"], "Not all 32 cards active throughout warm window")
    return dict(case=config["name"], group=config["id"], seed=config["seed"], strategy=strategy,
        num_npu=32, num_ssu=config["ssu"], passed=True, input=input_result,
        manifest_sha256=manifest_sha, config_sha256=sha(directory / "config.json"),
        input_config_sha256=metrics["config_sha256"], source_sha256=metrics["source_sha256"],
        native_summary_sha256=sha(directory / "native_summary.json.gz"), timing=timing_result,
        makespan_ms=end, warm=windows["warm"], full=windows["full"], slo_all_requests=full_slo,
        max_exported_demand_difference_gb_s=max_demand_error,
        physical_work_conservation_pass=True, exclusive_od_or_path0_asu_audit_pass=True)


def preserved_parity():
    result = {}
    for strategy in ("asu_baseline", "once"):
        directory = HERE / "validation" / ("parity8_checked_" + strategy)
        p = read(directory / "parity.json")
        check(sha(p["reference"]) == p["reference_sha256"], "Archived parity reference changed")
        check(all(p["fields"].values()), "Archived eight-card parity failed")
        for field in ("request_metrics", "microbatch_metrics"):
            check(p["output_sha256"][field] == p["reference_field_sha256"][field], "Original timing parity SHA mismatch")
        result[strategy] = dict(path=str((directory / "parity.json").relative_to(HERE)),
            all_checks_pass=True, request_metrics_exact=True, layer_metrics_exact=True,
            scalar_aggregation_difference=p["scalar_aggregation_difference"],
            reference_sha256=p["reference_sha256"], runner_sha256=read(directory / "metrics.json")["source_sha256"]["results/formula_ab_32npu_20260921/runner.py"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    args = parser.parse_args()
    configs = { (c["id"], c["seed"], c["ssu"]): (path, c)
                for path in sorted((HERE / "configs").glob("*.json")) for c in [read(path)] }
    expected = {(g, s, 3 if g == "R32_A200_B10" else 6) for g in GROUPS for s in SEEDS}
    expected |= {("XY12_16", s, 4) for s in SEEDS}
    check(set(configs) == expected, "Formal grid differs from 54 main + 9 four-disk controls")
    jobs = [(path, config, strategy, HERE / "runs" / path.stem / strategy)
            for path, config in configs.values() for strategy in STRATEGIES]
    missing = [str(directory.relative_to(HERE)) for _, _, _, directory in jobs if not (directory / "metrics.json").exists()]
    if missing:
        print(json.dumps(dict(status="incomplete", expected=63, missing_count=len(missing), missing=missing), ensure_ascii=False))
        return 2  # No partial verification file and no repeated polling.
    deployed = read(HERE / "deployment_source_manifest.json")["files"]
    results, failures, global_sources, pairings = [], [], {}, {}
    for path, config, strategy, directory in jobs:
        try:
            result = audit_case(directory, config, deployed)
            check(result["input_config_sha256"] == sha(path), "Original config SHA changed")
            for name, value in result.pop("source_sha256").items():
                check(name not in global_sources or global_sources[name] == value, "Source differs across hosts/strategies")
                global_sources[name] = value
            results.append(result)
            pairings.setdefault(path.stem, []).append(result)
            print(json.dumps(dict(event="audited", case=path.stem, strategy=strategy,
                warm_U_percent=100 * result["warm"]["fleet_utilization"],
                all_warm_compute_AB=result["warm"]["all_npus_compute_both_groups"])), flush=True)
        except Exception as error:
            failures.append(dict(case=path.stem, strategy=strategy, error=repr(error)))
            print(json.dumps(dict(event="audit_failure", **failures[-1])), flush=True)
    pairing_checks = []
    for name, rr in pairings.items():
        passed = len(rr) == 3 and all(len({r[key] for r in rr}) == 1
            for key in ("manifest_sha256", "config_sha256", "input_config_sha256"))
        passed = passed and len({r["input"]["independent_fingerprint"] for r in rr}) == 1
        passed = passed and len({r["input"]["request_id_sha256"] for r in rr}) == 1
        passed = passed and len({r["input"]["blocks_total"] for r in rr}) == 1
        passed = passed and len({r["input"]["bytes_total"] for r in rr}) == 1
        pairing_checks.append(dict(case=name, same_input_config_population_and_work=passed))
        if not passed:
            failures.append(dict(case=name, error="Three-strategy paired input mismatch/missing successful audit"))
    parity = preserved_parity()
    # Re-read each shared source at the end, rather than only trusting the
    # memoized checks performed as cases were consumed.
    sha.cache_clear()
    for name, value in global_sources.items():
        if sha(PROJECT / name) != value:
            failures.append(dict(source=name, error="Source changed during audit"))
    verification = dict(schema_version=1, complete=len(results) == 63, passed=not failures and len(results) == 63,
        expected_cases=63, audited_cases=len(results), failures=failures,
        audit_source_sha256=sha(Path(__file__)), original_8npu_parity=parity,
        recomputation="Independent standard-library implementation; no producer/simulator/audit helper imports",
        definitions=dict(time="ms", bandwidth="decimal GB/s", windows="warm [2000,4000); full [0,last completion)",
            utilization="overlap of physical compute intervals/(32*window duration)",
            SLO="completion-admission <= 1.5*(8*profile layer compute); numerical tolerance 1e-9 ms; full includes startup",
            demand="sum of each currently admitted request's exact per-disk bytes/C; extra next-request L0 demand not added; not actual SSD service",
            coverage="AB compute coverage is reported separately from active admission/completion occupancy"),
        pairing_checks=pairing_checks, source_sha256_current_and_across_hosts=global_sources,
        warm_missing_AB_coverage=[dict(case=r["case"], strategy=r["strategy"], npus=r["warm"]["missing_compute_group_npus"])
            for r in results if not r["warm"]["all_npus_compute_both_groups"]],
        cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(dict(status="PASS" if verification["passed"] else "FAIL", output=str(args.output),
        audited=len(results), failures=len(failures), warm_mix_incomplete_cases=len(verification["warm_missing_AB_coverage"]))))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
