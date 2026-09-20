"""Post-process native finite baseline/Once summaries without simulator imports.

``analyze(summary, requests_meta, config)`` accepts metadata keyed by integer or
JSON-string request IDs. Volumes in the metadata are GiB, timestamps are ms,
and every returned bandwidth is decimal GB/s. Intervals are half-open.

Two demand definitions are deliberately retained: active request profile demand
and the conservative pending *internal-layer* profile demand. Neither is actual
instantaneous service bandwidth or remaining-bytes/remaining-deadline demand.
"""
from __future__ import annotations

import math
from collections import defaultdict


GIB_TO_GB = 2**30 / 1e9
EPS_MS = 1e-9
EPS_BW = 1e-9


def _overlap(a, b, start, end):
    return max(0.0, min(float(b), end) - max(float(a), start))


def _ratio(value, denominator):
    return value / denominator if denominator > 0 else None


def _meta_for(metadata, request_id):
    if request_id in metadata:
        return metadata[request_id]
    return metadata[str(request_id)]


def _slo(rows, metadata, alpha, horizon_end):
    def one(sample):
        passed = sum(
            float(row["completion_time_ms"]) - float(row["admission_time_ms"])
            <= alpha * float(row["own_compute_ms"]) + EPS_MS
            for row in sample
        )
        return {
            "count": len(sample), "passed": passed,
            "rate": _ratio(passed, len(sample)),
            "completion_after_window_end_count": sum(
                float(row["completion_time_ms"]) > horizon_end + EPS_MS
                for row in sample
            ),
        }
    by_group = {
        group: one([
            row for row in rows
            if str(_meta_for(metadata, row["request_id"])["profile_group"]) == group
        ]) for group in ("A", "B")
    }
    return {**one(rows), "by_group": by_group}


def _demand_scan(intervals, ssu_count, start, end, capacity, near_capacity):
    """Aggregate same-time edges before observing an interval's demand.

    Each interval is (start, end, vector). A zero-duration intermediate state at
    simultaneous completion/admission cannot create a false peak or overload.
    """
    events = defaultdict(lambda: [0.0] * ssu_count)
    events[start]
    events[end]
    for a, b, rates in intervals:
        a, b = max(start, float(a)), min(end, float(b))
        if b <= a:
            continue
        for s, rate in enumerate(rates):
            events[a][s] += rate
            events[b][s] -= rate
    points = sorted(events)
    demand = [0.0] * ssu_count
    peaks = [0.0] * ssu_count
    peak_at = [start] * ssu_count
    over = [0.0] * ssu_count
    near = [0.0] * ssu_count
    integral = [0.0] * ssu_count
    any_over = any_near = total_over = total_near = 0.0
    peak_total = 0.0
    peak_total_at = start
    first_overload = None
    for index, t in enumerate(points[:-1]):
        for s in range(ssu_count):
            demand[s] += events[t][s]
            if abs(demand[s]) < 1e-10:
                demand[s] = 0.0
            if demand[s] < -1e-7:
                raise AssertionError("negative demand after event aggregation")
        dt = points[index + 1] - t
        if dt <= 0.0:
            continue
        overloaded = []
        near_flags = []
        for s, bw in enumerate(demand):
            if bw > peaks[s]:
                peaks[s], peak_at[s] = bw, t
            is_over = bw > capacity + EPS_BW
            is_near = bw >= near_capacity - EPS_BW
            over[s] += dt * is_over
            near[s] += dt * is_near
            integral[s] += bw * dt
            overloaded.append(is_over)
            near_flags.append(is_near)
        total = math.fsum(demand)
        if total > peak_total:
            peak_total, peak_total_at = total, t
        any_over += dt * any(overloaded)
        any_near += dt * any(near_flags)
        total_over += dt * (total > ssu_count * capacity + EPS_BW)
        total_near += dt * (total >= ssu_count * near_capacity - EPS_BW)
        if first_overload is None and any(overloaded):
            first_overload = {
                "start_ms": t, "end_ms": points[index + 1],
                "demand_gb_s_by_ssu": list(demand),
            }
    duration = end - start
    return {
        "start_ms": start, "end_ms": end, "duration_ms": duration,
        "bandwidth_unit": "decimal GB/s",
        "per_ssu_capacity_gb_s": capacity,
        "near_capacity_threshold_gb_s": near_capacity,
        "peak_gb_s_by_ssu": peaks,
        "peak_at_ms_by_ssu": peak_at,
        "max_single_ssu_gb_s": max(peaks, default=0.0),
        "peak_total_gb_s": peak_total,
        "peak_total_at_ms": peak_total_at,
        "mean_gb_s_by_ssu": [_ratio(v, duration) for v in integral],
        "overload_ms_by_ssu": over,
        "overload_fraction_by_ssu": [_ratio(v, duration) for v in over],
        "at_or_above_near_ms_by_ssu": near,
        "any_ssu_overload_ms": any_over,
        "any_ssu_overload_fraction": _ratio(any_over, duration),
        "any_ssu_at_or_above_near_ms": any_near,
        "any_ssu_at_or_above_near_fraction": _ratio(any_near, duration),
        "aggregate_overload_ms": total_over,
        "aggregate_at_or_above_near_ms": total_near,
        "no_ssu_overload": any_over <= EPS_MS,
        "strictly_under_capacity": all(v < capacity - EPS_BW for v in peaks),
        "first_overload_interval": first_overload,
    }


def _composition_scan(intervals, start, end, num_npu):
    events = defaultdict(lambda: [0, 0])
    events[start]
    events[end]
    for a, b, group in intervals:
        a, b = max(start, float(a)), min(end, float(b))
        if b <= a:
            continue
        index = 0 if group == "A" else 1
        events[a][index] += 1
        events[b][index] -= 1
    points = sorted(events)
    counts = [0, 0]
    joint = defaultdict(float)
    for index, t in enumerate(points[:-1]):
        counts = [counts[j] + events[t][j] for j in range(2)]
        if min(counts) < 0 or sum(counts) > num_npu:
            raise AssertionError("invalid active-request concurrency")
        joint[tuple(counts)] += points[index + 1] - t
    duration = end - start
    na = [0.0] * (num_npu + 1)
    for (a, _b), dt in joint.items():
        na[a] += dt
    return {
        "definition": "Admitted current requests, including compute and I/O stall",
        "nA_duration_ms": na,
        "nA_time_fraction": [_ratio(dt, duration) for dt in na],
        "joint_nA_nB": [
            {"n_A": a, "n_B": b, "idle_npu": num_npu-a-b,
             "duration_ms": dt, "time_fraction": _ratio(dt, duration)}
            for (a, b), dt in sorted(joint.items()) if dt > 0.0
        ],
    }


def _window_metrics(batches, request_rows, metadata, start, end, num_npu, alpha):
    compute = [0.0] * num_npu
    active = [0.0] * num_npu
    groups = {g: {"compute_card_ms": 0.0, "active_card_ms": 0.0,
                  "active_request_count": 0, "admitted_count": 0,
                  "completed_count": 0} for g in ("A", "B")}
    roles = [set() for _ in range(num_npu)]
    per_npu_groups = [{g: {"compute_card_ms": 0.0, "active_card_ms": 0.0}
                       for g in ("A", "B")} for _ in range(num_npu)]
    stalls = {kind: {"card_ms": 0.0, "count": 0, "by_group": {
        g: {"card_ms": 0.0, "count": 0} for g in ("A", "B")}}
        for kind in ("internal", "layer0")}
    for batch in batches:
        ids = batch["member_request_ids"]
        if len(ids) != 1:
            raise ValueError("audit expects batch_size=1")
        rid = ids[0]
        meta = _meta_for(metadata, rid)
        group = str(meta["profile_group"])
        if group not in groups:
            raise ValueError("profile_group must be A or B")
        npu = int(batch["npu_id"])
        a, b = float(batch["admission_time_ms"]), float(batch["completion_time_ms"])
        a_ms = _overlap(a, b, start, end)
        active[npu] += a_ms
        groups[group]["active_card_ms"] += a_ms
        per_npu_groups[npu][group]["active_card_ms"] += a_ms
        if a_ms > EPS_MS:
            roles[npu].add(group)
            groups[group]["active_request_count"] += 1
        groups[group]["admitted_count"] += int(start <= a < end)
        # Completion events use (start,end], so the final completion is counted
        # in the full run and adjacent windows do not double count events.
        groups[group]["completed_count"] += int(start < b <= end)
        for layer in batch["layer_metrics"]:
            c_start, c_end = float(layer["compute_start_ms"]), float(layer["compute_end_ms"])
            c_ms = _overlap(c_start, c_end, start, end)
            compute[npu] += c_ms
            groups[group]["compute_card_ms"] += c_ms
            per_npu_groups[npu][group]["compute_card_ms"] += c_ms
            wait = float(layer["io_barrier_wait_ms"])
            s_ms = _overlap(c_start-wait, c_start, start, end)
            if s_ms > EPS_MS:
                kind = "layer0" if int(layer["layer"]) == 0 else "internal"
                stalls[kind]["card_ms"] += s_ms
                stalls[kind]["count"] += 1
                stalls[kind]["by_group"][group]["card_ms"] += s_ms
                stalls[kind]["by_group"][group]["count"] += 1
    for stats in groups.values():
        stats["compute_active_utilization"] = _ratio(stats["compute_card_ms"], stats["active_card_ms"])
    duration = end-start
    admitted = [r for r in request_rows if start <= float(r["admission_time_ms"]) < end]
    per_npu = []
    for n in range(num_npu):
        for stats in per_npu_groups[n].values():
            stats["compute_active_utilization"] = _ratio(stats["compute_card_ms"], stats["active_card_ms"])
        per_npu.append({
            "npu_id": n, "utilization": compute[n]/duration,
            "compute_card_ms": compute[n], "active_card_ms": active[n],
            "idle_ms": max(0.0, duration-active[n]),
            "roles": sorted(roles[n]),
            "role": "+".join(sorted(roles[n])) or "idle",
            "by_group": per_npu_groups[n],
        })
    total_stall = stalls["internal"]["card_ms"] + stalls["layer0"]["card_ms"]
    return {
        "start_ms": start, "end_ms": end, "duration_ms": duration,
        "fleet_utilization": math.fsum(compute)/(num_npu*duration),
        "compute_card_ms": math.fsum(compute), "active_card_ms": math.fsum(active),
        "all_npus_active_whole_window": all(abs(a-duration) < 1e-6 for a in active),
        "per_npu": per_npu, "by_group": groups, "stall": stalls,
        "total_stall_card_ms": total_stall,
        "internal_stall_fraction": _ratio(stalls["internal"]["card_ms"], total_stall),
        "accounting_error_card_ms": math.fsum(active)-math.fsum(compute)-total_stall,
        "slo_admitted": _slo(admitted, metadata, alpha, end),
    }


def analyze(summary, requests_meta, config):
    """Return JSON-safe exact-window utilization, SLO and demand audits.

    Config: ``ssu`` (required), ``window_ms`` (default [2000,4000]),
    ``num_npu`` (default 8), ``disk_gb_s`` (default 40),
    ``near_gb_s`` (default 36), ``slo_alpha`` (default 1.5).
    Every metadata row needs profile_group, per_layer_us, disk_gib; the other
    requested metadata fields are retained by the caller and need not be read.
    """
    num_npu = int(config.get("num_npu", 8))
    ssu = int(config["ssu"])
    warm_start, warm_end = map(float, config.get("window_ms", [2000, 4000]))
    end = float(summary["makespan_ms"])
    cap = float(config.get("disk_gb_s", 40.0))
    near = float(config.get("near_gb_s", 36.0))
    alpha = float(config.get("slo_alpha", 1.5))
    if not 0 <= warm_start < warm_end <= end + EPS_MS:
        raise ValueError("warm window must lie inside the completed finite run")
    if int(summary["num_npu"]) != num_npu or int(summary["num_ssu"]) != ssu:
        raise ValueError("config topology differs from summary")
    batches, rows = summary["microbatch_metrics"], summary["request_metrics"]
    ordinary, pending, composition = [], [], []
    volume_errors = []
    for batch in batches:
        if len(batch["member_request_ids"]) != 1:
            raise ValueError("audit expects batch_size=1")
        rid = batch["member_request_ids"][0]
        meta = _meta_for(requests_meta, rid)
        c_s = float(meta["per_layer_us"])/1e6
        disk_gib = list(map(float, meta["disk_gib"]))
        if len(disk_gib) != ssu or c_s <= 0 or any(v < 0 for v in disk_gib):
            raise ValueError("invalid metadata compute time or disk volume vector")
        if "per_layer_kv_gb" in meta:
            volume_errors.append(abs(math.fsum(disk_gib)-float(meta["per_layer_kv_gb"])))
        rates = tuple(v*GIB_TO_GB/c_s for v in disk_gib)
        a, b = float(batch["admission_time_ms"]), float(batch["completion_time_ms"])
        ordinary.append((a, b, rates))
        composition.append((a, b, str(meta["profile_group"])))
        for layer in batch["layer_metrics"]:
            if int(layer["layer"]) >= 1:
                pending.append((float(layer["io_start_time_ms"]),
                                float(layer["io_ready_time_ms"]), rates))
    result = {
        "schema_version": 1, "num_npu": num_npu, "num_ssu": ssu,
        "definitions": {
            "utilization": "Exact overlap of compute intervals / (NPU count * wall window)",
            "group_utilization": "Group compute overlap / group admission-to-completion overlap",
            "stall_count": "Number of positive stall intervals intersecting the window",
            "count_windows": "admitted_count uses [start,end); completed_count uses (start,end]",
            "slo": "completion - admission <= alpha * own_compute_ms; final completions retained",
            "slo_alpha": alpha,
            "ordinary_demand": "Current request disk_GiB / own layer compute time over [admission,completion)",
            "ordinary_pending_demand": "Only k>=1: disk_GiB / own C over [io_start,all-SSU io_ready); conservative per-SSU envelope",
            "boundary_exemption": "No extra next-request L0 burst term; no entire time interval is removed",
            "demand_caveat": "Nominal profile demand, not physical service or remaining-bytes/remaining-deadline demand",
        },
        "input_audit": {"request_count": len(rows),
                        "max_layer_volume_sum_error_gib": max(volume_errors, default=0.0)},
        "slo_all_requests": _slo(rows, requests_meta, alpha, end),
    }
    for name, start, stop in (("full", 0.0, end), ("warm", warm_start, warm_end)):
        metrics = _window_metrics(batches, rows, requests_meta, start, stop, num_npu, alpha)
        metrics["ordinary_demand"] = _demand_scan(ordinary, ssu, start, stop, cap, near)
        metrics["ordinary_pending_demand"] = _demand_scan(pending, ssu, start, stop, cap, near)
        metrics["composition"] = _composition_scan(composition, start, stop, num_npu)
        result[name] = metrics
    return result
