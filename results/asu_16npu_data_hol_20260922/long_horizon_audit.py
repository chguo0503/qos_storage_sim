#!/usr/bin/env python3
"""Independent, read-only audit of completed raw layer/request records.

Does not import the runner, its audit helpers, or the simulator. Source raw
artifacts are never modified. Missing/incomplete cases are reported, not passed.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPS = 1e-8


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b, start, end):
    return max(0., min(float(b), end) - max(float(a), start))


def audit_case(path):
    required = ("native_summary.json.gz", "manifest.json.gz", "config.json", "command.json", "metrics.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return dict(status="pending", source=str(path.relative_to(ROOT)), missing=missing)
    original_sha = {name: sha(path / name) for name in required}
    s, manifest, cfg, command, reported = (read(path / name) for name in required)
    n, disks, layers = int(s["num_npu"]), int(s["num_ssu"]), int(s["n_layers"])
    end = float(s["makespan_ms"])
    meta = {int(row["request_id"]): row["load"] for row in manifest["requests"]}
    requests = s["request_metrics"]
    batches = s["microbatch_metrics"]
    request_ids = [int(row["request_id"]) for row in requests]
    checks = dict(completed_metrics_marker=reported.get("source_unchanged") is True,
        requests_all_drained=len(requests) == len(meta) == s["request_count"] == len(set(request_ids)),
        original_request_ids=set(request_ids) == set(meta),
        all_invariants=all(s["invariants"].values()),
        eight_layers=layers == 8 and all(len(batch["layer_metrics"]) == layers for batch in batches),
        manifest_sha=sha(path / "manifest.json.gz") == command["manifest_sha256"] == reported["manifest_sha256"],
        fingerprints_agree=manifest["input_fingerprint"] == s["input_fingerprint"] == reported["input_fingerprint"],
        source_hash_records_agree=command["source_sha256"] == reported["source_sha256"])
    current_sources = {p: (ROOT / p).is_file() and sha(ROOT / p) == value
                       for p, value in command["source_sha256"].items()}
    checks["source_files_currently_match"] = all(current_sources.values())
    expected_blocks = 0
    for row in manifest["requests"]:
        placement = manifest["placements"][row["placement_index"]]
        # Every stored A/B request repeats the same one-layer placement 8 times.
        assert len(placement) == 1
        expected_blocks += layers * len(placement[0])
    checks["block_conservation"] = expected_blocks == s["submitted_blocks"] == s["completed_blocks"]
    active, computes, waits = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    finish = [0.] * n
    demand_events = defaultdict(lambda: {"start": [], "end": []})
    demand_events[0.]
    demand_events[end]
    rates_by_id = {}
    for row in requests:
        rid, card = int(row["request_id"]), int(row["npu_id"])
        a, b = float(row["admission_time_ms"]), float(row["completion_time_ms"])
        group = meta[rid]["profile_group"]
        active[card].append((a, b, group, rid))
        finish[card] = max(finish[card], b)
        assert meta[rid]["npu_id"] == card
        assert math.isclose(row["own_compute_ms"], layers * meta[rid]["per_layer_us"] / 1000, abs_tol=1e-6)
        rates_by_id[rid] = [v * 2**30 / 1e9 / (meta[rid]["per_layer_us"] / 1e6) for v in meta[rid]["disk_gib"]]
        demand_events[a]["start"].append(rid)
        demand_events[b]["end"].append(rid)
    for batch in batches:
        assert len(batch["member_request_ids"]) == 1
        rid = int(batch["member_request_ids"][0]);card = int(batch["npu_id"])
        group = meta[rid]["profile_group"]
        previous = float(batch["admission_time_ms"])
        for layer in batch["layer_metrics"]:
            a, b = float(layer["compute_start_ms"]), float(layer["compute_end_ms"])
            wait = float(layer["io_barrier_wait_ms"])
            assert math.isclose(b - a, meta[rid]["per_layer_us"] / 1000, abs_tol=1e-7)
            assert math.isclose(a - previous, wait, abs_tol=1e-7)
            computes[card].append((a, b, group, rid))
            if wait > EPS:
                kind = "internal" if int(layer["layer"]) > 0 else ("startup_L0" if meta[rid]["generation"] == 0 else "cross_request_L0")
                waits[card].append((a - wait, a, kind, group, rid))
            previous = b
        assert math.isclose(previous, batch["completion_time_ms"], abs_tol=1e-7)
    for card in range(n):
        active[card].sort();computes[card].sort()
        assert active[card][0][0] == 0.
        assert all(abs(left[1] - right[0]) < EPS for left, right in zip(active[card], active[card][1:]))
        assert all(left[1] <= right[0] + EPS for left, right in zip(computes[card], computes[card][1:]))
    points = sorted(demand_events)
    current, segments = {}, []
    for i, t in enumerate(points[:-1]):
        for rid in demand_events[t]["end"]:
            del current[rid]
        for rid in demand_events[t]["start"]:
            current[rid] = rates_by_id[rid]
        stop = points[i + 1]
        if stop <= t:
            continue
        rates = [math.fsum(values[d] for values in current.values()) for d in range(disks)]
        segments.append((t, stop, rates, len(current)))

    def demand(start, stop):
        integral = [0.] * disks
        low, high = [math.inf] * disks, [-math.inf] * disks
        over, at_cap = [0.] * disks, [0.] * disks
        any_over = any_at_cap = 0.
        violations = []
        active_min = n
        for a, b, rates, num_active in segments:
            dt = overlap(a, b, start, stop)
            if dt <= 0:
                continue
            active_min = min(active_min, num_active)
            flags, equal_flags = [], []
            for d, rate in enumerate(rates):
                integral[d] += dt * rate
                low[d], high[d] = min(low[d], rate), max(high[d], rate)
                flags.append(rate > 40 + EPS)
                equal_flags.append(rate >= 40 - EPS)
                over[d] += dt * flags[-1]
                at_cap[d] += dt * equal_flags[-1]
            any_over += dt * any(flags)
            any_at_cap += dt * any(equal_flags)
            if any(equal_flags):
                violations.append(dict(start_ms=max(a, start), end_ms=min(b, stop), demand_GB_s=rates))
        duration = stop - start
        return dict(mean_GB_s_by_ssu=[v / duration for v in integral], min_GB_s_by_ssu=low,
            max_GB_s_by_ssu=high, overload_ms_by_ssu=over,
            overload_fraction_by_ssu=[v / duration for v in over], at_or_above_capacity_ms_by_ssu=at_cap,
            any_ssu_overload_fraction=any_over / duration,
            any_ssu_at_or_above_capacity_fraction=any_at_cap / duration,
            strict_under40_all_disks=all(v < 40 - EPS for v in high),
            at_or_above_capacity_intervals=violations, minimum_admitted_active_card_count=active_min)

    def window(start, stop):
        if stop > end + EPS:
            return dict(start_ms=start, end_ms=stop, status="unavailable")
        duration = stop - start
        per_card = []
        stalls = {kind: {"card_ms": 0., "interval_count": 0, "A_card_ms": 0., "B_card_ms": 0.}
                  for kind in ("internal", "startup_L0", "cross_request_L0")}
        for card in range(n):
            active_ms = math.fsum(overlap(a, b, start, stop) for a, b, _, _ in active[card])
            group_compute = {g: math.fsum(overlap(a, b, start, stop) for a, b, group, _ in computes[card] if group == g)
                             for g in ("A", "B")}
            group_ids = {g: sorted({rid for a, b, group, rid in computes[card]
                                   if group == g and overlap(a, b, start, stop) > EPS}) for g in ("A", "B")}
            card_stall = 0.
            for a, b, kind, group, _rid in waits[card]:
                dt = overlap(a, b, start, stop)
                if dt > EPS:
                    stalls[kind]["card_ms"] += dt
                    stalls[kind]["interval_count"] += 1
                    stalls[kind][group + "_card_ms"] += dt
                    card_stall += dt
            compute_ms = math.fsum(group_compute.values())
            assert abs(active_ms - compute_ms - card_stall) < 1e-5
            per_card.append(dict(npu_id=card, U_percent=100 * compute_ms / duration,
                compute_ms=compute_ms, active_ms=active_ms, stall_ms=card_stall,
                group_compute_ms=group_compute, group_computing_request_count={g: len(ids) for g, ids in group_ids.items()},
                actual_compute_both_AB=all(value > EPS for value in group_compute.values()),
                active_whole_window=abs(active_ms - duration) < 1e-6))
        cohort = [row for row in requests if start <= row["admission_time_ms"] < stop]
        passed = [row for row in cohort if row["completion_time_ms"] - row["admission_time_ms"] <= 1.5 * row["own_compute_ms"] + 1e-9]
        return dict(status="complete", start_ms=start, end_ms=stop, duration_ms=duration,
            U_percent=100 * math.fsum(card["compute_ms"] for card in per_card) / (n * duration),
            SLO_1p5_percent=100 * len(passed) / len(cohort) if cohort else None,
            admitted_count=len(cohort), passed_count=len(passed),
            cohort_completed_after_window_count=sum(row["completion_time_ms"] > stop for row in cohort),
            all_cards_active=all(card["active_whole_window"] for card in per_card),
            all_cards_actual_AB_compute=all(card["actual_compute_both_AB"] for card in per_card),
            mixed_card_count=sum(card["actual_compute_both_AB"] for card in per_card),
            ends_before_first_npu_finishes=stop <= min(finish) + EPS,
            per_npu=per_card, stall=stalls, nominal_demand=demand(start, stop))

    windows = {"warm_2_4": window(2000., 4000.), "long_2_60": window(2000., 60000.),
               "late_20_60": window(20000., 60000.), "late_40_60": window(40000., 60000.),
               "full": window(0., end)}
    bins = [window(float(a), min(float(a + 2000), end)) for a in range(0, math.ceil(end), 2000)]
    for row in bins:
        row["full_2second_window"] = abs(row["duration_ms"] - 2000) < EPS
    checks["warm_U_agrees_reported"] = abs(windows["warm_2_4"]["U_percent"] - reported["warm_U_percent"]) < 1e-8
    checks["warm_SLO_agrees_reported"] = abs(windows["warm_2_4"]["SLO_1p5_percent"] - reported["warm_SLO_1p5_percent"]) < 1e-8
    full_compute = math.fsum(row["compute_ms"] for row in windows["full"]["per_npu"])
    bins_compute = math.fsum(row["U_percent"] / 100 * n * row["duration_ms"] for row in bins)
    checks["per2s_compute_conservation"] = abs(full_compute - bins_compute) < 1e-5
    checks["per2s_SLO_population_conservation"] = sum(row["admitted_count"] for row in bins) == len(requests)
    checks["per2s_SLO_pass_conservation"] = sum(row["passed_count"] for row in bins) == windows["full"]["passed_count"]
    checks["raw_artifacts_unchanged"] = original_sha == {name: sha(path / name) for name in required}
    return dict(status="passed" if all(checks.values()) else "failed", source=str(path.relative_to(ROOT)),
        artifact_sha256=original_sha, manifest_sha256=original_sha["manifest.json.gz"],
        input_fingerprint=s["input_fingerprint"], strategy=command["strategy"], order=cfg["mode"],
        npu_count=n, ssu_count=disks, n_layers=layers, makespan_ms=end,
        request_count=len(requests), completed_blocks=expected_blocks,
        first_npu_finishes_ms=min(finish), final_completion_ms_by_npu=finish,
        per_card_input_role_counts=[dict(Counter(meta[rid]["profile_group"] for _, _, _, rid in intervals)) for intervals in active],
        checks=checks, source_hash_mismatches=[p for p, value in current_sources.items() if not value],
        exact_nominal_demand_segment_count=len(segments), windows=windows, consecutive_2s=bins,
        conclusion="An audited finite trajectory; a low early window alone does not establish long-lived low utilization.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="I1_s1_r18_two_cohorts_64s_seed7")
    ap.add_argument("--strategies", nargs="+", default=["asu_baseline", "once"])
    ap.add_argument("--output-prefix", type=Path, default=HERE / "long_horizon_audit")
    args = ap.parse_args()
    cases = {}
    for strategy in args.strategies:
        try:
            cases[strategy] = audit_case(HERE / "runs" / args.case / strategy)
        except Exception as error:
            cases[strategy] = dict(status="failed", error_type=type(error).__name__, detail=str(error))
    paired = all(row.get("status") == "passed" for row in cases.values())
    if paired:
        paired = len({row["manifest_sha256"] for row in cases.values()}) == 1
    result = dict(case=args.case, status="passed" if paired else "incomplete_or_failed", paired_identical_manifest=paired,
        audit_script_sha256=sha(Path(__file__)), definitions=dict(
            U="Exact raw compute interval intersections / (NPU count * wall window); IO waits are included in denominator",
            SLO="Uncensored admission cohort [start,end): completion-admission <=1.5*own_compute; numerator and denominator are recomputed",
            demand="Sum admitted-current-request V_disk/C over admission-to-completion, including IO stall; excludes extra cross-request L0 prefetch",
            bandwidth_unit="decimal GB/s, capacity40 per disk", bins="Half-open intervals; the last bin can be partial; no result is extrapolated",
            completeness="Completed metrics marker plus full raw requests/layers/blocks; absent/invalid cases never pass"), cases=cases)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 64秒输入的独立长窗口审计", "", "从原始层区间和请求记录重算；未导入运行器、仿真器或原统计函数。", "",
             "名义需求包含当前请求的 IO stall 时间，按历史口径排除额外跨请求 L0 预取；它不是 SSD 瞬时吞吐。", "",
             "| 策略 | 窗口(s) | NPU U | SLO×1.5 | 接纳/达标数 | 全卡活跃/实际算过AB | 每盘严格低于40 |", "|---|---|---:|---:|---:|---|---|"]
    for strategy, row in cases.items():
        if row.get("status") != "passed":
            lines += [f"| {strategy} | 未完成或审计失败 | — | — | — | — | — |"]
            continue
        for key in ("warm_2_4", "long_2_60", "late_20_60", "late_40_60"):
            w = row["windows"][key]
            lines.append(f"| {strategy} | [{w['start_ms']/1000:g},{w['end_ms']/1000:g}) | {w['U_percent']:.6f}% | {w['SLO_1p5_percent']:.6f}% | {w['admitted_count']}/{w['passed_count']} | {w['all_cards_active']}/{w['all_cards_actual_AB_compute']} | {w['nominal_demand']['strict_under40_all_disks']} |")
        d = row["windows"]["full"]["nominal_demand"]
        lines += ["", f"{strategy}：首次有卡排空为 {row['first_npu_finishes_ms']/1000:.6f} 秒，完整排空 {row['makespan_ms']/1000:.6f} 秒。全程逐盘需求峰值 {d['max_GB_s_by_ssu']} GB/s，超过40的时间比例 {d['overload_fraction_by_ssu']}。", ""]
    asu = cases.get("asu_baseline", {})
    if asu.get("status") == "passed":
        w = asu["windows"]
        lines += [f"ASU 的早期 [2,4) 利用率为 {w['warm_2_4']['U_percent']:.6f}%，而 [2,60) 为 {w['long_2_60']['U_percent']:.6f}%，[20,60) 为 {w['late_20_60']['U_percent']:.6f}%。这份输入没有保持早期的低水平，应披露为未达到长期低利用率目标的候选。", ""]
    lines += ["每两秒的完整 U/SLO/逐盘需求、warm每卡AB计算覆盖、内部层/首层等待计数与时长均在同名 JSON 中；末尾不足两秒的区间单独标记。", "", f"审计状态：`{result['status']}`；输入完全相同：`{paired}`。"]
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "cases": {strategy: {"status": row.get("status"),
        "windows": {k: {field: v.get(field) for field in ("U_percent", "SLO_1p5_percent", "all_cards_active", "mixed_card_count")} for k, v in row.get("windows", {}).items()}}
        for strategy, row in cases.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
