#!/usr/bin/env python3
"""Compare saved local and remote results without assuming phase equivalence."""

import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(directory):
    paths = sorted(directory.glob("*.json.gz"))
    if len(paths) > 1:
        raise AssertionError(f"Ambiguous source: {directory}")
    return paths[0] if paths else None


def differences(left, right):
    count = 0
    fields = Counter()
    examples = []
    largest = 0.0

    def walk(a, b, path="root"):
        nonlocal count, largest
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    record(a.get(key), b.get(key), path + "." + key)
                else:
                    walk(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                record(len(a), len(b), path + ".length")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + f"[{i}]")
        elif a != b:
            record(a, b, path)

    def record(a, b, path):
        nonlocal count, largest
        count += 1
        fields[path.rsplit(".", 1)[-1]] += 1
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            largest = max(largest, abs(a - b))
        if len(examples) < 12:
            examples.append({"path": path, "remote": a, "local": b})

    walk(left, right)
    return {"exact": count == 0, "difference_count": count,
            "different_fields": dict(fields), "max_numeric_absolute_difference": largest,
            "first_differences": examples}


def overlap(a, b, start, end):
    return max(0.0, min(b, end) - max(a, start))


def window(result, start, end):
    n = result["metadata"]["num_npu"]
    active, compute = [0.0] * n, [0.0] * n
    for batch in result["summary"]["microbatch_metrics"]:
        i = batch["npu_id"]
        active[i] += overlap(batch["admission_time_ms"], batch["completion_time_ms"], start, end)
        for layer in batch["layer_metrics"]:
            compute[i] += overlap(layer["compute_start_ms"], layer["compute_end_ms"], start, end)
    return {"utilization": sum(compute) / (n * (end - start)),
            "per_npu_utilization": [x / (end - start) for x in compute],
            "compute_npu_ms": sum(compute), "active_npu_ms": sum(active),
            "stall_npu_ms": sum(active) - sum(compute),
            "idle_npu_ms": n * (end - start) - sum(active)}


def compare(label, policy, remote_path, local_path):
    remote, local = read(remote_path), read(local_path)
    checks = {
        "same_input_fingerprint": remote["input_fingerprint"] == local["input_fingerprint"],
        "same_metadata": remote["metadata"] == local["metadata"],
        "same_strategy": remote["strategy"] == local["strategy"],
        "same_submit_seed": remote["submit_seed"] == local["submit_seed"],
        "same_collector_interval": remote["collector_interval_ms"] == local["collector_interval_ms"] == 5.0,
        "same_policy_config": remote["policy_config"] == local["policy_config"],
        "same_runner_sha256": remote["stress_runner_sha256"] == local["stress_runner_sha256"],
        "same_core_and_policy_sha256": remote["core_and_policy_sha256"] == local["core_and_policy_sha256"],
        "remote_invariants": all(remote["summary"]["invariants"].values()),
        "local_invariants": all(local["summary"]["invariants"].values()),
    }
    request_compare = differences(remote["summary"]["request_metrics"], local["summary"]["request_metrics"])
    batch_compare = differences(remote["summary"]["microbatch_metrics"], local["summary"]["microbatch_metrics"])
    by_request = [{r["request_id"]: r for r in result["summary"]["request_metrics"]} for result in (remote, local)]
    checks["same_complete_request_ids"] = set(by_request[0]) == set(by_request[1])
    matched_requests = []
    for rid in sorted(set(by_request[0]) & set(by_request[1])):
        a, b = by_request[0][rid], by_request[1][rid]
        row = {"input": label, "strategy": policy, "request_id": rid,
               "remote_npu": a["npu_id"], "local_npu": b["npu_id"], "same_npu": a["npu_id"] == b["npu_id"]}
        for field in ("admission_time_ms", "completion_time_ms", "io_stall_ms", "processing_latency_ms", "own_compute_ms"):
            row["remote_" + field] = a[field]
            row["local_" + field] = b[field]
            row["delta_" + field] = b[field] - a[field]
        matched_requests.append(row)
    rows = []
    for rw, lw in zip(remote["windows"], local["windows"]):
        assert (rw["start_ms"], rw["end_ms"]) == (lw["start_ms"], lw["end_ms"])
        start, end = rw["start_ms"], rw["end_ms"]
        rcalc, lcalc = window(remote, start, end), window(local, start, end)
        rows.append({"input": label, "strategy": policy, "start_ms": start, "end_ms": end,
            "input_fingerprint": local["input_fingerprint"], "configuration_pair_pass": all(checks.values()),
            "remote_result": str(remote_path), "local_result": str(local_path),
            "remote_python": remote["python_version"], "local_python": local["python_version"],
            "remote_saved_U": rw["mean_npu_utilization"], "local_saved_U": lw["mean_npu_utilization"],
            "saved_delta_U_pp": 100 * (lw["mean_npu_utilization"] - rw["mean_npu_utilization"]),
            "remote_U_recomputed_same_python": rcalc["utilization"],
            "local_U_recomputed_same_python": lcalc["utilization"],
            "recomputed_delta_U_pp": 100 * (lcalc["utilization"] - rcalc["utilization"]),
            "max_recomputed_per_npu_U_delta_pp": 100 * max(abs(a-b) for a,b in zip(rcalc["per_npu_utilization"], lcalc["per_npu_utilization"])),
            "remote_active_npu_ms": rcalc["active_npu_ms"], "local_active_npu_ms": lcalc["active_npu_ms"],
            "remote_stall_npu_ms": rcalc["stall_npu_ms"], "local_stall_npu_ms": lcalc["stall_npu_ms"],
            "remote_idle_npu_ms": rcalc["idle_npu_ms"], "local_idle_npu_ms": lcalc["idle_npu_ms"],
            "remote_makespan_ms": remote["summary"]["makespan_ms"],
            "local_makespan_ms": local["summary"]["makespan_ms"],
            "delta_makespan_ms": local["summary"]["makespan_ms"] - remote["summary"]["makespan_ms"],
            "remote_full_run_U": remote["summary"]["fleet_npu_compute_utilization"],
            "local_full_run_U": local["summary"]["fleet_npu_compute_utilization"],
            "request_metrics_exact": request_compare["exact"], "microbatch_metrics_exact": batch_compare["exact"],
            "remote_wall_seconds": remote["wall_seconds_total"], "local_wall_seconds": local["wall_seconds_total"]})
    audit = {"input": label, "strategy": policy, "checks": checks, "configuration_pair_pass": all(checks.values()),
        "remote_result": str(remote_path), "local_result": str(local_path),
        "remote_result_sha256": sha(remote_path), "local_result_sha256": sha(local_path),
        "request_metrics": request_compare, "microbatch_metrics": batch_compare,
        "max_matched_request_admission_delta_ms": max(abs(row["delta_admission_time_ms"]) for row in matched_requests),
        "max_matched_request_completion_delta_ms": max(abs(row["delta_completion_time_ms"]) for row in matched_requests)}
    return rows, matched_requests, audit


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, request_rows, audits, missing = [], [], [], []
    jobs = [(label, policy, HERE / "runs" / label / policy)
            for label in ("raw32_shuffled", "strong32_local", "strong32_stripe6_feasible")
            for policy in ("baseline", "once")]
    jobs.append(("raw32_local", "baseline", BASE / "native_assignment/runs/raw32_local/baseline_shared_local"))
    for label, policy, directory in jobs:
        remote = one(BASE / "screen/runs" / label / policy)
        local = one(directory)
        if remote is None or local is None:
            missing.append({"input": label, "strategy": policy,
                            "remote_missing": remote is None, "local_missing": local is None})
            continue
        new_rows, new_requests, audit = compare(label, policy, remote, local)
        rows.extend(new_rows); request_rows.extend(new_requests); audits.append(audit)
    write_csv(HERE / "cross_environment_comparison.csv", rows)
    write_csv(HERE / "matched_request_differences.csv", request_rows)
    artifact = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis_source_sha256": sha(Path(__file__)),
        "expected_pairs": len(jobs), "completed_pairs": len(audits), "missing": missing,
        "all_completed_configuration_pairs_pass": all(a["configuration_pair_pass"] for a in audits),
        "all_completed_request_and_layer_timelines_exact": all(a["request_metrics"]["exact"] and a["microbatch_metrics"]["exact"] for a in audits),
        "aggregation_note": "Saved U uses each original Python environment; recomputed U processes both saved timelines in this analysis interpreter. Exact metric-list equality is separate from near-equality of aggregate U.",
        "runtime_note": "Host, Python version and concurrent load differ. Wall seconds are preserved as observations, not an isolated CPU speed comparison.",
        "audits": audits}
    (HERE / "cross_environment_audit.json").write_text(json.dumps(artifact, indent=2) + "\n")
    max_saved_delta = max((abs(row["saved_delta_U_pp"]) for row in rows), default=0.0)
    max_recomputed_delta = max((abs(row["recomputed_delta_U_pp"]) for row in rows), default=0.0)
    max_makespan_delta = max((abs(row["delta_makespan_ms"]) for row in rows), default=0.0)
    lines = ["# 有限跨环境复核", "", f"完整配对 {len(audits)}/{len(jobs)}；只读比较相同冻结 manifest 与原策略。", "",
        f"已完成配对的全部配置检查通过：{artifact['all_completed_configuration_pairs_pass']}；全部请求和微批/层时间线逐项精确相等：{artifact['all_completed_request_and_layer_timelines_exact']}。最大保存 U 差 {max_saved_delta:.12g} 个百分点，同解释器重算 U 最大差 {max_recomputed_delta:.12g}，makespan 最大差 {max_makespan_delta:.12g} ms。",
        "",
        "同输入、同提交 seed、同核心和策略源码；本机 shared Baseline/Once 的用途是为本机执行策略提供锚点。不能预先假定跨 Python 的相位或逐请求时间线相同。",
        "", "| 输入 | 策略 | 窗口/ms | 远端 U% | 本机 U% | 差/百分点 | makespan 差/ms | 全请求/层时间线精确相同 |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['input']} | {row['strategy']} | {row['start_ms']:g}–{row['end_ms']:g} | {100*row['remote_saved_U']:.9f} | {100*row['local_saved_U']:.9f} | {row['saved_delta_U_pp']:.9g} | {row['delta_makespan_ms']:.9g} | {row['request_metrics_exact'] and row['microbatch_metrics_exact']} |")
    lines += ["", "数值详细表：`cross_environment/cross_environment_comparison.csv`。逐请求差：`matched_request_differences.csv`；指纹、参数、源码与递归结果对比：`cross_environment_audit.json`。",
        "", "保存的 U 可能仅因 Python 求和舍入而有微差，因此另在同一个本机解释器重算两套事件时间线的 U；同时单独比较全部 request_metrics、microbatch_metrics 与每个记录层事件，不能把均值相近当成时间线相同。",
        "", "raw32_local 本机 shared Baseline 复用 design 代理的 `native_assignment/runs/raw32_local/baseline_shared_local`，未重复运行。所有 5 ms 设置和物理位置保留。运行时长受机器、Python 和并行负载共同影响，不能据此单独给实现加速比。"]
    if missing:
        lines += ["", "未完成配对（保留，不补值）：", ""]
        lines += ["- " + json.dumps(item) for item in missing]
    (BASE / "notes/cross_environment_review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: artifact[key] for key in ("completed_pairs", "expected_pairs", "all_completed_configuration_pairs_pass", "all_completed_request_and_layer_timelines_exact", "missing")}))


if __name__ == "__main__":
    main()
