"""Validate, plot, and explain the Full-prefill microbatch experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import continuous_prefill_experiment as experiment


DEFAULT_INPUT = experiment.DEFAULT_OUTPUT
EXECUTION_MODEL = "full_prefill_layer_synchronous_microbatch_v1"
LABELS = {
    "baseline": "Baseline\n(Path 0)",
    "path_rr": "Path RR",
    "layer_once": "Layer once",
    "refresh8": "Refresh 8",
    "refresh1": "Refresh 1",
    "scheme_b_once": "Scheme B\nmembership",
    "scheme_b_periodic8": "Scheme B\n8 layers",
    "scheme_b_periodic4": "Scheme B\n4 layers",
    "scheme_b_periodic2": "Scheme B\n2 layers",
    "scheme_b_periodic1": "Scheme B\n1 layer",
    "scheme_b_after_l0": "L0 Path0\nthen Scheme B",
    "best_feasible": "Full-info\nEDF reference",
}
SCHEME_ORDER = (
    "scheme_b_once",
    "scheme_b_periodic8",
    "scheme_b_periodic4",
    "scheme_b_periodic2",
    "scheme_b_periodic1",
)


def _cohort(rows):
    latencies = [row["latency_ms"] for row in rows]
    return {
        "count": len(rows),
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "avg_admission_wait_ms": float(
            np.mean([row["admission_wait_ms"] for row in rows])
        ),
    }


def _paired_invariants(rows):
    summaries = [row["summary"] for row in rows.values()]
    fields = (
        "input_fingerprint",
        "request_count",
        "microbatch_count",
        "submitted_blocks",
        "expected_read_gb",
        "completed_request_layer_jobs",
    )
    checks = {
        field: len({summary[field] for summary in summaries}) == 1
        for field in fields
    }
    canonical_memberships = []
    for summary in summaries:
        by_npu = defaultdict(list)
        for batch in summary["microbatch_metrics"]:
            by_npu[batch["npu_id"]].append(batch)
        canonical_memberships.append(
            tuple(
                (
                    npu_id,
                    tuple(
                        tuple(batch["member_request_ids"])
                        for batch in sorted(
                            batches,
                            key=lambda row: row["admission_time_ms"],
                        )
                    ),
                )
                for npu_id, batches in sorted(by_npu.items())
            )
        )
    checks["canonical_microbatch_membership"] = all(
        membership == canonical_memberships[0]
        for membership in canonical_memberships
    )
    physical_compute = [
        summary["fleet_npu_compute_utilization"]
        * summary["num_npu"]
        * summary["makespan_ms"]
        for summary in summaries
    ]
    checks["physical_compute_work"] = bool(
        np.allclose(physical_compute, physical_compute[0], rtol=1e-10, atol=1e-7)
    )
    return checks


def _compute_only_reference(summary):
    requests = {
        row["request_id"]: row for row in summary["request_metrics"]
    }
    batches_by_npu = defaultdict(list)
    for batch in summary["microbatch_metrics"]:
        batches_by_npu[batch["npu_id"]].append(batch)
    total_compute_ms = 0.0
    active_window_ms = 0.0
    finishes = []
    for npu_id, batches in batches_by_npu.items():
        finish_ms = 0.0
        first_start_ms = None
        npu_compute_ms = 0.0
        for batch in sorted(batches, key=lambda row: row["batch_id"]):
            arrival_ready_ms = max(
                requests[request_id]["arrival_time_ms"]
                for request_id in batch["member_request_ids"]
            )
            start_ms = max(finish_ms, arrival_ready_ms)
            if first_start_ms is None:
                first_start_ms = start_ms
            finish_ms = start_ms + batch["compute_busy_ms"]
            npu_compute_ms += batch["compute_busy_ms"]
        total_compute_ms += npu_compute_ms
        active_window_ms += finish_ms - first_start_ms
        finishes.append((finish_ms, npu_id, npu_compute_ms))
    makespan_ms, critical_npu, critical_compute_ms = max(finishes)
    return {
        "makespan_lower_bound_ms": makespan_ms,
        "fleet_utilization_upper_bound": total_compute_ms
        / (summary["num_npu"] * makespan_ms),
        "active_window_utilization_upper_bound": total_compute_ms
        / active_window_ms,
        "critical_npu": critical_npu,
        "critical_npu_compute_ms": critical_compute_ms,
        "total_compute_ms": total_compute_ms,
    }


def analyze(payload):
    expected = tuple(payload["experiment"]["strategies"])
    rows = {row["strategy"]: row for row in payload["results"]}
    if not payload["complete"] or tuple(rows) != expected:
        raise ValueError("experiment result matrix is incomplete")
    if payload["experiment"].get("execution_model") != EXECUTION_MODEL:
        raise ValueError("refusing to analyze legacy request-interleaved results")
    if len({row["trace_hash"] for row in rows.values()}) != 1:
        raise ValueError("strategies did not use the same request trace")
    if not all(
        row["summary"].get("execution_model") == EXECUTION_MODEL
        and all(row["summary"]["invariants"].values())
        for row in rows.values()
    ):
        raise ValueError("a strategy used the wrong execution model or failed invariants")
    paired = _paired_invariants(rows)
    if not all(paired.values()):
        raise ValueError(f"strategy pairing failed: {paired}")

    compute_only = _compute_only_reference(rows["baseline"]["summary"])
    critical_npu = compute_only["critical_npu"]
    metrics = {}
    for name, row in rows.items():
        summary = row["summary"]
        initial = [item for item in summary["request_metrics"] if item["initial"]]
        new = [item for item in summary["request_metrics"] if not item["initial"]]
        layer0_barriers = [
            batch["layer_metrics"][0]["io_barrier_wait_ms"]
            for batch in summary["microbatch_metrics"]
        ]
        control = row["control"] or {}
        critical_batches = [
            batch
            for batch in summary["microbatch_metrics"]
            if batch["npu_id"] == critical_npu
        ]
        critical_initial = next(
            batch for batch in critical_batches if batch["batch_size"] == 8
        )
        metrics[name] = {
            "fleet_npu_compute_utilization": summary[
                "fleet_npu_compute_utilization"
            ],
            "active_window_npu_compute_utilization": summary[
                "active_window_npu_compute_utilization"
            ],
            "avg_batch_compute_fraction": summary["avg_batch_compute_fraction"],
            "makespan_ms": summary["makespan_ms"],
            "throughput_requests_per_s": summary["throughput_requests_per_s"],
            "avg_request_latency_ms": summary["avg_request_latency_ms"],
            "p95_request_latency_ms": summary["p95_request_latency_ms"],
            "p99_request_latency_ms": summary["p99_request_latency_ms"],
            "avg_batch_io_barrier_wait_ms": summary[
                "avg_batch_io_barrier_wait_ms"
            ],
            "p99_layer0_barrier_ms": float(np.percentile(layer0_barriers, 99)),
            "critical_npu_completion_ms": max(
                batch["completion_time_ms"] for batch in critical_batches
            ),
            "critical_npu_total_barrier_ms": sum(
                batch["io_barrier_wait_ms"] for batch in critical_batches
            ),
            "critical_npu_initial_layer0_barrier_ms": critical_initial[
                "layer_metrics"
            ][0]["io_barrier_wait_ms"],
            "ssd_mean_utilization": summary["ssd_mean_utilization"],
            "npu_link_mean_utilization": summary["npu_link_mean_utilization"],
            "control_evaluations": summary["control_evaluations"],
            "cir_commits": summary["cir_commits"],
            "cir_path_writes": summary["cir_path_writes"],
            "pressure_reports": summary["pressure_reports"],
            "wall_time_s": row["wall_time_s"],
            "update_trigger": control.get("update_trigger"),
            "initial_cohort": _cohort(initial),
            "new_cohort": _cohort(new),
        }

    baseline = metrics["baseline"]
    for values in metrics.values():
        values["fleet_delta_vs_baseline_pp"] = 100.0 * (
            values["fleet_npu_compute_utilization"]
            - baseline["fleet_npu_compute_utilization"]
        )
        values["makespan_delta_vs_baseline_ms"] = (
            values["makespan_ms"] - baseline["makespan_ms"]
        )
        values["barrier_delta_vs_baseline_ms"] = (
            values["avg_batch_io_barrier_wait_ms"]
            - baseline["avg_batch_io_barrier_wait_ms"]
        )

    best_scheme = max(
        SCHEME_ORDER,
        key=lambda name: metrics[name]["fleet_npu_compute_utilization"],
    )
    return {
        "configuration": payload["experiment"],
        "trace_hash": next(iter(rows.values()))["trace_hash"],
        "input_results_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "paired_invariants": paired,
        "workload_statistics": next(iter(rows.values()))["workload_statistics"],
        "compute_only_reference": compute_only,
        "metrics": metrics,
        "best_scheme_b_frequency": best_scheme,
        "best_non_reference_strategy": max(
            (name for name in expected if name != "best_feasible"),
            key=lambda name: metrics[name]["fleet_npu_compute_utilization"],
        ),
    }


def write_plot(output_dir: Path, analysis):
    names = tuple(analysis["configuration"]["strategies"])
    metrics = analysis["metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    positions = np.arange(len(names))
    colors = [
        "#F58518" if name.startswith("scheme_b") else "#4C78A8"
        for name in names
    ]
    colors[-1] = "#54A24B"
    fleet = [100 * metrics[name]["fleet_npu_compute_utilization"] for name in names]
    active = [
        100 * metrics[name]["active_window_npu_compute_utilization"]
        for name in names
    ]
    bars = axes[0].bar(positions - 0.19, fleet, width=0.38, color=colors,
                       label="Fleet / full trace")
    active_bars = axes[0].bar(
        positions + 0.19, active, width=0.38, color=colors, alpha=0.45,
        hatch="//", label="Per-NPU active window"
    )
    axes[0].axhline(
        100 * analysis["compute_only_reference"]["fleet_utilization_upper_bound"],
        color="#222222",
        linestyle=":",
        linewidth=1.5,
        label="Compute-only fleet upper bound",
    )
    axes[0].set(
        title="Full-prefill microbatch comparison (28 SSUs)",
        ylabel="NPU compute utilization (%)",
        xticks=positions,
        xticklabels=[LABELS[name] for name in names],
        ylim=(0, 100),
    )
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes[0].bar_label(active_bars, fmt="%.1f", fontsize=7, padding=2)
    axes[0].legend(loc="lower right")

    frequency_labels = ("Membership", "8", "4", "2", "1")
    scheme_util = [
        100 * metrics[name]["fleet_npu_compute_utilization"]
        for name in SCHEME_ORDER
    ]
    axes[1].plot(frequency_labels, scheme_util, marker="o", linewidth=2,
                 color="#F58518", label="Fleet NPU utilization")
    axes[1].set(
        title="Scheme B update-frequency sweep",
        xlabel="Membership updates plus periodic interval (batch layers)",
        ylabel="Fleet NPU compute utilization (%)",
        ylim=(0, 100),
    )
    axes[1].grid(alpha=0.3)
    writes_axis = axes[1].twinx()
    writes = [metrics[name]["cir_path_writes"] for name in SCHEME_ORDER]
    writes_axis.plot(frequency_labels, writes, marker="s", linestyle="--",
                     color="#E45756", label="CIR Path writes")
    writes_axis.set_ylabel("Runtime CIR Path writes")
    lines = axes[1].lines + writes_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="lower right")
    figure.tight_layout()
    output_path = output_dir / "01_full_prefill_microbatch_strategies.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_report(output_dir: Path, analysis):
    metrics = analysis["metrics"]
    config = analysis["configuration"]
    stats = analysis["workload_statistics"]
    baseline = metrics["baseline"]
    best_scheme_name = analysis["best_scheme_b_frequency"]
    best_scheme = metrics[best_scheme_name]
    compute_only = analysis["compute_only_reference"]
    lines = [
        "# Full-prefill microbatch result",
        "",
        "配置：128 NPU、batch 8、16 层、28 SSU。固定 microbatch 的所有成员逐层同步；本层所有成员 I/O 完成后才联合计算，最后一层整批同刻完成。",
        "",
        "Batch 层计算时间采用 `sum(member singleton layer time)`，这是固定吞吐、工作量守恒的代理，不是假设实测 batch kernel 加速。所有策略共用 SSD 40、NPU 50 和同一 ring-hash placement。",
        "",
        "| Strategy | Fleet util. | Active-window util. | vs baseline | Makespan (ms) | Avg batch I/O barrier (ms) | P99 latency (ms) | CIR writes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in config["strategies"]:
        row = metrics[name]
        lines.append(
            f"| {LABELS[name].replace(chr(10), ' ')} | "
            f"{100 * row['fleet_npu_compute_utilization']:.3f}% | "
            f"{100 * row['active_window_npu_compute_utilization']:.3f}% | "
            f"{row['fleet_delta_vs_baseline_pp']:+.3f} pp | "
            f"{row['makespan_ms']:.3f} | "
            f"{row['avg_batch_io_barrier_wait_ms']:.3f} | "
            f"{row['p99_request_latency_ms']:.3f} | "
            f"{row['cir_path_writes']} |"
        )
    lines.extend(
        (
            "",
            "## 结论口径",
            "",
            f"Scheme B 频率中最好的是 **{LABELS[best_scheme_name].replace(chr(10), ' ')}**，"
            f"fleet 利用率 {100 * best_scheme['fleet_npu_compute_utilization']:.3f}%，"
            f"相对 baseline {best_scheme['fleet_delta_vs_baseline_pp']:+.3f} pp。",
            "",
            "所有方案的 microbatch membership、块数、字节数和物理计算工作量完全一致。因此 fleet 利用率的差异不是 batch 计算量变化造成的；分子相同，差异来自 I/O barrier 改变 makespan。",
            "",
            f"即使完全删除存储等待，固定请求画像下的 compute-only fleet 上界也只有 "
            f"{100 * compute_only['fleet_utilization_upper_bound']:.3f}%。原因是 fleet 分母是 "
            f"128×全局 makespan，而 NPU {compute_only['critical_npu']} 单独拥有 "
            f"{compute_only['critical_npu_compute_ms']:.1f} ms 计算工作，决定了 "
            f"{compute_only['makespan_lower_bound_ms']:.1f} ms 的排空尾部；其他 NPU 先结束后仍计入分母。"
            f"Baseline 已达到这个上界的 "
            f"{100 * baseline['fleet_npu_compute_utilization'] / compute_only['fleet_utilization_upper_bound']:.3f}%。",
            "",
            f"Baseline 平均 batch I/O barrier 为 {baseline['avg_batch_io_barrier_wait_ms']:.3f} ms；"
            f"最佳 Scheme B 为 {best_scheme['avg_batch_io_barrier_wait_ms']:.3f} ms。"
            "这里的 barrier 是每层最慢成员的暴露存储等待，不能再解释成旧异步模型里的单请求 compute queue wait。",
            "",
            f"关键 NPU {compute_only['critical_npu']} 的初始 layer-0 barrier 在 baseline 中是 "
            f"{baseline['critical_npu_initial_layer0_barrier_ms']:.3f} ms，在最佳 Scheme B 中是 "
            f"{best_scheme['critical_npu_initial_layer0_barrier_ms']:.3f} ms。Scheme B 用稳态 "
            "`next-layer bytes / batch compute window` 分配 CIR，但 layer 0 没有该窗口；"
            "低稳态需求的 batch 会在 cold start 被低配，这个首层延迟会沿 16 层关键路径保留下来。",
            "",
            "Scheme B 的 membership 版本只在 microbatch 成员变化时重算；周期版本还在每 8/4/2/1 个真实 batch-layer equivalent 后评估。sticky placement 与固定成员使各稳态层的需求向量相同，所以更频繁评估理论上通常只会增加评估次数；只有 membership、shape 或层耗时改变时目标 CIR 才应变化。",
            "",
            "Layer 0 是 cold-start burst，没有上一层计算窗口可隐藏。它仍由相同命令级 SSD/NPU 数据面真实排队；报告没有把 layer-0 bytes / layer compute 伪称为可满足的稳态 CIR demand。",
            "",
            f"新增请求在 {stats['arrival_window_ms'][0]:.1f}–{stats['arrival_window_ms'][1]:.1f} ms 到达。"
            "Full-prefill 下它们不能插入正在运行的 batch；必须等待整批 16 层完成，再组成下一批。当前 trace 每 NPU 只有 1 个新增请求，因此第二波是 final singleton partial batch，并不是长期稳态 continuous batching。",
            "",
            "单位限制：画像文件用 `2^30` 计算 KV 大小，数值实际是 GiB，但历史字段名为 `_gb`。本轮为了与既有 SSD40/NPU50 实验配对，没有做 1.073741824 的十进制换算；绝对容量结论应按这一假设阅读。",
            "",
            "`best_feasible` 是保留相同 NPU↔SSD placement、SSD40、NPU50 和 batch barrier 的 full-information EDF reference，不会修改访问关系，也不宣称是数学全局最优。",
        )
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = analyze(payload)
    output_dir = args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    plot_path = write_plot(output_dir, result)
    report_path = write_report(output_dir, result)
    print(plot_path)
    print(report_path)


if __name__ == "__main__":
    main()
