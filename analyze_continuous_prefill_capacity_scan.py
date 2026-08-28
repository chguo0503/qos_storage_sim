"""Plot and explain whether the Scheme-B regression is caused by SSU count."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continuous_prefill_capacity_scan import OUTPUT
from continuous_prefill_experiment import DEFAULT_OUTPUT as MAIN_RESULTS


def analyze(payload):
    assert payload["complete"] is True
    rows = {
        (row["num_ssu"], row["strategy"]): row for row in payload["results"]
    }
    points = tuple(payload["ssu_points"])
    metrics = {}
    for num_ssu in points:
        baseline = rows[(num_ssu, "baseline")]
        scheme = rows[(num_ssu, "scheme_b_once")]
        assert baseline["trace_hash"] == scheme["trace_hash"]
        assert all(baseline["summary"]["invariants"].values())
        assert all(scheme["summary"]["invariants"].values())
        stats = baseline["workload_statistics"]
        total_compute_ms = sum(
            request["own_compute_ms"]
            for request in baseline["summary"]["request_metrics"]
        )
        compute_bound = baseline["critical_path"]["compute_only_lower_bound_ms"]
        metrics[str(num_ssu)] = {
            "offered_load_percent": 100.0
            * stats["initial_demand_to_ssd_capacity"],
            "hot_ssus": sum(
                demand > 40.0
                for demand in stats["initial_per_ssu_demand_gbps"]
            ),
            "compute_only_fleet_upper_bound_percent": 100.0
            * total_compute_ms
            / (baseline["summary"]["num_npu"] * compute_bound),
            "baseline": _row_metrics(baseline),
            "scheme_b_once": _row_metrics(scheme),
        }
        metrics[str(num_ssu)]["scheme_minus_baseline_fleet_pp"] = (
            metrics[str(num_ssu)]["scheme_b_once"]["fleet_percent"]
            - metrics[str(num_ssu)]["baseline"]["fleet_percent"]
        )
    return {
        "ssu_points": points,
        "metrics": metrics,
        "causal_28_ssu": _causal_28_ssu(),
    }


def _metrics_by_request(summary):
    return {row["request_id"]: row for row in summary["request_metrics"]}


def _request_finish(row):
    return row["arrival_time_ms"] + row["latency_ms"]


def _weighted_npu_io(rows, npu_id, field):
    selected = [row for row in rows if row["npu_id"] == npu_id]
    return float(
        np.average(
            [row[field] for row in selected],
            weights=[row["io_count"] for row in selected],
        )
    )


def _causal_28_ssu():
    payload = json.loads(MAIN_RESULTS.read_text())
    rows = {row["strategy"]: row for row in payload["results"]}
    baseline = rows["baseline"]["summary"]
    scheme = rows["scheme_b_once"]["summary"]
    periodic1 = rows["scheme_b_periodic1"]["summary"]
    baseline_requests = _metrics_by_request(baseline)
    scheme_requests = _metrics_by_request(scheme)

    critical = max(baseline_requests.values(), key=_request_finish)
    critical_id = critical["request_id"]
    critical_npu = critical["npu_id"]
    scheme_critical = scheme_requests[critical_id]
    baseline_admission = (
        critical["arrival_time_ms"] + critical["admission_wait_ms"]
    )
    scheme_admission = (
        scheme_critical["arrival_time_ms"]
        + scheme_critical["admission_wait_ms"]
    )

    initial_on_critical_npu = [
        row
        for row in baseline_requests.values()
        if row["npu_id"] == critical_npu and row["initial"]
    ]
    releaser = min(
        initial_on_critical_npu,
        key=lambda row: abs(_request_finish(row) - baseline_admission),
    )
    scheme_releaser = scheme_requests[releaser["request_id"]]

    stats = rows["baseline"]["workload_statistics"]
    initial_compute_ms = np.asarray(
        stats["per_npu_initial_layer_compute_ms"], dtype=np.float64
    )
    initial_demand = np.asarray(
        rows["baseline"]["workload_statistics"][
            "initial_per_ssu_demand_gbps"
        ],
        dtype=np.float64,
    )

    # The per-NPU demand vector is reconstructed from the Scheme-B initial
    # target metadata already embedded in the deterministic trace calculation.
    import sim
    from continuous_prefill_experiment import _scheme_b_initial
    from continuous_prefill_workload import prepare_continuous_prefill_workload

    workload = prepare_continuous_prefill_workload(
        sim.load_bw_table_cache(num_npu=baseline["num_npu"]),
        num_ssu=baseline["num_ssu"],
    )
    _, _, target = _scheme_b_initial(workload)
    per_npu_demand = np.asarray(
        [sum(row) for row in target.demands_gbps], dtype=np.float64
    )
    per_npu_grant = np.asarray(
        [sum(row) for row in target.grants_gbps], dtype=np.float64
    )
    critical_initial = [
        request
        for request in workload.initial_requests
        if request.npu_id == critical_npu
    ]
    short = [request for request in critical_initial if request.per_layer_us < 5_000.0]
    critical_layer_work_gb = sum(
        request.per_layer_kv_gb for request in critical_initial
    )
    critical_layer_compute_s = sum(
        request.per_layer_us for request in critical_initial
    ) / 1e6

    baseline_tail_by_npu = []
    scheme_tail_by_npu = []
    for npu_id in range(baseline["num_npu"]):
        baseline_tail_by_npu.append(
            max(
                _request_finish(row)
                for row in baseline_requests.values()
                if row["npu_id"] == npu_id
            )
        )
        scheme_tail_by_npu.append(
            max(
                _request_finish(row)
                for row in scheme_requests.values()
                if row["npu_id"] == npu_id
            )
        )
    tail_delta = np.asarray(scheme_tail_by_npu) - np.asarray(
        baseline_tail_by_npu
    )
    quartiles = []
    for index, ids in enumerate(np.array_split(np.argsort(per_npu_demand), 4), 1):
        quartiles.append(
            {
                "quartile": index,
                "mean_demand_gbps": float(np.mean(per_npu_demand[ids])),
                "mean_tail_delta_ms": float(np.mean(tail_delta[ids])),
            }
        )

    total_demand = float(sum(map(sum, target.demands_gbps)))
    total_grant = float(sum(map(sum, target.grants_gbps)))
    baseline_releaser_finish = _request_finish(releaser)
    scheme_releaser_finish = _request_finish(scheme_releaser)
    return {
        "critical_npu": critical_npu,
        "critical_request_id": critical_id,
        "slot_releaser_request_id": releaser["request_id"],
        "baseline_makespan_ms": baseline["makespan_ms"],
        "scheme_makespan_ms": scheme["makespan_ms"],
        "makespan_delta_ms": scheme["makespan_ms"] - baseline["makespan_ms"],
        "critical_admission_delta_ms": scheme_admission - baseline_admission,
        "critical_processing_delta_ms": (
            scheme_critical["processing_latency_ms"]
            - critical["processing_latency_ms"]
        ),
        "releaser_finish_delta_ms": (
            scheme_releaser_finish - baseline_releaser_finish
        ),
        "releaser_io_stall_delta_ms": (
            scheme_releaser["io_stall_ms"] - releaser["io_stall_ms"]
        ),
        "releaser_compute_queue_delta_ms": (
            scheme_releaser["compute_queue_wait_ms"]
            - releaser["compute_queue_wait_ms"]
        ),
        "critical_initial_layer_work_gb": critical_layer_work_gb,
        "critical_initial_layer_compute_ms": 1_000.0 * critical_layer_compute_s,
        "critical_steady_demand_gbps": float(per_npu_demand[critical_npu]),
        "critical_grant_gbps": float(per_npu_grant[critical_npu]),
        "critical_demand_rank_ascending": int(
            np.where(np.argsort(per_npu_demand) == critical_npu)[0][0] + 1
        ),
        "short_request_threshold_ms": 5.0,
        "short_request_count": len(short),
        "short_request_demand_gbps": [
            request.per_layer_kv_gb / (request.per_layer_us / 1e6)
            for request in short
        ],
        "short_compute_share_percent": 100.0
        * sum(request.per_layer_us for request in short)
        / sum(request.per_layer_us for request in critical_initial),
        "short_data_share_percent": 100.0
        * sum(request.per_layer_kv_gb for request in short)
        / critical_layer_work_gb,
        "initial_total_demand_gbps": total_demand,
        "initial_total_grant_gbps": total_grant,
        "initial_unmet_demand_percent": 100.0 * (total_demand - total_grant) / total_demand,
        "initial_hot_ssus": int(np.sum(initial_demand > sim.DISK_BW)),
        "demand_tail_delta_correlation": float(
            np.corrcoef(per_npu_demand, tail_delta)[0, 1]
        ),
        "demand_quartiles": quartiles,
        "avg_latency_delta_ms": (
            scheme["avg_request_latency_ms"] - baseline["avg_request_latency_ms"]
        ),
        "avg_admission_delta_ms": (
            scheme["avg_admission_wait_ms"] - baseline["avg_admission_wait_ms"]
        ),
        "avg_io_stall_delta_ms": (
            scheme["avg_io_stall_ms"] - baseline["avg_io_stall_ms"]
        ),
        "avg_compute_queue_delta_ms": (
            scheme["avg_compute_queue_wait_ms"]
            - baseline["avg_compute_queue_wait_ms"]
        ),
        "critical_npu_weighted_ssd_queue_baseline_ms": _weighted_npu_io(
            baseline["request_metrics"], critical_npu, "avg_ssd_queue_wait_ms"
        ),
        "critical_npu_weighted_ssd_queue_scheme_ms": _weighted_npu_io(
            scheme["request_metrics"], critical_npu, "avg_ssd_queue_wait_ms"
        ),
        "periodic1_same_makespan_as_once": (
            periodic1["makespan_ms"] == scheme["makespan_ms"]
        ),
        "periodic1_runtime_commits": periodic1["cir_commits"],
        "periodic1_path_writes": periodic1["cir_path_writes"],
        "initial_compute_ms_mean": float(np.mean(initial_compute_ms)),
    }


def _row_metrics(row):
    summary = row["summary"]
    return {
        "fleet_percent": 100.0 * summary["fleet_npu_compute_utilization"],
        "active_window_percent": 100.0
        * summary["active_window_npu_compute_utilization"],
        "makespan_ms": summary["makespan_ms"],
        "critical_gap_ms": row["critical_path"][
            "makespan_gap_above_compute_bound_ms"
        ],
        "critical_npu": row["critical_path"]["critical_npu"],
        "avg_latency_ms": summary["avg_request_latency_ms"],
        "p99_latency_ms": summary["p99_request_latency_ms"],
        "avg_io_stall_ms": summary["avg_io_stall_ms"],
        "avg_compute_queue_wait_ms": summary["avg_compute_queue_wait_ms"],
        "ssd_mean_utilization_percent": 100.0 * summary["ssd_mean_utilization"],
    }


def write_plot(output_dir, analysis):
    points = analysis["ssu_points"]
    metrics = analysis["metrics"]
    baseline = [metrics[str(point)]["baseline"] for point in points]
    scheme = [metrics[str(point)]["scheme_b_once"] for point in points]
    upper = [
        metrics[str(point)]["compute_only_fleet_upper_bound_percent"]
        for point in points
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    axes[0, 0].plot(
        points, [row["fleet_percent"] for row in baseline], marker="o",
        linewidth=2, label="Baseline (Path 0)"
    )
    axes[0, 0].plot(
        points, [row["fleet_percent"] for row in scheme], marker="s",
        linewidth=2, label="Scheme B once"
    )
    axes[0, 0].plot(
        points, upper, linestyle="--", color="black", label="Compute-only upper bound"
    )
    axes[0, 0].set_ylabel("Fleet NPU compute utilization (%)")
    axes[0, 0].set_title("Fleet result (zoomed)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        points, [row["critical_gap_ms"] for row in baseline], marker="o",
        linewidth=2, label="Baseline (Path 0)"
    )
    axes[0, 1].plot(
        points, [row["critical_gap_ms"] for row in scheme], marker="s",
        linewidth=2, label="Scheme B once"
    )
    axes[0, 1].set_ylabel("Makespan above compute-only bound (ms)")
    axes[0, 1].set_title("Critical NPU exposed idle")
    axes[0, 1].legend()

    axes[1, 0].plot(
        points, [row["avg_latency_ms"] for row in baseline], marker="o",
        linewidth=2, label="Baseline (Path 0)"
    )
    axes[1, 0].plot(
        points, [row["avg_latency_ms"] for row in scheme], marker="s",
        linewidth=2, label="Scheme B once"
    )
    axes[1, 0].set_ylabel("Average request latency (ms)")
    axes[1, 0].set_title("Average request latency")
    axes[1, 0].legend()

    axes[1, 1].plot(
        points, [row["avg_io_stall_ms"] for row in baseline], marker="o",
        linewidth=2, label="Baseline I/O stall"
    )
    axes[1, 1].plot(
        points, [row["avg_io_stall_ms"] for row in scheme], marker="s",
        linewidth=2, label="Scheme B I/O stall"
    )
    axes[1, 1].plot(
        points,
        [row["avg_compute_queue_wait_ms"] for row in baseline],
        marker="o", linestyle="--", label="Baseline compute queue"
    )
    axes[1, 1].plot(
        points,
        [row["avg_compute_queue_wait_ms"] for row in scheme],
        marker="s", linestyle="--", label="Scheme B compute queue"
    )
    axes[1, 1].set_ylabel("Average request wait (ms)")
    axes[1, 1].set_title("Where request waiting moves")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Number of SSUs")
        axis.set_xticks(points)
        axis.axvline(30, color="#777777", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.3)
    axes[0, 0].annotate(
        "first point with no\noverloaded SSU",
        xy=(30, metrics["30"]["baseline"]["fleet_percent"]),
        xytext=(31, metrics["30"]["baseline"]["fleet_percent"] - 0.25),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
    )
    figure.tight_layout()
    path = output_dir / "02_ssu_capacity_diagnosis.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_report(output_dir, analysis):
    metrics = analysis["metrics"]
    causal = analysis["causal_28_ssu"]
    no_hot_point = min(
        point
        for point in analysis["ssu_points"]
        if metrics[str(point)]["hot_ssus"] == 0
    )
    no_hot = metrics[str(no_hot_point)]
    overloaded = metrics[str(analysis["ssu_points"][0])]
    formal = metrics["28"]
    largest = metrics[str(analysis["ssu_points"][-1])]
    lines = [
        "# Continuous-batch：SSU 容量与 Scheme B 因果诊断",
        "",
        "| SSU | Offered load | Hot SSUs | Baseline fleet | Scheme B fleet | Scheme−baseline | Baseline critical gap | Scheme critical gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in analysis["ssu_points"]:
        row = metrics[str(point)]
        baseline = row["baseline"]
        scheme = row["scheme_b_once"]
        lines.append(
            f"| {point} | {row['offered_load_percent']:.1f}% | "
            f"{row['hot_ssus']} | {baseline['fleet_percent']:.4f}% | "
            f"{scheme['fleet_percent']:.4f}% | "
            f"{row['scheme_minus_baseline_fleet_pp']:+.4f} pp | "
            f"{baseline['critical_gap_ms']:.3f} ms | "
            f"{scheme['critical_gap_ms']:.3f} ms |"
        )
    lines.extend((
        "",
        "## 结论",
        "",
        f"**主要影响不是盘数，而是 Scheme B 的 CIR 目标与 cold-start / 短请求突发不匹配。** "
        f"28 SSU 的总 offered load 是 {formal['offered_load_percent']:.1f}%，"
        f"只有 {formal['hot_ssus']} 块盘略超 40 GB/s。增加到 {no_hot_point} SSU 后已没有任何热点盘，"
        f"但 Scheme B 仍比 baseline 低 {abs(no_hot['scheme_minus_baseline_fleet_pp']):.4f} pp，"
        f"关键路径仍多暴露 {no_hot['scheme_b_once']['critical_gap_ms'] - no_hot['baseline']['critical_gap_ms']:.3f} ms。"
        "因此加盘会缓解差距，却不能解释或修复策略本身。",
        "",
        f"在真正容量不足的 {analysis['ssu_points'][0]} SSU 点（offered load "
        f"{overloaded['offered_load_percent']:.1f}%）Scheme B 也没有反超，反而低 "
        f"{abs(overloaded['scheme_minus_baseline_fleet_pp']):.4f} pp。"
        f"即使加到 {analysis['ssu_points'][-1]} SSU，最终差值为 "
        f"{largest['scheme_minus_baseline_fleet_pp']:+.4f} pp；这个点用于观察收敛，不应作为正式容量点，"
        "因为大量过配会让 QoS 仲裁本身失去意义。",
        "",
        "容量选择建议：**保留 28 SSU 作为近容量压力点**；如果实验要求 ring-hash 后每块盘都严格不超过 40 GB/s，"
        f"则使用 **{no_hot_point} SSU** 作为 capacity-safe 对照点。现有证据不支持因为 Scheme B 较差而继续增加盘数。",
        "",
        "## 关键路径：119.744 ms 从哪里来",
        "",
        f"全局尾部是 NPU {causal['critical_npu']} 的新请求 {causal['critical_request_id']}。"
        f"Scheme B 的 makespan 比 baseline 多 {causal['makespan_delta_ms']:.3f} ms；"
        f"该请求的 admission 恰好晚 {causal['critical_admission_delta_ms']:.3f} ms，"
        f"而 admission 后处理时间只变化 {causal['critical_processing_delta_ms']:.6f} ms。"
        f"释放 batch slot 的初始请求 {causal['slot_releaser_request_id']} 也恰好晚完成 "
        f"{causal['releaser_finish_delta_ms']:.3f} ms。"
        "因此这不是指标搬运或尾请求自身变慢，而是早期 I/O bubble 将整个关键 NPU 时间线向后平移。",
        "",
        f"在 slot-releaser 上，暴露 I/O stall 增加 {causal['releaser_io_stall_delta_ms']:.3f} ms，"
        f"compute queue 反而减少 {abs(causal['releaser_compute_queue_delta_ms']):.3f} ms，"
        f"两者净和就是 {causal['releaser_finish_delta_ms']:.3f} ms。"
        "这说明必须看端到端关键链，不能只用平均 SSD queue 或平均 I/O stall 判断策略优劣。",
        "",
        "## Scheme B 为什么低估关键 NPU",
        "",
        f"Scheme B 用一层内 `sum(data) / sum(compute)` 作为 NPU×SSU 稳态 demand。"
        f"NPU {causal['critical_npu']} 的一层总数据为 {causal['critical_initial_layer_work_gb']:.6f} GB，"
        f"总计算为 {causal['critical_initial_layer_compute_ms']:.3f} ms，所以 demand 和 grant 都只有 "
        f"{causal['critical_grant_gbps']:.6f} GB/s（128 个 NPU 中从低到高第 "
        f"{causal['critical_demand_rank_ascending']}）。",
        "",
        f"但其中 {causal['short_request_count']} 个计算时间小于 "
        f"{causal['short_request_threshold_ms']:.0f} ms 的请求只占 "
        f"{causal['short_compute_share_percent']:.3f}% 计算，却占 "
        f"{causal['short_data_share_percent']:.3f}% 数据；它们各自的瞬时需求为 "
        f"{', '.join(f'{value:.1f}' for value in causal['short_request_demand_gbps'])} GB/s。"
        "batch8 在 layer 0 会同时发出这些 I/O，前面没有计算可以隐藏读取，"
        "所以长期平均 CIR 对启动突发明显偏低。",
        "",
        f"28 SSU 初始总 demand 为 {causal['initial_total_demand_gbps']:.3f} GB/s，"
        f"grant 为 {causal['initial_total_grant_gbps']:.3f} GB/s，未满足比例只有 "
        f"{causal['initial_unmet_demand_percent']:.3f}%。关键 NPU 的 demand 被完整 grant，"
        "所以它不是被 max-min 容量裁剪，而是控制器一开始就把它的需求估低了。",
        "",
        "## 次要影响与更新频率",
        "",
        f"从全体请求平均值看，Scheme B 让 I/O stall 改善 "
        f"{abs(causal['avg_io_stall_delta_ms']):.3f} ms，却让 compute queue 增加 "
        f"{causal['avg_compute_queue_delta_ms']:.3f} ms，最终平均 latency 增加 "
        f"{causal['avg_latency_delta_ms']:.3f} ms。它把服务机会转给高平均需求 NPU，"
        f"demand 与 NPU 尾部变化的相关系数为 {causal['demand_tail_delta_correlation']:.3f}。"
        f"最低需求四分位平均变慢 {causal['demand_quartiles'][0]['mean_tail_delta_ms']:.3f} ms，"
        f"最高需求四分位平均改善 {abs(causal['demand_quartiles'][-1]['mean_tail_delta_ms']):.3f} ms。"
        "这对 max-min 公平是合理的，却与本实验的 fleet makespan 目标不一致。",
        "",
        f"每 1 个 batch-layer 等价时间更新一次进行了 "
        f"{causal['periodic1_runtime_commits']} 次有效 commit 和 "
        f"{causal['periodic1_path_writes']} 次运行期 Path 写入，但 makespan 与 once "
        f"{'完全相同' if causal['periodic1_same_makespan_as_once'] else '不同'}。"
        "关键损失发生在第一次 slot 释放之前，此时 active set 基本没变；"
        "因此提高更新频率不是主要杠杆。",
        "",
        "## 应怎样改新策略",
        "",
        "下一版应保留稳态 max-min grant，但在 cold-start、新 admission 和预计 slot 释放前加入临时 burst/critical grant："
        "用近期要读的数据量除以真实可隐藏的 deadline/slack，而不是除以整批长请求的计算总和；"
        "同时提高预计最早释放 batch slot、或预计成为全局尾部的 NPU 权重。突发消退后再回落到当前 CIR。"
        "这个改动直接针对已观测到的 119.744 ms 因果链。",
        "",
        "## 指标限制",
        "",
        f"固定请求到 NPU 的映射使 compute-only fleet 上界只有 "
        f"{formal['compute_only_fleet_upper_bound_percent']:.4f}%。"
        f"baseline 在 28 SSU 已只比该上界低 "
        f"{formal['compute_only_fleet_upper_bound_percent'] - formal['baseline']['fleet_percent']:.4f} pp，"
        f"关键 makespan 仅高出 {formal['baseline']['critical_gap_ms']:.3f} ms。"
        "因此在这个有限 trace 上，任何只改存储 QoS、不改变 NPU 请求分配或计算负载的策略，"
        "都不可能获得显著 fleet 利用率提升。平均 latency 仍有优化空间，但它和 fleet 尾部是不同目标。",
        "",
    ))
    path = output_dir / "capacity_report.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    output_dir = args.input.parent
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(write_plot(output_dir, analysis))
    print(write_report(output_dir, analysis))


if __name__ == "__main__":
    main()
