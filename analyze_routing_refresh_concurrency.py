"""Analyze the final two-seed finite-issue Path-routing matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment as base_experiment


SEEDS = (42, 43)
SSUS = (8, 16, 28, 40, 56, 80, 112)
SIMULATION_STRATEGIES = (
    "baseline",
    "path_rr_baseline",
    "refresh1",
    "refresh8",
    "layer_once",
)
INPUT_STRATEGIES = SIMULATION_STRATEGIES
ORACLE_STRATEGY = "capacity_constrained_oracle"
ORACLE_CANDIDATE_STRATEGY = "demand_weighted_sjf_oracle_candidate"
STRATEGIES = SIMULATION_STRATEGIES + (ORACLE_STRATEGY,)
CATEGORIES = ("SS", "SL", "LS", "LL")
LABELS = {
    "baseline": "Final baseline (all I/O to Path 0)",
    "path_rr_baseline": "Static QoS: no Path-state reads (RR)",
    "refresh1": "Static QoS: read before every I/O",
    "refresh8": "Static QoS: read every 8 I/Os",
    "layer_once": "Static QoS: read once per layer",
    ORACLE_STRATEGY: "Capacity-constrained oracle (best feasible)",
}
COLORS = {
    "baseline": "#d62728",
    "path_rr_baseline": "#1f77b4",
    "refresh1": "#2ca02c",
    "refresh8": "#ff7f0e",
    "layer_once": "#9467bd",
    ORACLE_STRATEGY: "#111111",
}
MARKERS = {
    "baseline": "o",
    "path_rr_baseline": "s",
    "refresh1": "^",
    "refresh8": "D",
    "layer_once": "P",
    ORACLE_STRATEGY: "X",
}
EXPECTED_MODES = {
    "baseline": "fixed_path_zero",
    "path_rr_baseline": "stateless_round_robin",
    "refresh1": "pressure_aware",
    "refresh8": "pressure_aware",
    "layer_once": "pressure_aware",
}
EXPECTED_WINDOWS = {
    "baseline": None,
    "path_rr_baseline": None,
    "refresh1": 1,
    "refresh8": 8,
    "layer_once": None,
}


def validate(data):
    if not data["complete"]:
        raise AssertionError("routing matrix is incomplete")
    if tuple(data["selected_strategies"]) != INPUT_STRATEGIES:
        raise AssertionError("strategy set mismatch")

    runtime = data["experiment"]["runtime"]
    if runtime["num_npu"] != 128 or runtime["n_layers"] != 16:
        raise AssertionError("formal NPU/layer contract mismatch")
    if tuple(runtime["ssu_list"]) != SSUS or tuple(runtime["seeds"]) != SEEDS:
        raise AssertionError("formal SSU/seed contract mismatch")
    if runtime["arrival_delay_ms"] != [0.0, 5.0]:
        raise AssertionError("arrival-delay contract mismatch")
    if runtime["client_submit_batch_size"] != 1:
        raise AssertionError("submission batch is not one I/O")
    if runtime["client_issue_interval_us"] != 0.1:
        raise AssertionError("finite issue interval mismatch")

    expected_keys = {
        (seed, ssu, strategy)
        for seed in SEEDS
        for ssu in SSUS
        for strategy in INPUT_STRATEGIES
    }
    rows = data["results"]
    index = {
        (row["seed"], row["num_ssu"], row["strategy"]): row for row in rows
    }
    if set(index) != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError("matrix key mismatch")

    for seed in SEEDS:
        workload_hashes = set()
        for ssu in SSUS:
            paired = [
                index[(seed, ssu, strategy)] for strategy in INPUT_STRATEGIES
            ]
            if len({row["workload_fingerprint"] for row in paired}) != 1:
                raise AssertionError("unpaired workload")
            if len({row["placement_hash"] for row in paired}) != 1:
                raise AssertionError("unpaired placement")
            workload_hashes.add(paired[0]["workload_fingerprint"])

            simulation_rows = [
                index[(seed, ssu, strategy)]
                for strategy in SIMULATION_STRATEGIES
            ]
            request_ids = [
                tuple(item["request_id"] for item in row["request_metrics"])
                for row in simulation_rows
            ]
            if len(set(request_ids)) != 1 or len(request_ids[0]) != 128:
                raise AssertionError("unpaired request rows")
            arrival_vectors = [
                tuple(item["arrival_delay_ms"] for item in row["request_metrics"])
                for row in simulation_rows
            ]
            if len(set(arrival_vectors)) != 1:
                raise AssertionError("unpaired arrival delays")

            for row in simulation_rows:
                strategy = row["strategy"]
                config = row["config"]
                client = config["client_io"]
                summary = row["summary"]
                if row["kind"] != "simulation":
                    raise AssertionError("non-simulation strategy in data plane")
                if config["policy"] != "qos_static_cir":
                    raise AssertionError("simulation policy mismatch")
                if config["profile_name"] != (
                    "low_protect_cir_20_6_8_6_current_paths"
                ):
                    raise AssertionError("static profile mismatch")
                if config["category_cir_gbps"] != [20.0, 6.0, 8.0, 6.0]:
                    raise AssertionError("static CIR mismatch")
                if config["category_paths_per_group"] != [12, 4, 12, 4]:
                    raise AssertionError("static Path allocation mismatch")
                if client["submit_batch_size"] != 1:
                    raise AssertionError("strategy batch mismatch")
                if client["issue_interval_us"] != 0.1:
                    raise AssertionError("strategy issue interval mismatch")
                if client["path_selection_mode"] != EXPECTED_MODES[strategy]:
                    raise AssertionError("Path-selection mode mismatch")
                if client["pressure_window_io"] != EXPECTED_WINDOWS[strategy]:
                    raise AssertionError("pressure cadence mismatch")
                if not all(summary["invariants"].values()):
                    raise AssertionError("data-plane invariant failure")
                if strategy in ("baseline", "path_rr_baseline"):
                    if summary["pressure_reports"] != 0:
                        raise AssertionError("zero-telemetry strategy read pressure")
                elif summary["pressure_reports"] <= 0:
                    raise AssertionError("Static QoS strategy did not read pressure")
                if strategy == "baseline" and summary.get(
                    "enqueued_path_ids"
                ) != [0]:
                    raise AssertionError("final baseline used a Path other than Path 0")
        if len(workload_hashes) != 1:
            raise AssertionError("workload changed across SSU counts")
    return index


def validate_oracle_candidates(data, routing_index):
    if not data["complete"]:
        raise AssertionError("capacity-constrained oracle matrix is incomplete")
    experiment = data["experiment"]
    runtime = experiment["runtime"]
    if runtime["num_npu"] != 128 or runtime["n_layers"] != 16:
        raise AssertionError("oracle NPU/layer contract mismatch")
    if tuple(runtime["ssu_list"]) != SSUS or tuple(runtime["seeds"]) != SEEDS:
        raise AssertionError("oracle SSU/seed contract mismatch")
    if experiment["strategy"] != ORACLE_CANDIDATE_STRATEGY:
        raise AssertionError("oracle candidate strategy mismatch")
    optimality = experiment["optimality"]
    if optimality["exact_optimum_proven"]:
        raise AssertionError("heuristic oracle incorrectly claims exact optimality")

    rows = data["results"]
    index = {(row["seed"], row["num_ssu"]): row for row in rows}
    expected = {(seed, ssu) for seed in SEEDS for ssu in SSUS}
    if set(index) != expected or len(rows) != len(expected):
        raise AssertionError("oracle matrix key mismatch")

    for key, row in index.items():
        seed, ssu = key
        if row["strategy"] != ORACLE_CANDIDATE_STRATEGY:
            raise AssertionError("oracle row strategy mismatch")
        if row["kind"] != "feasible_oracle_candidate":
            raise AssertionError("oracle row kind mismatch")
        paired = routing_index[(seed, ssu, "baseline")]
        if row["workload_fingerprint"] != paired["workload_fingerprint"]:
            raise AssertionError("oracle workload is not paired")
        if row["placement_hash"] != paired["placement_hash"]:
            raise AssertionError("oracle placement is not paired")
        summary = row["summary"]
        if summary["policy"] != "per_ssd_full_visible_edf":
            raise AssertionError("oracle candidate policy mismatch")
        if summary["client_submit_batch_size"] != 1:
            raise AssertionError("oracle candidate batch mismatch")
        if summary["client_submit_interval_us"] != 0.1:
            raise AssertionError("oracle candidate issue interval mismatch")
        if summary["exact_optimum_proven"]:
            raise AssertionError("oracle row incorrectly claims exact optimality")
        if not all(summary["invariants"].values()):
            raise AssertionError("oracle data-plane invariant failure")
        if not summary["block_conservation"]["placement_targets_preserved"]:
            raise AssertionError("oracle changed block-to-SSD placement")
        if len(row["request_metrics"]) != 128:
            raise AssertionError("oracle request rows mismatch")
        request_ids = tuple(item["request_id"] for item in row["request_metrics"])
        paired_ids = tuple(item["request_id"] for item in paired["request_metrics"])
        if request_ids != paired_ids:
            raise AssertionError("oracle request IDs are not paired")
        arrivals = tuple(
            item["arrival_delay_ms"] for item in row["request_metrics"]
        )
        paired_arrivals = tuple(
            item["arrival_delay_ms"] for item in paired["request_metrics"]
        )
        if arrivals != paired_arrivals:
            raise AssertionError("oracle arrival delays are not paired")
    return index


def add_oracle_envelope(routing_index, candidate_index):
    """Add the best observed physically feasible schedule for every paired case."""
    selections = {}
    for seed in SEEDS:
        for ssu in SSUS:
            candidates = {
                strategy: routing_index[(seed, ssu, strategy)]
                for strategy in SIMULATION_STRATEGIES
            }
            candidates[ORACLE_CANDIDATE_STRATEGY] = candidate_index[(seed, ssu)]
            chosen_name, chosen_row = max(
                candidates.items(),
                key=lambda item: item[1]["summary"][
                    "avg_request_compute_fraction"
                ],
            )
            oracle_row = copy.deepcopy(chosen_row)
            oracle_row["strategy"] = ORACLE_STRATEGY
            oracle_row["kind"] = "feasible_oracle_envelope"
            oracle_row["config"] = {
                "name": ORACLE_STRATEGY,
                "exact_optimum_proven": False,
                "selection_scope": "paired_seed_and_ssu",
                "candidates": list(candidates),
                "physical_capacity_preserved": True,
            }
            candidate_values = {
                name: row["summary"]["avg_request_compute_fraction"]
                for name, row in candidates.items()
            }
            oracle_row["summary"]["oracle_envelope"] = {
                "chosen_candidate": chosen_name,
                "candidate_request_compute_fraction": candidate_values,
                "exact_optimum_proven": False,
                "interpretation": "best_observed_physically_feasible_schedule",
            }
            routing_index[(seed, ssu, ORACLE_STRATEGY)] = oracle_row
            selections[f"{seed}:{ssu}"] = oracle_row["summary"][
                "oracle_envelope"
            ]
    return selections


def aggregate(index):
    values = defaultdict(dict)
    for ssu in SSUS:
        for strategy in STRATEGIES:
            rows = [index[(seed, ssu, strategy)] for seed in SEEDS]
            request = [
                row["summary"]["avg_request_compute_fraction"] for row in rows
            ]
            fleet = [
                row["summary"]["fleet_npu_compute_utilization"] for row in rows
            ]
            makespan = [row["summary"]["makespan_ms"] for row in rows]
            pressure_reports = [
                row["summary"]["pressure_reports"] for row in rows
            ]
            values[(ssu, strategy)] = {
                "kind": (
                    "feasible_oracle_envelope"
                    if strategy == ORACLE_STRATEGY
                    else "simulation"
                ),
                "request_by_seed": dict(zip(SEEDS, request)),
                "request_mean": float(np.mean(request)),
                "request_min": min(request),
                "request_max": max(request),
                "fleet_by_seed": dict(zip(SEEDS, fleet)),
                "fleet_mean": float(np.mean(fleet)),
                "fleet_min": min(fleet),
                "fleet_max": max(fleet),
                "makespan_by_seed_ms": dict(zip(SEEDS, makespan)),
                "makespan_mean_ms": float(np.mean(makespan)),
                "makespan_min_ms": min(makespan),
                "makespan_max_ms": max(makespan),
                "pressure_reports_mean": float(np.mean(pressure_reports)),
                "pressure_telemetry_mb_mean": float(
                    np.mean(
                        [
                            row["summary"]["pressure_telemetry_mb"]
                            for row in rows
                        ]
                    )
                ),
                "pressure_telemetry_gib_mean": float(
                    np.mean(pressure_reports) * 256 * 4 / (1024**3)
                ),
                "avg_ssd_queue_wait_ms": float(
                    np.mean(
                        [
                            row["summary"]["avg_queue_wait_ms_per_block"]
                            for row in rows
                        ]
                    )
                ),
                "avg_npu_link_queue_wait_ms": float(
                    np.mean(
                        [
                            row["summary"]["avg_npu_link_queue_wait_ms"]
                            for row in rows
                        ]
                    )
                ),
                "request_compute_fraction_jain_mean": float(
                    np.mean(
                        [
                            row["summary"]["request_compute_fraction_jain"]
                            for row in rows
                        ]
                    )
                ),
                "category_request_mean": {
                    category: float(
                        np.mean(
                            [
                                row["summary"]["category_metrics"][category][
                                    "avg_request_compute_fraction"
                                ]
                                for row in rows
                            ]
                        )
                    )
                    for category in CATEGORIES
                },
            }
    return values


def _paired_requests(index, ssu, left, right):
    deltas = []
    for seed in SEEDS:
        lhs = {
            row["request_id"]: row
            for row in index[(seed, ssu, left)]["request_metrics"]
        }
        rhs = {
            row["request_id"]: row
            for row in index[(seed, ssu, right)]["request_metrics"]
        }
        deltas.extend(
            100.0
            * (
                lhs[request_id]["request_npu_utilization"]
                - rhs[request_id]["request_npu_utilization"]
            )
            for request_id in sorted(lhs)
        )
    return {
        "count": len(deltas),
        "mean_delta_pp": float(np.mean(deltas)),
        "median_delta_pp": float(np.median(deltas)),
        "improved": sum(delta > 1e-12 for delta in deltas),
        "regressed": sum(delta < -1e-12 for delta in deltas),
        "equal": sum(abs(delta) <= 1e-12 for delta in deltas),
    }


def _comparison(values, index, left, right):
    by_ssu = {}
    for ssu in SSUS:
        lhs = values[(ssu, left)]
        rhs = values[(ssu, right)]
        by_ssu[str(ssu)] = {
            "request_delta_pp": 100.0
            * (lhs["request_mean"] - rhs["request_mean"]),
            "fleet_delta_pp": 100.0
            * (lhs["fleet_mean"] - rhs["fleet_mean"]),
            "makespan_delta_ms": lhs["makespan_mean_ms"]
            - rhs["makespan_mean_ms"],
            "pressure_reports_delta": lhs["pressure_reports_mean"]
            - rhs["pressure_reports_mean"],
            "request_compute_fraction_jain_delta": lhs[
                "request_compute_fraction_jain_mean"
            ]
            - rhs["request_compute_fraction_jain_mean"],
            "paired_requests": _paired_requests(index, ssu, left, right),
            "category_request_delta_pp": {
                category: 100.0
                * (
                    lhs["category_request_mean"][category]
                    - rhs["category_request_mean"][category]
                )
                for category in CATEGORIES
            },
        }
    return {
        "left": left,
        "right": right,
        "by_ssu": by_ssu,
        "cross_ssu_request_delta_pp": float(
            np.mean([row["request_delta_pp"] for row in by_ssu.values()])
        ),
        "cross_ssu_fleet_delta_pp": float(
            np.mean([row["fleet_delta_pp"] for row in by_ssu.values()])
        ),
        "cross_ssu_makespan_delta_ms": float(
            np.mean([row["makespan_delta_ms"] for row in by_ssu.values()])
        ),
    }


def comparisons(values, index):
    pairs = {
        "rr_minus_final_baseline": ("path_rr_baseline", "baseline"),
        "refresh1_minus_final_baseline": ("refresh1", "baseline"),
        "refresh8_minus_final_baseline": ("refresh8", "baseline"),
        "layer_once_minus_final_baseline": ("layer_once", "baseline"),
        "oracle_minus_final_baseline": (ORACLE_STRATEGY, "baseline"),
        "refresh1_minus_refresh8": ("refresh1", "refresh8"),
        "refresh8_minus_layer_once": ("refresh8", "layer_once"),
    }
    return {
        name: _comparison(values, index, left, right)
        for name, (left, right) in pairs.items()
    }


def _style_axis(axis):
    axis.set_xticks(SSUS)
    axis.grid(alpha=0.25, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_overview(values, output_path):
    styles = ("s-", "o-", "^-", "D-", "P-", "x-")
    base_experiment.plot_results(
        {
            "experiment": {"n_layers": 16},
            "ssus": list(SSUS),
            "series": [
                {
                    "label": LABELS[strategy],
                    "style": style,
                    "values": [
                        100.0 * values[(ssu, strategy)]["request_mean"]
                        for ssu in SSUS
                    ],
                }
                for strategy, style in zip(STRATEGIES, styles)
            ],
            "ylabel": "Average NPU Utilization (%)",
            "title": "16-layer routing on shared SSD→NPU data plane",
        },
        output_path,
    )


def plot_system_metrics(values, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), constrained_layout=True)
    for strategy in STRATEGIES:
        axes[0].plot(
            SSUS,
            [100.0 * values[(ssu, strategy)]["fleet_mean"] for ssu in SSUS],
            marker=MARKERS[strategy],
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
        axes[1].plot(
            SSUS,
            [values[(ssu, strategy)]["makespan_mean_ms"] for ssu in SSUS],
            marker=MARKERS[strategy],
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
        axes[2].plot(
            SSUS,
            [
                values[(ssu, strategy)]["pressure_telemetry_gib_mean"]
                for ssu in SSUS
            ],
            marker=MARKERS[strategy],
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
    titles = (
        ("Fleet NPU compute utilization", "Percent"),
        ("Global makespan", "Milliseconds"),
        ("Path-pressure telemetry per run", "GiB"),
    )
    for axis, (title, ylabel) in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Number of SSUs")
        axis.set_ylabel(ylabel)
        _style_axis(axis)
    axes[0].legend(fontsize=8, frameon=False)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_deltas(comparison_data, output_path):
    figure, axis = plt.subplots(figsize=(11.8, 5.4), constrained_layout=True)
    series = (
        ("rr_minus_final_baseline", "path_rr_baseline"),
        ("refresh1_minus_final_baseline", "refresh1"),
        ("refresh8_minus_final_baseline", "refresh8"),
        ("layer_once_minus_final_baseline", "layer_once"),
    )
    for comparison_name, strategy in series:
        axis.plot(
            SSUS,
            [
                comparison_data[comparison_name]["by_ssu"][str(ssu)][
                    "request_delta_pp"
                ]
                for ssu in SSUS
            ],
            marker=MARKERS[strategy],
            color=COLORS[strategy],
            label=f"{LABELS[strategy]} − final baseline",
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_title("Request compute-fraction change relative to final Path-0 baseline")
    axis.set_xlabel("Number of SSUs")
    axis.set_ylabel("Delta (percentage points)")
    _style_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, ncol=2)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_report(values, comparison_data, oracle_selections, output_dir):
    lines = [
        "# 最终 Path 选路对比（共用物理 SSD→NPU 数据面）",
        "",
        "正式配置：128 NPU、16 层、SSU=`8/16/28/40/56/80/112`、seed=`42/43`、每 NPU 独立 `0–5 ms` 到达延迟、batch=1、每条 I/O 的发行间隔为 `0.1 us`。",
        "",
        "主图纵轴按要求简写为 `Average NPU Utilization (%)`，数值定义仍是 128 个请求等权的 `avg_request_compute_fraction`，不是 fleet utilization。",
        "",
        "五个可执行策略全部使用同一个 `qos_static_cir` SSD/QoS 数据面、同一个 `20/6/8/6` CIR 和 `12/4/12/4` Path 布局；唯一实验变量是 NPU 写入命令的 Path ID。最终 baseline 把全部 I/O 写入 Path 0；No-state Path RR 完全不读取 Path/I/O 状态，只在类别合法 Path 内轮转；三个 Static 策略分别每条 I/O、每 8 条 I/O、每层/SSU 读取一次压力表。",
        "",
        "黑色 Oracle 曲线不再使用删除跨 NPU 竞争的 fluid relaxation。它在每个 paired seed/SSU 中，从五个现有策略和一个 demand-weighted SJF Oracle 候选中选择 request compute fraction 最高的真实事件仿真；所有候选都保留原 block→SSD 映射、每盘单命令 40 GB/s、每 NPU 单接收队列 50 GB/s、原始到达时间和逐层依赖。它是当前找到的最佳可行包络，不声称已经证明数学精确最优。",
        "",
        "## Request compute fraction（双 seed 均值）",
        "",
        "| SSU | Final Path0 | No-state Path RR | Refresh1 | Refresh8 | Layer once | Capacity-constrained Oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ssu in SSUS:
        cells = [
            f"{100.0 * values[(ssu, strategy)]['request_mean']:.3f}%"
            for strategy in STRATEGIES
        ]
        lines.append(f"| {ssu} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 相对最终 Path-0 baseline",
            "",
            "正数表示 request compute fraction 更高；fleet 正数更高，makespan 负数更好。",
            "",
            "| 策略 | 跨七个 SSU request Δ | fleet Δ | makespan Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    comparisons_to_baseline = (
        ("No-state Path RR", "rr_minus_final_baseline"),
        ("Refresh1", "refresh1_minus_final_baseline"),
        ("Refresh8", "refresh8_minus_final_baseline"),
        ("Layer once", "layer_once_minus_final_baseline"),
        ("Capacity-constrained Oracle", "oracle_minus_final_baseline"),
    )
    for label, name in comparisons_to_baseline:
        row = comparison_data[name]
        lines.append(
            f"| {label} | {row['cross_ssu_request_delta_pp']:+.3f} pp | "
            f"{row['cross_ssu_fleet_delta_pp']:+.3f} pp | "
            f"{row['cross_ssu_makespan_delta_ms']:+.3f} ms |"
        )

    mid_ssus = (28, 40, 56, 80)
    refresh1_vs_baseline = comparison_data[
        "refresh1_minus_final_baseline"
    ]
    mid_refresh1_delta = float(
        np.mean(
            [
                refresh1_vs_baseline["by_ssu"][str(ssu)][
                    "request_delta_pp"
                ]
                for ssu in mid_ssus
            ]
        )
    )
    lines.extend(
        [
            "",
            "## 收益来自哪里",
            "",
            f"在用户关注的 28–80 SSU 区间，Refresh1 相对最终 Path-0 baseline 的 request compute fraction 平均提高 **{mid_refresh1_delta:+.3f} pp**。但它不是按每个 NPU 的 10/30 GB/s demand 动态配置 CIR：本矩阵的 CIR 始终是固定的类别预算 `20/6/8/6`。因此这里测到的是类别隔离、跨 Path 分散和压力感知选路的联合效果，不是 per-NPU 精确带宽匹配。",
            "",
            "下面按类别拆开 Refresh1−Path0；正数表示该类别变快。",
            "",
            "| SSU | SS | SL | LS | LL |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in mid_ssus:
        category = refresh1_vs_baseline["by_ssu"][str(ssu)][
            "category_request_delta_pp"
        ]
        lines.append(
            f"| {ssu} | {category['SS']:+.3f} pp | {category['SL']:+.3f} pp | "
            f"{category['LS']:+.3f} pp | {category['LL']:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "证据非常集中：中间 SSU 区间的均值提升主要由 SS 请求驱动，而 LL 在四个点全部回退。也就是说 Static QoS 重新分配了服务机会，并没有让所有输入同时变快。No-state Path RR 已经能通过多 Path 和 CIR 隔离获得一部分收益；压力感知选择进一步减少同类 Path 热点。",
            "",
            f"跨七个 SSU，Refresh1 的 request 指标相对 Path0 平均为 **{refresh1_vs_baseline['cross_ssu_request_delta_pp']:+.3f} pp**，但 fleet 为 **{refresh1_vs_baseline['cross_ssu_fleet_delta_pp']:+.3f} pp**、makespan 为 **{refresh1_vs_baseline['cross_ssu_makespan_delta_ms']:+.3f} ms**。原因是 request 指标让 128 个请求等权，而 fleet/makespan 由最晚完成的 LL 尾请求主导；QoS 改善大量短请求的同时延后尾部 LL，因此两组指标方向可以相反。",
            "",
            "## Capacity-constrained Oracle 的含义",
            "",
            "旧的水平 fluid 曲线已从最终图中删除，因为它为每个请求复制 SSD 容量，不能反映共享盘竞争。新的黑线由实际可运行 schedule 构成，所以 SSU 少时会随共享容量下降，也能合法报告 fleet 和 makespan。",
            "",
            "它仍不是带最优性证书的 exact optimum：原问题约有 170 万条非抢占命令，是带动态层 release、SSD→NPU 两阶段约束和非线性 request-utilization 目标的大规模 job-shop/coflow 排程。这里的 `exact_optimum_proven` 固定为 `false`。因此黑线应解读为 **unknown optimum 的可行下界**，而不是新的数学上界。",
            "",
            "该包络的选择目标就是主图的平均 request utilization，不是 fleet/makespan；它会优先改善大量短请求并允许 LL 尾部变慢。因此它在小 SSU 点明显提高主图指标，但不代表全局尾长也最优。",
            "",
            "| SSU | seed 42 选中 | seed 43 选中 |",
            "|---:|---|---|",
        ]
    )
    selection_labels = {
        **{strategy: LABELS[strategy] for strategy in SIMULATION_STRATEGIES},
        ORACLE_CANDIDATE_STRATEGY: "Demand-weighted SJF Oracle candidate",
    }
    for ssu in SSUS:
        chosen = [
            selection_labels[
                oracle_selections[f"{seed}:{ssu}"]["chosen_candidate"]
            ]
            for seed in SEEDS
        ]
        lines.append(f"| {ssu} | {chosen[0]} | {chosen[1]} |")

    lines.extend(
        [
            "",
            "## 五个标准策略中每个 SSU 的最佳策略",
            "",
            "| SSU | 最佳策略 | request compute fraction | 相对 Path0 |",
            "|---:|---|---:|---:|",
        ]
    )
    for ssu in SSUS:
        best = max(
            SIMULATION_STRATEGIES,
            key=lambda strategy: values[(ssu, strategy)]["request_mean"],
        )
        delta_pp = 100.0 * (
            values[(ssu, best)]["request_mean"]
            - values[(ssu, "baseline")]["request_mean"]
        )
        lines.append(
            f"| {ssu} | {LABELS[best]} | "
            f"{100.0 * values[(ssu, best)]['request_mean']:.3f}% | "
            f"{delta_pp:+.3f} pp |"
        )

    lines.extend(
        [
            "",
            "## 刷新频率与遥测",
            "",
            "| SSU | Layer reads | Refresh8 reads | Refresh1 reads | Refresh1−Refresh8 request | Refresh8−Layer request |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    per_io = comparison_data["refresh1_minus_refresh8"]
    window = comparison_data["refresh8_minus_layer_once"]
    for ssu in SSUS:
        reads = {
            strategy: values[(ssu, strategy)]["pressure_reports_mean"]
            for strategy in ("layer_once", "refresh8", "refresh1")
        }
        lines.append(
            f"| {ssu} | {reads['layer_once']:.0f} | {reads['refresh8']:.0f} | "
            f"{reads['refresh1']:.0f} | "
            f"{per_io['by_ssu'][str(ssu)]['request_delta_pp']:+.3f} pp | "
            f"{window['by_ssu'][str(ssu)]['request_delta_pp']:+.3f} pp |"
        )

    lines.extend(
        [
            "",
            "## 图",
            "",
            "![最终六曲线 NPU utilization 对比](01_routing_refresh_finite_issue.png)",
            "",
            "![五个普通策略和容量约束 Oracle 的系统指标](02_simulation_system_metrics.png)",
            "",
            "![相对最终 baseline 的差值](03_strategy_deltas_vs_final_baseline.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def analyze(input_path, output_dir, oracle_path=None):
    data = json.loads(input_path.read_text())
    index = validate(data)
    oracle_path = (
        input_path.with_name("capacity_constrained_oracle_results.json")
        if oracle_path is None
        else oracle_path
    )
    oracle_data = json.loads(oracle_path.read_text())
    candidate_index = validate_oracle_candidates(oracle_data, index)
    oracle_selections = add_oracle_envelope(index, candidate_index)
    values = aggregate(index)
    comparison_data = comparisons(values, index)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(values, output_dir / "01_routing_refresh_finite_issue.png")
    plot_system_metrics(values, output_dir / "02_simulation_system_metrics.png")
    plot_deltas(
        comparison_data,
        output_dir / "03_strategy_deltas_vs_final_baseline.png",
    )
    write_report(values, comparison_data, oracle_selections, output_dir)
    result = {
        "seeds": list(SEEDS),
        "ssu_list": list(SSUS),
        "strategies": list(STRATEGIES),
        "simulation_strategies": list(SIMULATION_STRATEGIES),
        "oracle": {
            "name": ORACLE_STRATEGY,
            "candidate_results": str(oracle_path),
            "interpretation": "best_observed_physically_feasible_schedule",
            "exact_optimum_proven": False,
            "selections": oracle_selections,
        },
        "aggregated": {
            f"{ssu}:{strategy}": row
            for (ssu, strategy), row in sorted(values.items())
        },
        "comparisons": comparison_data,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/routing_refresh_concurrency/results.json"),
    )
    parser.add_argument(
        "--oracle-input",
        type=Path,
        default=Path(
            "results/routing_refresh_concurrency/"
            "capacity_constrained_oracle_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/routing_refresh_concurrency"),
    )
    args = parser.parse_args()
    result = analyze(args.input, args.output_dir, args.oracle_input)
    print(
        f"validated {len(result['seeds'])} seeds × "
        f"{len(result['ssu_list'])} SSU points"
    )
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
