"""Analyze the two-seed four-strategy finite-issue sensitivity matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43)
SSUS = (40, 56, 80)
STRATEGIES = (
    "baseline",
    "static_cir_before",
    "static_cir_after",
    "fluid_no_contention_bound",
)
CATEGORIES = ("SS", "SL", "LS", "LL")
LABELS = {
    "baseline": "Baseline (NPU-RR)",
    "static_cir_before": "Static CIR before (20/4/12/4)",
    "static_cir_after": "Static CIR after (20/6/8/6)",
    "fluid_no_contention_bound": "Ideal fluid bound (no contention)",
}
COLORS = {
    "baseline": "#4c78a8",
    "static_cir_before": "#f58518",
    "static_cir_after": "#54a24b",
    "fluid_no_contention_bound": "#b8b51b",
}


def _request_fraction(row):
    key = (
        "avg_request_compute_fraction_upper_bound"
        if row["kind"] == "upper_bound"
        else "avg_request_compute_fraction"
    )
    return row["summary"][key]


def validate(data):
    if not data["complete"]:
        raise AssertionError("four-strategy concurrency matrix is incomplete")
    if tuple(data["selected_strategies"]) != STRATEGIES:
        raise AssertionError("strategy set mismatch")
    experiment = data["experiment"]
    runtime = experiment["runtime"]
    if runtime["num_npu"] != 128 or runtime["n_layers"] != 16:
        raise AssertionError("formal NPU/layer contract mismatch")
    if tuple(runtime["ssu_list"]) != SSUS or tuple(runtime["seeds"]) != SEEDS:
        raise AssertionError("formal SSU/seed contract mismatch")
    if runtime["arrival_delay_ms"] != [0.0, 5.0]:
        raise AssertionError("arrival-delay contract mismatch")
    rows = data["results"]
    expected_keys = {
        (seed, ssu, strategy)
        for seed in SEEDS
        for ssu in SSUS
        for strategy in STRATEGIES
    }
    index = {
        (row["seed"], row["num_ssu"], row["strategy"]): row for row in rows
    }
    if set(index) != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError("matrix key mismatch")
    expected_kinds = {
        "baseline": "simulation",
        "static_cir_before": "simulation",
        "static_cir_after": "simulation",
        "fluid_no_contention_bound": "upper_bound",
    }
    expected_static = {
        "static_cir_before": (
            "static_20_4_12_4_paths_12_4_12_4",
            [20.0, 4.0, 12.0, 4.0],
        ),
        "static_cir_after": (
            "low_protect_cir_20_6_8_6_current_paths",
            [20.0, 6.0, 8.0, 6.0],
        ),
    }
    for seed in SEEDS:
        workload_hashes = set()
        for ssu in SSUS:
            paired = [index[(seed, ssu, name)] for name in STRATEGIES]
            if len({row["workload_fingerprint"] for row in paired}) != 1:
                raise AssertionError("unpaired workload")
            if len({row["placement_hash"] for row in paired}) != 1:
                raise AssertionError("unpaired placement")
            workload_hashes.add(paired[0]["workload_fingerprint"])
            for row in paired:
                strategy = row["strategy"]
                if row["kind"] != expected_kinds[strategy]:
                    raise AssertionError("strategy kind mismatch")
                if row["kind"] == "simulation":
                    if not all(row["summary"]["invariants"].values()):
                        raise AssertionError("data-plane invariant failure")
                    config = row["config"]["client_io"]
                    if config["submit_batch_size"] != 1:
                        raise AssertionError("finite issue run is not per-I/O")
                    if config["issue_interval_us"] != 0.1:
                        raise AssertionError("finite issue interval mismatch")
                    if strategy in expected_static:
                        if row["config"]["policy"] != "qos_static_cir":
                            raise AssertionError("static policy mismatch")
                        if config["pressure_window_io"] != 8:
                            raise AssertionError("static pressure cadence mismatch")
                        profile_name, cir = expected_static[strategy]
                        if row["config"]["profile_name"] != profile_name:
                            raise AssertionError("static profile mismatch")
                        if row["config"]["category_cir_gbps"] != cir:
                            raise AssertionError("static CIR mismatch")
                        if row["config"]["category_paths_per_group"] != [
                            12,
                            4,
                            12,
                            4,
                        ]:
                            raise AssertionError("static Path allocation mismatch")
                    else:
                        if row["config"]["policy"] != "baseline_bypass":
                            raise AssertionError("baseline policy mismatch")
                        if config["pressure_window_io"] is not None:
                            raise AssertionError("baseline read Path pressure")
        if len(workload_hashes) != 1:
            raise AssertionError("workload changed across SSU counts")
    return index


def aggregate(index):
    values = defaultdict(dict)
    for ssu in SSUS:
        for strategy in STRATEGIES:
            rows = [index[(seed, ssu, strategy)] for seed in SEEDS]
            request = [_request_fraction(row) for row in rows]
            fleet = [
                row["summary"].get("fleet_npu_compute_utilization") for row in rows
            ]
            makespan = [row["summary"].get("makespan_ms") for row in rows]
            category_request_mean = (
                {
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
                }
                if all(row["kind"] == "simulation" for row in rows)
                else None
            )
            category_io_wait_mean_ms = (
                {
                    category: float(
                        np.mean(
                            [
                                row["summary"]["category_metrics"][category][
                                    "avg_io_wait_total_ms"
                                ]
                                for row in rows
                            ]
                        )
                    )
                    for category in CATEGORIES
                }
                if all(row["kind"] == "simulation" for row in rows)
                else None
            )
            values[(ssu, strategy)] = {
                "request_by_seed": dict(zip(SEEDS, request)),
                "request_mean": float(np.mean(request)),
                "request_min": min(request),
                "request_max": max(request),
                "fleet_by_seed": dict(zip(SEEDS, fleet)),
                "fleet_mean": (
                    float(np.mean(fleet)) if all(value is not None for value in fleet) else None
                ),
                "makespan_by_seed": dict(zip(SEEDS, makespan)),
                "makespan_mean_ms": (
                    float(np.mean(makespan))
                    if all(value is not None for value in makespan)
                    else None
                ),
                "category_request_mean": category_request_mean,
                "category_io_wait_mean_ms": category_io_wait_mean_ms,
            }
    return values


def _paired_comparison(values, left, right):
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
            "category_request_delta_pp": {
                category: 100.0
                * (
                    lhs["category_request_mean"][category]
                    - rhs["category_request_mean"][category]
                )
                for category in CATEGORIES
            },
            "category_io_wait_delta_ms": {
                category: lhs["category_io_wait_mean_ms"][category]
                - rhs["category_io_wait_mean_ms"][category]
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


def comparisons(values):
    return {
        "static_after_minus_before": _paired_comparison(
            values, "static_cir_after", "static_cir_before"
        ),
        "static_after_minus_baseline": _paired_comparison(
            values, "static_cir_after", "baseline"
        ),
    }


def tail_diagnostics(index):
    result = {}
    for seed in SEEDS:
        for ssu in SSUS:
            for strategy in STRATEGIES[:3]:
                row = index[(seed, ssu, strategy)]
                request = max(
                    row["request_metrics"],
                    key=lambda item: item["arrival_delay_ms"]
                    + item["request_compute_ms"]
                    + item["io_wait_total_ms"],
                )
                completion_ms = (
                    request["arrival_delay_ms"]
                    + request["request_compute_ms"]
                    + request["io_wait_total_ms"]
                )
                if not np.isclose(
                    completion_ms,
                    row["summary"]["makespan_ms"],
                    rtol=0.0,
                    atol=1e-8,
                ):
                    raise AssertionError("tail request does not explain makespan")
                result[f"{seed}:{ssu}:{strategy}"] = {
                    "request_id": request["request_id"],
                    "category": request["category"],
                    "arrival_delay_ms": request["arrival_delay_ms"],
                    "request_compute_ms": request["request_compute_ms"],
                    "io_wait_total_ms": request["io_wait_total_ms"],
                    "completion_ms": completion_ms,
                }
    return result


def plot(values, output_path):
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    markers = ("o", "s", "^", "<")
    for strategy, marker in zip(STRATEGIES, markers):
        means = [100.0 * values[(ssu, strategy)]["request_mean"] for ssu in SSUS]
        lower = [100.0 * values[(ssu, strategy)]["request_min"] for ssu in SSUS]
        upper = [100.0 * values[(ssu, strategy)]["request_max"] for ssu in SSUS]
        axis.fill_between(SSUS, lower, upper, color=COLORS[strategy], alpha=0.10)
        axis.plot(
            SSUS,
            means,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            linestyle="--" if strategy == "fluid_no_contention_bound" else "-",
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
    axis.set_title("Finite-issue sensitivity: mean per-request NPU compute fraction")
    axis.set_ylabel("Compute / (compute + exposed I/O stall)  (%)")
    axis.set_xlabel("Number of SSUs")
    axis.set_xticks(SSUS)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _load_zero_issue(path):
    data = json.loads(path.read_text())
    if not data["complete"]:
        raise AssertionError("zero-issue reference is incomplete")
    runtime = data["experiment"]["runtime"]
    if runtime["num_npu"] != 128 or runtime["n_layers"] != 16:
        raise AssertionError("zero-issue formal contract mismatch")
    if runtime["ssu_list"] != list(SSUS) or runtime["seeds"]["workload"] != 42:
        raise AssertionError("zero-issue seed/SSU mismatch")
    required = {
        "baseline",
        "current_refresh8",
        "tune__low_protect_cir_20_6_8_6_current_paths",
        "isolated_no_contention_bound",
    }
    if set(data["selected_strategies"]) != required:
        raise AssertionError("zero-issue strategy set mismatch")
    index = {(row["num_ssu"], row["strategy"]): row for row in data["results"]}
    expected_keys = {(ssu, strategy) for ssu in SSUS for strategy in required}
    if set(index) != expected_keys or len(data["results"]) != len(expected_keys):
        raise AssertionError("zero-issue matrix key mismatch")
    names = {
        "baseline": "baseline",
        "static_cir_before": "current_refresh8",
        "static_cir_after": "tune__low_protect_cir_20_6_8_6_current_paths",
    }
    return {
        (ssu, strategy): index[(ssu, source)]["summary"]
        for strategy, source in names.items()
        for ssu in SSUS
    }


def write_report(values, zero_issue, comparison_data, tails, output_dir):
    lines = [
        "# 四策略有限发行敏感性",
        "",
        "正式配置：128 NPU、16 层、SSU=40/56/80、seed=42/43、每 NPU 0–5 ms 独立到达延迟。三个可运行策略统一使用 batch=1、每条命令 0.1 us 发行间隔；两个 Static 策略统一 refresh8，只改变固定 CIR。",
        "",
        "0.1 us 是并发可见性的因果敏感性参数，不宣称是某款 NPU 的实测发行延迟。阴影表示两个 seed 的最小–最大范围，主曲线是两 seed 均值；图中严格只有用户指定的四个方案。",
        "",
        "## 两 seed 均值",
        "",
        "每格为 `request compute fraction / fleet compute utilization`；Fluid bound 没有联合 makespan，因此 fleet 为 N/A。",
        "",
        "| SSU | Baseline | Static before | Static after | Fluid bound |",
        "|---:|---:|---:|---:|---:|",
    ]
    for ssu in SSUS:
        cells = []
        for strategy in STRATEGIES:
            row = values[(ssu, strategy)]
            request = 100.0 * row["request_mean"]
            fleet = row["fleet_mean"]
            cells.append(
                f"{request:.3f}% / {100.0*fleet:.3f}%"
                if fleet is not None
                else f"{request:.3f}% / N/A"
            )
        lines.append(f"| {ssu} | " + " | ".join(cells) + " |")
    after_before = comparison_data["static_after_minus_before"]
    after_baseline = comparison_data["static_after_minus_baseline"]
    lines.extend(
        [
            "",
            "## 优化后的 Static CIR 改善了什么",
            "",
            "`Static after` 只把类别 CIR 从 `20/4/12/4` 改成 `20/6/8/6`；Path 数仍为 `12/4/12/4`，选路仍为 refresh8。正值表示优化后更好；makespan 负值表示优化后更快完成。",
            "",
            "| SSU | after − before request | after − before fleet | after − before makespan | after − baseline request | after − baseline fleet | after − baseline makespan |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in SSUS:
        tuned = after_before["by_ssu"][str(ssu)]
        baseline = after_baseline["by_ssu"][str(ssu)]
        lines.append(
            f"| {ssu} | {tuned['request_delta_pp']:+.3f} pp | {tuned['fleet_delta_pp']:+.3f} pp | {tuned['makespan_delta_ms']:+.3f} ms | {baseline['request_delta_pp']:+.3f} pp | {baseline['fleet_delta_pp']:+.3f} pp | {baseline['makespan_delta_ms']:+.3f} ms |"
        )
    lines.extend(
        [
            "",
            f"跨三个 SSU 等权平均，优化 CIR 相对优化前的 request 指标提高 **{after_before['cross_ssu_request_delta_pp']:+.3f} pp**、fleet 提高 **{after_before['cross_ssu_fleet_delta_pp']:+.3f} pp**，makespan 缩短 **{-after_before['cross_ssu_makespan_delta_ms']:.3f} ms**。相对 baseline，request 指标提高 **{after_baseline['cross_ssu_request_delta_pp']:+.3f} pp**，但 fleet 仍低 **{after_baseline['cross_ssu_fleet_delta_pp']:+.3f} pp**，平均 makespan 长 **{after_baseline['cross_ssu_makespan_delta_ms']:.3f} ms**。",
            "",
            "### 哪类输入变快或变慢",
            "",
            "下表仍是两个 seed 的等权均值。request 正值表示计算占比提高；I/O wait 负值表示暴露等待缩短。",
            "",
            "| SSU | 类别 | after − before request | after − before I/O wait | after − baseline request | after − baseline I/O wait |",
            "|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in SSUS:
        tuned_categories = after_before["by_ssu"][str(ssu)][
            "category_request_delta_pp"
        ]
        baseline_categories = after_baseline["by_ssu"][str(ssu)][
            "category_request_delta_pp"
        ]
        tuned_wait = after_before["by_ssu"][str(ssu)][
            "category_io_wait_delta_ms"
        ]
        baseline_wait = after_baseline["by_ssu"][str(ssu)][
            "category_io_wait_delta_ms"
        ]
        for category in CATEGORIES:
            lines.append(
                f"| {ssu} | {category} | {tuned_categories[category]:+.3f} pp | {tuned_wait[category]:+.3f} ms | {baseline_categories[category]:+.3f} pp | {baseline_wait[category]:+.3f} ms |"
            )
    lines.extend(
        [
            "",
            "把 4 GB/s 从 LS 转给 SL/LL 后，SL 和 LL 在三个 SSU 点都改善，LS 在三个点都回退；SS 的方向随容量而变。净效果仍为正，说明原先 `20/4/12/4` 对 SL/LL 保护不足，但 `20/6/8/6` 不是每类输入都更快的支配性解。",
            "",
            "## 为什么 request 更高，fleet 仍略低 baseline",
            "",
            "request 指标先对 128 个请求分别计算 `compute/(compute+暴露 I/O stall)` 再等权平均；fleet 则是 `总计算时间/(128×全局 makespan)`。因此大量请求变快可以提高前者，而一个长计算尾部请求稍慢就会拉长 makespan、压低后者。",
            "",
            "| Seed | SSU | 决定尾部的请求 | 类别 | baseline stall | Static after stall | makespan Δ |",
            "|---:|---:|---:|:---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for ssu in SSUS:
            baseline_tail = tails[f"{seed}:{ssu}:baseline"]
            after_tail = tails[f"{seed}:{ssu}:static_cir_after"]
            if (
                baseline_tail["request_id"] != after_tail["request_id"]
                or baseline_tail["category"] != after_tail["category"]
            ):
                raise AssertionError("paired policies changed the tail identity")
            delta = after_tail["completion_ms"] - baseline_tail["completion_ms"]
            lines.append(
                f"| {seed} | {ssu} | {baseline_tail['request_id']} | {baseline_tail['category']} | {baseline_tail['io_wait_total_ms']:.3f} ms | {after_tail['io_wait_total_ms']:.3f} ms | {delta:+.3f} ms |"
            )
    lines.extend(
        [
            "",
            "seed 42 的尾部 LL 请求在 Static after 下变慢；seed 43 则略快或相同。两个 seed 平均后尾长仍略增，所以当前结论是 request 平均稳定获益、fleet 有很小且 seed 敏感的损失，不能把旧的单 seed `-0.077 pp` 当作普遍常数。",
        ]
    )
    lines.extend(
        [
            "",
            "## Seed 42：有限发行相对历史零耗时模型",
            "",
            "该表只做同 seed 配对差值。有限发行同时把 atomic-8 改为 batch-1，并加入 0.1 us 时间推进，因此它是整体客户端发行模型敏感性，不应把全部差值只归因于发行延迟。",
            "",
            "| SSU | Baseline Δ | Static before Δ | Static after Δ |",
            "|---:|---:|---:|---:|",
        ]
    )
    for ssu in SSUS:
        deltas = []
        for strategy in STRATEGIES[:3]:
            finite = values[(ssu, strategy)]["request_by_seed"][42]
            zero = zero_issue[(ssu, strategy)]["avg_request_compute_fraction"]
            deltas.append(100.0 * (finite - zero))
        lines.append(
            f"| {ssu} | {deltas[0]:+.3f} pp | {deltas[1]:+.3f} pp | {deltas[2]:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "![Four-strategy finite issue](01_strategy_overview_finite_issue.png)",
            "",
            "无竞争上界的数学定义见 [IDEAL_NO_CONTENTION_BOUND_CN.md](../../doc/IDEAL_NO_CONTENTION_BOUND_CN.md)。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def analyze(input_path, zero_issue_path, output_dir):
    data = json.loads(input_path.read_text())
    index = validate(data)
    values = aggregate(index)
    comparison_data = comparisons(values)
    tails = tail_diagnostics(index)
    zero_issue = _load_zero_issue(zero_issue_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot(values, output_dir / "01_strategy_overview_finite_issue.png")
    write_report(values, zero_issue, comparison_data, tails, output_dir)
    analysis = {
        "seeds": list(SEEDS),
        "ssu_list": list(SSUS),
        "strategies": list(STRATEGIES),
        "aggregated": {
            f"{ssu}:{strategy}": row
            for (ssu, strategy), row in sorted(values.items())
        },
        "comparisons": comparison_data,
        "tail_diagnostics": tails,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/four_strategy_concurrency/results.json"),
    )
    parser.add_argument(
        "--zero-issue",
        type=Path,
        default=Path("results/four_strategy_focus/results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/four_strategy_concurrency"),
    )
    args = parser.parse_args()
    result = analyze(args.input, args.zero_issue, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
