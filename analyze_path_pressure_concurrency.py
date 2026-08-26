"""Analyze the Path-pressure atomicity/no-telemetry paired experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from path_pressure_concurrency_probe import (
    ARRIVAL_DELAY_MAX_MS,
    N_LAYERS,
    NUM_NPU,
    SSU_LIST,
    _code_fingerprint,
    runtime,
    strategies,
)


DEFAULT_OUTPUT_DIR = Path("results/path_pressure_concurrency")


LABELS = {
    "baseline_atomic8": "Baseline, atomic-8",
    "no_pressure_rr_atomic8": "No Path state, atomic-8",
    "refresh8_atomic8": "Refresh-8, atomic-8",
    "per_io_atomic8": "Per-I/O live, atomic-8",
    "refresh8_batch1_zero": "Refresh-8, batch1/zero-time",
    "per_io_batch1_zero": "Per-I/O, batch1/zero-time",
    "baseline_issue01us": "Baseline, 0.1us issue",
    "no_pressure_rr_issue01us": "No Path state, 0.1us issue",
    "refresh8_issue01us": "Refresh-8, 0.1us issue",
    "per_io_issue01us": "Per-I/O live, 0.1us issue",
}


def _mean(values):
    values = tuple(values)
    return sum(values) / len(values)


def load_results(output_dir, seeds):
    datasets = []
    for seed in seeds:
        path = output_dir / f"results_seed{seed}.json"
        data = json.loads(path.read_text())
        if not data["complete"]:
            raise AssertionError(f"incomplete experiment: {path}")
        datasets.append(data)
    return datasets


def _has_npu_reentry(order):
    closed = set()
    previous = None
    for item in order:
        current = item["npu_id"]
        if previous is not None and current != previous:
            closed.add(previous)
            if current in closed:
                return True
        previous = current
    return False


def validate(datasets, seeds):
    expected_names = {strategy.name for strategy in strategies()}
    expected_configs = {strategy.name: strategy for strategy in strategies()}
    for data, seed in zip(datasets, seeds):
        experiment = data["experiment"]
        if experiment["code_fingerprint"] != _code_fingerprint():
            raise AssertionError(f"seed {seed}: code fingerprint mismatch")
        if experiment["runtime"] != runtime(seed):
            raise AssertionError(f"seed {seed}: runtime mismatch")
        if experiment["runtime"]["num_npu"] != NUM_NPU:
            raise AssertionError("NPU contract mismatch")
        if experiment["runtime"]["n_layers"] != N_LAYERS:
            raise AssertionError("layer contract mismatch")
        if experiment["runtime"]["ssu_list"] != list(SSU_LIST):
            raise AssertionError("SSU contract mismatch")
        if experiment["runtime"]["arrival_delay_ms"] != [
            0.0,
            ARRIVAL_DELAY_MAX_MS,
        ]:
            raise AssertionError("arrival delay contract mismatch")
        if data["selected_strategies"] != [
            strategy.name for strategy in strategies()
        ]:
            raise AssertionError(f"seed {seed}: selected strategy mismatch")
        rows = data["results"]
        if len(rows) != len(SSU_LIST) * len(expected_names):
            raise AssertionError(f"seed {seed} row count mismatch")
        by_ssu = defaultdict(list)
        for row in rows:
            by_ssu[row["num_ssu"]].append(row)
            config = expected_configs[row["strategy"]]
            if row["config"] != asdict(config):
                raise AssertionError("strategy configuration mismatch")
            if row["seeds"] != experiment["runtime"]["seeds"]:
                raise AssertionError("row seed mismatch")
            if row["summary"]["policy"] != config.policy:
                raise AssertionError("row policy mismatch")
            request_ids = {item["request_id"] for item in row["request_metrics"]}
            if request_ids != set(range(NUM_NPU)):
                raise AssertionError("request ID set mismatch")
            if not all(row["summary"]["invariants"].values()):
                raise AssertionError(f"invariant failure: {row['strategy']}")
            if row["strategy"].startswith(("baseline", "no_pressure")):
                if row["summary"]["pressure_reports"] != 0:
                    raise AssertionError("no-state strategy read Path pressure")
        if set(by_ssu) != set(SSU_LIST):
            raise AssertionError(f"seed {seed}: SSU set mismatch")
        for ssu, ssu_rows in by_ssu.items():
            if {row["strategy"] for row in ssu_rows} != expected_names:
                raise AssertionError(f"seed {seed}, SSU {ssu}: strategy mismatch")
            if len({row["workload_fingerprint"] for row in ssu_rows}) != 1:
                raise AssertionError("unpaired workload")
            if len({row["placement_hash"] for row in ssu_rows}) != 1:
                raise AssertionError("unpaired placement")
            for row in ssu_rows:
                if row["strategy"].endswith("issue01us"):
                    order = row["summary"]["client_submission"]["order_sample"]
                    if not _has_npu_reentry(order):
                        raise AssertionError("finite issue model did not interleave NPUs")


def aggregate(datasets):
    groups = defaultdict(list)
    for data in datasets:
        for row in data["results"]:
            groups[(row["num_ssu"], row["strategy"])].append(row)
    result = {}
    for key, rows in groups.items():
        sample_switches = []
        sample_distinct_npus = []
        sample_has_reentry = []
        for row in rows:
            order = row["summary"]["client_submission"]["order_sample"]
            sample_switches.append(
                sum(
                    left["npu_id"] != right["npu_id"]
                    for left, right in zip(order, order[1:])
                )
            )
            sample_distinct_npus.append(len({item["npu_id"] for item in order}))
            sample_has_reentry.append(_has_npu_reentry(order))
        result[key] = {
            "request": _mean(
                row["summary"]["avg_request_compute_fraction"] for row in rows
            ),
            "fleet": _mean(
                row["summary"]["fleet_npu_compute_utilization"] for row in rows
            ),
            "makespan_ms": _mean(
                row["summary"]["makespan_ms"] for row in rows
            ),
            "pressure_reports": _mean(
                row["summary"]["pressure_reports"] for row in rows
            ),
            "telemetry_mb": _mean(
                row["summary"]["pressure_telemetry_mb"] for row in rows
            ),
            "submission_rounds": _mean(
                row["summary"]["client_submission"]["rounds"] for row in rows
            ),
            "multi_npu_rounds": _mean(
                row["summary"]["client_submission"]["multi_npu_rounds"]
                for row in rows
            ),
            "sample_npu_switches": _mean(sample_switches),
            "sample_distinct_npus": _mean(sample_distinct_npus),
            "sample_reentry_fraction": _mean(sample_has_reentry),
        }
    return result


def paired_request_delta(datasets, left, right):
    deltas = []
    for data in datasets:
        by_key = {
            (row["num_ssu"], row["strategy"]): row for row in data["results"]
        }
        for ssu in SSU_LIST:
            left_rows = {
                row["request_id"]: row
                for row in by_key[(ssu, left)]["request_metrics"]
            }
            right_rows = {
                row["request_id"]: row
                for row in by_key[(ssu, right)]["request_metrics"]
            }
            deltas.extend(
                100.0
                * (
                    left_rows[request_id]["request_npu_utilization"]
                    - right_rows[request_id]["request_npu_utilization"]
                )
                for request_id in left_rows
            )
    return {
        "mean_delta_pp": _mean(deltas),
        "median_delta_pp": float(np.median(deltas)),
        "improved": sum(delta > 1e-12 for delta in deltas),
        "regressed": sum(delta < -1e-12 for delta in deltas),
        "equal": sum(abs(delta) <= 1e-12 for delta in deltas),
        "count": len(deltas),
    }


def comparisons(aggregated, datasets):
    pairs = {
        "per_io_minus_refresh8_atomic8": (
            "per_io_atomic8",
            "refresh8_atomic8",
        ),
        "per_io_minus_refresh8_batch1_zero": (
            "per_io_batch1_zero",
            "refresh8_batch1_zero",
        ),
        "per_io_minus_refresh8_issue01us": (
            "per_io_issue01us",
            "refresh8_issue01us",
        ),
        "refresh8_minus_no_state_issue01us": (
            "refresh8_issue01us",
            "no_pressure_rr_issue01us",
        ),
        "per_io_minus_no_state_issue01us": (
            "per_io_issue01us",
            "no_pressure_rr_issue01us",
        ),
    }
    result = {}
    for name, (left, right) in pairs.items():
        by_ssu = {}
        for ssu in SSU_LIST:
            lhs = aggregated[(ssu, left)]
            rhs = aggregated[(ssu, right)]
            by_ssu[str(ssu)] = {
                "request_delta_pp": 100.0 * (lhs["request"] - rhs["request"]),
                "fleet_delta_pp": 100.0 * (lhs["fleet"] - rhs["fleet"]),
                "makespan_delta_ms": lhs["makespan_ms"] - rhs["makespan_ms"],
            }
        result[name] = {
            "left": left,
            "right": right,
            "by_ssu": by_ssu,
            "cross_ssu_request_delta_pp": _mean(
                row["request_delta_pp"] for row in by_ssu.values()
            ),
            "cross_ssu_fleet_delta_pp": _mean(
                row["fleet_delta_pp"] for row in by_ssu.values()
            ),
            "paired_requests": paired_request_delta(datasets, left, right),
        }
    for base in ("baseline", "no_pressure_rr", "refresh8", "per_io"):
        name = f"{base}_issue01us_minus_atomic8"
        left = f"{base}_issue01us"
        right = f"{base}_atomic8"
        result[name] = {
            "left": left,
            "right": right,
            "by_ssu": {
                str(ssu): {
                    "request_delta_pp": 100.0
                    * (
                        aggregated[(ssu, left)]["request"]
                        - aggregated[(ssu, right)]["request"]
                    ),
                    "fleet_delta_pp": 100.0
                    * (
                        aggregated[(ssu, left)]["fleet"]
                        - aggregated[(ssu, right)]["fleet"]
                    ),
                    "makespan_delta_ms": aggregated[(ssu, left)]["makespan_ms"]
                    - aggregated[(ssu, right)]["makespan_ms"],
                }
                for ssu in SSU_LIST
            },
            "paired_requests": paired_request_delta(datasets, left, right),
        }
    return result


def plot_primary(aggregated, comparisons_data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    finite_issue = (
        "baseline_issue01us",
        "no_pressure_rr_issue01us",
        "refresh8_issue01us",
        "per_io_issue01us",
    )
    colors = ("#4c566a", "#bf616a", "#d08770", "#5e81ac")
    for name, color in zip(finite_issue, colors):
        axes[0, 0].plot(
            SSU_LIST,
            [100.0 * aggregated[(ssu, name)]["request"] for ssu in SSU_LIST],
            marker="o",
            label=LABELS[name],
            color=color,
        )
        axes[1, 1].plot(
            SSU_LIST,
            [100.0 * aggregated[(ssu, name)]["fleet"] for ssu in SSU_LIST],
            marker="o",
            label=LABELS[name],
            color=color,
        )
    axes[0, 0].set_title("Request-side NPU compute fraction")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set_title("Fleet NPU compute utilization")
    axes[1, 1].set_ylabel("Percent")

    x = np.arange(len(SSU_LIST))
    width = 0.25
    for index, key in enumerate(
        (
            "per_io_minus_refresh8_atomic8",
            "per_io_minus_refresh8_batch1_zero",
            "per_io_minus_refresh8_issue01us",
        )
    ):
        values = [
            comparisons_data[key]["by_ssu"][str(ssu)]["request_delta_pp"]
            for ssu in SSU_LIST
        ]
        axes[0, 1].bar(
            x + (index - 1.0) * width,
            values,
            width,
            label=(
                "Atomic-8 / zero-time"
                if index == 0
                else (
                    "Batch-1 / zero-time"
                    if index == 1
                    else "Batch-1 / 0.1us issue"
                )
            ),
        )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x, SSU_LIST)
    axes[0, 1].set_title("Per-I/O live minus refresh-8")
    axes[0, 1].set_ylabel("Request fraction delta (pp)")
    axes[0, 1].legend(fontsize=8)

    no_state_pairs = (
        ("refresh8_minus_no_state_issue01us", "Refresh-8 minus no state"),
        ("per_io_minus_no_state_issue01us", "Per-I/O minus no state"),
    )
    for index, (key, label) in enumerate(no_state_pairs):
        values = [
            comparisons_data[key]["by_ssu"][str(ssu)]["request_delta_pp"]
            for ssu in SSU_LIST
        ]
        axes[1, 0].bar(x + (index - 0.5) * width, values, width, label=label)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(x, SSU_LIST)
    axes[1, 0].set_title("Value of Path pressure (0.1us issue)")
    axes[1, 0].set_ylabel("Request fraction delta (pp)")
    axes[1, 0].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("SSU count")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "01_atomicity_and_no_pressure.png", dpi=180)
    plt.close(fig)


def plot_submission(aggregated, output_dir):
    names = (
        "no_pressure_rr_atomic8",
        "refresh8_atomic8",
        "per_io_atomic8",
        "refresh8_batch1_zero",
        "per_io_batch1_zero",
        "no_pressure_rr_issue01us",
        "refresh8_issue01us",
        "per_io_issue01us",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    telemetry = [
        _mean(aggregated[(ssu, name)]["telemetry_mb"] for ssu in SSU_LIST)
        for name in names
    ]
    switches = [
        _mean(aggregated[(ssu, name)]["sample_npu_switches"] for ssu in SSU_LIST)
        for name in names
    ]
    labels = [LABELS[name] for name in names]
    axes[0].bar(x, telemetry, color="#5e81ac")
    axes[0].set_title("256-entry Path-table telemetry")
    axes[0].set_ylabel("MB per run (cross-SSU mean)")
    axes[1].bar(x, switches, color="#a3be8c")
    axes[1].set_title("Cross-NPU switches in first 256 batch events")
    axes[1].set_ylabel("Batch-event switches (cross-SSU mean)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "02_submission_and_telemetry.png", dpi=180)
    plt.close(fig)


def write_report(aggregated, comparisons_data, output_dir, seeds):
    lines = [
        "# Path pressure、提交原子性与无状态选路消融",
        "",
        f"正式配置：128 NPU、16 层、SSU={list(SSU_LIST)}、seed={list(seeds)}、每 NPU 0–5 ms 独立到达延迟。所有策略共享 workload 与 block placement，并保留同一 SSD40→NPU50 单命令数据面。",
        "",
        "`atomic8` 是旧模型：一个 ready NPU 在同一客户端事件内连续提交最多 8 条，发行耗时为 0。`batch1_zero` 只把 batch 改为 1，但发行仍为 0；当 ready time 唯一时仍不能代表真实重叠。`issue01us` 令每个 NPU 每 0.1 us 串行发行一条，使其他 NPU、SSD completion 和仲裁事件能够在两条命令之间插入。0.1 us 是因果敏感性参数，不宣称是某款 NPU 的实测发行延迟。",
        "",
        "## 两个 NPU 利用率指标",
        "",
        "| SSU | strategy | request compute | fleet compute | makespan | Path reads |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for ssu in SSU_LIST:
        for name in (
            "baseline_issue01us",
            "no_pressure_rr_issue01us",
            "refresh8_issue01us",
            "per_io_issue01us",
        ):
            row = aggregated[(ssu, name)]
            lines.append(
                f"| {ssu} | {LABELS[name]} | {100*row['request']:.4f}% | {100*row['fleet']:.4f}% | {row['makespan_ms']:.3f} ms | {row['pressure_reports']:.0f} |"
            )

    lines.extend(
        [
            "",
            "## Per-I/O 实时读取是否被 batch 原子性掩盖",
            "",
            "| SSU | atomic8/zero-time | batch1/zero-time | batch1/0.1us issue |",
            "|---:|---:|---:|---:|",
        ]
    )
    for ssu in SSU_LIST:
        old = comparisons_data["per_io_minus_refresh8_atomic8"]["by_ssu"][str(ssu)]
        zero = comparisons_data["per_io_minus_refresh8_batch1_zero"]["by_ssu"][str(ssu)]
        new = comparisons_data["per_io_minus_refresh8_issue01us"]["by_ssu"][str(ssu)]
        lines.append(
            f"| {ssu} | {old['request_delta_pp']:+.6f} pp | {zero['request_delta_pp']:+.6f} pp | {new['request_delta_pp']:+.6f} pp |"
        )
    old_paired = comparisons_data["per_io_minus_refresh8_atomic8"]["paired_requests"]
    zero_paired = comparisons_data["per_io_minus_refresh8_batch1_zero"]["paired_requests"]
    new_paired = comparisons_data["per_io_minus_refresh8_issue01us"]["paired_requests"]
    lines.extend(
        [
            "",
            f"atomic8 的 {old_paired['count']} 个配对 request：改善 {old_paired['improved']}、变慢 {old_paired['regressed']}、相同 {old_paired['equal']}；仅 batch1/zero-time：改善 {zero_paired['improved']}、变慢 {zero_paired['regressed']}、相同 {zero_paired['equal']}；加入 0.1 us 发行间隔后：改善 {new_paired['improved']}、变慢 {new_paired['regressed']}、相同 {new_paired['equal']}。",
            "",
            "## 完全不读 SSU Path 状态表",
            "",
            "无状态组只知道静态的类别合法 Path 池，按稳定起点轮转，运行时 Path-table 读取严格为 0。",
            "",
            "| SSU | refresh8 − no-state | per-I/O − no-state |",
            "|---:|---:|---:|",
        ]
    )
    for ssu in SSU_LIST:
        refresh = comparisons_data["refresh8_minus_no_state_issue01us"]["by_ssu"][str(ssu)]
        per_io = comparisons_data["per_io_minus_no_state_issue01us"]["by_ssu"][str(ssu)]
        lines.append(
            f"| {ssu} | {refresh['request_delta_pp']:+.6f} pp | {per_io['request_delta_pp']:+.6f} pp |"
        )
    lines.extend(
        [
            "",
            "## 图",
            "",
            "![Atomicity and no pressure](01_atomicity_and_no_pressure.png)",
            "",
            "![Submission rounds and telemetry](02_submission_and_telemetry.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def analyze(output_dir, seeds):
    datasets = load_results(output_dir, seeds)
    validate(datasets, seeds)
    aggregated = aggregate(datasets)
    comparisons_data = comparisons(aggregated, datasets)
    serializable = {
        "seeds": list(seeds),
        "aggregated": {
            f"{ssu}:{strategy}": values
            for (ssu, strategy), values in sorted(aggregated.items())
        },
        "comparisons": comparisons_data,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    plot_primary(aggregated, comparisons_data, output_dir)
    plot_submission(aggregated, output_dir)
    write_report(aggregated, comparisons_data, output_dir, seeds)
    return serializable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43))
    args = parser.parse_args()
    result = analyze(args.output_dir, tuple(args.seeds))
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
