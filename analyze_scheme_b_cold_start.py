"""Analyze the paired Layer-0 baseline -> Scheme-B hybrid experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_continuous_prefill import _compute_only_reference, _paired_invariants


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "full_prefill_microbatch"
    / "causal_layer0_comparison_results.json"
)
NAMES = ("baseline", "scheme_b_once", "scheme_b_after_l0")
LABELS = {
    "baseline": "Baseline (Path 0)",
    "scheme_b_once": "Original Scheme B",
    "scheme_b_after_l0": "Causal Layer 0 Path0 -> Scheme B",
}


def _batch_breakdown(summary, batch_size):
    batches = [
        batch
        for batch in summary["microbatch_metrics"]
        if batch["batch_size"] == batch_size
    ]
    layer0 = [batch["layer_metrics"][0]["io_barrier_wait_ms"] for batch in batches]
    later = [
        sum(layer["io_barrier_wait_ms"] for layer in batch["layer_metrics"][1:])
        for batch in batches
    ]
    return {
        "count": len(batches),
        "avg_layer0_barrier_ms": float(np.mean(layer0)),
        "avg_layer1_barrier_ms": float(
            np.mean(
                [batch["layer_metrics"][1]["io_barrier_wait_ms"] for batch in batches]
            )
        ),
        "avg_layer2_15_barrier_ms": float(
            np.mean(
                [
                    sum(
                        layer["io_barrier_wait_ms"]
                        for layer in batch["layer_metrics"][2:]
                    )
                    for batch in batches
                ]
            )
        ),
        "avg_later_layer_barrier_ms": float(np.mean(later)),
        "avg_total_barrier_ms": float(np.mean(np.asarray(layer0) + later)),
    }


def analyze(payload):
    rows = {row["strategy"]: row for row in payload["results"]}
    if set(rows) != set(NAMES):
        raise ValueError(f"expected exactly {NAMES}")
    if len({row["trace_hash"] for row in rows.values()}) != 1:
        raise ValueError("strategies used different traces")
    if not all(all(row["summary"]["invariants"].values()) for row in rows.values()):
        raise ValueError("a simulation invariant failed")
    paired = _paired_invariants(rows)
    if not all(paired.values()):
        raise ValueError(f"paired invariants failed: {paired}")

    compute_only = _compute_only_reference(rows["baseline"]["summary"])
    critical_npu = compute_only["critical_npu"]
    metrics = {}
    for name in NAMES:
        summary = rows[name]["summary"]
        critical_batches = [
            batch
            for batch in summary["microbatch_metrics"]
            if batch["npu_id"] == critical_npu
        ]
        critical_initial = next(
            batch for batch in critical_batches if batch["batch_size"] == 8
        )
        metrics[name] = {
            "fleet_utilization_percent": 100.0
            * summary["fleet_npu_compute_utilization"],
            "active_window_utilization_percent": 100.0
            * summary["active_window_npu_compute_utilization"],
            "makespan_ms": summary["makespan_ms"],
            "p99_request_latency_ms": summary["p99_request_latency_ms"],
            "avg_batch_barrier_ms": summary["avg_batch_io_barrier_wait_ms"],
            "initial_full_batches": _batch_breakdown(summary, 8),
            "final_partial_batches": _batch_breakdown(summary, 1),
            "critical_initial_layer0_ms": critical_initial["layer_metrics"][0][
                "io_barrier_wait_ms"
            ],
            "critical_initial_later_ms": sum(
                layer["io_barrier_wait_ms"]
                for layer in critical_initial["layer_metrics"][1:]
            ),
            "critical_final_completion_ms": max(
                batch["completion_time_ms"] for batch in critical_batches
            ),
            "cir_path_writes": summary["cir_path_writes"],
            "causal_layer_observations": summary["causal_layer_observations"],
            "wall_time_s": rows[name]["wall_time_s"],
        }

    baseline = metrics["baseline"]
    original = metrics["scheme_b_once"]
    hybrid = metrics["scheme_b_after_l0"]
    original_gap = original["makespan_ms"] - baseline["makespan_ms"]
    hybrid_gap = hybrid["makespan_ms"] - baseline["makespan_ms"]
    hybrid_vs_original = {
        "fleet_utilization_pp": hybrid["fleet_utilization_percent"]
        - original["fleet_utilization_percent"],
        "makespan_ms": hybrid["makespan_ms"] - original["makespan_ms"],
        "p99_request_latency_ms": hybrid["p99_request_latency_ms"]
        - original["p99_request_latency_ms"],
        "avg_batch_barrier_ms": hybrid["avg_batch_barrier_ms"]
        - original["avg_batch_barrier_ms"],
    }
    hybrid_vs_baseline = {
        "fleet_utilization_pp": hybrid["fleet_utilization_percent"]
        - baseline["fleet_utilization_percent"],
        "makespan_ms": hybrid_gap,
        "p99_request_latency_ms": hybrid["p99_request_latency_ms"]
        - baseline["p99_request_latency_ms"],
        "avg_batch_barrier_ms": hybrid["avg_batch_barrier_ms"]
        - baseline["avg_batch_barrier_ms"],
    }
    return {
        "configuration": {
            "num_npu": payload["experiment"]["num_npu"],
            "num_ssu": payload["experiment"]["num_ssu"],
            "batch_size": payload["experiment"]["batch_size"],
            "n_layers": payload["experiment"]["n_layers"],
            "seed": payload["experiment"]["seed"],
        },
        "paired_invariants": paired,
        "critical_npu": critical_npu,
        "compute_only_reference": compute_only,
        "metrics": metrics,
        "hybrid_vs_original": hybrid_vs_original,
        "hybrid_vs_baseline": hybrid_vs_baseline,
        "makespan_gap_recovered_percent": (
            100.0 * (original_gap - hybrid_gap) / original_gap
            if original_gap
            else 0.0
        ),
    }


def write_report(path, analysis):
    metrics = analysis["metrics"]
    lines = [
        "# Causal Scheme B after Layer 0",
        "",
        "配置：128 NPU、batch 8、16 layers、28 SSUs；三种策略使用同一请求、放置、microbatch membership、SSD40 与 NPU50 数据面。",
        "",
        "Layer 0 固定使用 baseline Path0 与基础 CIR。每个 NPU 完成自己的上一层 I/O 后，上报该层实际 bytes-by-SSU；控制器不接收仿真时钟、未来 placement 或 SSD 队列状态，原子更新 max-min CIR 后才提交下一层。cold Path0 使用固定 baseline CIR，不设置全局 fence。",
        "",
        "| Strategy | Fleet util. | Makespan (ms) | Initial L0 (ms) | Initial L1-15 (ms) | Avg batch barrier (ms) | P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        row = metrics[name]
        initial = row["initial_full_batches"]
        lines.append(
            f"| {LABELS[name]} | {row['fleet_utilization_percent']:.4f}% | "
            f"{row['makespan_ms']:.3f} | {initial['avg_layer0_barrier_ms']:.3f} | "
            f"{initial['avg_later_layer_barrier_ms']:.3f} | "
            f"{row['avg_batch_barrier_ms']:.3f} | {row['p99_request_latency_ms']:.3f} |"
        )
    baseline = metrics["baseline"]
    original = metrics["scheme_b_once"]
    hybrid = metrics["scheme_b_after_l0"]
    lines.extend(
        (
            "",
            "## Causal result",
            "",
            f"Original Scheme B makespan minus baseline: "
            f"{original['makespan_ms'] - baseline['makespan_ms']:+.3f} ms.",
            "",
            f"Hybrid makespan minus baseline: "
            f"{hybrid['makespan_ms'] - baseline['makespan_ms']:+.3f} ms.",
            "",
            f"The causal policy recovered {analysis['makespan_gap_recovered_percent']:.2f}% "
            f"of the original critical-path makespan gap; this does not mean its tail "
            f"or mean barrier matches baseline.",
            "",
            f"Critical NPU {analysis['critical_npu']} initial Layer-0 barriers: "
            f"baseline {baseline['critical_initial_layer0_ms']:.3f} ms, "
            f"original Scheme B {original['critical_initial_layer0_ms']:.3f} ms, "
            f"hybrid {hybrid['critical_initial_layer0_ms']:.3f} ms.",
            "",
            f"For that NPU, the causal hybrid leaves exposed wait in "
            f"Layer 1--15: baseline {baseline['critical_initial_later_ms']:.3f} ms, "
            f"original Scheme B {original['critical_initial_later_ms']:.3f} ms, "
            f"hybrid {hybrid['critical_initial_later_ms']:.3f} ms. This is physical "
            f"cold/warm overlap, not a simulator fence.",
            "",
            f"Across all initial full batches, Layer-0 barrier is "
            f"{hybrid['initial_full_batches']['avg_layer0_barrier_ms']:.3f} ms versus "
            f"{baseline['initial_full_batches']['avg_layer0_barrier_ms']:.3f} ms for "
            f"baseline. Early warm dedicated Paths consume grants while late cold "
            f"commands retain only the fixed baseline Path0 CIR, so the tail moves "
            f"into Layer 0 even though the critical NPU is unchanged.",
            "",
            f"Compared with original Scheme B, the hybrid changes fleet utilization "
            f"by {analysis['hybrid_vs_original']['fleet_utilization_pp']:+.3f} pp, "
            f"makespan by {analysis['hybrid_vs_original']['makespan_ms']:+.3f} ms, "
            f"and P99 by {analysis['hybrid_vs_original']['p99_request_latency_ms']:+.3f} ms.",
            "",
            f"Compared with baseline, it changes fleet utilization by "
            f"{analysis['hybrid_vs_baseline']['fleet_utilization_pp']:+.3f} pp, "
            f"makespan by {analysis['hybrid_vs_baseline']['makespan_ms']:+.3f} ms, "
            f"and P99 by {analysis['hybrid_vs_baseline']['p99_request_latency_ms']:+.3f} ms.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    output_dir = args.input.parent
    analysis_path = output_dir / "causal_layer0_comparison_analysis.json"
    report_path = output_dir / "causal_layer0_comparison_report.md"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    write_report(report_path, analysis)
    print(analysis_path)
    print(report_path)


if __name__ == "__main__":
    main()
