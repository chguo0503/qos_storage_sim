"""Analyze the complete causal Full-prefill strategy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_continuous_prefill import _compute_only_reference, _paired_invariants
from continuous_prefill_client import legacy_qos_config
from continuous_prefill_experiment import _source_fingerprint


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT / "results" / "full_prefill_microbatch" / "causal_full_matrix_results.json"
)
ORDER = (
    "baseline",
    "path_rr",
    "layer_once",
    "refresh8",
    "refresh1",
    "scheme_b_once",
    "scheme_b_periodic8",
    "scheme_b_periodic4",
    "scheme_b_periodic2",
    "scheme_b_periodic1",
    "scheme_b_after_l0",
    "best_feasible",
)
DEPLOYABLE = (
    "baseline",
    "path_rr",
    "layer_once",
    "refresh8",
    "refresh1",
    "scheme_b_after_l0",
)
LABELS = {
    "baseline": "Baseline Path0",
    "path_rr": "Path RR",
    "layer_once": "Layer once",
    "refresh8": "Refresh 8",
    "refresh1": "Refresh 1",
    "scheme_b_once": "Legacy Scheme B once",
    "scheme_b_periodic8": "Legacy Scheme B / 8",
    "scheme_b_periodic4": "Legacy Scheme B / 4",
    "scheme_b_periodic2": "Legacy Scheme B / 2",
    "scheme_b_periodic1": "Legacy Scheme B / 1",
    "scheme_b_after_l0": "Causal previous-layer Scheme B",
    "best_feasible": "Full-info EDF reference",
}


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def _batch_stats(batches):
    layer0 = [batch["layer_metrics"][0]["io_barrier_wait_ms"] for batch in batches]
    layer1 = [batch["layer_metrics"][1]["io_barrier_wait_ms"] for batch in batches]
    layer2_15 = [
        sum(layer["io_barrier_wait_ms"] for layer in batch["layer_metrics"][2:])
        for batch in batches
    ]
    total = [batch["io_barrier_wait_ms"] for batch in batches]
    per_layer = []
    for layer in range(16):
        values = [
            batch["layer_metrics"][layer]["io_barrier_wait_ms"]
            for batch in batches
        ]
        per_layer.append({"layer": layer, **_stats(values)})
    return {
        "count": len(batches),
        "layer0": _stats(layer0),
        "layer1": _stats(layer1),
        "layer2_15": _stats(layer2_15),
        "total": _stats(total),
        "per_layer": per_layer,
    }


def _metric(row, baseline):
    summary = row["summary"]
    requests = summary["request_metrics"]
    initial_latency = [item["latency_ms"] for item in requests if item["initial"]]
    new_latency = [item["latency_ms"] for item in requests if not item["initial"]]
    full = [item for item in summary["microbatch_metrics"] if item["batch_size"] == 8]
    partial = [item for item in summary["microbatch_metrics"] if item["batch_size"] == 1]
    last = max(
        summary["microbatch_metrics"], key=lambda item: item["completion_time_ms"]
    )
    last_later = sum(
        layer["io_barrier_wait_ms"] for layer in last["layer_metrics"][1:]
    )
    return {
        "fleet_percent": 100.0 * summary["fleet_npu_compute_utilization"],
        "fleet_delta_pp": 100.0
        * (
            summary["fleet_npu_compute_utilization"]
            - baseline["fleet_npu_compute_utilization"]
        ),
        "active_window_percent": 100.0
        * summary["active_window_npu_compute_utilization"],
        "makespan_ms": summary["makespan_ms"],
        "makespan_delta_ms": summary["makespan_ms"] - baseline["makespan_ms"],
        "throughput_requests_per_s": summary["throughput_requests_per_s"],
        "request_latency_initial": _stats(initial_latency),
        "request_latency_new": _stats(new_latency),
        "request_latency_p99_ms": summary["p99_request_latency_ms"],
        "avg_batch_barrier_ms": summary["avg_batch_io_barrier_wait_ms"],
        "initial_full_batches": _batch_stats(full),
        "final_partial_batches": _batch_stats(partial),
        "ssd_mean_utilization_percent": 100.0 * summary["ssd_mean_utilization"],
        "npu_link_mean_utilization_percent": 100.0
        * summary["npu_link_mean_utilization"],
        "pressure_reports": summary["pressure_reports"],
        "control_evaluations": summary["control_evaluations"],
        "cir_commits": summary["cir_commits"],
        "cir_path_writes": summary["cir_path_writes"],
        "causal_layer_observations": summary["causal_layer_observations"],
        "actual_last_finisher": {
            "npu_id": last["npu_id"],
            "batch_size": last["batch_size"],
            "completion_time_ms": last["completion_time_ms"],
            "layer0_barrier_ms": last["layer_metrics"][0]["io_barrier_wait_ms"],
            "layer1_15_barrier_ms": last_later,
            "total_barrier_ms": last["io_barrier_wait_ms"],
        },
    }


def analyze(payload):
    rows = {row["strategy"]: row for row in payload["results"]}
    if not payload["complete"] or tuple(rows) != ORDER:
        raise ValueError("the complete 12-strategy matrix is required")
    if payload["experiment"]["source_fingerprint"] != _source_fingerprint():
        raise ValueError("result and source fingerprints differ")
    if len({row["trace_hash"] for row in rows.values()}) != 1:
        raise ValueError("strategies used different traces")
    if not all(all(row["summary"]["invariants"].values()) for row in rows.values()):
        raise ValueError("a data-plane invariant failed")
    paired = _paired_invariants(rows)
    if not all(paired.values()):
        raise ValueError(f"paired invariants failed: {paired}")

    baseline = rows["baseline"]["summary"]
    metrics = {name: _metric(rows[name], baseline) for name in ORDER}
    causal_control = rows["scheme_b_after_l0"]["control"]
    if causal_control["policy_clock_input"] or causal_control["future_placement_input"]:
        raise ValueError("causal Scheme B contains a forbidden policy input")

    return {
        "configuration": payload["experiment"],
        "paired_invariants": paired,
        "compute_only_reference": _compute_only_reference(baseline),
        "causal_control": causal_control,
        "metrics": metrics,
        "deployable_best_fleet": max(
            DEPLOYABLE, key=lambda name: metrics[name]["fleet_percent"]
        ),
        "deployable_best_p99": min(
            DEPLOYABLE, key=lambda name: metrics[name]["request_latency_p99_ms"]
        ),
        "deployable_best_mean_barrier": min(
            DEPLOYABLE, key=lambda name: metrics[name]["avg_batch_barrier_ms"]
        ),
    }


def write_report(path, analysis):
    metrics = analysis["metrics"]
    detail_order = (
        "baseline",
        "path_rr",
        "layer_once",
        "refresh8",
        "refresh1",
        "scheme_b_once",
        "scheme_b_after_l0",
        "best_feasible",
    )
    lines = [
        "# Causal Full-prefill strategy comparison",
        "",
        "配置：128 NPU、batch 8、16 layers、28 SSUs；所有策略严格配对并共用 SSD40/NPU50 数据面。",
        "",
        "| Strategy | Fleet | Δ vs baseline | Active window | Makespan | P99 | Avg barrier | Initial L0 p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        row = metrics[name]
        lines.append(
            f"| {LABELS[name]} | {row['fleet_percent']:.4f}% | "
            f"{row['fleet_delta_pp']:+.4f} pp | {row['active_window_percent']:.4f}% | "
            f"{row['makespan_ms']:.3f} ms | {row['request_latency_p99_ms']:.3f} ms | "
            f"{row['avg_batch_barrier_ms']:.3f} ms | "
            f"{row['initial_full_batches']['layer0']['p99_ms']:.3f} ms |"
        )
    lines.extend(
        (
            "",
            "## Cohort and layer decomposition",
            "",
            "| Strategy | Initial req mean | Initial req p99 | New req mean | New req p99 | Full-batch L0 mean | Full-batch L1-15 mean | Final singleton total |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for name in detail_order:
        row = metrics[name]
        lines.append(
            f"| {LABELS[name]} | "
            f"{row['request_latency_initial']['mean_ms']:.3f} ms | "
            f"{row['request_latency_initial']['p99_ms']:.3f} ms | "
            f"{row['request_latency_new']['mean_ms']:.3f} ms | "
            f"{row['request_latency_new']['p99_ms']:.3f} ms | "
            f"{row['initial_full_batches']['layer0']['mean_ms']:.3f} ms | "
            f"{row['initial_full_batches']['layer1']['mean_ms'] + row['initial_full_batches']['layer2_15']['mean_ms']:.3f} ms | "
            f"{row['final_partial_batches']['total']['mean_ms']:.3f} ms |"
        )
    lines.extend(
        (
            "",
            "## Resource use and control cost",
            "",
            "| Strategy | SSD mean | NPU-link mean | Throughput | Pressure reads | Control evals | CIR commits | Path writes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for name in detail_order:
        row = metrics[name]
        lines.append(
            f"| {LABELS[name]} | {row['ssd_mean_utilization_percent']:.3f}% | "
            f"{row['npu_link_mean_utilization_percent']:.3f}% | "
            f"{row['throughput_requests_per_s']:.3f} req/s | "
            f"{row['pressure_reports']} | {row['control_evaluations']} | "
            f"{row['cir_commits']} | {row['cir_path_writes']} |"
        )
    lines.extend(
        (
            "",
            "## Actual last finisher",
            "",
            "| Strategy | NPU | Batch | L0 barrier | L1-15 barrier | Total barrier | Completion |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for name in detail_order:
        last = metrics[name]["actual_last_finisher"]
        lines.append(
            f"| {LABELS[name]} | {last['npu_id']} | {last['batch_size']} | "
            f"{last['layer0_barrier_ms']:.3f} ms | "
            f"{last['layer1_15_barrier_ms']:.3f} ms | "
            f"{last['total_barrier_ms']:.3f} ms | "
            f"{last['completion_time_ms']:.3f} ms |"
        )
    baseline = metrics["baseline"]
    causal = metrics["scheme_b_after_l0"]
    path0_cir = legacy_qos_config().path_cirs[0]
    lines.extend(
        (
            "",
            "## Result",
            "",
            f"Compute-only fleet upper bound is "
            f"{100 * analysis['compute_only_reference']['fleet_utilization_upper_bound']:.4f}%; "
            f"baseline is {baseline['fleet_percent']:.4f}%, leaving only "
            f"{100 * analysis['compute_only_reference']['fleet_utilization_upper_bound'] - baseline['fleet_percent']:.4f} pp.",
            "",
            f"Causal Scheme B matches baseline fleet/makespan because both finish on "
            f"NPU {causal['actual_last_finisher']['npu_id']}, but its P99 is "
            f"{causal['request_latency_p99_ms'] - baseline['request_latency_p99_ms']:+.3f} ms worse and "
            f"initial Layer-0 p99 is {causal['initial_full_batches']['layer0']['p99_ms']:.3f} ms "
            f"versus {baseline['initial_full_batches']['layer0']['p99_ms']:.3f} ms.",
            "",
            f"The causal controller reserves only the baseline configured Path-0 CIR "
            f"({path0_cir:.6f} GB/s) while cold batches exist. In baseline Path 0 is the "
            "only active Path and therefore receives all work-conserving surplus up to 40 GB/s. "
            "After warm dedicated Paths are enabled, their max-min CIRs consume nearly all SSD "
            "capacity, so unfinished Layer-0 traffic no longer receives that surplus. This mixed "
            "cold/warm arbitration—not the absence of global information—is the direct cause of "
            "the causal strategy's Layer-0 tail regression.",
            "",
            "Max-min grants optimize long-run demand fairness. The measured objective is instead "
            "batch barrier / P99 / final makespan, which is determined by the last block and the "
            "arrival order. Dynamic CIR cannot move ring-placed blocks to another SSD or create "
            "more than 40 GB/s per SSD, and it can destroy FCFS's favorable ordering.",
            "",
            "The five legacy Scheme-B frequencies are identical. They inspect future placement and are retained only as non-causal references. The full-info EDF row is an executable reference, not a mathematical optimum.",
            "",
            f"Deployable best P99: {LABELS[analysis['deployable_best_p99']]}. "
            f"Deployable best mean barrier: {LABELS[analysis['deployable_best_mean_barrier']]}. "
            f"Fleet best is a tie between baseline and causal Scheme B at four-decimal precision.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    analysis = analyze(json.loads(args.input.read_text()))
    output_dir = args.input.parent
    json_path = output_dir / "causal_full_matrix_analysis.json"
    report_path = output_dir / "causal_full_matrix_report.md"
    json_path.write_text(json.dumps(analysis, indent=2) + "\n")
    write_report(report_path, analysis)
    print(json_path)
    print(report_path)


if __name__ == "__main__":
    main()
