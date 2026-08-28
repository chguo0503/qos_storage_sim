"""Analyze and plot the balanced six-request fastest-cutoff matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import numpy as np

from six_request_experiment import CASES, LAYER_LIST, SLO_ALPHAS, SSU_LIST


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "six_request_fastest_cutoff" / "results.json"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
PLOT_ALPHA = 2.0
LABELS = {
    "baseline": "Baseline (Path 0)",
    "scheme_b": "Causal Scheme B",
    "full_info_edf": "Full-info EDF",
}
COLORS = {
    "baseline": "#4C78A8",
    "scheme_b": "#E45756",
    "full_info_edf": "#54A24B",
}
MARKERS = {
    "baseline": "o",
    "scheme_b": "^",
    "full_info_edf": "D",
}


def _alpha_key(alpha):
    return f"{alpha:g}"


def _row_key(row):
    return row["strategy"], row["num_ssu"], row["n_layers"]


def _validate(payload):
    experiment = payload["experiment"]
    rows = payload["results"]
    keys = [_row_key(row) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate result rows: {duplicates}")

    expected_strategies = {case.name for case in CASES}
    for n_layers in sorted({row["n_layers"] for row in rows}):
        for num_ssu in sorted(
            {row["num_ssu"] for row in rows if row["n_layers"] == n_layers}
        ):
            group = [
                row
                for row in rows
                if row["n_layers"] == n_layers and row["num_ssu"] == num_ssu
            ]
            if {row["strategy"] for row in group} != expected_strategies:
                raise ValueError(f"incomplete paired group at L={n_layers}, SSU={num_ssu}")
            for field in (
                "assignment_hash",
                "workload_hash",
                "placement_hash",
                "trace_hash",
                "simulator_input_fingerprint",
                "representative_profiles",
                "fleet_category_counts",
            ):
                values = {
                    json.dumps(row[field], sort_keys=True, separators=(",", ":"))
                    for row in group
                }
                if len(values) != 1:
                    raise ValueError(
                        f"unpaired {field} at L={n_layers}, SSU={num_ssu}"
                    )
            for row in group:
                if len(row["npu_rows"]) != experiment["num_npu"]:
                    raise ValueError("NPU rows do not cover the fleet")
                if len(row["request_rows"]) != (
                    experiment["num_npu"] * experiment["slo_requests_per_npu"]
                ):
                    raise ValueError("SLO request rows do not cover the fixed cohort")
                if not all(row["diagnostics"]["invariants"].values()):
                    raise ValueError("simulator invariant failed")
                mean_util = statistics.fmean(
                    item["utilization"] for item in row["npu_rows"]
                )
                if not math.isclose(
                    mean_util,
                    row["mean_npu_utilization"],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise ValueError("stored NPU utilization is inconsistent")
    return experiment, rows


def analyze(payload):
    experiment, rows = _validate(payload)
    strategies = [case.name for case in CASES]
    layers = sorted({row["n_layers"] for row in rows})
    ssus = sorted({row["num_ssu"] for row in rows})
    by_key = {_row_key(row): row for row in rows}
    metrics = {}
    invalid_slo = []
    for strategy in strategies:
        metrics[strategy] = {}
        for n_layers in layers:
            layer_metrics = {}
            for num_ssu in ssus:
                row = by_key[(strategy, num_ssu, n_layers)]
                slo_metrics = row["slo"]
                layer0_wait = statistics.fmean(
                    item["layer0_barrier_ms"] for item in row["request_rows"]
                )
                later_wait = statistics.fmean(
                    item["later_layer_barrier_ms"] for item in row["request_rows"]
                )
                layer_metrics[str(num_ssu)] = {
                    "mean_npu_utilization": row["mean_npu_utilization"],
                    "common_horizon_fleet_utilization": row[
                        "common_horizon_fleet_utilization"
                    ],
                    "ttft_slo": slo_metrics,
                    "slo_cohort_complete": row["slo_cohort_complete"],
                    "slo_guard_margin_ms": row["slo_guard_margin_ms"],
                    "cutoff_ms": row["cutoff_ms"],
                    "mean_layer0_barrier_ms": layer0_wait,
                    "mean_later_layer_barrier_ms": later_wait,
                    "wall_time_s": row["wall_time_s"],
                }
                if not row["slo_cohort_complete"]:
                    invalid_slo.append(
                        {
                            "strategy": strategy,
                            "n_layers": n_layers,
                            "num_ssu": num_ssu,
                            "completed": row["slo_cohort_completed_at_cutoff"],
                            "total": row["slo_cohort_size"],
                            "guard_margin_ms": row["slo_guard_margin_ms"],
                        }
                    )
            metrics[strategy][str(n_layers)] = layer_metrics

    deltas = {}
    for strategy in strategies:
        if strategy == "baseline":
            continue
        deltas[strategy] = {}
        for n_layers in layers:
            values = {}
            for num_ssu in ssus:
                candidate = metrics[strategy][str(n_layers)][str(num_ssu)]
                baseline = metrics["baseline"][str(n_layers)][str(num_ssu)]
                alpha = _alpha_key(PLOT_ALPHA)
                candidate_slo = candidate["ttft_slo"][alpha]["attainment"]
                baseline_slo = baseline["ttft_slo"][alpha]["attainment"]
                values[str(num_ssu)] = {
                    "npu_utilization_delta_pp": 100.0
                    * (
                        candidate["mean_npu_utilization"]
                        - baseline["mean_npu_utilization"]
                    ),
                    "ttft_slo_delta_pp": (
                        None
                        if candidate_slo is None or baseline_slo is None
                        else 100.0 * (candidate_slo - baseline_slo)
                    ),
                    "layer0_barrier_delta_ms": candidate[
                        "mean_layer0_barrier_ms"
                    ]
                    - baseline["mean_layer0_barrier_ms"],
                    "later_layer_barrier_delta_ms": candidate[
                        "mean_later_layer_barrier_ms"
                    ]
                    - baseline["mean_later_layer_barrier_ms"],
                }
            deltas[strategy][str(n_layers)] = values

    return {
        "schema_version": 1,
        "complete": bool(payload.get("complete")),
        "metric_scope": (
            "all NPU compute intervals clipped at the fastest sixth completion; "
            "fixed first-five request TTFT cohort"
        ),
        "num_npu": experiment["num_npu"],
        "requests_per_npu": experiment["requests_per_npu"],
        "slo_requests_per_npu": experiment["slo_requests_per_npu"],
        "layer_list": layers,
        "ssu_list": ssus,
        "strategies": strategies,
        "plot_alpha": PLOT_ALPHA,
        "slo_alphas": list(SLO_ALPHAS),
        "workload": {
            "name": "balanced_six_stress_v1",
            "representative_profiles": experiment["representative_profiles"],
            "description": (
                "A/B templates balance six-request per-NPU compute and KV work "
                "within 0.1%; fleet SS/SL/LS/LL counts are equal"
            ),
        },
        "metrics": metrics,
        "deltas_vs_baseline": deltas,
        "slo_guard": {
            "passed": not invalid_slo,
            "invalid_case_count": len(invalid_slo),
            "invalid_cases": invalid_slo,
        },
        "validation": {
            "paired_inputs": True,
            "survivorship_bias_excluded": not invalid_slo,
            "result_rows": len(rows),
        },
    }


def write_plot(path, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = analysis["layer_list"]
    ssus = analysis["ssu_list"]
    figure, axes = plt.subplots(
        2,
        len(layers),
        figsize=(4.2 * len(layers), 8.0),
        sharex="col",
        sharey="row",
    )
    for column, n_layers in enumerate(layers):
        utilization_axis = axes[0, column]
        slo_axis = axes[1, column]
        for strategy in analysis["strategies"]:
            cases = analysis["metrics"][strategy][str(n_layers)]
            utilization = [
                100.0 * cases[str(ssu)]["mean_npu_utilization"] for ssu in ssus
            ]
            slo = [
                cases[str(ssu)]["ttft_slo"][_alpha_key(PLOT_ALPHA)]["attainment"]
                for ssu in ssus
            ]
            slo = [np.nan if value is None else 100.0 * value for value in slo]
            style = {
                "color": COLORS[strategy],
                "marker": MARKERS[strategy],
                "linewidth": 2.0,
                "markersize": 5.5,
                "label": LABELS[strategy],
            }
            utilization_axis.plot(ssus, utilization, **style)
            slo_axis.plot(ssus, slo, **style)
        utilization_axis.set_title(f"{n_layers} layers")
        utilization_axis.grid(alpha=0.25)
        slo_axis.grid(alpha=0.25)
        slo_axis.set_xlabel("Number of SSUs")
        slo_axis.set_xticks(ssus)
        slo_axis.tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Mean NPU utilization (%)")
    axes[1, 0].set_ylabel(f"TTFT SLO attainment @ {PLOT_ALPHA:g}x (%)")
    axes[0, 0].set_ylim(0.0, 100.0)
    axes[1, 0].set_ylim(0.0, 100.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(analysis["strategies"]),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle("Balanced six-request closed-loop experiment", y=0.96)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_percent(value):
    return "invalid" if value is None else f"{100.0 * value:.2f}%"


def write_report(path, analysis):
    lines = [
        "# Balanced six-request fastest-cutoff experiment",
        "",
        "Each NPU runs six batch-1 prefill requests. The measurement cutoff is "
        "the first NPU to finish its sixth request. Mean NPU utilization clips "
        "all 128 NPU compute intervals at that cutoff. TTFT SLO uses the fixed "
        "first five requests per NPU and is valid only when all 640 have completed.",
        "",
        "The workload is a calibrated balanced stress input, not the natural "
        "distribution of all source profiles. Per-NPU six-request compute and KV "
        "work differ by less than 0.1%, and fleet SS/SL/LS/LL counts are equal.",
        "",
        f"SLO shown below: `TTFT <= {PLOT_ALPHA:g} × layers × per-layer compute time`.",
        "",
    ]
    guard = analysis["slo_guard"]
    lines.append(
        "SLO guard: **PASS** — every first-five cohort completed before cutoff."
        if guard["passed"]
        else f"SLO guard: **FAIL** in {guard['invalid_case_count']} cases; those "
        "points are marked invalid rather than computed from survivors."
    )
    for n_layers in analysis["layer_list"]:
        lines.extend(
            [
                "",
                f"## {n_layers} layers",
                "",
                "### Mean NPU utilization",
                "",
                "| Strategy | "
                + " | ".join(f"SSU {ssu}" for ssu in analysis["ssu_list"])
                + " |",
                "|---|" + "---:|" * len(analysis["ssu_list"]),
            ]
        )
        for strategy in analysis["strategies"]:
            values = [
                100.0
                * analysis["metrics"][strategy][str(n_layers)][str(ssu)][
                    "mean_npu_utilization"
                ]
                for ssu in analysis["ssu_list"]
            ]
            lines.append(
                f"| {LABELS[strategy]} | "
                + " | ".join(f"{value:.2f}%" for value in values)
                + " |"
            )
        lines.extend(
            [
                "",
                f"### TTFT SLO attainment @ {PLOT_ALPHA:g}x",
                "",
                "| Strategy | "
                + " | ".join(f"SSU {ssu}" for ssu in analysis["ssu_list"])
                + " |",
                "|---|" + "---:|" * len(analysis["ssu_list"]),
            ]
        )
        for strategy in analysis["strategies"]:
            values = [
                analysis["metrics"][strategy][str(n_layers)][str(ssu)][
                    "ttft_slo"
                ][_alpha_key(PLOT_ALPHA)]["attainment"]
                for ssu in analysis["ssu_list"]
            ]
            lines.append(
                f"| {LABELS[strategy]} | "
                + " | ".join(_format_percent(value) for value in values)
                + " |"
            )
    if not guard["passed"]:
        lines.extend(["", "## Invalid SLO cases", ""])
        for case in guard["invalid_cases"]:
            lines.append(
                f"- {case['strategy']}, layers={case['n_layers']}, "
                f"SSU={case['num_ssu']}: {case['completed']}/{case['total']} "
                f"cohort requests complete; guard margin "
                f"{case['guard_margin_ms']:.3f} ms."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", analysis)
    write_plot(args.output_dir / "01_six_request_layer_sweep.png", analysis)


if __name__ == "__main__":
    main()
