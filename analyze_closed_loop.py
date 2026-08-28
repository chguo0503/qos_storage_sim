"""Analyze the paired batch-1 closed-loop experiment.

The simulator emits only the middle, measured requests for each NPU.  This
script recomputes the two user-facing metrics from those rows:

* per-NPU TTFT SLO attainment, then an equal-weight mean across NPUs;
* physical compute utilization in each NPU's measured active window, then an
  equal-weight mean across NPUs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "closed_loop_batch1" / "results.json"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
DEFAULT_ANALYSIS = "analysis.json"
DEFAULT_REPORT = "report.md"
DEFAULT_PLOT = "01_ttft_slo_and_npu_utilization.png"
DEFAULT_IMPROVEMENT_PLOT = "02_scheme_b_realistic_improvements.png"

ALPHAS = (1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0)
PLOT_ALPHA = 2.0
CORE_STRATEGIES = (
    "baseline",
    "layer_once",
    "scheme_b",
    "full_info_edf",
)
IMPROVED_STRATEGIES = (
    "scheme_b_manifest",
    "scheme_b_coflow",
)
LABELS = {
    "baseline": "Baseline (Path 0)",
    "layer_once": "Layer once",
    "scheme_b": "Causal Scheme B",
    "full_info_edf": "Full-info EDF",
    "scheme_b_manifest": "Manifest Scheme B",
    "scheme_b_coflow": "Coflow Scheme B",
}
COLORS = {
    "baseline": "#4C78A8",
    "layer_once": "#F58518",
    "scheme_b": "#E45756",
    "full_info_edf": "#54A24B",
    "scheme_b_manifest": "#B279A2",
    "scheme_b_coflow": "#72B7B2",
}
MARKERS = {
    "baseline": "o",
    "layer_once": "s",
    "scheme_b": "^",
    "full_info_edf": "D",
    "scheme_b_manifest": "v",
    "scheme_b_coflow": "P",
}


def _alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def _percentiles(values) -> dict:
    return {
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _request_identity(row: dict) -> tuple:
    return (
        row["request_id"],
        row["npu_id"],
        row["sequence"],
        tuple(row["profile_key"]),
        row["category"],
        row["ideal_ttft_ms"],
    )


def _request_key(row: dict) -> tuple[int, int]:
    return row["npu_id"], row["sequence"]


def _validate_request_rows(row: dict, experiment: dict) -> dict[int, list[dict]]:
    num_npu = experiment["num_npu"]
    warmup = experiment["warmup_requests_per_npu"]
    measured = experiment["measured_requests_per_npu"]
    request_rows = row["request_rows"]
    expected_count = num_npu * measured
    if len(request_rows) != expected_count:
        raise ValueError(
            f"{row['strategy']}@SSU{row['num_ssu']} has {len(request_rows)} "
            f"measured requests, expected {expected_count}"
        )
    if len({item["request_id"] for item in request_rows}) != expected_count:
        raise ValueError("measured request_id values are not unique")

    by_npu = defaultdict(list)
    for item in request_rows:
        if item["ttft_ms"] <= 0.0 or item["ideal_ttft_ms"] <= 0.0:
            raise ValueError("TTFT values must be positive")
        expected_slowdown = item["ttft_ms"] / item["ideal_ttft_ms"]
        if not math.isclose(
            item["slowdown"], expected_slowdown, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("stored slowdown does not match TTFT / ideal TTFT")
        by_npu[item["npu_id"]].append(item)

    expected_sequences = list(range(warmup, warmup + measured))
    if set(by_npu) != set(range(num_npu)):
        raise ValueError("request rows do not cover every NPU exactly once")
    for npu_id, items in by_npu.items():
        items.sort(key=lambda item: item["sequence"])
        if [item["sequence"] for item in items] != expected_sequences:
            raise ValueError(f"NPU {npu_id} does not contain the measured sequence")
    return by_npu


def _validate_npu_rows(row: dict, experiment: dict) -> dict[int, dict]:
    num_npu = experiment["num_npu"]
    npu_rows = row["npu_rows"]
    if len(npu_rows) != num_npu:
        raise ValueError(
            f"{row['strategy']}@SSU{row['num_ssu']} has {len(npu_rows)} NPU rows, "
            f"expected {num_npu}"
        )
    by_npu = {item["npu_id"]: item for item in npu_rows}
    if set(by_npu) != set(range(num_npu)):
        raise ValueError("NPU utilization rows do not cover every NPU")
    for item in npu_rows:
        if item["active_window_ms"] <= 0.0:
            raise ValueError("NPU active window must be positive")
        utilization = item["compute_ms"] / item["active_window_ms"]
        if not math.isclose(
            utilization, item["utilization"], rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("stored NPU utilization does not match busy / active")
    return by_npu


def _strategy_set(rows: list[dict]) -> tuple[str, ...]:
    present = {row["strategy"] for row in rows}
    core = set(CORE_STRATEGIES)
    full = core | set(IMPROVED_STRATEGIES)
    if present == core:
        return CORE_STRATEGIES
    if present == full:
        return (*CORE_STRATEGIES, *IMPROVED_STRATEGIES)
    raise ValueError(
        "paired results must contain exactly the four core strategies or all six "
        f"strategies; found {sorted(present)}"
    )


def _validate(payload: dict) -> tuple[dict, list[dict], tuple[str, ...], list[int]]:
    experiment = payload.get("experiment", {})
    rows = payload.get("results", [])
    if not rows:
        return experiment, [], (), []

    required_experiment_fields = (
        "num_npu",
        "requests_per_npu",
        "warmup_requests_per_npu",
        "measured_requests_per_npu",
        "cooldown_requests_per_npu",
        "n_layers",
        "batch_size",
        "ssu_list",
    )
    missing = [field for field in required_experiment_fields if field not in experiment]
    if missing:
        raise ValueError(f"experiment metadata is missing {missing}")
    if experiment["batch_size"] != 1:
        raise ValueError("closed-loop analysis requires batch_size=1")
    if (
        experiment["warmup_requests_per_npu"]
        + experiment["measured_requests_per_npu"]
        + experiment["cooldown_requests_per_npu"]
        != experiment["requests_per_npu"]
    ):
        raise ValueError("warmup + measured + cooldown does not equal request count")

    keys = [(row["strategy"], row["num_ssu"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate strategy/SSU rows: {duplicates}")

    ssus = sorted({row["num_ssu"] for row in rows})
    grouped = {
        ssu: [row for row in rows if row["num_ssu"] == ssu] for ssu in ssus
    }
    strategy_order = _strategy_set(grouped[ssus[0]])
    expected_strategies = set(strategy_order)
    for ssu, group in grouped.items():
        if {row["strategy"] for row in group} != expected_strategies:
            raise ValueError(f"SSU {ssu} does not contain a complete paired strategy set")
        for field in (
            "assignment_hash",
            "workload_hash",
            "placement_hash",
            "trace_hash",
            "simulator_input_fingerprint",
            "category_counts",
        ):
            values = {
                json.dumps(row[field], sort_keys=True, separators=(",", ":"))
                for row in group
            }
            if len(values) != 1:
                raise ValueError(f"SSU {ssu} is not paired on {field}")
        for field in ("submitted_blocks", "expected_read_gb", "request_count"):
            values = {row["diagnostics"][field] for row in group}
            if len(values) != 1:
                raise ValueError(f"SSU {ssu} is not paired on diagnostic {field}")

    if payload.get("complete") and ssus != list(experiment["ssu_list"]):
        raise ValueError("complete result does not cover the configured SSU list")
    if len({row["assignment_hash"] for row in rows}) != 1:
        raise ValueError("request-to-NPU assignment changes across SSU counts")

    reference_identity = None
    for row in rows:
        by_npu = _validate_request_rows(row, experiment)
        npu_rows = _validate_npu_rows(row, experiment)
        identities = tuple(
            sorted(_request_identity(item) for items in by_npu.values() for item in items)
        )
        if reference_identity is None:
            reference_identity = identities
        elif identities != reference_identity:
            raise ValueError("measured request identities change across paired cases")
        for npu_id, items in by_npu.items():
            compute_ms = sum(item["compute_ms"] for item in items)
            if not math.isclose(
                compute_ms,
                npu_rows[npu_id]["compute_ms"],
                rel_tol=1e-10,
                abs_tol=1e-7,
            ):
                raise ValueError("request compute total does not match NPU busy time")
        if not all(row["diagnostics"]["invariants"].values()):
            raise ValueError(f"simulation invariants failed for {row['strategy']}")
        guard = row["diagnostics"]
        present = guard["npus_present_through_measurement"]
        full_load = guard["full_load_through_measurement"]
        margin_ms = guard["guard_margin_ms"]
        if not 0 <= present <= experiment["num_npu"]:
            raise ValueError("steady-load guard has an invalid NPU count")
        if full_load != (present == experiment["num_npu"]):
            raise ValueError("steady-load guard flag disagrees with its NPU count")
        if full_load != (margin_ms + 1e-9 >= 0.0):
            raise ValueError("steady-load guard flag disagrees with its time margin")

    return experiment, rows, strategy_order, ssus


def _case_metrics(row: dict, experiment: dict) -> dict:
    by_npu = _validate_request_rows(row, experiment)
    npu_rows = _validate_npu_rows(row, experiment)
    utilizations = [
        npu_rows[npu_id]["compute_ms"] / npu_rows[npu_id]["active_window_ms"]
        for npu_id in range(experiment["num_npu"])
    ]
    slo_by_alpha = {}
    slo_distribution = {}
    for alpha in ALPHAS:
        per_npu = [
            statistics.fmean(item["slowdown"] <= alpha for item in by_npu[npu_id])
            for npu_id in range(experiment["num_npu"])
        ]
        key = _alpha_key(alpha)
        slo_by_alpha[key] = statistics.fmean(per_npu)
        slo_distribution[key] = _percentiles(per_npu)

    requests = row["request_rows"]
    categories = {}
    for category in sorted({item["category"] for item in requests}):
        selected = [item for item in requests if item["category"] == category]
        categories[category] = {
            "count": len(selected),
            "mean_slowdown": statistics.fmean(item["slowdown"] for item in selected),
            "slo_attainment": {
                _alpha_key(alpha): statistics.fmean(
                    item["slowdown"] <= alpha for item in selected
                )
                for alpha in ALPHAS
            },
        }

    diagnostics = row["diagnostics"]
    return {
        "mean_npu_utilization": statistics.fmean(utilizations),
        "npu_utilization_distribution": _percentiles(utilizations),
        "ttft_slo_attainment": slo_by_alpha,
        "ttft_slo_per_npu_distribution": slo_distribution,
        "mean_ttft_ms": statistics.fmean(item["ttft_ms"] for item in requests),
        "mean_slowdown": statistics.fmean(item["slowdown"] for item in requests),
        "p95_slowdown": float(
            np.percentile([item["slowdown"] for item in requests], 95)
        ),
        "mean_layer0_barrier_ms": statistics.fmean(
            item["layer0_barrier_ms"] for item in requests
        ),
        "mean_layer1_15_barrier_ms": statistics.fmean(
            item["layer1_15_barrier_ms"] for item in requests
        ),
        "category_metrics": categories,
        "control_evaluations": diagnostics["control_evaluations"],
        "cir_commits": diagnostics["cir_commits"],
        "cir_path_writes": diagnostics["cir_path_writes"],
        "pressure_reports": diagnostics["pressure_reports"],
        "ssd_mean_utilization": diagnostics["ssd_mean_utilization"],
        "npu_link_mean_utilization": diagnostics["npu_link_mean_utilization"],
        "steady_load_guard": {
            "full_load_through_measurement": diagnostics[
                "full_load_through_measurement"
            ],
            "npus_present_through_measurement": diagnostics[
                "npus_present_through_measurement"
            ],
            "guard_margin_ms": diagnostics["guard_margin_ms"],
        },
    }


def _steady_load_guard(rows: list[dict], num_npu: int) -> dict:
    cases = {}
    failures = []
    for row in rows:
        diagnostics = row["diagnostics"]
        case = {
            "full_load_through_measurement": diagnostics[
                "full_load_through_measurement"
            ],
            "npus_present_through_measurement": diagnostics[
                "npus_present_through_measurement"
            ],
            "expected_npus": num_npu,
            "guard_margin_ms": diagnostics["guard_margin_ms"],
        }
        cases.setdefault(row["strategy"], {})[str(row["num_ssu"])] = case
        if not case["full_load_through_measurement"]:
            failures.append(
                {
                    "strategy": row["strategy"],
                    "num_ssu": row["num_ssu"],
                    **case,
                }
            )
    return {
        "evaluated": True,
        "passed": not failures,
        "failure_count": len(failures),
        "minimum_npus_present": min(
            case["npus_present_through_measurement"]
            for strategy_cases in cases.values()
            for case in strategy_cases.values()
        ),
        "minimum_guard_margin_ms": min(
            case["guard_margin_ms"]
            for strategy_cases in cases.values()
            for case in strategy_cases.values()
        ),
        "cases": cases,
        "failures": failures,
    }


def _paired_ttft(rows_by_key: dict, ssus: list[int]) -> dict:
    paired = {}
    for ssu in ssus:
        baseline = {
            _request_key(row): row
            for row in rows_by_key[("baseline", ssu)]["request_rows"]
        }
        scheme_b = {
            _request_key(row): row
            for row in rows_by_key[("scheme_b", ssu)]["request_rows"]
        }
        if set(baseline) != set(scheme_b):
            raise ValueError(f"SSU {ssu} Scheme B TTFT rows are not request-paired")
        deltas = [
            scheme_b[key]["ttft_ms"] - baseline[key]["ttft_ms"]
            for key in sorted(baseline)
        ]
        tolerance = 1e-9
        wins = sum(delta < -tolerance for delta in deltas)
        ties = sum(abs(delta) <= tolerance for delta in deltas)
        paired[str(ssu)] = {
            "request_count": len(deltas),
            "scheme_b_win_rate": wins / len(deltas),
            "tie_rate": ties / len(deltas),
            "baseline_win_rate": (len(deltas) - wins - ties) / len(deltas),
            "median_ttft_delta_ms": statistics.median(deltas),
            "mean_ttft_delta_ms": statistics.fmean(deltas),
            "p05_ttft_delta_ms": float(np.percentile(deltas, 5)),
            "p95_ttft_delta_ms": float(np.percentile(deltas, 95)),
            "delta_definition": "Scheme B TTFT minus baseline TTFT",
        }
    return paired


def analyze(payload: dict) -> dict:
    experiment, rows, strategy_order, ssus = _validate(payload)
    if not rows:
        return {
            "schema_version": 1,
            "complete": False,
            "empty": True,
            "alpha_grid": list(ALPHAS),
            "plot_alpha": PLOT_ALPHA,
            "strategy_order": [],
            "ssu_list": [],
            "metrics": {},
            "steady_load_guard": {
                "evaluated": False,
                "passed": None,
                "failure_count": 0,
                "failures": [],
            },
            "paired_ttft_scheme_b_vs_baseline": {},
            "validation": {
                "paired": True,
                "row_count": 0,
                "steady_load_guard_passed": None,
            },
        }

    by_key = {(row["strategy"], row["num_ssu"]): row for row in rows}
    metrics = {
        strategy: {
            str(ssu): _case_metrics(by_key[(strategy, ssu)], experiment)
            for ssu in ssus
        }
        for strategy in strategy_order
    }
    steady_load_guard = _steady_load_guard(rows, experiment["num_npu"])
    paired_ttft = _paired_ttft(by_key, ssus)
    baseline_deltas = {}
    for strategy in strategy_order:
        if strategy == "baseline":
            continue
        baseline_deltas[strategy] = {}
        for ssu in ssus:
            candidate = metrics[strategy][str(ssu)]
            baseline = metrics["baseline"][str(ssu)]
            baseline_deltas[strategy][str(ssu)] = {
                "npu_utilization_delta_pp": 100.0
                * (
                    candidate["mean_npu_utilization"]
                    - baseline["mean_npu_utilization"]
                ),
                "ttft_slo_delta_pp": {
                    _alpha_key(alpha): 100.0
                    * (
                        candidate["ttft_slo_attainment"][_alpha_key(alpha)]
                        - baseline["ttft_slo_attainment"][_alpha_key(alpha)]
                    )
                    for alpha in ALPHAS
                },
            }
    return {
        "schema_version": 1,
        "complete": bool(payload.get("complete")),
        "empty": False,
        "metric_scope": "middle measured requests in closed-loop batch=1 streams",
        "ttft_definition": experiment.get("ttft_definition"),
        "slo_definition": "TTFT <= alpha * 16-layer compute-only ideal TTFT",
        "npu_utilization_definition": (
            "per-NPU measured compute busy time / measured active window, then "
            "equal-weight mean across NPUs"
        ),
        "num_npu": experiment["num_npu"],
        "requests_per_npu": experiment["requests_per_npu"],
        "warmup_requests_per_npu": experiment["warmup_requests_per_npu"],
        "measured_requests_per_npu": experiment["measured_requests_per_npu"],
        "cooldown_requests_per_npu": experiment["cooldown_requests_per_npu"],
        "n_layers": experiment["n_layers"],
        "alpha_grid": list(ALPHAS),
        "plot_alpha": PLOT_ALPHA,
        "strategy_order": list(strategy_order),
        "core_strategies": list(CORE_STRATEGIES),
        "improved_strategies": [
            strategy for strategy in IMPROVED_STRATEGIES if strategy in strategy_order
        ],
        "ssu_list": ssus,
        "metrics": metrics,
        "deltas_vs_baseline": baseline_deltas,
        "paired_ttft_scheme_b_vs_baseline": paired_ttft,
        "steady_load_guard": steady_load_guard,
        "validation": {
            "paired": True,
            "steady_load_guard_passed": steady_load_guard["passed"],
            "row_count": len(rows),
            "strategies_per_ssu": len(strategy_order),
            "measured_request_rows_per_case": experiment["num_npu"]
            * experiment["measured_requests_per_npu"],
            "npu_rows_per_case": experiment["num_npu"],
            "request_assignment_fixed_across_ssus": True,
        },
    }


def _table(lines: list[str], analysis: dict, field, *, alpha=None, strategies=None):
    ssus = analysis["ssu_list"]
    strategies = tuple(strategies or analysis["strategy_order"])
    lines.extend(
        [
            "| Strategy | " + " | ".join(f"SSU {ssu}" for ssu in ssus) + " |",
            "|---|" + "---:|" * len(ssus),
        ]
    )
    for strategy in strategies:
        values = []
        for ssu in ssus:
            row = analysis["metrics"][strategy][str(ssu)]
            value = (
                row[field]
                if alpha is None
                else row[field][_alpha_key(alpha)]
            )
            values.append(f"{100.0 * value:.2f}%")
        lines.append(f"| {LABELS[strategy]} | " + " | ".join(values) + " |")


def write_report(path: Path, analysis: dict, plot_alpha: float):
    if analysis["empty"]:
        path.write_text(
            "# Closed-loop batch-1 analysis\n\nNo completed paired rows are available.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# Closed-loop batch-1 TTFT SLO and NPU utilization",
        "",
        f"Each of {analysis['num_npu']} NPUs owns {analysis['requests_per_npu']} "
        "fixed requests. The first "
        f"{analysis['warmup_requests_per_npu']} are warm-up, the middle "
        f"{analysis['measured_requests_per_npu']} are measured, and the final "
        f"{analysis['cooldown_requests_per_npu']} keep the shared SSD load active.",
        "",
        "TTFT is completion minus admission for one closed-loop request. Its SLO "
        "is `alpha × 16-layer compute-only TTFT`. Both SLO attainment and physical "
        "NPU utilization are computed per NPU first and then averaged with equal "
        "NPU weight.",
        "",
    ]
    guard = analysis["steady_load_guard"]
    lines.extend(["## Steady-load guard", ""])
    if guard["passed"]:
        lines.append(
            f"**PASS.** All {analysis['validation']['row_count']} cases retain all "
            f"{analysis['num_npu']} NPU streams through the global end of the "
            "measurement window. The minimum cooldown guard margin is "
            f"{guard['minimum_guard_margin_ms']:.3f} ms."
        )
    else:
        lines.extend(
            [
                f"**FAIL.** {guard['failure_count']} cases drain at least one NPU "
                "stream before the global measurement window ends; their steady-load "
                "metrics may be optimistic.",
                "",
                "| Strategy | SSU | NPUs still present | Expected | Guard margin |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for failure in guard["failures"]:
            lines.append(
                f"| {LABELS[failure['strategy']]} | {failure['num_ssu']} | "
                f"{failure['npus_present_through_measurement']} | "
                f"{failure['expected_npus']} | "
                f"{failure['guard_margin_ms']:.3f} ms |"
            )
    lines.extend(["", "## Mean physical NPU utilization", ""])
    _table(
        lines,
        analysis,
        "mean_npu_utilization",
        strategies=CORE_STRATEGIES,
    )
    lines.extend(
        [
            "",
            f"## TTFT SLO attainment at {plot_alpha:g}× ideal",
            "",
        ]
    )
    _table(
        lines,
        analysis,
        "ttft_slo_attainment",
        alpha=plot_alpha,
        strategies=CORE_STRATEGIES,
    )
    if analysis["improved_strategies"]:
        improvement_order = (
            "baseline",
            "scheme_b",
            *analysis["improved_strategies"],
        )
        lines.extend(
            [
                "",
                "## Realistic Scheme B improvements",
                "",
                f"The separate improvement figure compares these policies at "
                f"the same {plot_alpha:g}× SLO.",
                "",
            ]
        )
        _table(
            lines,
            analysis,
            "ttft_slo_attainment",
            alpha=plot_alpha,
            strategies=improvement_order,
        )
    lines.extend(
        [
            "",
            "## Scheme B paired difference from baseline",
            "",
            f"| SSU | Utilization delta | {plot_alpha:g}× SLO delta | "
            "TTFT win rate | Median TTFT delta | Layer-0 wait delta | "
            "Layer 1–15 wait delta |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in analysis["ssu_list"]:
        scheme = analysis["metrics"]["scheme_b"][str(ssu)]
        baseline = analysis["metrics"]["baseline"][str(ssu)]
        delta = analysis["deltas_vs_baseline"]["scheme_b"][str(ssu)]
        paired = analysis["paired_ttft_scheme_b_vs_baseline"][str(ssu)]
        lines.append(
            f"| {ssu} | {delta['npu_utilization_delta_pp']:+.2f} pp | "
            f"{delta['ttft_slo_delta_pp'][_alpha_key(plot_alpha)]:+.2f} pp | "
            f"{100.0 * paired['scheme_b_win_rate']:.2f}% | "
            f"{paired['median_ttft_delta_ms']:+.3f} ms | "
            f"{scheme['mean_layer0_barrier_ms'] - baseline['mean_layer0_barrier_ms']:+.2f} ms | "
            f"{scheme['mean_layer1_15_barrier_ms'] - baseline['mean_layer1_15_barrier_ms']:+.2f} ms |"
        )
    lines.extend(
        [
            "",
            "All seven SLO multipliers are retained in `analysis.json`. "
            "A negative paired TTFT delta means Scheme B is faster for that request. "
            "Full-info EDF preserves the same placement and SSD/NPU capacities; "
            "it is a clairvoyant feasible reference, not a proven optimum.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path,
    analysis: dict,
    plot_alpha: float,
    *,
    strategies=CORE_STRATEGIES,
    title="Closed-loop batch-1 storage QoS",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True)
    if analysis["empty"]:
        for axis in axes:
            axis.text(0.5, 0.5, "No completed paired rows", ha="center", va="center")
            axis.set_axis_off()
    else:
        for strategy in strategies:
            improved = strategy in IMPROVED_STRATEGIES
            style = "--" if improved else "-"
            ssus = analysis["ssu_list"]
            utilization = [
                100.0
                * analysis["metrics"][strategy][str(ssu)]["mean_npu_utilization"]
                for ssu in ssus
            ]
            attainment = [
                100.0
                * analysis["metrics"][strategy][str(ssu)]["ttft_slo_attainment"]
                [_alpha_key(plot_alpha)]
                for ssu in ssus
            ]
            for axis, values in zip(axes, (utilization, attainment)):
                axis.plot(
                    ssus,
                    values,
                    linestyle=style,
                    marker=MARKERS[strategy],
                    color=COLORS[strategy],
                    linewidth=2.2,
                    markersize=5.5,
                    label=LABELS[strategy],
                )
        axes[0].set_title("Mean physical NPU utilization")
        axes[0].set_ylabel("NPU utilization (%)")
        axes[1].set_title(f"TTFT SLO attainment ({plot_alpha:g}× ideal)")
        axes[1].set_ylabel("Requests meeting SLO (%)")
        for axis in axes:
            axis.set_xlabel("Number of SSUs")
            axis.set_xticks(analysis["ssu_list"])
            axis.set_ylim(0.0, 100.0)
            axis.grid(True, alpha=0.28)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=min(3, len(strategies)),
            frameon=False,
        )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_json(path: Path, value: dict):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-alpha",
        type=float,
        choices=ALPHAS,
        default=PLOT_ALPHA,
        help="SLO multiplier displayed in the report and PNG",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(payload)
    analysis["plot_alpha"] = args.plot_alpha
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / DEFAULT_ANALYSIS
    report_path = args.output_dir / DEFAULT_REPORT
    plot_path = args.output_dir / DEFAULT_PLOT
    improvement_plot_path = args.output_dir / DEFAULT_IMPROVEMENT_PLOT
    analysis["plot_files"] = [DEFAULT_PLOT]
    if analysis.get("improved_strategies"):
        analysis["plot_files"].append(DEFAULT_IMPROVEMENT_PLOT)
    _write_json(analysis_path, analysis)
    write_report(report_path, analysis, args.plot_alpha)
    write_plot(
        plot_path,
        analysis,
        args.plot_alpha,
        strategies=CORE_STRATEGIES,
        title="Closed-loop batch-1: four requested strategies",
    )
    if analysis.get("improved_strategies"):
        improvement_order = (
            "baseline",
            "scheme_b",
            *analysis["improved_strategies"],
        )
        write_plot(
            improvement_plot_path,
            analysis,
            args.plot_alpha,
            strategies=improvement_order,
            title="Closed-loop batch-1: realistic Scheme B improvements",
        )
    print(f"analysis: {analysis_path}")
    print(f"report:   {report_path}")
    print(f"plot:     {plot_path}")
    if analysis.get("improved_strategies"):
        print(f"plot:     {improvement_plot_path}")


if __name__ == "__main__":
    main()
