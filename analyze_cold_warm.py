"""Analyze routing/Scheme-B cold and warm six-request metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cold_warm_experiment import CASES, LAYER_LIST, SSU_LIST


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "cold_warm_modified" / "results.json"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
PLOT_ALPHA = 2.0
LABELS = {
    "modified_baseline": "Baseline + L0 prefetch",
    "modified_layer_once": "Read once/layer + L0 prefetch",
    "modified_refresh8": "Refresh8 + L0 prefetch",
    "modified_scheme_b": "Scheme B + manifest/CIR prefetch",
}
COLORS = {
    "modified_baseline": "#4C78A8",
    "modified_layer_once": "#59A14F",
    "modified_refresh8": "#F2CF5B",
    "modified_scheme_b": "#E45756",
}
MARKERS = {
    "modified_baseline": "o",
    "modified_layer_once": "D",
    "modified_refresh8": "s",
    "modified_scheme_b": "^",
}

_MIXED_RUN_FIELDS = {"schema_version", "source_fingerprint", "strategies"}


def _key(row):
    return row["strategy"], row["num_ssu"], row["n_layers"]


def _alpha_key(alpha):
    return f"{alpha:g}"


def _display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def merge_compatible_payloads(sources):
    """Merge paired strategy rows while preserving per-run provenance."""
    if len(sources) == 1:
        return sources[0][1]

    reference = {
        key: value
        for key, value in sources[0][1]["experiment"].items()
        if key not in _MIXED_RUN_FIELDS
    }
    rows_by_key = {}
    source_runs = []
    row_source = {}
    fingerprints_by_strategy = {}
    for path, payload in sources:
        experiment = payload["experiment"]
        comparable = {
            key: value
            for key, value in experiment.items()
            if key not in _MIXED_RUN_FIELDS
        }
        if comparable != reference:
            raise ValueError(f"incompatible experiment metadata: {path}")
        display_path = _display_path(path)
        source_runs.append(
            {
                "path": display_path,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_fingerprint": experiment.get("source_fingerprint"),
                "strategies_with_rows": sorted(
                    {row["strategy"] for row in payload["results"]}
                ),
            }
        )
        for row in payload["results"]:
            key = _key(row)
            if key in rows_by_key:
                raise ValueError(f"duplicate merged result row: {key}")
            rows_by_key[key] = row
            strategy = row["strategy"]
            previous = row_source.setdefault(strategy, display_path)
            if previous != display_path:
                raise ValueError(f"strategy spans multiple source runs: {strategy}")
            fingerprints_by_strategy[strategy] = experiment.get(
                "source_fingerprint"
            )

    present = {key[0] for key in rows_by_key}
    strategies = tuple(case.name for case in CASES if case.name in present)
    if set(strategies) != present:
        raise ValueError(f"unknown strategies in merged results: {present}")
    order = {strategy: index for index, strategy in enumerate(strategies)}
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            row["n_layers"],
            row["num_ssu"],
            order[row["strategy"]],
        ),
    )
    experiment = deepcopy(sources[-1][1]["experiment"])
    experiment.pop("source_fingerprint", None)
    experiment["source_fingerprint_scope"] = "per strategy; see provenance"
    experiment["source_fingerprints_by_strategy"] = fingerprints_by_strategy
    experiment["strategies"] = list(strategies)
    merged = {
        "schema_version": max(payload["schema_version"] for _, payload in sources),
        "complete": False,
        "selected_complete": True,
        "experiment": experiment,
        "results": rows,
        "comparison_provenance": {
            "kind": "merged_compatible_runs",
            "single_source": False,
            "source_runs": source_runs,
            "row_source": row_source,
            "compatibility": {
                "experiment_metadata_except_schema_source_and_strategies": True,
                "paired_row_fingerprints_checked_by_analyzer": True,
                "behavior_compatibility_audited": True,
            },
        },
    }
    _validate(merged)
    return merged


def _validate(payload):
    experiment = payload["experiment"]
    rows = payload["results"]
    strategies = tuple(
        experiment.get("strategies", (case.name for case in CASES))
    )
    keys = [_key(row) for row in rows]
    if any(count != 1 for count in Counter(keys).values()):
        raise ValueError("result rows contain duplicate strategy/SSU/layer keys")

    for n_layers in sorted({row["n_layers"] for row in rows}):
        for num_ssu in sorted(
            {row["num_ssu"] for row in rows if row["n_layers"] == n_layers}
        ):
            group = [
                row
                for row in rows
                if row["n_layers"] == n_layers and row["num_ssu"] == num_ssu
            ]
            if {row["strategy"] for row in group} != set(strategies):
                raise ValueError(
                    f"incomplete strategy set at layers={n_layers}, SSU={num_ssu}"
                )
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
                        f"unpaired {field} at layers={n_layers}, SSU={num_ssu}"
                    )

            for row in group:
                if not row["all_requests_completed"]:
                    raise ValueError("cold/warm metrics require a completed trace")
                if not row["survivorship_bias_excluded"]:
                    raise ValueError("survivorship-bias guard is not set")
                if len(row["request_rows"]) != experiment["num_npu"] * 6:
                    raise ValueError("request rows do not cover all six requests")
                for cohort, count in (("cold", 6), ("warm", 5)):
                    metrics = row["cohorts"][cohort]
                    if metrics["ttft_request_count"] != experiment["num_npu"] * count:
                        raise ValueError(f"{cohort} TTFT cohort has the wrong size")
                    if len(metrics["npu_rows"]) != experiment["num_npu"]:
                        raise ValueError(f"{cohort} utilization misses NPUs")
                    mean_util = sum(
                        item["utilization"] for item in metrics["npu_rows"]
                    ) / experiment["num_npu"]
                    if not math.isclose(
                        mean_util,
                        metrics["mean_npu_utilization"],
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    ):
                        raise ValueError(f"{cohort} utilization mean is inconsistent")
    return experiment, rows, strategies


def analyze(payload):
    experiment, rows, strategies = _validate(payload)
    layers = sorted({row["n_layers"] for row in rows})
    ssus = sorted({row["num_ssu"] for row in rows})
    by_key = {_key(row): row for row in rows}
    metrics = {}
    for strategy in strategies:
        metrics[strategy] = {}
        for n_layers in layers:
            metrics[strategy][str(n_layers)] = {}
            for num_ssu in ssus:
                row = by_key[(strategy, num_ssu, n_layers)]
                metrics[strategy][str(n_layers)][str(num_ssu)] = {
                    "cold": {
                        "mean_npu_utilization": row["cohorts"]["cold"][
                            "mean_npu_utilization"
                        ],
                        "mean_ttft_ms": row["cohorts"]["cold"]["mean_ttft_ms"],
                        "ttft_slo": row["cohorts"]["cold"]["slo"],
                    },
                    "warm": {
                        "mean_npu_utilization": row["cohorts"]["warm"][
                            "mean_npu_utilization"
                        ],
                        "mean_ttft_ms": row["cohorts"]["warm"]["mean_ttft_ms"],
                        "ttft_slo": row["cohorts"]["warm"]["slo"],
                    },
                    "first_request_only": row["first_request_only"],
                    "wall_time_s": row["wall_time_s"],
                }

    strategy_deltas = {}
    for strategy in strategies:
        if strategy == "modified_baseline":
            continue
        strategy_deltas[strategy] = {}
        for n_layers in layers:
            strategy_deltas[strategy][str(n_layers)] = {}
            for num_ssu in ssus:
                baseline = metrics["modified_baseline"][str(n_layers)][
                    str(num_ssu)
                ]
                candidate = metrics[strategy][str(n_layers)][str(num_ssu)]
                strategy_deltas[strategy][str(n_layers)][str(num_ssu)] = {
                    cohort: {
                        "npu_utilization_delta_pp": 100.0
                        * (
                            candidate[cohort]["mean_npu_utilization"]
                            - baseline[cohort]["mean_npu_utilization"]
                        ),
                        "ttft_slo_delta_pp": 100.0
                        * (
                            candidate[cohort]["ttft_slo"][
                                _alpha_key(PLOT_ALPHA)
                            ]["attainment"]
                            - baseline[cohort]["ttft_slo"][
                                _alpha_key(PLOT_ALPHA)
                            ]["attainment"]
                        ),
                    }
                    for cohort in ("cold", "warm")
                }

    return {
        "schema_version": 3,
        "complete": bool(payload.get("complete")),
        "selected_complete": bool(
            payload.get("selected_complete", payload.get("complete"))
        ),
        "num_npu": experiment["num_npu"],
        "requests_per_npu": experiment["requests_per_npu"],
        "layer_list": layers,
        "ssu_list": ssus,
        "strategies": list(strategies),
        "plot_alpha": PLOT_ALPHA,
        "metric_scope": {
            "cold": experiment["cold_definition"],
            "warm": experiment["warm_definition"],
            "ttft": experiment["ttft_definition"],
            "utilization": experiment["npu_utilization_definition"],
        },
        "metrics": metrics,
        "strategy_minus_modified_baseline": strategy_deltas,
        "modified_layer_once_minus_modified_baseline": strategy_deltas.get(
            "modified_layer_once", {}
        ),
        "modified_refresh8_minus_modified_baseline": strategy_deltas.get(
            "modified_refresh8", {}
        ),
        "modified_scheme_b_minus_modified_baseline": strategy_deltas.get(
            "modified_scheme_b", {}
        ),
        "validation": {
            "paired_inputs": True,
            "fixed_complete_cohorts": True,
            "survivorship_bias_excluded": True,
            "external_arrival_queue_wait_excluded_from_ttft": True,
            "result_rows": len(rows),
        },
        "comparison_provenance": payload.get("comparison_provenance"),
    }


def write_plot(path, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = analysis["layer_list"]
    ssus = analysis["ssu_list"]
    if len(layers) == 1 and len(ssus) == 1:
        n_layers = layers[0]
        num_ssu = ssus[0]
        strategies = analysis["strategies"]
        x = np.arange(len(strategies))
        width = 0.36
        figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
        for axis, (metric_name, ylabel) in zip(
            axes,
            (
                ("mean_npu_utilization", "Mean NPU utilization (%)"),
                ("ttft_slo", f"TTFT SLO attainment @ {PLOT_ALPHA:g}x (%)"),
            ),
        ):
            for cohort, offset, alpha in (
                ("cold", -width / 2.0, 1.0),
                ("warm", width / 2.0, 0.58),
            ):
                values = []
                for strategy in strategies:
                    point = analysis["metrics"][strategy][str(n_layers)][
                        str(num_ssu)
                    ][cohort]
                    value = (
                        point[metric_name]
                        if metric_name == "mean_npu_utilization"
                        else point[metric_name][_alpha_key(PLOT_ALPHA)]["attainment"]
                    )
                    values.append(100.0 * value)
                axis.bar(
                    x + offset,
                    values,
                    width,
                    color=[COLORS[strategy] for strategy in strategies],
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.5,
                    label=cohort,
                )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [LABELS[strategy].split(" +", 1)[0] for strategy in strategies]
            )
            axis.set_ylabel(ylabel)
            axis.set_ylim(0.0, 100.0)
            axis.grid(axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        figure.suptitle(
            f"{n_layers} layers, {num_ssu} SSUs: cold and warm requests",
            y=0.94,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    figure, axes = plt.subplots(
        2,
        len(layers),
        figsize=(4.2 * len(layers), 8.0),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    for column, n_layers in enumerate(layers):
        util_axis = axes[0, column]
        slo_axis = axes[1, column]
        for strategy in analysis["strategies"]:
            for cohort, linestyle in (("cold", "-"), ("warm", "--")):
                points = analysis["metrics"][strategy][str(n_layers)]
                util_axis.plot(
                    ssus,
                    [
                        100.0
                        * points[str(ssu)][cohort]["mean_npu_utilization"]
                        for ssu in ssus
                    ],
                    color=COLORS[strategy],
                    marker=MARKERS[strategy],
                    linestyle=linestyle,
                    linewidth=2.0,
                    markersize=5.5,
                    label=f"{LABELS[strategy]} — {cohort}",
                )
                slo_axis.plot(
                    ssus,
                    [
                        100.0
                        * points[str(ssu)][cohort]["ttft_slo"][
                            _alpha_key(PLOT_ALPHA)
                        ]["attainment"]
                        for ssu in ssus
                    ],
                    color=COLORS[strategy],
                    marker=MARKERS[strategy],
                    linestyle=linestyle,
                    linewidth=2.0,
                    markersize=5.5,
                )
        util_axis.set_title(f"{n_layers} layers")
        util_axis.grid(alpha=0.25)
        slo_axis.grid(alpha=0.25)
        slo_axis.set_xlabel("Number of SSUs")
        slo_axis.set_xticks(ssus)
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
    figure.suptitle("Modified strategies: cold and warm requests", y=0.96)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_ssu40_plot(path, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_ssu = 40
    layers = analysis["layer_list"]
    if num_ssu not in analysis["ssu_list"]:
        raise ValueError("SSU=40 is missing from the analyzed results")

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True)
    metric_specs = (
        ("mean_npu_utilization", "Mean NPU utilization (%)"),
        ("ttft_slo", f"TTFT SLO attainment @ {PLOT_ALPHA:g}x (%)"),
    )
    for axis, (metric_name, ylabel) in zip(axes, metric_specs):
        for strategy in analysis["strategies"]:
            for cohort, linestyle in (("cold", "-"), ("warm", "--")):
                values = []
                for n_layers in layers:
                    point = analysis["metrics"][strategy][str(n_layers)][
                        str(num_ssu)
                    ][cohort]
                    value = (
                        point[metric_name]
                        if metric_name == "mean_npu_utilization"
                        else point[metric_name][_alpha_key(PLOT_ALPHA)]["attainment"]
                    )
                    values.append(100.0 * value)
                axis.plot(
                    layers,
                    values,
                    color=COLORS[strategy],
                    marker=MARKERS[strategy],
                    linestyle=linestyle,
                    linewidth=2.2,
                    markersize=6.5,
                    label=f"{LABELS[strategy]} — {cohort}",
                )
        axis.set_xlabel("Number of layers")
        axis.set_ylabel(ylabel)
        axis.set_xticks(layers)
        axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(analysis["strategies"]),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle("SSU = 40: cold and warm requests", y=0.91)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_layer16_plot(path, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = 16
    ssus = analysis["ssu_list"]
    if n_layers not in analysis["layer_list"]:
        raise ValueError("16-layer results are missing from the analysis")

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True)
    metric_specs = (
        ("mean_npu_utilization", "Mean NPU utilization (%)"),
        ("ttft_slo", f"TTFT SLO attainment @ {PLOT_ALPHA:g}x (%)"),
    )
    for axis, (metric_name, ylabel) in zip(axes, metric_specs):
        for strategy in analysis["strategies"]:
            for cohort, linestyle in (("cold", "-"), ("warm", "--")):
                values = []
                for num_ssu in ssus:
                    point = analysis["metrics"][strategy][str(n_layers)][
                        str(num_ssu)
                    ][cohort]
                    value = (
                        point[metric_name]
                        if metric_name == "mean_npu_utilization"
                        else point[metric_name][_alpha_key(PLOT_ALPHA)]["attainment"]
                    )
                    values.append(100.0 * value)
                axis.plot(
                    ssus,
                    values,
                    color=COLORS[strategy],
                    marker=MARKERS[strategy],
                    linestyle=linestyle,
                    linewidth=2.2,
                    markersize=6.5,
                    label=f"{LABELS[strategy]} — {cohort}",
                )
        axis.set_xlabel("Number of SSUs")
        axis.set_ylabel(ylabel)
        axis.set_xticks(ssus)
        axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(analysis["strategies"]),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle("16 layers: cold and warm requests", y=0.91)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _percent(value):
    return f"{100.0 * value:.2f}%"


def _delta_pp(value):
    rendered = f"{value:+.2f} pp"
    return f"**{rendered}**" if value >= 10.0 else rendered


def write_report(path, analysis):
    lines = [
        "# Modified strategies: cold and warm six-request results",
        "",
        "Both views come from the same fully completed trace. Cold uses each "
        "NPU's request-0 admission through request-5 completion; warm starts at "
        "request-1 admission. TTFT is completion minus admission, so external "
        "arrival queue wait is not included.",
        "",
        f"Primary SLO: `TTFT <= {PLOT_ALPHA:g} × compute-only TTFT`.",
        "",
        "A strategy meets the requested utilization criterion only when its "
        "gain over baseline is at least `+10 pp`.",
    ]
    provenance = analysis.get("comparison_provenance")
    if provenance:
        lines.extend(
            [
                "",
                "Provenance: this comparison merges paired, behavior-compatible "
                "runs and preserves a separate source fingerprint for each "
                "strategy. It is not presented as a single-checkout rerun; see "
                "`comparison_results.json` for file hashes and row sources.",
            ]
        )
    for strategy in analysis["strategies"]:
        if strategy == "modified_baseline":
            continue
        for cohort in ("cold", "warm"):
            lines.extend(
                [
                    "",
                    f"## {LABELS[strategy]} minus baseline — {cohort}",
                    "",
                    "### Mean NPU utilization delta",
                    "",
                    "| Layers | "
                    + " | ".join(f"SSU {ssu}" for ssu in analysis["ssu_list"])
                    + " |",
                    "|---:|" + "---:|" * len(analysis["ssu_list"]),
                ]
            )
            for n_layers in analysis["layer_list"]:
                values = [
                    analysis["strategy_minus_modified_baseline"][strategy][
                        str(n_layers)
                    ][str(ssu)][cohort]["npu_utilization_delta_pp"]
                    for ssu in analysis["ssu_list"]
                ]
                lines.append(
                    f"| {n_layers} | "
                    + " | ".join(_delta_pp(value) for value in values)
                    + " |"
                )
            lines.extend(
                [
                    "",
                    f"### TTFT SLO attainment delta @ {PLOT_ALPHA:g}x",
                    "",
                    "| Layers | "
                    + " | ".join(f"SSU {ssu}" for ssu in analysis["ssu_list"])
                    + " |",
                    "|---:|" + "---:|" * len(analysis["ssu_list"]),
                ]
            )
            for n_layers in analysis["layer_list"]:
                values = [
                    analysis["strategy_minus_modified_baseline"][strategy][
                        str(n_layers)
                    ][str(ssu)][cohort]["ttft_slo_delta_pp"]
                    for ssu in analysis["ssu_list"]
                ]
                lines.append(
                    f"| {n_layers} | "
                    + " | ".join(f"{value:+.2f} pp" for value in values)
                    + " |"
                )
    for n_layers in analysis["layer_list"]:
        lines.extend(["", f"## {n_layers} layers", ""])
        for metric_name, title in (
            ("mean_npu_utilization", "Mean NPU utilization"),
            ("ttft_slo", f"TTFT SLO attainment @ {PLOT_ALPHA:g}x"),
        ):
            lines.extend(
                [
                    f"### {title}",
                    "",
                    "| Strategy / cohort | "
                    + " | ".join(
                        f"SSU {ssu}" for ssu in analysis["ssu_list"]
                    )
                    + " |",
                    "|---|" + "---:|" * len(analysis["ssu_list"]),
                ]
            )
            for strategy in analysis["strategies"]:
                for cohort in ("cold", "warm"):
                    values = []
                    for ssu in analysis["ssu_list"]:
                        row = analysis["metrics"][strategy][str(n_layers)][str(ssu)][
                            cohort
                        ]
                        value = (
                            row[metric_name]
                            if metric_name == "mean_npu_utilization"
                            else row[metric_name][_alpha_key(PLOT_ALPHA)]["attainment"]
                        )
                        values.append(_percent(value))
                    lines.append(
                        f"| {LABELS[strategy]} — {cohort} | "
                        + " | ".join(values)
                        + " |"
                    )
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_paths = tuple(args.input or (DEFAULT_INPUT,))
    payload = merge_compatible_payloads(
        tuple((path, json.loads(path.read_text())) for path in input_paths)
    )
    analysis = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(input_paths) > 1:
        (args.output_dir / "comparison_results.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", analysis)
    write_plot(args.output_dir / "01_cold_warm.png", analysis)
    write_ssu40_plot(args.output_dir / "02_ssu40_cold_warm.png", analysis)
    write_layer16_plot(args.output_dir / "03_layer16_cold_warm_by_ssu.png", analysis)


if __name__ == "__main__":
    main()
