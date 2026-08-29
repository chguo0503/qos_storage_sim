"""Analyze the paired 16-layer steady-state full-load experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT / "results" / "steady_state_full_load_layer16" / "results.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
PLOT_PATH_NAME = "01_steady_state_full_load.png"
LAYER_COUNT = 16
SSU_LIST = (16, 24, 40, 70)
STRATEGIES = ("baseline", "layer_once", "scheme_b")
STRATEGY_ALIASES = {
    "baseline": "baseline",
    "modified_baseline": "baseline",
    "layer_once": "layer_once",
    "read_once_per_layer": "layer_once",
    "modified_layer_once": "layer_once",
    "scheme_b": "scheme_b",
    "modified_scheme_b": "scheme_b",
}
LABELS = {
    "baseline": "Baseline",
    "layer_once": "Read once per layer",
    "scheme_b": "Scheme B",
}
COLORS = {
    "baseline": "#4C78A8",
    "layer_once": "#59A14F",
    "scheme_b": "#E45756",
}
MARKERS = {"baseline": "o", "layer_once": "D", "scheme_b": "^"}
PAIR_FIELDS = ("workload_hash", "placement_hash", "trace_hash")


def _canonical_strategy(value):
    try:
        return STRATEGY_ALIASES[value]
    except KeyError as error:
        raise ValueError(f"unsupported steady-state strategy: {value}") from error


def _result_key(row):
    return _canonical_strategy(row["strategy"]), int(row["num_ssu"])


def _validate_probability(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return value


def _validate(payload):
    if "experiment" not in payload or "results" not in payload:
        raise ValueError("expected payload.experiment and payload.results")
    experiment = payload["experiment"]
    if int(experiment["n_layers"]) != LAYER_COUNT:
        raise ValueError(f"expected {LAYER_COUNT} layers")
    if float(experiment["steady_state"]["slo_alpha"]) != 2.0:
        raise ValueError("steady-state primary SLO must use alpha=2")

    selected = []
    for row in payload["results"]:
        strategy = STRATEGY_ALIASES.get(row.get("strategy"))
        if (
            strategy in STRATEGIES
            and int(row.get("n_layers", -1)) == LAYER_COUNT
            and int(row.get("num_ssu", -1)) in SSU_LIST
        ):
            selected.append(row)

    keys = [_result_key(row) for row in selected]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate steady-state result rows: {duplicates}")
    expected = {(strategy, ssu) for strategy in STRATEGIES for ssu in SSU_LIST}
    missing = sorted(expected - set(keys))
    if missing:
        raise ValueError(f"missing steady-state result rows: {missing}")

    paired_hashes = {}
    for num_ssu in SSU_LIST:
        group = [row for row in selected if int(row["num_ssu"]) == num_ssu]
        if {_canonical_strategy(row["strategy"]) for row in group} != set(
            STRATEGIES
        ):
            raise ValueError(f"incomplete strategy set at SSU={num_ssu}")
        paired_hashes[str(num_ssu)] = {}
        for field in PAIR_FIELDS:
            values = {row[field] for row in group}
            if len(values) != 1:
                raise ValueError(f"unpaired {field} at SSU={num_ssu}")
            paired_hashes[str(num_ssu)][field] = next(iter(values))

        for row in group:
            _validate_probability(
                row["mean_npu_utilization"], "mean_npu_utilization"
            )
            _validate_probability(
                row["ttft_slo_attainment"], "ttft_slo_attainment"
            )
            _validate_probability(
                row["request_weighted_slo_attainment"],
                "request_weighted_slo_attainment",
            )
            if not math.isfinite(float(row["mean_ttft_ms"])):
                raise ValueError("mean_ttft_ms must be finite")
            if not math.isfinite(float(row["p99_ttft_ms"])):
                raise ValueError("p99_ttft_ms must be finite")
            if not row["measurement_blocks"]:
                raise ValueError("measurement_blocks must not be empty")
            if not row["invariants"] or not all(row["invariants"].values()):
                raise ValueError(
                    f"failed invariant for {_result_key(row)}: {row['invariants']}"
                )

    selected.sort(
        key=lambda row: (
            SSU_LIST.index(int(row["num_ssu"])),
            STRATEGIES.index(_canonical_strategy(row["strategy"])),
        )
    )
    return experiment, selected, paired_hashes


def analyze(payload):
    experiment, rows, paired_hashes = _validate(payload)
    by_key = {_result_key(row): row for row in rows}
    metrics = {strategy: {} for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for num_ssu in SSU_LIST:
            row = by_key[(strategy, num_ssu)]
            metrics[strategy][str(num_ssu)] = {
                "mean_npu_utilization": float(row["mean_npu_utilization"]),
                "ttft_slo_attainment_2x": float(row["ttft_slo_attainment"]),
                "request_weighted_slo_attainment_2x": float(
                    row["request_weighted_slo_attainment"]
                ),
                "mean_ttft_ms": float(row["mean_ttft_ms"]),
                "p99_ttft_ms": float(row["p99_ttft_ms"]),
                "measurement_request_count": int(
                    row.get("measurement_request_count", 0)
                ),
                "measurement_duration_ms": float(
                    row.get(
                        "measurement_duration_ms",
                        experiment["steady_state"]["measurement_ms"],
                    )
                ),
                "measurement_blocks": row["measurement_blocks"],
                "invariants": row["invariants"],
            }

    deltas = {}
    for strategy in STRATEGIES[1:]:
        deltas[strategy] = {}
        for num_ssu in SSU_LIST:
            baseline = metrics["baseline"][str(num_ssu)]
            candidate = metrics[strategy][str(num_ssu)]
            deltas[strategy][str(num_ssu)] = {
                "mean_npu_utilization_delta_pp": 100.0
                * (
                    candidate["mean_npu_utilization"]
                    - baseline["mean_npu_utilization"]
                ),
                "ttft_slo_attainment_2x_delta_pp": 100.0
                * (
                    candidate["ttft_slo_attainment_2x"]
                    - baseline["ttft_slo_attainment_2x"]
                ),
            }

    return {
        "schema_version": 1,
        "complete": bool(payload.get("complete", True)),
        "experiment": experiment,
        "n_layers": LAYER_COUNT,
        "ssu_list": list(SSU_LIST),
        "strategies": list(STRATEGIES),
        "metric_scope": {
            "mean_npu_utilization": (
                "mean across 128 NPUs of compute overlap divided by the same "
                "fixed steady-state measurement-window duration"
            ),
            "ttft_slo_attainment_2x": (
                "equal-weight mean across NPUs of admitted measurement requests "
                "whose completion-minus-admission is at most 2x compute-only TTFT"
            ),
            "request_weighted_slo_attainment_2x": "diagnostic only",
        },
        "metrics": metrics,
        "strategy_minus_baseline": deltas,
        "validation": {
            "complete_selected_grid": True,
            "paired_workload_placement_trace_by_ssu": True,
            "all_invariants_passed": True,
            "result_rows": len(rows),
            "paired_hashes_by_ssu": paired_hashes,
        },
    }


def _percent(value):
    return f"{100.0 * value:.2f}%"


def _delta_pp(value):
    return f"{value:+.2f} pp"


def write_plot(path, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), sharex=True)
    specifications = (
        ("mean_npu_utilization", "Mean NPU utilization (%)"),
        ("ttft_slo_attainment_2x", "TTFT processing SLO attainment @ 2x (%)"),
    )
    for axis, (field, ylabel) in zip(axes, specifications):
        for strategy in STRATEGIES:
            axis.plot(
                SSU_LIST,
                [
                    100.0 * analysis["metrics"][strategy][str(ssu)][field]
                    for ssu in SSU_LIST
                ],
                label=LABELS[strategy],
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linewidth=2.2,
                markersize=6.5,
            )
        axis.set_xlabel("Number of SSUs")
        axis.set_ylabel(ylabel)
        axis.set_xticks(SSU_LIST)
        axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].legend(loc="lower right", frameon=False)
    figure.suptitle("Steady-state full load, 16 layers")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(path, analysis):
    lines = [
        "# Steady-state full-load results",
        "",
        "Configuration: 128 NPUs, 16 layers, SSU = `16/24/40/70`. "
        "Only Baseline, Read once per layer, and Scheme B are included.",
        "",
        "Mean NPU utilization uses the fixed steady-state measurement window. "
        "TTFT SLO is `completion - admission <= 2 × compute-only TTFT`; external "
        "arrival-to-admission queue wait is excluded. The SLO shown below is "
        "equal-weighted across NPUs, not request-weighted.",
        "",
        "For every SSU count, all three strategies have exactly matching "
        "`workload_hash`, `placement_hash`, and `trace_hash`. All recorded "
        "simulation invariants passed.",
        "",
        "## Mean NPU utilization",
        "",
        "| Strategy | " + " | ".join(f"SSU {ssu}" for ssu in SSU_LIST) + " |",
        "|---|" + "---:|" * len(SSU_LIST),
    ]
    for strategy in STRATEGIES:
        lines.append(
            f"| {LABELS[strategy]} | "
            + " | ".join(
                _percent(
                    analysis["metrics"][strategy][str(ssu)][
                        "mean_npu_utilization"
                    ]
                )
                for ssu in SSU_LIST
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## TTFT processing SLO attainment @ 2x",
            "",
            "| Strategy | "
            + " | ".join(f"SSU {ssu}" for ssu in SSU_LIST)
            + " |",
            "|---|" + "---:|" * len(SSU_LIST),
        ]
    )
    for strategy in STRATEGIES:
        lines.append(
            f"| {LABELS[strategy]} | "
            + " | ".join(
                _percent(
                    analysis["metrics"][strategy][str(ssu)][
                        "ttft_slo_attainment_2x"
                    ]
                )
                for ssu in SSU_LIST
            )
            + " |"
        )

    for strategy in STRATEGIES[1:]:
        lines.extend(
            [
                "",
                f"## {LABELS[strategy]} minus Baseline",
                "",
                "| Metric | "
                + " | ".join(f"SSU {ssu}" for ssu in SSU_LIST)
                + " |",
                "|---|" + "---:|" * len(SSU_LIST),
                "| Mean NPU utilization | "
                + " | ".join(
                    _delta_pp(
                        analysis["strategy_minus_baseline"][strategy][str(ssu)][
                            "mean_npu_utilization_delta_pp"
                        ]
                    )
                    for ssu in SSU_LIST
                )
                + " |",
                "| TTFT SLO @ 2x | "
                + " | ".join(
                    _delta_pp(
                        analysis["strategy_minus_baseline"][strategy][str(ssu)][
                            "ttft_slo_attainment_2x_delta_pp"
                        ]
                    )
                    for ssu in SSU_LIST
                )
                + " |",
            ]
        )

    lines.extend(
        [
            "",
            "## Pairing validation",
            "",
            "| SSU | workload_hash | placement_hash | trace_hash |",
            "|---:|---|---|---|",
        ]
    )
    for num_ssu in SSU_LIST:
        hashes = analysis["validation"]["paired_hashes_by_ssu"][str(num_ssu)]
        lines.append(
            f"| {num_ssu} | `{hashes['workload_hash']}` | "
            f"`{hashes['placement_hash']}` | `{hashes['trace_hash']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "analysis.json"
    report_path = args.output_dir / "report.md"
    plot_path = args.output_dir / PLOT_PATH_NAME
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, analysis)
    write_plot(plot_path, analysis)
    print(f"Wrote {analysis_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
