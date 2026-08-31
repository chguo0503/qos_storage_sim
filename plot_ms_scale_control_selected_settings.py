"""Plot selected 32-NPU control settings across the measured SSU counts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_analysis/summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_plots"
)
SSU_COUNTS = (2, 3, 4, 5, 6, 10, 18)


class PlotDataError(ValueError):
    """Raised when the frozen analysis CSV does not contain the expected rows."""


@dataclass(frozen=True)
class SeriesSpec:
    case: str
    label: str
    color: str
    marker: str
    linestyle: str
    expected_ttl_ms: float
    expected_threshold_gbps: float
    expected_interval_ms: float


SERIES = (
    SeriesSpec("baseline", "Baseline", "#1f77b4", "s", "-", 0.0, 0.0, 0.0),
    SeriesSpec(
        "layer_once_ttl_0ms",
        "Layer once: TTL 0 ms",
        "#ff7f0e",
        "o",
        "-",
        0.0,
        0.0,
        0.0,
    ),
    SeriesSpec(
        "layer_once_ttl_2ms",
        "Layer once: TTL 2 ms",
        "#2ca02c",
        "^",
        "-",
        2.0,
        0.0,
        0.0,
    ),
    SeriesSpec(
        "layer_once_ttl_5ms",
        "Layer once: TTL 5 ms",
        "#d62728",
        "D",
        "-",
        5.0,
        0.0,
        0.0,
    ),
    SeriesSpec(
        "adaptive_t0_i25ms",
        "Adaptive: interval 25 ms",
        "#9467bd",
        "P",
        "-",
        0.0,
        0.0,
        25.0,
    ),
    SeriesSpec(
        "adaptive_t0_i100ms",
        "Adaptive: interval 100 ms",
        "#8c564b",
        "x",
        "-",
        0.0,
        0.0,
        100.0,
    ),
    SeriesSpec(
        "adaptive_t0_i200ms",
        "Adaptive: interval 200 ms",
        "#e377c2",
        "v",
        "-",
        0.0,
        0.0,
        200.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _as_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise PlotDataError(f"{context}: invalid {field!r}") from error


def _load_selected_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise PlotDataError(f"cannot read {path}: {error}") from error

    selected_cases = {series.case for series in SERIES}
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in source_rows:
        if row.get("case") not in selected_cases:
            continue
        try:
            key = (row["case"], int(row["num_ssu"]))
        except (KeyError, TypeError, ValueError) as error:
            raise PlotDataError(
                "selected row has an invalid case/num_ssu key"
            ) from error
        if key[1] not in SSU_COUNTS:
            continue
        if key in rows:
            raise PlotDataError(f"duplicate selected row: {key}")
        if row.get("all_invariants_passed") != "True":
            raise PlotDataError(f"{key}: simulator invariants did not all pass")
        if _as_float(row, "primary_slo_alpha", str(key)) != 2.0:
            raise PlotDataError(f"{key}: unexpected primary SLO alpha")
        if _as_float(row, "sensitivity_slo_alpha", str(key)) != 1.5:
            raise PlotDataError(f"{key}: alpha=1.5 sensitivity result is absent")
        rows[key] = row

    expected = {(series.case, ssu) for series in SERIES for ssu in SSU_COUNTS}
    missing = sorted(expected - set(rows))
    if missing:
        raise PlotDataError(f"missing selected rows: {missing}")

    for series in SERIES:
        for ssu in SSU_COUNTS:
            row = rows[(series.case, ssu)]
            actual = (
                _as_float(row, "pressure_ttl_ms", series.case),
                _as_float(row, "cir_write_threshold_gbps", series.case),
                _as_float(row, "min_interval_ms", series.case),
            )
            expected_knobs = (
                series.expected_ttl_ms,
                series.expected_threshold_gbps,
                series.expected_interval_ms,
            )
            if actual != expected_knobs:
                raise PlotDataError(
                    f"{series.case}/SSU{ssu}: knob mismatch "
                    f"{actual!r} != {expected_knobs!r}"
                )
    return rows


def _save_atomic(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=180,
        facecolor="white",
        metadata={
            "Title": path.stem,
            "Description": "32-NPU selected-setting SSU curve, including SSU 2-5",
        },
    )
    plt.close(fig)
    temporary.replace(path)


def _plot_metric(
    rows: dict[tuple[str, int], dict[str, str]],
    *,
    field: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.fancybox": True,
        }
    )
    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    for series in SERIES:
        values = [
            _as_float(rows[(series.case, ssu)], field, f"{series.case}/SSU{ssu}")
            for ssu in SSU_COUNTS
        ]
        axis.plot(
            SSU_COUNTS,
            values,
            label=series.label,
            color=series.color,
            marker=series.marker,
            linestyle=series.linestyle,
            linewidth=2.2,
            markersize=6.0,
        )

    axis.set(
        xlabel="Number of SSUs",
        ylabel=ylabel,
        xticks=SSU_COUNTS,
        yticks=range(0, 101, 20),
        ylim=(0, 100),
        title=title,
    )
    axis.grid(alpha=0.3)
    # Keep the legend clear of the newly added low-SSU points, where the
    # utilization and strict-SLO curves carry most of their separation.
    axis.legend(loc="lower right")
    fig.tight_layout()
    _save_atomic(fig, output)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = _load_selected_rows(input_path)

    utilization_path = output_dir / "01_mean_npu_utilization_vs_ssu.png"
    slo_path = output_dir / "02_ttft_slo_alpha1p5_vs_ssu.png"
    slo_alpha2_path = output_dir / "03_ttft_slo_alpha2_vs_ssu.png"
    _plot_metric(
        rows,
        field="mean_npu_utilization_pct",
        title="32-NPU mean utilization under selected CIR-control settings",
        ylabel="Average NPU Utilization (%)",
        output=utilization_path,
    )
    _plot_metric(
        rows,
        field="alpha_1p5_equal_npu_slo_pct",
        title="32-NPU TTFT SLO at 1.5× ideal under selected CIR-control settings",
        ylabel="Equal-NPU TTFT SLO Attainment (%)",
        output=slo_path,
    )
    _plot_metric(
        rows,
        field="equal_npu_slo_pct",
        title="32-NPU TTFT SLO at 2× ideal under selected CIR-control settings",
        ylabel="Equal-NPU TTFT SLO Attainment (%)",
        output=slo_alpha2_path,
    )
    print(f"wrote {utilization_path}")
    print(f"wrote {slo_path}")
    print(f"wrote {slo_alpha2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
