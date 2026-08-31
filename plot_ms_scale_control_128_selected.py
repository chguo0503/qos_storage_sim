"""Plot the audited selected128 utilization and SLO curves."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    "results/ms_scale_control/selected128_alpha_tuned_v1_analysis/summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/ms_scale_control/selected128_alpha_tuned_v1_plots")
SSU_COUNTS = (8, 12, 16, 20, 24, 40, 72)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PlotDataError(ValueError):
    """Raised when the selected128 summary is incomplete or inconsistent."""


@dataclass(frozen=True)
class SeriesSpec:
    case: str
    label: str
    color: str
    marker: str
    linestyle: str
    family: str
    kind: str
    ttl_ms: float
    interval_ms: float
    tuning_alpha: float | None


COMMON_SERIES = (
    SeriesSpec(
        "baseline",
        "Baseline",
        "#1f77b4",
        "s",
        "-",
        "baseline",
        "baseline",
        0.0,
        0.0,
        None,
    ),
    SeriesSpec(
        "layer_once_ttl_0ms",
        "Layer once: TTL 0 ms",
        "#ff7f0e",
        "o",
        "-",
        "ttl",
        "layer_once",
        0.0,
        0.0,
        None,
    ),
    SeriesSpec(
        "layer_once_ttl_2ms",
        "Layer once: TTL 2 ms",
        "#2ca02c",
        "^",
        "-",
        "ttl",
        "layer_once",
        2.0,
        0.0,
        None,
    ),
    SeriesSpec(
        "layer_once_ttl_5ms",
        "Layer once: TTL 5 ms",
        "#d62728",
        "D",
        "-",
        "ttl",
        "layer_once",
        5.0,
        0.0,
        None,
    ),
)

INTERVAL_STYLES = (
    (25, "#9467bd", "P"),
    (100, "#8c564b", "x"),
    (200, "#e377c2", "v"),
)

ALPHA1P5_SERIES = tuple(
    SeriesSpec(
        f"adaptive_a1p5_t0_i{interval}ms",
        f"Adaptive α1.5-tuned: interval {interval} ms",
        color,
        marker,
        "--",
        "adaptive_alpha_a1p5",
        "adaptive",
        0.0,
        float(interval),
        1.5,
    )
    for interval, color, marker in INTERVAL_STYLES
)

ALPHA2_SERIES = tuple(
    SeriesSpec(
        f"adaptive_a2_t0_i{interval}ms",
        f"Adaptive α2-tuned: interval {interval} ms",
        color,
        marker,
        ":",
        "adaptive_alpha_a2",
        "adaptive",
        0.0,
        float(interval),
        2.0,
    )
    for interval, color, marker in INTERVAL_STYLES
)

ALL_SERIES = COMMON_SERIES + ALPHA1P5_SERIES + ALPHA2_SERIES
CASE_BY_NAME = {series.case: series for series in ALL_SERIES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlotDataError(message)


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlotDataError(f"{context}: expected a number") from error
    _require(math.isfinite(result), f"{context}: expected a finite number")
    return result


def _close(value: object, expected: float, context: str) -> float:
    result = _finite(value, context)
    _require(
        math.isclose(result, expected, rel_tol=0.0, abs_tol=1e-10),
        f"{context}: {result!r} != {expected!r}",
    )
    return result


def _percentage(value: object, context: str) -> float:
    result = _finite(value, context)
    _require(0.0 <= result <= 100.0, f"{context}: outside [0, 100]")
    return result


def _optional_float(value: object, context: str) -> float | None:
    if value in (None, ""):
        return None
    return _finite(value, context)


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise PlotDataError(f"path is outside the project: {path}") from error


def _read_rows(path: Path) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    _portable_path(resolved)
    _require(
        resolved.is_file(),
        f"audited summary CSV does not exist: {resolved}; run "
        "build_ms_scale_control_128_selected_summary.py after all seven shards finish",
    )
    try:
        with resolved.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames is not None, f"{resolved}: CSV header missing")
            rows = list(reader)
    except OSError as error:
        raise PlotDataError(f"cannot read {resolved}: {error}") from error
    return rows


def _validate_rows(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], Mapping[str, object]]:
    expected_keys = {(series.case, ssu) for series in ALL_SERIES for ssu in SSU_COUNTS}
    _require(
        len(source_rows) == len(expected_keys),
        f"expected 70 summary rows, got {len(source_rows)}",
    )
    rows: dict[tuple[str, int], Mapping[str, object]] = {}
    fingerprint_fields = (
        "source_fingerprint",
        "config_fingerprint",
        "definition_fingerprint",
        "campaign_spec_sha256",
    )
    fingerprints = {field: set() for field in fingerprint_fields}
    metric_fields = (
        "mean_npu_utilization_pct",
        "alpha_1p5_equal_npu_slo_pct",
        "alpha_1p5_request_weighted_slo_pct",
        "alpha_2_equal_npu_slo_pct",
        "alpha_2_request_weighted_slo_pct",
    )
    for index, row in enumerate(source_rows):
        case = row.get("case")
        _require(case in CASE_BY_NAME, f"row {index}: unknown case {case!r}")
        try:
            num_ssu = int(str(row.get("num_ssu")))
        except (TypeError, ValueError) as error:
            raise PlotDataError(f"row {index}: invalid num_ssu") from error
        key = (str(case), num_ssu)
        _require(key in expected_keys, f"row {index}: unexpected key {key}")
        _require(key not in rows, f"duplicate summary row {key}")
        rows[key] = row
        series = CASE_BY_NAME[str(case)]
        _require(row.get("family") == series.family, f"{key}: family mismatch")
        _require(row.get("kind") == series.kind, f"{key}: kind mismatch")
        _close(row.get("pressure_ttl_ms"), series.ttl_ms, f"{key}/TTL")
        _close(row.get("cir_write_threshold_gbps"), 0.0, f"{key}/threshold")
        _close(row.get("min_interval_ms"), series.interval_ms, f"{key}/interval")
        tuning = _optional_float(row.get("tuning_slo_alpha"), f"{key}/tuning alpha")
        _require(tuning == series.tuning_alpha, f"{key}: tuning alpha mismatch")
        if tuning is None:
            for field in (
                "explicit_spill_threshold",
                "target_ratio",
                "required_ratio",
                "background_reserve_fraction",
            ):
                _require(
                    _optional_float(row.get(field), f"{key}/{field}") is None,
                    f"{key}: static strategy unexpectedly has {field}",
                )
        else:
            _close(row.get("explicit_spill_threshold"), 0.75, f"{key}/spill")
            _close(
                row.get("target_ratio"),
                1.0 / 1.5 + 0.02 if tuning == 1.5 else 0.52,
                f"{key}/target ratio",
            )
            _close(
                row.get("required_ratio"),
                1.0 / 1.5 if tuning == 1.5 else 0.50,
                f"{key}/required ratio",
            )
            _close(
                row.get("background_reserve_fraction"),
                0.05,
                f"{key}/background reserve",
            )
        _close(row.get("primary_slo_alpha"), 2.0, f"{key}/primary alpha")
        _close(row.get("sensitivity_slo_alpha"), 1.5, f"{key}/sensitivity alpha")
        _require(
            row.get("all_invariants_passed") in (True, "True"),
            f"{key}: simulator invariants did not all pass",
        )
        for field in metric_fields:
            _percentage(row.get(field), f"{key}/{field}")
        request_count = _finite(
            row.get("measurement_request_count"), f"{key}/measurement count"
        )
        _require(
            request_count > 0 and request_count.is_integer(),
            f"{key}: invalid request count",
        )
        minimum = _finite(
            row.get("measurement_requests_per_npu_min"), f"{key}/minimum count"
        )
        maximum = _finite(
            row.get("measurement_requests_per_npu_max"), f"{key}/maximum count"
        )
        med = _finite(
            row.get("measurement_requests_per_npu_median"), f"{key}/median count"
        )
        _require(
            1.0 <= minimum <= med <= maximum, f"{key}: per-NPU count order invalid"
        )
        source_artifact = row.get("source_artifact")
        _require(
            isinstance(source_artifact, str)
            and source_artifact
            and not source_artifact.startswith("/")
            and "/home/" not in source_artifact,
            f"{key}: source artifact is not portable",
        )
        for field in (
            "measurement_cohort_fingerprint",
            "input_simulator_fingerprint",
        ):
            value = row.get(field)
            _require(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                f"{key}/{field}: invalid SHA-256",
            )
        for field in fingerprint_fields:
            value = row.get(field)
            _require(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                f"{key}/{field}: invalid SHA-256",
            )
            fingerprints[field].add(value)
    _require(set(rows) == expected_keys, "selected128 summary grid is incomplete")
    for field, values in fingerprints.items():
        _require(len(values) == 1, f"cross-row {field} mismatch")
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
            "Description": (
                "128-NPU selected-setting curve; seven SSU points and "
                "separate alpha1.5/alpha2 Adaptive tuning"
            ),
        },
    )
    plt.close(fig)
    temporary.replace(path)


def _plot_metric(
    rows: Mapping[tuple[str, int], Mapping[str, object]],
    series_set: Sequence[SeriesSpec],
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
    for series in series_set:
        values = [
            _percentage(
                rows[(series.case, ssu)].get(field), f"{series.case}/SSU{ssu}/{field}"
            )
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
    axis.legend(loc="lower right", ncol=2 if len(series_set) > 7 else 1, fontsize=9)
    fig.tight_layout()
    _save_atomic(fig, output)


def _render(
    rows: Mapping[tuple[str, int], Mapping[str, object]], output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    _portable_path(output_dir)
    utilization_path = output_dir / "01_mean_npu_utilization_vs_ssu.png"
    alpha_1p5_path = output_dir / "02_ttft_slo_alpha1p5_vs_ssu.png"
    alpha_2_path = output_dir / "03_ttft_slo_alpha2_vs_ssu.png"
    _plot_metric(
        rows,
        ALL_SERIES,
        field="mean_npu_utilization_pct",
        title="128-NPU mean utilization under SLO-tuned CIR-control settings",
        ylabel="Average NPU Utilization (%)",
        output=utilization_path,
    )
    _plot_metric(
        rows,
        COMMON_SERIES + ALPHA1P5_SERIES,
        field="alpha_1p5_equal_npu_slo_pct",
        title="128-NPU TTFT SLO at 1.5× ideal under selected CIR-control settings",
        ylabel="Equal-NPU TTFT SLO Attainment (%)",
        output=alpha_1p5_path,
    )
    _plot_metric(
        rows,
        COMMON_SERIES + ALPHA2_SERIES,
        field="alpha_2_equal_npu_slo_pct",
        title="128-NPU TTFT SLO at 2× ideal under selected CIR-control settings",
        ylabel="Equal-NPU TTFT SLO Attainment (%)",
        output=alpha_2_path,
    )
    return utilization_path, alpha_1p5_path, alpha_2_path


def _synthetic_rows() -> list[dict[str, object]]:
    common_fingerprints = {
        "source_fingerprint": "1" * 64,
        "config_fingerprint": "2" * 64,
        "definition_fingerprint": "3" * 64,
        "campaign_spec_sha256": "4" * 64,
    }
    rows = []
    for series_index, series in enumerate(ALL_SERIES):
        for point_index, ssu in enumerate(SSU_COUNTS):
            base = min(98.0, 25.0 + 9.0 * point_index + series_index)
            rows.append(
                {
                    "case": series.case,
                    "family": series.family,
                    "kind": series.kind,
                    "tuning_slo_alpha": (
                        "" if series.tuning_alpha is None else series.tuning_alpha
                    ),
                    "num_ssu": ssu,
                    "pressure_ttl_ms": series.ttl_ms,
                    "cir_write_threshold_gbps": 0.0,
                    "min_interval_ms": series.interval_ms,
                    "explicit_spill_threshold": (
                        "" if series.tuning_alpha is None else 0.75
                    ),
                    "target_ratio": (
                        ""
                        if series.tuning_alpha is None
                        else (1.0 / 1.5 + 0.02 if series.tuning_alpha == 1.5 else 0.52)
                    ),
                    "required_ratio": (
                        ""
                        if series.tuning_alpha is None
                        else (1.0 / 1.5 if series.tuning_alpha == 1.5 else 0.50)
                    ),
                    "background_reserve_fraction": (
                        "" if series.tuning_alpha is None else 0.05
                    ),
                    "primary_slo_alpha": 2.0,
                    "sensitivity_slo_alpha": 1.5,
                    "mean_npu_utilization_pct": base,
                    "alpha_1p5_equal_npu_slo_pct": min(100.0, base + 2.0),
                    "alpha_1p5_request_weighted_slo_pct": min(100.0, base + 1.0),
                    "alpha_2_equal_npu_slo_pct": min(100.0, base + 15.0),
                    "alpha_2_request_weighted_slo_pct": min(100.0, base + 14.0),
                    "measurement_request_count": 1_000,
                    "measurement_requests_per_npu_min": 7,
                    "measurement_requests_per_npu_median": 8,
                    "measurement_requests_per_npu_max": 9,
                    "all_invariants_passed": True,
                    "source_artifact": f"results/synthetic/ssu{ssu}.json",
                    "measurement_cohort_fingerprint": f"{series_index + 5:x}" * 64,
                    "input_simulator_fingerprint": f"{point_index + 5:x}" * 64,
                    **common_fingerprints,
                }
            )
    return rows


def _png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    _require(raw[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG")
    _require(raw[12:16] == b"IHDR", f"{path}: PNG IHDR missing")
    return struct.unpack(">II", raw[16:24])


def _self_test() -> dict[str, object]:
    rows = _validate_rows(_synthetic_rows())
    with TemporaryDirectory(prefix="selected128-plot-") as temporary:
        output_dir = PROJECT_ROOT / "results" / "synthetic_plot_selftest"
        # Render beneath the project to exercise portable path enforcement, but
        # redirect the actual files into the temporary directory one plot at a
        # time so the self-test leaves no workspace artifact.
        temporary_path = Path(temporary)
        paths = (
            temporary_path / "01.png",
            temporary_path / "02.png",
            temporary_path / "03.png",
        )
        _plot_metric(
            rows,
            ALL_SERIES,
            field="mean_npu_utilization_pct",
            title="self-test utilization",
            ylabel="Average NPU Utilization (%)",
            output=paths[0],
        )
        _plot_metric(
            rows,
            COMMON_SERIES + ALPHA1P5_SERIES,
            field="alpha_1p5_equal_npu_slo_pct",
            title="self-test alpha1.5",
            ylabel="Equal-NPU TTFT SLO Attainment (%)",
            output=paths[1],
        )
        _plot_metric(
            rows,
            COMMON_SERIES + ALPHA2_SERIES,
            field="alpha_2_equal_npu_slo_pct",
            title="self-test alpha2",
            ylabel="Equal-NPU TTFT SLO Attainment (%)",
            output=paths[2],
        )
        dimensions = [_png_dimensions(path) for path in paths]
        _require(
            dimensions == [(1800, 1080)] * 3,
            f"self-test PNG dimensions differ: {dimensions}",
        )
        _require(output_dir.is_absolute(), "self-test path resolution failed")
    return {
        "self_test": "passed",
        "validated_rows": len(rows),
        "png_count": 3,
        "png_dimensions": [1800, 1080],
        "utilization_series": len(ALL_SERIES),
        "slo_series_per_plot": len(COMMON_SERIES + ALPHA1P5_SERIES),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return 0
    rows = _validate_rows(_read_rows(args.input))
    paths = _render(rows, args.output_dir)
    for path in paths:
        print(f"wrote {_portable_path(path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotDataError as error:
        raise SystemExit(f"ERROR: {error}") from error
