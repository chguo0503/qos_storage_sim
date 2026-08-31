"""Validate and plot the paired random-catalog 128-NPU SSU curve.

This is deliberately a post-processing-only tool.  It does not import or
modify an experiment runner.  The two artifacts may have been produced by
different runner revisions, but they must describe the same immutable request
schedule.  Strategy rows are paired only through their input fingerprints at
the same SSU topology; measurement-window request cohorts are not treated as
request-by-request pairs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import sim
from random_steady_state_workload import build_steady_state_profile_schedule


SCHEMA_VERSION = 1
EXPECTED_STRATEGIES = (
    "baseline",
    "layer_once",
    "adaptive_v2_1_25ms",
)
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "layer_once": "Once per layer",
    "adaptive_v2_1_25ms": "Adaptive V2.1 (25 ms)",
}
STRATEGY_COLORS = {
    "baseline": "#64748B",
    "layer_once": "#0284C7",
    "adaptive_v2_1_25ms": "#EA580C",
}
STRATEGY_MARKERS = {
    "baseline": "o",
    "layer_once": "s",
    "adaptive_v2_1_25ms": "D",
}

INPUT_FINGERPRINT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "workload",
    "placement",
    "trace",
    "simulator",
)
CROSS_TOPOLOGY_FINGERPRINT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
)

# Frozen full-64-request schedule statistic.  The plot labels this as the
# long-run raw-demand knee, not as a measurement-window cohort demand.
EXPECTED_RAW_KNEE_SSU = 22.465191487464285

# Given optimistic NPU-link SLO boundary.  For each strategy row the ceiling is
# computed from that row's own measurement cohort; cohorts are never intersected
# or compared request by request across strategies.
LINK_SLO_DEMAND_THRESHOLD_GBPS = 103.333

OUTPUT_CSV = "random_ssu_curve.csv"
OUTPUT_JSON = "random_ssu_curve.json"
OUTPUT_PNG = "random_ssu_curve.png"


class ValidationError(ValueError):
    """An input artifact is not safe to include in the aggregate curve."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    payload: Mapping[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _finite_number(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite, got {result!r}")
    return result


def _fraction(value, name: str) -> float:
    result = _finite_number(value, name)
    _require(-1e-12 <= result <= 1.0 + 1e-12, f"{name} is outside [0, 1]")
    return min(1.0, max(0.0, result))


def _close(actual: float, expected: float, name: str, *, tol=1e-9) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol),
        f"{name} mismatch: actual={actual!r}, expected={expected!r}",
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_artifact(path: Path) -> Artifact:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValidationError(f"cannot read {resolved}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {resolved}: {error}") from error
    _require(isinstance(payload, dict), f"{resolved} must contain a JSON object")
    _require(
        payload.get("selected_complete") is True,
        f"{resolved}: selected experiment cases are incomplete",
    )
    _require(
        payload.get("source_stable_during_run") is True,
        f"{resolved}: source was not stable during the recorded run",
    )
    _require(
        payload.get("config_stable_during_run") is True,
        f"{resolved}: config was not stable during the recorded run",
    )
    _require(
        payload.get("source_fingerprint") == payload.get("ending_source_fingerprint"),
        f"{resolved}: start/end source fingerprints differ",
    )
    _require(
        payload.get("config_fingerprint") == payload.get("ending_config_fingerprint"),
        f"{resolved}: start/end config fingerprints differ",
    )
    _require(
        isinstance(payload.get("results"), list) and payload["results"],
        f"{resolved}: results must be a non-empty list",
    )
    return Artifact(resolved, payload)


def _schedule_metadata(artifacts: Sequence[Artifact]) -> dict:
    metadata_rows = []
    for artifact in artifacts:
        metadata = artifact.payload.get("schedule_metadata")
        _require(
            isinstance(metadata, dict),
            f"{artifact.path}: schedule_metadata is missing",
        )
        required = {
            *CROSS_TOPOLOGY_FINGERPRINT_FIELDS,
            "mode",
            "seed",
            "num_npu",
            "requests_per_npu",
        }
        _require(
            required <= set(metadata),
            f"{artifact.path}: incomplete schedule_metadata",
        )
        metadata_rows.append(metadata)
    reference = metadata_rows[0]
    for metadata, artifact in zip(metadata_rows[1:], artifacts[1:]):
        for field in required:
            _require(
                metadata[field] == reference[field],
                f"{artifact.path}: cross-file schedule field {field!r} differs",
            )
    return dict(reference)


def _validate_summary(row: Mapping[str, object], artifact: Path) -> None:
    strategy = str(row.get("strategy"))
    num_ssu = int(row.get("num_ssu", -1))
    prefix = f"{artifact}: {strategy}/SSU{num_ssu}"
    _require(row.get("status") == "ok", f"{prefix}: status is not ok")
    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{prefix}: steady_summary is missing")
    _require(
        summary.get("mode") == "steady_state_full_load",
        f"{prefix}: wrong simulator mode",
    )
    _require(int(summary.get("num_ssu", -1)) == num_ssu, f"{prefix}: SSU mismatch")
    _require(int(summary.get("num_npu", -1)) == 128, f"{prefix}: NPU mismatch")
    invariants = summary.get("invariants")
    _require(isinstance(invariants, dict) and invariants, f"{prefix}: no invariants")
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{prefix}: failed invariants: {failed}")

    request_rows = summary.get("request_rows")
    _require(
        isinstance(request_rows, list) and request_rows, f"{prefix}: no request rows"
    )
    _require(
        int(summary.get("measurement_request_count", -1)) == len(request_rows),
        f"{prefix}: measurement request count does not match request_rows",
    )
    request_ids = [int(request["request_id"]) for request in request_rows]
    _require(
        len(request_ids) == len(set(request_ids)),
        f"{prefix}: duplicate request IDs in measurement cohort",
    )

    by_npu = defaultdict(list)
    for request in request_rows:
        _require(
            isinstance(request.get("slo_met"), bool),
            f"{prefix}: request slo_met is not boolean",
        )
        npu_id = int(request["npu_id"])
        _require(0 <= npu_id < 128, f"{prefix}: invalid request NPU {npu_id}")
        by_npu[npu_id].append(request["slo_met"])
    _require(
        set(by_npu) == set(range(128)),
        f"{prefix}: measurement cohort does not sample every NPU",
    )
    equal_npu = statistics.fmean(statistics.fmean(values) for values in by_npu.values())
    request_weighted = statistics.fmean(request["slo_met"] for request in request_rows)
    _close(
        _fraction(summary.get("ttft_slo_attainment"), f"{prefix} equal-NPU SLO"),
        equal_npu,
        f"{prefix} equal-NPU SLO recomputation",
    )
    _close(
        _fraction(
            summary.get("request_weighted_slo_attainment"),
            f"{prefix} request-weighted SLO",
        ),
        request_weighted,
        f"{prefix} request-weighted SLO recomputation",
    )

    npu_utilizations = summary.get("npu_utilizations")
    _require(
        isinstance(npu_utilizations, list) and len(npu_utilizations) == 128,
        f"{prefix}: npu_utilizations must have 128 entries",
    )
    npu_utilizations = [
        _fraction(value, f"{prefix} NPU utilization") for value in npu_utilizations
    ]
    _close(
        _fraction(summary.get("mean_npu_utilization"), f"{prefix} mean NPU util"),
        statistics.fmean(npu_utilizations),
        f"{prefix} mean NPU utilization recomputation",
    )

    ssd_utilizations = summary.get("measurement_ssd_utilizations")
    _require(
        isinstance(ssd_utilizations, list) and len(ssd_utilizations) == num_ssu,
        f"{prefix}: measurement_ssd_utilizations has the wrong length",
    )
    ssd_utilizations = [
        _fraction(value, f"{prefix} SSD utilization") for value in ssd_utilizations
    ]
    _close(
        _fraction(
            summary.get("measurement_ssd_mean_utilization"),
            f"{prefix} mean SSD util",
        ),
        statistics.fmean(ssd_utilizations),
        f"{prefix} mean SSD utilization recomputation",
    )

    stats = row.get("workload_statistics")
    _require(isinstance(stats, dict), f"{prefix}: workload_statistics is missing")
    raw_knee = _finite_number(stats.get("capacity_knee_ssu"), f"{prefix} raw knee")
    _close(
        raw_knee,
        EXPECTED_RAW_KNEE_SSU,
        f"{prefix} full-schedule raw knee",
        tol=1e-6,
    )


def _collect_rows(artifacts: Sequence[Artifact]) -> dict[tuple[str, int], dict]:
    rows = {}
    for artifact in artifacts:
        for original in artifact.payload["results"]:
            _require(
                isinstance(original, dict), f"{artifact.path}: result is not an object"
            )
            row = dict(original)
            strategy = str(row.get("strategy"))
            num_ssu = int(row.get("num_ssu", -1))
            _require(
                strategy in EXPECTED_STRATEGIES,
                f"{artifact.path}: unexpected strategy {strategy!r}",
            )
            _require(num_ssu > 0, f"{artifact.path}: invalid SSU count {num_ssu}")
            key = strategy, num_ssu
            _require(key not in rows, f"duplicate result row across artifacts: {key}")
            fingerprints = row.get("input_fingerprints")
            _require(
                isinstance(fingerprints, dict)
                and set(fingerprints) == set(INPUT_FINGERPRINT_FIELDS),
                f"{artifact.path}: {strategy}/SSU{num_ssu} has incomplete input fingerprints",
            )
            _require(
                all(
                    isinstance(value, str) and value for value in fingerprints.values()
                ),
                f"{artifact.path}: {strategy}/SSU{num_ssu} has an empty fingerprint",
            )
            _validate_summary(row, artifact.path)
            row["_artifact"] = str(artifact.path)
            rows[key] = row
    return rows


def _validate_pairing(
    rows: Mapping[tuple[str, int], Mapping[str, object]],
    schedule_metadata: Mapping[str, object],
) -> tuple[list[int], dict[str, str], dict[str, str]]:
    ssus = sorted({num_ssu for _, num_ssu in rows})
    _require(len(ssus) >= 2, "the final curve needs at least two SSU points")
    simulator_by_ssu = {}
    for num_ssu in ssus:
        group = {
            strategy: rows[(strategy, num_ssu)]
            for strategy in EXPECTED_STRATEGIES
            if (strategy, num_ssu) in rows
        }
        _require(
            set(group) == set(EXPECTED_STRATEGIES),
            f"SSU{num_ssu} does not contain exactly the three expected strategies",
        )
        for field in INPUT_FINGERPRINT_FIELDS:
            values = {row["input_fingerprints"][field] for row in group.values()}
            _require(
                len(values) == 1,
                f"SSU{num_ssu} is not paired on input fingerprint {field!r}",
            )
        simulator_by_ssu[str(num_ssu)] = next(iter(group.values()))[
            "input_fingerprints"
        ]["simulator"]

    global_fingerprints = {}
    for field in CROSS_TOPOLOGY_FINGERPRINT_FIELDS:
        values = {row["input_fingerprints"][field] for row in rows.values()}
        _require(
            len(values) == 1,
            f"cross-file fingerprint {field!r} differs across topologies",
        )
        value = values.pop()
        _require(
            schedule_metadata[field] == value,
            f"schedule_metadata {field!r} does not match result rows",
        )
        global_fingerprints[field] = value

    # A simulator input fingerprint contains placement, so it is topology
    # specific.  If an SSU appears in both files duplicate-row validation above
    # prevents ambiguous aggregation; same-SSU three-way equality is enforced.
    return ssus, global_fingerprints, simulator_by_ssu


def _rebuild_schedule(
    metadata: Mapping[str, object], global_fingerprints: Mapping[str, str]
):
    num_npu = int(metadata["num_npu"])
    table = sim.load_bw_table_cache(num_npu=num_npu)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=str(metadata["mode"]),
        seed=int(metadata["seed"]),
        num_npu=num_npu,
        requests_per_npu=int(metadata["requests_per_npu"]),
    )
    rebuilt = schedule.as_fingerprint_dict()
    for field in CROSS_TOPOLOGY_FINGERPRINT_FIELDS:
        _require(
            rebuilt[field] == global_fingerprints[field],
            f"rebuilt schedule fingerprint {field!r} differs from artifacts",
        )
    assignments = {
        int(request_id): (
            int(npu_id),
            int(sequence),
            str(category),
            tuple(profile_key),
        )
        for request_id, npu_id, sequence, category, profile_key in schedule.assignments
    }
    _require(
        len(assignments) == len(schedule.assignments),
        "rebuilt schedule contains duplicate request IDs",
    )
    return table, assignments


def _link_ceiling(
    row: Mapping[str, object],
    table: Mapping,
    assignments: Mapping[int, tuple[int, int, str, tuple[int, int]]],
) -> dict[str, float | int]:
    summary = row["steady_summary"]
    request_rows = summary["request_rows"]
    prefix = f"{row['strategy']}/SSU{row['num_ssu']}"
    by_npu = defaultdict(list)
    ineligible_count = 0
    ineligible_slo_met = 0
    for request in request_rows:
        request_id = int(request["request_id"])
        _require(request_id in assignments, f"{prefix}: unknown request {request_id}")
        npu_id, sequence, category, profile_key = assignments[request_id]
        _require(int(request["npu_id"]) == npu_id, f"{prefix}: request NPU mismatch")
        _require(int(request["sequence"]) == sequence, f"{prefix}: sequence mismatch")
        _require(str(request["category"]) == category, f"{prefix}: category mismatch")
        demand = _finite_number(table[profile_key][0], f"{prefix}: profile demand")
        eligible = demand <= LINK_SLO_DEMAND_THRESHOLD_GBPS
        by_npu[npu_id].append(eligible)
        if not eligible:
            ineligible_count += 1
            ineligible_slo_met += int(request["slo_met"])

    _require(
        set(by_npu) == set(range(128)),
        f"{prefix}: link ceiling cannot be equal-NPU weighted",
    )
    _require(
        ineligible_slo_met == 0,
        f"{prefix}: D>{LINK_SLO_DEMAND_THRESHOLD_GBPS} request met SLO; "
        "the configured link bound is not a valid ceiling",
    )
    ceiling_equal = statistics.fmean(
        statistics.fmean(values) for values in by_npu.values()
    )
    ceiling_request = statistics.fmean(
        eligible for values in by_npu.values() for eligible in values
    )
    actual_equal = _fraction(summary["ttft_slo_attainment"], f"{prefix}: SLO")
    actual_request = _fraction(
        summary["request_weighted_slo_attainment"], f"{prefix}: request SLO"
    )
    _require(
        ceiling_equal + 1e-9 >= actual_equal,
        f"{prefix}: equal-NPU actual SLO exceeds optimistic link ceiling",
    )
    _require(
        ceiling_request + 1e-9 >= actual_request,
        f"{prefix}: request SLO exceeds optimistic link ceiling",
    )
    return {
        "link_ineligible_request_count": ineligible_count,
        "link_ineligible_slo_met_count": ineligible_slo_met,
        "link_slo_ceiling_equal_npu_pct": 100.0 * ceiling_equal,
        "actual_to_link_ceiling_equal_npu_gap_pp": 100.0
        * (ceiling_equal - actual_equal),
        "link_slo_ceiling_request_weighted_pct": 100.0 * ceiling_request,
        "actual_to_link_ceiling_request_weighted_gap_pp": 100.0
        * (ceiling_request - actual_request),
    }


def _compact_row(row, table, assignments) -> dict:
    summary = row["steady_summary"]
    npu_utilizations = [float(value) for value in summary["npu_utilizations"]]
    ssd_utilizations = [
        float(value) for value in summary["measurement_ssd_utilizations"]
    ]
    compact = {
        "ssu": int(row["num_ssu"]),
        "strategy": str(row["strategy"]),
        "mean_npu_utilization_pct": 100.0 * float(summary["mean_npu_utilization"]),
        "npu_utilization_min_pct": 100.0 * min(npu_utilizations),
        "npu_utilization_p10_pct": 100.0 * _percentile(npu_utilizations, 10.0),
        "equal_npu_ttft_slo_pct": 100.0 * float(summary["ttft_slo_attainment"]),
        "request_weighted_ttft_slo_pct": 100.0
        * float(summary["request_weighted_slo_attainment"]),
        "measurement_request_count": int(summary["measurement_request_count"]),
        "ssd_mean_utilization_pct": 100.0
        * float(summary["measurement_ssd_mean_utilization"]),
        "ssd_max_utilization_pct": 100.0 * max(ssd_utilizations),
        "wall_time_s": float(row["wall_time_s"]),
        "source_artifact": row["_artifact"],
    }
    compact.update(_link_ceiling(row, table, assignments))
    return compact


CSV_FIELDS = (
    "ssu",
    "strategy",
    "mean_npu_utilization_pct",
    "npu_utilization_min_pct",
    "npu_utilization_p10_pct",
    "equal_npu_ttft_slo_pct",
    "request_weighted_ttft_slo_pct",
    "measurement_request_count",
    "ssd_mean_utilization_pct",
    "ssd_max_utilization_pct",
    "link_ineligible_request_count",
    "link_ineligible_slo_met_count",
    "link_slo_ceiling_equal_npu_pct",
    "actual_to_link_ceiling_equal_npu_gap_pp",
    "link_slo_ceiling_request_weighted_pct",
    "actual_to_link_ceiling_request_weighted_gap_pp",
    "wall_time_s",
    "source_artifact",
)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    _atomic_write_text(path, buffer.getvalue())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _plot(
    path: Path, rows: Sequence[Mapping[str, object]], ssus: Sequence[int]
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), layout="constrained")
    panels = (
        (axes[0], "mean_npu_utilization_pct", "Mean NPU utilization"),
        (axes[1], "equal_npu_ttft_slo_pct", "Equal-NPU TTFT SLO"),
    )
    by_key = {(row["strategy"], int(row["ssu"])): row for row in rows}

    for axis, field, title in panels:
        for strategy in EXPECTED_STRATEGIES:
            values = [float(by_key[(strategy, ssu)][field]) for ssu in ssus]
            axis.plot(
                ssus,
                values,
                color=STRATEGY_COLORS[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=6.5,
                linewidth=2.2,
                label=STRATEGY_LABELS[strategy],
            )
        axis.set_title(title)
        axis.set_xlabel("SSU count (40 GB/s each)")
        axis.set_ylabel("Percent")
        axis.set_ylim(0.0, 101.5)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
        axis.grid(True, axis="y")

    ssd_axis = axes[2]
    for strategy in EXPECTED_STRATEGIES:
        means = [
            float(by_key[(strategy, ssu)]["ssd_mean_utilization_pct"]) for ssu in ssus
        ]
        maxima = [
            float(by_key[(strategy, ssu)]["ssd_max_utilization_pct"]) for ssu in ssus
        ]
        color = STRATEGY_COLORS[strategy]
        ssd_axis.fill_between(ssus, means, maxima, color=color, alpha=0.11)
        ssd_axis.plot(
            ssus,
            means,
            color=color,
            marker=STRATEGY_MARKERS[strategy],
            markersize=6.0,
            linewidth=2.1,
        )
        ssd_axis.plot(ssus, maxima, color=color, linewidth=1.25, linestyle="--")
    ssd_axis.set_title("SSD utilization: mean to hottest SSU")
    ssd_axis.set_xlabel("SSU count (40 GB/s each)")
    ssd_axis.set_ylabel("Percent")
    ssd_axis.set_ylim(0.0, 101.5)
    ssd_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ssd_axis.grid(True, axis="y")
    ssd_axis.text(
        0.02,
        0.03,
        "solid = mean   dashed = max   band = gap",
        transform=ssd_axis.transAxes,
        color="#64748B",
        fontsize=8.5,
    )

    for axis in axes:
        axis.axvline(
            EXPECTED_RAW_KNEE_SSU,
            color="#D97706",
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            alpha=0.95,
            zorder=0,
        )
        axis.set_xticks(ssus)
        axis.set_xlim(min(ssus) - 1.0, max(ssus) + 1.5)
    axes[0].annotate(
        "long-run raw knee\n22.465 SSUs",
        xy=(EXPECTED_RAW_KNEE_SSU, 99.0),
        xytext=(8, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#92400E",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFFBEB", "ec": "#FCD34D"},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        fontsize=10,
    )
    fig.suptitle(
        "128-NPU random-catalog steady-state SSU curve",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.965,
        "Aggregate windows only; strategy-specific measurement cohorts are not request-paired",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#64748B",
    )
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=260,
        facecolor="white",
        metadata={
            "Title": "128-NPU random-catalog steady-state SSU curve",
            "Description": "Aggregate paired-input strategy comparison",
        },
    )
    plt.close(fig)
    temporary.replace(path)


def build_outputs(low_results: Path, ssu40_results: Path, output_dir: Path) -> dict:
    artifacts = (_read_artifact(low_results), _read_artifact(ssu40_results))
    metadata = _schedule_metadata(artifacts)
    rows = _collect_rows(artifacts)
    ssus, global_fingerprints, simulator_by_ssu = _validate_pairing(rows, metadata)
    table, assignments = _rebuild_schedule(metadata, global_fingerprints)
    compact_rows = [
        _compact_row(rows[(strategy, num_ssu)], table, assignments)
        for num_ssu in ssus
        for strategy in EXPECTED_STRATEGIES
    ]

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / OUTPUT_CSV
    json_path = output / OUTPUT_JSON
    png_path = output / OUTPUT_PNG
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": [str(artifact.path) for artifact in artifacts],
        "ssu_points": ssus,
        "strategies": list(EXPECTED_STRATEGIES),
        "global_input_fingerprints": global_fingerprints,
        "simulator_input_fingerprint_by_ssu": simulator_by_ssu,
        "pairing_validation": {
            "same_ssu_fields": list(INPUT_FINGERPRINT_FIELDS),
            "cross_topology_fields": list(CROSS_TOPOLOGY_FINGERPRINT_FIELDS),
            "all_three_strategies_present_per_ssu": True,
            "all_status_ok_and_invariants_true": True,
            "measurement_cohort_semantics": (
                "strategy-specific wall-time admission cohorts; aggregate metrics "
                "only, not request-by-request paired"
            ),
        },
        "long_run_raw_knee_ssu": EXPECTED_RAW_KNEE_SSU,
        "link_slo_ceiling_model": {
            "optimistic_demand_threshold_gbps": LINK_SLO_DEMAND_THRESHOLD_GBPS,
            "ineligible_condition": "required_bw_input_gbps > threshold",
            "ceiling_scope": "computed independently from each row's request_rows",
        },
        "rows": compact_rows,
    }
    _write_csv(csv_path, compact_rows)
    _write_json(json_path, summary)
    _plot(png_path, compact_rows, ssus)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path),
        "ssu_points": ssus,
        "row_count": len(compact_rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-results", type=Path, required=True)
    parser.add_argument("--ssu40-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        result = build_outputs(
            args.low_results,
            args.ssu40_results,
            args.output_dir,
        )
    except ValidationError as error:
        raise SystemExit(f"validation failed: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
