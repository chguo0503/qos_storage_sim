"""Analyze paired 32-NPU warm/full-load Scheme-B controller results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "steady_state_32npu_normalized_slo" / "results.json"
DEFAULT_ANALYSIS = DEFAULT_INPUT.with_name("analysis.json")
DEFAULT_REPORT = DEFAULT_INPUT.with_name("report.md")
PRIMARY = ("baseline", "current_scheme_b", "new_scheme_b")
CATEGORIES = ("SS", "SL", "LS", "LL")


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _request_metrics(request_rows):
    by_category = {category: [] for category in CATEGORIES}
    by_npu = {}
    for row in request_rows:
        by_category[row["category"]].append(row)
        by_npu.setdefault(int(row["npu_id"]), []).append(row)
    category = {}
    for name, rows in by_category.items():
        normalized = [row["ttft_ms"] / row["ideal_ttft_ms"] for row in rows]
        category[name] = {
            "count": len(rows),
            "slo_attainment": (
                sum(bool(row["slo_met"]) for row in rows) / len(rows) if rows else None
            ),
            "mean_normalized_ttft": statistics.fmean(normalized) if rows else None,
            "p99_normalized_ttft": _percentile(normalized, 99),
        }
    populated_category_slos = [
        value["slo_attainment"]
        for value in category.values()
        if value["slo_attainment"] is not None
    ]
    per_npu_slos = [
        sum(bool(row["slo_met"]) for row in rows) / len(rows)
        for rows in by_npu.values()
        if rows
    ]
    normalized = [row["ttft_ms"] / row["ideal_ttft_ms"] for row in request_rows]
    return {
        "request_count": len(request_rows),
        "npu_count": len(by_npu),
        "request_weighted_slo_attainment": (
            sum(bool(row["slo_met"]) for row in request_rows) / len(request_rows)
            if request_rows
            else None
        ),
        "equal_npu_slo_attainment": (
            statistics.fmean(per_npu_slos) if per_npu_slos else None
        ),
        "equal_category_slo_attainment": (
            statistics.fmean(populated_category_slos)
            if populated_category_slos
            else None
        ),
        "mean_normalized_ttft": statistics.fmean(normalized) if normalized else None,
        "p99_normalized_ttft": _percentile(normalized, 99),
        "category": category,
    }


def analyze(payload):
    rows = {(row["strategy"], int(row["num_ssu"])): row for row in payload["results"]}
    ssu_list = tuple(int(value) for value in payload["experiment"]["ssu_list"])
    missing = [
        (strategy, num_ssu)
        for num_ssu in ssu_list
        for strategy in PRIMARY
        if (strategy, num_ssu) not in rows
    ]
    if missing:
        raise ValueError(f"primary result matrix is incomplete: {missing}")

    result_rows = []
    intersections = {}
    for num_ssu in ssu_list:
        group = [rows[(strategy, num_ssu)] for strategy in PRIMARY]
        for field in (
            "assignment_hash",
            "workload_hash",
            "placement_hash",
            "trace_hash",
            "simulator_input_fingerprint",
        ):
            if len({row[field] for row in group}) != 1:
                raise ValueError(f"unpaired {field} at SSU={num_ssu}")
        request_maps = {
            row["strategy"]: {
                int(request["request_id"]): request
                for request in row["steady_summary"]["request_rows"]
            }
            for row in group
        }
        common_ids = set.intersection(
            *(set(request_map) for request_map in request_maps.values())
        )
        intersections[str(num_ssu)] = {
            "request_count": len(common_ids),
            "request_ids": sorted(common_ids),
            "strategies": {
                strategy: _request_metrics(
                    [request_maps[strategy][request_id] for request_id in common_ids]
                )
                for strategy in PRIMARY
            },
        }

        for row in group:
            summary = row["steady_summary"]
            request_metrics = _request_metrics(summary["request_rows"])
            npu_utils = summary["npu_utilizations"]
            blocks = summary["measurement_blocks"]
            result_rows.append(
                {
                    "strategy": row["strategy"],
                    "num_ssu": num_ssu,
                    "mean_npu_utilization": summary["mean_npu_utilization"],
                    "p10_npu_utilization": _percentile(npu_utils, 10),
                    "min_npu_utilization": min(npu_utils),
                    "ttft_slo_attainment": summary["ttft_slo_attainment"],
                    "request_weighted_slo_attainment": summary[
                        "request_weighted_slo_attainment"
                    ],
                    "equal_category_slo_attainment": request_metrics[
                        "equal_category_slo_attainment"
                    ],
                    "mean_normalized_ttft": request_metrics["mean_normalized_ttft"],
                    "p99_normalized_ttft": request_metrics["p99_normalized_ttft"],
                    "category": request_metrics["category"],
                    "measurement_request_count": summary["measurement_request_count"],
                    "admissions_per_second": summary["measurement_request_count"]
                    * 1000.0
                    / summary["measurement_duration_ms"],
                    "ssd_mean_utilization": summary["measurement_ssd_mean_utilization"],
                    "ssd_utilization_min": min(summary["measurement_ssd_utilizations"]),
                    "ssd_utilization_max": max(summary["measurement_ssd_utilizations"]),
                    "npu_link_mean_utilization": summary[
                        "measurement_npu_link_mean_utilization"
                    ],
                    "block_npu_utilization_range": max(
                        block["npu_utilization"] for block in blocks
                    )
                    - min(block["npu_utilization"] for block in blocks),
                    "control_evaluations": summary["control_evaluations"],
                    "cir_commits": summary["cir_commits"],
                    "cir_path_writes": summary["cir_path_writes"],
                    "wall_time_s": row["wall_time_s"],
                }
            )

    by_key = {(row["strategy"], row["num_ssu"]): row for row in result_rows}
    deltas = {}
    for num_ssu in ssu_list:
        new = by_key[("new_scheme_b", num_ssu)]
        deltas[str(num_ssu)] = {}
        for reference in ("baseline", "current_scheme_b"):
            old = by_key[(reference, num_ssu)]
            deltas[str(num_ssu)][f"new_minus_{reference}"] = {
                "mean_npu_utilization_pp": 100.0
                * (new["mean_npu_utilization"] - old["mean_npu_utilization"]),
                "ttft_slo_attainment_pp": 100.0
                * (new["ttft_slo_attainment"] - old["ttft_slo_attainment"]),
                "equal_category_slo_pp": 100.0
                * (
                    new["equal_category_slo_attainment"]
                    - old["equal_category_slo_attainment"]
                ),
                "admissions_per_second_percent": 100.0
                * (new["admissions_per_second"] / old["admissions_per_second"] - 1.0),
            }
    return {
        "schema_version": 1,
        "source": str(DEFAULT_INPUT),
        "primary_strategies": list(PRIMARY),
        "ssu_list": list(ssu_list),
        "experiment": payload["experiment"],
        "rows": result_rows,
        "paired_request_intersections": intersections,
        "deltas": deltas,
    }


def render_markdown(analysis):
    lines = [
        "# 32-NPU warm/full-load Scheme-B analysis",
        "",
        "All utilization values use each strategy's fixed 2,000-ms steady window. "
        "The paired-request section additionally restricts TTFT to request IDs "
        "that occur inside all three strategies' measurement cohorts.",
        "",
        "| SSU | Strategy | NPU util | TTFT SLO | Category-balanced SLO | "
        "Admissions/s | SSD util | NPU-link util | Control evals |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["rows"]:
        lines.append(
            "| {num_ssu} | {strategy} | {util:.2%} | {slo:.2%} | "
            "{cat:.2%} | {rate:.1f} | {ssd:.2%} | {link:.2%} | {control} |".format(
                num_ssu=row["num_ssu"],
                strategy=row["strategy"],
                util=row["mean_npu_utilization"],
                slo=row["ttft_slo_attainment"],
                cat=row["equal_category_slo_attainment"],
                rate=row["admissions_per_second"],
                ssd=row["ssd_mean_utilization"],
                link=row["npu_link_mean_utilization"],
                control=row["control_evaluations"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired-request TTFT cohort",
            "",
            "| SSU | Common requests | Strategy | Equal-NPU SLO | "
            "Category-balanced SLO | Mean normalized TTFT |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for num_ssu in analysis["ssu_list"]:
        group = analysis["paired_request_intersections"][str(num_ssu)]
        for strategy in PRIMARY:
            metrics = group["strategies"][strategy]
            lines.append(
                "| {ssu} | {count} | {strategy} | {npu:.2%} | {cat:.2%} | "
                "{ttft:.3f} |".format(
                    ssu=num_ssu,
                    count=group["request_count"],
                    strategy=strategy,
                    npu=metrics["equal_npu_slo_attainment"],
                    cat=metrics["equal_category_slo_attainment"],
                    ttft=metrics["mean_normalized_ttft"],
                )
            )
    lines.extend(
        [
            "",
            "## New Scheme B deltas",
            "",
            "| SSU | Reference | NPU util | TTFT SLO | Category-balanced SLO | "
            "Admissions/s |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in analysis["ssu_list"]:
        for key, delta in analysis["deltas"][str(num_ssu)].items():
            lines.append(
                "| {ssu} | {ref} | {util:+.2f} pp | {slo:+.2f} pp | "
                "{cat:+.2f} pp | {rate:+.2f}% |".format(
                    ssu=num_ssu,
                    ref=key.removeprefix("new_minus_"),
                    util=delta["mean_npu_utilization_pp"],
                    slo=delta["ttft_slo_attainment_pp"],
                    cat=delta["equal_category_slo_pp"],
                    rate=delta["admissions_per_second_percent"],
                )
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    analysis = analyze(json.loads(args.input.read_text()))
    args.analysis.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    args.report.write_text(render_markdown(analysis))
    print(f"wrote {args.analysis}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
