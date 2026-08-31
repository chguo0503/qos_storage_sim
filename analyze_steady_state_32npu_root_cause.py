"""Reproducible root-cause analysis for the 32-NPU warm experiment.

This analyzer is deliberately independent of ``analyze_steady_state_32npu.py``.
It combines the primary experiment with the admission/oracle/NPU-bypass
diagnostics, keeps scientifically invalid rows out of numeric performance
tables, and reports a request-ID intersection for cohort-safe SLO comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "steady_state_32npu_normalized_slo"
DEFAULT_RESULTS = RESULT_DIR / "results.json"
DEFAULT_DIAGNOSTICS = RESULT_DIR / "diagnostics.json"
DEFAULT_ANALYSIS = RESULT_DIR / "root_cause_analysis.json"
DEFAULT_REPORT = RESULT_DIR / "root_cause_report.md"

CATEGORIES = ("SS", "SL", "LS", "LL")
MAIN_STRATEGIES = ("baseline", "current_scheme_b", "new_scheme_b")
TABLE_STRATEGIES = (*MAIN_STRATEGIES, "admission")
PAIRED_STRATEGIES = ("baseline", "current_scheme_b", "admission")
ADMISSION_SOURCE_NAME = "scheme_b_slo_admission"
PAIR_FIELDS = (
    "assignment_hash",
    "workload_hash",
    "placement_hash",
    "trace_hash",
    "simulator_input_fingerprint",
)
MAIN_SOURCE_FILES = (
    "sim.py",
    "policy_logic.py",
    "strategy_profiles.py",
    "continuous_batch_control.py",
    "continuous_batch_sim.py",
    "continuous_prefill_client.py",
    "continuous_prefill_workload.py",
    "six_request_workload.py",
    "steady_state_workload.py",
    "steady_state_32npu_experiment.py",
    "scheme_b_prefill.py",
    "data",
)
DIAGNOSTIC_SOURCE_FILES = (
    "sim.py",
    "policy_logic.py",
    "continuous_batch_control.py",
    "continuous_batch_sim.py",
    "continuous_prefill_client.py",
    "continuous_prefill_workload.py",
    "six_request_workload.py",
    "steady_state_workload.py",
    "steady_state_32npu_experiment.py",
    "steady_state_32npu_diagnostics.py",
    "slo_admission_scheme_b.py",
    "data",
)


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simulation_source_fingerprint(seed: bytes, names) -> str:
    digest = hashlib.sha256(seed)
    for name in names:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _rows_by_key(payload, source_name):
    result = {}
    for row in payload.get("results", ()):
        key = (str(row["strategy"]), int(row["num_ssu"]))
        if key in result:
            raise ValueError(f"duplicate {source_name} row: {key}")
        result[key] = row
    return result


def _valid_summary(row, label):
    status = row.get("status", "ok")
    if status != "ok":
        raise ValueError(f"invalid row cannot enter numeric analysis: {label}")
    summary = row.get("steady_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"missing steady summary: {label}")
    if summary.get("mode") != "steady_state_full_load":
        raise ValueError(f"wrong simulator mode: {label}")
    invariants = summary.get("invariants", {})
    failed = sorted(name for name, holds in invariants.items() if not holds)
    if not invariants or failed:
        raise ValueError(f"failed steady-state invariants for {label}: {failed}")
    return summary


def _slo_metrics(request_rows):
    rows = tuple(request_rows)
    by_npu = {}
    by_category = {category: [] for category in CATEGORIES}
    for row in rows:
        category = row["category"]
        if category not in by_category:
            raise ValueError(f"unknown workload category: {category}")
        by_npu.setdefault(int(row["npu_id"]), []).append(row)
        by_category[category].append(row)

    per_npu = [
        statistics.fmean(bool(row["slo_met"]) for row in npu_rows)
        for npu_rows in by_npu.values()
        if npu_rows
    ]
    normalized = [
        float(row["ttft_ms"]) / float(row["ideal_ttft_ms"]) for row in rows
    ]
    category_metrics = {}
    for category in CATEGORIES:
        category_rows = by_category[category]
        category_metrics[category] = {
            "request_count": len(category_rows),
            "slo_attainment": (
                statistics.fmean(bool(row["slo_met"]) for row in category_rows)
                if category_rows
                else None
            ),
        }
    return {
        "request_count": len(rows),
        "sampled_npu_count": len(by_npu),
        "equal_npu_slo_attainment": (
            statistics.fmean(per_npu) if per_npu else None
        ),
        "request_weighted_slo_attainment": (
            statistics.fmean(bool(row["slo_met"]) for row in rows)
            if rows
            else None
        ),
        "mean_normalized_ttft": (
            statistics.fmean(normalized) if normalized else None
        ),
        "category_slo": category_metrics,
    }


def _strategy_metrics(strategy, row):
    summary = _valid_summary(row, f"{strategy}/SSU{row['num_ssu']}")
    npu_utilizations = tuple(float(value) for value in summary["npu_utilizations"])
    if not npu_utilizations:
        raise ValueError(f"empty NPU utilization vector: {strategy}")
    slo = _slo_metrics(summary["request_rows"])
    if not math.isclose(
        slo["equal_npu_slo_attainment"],
        float(summary["ttft_slo_attainment"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"equal-NPU SLO mismatch: {strategy}/SSU{row['num_ssu']}")
    if not math.isclose(
        slo["request_weighted_slo_attainment"],
        float(summary["request_weighted_slo_attainment"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"request-weighted SLO mismatch: {strategy}/SSU{row['num_ssu']}"
        )
    return {
        "strategy": strategy,
        "source_strategy": row["strategy"],
        "num_ssu": int(row["num_ssu"]),
        "status": "ok",
        "measurement_request_count": slo["request_count"],
        "measurement_duration_ms": float(summary["measurement_duration_ms"]),
        "admissions_per_second": (
            1000.0 * slo["request_count"] / float(summary["measurement_duration_ms"])
        ),
        "npu_utilization": {
            "min": min(npu_utilizations),
            "p10": _percentile(npu_utilizations, 10),
            "mean": statistics.fmean(npu_utilizations),
        },
        "equal_npu_slo_attainment": slo["equal_npu_slo_attainment"],
        "request_weighted_slo_attainment": slo[
            "request_weighted_slo_attainment"
        ],
        "mean_normalized_ttft": slo["mean_normalized_ttft"],
        "category_slo": slo["category_slo"],
        "ssd_mean_utilization": float(summary["measurement_ssd_mean_utilization"]),
        "npu_link_mean_utilization": float(
            summary["measurement_npu_link_mean_utilization"]
        ),
        "control_evaluations": int(summary["control_evaluations"]),
        "cir_commits": int(summary["cir_commits"]),
    }


def _assert_paired(rows, num_ssu, context):
    for field in PAIR_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"unpaired {field} at SSU={num_ssu} ({context})")


def _pp(new_value, reference_value):
    if new_value is None or reference_value is None:
        return None
    return 100.0 * (float(new_value) - float(reference_value))


def _metric_deltas(new, reference):
    return {
        "mean_npu_utilization_pp": _pp(
            new["npu_utilization"]["mean"],
            reference["npu_utilization"]["mean"],
        ),
        "equal_npu_slo_attainment_pp": _pp(
            new["equal_npu_slo_attainment"],
            reference["equal_npu_slo_attainment"],
        ),
        "request_weighted_slo_attainment_pp": _pp(
            new["request_weighted_slo_attainment"],
            reference["request_weighted_slo_attainment"],
        ),
        "category_slo_pp": {
            category: _pp(
                new["category_slo"][category]["slo_attainment"],
                reference["category_slo"][category]["slo_attainment"],
            )
            for category in CATEGORIES
        },
    }


def _request_map(row):
    summary = _valid_summary(
        row, f"{row['strategy']}/SSU{row['num_ssu']}/paired-cohort"
    )
    result = {}
    for request in summary["request_rows"]:
        request_id = int(request["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate request ID in measurement cohort: {request_id}")
        result[request_id] = request
    return result


def _paired_request_intersection(num_ssu, source_rows):
    request_maps = {
        strategy: _request_map(source_rows[strategy])
        for strategy in PAIRED_STRATEGIES
    }
    common_ids = sorted(
        set.intersection(*(set(rows) for rows in request_maps.values()))
    )
    if not common_ids:
        raise ValueError(f"empty paired request intersection at SSU={num_ssu}")
    for request_id in common_ids:
        identity = {
            (
                int(request_maps[strategy][request_id]["npu_id"]),
                int(request_maps[strategy][request_id]["sequence"]),
                request_maps[strategy][request_id]["category"],
                float(request_maps[strategy][request_id]["ideal_ttft_ms"]),
            )
            for strategy in PAIRED_STRATEGIES
        }
        if len(identity) != 1:
            raise ValueError(
                f"request identity mismatch at SSU={num_ssu}, request={request_id}"
            )
    metrics = {
        strategy: _slo_metrics(
            [request_maps[strategy][request_id] for request_id in common_ids]
        )
        for strategy in PAIRED_STRATEGIES
    }
    return {
        "num_ssu": int(num_ssu),
        "strategies": list(PAIRED_STRATEGIES),
        "original_cohort_sizes": {
            strategy: len(rows) for strategy, rows in request_maps.items()
        },
        "common_request_count": len(common_ids),
        "common_request_ids": common_ids,
        "metrics": metrics,
        "admission_deltas_pp": {
            f"admission_minus_{reference}": _metric_deltas(
                {
                    "npu_utilization": {"mean": None},
                    **metrics["admission"],
                },
                {
                    "npu_utilization": {"mean": None},
                    **metrics[reference],
                },
            )
            for reference in ("baseline", "current_scheme_b")
        },
    }


def _invalid_oracle(row, requests_per_npu):
    if row.get("status") == "ok":
        raise ValueError("valid oracle passed to invalid-row formatter")
    failed_values = row.get("failed_invariants", {})
    failed = sorted(name for name, holds in failed_values.items() if not holds)
    diagnostics = row.get("failure_diagnostics", {})
    counts = [int(value) for value in diagnostics.get("request_counts_by_npu", ())]
    completed = [
        int(value) for value in diagnostics.get("completed_by_npu_at_stop", ())
    ]
    zero_sample_npus = [npu for npu, count in enumerate(counts) if count == 0]
    exhausted_npus = [
        npu for npu, count in enumerate(completed) if count >= requests_per_npu
    ]
    reasons = []
    if "all_npus_sampled_for_slo" in failed:
        reasons.append(
            "all_npus_sampled_for_slo=false: "
            f"zero-sample NPUs={zero_sample_npus}"
        )
    if "no_backlog_exhaustion" in failed:
        reasons.append(
            "no_backlog_exhaustion=false: finite-prefix exhausted NPUs="
            f"{exhausted_npus}"
        )
    for invariant in failed:
        if invariant not in ("all_npus_sampled_for_slo", "no_backlog_exhaustion"):
            reasons.append(f"{invariant}=false")
    return {
        "strategy": row["strategy"],
        "num_ssu": int(row["num_ssu"]),
        "status": row.get("status", "invalid"),
        "included_in_numeric_performance_tables": False,
        "failed_invariants": failed,
        "reasons": reasons,
        "zero_sample_npu_ids": zero_sample_npus,
        "finite_prefix_exhausted_npu_ids": exhausted_npus,
        "failure_diagnostics": diagnostics,
    }


def analyze(
    results_payload,
    diagnostics_payload,
    *,
    input_metadata=None,
    current_source_fingerprints=None,
):
    if not results_payload.get("primary_three_strategy_complete"):
        raise ValueError("primary three-strategy result matrix is incomplete")
    if not diagnostics_payload.get("all_cases_finished"):
        raise ValueError("diagnostic cases have not all finished")

    main_rows = _rows_by_key(results_payload, "primary")
    diagnostic_rows = _rows_by_key(diagnostics_payload, "diagnostic")
    ssu_list = tuple(int(value) for value in results_payload["experiment"]["ssu_list"])
    requests_per_npu = int(
        results_payload["experiment"]["requests_per_npu_prefix"]
    )

    table_rows = []
    paired_intersections = []
    admission_deltas = {}
    for num_ssu in ssu_list:
        source_rows = {
            strategy: main_rows[(strategy, num_ssu)] for strategy in MAIN_STRATEGIES
        }
        source_rows["admission"] = diagnostic_rows[
            (ADMISSION_SOURCE_NAME, num_ssu)
        ]
        _assert_paired(
            list(source_rows.values()), num_ssu, "main plus SLO admission"
        )
        metrics = {
            strategy: _strategy_metrics(strategy, source_rows[strategy])
            for strategy in TABLE_STRATEGIES
        }
        table_rows.extend(metrics[strategy] for strategy in TABLE_STRATEGIES)
        admission_deltas[str(num_ssu)] = {
            f"admission_minus_{reference}": _metric_deltas(
                metrics["admission"], metrics[reference]
            )
            for reference in ("baseline", "current_scheme_b")
        }
        paired_intersections.append(
            _paired_request_intersection(num_ssu, source_rows)
        )

    oracle_rows = sorted(
        (
            row
            for (strategy, _), row in diagnostic_rows.items()
            if strategy == "released_io_oracle"
        ),
        key=lambda row: int(row["num_ssu"]),
    )
    valid_oracles = [
        _strategy_metrics("released_io_oracle", row)
        for row in oracle_rows
        if row.get("status") == "ok"
    ]
    invalid_oracles = [
        _invalid_oracle(row, requests_per_npu)
        for row in oracle_rows
        if row.get("status") != "ok"
    ]

    baseline_18 = main_rows[("baseline", 18)]
    new_18 = main_rows[("new_scheme_b", 18)]
    baseline_bypass = diagnostic_rows[("baseline_npu_bypass", 18)]
    new_bypass = diagnostic_rows[("new_scheme_b_npu_bypass", 18)]
    oracle_18 = diagnostic_rows[("released_io_oracle", 18)]
    bypass_source_rows = (
        baseline_18,
        baseline_bypass,
        new_18,
        new_bypass,
        oracle_18,
    )
    _assert_paired(bypass_source_rows, 18, "NPU bypass and released-I/O oracle")
    bypass_metrics = {
        "baseline": _strategy_metrics("baseline", baseline_18),
        "baseline_npu_bypass": _strategy_metrics(
            "baseline_npu_bypass", baseline_bypass
        ),
        "new_scheme_b": _strategy_metrics("new_scheme_b", new_18),
        "new_scheme_b_npu_bypass": _strategy_metrics(
            "new_scheme_b_npu_bypass", new_bypass
        ),
    }
    npu_bypass = {
        "num_ssu": 18,
        "physical_npu_bw_gbps": {
            "baseline": 50.0,
            "baseline_npu_bypass": baseline_bypass["physical_npu_bw_gbps"],
            "new_scheme_b": 50.0,
            "new_scheme_b_npu_bypass": new_bypass["physical_npu_bw_gbps"],
        },
        "allocator_npu_cap_gbps": {
            "baseline": None,
            "baseline_npu_bypass": baseline_bypass.get("allocator_npu_cap_gbps"),
            "new_scheme_b": 50.0,
            "new_scheme_b_npu_bypass": new_bypass.get(
                "allocator_npu_cap_gbps"
            ),
        },
        "rows": [
            bypass_metrics[strategy]
            for strategy in (
                "baseline",
                "baseline_npu_bypass",
                "new_scheme_b",
                "new_scheme_b_npu_bypass",
            )
        ],
        "bypass_minus_original_pp": {
            "baseline": _metric_deltas(
                bypass_metrics["baseline_npu_bypass"],
                bypass_metrics["baseline"],
            ),
            "new_scheme_b": _metric_deltas(
                bypass_metrics["new_scheme_b_npu_bypass"],
                bypass_metrics["new_scheme_b"],
            ),
        },
    }

    current_source_fingerprints = current_source_fingerprints or {
        "results": results_payload["experiment"]["source_fingerprint"],
        "diagnostics": diagnostics_payload["source_fingerprint"],
    }
    stored_source_fingerprints = {
        "results": results_payload["experiment"]["source_fingerprint"],
        "diagnostics": diagnostics_payload["source_fingerprint"],
    }
    source_matches_current = {
        name: stored_source_fingerprints[name]
        == current_source_fingerprints[name]
        for name in ("results", "diagnostics")
    }
    hard_warnings = []
    for name, matches in source_matches_current.items():
        if not matches:
            hard_warnings.append(
                f"{name} source fingerprint does not match current source; "
                "numeric rows are stale and must not be treated as final"
            )

    return {
        "schema_version": 1,
        "inputs": input_metadata or {},
        "integrity": {
            "primary_three_strategy_complete": True,
            "diagnostic_all_cases_finished": True,
            "diagnostic_complete": bool(diagnostics_payload.get("complete")),
            "diagnostic_complete_false_is_explained_by_invalid_oracles": (
                not diagnostics_payload.get("complete") and bool(invalid_oracles)
            ),
            "invalid_rows_excluded_from_numeric_performance_tables": True,
            "paired_fields": list(PAIR_FIELDS),
            "stored_source_fingerprints": stored_source_fingerprints,
            "current_source_fingerprints": current_source_fingerprints,
            "source_matches_current": source_matches_current,
            "provenance_ok": all(source_matches_current.values()),
            "hard_warnings": hard_warnings,
        },
        "ssu_list": list(ssu_list),
        "categories": list(CATEGORIES),
        "strategy_table": table_rows,
        "admission_deltas_pp": admission_deltas,
        "paired_request_intersections": paired_intersections,
        "ssu18_npu_bypass": npu_bypass,
        "valid_oracles": valid_oracles,
        "invalid_oracles": invalid_oracles,
    }


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _pp_text(value):
    return "—" if value is None else f"{float(value):+.2f} pp"


def _ids(values):
    return "—" if not values else ",".join(str(value) for value in values)


def _append_metric_table(lines, rows, *, include_caps=None):
    header = ["SSU", "策略"]
    align = ["---:", "---"]
    if include_caps is not None:
        header.extend(["物理 NPU GB/s", "分配器 NPU cap"])
        align.extend(["---:", "---:"])
    header.extend(
        [
            "NPU util min",
            "p10",
            "mean",
            "Equal-NPU SLO",
            "Request SLO",
            "SS",
            "SL",
            "LS",
            "LL",
            "测量请求",
        ]
    )
    align.extend(["---:"] * 10)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(align) + "|")
    for row in rows:
        values = [str(row["num_ssu"]), row["strategy"]]
        if include_caps is not None:
            physical = include_caps["physical_npu_bw_gbps"][row["strategy"]]
            allocator = include_caps["allocator_npu_cap_gbps"][row["strategy"]]
            values.extend(
                [
                    f"{physical:g}" if physical is not None else "—",
                    f"{allocator:g}" if allocator is not None else "—",
                ]
            )
        values.extend(
            [
                _pct(row["npu_utilization"]["min"]),
                _pct(row["npu_utilization"]["p10"]),
                _pct(row["npu_utilization"]["mean"]),
                _pct(row["equal_npu_slo_attainment"]),
                _pct(row["request_weighted_slo_attainment"]),
                *(
                    _pct(row["category_slo"][category]["slo_attainment"])
                    for category in CATEGORIES
                ),
                str(row["measurement_request_count"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")


def render_markdown(analysis):
    lines = [
        "# 32-NPU warm/full-load root-cause analysis",
        "",
        "本报告由主实验 `results.json` 和诊断实验 `diagnostics.json` 独立重算。"
        "所有利用率都来自各策略自身固定 2,000-ms 测量窗；SLO 同时给出 Equal-NPU "
        "和 request-weighted 口径。",
        "",
    ]
    if analysis["integrity"]["hard_warnings"]:
        lines.extend(
            [
                "> [!WARNING]",
                "> **Provenance hard warning：当前数字不可作为最终结论。**",
                *(
                    f"> - {warning}"
                    for warning in analysis["integrity"]["hard_warnings"]
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "输入 artifact 的 stored source fingerprint 与当前源码一致。",
                "",
            ]
        )
    lines.extend(["## Baseline / Current / New / Admission", ""])
    _append_metric_table(lines, analysis["strategy_table"])

    lines.extend(
        [
            "",
            "### Admission 相对收益（原始固定窗口）",
            "",
            "下表是 admission 减去 reference 的百分点。请求 cohort 会随策略变化，"
            "因此 TTFT 的严格同请求比较见下一节。",
            "",
            "| SSU | Reference | Mean NPU util | Equal-NPU SLO | Request SLO | "
            "SS | SL | LS | LL |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in analysis["ssu_list"]:
        for reference in ("baseline", "current_scheme_b"):
            delta = analysis["admission_deltas_pp"][str(num_ssu)][
                f"admission_minus_{reference}"
            ]
            lines.append(
                "| {ssu} | {reference} | {util} | {equal} | {request} | "
                "{SS} | {SL} | {LS} | {LL} |".format(
                    ssu=num_ssu,
                    reference=reference,
                    util=_pp_text(delta["mean_npu_utilization_pp"]),
                    equal=_pp_text(delta["equal_npu_slo_attainment_pp"]),
                    request=_pp_text(
                        delta["request_weighted_slo_attainment_pp"]
                    ),
                    **{
                        category: _pp_text(delta["category_slo_pp"][category])
                        for category in CATEGORIES
                    },
                )
            )

    lines.extend(
        [
            "",
            "## Paired request intersection：baseline/current/admission",
            "",
            "每个 SSU 仅保留三个策略测量 cohort 中 request ID 的交集；NPU、sequence、"
            "category 和 ideal TTFT 也逐请求核对一致。该表用于排除不同 admission "
            "时刻造成的 cohort 偏差。",
            "",
        ]
    )
    for intersection in analysis["paired_request_intersections"]:
        sizes = intersection["original_cohort_sizes"]
        lines.extend(
            [
                f"### SSU {intersection['num_ssu']}",
                "",
                "原 cohort：baseline={baseline}，current={current}，admission={admission}；"
                "交集={common}。".format(
                    baseline=sizes["baseline"],
                    current=sizes["current_scheme_b"],
                    admission=sizes["admission"],
                    common=intersection["common_request_count"],
                ),
                "",
                "| 策略 | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy in PAIRED_STRATEGIES:
            metrics = intersection["metrics"][strategy]
            lines.append(
                "| {strategy} | {equal} | {request} | {SS} | {SL} | {LS} | "
                "{LL} |".format(
                    strategy=strategy,
                    equal=_pct(metrics["equal_npu_slo_attainment"]),
                    request=_pct(metrics["request_weighted_slo_attainment"]),
                    **{
                        category: _pct(
                            metrics["category_slo"][category]["slo_attainment"]
                        )
                        for category in CATEGORIES
                    },
                )
            )
        lines.extend(
            [
                "",
                "| Admission minus | Equal-NPU SLO | Request SLO | SS | SL | "
                "LS | LL |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for reference in ("baseline", "current_scheme_b"):
            delta = intersection["admission_deltas_pp"][
                f"admission_minus_{reference}"
            ]
            lines.append(
                "| {reference} | {equal} | {request} | {SS} | {SL} | {LS} | "
                "{LL} |".format(
                    reference=reference,
                    equal=_pp_text(delta["equal_npu_slo_attainment_pp"]),
                    request=_pp_text(
                        delta["request_weighted_slo_attainment_pp"]
                    ),
                    **{
                        category: _pp_text(delta["category_slo_pp"][category])
                        for category in CATEGORIES
                    },
                )
            )
        lines.append("")

    bypass = analysis["ssu18_npu_bypass"]
    lines.extend(
        [
            "## SSU18 NPU-link bypass",
            "",
            "Bypass 将物理 NPU link 提升到 1,000,000 GB/s；new Scheme B 的"
            " allocator NPU cap 同时提升，baseline 没有动态 allocator cap。",
            "",
        ]
    )
    _append_metric_table(lines, bypass["rows"], include_caps=bypass)
    lines.extend(
        [
            "",
            "| 策略 | Bypass mean util | Bypass Equal-NPU SLO | "
            "Bypass Request SLO |",
            "|---|---:|---:|---:|",
        ]
    )
    for strategy in ("baseline", "new_scheme_b"):
        delta = bypass["bypass_minus_original_pp"][strategy]
        lines.append(
            "| {strategy} | {util} | {equal} | {request} |".format(
                strategy=strategy,
                util=_pp_text(delta["mean_npu_utilization_pp"]),
                equal=_pp_text(delta["equal_npu_slo_attainment_pp"]),
                request=_pp_text(delta["request_weighted_slo_attainment_pp"]),
            )
        )

    lines.extend(["", "## Valid released-I/O oracle", ""])
    _append_metric_table(lines, analysis["valid_oracles"])
    lines.extend(
        [
            "",
            "SSU18 oracle 是有效 steady-state row，可作为已释放 I/O 可见条件下的"
            "调度参考；它不是不可实现的全未来信息最优解。",
            "",
            "## Invalid oracle diagnostics",
            "",
            "以下 row 只报告科学无效原因，不进入上面的任何利用率或 SLO 数字表。",
            "",
            "| SSU | Failed invariants | Zero-sample NPU IDs | "
            "Finite-prefix exhausted NPU IDs |",
            "|---:|---|---|---|",
        ]
    )
    for row in analysis["invalid_oracles"]:
        lines.append(
            "| {ssu} | {failed} | {zero} | {exhausted} |".format(
                ssu=row["num_ssu"],
                failed=", ".join(row["failed_invariants"]),
                zero=_ids(row["zero_sample_npu_ids"]),
                exhausted=_ids(row["finite_prefix_exhausted_npu_ids"]),
            )
        )

    lines.extend(
        [
            "",
            "## Root-cause reading",
            "",
            "- Admission 与 baseline/current 在相同 workload、placement、trace 和 "
            "simulator input 上配对；其 SLO 改善不能解释为输入变化。",
            "- Admission 的 SLO 收益应优先使用 paired-request 表；固定窗口利用率仍"
            "是正确的系统时间积分，但不能按 request ID 配对。",
            "- SSU18 的 NPU bypass 只给出移除 50-GB/s link/cap 后的敏感性，"
            "不能单独证明所有差距都由 NPU link 导致。",
            "- SSU6/10 oracle 因有限前缀耗尽且部分 NPU 无 SLO 样本而无效，"
            "不能引用其 SLO/利用率，也不能把它们当上界。",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    input_metadata = {
        "results": {
            "path": _display_path(args.results),
            "sha256": _sha256(args.results),
        },
        "diagnostics": {
            "path": _display_path(args.diagnostics),
            "sha256": _sha256(args.diagnostics),
        },
    }
    analysis = analyze(
        json.loads(args.results.read_text()),
        json.loads(args.diagnostics.read_text()),
        input_metadata=input_metadata,
        current_source_fingerprints={
            "results": _simulation_source_fingerprint(
                b"steady-state-32npu-normalized-slo-v1\0", MAIN_SOURCE_FILES
            ),
            "diagnostics": _simulation_source_fingerprint(
                b"steady-state-32npu-diagnostics-v1\0",
                DIAGNOSTIC_SOURCE_FILES,
            ),
        },
    )
    _write(args.analysis, json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    _write(args.report, render_markdown(analysis))
    print(f"wrote {args.analysis}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
