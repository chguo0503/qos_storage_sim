"""Validate, aggregate and plot the retained routing strategies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from experiment import plot_results
import routing_refresh_concurrency_experiment as experiment
import sim

DEFAULT_RESULTS = experiment.DEFAULT_OUTPUT_DIR / "results.json"
DEFAULT_ORACLE_RESULTS = (
    experiment.DEFAULT_OUTPUT_DIR / "capacity_constrained_oracle_results.json"
)
DEFAULT_SCHEME_B_RESULTS = (
    experiment.DEFAULT_OUTPUT_DIR / "scheme_b_prefill_results.json"
)
DEFAULT_ANALYSIS = experiment.DEFAULT_OUTPUT_DIR / "analysis.json"
DEFAULT_REPORT = experiment.DEFAULT_OUTPUT_DIR / "report.md"
DEFAULT_PLOT = (
    experiment.DEFAULT_OUTPUT_DIR / "01_routing_refresh_finite_issue.png"
)
ORACLE_STRATEGY = "demand_weighted_sjf_oracle_candidate"
SCHEME_B_STRATEGY = "scheme_b_prefill"
ROUTING_STRATEGY_ORDER = (
    "baseline",
    "refresh8",
)
STRATEGY_ORDER = (*ROUTING_STRATEGY_ORDER, SCHEME_B_STRATEGY, ORACLE_STRATEGY)
LABELS = {
    "baseline": "Baseline (Path 0)",
    "refresh8": "Refresh every 8 I/Os",
    SCHEME_B_STRATEGY: "Scheme B (one-shot)",
    ORACLE_STRATEGY: "Best feasible reference",
}
STYLES = {
    "baseline": "o-",
    "refresh8": "D-",
    SCHEME_B_STRATEGY: "P-",
    ORACLE_STRATEGY: "*-",
}


def _load(path):
    payload = json.loads(path.read_text())
    assert payload["complete"] is True
    return payload


def _pair_key(row):
    return row["seed"], row["num_ssu"]


def _validate(routing, oracle, scheme_b):
    runtime = routing["experiment"]["runtime"]
    assert runtime["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
    assert oracle["experiment"]["runtime"] == runtime
    assert scheme_b["experiment"]["runtime"] == runtime
    data_fingerprint = routing["experiment"]["data_fingerprint"]
    assert oracle["experiment"]["data_fingerprint"] == data_fingerprint
    assert scheme_b["experiment"]["data_fingerprint"] == data_fingerprint
    assert oracle["experiment"]["strategy"] == ORACLE_STRATEGY
    assert scheme_b["experiment"]["strategy"] == SCHEME_B_STRATEGY
    assert set(ROUTING_STRATEGY_ORDER) <= set(routing["selected_strategies"])
    routing_by_pair = defaultdict(list)
    for row in routing["results"]:
        if row["strategy"] not in ROUTING_STRATEGY_ORDER:
            continue
        assert row["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
        assert all(row["summary"]["invariants"].values())
        routing_by_pair[_pair_key(row)].append(row)
    oracle_by_pair = {_pair_key(row): row for row in oracle["results"]}
    scheme_b_by_pair = {_pair_key(row): row for row in scheme_b["results"]}
    expected_pairs = {
        (seed, ssu)
        for seed in runtime["seeds"]
        for ssu in runtime["ssu_list"]
    }
    assert set(routing_by_pair) == expected_pairs
    assert set(oracle_by_pair) == expected_pairs
    assert set(scheme_b_by_pair) == expected_pairs
    assert len(oracle["results"]) == len(expected_pairs)
    assert len(scheme_b["results"]) == len(expected_pairs)
    for pair, rows in routing_by_pair.items():
        seed, _ = pair
        assert {row["strategy"] for row in rows} == set(ROUTING_STRATEGY_ORDER)
        assert all(row["seeds"] == runtime["seed_bundles"][str(seed)] for row in rows)
        fingerprints = {
            (row["workload_fingerprint"], row["placement_hash"])
            for row in rows
        }
        oracle_row = oracle_by_pair[pair]
        scheme_b_row = scheme_b_by_pair[pair]
        fingerprints.add(
            (
                oracle_row["workload_fingerprint"],
                oracle_row["placement_hash"],
            )
        )
        fingerprints.add(
            (
                scheme_b_row["workload_fingerprint"],
                scheme_b_row["placement_hash"],
            )
        )
        assert len(fingerprints) == 1
        assert all(oracle_row["summary"]["invariants"].values())
        assert oracle_row["seeds"] == runtime["seed_bundles"][str(seed)]
        assert scheme_b_row["strategy"] == SCHEME_B_STRATEGY
        assert scheme_b_row["seeds"] == runtime["seed_bundles"][str(seed)]
        assert scheme_b_row["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
        assert all(scheme_b_row["summary"]["invariants"].values())
        assert scheme_b_row["summary"]["pressure_reports"] == 0
    return runtime, routing_by_pair, oracle_by_pair, scheme_b_by_pair


def _mean(rows, field):
    return statistics.fmean(row["summary"][field] for row in rows)


def analyze(routing, oracle, scheme_b):
    runtime, routing_by_pair, oracle_by_pair, scheme_b_by_pair = _validate(
        routing, oracle, scheme_b
    )
    by_strategy_ssu = defaultdict(list)
    for pair, rows in routing_by_pair.items():
        _, ssu = pair
        for row in rows:
            by_strategy_ssu[(row["strategy"], ssu)].append(row)
        scheme_b_row = scheme_b_by_pair[pair]
        by_strategy_ssu[(SCHEME_B_STRATEGY, ssu)].append(scheme_b_row)
        by_strategy_ssu[(ORACLE_STRATEGY, ssu)].append(oracle_by_pair[pair])

    metrics = {}
    category_metrics = {}
    for strategy in STRATEGY_ORDER:
        metrics[strategy] = {}
        category_metrics[strategy] = {}
        for ssu in runtime["ssu_list"]:
            rows = by_strategy_ssu[(strategy, ssu)]
            metrics[strategy][str(ssu)] = {
                "avg_request_compute_fraction": _mean(
                    rows, "avg_request_compute_fraction"
                ),
                "fleet_npu_compute_utilization": _mean(
                    rows, "fleet_npu_compute_utilization"
                ),
                "makespan_ms": _mean(rows, "makespan_ms"),
                "pressure_reports": _mean(rows, "pressure_reports"),
            }
            category_metrics[strategy][str(ssu)] = {
                category: statistics.fmean(
                    row["summary"]["category_metrics"][category][
                        "avg_request_compute_fraction"
                    ]
                    for row in rows
                )
                for category in sim.WORKLOAD_CATEGORIES
            }

    scheme_b_control = {}
    for ssu in runtime["ssu_list"]:
        grants = [
            row["summary"]["scheme_b_grant"]
            for row in by_strategy_ssu[(SCHEME_B_STRATEGY, ssu)]
        ]
        scheme_b_control[str(ssu)] = {
            "grant_to_demand_fraction": statistics.fmean(
                grant["total_grant_gbps"] / grant["total_demand_gbps"]
                for grant in grants
            ),
            "saturated_ssus": statistics.fmean(
                sum(
                    value >= grant["ssd_cap_gbps"] - 1e-9
                    for value in grant["per_ssu_grant_gbps"]
                )
                for grant in grants
            ),
        }
    return {
        "placement_mode": sim.PLACEMENT_BLOCK_RING_HASH,
        "num_npu": runtime["num_npu"],
        "n_layers": runtime["n_layers"],
        "seeds": runtime["seeds"],
        "ssu_list": runtime["ssu_list"],
        "strategy_order": list(STRATEGY_ORDER),
        "metrics": metrics,
        "category_metrics": category_metrics,
        "scheme_b_control": scheme_b_control,
        "optimality_note": (
            "Best feasible reference is the measured physical-capacity-preserving "
            "oracle candidate; it is not a proven mathematical optimum."
        ),
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_report(path, analysis):
    ssus = analysis["ssu_list"]
    lines = [
        "# Ring-hash refresh strategy results",
        "",
        "All rows use 128 NPUs, 16 layers, 0–5 ms arrival delay and the same "
        "block ring-hash placement. One block stays on the same SSU for all layers.",
        "Scheme B computes one max-min NPU×SSU grant for the admitted batch, "
        "writes it once, and reuses it across all prefill layers.",
        "",
        "Values below are the two-seed mean per-request NPU compute fraction.",
        "",
        "| Strategy | " + " | ".join(f"SSU {ssu}" for ssu in ssus) + " |",
        "|---|" + "---:|" * len(ssus),
    ]
    for strategy in analysis["strategy_order"]:
        values = [
            100.0
            * analysis["metrics"][strategy][str(ssu)][
                "avg_request_compute_fraction"
            ]
            for ssu in ssus
        ]
        lines.append(
            f"| {LABELS[strategy]} | "
            + " | ".join(f"{value:.3f}%" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scheme B paired deltas",
            "",
            "| SSU | vs baseline | vs Refresh8 | SS class vs Refresh8 | saturated SSUs |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in ssus:
        key = str(ssu)
        scheme_b_value = analysis["metrics"][SCHEME_B_STRATEGY][key][
            "avg_request_compute_fraction"
        ]
        baseline_value = analysis["metrics"]["baseline"][key][
            "avg_request_compute_fraction"
        ]
        refresh_strategy = "refresh8"
        refresh_value = analysis["metrics"][refresh_strategy][key][
            "avg_request_compute_fraction"
        ]
        scheme_b_ss = analysis["category_metrics"][SCHEME_B_STRATEGY][key]["SS"]
        refresh_ss = analysis["category_metrics"][refresh_strategy][key]["SS"]
        saturated = analysis["scheme_b_control"][key]["saturated_ssus"]
        lines.append(
            f"| {ssu} | {100.0 * (scheme_b_value - baseline_value):+.3f} pp | "
            f"{100.0 * (scheme_b_value - refresh_value):+.3f} pp | "
            f"{100.0 * (scheme_b_ss - refresh_ss):+.3f} pp | "
            f"{saturated:.1f}/{ssu} |"
        )
    lines.extend(
        [
            "",
            "Scheme B uses one per-NPU Path on every SSU and configures the "
            "max-min grant once before the admitted batch launches. The 0–5 ms "
            "arrival vector is retained as launch jitter; no Path-pressure table "
            "is read during execution.",
            "",
            "At 8 SSUs, equal max-min sharing improves the mean under extreme "
            "contention. From 28 through 56 SSUs every disk remains saturated, "
            "and equal flow fairness removes the original 20/40 GB/s protection "
            "for the latency-sensitive SS class; the resulting last-block wait "
            "makes Scheme B slower than pressure-aware routing. The executable "
            "oracle reference has the highest measured mean at every SSU count.",
            "",
            "`Best feasible reference` is an executable capacity-preserving oracle "
            "candidate, not a proven exact optimum.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_plot(path, analysis):
    ssus = analysis["ssu_list"]
    plot_results(
        {
            "ssus": ssus,
            "title": "16-layer routing with block ring-hash placement",
            "ylabel": "Average NPU Utilization (%)",
            "series": [
                {
                    "label": LABELS[strategy],
                    "style": STYLES[strategy],
                    "values": [
                        100.0
                        * analysis["metrics"][strategy][str(ssu)][
                            "avg_request_compute_fraction"
                        ]
                        for ssu in ssus
                    ],
                }
                for strategy in analysis["strategy_order"]
            ],
            "legend": {
                "loc": "lower center",
                "ncol": 4,
                "row_major": True,
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--oracle-results", type=Path, default=DEFAULT_ORACLE_RESULTS
    )
    parser.add_argument(
        "--scheme-b-results", type=Path, default=DEFAULT_SCHEME_B_RESULTS
    )
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()
    analysis = analyze(
        _load(args.results),
        _load(args.oracle_results),
        _load(args.scheme_b_results),
    )
    _write_json(args.analysis, analysis)
    write_report(args.report, analysis)
    write_plot(args.plot, analysis)
    print("analysis:", args.analysis)
    print("report:", args.report)
    print("plot:", args.plot)


if __name__ == "__main__":
    main()
