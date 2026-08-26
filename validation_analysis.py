"""Cross-seed validation for baseline and four predeclared candidates.

The seed-42 input is the complete formal matrix and is filtered to the five
strategies below.  The seed-43 input must be an exact selected-strategy run.
Ranks in this module are restricted to the four non-baseline candidates; they
are not ranks over the complete formal strategy registry.  This is a
deterministic two-seed sensitivity check, not a significance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_STRATEGIES = (
    "baseline",
    "current_refresh8",
    "tune__low_protect_cir_20_6_8_6_current_paths",
    "tune__low_protect_cir_20_5_10_5_paths_12_5_10_5",
    "demand_maxmin",
)
NON_BASELINE_STRATEGIES = EXPECTED_STRATEGIES[1:]
FORMAL_SSUS = (40, 56, 80)
FORMAL_NUM_NPU = 128
FORMAL_LAYERS = 16
SEED_OFFSETS = {
    "workload": 0,
    "placement": 1_000_003,
    "submit_order": 2_000_003,
    "arrival_delay": 3_000_003,
}
DEFAULT_SEED42 = Path("results/full_analysis/results.json")
DEFAULT_SEED43 = Path("results/full_analysis_seed43/results.json")
DEFAULT_OUTPUT_DIR = Path("results/full_analysis")

STRATEGY_LABELS = {
    "baseline": "Baseline",
    "current_refresh8": "Current static",
    "tune__low_protect_cir_20_6_8_6_current_paths": "CIR 20/6/8/6",
    "tune__low_protect_cir_20_5_10_5_paths_12_5_10_5": (
        "CIR 20/5/10/5 + paths"
    ),
    "demand_maxmin": "Demand max-min",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_seed_bundle(seed):
    return {
        name: int(seed + offset) for name, offset in SEED_OFFSETS.items()
    }


def load_result(path):
    return json.loads(Path(path).read_text())


def _validate_runtime(data, seed, source_name):
    _require(data["complete"] is True, "%s is not complete" % source_name)
    runtime = data["experiment"]["runtime"]
    _require(runtime["mode"] == "formal", "%s is not a formal run" % source_name)
    _require(
        runtime["num_npu"] == FORMAL_NUM_NPU,
        "%s num_npu must be 128" % source_name,
    )
    _require(
        runtime["n_layers"] == FORMAL_LAYERS,
        "%s n_layers must be 16" % source_name,
    )
    _require(
        tuple(runtime["ssu_list"]) == FORMAL_SSUS,
        "%s SSUs must be 40,56,80" % source_name,
    )
    _require(
        runtime["seeds"] == expected_seed_bundle(seed),
        "%s seed bundle does not match seed %d" % (source_name, seed),
    )


def _validate_strategy_scope(data42, data43):
    selected42 = tuple(data42["selected_strategies"])
    selected43 = tuple(data43["selected_strategies"])
    _require(
        len(selected42) == len(set(selected42)),
        "seed42 selected_strategies contains duplicates",
    )
    _require(
        set(EXPECTED_STRATEGIES).issubset(selected42),
        "seed42 full matrix does not contain all five validation strategies",
    )
    _require(
        len(selected43) == len(EXPECTED_STRATEGIES)
        and set(selected43) == set(EXPECTED_STRATEGIES),
        "seed43 selected_strategies must be exactly the five validation strategies",
    )
    result_strategies43 = {row["strategy"] for row in data43["results"]}
    _require(
        result_strategies43 == set(EXPECTED_STRATEGIES),
        "seed43 result rows must contain exactly the five validation strategies",
    )


def _extract_paired_rows(data, seed, source_name):
    indexed = {}
    for row in data["results"]:
        strategy = row["strategy"]
        if strategy not in EXPECTED_STRATEGIES:
            continue
        key = (int(row["num_ssu"]), strategy)
        _require(key not in indexed, "%s contains duplicate %s" % (source_name, key))
        indexed[key] = row

    expected_keys = {
        (ssu, strategy)
        for ssu in FORMAL_SSUS
        for strategy in EXPECTED_STRATEGIES
    }
    _require(
        set(indexed) == expected_keys,
        "%s does not contain one row for every SSU/strategy pair" % source_name,
    )

    workload_hashes = set()
    pairing = {}
    for ssu in FORMAL_SSUS:
        rows = [indexed[(ssu, strategy)] for strategy in EXPECTED_STRATEGIES]
        workload = {row["workload_fingerprint"] for row in rows}
        placement = {row["placement_hash"] for row in rows}
        _require(
            len(workload) == 1,
            "%s SSU=%d workload hashes are not paired" % (source_name, ssu),
        )
        _require(
            len(placement) == 1,
            "%s SSU=%d placement hashes are not paired" % (source_name, ssu),
        )
        workload_hash = next(iter(workload))
        placement_hash = next(iter(placement))
        workload_hashes.add(workload_hash)
        pairing[str(ssu)] = {
            "workload_fingerprint": workload_hash,
            "placement_hash": placement_hash,
        }
        for row in rows:
            _require(
                row["seeds"] == expected_seed_bundle(seed),
                "%s row seed bundle mismatch at SSU=%d strategy=%s"
                % (source_name, ssu, row["strategy"]),
            )
            _require(
                row["summary"]["workload_fingerprint"] == workload_hash,
                "%s summary workload hash mismatch" % source_name,
            )
            _require(
                row["summary"]["placement_hash"] == placement_hash,
                "%s summary placement hash mismatch" % source_name,
            )
            _require(
                len(row["request_metrics"]) == FORMAL_NUM_NPU,
                "%s row does not contain 128 request metrics" % source_name,
            )
    _require(
        len(workload_hashes) == 1,
        "%s workload fingerprint changes across SSU counts" % source_name,
    )
    return indexed, pairing


def validate_sources(data42, data43):
    """Validate both formal inputs and return their five-strategy row indexes."""
    _validate_runtime(data42, 42, "seed42")
    _validate_runtime(data43, 43, "seed43")
    _validate_strategy_scope(data42, data43)
    index42, pairing42 = _extract_paired_rows(data42, 42, "seed42")
    index43, pairing43 = _extract_paired_rows(data43, 43, "seed43")
    return {
        42: index42,
        43: index43,
    }, {
        "42": pairing42,
        "43": pairing43,
    }


def _direction(value):
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "zero"


def _metric(row, name):
    return float(row["summary"][name])


def _rank_strategies(rows, gain_name):
    ranked = sorted(
        NON_BASELINE_STRATEGIES,
        key=lambda strategy: (
            -fmean(rows[(ssu, strategy)][gain_name] for ssu in FORMAL_SSUS),
            strategy,
        ),
    )
    return [
        {
            "rank": rank,
            "strategy": strategy,
            "mean_gain_pp": fmean(
                rows[(ssu, strategy)][gain_name] for ssu in FORMAL_SSUS
            ),
            "gains_by_ssu_pp": {
                str(ssu): rows[(ssu, strategy)][gain_name]
                for ssu in FORMAL_SSUS
            },
        }
        for rank, strategy in enumerate(ranked, start=1)
    ]


def analyze_sources(data42, data43):
    """Return paired gains, rankings, and seed-43 robustness of the seed-42 winner."""
    indexes, pairing = validate_sources(data42, data43)
    comparison_rows = {}
    flat_rows = []
    for seed in (42, 43):
        source = indexes[seed]
        for ssu in FORMAL_SSUS:
            baseline = source[(ssu, "baseline")]
            baseline_request = _metric(baseline, "avg_request_compute_fraction")
            baseline_fleet = _metric(baseline, "fleet_npu_compute_utilization")
            for strategy in EXPECTED_STRATEGIES:
                row = source[(ssu, strategy)]
                item = {
                    "seed": seed,
                    "num_ssu": ssu,
                    "strategy": strategy,
                    "request_compute_fraction": _metric(
                        row, "avg_request_compute_fraction"
                    ),
                    "fleet_npu_compute_utilization": _metric(
                        row, "fleet_npu_compute_utilization"
                    ),
                }
                item["request_gain_pp"] = 100.0 * (
                    item["request_compute_fraction"] - baseline_request
                )
                item["fleet_gain_pp"] = 100.0 * (
                    item["fleet_npu_compute_utilization"] - baseline_fleet
                )
                comparison_rows[(seed, ssu, strategy)] = item
                flat_rows.append(item)

    rankings = {}
    for seed in (42, 43):
        seed_rows = {
            (ssu, strategy): comparison_rows[(seed, ssu, strategy)]
            for ssu in FORMAL_SSUS
            for strategy in EXPECTED_STRATEGIES
        }
        rankings[str(seed)] = {
            "request_gain": _rank_strategies(seed_rows, "request_gain_pp"),
            "fleet_gain": _rank_strategies(seed_rows, "fleet_gain_pp"),
        }

    winner = rankings["42"]["request_gain"][0]["strategy"]
    request_rank43 = next(
        row["rank"]
        for row in rankings["43"]["request_gain"]
        if row["strategy"] == winner
    )
    fleet_rank42 = next(
        row["rank"]
        for row in rankings["42"]["fleet_gain"]
        if row["strategy"] == winner
    )
    fleet_rank43 = next(
        row["rank"]
        for row in rankings["43"]["fleet_gain"]
        if row["strategy"] == winner
    )
    request_direction_by_ssu = {
        str(ssu): {
            "seed42": _direction(
                comparison_rows[(42, ssu, winner)]["request_gain_pp"]
            ),
            "seed43": _direction(
                comparison_rows[(43, ssu, winner)]["request_gain_pp"]
            ),
        }
        for ssu in FORMAL_SSUS
    }
    fleet_direction_by_ssu = {
        str(ssu): {
            "seed42": _direction(
                comparison_rows[(42, ssu, winner)]["fleet_gain_pp"]
            ),
            "seed43": _direction(
                comparison_rows[(43, ssu, winner)]["fleet_gain_pp"]
            ),
        }
        for ssu in FORMAL_SSUS
    }
    mean_request42 = fmean(
        comparison_rows[(42, ssu, winner)]["request_gain_pp"]
        for ssu in FORMAL_SSUS
    )
    mean_request43 = fmean(
        comparison_rows[(43, ssu, winner)]["request_gain_pp"]
        for ssu in FORMAL_SSUS
    )
    mean_fleet42 = fmean(
        comparison_rows[(42, ssu, winner)]["fleet_gain_pp"]
        for ssu in FORMAL_SSUS
    )
    mean_fleet43 = fmean(
        comparison_rows[(43, ssu, winner)]["fleet_gain_pp"]
        for ssu in FORMAL_SSUS
    )
    robustness = {
        "seed42_request_winner": winner,
        "ranking_scope": "four_predeclared_nonbaseline_candidates_only",
        "request_mean_gain_pp": {"42": mean_request42, "43": mean_request43},
        "request_mean_direction": {
            "42": _direction(mean_request42),
            "43": _direction(mean_request43),
        },
        "request_direction_by_ssu": request_direction_by_ssu,
        "request_direction_held_all_ssus": all(
            value["seed42"] == value["seed43"]
            for value in request_direction_by_ssu.values()
        ),
        "request_rank": {"42": 1, "43": request_rank43},
        "request_rank_held": request_rank43 == 1,
        "fleet_mean_gain_pp": {"42": mean_fleet42, "43": mean_fleet43},
        "fleet_mean_direction": {
            "42": _direction(mean_fleet42),
            "43": _direction(mean_fleet43),
        },
        "fleet_direction_by_ssu": fleet_direction_by_ssu,
        "fleet_direction_held_all_ssus": all(
            value["seed42"] == value["seed43"]
            for value in fleet_direction_by_ssu.values()
        ),
        "fleet_rank": {"42": fleet_rank42, "43": fleet_rank43},
        "fleet_rank_held": fleet_rank43 == fleet_rank42,
    }
    return {
        "schema_version": 1,
        "contract": {
            "num_npu": FORMAL_NUM_NPU,
            "n_layers": FORMAL_LAYERS,
            "ssu_list": list(FORMAL_SSUS),
            "seeds": [42, 43],
            "strategies": list(EXPECTED_STRATEGIES),
            "seed42_scope": "full_matrix_filtered_to_expected_five",
            "seed43_scope": "exact_selected_five",
        },
        "pairing": pairing,
        "comparisons": flat_rows,
        "rankings": rankings,
        "seed43_robustness": robustness,
        "interpretation": (
            "Two deterministic seeds provide a sensitivity check only; they do "
            "not establish statistical significance."
        ),
    }


def render_validation_plot(analysis, output_path):
    """Plot request and fleet gains for every seed/SSU combination."""
    lookup = {
        (row["seed"], row["num_ssu"], row["strategy"]): row
        for row in analysis["comparisons"]
    }
    groups = [(seed, ssu) for seed in (42, 43) for ssu in FORMAL_SSUS]
    x_positions = list(range(len(groups)))
    width = 0.16
    colors = ("#7f7f7f", "#4c78a8", "#f58518", "#54a24b", "#b279a2")
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)
    for strategy_index, strategy in enumerate(EXPECTED_STRATEGIES):
        offset = (strategy_index - 2) * width
        positions = [position + offset for position in x_positions]
        request_values = [
            lookup[(seed, ssu, strategy)]["request_gain_pp"]
            for seed, ssu in groups
        ]
        fleet_values = [
            lookup[(seed, ssu, strategy)]["fleet_gain_pp"]
            for seed, ssu in groups
        ]
        axes[0].bar(
            positions,
            request_values,
            width,
            label=STRATEGY_LABELS[strategy],
            color=colors[strategy_index],
        )
        axes[1].bar(
            positions,
            fleet_values,
            width,
            color=colors[strategy_index],
        )
    axes[0].set_ylabel("Request gain vs baseline (pp)")
    axes[1].set_ylabel("Fleet gain vs baseline (pp)")
    axes[1].set_xlabel("Workload seed / SSU count")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(
        ["seed %d\n%d SSU" % group for group in groups]
    )
    figure.suptitle("Seed sensitivity: paired gains at 40/56/80 SSUs")
    figure.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.955))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_markdown(analysis):
    robustness = analysis["seed43_robustness"]
    winner = robustness["seed42_request_winner"]
    lines = [
        "# Seed 42/43 validation",
        "",
        "Formal contract and paired fingerprint checks passed for 128 NPUs, "
        "16 layers, and 40/56/80 SSUs.",
        "",
        "Primary ranking uses the mean request compute-fraction gain over the "
        "three SSU points. Rankings cover only the four predeclared "
        "non-baseline candidates shown below, not the full formal registry.",
        "",
        "| Seed | Request ranking |",
        "|---:|---|",
    ]
    for seed in (42, 43):
        ranking = analysis["rankings"][str(seed)]["request_gain"]
        text = "; ".join(
            "%d. %s (%+.3f pp)"
            % (row["rank"], STRATEGY_LABELS[row["strategy"]], row["mean_gain_pp"])
            for row in ranking
        )
        lines.append("| %d | %s |" % (seed, text))
    lines.extend(
        [
            "",
            "Seed-42 winner within these four candidates: `%s` (request mean "
            "%+.3f pp at seed 42, %+.3f pp at seed 43)."
            % (
                winner,
                robustness["request_mean_gain_pp"]["42"],
                robustness["request_mean_gain_pp"]["43"],
            ),
            "",
            "- Request-gain direction held at all SSUs: **%s**."
            % robustness["request_direction_held_all_ssus"],
            "- Request rank remained first among these four candidates at "
            "seed 43: **%s** (rank %d)."
            % (robustness["request_rank_held"], robustness["request_rank"]["43"]),
            "- Fleet-gain direction held at all SSUs: **%s**."
            % robustness["fleet_direction_held_all_ssus"],
            "- Fleet rank among these four candidates was preserved: **%s** "
            "(seed42 %d, seed43 %d)."
            % (
                robustness["fleet_rank_held"],
                robustness["fleet_rank"]["42"],
                robustness["fleet_rank"]["43"],
            ),
            "",
            "![Seed validation](12_seed_validation.png)",
            "",
            "This is a sensitivity check over two deterministic seeds. It does "
            "not establish statistical significance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def run_validation(seed42_path, seed43_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_sources(load_result(seed42_path), load_result(seed43_path))
    analysis["inputs"] = {
        "seed42": str(Path(seed42_path)),
        "seed43": str(Path(seed43_path)),
    }
    figure_path = output_dir / "12_seed_validation.png"
    render_validation_plot(analysis, figure_path)
    write_json(output_dir / "validation_analysis.json", analysis)
    (output_dir / "validation_report.md").write_text(build_markdown(analysis))
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42", type=Path, default=DEFAULT_SEED42)
    parser.add_argument("--seed43", type=Path, default=DEFAULT_SEED43)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analysis = run_validation(args.seed42, args.seed43, args.output_dir)
    robustness = analysis["seed43_robustness"]
    print("validation: %s" % (args.output_dir / "validation_analysis.json"))
    print("report:     %s" % (args.output_dir / "validation_report.md"))
    print("figure:     %s" % (args.output_dir / "12_seed_validation.png"))
    print("seed42 winner: %s" % robustness["seed42_request_winner"])
    print("seed43 rank held: %s" % robustness["request_rank_held"])


if __name__ == "__main__":
    main()
