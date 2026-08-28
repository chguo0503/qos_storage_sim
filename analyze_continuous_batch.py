"""Validate, aggregate and plot the continuous-batch Scheme-B trace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import continuous_batch_experiment as experiment
import sim


DEFAULT_INPUT = experiment.DEFAULT_OUTPUT
DEFAULT_DIR = DEFAULT_INPUT.parent
POLICY_ORDER = tuple(experiment.POLICIES)
PLOT_POLICIES = (
    "fixed_initial",
    "fresh_every_epoch",
    "per_path10_max8",
    "global5_max8",
    "global5_guard80_max8",
    "global5_guard90_max8",
    "periodic8",
)
LABELS = {
    "fixed_initial": "Initial CIR only",
    "fresh_every_epoch": "Fresh every epoch",
    "per_path10_max8": "Any Path >10%",
    "global2_max8": "Global deficit >2%",
    "global5_max8": "Global deficit >5%",
    "global5_guard80_max8": "Global 5% + resource 80%",
    "global5_guard90_max8": "Global 5% + resource 90%",
    "global10_max8": "Global deficit >10%",
    "periodic8": "Periodic every 8",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
STYLES = {
    policy: {"marker": MARKERS[index], "linewidth": 1.9, "markersize": 5}
    for index, policy in enumerate(POLICY_ORDER)
}
MEAN_METRICS = (
    "mean_target_cir_coverage",
    "mean_cir_fidelity",
    "mean_under_target_fraction",
    "mean_over_demand_gbps",
    "mean_min_npu_coverage",
    "mean_min_ssu_coverage",
    "mean_starved_active_flow_fraction",
    "commit_epoch_fraction",
    "path_writes_per_epoch",
    "config_bytes_per_epoch",
    "mean_touched_ssus_per_commit",
)
MIN_METRICS = (
    "min_target_cir_coverage",
    "worst_npu_coverage",
    "worst_ssu_coverage",
)


def _mean(rows, field):
    return statistics.fmean(row[field] for row in rows)


def _aggregate_rows(rows):
    measured_epochs = rows[0]["measured_controller_epochs"]
    return {
        "mean_target_churn": _mean(rows, "mean_target_churn"),
        "max_target_churn": max(row["max_target_churn"] for row in rows),
        "mean_target_grant_fraction": _mean(rows, "mean_target_grant_fraction"),
        "mean_active_npu_ssu_flows": _mean(rows, "mean_active_npu_ssu_flows"),
        "mean_batch_compute_window_ms": _mean(rows, "mean_batch_compute_window_ms"),
        "measured_replacement_rate": _mean(rows, "measured_replacement_rate"),
        "appended_tokens_per_epoch": _mean(rows, "appended_tokens")
        / measured_epochs,
        "new_block_events_per_epoch": _mean(rows, "new_block_events")
        / measured_epochs,
        "aggregate_flow_activations_per_epoch": _mean(
            rows, "aggregate_flow_activations"
        )
        / measured_epochs,
        "growth_gb_per_measured_epoch": _mean(
            rows, "growth_gb_per_measured_epoch"
        ),
        "policies": {
            policy: {
                **{
                    metric: statistics.fmean(
                        row["policies"][policy][metric] for row in rows
                    )
                    for metric in MEAN_METRICS
                },
                **{
                    metric: min(row["policies"][policy][metric] for row in rows)
                    for metric in MIN_METRICS
                },
                "max_pending_age": max(
                    row["policies"][policy]["max_pending_age"] for row in rows
                ),
                "commit_reasons": dict(
                    sum(
                        (
                            Counter(row["policies"][policy]["commit_reasons"])
                            for row in rows
                        ),
                        Counter(),
                    )
                ),
            }
            for policy in POLICY_ORDER
        },
    }


def _validate(payload):
    spec = payload["experiment"]
    assert payload["complete"] is True
    assert payload["schema_version"] == experiment.SCHEMA_VERSION
    assert spec["code_fingerprint"] == experiment._fingerprint()
    assert spec["control_plane_only"] is True
    assert spec["end_to_end_npu_utilization"] is False
    assert spec["placement_mode"] == sim.PLACEMENT_BLOCK_RING_HASH
    assert tuple(spec["policies"]) == POLICY_ORDER

    expected = set()
    for seed in spec["seeds"]:
        for growth in spec["growth_modes"]:
            expected.update(
                (seed, ssu, spec["batch_size"], growth)
                for ssu in spec["ssu_list"]
            )
            expected.update(
                (seed, 40, batch, growth)
                for batch in spec["batch_size_sweep"]
            )
    actual = {
        (row["seed"], row["num_ssu"], row["batch_size"], row["growth_mode"])
        for row in payload["results"]
    }
    assert actual == expected

    ssd_cap = spec["ssd_capacity_gbps"]
    npu_cap = spec["npu_capacity_gbps"]
    for row in payload["results"]:
        assert row["source_fingerprint"] == spec["source_fingerprint"]
        for policy in POLICY_ORDER:
            values = row["policies"][policy]
            assert values["max_ssu_cir_sum_gbps"] <= ssd_cap + 1e-8
            assert values["max_npu_cir_sum_gbps"] <= npu_cap + 1e-8

    paired = defaultdict(list)
    for row in payload["results"]:
        paired[(row["seed"], row["num_ssu"], row["batch_size"])].append(row)
    for rows in paired.values():
        assert len(rows) == len(spec["growth_modes"])
        assert len({row["trace_fingerprint"] for row in rows}) == 1
        assert len({row["workload_fingerprint"] for row in rows}) == 1


def analyze(payload):
    _validate(payload)
    spec = payload["experiment"]
    grouped = defaultdict(list)
    for row in payload["results"]:
        grouped[(row["growth_mode"], row["num_ssu"], row["batch_size"])].append(
            row
        )

    main = {
        growth: {
            str(ssu): _aggregate_rows(
                grouped[(growth, ssu, spec["batch_size"])]
            )
            for ssu in spec["ssu_list"]
        }
        for growth in spec["growth_modes"]
    }
    sensitivity = {
        growth: {
            str(batch): _aggregate_rows(grouped[(growth, 40, batch)])
            for batch in spec["batch_size_sweep"]
        }
        for growth in spec["growth_modes"]
    }
    paired_delta = {}
    for ssu in spec["ssu_list"]:
        static = main["static_kv"][str(ssu)]
        growth = main["decode_grow_1token"][str(ssu)]
        paired_delta[str(ssu)] = {
            policy: {
                metric: growth["policies"][policy][metric]
                - static["policies"][policy][metric]
                for metric in (
                    "mean_target_cir_coverage",
                    "commit_epoch_fraction",
                    "path_writes_per_epoch",
                )
            }
            for policy in POLICY_ORDER
        }
    return {
        "scope": "control-plane trace; not end-to-end NPU utilization",
        "model": {
            key: spec[key]
            for key in (
                "num_npu",
                "batch_size",
                "batch_size_sweep",
                "ssu_list",
                "growth_modes",
                "epochs",
                "warmup_epochs",
                "replacement_probability",
                "batch_compute_model",
                "ssd_capacity_gbps",
                "npu_capacity_gbps",
                "placement_mode",
            )
        },
        "main": main,
        "batch_size_sensitivity_ssu40": sensitivity,
        "decode_growth_minus_static": paired_delta,
    }


def _finish_axis(axis, *, xlabel, ylabel, title, xticks, ylim=None):
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.3)


def write_plots(output_dir, analysis):
    ssus = analysis["model"]["ssu_list"]
    rows = analysis["main"]["decode_grow_1token"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for policy in PLOT_POLICIES:
        values = [rows[str(ssu)]["policies"][policy] for ssu in ssus]
        axes[0].plot(
            ssus,
            [100.0 * value["mean_target_cir_coverage"] for value in values],
            label=LABELS[policy],
            **STYLES[policy],
        )
        axes[1].plot(
            ssus,
            [100.0 * value["commit_epoch_fraction"] for value in values],
            label=LABELS[policy],
            **STYLES[policy],
        )
        axes[2].plot(
            ssus,
            [value["path_writes_per_epoch"] for value in values],
            label=LABELS[policy],
            **STYLES[policy],
        )
    _finish_axis(
        axes[0],
        xlabel="Number of SSUs",
        ylabel="Target CIR coverage (%)",
        title="Grant coverage",
        xticks=ssus,
        ylim=(0, 101),
    )
    _finish_axis(
        axes[1],
        xlabel="Number of SSUs",
        ylabel="Epochs with a CIR commit (%)",
        title="Commit frequency",
        xticks=ssus,
        ylim=(0, 101),
    )
    _finish_axis(
        axes[2],
        xlabel="Number of SSUs",
        ylabel="Changed Path CIRs / epoch",
        title="Configuration writes",
        xticks=ssus,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Scheme-B CIR control under continuous batching (batch=8, +1 SSD token/epoch)",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    first = output_dir / "01_continuous_batch_cir_control.png"
    figure.savefig(first, dpi=180, bbox_inches="tight")
    plt.close(figure)

    batches = analysis["model"]["batch_size_sweep"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for growth, linestyle in (("static_kv", "--"), ("decode_grow_1token", "-")):
        rows = analysis["batch_size_sensitivity_ssu40"][growth]
        for policy in ("global5_guard80_max8", "global5_guard90_max8"):
            values = [rows[str(batch)]["policies"][policy] for batch in batches]
            label = "%s, %s" % (
                LABELS[policy],
                "decode growth" if growth == "decode_grow_1token" else "static KV",
            )
            style = dict(STYLES[policy], linestyle=linestyle)
            axes[0].plot(
                batches,
                [100.0 * value["mean_target_cir_coverage"] for value in values],
                label=label,
                **style,
            )
            axes[1].plot(
                batches,
                [100.0 * value["commit_epoch_fraction"] for value in values],
                label=label,
                **style,
            )
            axes[2].plot(
                batches,
                [100.0 * value["worst_npu_coverage"] for value in values],
                label=label,
                **style,
            )
    _finish_axis(
        axes[0],
        xlabel="Continuous batch size",
        ylabel="Target CIR coverage (%)",
        title="Aggregate coverage",
        xticks=batches,
        ylim=(75, 101),
    )
    _finish_axis(
        axes[1],
        xlabel="Continuous batch size",
        ylabel="Epochs with a CIR commit (%)",
        title="Commit frequency",
        xticks=batches,
        ylim=(0, 101),
    )
    _finish_axis(
        axes[2],
        xlabel="Continuous batch size",
        ylabel="Worst NPU target coverage (%)",
        title="Tail protection",
        xticks=batches,
        ylim=(70, 101),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    second = output_dir / "02_continuous_batch_size_sensitivity.png"
    figure.savefig(second, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for policy in (
        "per_path10_max8",
        "global5_max8",
        "global5_guard80_max8",
        "global5_guard90_max8",
    ):
        delta = [
            analysis["decode_growth_minus_static"][str(ssu)][policy]
            for ssu in ssus
        ]
        axes[0].plot(
            ssus,
            [100.0 * value["commit_epoch_fraction"] for value in delta],
            label=LABELS[policy],
            **STYLES[policy],
        )
        axes[1].plot(
            ssus,
            [value["path_writes_per_epoch"] for value in delta],
            label=LABELS[policy],
            **STYLES[policy],
        )
    _finish_axis(
        axes[0],
        xlabel="Number of SSUs",
        ylabel="Commit-frequency delta (pp)",
        title="Decode growth minus static KV",
        xticks=ssus,
    )
    _finish_axis(
        axes[1],
        xlabel="Number of SSUs",
        ylabel="Extra changed Path CIRs / epoch",
        title="Decode-growth write overhead",
        xticks=ssus,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    third = output_dir / "03_decode_growth_sensitivity.png"
    figure.savefig(third, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return first, second, third


def write_report(path, analysis):
    rows = analysis["main"]["decode_grow_1token"]
    lines = [
        "# Continuous-batch Scheme-B control results",
        "",
        "This is a control-plane trace, not an end-to-end NPU-utilization result. "
        "Every target and committed grant obeys SSD 40 GB/s and NPU 50 GB/s limits.",
        "",
        "The main case has 128 always-busy NPUs, batch 8, a 2.5% per-slot "
        "replacement probability per epoch, serial-equivalent batch compute, and "
        "one newly SSD-resident decode token per surviving request per epoch.",
        "",
        "## Resource-guarded global controller",
        "",
        "| SSU | Target churn | 80% guard coverage | 80% worst NPU | 80% commits | 80% writes/epoch | 90% guard coverage | 90% worst NPU | 90% commits | 90% writes/epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ssu in analysis["model"]["ssu_list"]:
        row = rows[str(ssu)]
        p80 = row["policies"]["global5_guard80_max8"]
        p90 = row["policies"]["global5_guard90_max8"]
        lines.append(
            "| %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.1f | %.2f%% | %.2f%% | %.2f%% | %.1f |"
            % (
                ssu,
                100.0 * row["mean_target_churn"],
                100.0 * p80["mean_target_cir_coverage"],
                100.0 * p80["worst_npu_coverage"],
                100.0 * p80["commit_epoch_fraction"],
                p80["path_writes_per_epoch"],
                100.0 * p90["mean_target_cir_coverage"],
                100.0 * p90["worst_npu_coverage"],
                100.0 * p90["commit_epoch_fraction"],
                p90["path_writes_per_epoch"],
            )
        )

    lines.extend(
        [
            "",
            "## Low-frequency versus per-NPU freshness",
            "",
            "| SSU | Global-5 coverage | Global-5 worst NPU | Global-5 commits | Periodic-8 coverage | Periodic-8 worst NPU | Periodic-8 commits |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in analysis["model"]["ssu_list"]:
        row = rows[str(ssu)]
        global5 = row["policies"]["global5_max8"]
        periodic = row["policies"]["periodic8"]
        lines.append(
            "| %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |"
            % (
                ssu,
                100.0 * global5["mean_target_cir_coverage"],
                100.0 * global5["worst_npu_coverage"],
                100.0 * global5["commit_epoch_fraction"],
                100.0 * periodic["mean_target_cir_coverage"],
                100.0 * periodic["worst_npu_coverage"],
                100.0 * periodic["commit_epoch_fraction"],
            )
        )

    lines.extend(
        [
            "",
            "## Reference controllers",
            "",
            "| SSU | Fixed coverage | Any-Path commits | Any-Path writes/epoch | Fresh writes/epoch | Periodic-8 coverage | Periodic-8 commits |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in analysis["model"]["ssu_list"]:
        row = rows[str(ssu)]
        fixed = row["policies"]["fixed_initial"]
        per_path = row["policies"]["per_path10_max8"]
        fresh = row["policies"]["fresh_every_epoch"]
        periodic = row["policies"]["periodic8"]
        lines.append(
            "| %d | %.2f%% | %.2f%% | %.1f | %.1f | %.2f%% | %.2f%% |"
            % (
                ssu,
                100.0 * fixed["mean_target_cir_coverage"],
                100.0 * per_path["commit_epoch_fraction"],
                per_path["path_writes_per_epoch"],
                fresh["path_writes_per_epoch"],
                100.0 * periodic["mean_target_cir_coverage"],
                100.0 * periodic["commit_epoch_fraction"],
            )
        )

    lines.extend(
        [
            "",
            "`Path writes/epoch` counts every exact floating-point grant entry "
            "that changed. It is an upper bound until a real SSU CIR register "
            "granularity and update protocol are specified.",
        ]
    )

    lines.extend(
        [
            "",
            "## Decode-growth sensitivity",
            "",
            "| SSU | Appended tokens/epoch | New blocks/epoch | New NPU-SSU flows/epoch | 80% guard commit delta | 80% guard write delta |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in analysis["model"]["ssu_list"]:
        row = rows[str(ssu)]
        delta = analysis["decode_growth_minus_static"][str(ssu)][
            "global5_guard80_max8"
        ]
        lines.append(
            "| %d | %.1f | %.1f | %.1f | %+.2f pp | %+.1f |"
            % (
                ssu,
                row["appended_tokens_per_epoch"],
                row["new_block_events_per_epoch"],
                row["aggregate_flow_activations_per_epoch"],
                100.0 * delta["commit_epoch_fraction"],
                delta["path_writes_per_epoch"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "analysis.json"
    report_path = args.output_dir / "report.md"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    write_report(report_path, analysis)
    plots = write_plots(args.output_dir, analysis)
    print("analysis:", analysis_path)
    print("report:", report_path)
    print("plots:", ", ".join(map(str, plots)))


if __name__ == "__main__":
    main()
