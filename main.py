import argparse
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import (
    simulate_continuous,
    load_bw_table_cache,
    IOSchedulingConfig,
    QOS_LAYOUT_EIGHT_GROUP,
    QOS_LAYOUT_THREE_TIER,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BATCH_DISPATCH_INTERVAL_US = 200.0
ADAPTIVE_BATCH_HEADROOM = 1.1
QOS_QUEUE_MAX_ACTIVE_FLOWS = 4
EXPERIMENT_SUFFIX = "qd4_b200us"
QOS_QUEUE_COUNT = 24
QOS_LAYOUT = QOS_LAYOUT_THREE_TIER
RESULT_SUFFIX = f"q24_{EXPERIMENT_SUFFIX}"
PICKLE_FILE = os.path.join(
    RESULTS_DIR,
    f"io_scheduling_sweep_{RESULT_SUFFIX}_pir_uncapped.pkl",
)
QOS_POLICY_VERSION = "cir_wrr_pir_uncapped_qdepth4_batch200us_v2"
FAIR_POLICY_VERSION = "single_path_qdepth4_v1"
ADAPTIVE_BATCH_POLICY_VERSION = "payload_over_required_bw_v1"

NPU_COUNT = 128
N_LAYERS = 16
N_REQ = 1
# SEEDS = [42, 123, 456]
SEEDS = [42]
SSU_LIST = [16, 28, 40, 56, 84, 112]
# LS_RATIO_LIST = [0.25, 0.5, 0.75, 1.0]
# SSU_LIST = [8, 16, 28, 40, 56, 84, 112]
LS_RATIO_LIST = [0.5]

# Each mode: (name, policy, io_sched)
MODES = [
    (
        "baseline_fair",
        "fair",
        IOSchedulingConfig(
            batch_dispatch_interval_us=BATCH_DISPATCH_INTERVAL_US,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    (
        "baseline_qwrr",
        "queue_wrr",
        IOSchedulingConfig(
            batch_dispatch_interval_us=BATCH_DISPATCH_INTERVAL_US,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    (
        "batched8_qwrr",
        "queue_wrr",
        IOSchedulingConfig(
            io_dispatch_mode="batched",
            batch_size=8,
            batch_dispatch_interval_us=BATCH_DISPATCH_INTERVAL_US,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    (
        "traffic_aware8_qwrr",
        "queue_wrr",
        IOSchedulingConfig(
            io_dispatch_mode="traffic_aware_batched",
            batch_size=8,
            batch_dispatch_interval_us=BATCH_DISPATCH_INTERVAL_US,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    (
        "batched8_adaptive_qwrr",
        "queue_wrr",
        IOSchedulingConfig(
            io_dispatch_mode="batched",
            batch_size=8,
            batch_interval_mode="demand_aware",
            batch_dispatch_headroom=ADAPTIVE_BATCH_HEADROOM,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    (
        "traffic_aware8_adaptive_qwrr",
        "queue_wrr",
        IOSchedulingConfig(
            io_dispatch_mode="traffic_aware_batched",
            batch_size=8,
            batch_interval_mode="demand_aware",
            batch_dispatch_headroom=ADAPTIVE_BATCH_HEADROOM,
            qos_queue_max_active_flows=QOS_QUEUE_MAX_ACTIVE_FLOWS,
        ),
    ),
    # ('tb_capped_qwrr',         'queue_wrr', IOSchedulingConfig(token_bucket_enabled=True, token_bucket_pir_cap=True)),
    # ('tb_uncapped_qwrr',       'queue_wrr', IOSchedulingConfig(token_bucket_enabled=True, token_bucket_pir_cap=False)),
    # ('batched8_fair',          'fair',      IOSchedulingConfig(io_dispatch_mode='batched', batch_size=8)),
    # ('traffic_aware8_fair',    'fair',      IOSchedulingConfig(io_dispatch_mode='traffic_aware_batched', batch_size=8)),
]


def configure_qos_experiment(qos_queue_count):
    """Select the supported queue topology and isolate its result artifacts."""
    global QOS_QUEUE_COUNT, QOS_LAYOUT, RESULT_SUFFIX, PICKLE_FILE
    if qos_queue_count == 24:
        qos_layout = QOS_LAYOUT_THREE_TIER
    elif qos_queue_count == 256:
        qos_layout = QOS_LAYOUT_EIGHT_GROUP
    else:
        raise ValueError("qos_queue_count must be 24 or 256")

    QOS_QUEUE_COUNT = qos_queue_count
    QOS_LAYOUT = qos_layout
    RESULT_SUFFIX = f"q{qos_queue_count}_{EXPERIMENT_SUFFIX}"
    PICKLE_FILE = os.path.join(
        RESULTS_DIR,
        f"io_scheduling_sweep_{RESULT_SUFFIX}_pir_uncapped.pkl",
    )
    for _, _, io_sched in MODES:
        io_sched.qos_queue_count = qos_queue_count
        io_sched.qos_layout = qos_layout


# Display labels
LABELS = {
    "baseline_qwrr": "Baseline (queue_wrr)",
    "batched8_qwrr": "Batched(8, fixed 200us) + queue_wrr",
    "traffic_aware8_qwrr": "TA(8, fixed 200us) + queue_wrr",
    "batched8_adaptive_qwrr": "Batched(8, adaptive) + queue_wrr",
    "traffic_aware8_adaptive_qwrr": "TA(8, adaptive) + queue_wrr",
    "tb_capped_qwrr": "TB Capped (80us)",
    "tb_uncapped_qwrr": "TB Uncapped (80us)",
    "baseline_fair": "Baseline (fair)",
    "batched8_fair": "Batched(8) + fair",
    "traffic_aware8_fair": "TA(8) + fair",
}

# Plot styling
COLORS = {
    "baseline_qwrr": "#888888",
    "batched8_qwrr": "#4C72B0",
    "traffic_aware8_qwrr": "#55A868",
    "batched8_adaptive_qwrr": "#DD8452",
    "traffic_aware8_adaptive_qwrr": "#C44E52",
    "tb_capped_qwrr": "#C44E52",
    "tb_uncapped_qwrr": "#DD8452",
    "baseline_fair": "#AAAAAA",
    "batched8_fair": "#8172B3",
    "traffic_aware8_fair": "#937860",
}
MARKERS = {
    "baseline_qwrr": "s",
    "batched8_qwrr": "o",
    "traffic_aware8_qwrr": "D",
    "batched8_adaptive_qwrr": "^",
    "traffic_aware8_adaptive_qwrr": "v",
    "tb_capped_qwrr": "^",
    "tb_uncapped_qwrr": "v",
    "baseline_fair": "P",
    "batched8_fair": "X",
    "traffic_aware8_fair": "p",
}
LINESTYLES = {
    "baseline_qwrr": "-",
    "batched8_qwrr": "-",
    "traffic_aware8_qwrr": "-",
    "batched8_adaptive_qwrr": "--",
    "traffic_aware8_adaptive_qwrr": "--",
    "tb_capped_qwrr": "--",
    "tb_uncapped_qwrr": "--",
    "baseline_fair": ":",
    "batched8_fair": ":",
    "traffic_aware8_fair": ":",
}


def run_sweep(force=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not force and os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            cached = pickle.load(f)

        target_modes = [mode_name for mode_name, _, _ in MODES]
        cached_results = cached["results"]
        seeds_match = list(cached.get("seeds", [])) == list(SEEDS)
        qos_policy_version_matches = (
            cached.get("qos_policy_version") == QOS_POLICY_VERSION
        )
        fair_policy_version_matches = (
            cached.get("fair_policy_version") == FAIR_POLICY_VERSION
        )
        adaptive_batch_config_matches = (
            cached.get("adaptive_batch_policy_version") == ADAPTIVE_BATCH_POLICY_VERSION
            and cached.get("adaptive_batch_headroom") == ADAPTIVE_BATCH_HEADROOM
        )
        qos_config_matches = (
            cached.get("qos_queue_count") == QOS_QUEUE_COUNT
            and cached.get("qos_layout") == QOS_LAYOUT
            and cached.get("batch_dispatch_interval_us") == BATCH_DISPATCH_INTERVAL_US
            and cached.get("qos_queue_max_active_flows") == QOS_QUEUE_MAX_ACTIVE_FLOWS
        )
        qos_cache_sensitive_modes = {
            mode_name
            for mode_name, policy, io_sched in MODES
            if policy in ("queue_wrr", "urgency_driven")
            or io_sched.io_dispatch_mode != "all_at_once"
        }
        fair_modes = {
            mode_name
            for mode_name, policy, _ in MODES
            if policy in ("fair", "demand_driven")
        }
        adaptive_modes = {
            mode_name
            for mode_name, _, io_sched in MODES
            if io_sched.batch_interval_mode == "demand_aware"
        }
        missing_cases = [
            (ssu, ls_ratio, mode_name)
            for ssu in SSU_LIST
            for ls_ratio in LS_RATIO_LIST
            for mode_name in target_modes
            if not seeds_match
            or (
                mode_name in qos_cache_sensitive_modes
                and (not qos_policy_version_matches or not qos_config_matches)
            )
            or (mode_name in fair_modes and not fair_policy_version_matches)
            or (mode_name in adaptive_modes and not adaptive_batch_config_matches)
            or (ssu, ls_ratio, mode_name) not in cached_results
            or len(cached_results[(ssu, ls_ratio, mode_name)]) != len(SEEDS)
        ]

        if not missing_cases:
            return _active_sweep_view(cached)

        bw = load_bw_table_cache(num_npu=NPU_COUNT)
        results = dict(cached["results"])
        mode_names = list(
            dict.fromkeys(list(cached["mode_names"]) + [m for m, _, _ in MODES])
        )
        mode_policies = dict(cached.get("mode_policies", {}))
        for m, pol, _ in MODES:
            mode_policies[m] = pol
        ssu_list = sorted(set(cached["ssu_list"]) | set(SSU_LIST))
        ls_ratio_list = sorted(set(cached["ls_ratio_list"]) | set(LS_RATIO_LIST))
        _run_missing(bw, results, mode_policies, missing_cases)

        data = {
            "ssu_list": ssu_list,
            "ls_ratio_list": ls_ratio_list,
            "mode_names": mode_names,
            "results": results,
            "seeds": SEEDS,
            "mode_policies": mode_policies,
            "qos_policy_version": QOS_POLICY_VERSION,
            "fair_policy_version": FAIR_POLICY_VERSION,
            "adaptive_batch_policy_version": ADAPTIVE_BATCH_POLICY_VERSION,
            "adaptive_batch_headroom": ADAPTIVE_BATCH_HEADROOM,
            "qos_queue_count": QOS_QUEUE_COUNT,
            "qos_layout": QOS_LAYOUT,
            "batch_dispatch_interval_us": BATCH_DISPATCH_INTERVAL_US,
            "qos_queue_max_active_flows": QOS_QUEUE_MAX_ACTIVE_FLOWS,
        }
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
        return _active_sweep_view(data)

    bw = load_bw_table_cache(num_npu=NPU_COUNT)
    results = {}
    mode_names = [m for m, _, _ in MODES]
    mode_policies = {m: pol for m, pol, _ in MODES}
    ssu_list = SSU_LIST
    ls_ratio_list = LS_RATIO_LIST
    cases = [
        (ssu, ls_ratio, mode_name)
        for ssu in SSU_LIST
        for ls_ratio in LS_RATIO_LIST
        for mode_name in mode_names
    ]
    _run_missing(bw, results, mode_policies, cases)

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(
            {
                "ssu_list": ssu_list,
                "ls_ratio_list": ls_ratio_list,
                "mode_names": mode_names,
                "results": results,
                "seeds": SEEDS,
                "mode_policies": mode_policies,
                "qos_policy_version": QOS_POLICY_VERSION,
                "fair_policy_version": FAIR_POLICY_VERSION,
                "adaptive_batch_policy_version": ADAPTIVE_BATCH_POLICY_VERSION,
                "adaptive_batch_headroom": ADAPTIVE_BATCH_HEADROOM,
                "qos_queue_count": QOS_QUEUE_COUNT,
                "qos_layout": QOS_LAYOUT,
                "batch_dispatch_interval_us": BATCH_DISPATCH_INTERVAL_US,
                "qos_queue_max_active_flows": QOS_QUEUE_MAX_ACTIVE_FLOWS,
            },
            f,
        )
    return {
        "ssu_list": ssu_list,
        "ls_ratio_list": ls_ratio_list,
        "mode_names": mode_names,
        "results": results,
        "seeds": SEEDS,
        "mode_policies": mode_policies,
        "qos_policy_version": QOS_POLICY_VERSION,
        "fair_policy_version": FAIR_POLICY_VERSION,
        "adaptive_batch_policy_version": ADAPTIVE_BATCH_POLICY_VERSION,
        "adaptive_batch_headroom": ADAPTIVE_BATCH_HEADROOM,
        "qos_queue_count": QOS_QUEUE_COUNT,
        "qos_layout": QOS_LAYOUT,
        "batch_dispatch_interval_us": BATCH_DISPATCH_INTERVAL_US,
        "qos_queue_max_active_flows": QOS_QUEUE_MAX_ACTIVE_FLOWS,
    }


def _active_sweep_view(data):
    """Return only the dimensions selected for this invocation of main.py."""
    active_modes = [mode_name for mode_name, _, _ in MODES]
    return {
        "ssu_list": list(SSU_LIST),
        "ls_ratio_list": list(LS_RATIO_LIST),
        "mode_names": active_modes,
        "results": data["results"],
        "seeds": list(SEEDS),
        "mode_policies": {
            mode_name: data.get("mode_policies", {}).get(mode_name, policy)
            for mode_name, policy, _ in MODES
        },
        "qos_policy_version": data.get("qos_policy_version"),
        "fair_policy_version": data.get("fair_policy_version"),
        "adaptive_batch_policy_version": data.get("adaptive_batch_policy_version"),
        "adaptive_batch_headroom": data.get("adaptive_batch_headroom"),
        "qos_queue_count": data.get("qos_queue_count"),
        "qos_layout": data.get("qos_layout"),
        "batch_dispatch_interval_us": data.get("batch_dispatch_interval_us"),
        "qos_queue_max_active_flows": data.get("qos_queue_max_active_flows"),
    }


def _run_missing(bw, results, mode_policies, missing_cases):
    mode_scheds = {m: cfg for m, _, cfg in MODES}
    total = len(missing_cases) * len(SEEDS)
    done = 0
    for ssu, ls_ratio, mode_name in missing_cases:
        if mode_name not in mode_scheds:
            continue
        io_sched = mode_scheds[mode_name]
        policy = mode_policies[mode_name]
        key = (ssu, ls_ratio, mode_name)
        utils = []
        for seed in SEEDS:
            npus, _ = simulate_continuous(
                bw_table=bw,
                policy=policy,
                num_requests_per_npu=N_REQ,
                num_npu=NPU_COUNT,
                num_disk=ssu,
                n_layers=N_LAYERS,
                ls_ratio=ls_ratio,
                rng=np.random.RandomState(seed),
                io_sched=io_sched,
            )
            valid = [n for n in npus if n.compute_end_time > 0]
            avg_util = sum(
                n.total_compute_ms / n.compute_end_time * 100 for n in valid
            ) / max(1, len(valid))
            utils.append(avg_util)
            done += 1
            print(
                "  [%d/%d] ssu=%3d ls=%.2f %-25s seed=%d util=%.1f%%"
                % (done, total, ssu, ls_ratio, mode_name, seed, avg_util)
            )
        results[key] = utils
        print(
            "  ssu=%3d ls=%.2f %-25s mean=%.1f%%"
            % (ssu, ls_ratio, mode_name, np.mean(utils))
        )


def plot(data):
    ssu_list = data["ssu_list"]
    ls_ratio_list = data["ls_ratio_list"]
    mode_names = data["mode_names"]
    results = data["results"]

    # --- Chart 1: queue_wrr modes at ls_ratio=0.5 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    qwrr_modes = [m for m in mode_names if "qwrr" in m]
    ls_default = 0.5
    for mode_name in qwrr_modes:
        means = [
            np.mean(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        stds = [
            np.std(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        ax.errorbar(
            ssu_list,
            means,
            yerr=stds,
            label=LABELS.get(mode_name, mode_name),
            color=COLORS.get(mode_name, "#333333"),
            marker=MARKERS.get(mode_name, "o"),
            linestyle=LINESTYLES.get(mode_name, "-"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )
    ax.set_xlabel("SSU (Disk) Count", fontsize=13)
    ax.set_ylabel("NPU Utilization (%)", fontsize=13)
    ax.set_title(
        "IO Scheduling (queue_wrr): NPU Utilization vs SSU\n"
        "(NPU=%d, layers=%d, ls_ratio=%.2f, QoS queues/SSU=%d)"
        % (NPU_COUNT, N_LAYERS, ls_default, QOS_QUEUE_COUNT),
        fontsize=14,
    )
    ax.set_xticks(ssu_list)
    if qwrr_modes:
        ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    ax2 = ax.twiny()
    ratios = [NPU_COUNT / ssu for ssu in ssu_list]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ssu_list)
    ax2.set_xticklabels([f"{r:.1f}" for r in ratios])
    ax2.set_xlabel("NPU/SSU Ratio", fontsize=13)

    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, f"io_sched_16L_qwrr_{RESULT_SUFFIX}.png")
    if qwrr_modes:
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        print("Chart saved to %s" % out1)
    else:
        plt.close(fig)

    # --- Chart 2: fair modes at ls_ratio=0.5 ---
    fig2, ax2p = plt.subplots(figsize=(12, 7))
    fair_modes = [m for m in mode_names if m.endswith("_fair")]
    for mode_name in fair_modes:
        means = [
            np.mean(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        stds = [
            np.std(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        ax2p.errorbar(
            ssu_list,
            means,
            yerr=stds,
            label=LABELS.get(mode_name, mode_name),
            color=COLORS.get(mode_name, "#333333"),
            marker=MARKERS.get(mode_name, "o"),
            linestyle=LINESTYLES.get(mode_name, "-"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )
    ax2p.set_xlabel("SSU (Disk) Count", fontsize=13)
    ax2p.set_ylabel("NPU Utilization (%)", fontsize=13)
    ax2p.set_title(
        "IO Scheduling (fair): NPU Utilization vs SSU\n(NPU=%d, layers=%d, ls_ratio=%.2f)"
        % (NPU_COUNT, N_LAYERS, ls_default),
        fontsize=14,
    )
    ax2p.set_xticks(ssu_list)
    ax2p.legend(fontsize=10, loc="lower right")
    ax2p.grid(True, alpha=0.3)
    ax2p.set_ylim(0, 100)

    ax3 = ax2p.twiny()
    ax3.set_xlim(ax2p.get_xlim())
    ax3.set_xticks(ssu_list)
    ax3.set_xticklabels([f"{r:.1f}" for r in ratios])
    ax3.set_xlabel("NPU/SSU Ratio", fontsize=13)

    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, f"io_sched_16L_fair_{RESULT_SUFFIX}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print("Chart saved to %s" % out2)

    # --- Chart 3: ls_ratio sweep for key modes at SSU=56 ---
    fig3, ax3p = plt.subplots(figsize=(12, 7))
    key_modes = [
        "baseline_qwrr",
        "batched8_qwrr",
        "traffic_aware8_qwrr",
        "batched8_adaptive_qwrr",
        "traffic_aware8_adaptive_qwrr",
        "tb_capped_qwrr",
        "baseline_fair",
        "batched8_fair",
    ]
    ssu_fixed = 56
    for mode_name in key_modes:
        if mode_name not in mode_names:
            continue
        means = [
            np.mean(results.get((ssu_fixed, lr, mode_name), [0]))
            for lr in ls_ratio_list
        ]
        stds = [
            np.std(results.get((ssu_fixed, lr, mode_name), [0])) for lr in ls_ratio_list
        ]
        ax3p.errorbar(
            ls_ratio_list,
            means,
            yerr=stds,
            label=LABELS.get(mode_name, mode_name),
            color=COLORS.get(mode_name, "#333333"),
            marker=MARKERS.get(mode_name, "o"),
            linestyle=LINESTYLES.get(mode_name, "-"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )
    ax3p.set_xlabel("LS Ratio", fontsize=13)
    ax3p.set_ylabel("NPU Utilization (%)", fontsize=13)
    ax3p.set_title(
        "IO Scheduling: NPU Utilization vs LS Ratio\n"
        "(NPU=%d, SSU=%d, layers=%d, QoS queues/SSU=%d)"
        % (NPU_COUNT, ssu_fixed, N_LAYERS, QOS_QUEUE_COUNT),
        fontsize=14,
    )
    ax3p.set_xticks(ls_ratio_list)
    ax3p.legend(fontsize=10, loc="lower right")
    ax3p.grid(True, alpha=0.3)
    ax3p.set_ylim(0, 100)
    plt.tight_layout()
    out3 = os.path.join(RESULTS_DIR, f"io_sched_16L_ls_ratio_{RESULT_SUFFIX}.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print("Chart saved to %s" % out3)

    # --- Chart 4: All modes combined at ls_ratio=0.5 ---
    fig4, ax4p = plt.subplots(figsize=(14, 8))
    for mode_name in mode_names:
        means = [
            np.mean(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        stds = [
            np.std(results.get((ssu, ls_default, mode_name), [0])) for ssu in ssu_list
        ]
        ax4p.errorbar(
            ssu_list,
            means,
            yerr=stds,
            label=LABELS.get(mode_name, mode_name),
            color=COLORS.get(mode_name, "#333333"),
            marker=MARKERS.get(mode_name, "o"),
            linestyle=LINESTYLES.get(mode_name, "-"),
            linewidth=2,
            markersize=7,
            capsize=3,
        )
    ax4p.set_xlabel("SSU (Disk) Count", fontsize=13)
    ax4p.set_ylabel("NPU Utilization (%)", fontsize=13)
    ax4p.set_title(
        "IO Scheduling All Modes: NPU Utilization vs SSU\n"
        "(NPU=%d, layers=%d, ls_ratio=%.2f, QoS queues/SSU=%d)"
        % (NPU_COUNT, N_LAYERS, ls_default, QOS_QUEUE_COUNT),
        fontsize=14,
    )
    ax4p.set_xticks(ssu_list)
    ax4p.legend(fontsize=9, loc="lower right", ncol=2)
    ax4p.grid(True, alpha=0.3)
    ax4p.set_ylim(0, 100)

    ax5 = ax4p.twiny()
    ax5.set_xlim(ax4p.get_xlim())
    ax5.set_xticks(ssu_list)
    ax5.set_xticklabels([f"{r:.1f}" for r in ratios])
    ax5.set_xlabel("NPU/SSU Ratio", fontsize=13)

    plt.tight_layout()
    out4 = os.path.join(RESULTS_DIR, f"io_sched_16L_all_{RESULT_SUFFIX}.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    print("Chart saved to %s" % out4)

    # Print summary table at ls_ratio=0.5
    print("\n=== Summary Table (ls_ratio=0.5, layers=%d) ===" % N_LAYERS)
    header = "%-6s" % "SSU"
    for mode_name in mode_names:
        label = LABELS.get(mode_name, mode_name)
        header += "  %-18s" % label[:18]
    print(header)
    for ssu in ssu_list:
        row = "%-6d" % ssu
        for mode_name in mode_names:
            mean = np.mean(results.get((ssu, ls_default, mode_name), [0]))
            row += "  %-18.1f" % mean
        print(row)

    # Print delta table: batched vs baseline within same policy
    print("\n=== Delta: Batched vs Baseline (ls_ratio=0.5) ===")
    for policy_suffix, policy_label in [("qwrr", "queue_wrr"), ("fair", "fair")]:
        baseline_key = "baseline_%s" % policy_suffix
        delta_modes = [
            ("batched8_%s" % policy_suffix, "batched delta"),
            ("traffic_aware8_%s" % policy_suffix, "TA delta"),
            ("batched8_adaptive_%s" % policy_suffix, "adaptive batch delta"),
            (
                "traffic_aware8_adaptive_%s" % policy_suffix,
                "adaptive TA delta",
            ),
            ("tb_capped_%s" % policy_suffix, "TB capped delta"),
        ]
        delta_modes = [item for item in delta_modes if item[0] in mode_names]
        if baseline_key not in mode_names or not delta_modes:
            continue
        print("\n  %s:" % policy_label)
        header = "  %-6s  %-10s" % ("SSU", "baseline")
        for _, label in delta_modes:
            header += "  %-16s" % label
        print(header)
        for ssu in ssu_list:
            b = np.mean(results.get((ssu, ls_default, baseline_key), [0]))
            row = "  %-6d  %-10.1f" % (ssu, b)
            for mode_name, _ in delta_modes:
                delta = np.mean(results[(ssu, ls_default, mode_name)]) - b
                row += "  %-+16.1f" % delta
            print(row)

    # Print ls_ratio table for key modes at SSU=56
    print("\n=== ls_ratio Sweep at SSU=56 ===")
    header2 = "%-8s" % "ls_ratio"
    for mode_name in key_modes:
        if mode_name in mode_names:
            header2 += "  %-18s" % LABELS.get(mode_name, mode_name)[:18]
    print(header2)
    for lr in ls_ratio_list:
        row = "%-8.2f" % lr
        for mode_name in key_modes:
            if mode_name in mode_names:
                mean = np.mean(results.get((ssu_fixed, lr, mode_name), [0]))
                row += "  %-18.1f" % mean
        print(row)


def plot_queue_count_comparison():
    """Compare q24 and q256 QoS modes after both sweep caches exist."""
    queue_counts = (24, 256)
    comparison_modes = (
        "baseline_qwrr",
        "batched8_qwrr",
        "traffic_aware8_qwrr",
        "batched8_adaptive_qwrr",
        "traffic_aware8_adaptive_qwrr",
    )
    datasets = {}
    for queue_count in queue_counts:
        cache_path = os.path.join(
            RESULTS_DIR,
            "io_scheduling_sweep_"
            f"q{queue_count}_{EXPERIMENT_SUFFIX}_pir_uncapped.pkl",
        )
        if not os.path.exists(cache_path):
            return
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if (
            data.get("qos_policy_version") != QOS_POLICY_VERSION
            or data.get("qos_queue_count") != queue_count
            or data.get("batch_dispatch_interval_us") != BATCH_DISPATCH_INTERVAL_US
            or data.get("qos_queue_max_active_flows") != QOS_QUEUE_MAX_ACTIVE_FLOWS
            or data.get("adaptive_batch_policy_version")
            != ADAPTIVE_BATCH_POLICY_VERSION
            or data.get("adaptive_batch_headroom") != ADAPTIVE_BATCH_HEADROOM
        ):
            return
        required_keys = {
            (ssu, 0.5, mode_name) for ssu in SSU_LIST for mode_name in comparison_modes
        }
        if not required_keys.issubset(data["results"]):
            return
        datasets[queue_count] = data

    fig, ax = plt.subplots(figsize=(13, 8))
    for queue_count in queue_counts:
        data = datasets[queue_count]
        results = data["results"]
        for mode_name in comparison_modes:
            means = [np.mean(results[(ssu, 0.5, mode_name)]) for ssu in SSU_LIST]
            stds = [np.std(results[(ssu, 0.5, mode_name)]) for ssu in SSU_LIST]
            kwargs = {}
            if queue_count == 256:
                kwargs["markerfacecolor"] = "white"
            ax.errorbar(
                SSU_LIST,
                means,
                yerr=stds,
                label=f"q{queue_count} {LABELS[mode_name]}",
                color=COLORS[mode_name],
                marker=MARKERS[mode_name],
                linestyle="-" if queue_count == 24 else "--",
                linewidth=2,
                markersize=8,
                capsize=4,
                **kwargs,
            )

    ax.set_xlabel("SSU (Disk) Count", fontsize=13)
    ax.set_ylabel("NPU Utilization (%)", fontsize=13)
    ax.set_title(
        "QoS Queue Count Comparison: q24 vs q256\n"
        "(NPU=%d, layers=%d, ls_ratio=0.50, PIR uncapped)" % (NPU_COUNT, N_LAYERS),
        fontsize=14,
    )
    ax.set_xticks(SSU_LIST)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    plt.tight_layout()
    output_path = os.path.join(
        RESULTS_DIR,
        f"io_sched_16L_q24_vs_q256_{EXPERIMENT_SUFFIX}.png",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Comparison chart saved to %s" % output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QoS storage simulation sweeps.")
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    parser.add_argument("--fair-only", action="store_true", help="run only fair")
    parser.add_argument(
        "--qos-queues",
        type=int,
        choices=(24, 256),
        default=24,
        help="QoS queues per SSU (default: 24)",
    )
    args = parser.parse_args()
    configure_qos_experiment(args.qos_queues)
    if args.fair_only:
        MODES[:] = [mode for mode in MODES if mode[1] == "fair"]
    data = run_sweep(force=args.force)
    plot(data)
    plot_queue_count_comparison()
