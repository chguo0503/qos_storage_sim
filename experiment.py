"""比较共用 SSD→NPU 两级数据面的 baseline 与静态 QoS 8/1。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import (
    ARRIVAL_DELAY_MAX_MS,
    CLIENT_SUBMIT_BATCH_SIZE,
    DISK_BW,
    NPU_BW_LIMIT,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_STATIC_CIR,
    QOS_ROUTING_CATEGORIES,
    PRESSURE_READ_INTERVAL,
    load_bw_table_cache,
    prepare_simulation_inputs,
    simulate_continuous,
)
from strategy_profiles import CURRENT_STATIC


NUM_NPU = 128
SSU_LIST = (40, 56, 80)
LS_RATIO = 0.5
DEFAULT_LAYERS = 16
WORKLOAD_SEED = 42
PLACEMENT_SEED_OFFSET = 1_000_003
SUBMIT_ORDER_SEED_OFFSET = 2_000_003
ARRIVAL_DELAY_SEED_OFFSET = 3_000_003

PATH_COUNT = 256
GROUP_COUNT = 8
CATEGORY_CIR_GBPS = CURRENT_STATIC.category_cir_gbps
CATEGORY_PATHS_PER_GROUP = CURRENT_STATIC.category_paths_per_group
SCHEMA_VERSION = 6


def category_path_cirs():
    """把四类全盘 CIR 均匀展开到 8 组、256 个 Path。"""
    return CURRENT_STATIC.path_cirs()


def qos_config():
    return CURRENT_STATIC.hardware_config()


def _sha256_files(*names):
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in names:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _data_fingerprint(table):
    data = [[list(key), list(table[key])] for key in sorted(table)]
    payload = json.dumps(data, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seeds(seed):
    return {
        "workload": seed,
        "placement": seed + PLACEMENT_SEED_OFFSET,
        "submit_order": seed + SUBMIT_ORDER_SEED_OFFSET,
        "arrival_delay": seed + ARRIVAL_DELAY_SEED_OFFSET,
    }


def routing_comparison_spec(table, ssu_list, n_layers, seed):
    return {
        "schema_version": SCHEMA_VERSION,
        "code_fingerprint": _sha256_files(
            "sim.py",
            "advanced_policies.py",
            "strategy_profiles.py",
            "experiment.py",
        ),
        "data_fingerprint": _data_fingerprint(table),
        "num_npu": NUM_NPU,
        "num_ssu": list(ssu_list),
        "n_layers": n_layers,
        "ls_ratio": LS_RATIO,
        "seeds": _seeds(seed),
        "arrival_delay_ms": [0.0, ARRIVAL_DELAY_MAX_MS],
        "disk_bw_gbps": DISK_BW,
        "npu_cap_gbps": NPU_BW_LIMIT,
        "backend": {
            "model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
            "ssd_service": "io_size_gb / disk_bw_gbps",
            "ssd_max_active_io": 1,
            "npu_service": "io_size_gb / npu_cap_gbps",
            "npu_max_active_io": 1,
            "block_visible_after": "npu_link_completion",
        },
        "qos": {
            "paths": PATH_COUNT,
            "groups": GROUP_COUNT,
            "category_order": list(QOS_ROUTING_CATEGORIES),
            "category_cir_gbps": list(CATEGORY_CIR_GBPS),
            "category_paths_per_group": list(CATEGORY_PATHS_PER_GROUP),
            "pir": "uncapped",
            "pressure_read_interval": PRESSURE_READ_INTERVAL,
            "path_selection": "per_io",
            "client_submit_batch_size": CLIENT_SUBMIT_BATCH_SIZE,
        },
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _summary(full):
    disks = full["disk_stats"]
    links = full["npu_link_stats"]
    blocks = sum(disk["blocks_enqueued"] for disk in disks)
    queue_wait = sum(disk["total_queue_wait_ms"] for disk in disks)
    link_dispatches = sum(link["dispatches"] for link in links)
    link_queue_wait = sum(
        link["avg_queue_wait_ms"] * link["dispatches"] for link in links
    )
    conservation = full["block_conservation"]
    invariants = {
        "requests_completed": full["completed_requests"] == full["total_requests"],
        "blocks_conserved": conservation["expected"]
        == conservation["submitted"]
        == conservation["completed"],
        "placement_preserved": conservation["placement_targets_preserved"],
        "bytes_conserved": math.isclose(
            conservation["expected_read_gb"],
            conservation["ssd_completed_read_gb"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        )
        and math.isclose(
            conservation["expected_read_gb"],
            conservation["completed_read_gb"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        ),
        "npu_cap_respected": full["npu_link_peak_effective_bw_gbps"]
        <= NPU_BW_LIMIT + 1e-9,
        "single_backend_io": max(
            (disk["max_backend_active_io"] for disk in disks), default=0
        )
        <= 1,
        "single_npu_link_io": max(
            (link["max_active_io"] for link in links), default=0
        )
        <= 1,
        "queues_drained": all(disk["outstanding_blocks"] == 0 for disk in disks)
        and all(link["outstanding_io"] == 0 for link in links),
    }
    return {
        "policy": full["policy"],
        "backend_model": full["backend_model"],
        "data_plane_stages": full["data_plane_stages"],
        "backend_capacity_gbps": full["backend_capacity_gbps"],
        "npu_bw_limit_gbps": full["npu_bw_limit_gbps"],
        "avg_request_compute_fraction": full["avg_request_compute_fraction"],
        "fleet_npu_compute_utilization": full["fleet_npu_compute_utilization"],
        "request_compute_fraction_jain": full["request_compute_fraction_jain"],
        "makespan_ms": full["makespan_ms"],
        "throughput_requests_per_s": full["throughput_requests_per_s"],
        "category_metrics": full["category_metrics"],
        "npu_link_utilization": full["npu_link_utilization"],
        "npu_link_peak_raw_bw_gbps": full["npu_link_peak_raw_bw_gbps"],
        "npu_link_peak_effective_bw_gbps": full["npu_link_peak_effective_bw_gbps"],
        "npu_link_cap_hit_fraction": full["npu_link_cap_hit_fraction"],
        "npu_link_busy_fraction": full["npu_link_busy_fraction"],
        "avg_npu_link_queue_wait_ms": (
            link_queue_wait / link_dispatches if link_dispatches else 0.0
        ),
        "max_npu_link_queue_wait_ms": max(
            (link["max_queue_wait_ms"] for link in links), default=0.0
        ),
        "max_npu_link_outstanding_io": max(
            (link["max_outstanding_io"] for link in links), default=0
        ),
        "ssu_active_time_utilization": _mean(
            [disk["active_time_utilization"] for disk in disks]
        ),
        "ssu_effective_bandwidth_utilization": _mean(
            [disk["effective_bandwidth_utilization"] for disk in disks]
        ),
        "avg_queue_wait_ms_per_block": queue_wait / blocks if blocks else 0.0,
        "max_queue_wait_ms": max(
            (disk["max_queue_wait_ms"] for disk in disks), default=0.0
        ),
        "max_path_outstanding_io": max(
            (disk["max_path_outstanding_io"] for disk in disks), default=0
        ),
        "pressure_reports": sum(disk["pressure_reports"] for disk in disks),
        "pressure_telemetry_mb": sum(
            disk["pressure_reports"] for disk in disks
        )
        * 256
        * 4
        / 1_000_000,
        "backend_dispatches": sum(disk["backend_dispatches"] for disk in disks),
        "blocks_enqueued": blocks,
        "workload_fingerprint": full["workload_fingerprint"],
        "placement_hash": full["placement_hash"],
        "block_conservation": conservation,
        "invariants": invariants,
        "request_metrics": full["request_metrics"],
    }


def _comparison(baseline, qos):
    p95_delta = {
        category: qos["category_metrics"][category]["p95_ttft_ms"]
        - baseline["category_metrics"][category]["p95_ttft_ms"]
        for category in QOS_ROUTING_CATEGORIES
    }
    return {
        "request_compute_fraction_delta_pp": 100
        * (
            qos["avg_request_compute_fraction"]
            - baseline["avg_request_compute_fraction"]
        ),
        "fleet_compute_utilization_delta_pp": 100
        * (
            qos["fleet_npu_compute_utilization"]
            - baseline["fleet_npu_compute_utilization"]
        ),
        "p95_delta_ms": p95_delta,
        "p95_pass": all(delta <= 1e-9 for delta in p95_delta.values()),
        "fairness_pass": qos["request_compute_fraction_jain"]
        >= baseline["request_compute_fraction_jain"] - 1e-12,
    }


def run_routing_comparison_case(
    table, num_ssu, n_layers=DEFAULT_LAYERS, seed=WORKLOAD_SEED
):
    seeds = _seeds(seed)
    prepared = prepare_simulation_inputs(
        table,
        total_requests=NUM_NPU,
        n_layers=n_layers,
        num_disk=num_ssu,
        ls_ratio=LS_RATIO,
        workload_seed=seeds["workload"],
        placement_seed=seeds["placement"],
        arrival_delay_seed=seeds["arrival_delay"],
        arrival_delay_max_ms=ARRIVAL_DELAY_MAX_MS,
    )
    common = dict(
        num_npu=NUM_NPU,
        num_disk=num_ssu,
        n_layers=n_layers,
        submit_order_seed=seeds["submit_order"],
        prepared_inputs=prepared,
    )
    _, baseline_full = simulate_continuous(
        table, policy=POLICY_BASELINE_BYPASS, **common
    )
    _, qos_full = simulate_continuous(
        table,
        policy=POLICY_QOS_STATIC_CIR,
        qos_config=qos_config(),
        **common,
    )
    baseline = _summary(baseline_full)
    qos = _summary(qos_full)
    assert baseline["workload_fingerprint"] == qos["workload_fingerprint"]
    assert baseline["placement_hash"] == qos["placement_hash"]
    assert baseline["backend_model"] == qos["backend_model"]
    assert baseline["data_plane_stages"] == qos["data_plane_stages"]
    assert baseline["backend_capacity_gbps"] == qos["backend_capacity_gbps"]
    assert baseline["npu_bw_limit_gbps"] == qos["npu_bw_limit_gbps"]
    assert all(baseline["invariants"].values())
    assert all(qos["invariants"].values())
    qos["comparison_vs_baseline"] = _comparison(baseline, qos)
    return {
        "num_ssu": num_ssu,
        "seeds": seeds,
        "baseline": baseline,
        "qos": qos,
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _run_case(args):
    return run_routing_comparison_case(*args)


def run_sweep(
    result_path,
    ssu_list=SSU_LIST,
    n_layers=DEFAULT_LAYERS,
    seed=WORKLOAD_SEED,
    workers=1,
    rerun=False,
):
    ssu_list = tuple(ssu_list)
    table = load_bw_table_cache(num_npu=NUM_NPU)
    spec = routing_comparison_spec(table, ssu_list, n_layers, seed)
    rows = {}
    if result_path.exists() and not rerun:
        cached = json.loads(result_path.read_text())
        if cached.get("experiment") == spec:
            rows = {row["num_ssu"]: row for row in cached["results"]}

    def checkpoint():
        _write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": spec,
                "results": [rows[ssu] for ssu in ssu_list if ssu in rows],
            },
        )

    pending = [(table, ssu, n_layers, seed) for ssu in ssu_list if ssu not in rows]
    if workers == 1 or len(pending) <= 1:
        for args in pending:
            print(f"运行 SSU={args[1]} ...", flush=True)
            row = _run_case(args)
            rows[row["num_ssu"]] = row
            checkpoint()
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = [pool.submit(_run_case, args) for args in pending]
            for future in as_completed(futures):
                row = future.result()
                rows[row["num_ssu"]] = row
                checkpoint()
                print(f"完成 SSU={row['num_ssu']}", flush=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": spec,
        "results": [rows[ssu] for ssu in ssu_list],
    }


def plot_results(data, output_path):
    rows = data["results"]
    ssus = [row["num_ssu"] for row in rows]
    baseline = [100 * row["baseline"]["avg_request_compute_fraction"] for row in rows]
    qos = [100 * row["qos"]["avg_request_compute_fraction"] for row in rows]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(ssus, baseline, "s-", linewidth=2.2, label="Baseline")
    axis.plot(ssus, qos, "o-", linewidth=2.2, label="QoS 8/1")
    axis.set(
        xlabel="Number of SSUs",
        ylabel="Average request compute fraction (%)",
        xticks=ssus,
        ylim=(0, 100),
        title=(
            f"{data['experiment']['n_layers']}-layer routing on shared "
            f"SSD→NPU data plane"
        ),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_results(data):
    print("\nSSU  Baseline  QoS 8/1  Delta")
    for row in data["results"]:
        baseline = 100 * row["baseline"]["avg_request_compute_fraction"]
        qos = 100 * row["qos"]["avg_request_compute_fraction"]
        print(
            f"{row['num_ssu']:>3}  {baseline:>7.2f}%  {qos:>7.2f}%  "
            f"{qos - baseline:>+7.2f}pp"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--seed", type=int, default=WORKLOAD_SEED)
    parser.add_argument(
        "--ssu-list",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=SSU_LIST,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/routing_comparison")
    )
    args = parser.parse_args()
    result_path = args.output_dir / "results.json"
    data = run_sweep(
        result_path,
        ssu_list=args.ssu_list,
        n_layers=args.layers,
        seed=args.seed,
        workers=args.workers,
        rerun=args.rerun,
    )
    plot_results(data, args.output_dir / "comparison.png")
    print_results(data)


if __name__ == "__main__":
    main()
