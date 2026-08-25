"""运行带 50 GB/s NPU 聚合上限的 Baseline 与静态 QoS 路由消融。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import (
    NPU_BW_LIMIT,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_STATIC_CIR,
    QOS_ROUTING_CATEGORIES,
    StaticQoSConfig,
    load_bw_table_cache,
    prepare_simulation_inputs,
    simulate_continuous,
)


NUM_NPU = 128  # 客户端/NPU 数量；所有策略和 SSU 点保持一致。
SSU_LIST = (40, 56)  # 默认只运行用户指定的两个主要 SSU 配比点。
LS_RATIO = 0.5  # 请求生成时短/长 NQL 子类的混合比例。
DEFAULT_LAYERS = 16  # 默认使用完整的 16 层 layerwise workload。
WORKLOAD_SEED = 42  # 请求画像抽样 RNG；不参与 placement 或提交排序。
PLACEMENT_SEED_OFFSET = 1_000_003  # 从 workload seed 派生独立 placement seed。
SUBMIT_ORDER_SEED_OFFSET = 2_000_003  # 派生独立的同 timestamp 提交排序 seed。
CLIENT_SUBMIT_BATCH_SIZE = 8  # 每轮每个 NPU 最多提交的 I/O 数。
CLIENT_SUBMIT_INTERVAL_US = 0.0  # 两轮提交的模拟间隔；0 表示无额外发送延迟。
DISK_BW = 40.0  # 每块 SSU 的物理后端带宽，单位 GB/s。
PATH_COUNT = 256  # 每块 SSU 的 QoS Path 数。
GROUP_COUNT = 8  # 每块 SSU 的 QoS Group 数。
PATHS_PER_GROUP = 32  # 每个 Group 的 Path 数。
CATEGORY_CIR_GBPS = (20.0, 4.0, 12.0, 4.0)  # SS/SL/LS/LL 全盘 CIR 预算。
CATEGORY_PATHS_PER_GROUP = (12, 4, 12, 4)  # 每组四类 Path 数。
ROUTING_COMPARISON_SCHEMA_VERSION = 3  # v3 首次明确包含真实 NPU 50 GB/s cap。

# 名称、压力读取间隔、Path 绑定粒度、图例。0/1 与 8/1 只改变遥测刷新频率；
# 8/1 与 8/8 只改变连续 I/O 的 Path 绑定粒度。
ROUTING_VARIANTS = (
    ("layer_once_per_io", 0, 1, "0/1: layer snapshot, per-I/O path"),
    ("refresh8_per_io", 8, 1, "8/1: refresh every 8, per-I/O path"),
    ("refresh8_bind8", 8, 8, "8/8: refresh every 8, bind 8 I/Os"),
)


def category_path_cirs(
    budgets=CATEGORY_CIR_GBPS,
    path_counts=CATEGORY_PATHS_PER_GROUP,
):
    """把四类全盘 CIR 预算均匀展开到 8 组、256 个 Path。"""
    budgets = tuple(float(value) for value in budgets)
    path_counts = tuple(int(value) for value in path_counts)
    if len(budgets) != 4 or len(path_counts) != 4:
        raise ValueError("CIR 和 Path 配置必须按 SS/SL/LS/LL 各给四项")
    if min(budgets) < 0.0 or sum(budgets) > DISK_BW + 1e-12:
        raise ValueError("CIR 总和必须位于 [0, 40] GB/s")
    if min(path_counts) <= 0 or sum(path_counts) != PATHS_PER_GROUP:
        raise ValueError("每组四类 Path 数必须为正且总和为 32")
    result = []
    for _ in range(GROUP_COUNT):
        for budget, count in zip(budgets, path_counts):
            result.extend([budget / (GROUP_COUNT * count)] * count)
    return tuple(result)


def qos_config():
    """构造只在 SSU 初始化时写入一次的静态 CIR/PIR/WRR 配置。"""
    return StaticQoSConfig(
        path_cirs=category_path_cirs(),
        path_pirs=(float("inf"),) * PATH_COUNT,
        path_weights=(1.0,) * PATH_COUNT,
        group_weights=(1.0,) * GROUP_COUNT,
        category_paths_per_group=CATEGORY_PATHS_PER_GROUP,
    )


def _code_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for filename in ("sim.py", "experiment.py"):
        digest.update(filename.encode("utf-8"))
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def _bw_table_fingerprint(bw_table):
    normalized = [
        [list(key), [float(value) for value in bw_table[key]]]
        for key in sorted(bw_table)
    ]
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_roles(seed):
    seed = int(seed)
    return {
        "workload_seed": seed,
        "placement_seed": seed + PLACEMENT_SEED_OFFSET,
        "submit_order_seed": seed + SUBMIT_ORDER_SEED_OFFSET,
    }


def routing_comparison_spec(*, bw_table, ssu_list, n_layers, seed):
    """返回 cache 和结果文件使用的完整、可复现静态口径。"""
    cir_payload = ",".join(
        f"{value:.17g}" for value in category_path_cirs()
    )
    return {
        "schema_version": ROUTING_COMPARISON_SCHEMA_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "bw_table_fingerprint": _bw_table_fingerprint(bw_table),
        "num_npu": NUM_NPU,
        "ssu_list": [int(value) for value in ssu_list],
        "n_layers": int(n_layers),
        "ls_ratio": LS_RATIO,
        "seed_roles": _seed_roles(seed),
        "disk_bw_gbps": DISK_BW,
        "npu_aggregate_cap_gbps": NPU_BW_LIMIT,
        "npu_cap_allocation": "proportional_raw_rate_work_conserving_v1",
        "path_count": PATH_COUNT,
        "group_count": GROUP_COUNT,
        "category_order": list(QOS_ROUTING_CATEGORIES),
        "category_cir_gbps": list(CATEGORY_CIR_GBPS),
        "category_paths_per_group": list(CATEGORY_PATHS_PER_GROUP),
        "path_cirs_sha256": hashlib.sha256(
            cir_payload.encode("ascii")
        ).hexdigest(),
        "pir": "uncapped",
        "path_wrr_weights": "all_1",
        "group_wrr_weights": "all_1",
        "backend_model": "one_nonpreemptive_active_io_per_ssu",
        "baseline_arbitration": "per_npu_fcfs_then_io_rr",
        "qos_arbitration": "cir_then_group_wrr_then_path_wrr_then_rr",
        "pressure_abi": "256_active_plus_pending_io_counts",
        "client_submit_batch_size": CLIENT_SUBMIT_BATCH_SIZE,
        "client_submit_interval_us": CLIENT_SUBMIT_INTERVAL_US,
        "same_timestamp_visibility": "seeded_shuffle_plan_then_immediate_enqueue",
        "routing_variants": [
            {
                "name": name,
                "pressure_read_interval": read_interval,
                "path_binding_batch_size": binding_size,
                "label": label,
            }
            for name, read_interval, binding_size, label in ROUTING_VARIANTS
        ],
    }


def _summary(full):
    """保留比较、诊断和正确性验证需要的结果。"""
    disk_stats = full["disk_stats"]
    total_blocks = sum(row["blocks_enqueued"] for row in disk_stats)
    total_queue_wait = sum(row["total_queue_wait_ms"] for row in disk_stats)
    path_rows = [
        path for disk in disk_stats for path in disk["paths"].values()
    ]
    invariants = {
        "all_requests_completed": (
            full["completed_requests"] == full["total_requests"]
        ),
        "submit_complete_exactly_once": (
            full["block_conservation"]["expected"]
            == full["block_conservation"]["submitted"]
            == full["block_conservation"]["completed"]
        ),
        "placement_targets_preserved": full["block_conservation"][
            "placement_targets_preserved"
        ],
        "bytes_conserved": bool(
            np.isclose(
                full["block_conservation"]["expected_read_gb"],
                full["block_conservation"]["completed_read_gb"],
                rtol=1e-10,
                atol=1e-9,
            )
        ),
        "npu_cap_respected": (
            full["npu_link_peak_effective_bw_gbps"]
            <= NPU_BW_LIMIT + 1e-9
        ),
        "one_backend_io_per_ssu": max(
            (row["max_backend_active_io"] for row in disk_stats), default=0
        )
        <= 1,
        "no_outstanding_at_end": all(
            row["outstanding_blocks"] == 0 for row in disk_stats
        ),
    }
    return {
        "policy": full["policy"],
        "avg_request_compute_fraction": full["avg_request_compute_fraction"],
        "avg_npu_utilization": full["avg_request_compute_fraction"],
        "fleet_npu_compute_utilization": full["fleet_npu_compute_utilization"],
        "request_compute_fraction_jain": full["request_compute_fraction_jain"],
        "makespan_ms": full["makespan_ms"],
        "throughput_requests_per_s": full["throughput_requests_per_s"],
        "category_metrics": full["category_metrics"],
        "npu_link_utilization": full["npu_link_utilization"],
        "npu_link_total_read_gb": full["npu_link_total_read_gb"],
        "npu_link_peak_raw_bw_gbps": full["npu_link_peak_raw_bw_gbps"],
        "npu_link_peak_effective_bw_gbps": full[
            "npu_link_peak_effective_bw_gbps"
        ],
        "npu_link_mean_effective_bw_gbps_per_npu": full[
            "npu_link_mean_effective_bw_gbps_per_npu"
        ],
        "npu_link_cap_hit_fraction": full["npu_link_cap_hit_fraction"],
        "ssu_active_time_utilization_mean": float(
            np.mean([row["active_time_utilization"] for row in disk_stats])
        ),
        "ssu_effective_bandwidth_utilization_mean": float(
            np.mean(
                [row["effective_bandwidth_utilization"] for row in disk_stats]
            )
        ),
        "avg_queue_wait_ms_per_block": (
            total_queue_wait / total_blocks if total_blocks else 0.0
        ),
        "max_queue_wait_ms": max(
            (row["max_queue_wait_ms"] for row in disk_stats), default=0.0
        ),
        "max_path_outstanding_io": max(
            (row["max_outstanding_io"] for row in path_rows), default=0
        ),
        "pressure_reports": sum(row["pressure_reports"] for row in disk_stats),
        "backend_dispatches": sum(
            row["backend_dispatches"] for row in disk_stats
        ),
        "blocks_enqueued": total_blocks,
        "max_backend_active_io": max(
            (row["max_backend_active_io"] for row in disk_stats), default=0
        ),
        "client_submission": full["client_submission"],
        "workload_fingerprint": full["workload_fingerprint"],
        "placement_hash": full["placement_hash"],
        "block_conservation": full["block_conservation"],
        "invariants": invariants,
    }


def _compare(baseline, variant):
    p95_delta = {
        category: (
            variant["category_metrics"][category]["p95_ttft_ms"]
            - baseline["category_metrics"][category]["p95_ttft_ms"]
        )
        for category in QOS_ROUTING_CATEGORIES
    }
    return {
        "utilization_delta_pp": 100.0
        * (
            variant["avg_request_compute_fraction"]
            - baseline["avg_request_compute_fraction"]
        ),
        "fleet_utilization_delta_pp": 100.0
        * (
            variant["fleet_npu_compute_utilization"]
            - baseline["fleet_npu_compute_utilization"]
        ),
        "p95_delta_ms": p95_delta,
        "p95_pass": all(value <= 1e-9 for value in p95_delta.values()),
        "fairness_pass": (
            variant["request_compute_fraction_jain"] + 1e-12
            >= baseline["request_compute_fraction_jain"]
        ),
    }


def _assert_paired(reference, candidate):
    for key in ("workload_fingerprint", "placement_hash"):
        if reference[key] != candidate[key]:
            raise AssertionError(f"配对策略的 {key} 不一致")
    if reference["block_conservation"]["expected_read_gb"] != candidate[
        "block_conservation"
    ]["expected_read_gb"]:
        raise AssertionError("配对策略的 block 总字节不一致")
    if not all(reference["invariants"].values()):
        raise AssertionError("baseline 正确性不变量失败")
    if not all(candidate["invariants"].values()):
        raise AssertionError("QoS 正确性不变量失败")


def run_routing_comparison_case(
    bw_table,
    num_ssu,
    n_layers=DEFAULT_LAYERS,
    seed=WORKLOAD_SEED,
):
    """在一个 SSU 点复用一次 capped baseline，运行 0/1、8/1、8/8。"""
    seeds = _seed_roles(seed)
    prepared = prepare_simulation_inputs(
        bw_table,
        total_requests=NUM_NPU,
        n_layers=int(n_layers),
        num_disk=int(num_ssu),
        ls_ratio=LS_RATIO,
        workload_seed=seeds["workload_seed"],
        placement_seed=seeds["placement_seed"],
        placement_mode="random",
    )
    common = {
        "num_npu": NUM_NPU,
        "num_disk": int(num_ssu),
        "n_layers": int(n_layers),
        "ls_ratio": LS_RATIO,
        "placement_mode": "random",
        "client_submit_batch_size": CLIENT_SUBMIT_BATCH_SIZE,
        "client_submit_interval_us": CLIENT_SUBMIT_INTERVAL_US,
        "workload_seed": seeds["workload_seed"],
        "placement_seed": seeds["placement_seed"],
        "submit_order_seed": seeds["submit_order_seed"],
        "npu_bw_limit": NPU_BW_LIMIT,
        "prepared_inputs": prepared,
    }
    _, baseline_full = simulate_continuous(
        bw_table, policy=POLICY_BASELINE_BYPASS, **common
    )
    baseline = _summary(baseline_full)
    variants = {}
    static_qos = qos_config()
    for name, read_interval, binding_size, label in ROUTING_VARIANTS:
        _, qos_full = simulate_continuous(
            bw_table,
            policy=POLICY_QOS_STATIC_CIR,
            qos_config=static_qos,
            pressure_read_interval=read_interval,
            path_binding_batch_size=binding_size,
            **common,
        )
        qos = _summary(qos_full)
        _assert_paired(baseline, qos)
        variants[name] = {
            "label": label,
            "pressure_read_interval": read_interval,
            "path_binding_batch_size": binding_size,
            **qos,
            "comparison_vs_baseline": _compare(baseline, qos),
        }
    return {
        "num_ssu": int(num_ssu),
        "seed_roles": seeds,
        "baseline": baseline,
        "variants": variants,
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_case_worker(payload):
    bw_table, num_ssu, n_layers, seed = payload
    return run_routing_comparison_case(bw_table, num_ssu, n_layers, seed)


def run_routing_comparison_sweep(
    result_json,
    *,
    ssu_list=SSU_LIST,
    n_layers=DEFAULT_LAYERS,
    seed=WORKLOAD_SEED,
    rerun=False,
    workers=1,
):
    """运行/续跑 capped 路由消融，每完成一个 SSU 点就原子 checkpoint。"""
    ssu_list = tuple(int(value) for value in ssu_list)
    if not ssu_list or min(ssu_list) <= 0 or len(set(ssu_list)) != len(ssu_list):
        raise ValueError("ssu_list 必须由不重复的正整数构成")
    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    bw_table = load_bw_table_cache(num_npu=NUM_NPU)
    spec = routing_comparison_spec(
        bw_table=bw_table,
        ssu_list=ssu_list,
        n_layers=n_layers,
        seed=seed,
    )
    cached = {}
    if result_json.exists() and not rerun:
        data = json.loads(result_json.read_text(encoding="utf-8"))
        if (
            data.get("schema_version") == ROUTING_COMPARISON_SCHEMA_VERSION
            and data.get("experiment") == spec
        ):
            cached = {row["num_ssu"]: row for row in data.get("results", [])}

    def checkpoint():
        _write_json(
            result_json,
            {
                "schema_version": ROUTING_COMPARISON_SCHEMA_VERSION,
                "experiment": spec,
                "results": [cached[key] for key in ssu_list if key in cached],
            },
        )

    payloads = [
        (bw_table, value, n_layers, seed)
        for value in ssu_list
        if value not in cached
    ]
    if workers == 1 or len(payloads) <= 1:
        for payload in payloads:
            print(f"运行 capped 路由消融 SSU={payload[1]} ...", flush=True)
            row = _run_case_worker(payload)
            cached[row["num_ssu"]] = row
            checkpoint()
    else:
        worker_count = min(workers, len(payloads))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_case_worker, payload): payload[1]
                for payload in payloads
            }
            for future in as_completed(futures):
                row = future.result()
                cached[row["num_ssu"]] = row
                checkpoint()
                print(f"完成 capped 路由消融 SSU={row['num_ssu']}", flush=True)
    return {
        "schema_version": ROUTING_COMPARISON_SCHEMA_VERSION,
        "experiment": spec,
        "results": [cached[key] for key in ssu_list],
    }


def plot_routing_comparison(data, output_path):
    rows = data["results"]
    ssus = [row["num_ssu"] for row in rows]
    series = [
        (
            "capped baseline",
            [
                100.0 * row["baseline"]["avg_request_compute_fraction"]
                for row in rows
            ],
            "#777777",
            "s-",
        )
    ]
    colors = ("#009E73", "#0072B2", "#D55E00")
    markers = ("o-", "^-", "D-")
    for (name, _, _, label), color, marker in zip(
        ROUTING_VARIANTS, colors, markers
    ):
        series.append(
            (
                label,
                [
                    100.0
                    * row["variants"][name]["avg_request_compute_fraction"]
                    for row in rows
                ],
                color,
                marker,
            )
        )
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for label, values, color, marker in series:
        axis.plot(
            ssus,
            values,
            marker,
            color=color,
            linewidth=2.2,
            markersize=8,
            label=label,
        )
    axis.set(
        xlabel="Number of SSUs",
        ylabel="Average request compute fraction (%)",
        xticks=ssus,
        ylim=(0, 100),
        title=(
            f"Capped routing comparison ({data['experiment']['n_layers']} layers, "
            f"NPU aggregate cap={NPU_BW_LIMIT:g} GB/s)"
        ),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_routing_comparison(data):
    print(
        "\nSSU  capped baseline  0/1       8/1       8/8       "
        "8/1-0/1   8/1-8/8"
    )
    for row in data["results"]:
        variants = row["variants"]
        baseline = 100.0 * row["baseline"]["avg_request_compute_fraction"]
        layer_once = 100.0 * variants["layer_once_per_io"][
            "avg_request_compute_fraction"
        ]
        refresh8 = 100.0 * variants["refresh8_per_io"][
            "avg_request_compute_fraction"
        ]
        bind8 = 100.0 * variants["refresh8_bind8"][
            "avg_request_compute_fraction"
        ]
        print(
            f"{row['num_ssu']:>3}  {baseline:>14.2f}%  {layer_once:>7.2f}%  "
            f"{refresh8:>7.2f}%  {bind8:>7.2f}%  "
            f"{refresh8-layer_once:>+8.2f}pp  {refresh8-bind8:>+8.2f}pp"
        )


def _parse_ssu_list(value):
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SSU 列表必须是逗号分隔整数") from exc
    if not result or min(result) <= 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("SSU 列表必须是互不重复的正整数")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--seed", type=int, default=WORKLOAD_SEED)
    parser.add_argument(
        "--ssu-list",
        type=_parse_ssu_list,
        default=SSU_LIST,
        help="逗号分隔的 SSU 数，默认 40,56",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="SSU 点并发数；16 层默认 1 以控制内存",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/routing_comparison_capped_v3"),
    )
    # 兼容旧命令；当前程序无论是否传入该开关都只运行 routing comparison。
    parser.add_argument(
        "--compare-routing", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.layers <= 0:
        parser.error("--layers 必须为正整数")
    if args.workers <= 0:
        parser.error("--workers 必须为正整数")
    result_json = args.output_dir / "routing_comparison_capped.json"
    data = run_routing_comparison_sweep(
        result_json,
        ssu_list=args.ssu_list,
        n_layers=args.layers,
        seed=args.seed,
        rerun=args.rerun,
        workers=args.workers,
    )
    plot_routing_comparison(
        data, args.output_dir / "routing_comparison_capped.png"
    )
    print_routing_comparison(data)
    print(f"结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
