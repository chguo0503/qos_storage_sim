"""Baseline、静态 QoS 与在线动态 CIR 共用两级数据面的离散事件仿真。

所有策略共享 workload、placement、I/O 提交节奏和数据面上限。静态 QoS
保持初始化 CIR 不变；动态策略只在非抢占命令边界原子更新下一轮 CIR。
策略选出命令后都先使用单命令 SSD 40 GB/s 服务，再进入每 NPU
独立的 50 GB/s FCFS 接收队列；只有链路完成后 block 才对计算可见。
"""

from __future__ import annotations

import ast
import bisect
from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq
import hashlib
import json
import math
import os
import struct
from typing import Optional

import numpy as np

from advanced_policies import (
    capped_input_demands,
    capped_proportional_demands,
    global_link_aware_priority,
    max_min_work_conserving_rates,
    omniscient_edf_key,
    proportional_capacity_grants,
    slack_link_guarded_demands,
)


# ---------------------------------------------------------------------------
# 实验常量
# ---------------------------------------------------------------------------

NUM_NPU = 32
NUM_DISK = 8
DISK_BW = 40.0
NPU_BW_LIMIT = 50.0
ARRIVAL_DELAY_MAX_MS = 5.0
PRESSURE_READ_INTERVAL = 8
CLIENT_SUBMIT_BATCH_SIZE = 8
SIM_N_LAYERS = 8
BLOCK_SIZE = 128

SEQ_SHORT_BOUNDARY = 80
NQL_LONG_BOUNDARY = 512

POLICY_BASELINE_BYPASS = "baseline_bypass"
POLICY_QOS_STATIC_CIR = "qos_static_cir"
POLICY_QOS_DYNAMIC_JOINT_CIR = "qos_dynamic_joint_cir"
POLICY_QOS_DYNAMIC_CIR = POLICY_QOS_DYNAMIC_JOINT_CIR
POLICY_QOS_DEMAND_MAXMIN = "qos_demand_maxmin"
POLICY_PER_SSD_FULL_VISIBLE_EDF = "per_ssd_full_visible_edf"
POLICY_OMNISCIENT_EDF = POLICY_PER_SSD_FULL_VISIBLE_EDF
POLICY_GLOBAL_LINK_AWARE = "global_link_aware_online"

QOS_POLICIES = (POLICY_QOS_STATIC_CIR,)
PATH_QOS_POLICIES = QOS_POLICIES + (POLICY_QOS_DYNAMIC_JOINT_CIR,)
SUPPORTED_POLICIES = (
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_STATIC_CIR,
    POLICY_QOS_DYNAMIC_JOINT_CIR,
    POLICY_QOS_DEMAND_MAXMIN,
    POLICY_PER_SSD_FULL_VISIBLE_EDF,
    POLICY_GLOBAL_LINK_AWARE,
)

RESULT_SCHEMA_VERSION = 5

QOS_ROUTING_CATEGORIES = ("SS", "SL", "LS", "LL")
PATH_SELECTION_PRESSURE_AWARE = "pressure_aware"
PATH_SELECTION_STATELESS_RR = "stateless_round_robin"
PATH_SELECTION_FIXED_PATH_ZERO = "fixed_path_zero"

# 事件编号同时决定相同时间戳下的处理顺序：编号越小，越先处理。
COMPUTE_DONE = 0
DISK_COMPLETION = 1
NPU_LINK_COMPLETION = 2
CLIENT_SUBMISSION = 3
DISK_SCHEDULE = 4

_EPS = 1e-12


@dataclass(frozen=True)
class ClientIOConfig:
    """客户端提交粒度和 Path 压力刷新方式。"""

    name: str
    pressure_window_io: Optional[int]
    submit_batch_size: int = CLIENT_SUBMIT_BATCH_SIZE
    path_pool_mode: str = "category_shared"
    issue_interval_us: float = 0.0
    path_selection_mode: str = PATH_SELECTION_PRESSURE_AWARE


DYNAMIC_CIR_DEMAND_PROPORTIONAL = "demand_proportional"
DYNAMIC_CIR_SLACK_LINK_GUARDED = "slack_link_guarded"
DYNAMIC_CIR_CONTROL_MODES = (
    DYNAMIC_CIR_DEMAND_PROPORTIONAL,
    DYNAMIC_CIR_SLACK_LINK_GUARDED,
)
DYNAMIC_ROUTE_LEAST_WORK = "least_projected_work"
DYNAMIC_ROUTE_FIXED = "fixed_owned_path"
DYNAMIC_ROUTE_MODES = (DYNAMIC_ROUTE_LEAST_WORK, DYNAMIC_ROUTE_FIXED)


@dataclass(frozen=True)
class DynamicCIRPolicyConfig:
    """Fixed control-plane semantics for the online dynamic-CIR policy."""

    mode: str = DYNAMIC_CIR_DEMAND_PROPORTIONAL
    paths_per_npu: int = 2
    routing_mode: str = DYNAMIC_ROUTE_LEAST_WORK

    def __post_init__(self):
        if self.mode not in DYNAMIC_CIR_CONTROL_MODES:
            raise ValueError("unknown dynamic CIR control mode")
        if self.paths_per_npu != 2:
            raise ValueError("dynamic CIR hardware uses exactly two Paths per NPU")
        if self.routing_mode not in DYNAMIC_ROUTE_MODES:
            raise ValueError("unknown dynamic Path routing mode")


DEFAULT_DYNAMIC_CIR_POLICY_CONFIG = DynamicCIRPolicyConfig()


QOS_REFRESH_EVERY_IO = ClientIOConfig("per_io_live", 1)
QOS_REFRESH_EVERY_8_IO = ClientIOConfig("refresh8", PRESSURE_READ_INTERVAL)
QOS_SNAPSHOT_PER_LAYER = ClientIOConfig("per_layer_snapshot", None)
DEFAULT_CLIENT_IO_CONFIG = QOS_REFRESH_EVERY_8_IO


def ssd_command_service_time_ms(size_gb, bandwidth_gbps):
    """Baseline/QoS 共用的单条 SSD 命令数据面时间。"""
    size_gb = float(size_gb)
    bandwidth_gbps = float(bandwidth_gbps)
    if size_gb <= 0.0 or bandwidth_gbps <= 0.0:
        raise ValueError("I/O 大小和 SSD 带宽必须为正")
    return size_gb / bandwidth_gbps * 1000.0


def npu_link_service_time_ms(size_gb, bandwidth_gbps=NPU_BW_LIMIT):
    """一条已离开 SSD 的数据在 NPU 接收队列中的服务时间。"""
    size_gb = float(size_gb)
    bandwidth_gbps = float(bandwidth_gbps)
    if size_gb <= 0.0 or bandwidth_gbps <= 0.0:
        raise ValueError("I/O 大小和 NPU 带宽必须为正")
    return size_gb / bandwidth_gbps * 1000.0


# ---------------------------------------------------------------------------
# 工作负载生成与数据块放置
# ---------------------------------------------------------------------------


def load_bw_table_cache(results_dir=None, num_npu=None):
    """读取示例程序使用的请求画像表（带宽、计算时间和每层 KV 大小）。"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = results_dir or os.path.join(project_dir, "results")
    num_npu = NUM_NPU if num_npu is None else num_npu
    cache_file = os.path.join(results_dir, f"bw_table_cache_v2_{num_npu}npu.npz")

    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as cached:
            raw = cached["table"].item()
        source = cache_file
    else:
        source = os.path.join(project_dir, "data")
        with open(source, encoding="utf-8") as file:
            raw = ast.literal_eval(file.read())

    table = {}
    for raw_key, raw_value in raw.items():
        key = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        values = tuple(raw_value)
        if len(values) == 3:
            required_bw, per_layer_us, ttft_ms = values
            per_layer_kv_gb = required_bw * per_layer_us / 1e6
            values = (required_bw, per_layer_us, ttft_ms, per_layer_kv_gb)
        table[key] = values
    print(f"  已从 {source} 加载 {len(table)} 条请求画像")
    return table


def classify_request(seq_len_k, nql):
    """根据输入序列长度和 NQL，把请求分成 SS、SL、LS 或 LL。"""
    short_sequence = seq_len_k <= SEQ_SHORT_BOUNDARY
    long_nql = nql >= NQL_LONG_BOUNDARY
    if short_sequence and not long_nql:
        return "SS"
    if short_sequence and long_nql:
        return "SL"
    if not short_sequence and not long_nql:
        return "LS"
    return "LL"


def _load_from_key(bw_table, key, request_id):
    required_bw, per_layer_us, _, per_layer_kv_gb = bw_table[key]
    return {
        "request_id": request_id,
        "npu_id": request_id,
        "seq_len_k": key[0],
        "nql": key[1],
        "per_layer_us": per_layer_us,
        "per_layer_kv_gb": per_layer_kv_gb,
        # 输入画像中的 required_bw 只用于记录和配对校验。
        "required_bw_input_gbps": float(required_bw),
        "category": classify_request(*key),
    }


def generate_request_loads(bw_table, rng, count, ls_ratio=None):
    """生成实验使用的 SS/SL/LS/LL 混合请求列表。"""
    keys = list(bw_table)
    if ls_ratio is None:
        return [
            _load_from_key(bw_table, keys[rng.randint(len(keys))], request_id)
            for request_id in range(count)
        ]

    keys_by_category = {
        category: [key for key in keys if classify_request(*key) == category]
        for category in QOS_ROUTING_CATEGORIES
    }
    short_count = count // 2
    long_count = count - short_count
    category_counts = {
        "SS": short_count - int(round(short_count * ls_ratio)),
        "SL": int(round(short_count * ls_ratio)),
        "LS": int(round(long_count * ls_ratio)),
        "LL": long_count - int(round(long_count * ls_ratio)),
    }

    loads = []
    # 先保持原生成器的类别顺序，全部生成完成后再统一打乱。
    for category in ("SS", "SL", "LL", "LS"):
        choices = keys_by_category[category]
        for _ in range(category_counts[category]):
            key = choices[rng.randint(len(choices))]
            loads.append(_load_from_key(bw_table, key, len(loads)))
    rng.shuffle(loads)
    for request_id, load in enumerate(loads):
        load["request_id"] = request_id
        load["npu_id"] = request_id
    return loads


def calculate_token_partition(seq_len_k, nql):
    """返回总词元数、NPU 计算的词元数、需要从 SSD 读取的词元数。"""
    total_tokens = int(round(float(seq_len_k) * 1024.0))
    npu_tokens = int(round(float(nql)))
    return total_tokens, npu_tokens, total_tokens - npu_tokens


def build_block_placement(loads, rng, mode, n_layers, num_disk):
    """把每一层 KV 切成数据块，并决定每个数据块放在哪块 SSD 上。"""
    if mode not in {"random", "roundrobin"}:
        raise ValueError("placement_mode 只能是 random 或 roundrobin")
    placement_by_request = {}
    for load in loads:
        _, _, ssd_tokens = calculate_token_partition(load["seq_len_k"], load["nql"])
        block_sizes = []
        if ssd_tokens > 0 and load["per_layer_kv_gb"] > 0:
            gb_per_token = load["per_layer_kv_gb"] / ssd_tokens
            block_count = int(np.ceil(ssd_tokens / BLOCK_SIZE))
            block_sizes = [
                min(BLOCK_SIZE, ssd_tokens - index * BLOCK_SIZE) * gb_per_token
                for index in range(block_count)
            ]

        layers = {}
        for layer in range(n_layers):
            blocks = []
            for block_index, block_gb in enumerate(block_sizes):
                if mode == "roundrobin":
                    disk_id = block_index % num_disk
                else:
                    disk_id = int(rng.randint(num_disk))
                blocks.append((disk_id, float(block_gb)))
            layers[layer] = tuple(blocks)
        placement_by_request[load["request_id"]] = layers
    return placement_by_request


def _stable_json_hash(value):
    """对只含 JSON 基本类型的规范化输入计算可复现 SHA-256。"""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workload_fingerprint(request_loads):
    """返回与运行策略无关的 workload fingerprint。"""
    fields = (
        "request_id",
        "seq_len_k",
        "nql",
        "per_layer_us",
        "per_layer_kv_gb",
        "required_bw_input_gbps",
        "category",
        "arrival_time",
    )
    canonical = [
        {field_name: load[field_name] for field_name in fields}
        for load in sorted(request_loads, key=lambda row: row["request_id"])
    ]
    return _stable_json_hash(canonical)


def placement_fingerprint(placement_by_request):
    """流式计算 immutable block placement 的 SHA-256。"""
    digest = hashlib.sha256()
    for request_id in sorted(placement_by_request):
        layers = placement_by_request[request_id]
        for layer in sorted(layers):
            for block_index, (disk_id, block_gb) in enumerate(layers[layer]):
                digest.update(
                    struct.pack(
                        "!IIIId",
                        int(request_id),
                        int(layer),
                        block_index,
                        int(disk_id),
                        float(block_gb),
                    )
                )
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedSimulationInputs:
    """一次配对实验共享的 workload 与 immutable placement artifact。"""

    request_loads: tuple[dict, ...]
    placement_by_request: dict
    workload_seed: int
    placement_seed: int
    workload_hash: str
    placement_hash: str
    n_layers: int
    num_disk: int
    placement_mode: str
    arrival_delay_seed: int = 0
    arrival_delay_max_ms: float = 0.0


def prepare_simulation_inputs(
    bw_table,
    *,
    total_requests,
    n_layers,
    num_disk,
    ls_ratio=None,
    workload_seed=42,
    placement_seed=43,
    arrival_delay_seed=44,
    arrival_delay_max_ms=ARRIVAL_DELAY_MAX_MS,
    placement_mode="random",
):
    """用彼此独立的 RNG 生成可在策略间严格复用的输入。"""
    workload_rng = np.random.RandomState(int(workload_seed))
    placement_rng = np.random.RandomState(int(placement_seed))
    arrival_rng = np.random.RandomState(int(arrival_delay_seed))
    loads = generate_request_loads(
        bw_table,
        workload_rng,
        int(total_requests),
        ls_ratio=ls_ratio,
    )
    for request_id, request in enumerate(loads):
        request["request_id"] = request_id
        request["npu_id"] = request_id
        request["arrival_time"] = float(
            arrival_rng.uniform(0.0, float(arrival_delay_max_ms))
        )
    placement = build_block_placement(
        loads,
        placement_rng,
        placement_mode,
        int(n_layers),
        int(num_disk),
    )
    return PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placement,
        workload_seed=int(workload_seed),
        placement_seed=int(placement_seed),
        workload_hash=workload_fingerprint(loads),
        placement_hash=placement_fingerprint(placement),
        n_layers=int(n_layers),
        num_disk=int(num_disk),
        placement_mode=placement_mode,
        arrival_delay_seed=int(arrival_delay_seed),
        arrival_delay_max_ms=float(arrival_delay_max_ms),
    )


# ---------------------------------------------------------------------------
# SSD 硬件的静态 QoS 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticQoSConfig:
    """每块 SSD 初始化时写入一次、之后不再修改的 QoS 寄存器配置。

    每个元组的下标就是 QoS Path ID。例如 ``path_cirs[17]`` 表示
    Path 17 的 CIR。``category_paths_per_group`` 描述每个组内连续排列的
    SS、SL、LS、LL Path 数量。

    注意：CIR/PIR 是静态硬件配置；运行时的离散服务顺序不是配置改写。

    字段说明
    --------
    ``path_cirs``（各路径的保证带宽）：
        256 个 Path 各自的 CIR，单位为 GB/s。
    ``path_pirs``（各路径的峰值带宽）：
        256 个 Path 各自的 PIR，单位为 GB/s。
    ``path_weights``（各路径的权重）：
        256 个 Path 的组内 WRR 权重。
    ``group_weights``（各组的权重）：
        8 个组的组间 WRR 权重。
    ``category_paths_per_group``（每组的类别布局）：
        每个组内 SS、SL、LS、LL 各占多少个 Path。本示例程序使用
        ``(12, 4, 12, 4)``：组内偏移 0~11 属于 SS，12~15 属于 SL，
        16~27 属于 LS，28~31 属于 LL。

    ``frozen=True`` 表示对象创建后不能重新给这些字段赋值。结合下面的元组
    转换，可以从代码结构上保证运行期间不会动态修改 CIR/PIR/WRR 配置。
    """

    path_cirs: tuple[float, ...]
    path_pirs: tuple[float, ...]
    path_weights: tuple[float, ...]
    group_weights: tuple[float, ...]
    category_paths_per_group: tuple[int, int, int, int]

    def __post_init__(self):
        # 初始化时统一转成元组。这样即使调用方传入列表，创建完成后也不能
        # 再从外部修改这些硬件配置表。
        for field_name in (
            "path_cirs",
            "path_pirs",
            "path_weights",
            "group_weights",
            "category_paths_per_group",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

        path_count = len(self.path_cirs)
        if path_count != 256 or len(self.group_weights) != 8:
            raise ValueError("静态 QoS 硬件必须包含 256 个 Path 和 8 个 Group")
        if not (path_count == len(self.path_pirs) == len(self.path_weights)):
            raise ValueError("CIR、PIR 和 Path 权重数组的长度必须相同")
        if len(self.category_paths_per_group) != len(QOS_ROUTING_CATEGORIES) or any(
            count <= 0 for count in self.category_paths_per_group
        ):
            raise ValueError("每个 Group 都必须为 SS、SL、LS、LL 分配 Path")
        if sum(self.category_paths_per_group) != self.paths_per_group:
            raise ValueError("四个类别的 Path 数量之和必须正好填满一个 Group")
        if any(cir < 0.0 for cir in self.path_cirs):
            raise ValueError("Path CIR 不能为负数")
        if any(pir < cir for cir, pir in zip(self.path_cirs, self.path_pirs)):
            raise ValueError("每个 Path 都必须满足 CIR <= PIR")

    @property
    def path_count(self):
        return len(self.path_cirs)

    @property
    def group_count(self):
        return len(self.group_weights)

    @property
    def paths_per_group(self):
        return self.path_count // self.group_count


@dataclass(frozen=True)
class ClientRoutingConfig:
    """客户端已知的静态 QoS 镜像和稳定轮转起点。"""

    qos_config: StaticQoSConfig
    disk_bw: float
    start_offset: int = 0


@dataclass(frozen=True)
class CandidatePathEstimate:
    """用 count-only 遥测估算的一块 SSU 下一条 I/O 结果。"""

    path_id: int
    finish_time_s: float
    effective_path_rate_gbps: float
    near_term_rate_gbps: float
    estimated_old_backlog_gb: float


@dataclass(frozen=True)
class _QoSCountAnalysis:
    """完全由一份 256-count 派生的客户端侧聚合；不含硬件隐藏状态。"""

    counts: tuple[int, ...]
    group_io_counts: tuple[int, ...]
    active_paths_per_group: tuple[int, ...]
    active_path_weights: tuple[float, ...]
    active_group_weight_sum: float
    active_cir_sum: float


def _analyze_qos_counts(path_io_counts, qos_config):
    # SSD report 已经是不可变整数 tuple；直接保留其对象身份，既避免每次
    # 256 项复制，也让同一未变化 snapshot 能安全命中客户端缓存。纯函数
    # 调用方若传入 list/array，仍在这里复制成不可变 tuple。
    counts = (
        path_io_counts
        if isinstance(path_io_counts, tuple)
        else tuple(int(count) for count in path_io_counts)
    )
    path_count = len(qos_config.path_cirs)
    group_count = len(qos_config.group_weights)
    paths_per_group = path_count // group_count
    if len(counts) != path_count:
        raise ValueError("Path 压力快照长度必须恰好为 256")
    group_io_counts = [0] * group_count
    active_paths = [0] * group_count
    active_weights = [0.0] * group_count
    group_weights = qos_config.group_weights
    path_weights = qos_config.path_weights
    path_cirs = qos_config.path_cirs
    path_pirs = qos_config.path_pirs
    group_weight_sum = 0.0
    cir_sum = 0.0
    for path_id, count in enumerate(counts):
        group_id = path_id // paths_per_group
        group_io_counts[group_id] += count
        if count <= 0:
            continue
        if active_paths[group_id] == 0:
            group_weight_sum += group_weights[group_id]
        active_paths[group_id] += 1
        active_weights[group_id] += path_weights[path_id]
        cir_sum += min(path_cirs[path_id], path_pirs[path_id])
    return _QoSCountAnalysis(
        counts=counts,
        group_io_counts=tuple(group_io_counts),
        active_paths_per_group=tuple(active_paths),
        active_path_weights=tuple(active_weights),
        active_group_weight_sum=group_weight_sum,
        active_cir_sum=cir_sum,
    )


def _projection_choices_from_qos_counts(
    *,
    block_size_gb,
    representative_block_gb,
    allowed_path_ids,
    routing_config,
    counts,
    group_io_counts,
    active_paths_per_group,
    active_path_weights,
    active_group_weight_sum,
    active_cir_sum,
    projected_work_gb=None,
):
    """计算与 start_offset 无关的最优候选集合。

    原比较键依次为 F、Path count、Group count、轮转距离和 Path ID。
    前三项与请求的稳定轮转起点无关，因此同一不可变 snapshot 可缓存这份
    通常很小的候选集合；最后两项仍在每次请求时独立计算。
    """
    qos = routing_config.qos_config
    allowed = tuple(allowed_path_ids)
    path_cirs = qos.path_cirs
    path_pirs = qos.path_pirs
    path_weights = qos.path_weights
    group_weights = qos.group_weights
    paths_per_group = len(path_cirs) // len(group_weights)
    disk_bw = routing_config.disk_bw
    best_primary = None
    choices = []
    # 同一 group 中服务输入完全相同的 Path 有完全相同的长期 rate。先把它们
    # 合并，并在每个 rate 等价类内淘汰 old_work/count 更差者；最终平局 Path
    # 仍全部保留，所以稳定哈希的轮转结果不变。该归并不假设 WRR 权重为 1。
    equivalence = {}
    for allowed_index, path_id in enumerate(allowed):
        group_id = path_id // paths_per_group
        was_empty = counts[path_id] == 0
        path_pir = path_pirs[path_id]
        path_weight = path_weights[path_id]
        path_cir = path_cirs[path_id]
        old_work = (
            projected_work_gb[path_id]
            if projected_work_gb is not None
            else counts[path_id] * representative_block_gb
        )
        rate_class = (
            group_id,
            was_empty,
            path_cir,
            path_pir,
            path_weight,
        )
        dominance_rank = (old_work, counts[path_id])
        current = equivalence.get(rate_class)
        record = (allowed_index, path_id, old_work, counts[path_id])
        if current is None or dominance_rank < current[0]:
            equivalence[rate_class] = (dominance_rank, [record])
        elif dominance_rank == current[0]:
            current[1].append(record)

    for rate_class, (_, equivalent_paths) in equivalence.items():
        group_id, was_empty, path_cir, path_pir, path_weight = rate_class
        group_weight = group_weights[group_id]
        base_rate = min(path_cir, path_pir)
        cir_sum = active_cir_sum + (base_rate if was_empty else 0.0)
        group_weight_sum = active_group_weight_sum
        path_weight_sum = active_path_weights[group_id]
        if was_empty:
            path_weight_sum += path_weight
            if active_paths_per_group[group_id] == 0:
                group_weight_sum += group_weight
        if group_weight_sum <= _EPS or path_weight_sum <= _EPS:
            continue
        remaining = max(0.0, disk_bw - cir_sum)
        group_extra = remaining * group_weight / group_weight_sum
        path_extra = group_extra * path_weight / path_weight_sum
        rate = min(path_pir, base_rate + path_extra)
        if rate <= _EPS:
            continue
        allowed_index, path_id, old_work, path_count = equivalent_paths[0]
        finish_s = (old_work + block_size_gb) / rate
        primary = (
            finish_s,
            path_count,
            group_io_counts[group_id],
        )
        equivalent_choices = [
            (index, candidate_path, finish_s, rate, candidate_old_work)
            for index, candidate_path, candidate_old_work, _ in equivalent_paths
        ]
        if best_primary is None or primary < best_primary:
            best_primary = primary
            choices = equivalent_choices
        elif primary == best_primary:
            choices.extend(equivalent_choices)
    if not choices:
        raise RuntimeError("合法 Path 池没有正的静态长期服务率")
    choices.sort(key=lambda choice: choice[0])
    return (
        tuple(choice[0] for choice in choices),
        tuple(choices),
    )


def _select_projection_choice(
    choices,
    *,
    allowed_count,
    start_offset,
    block_size_gb,
    disk_bw,
):
    offset = start_offset % allowed_count
    allowed_indices, records = choices
    position = bisect.bisect_left(allowed_indices, offset)
    if position == len(records):
        position = 0
    selected = records[position]
    return CandidatePathEstimate(
        path_id=selected[1],
        finish_time_s=selected[2],
        effective_path_rate_gbps=selected[3],
        near_term_rate_gbps=min(
            disk_bw,
            block_size_gb / max(selected[2], _EPS),
        ),
        estimated_old_backlog_gb=selected[4],
    )


def _estimate_from_qos_projection(
    *,
    block_size_gb,
    representative_block_gb,
    allowed_path_ids,
    routing_config,
    counts,
    group_io_counts,
    active_paths_per_group,
    active_path_weights,
    active_group_weight_sum,
    active_cir_sum,
    projected_work_gb=None,
):
    allowed = tuple(allowed_path_ids)
    choices = _projection_choices_from_qos_counts(
        block_size_gb=block_size_gb,
        representative_block_gb=representative_block_gb,
        allowed_path_ids=allowed,
        routing_config=routing_config,
        counts=counts,
        group_io_counts=group_io_counts,
        active_paths_per_group=active_paths_per_group,
        active_path_weights=active_path_weights,
        active_group_weight_sum=active_group_weight_sum,
        active_cir_sum=active_cir_sum,
        projected_work_gb=projected_work_gb,
    )
    return _select_projection_choice(
        choices,
        allowed_count=len(allowed),
        start_offset=routing_config.start_offset,
        block_size_gb=block_size_gb,
        disk_bw=routing_config.disk_bw,
    )


class _QoSPlanningShadow:
    """一轮 plan 内由 snapshot 派生、只记录本 NPU 新计划的可变 shadow。"""

    def __init__(self, analysis, representative_block_gb):
        self.counts = list(analysis.counts)
        self.group_io_counts = list(analysis.group_io_counts)
        self.active_paths_per_group = list(analysis.active_paths_per_group)
        self.active_path_weights = list(analysis.active_path_weights)
        self.active_group_weight_sum = analysis.active_group_weight_sum
        self.active_cir_sum = analysis.active_cir_sum
        self.projected_work_gb = [
            count * representative_block_gb for count in analysis.counts
        ]

    def estimate(
        self,
        *,
        block_size_gb,
        representative_block_gb,
        allowed_path_ids,
        routing_config,
    ):
        return _estimate_from_qos_projection(
            block_size_gb=block_size_gb,
            representative_block_gb=representative_block_gb,
            allowed_path_ids=allowed_path_ids,
            routing_config=routing_config,
            counts=self.counts,
            group_io_counts=self.group_io_counts,
            active_paths_per_group=self.active_paths_per_group,
            active_path_weights=self.active_path_weights,
            active_group_weight_sum=self.active_group_weight_sum,
            active_cir_sum=self.active_cir_sum,
            projected_work_gb=self.projected_work_gb,
        )

    def add(self, path_id, block_size_gb, qos_config, io_count=1):
        if io_count <= 0:
            raise ValueError("shadow 新增 I/O 数必须为正数")
        group_id = path_id // (
            len(qos_config.path_cirs) // len(qos_config.group_weights)
        )
        if self.counts[path_id] == 0:
            if self.active_paths_per_group[group_id] == 0:
                self.active_group_weight_sum += qos_config.group_weights[group_id]
            self.active_paths_per_group[group_id] += 1
            self.active_path_weights[group_id] += qos_config.path_weights[path_id]
            self.active_cir_sum += min(
                qos_config.path_cirs[path_id], qos_config.path_pirs[path_id]
            )
        self.counts[path_id] += io_count
        self.group_io_counts[group_id] += io_count
        self.projected_work_gb[path_id] += block_size_gb


# ---------------------------------------------------------------------------
# 客户端选路：这一节被设计成可以直接移植到其他项目
# ---------------------------------------------------------------------------

# 初学者可以把客户端的完整工作理解为下面 5 步：
#
# 1. 数据放置模块先决定“每个数据块位于哪块 SSD”。这一步可以使用
#    环形哈希，但不属于本函数的职责。
# 2. 每块 SSD 按配置的压力读取间隔分别汇报自己的 256 个 Path 压力：
#       Path 压力 = 正在执行的 I/O 数 + 排队等待的 I/O 数
# 3. 客户端根据请求类别取得该类别允许使用的 Path 池。
# 4. 客户端调用 client_select_qos_paths()；每条 I/O 独立决定一个 Path。
# 5. 客户端把返回的 Path ID 写入对应 I/O 的请求字段（例如 SQE DW2），
#    再把请求提交给 SSD。
#
# 需要特别区分：上游负责选择 SSD，本节只负责选择该 SSD 内部的 QoS Path。
# 客户端只是填写 Path ID，不会租用、独占或预留 Path。非空 Path 仍然可选。


def client_category_paths(  # 定义“请求类别到可用 Path 池”的初始化函数。
    category: str,  # 输入一：当前请求类别，例如 "SS" 或 "LL"。
    qos_config: StaticQoSConfig,  # 输入二：客户端初始化时保存的静态 QoS 布局。
) -> tuple[int, ...]:  # 输出：该类别允许使用的全部 Path ID，使用元组保存。
    """生成某个请求类别可以使用的全部 Path ID。

    这个函数什么时候调用？
    ----------------------
    在客户端初始化阶段调用一次即可。真实客户端可以把四个类别的结果缓存为：

    类别池缓存示例：``category_path_pools = {"SS": (...), "SL": (...), ...}``

    运行时不需要重复计算，也不需要向 SSD 申请 Path。

    输入
    ----
    参数 ``category``（请求类别）：
        请求类别，只能是 ``"SS"``、``"SL"``、``"LS"`` 或 ``"LL"``。
    参数 ``qos_config``（静态 QoS 配置）：
        客户端与 SSD 初始化代码共同知道的静态布局。这里面只有 CIR/PIR、
        权重和类别布局，不包含任何运行时队列状态。

    输出
    ----
    返回类型：``tuple[int, ...]``
        该类别在所有 8 个组中拥有的 Path ID。例如 LL 每组有 4 个 Path，
        8 个组一共返回 32 个 Path ID。

    为什么需要这个函数？
    --------------------
    ``client_select_qos_paths`` 本身不理解 SS/SL/LS/LL。调用方只把本类别允许
    使用的 Path ID 传进去，就自然实现了“每个类别只使用自己的 Path 池”。
    """
    # 第一步：确定 category 在 (SS, SL, LS, LL) 中排第几个。
    category_index = QOS_ROUTING_CATEGORIES.index(category)  # 查出类别序号。

    # 第二步：计算该类别在一个组内的起始偏移。
    # 例如每组布局为 (12, 4, 12, 4)，LL 前面共有 12+4+12=28 个 Path，
    # 所以 LL 在每个组内从偏移 28 开始。
    category_offset = sum(  # 计算该类别在一个组内的起始位置。
        qos_config.category_paths_per_group[:category_index]  # 累加前面类别的 Path 数。
    )  # 得到组内偏移，例如 LL 的偏移是 28。
    category_count = qos_config.category_paths_per_group[  # 读取每组本类别 Path 数。
        category_index  # 用上面得到的类别序号访问静态布局。
    ]  # 例如 LL 每组有 4 个 Path。

    # 第三步：把“组号 + 组内偏移”转换为全局 Path ID。
    # 外层先遍历类别内偏移、内层再遍历 8 个组，可以让同压力平局时优先
    # 跨组分散。例如 LL 的开头是 28、60、92、124，而不是 28、29、30、31。
    path_ids = []  # 创建普通列表，按跨组交错顺序收集全局 Path ID。
    for local_offset in range(category_count):  # 依次处理类别内部的第几个 Path。
        for group_id in range(qos_config.group_count):  # 让同一偏移先跨全部 8 个组。
            group_start = (  # 计算当前组的第一个全局 Path ID。
                group_id * qos_config.paths_per_group  # 组号乘以每组 Path 数。
            )  # 例如第 3 组的起点是 3 * 32 = 96。
            path_id = (  # 计算一个 Path 的全局 ID。
                group_start  # 从当前组的全局起点开始。
                + category_offset  # 加上当前类别在组内的起始偏移。
                + local_offset  # 再加当前类别内部的第几个 Path。
            )  # 例如第 3 组的第 2 个 LL Path 是 96 + 28 + 2 = 126。
            path_ids.append(path_id)  # 把算出的 Path ID 追加到结果列表末尾。
    return tuple(path_ids)  # 转成不可变元组，作为该类别的完整 Path 池返回。


def allocate_demand_path_tickets(request_loads, total_paths=256):
    """按每 NPU 的 capped demand 用 largest remainder 静态分配 Path 票据。"""
    ordered = sorted(request_loads, key=lambda load: load["request_id"])
    demands = [min(float(load["required_bw_input_gbps"]), NPU_BW_LIMIT) for load in ordered]
    free = set(range(len(ordered)))
    fixed = set()
    remaining_paths = total_paths
    while True:
        free_demand = sum(demands[index] for index in free)
        projected = {
            index: remaining_paths * demands[index] / free_demand for index in free
        }
        below_minimum = [index for index, quota in projected.items() if quota < 1.0]
        if not below_minimum:
            break
        for index in below_minimum:
            free.remove(index)
            fixed.add(index)
            remaining_paths -= 1

    free_demand = sum(demands[index] for index in free)
    quotas = [1.0] * len(ordered)
    for index in free:
        quotas[index] = remaining_paths * demands[index] / free_demand
    counts = [1 if index in fixed else int(math.floor(quotas[index])) for index in range(len(ordered))]
    remaining = total_paths - sum(counts)
    remainder_order = sorted(
        free,
        key=lambda index: (-(quotas[index] - math.floor(quotas[index])), index),
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return {
        load["request_id"]: counts[index] for index, load in enumerate(ordered)
    }


def build_demand_ticket_path_pools(request_loads, num_disk, total_paths=256):
    """把 256 个等价 Path 在每块盘上静态划给各 NPU。"""
    ticket_counts = allocate_demand_path_tickets(request_loads, total_paths)
    request_ids = sorted(ticket_counts)
    pools = {}
    for disk_id in range(num_disk):
        permutation = tuple(
            (disk_id * 29 + position * 73) % total_paths
            for position in range(total_paths)
        )
        cursor = 0
        for request_id in request_ids:
            next_cursor = cursor + ticket_counts[request_id]
            pools[(request_id, disk_id)] = permutation[cursor:next_cursor]
            cursor = next_cursor
    return pools


def build_dynamic_npu_path_pools(
    num_npu,
    num_disk,
    total_paths=256,
    paths_per_npu=2,
):
    """Build workload-independent, mutually exclusive per-NPU Path pools."""
    lane_width = total_paths // paths_per_npu
    if paths_per_npu != 2 or num_npu > lane_width:
        raise ValueError("dynamic CIR requires at most 128 NPUs with two Paths each")
    pools = {}
    for disk_id in range(num_disk):
        for npu_id in range(num_npu):
            pools[(npu_id, disk_id)] = tuple(
                npu_id + lane * lane_width for lane in range(paths_per_npu)
            )
    return pools


def client_select_dynamic_owned_paths(
    *,
    block_sizes_gb,
    path_io_counts,
    allowed_path_ids,
    start_offset=0,
):
    """Route each I/O only inside one NPU's exclusive two-Path pool."""
    sizes = tuple(float(size) for size in block_sizes_gb)
    allowed = tuple(allowed_path_ids)
    representative_gb = sorted(sizes)[len(sizes) // 2]
    counts = list(path_io_counts)
    projected_work_gb = [count * representative_gb for count in counts]
    selected = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda index: (-sizes[index], index))
    offset = start_offset % len(allowed)
    for index in order:
        path_id = min(
            allowed,
            key=lambda candidate: (
                projected_work_gb[candidate],
                counts[candidate],
                (allowed.index(candidate) - offset) % len(allowed),
                candidate,
            ),
        )
        selected[index] = path_id
        counts[path_id] += 1
        projected_work_gb[path_id] += sizes[index]
    return selected


def _select_qos_paths_from_analysis(
    sizes, analysis, allowed_path_ids, routing_config
):
    qos = routing_config.qos_config
    allowed = tuple(allowed_path_ids)
    representative_gb = sorted(sizes)[len(sizes) // 2]
    if len(sizes) == 1:
        estimate = _estimate_from_qos_projection(
            block_size_gb=sizes[0],
            representative_block_gb=representative_gb,
            allowed_path_ids=allowed,
            routing_config=routing_config,
            counts=analysis.counts,
            group_io_counts=analysis.group_io_counts,
            active_paths_per_group=analysis.active_paths_per_group,
            active_path_weights=analysis.active_path_weights,
            active_group_weight_sum=analysis.active_group_weight_sum,
            active_cir_sum=analysis.active_cir_sum,
        )
        return [estimate.path_id]

    shadow = _QoSPlanningShadow(analysis, representative_gb)
    selected = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda index: (-sizes[index], index))
    for index in order:
        block_gb = sizes[index]
        estimate = shadow.estimate(
            block_size_gb=block_gb,
            representative_block_gb=representative_gb,
            allowed_path_ids=allowed,
            routing_config=routing_config,
        )
        selected[index] = estimate.path_id
        shadow.add(estimate.path_id, block_gb, qos)
    return selected


def client_select_qos_paths(
    *, block_sizes_gb, path_io_counts, allowed_path_ids, routing_config
):
    """按预计完成时间为同一 SSU 的一批 I/O 选择合法 QoS Path。

    硬件输入只有完整的 256 项 active+pending I/O-count。旧 I/O 字节数用
    当前窗口的中位 block 大小估算；每规划一条 I/O 后立即更新本地 shadow。
    """
    sizes = tuple(float(size) for size in block_sizes_gb)
    if not sizes:
        return []
    if any(not np.isfinite(size) or size <= 0.0 for size in sizes):
        raise ValueError("block 大小必须是有限正数")

    qos = routing_config.qos_config
    counts = tuple(path_io_counts)
    if len(counts) != qos.path_count:
        raise ValueError("Path 压力快照长度必须恰好为 256")
    if any(
        not isinstance(count, (int, np.integer)) or int(count) < 0 for count in counts
    ):
        raise ValueError("Path 压力必须是非负整数 I/O-count")

    allowed = tuple(allowed_path_ids)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("合法 Path 池不能为空且不能包含重复 ID")
    if any(
        not isinstance(path_id, (int, np.integer))
        or path_id < 0
        or path_id >= qos.path_count
        for path_id in allowed
    ):
        raise ValueError("合法 Path ID 超出 0..255")

    return _select_qos_paths_from_analysis(
        sizes,
        _analyze_qos_counts(counts, qos),
        allowed,
        routing_config,
    )


# SSD 侧 I/O 对象与静态 QoS 调度器
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class BlockIOFlow:
    npu_id: int  # 发起这条 I/O 的 NPU 编号。
    layer: int  # 这条 I/O 所读取的模型层号。
    block_idx: int  # 这条 I/O 对应的逻辑数据块编号。
    disk_id: int  # 数据放置阶段已经选定的目标 SSD 编号。
    total_gb: float  # 这条仿真流需要传输的总数据量，单位为 GB。
    queue_id: int  # 客户端选择的 Path ID；基线绕过模式使用 -1。
    block_count: int  # 这条仿真流代表的逻辑数据块数量，用于计算 Path 压力。
    enqueue_time: float  # 这条 I/O 进入 SSD 的仿真时间。
    request_id: int = -1  # 全局请求编号；独立调度器单测可使用默认值。
    demand_gbps: float = 0.0
    deadline_time: float = float("inf")
    layer_work_gb: float = 0.0
    remaining_gb: float = field(init=False)  # 尚未传输的数据量，由初始化函数赋值。
    active: bool = field(default=False, init=False)  # 是否是整块 SSD 当前唯一的后端 I/O。
    bw: float = field(default=0.0, init=False)  # 当前 SSD 命令服务阶段的速率。
    start_time: float = field(init=False)  # 最近一次开始按当前带宽传输的时刻。
    end_time: float = field(default=float("inf"), init=False)  # 预计完成时刻。
    queue: "PathQueue | None" = field(default=None, init=False)  # 反向指向选中的 Path。
    link_enqueue_time: float = field(default=float("inf"), init=False)
    link_start_time: float = field(default=float("inf"), init=False)
    link_end_time: float = field(default=float("inf"), init=False)
    ssd_queue_wait_ms: float = field(default=0.0, init=False)

    def __post_init__(self):  # 数据类创建完成后初始化两个由输入字段推导的运行时字段。
        self.remaining_gb = self.total_gb  # 初始时全部数据都尚未传输。
        self.start_time = self.enqueue_time  # 第一次进度结算从入队时刻开始计算。

    def remaining_io_count(self):  # 把剩余数据量换算成尚未完成的逻辑 I/O 数。
        """返回当前仿真流中尚未完成的数据块数量，供 Path 压力报告使用。"""
        if self.remaining_gb <= _EPS:  # 剩余量接近 0 时认为整条流已经完成。
            return 0  # 已完成的流不再贡献 Path 压力。
        if self.block_count <= 1:  # QoS 模式通常一条流只代表一个数据块。
            return 1  # 只要还有数据未传完，这一个逻辑 I/O 就仍未完成。
        io_size_gb = self.total_gb / self.block_count  # 估算每个聚合前数据块的大小。
        remaining_count = int(  # 把剩余数据量向上取整为尚未完成的数据块数。
            np.ceil(self.remaining_gb / io_size_gb - 1e-10)  # 微小量用于消除浮点边界误差。
        )  # 得到按数据量估算的剩余逻辑 I/O 数。
        remaining_count = max(1, remaining_count)  # 尚有数据时至少算一个未完成 I/O。
        return min(  # 防止浮点误差导致结果超过原始数据块总数。
            self.block_count,  # 剩余数量的最大值是这条流包含的数据块总数。
            remaining_count,  # 返回修正后的剩余逻辑 I/O 数。
        )  # 该结果会被加到 Path 的排队 I/O 数上形成压力报告。


class PathQueue:
    """一个 QoS Path 的 FCFS 队列。

    ``active_flow`` 只在该 Path 被整盘仲裁器选中时非空。所有 Path 合计最多
    只有一个 active flow；其他 Path 即使非空，也只保存等待中的 I/O。
    """

    def __init__(  # 在 SSD 初始化时创建一个静态 QoS Path。
        self,  # 当前 Path 队列对象。
        path_id,  # 这个 Path 的全局编号，也是客户端最终写入 DW2 的值。
        cir,  # 初始化保证带宽；动态策略可在非抢占命令边界原子更新。
        pir,  # 初始化时写入的峰值带宽，运行期间不再修改。
        path_weight,  # 初始化时写入的组内 WRR 权重。
        group_id,  # 这个 Path 所属的组编号。
    ):
        self.path_id = path_id  # 保存全局 Path ID，便于报告和统计。
        self.cir = float(cir)  # 保存当前有效 CIR，并统一转换成浮点数。
        self.pir = float(pir)  # 保存静态 PIR，并统一转换成浮点数。
        self.path_weight = float(path_weight)  # 保存静态组内 WRR 权重。
        self.group_id = group_id  # 保存所属组，用于两级 WRR 仲裁。
        self.pending = deque()  # 创建先入先出等待队列；非空 Path 仍可继续加入 I/O。
        self.pending_gb = 0.0
        self.pending_io_count = 0  # 当前排队等待的逻辑 I/O 数初始为 0。
        self.active_flow = None  # 当前正在服务的 I/O 初始为空。
        self.max_outstanding_io = 0  # 历史最大 Path 压力初始为 0。
        self.virtual_finish = 0.0  # 离散 CIR/WRR 仲裁使用的虚拟完成标签。

    def io_count(self):  # 计算并返回 QoS 向客户端汇报的当前 Path 压力。
        if self.active_flow is None:  # 当前 Path 没有正在服务的 I/O。
            active_count = 0  # 没有正在服务的 I/O，因此正在执行数量为 0。
        else:  # 当前 Path 存在正在服务的 I/O。
            active_count = self.active_flow.remaining_io_count()  # 计算尚未完成的数量。
        return self.pending_io_count + active_count  # 压力等于正在执行数加排队数。

    def enqueue(self, flow):  # 把客户端已经选好 Path 的一个 I/O 放入该 Path。
        """接收 I/O；即使 Path 非空也允许继续加入等待队列。"""
        flow.queue = self  # 在 I/O 对象中记录它属于当前 Path。
        self.pending.append(flow)  # 无条件追加到先入先出等待队列，绝不要求压力为 0。
        self.pending_gb += flow.total_gb
        self.pending_io_count += flow.block_count  # 把新加入的数据块数计入排队压力。
        self.max_outstanding_io = max(  # 更新这个 Path 曾经观察到的最大压力。
            self.max_outstanding_io,  # 保留历史最大压力。
            self.io_count(),  # 与加入当前 I/O 后的最新压力比较。
        )  # 得到更新后的最大未完成 I/O 数。
        return flow

    def peek(self):
        """返回 FCFS 队头，但不把它交给后端。"""
        return self.pending[0] if self.pending else None

    def activate_next(self):  # 仲裁胜出后让队头 I/O 进入唯一后端执行位。
        if self.active_flow is not None or not self.pending:  # 已在执行或队列为空时不能启动。
            return None  # 没有新启动的 I/O，用空值告诉调用方。
        flow = self.pending.popleft()  # 按先入先出规则取出最早到达的 I/O。
        self.pending_gb -= flow.total_gb
        self.pending_io_count -= flow.block_count  # 它已离开等待队列，扣除排队压力。
        self.active_flow = flow  # 把它登记为当前正在执行的 I/O。
        return flow  # 把新启动的 I/O 返回给整盘带宽调度器。

    def complete(self, flow):  # 当前 I/O 完成后更新 Path，并尝试启动队头 I/O。
        if self.active_flow is not flow:
            raise RuntimeError("完成的 I/O 不是当前 Path 的后端 active I/O")
        self.active_flow = None  # 清空正在执行位置，使 Path 可以启动下一条 I/O。

    def is_active(self):  # 判断这个 Path 当前是否有 I/O 正在执行。
        return self.active_flow is not None  # 有正在执行的 I/O 时返回真，否则返回假。

    def has_work(self):
        """返回该 Path 是否有正在执行或等待中的 I/O。"""
        return self.active_flow is not None or bool(self.pending)


class DiskState:
    """事件循环与磁盘调度器共同使用的一块物理 SSD 状态。"""

    def __init__(self, disk_id):
        self.disk_id = disk_id
        self.active_flows = []
        self.generation = 0
        self.busy_time = 0.0
        self.idle_time = 0.0
        self.last_event_time = 0.0
        self.surplus_bw_integral = 0.0
        self.total_bw_integral = 0.0
        self.completed_bytes_gb = 0.0
        self.scheduler = None


class GlobalLinkCoordinator:
    """Coordinate idle SSDs using only online, already-visible queue state.

    The coordinator never moves data or reserves NPU bandwidth.  It predicts the
    FCFS link completion of each disk-local queue-head candidate from three
    pieces of state: the NPU's active command, its arrived pending commands, and
    commands already active on other SSDs.  Future client submissions and future
    layers are deliberately absent.
    """

    def __init__(self, npus, disk_states):
        self.npus = npus
        self.disk_states = disk_states
        self._timeline_cache = [None] * len(npus)
        self._timeline_dirty = [True] * len(npus)
        self.predictions = 0
        self.selections = 0
        self.timeline_rebuilds = 0

    @staticmethod
    def _reservation_key(flow, ssd_end_time):
        return (
            ssd_end_time,
            flow.request_id,
            flow.layer,
            flow.block_idx,
            flow.disk_id,
        )

    def mark_npu_dirty(self, npu_id):
        self._timeline_dirty[npu_id] = True

    def _rebuild_timeline(self, npu_id):
        npu = self.npus[npu_id]
        link_ready_time = 0.0
        if npu.link_active_flow is not None:
            link_ready_time = npu.link_active_flow.link_end_time
        for flow in npu.link_pending:
            link_ready_time = max(link_ready_time, flow.link_enqueue_time)
            link_ready_time += npu_link_service_time_ms(flow.total_gb)

        reservations = []
        for disk_state in self.disk_states:
            for flow in disk_state.active_flows:
                if flow.npu_id == npu_id:
                    reservations.append(
                        (self._reservation_key(flow, flow.end_time), flow)
                    )
        reservations.sort(key=lambda item: item[0])

        reservation_keys = []
        prefix_link_end_times = []
        predicted_end = link_ready_time
        for reservation_key, flow in reservations:
            predicted_end = max(predicted_end, reservation_key[0])
            predicted_end += npu_link_service_time_ms(flow.total_gb)
            reservation_keys.append(reservation_key)
            prefix_link_end_times.append(predicted_end)

        timeline = (
            tuple(reservation_keys),
            tuple(prefix_link_end_times),
            link_ready_time,
        )
        self._timeline_cache[npu_id] = timeline
        self._timeline_dirty[npu_id] = False
        self.timeline_rebuilds += 1
        return timeline

    def predict_link_completion(self, flow, current_time, disk_bw):
        """Predict this candidate's exact FCFS completion among known work."""
        npu_id = flow.npu_id
        timeline = (
            self._rebuild_timeline(npu_id)
            if self._timeline_dirty[npu_id]
            else self._timeline_cache[npu_id]
        )
        reservation_keys, prefix_link_end_times, link_ready_time = timeline
        ssd_end_time = current_time + ssd_command_service_time_ms(
            flow.total_gb, disk_bw
        )
        candidate_key = self._reservation_key(flow, ssd_end_time)
        position = bisect.bisect_left(reservation_keys, candidate_key)
        predecessor_end = (
            prefix_link_end_times[position - 1]
            if position
            else link_ready_time
        )
        self.predictions += 1
        return max(predecessor_end, ssd_end_time) + npu_link_service_time_ms(
            flow.total_gb
        )

    def choose(self, candidates, current_time, disk_bw):
        """Choose one disk-local per-NPU FCFS head without ever idling."""
        best_flow = candidates[0]
        best_predicted_end = self.predict_link_completion(
            best_flow, current_time, disk_bw
        )
        best_priority = global_link_aware_priority(
            best_flow,
            best_predicted_end,
            self.npus[best_flow.npu_id].pending_blocks[best_flow.layer],
        )
        for flow in candidates[1:]:
            predicted_end = self.predict_link_completion(
                flow, current_time, disk_bw
            )
            pending_layer_blocks = self.npus[flow.npu_id].pending_blocks[flow.layer]
            priority = global_link_aware_priority(
                flow, predicted_end, pending_layer_blocks
            )
            if priority < best_priority:
                best_flow = flow
                best_priority = priority
        self.selections += 1
        return best_flow


@dataclass
class _DynamicLayerDemand:
    input_demand_gbps: float
    deadline_ms: float
    remaining_ssd_gb_by_disk: dict
    remaining_ssd_gb_total: float


class DynamicCIRController:
    """Maintain only currently visible per-NPU layer work for CIR adverts."""

    def __init__(self, npus, config):
        self.npus = npus
        self.config = config
        self.layers = {}

    def register_layer(
        self,
        *,
        npu_id,
        layer,
        input_demand_gbps,
        deadline_ms,
        work_by_disk,
    ):
        self.layers[(npu_id, layer)] = _DynamicLayerDemand(
            input_demand_gbps=float(input_demand_gbps),
            deadline_ms=float(deadline_ms),
            remaining_ssd_gb_by_disk=dict(work_by_disk),
            remaining_ssd_gb_total=sum(work_by_disk.values()),
        )

    def complete_ssd(self, flow):
        key = (flow.npu_id, flow.layer)
        state = self.layers[key]
        remaining = state.remaining_ssd_gb_by_disk[flow.disk_id] - flow.total_gb
        state.remaining_ssd_gb_total -= flow.total_gb
        if remaining <= _EPS:
            del state.remaining_ssd_gb_by_disk[flow.disk_id]
        else:
            state.remaining_ssd_gb_by_disk[flow.disk_id] = remaining
        if not state.remaining_ssd_gb_by_disk:
            del self.layers[key]

    def _link_backlog_gb(self, npu_id, current_time):
        npu = self.npus[npu_id]
        active_gb = (
            max(0.0, npu.link_active_flow.link_end_time - current_time)
            * NPU_BW_LIMIT
            / 1000.0
            if npu.link_active_flow is not None
            else 0.0
        )
        return active_gb + npu.link_pending_gb

    def demands_for_layer(self, npu_id, layer, current_time):
        state = self.layers[(npu_id, layer)]
        work_by_disk = state.remaining_ssd_gb_by_disk
        if self.config.mode == DYNAMIC_CIR_DEMAND_PROPORTIONAL:
            return capped_input_demands(
                NPU_BW_LIMIT,
                state.input_demand_gbps,
                work_by_disk,
            )
        return slack_link_guarded_demands(
            NPU_BW_LIMIT,
            current_time,
            state.deadline_ms,
            self._link_backlog_gb(npu_id, current_time),
            work_by_disk,
        )

    def demand_for_disk(self, npu_id, layer, disk_id, current_time):
        state = self.layers[(npu_id, layer)]
        total_work = {0: state.remaining_ssd_gb_total}
        if self.config.mode == DYNAMIC_CIR_DEMAND_PROPORTIONAL:
            total_demand = capped_input_demands(
                NPU_BW_LIMIT,
                state.input_demand_gbps,
                total_work,
            )[0]
        else:
            total_demand = slack_link_guarded_demands(
                NPU_BW_LIMIT,
                current_time,
                state.deadline_ms,
                self._link_backlog_gb(npu_id, current_time),
                total_work,
            )[0]
        return (
            total_demand
            * state.remaining_ssd_gb_by_disk[disk_id]
            / state.remaining_ssd_gb_total
        )


def _weighted_capped_split(capacity, items, weights, limits):
    """工作保守的加权带宽分配，每个对象还有各自的带宽上限。"""
    grants = {item: 0.0 for item in items}
    remaining = max(0.0, float(capacity))
    eligible = {item for item in items if weights[item] > _EPS and limits[item] > _EPS}
    while remaining > _EPS and eligible:
        total_weight = sum(weights[item] for item in eligible)
        unit = remaining / total_weight
        saturated = [
            item
            for item in eligible
            if limits[item] - grants[item] <= unit * weights[item] + _EPS
        ]
        if not saturated:
            for item in eligible:
                grants[item] += unit * weights[item]
            break
        for item in saturated:
            addition = max(0.0, limits[item] - grants[item])
            grants[item] += addition
            remaining -= addition
            eligible.remove(item)
    return grants


def _static_qos_service_rates(backlogged_paths, disk_bw, group_weights):
    """按当前有效 CIR 计算离散仲裁长期服务率，不并行传输 I/O。

    CIR 先形成基础服务权，剩余服务机会再按组间和组内 WRR 分配。返回值只用于
    计算各 Path 的虚拟完成标签；真正进入后端的始终只有一个 I/O。
    """
    assigned = {path: path.cir for path in backlogged_paths}
    remaining = max(0.0, float(disk_bw) - sum(assigned.values()))
    paths_by_group = defaultdict(list)
    for path in backlogged_paths:
        paths_by_group[path.group_id].append(path)
    if remaining <= _EPS:
        return assigned

    active_groups = [
        group_id
        for group_id, paths in paths_by_group.items()
        if group_weights[group_id] > _EPS
        and any(path.path_weight > _EPS for path in paths)
    ]
    active_group_weight = sum(group_weights[group_id] for group_id in active_groups)
    for group_id in active_groups:
        paths = paths_by_group[group_id]
        path_weight = sum(path.path_weight for path in paths if path.path_weight > _EPS)
        group_grant = remaining * group_weights[group_id] / active_group_weight
        for path in paths:
            if path.path_weight > _EPS:
                assigned[path] += group_grant * path.path_weight / path_weight
    return assigned


class DiskIOScheduler:
    """SSD 侧调度器；公共仿真入口只会传入本项目支持的两种策略。"""

    def __init__(  # 创建一块 SSD 的调度器，并在这里一次性安装静态 QoS 配置。
        self,  # 当前 SSD 调度器对象。
        disk_state,  # 事件循环共享的这块 SSD 的运行状态。
        policy,  # 磁盘策略：基线绕过或静态 CIR QoS。
        disk_bw,  # 这块物理 SSD 的总带宽，单位为 GB/s。
        qos_config=None,  # 静态 QoS 寄存器配置；基线绕过模式不使用。
        global_link_coordinator=None,
        dynamic_cir_controller=None,
        dynamic_path_owners=None,
        dynamic_cir_config=None,
    ):
        self.state = disk_state  # 保存这块 SSD 的共享状态。
        self.policy = policy  # 保存本次仿真选用的磁盘策略。
        self.disk_bw = float(disk_bw)  # 保存物理带宽，并统一转换成浮点数。
        self.group_weights = ()  # 组间 WRR 权重稍后从静态配置复制一次。
        self.paths = {}  # 创建“Path ID -> PathQueue”的完整映射。
        self.baseline_queues = defaultdict(deque)  # Baseline 每个 NPU 的 FCFS I/O 队列。
        self.baseline_rr_sources = deque()  # 有积压的 NPU 按逐 I/O RR 排列。
        self.baseline_sources_in_rr = set()  # 防止同一 NPU 重复进入 RR 环。
        self.demand_queues = defaultdict(deque)
        self.demand_targets = {}
        self.demand_virtual_finish = {}
        self._demand_rates = None
        self._demand_finish_heap = None
        self.oracle_heap = []
        self.global_queues = defaultdict(deque)
        self.global_link_coordinator = global_link_coordinator
        self.dynamic_cir_controller = dynamic_cir_controller
        self.dynamic_path_owners = dynamic_path_owners or {}
        self.dynamic_cir_config = dynamic_cir_config
        self.dynamic_cir_epochs = 0
        self.dynamic_cir_total_gbps = 0.0
        self.dynamic_cir_max_total_gbps = 0.0
        self._dynamic_pending_path_ids = set()
        self._dynamic_nonzero_cir_paths = set()
        self.qos_rr_cursor = 0  # QoS 虚拟完成标签完全相同时的最终 RR 起点。
        self._qos_floor_heap = []
        self._qos_floor_versions = [0] * 256
        # 只有“哪些 Path 的 pending 队列非空”改变时，长期服务率才会
        # 改变。同一组积压下的逐 I/O 仲裁复用该结果，不改变选择顺序。
        self._qos_backlogged_paths_cache = None
        self._qos_service_rates_cache = None
        self._qos_finish_heap = None
        self._qos_finish_buckets = None
        self.backend_dispatches = 0  # 整块 SSD 实际下发到后端的 I/O 数。
        self.max_backend_active_io = 0  # 硬件语义要求该统计永远不超过 1。

        self.flows_enqueued = 0  # 累计收到的仿真 I/O 流数量。
        self.blocks_enqueued = 0  # 累计收到的逻辑数据块数量。
        self.outstanding_blocks = 0  # 当前整块 SSD 尚未完成的数据块数量。
        self.max_outstanding_blocks = 0  # 整块 SSD 的历史最大未完成数据块数。
        self.dispatch_cycles = 0  # 累计执行后端选取或续跑检查的次数。
        self.dispatch_events = 0  # 累计创建后端调度事件的次数。
        self.pressure_reports = 0  # 客户端累计读取 256-Path 压力快照的次数。
        self.pending_dispatch_time = None  # 当前已经预约的调度时刻。
        self.dispatch_generation = 0  # 用于识别过期调度事件的版本号。
        self.total_queue_wait_ms = 0.0  # 所有逻辑数据块累计等待时间。
        self.max_queue_wait_ms = 0.0  # 单条 I/O 观察到的最大等待时间。
        self._path_io_counts = [0] * 256 if policy in PATH_QOS_POLICIES else []
        self.enqueued_path_ids = set()
        self._path_io_snapshot_cache = None
        self._group_io_counts = [0] * 8 if policy in PATH_QOS_POLICIES else []
        self._active_paths_per_group = [0] * 8 if policy in PATH_QOS_POLICIES else []
        self._active_path_weights = [0.0] * 8 if policy in PATH_QOS_POLICIES else []
        self._active_group_weight_sum = 0.0
        self._active_cir_sum = 0.0

        if policy in QOS_POLICIES:  # 所有静态 QoS 数据面策略都创建 256 个 Path。
            # 创建 SSD 时只检查并复制一次硬件寄存器配置。
            # 运行时只改变队列、虚拟服务标签和当前后端 I/O，绝不会重写
            # CIR、PIR、Path 权重或组权重。
            if sum(qos_config.path_cirs) > self.disk_bw + _EPS:  # 检查总保证带宽可实现。
                raise ValueError(  # 配置超过物理能力时立即指出静态配置错误。
                    "所有 Path 的 CIR 总和超过了 SSD 物理带宽"  # 给出明确中文错误原因。
                )  # 结束错误构造。
            if any(np.isfinite(pir) for pir in qos_config.path_pirs):
                raise ValueError(
                    "当前命令级 QoS 模型只支持不限 PIR；"
                    "有限 PIR 需要硬件 token-bucket 桶深与 burst 规格"
                )
            self.group_weights = tuple(  # 一次性复制静态组间 WRR 权重。
                qos_config.group_weights  # 数据来源是初始化配置，不是运行时客户端。
            )  # 元组形式避免调度过程中意外改写。
            for path_id in range(qos_config.path_count):  # 依次创建 Path 0 到 Path 255。
                group_id = (  # 根据连续布局计算当前 Path 所属组。
                    path_id // qos_config.paths_per_group  # 全局编号整除每组 Path 数。
                )  # 得到 0 到 7 的组编号。
                self.paths[path_id] = PathQueue(  # 把新 Path 放到可按 ID 查找的映射中。
                    path_id,  # Path 的全局编号。
                    qos_config.path_cirs[path_id],  # 初始化时写入该 Path 的 CIR。
                    qos_config.path_pirs[path_id],  # 初始化时写入该 Path 的 PIR。
                    qos_config.path_weights[path_id],  # 初始化时写入该 Path 的组内权重。
                    group_id,  # 初始化时写入该 Path 的所属组。
                )  # 运行期间只改变队列状态和离散服务顺序，不重建静态配置。
        elif policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
            self.group_weights = (1.0,) * 8
            for path_id in range(256):
                self.paths[path_id] = PathQueue(
                    path_id,
                    0.0,
                    float("inf"),
                    1.0,
                    path_id // 32,
                )
        disk_state.scheduler = self  # 让共享 SSD 状态反向引用刚创建的调度器。

    @property  # 允许调用方像读取普通字段一样判断当前是否为 QoS 模式。
    def is_qos(self):  # 判断 I/O 是否需要按客户端填写的 Path ID 入队。
        return self.policy in PATH_QOS_POLICIES

    def report_path_io_counts(  # 定义 QoS 向客户端提供 Path 压力的接口。
        self,  # 当前这块 SSD 的调度器对象。
        current_time,  # 客户端读取报告时的仿真时间。
    ):
        """返回这块 SSD 当前每个 Path 的未完成 I/O 数快照。

        这是客户端从 QoS 获取的唯一动态遥测信号。返回元组的下标是 Path ID，
        数值是该 Path 的正在执行 I/O 数与排队等待 I/O 数之和。

        真实实现应把这里的直接函数调用替换为 QoS 遥测传输，再把接收到的快照
        传给 ``client_select_qos_paths``。

        当前仿真假设报告零延迟，并且一个客户端提交后能立即被下一个客户端看到。
        真实系统若采用周期上报，客户端还应记录“已经本地提交、但尚未反映到下一份
        硬件报告”的 I/O。元组不可修改只表示数据结构固定，不表示报告永不过期。
        """
        self.pressure_reports += 1  # 一次调用对应一次完整 256-Path 压力读取。
        self.settle(current_time)  # 先把所有传输进度推进到报告时刻，避免读取旧状态。
        if self._path_io_snapshot_cache is None:
            self._path_io_snapshot_cache = tuple(self._path_io_counts)
        return self._path_io_snapshot_cache

    def report_path_pressure_analysis(self, current_time):
        """仿真内部复用增量聚合，数值等价于分析完整 256-count。"""
        self.pressure_reports += 1
        self.settle(current_time)
        return _QoSCountAnalysis(
            counts=self._path_io_counts,
            group_io_counts=tuple(self._group_io_counts),
            active_paths_per_group=tuple(self._active_paths_per_group),
            active_path_weights=tuple(self._active_path_weights),
            active_group_weight_sum=self._active_group_weight_sum,
            active_cir_sum=self._active_cir_sum,
        )

    def _change_path_io_count(self, path_id, delta):
        old_count = self._path_io_counts[path_id]
        new_count = old_count + delta
        path = self.paths[path_id]
        group_id = path.group_id
        self._path_io_counts[path_id] = new_count
        self._group_io_counts[group_id] += delta
        if old_count == 0 and new_count > 0:
            if self._active_paths_per_group[group_id] == 0:
                self._active_group_weight_sum += self.group_weights[group_id]
            self._active_paths_per_group[group_id] += 1
            self._active_path_weights[group_id] += path.path_weight
            self._active_cir_sum += path.cir
        elif new_count == 0:
            self._active_paths_per_group[group_id] -= 1
            self._active_path_weights[group_id] -= path.path_weight
            self._active_cir_sum -= path.cir
            if self._active_paths_per_group[group_id] == 0:
                self._active_group_weight_sum -= self.group_weights[group_id]
        self._path_io_snapshot_cache = None

    def _account_until(self, current_time):  # 把整盘忙闲时间统计推进到指定时刻。
        if current_time <= self.state.last_event_time:  # 时间没有前进时无须重复累计。
            return  # 保持统计不变并直接结束。
        duration = current_time - self.state.last_event_time  # 计算本次新增时间区间。
        if self.state.active_flows:  # 至少存在一条正在执行的流时，SSD 处于忙状态。
            self.state.busy_time += duration  # 累加物理 SSD 忙碌时间。
            used_bw = sum(  # 汇总当前全部正在执行流实际获得的带宽。
                flow.bw  # 每条流贡献本轮仲裁分给它的带宽。
                for flow in self.state.active_flows  # 遍历整盘正在执行流集合。
                if flow.active  # 忽略已经完成但尚未从集合清理的流。
            )  # 得到当前时刻整盘已使用带宽。
            unused_bw = max(0.0, self.disk_bw - used_bw)  # 计算未被利用的物理带宽。
            self.state.surplus_bw_integral += unused_bw * duration  # 累加空余带宽积分。
            self.state.total_bw_integral += self.disk_bw * duration  # 累加总带宽积分。
        else:  # 没有正在执行的流时，SSD 处于空闲状态。
            self.state.idle_time += duration  # 累加物理 SSD 空闲时间。
        self.state.last_event_time = current_time  # 记录统计已推进到的最新时刻。

    def settle(self, current_time):  # 在生成压力报告前，先结算所有 I/O 的传输进度。
        """按照旧带宽把正在执行的流推进到 ``current_time`` 所表示的时刻。"""
        if current_time <= self.state.last_event_time + _EPS:  # 时间未前进时没有传输增量。
            return  # 不改变剩余数据量，直接结束。
        self._account_until(current_time)  # 先更新整盘忙闲与带宽利用统计。
        for flow in self.state.active_flows:  # 逐条推进当前整盘正在执行的 I/O 流。
            elapsed_ms = max(  # 计算这条流使用当前带宽持续了多少毫秒。
                0.0,  # 防止浮点误差产生负时间。
                current_time - flow.start_time,  # 当前时刻减去上次结算时刻。
            )  # 得到非负的传输持续时间。
            if (  # 只有真正执行且拥有正带宽、正持续时间的流才会传输数据。
                flow.active  # 这条流仍在执行集合中。
                and flow.bw > 0  # 这条流本轮获得了带宽。
                and elapsed_ms > 0  # 距离上次结算确实经过了时间。
            ):
                transferred = min(  # 计算本时间段实际完成的数据量。
                    flow.remaining_gb,  # 实际传输量不能超过尚未完成的数据量。
                    flow.bw * elapsed_ms / 1000.0,  # 带宽乘时间，并把毫秒换算成秒。
                )  # 得到本次可从剩余量中扣除的数据量。
                flow.remaining_gb -= transferred  # 更新这条流尚未传输的数据量。
            flow.start_time = current_time  # 下一次结算从当前时刻继续计算。
            if flow.remaining_gb <= _EPS:  # 剩余量接近 0 时认为 I/O 已经完成。
                flow.remaining_gb = 0.0  # 清除浮点尾差，明确标记没有剩余数据。
                flow.end_time = current_time  # 把预计完成时刻更新为当前时刻。

    def _activate_flow(self, flow, current_time):  # 让 Path 队头 I/O 加入整盘带宽竞争。
        if self.state.active_flows:
            raise RuntimeError("同一块 SSD 后端不能同时执行多个 I/O")
        wait_ms = max(  # 计算这条 I/O 在 Path 队列中等待了多久。
            0.0,  # 防止浮点误差产生负等待时间。
            current_time - flow.enqueue_time,  # 启动时刻减去进入 SSD 的时刻。
        )  # 得到非负等待时间，单位为毫秒。
        flow.ssd_queue_wait_ms = wait_ms
        self.total_queue_wait_ms += (  # 累加所有逻辑数据块的总等待时间。
            wait_ms * flow.block_count  # 聚合流需要按其代表的数据块数量加权。
        )  # 完成总等待时间更新。
        self.max_queue_wait_ms = max(  # 更新单条 I/O 的历史最大等待时间。
            self.max_queue_wait_ms,  # 保留过去的最大值。
            wait_ms,  # 与当前 I/O 的等待时间比较。
        )  # 得到新的最大等待时间。
        flow.active = True  # 只有仲裁胜出的这一条 I/O 才是 backend active。
        flow.start_time = current_time  # 记录这条流进入唯一后端执行位的时刻。
        flow.bw = 0.0  # 尚未执行下一轮仲裁，所以初始实时带宽为 0。
        flow.end_time = float("inf")  # 获得正带宽前无法估算完成时刻。
        self.state.active_flows.append(flow)  # 加入整块 SSD 的正在执行流集合。

    def _enqueue_baseline(self, flow):
        """把一个 Baseline I/O 放入其 NPU 的 FCFS 队列并加入 RR 环。"""
        source_id = flow.npu_id
        self.baseline_queues[source_id].append(flow)
        if source_id not in self.baseline_sources_in_rr:
            self.baseline_rr_sources.append(source_id)
            self.baseline_sources_in_rr.add(source_id)

    def _pop_baseline_rr(self):
        """从有积压的 NPU 中逐源 RR，每次只取一个逻辑 I/O。"""
        while self.baseline_rr_sources:
            source_id = self.baseline_rr_sources.popleft()
            self.baseline_sources_in_rr.discard(source_id)
            queue = self.baseline_queues[source_id]
            if not queue:
                self.baseline_queues.pop(source_id, None)
                continue
            flow = queue.popleft()
            if queue:
                self.baseline_rr_sources.append(source_id)
                self.baseline_sources_in_rr.add(source_id)
            else:
                self.baseline_queues.pop(source_id, None)
            return flow
        return None

    def _enqueue_demand_aware(self, flow):
        source_id = flow.npu_id
        queue = self.demand_queues[source_id]
        if not queue:
            active_finish = [
                self.demand_virtual_finish[source]
                for source, source_queue in self.demand_queues.items()
                if source_queue and source != source_id
            ]
            self.demand_virtual_finish[source_id] = (
                min(active_finish) if active_finish else 0.0
            )
            self.demand_targets[source_id] = flow.demand_gbps
            self._demand_rates = None
            self._demand_finish_heap = None
        queue.append(flow)

    def _build_demand_finish_heap(self):
        active = {
            source: self.demand_targets[source]
            for source, queue in self.demand_queues.items()
            if queue
        }
        self._demand_rates = max_min_work_conserving_rates(self.disk_bw, active)
        self._demand_finish_heap = []
        for source, rate in self._demand_rates.items():
            head = self.demand_queues[source][0]
            finish = self.demand_virtual_finish[source] + head.total_gb / rate
            heapq.heappush(self._demand_finish_heap, (finish, source))

    def _pop_demand_aware(self):
        if self._demand_finish_heap is None:
            self._build_demand_finish_heap()
        if not self._demand_finish_heap:
            return None
        finish, source = heapq.heappop(self._demand_finish_heap)
        queue = self.demand_queues[source]
        flow = queue.popleft()
        self.demand_virtual_finish[source] = finish
        if queue:
            next_finish = finish + queue[0].total_gb / self._demand_rates[source]
            heapq.heappush(self._demand_finish_heap, (next_finish, source))
        else:
            del self.demand_queues[source]
            del self.demand_targets[source]
            self._demand_rates = None
            self._demand_finish_heap = None
        return flow

    def _enqueue_oracle(self, flow):
        heapq.heappush(self.oracle_heap, (omniscient_edf_key(flow), flow))

    def _pop_oracle(self):
        return heapq.heappop(self.oracle_heap)[1] if self.oracle_heap else None

    def _enqueue_global_link_aware(self, flow):
        self.global_queues[flow.npu_id].append(flow)

    def _pop_global_link_aware(self, current_time):
        candidates = [queue[0] for queue in self.global_queues.values() if queue]
        if not candidates:
            return None
        flow = self.global_link_coordinator.choose(
            candidates, current_time, self.disk_bw
        )
        queue = self.global_queues[flow.npu_id]
        queue.popleft()
        if not queue:
            del self.global_queues[flow.npu_id]
        return flow

    def _publish_qos_floor(self, path):
        path_id = path.path_id
        self._qos_floor_versions[path_id] += 1
        if path.has_work():
            heapq.heappush(
                self._qos_floor_heap,
                (
                    path.virtual_finish,
                    path_id,
                    self._qos_floor_versions[path_id],
                ),
            )

    def _qos_virtual_floor(self, excluded_path=None):
        while self._qos_floor_heap:
            virtual_finish, path_id, version = self._qos_floor_heap[0]
            path = self.paths[path_id]
            if (
                version != self._qos_floor_versions[path_id]
                or not path.has_work()
                or virtual_finish != path.virtual_finish
            ):
                heapq.heappop(self._qos_floor_heap)
                continue
            return virtual_finish
        return 0.0

    def _invalidate_qos_arbitration_cache(self):
        """使下一次离散仲裁重建 backlogged Path 集合与服务率。"""
        self._qos_backlogged_paths_cache = None
        self._qos_service_rates_cache = None
        self._qos_finish_heap = None
        self._qos_finish_buckets = None

    def apply_dynamic_cir_epoch(self, current_time):
        """Atomically configure online CIRs for the next nonpreemptive command."""
        backlogged_by_source = defaultdict(list)
        source_layers = {}
        for path_id in sorted(self._dynamic_pending_path_ids):
            path = self.paths[path_id]
            source = self.dynamic_path_owners[path.path_id]
            backlogged_by_source[source].append(path)
            source_layers[source] = path.peek().layer

        demands = {
            source: self.dynamic_cir_controller.demand_for_disk(
                source,
                source_layers[source],
                self.state.disk_id,
                current_time,
            )
            for source in sorted(backlogged_by_source)
        }
        grants = (
            proportional_capacity_grants(self.disk_bw, demands)
            if demands
            else {}
        )
        next_cirs = {}
        for source, paths in backlogged_by_source.items():
            work_by_path = {
                path.path_id: path.pending_gb
                for path in paths
            }
            source_work = sum(work_by_path.values())
            for path_id, work_gb in work_by_path.items():
                next_cirs[path_id] = grants[source] * work_gb / source_work

        total_cir = sum(next_cirs.values())
        if total_cir > self.disk_bw:
            last_path = max(
                path_id for path_id, cir in next_cirs.items() if cir > 0.0
            )
            next_cirs[last_path] -= total_cir - self.disk_bw
            total_cir = self.disk_bw
        affected_paths = self._dynamic_nonzero_cir_paths | set(next_cirs)
        changed = any(
            abs(self.paths[path_id].cir - next_cirs.get(path_id, 0.0)) > _EPS
            for path_id in affected_paths
        )
        if changed:
            for path_id in affected_paths:
                self.paths[path_id].cir = next_cirs.get(path_id, 0.0)
            self._active_cir_sum = sum(
                cir
                for path_id, cir in next_cirs.items()
                if self._path_io_counts[path_id] > 0
            )
            self._invalidate_qos_arbitration_cache()
            self.dynamic_cir_epochs += 1
            self._dynamic_nonzero_cir_paths = {
                path_id for path_id, cir in next_cirs.items() if cir > _EPS
            }
        self.dynamic_cir_total_gbps = total_cir
        self.dynamic_cir_max_total_gbps = max(
            self.dynamic_cir_max_total_gbps,
            total_cir,
        )
        return grants

    def _push_qos_candidate(self, path):
        """把一个 Path 的当前队头加入 finish-tag bucket。"""
        rate = self._qos_service_rates_cache.get(path, 0.0)
        head = path.peek()
        if head is None or rate <= _EPS or path.pir <= _EPS:
            return
        finish_tag = path.virtual_finish + head.total_gb / rate
        path_ids = self._qos_finish_buckets.get(finish_tag)
        if path_ids is None:
            path_ids = []
            self._qos_finish_buckets[finish_tag] = path_ids
            heapq.heappush(self._qos_finish_heap, finish_tag)
        bisect.insort(path_ids, path.path_id)

    def _build_qos_arbitration_cache(self):
        """仅在非空 Path 集合改变时重建服务率和候选 heap。"""
        backlogged = [path for path in self.paths.values() if path.pending]
        self._qos_backlogged_paths_cache = backlogged
        self._qos_service_rates_cache = _static_qos_service_rates(
            backlogged, self.disk_bw, self.group_weights
        )
        self._qos_finish_heap = []
        self._qos_finish_buckets = {}
        for path in backlogged:
            self._push_qos_candidate(path)
        if backlogged and not self._qos_finish_heap:
            raise RuntimeError("QoS 有待处理 I/O，但 CIR/PIR/WRR 没有产生可服务 Path")

    def _select_qos_path(self):
        """按 CIR、两级 WRR 和最终 RR 选择一个非空 Path。"""
        if self._qos_finish_heap is None:
            self._build_qos_arbitration_cache()
        if not self._qos_backlogged_paths_cache:
            return None
        if not self._qos_finish_heap:
            raise RuntimeError("QoS 有待处理 I/O，但 CIR/PIR/WRR 没有产生可服务 Path")

        best_finish = heapq.heappop(self._qos_finish_heap)
        tied_tags = [best_finish]
        while self._qos_finish_heap and self._qos_finish_heap[0] <= best_finish + _EPS:
            tied_tags.append(heapq.heappop(self._qos_finish_heap))

        selected_key = None
        selected_position = None
        for finish_tag in tied_tags:
            path_ids = self._qos_finish_buckets[finish_tag]
            position = bisect.bisect_left(path_ids, self.qos_rr_cursor)
            if position == len(path_ids):
                position = 0
            path_id = path_ids[position]
            key = (
                (path_id - self.qos_rr_cursor) % len(self.paths),
                path_id,
                finish_tag,
            )
            if selected_key is None or key < selected_key:
                selected_key = key
                selected_position = position

        _, selected_path_id, selected_finish = selected_key
        selected_ids = self._qos_finish_buckets[selected_finish]
        selected_ids.pop(selected_position)
        for finish_tag in tied_tags:
            if self._qos_finish_buckets[finish_tag]:
                heapq.heappush(self._qos_finish_heap, finish_tag)
            else:
                del self._qos_finish_buckets[finish_tag]
        selected = self.paths[selected_path_id]
        selected.virtual_finish = selected_finish
        self._publish_qos_floor(selected)
        self.qos_rr_cursor = (selected.path_id + 1) % len(self.paths)
        return selected

    def _dispatch_one(self, current_time):
        """数据面空闲时只仲裁并启动一条命令。

        Baseline 与 QoS 共用完全相同的 SSD 服务时间：命令大小除以
        整盘服务能力。CIR/WRR 只决定 QoS 选中哪条命令，不改变
        命令本身的服务成本。
        """
        if self.state.active_flows:
            return self.state.active_flows[0]

        selected_path = None
        if self.policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
            self.apply_dynamic_cir_epoch(current_time)
        if self.is_qos:
            selected_path = self._select_qos_path()
            flow = selected_path.activate_next() if selected_path is not None else None
            if selected_path is not None:
                if (
                    self.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
                    and not selected_path.pending
                ):
                    self._dynamic_pending_path_ids.discard(selected_path.path_id)
                if selected_path.pending:
                    self._push_qos_candidate(selected_path)
                else:
                    # Path 离开 backlogged 集合后才需要重算长期服务率。
                    self._invalidate_qos_arbitration_cache()
        elif self.policy == POLICY_QOS_DEMAND_MAXMIN:
            flow = self._pop_demand_aware()
        elif self.policy == POLICY_OMNISCIENT_EDF:
            flow = self._pop_oracle()
        elif self.policy == POLICY_GLOBAL_LINK_AWARE:
            flow = self._pop_global_link_aware(current_time)
        else:
            flow = self._pop_baseline_rr()
        if flow is None:
            return None

        self._activate_flow(flow, current_time)
        command_service_gbps = self.disk_bw
        flow.bw = command_service_gbps
        flow.end_time = current_time + ssd_command_service_time_ms(
            flow.remaining_gb, command_service_gbps
        )
        if self.policy == POLICY_GLOBAL_LINK_AWARE:
            self.global_link_coordinator.mark_npu_dirty(flow.npu_id)
        self.backend_dispatches += 1
        self.max_backend_active_io = max(
            self.max_backend_active_io, len(self.state.active_flows)
        )
        return flow

    def enqueue_many(self, flows, current_time):  # 接收客户端已经填写好 Path ID 的 I/O。
        self.settle(current_time)  # 先把旧 I/O 的进度推进到本批请求的到达时刻。
        for flow in flows:  # 按客户端给出的顺序逐个接收本批 I/O。
            if self.is_qos:  # 静态 QoS 模式需要按客户端填写的 Path ID 入队。
                if (
                    self.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
                    and self.dynamic_path_owners[flow.queue_id] != flow.npu_id
                ):
                    raise ValueError("dynamic CIR I/O selected a Path owned by another NPU")
                selected_path = self.paths[flow.queue_id]  # 用 Path ID 找到目标 Path。
                was_empty = not selected_path.has_work()
                pending_was_empty = not selected_path.pending
                selected_path.enqueue(flow)  # 非空 Path 也照常接收并排队。
                self.enqueued_path_ids.add(flow.queue_id)
                if self.policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
                    self._dynamic_pending_path_ids.add(flow.queue_id)
                self._change_path_io_count(flow.queue_id, flow.block_count)
                if pending_was_empty:
                    self._invalidate_qos_arbitration_cache()
                if was_empty:
                    selected_path.virtual_finish = self._qos_virtual_floor(
                        excluded_path=selected_path
                    )
                    self._publish_qos_floor(selected_path)
            elif self.policy == POLICY_QOS_DEMAND_MAXMIN:
                flow.queue_id = -1
                self._enqueue_demand_aware(flow)
            elif self.policy == POLICY_OMNISCIENT_EDF:
                flow.queue_id = -1
                self._enqueue_oracle(flow)
            elif self.policy == POLICY_GLOBAL_LINK_AWARE:
                flow.queue_id = -1
                self._enqueue_global_link_aware(flow)
            else:  # 基线绕过模式没有 QoS Path 这一层。
                flow.queue_id = -1  # 用 -1 明确标记该 I/O 没有 Path ID。
                self._enqueue_baseline(flow)  # 等待后端按 NPU 源队列逐 I/O RR。
            self.flows_enqueued += 1  # 累计调度器收到的仿真流数量。
            self.blocks_enqueued += flow.block_count  # 累计收到的逻辑数据块数量。
            self.outstanding_blocks += flow.block_count  # 增加整盘未完成数据块数。
        self.max_outstanding_blocks = max(  # 更新整块 SSD 的历史最大未完成数据块数。
            self.max_outstanding_blocks,  # 保留过去观察到的最大值。
            self.outstanding_blocks,  # 与本批 I/O 入队后的当前值比较。
        )  # 得到新的历史最大值。

    def complete_ready_flows(self, current_time):  # 完成到期 I/O，并更新下一份 Path 压力。
        self.settle(current_time)  # 先把全部流的剩余数据量结算到当前时刻。
        if len(self.state.active_flows) > 1:
            raise RuntimeError("同一块 SSD 后端观察到多个 active I/O")
        completed = []  # 创建本次确认完成的 I/O 列表。
        for flow in self.state.active_flows:  # 逐条检查整盘正在执行的 I/O 流。
            if flow.remaining_gb <= _EPS:  # 剩余数据量接近 0 表示这条流已经完成。
                completed.append(flow)  # 记住它，后面统一更新整盘和 Path 状态。
        if not completed:  # 当前时刻没有任何 I/O 完成时，无须改变队列状态。
            return []  # 返回空列表给事件循环。

        completed_set = set(completed)  # 转成集合，便于快速判断某条流是否已完成。
        still_active = []  # 创建仍需保留在整盘执行集合中的流列表。
        for flow in self.state.active_flows:  # 再次按原顺序遍历整盘执行集合。
            if flow not in completed_set:  # 只保留本次尚未完成的流。
                still_active.append(flow)  # 加入新的整盘执行集合。
        self.state.active_flows = still_active  # 用过滤后的列表替换旧执行集合。

        completed_block_count = 0  # 统计本次完成了多少个逻辑数据块。
        for flow in completed:  # 遍历本次全部完成流并累加其逻辑数据块数量。
            completed_block_count += flow.block_count  # 聚合流可能代表多个数据块。
        self.outstanding_blocks -= completed_block_count  # 从整盘未完成数量中扣除。

        for flow in completed:  # 逐条更新已完成流及其所选 Path 的状态。
            flow.active = False  # 标记该流不再参与整盘带宽竞争。
            self.state.completed_bytes_gb += flow.total_gb
            if self.policy == POLICY_GLOBAL_LINK_AWARE:
                self.global_link_coordinator.mark_npu_dirty(flow.npu_id)
            if flow.queue is None:  # 基线绕过流没有关联的 QoS Path。
                continue  # 基线流无需更新 Path 队列，继续处理下一条完成流。
            path = flow.queue  # 取得客户端当初为这条 I/O 选择的 Path。
            path.complete(flow)  # 清除已完成 I/O；下一条必须重新参加整盘仲裁。
            if not path.has_work():
                self._publish_qos_floor(path)
            self._change_path_io_count(flow.queue_id, -flow.block_count)
        return completed  # 返回完成流，供事件循环通知对应 NPU。

    def request_dispatch(self, current_time, event_heap):
        """为这块 SSD 在当前时间点安排一次后端调度事件。"""
        if (
            self.pending_dispatch_time is not None
            and abs(self.pending_dispatch_time - current_time) <= _EPS
        ):
            return
        self.dispatch_generation += 1
        self.pending_dispatch_time = current_time
        heapq.heappush(
            event_heap,
            (
                current_time,
                DISK_SCHEDULE,
                self.state.disk_id,
                0,
                self.dispatch_generation,
            ),
        )
        self.dispatch_events += 1

    def dispatch(self, current_time, event_heap, schedule_completion=True):
        self.pending_dispatch_time = None
        self.settle(current_time)
        flow = self._dispatch_one(current_time)
        self.dispatch_cycles += 1
        if schedule_completion:
            self._schedule_next_completion(current_time, event_heap)
        return flow

    def _schedule_next_completion(self, current_time, event_heap):
        self.state.generation += 1
        finite_end_times = [
            flow.end_time
            for flow in self.state.active_flows
            if flow.active and np.isfinite(flow.end_time)
        ]
        if not finite_end_times:
            return
        event_time = max(current_time + _EPS, min(finite_end_times))
        heapq.heappush(
            event_heap,
            (
                event_time,
                DISK_COMPLETION,
                self.state.disk_id,
                0,
                self.state.generation,
            ),
        )


# ---------------------------------------------------------------------------
# NPU 请求状态与逐层预取流水线
# ---------------------------------------------------------------------------


class NPUState:
    """一个 NPU 的可变状态；每个 NPU 同一时间只处理一个请求。"""

    def __init__(self, npu_id):
        self.npu_id = npu_id
        self.per_request_io_detail = []
        self.done = True
        self.current_load = {}
        # NPU link 统计跨请求保留；每个 NPU 独立拥有同一个聚合上限。
        self.link_last_update_ms = 0.0
        self.link_current_bw_gbps = 0.0
        self.link_bw_integral_gb = 0.0
        self.link_capacity_integral_gb = 0.0
        self.link_cap_hit_time_ms = 0.0
        self.link_peak_raw_bw_gbps = 0.0
        self.link_peak_effective_bw_gbps = 0.0
        self.link_completed_gb = 0.0
        self.link_pending = deque()
        self.link_pending_gb = 0.0
        self.link_active_flow = None
        self.link_max_outstanding_io = 0
        self.link_max_active_io = 0
        self.link_dispatches = 0
        self.link_total_queue_wait_ms = 0.0
        self.link_max_queue_wait_ms = 0.0
        self.link_total_io_latency_ms = 0.0
        self.link_max_io_latency_ms = 0.0
        self.total_compute_ms = 0.0
        self.completion_generation = 0

    def start_request(self, load, block_placement, start_time):
        """FIFO 调度器分配新请求时，重置这一请求对应的运行状态。"""
        self.current_load = load
        self.done = False

        self.per_layer_us = float(load["per_layer_us"])
        self.category = load["category"]
        self.block_placement = block_placement

        self.request_start_time = start_time
        self.arrival_time = float(load["arrival_time"])
        self.queueing_delay_ms = start_time - self.arrival_time
        self.processing_ttft_ms = 0.0
        self.ttft_ms = None

        self.compute_done_up_to = -1
        self.compute_active = False
        self.compute_generation = 0
        self.pending_blocks = {}
        self.kv_ready_layers = set()
        self.per_layer_kv_load_start = {}
        self.current_compute_expected_end_ms = None

        self.last_compute_end_time = start_time
        self.io_wait_L0_ms = 0.0
        self.io_wait_L1_ms = 0.0
        self.io_wait_L2plus_ms = 0.0
        self.request_io_count = 0
        self.request_ssd_queue_wait_ms = 0.0
        self.request_link_queue_wait_ms = 0.0
        self.request_e2e_io_latency_ms = 0.0
        self.request_max_ssd_queue_wait_ms = 0.0
        self.request_max_link_queue_wait_ms = 0.0


class _SimulationContext:
    def __init__(
        self,
        *,
        event_heap,
        disk_states,
        npus,
        n_layers,
        policy,
        qos_config,
        client_io_config,
        request_path_pools,
        submit_order_seed,
        placement_by_request,
        workload_hash,
        placement_hash,
        workload_seed,
        placement_seed,
        arrival_delay_seed,
        arrival_delay_max_ms,
        global_link_coordinator,
        dynamic_cir_controller,
        dynamic_cir_config,
    ):
        self.event_heap = event_heap
        self.disk_states = disk_states
        self.npus = npus
        self.n_layers = n_layers
        self.workload_hash = workload_hash
        self.placement_hash = placement_hash
        self.workload_seed = workload_seed
        self.placement_seed = placement_seed
        self.policy = policy
        self.client_io_config = client_io_config
        self.client_request_path_pools = request_path_pools
        self.submit_order_seed = submit_order_seed
        self.arrival_delay_seed = arrival_delay_seed
        self.arrival_delay_max_ms = arrival_delay_max_ms
        self.global_link_coordinator = global_link_coordinator
        self.dynamic_cir_controller = dynamic_cir_controller
        self.dynamic_cir_config = dynamic_cir_config
        # 提交顺序使用独立随机源，不能消耗请求抽样或数据放置的随机序列。
        self.submit_rng = np.random.RandomState(submit_order_seed)

        # 下面是客户端选路的初始化阶段：提前缓存四个请求类别各自的合法 Path 池。
        self.client_path_pools = {}  # 创建“请求类别 -> Path ID 元组”的空字典。
        self.client_qos_config = qos_config  # 保存只读静态配置镜像，供预计带宽计算。
        if policy in QOS_POLICIES:  # 所有静态 QoS 数据面策略都需要合法类别 Path 池。
            for category in QOS_ROUTING_CATEGORIES:  # 依次处理 SS、SL、LS、LL。
                category_paths = client_category_paths(  # 计算当前类别的合法 Path 池。
                    category=category,  # 传入当前正在初始化的请求类别。
                    qos_config=qos_config,  # 传入客户端和 SSD 共同知道的静态布局。
                )  # 得到当前类别跨全部组的 Path ID 元组。
                self.client_path_pools[category] = category_paths  # 缓存结果供运行时查表。
        self.completed_requests = 0
        self.stale_events = 0
        self.event_counts = defaultdict(int)

        # 客户端提交器按 ready_time 驱动；同一轮每个 NPU 最多提交一个 batch。
        self.client_submission_states = {}
        # 每个 NPU 同时最多预取一个 layer；该 layer 的各 SSU 状态具有相同
        # ready_time，因此队头 RR 既保持 SSU 公平，也能把每轮扫描限制在 NPU 数。
        self.client_submission_queues = defaultdict(deque)
        self.next_client_submission_id = 0
        self.client_submission_generation = 0
        self.pending_client_submission_time = None
        self.client_submission_rounds = 0
        self.client_submission_batches = 0
        self.client_multi_npu_rounds = 0
        self.client_max_npus_per_round = 0
        self.client_submission_order_sample = []
        # 每个 NPU 独立串行发行命令。默认间隔为 0，完全复现历史的零耗时
        # 提交；正间隔允许其他 NPU、SSD 完成和仲裁事件在两次发行间插入。
        self.client_next_issue_time = [0.0] * len(npus)
        # 用紧凑三态数组审计每个 placement block：0 未提交、1 已提交、2 已完成。
        self.placement_by_request = placement_by_request
        self.block_offsets = {}
        self.expected_block_count = 0
        self.expected_read_gb = 0.0
        for request_id, layers in placement_by_request.items():
            for layer, blocks in layers.items():
                self.block_offsets[(request_id, layer)] = self.expected_block_count
                for _, block_gb in blocks:
                    self.expected_read_gb += block_gb
                self.expected_block_count += len(blocks)
        self.block_states = bytearray(self.expected_block_count)
        self.submitted_block_count = 0
        self.completed_block_count = 0
        self.pending_npu_link_start_ids = set()


@dataclass
class _ClientSubmissionState:
    """一个 ``(request, layer, SSU)`` 尚未提交完的客户端 I/O 序列。"""

    state_id: int
    npu_id: int
    request_id: int
    category: str
    layer: int
    disk_id: int
    blocks: tuple
    ready_time: float
    start_offset: int
    allowed_path_ids: tuple
    demand_gbps: float
    deadline_time: float
    layer_work_gb: float
    cursor: int = 0
    planned_path_ids: list = field(default_factory=list)


def _new_flow(  # 把客户端的一条数据块读取请求转换成仿真 I/O 对象。
    npu,  # 当前请求所在的 NPU。
    layer,  # 当前读取的模型层号。
    block,  # 数据块描述，至少包含编号和大小。
    disk_id,  # 数据放置阶段已经选定的目标 SSD 编号。
    queue_id,  # QoS Path ID；基线绕过策略使用 -1。
    block_count,  # 该仿真流代表多少个逻辑数据块 I/O。
    current_time,  # 客户端提交该 I/O 的仿真时间。
    demand_gbps,
    deadline_time,
    layer_work_gb,
):
    """创建仿真 I/O；真实客户端应在对应位置构造 KV 请求并填写 SQE DW2。"""
    return BlockIOFlow(  # 创建并返回一条新的仿真 I/O 流。
        npu_id=npu.npu_id,  # 保存发起请求的 NPU 编号。
        layer=layer,  # 保存这条 I/O 对应的模型层号。
        block_idx=block["block_idx"],  # 保存逻辑数据块编号。
        disk_id=disk_id,  # 保存目标 SSD 编号。
        total_gb=block["gb"],  # 保存需要传输的总数据量，单位 GB。
        queue_id=queue_id,  # 保存客户端选择的 QoS Path ID。
        block_count=block_count,  # 保存聚合前的逻辑数据块数量。
        enqueue_time=current_time,  # 保存进入 SSD 队列的时间。
        request_id=npu.current_load["request_id"],  # 保存跨 NPU 稳定的请求 ID。
        demand_gbps=demand_gbps,
        deadline_time=deadline_time,
        layer_work_gb=layer_work_gb,
    )  # 返回完整的仿真 I/O 对象。


def _register_submitted_flow(context, flow):
    """在真正 enqueue 前验证 placement，并登记一次性 submit。"""
    expected = context.placement_by_request[flow.request_id][flow.layer][flow.block_idx]
    if flow.disk_id != expected[0]:
        raise AssertionError("block 被提交到非 placement SSU")
    if not math.isclose(flow.total_gb, expected[1], rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("block 大小与 placement 不一致")
    block_id = context.block_offsets[(flow.request_id, flow.layer)] + flow.block_idx
    if context.block_states[block_id] != 0:
        raise AssertionError("同一个 block 被重复 submit")
    context.block_states[block_id] = 1
    context.submitted_block_count += 1


def _register_completed_flow(context, flow):
    """登记一次性 completion，并再次验证目标 SSU。"""
    expected = context.placement_by_request[flow.request_id][flow.layer][flow.block_idx]
    if expected[0] != flow.disk_id:
        raise AssertionError("completion 的 SSU 与 submit 目标不一致")
    block_id = context.block_offsets[(flow.request_id, flow.layer)] + flow.block_idx
    if context.block_states[block_id] != 1:
        raise AssertionError("block 未 submit 或已 complete")
    context.block_states[block_id] = 2
    context.completed_block_count += 1


def _request_client_submission(context):
    """为所有未完成提交状态预约最早的一轮客户端提交事件。"""
    if not context.client_submission_states:
        return
    ready_times = [
        max(
            context.client_submission_states[state_ids[0]].ready_time,
            context.client_next_issue_time[npu_id],
        )
        for npu_id, state_ids in context.client_submission_queues.items()
        if state_ids
    ]
    if not ready_times:
        return
    next_time = min(ready_times)
    pending = context.pending_client_submission_time
    if pending is not None and pending <= next_time + _EPS:
        return
    context.client_submission_generation += 1
    context.pending_client_submission_time = next_time
    heapq.heappush(
        context.event_heap,
        (
            next_time,
            CLIENT_SUBMISSION,
            0,
            0,
            context.client_submission_generation,
        ),
    )


def _client_submit_layer_blocks(context, npu, layer, blocks, current_time):
    """把一层拆成独立的 ``(request, layer, SSU)`` 提交状态。

    数据块此时只完成 SSU 放置，不会原子地提交整层。全局提交器稍后按
    ready_time 取状态；同一时刻每轮随机排列 NPU，且每个 NPU 只发一个 batch。
    """
    blocks_by_disk = defaultdict(list)
    for block in blocks:
        blocks_by_disk[block["disk"]].append(block)

    layer_work_by_disk = {
        disk_id: sum(block["gb"] for block in disk_blocks)
        for disk_id, disk_blocks in blocks_by_disk.items()
    }
    deadline_time = (
        current_time
        if layer == 0
        else npu.current_compute_expected_end_ms
    )
    if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
        context.dynamic_cir_controller.register_layer(
            npu_id=npu.npu_id,
            layer=layer,
            input_demand_gbps=npu.current_load["required_bw_input_gbps"],
            deadline_ms=deadline_time,
            work_by_disk=layer_work_by_disk,
        )
        demand_by_disk = context.dynamic_cir_controller.demands_for_layer(
            npu.npu_id,
            layer,
            current_time,
        )
    elif context.policy in (
        POLICY_QOS_DEMAND_MAXMIN,
        POLICY_OMNISCIENT_EDF,
        POLICY_GLOBAL_LINK_AWARE,
    ):
        demand_by_disk = capped_proportional_demands(
            NPU_BW_LIMIT,
            npu.per_layer_us / 1e6,
            layer_work_by_disk,
        )
    else:
        demand_by_disk = {}

    # 保留 block placement 中各 SSU 的首次出现顺序。随机 placement 会让不同
    # NPU 自然错开目标 SSU，避免所有 NPU 按 0,1,2... 同步扫盘的仿真假象。
    for disk_id, disk_block_list in blocks_by_disk.items():
        disk_blocks = tuple(disk_block_list)
        layer_work_gb = layer_work_by_disk[disk_id]
        demand_gbps = demand_by_disk.get(disk_id, 0.0)
        start_offset = 0
        allowed_path_ids = ()
        if context.policy in PATH_QOS_POLICIES:
            if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
                allowed_paths = context.client_request_path_pools[
                    (npu.npu_id, disk_id)
                ]
            else:
                allowed_paths = (
                    context.client_request_path_pools[
                        (npu.current_load["request_id"], disk_id)
                    ]
                    if context.client_io_config.path_pool_mode
                    == "per_npu_demand_tickets"
                    else context.client_path_pools[npu.category]
                )
            allowed_path_ids = allowed_paths
            start_offset = (
                npu.current_load["request_id"] + layer * 13 + disk_id * 29
            ) % len(allowed_paths)
        state_id = context.next_client_submission_id
        context.next_client_submission_id += 1
        context.client_submission_states[state_id] = _ClientSubmissionState(
            state_id=state_id,
            npu_id=npu.npu_id,
            request_id=npu.current_load["request_id"],
            category=npu.category,
            layer=layer,
            disk_id=disk_id,
            blocks=disk_blocks,
            ready_time=current_time,
            start_offset=start_offset,
            allowed_path_ids=allowed_path_ids,
            demand_gbps=demand_gbps,
            deadline_time=deadline_time,
            layer_work_gb=layer_work_gb,
        )
        context.client_submission_queues[npu.npu_id].append(state_id)
    _request_client_submission(context)


def _plan_qos_pressure_window(context, state, current_time):
    """读取一次 Path 压力，并规划下一个压力窗口内的 Path ID。"""
    window_start = len(state.planned_path_ids)
    if (
        context.client_io_config.path_selection_mode
        == PATH_SELECTION_FIXED_PATH_ZERO
    ):
        state.planned_path_ids.extend(
            0 for _ in range(window_start, len(state.blocks))
        )
        return
    if (
        context.client_io_config.path_selection_mode
        == PATH_SELECTION_STATELESS_RR
    ):
        allowed = state.allowed_path_ids
        state.planned_path_ids.extend(
            allowed[(state.start_offset + index) % len(allowed)]
            for index in range(window_start, len(state.blocks))
        )
        return
    pressure_window_io = context.client_io_config.pressure_window_io
    window_end = (
        len(state.blocks)
        if pressure_window_io is None
        else min(len(state.blocks), window_start + pressure_window_io)
    )
    scheduler = context.disk_states[state.disk_id].scheduler
    window = state.blocks[window_start:window_end]
    sizes = tuple(block["gb"] for block in window)
    if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR:
        if context.dynamic_cir_config.routing_mode == DYNAMIC_ROUTE_FIXED:
            selected = state.allowed_path_ids[
                state.start_offset % len(state.allowed_path_ids)
            ]
            path_ids = [selected] * len(sizes)
        else:
            path_ids = client_select_dynamic_owned_paths(
                block_sizes_gb=sizes,
                path_io_counts=scheduler.report_path_io_counts(current_time),
                allowed_path_ids=state.allowed_path_ids,
                start_offset=state.start_offset,
            )
    else:
        pressure = scheduler.report_path_pressure_analysis(current_time=current_time)
        path_ids = _select_qos_paths_from_analysis(
            sizes,
            pressure,
            state.allowed_path_ids,
            ClientRoutingConfig(
                qos_config=context.client_qos_config,
                disk_bw=scheduler.disk_bw,
                start_offset=state.start_offset,
            ),
        )
    state.planned_path_ids.extend(path_ids)


def _submit_client_batch(context, state, current_time):
    """按配置提交一个状态接下来的一个 I/O batch。"""
    npu = context.npus[state.npu_id]
    scheduler = context.disk_states[state.disk_id].scheduler
    submit_end = min(
        len(state.blocks), state.cursor + context.client_io_config.submit_batch_size
    )
    while state.cursor < submit_end:
        if context.policy in PATH_QOS_POLICIES:
            if state.cursor == len(state.planned_path_ids):
                _plan_qos_pressure_window(context, state, current_time)
            chunk_end = min(submit_end, len(state.planned_path_ids))
            path_ids = state.planned_path_ids[state.cursor : chunk_end]
        else:
            chunk_end = submit_end
            path_ids = [-1] * (chunk_end - state.cursor)

        flows = []
        for block, path_id in zip(state.blocks[state.cursor : chunk_end], path_ids):
            flows.append(
                _new_flow(
                    npu=npu,
                    layer=state.layer,
                    block=block,
                    disk_id=state.disk_id,
                    queue_id=path_id,
                    block_count=1,
                    current_time=current_time,
                    demand_gbps=state.demand_gbps,
                    deadline_time=state.deadline_time,
                    layer_work_gb=state.layer_work_gb,
                )
            )
        for flow in flows:
            _register_submitted_flow(context, flow)
        scheduler.enqueue_many(flows=flows, current_time=current_time)
        state.cursor = chunk_end

    scheduler.request_dispatch(current_time, context.event_heap)
    return state.cursor == len(state.blocks)


def _handle_client_submission(context, generation, current_time):
    """执行一轮同 ready_time、NPU 无放回随机排列的 batch 提交。"""
    if generation != context.client_submission_generation:
        context.stale_events += 1
        return
    context.pending_client_submission_time = None

    ready_npus = []
    for npu_id, state_ids in context.client_submission_queues.items():
        if not state_ids:
            continue
        state = context.client_submission_states[state_ids[0]]
        if (
            state.ready_time <= current_time + _EPS
            and context.client_next_issue_time[npu_id] <= current_time + _EPS
        ):
            ready_npus.append(npu_id)
    if not ready_npus:
        _request_client_submission(context)
        return

    ready_npus.sort()
    context.submit_rng.shuffle(ready_npus)
    context.client_submission_rounds += 1
    round_id = context.client_submission_rounds
    context.client_submission_batches += len(ready_npus)
    context.client_max_npus_per_round = max(
        context.client_max_npus_per_round, len(ready_npus)
    )
    if len(ready_npus) > 1:
        context.client_multi_npu_rounds += 1

    for npu_id in ready_npus:
        state_ids = context.client_submission_queues[npu_id]
        state_id = state_ids.popleft()
        state = context.client_submission_states[state_id]
        batch_size = min(
            context.client_io_config.submit_batch_size,
            len(state.blocks) - state.cursor,
        )
        finished = _submit_client_batch(context, state, current_time)
        context.client_next_issue_time[npu_id] = current_time + (
            batch_size * context.client_io_config.issue_interval_us / 1000.0
        )
        disk_id = state.disk_id
        if len(context.client_submission_order_sample) < 256:
            context.client_submission_order_sample.append(
                {
                    "time_ms": current_time,
                    "round": round_id,
                    "npu_id": npu_id,
                    "request_id": state.request_id,
                    "layer": state.layer,
                    "disk_id": disk_id,
                    "io_count": batch_size,
                }
            )
        if finished:
            del context.client_submission_states[state.state_id]
        else:
            state_ids.append(state_id)
        if not state_ids:
            del context.client_submission_queues[npu_id]

    _request_client_submission(context)


def _start_layer_io(  # 准备一层的数据块，并把它们交给客户端选路入口。
    context,  # 仿真上下文，里面保存各 SSD、Path 池和事件队列。
    npu,  # 当前执行请求的 NPU 状态，其中包含数据块放置结果。
    layer,  # 本次准备读取的层号。
    current_time,  # 本次读取开始的仿真时间，单位为毫秒。
):
    """启动一层 KV 读取。

    第 0 层在请求开始时读取；从第 1 层开始，计算第 k 层时预取第 k+1 层。
    """
    if layer < 0 or layer >= context.n_layers:  # 层号不在有效范围时不能创建 I/O。
        return  # 直接结束，避免访问不存在的层。
    if layer in npu.per_layer_kv_load_start:  # 已启动过的层不能重复提交和重复选路。
        return  # 保持每层只提交一次。

    placement = npu.block_placement.get(  # 读取上游已经算好的“数据块 -> SSD”结果。
        layer,  # 只读取当前层的数据放置。
        (),  # 当前层没有放置记录时使用空元组。
    )  # 这里已经确定目标 SSD，但尚未选择 SSD 内部的 Path。
    blocks = []  # 创建待选路的数据块列表。
    for block_index, placed_block in enumerate(placement):  # 按原顺序遍历放置结果。
        client_block = {  # 为客户端提交入口构造一个简单的数据块描述。
            "disk": placed_block[0],  # 上游选出的目标 SSD 编号，选路函数不修改它。
            "gb": placed_block[1],  # 当前数据块的传输大小，单位为 GB。
            "block_idx": block_index,  # 保存原始顺序，用于完成通知和结果统计。
        }  # 至此只有 SSD 编号，还没有 Path ID。
        blocks.append(client_block)  # 保持顺序加入列表，后面按相同顺序接收 Path ID。
    npu.per_layer_kv_load_start[layer] = current_time  # 记录这一层开始读取的时刻。
    npu.pending_blocks[layer] = len(blocks)  # 记录这一层还有多少数据块尚未完成。

    if not blocks:  # 没有需要从 SSD 读取的数据块时，无须进入客户端选路流程。
        _mark_layer_io_ready(npu, layer, current_time)  # 直接把这一层标记为读取完成。
        return  # 结束当前函数，避免提交空请求。
    _client_submit_layer_blocks(
        context=context,
        npu=npu,
        layer=layer,
        blocks=blocks,
        current_time=current_time,
    )


def _mark_layer_io_ready(npu, layer, current_time):
    npu.kv_ready_layers.add(layer)


def _try_start_compute(context, npu, current_time):
    if npu.done or npu.compute_active:
        return False
    layer = npu.compute_done_up_to + 1
    if layer >= context.n_layers or layer not in npu.kv_ready_layers:
        return False

    wait_from = npu.request_start_time if layer == 0 else npu.last_compute_end_time
    io_wait_ms = max(0.0, current_time - wait_from)
    if layer == 0:
        npu.io_wait_L0_ms += io_wait_ms
    elif layer == 1:
        npu.io_wait_L1_ms += io_wait_ms
    else:
        npu.io_wait_L2plus_ms += io_wait_ms

    compute_ms = npu.per_layer_us / 1000.0
    npu.current_compute_expected_end_ms = current_time + compute_ms
    npu.compute_generation += 1
    npu.compute_active = True
    heapq.heappush(
        context.event_heap,
        (
            npu.current_compute_expected_end_ms,
            COMPUTE_DONE,
            npu.npu_id,
            layer,
            npu.compute_generation,
        ),
    )

    # 逐层预取：计算第 k 层的同时，读取第 k+1 层。
    _start_layer_io(
        context,
        npu,
        layer + 1,
        current_time,
    )
    return True


def _handle_completed_flow(context, flow, current_time):
    _register_completed_flow(context, flow)
    npu = context.npus[flow.npu_id]
    layer = flow.layer
    npu.pending_blocks[layer] -= flow.block_count
    if npu.pending_blocks[layer] > 0:
        return
    _mark_layer_io_ready(npu, layer, current_time)
    _try_start_compute(context, npu, current_time)


def _archive_request(npu, n_layers):
    compute_ms = n_layers * npu.per_layer_us / 1000.0
    io_wait_ms = npu.io_wait_L0_ms + npu.io_wait_L1_ms + npu.io_wait_L2plus_ms
    npu.per_request_io_detail.append(
        {
            "request_id": npu.current_load["request_id"],
            "category": npu.category,
            "npu_id": npu.npu_id,
            "seq_len_k": npu.current_load["seq_len_k"],
            "nql": npu.current_load["nql"],
            "per_layer_us": npu.current_load["per_layer_us"],
            "per_layer_kv_gb": npu.current_load["per_layer_kv_gb"],
            "required_bw_input_gbps": npu.current_load[
                "required_bw_input_gbps"
            ],
            "arrival_delay_ms": npu.arrival_time,
            "queueing_delay_ms": npu.queueing_delay_ms,
            "processing_ttft_ms": npu.processing_ttft_ms,
            "ttft_ms": npu.ttft_ms,
            "io_wait_L0_ms": npu.io_wait_L0_ms,
            "io_wait_L1_ms": npu.io_wait_L1_ms,
            "io_wait_L2plus_ms": npu.io_wait_L2plus_ms,
            "io_wait_total_ms": io_wait_ms,
            "avg_ssd_queue_wait_ms": npu.request_ssd_queue_wait_ms
            / npu.request_io_count,
            "max_ssd_queue_wait_ms": npu.request_max_ssd_queue_wait_ms,
            "avg_npu_link_queue_wait_ms": npu.request_link_queue_wait_ms
            / npu.request_io_count,
            "max_npu_link_queue_wait_ms": npu.request_max_link_queue_wait_ms,
            "avg_end_to_end_io_latency_ms": npu.request_e2e_io_latency_ms
            / npu.request_io_count,
            "io_count": npu.request_io_count,
            "request_compute_ms": compute_ms,
            "request_npu_utilization": (
                compute_ms / (compute_ms + io_wait_ms)
                if compute_ms + io_wait_ms > 0
                else 0.0
            ),
        }
    )


def _handle_compute_done(context, npu_id, layer, generation, current_time):
    npu = context.npus[npu_id]
    if generation != npu.compute_generation:
        context.stale_events += 1
        return

    npu.compute_active = False
    npu.compute_done_up_to = layer
    npu.last_compute_end_time = current_time
    npu.total_compute_ms += npu.per_layer_us / 1000.0
    if layer + 1 < context.n_layers:
        _try_start_compute(context, npu, current_time)
        return

    npu.done = True
    npu.processing_ttft_ms = current_time - npu.request_start_time
    npu.ttft_ms = current_time - npu.arrival_time
    _archive_request(npu, context.n_layers)
    context.completed_requests += 1


def _account_npu_link_until(npu, current_time):
    """按上一次有效速率积分一个 NPU 的链路占用。

    单服务器有数据时始终以 50 GB/s 工作，因此历史字段
    ``link_cap_hit_time_ms`` 在新模型中等于 link busy time。
    """
    if current_time <= npu.link_last_update_ms + _EPS:
        return
    duration_ms = current_time - npu.link_last_update_ms
    npu.link_bw_integral_gb += npu.link_current_bw_gbps * duration_ms / 1000.0
    npu.link_capacity_integral_gb += NPU_BW_LIMIT * duration_ms / 1000.0
    if npu.link_current_bw_gbps >= NPU_BW_LIMIT - 1e-9:
        npu.link_cap_hit_time_ms += duration_ms
    npu.link_last_update_ms = current_time


def _start_next_npu_link_io(context, npu, current_time):
    """用单服务器 FCFS 模型启动一条 NPU 接收传输。"""
    if npu.link_active_flow is not None or not npu.link_pending:
        return None
    _account_npu_link_until(npu, current_time)
    flow = npu.link_pending.popleft()
    npu.link_pending_gb -= flow.total_gb
    npu.link_active_flow = flow
    npu.link_max_active_io = max(npu.link_max_active_io, 1)
    npu.link_current_bw_gbps = NPU_BW_LIMIT
    npu.link_peak_raw_bw_gbps = max(
        npu.link_peak_raw_bw_gbps, NPU_BW_LIMIT
    )
    npu.link_peak_effective_bw_gbps = max(
        npu.link_peak_effective_bw_gbps, NPU_BW_LIMIT
    )
    npu.link_dispatches += 1
    flow.link_start_time = current_time
    link_queue_wait_ms = max(0.0, current_time - flow.link_enqueue_time)
    npu.link_total_queue_wait_ms += link_queue_wait_ms
    npu.link_max_queue_wait_ms = max(
        npu.link_max_queue_wait_ms, link_queue_wait_ms
    )
    flow.link_end_time = current_time + npu_link_service_time_ms(flow.total_gb)
    npu.completion_generation += 1
    heapq.heappush(
        context.event_heap,
        (
            max(current_time + _EPS, flow.link_end_time),
            NPU_LINK_COMPLETION,
            npu.npu_id,
            0,
            npu.completion_generation,
        ),
    )
    if context.global_link_coordinator is not None:
        context.global_link_coordinator.mark_npu_dirty(npu.npu_id)
    return flow


def _enqueue_npu_link_io(context, flow, current_time):
    """SSD 命令完成后把数据放入对应 NPU 的接收队列。"""
    npu = context.npus[flow.npu_id]
    flow.link_enqueue_time = current_time
    npu.link_pending.append(flow)
    npu.link_pending_gb += flow.total_gb
    npu.link_max_outstanding_io = max(
        npu.link_max_outstanding_io,
        len(npu.link_pending) + (1 if npu.link_active_flow is not None else 0),
    )
    context.pending_npu_link_start_ids.add(npu.npu_id)
    if context.global_link_coordinator is not None:
        context.global_link_coordinator.mark_npu_dirty(npu.npu_id)


def _flush_pending_npu_link_starts(context, current_time):
    """同时刻的所有 SSD 完成到齐后，用策略无关键确定链路 FCFS。"""
    pending_ids = sorted(context.pending_npu_link_start_ids)
    context.pending_npu_link_start_ids.clear()
    for npu_id in pending_ids:
        npu = context.npus[npu_id]
        new_arrivals = []
        while (
            npu.link_pending
            and abs(npu.link_pending[-1].link_enqueue_time - current_time) <= _EPS
        ):
            new_arrivals.append(npu.link_pending.pop())
        new_arrivals.reverse()
        new_arrivals.sort(
            key=lambda flow: (
                flow.link_enqueue_time,
                flow.request_id,
                flow.layer,
                flow.block_idx,
                flow.disk_id,
            )
        )
        npu.link_pending.extend(new_arrivals)
        _start_next_npu_link_io(context, npu, current_time)


def _handle_disk_service_completion(context, disk_id, generation, current_time):
    """结束 SSD 命令服务，立即释放盘槽并转入 NPU 数据面。"""
    disk_state = context.disk_states[disk_id]
    if generation != disk_state.generation:
        context.stale_events += 1
        return
    scheduler = disk_state.scheduler
    completed = scheduler.complete_ready_flows(current_time)
    if not completed:
        raise AssertionError("预期完成的 SSD active I/O 未从后端移除")
    for flow in completed:
        if context.dynamic_cir_controller is not None:
            context.dynamic_cir_controller.complete_ssd(flow)
        _enqueue_npu_link_io(context, flow, current_time)
    scheduler.request_dispatch(current_time, context.event_heap)


def _handle_npu_link_completion(context, npu_id, generation, current_time):
    """数据经 50 GB/s NPU 接收服务后，才对计算流水线可见。"""
    npu = context.npus[npu_id]
    if generation != npu.completion_generation or npu.link_active_flow is None:
        context.stale_events += 1
        return
    flow = npu.link_active_flow
    if current_time + _EPS < flow.link_end_time:
        context.stale_events += 1
        return
    _account_npu_link_until(npu, current_time)
    npu.link_completed_gb += flow.total_gb
    io_latency_ms = max(0.0, current_time - flow.enqueue_time)
    npu.link_total_io_latency_ms += io_latency_ms
    npu.link_max_io_latency_ms = max(npu.link_max_io_latency_ms, io_latency_ms)
    link_queue_wait_ms = flow.link_start_time - flow.link_enqueue_time
    npu.request_io_count += 1
    npu.request_ssd_queue_wait_ms += flow.ssd_queue_wait_ms
    npu.request_link_queue_wait_ms += link_queue_wait_ms
    npu.request_e2e_io_latency_ms += io_latency_ms
    npu.request_max_ssd_queue_wait_ms = max(
        npu.request_max_ssd_queue_wait_ms, flow.ssd_queue_wait_ms
    )
    npu.request_max_link_queue_wait_ms = max(
        npu.request_max_link_queue_wait_ms, link_queue_wait_ms
    )
    npu.link_active_flow = None
    npu.link_current_bw_gbps = 0.0
    if context.global_link_coordinator is not None:
        context.global_link_coordinator.mark_npu_dirty(npu.npu_id)
    _handle_completed_flow(context, flow, current_time)
    _start_next_npu_link_io(context, npu, current_time)


# ---------------------------------------------------------------------------
# 结果汇总与公共仿真入口
# ---------------------------------------------------------------------------


def _build_summary(context, current_time, total_requests, events_processed):
    for npu in context.npus:
        _account_npu_link_until(npu, current_time)

    if context.submitted_block_count != context.expected_block_count:
        raise AssertionError("仿真结束时 submit block 集合与 placement 不一致")
    if context.completed_block_count != context.expected_block_count:
        raise AssertionError("仿真结束时 complete block 集合与 placement 不一致")
    if any(state != 2 for state in context.block_states):
        raise AssertionError("仿真结束时存在未提交或未完成 block")
    if context.client_submission_states:
        raise AssertionError("仿真结束时仍有未清理的客户端提交状态")
    if context.pending_npu_link_start_ids:
        raise AssertionError("仿真结束时仍有未启动的 NPU 链路队列")
    for disk_state in context.disk_states:
        scheduler = disk_state.scheduler
        if scheduler.outstanding_blocks != 0 or disk_state.active_flows:
            raise AssertionError("仿真结束时 SSU 仍有 outstanding/active I/O")
        if scheduler.max_backend_active_io > 1:
            raise AssertionError("SSU 后端 active I/O 超过 1")
    for npu in context.npus:
        if npu.link_active_flow is not None or npu.link_pending:
            raise AssertionError("仿真结束时 NPU 接收队列未排空")
        if npu.link_max_active_io > 1:
            raise AssertionError("NPU 接收数据面 active I/O 超过 1")
    if any(
        npu.link_peak_effective_bw_gbps > NPU_BW_LIMIT + 1e-9 for npu in context.npus
    ):
        raise AssertionError("历史 NPU 聚合峰值超过物理上限")

    request_metrics = [
        detail for npu in context.npus for detail in npu.per_request_io_detail
    ]
    ttfts = [item["ttft_ms"] for item in request_metrics]
    utilizations = [item["request_npu_utilization"] for item in request_metrics]

    category_metrics = {}
    for category in QOS_ROUTING_CATEGORIES:
        rows = [item for item in request_metrics if item["category"] == category]
        category_ttfts = [item["ttft_ms"] for item in rows]
        category_utils = [item["request_npu_utilization"] for item in rows]
        category_metrics[category] = {
            "count": len(rows),
            "p50_ttft_ms": (float(np.percentile(category_ttfts, 50)) if rows else 0.0),
            "p95_ttft_ms": (float(np.percentile(category_ttfts, 95)) if rows else 0.0),
            "p99_ttft_ms": (float(np.percentile(category_ttfts, 99)) if rows else 0.0),
            "max_ttft_ms": max(category_ttfts) if rows else 0.0,
            "avg_request_compute_fraction": (
                float(np.mean(category_utils)) if rows else 0.0
            ),
            "p10_request_compute_fraction": (
                float(np.percentile(category_utils, 10)) if rows else 0.0
            ),
            "p50_request_compute_fraction": (
                float(np.percentile(category_utils, 50)) if rows else 0.0
            ),
            "avg_io_wait_total_ms": (
                float(np.mean([row["io_wait_total_ms"] for row in rows]))
                if rows
                else 0.0
            ),
            "avg_ssd_queue_wait_ms": (
                float(np.mean([row["avg_ssd_queue_wait_ms"] for row in rows]))
                if rows
                else 0.0
            ),
            "avg_npu_link_queue_wait_ms": (
                float(
                    np.mean([row["avg_npu_link_queue_wait_ms"] for row in rows])
                )
                if rows
                else 0.0
            ),
        }

    disk_stats = []
    for disk_state in context.disk_states:
        scheduler = disk_state.scheduler
        scheduler.settle(current_time)
        elapsed = disk_state.busy_time + disk_state.idle_time
        effective_utilization = (
            disk_state.completed_bytes_gb / (scheduler.disk_bw * current_time / 1000.0)
            if current_time > 0.0
            else 0.0
        )
        disk_stats.append(
            {
                "disk_id": disk_state.disk_id,
                "busy_time_ms": disk_state.busy_time,
                "idle_time_ms": disk_state.idle_time,
                "utilization": disk_state.busy_time / elapsed if elapsed > 0 else 0.0,
                "active_time_utilization": (
                    disk_state.busy_time / current_time if current_time > 0.0 else 0.0
                ),
                "effective_bandwidth_utilization": effective_utilization,
                "completed_bytes_gb": disk_state.completed_bytes_gb,
                "unused_bw_fraction": (
                    disk_state.surplus_bw_integral / disk_state.total_bw_integral
                    if disk_state.total_bw_integral > 0
                    else 0.0
                ),
                "flows_enqueued": scheduler.flows_enqueued,
                "blocks_enqueued": scheduler.blocks_enqueued,
                "outstanding_blocks": scheduler.outstanding_blocks,
                "max_outstanding_blocks": scheduler.max_outstanding_blocks,
                "dispatch_cycles": scheduler.dispatch_cycles,
                "dispatch_events_scheduled": scheduler.dispatch_events,
                "pressure_reports": scheduler.pressure_reports,
                "backend_dispatches": scheduler.backend_dispatches,
                "max_backend_active_io": scheduler.max_backend_active_io,
                "dynamic_cir_epochs": scheduler.dynamic_cir_epochs,
                "dynamic_cir_total_gbps": scheduler.dynamic_cir_total_gbps,
                "dynamic_cir_max_total_gbps": scheduler.dynamic_cir_max_total_gbps,
                "total_queue_wait_ms": scheduler.total_queue_wait_ms,
                "max_queue_wait_ms": scheduler.max_queue_wait_ms,
                "queue_count": len(scheduler.paths),
                "enqueued_path_ids": sorted(scheduler.enqueued_path_ids),
                "max_path_outstanding_io": max(
                    (path.max_outstanding_io for path in scheduler.paths.values()),
                    default=0,
                ),
            }
        )

    avg_request_compute_fraction = float(np.mean(utilizations)) if utilizations else 0.0
    total_compute_ms = sum(npu.total_compute_ms for npu in context.npus)
    fleet_compute_utilization = (
        total_compute_ms / (len(context.npus) * current_time)
        if context.npus and current_time > 0.0
        else 0.0
    )
    total_read_gb = sum(npu.link_completed_gb for npu in context.npus)
    link_capacity_gb = sum(npu.link_capacity_integral_gb for npu in context.npus)
    link_integral_gb = sum(npu.link_bw_integral_gb for npu in context.npus)
    npu_link_utilization = (
        total_read_gb / link_capacity_gb if link_capacity_gb > 0.0 else 0.0
    )
    if not math.isclose(
        link_integral_gb,
        total_read_gb,
        rel_tol=1e-8,
        abs_tol=1e-9,
    ):
        raise AssertionError("NPU link 带宽积分与完成字节不守恒")
    expected_read_gb = context.expected_read_gb
    disk_completed_gb = sum(
        disk_state.completed_bytes_gb for disk_state in context.disk_states
    )
    if not math.isclose(
        disk_completed_gb,
        expected_read_gb,
        rel_tol=1e-10,
        abs_tol=1e-9,
    ):
        raise AssertionError("SSD 阶段完成字节与 placement 字节不守恒")
    if not math.isclose(
        total_read_gb,
        expected_read_gb,
        rel_tol=1e-10,
        abs_tol=1e-9,
    ):
        raise AssertionError("完成读取字节与 placement 字节不守恒")
    utility_sum = sum(utilizations)
    utility_square_sum = sum(value * value for value in utilizations)
    jain_fairness = (
        utility_sum * utility_sum / (len(utilizations) * utility_square_sum)
        if utilizations and utility_square_sum > 0.0
        else 0.0
    )
    npu_link_stats = [
        {
            "npu_id": npu.npu_id,
            "completed_gb": npu.link_completed_gb,
            "bandwidth_integral_gb": npu.link_bw_integral_gb,
            "capacity_integral_gb": npu.link_capacity_integral_gb,
            "peak_raw_bw_gbps": npu.link_peak_raw_bw_gbps,
            "peak_effective_bw_gbps": npu.link_peak_effective_bw_gbps,
            "busy_time_ms": npu.link_cap_hit_time_ms,
            "cap_hit_time_ms": npu.link_cap_hit_time_ms,
            "cap_hit_fraction": (
                npu.link_cap_hit_time_ms / current_time if current_time > 0.0 else 0.0
            ),
            "dispatches": npu.link_dispatches,
            "max_active_io": npu.link_max_active_io,
            "outstanding_io": len(npu.link_pending)
            + (1 if npu.link_active_flow is not None else 0),
            "max_outstanding_io": npu.link_max_outstanding_io,
            "avg_queue_wait_ms": (
                npu.link_total_queue_wait_ms / npu.link_dispatches
                if npu.link_dispatches
                else 0.0
            ),
            "max_queue_wait_ms": npu.link_max_queue_wait_ms,
            "avg_end_to_end_io_latency_ms": (
                npu.link_total_io_latency_ms / npu.link_dispatches
                if npu.link_dispatches
                else 0.0
            ),
            "max_end_to_end_io_latency_ms": npu.link_max_io_latency_ms,
        }
        for npu in context.npus
    ]

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "policy": context.policy,
        "supported_policies": list(SUPPORTED_POLICIES),
        "backend_model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
        "data_plane_stages": {
            "ssd": {
                "discipline": "policy_select_then_one_nonpreemptive_command",
                "service_time": "io_size_gb / disk_bw_gbps",
                "max_active_io": 1,
            },
            "npu_link": {
                "discipline": "fcfs_store_and_forward",
                "service_time": "io_size_gb / npu_bw_limit_gbps",
                "max_active_io": 1,
            },
            "block_visible_after": "npu_link_completion",
            "path_pressure_released_after": "ssd_completion",
            "intermediate_buffer": "unbounded_store_and_forward",
        },
        "pressure_read_interval": (
            context.client_io_config.pressure_window_io
            if context.policy in PATH_QOS_POLICIES
            else None
        ),
        "pressure_read_mode": (
            context.client_io_config.name
            if context.policy in PATH_QOS_POLICIES
            else "none"
        ),
        "path_selection": (
            (
                "per_io"
                if context.client_io_config.path_selection_mode
                == PATH_SELECTION_PRESSURE_AWARE
                else context.client_io_config.path_selection_mode
            )
            if context.policy in PATH_QOS_POLICIES
            else "none"
        ),
        "path_pool_mode": (
            "exclusive_two_per_npu"
            if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
            else (
                context.client_io_config.path_pool_mode
                if context.policy in QOS_POLICIES
                else "none"
            )
        ),
        "client_submit_batch_size": context.client_io_config.submit_batch_size,
        "client_submit_interval_us": context.client_io_config.issue_interval_us,
        "client_submission_order": "ready_time_then_seeded_shuffle_round",
        "client_ssu_state_order": "placement_first_occurrence_round_robin",
        "submit_order_seed": context.submit_order_seed,
        "workload_seed": context.workload_seed,
        "placement_seed": context.placement_seed,
        "arrival_delay_seed": context.arrival_delay_seed,
        "arrival_delay_max_ms": context.arrival_delay_max_ms,
        "workload_fingerprint": context.workload_hash,
        "placement_hash": context.placement_hash,
        "placement_algorithm_version": "immutable_block_placement_v2_compact",
        "same_timestamp_visibility": "seeded_shuffle_plan_then_immediate_enqueue",
        "npu_bw_limit_gbps": NPU_BW_LIMIT,
        "npu_cap_allocation": "single_server_fcfs_store_and_forward_v1",
        "client_submission": {
            "rounds": context.client_submission_rounds,
            "batches": context.client_submission_batches,
            "multi_npu_rounds": context.client_multi_npu_rounds,
            "max_npus_per_round": context.client_max_npus_per_round,
            "order_sample": context.client_submission_order_sample,
        },
        "backend_capacity_gbps": (
            context.disk_states[0].scheduler.disk_bw if context.disk_states else 0.0
        ),
        "qos_static_configuration": context.policy in QOS_POLICIES,
        "qos_dynamic_configuration": (
            context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
        ),
        "qos_client_routing": (
            {
                PATH_SELECTION_PRESSURE_AWARE: (
                    "strict_category_group_aware_sed_nonexclusive"
                ),
                PATH_SELECTION_STATELESS_RR: (
                    "strict_category_stateless_round_robin_nonexclusive"
                ),
                PATH_SELECTION_FIXED_PATH_ZERO: (
                    "all_npus_all_io_fixed_to_path_zero"
                ),
            }[context.client_io_config.path_selection_mode]
            if context.policy in QOS_POLICIES
            else (
                "per_npu_exclusive_two_path_least_projected_work"
                if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
                else "none"
            )
        ),
        "dynamic_cir_control": (
            {
                "mode": context.dynamic_cir_config.mode,
                "paths_per_npu": context.dynamic_cir_config.paths_per_npu,
                "routing_mode": context.dynamic_cir_config.routing_mode,
                "epochs": sum(
                    disk_state.scheduler.dynamic_cir_epochs
                    for disk_state in context.disk_states
                ),
                "final_total_cir_gbps": [
                    disk_state.scheduler.dynamic_cir_total_gbps
                    for disk_state in context.disk_states
                ],
                "max_total_cir_gbps": max(
                    (
                        disk_state.scheduler.dynamic_cir_max_total_gbps
                        for disk_state in context.disk_states
                    ),
                    default=0.0,
                ),
            }
            if context.policy == POLICY_QOS_DYNAMIC_JOINT_CIR
            else None
        ),
        "global_link_coordination": (
            {
                "scope": "submitted_queue_heads_and_committed_data_plane_only",
                "disk_candidates": "per_npu_fcfs_head",
                "downstream_prediction": "ssd_reservation_then_npu_fcfs_completion",
                "predictions": context.global_link_coordinator.predictions,
                "selections": context.global_link_coordinator.selections,
                "timeline_rebuilds": context.global_link_coordinator.timeline_rebuilds,
            }
            if context.policy == POLICY_GLOBAL_LINK_AWARE
            else None
        ),
        "total_requests": total_requests,
        "completed_requests": context.completed_requests,
        "makespan_ms": current_time,
        "events_processed": events_processed,
        "stale_events": context.stale_events,
        "event_counts": dict(context.event_counts),
        "avg_ttft_ms": float(np.mean(ttfts)) if ttfts else 0.0,
        "avg_processing_ttft_ms": (
            float(np.mean([item["processing_ttft_ms"] for item in request_metrics]))
            if request_metrics
            else 0.0
        ),
        "avg_queueing_delay_ms": (
            float(np.mean([item["queueing_delay_ms"] for item in request_metrics]))
            if request_metrics
            else 0.0
        ),
        "avg_request_compute_fraction": avg_request_compute_fraction,
        "fleet_npu_compute_utilization": fleet_compute_utilization,
        "request_compute_fraction_jain": jain_fairness,
        "npu_link_utilization": npu_link_utilization,
        "npu_link_total_read_gb": total_read_gb,
        "npu_link_bandwidth_integral_gb": link_integral_gb,
        "npu_link_capacity_integral_gb": link_capacity_gb,
        "npu_link_peak_raw_bw_gbps": max(
            (row["peak_raw_bw_gbps"] for row in npu_link_stats),
            default=0.0,
        ),
        "npu_link_peak_effective_bw_gbps": max(
            (row["peak_effective_bw_gbps"] for row in npu_link_stats),
            default=0.0,
        ),
        "npu_link_mean_effective_bw_gbps_per_npu": (
            total_read_gb / (current_time / 1000.0) / len(context.npus)
            if current_time > 0.0 and context.npus
            else 0.0
        ),
        "npu_link_cap_hit_fraction": (
            sum(row["cap_hit_time_ms"] for row in npu_link_stats)
            / (len(context.npus) * current_time)
            if current_time > 0.0 and context.npus
            else 0.0
        ),
        "npu_link_busy_fraction": (
            sum(row["busy_time_ms"] for row in npu_link_stats)
            / (len(context.npus) * current_time)
            if current_time > 0.0 and context.npus
            else 0.0
        ),
        "npu_link_stats": npu_link_stats,
        "block_conservation": {
            "expected": context.expected_block_count,
            "submitted": context.submitted_block_count,
            "completed": context.completed_block_count,
            "placement_targets_preserved": True,
            "expected_read_gb": expected_read_gb,
            "ssd_completed_read_gb": disk_completed_gb,
            "completed_read_gb": total_read_gb,
        },
        "throughput_requests_per_s": (
            context.completed_requests / current_time * 1000.0
            if current_time > 0
            else 0.0
        ),
        "request_metrics": request_metrics,
        "category_metrics": category_metrics,
        "disk_stats": disk_stats,
    }


def simulate_continuous(
    bw_table,
    *,
    policy,
    num_npu=NUM_NPU,
    num_disk=NUM_DISK,
    n_layers=SIM_N_LAYERS,
    ls_ratio=None,
    qos_config=None,
    client_io_config=DEFAULT_CLIENT_IO_CONFIG,
    dynamic_cir_config=DEFAULT_DYNAMIC_CIR_POLICY_CONFIG,
    placement_mode="random",
    disk_bw=DISK_BW,
    submit_order_seed=42,
    workload_seed=42,
    placement_seed=43,
    arrival_delay_seed=44,
    arrival_delay_max_ms=ARRIVAL_DELAY_MAX_MS,
    prepared_inputs=None,
):
    """运行一个请求/NPU 的共享两阶段数据面仿真。"""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"policy 只能是 {SUPPORTED_POLICIES} 之一")
    if policy in QOS_POLICIES and qos_config is None:
        raise ValueError("静态 QoS 数据面策略必须提供 StaticQoSConfig")

    total_requests = num_npu
    if prepared_inputs is not None:
        if prepared_inputs.n_layers != n_layers:
            raise ValueError("prepared placement 的 layer 数与仿真不一致")
        if prepared_inputs.num_disk != num_disk:
            raise ValueError("prepared placement 的 SSU 数与仿真不一致")
        if len(prepared_inputs.request_loads) != total_requests:
            raise ValueError("prepared workload 的请求数与仿真不一致")
        # Dispatcher 会在分派前复制每个 request；placement 在仿真中只读，
        # 因而配对策略可安全共享同一份大对象，避免重复生成和深拷贝。
        request_loads = list(prepared_inputs.request_loads)
        placement_by_request = prepared_inputs.placement_by_request
        actual_workload_seed = prepared_inputs.workload_seed
        actual_placement_seed = prepared_inputs.placement_seed
        actual_arrival_delay_seed = prepared_inputs.arrival_delay_seed
        actual_arrival_delay_max_ms = prepared_inputs.arrival_delay_max_ms
        workload_hash = prepared_inputs.workload_hash
        placement_hash = prepared_inputs.placement_hash
        placement_mode = prepared_inputs.placement_mode
    else:
        workload_rng = np.random.RandomState(int(workload_seed))
        arrival_rng = np.random.RandomState(int(arrival_delay_seed))
        request_loads = generate_request_loads(
            bw_table, workload_rng, total_requests, ls_ratio=ls_ratio
        )
        for request_id, request in enumerate(request_loads):
            request["request_id"] = request_id
            request["npu_id"] = request_id
            request["arrival_time"] = float(
                arrival_rng.uniform(0.0, float(arrival_delay_max_ms))
            )
        placement_rng = np.random.RandomState(int(placement_seed))
        placement_by_request = build_block_placement(
            request_loads,
            placement_rng,
            placement_mode,
            n_layers,
            num_disk,
        )
        actual_workload_seed = int(workload_seed)
        actual_placement_seed = int(placement_seed)
        actual_arrival_delay_seed = int(arrival_delay_seed)
        actual_arrival_delay_max_ms = float(arrival_delay_max_ms)
        workload_hash = workload_fingerprint(request_loads)
        placement_hash = placement_fingerprint(placement_by_request)

    npus = [NPUState(npu_id) for npu_id in range(num_npu)]
    disk_states = [DiskState(disk_id) for disk_id in range(num_disk)]
    event_heap = []
    global_link_coordinator = (
        GlobalLinkCoordinator(npus, disk_states)
        if policy == POLICY_GLOBAL_LINK_AWARE
        else None
    )
    dynamic_path_pools = (
        build_dynamic_npu_path_pools(
            num_npu,
            num_disk,
            paths_per_npu=dynamic_cir_config.paths_per_npu,
        )
        if policy == POLICY_QOS_DYNAMIC_JOINT_CIR
        else {}
    )
    dynamic_cir_controller = (
        DynamicCIRController(npus, dynamic_cir_config)
        if policy == POLICY_QOS_DYNAMIC_JOINT_CIR
        else None
    )
    for disk_state in disk_states:
        dynamic_path_owners = {
            path_id: npu_id
            for (npu_id, disk_id), path_ids in dynamic_path_pools.items()
            if disk_id == disk_state.disk_id
            for path_id in path_ids
        }
        DiskIOScheduler(
            disk_state,
            policy,
            disk_bw,
            qos_config if policy in QOS_POLICIES else None,
            global_link_coordinator,
            dynamic_cir_controller,
            dynamic_path_owners,
            dynamic_cir_config,
        )

    request_path_pools = (
        dynamic_path_pools
        if policy == POLICY_QOS_DYNAMIC_JOINT_CIR
        else (
            build_demand_ticket_path_pools(request_loads, num_disk)
            if policy in QOS_POLICIES
            and client_io_config.path_pool_mode == "per_npu_demand_tickets"
            else {}
        )
    )

    context = _SimulationContext(
        event_heap=event_heap,
        disk_states=disk_states,
        npus=npus,
        n_layers=n_layers,
        policy=policy,
        qos_config=qos_config,
        client_io_config=client_io_config,
        request_path_pools=request_path_pools,
        submit_order_seed=int(submit_order_seed),
        placement_by_request=placement_by_request,
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        workload_seed=actual_workload_seed,
        placement_seed=actual_placement_seed,
        arrival_delay_seed=actual_arrival_delay_seed,
        arrival_delay_max_ms=actual_arrival_delay_max_ms,
        global_link_coordinator=global_link_coordinator,
        dynamic_cir_controller=dynamic_cir_controller,
        dynamic_cir_config=dynamic_cir_config,
    )

    for request in request_loads:
        npu = npus[request["npu_id"]]
        request_start_time = float(request["arrival_time"])
        npu.start_request(
            request,
            placement_by_request[request["request_id"]],
            request_start_time,
        )
        _start_layer_io(context, npu, 0, request_start_time)
        _try_start_compute(context, npu, request_start_time)

    current_time = 0.0
    events_processed = 0

    while context.completed_requests < total_requests:
        if not event_heap:
            raise RuntimeError("事件队列已为空，但仍有请求没有完成")

        event_time, event_type, resource_id, value, generation = heapq.heappop(
            event_heap
        )
        current_time = max(current_time, event_time)
        events_processed += 1
        context.event_counts[event_type] += 1

        if event_type == COMPUTE_DONE:
            _handle_compute_done(context, resource_id, value, generation, current_time)
            continue

        if event_type == CLIENT_SUBMISSION:
            _handle_client_submission(context, generation, current_time)
            continue

        if event_type == DISK_COMPLETION:
            _handle_disk_service_completion(
                context, resource_id, generation, current_time
            )
            next_is_same_time_disk_completion = (
                bool(event_heap)
                and abs(event_heap[0][0] - current_time) <= _EPS
                and event_heap[0][1] == DISK_COMPLETION
            )
            if not next_is_same_time_disk_completion:
                _flush_pending_npu_link_starts(context, current_time)
            continue

        if event_type == NPU_LINK_COMPLETION:
            _handle_npu_link_completion(
                context, resource_id, generation, current_time
            )
            continue
        disk_state = disk_states[resource_id]
        disk_scheduler = disk_state.scheduler
        if generation != disk_scheduler.dispatch_generation:
            context.stale_events += 1
            continue
        disk_scheduler.dispatch(current_time, event_heap, schedule_completion=True)

    return npus, _build_summary(context, current_time, total_requests, events_processed)
