"""Pressure-aware 路由策略使用的精简离散事件仿真。

所有方案共享相同的静态 QoS、单命令 SSD 40 GB/s 服务和每 NPU 独立的
50 GB/s FCFS 接收队列。唯一策略变量是客户端读取 Path pressure 的频率。
每个 token block 通过一致性哈希选择一个 SSU，并在所有层复用该放置。

Scheme B 也使用同一数据面，但允许每块 SSD 写入独立的静态 CIR 表，
并让每个 NPU 在所有 SSD 上使用自己的专属 Path。
"""

from __future__ import annotations

import ast
import bisect
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import heapq
import json
import math
import os
import struct
from typing import Optional

import numpy as np

from policy_logic import (
    PathPressureSnapshot,
    baseline_path_ids,
    category_path_ids,
    hardware_view,
    pressure_aware_path_ids,
)

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

POLICY_QOS_STATIC_CIR = "qos_static_cir"
POLICY_PER_SSD_FULL_VISIBLE_EDF = "per_ssd_full_visible_edf"
POLICY_OMNISCIENT_EDF = POLICY_PER_SSD_FULL_VISIBLE_EDF
SUPPORTED_POLICIES = (POLICY_QOS_STATIC_CIR, POLICY_PER_SSD_FULL_VISIBLE_EDF)
RESULT_SCHEMA_VERSION = 8
WORKLOAD_CATEGORIES = ("SS", "SL", "LS", "LL")
QOS_ROUTING_CATEGORIES = WORKLOAD_CATEGORIES

PLACEMENT_BLOCK_RING_HASH = "block_ring_hash"
BLOCK_RING_VIRTUAL_NODES = 256
PLACEMENT_MODES = (PLACEMENT_BLOCK_RING_HASH,)

PATH_SELECTION_PRESSURE_AWARE = "pressure_aware"
PATH_SELECTION_FIXED_PATH_ZERO = "fixed_path_zero"
PATH_SELECTION_MODES = (
    PATH_SELECTION_PRESSURE_AWARE,
    PATH_SELECTION_FIXED_PATH_ZERO,
)

COMPUTE_DONE = 0
DISK_COMPLETION = 1
NPU_LINK_COMPLETION = 2
CLIENT_SUBMISSION = 3
DISK_SCHEDULE = 4
_EPS = 1e-12
_MAX_CIR_WRITE_THRESHOLD_GBPS = 0.05


@dataclass(frozen=True)
class ClientIOConfig:
    """客户端的提交粒度、发行间隔和 Path pressure 刷新频率。"""

    name: str
    pressure_window_io: Optional[int]
    submit_batch_size: int = CLIENT_SUBMIT_BATCH_SIZE
    issue_interval_us: float = 0.0
    path_selection_mode: str = PATH_SELECTION_PRESSURE_AWARE

    def __post_init__(self):
        if self.path_selection_mode not in PATH_SELECTION_MODES:
            raise ValueError(f"path_selection_mode 只能是 {PATH_SELECTION_MODES}")
        if self.submit_batch_size <= 0:
            raise ValueError("submit_batch_size 必须为正数")
        if self.issue_interval_us < 0.0:
            raise ValueError("issue_interval_us 不能为负数")
        if self.pressure_window_io is not None and self.pressure_window_io <= 0:
            raise ValueError("pressure_window_io 必须为正数或 None")
        if (
            self.path_selection_mode == PATH_SELECTION_FIXED_PATH_ZERO
            and self.pressure_window_io is not None
        ):
            raise ValueError("baseline 不读取 Path pressure")


QOS_REFRESH_EVERY_8_IO = ClientIOConfig("refresh8", PRESSURE_READ_INTERVAL)
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


def capped_proportional_demands(capacity_gbps, compute_time_s, work_by_resource):
    total_work_gb = sum(work_by_resource.values())
    capped_total_gbps = min(total_work_gb / compute_time_s, capacity_gbps)
    return {
        resource: capped_total_gbps * work_gb / total_work_gb
        for resource, work_gb in work_by_resource.items()
    }


def omniscient_edf_key(flow):
    return (
        flow.deadline_time,
        flow.layer_work_gb,
        flow.enqueue_time,
        flow.request_id,
        flow.layer,
        flow.block_idx,
        flow.disk_id,
    )

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
        for category in WORKLOAD_CATEGORIES
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

def _sha256_ring_position(namespace, first, second):
    payload = namespace + struct.pack("!QQ", int(first), int(second))
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")

@lru_cache(maxsize=None)
def _block_hash_ring(num_disk):
    entries = sorted(
        (
            _sha256_ring_position(
                b"qos_storage_sim:block_ring_hash:vnode:v1\0",
                disk_id,
                virtual_node,
            ),
            disk_id,
        )
        for disk_id in range(num_disk)
        for virtual_node in range(BLOCK_RING_VIRTUAL_NODES)
    )
    return tuple(position for position, _ in entries), tuple(
        disk_id for _, disk_id in entries
    )

def block_ring_hash_disk_id(request_id, block_index, num_disk):
    """稳定 consistent-hash：block key 不含 layer，ring 节点不含 SSU 总数。"""
    positions, disk_ids = _block_hash_ring(num_disk)
    block_position = _sha256_ring_position(
        b"qos_storage_sim:block_ring_hash:block:v1\0",
        request_id,
        block_index,
    )
    ring_index = bisect.bisect_left(positions, block_position) % len(positions)
    return disk_ids[ring_index]

def build_block_placement(loads, mode, n_layers, num_disk):
    """将每个 token block 做 ring hash，并在所有层复用同一 SSU。"""
    if mode != PLACEMENT_BLOCK_RING_HASH:
        raise ValueError(f"placement_mode 只能是 {PLACEMENT_BLOCK_RING_HASH}")
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
        disk_ids = tuple(
            block_ring_hash_disk_id(load["request_id"], block_index, num_disk)
            for block_index in range(len(block_sizes))
        )
        layer_blocks = tuple(zip(disk_ids, map(float, block_sizes)))
        placement_by_request[load["request_id"]] = {
            layer: layer_blocks for layer in range(n_layers)
        }
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
    placement_mode=PLACEMENT_BLOCK_RING_HASH,
):
    """生成可被所有 refresh 方案严格复用的 workload 与 ring placement。"""
    workload_rng = np.random.RandomState(int(workload_seed))
    arrival_rng = np.random.RandomState(int(arrival_delay_seed))
    loads = generate_request_loads(
        bw_table, workload_rng, int(total_requests), ls_ratio=ls_ratio
    )
    for request_id, request in enumerate(loads):
        request["request_id"] = request_id
        request["npu_id"] = request_id
        request["arrival_time"] = float(
            arrival_rng.uniform(0.0, float(arrival_delay_max_ms))
        )
    placement = build_block_placement(
        loads, placement_mode, int(n_layers), int(num_disk)
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

@dataclass(frozen=True)
class StaticQoSConfig:
    """一块 SSD 的不可变初始 QoS 寄存器镜像。

    每个元组的下标就是 QoS Path ID。例如 ``path_cirs[17]`` 表示
    Path 17 的 CIR。``category_labels`` 和 ``category_paths_per_group``
    描述每个组内连续排列的硬件路由类别及其 Path 数量。

    静态策略始终复用这份初值。动态实验只修改调度器复制出的运行时 CIR，
    不会回写本对象；PIR、Path 权重和 Group 权重仍保持不变。

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
        每个组内各硬件类别占多少个 Path。legacy 四类配置使用
        ``(12, 4, 12, 4)``：组内偏移 0~11 属于 SS，12~15 属于 SL，
        16~27 属于 LS，28~31 属于 LL。
    ``category_labels``（硬件类别名称）：
        与布局和 CIR 预算等长；默认值保留 legacy SS/SL/LS/LL ABI。

    ``frozen=True`` 表示对象创建后不能重新给这些字段赋值。结合下面的元组
    转换，可以保证初始配置不会被运行时 CIR 提交意外改写。
    """

    path_cirs: tuple[float, ...]
    path_pirs: tuple[float, ...]
    path_weights: tuple[float, ...]
    group_weights: tuple[float, ...]
    category_paths_per_group: tuple[int, ...]
    category_labels: tuple[str, ...] = QOS_ROUTING_CATEGORIES

    def __post_init__(self):
        # 初始化时统一转成元组。这样即使调用方传入列表，创建完成后也不能
        # 再从外部修改这些硬件配置表。
        for field_name in (
            "path_cirs",
            "path_pirs",
            "path_weights",
            "group_weights",
            "category_paths_per_group",
            "category_labels",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

        path_count = len(self.path_cirs)
        if path_count != 256 or len(self.group_weights) != 8:
            raise ValueError("静态 QoS 硬件必须包含 256 个 Path 和 8 个 Group")
        if not (path_count == len(self.path_pirs) == len(self.path_weights)):
            raise ValueError("CIR、PIR 和 Path 权重数组的长度必须相同")
        if len(set(self.category_labels)) != len(self.category_labels):
            raise ValueError("hardware routing class labels must be unique")
        if len(self.category_paths_per_group) != len(self.category_labels) or any(
            count <= 0 for count in self.category_paths_per_group
        ):
            raise ValueError("每个 Group 都必须为每个硬件路由类别分配 Path")
        if sum(self.category_paths_per_group) != self.paths_per_group:
            raise ValueError("所有类别的 Path 数量之和必须正好填满一个 Group")
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


def client_category_paths(category, qos_config):
    """Return category-legal Path IDs through the portable policy module."""
    return category_path_ids(category, hardware_view(qos_config))


def _select_qos_paths_from_analysis(
    sizes, analysis, allowed_path_ids, routing_config
):
    """Adapt one pressure window to the DES-independent routing policy."""
    return list(
        pressure_aware_path_ids(
            sizes,
            analysis,
            allowed_path_ids,
            hardware_view(routing_config.qos_config),
            disk_bw_gbps=routing_config.disk_bw,
            start_offset=routing_config.start_offset,
        )
    )


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
        # Exact service accounting, including commands that cross an external
        # measurement boundary.  Completion counters alone cannot attribute a
        # partial command to the interval in which its bytes were transferred.
        self.served_gb_by_npu = defaultdict(float)
        self.scheduler = None

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
    """一块 SSD 的静态 CIR/WRR 命令仲裁器。"""

    def __init__(
        self,
        disk_state,
        policy,
        disk_bw,
        qos_config=None,
        oracle_priority=None,
        *,
        pressure_ttl_ms=0.0,
        cir_write_threshold_gbps=0.0,
    ):
        pressure_ttl_ms = float(pressure_ttl_ms)
        cir_write_threshold_gbps = float(cir_write_threshold_gbps)
        if not math.isfinite(pressure_ttl_ms) or pressure_ttl_ms < 0.0:
            raise ValueError("pressure_ttl_ms 必须是非负有限值")
        if (
            not math.isfinite(cir_write_threshold_gbps)
            or cir_write_threshold_gbps < 0.0
            or cir_write_threshold_gbps > _MAX_CIR_WRITE_THRESHOLD_GBPS
        ):
            raise ValueError(
                "cir_write_threshold_gbps 必须是 [0, 0.05] 内的有限值"
            )
        self.state = disk_state
        self.policy = policy
        self.disk_bw = float(disk_bw)
        self.pressure_ttl_ms = pressure_ttl_ms
        self.cir_write_threshold_gbps = cir_write_threshold_gbps
        self._pressure_cache = None
        self._pressure_cache_time = None
        self.paths = {}
        self.group_weights = ()
        self.oracle_heap = []
        self.oracle_priority = oracle_priority or omniscient_edf_key
        if policy == POLICY_QOS_STATIC_CIR:
            if sum(qos_config.path_cirs) > self.disk_bw + _EPS:
                raise ValueError("所有 Path 的 CIR 总和超过了 SSD 物理带宽")
            if any(np.isfinite(pir) for pir in qos_config.path_pirs):
                raise ValueError("当前命令级 QoS 模型只支持不限 PIR")
            self.group_weights = qos_config.group_weights
            self.paths = {
                path_id: PathQueue(
                    path_id,
                    qos_config.path_cirs[path_id],
                    qos_config.path_pirs[path_id],
                    qos_config.path_weights[path_id],
                    path_id // qos_config.paths_per_group,
                )
                for path_id in range(qos_config.path_count)
            }
        self.qos_rr_cursor = 0
        self._qos_floor_heap = []
        self._qos_floor_versions = [0] * len(self.paths)
        self._qos_backlogged_paths_cache = None
        self._qos_service_rates_cache = None
        self._qos_finish_heap = None
        self._qos_finish_buckets = None
        self._path_io_counts = [0] * len(self.paths)
        group_count = len(self.group_weights)
        self._group_io_counts = [0] * group_count
        self._active_paths_per_group = [0] * group_count
        self._active_path_weights = [0.0] * group_count
        self._active_group_weight_sum = 0.0
        self._active_cir_sum = 0.0
        self.backend_dispatches = 0
        self.max_backend_active_io = 0
        self.flows_enqueued = 0
        self.blocks_enqueued = 0
        self.outstanding_blocks = 0
        self.max_outstanding_blocks = 0
        self.dispatch_cycles = 0
        self.dispatch_events = 0
        self.pressure_reports = 0
        self.pressure_cache_hits = 0
        self.pending_dispatch_time = None
        self.dispatch_generation = 0
        self.total_queue_wait_ms = 0.0
        self.max_queue_wait_ms = 0.0
        self.enqueued_path_ids = set()
        disk_state.scheduler = self

    def report_path_pressure_analysis(self, current_time):
        self.settle(current_time)
        if (
            self.pressure_ttl_ms > 0.0
            and self._pressure_cache is not None
            and current_time
            < self._pressure_cache_time + self.pressure_ttl_ms
        ):
            self.pressure_cache_hits += 1
            return self._pressure_cache
        self.pressure_reports += 1
        snapshot = PathPressureSnapshot(
            counts=tuple(self._path_io_counts),
            group_io_counts=tuple(self._group_io_counts),
            active_paths_per_group=tuple(self._active_paths_per_group),
            active_path_weights=tuple(self._active_path_weights),
            active_group_weight_sum=self._active_group_weight_sum,
            active_cir_sum=self._active_cir_sum,
        )
        if self.pressure_ttl_ms > 0.0:
            self._pressure_cache = snapshot
            self._pressure_cache_time = float(current_time)
        return snapshot

    def update_path_cirs(self, path_cirs, current_time, *, forced_path_ids=()):
        """Atomically replace the runtime CIR table for the next arbitration.

        An in-flight SSD command remains non-preemptive at ``disk_bw`` and keeps
        its already scheduled completion time.  Only a later command selection
        observes the new CIRs.  Virtual-finish history is deliberately retained.
        ``forced_path_ids`` lets a fleet-wide planner commit exact compensating
        entries that would otherwise be suppressed by this disk's write threshold.
        """
        path_cirs = tuple(float(cir) for cir in path_cirs)
        if self.policy != POLICY_QOS_STATIC_CIR:
            raise RuntimeError("runtime CIR updates require the QoS policy")
        try:
            forced_path_ids = tuple(forced_path_ids)
        except TypeError as exc:
            raise ValueError("forced Path IDs must be an iterable of integers") from exc
        if any(
            isinstance(path_id, (bool, np.bool_))
            or not isinstance(path_id, (int, np.integer))
            for path_id in forced_path_ids
        ):
            raise ValueError("forced Path IDs must be integers")
        forced_path_ids = frozenset(int(path_id) for path_id in forced_path_ids)
        if any(path_id not in self.paths for path_id in forced_path_ids):
            raise ValueError("forced Path ID is outside the runtime CIR table")
        if len(path_cirs) != len(self.paths):
            raise ValueError("runtime CIR table has the wrong Path count")
        if any(cir < 0.0 for cir in path_cirs):
            raise ValueError("runtime Path CIR cannot be negative")
        if sum(path_cirs) > self.disk_bw + _EPS:
            raise ValueError("runtime Path CIR sum exceeds SSD bandwidth")
        if any(
            cir > self.paths[path_id].pir + _EPS
            for path_id, cir in enumerate(path_cirs)
        ):
            raise ValueError("runtime Path CIR exceeds its PIR")

        if self.cir_write_threshold_gbps <= 0.0:
            # Preserve the original zero-threshold path, including comparison
            # tolerance and Path-ID write order, for exact default regression.
            changes = [
                (path_id, cir)
                for path_id, cir in enumerate(path_cirs)
                if abs(self.paths[path_id].cir - cir) > _EPS
            ]
        else:
            threshold = max(_EPS, self.cir_write_threshold_gbps)
            selected = {}
            compensating_decreases = []
            for path_id, cir in enumerate(path_cirs):
                old_cir = self.paths[path_id].cir
                delta = cir - old_cir
                active_set_change = (old_cir > 0.0) != (cir > 0.0)
                forced_change = path_id in forced_path_ids and abs(delta) > _EPS
                if forced_change or active_set_change or abs(delta) > threshold:
                    selected[path_id] = (path_id, cir, delta)
                elif delta < 0.0:
                    compensating_decreases.append((path_id, cir, delta))

            if selected:
                def resulting_cir_sum():
                    return math.fsum(
                        selected[path_id][1]
                        if path_id in selected
                        else self.paths[path_id].cir
                        for path_id in range(len(self.paths))
                    )

                proposed_sum = resulting_cir_sum()
                if proposed_sum > self.disk_bw + _EPS:
                    # A sparse mix of old and new registers can exceed capacity
                    # even when both complete tables are valid.  Largest
                    # suppressed decreases first gives the minimum number of
                    # additional writes needed to restore the capacity bound.
                    compensating_decreases.sort(
                        key=lambda change: (change[2], change[0])
                    )
                    for change in compensating_decreases:
                        selected[change[0]] = change
                        proposed_sum = resulting_cir_sum()
                        if proposed_sum <= self.disk_bw + _EPS:
                            break
                if proposed_sum > self.disk_bw + _EPS:
                    raise AssertionError(
                        "sparse CIR update cannot preserve SSD capacity"
                    )

            # A real register sequence must release capacity before consuming
            # it.  Within each direction Path ID keeps the order deterministic.
            changes = [
                (path_id, cir)
                for path_id, cir, _ in sorted(
                    selected.values(),
                    key=lambda change: (
                        0 if change[2] < 0.0 else 1,
                        change[0],
                    ),
                )
            ]
        if not changes:
            return 0
        self.settle(current_time)
        for path_id, cir in changes:
            self.paths[path_id].cir = cir
        # A pressure snapshot also carries active_cir_sum.  Any real register
        # write therefore invalidates it even when its TTL has not expired.
        self._pressure_cache = None
        self._pressure_cache_time = None
        if (
            self.cir_write_threshold_gbps > 0.0
            and math.fsum(path.cir for path in self.paths.values())
            > self.disk_bw + _EPS
        ):
            raise AssertionError("sparse CIR update exceeded SSD capacity")
        self._active_cir_sum = sum(
            self.paths[path_id].cir
            for path_id, count in enumerate(self._path_io_counts)
            if count > 0
        )
        self._invalidate_qos_arbitration_cache()
        return len(changes)

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
    def _account_until(self, current_time):
        if current_time <= self.state.last_event_time:
            return
        duration = current_time - self.state.last_event_time
        if self.state.active_flows:
            self.state.busy_time += duration
            flow = self.state.active_flows[0]
            used_bw = flow.bw if flow.active else 0.0
            self.state.served_gb_by_npu[flow.npu_id] += (
                used_bw * duration / 1000.0
            )
            self.state.surplus_bw_integral += max(0.0, self.disk_bw - used_bw) * duration
            self.state.total_bw_integral += self.disk_bw * duration
        else:
            self.state.idle_time += duration
        self.state.last_event_time = current_time

    def settle(self, current_time):
        if current_time <= self.state.last_event_time + _EPS:
            return
        self._account_until(current_time)
        if self.state.active_flows:
            flow = self.state.active_flows[0]
            elapsed_ms = max(0.0, current_time - flow.start_time)
            if flow.active and flow.bw > 0.0 and elapsed_ms > 0.0:
                flow.remaining_gb -= min(
                    flow.remaining_gb, flow.bw * elapsed_ms / 1000.0
                )
            flow.start_time = current_time
            if flow.remaining_gb <= _EPS:
                flow.remaining_gb = 0.0
                flow.end_time = current_time

    def _activate_flow(self, flow, current_time):
        if self.state.active_flows:
            raise RuntimeError("同一块 SSD 后端不能同时执行多个 I/O")
        wait_ms = max(0.0, current_time - flow.enqueue_time)
        flow.ssd_queue_wait_ms = wait_ms
        self.total_queue_wait_ms += wait_ms * flow.block_count
        self.max_queue_wait_ms = max(self.max_queue_wait_ms, wait_ms)
        flow.active = True
        flow.start_time = current_time
        flow.bw = 0.0
        flow.end_time = float("inf")
        self.state.active_flows.append(flow)

    def _publish_qos_floor(self, path):
        path_id = path.path_id
        self._qos_floor_versions[path_id] += 1
        if path.has_work():
            heapq.heappush(
                self._qos_floor_heap,
                (path.virtual_finish, path_id, self._qos_floor_versions[path_id]),
            )

    def _qos_virtual_floor(self):
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
        self._qos_backlogged_paths_cache = None
        self._qos_service_rates_cache = None
        self._qos_finish_heap = None
        self._qos_finish_buckets = None

    def _push_qos_candidate(self, path):
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
        if self._qos_finish_heap is None:
            self._build_qos_arbitration_cache()
        if not self._qos_backlogged_paths_cache:
            return None
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
        if self.state.active_flows:
            return self.state.active_flows[0]
        if self.policy == POLICY_QOS_STATIC_CIR:
            selected_path = self._select_qos_path()
            flow = selected_path.activate_next() if selected_path is not None else None
            if selected_path is not None:
                if selected_path.pending:
                    self._push_qos_candidate(selected_path)
                else:
                    self._invalidate_qos_arbitration_cache()
        else:
            flow = heapq.heappop(self.oracle_heap)[1] if self.oracle_heap else None
        if flow is None:
            return None
        self._activate_flow(flow, current_time)
        flow.bw = self.disk_bw
        flow.end_time = current_time + ssd_command_service_time_ms(
            flow.remaining_gb, self.disk_bw
        )
        self.backend_dispatches += 1
        self.max_backend_active_io = max(
            self.max_backend_active_io, len(self.state.active_flows)
        )
        return flow

    def enqueue_many(self, flows, current_time):
        self.settle(current_time)
        for flow in flows:
            if self.policy == POLICY_QOS_STATIC_CIR:
                selected_path = self.paths[flow.queue_id]
                was_empty = not selected_path.has_work()
                pending_was_empty = not selected_path.pending
                selected_path.enqueue(flow)
                self.enqueued_path_ids.add(flow.queue_id)
                self._change_path_io_count(flow.queue_id, flow.block_count)
                if pending_was_empty:
                    self._invalidate_qos_arbitration_cache()
                if was_empty:
                    selected_path.virtual_finish = self._qos_virtual_floor()
                    self._publish_qos_floor(selected_path)
            else:
                flow.queue_id = -1
                heapq.heappush(
                    self.oracle_heap,
                    (self.oracle_priority(flow), flow),
                )
            self.flows_enqueued += 1
            self.blocks_enqueued += flow.block_count
            self.outstanding_blocks += flow.block_count
        self.max_outstanding_blocks = max(
            self.max_outstanding_blocks, self.outstanding_blocks
        )

    def complete_ready_flows(self, current_time):
        self.settle(current_time)
        if len(self.state.active_flows) > 1:
            raise RuntimeError("同一块 SSD 后端观察到多个 active I/O")
        if not self.state.active_flows:
            return []
        flow = self.state.active_flows[0]
        if flow.remaining_gb > _EPS:
            return []
        self.state.active_flows.clear()
        self.outstanding_blocks -= flow.block_count
        flow.active = False
        self.state.completed_bytes_gb += flow.total_gb
        path = flow.queue
        if path is not None:
            path.complete(flow)
            if not path.has_work():
                self._publish_qos_floor(path)
            self._change_path_io_count(flow.queue_id, -flow.block_count)
        return [flow]

    def request_dispatch(self, current_time, event_heap):
        # A non-preemptive command already owns the SSD.  Its completion
        # handler will request the next dispatch, so enqueue-time dispatch
        # events while busy are redundant and only create stale completions.
        if self.state.active_flows:
            return
        if (
            self.pending_dispatch_time is not None
            and abs(self.pending_dispatch_time - current_time) <= _EPS
        ):
            return
        self.dispatch_generation += 1
        self.pending_dispatch_time = current_time
        heapq.heappush(
            event_heap,
            (current_time, DISK_SCHEDULE, self.state.disk_id, 0, self.dispatch_generation),
        )
        self.dispatch_events += 1

    def dispatch(self, current_time, event_heap):
        self.pending_dispatch_time = None
        self.settle(current_time)
        flow = self._dispatch_one(current_time)
        self.dispatch_cycles += 1
        self._schedule_next_completion(current_time, event_heap)
        return flow

    def _schedule_next_completion(self, current_time, event_heap):
        self.state.generation += 1
        if self.state.active_flows:
            flow = self.state.active_flows[0]
        else:
            flow = None
        if flow is not None and flow.active and math.isfinite(flow.end_time):
            heapq.heappush(
                event_heap,
                (
                    max(current_time + _EPS, flow.end_time),
                    DISK_COMPLETION,
                    self.state.disk_id,
                    0,
                    self.state.generation,
                ),
            )

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
        qos_configs_by_disk,
        npu_dedicated_paths,
        client_io_config,
        submit_order_seed,
        placement_by_request,
        workload_hash,
        placement_hash,
        workload_seed,
        placement_seed,
        arrival_delay_seed,
        arrival_delay_max_ms,
    ):
        self.event_heap = event_heap
        self.disk_states = disk_states
        self.npus = npus
        self.n_layers = n_layers
        self.policy = policy
        self.client_io_config = client_io_config
        self.qos_configs_by_disk = qos_configs_by_disk
        self.npu_dedicated_paths = npu_dedicated_paths
        self.submit_order_seed = submit_order_seed
        self.submit_rng = np.random.RandomState(submit_order_seed)
        self.workload_hash = workload_hash
        self.placement_hash = placement_hash
        self.workload_seed = workload_seed
        self.placement_seed = placement_seed
        self.placement_mode = PLACEMENT_BLOCK_RING_HASH
        self.arrival_delay_seed = arrival_delay_seed
        self.arrival_delay_max_ms = arrival_delay_max_ms
        self.placement_by_request = placement_by_request
        self.routing_categories = QOS_ROUTING_CATEGORIES
        self.client_path_pools_by_disk = (
            tuple(
                {
                    category: client_category_paths(category, disk_qos_config)
                    for category in self.routing_categories
                }
                for disk_qos_config in qos_configs_by_disk
            )
            if policy == POLICY_QOS_STATIC_CIR
            and npu_dedicated_paths is None
            else ()
        )
        self.completed_requests = 0
        self.stale_events = 0
        self.event_counts = defaultdict(int)
        self.client_submission_states = {}
        self.client_submission_queues = defaultdict(deque)
        self.next_client_submission_id = 0
        self.client_submission_generation = 0
        self.pending_client_submission_time = None
        self.client_submission_rounds = 0
        self.client_submission_batches = 0
        self.client_multi_npu_rounds = 0
        self.client_max_npus_per_round = 0
        self.client_submission_order_sample = []
        self.routing_category_state_counts = defaultdict(int)
        self.routing_category_io_counts = defaultdict(int)
        self.routing_category_gb = defaultdict(float)
        self.client_next_issue_time = [0.0] * len(npus)
        self.block_offsets = {}
        self.expected_block_count = 0
        self.expected_read_gb = 0.0
        for request_id, layers in placement_by_request.items():
            for layer, blocks in layers.items():
                self.block_offsets[(request_id, layer)] = self.expected_block_count
                self.expected_read_gb += sum(block_gb for _, block_gb in blocks)
                self.expected_block_count += len(blocks)
        self.block_states = bytearray(self.expected_block_count)
        self.submitted_block_count = 0
        self.completed_block_count = 0
        self.pending_npu_link_start_ids = set()


@dataclass
class _ClientSubmissionState:
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
    remaining_submit_gb: float
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
    blocks_by_disk = defaultdict(list)
    for block in blocks:
        blocks_by_disk[block["disk"]].append(block)
    layer_work_by_disk = {
        disk_id: sum(block["gb"] for block in disk_blocks)
        for disk_id, disk_blocks in blocks_by_disk.items()
    }
    demand_by_disk = (
        capped_proportional_demands(
            NPU_BW_LIMIT, npu.per_layer_us / 1e6, layer_work_by_disk
        )
        if context.policy == POLICY_PER_SSD_FULL_VISIBLE_EDF
        else {}
    )
    for disk_id, disk_blocks_list in blocks_by_disk.items():
        disk_blocks = tuple(disk_blocks_list)
        disk_work_gb = layer_work_by_disk[disk_id]
        allowed_paths = (
            (context.npu_dedicated_paths[npu.npu_id],)
            if context.npu_dedicated_paths is not None
            else context.client_path_pools_by_disk[disk_id][npu.category]
            if context.policy == POLICY_QOS_STATIC_CIR
            else ()
        )
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
            start_offset=(
                npu.current_load["request_id"] + layer * 13 + disk_id * 29
            ) % len(allowed_paths) if allowed_paths else 0,
            allowed_path_ids=allowed_paths,
            demand_gbps=demand_by_disk.get(disk_id, 0.0),
            deadline_time=(
                current_time if layer == 0 else npu.current_compute_expected_end_ms
            ),
            layer_work_gb=disk_work_gb,
            remaining_submit_gb=disk_work_gb,
        )
        context.routing_category_state_counts[npu.category] += 1
        context.routing_category_io_counts[npu.category] += len(disk_blocks)
        context.routing_category_gb[npu.category] += disk_work_gb
        context.client_submission_queues[npu.npu_id].append(state_id)
    _request_client_submission(context)


def _plan_qos_pressure_window(context, state, current_time):
    window_start = len(state.planned_path_ids)
    if context.npu_dedicated_paths is not None:
        state.planned_path_ids.extend(
            state.allowed_path_ids[0]
            for _ in range(window_start, len(state.blocks))
        )
        return
    if context.client_io_config.path_selection_mode == PATH_SELECTION_FIXED_PATH_ZERO:
        state.planned_path_ids.extend(
            baseline_path_ids(len(state.blocks) - window_start)
        )
        return
    pressure_window_io = context.client_io_config.pressure_window_io
    window_end = (
        len(state.blocks)
        if pressure_window_io is None
        else min(len(state.blocks), window_start + pressure_window_io)
    )
    scheduler = context.disk_states[state.disk_id].scheduler
    sizes = tuple(block["gb"] for block in state.blocks[window_start:window_end])
    pressure = scheduler.report_path_pressure_analysis(current_time)
    state.planned_path_ids.extend(
        _select_qos_paths_from_analysis(
            sizes,
            pressure,
            state.allowed_path_ids,
            ClientRoutingConfig(
                qos_config=context.qos_configs_by_disk[state.disk_id],
                disk_bw=scheduler.disk_bw,
                start_offset=state.start_offset,
            ),
        )
    )


def _submit_client_batch(context, state, current_time):
    npu = context.npus[state.npu_id]
    scheduler = context.disk_states[state.disk_id].scheduler
    submit_end = min(
        len(state.blocks), state.cursor + context.client_io_config.submit_batch_size
    )
    while state.cursor < submit_end:
        if context.policy == POLICY_QOS_STATIC_CIR:
            if state.cursor == len(state.planned_path_ids):
                _plan_qos_pressure_window(context, state, current_time)
            chunk_end = min(submit_end, len(state.planned_path_ids))
            path_ids = state.planned_path_ids[state.cursor:chunk_end]
        else:
            chunk_end = submit_end
            path_ids = [-1] * (chunk_end - state.cursor)
        flows = [
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
            for block, path_id in zip(
                state.blocks[state.cursor:chunk_end], path_ids
            )
        ]
        for flow in flows:
            _register_submitted_flow(context, flow)
        scheduler.enqueue_many(flows, current_time)
        state.remaining_submit_gb -= sum(flow.total_gb for flow in flows)
        state.cursor = chunk_end
    scheduler.request_dispatch(current_time, context.event_heap)
    return state.cursor == len(state.blocks)


def _handle_client_submission(context, generation, current_time):
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
    context.client_multi_npu_rounds += int(len(ready_npus) > 1)
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
        if len(context.client_submission_order_sample) < 256:
            context.client_submission_order_sample.append(
                {
                    "time_ms": current_time,
                    "round": round_id,
                    "npu_id": npu_id,
                    "request_id": state.request_id,
                    "layer": state.layer,
                    "disk_id": state.disk_id,
                    "io_count": batch_size,
                }
            )
        if finished:
            del context.client_submission_states[state_id]
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
    disk_state = context.disk_states[disk_id]
    if generation != disk_state.generation:
        context.stale_events += 1
        return
    scheduler = disk_state.scheduler
    completed = scheduler.complete_ready_flows(current_time)
    if not completed:
        raise AssertionError("预期完成的 SSD active I/O 未从后端移除")
    for flow in completed:
        _enqueue_npu_link_io(context, flow, current_time)
    scheduler.request_dispatch(current_time, context.event_heap)


def _handle_npu_link_completion(context, npu_id, generation, current_time):
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
    _handle_completed_flow(context, flow, current_time)
    _start_next_npu_link_io(context, npu, current_time)

def _build_summary(context, current_time, total_requests, events_processed):
    for npu in context.npus:
        _account_npu_link_until(npu, current_time)
    if context.submitted_block_count != context.expected_block_count:
        raise AssertionError("仿真结束时 submit block 集合与 placement 不一致")
    if context.completed_block_count != context.expected_block_count:
        raise AssertionError("仿真结束时 complete block 集合与 placement 不一致")
    if any(state != 2 for state in context.block_states):
        raise AssertionError("仿真结束时存在未提交或未完成 block")
    if context.client_submission_states or context.pending_npu_link_start_ids:
        raise AssertionError("仿真结束时仍有未清理的提交或链路状态")
    for disk_state in context.disk_states:
        scheduler = disk_state.scheduler
        if scheduler.outstanding_blocks or disk_state.active_flows:
            raise AssertionError("仿真结束时 SSU 仍有 outstanding/active I/O")
        if scheduler.max_backend_active_io > 1:
            raise AssertionError("SSU 后端 active I/O 超过 1")
    for npu in context.npus:
        if npu.link_active_flow is not None or npu.link_pending:
            raise AssertionError("仿真结束时 NPU 接收队列未排空")
        if npu.link_max_active_io > 1:
            raise AssertionError("NPU 接收数据面 active I/O 超过 1")
        if npu.link_peak_effective_bw_gbps > NPU_BW_LIMIT + 1e-9:
            raise AssertionError("历史 NPU 聚合峰值超过物理上限")

    request_metrics = [
        detail for npu in context.npus for detail in npu.per_request_io_detail
    ]
    ttfts = [item["ttft_ms"] for item in request_metrics]
    utilizations = [item["request_npu_utilization"] for item in request_metrics]
    category_metrics = {}
    for category in WORKLOAD_CATEGORIES:
        rows = [row for row in request_metrics if row["category"] == category]
        row_ttfts = [row["ttft_ms"] for row in rows]
        row_utils = [row["request_npu_utilization"] for row in rows]
        category_metrics[category] = {
            "count": len(rows),
            "p50_ttft_ms": float(np.percentile(row_ttfts, 50)) if rows else 0.0,
            "p95_ttft_ms": float(np.percentile(row_ttfts, 95)) if rows else 0.0,
            "p99_ttft_ms": float(np.percentile(row_ttfts, 99)) if rows else 0.0,
            "max_ttft_ms": max(row_ttfts) if rows else 0.0,
            "avg_request_compute_fraction": float(np.mean(row_utils)) if rows else 0.0,
            "p10_request_compute_fraction": float(np.percentile(row_utils, 10)) if rows else 0.0,
            "p50_request_compute_fraction": float(np.percentile(row_utils, 50)) if rows else 0.0,
            "avg_io_wait_total_ms": float(np.mean([r["io_wait_total_ms"] for r in rows])) if rows else 0.0,
            "avg_ssd_queue_wait_ms": float(np.mean([r["avg_ssd_queue_wait_ms"] for r in rows])) if rows else 0.0,
            "avg_npu_link_queue_wait_ms": float(np.mean([r["avg_npu_link_queue_wait_ms"] for r in rows])) if rows else 0.0,
        }

    disk_stats = []
    for disk_state in context.disk_states:
        scheduler = disk_state.scheduler
        scheduler.settle(current_time)
        elapsed = disk_state.busy_time + disk_state.idle_time
        disk_stats.append(
            {
                "disk_id": disk_state.disk_id,
                "busy_time_ms": disk_state.busy_time,
                "idle_time_ms": disk_state.idle_time,
                "utilization": disk_state.busy_time / elapsed if elapsed else 0.0,
                "active_time_utilization": disk_state.busy_time / current_time if current_time else 0.0,
                "effective_bandwidth_utilization": (
                    disk_state.completed_bytes_gb
                    / (scheduler.disk_bw * current_time / 1000.0)
                    if current_time else 0.0
                ),
                "completed_bytes_gb": disk_state.completed_bytes_gb,
                "unused_bw_fraction": (
                    disk_state.surplus_bw_integral / disk_state.total_bw_integral
                    if disk_state.total_bw_integral else 0.0
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

    total_compute_ms = sum(npu.total_compute_ms for npu in context.npus)
    total_read_gb = sum(npu.link_completed_gb for npu in context.npus)
    link_capacity_gb = sum(npu.link_capacity_integral_gb for npu in context.npus)
    link_integral_gb = sum(npu.link_bw_integral_gb for npu in context.npus)
    if not math.isclose(link_integral_gb, total_read_gb, rel_tol=1e-8, abs_tol=1e-9):
        raise AssertionError("NPU link 带宽积分与完成字节不守恒")
    disk_completed_gb = sum(d.completed_bytes_gb for d in context.disk_states)
    if not math.isclose(disk_completed_gb, context.expected_read_gb, rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError("SSD 阶段完成字节与 placement 字节不守恒")
    if not math.isclose(total_read_gb, context.expected_read_gb, rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError("完成读取字节与 placement 字节不守恒")
    utility_sum = sum(utilizations)
    utility_square_sum = sum(value * value for value in utilizations)
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
            "cap_hit_fraction": npu.link_cap_hit_time_ms / current_time if current_time else 0.0,
            "dispatches": npu.link_dispatches,
            "max_active_io": npu.link_max_active_io,
            "outstanding_io": len(npu.link_pending) + int(npu.link_active_flow is not None),
            "max_outstanding_io": npu.link_max_outstanding_io,
            "avg_queue_wait_ms": npu.link_total_queue_wait_ms / npu.link_dispatches if npu.link_dispatches else 0.0,
            "max_queue_wait_ms": npu.link_max_queue_wait_ms,
            "avg_end_to_end_io_latency_ms": npu.link_total_io_latency_ms / npu.link_dispatches if npu.link_dispatches else 0.0,
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
            "ssd": {"discipline": "static_qos_then_one_nonpreemptive_command", "service_time": "io_size_gb / disk_bw_gbps", "max_active_io": 1},
            "npu_link": {"discipline": "fcfs_store_and_forward", "service_time": "io_size_gb / npu_bw_limit_gbps", "max_active_io": 1},
            "block_visible_after": "npu_link_completion",
            "path_pressure_released_after": "ssd_completion",
            "intermediate_buffer": "unbounded_store_and_forward",
        },
        "pressure_read_interval": (
            context.client_io_config.pressure_window_io
            if context.policy == POLICY_QOS_STATIC_CIR
            and context.npu_dedicated_paths is None
            else None
        ),
        "pressure_read_mode": (
            context.client_io_config.name
            if context.policy == POLICY_QOS_STATIC_CIR
            and context.npu_dedicated_paths is None
            else "none"
        ),
        "path_selection": (
            "npu_dedicated"
            if context.npu_dedicated_paths is not None
            else (
                "per_io"
                if context.client_io_config.path_selection_mode == PATH_SELECTION_PRESSURE_AWARE
                else context.client_io_config.path_selection_mode
            )
            if context.policy == POLICY_QOS_STATIC_CIR else "none"
        ),
        "path_pool_mode": (
            "npu_dedicated"
            if context.npu_dedicated_paths is not None
            else "category_shared"
        ),
        **(
            {
                "npu_dedicated_paths": list(context.npu_dedicated_paths),
                "npu_dedicated_path_group_counts": [
                    sum(
                        path_id // context.qos_configs_by_disk[0].paths_per_group
                        == group_id
                        for path_id in context.npu_dedicated_paths
                    )
                    for group_id in range(
                        context.qos_configs_by_disk[0].group_count
                    )
                ],
                "qos_configuration_scope": "per_ssd_static_cir_table",
                "qos_cir_sum_gbps_by_ssd": [
                    sum(config.path_cirs)
                    for config in context.qos_configs_by_disk
                ],
                "path_pressure_reads": sum(
                    row["pressure_reports"] for row in disk_stats
                ),
            }
            if context.npu_dedicated_paths is not None
            else {}
        ),
        "routing_category_policy": {"scope": "request"},
        "routing_category_distribution": {
            category: {
                "states": context.routing_category_state_counts[category],
                "io_count": context.routing_category_io_counts[category],
                "gb": context.routing_category_gb[category],
            }
            for category in context.routing_categories
        },
        "client_submit_batch_size": context.client_io_config.submit_batch_size,
        "client_submit_interval_us": context.client_io_config.issue_interval_us,
        "client_submission_order": "ready_time_then_seeded_shuffle_round",
        "client_ssu_state_order": "placement_first_occurrence_round_robin",
        "submit_order_seed": context.submit_order_seed,
        "workload_seed": context.workload_seed,
        "placement_seed": context.placement_seed,
        "placement_mode": PLACEMENT_BLOCK_RING_HASH,
        "placement_ring_virtual_nodes": BLOCK_RING_VIRTUAL_NODES,
        "placement_ring_hash_version": "sha256_u64_pair_v1",
        "arrival_delay_seed": context.arrival_delay_seed,
        "arrival_delay_max_ms": context.arrival_delay_max_ms,
        "workload_fingerprint": context.workload_hash,
        "placement_hash": context.placement_hash,
        "placement_algorithm_version": "immutable_block_ring_hash_v1",
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
        "backend_capacity_gbps": context.disk_states[0].scheduler.disk_bw if context.disk_states else 0.0,
        "qos_static_configuration": context.policy == POLICY_QOS_STATIC_CIR,
        "qos_client_routing": (
                "one_dedicated_path_per_npu"
            if context.npu_dedicated_paths is not None
            else {
                    PATH_SELECTION_FIXED_PATH_ZERO: "all_npus_all_io_fixed_to_path_zero",
                    PATH_SELECTION_PRESSURE_AWARE: "strict_category_group_aware_sed_nonexclusive",
                }[context.client_io_config.path_selection_mode]
            if context.policy == POLICY_QOS_STATIC_CIR else "none"
        ),
        "total_requests": total_requests,
        "completed_requests": context.completed_requests,
        "makespan_ms": current_time,
        "events_processed": events_processed,
        "stale_events": context.stale_events,
        "event_counts": dict(context.event_counts),
        "avg_ttft_ms": float(np.mean(ttfts)) if ttfts else 0.0,
        "avg_processing_ttft_ms": float(np.mean([r["processing_ttft_ms"] for r in request_metrics])) if request_metrics else 0.0,
        "avg_queueing_delay_ms": float(np.mean([r["queueing_delay_ms"] for r in request_metrics])) if request_metrics else 0.0,
        "avg_request_compute_fraction": float(np.mean(utilizations)) if utilizations else 0.0,
        "fleet_npu_compute_utilization": total_compute_ms / (len(context.npus) * current_time) if context.npus and current_time else 0.0,
        "request_compute_fraction_jain": utility_sum * utility_sum / (len(utilizations) * utility_square_sum) if utilizations and utility_square_sum else 0.0,
        "npu_link_utilization": total_read_gb / link_capacity_gb if link_capacity_gb else 0.0,
        "npu_link_total_read_gb": total_read_gb,
        "npu_link_bandwidth_integral_gb": link_integral_gb,
        "npu_link_capacity_integral_gb": link_capacity_gb,
        "npu_link_peak_raw_bw_gbps": max((r["peak_raw_bw_gbps"] for r in npu_link_stats), default=0.0),
        "npu_link_peak_effective_bw_gbps": max((r["peak_effective_bw_gbps"] for r in npu_link_stats), default=0.0),
        "npu_link_mean_effective_bw_gbps_per_npu": total_read_gb / (current_time / 1000.0) / len(context.npus) if current_time and context.npus else 0.0,
        "npu_link_cap_hit_fraction": sum(r["cap_hit_time_ms"] for r in npu_link_stats) / (len(context.npus) * current_time) if current_time and context.npus else 0.0,
        "npu_link_busy_fraction": sum(r["busy_time_ms"] for r in npu_link_stats) / (len(context.npus) * current_time) if current_time and context.npus else 0.0,
        "npu_link_stats": npu_link_stats,
        "block_conservation": {
            "expected": context.expected_block_count,
            "submitted": context.submitted_block_count,
            "completed": context.completed_block_count,
            "placement_targets_preserved": True,
            "expected_read_gb": context.expected_read_gb,
            "ssd_completed_read_gb": disk_completed_gb,
            "completed_read_gb": total_read_gb,
        },
        "throughput_requests_per_s": context.completed_requests / current_time * 1000.0 if current_time else 0.0,
        "request_metrics": request_metrics,
        "category_metrics": category_metrics,
        "disk_stats": disk_stats,
    }

def simulate_continuous(
    bw_table,
    *,
    policy=POLICY_QOS_STATIC_CIR,
    num_npu=NUM_NPU,
    num_disk=NUM_DISK,
    n_layers=SIM_N_LAYERS,
    ls_ratio=None,
    qos_config=None,
    qos_configs_by_disk=None,
    npu_dedicated_paths=None,
    client_io_config=DEFAULT_CLIENT_IO_CONFIG,
    placement_mode=PLACEMENT_BLOCK_RING_HASH,
    disk_bw=DISK_BW,
    submit_order_seed=42,
    workload_seed=42,
    placement_seed=43,
    arrival_delay_seed=44,
    arrival_delay_max_ms=ARRIVAL_DELAY_MAX_MS,
    prepared_inputs=None,
):
    """运行一个请求/NPU 的 ring-placement 静态 QoS 仿真。

    ``qos_config`` 保留原有语义：将一份配置广播到所有 SSD。
    ``qos_configs_by_disk`` 则按 disk_id 提供每块 SSD 的配置。
    ``npu_dedicated_paths[npu_id]`` 设置该 NPU 在所有 SSD 上使用的
    专属 Path；该模式不读取 Path pressure。
    """
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"policy 只能是 {SUPPORTED_POLICIES}")
    disk_qos_configs = ()
    if policy == POLICY_QOS_STATIC_CIR:
        if qos_configs_by_disk is None:
            if qos_config is None:
                raise ValueError("Static QoS 策略必须提供 qos_config")
            disk_qos_configs = (qos_config,) * num_disk
        else:
            disk_qos_configs = tuple(qos_configs_by_disk)
            if len(disk_qos_configs) != num_disk:
                raise ValueError("qos_configs_by_disk 必须每块 SSD 提供一份配置")
        if npu_dedicated_paths is None and any(
            config.category_labels != QOS_ROUTING_CATEGORIES
            for config in disk_qos_configs
        ):
            raise ValueError("精简仿真只保留固定 SS/SL/LS/LL 硬件类别")
        if npu_dedicated_paths is not None:
            npu_dedicated_paths = tuple(npu_dedicated_paths)
            if len(npu_dedicated_paths) != num_npu:
                raise ValueError("npu_dedicated_paths 必须每个 NPU 提供一个 Path")
            if len(set(npu_dedicated_paths)) != num_npu:
                raise ValueError("专属 Path 不能被多个 NPU 共享")
            if any(
                path_id < 0 or path_id >= config.path_count
                for path_id in npu_dedicated_paths
                for config in disk_qos_configs
            ):
                raise ValueError("专属 Path ID 超出 QoS Path 表")
    elif npu_dedicated_paths is not None:
        raise ValueError("专属 Path 只用于静态 QoS 策略")
    if prepared_inputs is not None:
        if prepared_inputs.n_layers != n_layers or prepared_inputs.num_disk != num_disk:
            raise ValueError("prepared placement 维度与仿真不一致")
        if len(prepared_inputs.request_loads) != num_npu:
            raise ValueError("prepared workload 请求数与仿真不一致")
        if prepared_inputs.placement_mode != PLACEMENT_BLOCK_RING_HASH:
            raise ValueError("prepared placement 必须使用 block_ring_hash")
        request_loads = list(prepared_inputs.request_loads)
        placement_by_request = prepared_inputs.placement_by_request
        workload_hash = prepared_inputs.workload_hash
        placement_hash = prepared_inputs.placement_hash
        actual_workload_seed = prepared_inputs.workload_seed
        actual_placement_seed = prepared_inputs.placement_seed
        actual_arrival_delay_seed = prepared_inputs.arrival_delay_seed
        actual_arrival_delay_max_ms = prepared_inputs.arrival_delay_max_ms
    else:
        prepared_inputs = prepare_simulation_inputs(
            bw_table,
            total_requests=num_npu,
            n_layers=n_layers,
            num_disk=num_disk,
            ls_ratio=ls_ratio,
            workload_seed=workload_seed,
            placement_seed=placement_seed,
            arrival_delay_seed=arrival_delay_seed,
            arrival_delay_max_ms=arrival_delay_max_ms,
            placement_mode=placement_mode,
        )
        request_loads = list(prepared_inputs.request_loads)
        placement_by_request = prepared_inputs.placement_by_request
        workload_hash = prepared_inputs.workload_hash
        placement_hash = prepared_inputs.placement_hash
        actual_workload_seed = prepared_inputs.workload_seed
        actual_placement_seed = prepared_inputs.placement_seed
        actual_arrival_delay_seed = prepared_inputs.arrival_delay_seed
        actual_arrival_delay_max_ms = prepared_inputs.arrival_delay_max_ms

    npus = [NPUState(npu_id) for npu_id in range(num_npu)]
    disk_states = [DiskState(disk_id) for disk_id in range(num_disk)]
    event_heap = []
    for disk_state in disk_states:
        disk_qos_config = (
            disk_qos_configs[disk_state.disk_id]
            if policy == POLICY_QOS_STATIC_CIR
            else None
        )
        DiskIOScheduler(disk_state, policy, disk_bw, disk_qos_config)
    context = _SimulationContext(
        event_heap=event_heap,
        disk_states=disk_states,
        npus=npus,
        n_layers=n_layers,
        policy=policy,
        qos_configs_by_disk=disk_qos_configs,
        npu_dedicated_paths=npu_dedicated_paths,
        client_io_config=client_io_config,
        submit_order_seed=int(submit_order_seed),
        placement_by_request=placement_by_request,
        workload_hash=workload_hash,
        placement_hash=placement_hash,
        workload_seed=actual_workload_seed,
        placement_seed=actual_placement_seed,
        arrival_delay_seed=actual_arrival_delay_seed,
        arrival_delay_max_ms=actual_arrival_delay_max_ms,
    )
    for request in request_loads:
        npu = npus[request["npu_id"]]
        start_time = float(request["arrival_time"])
        npu.start_request(
            request, placement_by_request[request["request_id"]], start_time
        )
        _start_layer_io(context, npu, 0, start_time)
        _try_start_compute(context, npu, start_time)

    current_time = 0.0
    events_processed = 0
    while context.completed_requests < num_npu:
        if not event_heap:
            raise RuntimeError("事件队列已为空，但仍有请求没有完成")
        event_time, event_type, resource_id, value, generation = heapq.heappop(
            event_heap
        )
        current_time = max(current_time, event_time)
        events_processed += 1
        context.event_counts[event_type] += 1
        if event_type == COMPUTE_DONE:
            _handle_compute_done(
                context, resource_id, value, generation, current_time
            )
        elif event_type == CLIENT_SUBMISSION:
            _handle_client_submission(context, generation, current_time)
        elif event_type == DISK_COMPLETION:
            _handle_disk_service_completion(
                context, resource_id, generation, current_time
            )
            next_same_time_disk_completion = (
                bool(event_heap)
                and abs(event_heap[0][0] - current_time) <= _EPS
                and event_heap[0][1] == DISK_COMPLETION
            )
            if not next_same_time_disk_completion:
                _flush_pending_npu_link_starts(context, current_time)
        elif event_type == NPU_LINK_COMPLETION:
            _handle_npu_link_completion(
                context, resource_id, generation, current_time
            )
        else:
            scheduler = disk_states[resource_id].scheduler
            if generation != scheduler.dispatch_generation:
                context.stale_events += 1
            else:
                scheduler.dispatch(current_time, event_heap)
    return npus, _build_summary(
        context, current_time, num_npu, events_processed
    )
