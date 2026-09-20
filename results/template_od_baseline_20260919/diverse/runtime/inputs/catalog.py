"""Catalog loading and one-request-per-NPU input generation.

This module owns file access and random input generation. The core receives
already prepared inputs and has no dependency on this package or project data.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import numpy as np
from simulator.core import sim
from simulator.core.sim import (
    NUM_NPU, WORKLOAD_CATEGORIES, ARRIVAL_DELAY_MAX_MS, PLACEMENT_BLOCK_RING_HASH,
    classify_request, build_block_placement, PreparedSimulationInputs,
    workload_fingerprint, placement_fingerprint,
)

def load_bw_table_cache(results_dir=None, num_npu=None):
    """读取示例程序使用的请求画像表（带宽、计算时间和每层 KV 大小）。"""
    project_dir = str(Path(__file__).resolve().parents[1])
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

def simulate_continuous(bw_table, **kwargs):
    """Convenience input runner for the legacy single-request core."""
    if kwargs.get("prepared_inputs") is None:
        kwargs["prepared_inputs"] = prepare_simulation_inputs(
            bw_table, total_requests=kwargs.get("num_npu", sim.NUM_NPU),
            n_layers=kwargs.get("n_layers", sim.SIM_N_LAYERS),
            num_disk=kwargs.get("num_disk", sim.NUM_DISK),
            **{name: kwargs[name] for name in (
                "ls_ratio", "workload_seed", "placement_seed", "arrival_delay_seed",
                "arrival_delay_max_ms", "placement_mode") if name in kwargs},
        )
    return sim.simulate_continuous(**kwargs)
