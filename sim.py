import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import itertools
import ast
import logging
import time as _time
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq

sim_logger = logging.getLogger("storage_sim")
sim_logger.setLevel(logging.WARNING)

SIM_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "sim_debug.log"
)

NUM_NPU = 32
NUM_DISK = 8
DISK_BW = 40.0
NPU_BW_LIMIT = 50.0
MODEL_N_LAYERS = 78
SIM_N_LAYERS = 8
BLOCK_SIZE = 128

SEQ_LENS = [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 200]
NQLS = [64, 128, 256, 512, 1024, 2048, 4096]

SEQ_SHORT_BOUNDARY = 80
NQL_LONG_BOUNDARY = 512


def set_classification_boundaries(seq_short_boundary=80, nql_long_boundary=512):
    global SEQ_SHORT_BOUNDARY, NQL_LONG_BOUNDARY
    SEQ_SHORT_BOUNDARY = seq_short_boundary
    NQL_LONG_BOUNDARY = nql_long_boundary


def load_bw_table_cache(results_dir=None, num_npu=None):
    import os as _os

    project_dir = _os.path.dirname(_os.path.abspath(__file__))
    if results_dir is None:
        results_dir = _os.path.join(project_dir, "results")
    if num_npu is None:
        num_npu = NUM_NPU
    cache_file = _os.path.join(results_dir, f"bw_table_cache_v2_{num_npu}npu.npz")
    if _os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=True) as cached:
            raw = cached["table"].item()
        source = cache_file
    else:
        data_file = _os.path.join(project_dir, "data")
        if not _os.path.exists(data_file):
            return None
        with open(data_file, encoding="utf-8") as file:
            raw = ast.literal_eval(file.read())
        if not isinstance(raw, dict):
            raise ValueError(f"bandwidth data must be a dict: {data_file}")
        source = data_file

    bw_table = {}
    for k, v in raw.items():
        key = ast.literal_eval(k) if isinstance(k, str) and k.startswith("(") else k
        vals = tuple(v)
        if len(vals) == 3:
            req_bw, per_layer_us, ttft_ms = vals
            per_layer_kv_gb = req_bw * per_layer_us * 1e-6
            bw_table[key] = (req_bw, per_layer_us, ttft_ms, per_layer_kv_gb)
        else:
            bw_table[key] = vals
    print(f"  {len(bw_table)} entries loaded from {source}")
    return bw_table


def classify_request(seq_len_k, nql):
    short_seq = seq_len_k <= SEQ_SHORT_BOUNDARY
    long_nql = nql >= NQL_LONG_BOUNDARY
    if short_seq and not long_nql:
        return "SS"
    elif short_seq and long_nql:
        return "SL"
    elif not short_seq and not long_nql:
        return "LS"
    else:
        return "LL"


def generate_npu_loads(
    bw_table, rng, load_profile="mixed", total_bw_cap=None, ls_ratio=None, num_npu=None
):
    loads = []
    all_keys = list(bw_table.keys())
    if num_npu is None:
        num_npu = NUM_NPU
    if load_profile == "heavy":
        seq_choices, nql_choices = [160, 176, 192, 200], [512, 1024, 2048, 4096]
    elif load_profile == "light":
        seq_choices, nql_choices = [32, 48, 64], [256, 512]
    else:
        seq_choices, nql_choices = SEQ_LENS, NQLS
    if ls_ratio is not None and load_profile == "mixed":
        short_keys = [k for k in all_keys if k[0] <= SEQ_SHORT_BOUNDARY]
        long_keys = [k for k in all_keys if k[0] > SEQ_SHORT_BOUNDARY]
        ss_sl_keys = [k for k in short_keys if k[1] < NQL_LONG_BOUNDARY]
        sl_sl_keys = [k for k in short_keys if k[1] >= NQL_LONG_BOUNDARY]
        ls_lg_keys = [k for k in long_keys if k[1] < NQL_LONG_BOUNDARY]
        ll_lg_keys = [k for k in long_keys if k[1] >= NQL_LONG_BOUNDARY]
        n_s = num_npu // 2
        n_l = num_npu - n_s
        n_s_sl = int(round(n_s * ls_ratio))
        n_s_ss = n_s - n_s_sl
        n_l_ls = int(round(n_l * ls_ratio))
        n_l_ll = n_l - n_l_ls
        npu_id = 0
        for _ in range(n_s_ss):
            key = (
                ss_sl_keys[rng.randint(len(ss_sl_keys))]
                if ss_sl_keys
                else short_keys[rng.randint(len(short_keys))]
            )
            req_bw, per_layer_us, ttft_ms, per_layer_kv_gb = bw_table[key]
            loads.append(
                {
                    "npu_id": npu_id,
                    "seq_len_k": key[0],
                    "nql": key[1],
                    "required_bw": req_bw,
                    "per_layer_us": per_layer_us,
                    "ttft_ideal_ms": ttft_ms,
                    "per_layer_kv_gb": per_layer_kv_gb,
                    "category": classify_request(key[0], key[1]),
                }
            )
            npu_id += 1
        for _ in range(n_s_sl):
            key = (
                sl_sl_keys[rng.randint(len(sl_sl_keys))]
                if sl_sl_keys
                else short_keys[rng.randint(len(short_keys))]
            )
            req_bw, per_layer_us, ttft_ms, per_layer_kv_gb = bw_table[key]
            loads.append(
                {
                    "npu_id": npu_id,
                    "seq_len_k": key[0],
                    "nql": key[1],
                    "required_bw": req_bw,
                    "per_layer_us": per_layer_us,
                    "ttft_ideal_ms": ttft_ms,
                    "per_layer_kv_gb": per_layer_kv_gb,
                    "category": classify_request(key[0], key[1]),
                }
            )
            npu_id += 1
        for _ in range(n_l_ll):
            key = (
                ll_lg_keys[rng.randint(len(ll_lg_keys))]
                if ll_lg_keys
                else long_keys[rng.randint(len(long_keys))]
            )
            req_bw, per_layer_us, ttft_ms, per_layer_kv_gb = bw_table[key]
            loads.append(
                {
                    "npu_id": npu_id,
                    "seq_len_k": key[0],
                    "nql": key[1],
                    "required_bw": req_bw,
                    "per_layer_us": per_layer_us,
                    "ttft_ideal_ms": ttft_ms,
                    "per_layer_kv_gb": per_layer_kv_gb,
                    "category": classify_request(key[0], key[1]),
                }
            )
            npu_id += 1
        for _ in range(n_l_ls):
            key = (
                ls_lg_keys[rng.randint(len(ls_lg_keys))]
                if ls_lg_keys
                else long_keys[rng.randint(len(long_keys))]
            )
            req_bw, per_layer_us, ttft_ms, per_layer_kv_gb = bw_table[key]
            loads.append(
                {
                    "npu_id": npu_id,
                    "seq_len_k": key[0],
                    "nql": key[1],
                    "required_bw": req_bw,
                    "per_layer_us": per_layer_us,
                    "ttft_ideal_ms": ttft_ms,
                    "per_layer_kv_gb": per_layer_kv_gb,
                    "category": classify_request(key[0], key[1]),
                }
            )
            npu_id += 1
        rng.shuffle(loads)
        for i, l in enumerate(loads):
            l["npu_id"] = i
    else:
        for i in range(num_npu):
            if load_profile == "mixed":
                key = all_keys[rng.randint(len(all_keys))]
            else:
                sl = seq_choices[rng.randint(len(seq_choices))]
                nql = nql_choices[rng.randint(len(nql_choices))]
                key = (sl, nql)
            req_bw, per_layer_us, ttft_ms, per_layer_kv_gb = bw_table[key]
            loads.append(
                {
                    "npu_id": i,
                    "seq_len_k": key[0],
                    "nql": key[1],
                    "required_bw": req_bw,
                    "per_layer_us": per_layer_us,
                    "ttft_ideal_ms": ttft_ms,
                    "per_layer_kv_gb": per_layer_kv_gb,
                    "category": classify_request(key[0], key[1]),
                }
            )
    if total_bw_cap is not None:
        total_bw = sum(l["required_bw"] for l in loads)
        while total_bw > total_bw_cap and len(loads) > 1:
            worst = max(range(len(loads)), key=lambda i: loads[i]["required_bw"])
            total_bw -= loads[worst]["required_bw"]
            loads.pop(worst)
        for i, l in enumerate(loads):
            l["npu_id"] = i
    return loads


# ── Block-level placement ──────────────────────────────────────────────────────


def calculate_token_partition(seq_len_k, nql):
    """Return total, NPU-computed, and SSU-read token counts for a request."""
    total_tokens_value = float(seq_len_k) * 1024.0
    npu_tokens_value = float(nql)
    if not np.isfinite(total_tokens_value) or total_tokens_value < 0:
        raise ValueError("seq_len_k must describe a non-negative token count")
    if not np.isfinite(npu_tokens_value) or npu_tokens_value < 0:
        raise ValueError("nql must be a non-negative token count")

    total_tokens = int(round(total_tokens_value))
    npu_tokens = int(round(npu_tokens_value))
    if not np.isclose(total_tokens_value, total_tokens):
        raise ValueError("seq_len_k * 1024 must be a whole number of tokens")
    if not np.isclose(npu_tokens_value, npu_tokens):
        raise ValueError("nql must be a whole number of tokens")
    if npu_tokens > total_tokens:
        raise ValueError(
            "nql cannot exceed the request token count: "
            f"{npu_tokens} > {total_tokens}"
        )
    return total_tokens, npu_tokens, total_tokens - npu_tokens


def build_block_placement(
    loads, rng, mode="random", n_layers=SIM_N_LAYERS, num_disk=NUM_DISK
):
    result = {}
    global_disk_loads = np.zeros(num_disk)
    for load in loads:
        npu_id = load["npu_id"]
        per_layer_kv_gb = load["per_layer_kv_gb"]
        _, _, ssu_tokens = calculate_token_partition(load["seq_len_k"], load["nql"])
        if per_layer_kv_gb < 0:
            raise ValueError("per_layer_kv_gb must be non-negative")
        if ssu_tokens == 0 and per_layer_kv_gb > 1e-12:
            raise ValueError(
                "per_layer_kv_gb must be zero when every input token is computed "
                "on the NPU"
            )

        block_sizes_gb = []
        if ssu_tokens > 0 and per_layer_kv_gb > 0:
            kv_gb_per_token = per_layer_kv_gb / ssu_tokens
            n_blocks = int(np.ceil(ssu_tokens / BLOCK_SIZE))
            block_sizes_gb = [
                min(BLOCK_SIZE, ssu_tokens - block_idx * BLOCK_SIZE) * kv_gb_per_token
                for block_idx in range(n_blocks)
            ]

        placement = {}
        local_disk_loads = np.zeros(num_disk)
        for layer in range(n_layers):
            layer_blocks = []
            for block_idx, block_gb in enumerate(block_sizes_gb):
                if mode == "random":
                    disk = int(rng.randint(num_disk))
                elif mode == "roundrobin":
                    disk = block_idx % num_disk
                elif mode == "local_balanced":
                    disk = int(np.argmin(local_disk_loads))
                    local_disk_loads[disk] += block_gb
                    global_disk_loads[disk] += block_gb
                elif mode == "load_aware":
                    disk = int(np.argmin(global_disk_loads))
                    global_disk_loads[disk] += block_gb
                else:
                    disk = int(rng.randint(num_disk))
                if mode not in ("local_balanced", "load_aware"):
                    global_disk_loads[disk] += block_gb
                layer_blocks.append({"disk": disk, "gb": block_gb})
            placement[layer] = layer_blocks
        result[npu_id] = placement
    return result


# ── Event-driven simulation core ──────────────────────────────────────────────


KV_BLOCK_DONE = 0

COMPUTE_DONE = 1

NPU_START = 2

NPU_RESTART = 3

DISK_COMPLETION = 4

REQUEST_ARRIVAL = 5

BATCH_DISPATCH = 6

TOKEN_REFILL = 7

DISK_REBALANCE = 8


_flow_id_counter = 0


class BlockIOFlow:
    def __init__(
        self,
        npu_id,
        layer,
        block_idx,
        disk_id,
        total_gb,
        bw,
        start_time,
        priority=0,
        demand_bw=0.0,
        queue_id=-1,
        category="SS",
        block_count=1,
    ):
        global _flow_id_counter
        self.npu_id = npu_id
        self.layer = layer
        self.block_idx = block_idx
        self.disk_id = disk_id
        self.total_gb = total_gb
        self.remaining_gb = total_gb
        self.bw = bw
        self.start_time = start_time
        self.end_time = start_time + total_gb / bw * 1000 if bw > 0 else float("inf")
        self.active = True
        self.priority = priority
        self.flow_id = _flow_id_counter
        _flow_id_counter += 1
        self.demand_bw = demand_bw
        self.queue_id = queue_id
        self._queue = None
        self.category = category
        self.block_count = block_count

    def update_bw(self, new_bw, current_time):
        if current_time >= self.end_time:
            self.remaining_gb = 0
            self.active = False
            self.bw = 0
            self.end_time = current_time
            return
        if self.bw > 0:
            done_gb = self.bw * (current_time - self.start_time) / 1000
            self.remaining_gb = max(self.remaining_gb - done_gb, 0)
        if self.remaining_gb < 1e-12:
            self.remaining_gb = 0
            self.active = False
            self.bw = 0
            self.end_time = current_time
            return
        if new_bw <= 0:
            self.bw = 0
            self.start_time = current_time
            self.end_time = float("inf")
            return
        self.bw = new_bw
        self.start_time = current_time
        self.end_time = current_time + self.remaining_gb / new_bw * 1000


class DiskState:
    def __init__(self, disk_id, disk_bw=DISK_BW, queue_scheduler=None):
        self.disk_id = disk_id
        self.disk_bw = disk_bw
        self.active_flows = []
        self.generation = 0
        self.busy_time = 0.0
        self.idle_time = 0.0
        self.last_event_time = 0.0
        self.surplus_bw_integral = 0.0
        self.total_bw_integral = 0.0
        self.n_idle_to_busy_events = 0
        self.n_idle_events = 0
        self.queue_scheduler = queue_scheduler

    def add_flow(self, flow, current_time=0.0, disk_bw=DISK_BW):
        was_idle = len(self.active_flows) == 0
        if self.last_event_time < current_time:
            dur = current_time - self.last_event_time
            if was_idle:
                self.idle_time += dur
            else:
                self.busy_time += dur
                actual_used = sum(f.bw for f in self.active_flows if f.active)
                self.surplus_bw_integral += max(0, disk_bw - actual_used) * dur
                self.total_bw_integral += disk_bw * dur
            self.last_event_time = current_time
        if was_idle:
            self.n_idle_to_busy_events += 1
        self.active_flows.append(flow)

    def remove_flow(self, flow, current_time=0.0, disk_bw=DISK_BW):
        was_busy = len(self.active_flows) > 0
        if was_busy and self.last_event_time < current_time:
            dur = current_time - self.last_event_time
            self.busy_time += dur
            n_flows = len(self.active_flows)
            used_bw = min(disk_bw, n_flows * (disk_bw / n_flows)) if n_flows > 0 else 0
            actual_used = sum(f.bw for f in self.active_flows if f.active)
            self.surplus_bw_integral += max(0, disk_bw - actual_used) * dur
            self.total_bw_integral += disk_bw * dur
            self.last_event_time = current_time
        self.active_flows.remove(flow)
        if len(self.active_flows) == 0:
            self.n_idle_events += 1

    def remove_flows(self, flows, current_time=0.0, disk_bw=DISK_BW):
        """Remove multiple completed flows with one linear container rebuild."""
        if not flows:
            return
        if self.active_flows and self.last_event_time < current_time:
            duration = current_time - self.last_event_time
            self.busy_time += duration
            actual_used = sum(flow.bw for flow in self.active_flows if flow.active)
            self.surplus_bw_integral += max(0, disk_bw - actual_used) * duration
            self.total_bw_integral += disk_bw * duration
            self.last_event_time = current_time
        removed = set(flows)
        self.active_flows = [flow for flow in self.active_flows if flow not in removed]
        if not self.active_flows:
            self.n_idle_events += 1

    def earliest_end_time(self):
        if not self.active_flows:
            return float("inf")
        return min(f.end_time for f in self.active_flows)

    def push_next_event(self, event_heap, current_time):
        if not self.active_flows:
            return
        t = self.earliest_end_time()
        if t <= current_time:
            t = current_time + 1e-12
        self.generation += 1
        heapq.heappush(
            event_heap, (t, DISK_COMPLETION, self.disk_id, 0, self.generation)
        )


class NPUState:
    def __init__(self, load, block_placement, n_layers=SIM_N_LAYERS):
        self.npu_id = load["npu_id"]
        self.per_layer_us = load["per_layer_us"]
        self.per_layer_kv_gb = load["per_layer_kv_gb"]
        self.block_placement = block_placement
        self.kv_loaded_up_to = -1
        self.compute_done_up_to = -1
        self.compute_end_time = 0.0
        self.ttft_ms = None
        self.done = False
        self.pending_blocks = {}
        self.active_block_flows = defaultdict(set)
        self.total_compute_ms = 0.0
        self.started = False
        self.bw_priority = load["required_bw"]
        self.required_bw = load["required_bw"]
        self.ttft_ideal_ms = load["ttft_ideal_ms"] * n_layers / MODEL_N_LAYERS
        self.n_blocks_per_layer = len(block_placement[0]) if block_placement else 1
        self.request_count = 0
        self.ttft_list = []
        self.request_queue = []
        self.current_load = load
        self.request_start_time = 0.0
        self.seq_len_k = load["seq_len_k"]
        self.nql = load["nql"]
        (
            self.total_input_tokens,
            self.npu_compute_tokens,
            self.ssu_read_tokens,
        ) = calculate_token_partition(self.seq_len_k, self.nql)
        self.category = load["category"]
        self.layer_trace = {}
        self.trace_enabled = False
        self.layer0_kv_end_time = None
        self.layer1_kv_end_time = None
        self.io_wait_layers_2plus_ms = 0.0
        self.io_wait_L0_ms = 0.0
        self.io_wait_L1_ms = 0.0
        self.io_wait_L2plus_ms = 0.0
        self.npu_idle_ms = 0.0
        self.last_compute_end_time = 0.0
        self.per_request_io_detail = []
        self.current_request_io_waits = {}
        self.current_request_kv_actual_dur = {}
        self.per_layer_kv_load_start = {}
        self.instance_id = -1
        self.arrival_time = 0.0
        self.queueing_delay_ms = 0.0
        self.processing_ttft_ms = 0.0
        self.batch_states = {}
        self._kv_ready_layers = set()
        self._compute_active = False
        self._request_archived = False
        self._batches_dispatched = 0

    def reset_for_next_request(
        self, load, block_placement, event_time, n_layers=SIM_N_LAYERS
    ):
        if (
            self.request_start_time is not None
            and event_time > self.request_start_time
            and self.ttft_ms is not None
            and not self._request_archived
        ):
            per_layer_compute_ms = self.per_layer_us / 1000.0
            ideal_L1_kv_dur = (
                self.per_layer_kv_gb / NPU_BW_LIMIT * 1000
                if self.per_layer_kv_gb and self.per_layer_kv_gb > 0
                else 0
            )
            actual_L0_kv_dur = (
                (self.layer0_kv_end_time - self.request_start_time)
                if self.layer0_kv_end_time is not None
                and self.request_start_time is not None
                else 0
            )
            ideal_excl012 = (
                actual_L0_kv_dur
                + max(ideal_L1_kv_dur, per_layer_compute_ms)
                + (n_layers - 1) * per_layer_compute_ms
            )
            infl_ex012 = (
                self.ttft_ms / ideal_excl012 if ideal_excl012 > 0 else float("inf")
            )
            self.per_request_io_detail.append(
                {
                    "request_id": self.current_load.get("request_id"),
                    "io_wait_L0_ms": self.io_wait_L0_ms,
                    "io_wait_L1_ms": self.io_wait_L1_ms,
                    "io_wait_L2plus_ms": self.io_wait_L2plus_ms,
                    "io_wait_total_ms": self.io_wait_L0_ms
                    + self.io_wait_L1_ms
                    + self.io_wait_L2plus_ms,
                    "category": self.category,
                    "seq_len_k": self.seq_len_k,
                    "nql": self.nql,
                    "total_input_tokens": self.total_input_tokens,
                    "npu_compute_tokens": self.npu_compute_tokens,
                    "ssu_read_tokens": self.ssu_read_tokens,
                    "ttft_ms": self.ttft_ms,
                    "processing_ttft_ms": self.processing_ttft_ms,
                    "queueing_delay_ms": self.queueing_delay_ms,
                    "arrival_time": self.arrival_time,
                    "request_start_time": self.request_start_time,
                    "instance_id": self.instance_id,
                    "npu_id": self.npu_id,
                    "ideal_excl012_ms": ideal_excl012,
                    "infl_ex012": infl_ex012,
                    "per_layer_io_waits": dict(self.current_request_io_waits),
                    "total_compute_ms": self.total_compute_ms,
                    "n_blocks_per_layer": self.n_blocks_per_layer,
                    "per_layer_kv_gb": self.per_layer_kv_gb,
                    "per_layer_kv_actual_dur_ms": dict(
                        self.current_request_kv_actual_dur
                    ),
                }
            )
            self._request_archived = True
        for layer_flows in self.active_block_flows.values():
            for f in layer_flows:
                f.active = False
        self.per_layer_us = load["per_layer_us"]
        self.per_layer_kv_gb = load["per_layer_kv_gb"]
        self.block_placement = block_placement
        self.kv_loaded_up_to = -1
        self.compute_done_up_to = -1
        self.compute_end_time = event_time
        self.ttft_ms = None
        self.processing_ttft_ms = 0.0
        self.queueing_delay_ms = 0.0
        self.done = False
        self.pending_blocks = {}
        self.active_block_flows = defaultdict(set)
        self.started = False
        self.bw_priority = load["required_bw"]
        self.required_bw = load["required_bw"]
        self.ttft_ideal_ms = load["ttft_ideal_ms"] * n_layers / MODEL_N_LAYERS
        self.n_blocks_per_layer = len(block_placement[0]) if block_placement else 1
        self.current_load = load
        self.request_start_time = event_time
        self.seq_len_k = load["seq_len_k"]
        self.nql = load["nql"]
        (
            self.total_input_tokens,
            self.npu_compute_tokens,
            self.ssu_read_tokens,
        ) = calculate_token_partition(self.seq_len_k, self.nql)
        self.category = load["category"]
        self.layer_trace = {}
        self.layer0_kv_end_time = None
        self.layer1_kv_end_time = None
        self.io_wait_layers_2plus_ms = 0.0
        self.io_wait_L0_ms = 0.0
        self.io_wait_L1_ms = 0.0
        self.io_wait_L2plus_ms = 0.0
        self.current_request_io_waits = {}
        self.current_request_kv_actual_dur = {}
        self.per_layer_kv_load_start = {}
        self.last_compute_end_time = event_time
        self.batch_states = {}
        self._kv_ready_layers = set()
        self._compute_active = False
        self._request_archived = False


class GlobalScheduler:
    L1_STRATEGIES = (
        "round_robin",
        "least_loaded",
        "length_grouped",
        "pressure_balanced",
    )

    L2_STRATEGIES = ("round_robin", "least_loaded", "random")

    def __init__(
        self,
        instance_config,
        l1_strategy="round_robin",
        l2_strategy="round_robin",
        rng_seed=42,
    ):
        self.instance_config = instance_config
        self.l1_strategy = l1_strategy
        self.l2_strategy = l2_strategy
        self.rng = random.Random(rng_seed)
        self.instances = []
        self.npu_to_instance = {}
        self.instance_npus = {}
        self._l1_rr_idx = 0
        self._l2_rr_idx = {}
        self._instance_queue_lens = {}
        self._instance_npu_queue_lens = {}
        self._instance_avg_req_bw = {}
        self._instance_req_bw_sum = {}
        self._instance_active_req_bw = {}
        self._instance_npu_busy = {}
        self._global_pending_queue = []
        self._n_layers = SIM_N_LAYERS
        for inst_id, npu_ids in instance_config.items():
            self.instances.append(inst_id)
            self.instance_npus[inst_id] = list(npu_ids)
            self._l2_rr_idx[inst_id] = 0
            self._instance_queue_lens[inst_id] = 0
            self._instance_npu_queue_lens[inst_id] = {nid: 0 for nid in npu_ids}
            self._instance_avg_req_bw[inst_id] = 0.0
            self._instance_req_bw_sum[inst_id] = 0.0
            self._instance_active_req_bw[inst_id] = 0.0
            self._instance_npu_busy[inst_id] = {nid: False for nid in npu_ids}
            for nid in npu_ids:
                self.npu_to_instance[nid] = inst_id

    def dispatch(self, request):
        inst_id = self._select_instance(request)
        npu_id = self._select_npu(inst_id, request)
        self._instance_queue_lens[inst_id] += 1
        self._instance_npu_queue_lens[inst_id][npu_id] += 1
        req_bw = request.get("required_bw", 0.0)
        self._instance_req_bw_sum[inst_id] += req_bw
        n_dispatched = sum(self._instance_npu_queue_lens[inst_id].values())
        self._instance_avg_req_bw[inst_id] = (
            self._instance_req_bw_sum[inst_id] / n_dispatched
            if n_dispatched > 0
            else 0.0
        )
        return inst_id, npu_id

    def dispatch_online(self, request, npus_map, instance_sync, event_time):
        inst_id, npu_id = self._select_idle_npu(request, npus_map, event_time)
        if inst_id is not None and npu_id is not None:
            self._instance_npu_busy[inst_id][npu_id] = True
            req_bw = request.get("required_bw", 0.0)
            self._instance_queue_lens[inst_id] += 1
            self._instance_npu_queue_lens[inst_id][npu_id] += 1
            self._instance_req_bw_sum[inst_id] += req_bw
            self._instance_active_req_bw[inst_id] += req_bw
            n_dispatched = sum(self._instance_npu_queue_lens[inst_id].values())
            self._instance_avg_req_bw[inst_id] = (
                self._instance_req_bw_sum[inst_id] / n_dispatched
                if n_dispatched > 0
                else 0.0
            )
            request["npu_id"] = npu_id
            request["instance_id"] = inst_id
            return inst_id, npu_id, request
        self._global_pending_queue.append(request)
        sim_logger.debug(
            "    dispatch_online: no idle NPU, queued globally, pending=%d",
            len(self._global_pending_queue),
        )
        return None, None, None

    def _select_idle_npu(self, request, npus_map, event_time):
        inst_id = self._select_instance_online(request, npus_map, None, event_time)
        if inst_id is None:
            return None, None
        idle_npus = [
            nid
            for nid in self.instance_npus[inst_id]
            if not self._instance_npu_busy[inst_id].get(nid, False)
        ]
        if not idle_npus:
            for other_inst in self.instances:
                if other_inst == inst_id:
                    continue
                idle_npus = [
                    nid
                    for nid in self.instance_npus[other_inst]
                    if not self._instance_npu_busy[other_inst].get(nid, False)
                ]
                if idle_npus:
                    inst_id = other_inst
                    break
            else:
                return None, None
        npu_id = self._select_npu_from_idle(inst_id, idle_npus)
        return inst_id, npu_id

    def _select_npu_from_idle(self, inst_id, idle_npus):
        if self.l2_strategy == "round_robin":
            idx = self._l2_rr_idx[inst_id] % len(idle_npus)
            self._l2_rr_idx[inst_id] += 1
            return idle_npus[idx]
        elif self.l2_strategy == "least_loaded":
            return min(
                idle_npus, key=lambda nid: self._instance_npu_queue_lens[inst_id][nid]
            )
        elif self.l2_strategy == "random":
            return self.rng.choice(idle_npus)
        return idle_npus[0]

    def try_dispatch_pending(
        self,
        event_time,
        npus_map,
        instance_sync,
        event_heap,
        disk_states,
        compute_tag_counter,
        continuous,
        request_loads_map,
        rng,
        n_layers,
        placement_mode,
        disk_bw,
        npus,
        policy="fair",
        io_mode="prefetch",
        io_sched=None,
    ):
        dispatched_any = False
        while self._global_pending_queue:
            inst_id, npu_id = self._select_idle_npu(
                self._global_pending_queue[0], npus_map, event_time
            )
            if inst_id is None or npu_id is None:
                break
            request = self._global_pending_queue.pop(0)
            self._instance_npu_busy[inst_id][npu_id] = True
            req_bw = request.get("required_bw", 0.0)
            self._instance_queue_lens[inst_id] += 1
            self._instance_npu_queue_lens[inst_id][npu_id] += 1
            self._instance_req_bw_sum[inst_id] += req_bw
            self._instance_active_req_bw[inst_id] += req_bw
            n_dispatched = sum(self._instance_npu_queue_lens[inst_id].values())
            self._instance_avg_req_bw[inst_id] = (
                self._instance_req_bw_sum[inst_id] / n_dispatched
                if n_dispatched > 0
                else 0.0
            )
            request["npu_id"] = npu_id
            request["instance_id"] = inst_id
            npu = npus_map.get(npu_id)
            if npu is not None:
                next_bp = build_block_placement(
                    [request],
                    rng,
                    mode=placement_mode,
                    n_layers=n_layers,
                    num_disk=len(disk_states),
                )
                bp_for_npu = next_bp.get(npu_id)
                if bp_for_npu is None and next_bp:
                    bp_for_npu = next_bp[next(iter(next_bp))]
                npu.reset_for_next_request(request, bp_for_npu, event_time, n_layers)
                npu.instance_id = inst_id
                npu.started = True
                npu.request_count += 1
                npu.request_start_time = event_time
                npu.arrival_time = request.get("arrival_time", event_time)
                npu.queueing_delay_ms = event_time - npu.arrival_time
                sim_logger.debug(
                    "      DISPATCH: npu=%d req#%d at t=%.4f seq=%dK nql=%d cat=%s",
                    npu_id,
                    npu.request_count,
                    event_time,
                    npu.seq_len_k,
                    npu.nql,
                    npu.category,
                )
                start_kv_load(
                    npu,
                    0,
                    disk_states,
                    event_heap,
                    event_time,
                    policy,
                    n_layers,
                    disk_bw,
                    npus,
                    io_mode,
                    io_sched,
                )
            sim_logger.info(
                "    DISPATCHED: npu=%d inst=%d at t=%.4f policy=%s pending=%d",
                npu_id,
                inst_id,
                event_time,
                policy,
                len(self._global_pending_queue),
            )
            dispatched_any = True
        return dispatched_any

    def on_npu_request_complete(
        self,
        inst_id,
        npu_id,
        event_time,
        npus_map,
        instance_sync,
        event_heap,
        disk_states,
        compute_tag_counter,
        continuous,
        request_loads_map,
        rng,
        n_layers,
        placement_mode,
        disk_bw,
        npus,
        policy="fair",
        io_mode="prefetch",
        io_sched=None,
    ):
        self._instance_npu_busy[inst_id][npu_id] = False
        self._instance_queue_lens[inst_id] = max(
            0, self._instance_queue_lens[inst_id] - 1
        )
        self._instance_npu_queue_lens[inst_id][npu_id] = max(
            0, self._instance_npu_queue_lens[inst_id][npu_id] - 1
        )
        npu_obj = npus_map.get(npu_id)
        req_bw = npu_obj.required_bw if npu_obj is not None else 0.0
        self._instance_req_bw_sum[inst_id] = max(
            0, self._instance_req_bw_sum[inst_id] - req_bw
        )
        self._instance_active_req_bw[inst_id] = max(
            0, self._instance_active_req_bw[inst_id] - req_bw
        )
        n_dispatched = sum(self._instance_npu_queue_lens[inst_id].values())
        self._instance_avg_req_bw[inst_id] = (
            self._instance_req_bw_sum[inst_id] / n_dispatched
            if n_dispatched > 0
            else 0.0
        )
        sim_logger.info(
            "    NPU REQUEST COMPLETE: npu=%d inst=%d at t=%.4f, pending_global=%d",
            npu_id,
            inst_id,
            event_time,
            len(self._global_pending_queue),
        )
        self.try_dispatch_pending(
            event_time,
            npus_map,
            instance_sync,
            event_heap,
            disk_states,
            compute_tag_counter,
            continuous,
            request_loads_map,
            rng,
            n_layers,
            placement_mode,
            disk_bw,
            npus,
            policy,
            io_mode,
            io_sched,
        )

    def on_npu_idle(self, npu_id, event_time):
        pass

    def on_request_done(self, npu_id):
        inst_id = self.npu_to_instance[npu_id]
        self._instance_queue_lens[inst_id] = max(
            0, self._instance_queue_lens[inst_id] - 1
        )
        self._instance_npu_queue_lens[inst_id][npu_id] = max(
            0, self._instance_npu_queue_lens[inst_id][npu_id] - 1
        )

    def _select_instance(self, request):
        if self.l1_strategy == "round_robin":
            inst_id = self.instances[self._l1_rr_idx % len(self.instances)]
            self._l1_rr_idx += 1
            return inst_id
        elif self.l1_strategy == "least_loaded":
            return min(self.instances, key=lambda i: self._instance_queue_lens[i])
        elif self.l1_strategy == "length_grouped":
            seq_len_k = request.get("seq_len_k", 0)
            nql = request.get("nql", 0)
            if seq_len_k > SEQ_SHORT_BOUNDARY and nql >= NQL_LONG_BOUNDARY:
                target_inst = self.instances[-1]
            elif seq_len_k > SEQ_SHORT_BOUNDARY:
                target_inst = (
                    self.instances[-1] if len(self.instances) > 1 else self.instances[0]
                )
            else:
                target_inst = self.instances[0]
            return target_inst
        elif self.l1_strategy == "pressure_balanced":

            def pressure(inst_id):
                avg = self._instance_avg_req_bw.get(inst_id, 0.0)
                queue_w = self._instance_queue_lens[inst_id]
                return avg + queue_w * 0.5

            return min(self.instances, key=pressure)
        else:
            return self.instances[0]

    def _select_instance_online(
        self, request, npus_map, instance_sync, event_time, idle_instances=None
    ):
        if idle_instances is None:
            idle_instances = [
                i
                for i in self.instances
                if any(
                    not self._instance_npu_busy[i].get(nid, False)
                    for nid in self.instance_npus[i]
                )
            ]
        if not idle_instances:
            return None
        is_short = request.get("seq_len_k", 0) <= SEQ_SHORT_BOUNDARY
        is_low_nql = request.get("nql", 0) < NQL_LONG_BOUNDARY
        if self.l1_strategy == "round_robin":
            available = idle_instances
            inst_id = available[self._l1_rr_idx % len(available)]
            self._l1_rr_idx += 1
            return inst_id
        elif self.l1_strategy == "least_loaded":
            return min(
                idle_instances,
                key=lambda i: sum(
                    1
                    for nid in self.instance_npus[i]
                    if self._instance_npu_busy[i].get(nid, False)
                ),
            )
        elif self.l1_strategy == "length_grouped":
            if is_short and is_low_nql:
                available_short = idle_instances
                return available_short[0]
            elif not is_short:
                available_long = list(reversed(idle_instances))
                return available_long[0]
            else:
                available = idle_instances
                return available[-1] if len(available) > 1 else available[0]
        elif self.l1_strategy == "pressure_balanced":

            def pressure(inst_id):
                avg = self._instance_active_req_bw.get(inst_id, 0.0)
                pending_w = len(self._global_pending_queue)
                return avg + pending_w * 0.5

            return min(idle_instances, key=pressure)
        else:
            return idle_instances[0] if idle_instances else None

    def _select_npu(self, inst_id, request):
        npu_ids = self.instance_npus[inst_id]
        if self.l2_strategy == "round_robin":
            idx = self._l2_rr_idx[inst_id] % len(npu_ids)
            self._l2_rr_idx[inst_id] += 1
            return npu_ids[idx]
        elif self.l2_strategy == "least_loaded":
            return min(
                npu_ids, key=lambda nid: self._instance_npu_queue_lens[inst_id][nid]
            )
        elif self.l2_strategy == "random":
            return self.rng.choice(npu_ids)
        else:
            return npu_ids[0]


NUM_BW_TIERS = 3
QUEUES_PER_TIER = 8  # 3 tiers × 8 queues = 24 queues per disk
DEFAULT_QOS_QUEUE_COUNT = NUM_BW_TIERS * QUEUES_PER_TIER
QOS_LAYOUT_THREE_TIER = "three_tier"
QOS_LAYOUT_EIGHT_GROUP = "eight_group"
EIGHT_GROUP_COUNT = 8

# demand_bw thresholds (GB/s) — derived from typical KV block sizes and compute times
BW_TIER_HIGH_THRESH = 100.0  # above this → high-bw tier
BW_TIER_MID_THRESH = 10.0  # above this → medium-bw tier; at or below → low-bw tier

# CIR per queue in each tier (GB/s). Active queues first receive this guaranteed
# bandwidth, then borrow the remaining disk bandwidth according to their WRR
# weights. Queue PIR is infinite, so the physical disk bandwidth is the only
# bandwidth cap while a queue has data to send.
BW_TIER_CIR = [2.0, 0.5, 0.1]
BW_TIER_WRR = [4.0, 2.0, 1.0]
BW_TIER_CIR_BUDGETS = [value * QUEUES_PER_TIER for value in BW_TIER_CIR]


@dataclass
class IOSchedulingConfig:
    """Configuration for IO dispatch mode and token bucket scheduling."""

    io_dispatch_mode: str = (
        "all_at_once"  # 'all_at_once'|'batched'|'traffic_aware_batched'
    )
    batch_size: int = 8  # number of ASUs per batch
    batch_interval_mode: str = "fixed"  # 'fixed'|'demand_aware'
    batch_dispatch_interval_us: float = 200.0
    batch_dispatch_headroom: float = 1.0
    batch_min_dispatch_interval_us: float = 0.0
    batch_max_dispatch_interval_us: float = float("inf")
    qos_queue_count: int = DEFAULT_QOS_QUEUE_COUNT
    qos_layout: str = QOS_LAYOUT_THREE_TIER
    qos_queue_max_active_flows: int = 4
    token_bucket_enabled: bool = False
    token_bucket_refill_us: float = (
        80.0  # hardware constraint: token refill interval (us)
    )
    token_bucket_pir_cap: bool = (
        True  # True=cap at PIR, False=allow surplus beyond tokens
    )


class TokenBucket:
    def __init__(
        self, disk_id, pir_gb_per_s=DISK_BW, refill_interval_us=80.0, capped=True
    ):
        self.disk_id = disk_id
        self.pir_gb_per_s = pir_gb_per_s
        self.refill_interval_us = refill_interval_us
        self.refill_interval_ms = refill_interval_us / 1000.0
        self.max_tokens_gb = pir_gb_per_s * refill_interval_us / 1e6
        self.tokens_gb = self.max_tokens_gb  # start with full bucket
        self.last_refill_time_ms = 0.0
        self.capped = capped

    def refill(self, current_time_ms):
        """Refill tokens based on elapsed 80us intervals. Use-it-or-lose-it.

        Returns True if refill happened."""
        elapsed_ms = current_time_ms - self.last_refill_time_ms
        if elapsed_ms < self.refill_interval_ms - 1e-12 or not np.isfinite(elapsed_ms):
            return False
        n_intervals = int(elapsed_ms / self.refill_interval_ms)
        if n_intervals <= 0:
            return False
        # Use-it-or-lose-it: reset to one interval's worth, don't accumulate
        self.tokens_gb = self.max_tokens_gb
        self.last_refill_time_ms += n_intervals * self.refill_interval_ms
        return True

    def try_consume(self, gb):
        """Try to consume tokens for data transfer.

        Returns (allowed_gb, tokens_depleted).

        If capped and tokens run out, returns partial amount."""
        if not self.capped:
            # Uncapped: track consumption but don't limit
            self.tokens_gb = max(0, self.tokens_gb - gb)
            return (gb, False)
        if gb <= self.tokens_gb:
            self.tokens_gb -= gb
            return (gb, False)
        else:
            # Tokens depleted — can only transfer what's left
            allowed = self.tokens_gb
            self.tokens_gb = 0
            return (allowed, True)

    def has_tokens(self):
        """Whether tokens remain for this interval."""
        return self.tokens_gb > 1e-15


class BatchState:
    """Tracks per-(NPU, layer) batch dispatch progress for traffic orchestration."""

    __slots__ = (
        "npu_id",
        "layer",
        "batches",
        "current_batch_idx",
        "total_blocks",
        "remaining_blocks",
        "remaining_by_disk",
        "remaining_count",
    )

    def __init__(
        self, npu_id, layer, batches=None, total_blocks=0, remaining_blocks=None
    ):
        self.npu_id = npu_id
        self.layer = layer
        self.batches = batches or []  # list[list[dict]] for 'batched' mode
        self.current_batch_idx = 0
        self.total_blocks = total_blocks
        self.remaining_blocks = []
        self.remaining_by_disk = defaultdict(deque)
        for block in remaining_blocks or []:
            self.remaining_by_disk[block["disk"]].append(block)
        self.remaining_count = sum(
            len(blocks) for blocks in self.remaining_by_disk.values()
        )


def _demand_bw_to_tier(demand_bw):
    """Map a demand_bw value to a tier index based on thresholds."""
    if demand_bw > BW_TIER_HIGH_THRESH:
        return 0  # high-bw
    elif demand_bw > BW_TIER_MID_THRESH:
        return 1  # medium-bw
    else:
        return 2  # low-bw


def qos_tier_queue_counts(qos_queue_count=DEFAULT_QOS_QUEUE_COUNT):
    """Split a total queue count as evenly as possible across three tiers."""
    if not isinstance(qos_queue_count, int) or qos_queue_count < NUM_BW_TIERS:
        raise ValueError(
            f"qos_queue_count must be an integer >= {NUM_BW_TIERS}: "
            f"{qos_queue_count!r}"
        )
    base, remainder = divmod(qos_queue_count, NUM_BW_TIERS)
    return tuple(base + (tier < remainder) for tier in range(NUM_BW_TIERS))


def flow_to_queue_id(
    flow,
    num_npu=128,
    policy="queue_wrr",
    tier_queue_counts=None,
):
    """Map a BlockIOFlow to a queue ID.

    fair/demand_driven: single queue (queue 0), all flows active.
    queue_wrr/urgency_driven: tier-based mapping with round-robin within tier.
    """
    if policy in ("fair", "demand_driven"):
        return 0
    if tier_queue_counts is None:
        tier_queue_counts = qos_tier_queue_counts()
    tier = _demand_bw_to_tier(flow.demand_bw)
    tier_base = sum(tier_queue_counts[:tier])
    rr_offset = (flow.npu_id + flow.layer) % tier_queue_counts[tier]
    return tier_base + rr_offset


def build_default_queue_config(
    qos_queue_count=DEFAULT_QOS_QUEUE_COUNT,
    disk_bw=DISK_BW,
):
    """Build a tiered CIR/PIR/WRR configuration for one physical disk.

    Tiers are defined by demand_bw ranges, not by SS/SL/LS/LL labels.
    CIR is the active queue's guarantee. Surplus disk bandwidth is distributed
    by WRR weight, and infinite PIR permits borrowing up to physical capacity.
    Tier CIR budgets remain 16/4/0.8 GB/s as queue count changes.
    """
    tier_queue_counts = qos_tier_queue_counts(qos_queue_count)
    config = {}
    base_idx = 0
    for tier in range(NUM_BW_TIERS):
        queue_cir = BW_TIER_CIR_BUDGETS[tier] / tier_queue_counts[tier]
        for q in range(tier_queue_counts[tier]):
            qid = base_idx + q
            config[qid] = {
                "cir": queue_cir,
                "pir": float("inf"),
                "wrr_weight": BW_TIER_WRR[tier],
                "category": f"T{tier}",  # tier label for debugging
                "group_id": tier,
            }
        base_idx += tier_queue_counts[tier]
    return config


def build_eight_group_queue_config(qos_queue_count=256, disk_bw=DISK_BW):
    """Build 8 equal groups with equal per-queue CIR and group WRR weight 1."""
    if qos_queue_count % EIGHT_GROUP_COUNT != 0:
        raise ValueError(
            f"eight_group qos_queue_count must be divisible by {EIGHT_GROUP_COUNT}: "
            f"{qos_queue_count}"
        )
    queues_per_group = qos_queue_count // EIGHT_GROUP_COUNT
    queue_cir = float(disk_bw) / qos_queue_count
    return {
        queue_id: {
            "cir": queue_cir,
            "pir": float("inf"),
            "wrr_weight": 1.0,
            "category": f"G{queue_id // queues_per_group}",
            "group_id": queue_id // queues_per_group,
        }
        for queue_id in range(qos_queue_count)
    }


class DiskQueue:
    """Single FCFS queue with up to ``max_depth`` concurrently active flows."""

    __slots__ = (
        "queue_id",
        "cir",
        "pir",
        "wrr_weight",
        "pending",
        "active_flows",
        "assigned_bw",
        "bytes_served",
        "n_activations",
        "category",
        "group_id",
        "max_depth",
        "max_active_flows_observed",
        "_per_flow_shares",
    )

    def __init__(
        self,
        queue_id,
        cir=0.0,
        pir=0.0,
        wrr_weight=1.0,
        category="SS",
        group_id=0,
        max_depth=1,
    ):
        self.queue_id = queue_id
        self.cir = cir
        self.pir = pir
        self.wrr_weight = wrr_weight
        self.pending = deque()
        self.active_flows = []  # active flow(s) — max_depth controls concurrency
        self.assigned_bw = 0.0
        self.bytes_served = 0.0
        self.n_activations = 0
        self.category = category
        self.group_id = group_id
        self.max_depth = max_depth  # max concurrent flows per queue
        self.max_active_flows_observed = 0
        self._per_flow_shares = []  # per-flow bandwidth shares

    def is_active(self):
        return len(self.active_flows) > 0

    def is_empty(self):
        return len(self.active_flows) == 0 and len(self.pending) == 0

    def enqueue(self, flow):
        """Add flow to tail of queue. Activate if below max_depth."""
        flow._queue = self
        self.pending.append(flow)
        return self._try_activate()

    def _try_activate(self):
        """Activate pending flows up to max_depth."""
        activated = []
        while len(self.active_flows) < self.max_depth and self.pending:
            flow = self.pending.popleft()
            self.active_flows.append(flow)
            self.n_activations += 1
            activated.append(flow)
        self.max_active_flows_observed = max(
            self.max_active_flows_observed,
            len(self.active_flows),
        )
        return activated

    def complete_flow(self, flow, current_time):
        """Complete a flow and return it together with newly activated flows."""
        completed, activated = self.complete_flows([flow], current_time)
        return (completed[0] if completed else None), activated

    def complete_flows(self, flows, current_time):
        """Complete several flows with one linear rebuild of the active list."""
        requested = set(flows)
        completed = [flow for flow in self.active_flows if flow in requested]
        if not completed:
            return [], []
        completed_set = set(completed)
        for flow in completed:
            flow.active = False
            self.bytes_served += getattr(flow, "total_gb", flow.remaining_gb)
        self.active_flows = [
            flow for flow in self.active_flows if flow not in completed_set
        ]
        return completed, self._try_activate()


# ── Fixed-bandwidth QoS disk scheduling ──────────────────────────────────────


_SIM_EPS = 1e-12


def _max_min_fair_shares(flows, capacity):
    """Return weighted max-min shares for active (possibly aggregated) flows."""
    if not flows or capacity <= 0:
        return {flow: 0.0 for flow in flows}

    weights = {flow: max(1, int(getattr(flow, "block_count", 1))) for flow in flows}
    demands = {
        flow: flow.demand_bw if flow.demand_bw > 0 else float("inf") for flow in flows
    }
    equal_share = float(capacity) / sum(weights.values())
    if all(demand >= equal_share for demand in demands.values()):
        return {flow: equal_share * weights[flow] for flow in flows}

    ordered = sorted(flows, key=lambda flow: demands[flow])
    allocations = {}
    remaining = float(capacity)
    remaining_weight = sum(weights.values())

    for index, flow in enumerate(ordered):
        equal_share = remaining / remaining_weight
        if demands[flow] <= equal_share:
            allocation = demands[flow] * weights[flow]
            allocations[flow] = allocation
            remaining -= allocation
            remaining_weight -= weights[flow]
            continue
        for unsatisfied in ordered[index:]:
            allocations[unsatisfied] = equal_share * weights[unsatisfied]
        remaining = 0.0
        break

    return allocations


def _uncapped_wrr_queue_bandwidth(active_queue_flows, capacity):
    """Allocate CIR first, then lend surplus by WRR up to each queue's PIR."""
    assignments = {}
    limits = {}
    for queue in active_queue_flows:
        limits[queue] = queue.pir
        assignments[queue] = min(queue.cir, limits[queue])

    remaining = max(0.0, float(capacity) - sum(assignments.values()))
    eligible = {
        queue
        for queue in active_queue_flows
        if queue.wrr_weight > 0 and assignments[queue] < limits[queue] - _SIM_EPS
    }

    while remaining > _SIM_EPS and eligible:
        total_weight = sum(queue.wrr_weight for queue in eligible)
        if total_weight <= _SIM_EPS:
            break
        surplus_per_weight = remaining / total_weight
        saturated = [
            queue
            for queue in eligible
            if limits[queue] - assignments[queue]
            <= surplus_per_weight * queue.wrr_weight + _SIM_EPS
        ]
        if not saturated:
            for queue in eligible:
                assignments[queue] += surplus_per_weight * queue.wrr_weight
            remaining = 0.0
            break
        for queue in saturated:
            addition = max(0.0, limits[queue] - assignments[queue])
            assignments[queue] += addition
            remaining = max(0.0, remaining - addition)
            eligible.remove(queue)

    return assignments


def _eight_group_queue_bandwidth(active_queue_flows, capacity, group_weights):
    """Guarantee queue CIR, then share surplus by group and equally within it."""
    assignments = {queue: min(queue.cir, queue.pir) for queue in active_queue_flows}
    remaining = max(0.0, float(capacity) - sum(assignments.values()))
    queues_by_group = defaultdict(list)
    for queue in active_queue_flows:
        queues_by_group[queue.group_id].append(queue)
    total_group_weight = sum(group_weights[group_id] for group_id in queues_by_group)
    if remaining <= _SIM_EPS or total_group_weight <= _SIM_EPS:
        return assignments

    for group_id, queues in queues_by_group.items():
        group_surplus = remaining * group_weights[group_id] / total_group_weight
        queue_surplus = group_surplus / len(queues)
        for queue in queues:
            assignments[queue] += queue_surplus
    return assignments


def _uncapped_weighted_flow_shares(flows, capacity):
    """Share queue bandwidth by virtual-flow weight without a demand/PIR cap."""
    if not flows or capacity <= 0:
        return {flow: 0.0 for flow in flows}
    weights = {flow: max(1, int(getattr(flow, "block_count", 1))) for flow in flows}
    total_weight = sum(weights.values())
    return {flow: capacity * weights[flow] / total_weight for flow in flows}


class DiskIOScheduler:
    """Own the logical queues of one disk and assign their bandwidth.

    QoS policies guarantee each active queue's configured CIR, then lend the
    remaining disk bandwidth by WRR weight with uncapped queue PIR. The fair
    policy shares physical disk bandwidth directly among all active flows.
    """

    def __init__(self, disk_state, policy="queue_wrr", disk_bw=DISK_BW, io_sched=None):
        self.disk_state = disk_state
        self.policy = policy
        self.disk_bw = float(disk_bw)
        self.io_sched = io_sched or IOSchedulingConfig()
        self.qos_queue_count = self.io_sched.qos_queue_count
        self.qos_layout = self.io_sched.qos_layout
        if self.qos_layout == QOS_LAYOUT_THREE_TIER:
            self.group_queue_counts = qos_tier_queue_counts(self.qos_queue_count)
            self.group_cir_gbps = tuple(
                BW_TIER_CIR_BUDGETS[tier] / self.group_queue_counts[tier]
                for tier in range(NUM_BW_TIERS)
            )
            self.group_weights = tuple(BW_TIER_WRR)
            queue_config = build_default_queue_config(
                self.qos_queue_count, self.disk_bw
            )
        elif self.qos_layout == QOS_LAYOUT_EIGHT_GROUP:
            if self.qos_queue_count % EIGHT_GROUP_COUNT != 0:
                raise ValueError(
                    "eight_group qos_queue_count must be divisible by "
                    f"{EIGHT_GROUP_COUNT}: {self.qos_queue_count}"
                )
            queues_per_group = self.qos_queue_count // EIGHT_GROUP_COUNT
            self.group_queue_counts = (queues_per_group,) * EIGHT_GROUP_COUNT
            self.group_cir_gbps = (
                self.disk_bw / self.qos_queue_count,
            ) * EIGHT_GROUP_COUNT
            self.group_weights = (1.0,) * EIGHT_GROUP_COUNT
            queue_config = build_eight_group_queue_config(
                self.qos_queue_count, self.disk_bw
            )
        else:
            raise ValueError(f"unknown qos_layout: {self.qos_layout}")
        self.tier_queue_counts = self.group_queue_counts
        self.tier_cir_gbps = self.group_cir_gbps
        self.queues = {}
        self.active_queues = set()
        self.n_flows_enqueued = 0
        self.n_blocks_enqueued = 0
        self.n_bandwidth_updates = 0
        self.n_token_events = 0
        self.n_rebalance_events = 0
        self.outstanding_blocks = 0
        self.max_outstanding_blocks = 0
        self.max_observed_queue_bw = defaultdict(float)
        self.pending_rebalance_time = None
        self.rebalance_generation = 0

        if policy in ("fair", "demand_driven"):
            self.queues[0] = DiskQueue(
                queue_id=0,
                cir=self.disk_bw,
                pir=self.disk_bw,
                wrr_weight=1.0,
                category="fair",
                group_id=0,
                max_depth=self.io_sched.qos_queue_max_active_flows,
            )
            self.configured_queue_bandwidth = self.disk_bw
        else:
            self.configured_queue_bandwidth = sum(
                config["cir"] for config in queue_config.values()
            )
            if self.configured_queue_bandwidth > self.disk_bw + _SIM_EPS:
                raise ValueError(
                    "QoS CIR total exceeds physical disk bandwidth: "
                    f"{self.configured_queue_bandwidth:.3f} > {self.disk_bw:.3f} GB/s"
                )
            for queue_id, config in queue_config.items():
                self.queues[queue_id] = DiskQueue(
                    queue_id=queue_id,
                    cir=config["cir"],
                    pir=config["pir"],
                    wrr_weight=config["wrr_weight"],
                    category=config["category"],
                    group_id=config["group_id"],
                    max_depth=self.io_sched.qos_queue_max_active_flows,
                )

        self.token_bucket = None
        if self.io_sched.token_bucket_enabled:
            self.token_bucket = TokenBucket(
                disk_id=disk_state.disk_id,
                pir_gb_per_s=self.disk_bw,
                refill_interval_us=self.io_sched.token_bucket_refill_us,
                capped=self.io_sched.token_bucket_pir_cap,
            )

        disk_state.queue_scheduler = self

    def queue_id_for_flow(self, flow):
        if self.policy in ("fair", "demand_driven"):
            return 0
        if self.qos_layout == QOS_LAYOUT_EIGHT_GROUP:
            queues_per_group = self.group_queue_counts[0]
            group_id = flow.npu_id % EIGHT_GROUP_COUNT
            group_offset = (flow.npu_id // EIGHT_GROUP_COUNT + flow.layer) % (
                queues_per_group
            )
            return group_id * queues_per_group + group_offset
        return flow_to_queue_id(
            flow,
            policy=self.policy,
            tier_queue_counts=self.group_queue_counts,
        )

    def _account_until(self, current_time):
        state = self.disk_state
        if current_time <= state.last_event_time:
            return
        duration = current_time - state.last_event_time
        if state.active_flows:
            state.busy_time += duration
            used_bw = sum(flow.bw for flow in state.active_flows if flow.active)
            state.surplus_bw_integral += max(0.0, self.disk_bw - used_bw) * duration
            state.total_bw_integral += self.disk_bw * duration
        else:
            state.idle_time += duration
        state.last_event_time = current_time

    def settle(self, current_time):
        """Advance every active flow to current_time using its old bandwidth."""
        if current_time <= self.disk_state.last_event_time + _SIM_EPS:
            return
        self._account_until(current_time)
        transferred_gb = 0.0
        for flow in list(self.disk_state.active_flows):
            elapsed_ms = max(0.0, current_time - flow.start_time)
            if flow.active and flow.bw > 0 and elapsed_ms > 0:
                done_gb = min(flow.remaining_gb, flow.bw * elapsed_ms / 1000.0)
                flow.remaining_gb = max(0.0, flow.remaining_gb - done_gb)
                transferred_gb += done_gb
            flow.start_time = current_time
            if flow.remaining_gb <= _SIM_EPS:
                flow.remaining_gb = 0.0
                flow.end_time = current_time

        if self.token_bucket is not None and transferred_gb > 0:
            self.token_bucket.tokens_gb = max(
                0.0, self.token_bucket.tokens_gb - transferred_gb
            )

    def request_redistribution(self, current_time, event_heap):
        """Coalesce all redistribution requests for one disk and timestamp."""
        if (
            self.pending_rebalance_time is not None
            and abs(self.pending_rebalance_time - current_time) <= _SIM_EPS
        ):
            return
        self.rebalance_generation += 1
        self.pending_rebalance_time = current_time
        heapq.heappush(
            event_heap,
            (
                current_time,
                DISK_REBALANCE,
                self.disk_state.disk_id,
                0,
                self.rebalance_generation,
            ),
        )
        self.n_rebalance_events += 1

    def _activate_new_flows(self, flows, current_time):
        for flow in flows:
            flow.start_time = current_time
            flow.bw = 0.0
            flow.end_time = float("inf")
            self.disk_state.add_flow(flow, current_time, self.disk_bw)
            self.active_queues.add(flow._queue)

    def enqueue_many(self, flows, current_time):
        if not flows:
            return
        self.settle(current_time)
        for flow in flows:
            queue = self.queues[flow.queue_id]
            activated = queue.enqueue(flow)
            self._activate_new_flows(activated, current_time)
            self.n_flows_enqueued += 1
            self.n_blocks_enqueued += flow.block_count
            self.outstanding_blocks += flow.block_count
            self.max_outstanding_blocks = max(
                self.max_outstanding_blocks,
                self.outstanding_blocks,
            )

    def complete_ready_flows(self, current_time):
        self.settle(current_time)
        completed = [
            flow
            for flow in self.disk_state.active_flows
            if flow.remaining_gb <= _SIM_EPS
        ]
        self.outstanding_blocks = max(
            0,
            self.outstanding_blocks - sum(flow.block_count for flow in completed),
        )
        self.disk_state.remove_flows(completed, current_time, self.disk_bw)
        completed_by_queue = defaultdict(list)
        for flow in completed:
            completed_by_queue[flow._queue].append(flow)
        for queue, queue_flows in completed_by_queue.items():
            _, activated = queue.complete_flows(queue_flows, current_time)
            self._activate_new_flows(activated, current_time)
            if queue.active_flows:
                self.active_queues.add(queue)
            else:
                self.active_queues.discard(queue)
                queue.assigned_bw = 0.0
                queue._per_flow_shares = []
        return completed

    def _flow_allocations(self):
        active_flows = [flow for flow in self.disk_state.active_flows if flow.active]
        if self.policy in ("fair", "demand_driven"):
            queue = self.queues[0]
            queue.assigned_bw = self.disk_bw if active_flows else 0.0
            self.max_observed_queue_bw[0] = max(
                self.max_observed_queue_bw[0], queue.assigned_bw
            )
            allocations = _max_min_fair_shares(active_flows, self.disk_bw)
            queue._per_flow_shares = [allocations[flow] for flow in active_flows]
            return allocations

        allocations = {}
        active_queue_flows = {}
        for queue in tuple(self.active_queues):
            flows = [flow for flow in queue.active_flows if flow.active]
            if flows:
                active_queue_flows[queue] = flows
            else:
                self.active_queues.discard(queue)
                queue.assigned_bw = 0.0
                queue._per_flow_shares = []
        if self.qos_layout == QOS_LAYOUT_EIGHT_GROUP:
            queue_bandwidth = _eight_group_queue_bandwidth(
                active_queue_flows,
                self.disk_bw,
                self.group_weights,
            )
        else:
            queue_bandwidth = _uncapped_wrr_queue_bandwidth(
                active_queue_flows, self.disk_bw
            )
        for queue, flows in active_queue_flows.items():
            queue.assigned_bw = queue_bandwidth[queue]
            self.max_observed_queue_bw[queue.queue_id] = max(
                self.max_observed_queue_bw[queue.queue_id], queue.assigned_bw
            )
            shares = _uncapped_weighted_flow_shares(flows, queue.assigned_bw)
            queue._per_flow_shares = [shares[flow] for flow in flows]
            allocations.update(shares)
        return allocations

    def redistribute(self, current_time, event_heap):
        if (
            self.pending_rebalance_time is not None
            and self.pending_rebalance_time <= current_time + _SIM_EPS
        ):
            self.pending_rebalance_time = None
            self.rebalance_generation += 1
        self.settle(current_time)
        if self.token_bucket is not None:
            self.token_bucket.refill(current_time)

        allocations = self._flow_allocations()
        if (
            self.token_bucket is not None
            and self.token_bucket.capped
            and not self.token_bucket.has_tokens()
        ):
            allocations = {flow: 0.0 for flow in allocations}

        for flow in self.disk_state.active_flows:
            new_bw = max(0.0, allocations.get(flow, 0.0))
            flow.start_time = current_time
            if flow.remaining_gb <= _SIM_EPS:
                flow.bw = 0.0
                flow.end_time = current_time + _SIM_EPS
            else:
                flow.bw = new_bw
                flow.end_time = (
                    current_time + flow.remaining_gb / new_bw * 1000.0
                    if new_bw > 0
                    else float("inf")
                )

        self.n_bandwidth_updates += 1
        self._schedule_next_event(current_time, event_heap)

    def _schedule_next_event(self, current_time, event_heap):
        self.disk_state.generation += 1
        generation = self.disk_state.generation
        finite_ends = [
            flow.end_time
            for flow in self.disk_state.active_flows
            if flow.active and np.isfinite(flow.end_time)
        ]
        if finite_ends:
            event_time = min(finite_ends)
            if event_time <= current_time:
                event_time = current_time + _SIM_EPS
            heapq.heappush(
                event_heap,
                (
                    event_time,
                    DISK_COMPLETION,
                    self.disk_state.disk_id,
                    0,
                    generation,
                ),
            )

        bucket = self.token_bucket
        active_bw = sum(flow.bw for flow in self.disk_state.active_flows if flow.active)
        if bucket is None or not bucket.capped or not self.disk_state.active_flows:
            return

        next_refill = bucket.last_refill_time_ms + bucket.refill_interval_ms
        if active_bw > 0 and bucket.tokens_gb > _SIM_EPS:
            depletion_time = current_time + bucket.tokens_gb / active_bw * 1000.0
            token_event_time = min(next_refill, depletion_time)
        else:
            token_event_time = next_refill
        if token_event_time <= current_time:
            token_event_time = current_time + _SIM_EPS
        heapq.heappush(
            event_heap,
            (
                token_event_time,
                TOKEN_REFILL,
                self.disk_state.disk_id,
                0,
                generation,
            ),
        )
        self.n_token_events += 1


def _disk_pressure(disk_state):
    return disk_state.queue_scheduler.outstanding_blocks


def _select_traffic_aware_batch(batch_state, batch_size, disk_states):
    """Select a pressure-balanced batch without rescanning every remaining block."""
    selected = []
    selected_per_disk = defaultdict(int)
    base_pressure = {}
    candidate_heap = []
    for disk_id, blocks in batch_state.remaining_by_disk.items():
        if not blocks:
            continue
        base_pressure[disk_id] = _disk_pressure(disk_states[disk_id])
        heapq.heappush(
            candidate_heap,
            (base_pressure[disk_id], disk_id, blocks[0]["block_idx"]),
        )

    while candidate_heap and len(selected) < batch_size:
        _, disk_id, _ = heapq.heappop(candidate_heap)
        blocks = batch_state.remaining_by_disk[disk_id]
        block = blocks.popleft()
        selected.append(block)
        selected_per_disk[disk_id] += 1
        batch_state.remaining_count -= 1
        if blocks:
            heapq.heappush(
                candidate_heap,
                (
                    base_pressure[disk_id] + selected_per_disk[disk_id],
                    disk_id,
                    blocks[0]["block_idx"],
                ),
            )
    return selected


def _dispatch_block_descriptors(
    npu,
    layer,
    block_descriptors,
    disk_states,
    event_heap,
    current_time,
    policy,
):
    grouped_blocks = {}
    for block in block_descriptors:
        disk_id = block["disk"]
        if disk_id < 0 or disk_id >= len(disk_states):
            raise ValueError(f"invalid disk id {disk_id}")
        key = (disk_id, block["gb"])
        group = grouped_blocks.setdefault(
            key,
            {
                "disk": disk_id,
                "gb": 0.0,
                "block_idx": block["block_idx"],
                "block_count": 0,
            },
        )
        group["gb"] += block["gb"]
        group["block_count"] += 1

    by_disk = defaultdict(list)
    for group in grouped_blocks.values():
        disk_id = group["disk"]
        flow = BlockIOFlow(
            npu_id=npu.npu_id,
            layer=layer,
            block_idx=group["block_idx"],
            disk_id=disk_id,
            total_gb=group["gb"],
            bw=0.0,
            start_time=current_time,
            priority=npu.bw_priority,
            demand_bw=npu.required_bw,
            category=npu.category,
            block_count=group["block_count"],
        )
        queue_scheduler = disk_states[disk_id].queue_scheduler
        flow.queue_id = queue_scheduler.queue_id_for_flow(flow)
        npu.active_block_flows[layer].add(flow)
        by_disk[disk_id].append(flow)

    for disk_id, flows in by_disk.items():
        scheduler = disk_states[disk_id].queue_scheduler
        scheduler.enqueue_many(flows, current_time)
        scheduler.request_redistribution(current_time, event_heap)


def start_kv_load(
    npu,
    layer,
    disk_states,
    event_heap,
    current_time,
    policy,
    n_layers,
    disk_bw,
    npus,
    io_mode="prefetch",
    io_sched=None,
):
    """Start one layer's KV reads according to the configured dispatch mode."""
    if layer < 0 or layer >= n_layers or layer in npu.per_layer_kv_load_start:
        return False

    io_sched = io_sched or IOSchedulingConfig()
    placement = npu.block_placement.get(layer, [])
    blocks = [
        {"disk": block["disk"], "gb": block["gb"], "block_idx": block_idx}
        for block_idx, block in enumerate(placement)
    ]
    npu.per_layer_kv_load_start[layer] = current_time
    npu.pending_blocks[layer] = len(blocks)
    if npu.trace_enabled:
        npu.layer_trace.setdefault(layer, {})["kv_start_ms"] = current_time

    if not blocks:
        npu._kv_ready_layers.add(layer)
        npu.current_request_kv_actual_dur[layer] = 0.0
        if layer == 0:
            npu.layer0_kv_end_time = current_time
        elif layer == 1:
            npu.layer1_kv_end_time = current_time
        if npu.trace_enabled:
            npu.layer_trace.setdefault(layer, {})["kv_end_ms"] = current_time
        while npu.kv_loaded_up_to + 1 in npu._kv_ready_layers:
            npu.kv_loaded_up_to += 1
        return True

    dispatch_mode = io_sched.io_dispatch_mode
    batch_size = max(1, int(io_sched.batch_size))
    if dispatch_mode == "all_at_once":
        first_batch = blocks
    elif dispatch_mode == "batched":
        batches = [
            blocks[index : index + batch_size]
            for index in range(0, len(blocks), batch_size)
        ]
        batch_state = BatchState(
            npu.npu_id,
            layer,
            batches=batches,
            total_blocks=len(blocks),
        )
        batch_state.current_batch_idx = 1
        npu.batch_states[layer] = batch_state
        first_batch = batches[0]
    elif dispatch_mode == "traffic_aware_batched":
        batch_state = BatchState(
            npu.npu_id,
            layer,
            total_blocks=len(blocks),
            remaining_blocks=list(blocks),
        )
        npu.batch_states[layer] = batch_state
        first_batch = _select_traffic_aware_batch(batch_state, batch_size, disk_states)
        batch_state.current_batch_idx = 1
    else:
        raise ValueError(f"unknown io_dispatch_mode: {dispatch_mode}")

    _record_batch_dispatch(npu, layer, current_time)
    _dispatch_block_descriptors(
        npu,
        layer,
        first_batch,
        disk_states,
        event_heap,
        current_time,
        policy,
    )
    if dispatch_mode != "all_at_once":
        _schedule_next_batch_dispatch(
            npu,
            layer,
            batch_state,
            event_heap,
            current_time,
            io_sched,
            first_batch,
        )
    return True


def _record_batch_dispatch(npu, layer, current_time):
    npu._batches_dispatched += 1
    if npu.trace_enabled:
        layer_trace = npu.layer_trace.setdefault(layer, {})
        layer_trace.setdefault("batch_dispatch_ms", []).append(current_time)


def _batch_state_has_undispatched(batch_state, dispatch_mode):
    if dispatch_mode == "batched":
        return batch_state.current_batch_idx < len(batch_state.batches)
    if dispatch_mode == "traffic_aware_batched":
        return batch_state.remaining_count > 0
    return False


def _choose_batch_dispatch_interval_us(npu, dispatched_batch, io_sched):
    if io_sched.batch_interval_mode == "fixed":
        return io_sched.batch_dispatch_interval_us

    batch_gb = sum(block["gb"] for block in dispatched_batch)
    target_bw = npu.required_bw * io_sched.batch_dispatch_headroom
    if batch_gb <= _SIM_EPS or target_bw <= _SIM_EPS:
        interval_us = io_sched.batch_dispatch_interval_us
    else:
        interval_us = batch_gb / target_bw * 1e6
    return min(
        max(interval_us, io_sched.batch_min_dispatch_interval_us),
        io_sched.batch_max_dispatch_interval_us,
    )


def _schedule_next_batch_dispatch(
    npu,
    layer,
    batch_state,
    event_heap,
    current_time,
    io_sched,
    dispatched_batch,
):
    if not _batch_state_has_undispatched(
        batch_state,
        io_sched.io_dispatch_mode,
    ):
        return False
    interval_us = _choose_batch_dispatch_interval_us(
        npu,
        dispatched_batch,
        io_sched,
    )
    interval_ms = interval_us / 1000.0
    if npu.trace_enabled:
        layer_trace = npu.layer_trace.setdefault(layer, {})
        layer_trace.setdefault("batch_interval_us", []).append(interval_us)
    heapq.heappush(
        event_heap,
        (
            current_time + interval_ms,
            BATCH_DISPATCH,
            npu.npu_id,
            layer,
            batch_state.current_batch_idx,
        ),
    )
    return True


def _dispatch_next_batch(
    npu,
    layer,
    disk_states,
    event_heap,
    current_time,
    policy,
    io_sched,
):
    if npu.pending_blocks.get(layer, 0) <= 0:
        return False
    batch_state = npu.batch_states.get(layer)
    if batch_state is None:
        return False

    if io_sched.io_dispatch_mode == "batched":
        if batch_state.current_batch_idx >= len(batch_state.batches):
            return False
        batch = batch_state.batches[batch_state.current_batch_idx]
        batch_state.current_batch_idx += 1
    elif io_sched.io_dispatch_mode == "traffic_aware_batched":
        if batch_state.remaining_count <= 0:
            return False
        batch = _select_traffic_aware_batch(
            batch_state,
            max(1, int(io_sched.batch_size)),
            disk_states,
        )
        batch_state.current_batch_idx += 1
    else:
        return False

    _record_batch_dispatch(npu, layer, current_time)
    _dispatch_block_descriptors(
        npu,
        layer,
        batch,
        disk_states,
        event_heap,
        current_time,
        policy,
    )
    _schedule_next_batch_dispatch(
        npu,
        layer,
        batch_state,
        event_heap,
        current_time,
        io_sched,
        batch,
    )
    return True


# ── Event loop ────────────────────────────────────────────────────────────────


class _SimulationContext:
    def __init__(
        self,
        scheduler,
        event_heap,
        disk_states,
        npus,
        request_loads_map,
        rng,
        n_layers,
        placement_mode,
        disk_bw,
        policy,
        io_mode,
        io_sched,
    ):
        self.scheduler = scheduler
        self.event_heap = event_heap
        self.disk_states = disk_states
        self.npus = npus
        self.npus_map = {npu.npu_id: npu for npu in npus}
        self.request_loads_map = request_loads_map
        self.rng = rng
        self.n_layers = n_layers
        self.placement_mode = placement_mode
        self.disk_bw = disk_bw
        self.policy = policy
        self.io_mode = io_mode
        self.io_sched = io_sched
        self.compute_tag_counter = {npu.npu_id: 0 for npu in npus}
        self.instance_sync = {}
        self.completed_requests = 0
        self.stale_events = 0
        self.event_counts = defaultdict(int)


def _archive_current_request(npu, n_layers):
    if npu._request_archived or npu.ttft_ms is None:
        return
    per_layer_compute_ms = npu.per_layer_us / 1000.0
    ideal_l1_kv_ms = (
        npu.per_layer_kv_gb / NPU_BW_LIMIT * 1000.0 if npu.per_layer_kv_gb > 0 else 0.0
    )
    actual_l0_kv_ms = (
        npu.layer0_kv_end_time - npu.request_start_time
        if npu.layer0_kv_end_time is not None and npu.request_start_time is not None
        else 0.0
    )
    ideal_excl012_ms = (
        actual_l0_kv_ms
        + max(ideal_l1_kv_ms, per_layer_compute_ms)
        + (n_layers - 1) * per_layer_compute_ms
    )
    inflation = npu.ttft_ms / ideal_excl012_ms if ideal_excl012_ms > 0 else float("inf")
    npu.per_request_io_detail.append(
        {
            "request_id": npu.current_load.get("request_id"),
            "io_wait_L0_ms": npu.io_wait_L0_ms,
            "io_wait_L1_ms": npu.io_wait_L1_ms,
            "io_wait_L2plus_ms": npu.io_wait_L2plus_ms,
            "io_wait_total_ms": npu.io_wait_L0_ms
            + npu.io_wait_L1_ms
            + npu.io_wait_L2plus_ms,
            "category": npu.category,
            "seq_len_k": npu.seq_len_k,
            "nql": npu.nql,
            "total_input_tokens": npu.total_input_tokens,
            "npu_compute_tokens": npu.npu_compute_tokens,
            "ssu_read_tokens": npu.ssu_read_tokens,
            "ttft_ms": npu.ttft_ms,
            "processing_ttft_ms": npu.processing_ttft_ms,
            "queueing_delay_ms": npu.queueing_delay_ms,
            "arrival_time": npu.arrival_time,
            "request_start_time": npu.request_start_time,
            "instance_id": npu.instance_id,
            "npu_id": npu.npu_id,
            "ideal_excl012_ms": ideal_excl012_ms,
            "infl_ex012": inflation,
            "per_layer_io_waits": dict(npu.current_request_io_waits),
            "request_compute_ms": n_layers * per_layer_compute_ms,
            "total_compute_ms": npu.total_compute_ms,
            "n_blocks_per_layer": npu.n_blocks_per_layer,
            "per_layer_kv_gb": npu.per_layer_kv_gb,
            "per_layer_kv_actual_dur_ms": dict(npu.current_request_kv_actual_dur),
        }
    )
    npu._request_archived = True


def _try_start_compute(context, npu, current_time):
    if npu.done or npu._compute_active:
        return False
    layer = npu.compute_done_up_to + 1
    if layer >= context.n_layers or layer not in npu._kv_ready_layers:
        return False

    wait_from = npu.request_start_time if layer == 0 else npu.last_compute_end_time
    io_wait_ms = max(0.0, current_time - wait_from)
    npu.current_request_io_waits[layer] = io_wait_ms
    npu.npu_idle_ms += io_wait_ms
    if layer == 0:
        npu.io_wait_L0_ms += io_wait_ms
    elif layer == 1:
        npu.io_wait_L1_ms += io_wait_ms
    else:
        npu.io_wait_L2plus_ms += io_wait_ms
        npu.io_wait_layers_2plus_ms += io_wait_ms

    duration_ms = max(0.0, npu.per_layer_us / 1000.0)
    end_time = current_time + duration_ms
    context.compute_tag_counter[npu.npu_id] += 1
    generation = context.compute_tag_counter[npu.npu_id]
    npu._compute_active = True
    npu.compute_end_time = end_time
    npu.total_compute_ms += duration_ms
    if npu.trace_enabled:
        npu.layer_trace.setdefault(layer, {})["compute_start_ms"] = current_time
        npu.layer_trace[layer]["compute_end_ms"] = end_time
    heapq.heappush(
        context.event_heap,
        (end_time, COMPUTE_DONE, npu.npu_id, layer, generation),
    )

    if context.io_mode == "prefetch":
        start_kv_load(
            npu,
            layer + 1,
            context.disk_states,
            context.event_heap,
            current_time,
            context.policy,
            context.n_layers,
            context.disk_bw,
            context.npus,
            context.io_mode,
            context.io_sched,
        )
    return True


def _handle_completed_flow(context, flow, current_time):
    npu = context.npus_map[flow.npu_id]
    layer = flow.layer
    layer_flows = npu.active_block_flows.get(layer)
    if layer_flows is not None:
        layer_flows.discard(flow)
    npu.pending_blocks[layer] = max(
        0,
        npu.pending_blocks.get(layer, 0) - flow.block_count,
    )

    if npu.pending_blocks[layer] > 0:
        return

    npu.batch_states.pop(layer, None)
    npu._kv_ready_layers.add(layer)
    load_start = npu.per_layer_kv_load_start.get(layer, current_time)
    npu.current_request_kv_actual_dur[layer] = current_time - load_start
    if layer == 0:
        npu.layer0_kv_end_time = current_time
    elif layer == 1:
        npu.layer1_kv_end_time = current_time
    while npu.kv_loaded_up_to + 1 in npu._kv_ready_layers:
        npu.kv_loaded_up_to += 1
    if npu.trace_enabled:
        npu.layer_trace.setdefault(layer, {})["kv_end_ms"] = current_time
    _try_start_compute(context, npu, current_time)


def _kick_ready_computes(context, current_time):
    for npu in context.npus:
        if npu.started and not npu.done:
            _try_start_compute(context, npu, current_time)


def _dispatch_waiting_requests(context, current_time):
    context.scheduler.try_dispatch_pending(
        current_time,
        context.npus_map,
        context.instance_sync,
        context.event_heap,
        context.disk_states,
        context.compute_tag_counter,
        True,
        context.request_loads_map,
        context.rng,
        context.n_layers,
        context.placement_mode,
        context.disk_bw,
        context.npus,
        context.policy,
        context.io_mode,
        context.io_sched,
    )
    _kick_ready_computes(context, current_time)


def _handle_compute_done(context, npu_id, layer, generation, current_time):
    npu = context.npus_map[npu_id]
    if (
        generation != context.compute_tag_counter[npu_id]
        or not npu._compute_active
        or layer != npu.compute_done_up_to + 1
    ):
        context.stale_events += 1
        return

    npu._compute_active = False
    npu.compute_done_up_to = layer
    npu.last_compute_end_time = current_time
    npu.compute_end_time = current_time

    if layer + 1 < context.n_layers:
        if context.io_mode != "prefetch":
            start_kv_load(
                npu,
                layer + 1,
                context.disk_states,
                context.event_heap,
                current_time,
                context.policy,
                context.n_layers,
                context.disk_bw,
                context.npus,
                context.io_mode,
                context.io_sched,
            )
        _try_start_compute(context, npu, current_time)
        return

    npu.done = True
    npu.processing_ttft_ms = current_time - npu.request_start_time
    npu.ttft_ms = current_time - npu.arrival_time
    npu.ttft_list.append(npu.ttft_ms)
    _archive_current_request(npu, context.n_layers)
    context.completed_requests += 1

    context.scheduler.on_npu_request_complete(
        npu.instance_id,
        npu.npu_id,
        current_time,
        context.npus_map,
        context.instance_sync,
        context.event_heap,
        context.disk_states,
        context.compute_tag_counter,
        True,
        context.request_loads_map,
        context.rng,
        context.n_layers,
        context.placement_mode,
        context.disk_bw,
        context.npus,
        context.policy,
        context.io_mode,
        context.io_sched,
    )
    _kick_ready_computes(context, current_time)


def _make_request_loads(
    bw_table,
    rng,
    total_requests,
    load_profile,
    total_bw_cap,
    ls_ratio,
    arrival_interval_ms,
):
    loads = generate_npu_loads(
        bw_table,
        rng,
        load_profile=load_profile,
        total_bw_cap=total_bw_cap,
        ls_ratio=ls_ratio,
        num_npu=total_requests,
    )
    for request_id, load in enumerate(loads):
        load["request_id"] = request_id
        load["arrival_time"] = request_id * arrival_interval_ms
    return loads


def _validate_instance_config(instance_config, num_npu):
    configured_npus = [
        npu_id for npu_ids in instance_config.values() for npu_id in npu_ids
    ]
    if sorted(configured_npus) != list(range(num_npu)):
        raise ValueError(
            "instance_config must contain every NPU id exactly once: "
            f"expected 0..{num_npu - 1}"
        )


def _build_simulation_summary(context, current_time, total_requests, events_processed):
    request_metrics = [
        detail for npu in context.npus for detail in npu.per_request_io_detail
    ]
    ttfts = [ttft for npu in context.npus for ttft in npu.ttft_list]
    processing_ttfts = [item["processing_ttft_ms"] for item in request_metrics]
    disk_stats = []
    for disk_state in context.disk_states:
        scheduler = disk_state.queue_scheduler
        scheduler.settle(current_time)
        elapsed = disk_state.busy_time + disk_state.idle_time
        queue_stats = {
            queue_id: {
                "default_bw_gbps": queue.cir,
                "group_id": queue.group_id,
                "max_observed_bw_gbps": scheduler.max_observed_queue_bw[queue_id],
                "bytes_served_gb": queue.bytes_served,
                "activations": queue.n_activations,
                "max_active_flows": queue.max_depth,
                "max_active_flows_observed": queue.max_active_flows_observed,
            }
            for queue_id, queue in scheduler.queues.items()
        }
        disk_stats.append(
            {
                "disk_id": disk_state.disk_id,
                "busy_time_ms": disk_state.busy_time,
                "idle_time_ms": disk_state.idle_time,
                "utilization": disk_state.busy_time / elapsed if elapsed > 0 else 0.0,
                "unused_bw_fraction": (
                    disk_state.surplus_bw_integral / disk_state.total_bw_integral
                    if disk_state.total_bw_integral > 0
                    else 0.0
                ),
                "flows_enqueued": scheduler.n_flows_enqueued,
                "blocks_enqueued": scheduler.n_blocks_enqueued,
                "outstanding_blocks": scheduler.outstanding_blocks,
                "max_outstanding_blocks": scheduler.max_outstanding_blocks,
                "bandwidth_updates": scheduler.n_bandwidth_updates,
                "token_events_scheduled": scheduler.n_token_events,
                "rebalance_events_scheduled": scheduler.n_rebalance_events,
                "configured_queue_bandwidth_gbps": (
                    scheduler.configured_queue_bandwidth
                ),
                "queue_count": len(scheduler.queues),
                "qos_layout": scheduler.qos_layout,
                "group_queue_counts": scheduler.group_queue_counts,
                "group_cir_gbps": scheduler.group_cir_gbps,
                "group_weights": scheduler.group_weights,
                "tier_queue_counts": scheduler.tier_queue_counts,
                "tier_cir_gbps": scheduler.tier_cir_gbps,
                "queues": queue_stats,
            }
        )

    npu_utilizations = [
        npu.total_compute_ms / npu.compute_end_time
        for npu in context.npus
        if npu.compute_end_time > 0
    ]
    qos_scheduler = context.disk_states[0].queue_scheduler
    return {
        "policy": context.policy,
        "io_dispatch_mode": context.io_sched.io_dispatch_mode,
        "batch_interval_mode": context.io_sched.batch_interval_mode,
        "batch_dispatch_interval_us": context.io_sched.batch_dispatch_interval_us,
        "batch_dispatch_headroom": context.io_sched.batch_dispatch_headroom,
        "batch_min_dispatch_interval_us": (
            context.io_sched.batch_min_dispatch_interval_us
        ),
        "batch_max_dispatch_interval_us": (
            context.io_sched.batch_max_dispatch_interval_us
        ),
        "fixed_qos_queue_bandwidth": False,
        "qos_cir_guaranteed": context.policy not in ("fair", "demand_driven"),
        "qos_surplus_borrowing": context.policy not in ("fair", "demand_driven"),
        "qos_queue_pir_uncapped": context.policy not in ("fair", "demand_driven"),
        "qos_layout": context.io_sched.qos_layout,
        "qos_queue_defaults_gbps": qos_scheduler.group_cir_gbps,
        "qos_queue_count": context.io_sched.qos_queue_count,
        "qos_queue_max_active_flows": (context.io_sched.qos_queue_max_active_flows),
        "qos_group_queue_counts": qos_scheduler.group_queue_counts,
        "qos_group_cir_gbps": qos_scheduler.group_cir_gbps,
        "qos_group_weights": qos_scheduler.group_weights,
        "qos_tier_queue_counts": qos_scheduler.tier_queue_counts,
        "qos_tier_cir_gbps": qos_scheduler.tier_cir_gbps,
        "total_requests": total_requests,
        "completed_requests": context.completed_requests,
        "makespan_ms": current_time,
        "events_processed": events_processed,
        "stale_events": context.stale_events,
        "event_counts": dict(context.event_counts),
        "avg_ttft_ms": float(np.mean(ttfts)) if ttfts else 0.0,
        "avg_processing_ttft_ms": (
            float(np.mean(processing_ttfts)) if processing_ttfts else 0.0
        ),
        "avg_queueing_delay_ms": (
            float(np.mean([item["queueing_delay_ms"] for item in request_metrics]))
            if request_metrics
            else 0.0
        ),
        "avg_npu_utilization": (
            float(np.mean(npu_utilizations)) if npu_utilizations else 0.0
        ),
        "throughput_requests_per_s": (
            context.completed_requests / current_time * 1000.0
            if current_time > 0
            else 0.0
        ),
        "batches_dispatched": sum(npu._batches_dispatched for npu in context.npus),
        "request_metrics": request_metrics,
        "disk_stats": disk_stats,
    }


def simulate_continuous(
    bw_table,
    policy="queue_wrr",
    num_requests_per_npu=1,
    num_npu=NUM_NPU,
    num_disk=NUM_DISK,
    n_layers=SIM_N_LAYERS,
    ls_ratio=None,
    rng=None,
    io_sched=None,
    placement_mode="random",
    disk_bw=DISK_BW,
    io_mode="prefetch",
    load_profile="mixed",
    total_bw_cap=None,
    instance_config=None,
    l1_strategy="round_robin",
    l2_strategy="round_robin",
    arrival_interval_ms=0.0,
    max_events=10_000_000,
    trace=False,
):
    """Run the complete event-driven simulation.

    QoS policies guarantee each active logical queue its default CIR. Idle
    bandwidth is borrowed by active queues according to WRR weights, with
    uncapped queue PIR and the physical disk bandwidth as the global limit.
    """
    if not bw_table:
        raise ValueError("bw_table must not be empty")
    if num_requests_per_npu <= 0 or num_npu <= 0 or num_disk <= 0 or n_layers <= 0:
        raise ValueError("request, NPU, disk, and layer counts must be positive")
    if disk_bw <= 0:
        raise ValueError("disk_bw must be positive")
    if arrival_interval_ms < 0:
        raise ValueError("arrival_interval_ms must be non-negative")
    if policy not in ("fair", "demand_driven", "queue_wrr", "urgency_driven"):
        raise ValueError(f"unknown policy: {policy}")
    if io_mode not in ("prefetch", "sequential"):
        raise ValueError(f"unknown io_mode: {io_mode}")

    rng = rng or np.random.RandomState(42)
    io_sched = io_sched or IOSchedulingConfig()
    if io_sched.io_dispatch_mode not in (
        "all_at_once",
        "batched",
        "traffic_aware_batched",
    ):
        raise ValueError(f"unknown io_dispatch_mode: {io_sched.io_dispatch_mode}")
    if io_sched.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if io_sched.batch_interval_mode not in ("fixed", "demand_aware"):
        raise ValueError(f"unknown batch_interval_mode: {io_sched.batch_interval_mode}")
    if (
        not np.isfinite(io_sched.batch_dispatch_interval_us)
        or io_sched.batch_dispatch_interval_us <= 0
    ):
        raise ValueError("batch_dispatch_interval_us must be finite and positive")
    if (
        not np.isfinite(io_sched.batch_dispatch_headroom)
        or io_sched.batch_dispatch_headroom <= 0
    ):
        raise ValueError("batch_dispatch_headroom must be finite and positive")
    if (
        not np.isfinite(io_sched.batch_min_dispatch_interval_us)
        or io_sched.batch_min_dispatch_interval_us < 0
    ):
        raise ValueError(
            "batch_min_dispatch_interval_us must be finite and non-negative"
        )
    if (
        np.isnan(io_sched.batch_max_dispatch_interval_us)
        or io_sched.batch_max_dispatch_interval_us <= 0
        or io_sched.batch_max_dispatch_interval_us
        < io_sched.batch_min_dispatch_interval_us
    ):
        raise ValueError(
            "batch_max_dispatch_interval_us must be positive and no smaller "
            "than batch_min_dispatch_interval_us"
        )
    if (
        not isinstance(io_sched.qos_queue_max_active_flows, int)
        or isinstance(io_sched.qos_queue_max_active_flows, bool)
        or io_sched.qos_queue_max_active_flows <= 0
    ):
        raise ValueError("qos_queue_max_active_flows must be a positive integer")
    if io_sched.qos_layout == QOS_LAYOUT_THREE_TIER:
        qos_tier_queue_counts(io_sched.qos_queue_count)
    elif io_sched.qos_layout == QOS_LAYOUT_EIGHT_GROUP:
        if (
            io_sched.qos_queue_count < EIGHT_GROUP_COUNT
            or io_sched.qos_queue_count % EIGHT_GROUP_COUNT != 0
        ):
            raise ValueError(
                "eight_group qos_queue_count must be a positive multiple of "
                f"{EIGHT_GROUP_COUNT}: {io_sched.qos_queue_count}"
            )
    else:
        raise ValueError(f"unknown qos_layout: {io_sched.qos_layout}")
    if io_sched.token_bucket_enabled and io_sched.token_bucket_refill_us <= 0:
        raise ValueError("token_bucket_refill_us must be positive")

    total_requested = num_npu * num_requests_per_npu
    request_loads = _make_request_loads(
        bw_table,
        rng,
        total_requested,
        load_profile,
        total_bw_cap,
        ls_ratio,
        arrival_interval_ms,
    )
    if not request_loads:
        raise ValueError("load generation produced no requests")
    total_requests = len(request_loads)
    request_loads_map = {load["request_id"]: load for load in request_loads}

    placeholder_placement = {layer: [] for layer in range(n_layers)}
    npus = []
    for npu_id in range(num_npu):
        placeholder_load = dict(request_loads[npu_id % total_requests])
        placeholder_load["npu_id"] = npu_id
        npu = NPUState(placeholder_load, placeholder_placement, n_layers)
        npu.request_start_time = None
        npu.trace_enabled = trace
        npus.append(npu)

    if instance_config is None:
        instance_config = {0: list(range(num_npu))}
    _validate_instance_config(instance_config, num_npu)
    scheduler = GlobalScheduler(
        instance_config,
        l1_strategy=l1_strategy,
        l2_strategy=l2_strategy,
    )

    disk_states = [DiskState(disk_id, disk_bw) for disk_id in range(num_disk)]
    for disk_state in disk_states:
        DiskIOScheduler(disk_state, policy, disk_bw, io_sched)

    event_heap = []
    for request in request_loads:
        heapq.heappush(
            event_heap,
            (
                request["arrival_time"],
                REQUEST_ARRIVAL,
                request["request_id"],
                0,
                0,
            ),
        )

    context = _SimulationContext(
        scheduler,
        event_heap,
        disk_states,
        npus,
        request_loads_map,
        rng,
        n_layers,
        placement_mode,
        disk_bw,
        policy,
        io_mode,
        io_sched,
    )

    current_time = 0.0
    events_processed = 0
    while context.completed_requests < total_requests:
        if not event_heap:
            active_flows = sum(len(state.active_flows) for state in disk_states)
            raise RuntimeError(
                "simulation deadlocked before all requests completed: "
                f"completed={context.completed_requests}/{total_requests}, "
                f"active_flows={active_flows}, "
                f"pending_requests={len(scheduler._global_pending_queue)}"
            )
        if events_processed >= max_events:
            raise RuntimeError(
                f"max_events={max_events} reached at t={current_time:.6f} ms"
            )

        event_time, event_type, resource_id, value, generation = heapq.heappop(
            event_heap
        )
        if event_time + _SIM_EPS < current_time:
            raise RuntimeError("event time moved backwards")
        current_time = max(current_time, event_time)
        events_processed += 1
        context.event_counts[event_type] += 1

        if event_type == REQUEST_ARRIVAL:
            scheduler._global_pending_queue.append(dict(request_loads_map[resource_id]))
            _dispatch_waiting_requests(context, current_time)
        elif event_type == DISK_COMPLETION:
            disk_state = disk_states[resource_id]
            if generation != disk_state.generation:
                context.stale_events += 1
                continue
            completed_flows = disk_state.queue_scheduler.complete_ready_flows(
                current_time
            )
            for flow in completed_flows:
                _handle_completed_flow(context, flow, current_time)
            disk_state.queue_scheduler.request_redistribution(current_time, event_heap)
        elif event_type == BATCH_DISPATCH:
            npu = context.npus_map[resource_id]
            batch_state = npu.batch_states.get(value)
            if batch_state is None or generation != batch_state.current_batch_idx:
                context.stale_events += 1
                continue
            if not _dispatch_next_batch(
                npu,
                value,
                context.disk_states,
                context.event_heap,
                current_time,
                context.policy,
                context.io_sched,
            ):
                context.stale_events += 1
        elif event_type == TOKEN_REFILL:
            disk_state = disk_states[resource_id]
            if generation != disk_state.generation:
                context.stale_events += 1
                continue
            disk_state.queue_scheduler.request_redistribution(current_time, event_heap)
        elif event_type == DISK_REBALANCE:
            scheduler = disk_states[resource_id].queue_scheduler
            if (
                generation != scheduler.rebalance_generation
                or scheduler.pending_rebalance_time is None
                or abs(scheduler.pending_rebalance_time - current_time) > _SIM_EPS
            ):
                context.stale_events += 1
                continue
            scheduler.pending_rebalance_time = None
            scheduler.redistribute(current_time, event_heap)
        elif event_type == COMPUTE_DONE:
            _handle_compute_done(context, resource_id, value, generation, current_time)
        else:
            raise RuntimeError(f"unhandled event type: {event_type}")

    summary = _build_simulation_summary(
        context, current_time, total_requests, events_processed
    )
    return npus, summary
