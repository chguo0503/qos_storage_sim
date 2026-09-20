"""Small deterministic synthetic traces for examples and debugging."""

import random

from simulator.core import sim
from simulator.core.continuous_batch_sim import ContinuousBatchRequest
from simulator.policies.common import IO_GIB


def build_requests(*, num_npu=32, num_ssu=3, requests_per_npu=4, seed=7):
    """Each card gets both profiles when count >= 2; order is shuffled per card.

    Profile (blocks, compute_ms): (8, 0.15) and (32, 2.0). These are explicitly
    synthetic values, not measurements selected from data. All arrivals are 0.
    """
    if min(num_npu, num_ssu, requests_per_npu) <= 0:
        raise ValueError("input dimensions must be positive")
    rng = random.Random(seed)
    requests = []
    for npu in range(num_npu):
        profiles = [(8, 0.15), (32, 2.0)] * ((requests_per_npu + 1) // 2)
        profiles = profiles[:requests_per_npu]
        rng.shuffle(profiles)
        for blocks, compute_ms in profiles:
            rid = len(requests)
            miss = 256
            seq = (blocks * sim.BLOCK_SIZE + miss) / 1024.0
            load = dict(request_id=rid, npu_id=npu, seq_len_k=seq, nql=miss,
                        category=sim.classify_request(seq, miss),
                        per_layer_us=1000 * compute_ms, per_layer_kv_gb=blocks * IO_GIB,
                        required_bw_input_gbps=1000 * blocks * IO_GIB / compute_ms,
                        source="synthetic_example")
            placement = (tuple((sim.block_ring_hash_disk_id(rid, b, num_ssu), IO_GIB)
                               for b in range(blocks)),)
            requests.append(ContinuousBatchRequest(rid, npu, 0.0, load, placement))
    return tuple(requests)
