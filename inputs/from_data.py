"""Build finite randomized input from exact catalog rows, without interpolation."""

import random

from inputs.authenticated import load_authenticated_bw_table
from simulator.core import sim
from simulator.core.continuous_batch_sim import ContinuousBatchRequest
from simulator.policies.common import IO_GIB


def build_requests(*, num_npu=32, num_ssu=3, requests_per_npu=2, seed=7,
                   keys=((32, 256), (128, 4096))):
    if min(num_npu, num_ssu, requests_per_npu) <= 0 or not keys:
        raise ValueError("positive input dimensions and at least one catalog key are required")
    table, provenance = load_authenticated_bw_table(num_npu)
    for key in keys:
        if key not in table:
            raise ValueError(f"catalog profile is absent: {key}")
        _, _, hit = sim.calculate_token_partition(*key)
        # The shared-path adapter models equal 176-KiB commands. Reject exact
        # half-block profiles here instead of silently padding measured data.
        if hit <= 0 or hit % sim.BLOCK_SIZE or table[key][3] != hit / sim.BLOCK_SIZE * IO_GIB:
            raise ValueError(f"{key} is not an exact 176-KiB-block profile; use a native input runner")
    rng = random.Random(seed)
    requests = []
    for npu in range(num_npu):
        for _ in range(requests_per_npu):
            key = rng.choice(keys)
            rid = len(requests)
            bandwidth, compute_us, _, volume = table[key]
            load = dict(request_id=rid, npu_id=npu, seq_len_k=key[0], nql=key[1],
                        category=sim.classify_request(*key), per_layer_us=compute_us,
                        per_layer_kv_gb=volume, required_bw_input_gbps=bandwidth,
                        source="direct_data_row")
            blocks = int(volume / IO_GIB)
            placement = (tuple((sim.block_ring_hash_disk_id(rid, b, num_ssu), IO_GIB)
                               for b in range(blocks)),)
            requests.append(ContinuousBatchRequest(rid, npu, 0.0, load, placement))
    return tuple(requests), provenance
