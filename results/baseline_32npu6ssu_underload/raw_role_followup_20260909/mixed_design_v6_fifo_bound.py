#!/usr/bin/env python3
"""Recompute the conditional Baseline FIFO bound from frozen input values.

Queue semantics are a separately reviewed code argument, not a formal program
verification.  This script verifies the numerical hypotheses and records source
hashes.  It is not a simulation or an assertion about multi-Path Once arbitration.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from run_baseline_npu32_stress import load_manifest
import mixed_design_v6
import mixed_design_v7


def main():
    path = HERE / "inputs/raw200_three_l20_fixed_seed7.json.gz"
    requests, metadata = load_manifest(path)
    block = 176 * 1024 / 2**30
    role_max = {"long": 0.0, "short": 0.0}
    max_layer_blocks = 0
    max_long_volume = 0.0
    max_role_rate = {"long": 0.0, "short": 0.0}
    long_compute_ms = set()
    short_compute_ms = {}
    for request in requests:
        assert len(request.placement) == 1
        layer = request.placement[0]
        assert all(abs(amount - block) < 1e-14 for _, amount in layer)
        role = request.load["role"]
        per_disk = [0.0] * 6
        for ssu, amount in layer:
            per_disk[ssu] += amount
        c_ms = float(request.load["per_layer_us"]) / 1000
        role_max[role] = max(role_max[role], max(per_disk))
        max_role_rate[role] = max(max_role_rate[role], max(per_disk) / (c_ms / 1000))
        max_layer_blocks = max(max_layer_blocks, len(layer))
        if role == "long":
            long_compute_ms.add(c_ms)
            max_long_volume = max(max_long_volume, sum(per_disk))
        else:
            short_compute_ms[int(request.load["seq_len_k"])] = c_ms
    assert len(long_compute_ms) == 1
    long_c = next(iter(long_compute_ms))
    issue_ms = max_layer_blocks * 0.1 / 1000.0
    outstanding_per_disk = 20 * role_max["long"] + 12 * role_max["short"]
    disk_ms = outstanding_per_disk / 40.0 * 1000
    link_ms = max_long_volume / 50.0 * 1000
    read_bound = issue_ms + disk_ms + link_ms
    assert read_bound < long_c
    guard_end = 8 * 8 * long_c + read_bound
    variants = []
    for name, module in (("v6", mixed_design_v6), ("v7", mixed_design_v7)):
        queues, desc = module.build_queues(requests, metadata, 7, "ordered")
        earliest_next_long = []
        for npu in range(20, 32):
            first_long = next(i for i, r in enumerate(queues[npu]) if r.load["role"] == "long")
            prefix = queues[npu][:first_long]
            pure = sum(8 * float(r.load["per_layer_us"]) / 1000 for r in prefix)
            predecessor_c = float(prefix[-1].load["per_layer_us"]) / 1000
            earliest_next_long.append(pure - predecessor_c)
        earliest = min(earliest_next_long)
        assert guard_end < earliest
        variants.append(dict(variant=name, design_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            earliest_next_Long_L0_ms=earliest, guard_exit_upper_ms=guard_end,
            margin_before_first_Long_L0_ms=earliest - guard_end,
            inequality_closes_role_assumption=True))
    result = dict(scope="Baseline ordered only: fixed batch1, one-layer lookahead, all disk commands Path0 FIFO, no extra delay/control",
        source_manifest=str(path), source_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ("sim.py", "continuous_batch_sim.py", "shared_path_sim_adapter.py", "shared_path_baseline.py")},
        block_gib=block, max_long_blocks_per_disk=round(role_max["long"] / block),
        max_short_blocks_per_disk=round(role_max["short"] / block), max_layer_blocks=max_layer_blocks,
        client_issue_upper_ms=issue_ms, outstanding_layer_work_per_disk_gib=outstanding_per_disk,
        disk_residual_upper_ms=disk_ms, own_layer_receive_upper_ms=link_ms,
        layer_release_to_ready_upper_ms=read_bound, long_layer_compute_ms=long_c,
        positive_hiding_margin_ms=long_c - read_bound,
        variants=variants, full_run_current_Long_count_upper=29,
        full_run_current_nominal_disk_rate_upper_gib_s=29 * max_role_rate["long"] + 3 * max_role_rate["short"],
        read_bound_not_used_as_Once_bound=True,
        caveat="The numerical implications depend on reviewed FIFO/one-layer scheduler semantics. No simulated U, mixed-role duration, or Once capacity is inferred.")
    (HERE / "mixed_design_v6_fifo_bound.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
