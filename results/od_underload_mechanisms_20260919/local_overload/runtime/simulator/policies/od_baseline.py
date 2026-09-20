"""OD baseline: one exclusive path per execution NPU on every SSD.

CIR is capacity/N; PIR is unlimited. Actual bandwidth may borrow idle
capacity through the native group/path arbitration.
"""

import math


def queue_depth_per_npu(total_depth_per_ssu, num_npu):
    """Split command slots permanently; idle NPUs cannot lend queue entries."""
    od_npu_path_ids(num_npu)
    if total_depth_per_ssu is None:
        return None
    if (isinstance(total_depth_per_ssu, bool)
            or not isinstance(total_depth_per_ssu, int)
            or total_depth_per_ssu <= 0
            or total_depth_per_ssu % num_npu):
        raise ValueError("OD per-SSD queue depth must be a positive integer divisible by num_npu")
    return total_depth_per_ssu // num_npu


def od_npu_path_ids(num_npu):
    """Spread the exclusive paths evenly across the eight hardware groups."""
    if not isinstance(num_npu, int) or isinstance(num_npu, bool) or not 1 <= num_npu <= 256:
        raise ValueError("OD baseline requires 1 <= num_npu <= 256")
    return tuple((npu % 8) * 32 + npu // 8 for npu in range(num_npu))


def path_ids(io_count, npu_id, num_npu):
    """Route by the execution NPU, including next-request layer-zero prefetch."""
    return (od_npu_path_ids(num_npu)[npu_id],) * io_count


def qos_config(num_npu, disk_bw_gib_s=40.0):
    from simulator.core.sim import StaticQoSConfig
    if not math.isfinite(disk_bw_gib_s) or disk_bw_gib_s <= 0:
        raise ValueError("disk bandwidth must be finite and positive")
    paths = set(od_npu_path_ids(num_npu))
    return StaticQoSConfig(
        path_cirs=tuple(disk_bw_gib_s / num_npu if p in paths else 0.0 for p in range(256)),
        path_pirs=(float("inf"),) * 256,
        path_weights=tuple(1.0 if p in paths else 0.0 for p in range(256)),
        group_weights=(1.0,) * 8,
        category_paths_per_group=(12, 4, 12, 4),
    )


def configuration(num_npu, disk_bw_gib_s=40.0):
    return {
        "canonical_strategy": "od_baseline",
        "usable_path_count_per_ssu": num_npu,
        "npu_path_ids": list(od_npu_path_ids(num_npu)),
        "exclusive_per_npu": True,
        "per_npu_cir_gib_s": disk_bw_gib_s / num_npu,
        "path_pir": "unlimited",
        "idle_capacity_borrowing": True,
        "surplus_policy": "native equal-weight active-group WRR, then active-path WRR",
        "unused_od_path_cir_and_weight": 0,
        "cross_request_layer0_uses_execution_npu_path": True,
        "runtime_cir_updates": False,
        "within_path_order": "FIFO, nonpreemptive I/O",
    }


def main():
    from collections import Counter
    paths = od_npu_path_ids(32)
    config = qos_config(32)
    assert len(set(paths)) == 32
    assert Counter(p // 32 for p in paths) == Counter({g: 4 for g in range(8)})
    assert sum(config.path_cirs) == 40.0
    assert all(config.path_cirs[p] == 1.25 for p in paths)
    assert all(config.path_cirs[p] == config.path_weights[p] == 0
               for p in set(range(256)) - set(paths))
    assert all(math.isinf(v) for v in config.path_pirs)
    assert all(path_ids(2, n, 32) == (paths[n], paths[n]) for n in range(32))
    for count in (1, 7, 33, 256):
        assert len(set(od_npu_path_ids(count))) == count
    print("od_baseline: PASS (32 exclusive paths, CIR=1.25 GiB/s, unlimited PIR)")


if __name__ == "__main__":
    main()
