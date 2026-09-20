"""Original Once routing engine, supplied with an externally sampled snapshot.

One call plans a request-layer-SSU's I/Os. The experiment adapter determines
when the shared snapshot is collected (5 ms or slower); this function cannot
refresh it. It preserves the original category-legal candidate pool and static
CIR/PIR/rate estimate, and projects each newly planned I/O into a local shadow.
It does NOT choose one single Path for the whole layer or use a live SSD FIFO.
"""

from simulator.policies.common import IO_GIB


def once_path_ids(io_count, snapshot, allowed_path_ids, qos, start_offset=0,
                  disk_bw_gib_s=40.0):
    # Reuse, rather than subtly reimplement, the project's established pure
    # routing engine. That module currently imports the project's NumPy-based
    # allocation helper; this thin wrapper otherwise has no simulator state.
    from simulator.policies.policy_logic import layer_once_path_ids
    return layer_once_path_ids((IO_GIB,) * io_count, snapshot, allowed_path_ids,
                               qos, disk_bw_gbps=disk_bw_gib_s,
                               start_offset=start_offset)


def main():
    from simulator.policies.policy_logic import category_path_ids
    from simulator.policies.common import pressure_from_counts
    from simulator.policies.profiles import FINAL_STATIC
    config = FINAL_STATIC.hardware_config()
    snapshot = pressure_from_counts((0,) * 256, config)
    allowed = category_path_ids("SS", config)
    planned = once_path_ids(8, snapshot, allowed, config)
    assert len(planned) == 8 and set(planned).issubset(allowed)
    assert planned == once_path_ids(8, snapshot, allowed, config)
    assert snapshot.counts == (0,) * 256
    assert once_path_ids(0, snapshot, allowed, config) == ()
    congested = pressure_from_counts((1000,) + (0,) * 255, config)
    assert once_path_ids(1, congested, (0, 32), config) == (32,)
    print("once: PASS (legal deterministic paths, immutable snapshot, congestion avoidance)")


if __name__ == "__main__":
    main()
