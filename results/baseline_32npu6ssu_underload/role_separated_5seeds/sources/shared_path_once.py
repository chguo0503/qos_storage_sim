"""Original Once routing engine, supplied with an externally sampled snapshot.

One call plans a request-layer-SSU's I/Os. The experiment adapter determines
when the shared snapshot is collected (5 ms or slower); this function cannot
refresh it. It preserves the original category-legal candidate pool and static
CIR/PIR/rate estimate, and projects each newly planned I/O into a local shadow.
It does NOT choose one single Path for the whole layer or use a live SSD FIFO.
"""

from shared_path_common import IO_GIB


def once_path_ids(io_count, snapshot, allowed_path_ids, qos, start_offset=0,
                  disk_bw_gib_s=40.0):
    # Reuse, rather than subtly reimplement, the project's established pure
    # routing engine. That module currently imports the project's NumPy-based
    # allocation helper; this thin wrapper otherwise has no simulator state.
    from policy_logic import layer_once_path_ids
    return layer_once_path_ids((IO_GIB,) * io_count, snapshot, allowed_path_ids,
                               qos, disk_bw_gbps=disk_bw_gib_s,
                               start_offset=start_offset)


if __name__ == "__main__":
    from policy_logic import category_path_ids
    from shared_path_common import pressure_from_counts
    from strategy_profiles import FINAL_STATIC
    config = FINAL_STATIC.hardware_config()
    snapshot = pressure_from_counts((0,) * 256, config)
    print(once_path_ids(8, snapshot, category_path_ids("SS", config), config))
