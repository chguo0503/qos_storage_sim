"""ASU baseline: all NPU I/O shares each SSD's Path0.

The original static table is retained; only Path0 receives ASU traffic.
"""

def baseline_path_ids(io_count, path_id=0):
    return (path_id,) * io_count


def qos_config():
    from simulator.config import static_qos_config
    return static_qos_config()


def configuration(num_npu):
    return {
        "canonical_strategy": "asu_baseline",
        "usable_path_count_per_ssu": 1,
        "npu_path_ids": [0] * num_npu,
        "exclusive_per_npu": False,
        "per_npu_cir_gib_s": None,
        "path_pir": "unlimited",
        "idle_capacity_borrowing": True,
        "surplus_policy": "native equal-weight active-group WRR, then active-path WRR",
        "unused_od_path_cir_and_weight": None,
        "cross_request_layer0_uses_execution_npu_path": False,
        "runtime_cir_updates": False,
        "within_path_order": "FIFO, nonpreemptive I/O",
    }


def main():
    import math
    from simulator.policies.profiles import FINAL_STATIC
    assert baseline_path_ids(8) == (0,) * 8
    assert baseline_path_ids(0) == ()
    config = qos_config()
    assert config == FINAL_STATIC.hardware_config()
    assert len(config.path_cirs) == 256
    assert all(math.isinf(v) for v in config.path_pirs)
    assert configuration(32)["npu_path_ids"] == [0] * 32
    print("asu_baseline: PASS (all NPU I/O uses Path0; static table preserved)")


if __name__ == "__main__":
    main()
