"""Compatibility facade for the independent ASU and OD baseline modules.

Legacy ``baseline`` retains ASU semantics. Other policies keep the original
static register table through ``baseline_qos_config``.
"""

from simulator.policies import asu_baseline, od_baseline
from simulator.policies.asu_baseline import baseline_path_ids
from simulator.policies.od_baseline import od_npu_path_ids

BASELINE_STRATEGIES = ("asu_baseline", "od_baseline")


def canonical_baseline_name(strategy):
    return "asu_baseline" if strategy == "baseline" else strategy


def baseline_qos_config(strategy, num_npu, disk_bw_gib_s=40.0):
    if canonical_baseline_name(strategy) == "od_baseline":
        return od_baseline.qos_config(num_npu, disk_bw_gib_s)
    return asu_baseline.qos_config()


def baseline_configuration(strategy, num_npu, disk_bw_gib_s=40.0):
    strategy = canonical_baseline_name(strategy)
    if strategy == "asu_baseline":
        return asu_baseline.configuration(num_npu)
    if strategy == "od_baseline":
        return od_baseline.configuration(num_npu, disk_bw_gib_s)
    return None


def main():
    asu_baseline.main()
    od_baseline.main()
    assert baseline_configuration("baseline", 32) == baseline_configuration("asu_baseline", 32)
    assert baseline_qos_config("once", 32) == baseline_qos_config("baseline", 32)
    assert baseline_configuration("once", 32) is None
    print("baselines: PASS (legacy alias and facade compatibility)")


if __name__ == "__main__":
    main()
