"""Baseline / Once integration used by the two retained result studies."""

import pytest
import sim
from continuous_batch_sim import ContinuousBatchRequest, simulate_continuous_batch
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from shared_path_sim_adapter import shared_path_adapter, IO_GIB


def example_trace():
    requests = []
    for npu in range(4):
        for generation in range(3):
            rid = npu * 100 + generation
            load = {"request_id": rid, "npu_id": npu, "seq_len_k": 32,
                    "nql": 128, "category": "SS", "per_layer_us": 2000.0,
                    "per_layer_kv_gb": 32 * IO_GIB, "required_bw_input_gbps": 1.0}
            placement = (tuple((i % 2, IO_GIB) for i in range(32)),)
            requests.append(ContinuousBatchRequest(rid, npu, generation * 5., load, placement))
    return tuple(requests)


def run(policy, trace=None):
    trace = example_trace() if trace is None else trace
    original_ids = tuple(r.npu_id for r in trace)
    with shared_path_adapter(policy) as adapter:
        summary = simulate_continuous_batch(trace, num_npu=4, num_ssu=2,
            n_layers=8, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=static_qos_config(), cross_request_layer0_prefetch=True,
            client_io_config=routing_strategy_specs()[1].client_config(),
            pressure_ttl_ms=5, submit_order_seed=17)
    assert tuple(r.npu_id for r in trace) == original_ids
    return summary, adapter.statistics(), adapter.assignment_log


@pytest.mark.parametrize("policy", ("baseline", "once"))
def test_baseline_and_once_obey_clock_and_conserve_io(policy):
    summary, stats, assignments = run(policy)
    assert all(summary["invariants"].values())
    assert summary["request_count"] == 12
    assert stats["reserved_blocks"] == stats["acknowledged_blocks"] == 12 * 8 * 32
    assert stats["ledger_end_counts_by_ssu"] == [0, 0]
    assert stats["min_ledger_count"] == 0
    times = stats["collector_times_ms"]
    assert times == [5.0 * i for i in range(len(times))]
    assert stats["fresh_reads_by_ssu"] == [len(times)] * 2
    assert stats["native_fresh_reads_by_ssu"] == [len(times)] * 2
    assert stats["max_snapshot_age_ms"] <= 5 + 1e-9
    assert stats["cir_write_events"] == []
    assert len(assignments) == 0
    assert stats["reorder_calls"] == 0


def test_baseline_periodic_observation_does_not_change_native_data_plane():
    expected, _, _ = run("baseline")
    actual = simulate_continuous_batch(example_trace(), num_npu=4, num_ssu=2,
        n_layers=8, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
        qos_config=static_qos_config(), cross_request_layer0_prefetch=True,
        client_io_config=routing_strategy_specs()[0].client_config(),
        pressure_ttl_ms=0, submit_order_seed=17)
    assert expected["makespan_ms"] == pytest.approx(actual["makespan_ms"], abs=1e-9)
    for left, right in zip(expected["request_metrics"], actual["request_metrics"]):
        assert left["completion_time_ms"] == pytest.approx(right["completion_time_ms"], abs=1e-9)
