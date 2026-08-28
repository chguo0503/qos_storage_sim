from collections import defaultdict

import sim
from closed_loop_scheme_b import allocate_coflow_grants
from closed_loop_workload import FORMAL_WINDOWS, prepare_closed_loop_workload
from continuous_batch_sim import (
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import legacy_qos_config, legacy_strategy_specs


def _table():
    return sim.load_bw_table_cache(num_npu=128)


def test_closed_loop_assignment_is_balanced_and_independent_of_ssu_count():
    first = prepare_closed_loop_workload(
        _table(), num_npu=4, requests_per_npu=8, n_layers=2, num_ssu=8
    )
    repeated = prepare_closed_loop_workload(
        _table(), num_npu=4, requests_per_npu=8, n_layers=2, num_ssu=8
    )
    wider = prepare_closed_loop_workload(
        _table(), num_npu=4, requests_per_npu=8, n_layers=2, num_ssu=16
    )
    assert first.trace_hash == repeated.trace_hash
    assert first.statistics["assignment_hash"] == wider.statistics["assignment_hash"]
    assert first.placement_hash != wider.placement_hash
    by_npu = defaultdict(list)
    for request in first.requests:
        by_npu[request.npu_id].append(request)
    assert all(
        sorted(request.category for request in requests)
        == sorted(sim.WORKLOAD_CATEGORIES * 2)
        for requests in by_npu.values()
    )


def test_batch1_fifo_admits_next_request_at_previous_completion():
    workload = prepare_closed_loop_workload(
        _table(), num_npu=2, requests_per_npu=3, n_layers=2, num_ssu=2
    )
    baseline = next(
        spec for spec in legacy_strategy_specs() if spec.name == "baseline"
    )
    summary = simulate_continuous_batch(
        requests_from_continuous_prefill_workload(workload),
        num_npu=2,
        num_ssu=2,
        n_layers=2,
        batch_size=1,
        qos_config=legacy_qos_config(),
        client_io_config=baseline.client_config(),
    )
    assert all(summary["invariants"].values())
    by_npu = defaultdict(list)
    for row in summary["request_metrics"]:
        by_npu[row["npu_id"]].append(row)
    for rows in by_npu.values():
        rows.sort(key=lambda row: row["request_id"])
        assert all(
            current["admission_time_ms"] == previous["completion_time_ms"]
            for previous, current in zip(rows, rows[1:])
        )
        assert all(
            row["processing_latency_ms"]
            == row["completion_time_ms"] - row["admission_time_ms"]
            for row in rows
        )


def test_formal_windows_have_equal_profile_multisets_per_npu():
    workload = prepare_closed_loop_workload(_table(), num_ssu=28)
    by_npu = defaultdict(list)
    for request in workload.requests:
        by_npu[request.npu_id].append(request)
    for requests in by_npu.values():
        requests.sort(key=lambda request: request.stream_id)
    reference = by_npu[0]
    for _, start, count in FORMAL_WINDOWS:
        expected = sorted(request.profile_key for request in reference[start : start + count])
        expected_categories = sorted(
            request.category for request in reference[start : start + count]
        )
        for requests in by_npu.values():
            selected = requests[start : start + count]
            assert sorted(request.profile_key for request in selected) == expected
            assert sorted(request.category for request in selected) == expected_categories


def test_coflow_grants_preserve_each_npu_work_shape_and_capacity():
    grants = allocate_coflow_grants(
        ((10.0, 30.0), (30.0, 10.0)),
        (1000.0, 1000.0),
    )
    assert grants == ((10.0, 30.0), (30.0, 10.0))
    assert all(sum(grants[npu][ssu] for npu in range(2)) <= 40.0 for ssu in range(2))
    assert all(sum(row) <= 50.0 for row in grants)

    constrained = allocate_coflow_grants(
        ((10.0,), (30.0,)),
        (1000.0, 1000.0),
        ssd_cap=20.0,
    )
    assert constrained == ((5.0,), (15.0,))
