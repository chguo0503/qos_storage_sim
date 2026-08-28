"""Black-box contract tests for full-prefill microbatch progression.

These tests deliberately exercise only the public request/simulation API and
the JSON-safe trace in ``microbatch_metrics``.  The SSD40 command server and
the per-NPU NPU50 receive link remain the production data plane.
"""

from __future__ import annotations

import pytest

import sim
from analyze_continuous_prefill import _paired_invariants
from continuous_batch_sim import (
    CausalLayerControlConfig,
    CausalLayerObservation,
    CausalLayerSnapshot,
    CausalMaxMinSchemeBController,
    CIRControlDecision,
    ContinuousBatchRequest,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    legacy_qos_config,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id


def _request(
    request_id: int,
    *,
    npu_id: int = 0,
    arrival_ms: float = 0.0,
    per_layer_compute_ms: float = 1.0,
    placement=None,
    n_layers: int = 1,
) -> ContinuousBatchRequest:
    if placement is None:
        placement = tuple((((0, 0.001),)) for _ in range(n_layers))
    return ContinuousBatchRequest(
        request_id=request_id,
        npu_id=npu_id,
        arrival_time_ms=arrival_ms,
        load={
            "request_id": request_id,
            "npu_id": npu_id,
            "category": "SS",
            "per_layer_us": per_layer_compute_ms * 1_000.0,
            "initial": arrival_ms == 0.0,
        },
        placement=tuple(placement),
    )


def _run(requests, *, n_layers: int, batch_size: int, num_ssu: int = 1):
    summary = simulate_continuous_batch(
        tuple(requests),
        num_npu=1,
        num_ssu=num_ssu,
        n_layers=n_layers,
        batch_size=batch_size,
        policy=sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
    )
    assert all(summary["invariants"].values())
    return summary


def _batches(summary):
    return sorted(summary["microbatch_metrics"], key=lambda row: row["batch_id"])


def _requests(summary):
    return {
        row["request_id"]: row
        for row in summary["request_metrics"]
    }


def test_eight_simultaneous_requests_form_one_fixed_microbatch():
    summary = _run(
        (_request(request_id, n_layers=2) for request_id in range(8)),
        n_layers=2,
        batch_size=8,
    )

    batches = _batches(summary)
    assert len(batches) == 1
    batch = batches[0]
    assert tuple(batch["member_request_ids"]) == tuple(range(8))
    assert batch["batch_size"] == 8
    assert len(batch["layer_metrics"]) == 2

    requests = _requests(summary)
    assert {row["batch_id"] for row in requests.values()} == {batch["batch_id"]}
    assert {
        row["admission_time_ms"] for row in requests.values()
    } == {batch["admission_time_ms"]}


def test_request_arriving_mid_batch_cannot_join_or_release_one_slot():
    initial = (
        _request(0, per_layer_compute_ms=5.0, n_layers=2),
        _request(1, per_layer_compute_ms=5.0, n_layers=2),
    )
    late = _request(
        2,
        arrival_ms=0.5,
        per_layer_compute_ms=1.0,
        n_layers=2,
    )
    summary = _run((*initial, late), n_layers=2, batch_size=2)

    first, second = _batches(summary)
    assert tuple(first["member_request_ids"]) == (0, 1)
    assert tuple(second["member_request_ids"]) == (2,)
    assert second["admission_time_ms"] == pytest.approx(
        first["completion_time_ms"]
    )
    assert _requests(summary)[2]["admission_time_ms"] == pytest.approx(
        first["completion_time_ms"]
    )


def test_batch_compute_waits_for_slowest_member_io_barrier():
    fast = _request(0, placement=((((0, 0.001),)),), n_layers=1)
    slow = _request(1, placement=((((1, 0.040),)),), n_layers=1)
    summary = _run((fast, slow), n_layers=1, batch_size=2, num_ssu=2)

    batch = _batches(summary)[0]
    layer = batch["layer_metrics"][0]
    assert layer["layer"] == 0
    assert layer["compute_start_ms"] == pytest.approx(
        layer["io_ready_time_ms"]
    )
    assert layer["io_ready_time_ms"] >= 1.8 - 1e-9
    assert layer["io_barrier_wait_ms"] == pytest.approx(
        layer["compute_start_ms"] - batch["admission_time_ms"]
    )


def test_joint_compute_duration_is_sum_of_member_compute_work():
    summary = _run(
        (
            _request(0, per_layer_compute_ms=2.0, n_layers=3),
            _request(1, per_layer_compute_ms=3.0, n_layers=3),
        ),
        n_layers=3,
        batch_size=2,
    )

    batch = _batches(summary)[0]
    for layer in batch["layer_metrics"]:
        assert layer["compute_duration_ms"] == pytest.approx(5.0)
        assert layer["compute_end_ms"] - layer["compute_start_ms"] == pytest.approx(
            5.0
        )
    assert batch["compute_busy_ms"] == pytest.approx(15.0)


def test_next_layer_prefetch_starts_with_previous_layer_compute():
    summary = _run(
        (
            _request(0, per_layer_compute_ms=10.0, n_layers=2),
            _request(1, per_layer_compute_ms=10.0, n_layers=2),
        ),
        n_layers=2,
        batch_size=2,
    )

    layer0, layer1 = _batches(summary)[0]["layer_metrics"]
    assert layer1["io_start_time_ms"] == pytest.approx(
        layer0["compute_start_ms"]
    )
    assert layer1["io_ready_time_ms"] < layer0["compute_end_ms"]
    assert layer1["compute_start_ms"] == pytest.approx(
        layer0["compute_end_ms"]
    )


def test_layer_15_completes_every_batch_member_at_same_timestamp():
    summary = _run(
        (
            _request(0, per_layer_compute_ms=0.1, n_layers=16),
            _request(1, per_layer_compute_ms=0.2, n_layers=16),
        ),
        n_layers=16,
        batch_size=2,
    )

    batch = _batches(summary)[0]
    layer15 = batch["layer_metrics"][-1]
    requests = _requests(summary)
    assert layer15["layer"] == 15
    assert batch["completion_time_ms"] == pytest.approx(
        layer15["compute_end_ms"]
    )
    assert requests[0]["completion_time_ms"] == pytest.approx(
        batch["completion_time_ms"]
    )
    assert requests[1]["completion_time_ms"] == pytest.approx(
        batch["completion_time_ms"]
    )


def test_final_partial_microbatch_drains_when_no_future_arrivals_remain():
    summary = _run(
        (_request(request_id, n_layers=2) for request_id in range(3)),
        n_layers=2,
        batch_size=2,
    )

    batches = _batches(summary)
    assert [batch["batch_size"] for batch in batches] == [2, 1]
    assert tuple(batches[0]["member_request_ids"]) == (0, 1)
    assert tuple(batches[1]["member_request_ids"]) == (2,)
    assert batches[1]["admission_time_ms"] == pytest.approx(
        batches[0]["completion_time_ms"]
    )
    assert summary["request_count"] == 3


def test_physical_compute_busy_counts_joint_batch_once_not_per_member():
    n_layers = 4
    summary = _run(
        (
            _request(0, per_layer_compute_ms=2.0, n_layers=n_layers),
            _request(1, per_layer_compute_ms=3.0, n_layers=n_layers),
        ),
        n_layers=n_layers,
        batch_size=2,
    )

    batch = _batches(summary)[0]
    expected_physical_busy_ms = n_layers * (2.0 + 3.0)
    reported_physical_busy_ms = (
        summary["fleet_npu_compute_utilization"]
        * summary["num_npu"]
        * summary["makespan_ms"]
    )
    assert batch["compute_busy_ms"] == pytest.approx(expected_physical_busy_ms)
    assert reported_physical_busy_ms == pytest.approx(expected_physical_busy_ms)

    request_rows = _requests(summary)
    assert all(
        row["batch_compute_ms"] == pytest.approx(expected_physical_busy_ms)
        for row in request_rows.values()
    )
    duplicated_request_view_ms = sum(
        row["batch_compute_ms"] for row in request_rows.values()
    )
    assert duplicated_request_view_ms == pytest.approx(
        2.0 * expected_physical_busy_ms
    )
    assert reported_physical_busy_ms != pytest.approx(
        duplicated_request_view_ms
    )


def test_paired_membership_ignores_strategy_dependent_global_batch_ids():
    def summary(batch_id, members):
        return {
            "input_fingerprint": "same-input",
            "request_count": 2,
            "microbatch_count": 1,
            "submitted_blocks": 2,
            "expected_read_gb": 1.0,
            "completed_request_layer_jobs": 2,
            "fleet_npu_compute_utilization": 0.5,
            "num_npu": 1,
            "makespan_ms": 10.0,
            "microbatch_metrics": [{
                "batch_id": batch_id,
                "npu_id": 0,
                "admission_time_ms": 0.0,
                "member_request_ids": list(members),
            }],
        }

    paired = _paired_invariants({
        "a": {"summary": summary(7, (0, 1))},
        "b": {"summary": summary(99, (0, 1))},
    })
    assert all(paired.values())
    mismatched = _paired_invariants({
        "a": {"summary": summary(7, (0, 1))},
        "b": {"summary": summary(99, (0, 2))},
    })
    assert not mismatched["canonical_microbatch_membership"]


def test_causal_controller_uses_observed_bytes_and_cold_membership_only():
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(2))
    controller = CausalMaxMinSchemeBController(
        paths, cold_path_id=0, cold_path_cir_gbps=0.25, path_count=PATH_COUNT
    )
    snapshot = CausalLayerSnapshot(
        num_npu=2,
        num_ssu=1,
        active_batches=(
            CausalLayerObservation(0, 0, 0, (1.0,), 100.0),
            CausalLayerObservation(1, 1, -1, (), 100.0),
        ),
    )
    decision = controller(snapshot)
    cirs = decision.path_cirs_by_ssu[0]
    assert cirs[paths[0]] == pytest.approx(10.0)
    assert cirs[0] == pytest.approx(0.25)
    assert sum(cirs) == pytest.approx(10.25)
    assert controller(snapshot) is None


def test_hybrid_layer0_matches_baseline_then_uses_scheme_b():
    requests = tuple(
        _request(
            request_id,
            npu_id=request_id,
            per_layer_compute_ms=5.0,
            placement=((((0, 0.001),)),),
            n_layers=2,
        )
        for request_id in range(2)
    )
    common = dict(
        requests=requests,
        num_npu=2,
        num_ssu=1,
        n_layers=2,
        batch_size=1,
        submit_order_seed=42,
    )
    baseline = simulate_continuous_batch(
        qos_config=legacy_qos_config(),
        client_io_config=sim.ClientIOConfig(
            "test_baseline", None, path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO
        ),
        **common,
    )

    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(2))
    configs = qos_configs_from_path_cirs(((0.0,) * PATH_COUNT,))
    hybrid = simulate_continuous_batch(
        qos_configs_by_ssu=configs,
        npu_dedicated_paths=paths,
        layer0_path_id=0,
        client_io_config=scheme_b_client_config("test_after_l0"),
        causal_control=CausalLayerControlConfig(
            CausalMaxMinSchemeBController(
                paths,
                cold_path_id=0,
                cold_path_cir_gbps=legacy_qos_config().path_cirs[0],
                path_count=PATH_COUNT,
            )
        ),
        **common,
    )

    baseline_l0 = {
        row["npu_id"]: row["layer_metrics"][0]["io_ready_time_ms"]
        for row in baseline["microbatch_metrics"]
    }
    hybrid_l0 = {
        row["npu_id"]: row["layer_metrics"][0]["io_ready_time_ms"]
        for row in hybrid["microbatch_metrics"]
    }
    assert hybrid_l0 == pytest.approx(baseline_l0)
    assert hybrid["routing_mode"] == "layer0_path0_then_causal_npu_dedicated"
    assert all(hybrid["invariants"].values())
    assert hybrid["causal_layer_observations"] == 4


def test_causal_snapshot_contains_completed_previous_layer_not_future_placement():
    requests = (
        _request(
            0,
            per_layer_compute_ms=5.0,
            placement=((((0, 0.001),)), (((1, 0.003),))),
            n_layers=2,
        ),
    )
    snapshots = []

    def observe(snapshot):
        snapshots.append(snapshot)
        return CIRControlDecision(((0.0,) * PATH_COUNT,) * 2)

    simulate_continuous_batch(
        requests,
        num_npu=1,
        num_ssu=2,
        n_layers=2,
        batch_size=1,
        qos_configs_by_ssu=qos_configs_from_path_cirs(
            ((0.0,) * PATH_COUNT,) * 2
        ),
        npu_dedicated_paths=(cold_start_hybrid_path_id(0),),
        layer0_path_id=0,
        client_io_config=scheme_b_client_config("causal_snapshot"),
        causal_control=CausalLayerControlConfig(observe),
    )

    first_warm = next(
        snapshot.active_batches[0]
        for snapshot in snapshots
        if snapshot.active_batches
        and snapshot.active_batches[0].observed_layer == 0
    )
    assert first_warm.observed_work_gb_by_ssu == pytest.approx((0.001, 0.0))
    assert not hasattr(snapshots[0], "time_ms")
