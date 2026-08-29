"""Parity and invariant tests for the DES-independent policy module."""

from types import SimpleNamespace

import pytest

import sim
from continuous_batch_sim import (
    CausalLayerObservation as ExistingCausalObservation,
    CausalLayerSnapshot as ExistingCausalSnapshot,
    CausalMaxMinSchemeBController,
)
from policy_logic import (
    CausalLayerObservation,
    ManifestDemand,
    OracleFlowView,
    baseline_path_ids,
    category_path_ids,
    hardware_view,
    oracle_priority_key,
    plan_causal_scheme_b,
    plan_scheme_b,
    pressure_aware_path_ids,
    pressure_snapshot,
    refresh8_path_ids,
    validate_scheme_b_plan,
)
from scheme_b_prefill import cold_start_hybrid_path_id
from strategy_profiles import FINAL_STATIC


def _pressure_fixture():
    counts = [0] * 256
    for path_id, count in ((0, 5), (1, 2), (32, 3), (64, 1), (96, 4)):
        counts[path_id] = count
    qos = FINAL_STATIC.hardware_config()
    group_counts = [0] * 8
    active_paths = [0] * 8
    active_weights = [0.0] * 8
    active_group_weight = 0.0
    active_cir = 0.0
    for path_id, count in enumerate(counts):
        if not count:
            continue
        group_id = path_id // 32
        group_counts[group_id] += count
        active_paths[group_id] += 1
        active_weights[group_id] += qos.path_weights[path_id]
        active_cir += min(qos.path_cirs[path_id], qos.path_pirs[path_id])
    active_group_weight = sum(
        qos.group_weights[group_id]
        for group_id, count in enumerate(active_paths)
        if count
    )
    return SimpleNamespace(
        counts=tuple(counts),
        group_io_counts=tuple(group_counts),
        active_paths_per_group=tuple(active_paths),
        active_path_weights=tuple(active_weights),
        active_group_weight_sum=active_group_weight,
        active_cir_sum=active_cir,
    )


def test_baseline_is_fixed_path_zero_without_external_state():
    assert baseline_path_ids(5) == (0, 0, 0, 0, 0)


def test_category_path_mapping_matches_existing_client():
    existing = FINAL_STATIC.hardware_config()
    pure = hardware_view(existing)
    for category in existing.category_labels:
        assert category_path_ids(category, pure) == sim.client_category_paths(
            category, existing
        )


@pytest.mark.parametrize(
    "sizes,golden",
    [
        ((0.001,), (130,)),
        (
            (0.001, 0.004, 0.002, 0.008, 0.003, 0.005, 0.006, 0.007),
            (195, 66, 163, 130, 131, 226, 194, 162),
        ),
    ],
)
def test_pressure_routing_is_numerically_identical_to_existing_sim(sizes, golden):
    existing_qos = FINAL_STATIC.hardware_config()
    pure_qos = hardware_view(existing_qos)
    existing_pressure = _pressure_fixture()
    pure_pressure = pressure_snapshot(existing_pressure)
    allowed = sim.client_category_paths("SS", existing_qos)
    config = sim.ClientRoutingConfig(existing_qos, sim.DISK_BW, start_offset=17)
    expected = tuple(
        sim._select_qos_paths_from_analysis(
            sizes, existing_pressure, allowed, config
        )
    )
    actual = pressure_aware_path_ids(
        sizes,
        pure_pressure,
        allowed,
        pure_qos,
        disk_bw_gbps=sim.DISK_BW,
        start_offset=17,
    )
    assert actual == expected == golden
    assert refresh8_path_ids(
        sizes,
        pure_pressure,
        allowed,
        pure_qos,
        disk_bw_gbps=sim.DISK_BW,
        start_offset=17,
    ) == expected


def test_refresh8_rejects_more_than_one_pressure_window():
    qos = hardware_view(FINAL_STATIC.hardware_config())
    pressure = pressure_snapshot(_pressure_fixture())
    with pytest.raises(ValueError, match="at most eight"):
        refresh8_path_ids(
            (0.001,) * 9,
            pressure,
            category_path_ids("SS", qos),
            qos,
        )


def test_scheme_b_allocates_10_and_30_gbps_without_false_equal_split():
    plan = plan_scheme_b(
        (
            ManifestDemand(0, 0, 1.0, (10.0,)),
            ManifestDemand(1, 1, 1.0, (30.0,)),
        ),
        num_npu=2,
        num_ssu=1,
    )
    assert plan.demands_gbps == ((10.0,), (30.0,))
    assert plan.grants_gbps == ((10.0,), (30.0,))
    assert plan.path_cirs_by_ssu[0][plan.path_by_npu[0]] == 10.0
    assert plan.path_cirs_by_ssu[0][plan.path_by_npu[1]] == 30.0
    assert validate_scheme_b_plan(plan)


def test_scheme_b_target_has_stable_golden_hash():
    actual = plan_scheme_b(
        (
            ManifestDemand(10, 0, 0.02, (0.4, 0.2)),
            ManifestDemand(11, 1, 0.025, (0.1, 0.5)),
        ),
        num_npu=2,
        num_ssu=2,
    )
    assert actual.demands_gbps == ((20.0, 10.0), (4.0, 20.0))
    assert actual.grants_gbps == ((20.0, 10.0), (4.0, 20.0))
    assert (
        actual.target_hash
        == "18ce78817df2d88ee5ab9ed802baeb1afd8b864d58317ed1deaaf8e32a9b4f7f"
    )
    assert validate_scheme_b_plan(actual)


def test_causal_scheme_b_matches_existing_controller_cir_table():
    path_by_npu = tuple(cold_start_hybrid_path_id(npu) for npu in range(3))
    pure_observations = (
        CausalLayerObservation(10, 0, 0, 20.0, (0.4, 0.2)),
        CausalLayerObservation(11, 1, 0, 25.0, (0.1, 0.5)),
        CausalLayerObservation(12, 2, -1, 30.0, (0.3, 0.3)),
    )
    pure = plan_causal_scheme_b(
        pure_observations,
        num_npu=3,
        num_ssu=2,
        cold_path_id=0,
        cold_path_cir_gbps=2.5,
        path_by_npu=path_by_npu,
    )

    existing_observations = tuple(
        ExistingCausalObservation(
            batch_id=row.request_id,
            npu_id=row.npu_id,
            observed_layer=row.observed_layer,
            observed_work_gb_by_ssu=row.observed_work_gb_by_ssu,
            compute_budget_ms=row.compute_budget_ms,
        )
        for row in pure_observations
    )
    existing = CausalMaxMinSchemeBController(
        path_by_npu,
        cold_path_id=0,
        cold_path_cir_gbps=2.5,
        path_count=256,
    )(
        ExistingCausalSnapshot(
            num_npu=3,
            num_ssu=2,
            active_batches=existing_observations,
        )
    )
    assert pure.path_cirs_by_ssu == existing.path_cirs_by_ssu
    assert validate_scheme_b_plan(pure)


def test_oracle_priority_matches_retained_capacity_candidate_formula():
    flow = OracleFlowView(10.0, 2.0, 3.0, 4.0, 5, 6, 7, 8)
    key = oracle_priority_key(flow)
    assert key == (
        2.0 / 10.0**0.25,
        3.0,
        2.0,
        4.0,
        5,
        6,
        7,
        8,
    )
