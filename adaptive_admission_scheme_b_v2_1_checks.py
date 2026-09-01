"""Fast deterministic checks for Adaptive V2.1 decision diagnostics.

Run directly with::

    python -B adaptive_admission_scheme_b_v2_1_checks.py

The checks execute controller/allocator calls only; they do not run the DES or
write result artifacts.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from adaptive_admission_scheme_b_v2_1 import (
    REJECTION_NPU_CAPACITY,
    REJECTION_SSU_CAPACITY,
    RESIDUAL_MODE_COFLOW,
    RESIDUAL_MODE_EXPLICIT,
    SELECTION_MODE_GREEDY,
    SELECTION_MODE_PREFERRED,
    SELECTION_MODE_REQUIRED,
    AdaptiveAdmissionSchemeBControllerV2_1,
    allocate_adaptive_admission_grants,
    replay_admission_selection,
)
from continuous_batch_sim import CIRControlSnapshot, ControlRequestView


def _request(
    npu_id: int,
    demand_gbps_by_ssu: tuple[float, ...],
    *,
    request_id: int | None = None,
    compute_ms: float = 100.0,
) -> ControlRequestView:
    compute_s = compute_ms / 1000.0
    return ControlRequestView(
        request_id=100 + npu_id if request_id is None else request_id,
        npu_id=npu_id,
        category="diagnostic",
        per_layer_compute_ms=compute_ms,
        compute_done_up_to=0,
        remaining_layers=1,
        next_layer_work_gb_by_ssu=tuple(
            amount * compute_s for amount in demand_gbps_by_ssu
        ),
        waiting_for_io=True,
        remaining_work_gb_by_ssu=tuple(
            amount * compute_s for amount in demand_gbps_by_ssu
        ),
        remaining_compute_budget_ms=compute_ms,
    )


def _snapshot(
    requests: tuple[ControlRequestView, ...],
    *,
    num_npu: int,
    num_ssu: int,
    evaluation: int = 1,
    time_ms: float = 25.0,
    trigger_reasons: tuple[str, ...] | None = None,
):
    values = {
        "time_ms": time_ms,
        "evaluation": evaluation,
        "layer_jobs_since_previous": 7,
        "num_npu": num_npu,
        "num_ssu": num_ssu,
        "active_requests": requests,
        "current_path_cirs_by_ssu": tuple(
            tuple(0.0 for _ in range(num_npu)) for _ in range(num_ssu)
        ),
    }
    if trigger_reasons is not None:
        values["trigger_reasons"] = trigger_reasons
        return SimpleNamespace(**values)
    return CIRControlSnapshot(**values)


class AdaptiveAdmissionDiagnosticsChecks(unittest.TestCase):
    def test_default_off_and_enabled_diagnostics_do_not_change_decisions(self):
        plain = AdaptiveAdmissionSchemeBControllerV2_1((0, 1, 2))
        traced = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1, 2), record_diagnostics=True
        )
        snapshots = (
            _snapshot(
                tuple(_request(npu, (30.0,)) for npu in range(3)),
                num_npu=3,
                num_ssu=1,
                evaluation=1,
            ),
            _snapshot(
                tuple(_request(npu, (30.0,)) for npu in range(3)),
                num_npu=3,
                num_ssu=1,
                evaluation=2,
                time_ms=125.0,
            ),
        )

        self.assertFalse(plain.record_diagnostics)
        for snapshot in snapshots:
            self.assertEqual(plain(snapshot), traced(snapshot))
            self.assertEqual(plain.last_allocation, traced.last_allocation)
            self.assertEqual(
                plain.selected_request_by_npu,
                traced.selected_request_by_npu,
            )
            self.assertEqual(
                plain.residual_mode_evaluations,
                traced.residual_mode_evaluations,
            )
        self.assertEqual(plain.diagnostics, [])
        self.assertEqual(len(traced.diagnostics), 2)

    def test_records_snapshot_manifest_fallback_and_controller_demand(self):
        direct = _request(0, (10.0, 20.0), compute_ms=100.0)
        fallback = ControlRequestView(
            request_id=101,
            npu_id=1,
            category="fallback",
            per_layer_compute_ms=50.0,
            compute_done_up_to=0,
            remaining_layers=2,
            next_layer_work_gb_by_ssu=(0.5, 0.25),
            waiting_for_io=False,
            remaining_work_gb_by_ssu=(),
            remaining_compute_budget_ms=0.0,
            prefetch_only=True,
        )
        snapshot = _snapshot(
            (direct, fallback),
            num_npu=2,
            num_ssu=2,
            evaluation=9,
            time_ms=321.0,
            trigger_reasons=("admission", "cross_request_l0_prefetch"),
        )
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1), record_diagnostics=True
        )

        controller(snapshot)
        record = controller.diagnostics[0]
        self.assertEqual(record.snapshot_time_ms, 321.0)
        self.assertEqual(record.snapshot_evaluation, 9)
        self.assertEqual(record.layer_jobs_since_previous, 7)
        self.assertEqual(
            record.trigger_reasons,
            ("admission", "cross_request_l0_prefetch"),
        )
        self.assertEqual(record.request_by_npu, ((0, 100), (1, 101)))
        self.assertEqual(
            record.remaining_work_gb_by_npu_ssu,
            ((1.0, 2.0), (1.0, 0.5)),
        )
        self.assertEqual(record.remaining_compute_s_by_npu, (0.1, 0.1))
        self.assertEqual(
            record.controller_demand_gbps_by_npu_ssu,
            ((10.0, 20.0), (10.0, 5.0)),
        )
        self.assertEqual(record.selection_mode, SELECTION_MODE_PREFERRED)
        self.assertEqual(record.selected_npu_ids, (0, 1))
        self.assertEqual(record.rejected_npu_ids, ())
        self.assertEqual(record.residual_mode, RESIDUAL_MODE_COFLOW)
        self.assertIsNone(record.v2_floor_grants_gbps)

    def test_required_floor_feasible_mode_is_distinguished(self):
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1), record_diagnostics=True
        )
        snapshot = _snapshot(
            (_request(0, (39.5,)), _request(1, (39.5,))),
            num_npu=2,
            num_ssu=1,
        )

        controller(snapshot)
        record = controller.diagnostics[0]
        self.assertEqual(record.selection_mode, SELECTION_MODE_REQUIRED)
        self.assertEqual(record.effective_target_ratio, 0.5)
        self.assertEqual(record.selected_npu_ids, (0, 1))
        self.assertEqual(record.candidate_order, ())
        self.assertEqual(record.residual_mode, RESIDUAL_MODE_COFLOW)

    def test_greedy_pinning_and_ssu_capacity_rejection_are_exact(self):
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1, 2), record_diagnostics=True
        )
        controller.selected_request_by_npu = {2: 102}
        snapshot = _snapshot(
            tuple(_request(npu, (30.0,)) for npu in range(3)),
            num_npu=3,
            num_ssu=1,
        )

        controller(snapshot)
        record = controller.diagnostics[0]
        self.assertEqual(record.previous_selected_request_by_npu, ((2, 102),))
        self.assertEqual(record.previous_pinned_npu_ids, (2,))
        self.assertEqual(record.selection_mode, SELECTION_MODE_GREEDY)
        self.assertEqual(record.candidate_order, (0, 1))
        self.assertEqual(record.selected_npu_ids, (2, 0))
        self.assertEqual(record.rejected_npu_ids, (1,))
        self.assertAlmostEqual(record.selected_fraction, 2.0 / 3.0)
        self.assertEqual(record.residual_mode, RESIDUAL_MODE_EXPLICIT)

        rejection = record.capacity_rejections[0]
        self.assertEqual(rejection.npu_id, 1)
        self.assertEqual(rejection.stage, "greedy_candidate")
        self.assertEqual(rejection.rejection_reason, REJECTION_SSU_CAPACITY)
        self.assertEqual(rejection.violating_ssu_ids, (0,))
        self.assertAlmostEqual(rejection.target_gbps_by_ssu[0], 15.6)
        self.assertAlmostEqual(
            rejection.admission_remaining_before_gbps_by_ssu[0], 6.8
        )

        self.assertIsNotNone(record.v2_floor_grants_gbps)
        np.testing.assert_allclose(
            record.grants_gbps_by_npu_ssu,
            ((19.0,), (2.0,), (19.0,)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            record.v2_floor_grants_gbps,
            ((15.6,), (0.0,), (15.6,)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            record.v2_background_grants_gbps,
            ((0.0,), (2.0,), (0.0,)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            record.v2_selected_tail_grants_gbps,
            ((3.4,), (0.0,), (3.4,)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            record.v2_spill_tail_grants_gbps,
            ((0.0,), (0.0,), (0.0,)),
            rtol=0.0,
            atol=1e-12,
        )
        component_sum = np.sum(
            np.asarray(record.v2_floor_grants_gbps)
            + np.asarray(record.v2_background_grants_gbps)
            + np.asarray(record.v2_selected_tail_grants_gbps)
            + np.asarray(record.v2_spill_tail_grants_gbps),
            axis=None,
        )
        self.assertAlmostEqual(
            float(component_sum),
            math.fsum(
                value for row in record.grants_gbps_by_npu_ssu for value in row
            ),
        )

    def test_npu_capacity_rejection_precedes_ssu_capacity_check(self):
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1), record_diagnostics=True
        )
        snapshot = _snapshot(
            (_request(0, (60.0, 60.0)), _request(1, (5.0, 5.0))),
            num_npu=2,
            num_ssu=2,
        )

        controller(snapshot)
        record = controller.diagnostics[0]
        self.assertEqual(record.selection_mode, SELECTION_MODE_GREEDY)
        self.assertEqual(record.candidate_order, (1, 0))
        self.assertEqual(record.selected_npu_ids, (1,))
        self.assertEqual(record.rejected_npu_ids, (0,))
        rejection = record.capacity_rejections[0]
        self.assertEqual(rejection.rejection_reason, REJECTION_NPU_CAPACITY)
        self.assertEqual(rejection.violating_ssu_ids, ())
        self.assertAlmostEqual(rejection.target_sum_gbps, 62.4)
        self.assertEqual(rejection.npu_capacity_gbps, 50.0)

    def test_failed_pin_is_retried_in_the_normal_candidate_order(self):
        demand = ((8.0, 8.0), (16.0, 0.0), (30.0, 0.0), (0.0, 0.0))
        replay = replay_admission_selection(
            demand,
            target_ratio=0.5,
            required_ratio=0.5,
            background_reserve_fraction=0.05,
            pinned_npu_ids=(2,),
            ssd_caps=(10.0, 10.0),
            npu_caps=(10.0, 10.0, 10.0, 10.0),
        )
        allocation = allocate_adaptive_admission_grants(
            demand,
            target_ratio=0.5,
            required_ratio=0.5,
            background_reserve_fraction=0.05,
            pinned_npu_ids=(2,),
            ssd_caps=(10.0, 10.0),
            npu_caps=(10.0, 10.0, 10.0, 10.0),
        )

        self.assertEqual(replay.selected_npu_ids, allocation.selected_npu_ids)
        self.assertEqual(replay.candidate_order, (0, 1, 2))
        npu2_attempts = [
            attempt for attempt in replay.attempts if attempt.npu_id == 2
        ]
        self.assertEqual(
            [attempt.stage for attempt in npu2_attempts],
            ["pinned", "greedy_candidate"],
        )
        self.assertEqual(
            [attempt.attempt_index for attempt in replay.attempts],
            list(range(len(replay.attempts))),
        )
        self.assertEqual(
            [attempt.rejection_reason for attempt in npu2_attempts],
            [REJECTION_NPU_CAPACITY, REJECTION_NPU_CAPACITY],
        )

    def test_selected_fraction_threshold_is_strict(self):
        snapshot = _snapshot(
            tuple(_request(npu, (24.0,)) for npu in range(4)),
            num_npu=4,
            num_ssu=1,
        )
        at_threshold = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1, 2, 3),
            explicit_spill_threshold=0.75,
            record_diagnostics=True,
        )
        above_threshold = AdaptiveAdmissionSchemeBControllerV2_1(
            (0, 1, 2, 3),
            explicit_spill_threshold=0.7500001,
            record_diagnostics=True,
        )

        at_threshold(snapshot)
        above_threshold(snapshot)
        self.assertEqual(at_threshold.diagnostics[0].selected_fraction, 0.75)
        self.assertEqual(
            at_threshold.diagnostics[0].residual_mode,
            RESIDUAL_MODE_COFLOW,
        )
        self.assertEqual(
            above_threshold.diagnostics[0].residual_mode,
            RESIDUAL_MODE_EXPLICIT,
        )

    def test_pure_replay_matches_allocator_across_random_inputs(self):
        rng = np.random.default_rng(20260902)
        for _ in range(300):
            num_npu = int(rng.integers(1, 12))
            num_ssu = int(rng.integers(1, 6))
            demand = rng.uniform(0.0, 80.0, size=(num_npu, num_ssu))
            demand[rng.random((num_npu, num_ssu)) < 0.2] = 0.0
            pinned = tuple(
                int(npu)
                for npu in rng.choice(
                    num_npu,
                    size=int(rng.integers(0, num_npu + 1)),
                    replace=False,
                )
            )
            target_ratio = float(rng.uniform(0.3, 0.9))
            required_ratio = float(rng.uniform(0.1, target_ratio))
            reserve = float(rng.uniform(0.0, 0.25))
            ssd_caps = tuple(
                float(value) for value in rng.uniform(5.0, 60.0, size=num_ssu)
            )
            npu_caps = tuple(
                float(value) for value in rng.uniform(10.0, 80.0, size=num_npu)
            )
            original = demand.copy()
            replay = replay_admission_selection(
                demand,
                target_ratio=target_ratio,
                required_ratio=required_ratio,
                background_reserve_fraction=reserve,
                pinned_npu_ids=pinned,
                ssd_caps=ssd_caps,
                npu_caps=npu_caps,
            )
            allocation = allocate_adaptive_admission_grants(
                demand,
                target_ratio=target_ratio,
                required_ratio=required_ratio,
                background_reserve_fraction=reserve,
                pinned_npu_ids=pinned,
                ssd_caps=ssd_caps,
                npu_caps=npu_caps,
            )
            self.assertEqual(replay.selected_npu_ids, allocation.selected_npu_ids)
            np.testing.assert_array_equal(demand, original)

    def test_missing_trigger_reasons_records_empty_tuple(self):
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            (0,), record_diagnostics=True
        )
        controller(
            _snapshot((_request(0, (5.0,)),), num_npu=1, num_ssu=1)
        )
        self.assertEqual(controller.diagnostics[0].trigger_reasons, ())


if __name__ == "__main__":
    unittest.main()
