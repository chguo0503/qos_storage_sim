"""Fast, deterministic correctness checks for protected_floor_scheme_b.

This file deliberately does not use the project's retired ``test_*.py`` naming
pattern.  Run it directly with ``python -B protected_floor_scheme_b_checks.py``.
It performs no simulation and writes no result artifact.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from protected_floor_scheme_b import (
    ProtectedFloorSchemeBController,
    allocate_protected_floor_grants,
    reconcile_protected_floor_cirs,
)
from continuous_batch_sim import (
    CIRControlDecision,
    CIRControlSnapshot,
    ControlRequestView,
    _apply_cir_decision,
)
from slo_admission_scheme_b_v2 import allocate_slo_admission_grants_v2


SSD_TOLERANCE = 1e-12
NPU_TOLERANCE = 1e-9


class ProtectedFloorSchemeBChecks(unittest.TestCase):
    @staticmethod
    def controller_snapshot(current):
        """Return one two-SSU public snapshot with a 10 GB/s/SSU demand."""

        return CIRControlSnapshot(
            time_ms=25.0,
            evaluation=1,
            layer_jobs_since_previous=1,
            num_npu=1,
            num_ssu=2,
            active_requests=(
                ControlRequestView(
                    request_id=7,
                    npu_id=0,
                    category="SS",
                    per_layer_compute_ms=100.0,
                    compute_done_up_to=0,
                    remaining_layers=1,
                    next_layer_work_gb_by_ssu=(1.0, 1.0),
                    waiting_for_io=False,
                    remaining_work_gb_by_ssu=(1.0, 1.0),
                    remaining_compute_budget_ms=100.0,
                ),
            ),
            current_path_cirs_by_ssu=current,
        )

    def assert_capacity_safe_sequence(
        self,
        actual,
        reconciliation,
        *,
        path_by_npu,
        ssd_caps=40.0,
        npu_caps=50.0,
    ):
        """Apply every planned register write and check every prefix."""

        table = [list(map(float, row)) for row in actual]
        ssd_limits = (
            (float(ssd_caps),) * len(table)
            if isinstance(ssd_caps, (int, float))
            else tuple(map(float, ssd_caps))
        )
        npu_limits = (
            (float(npu_caps),) * len(path_by_npu)
            if isinstance(npu_caps, (int, float))
            else tuple(map(float, npu_caps))
        )
        phase_rank = {"decrease": 0, "increase": 1}
        previous_phase = 0

        for sequence_index, change in enumerate(reconciliation.ordered_changes):
            self.assertEqual(change.sequence_index, sequence_index)
            current_phase = phase_rank[change.phase]
            self.assertGreaterEqual(current_phase, previous_phase)
            previous_phase = current_phase
            self.assertAlmostEqual(
                table[change.ssu_id][change.path_id],
                change.old_gbps,
                delta=1e-15,
            )
            table[change.ssu_id][change.path_id] = change.new_gbps

            for ssu_id, row in enumerate(table):
                self.assertLessEqual(
                    math.fsum(row),
                    ssd_limits[ssu_id] + SSD_TOLERANCE,
                )
            for npu_id, path_id in enumerate(path_by_npu):
                self.assertLessEqual(
                    math.fsum(row[path_id] for row in table),
                    npu_limits[npu_id] + NPU_TOLERANCE,
                )

        expected = reconciliation.install_path_cirs_by_ssu
        self.assertEqual(len(table), len(expected))
        for actual_row, expected_row in zip(table, expected):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, delta=1e-15)

    def test_protected_allocation_is_exactly_v2_floor(self):
        demand = ((30.0,), (30.0,), (30.0,))
        source = allocate_slo_admission_grants_v2(demand)
        protected = allocate_protected_floor_grants(demand)

        self.assertEqual(
            protected.protected_floor_grants_gbps,
            source.floor_grants_gbps,
        )
        self.assertGreater(
            math.fsum(
                value
                for stage in (
                    source.background_grants_gbps,
                    source.selected_tail_grants_gbps,
                    source.spill_tail_grants_gbps,
                )
                for row in stage
                for value in row
            ),
            0.0,
        )
        for required_row, floor_row in zip(
            protected.required_floor_grants_gbps,
            protected.protected_floor_grants_gbps,
        ):
            for required, floor in zip(required_row, floor_row):
                self.assertLessEqual(required, floor + NPU_TOLERANCE)

    def test_sub_nanogbps_ssu_overflow_forces_compensating_decrease(self):
        """Regression for the DES-1e-12 versus PFO-1e-9 mismatch."""

        increment = 5e-10
        actual = ((20.0, 20.0),)
        ideal = ((20.0 + increment, 20.0 - increment),)
        required = ((20.0 + increment, 0.0),)
        result = reconcile_protected_floor_cirs(
            actual,
            ideal,
            required,
            path_by_npu=(0, 1),
            deadband_gbps=0.05,
        )

        self.assertEqual(
            [change.phase for change in result.ordered_changes],
            ["decrease", "increase"],
        )
        self.assertEqual(
            [change.reason for change in result.ordered_changes],
            ["ssd_capacity_compensation", "required_hard_floor"],
        )
        self.assertLessEqual(
            math.fsum(result.install_path_cirs_by_ssu[0]),
            40.0 + SSD_TOLERANCE,
        )
        self.assert_capacity_safe_sequence(
            actual,
            result,
            path_by_npu=(0, 1),
        )

    def test_sub_nanogbps_ssu_repair_is_accepted_by_des_apply(self):
        """Run the numeric regression through the DES decision boundary too."""

        increment = 5e-10
        result = reconcile_protected_floor_cirs(
            ((20.0, 20.0),),
            ((20.0 + increment, 20.0 - increment),),
            ((20.0 + increment, 0.0),),
            path_by_npu=(0, 1),
            deadband_gbps=0.05,
        )

        class FakeScheduler:
            def __init__(self):
                self.disk_bw = 40.0
                self.cir_write_threshold_gbps = 0.0
                self.paths = {
                    0: SimpleNamespace(cir=20.0, pir=float("inf")),
                    1: SimpleNamespace(cir=20.0, pir=float("inf")),
                }

            def update_path_cirs(self, values, _time_ms, *, forced_path_ids=()):
                self.assert_no_forced_paths = tuple(forced_path_ids)
                changed = 0
                for path_id, value in enumerate(values):
                    if abs(self.paths[path_id].cir - value) > 1e-12:
                        self.paths[path_id].cir = float(value)
                        changed += 1
                return changed

        scheduler = FakeScheduler()
        context = SimpleNamespace(
            num_ssu=1,
            disks=(SimpleNamespace(scheduler=scheduler),),
            npu_dedicated_paths=(0, 1),
            npu_bw_gbps=50.0,
            disk_bw_gbps=40.0,
            cir_write_transactions_by_ssu=[0],
            cir_path_writes_by_ssu=[0],
            cir_commits=0,
            cir_path_writes=0,
            max_cir_sum_gbps=0.0,
            max_actual_cir_sum_gbps_by_ssu=[0.0],
            max_actual_npu_cir_sum_gbps_by_npu=[0.0, 0.0],
        )
        _apply_cir_decision(
            context,
            CIRControlDecision(result.install_path_cirs_by_ssu),
            1.0,
        )

        installed = tuple(path.cir for path in scheduler.paths.values())
        self.assertEqual(installed, result.install_path_cirs_by_ssu[0])
        self.assertLessEqual(math.fsum(installed), 40.0 + SSD_TOLERANCE)
        self.assertEqual(context.cir_path_writes, 2)

    def test_sub_nanogbps_npu_overflow_uses_des_one_nanogbps_tolerance(self):
        increment = 5e-10
        actual = ((25.0,), (25.0,))
        result = reconcile_protected_floor_cirs(
            actual,
            ((25.0 + increment,), (25.0 - increment,)),
            ((25.0 + increment,), (0.0,)),
            path_by_npu=(0,),
            deadband_gbps=0.05,
        )

        self.assertEqual(
            [change.reason for change in result.ordered_changes],
            ["required_hard_floor"],
        )
        self.assertEqual(result.capacity_compensation_decrease_count, 0)
        self.assertGreater(
            math.fsum(row[0] for row in result.install_path_cirs_by_ssu),
            50.0 + SSD_TOLERANCE,
        )
        self.assertLessEqual(
            math.fsum(row[0] for row in result.install_path_cirs_by_ssu),
            50.0 + NPU_TOLERANCE,
        )

    def test_positive_non_dedicated_actual_cir_is_rejected(self):
        actual = ((0.0, 1.0, 0.0, 2.0, 0.03),)
        ideal = ((0.0, 5.2, 0.0, 0.0, 0.0),)
        required = ((0.0, 5.0, 0.0, 0.0, 0.0),)

        with self.assertRaisesRegex(ValueError, "exclusive ownership"):
            reconcile_protected_floor_cirs(
                actual,
                ideal,
                required,
                path_by_npu=(1, 3),
                deadband_gbps=0.05,
            )

    def test_controller_ownership_rejection_does_not_mutate_pin_state(self):
        controller = ProtectedFloorSchemeBController((1, 3), deadband_gbps=0.05)
        controller.selected_request_by_npu = {0: 99}
        snapshot = CIRControlSnapshot(
            time_ms=0.0,
            evaluation=1,
            layer_jobs_since_previous=0,
            num_npu=2,
            num_ssu=1,
            active_requests=(),
            current_path_cirs_by_ssu=((0.0, 1.0, 0.0, 2.0, 0.03),),
        )

        with self.assertRaisesRegex(ValueError, "exclusive ownership"):
            controller(snapshot)
        self.assertEqual(controller.selected_request_by_npu, {0: 99})
        self.assertIsNone(controller.last_plan)
        self.assertEqual(controller.decisions, 0)

    def test_default_mask_is_exactly_explicit_all_true(self):
        snapshot = self.controller_snapshot(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            )
        )
        default_controller = ProtectedFloorSchemeBController((1,), deadband_gbps=0.05)
        explicit_controller = ProtectedFloorSchemeBController(
            (1,),
            deadband_gbps=0.05,
            materialized_ssu_mask=(True, True),
        )

        default_decision = default_controller(snapshot)
        explicit_decision = explicit_controller(snapshot)
        self.assertEqual(default_decision, explicit_decision)
        self.assertEqual(
            default_controller.last_plan.reconciliation,
            explicit_controller.last_plan.reconciliation,
        )
        self.assertEqual(
            default_controller.last_plan.materialized_ssu_mask,
            (True, True),
        )
        self.assertEqual(
            default_decision.path_cirs_by_ssu,
            (
                (0.0, 5.2, 0.0, 0.0),
                (0.0, 5.2, 0.0, 0.0),
            ),
        )

    def test_mask_clears_cold_ssu_and_preserves_hot_floor_safety(self):
        caller_mask = [False, True]
        controller = ProtectedFloorSchemeBController(
            (1,),
            deadband_gbps=0.05,
            materialized_ssu_mask=caller_mask,
        )
        caller_mask[:] = [True, False]
        snapshot = self.controller_snapshot(
            (
                (0.0, 3.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            )
        )

        decision = controller(snapshot)
        plan = controller.last_plan
        self.assertEqual(controller.materialized_ssu_mask, (False, True))
        with self.assertRaises(AttributeError):
            controller.materialized_ssu_mask = (True, True)
        self.assertEqual(plan.materialized_ssu_mask, (False, True))
        self.assertEqual(
            plan.allocation.protected_floor_grants_gbps,
            ((5.2, 5.2),),
        )
        self.assertEqual(
            plan.reconciliation.ideal_path_cirs_by_ssu,
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 5.2, 0.0, 0.0),
            ),
        )
        self.assertEqual(
            plan.reconciliation.required_path_cirs_by_ssu,
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 5.0, 0.0, 0.0),
            ),
        )
        self.assertEqual(
            decision.path_cirs_by_ssu,
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 5.2, 0.0, 0.0),
            ),
        )
        self.assertEqual(
            [change.phase for change in plan.reconciliation.ordered_changes],
            ["decrease", "increase"],
        )
        self.assert_capacity_safe_sequence(
            snapshot.current_path_cirs_by_ssu,
            plan.reconciliation,
            path_by_npu=(1,),
        )
        self.assertGreaterEqual(
            decision.path_cirs_by_ssu[1][1] + NPU_TOLERANCE,
            plan.reconciliation.required_path_cirs_by_ssu[1][1],
        )

    def test_materialized_mask_requires_boolean_topology_match(self):
        with self.assertRaisesRegex(ValueError, "one boolean per SSU"):
            ProtectedFloorSchemeBController((1,), materialized_ssu_mask=(True, 0))

        controller = ProtectedFloorSchemeBController(
            (1,), materialized_ssu_mask=(True,)
        )
        with self.assertRaisesRegex(ValueError, "exactly one entry per SSU"):
            controller(
                self.controller_snapshot(
                    (
                        (0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 0.0),
                    )
                )
            )

    def test_ssu_and_npu_compensation_sequences_are_capacity_safe(self):
        ssu_actual = ((20.01, 19.99),)
        ssu_result = reconcile_protected_floor_cirs(
            ssu_actual,
            ((20.04, 19.96),),
            ((20.04, 0.0),),
            path_by_npu=(0, 1),
            deadband_gbps=0.05,
        )
        self.assertEqual(
            [change.reason for change in ssu_result.ordered_changes],
            ["ssd_capacity_compensation", "required_hard_floor"],
        )
        self.assert_capacity_safe_sequence(
            ssu_actual,
            ssu_result,
            path_by_npu=(0, 1),
        )

        npu_actual = ((25.01, 0.0), (24.99, 0.0))
        npu_result = reconcile_protected_floor_cirs(
            npu_actual,
            ((25.04, 0.0), (24.96, 0.0)),
            ((25.04, 0.0), (0.0, 0.0)),
            path_by_npu=(0,),
            deadband_gbps=0.05,
        )
        self.assertEqual(
            [change.reason for change in npu_result.ordered_changes],
            ["npu_capacity_compensation", "required_hard_floor"],
        )
        self.assert_capacity_safe_sequence(
            npu_actual,
            npu_result,
            path_by_npu=(0,),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
