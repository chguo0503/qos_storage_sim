"""Pure row/column feasibility and native 32-NPU multi-SSU integration."""

from types import SimpleNamespace
import random
import unittest

import continuous_batch_sim as native
from continuous_batch_sim import CIRControlConfig, ContinuousBatchRequest
from continuous_prefill_client import qos_configs_from_path_cirs, routing_strategy_specs
from multi_ssu_qos_controller import (
    IO_GIB, MultiSSUController, PendingCoflow, choose_coflow_rates,
    coflow_solo_ms, multi_ssu_control_events, snapshot_pending_coflows,
    serial_stall_objective, stall_interchange_order,
)
from policy_logic import dedicated_path_id


class MultiSSUControllerTests(unittest.TestCase):
    def assert_feasible(self, rates, disk=40, link=50):
        for row in rates:
            self.assertLessEqual(sum(row), link + 1e-9)
            self.assertTrue(all(value >= 0 for value in row))
        for column in zip(*rates):
            self.assertLessEqual(sum(column), disk + 1e-9)

    def test_one_coflow_is_link_limited_not_six_times_forty(self):
        job = PendingCoflow(0, 1, 1, (0.1,) * 6, 10, 10)
        rates = choose_coflow_rates((job,), 0, num_npu=32, num_ssu=6)
        self.assert_feasible(rates)
        self.assertAlmostEqual(sum(rates[0]), 50)
        for value in rates[0]:
            self.assertAlmostEqual(value, 50 / 6)
        self.assertAlmostEqual(coflow_solo_ms((0.1,) * 6), 12)

    def test_disjoint_coflows_can_fill_different_ssus(self):
        jobs = (PendingCoflow(0, 1, 1, (.1, 0), 1, 2),
                PendingCoflow(1, 2, 1, (0, .2), 2, 3))
        rates = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=2)
        self.assertEqual(rates, ((40, 0), (0, 40)))

    def test_hotspot_keeps_each_coflow_components_proportional(self):
        jobs = (PendingCoflow(0, 1, 1, (.4, .1), 1, 10),
                PendingCoflow(1, 2, 1, (.1, .4), 2, 10))
        for mode in ("deadline", "deadline_reserve", "least_slack", "demand",
                     "stall_interchange"):
            with self.subTest(mode=mode):
                rates = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=2,
                                            mode=mode)
                self.assert_feasible(rates)
                for job, row in zip(jobs, rates):
                    self.assertAlmostEqual(row[0] / job.remaining_gib_by_ssu[0],
                                           row[1] / job.remaining_gib_by_ssu[1])

    def test_deadline_reservation_can_hide_both_instead_of_early_monopoly(self):
        jobs = (PendingCoflow(0, 1, 1, (.01,), 1, 1),
                PendingCoflow(1, 2, 1, (.02,), 2, 2))
        edf = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=1)
        reserve = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=1,
                                      mode="deadline_reserve")
        self.assertEqual(edf, ((40,), (0,)))
        self.assertEqual(reserve, ((30,), (10,)))

    def test_stall_interchange_serial_two_job_example_19_to_11(self):
        # Deliberately a continuous-fluid arithmetic example, not a claim that
        # 0.4 GiB is an integer number of 176 KiB I/Os. Native smoke tests below
        # use only full I/Os and preserve every original simulator invariant.
        jobs = (PendingCoflow(0, 1, 1, (.4,), 0, 10),
                PendingCoflow(1, 2, 1, (.04,), 2, 10))
        improved = stall_interchange_order(jobs, 0)
        self.assertAlmostEqual(serial_stall_objective(jobs, 0), 19)
        self.assertAlmostEqual(serial_stall_objective(improved, 0), 11)
        self.assertEqual(tuple(job.npu_id for job in improved), (1, 0))
        rates = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=1,
                                    mode="stall_interchange")
        self.assertEqual(rates, ((0,), (40,)))

    def test_stall_interchange_full_io_version_of_two_job_example(self):
        unit_ms = 10 * IO_GIB * 1000 / 40
        jobs = (PendingCoflow(0, 1, 1, (100 * IO_GIB,), 0, 10),
                PendingCoflow(1, 2, 1, (10 * IO_GIB,), 2 * unit_ms, 10))
        improved = stall_interchange_order(jobs, 0)
        self.assertAlmostEqual(serial_stall_objective(jobs, 0) / unit_ms, 19)
        self.assertAlmostEqual(serial_stall_objective(improved, 0) / unit_ms, 11)

    def test_interchange_accounts_for_elapsed_prefix_not_pair_local_zero(self):
        # The final two jobs have zero tardiness if incorrectly treated at t=0.
        # Behind the 100-ms first job their order does matter: 120 -> 111 ms.
        jobs = (PendingCoflow(0, 1, 1, (4.0,), 0, 100),
                PendingCoflow(1, 2, 1, (.4,), 100, 100),
                PendingCoflow(2, 3, 1, (.04,), 101, 100))
        self.assertEqual(serial_stall_objective(jobs[1:], 0), 0)
        improved = stall_interchange_order(jobs, 0)
        self.assertEqual(tuple(job.npu_id for job in improved), (0, 2, 1))
        self.assertAlmostEqual(serial_stall_objective(jobs, 0), 120)
        self.assertAlmostEqual(serial_stall_objective(improved, 0), 111)

    def test_interchange_surrogate_is_monotone_for_seeded_visible_jobs(self):
        rng = random.Random(9127)
        for _ in range(100):
            jobs = tuple(PendingCoflow(
                n, n, 1, tuple(rng.randint(0, 1500) * IO_GIB for _ in range(7)),
                rng.uniform(-10, 100), 20) for n in range(32))
            ordered = tuple(sorted(jobs, key=lambda job: job.deadline_ms))
            improved = stall_interchange_order(ordered, 5)
            self.assertLessEqual(serial_stall_objective(improved, 5),
                                 serial_stall_objective(ordered, 5) + 1e-9)
            self.assertCountEqual(improved, ordered)
            rates = choose_coflow_rates(jobs, 5, num_npu=32, num_ssu=7,
                                        mode="stall_interchange")
            self.assert_feasible(rates)

    def test_idle_lanes_and_all_128_unique_paths(self):
        self.assertEqual(len({dedicated_path_id(i) for i in range(128)}), 128)
        self.assertEqual(choose_coflow_rates((), 10, num_npu=32, num_ssu=7),
                         ((0,) * 7,) * 32)
        with self.assertRaisesRegex(ValueError, "128"):
            choose_coflow_rates((), 0, num_npu=129, num_ssu=6)

    def test_duplicate_layers_require_a_different_prefetch_model(self):
        job = PendingCoflow(0, 1, 1, (.1,), 1, 1)
        with self.assertRaisesRegex(ValueError, "currently released"):
            choose_coflow_rates((job, job), 0, num_npu=2, num_ssu=1)

    def test_snapshot_cross_request_budget_is_predecessor_compute(self):
        current = SimpleNamespace(per_layer_compute_ms=2)
        successor = SimpleNamespace(
            io_started=[1] * 8, io_ready=[0] * 8,
            placement_groups=(((0, (), 10 * IO_GIB), (1, (), 2 * IO_GIB)),),
            completed_gb_by_layer_ssu=[[3 * IO_GIB, IO_GIB]] * 8)
        # Deliberately no SSD queues/events/active service anywhere in context.
        metric = SimpleNamespace(compute_end_ms=7, compute_duration_ms=2)
        batch = SimpleNamespace(member_request_ids=(10,), layer_metrics=[metric] * 8)
        npu = SimpleNamespace(npu_id=0, active_batch=batch, compute_active=(0, 7),
                              layer0_prefetch_request_id=11)
        context = SimpleNamespace(npus=[npu], n_layers=8, num_ssu=2,
                                  requests={10: current, 11: successor})
        jobs = snapshot_pending_coflows(context, 6)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].request_id, 11)
        self.assertEqual(jobs[0].remaining_blocks, (7, 1))
        self.assertEqual(jobs[0].deadline_ms, 7)
        self.assertEqual(jobs[0].compute_ms, 2)
        npu.layer0_prefetch_request_id = None
        self.assertEqual(snapshot_pending_coflows(context, 6), ())

    @staticmethod
    def native_case(num_ssu, mode, min_interval_ms=0, log_limit=5):
        requests = tuple(ContinuousBatchRequest(
            request_id=i, npu_id=i % 32, arrival_time_ms=0 if i < 32 else .01,
            load={"category": "SS", "per_layer_us": 200 + (i % 3) * 100},
            placement=(tuple((block % num_ssu, IO_GIB)
                             for block in range(num_ssu + 2)),))
            for i in range(64))
        controller = MultiSSUController(mode, log_limit=log_limit)
        previous = (native._handle_control, native._handle_compute_schedule,
                    native._mark_layer_io_ready, native._handle_link_completion,
                    native._queue_control_event)
        with multi_ssu_control_events(controller):
            summary = native.simulate_continuous_batch(
                requests, num_npu=32, num_ssu=num_ssu, n_layers=8, batch_size=1,
                qos_configs_by_ssu=qos_configs_from_path_cirs(((0,) * 256,) * num_ssu),
                npu_dedicated_paths=tuple(dedicated_path_id(i) for i in range(32)),
                client_io_config=routing_strategy_specs()[0].client_config(),
                cross_request_layer0_prefetch=True, disk_bw_gbps=40, npu_bw_gbps=50,
                control=CIRControlConfig(callback=controller, on_batch_boundary=True,
                                         min_interval_ms=min_interval_ms))
        assert previous == (native._handle_control, native._handle_compute_schedule,
                            native._mark_layer_io_ready, native._handle_link_completion,
                            native._queue_control_event)
        return summary, controller

    def test_native_32_npu_six_and_seven_ssu_all_invariants(self):
        for num_ssu in (6, 7):
            for mode in ("demand", "deadline", "stall_interchange"):
                with self.subTest(num_ssu=num_ssu, mode=mode):
                    summary, controller = self.native_case(num_ssu, mode)
                    self.assertTrue(all(summary["invariants"].values()),
                                    summary["invariants"])
                    self.assertEqual(len(summary["request_metrics"]), 64)
                    self.assertGreater(controller.evaluations, 5)
                    self.assertEqual(len(controller.decisions), 5)
                    self.assertGreater(controller.ssu_component_done_events, 0)
                    for decision in controller.decisions:
                        self.assert_feasible(decision["grants_gib_s"])

    def test_native_min_interval_and_reserve_strategy(self):
        summary, controller = self.native_case(7, "deadline_reserve", .02,
                                              log_limit=10000)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertFalse(controller.statistics()["hardware_control_latency_modeled"])
        self.assertGreater(controller.statistics()["mean_decision_wall_us"], 0)
        self.assertEqual(len(controller.decisions), controller.evaluations)
        times = [decision["time_ms"] for decision in controller.decisions]
        self.assertTrue(all(b - a >= .02 - 1e-9 for a, b in zip(times, times[1:])))
        self.assertLess(controller.evaluations, controller.ssu_component_done_events)


if __name__ == "__main__":
    unittest.main()
