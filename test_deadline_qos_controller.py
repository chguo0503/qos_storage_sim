import unittest
from unittest.mock import patch

from deadline_qos_controller import (
    PendingLayer, DeadlineController, choose_deadline_rates, deadline_control_events,
)


class DeadlineTests(unittest.TestCase):
    def test_earliest_deadline_not_largest_work(self):
        jobs = (PendingLayer(0, 1, 1, 2, 1, 1), PendingLayer(1, 2, 1, 800, 2, 3))
        self.assertEqual(choose_deadline_rates(jobs, 0), (40, 0, 0, 0))
        self.assertEqual(choose_deadline_rates(jobs, 0, mode="least_slack"), (0, 40, 0, 0))

    def test_ready_ignored_and_zero_when_idle(self):
        self.assertEqual(choose_deadline_rates((), 1), (0, 0, 0, 0))
        self.assertEqual(choose_deadline_rates((PendingLayer(0, 1, 1, 0, 0, 1),), 1), (0, 0, 0, 0))

    def test_real_sim_hooks_preserve_core_and_restore(self):
        import continuous_batch_sim as native
        from run_stall_policy_experiments import Profile, run_case
        old = native._handle_compute_schedule, native._handle_control, native._mark_layer_io_ready
        controller = DeadlineController()
        with patch("run_stall_policy_experiments.ManifestCIRController", return_value=controller), \
             deadline_control_events(controller):
            result = run_case((Profile(8, 0.3), Profile(32, 0.6), Profile(128, 2), Profile(256, 4)),
                              "dedicated_demand", finite_count=2)
        self.assertEqual(old, (native._handle_compute_schedule, native._handle_control, native._mark_layer_io_ready))
        self.assertTrue(all(result["summary"]["invariants"].values()))
        self.assertTrue(controller.decisions)
        self.assertTrue(any(d["jobs"] for d in controller.decisions))
        self.assertTrue(all(sum(d["cirs"]) <= 40 for d in controller.decisions))


if __name__ == "__main__":
    unittest.main()
