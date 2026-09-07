import unittest
from types import SimpleNamespace

from layer_budget_controller import LayerBudgetController, allocate_utilization_rates


class BudgetTests(unittest.TestCase):
    def test_under_capacity_is_unchanged(self):
        self.assertEqual(allocate_utilization_rates([1, 2, 20, 10]), (1, 2, 20, 10))

    def test_floor_capacity_and_no_overallocation(self):
        demands = (2, 8, 15, 30)
        rates = allocate_utilization_rates(demands)
        self.assertAlmostEqual(sum(rates), 40)
        for r, d in zip(rates, demands):
            self.assertGreaterEqual(r, 0.5 * 40 * d / sum(demands))
            self.assertLessEqual(r, d)
        self.assertGreater(sum(r / d for r, d in zip(rates, demands)), 4 * 40 / sum(demands))

    def test_full_floor_equals_proportional(self):
        demands = (2, 8, 15, 30)
        rates = allocate_utilization_rates(demands, floor_fraction=1)
        for r, d in zip(rates, demands):
            self.assertAlmostEqual(r, 40 * d / sum(demands))

    def test_zero_demand_no_division_or_grant(self):
        self.assertEqual(allocate_utilization_rates([0, 0, 0, 0]), (0, 0, 0, 0))
        self.assertEqual(allocate_utilization_rates([0, 0, 0, 80]), (0, 0, 0, 40))

    def test_prefetch_uses_previous_compute_budget(self):
        def request(rid, work, compute, prefetch=False):
            return SimpleNamespace(request_id=rid, npu_id=0, per_layer_compute_ms=compute,
                                   next_layer_work_gb_by_ssu=(work,), prefetch_only=prefetch)
        snapshot = SimpleNamespace(num_npu=4, time_ms=1,
                                   active_requests=(request(1, 0.001, 1), request(2, 0.01, 10, True)))
        controller = LayerBudgetController()
        output = controller(snapshot)
        self.assertAlmostEqual(output.path_cirs_by_ssu[0][0], 10)
        self.assertEqual(controller.decisions[0]["budget"], "transition")


if __name__ == "__main__":
    unittest.main()
