"""Policy-boundary checks; run directly with python this_file.py."""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

from policy_logic import category_path_ids, hardware_view
from shared_path_common import pressure_from_counts
from shared_path_once import once_path_ids
from strategy_profiles import FINAL_STATIC
from slo_router import RequestBudget, RouterConfig, VARIANTS, choose_path_pool, slo_path_ids


class SloRoutingTests(unittest.TestCase):
    def setUp(self):
        self.qos = hardware_view(FINAL_STATIC.hardware_config())
        self.snapshot = pressure_from_counts((0,) * self.qos.path_count, self.qos)
        self.allowed = category_path_ids("SL", self.qos)
        self.slack = RequestBudget(342.0, 228.0, 8, (.0125, .0125, .0125))
        self.urgent = RequestBudget(238.0, 228.0, 8, (.0125, .0125, .0125))

    def test_all_categories_keep_only_legal_paths_and_cover_all_groups(self):
        for category in self.qos.category_labels:
            allowed = category_path_ids(category, self.qos)
            for config in VARIANTS.values():
                decision = choose_path_pool(allowed, self.qos, budget=self.slack, config=config)
                self.assertTrue(set(decision.path_ids) <= set(allowed))
                self.assertEqual({p // self.qos.paths_per_group for p in decision.path_ids}, set(range(8)))
                self.assertEqual(len(decision.path_ids), 8 * config.slack_paths_per_group)

    def test_urgency_changes_service_opportunity_without_illegal_routes(self):
        relaxed = slo_path_ids(64, self.snapshot, self.allowed, self.qos, budget=self.slack)
        urgent = slo_path_ids(64, self.snapshot, self.allowed, self.qos, budget=self.urgent)
        self.assertEqual(len(set(relaxed)), 8)
        self.assertEqual(len(set(urgent)), 32)
        self.assertNotEqual(relaxed, urgent)
        self.assertTrue(set(relaxed + urgent) <= set(self.allowed))

    def test_urgent_and_unknown_budget_are_exact_original_once(self):
        counts = tuple((p * 7 + 3) % 11 for p in range(256))
        snapshot = pressure_from_counts(counts, self.qos)
        expected = once_path_ids(47, snapshot, self.allowed, self.qos, start_offset=13)
        for budget in (self.urgent, None, RequestBudget(100., 228., 8, (.0125,) * 3)):
            actual = slo_path_ids(47, snapshot, self.allowed, self.qos, budget=budget, start_offset=13)
            self.assertEqual(actual, expected)

    def test_counts_qos_and_budget_are_unchanged(self):
        before = (self.snapshot, self.qos, self.slack)
        first = slo_path_ids(64, self.snapshot, self.allowed, self.qos, budget=self.slack)
        second = slo_path_ids(64, self.snapshot, self.allowed, self.qos, budget=self.slack)
        self.assertEqual(before, (self.snapshot, self.qos, self.slack))
        self.assertEqual(first, second)

    def test_real_a_b_profiles_select_different_modes(self):
        for category, v, c, expected in (
            ("LS", .171539306640625, 6.024011548589167, "urgent_full_pool"),
            ("SL", .03759765625, 28.592841995880783, "slack_shared_prefix"),
        ):
            budget = RequestBudget(1.5 * 8 * c, 8 * c, 8, (v / 3,) * 3)
            decision = choose_path_pool(category_path_ids(category, self.qos), self.qos, budget=budget)
            self.assertEqual(decision.mode, expected)

    def test_invalid_config_rejected_and_empty_io_is_safe(self):
        with self.assertRaises(ValueError):
            RouterConfig(slack_paths_per_group=0)
        self.assertEqual(slo_path_ids(0, self.snapshot, (), self.qos, budget=None), ())


if __name__ == "__main__":
    unittest.main()
