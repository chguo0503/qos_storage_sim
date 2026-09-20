"""Routing rules used by the two retained Baseline / Once experiments."""

import unittest

from policy_logic import category_path_ids, layer_once_path_ids
from shared_path_baseline import baseline_path_ids
from shared_path_common import IO_BYTES, IO_GIB, pressure_from_counts
from shared_path_once import once_path_ids
from strategy_profiles import FINAL_STATIC


class SharedRoutingTests(unittest.TestCase):
    def setUp(self):
        self.qos = FINAL_STATIC.hardware_config()

    def test_fixed_io_units_and_baseline_without_state(self):
        self.assertEqual(IO_BYTES, 180224)
        self.assertEqual(baseline_path_ids(4), (0, 0, 0, 0))
        self.assertEqual(baseline_path_ids(0), ())

    def test_once_is_byte_identical_to_established_routing_decision(self):
        counts = tuple((p * 13) % 21 for p in range(256))
        snapshot = pressure_from_counts(counts, self.qos)
        for category in ("SS", "SL", "LS", "LL"):
            allowed = category_path_ids(category, self.qos)
            expected = layer_once_path_ids((IO_GIB,) * 17, snapshot, allowed,
                                           self.qos, start_offset=11)
            actual = once_path_ids(17, snapshot, allowed, self.qos, start_offset=11)
            self.assertEqual(actual, expected)
            self.assertTrue(set(actual) <= set(allowed))
        self.assertEqual(snapshot.counts, counts)

    def test_pressure_view_rebuilds_rates_as_well_as_counts(self):
        ledger = [0] * 256
        ledger[0], ledger[1], ledger[32] = 2, 3, 4
        view = pressure_from_counts(ledger, self.qos)
        self.assertEqual(view.group_io_counts[:2], (5, 4))
        self.assertEqual(view.active_paths_per_group[:2], (2, 1))
        self.assertEqual(view.active_path_weights[:2], (2, 1))
        self.assertEqual(view.active_group_weight_sum, 2)
        self.assertAlmostEqual(view.active_cir_sum, .625)


if __name__ == "__main__":
    unittest.main()
