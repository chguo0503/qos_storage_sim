"""Pure-policy tests; no simulator state, future queue, or physical experiment."""

from dataclasses import replace
import unittest

from policy_logic import category_path_ids, layer_once_path_ids
from shared_path_baseline import baseline_path_ids
from shared_path_common import IO_BYTES, IO_GIB, pressure_from_counts, solo_io_ms
from shared_path_new_once import new_once_path_ids
from shared_path_once import once_path_ids
from shared_path_strategy1 import KnownNPU, choose_npu
from shared_path_strategy2 import PendingIO, reorder_pending
from strategy_profiles import FINAL_STATIC


class SharedRoutingTests(unittest.TestCase):
    def setUp(self):
        self.qos = FINAL_STATIC.hardware_config()
        self.empty = pressure_from_counts((0,) * 256, self.qos)
        self.allowed = category_path_ids("SS", self.qos)

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

    def test_new_once_matches_once_when_ledger_and_snapshot_agree(self):
        ledger = tuple((p * 7) % 9 for p in range(256))
        snapshot = pressure_from_counts(ledger, self.qos)
        self.assertEqual(new_once_path_ids(12, snapshot, ledger, self.allowed, self.qos),
                         once_path_ids(12, snapshot, self.allowed, self.qos))

    def test_reservations_coordinate_clients_using_same_old_snapshot(self):
        first = once_path_ids(8, self.empty, self.allowed, self.qos)
        repeated = once_path_ids(8, self.empty, self.allowed, self.qos)
        self.assertEqual(first, repeated)
        ledger = [0] * 256
        for path in first:
            ledger[path] += 1
        before = tuple(ledger)
        coordinated = new_once_path_ids(8, self.empty, ledger, self.allowed, self.qos)
        self.assertNotEqual(first, coordinated)
        self.assertTrue(set(first).isdisjoint(coordinated))
        self.assertEqual(tuple(ledger), before)  # caller performs the atomic commit

    def test_hbm_ack_releases_old_snapshot_ghost_work(self):
        old_counts = [0] * 256
        old_counts[0] = 100
        stale = pressure_from_counts(old_counts, self.qos)
        ledger = [0] * 256  # all formerly sampled work now HBM-acknowledged
        original = once_path_ids(1, stale, (0, 32), self.qos)
        current = new_once_path_ids(1, stale, ledger, (0, 32), self.qos)
        self.assertEqual(original, (32,))
        self.assertEqual(current, (0,))

    def test_ledger_view_rebuilds_rates_as_well_as_counts(self):
        ledger = [0] * 256
        ledger[0], ledger[1], ledger[32] = 2, 3, 4
        view = pressure_from_counts(ledger, self.qos)
        self.assertEqual(view.group_io_counts[:2], (5, 4))
        self.assertEqual(view.active_paths_per_group[:2], (2, 1))
        self.assertEqual(view.active_path_weights[:2], (2, 1))
        self.assertEqual(view.active_group_weight_sum, 2)
        self.assertAlmostEqual(view.active_cir_sum, .625)

    def test_shared_pool_can_accept_more_than_256_client_reservations(self):
        ledger = [0] * 256
        for _client in range(300):
            selected = new_once_path_ids(1, self.empty, ledger, self.allowed, self.qos)
            ledger[selected[0]] += 1
        self.assertEqual(sum(ledger), 300)
        self.assertTrue(all(p in self.allowed for p, count in enumerate(ledger) if count))

    def test_negative_or_wrong_ssu_ledger_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same SSU"):
            new_once_path_ids(1, self.empty, (0,) * 255, self.allowed, self.qos)
        ledger = [0] * 256
        ledger[0] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            new_once_path_ids(1, self.empty, ledger, self.allowed, self.qos)


class SharedAssignmentTests(unittest.TestCase):
    def test_only_known_backlog_and_ids_need_not_be_indices(self):
        backlogs = (KnownNPU(8, 10, (0, 0), 1), KnownNPU(3, 3, (0, 0), 1))
        result = choose_npu(5, backlogs, (0, 0), 1)
        self.assertEqual(result.npu_id, 3)
        self.assertEqual(result.scores_ms, (23, 16))
        self.assertEqual(backlogs[0].remaining_compute_ms, 10)

    def test_receive_and_disk_work_can_outweigh_small_compute_backlog(self):
        backlogs = (KnownNPU(0, 1, (50000, 50000), 1), KnownNPU(1, 2, (0, 0), 1))
        result = choose_npu(0, backlogs, (1, 1), 1)
        self.assertEqual(result.npu_id, 1)
        self.assertGreater(result.details[0]["receive_link_ms"], 300)
        self.assertAlmostEqual(solo_io_ms((50000, 50000)),
                               100000 * IO_GIB * 1000 / 50)

    def test_coflow_known_work_sharing_can_change_compute_only_choice(self):
        backlogs = (KnownNPU(0, 3, (600,), 1), KnownNPU(1, 2, (0,), 1))
        result = choose_npu(0, backlogs, (400,), 1)
        self.assertEqual(result.npu_id, 0)
        self.assertGreater(result.details[1]["known_work_lanes_by_ssu"][0],
                           result.details[0]["known_work_lanes_by_ssu"][0])
        self.assertTrue(all(not d["is_completion_guarantee"] for d in result.details))

    def test_periodic_counts_are_recorded_not_a_common_assignment_floor(self):
        backlogs = (KnownNPU(0, 100, (1000,), 1),)
        without = choose_npu(0, backlogs, (1,), 1)
        with_counts = choose_npu(0, backlogs, (1,), 1,
                                 snapshot_counts_by_ssu=((1000,) + (0,) * 255,))
        self.assertEqual(with_counts.details[0]["sampled_pending_counts_by_ssu"], (1000,))
        self.assertEqual(with_counts.scores_ms, without.scores_ms)


class SamePathReorderTests(unittest.TestCase):
    def setUp(self):
        self.a = PendingIO(10, 0, 1, 2, 7, 0, 0, 10, 100)
        self.b = PendingIO(11, 1, 2, 1, 7, 0, .1, 1, 200)

    def test_only_pending_ids_reordered_by_carried_deadline(self):
        queue = (self.a, self.b)
        self.assertEqual(reorder_pending(queue, .5), (11, 10))
        self.assertEqual(queue, (self.a, self.b))
        self.assertEqual(reorder_pending((), .5), ())

    def test_component_size_is_only_a_tie_break_not_future_progress(self):
        a = replace(self.a, deadline_ms=1, layer_io_count=10)
        self.assertEqual(reorder_pending((self.b, a), .5), (10, 11))
        # Equal metadata is ordered by stable enqueue time/ID, not object ID.
        b = replace(a, io_id=11)
        self.assertEqual(reorder_pending((b, a), .5), (10, 11))

    def test_cross_path_cross_ssu_and_future_inputs_are_rejected(self):
        for other in (replace(self.b, path_id=8), replace(self.b, ssu_id=1)):
            with self.assertRaisesRegex(ValueError, "one SSU"):
                reorder_pending((self.a, other), 1)
        with self.assertRaisesRegex(ValueError, "future"):
            reorder_pending((self.a, self.b), .05)


if __name__ == "__main__":
    unittest.main()
