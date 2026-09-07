"""Pure arithmetic/causality checks, not claims of simulator improvement."""

import json
import math
import unittest
from functools import partial

from experimental_shared_assignment import choose_candidate, prefetch_release_ms
from shared_path_common import IO_GIB
from shared_path_strategy1 import AssignmentChoice, KnownNPU


MODES = ("compute", "fair_pipeline", "compute_guarded_mix")


class CandidateTests(unittest.TestCase):
    def test_compute_uses_all_known_compute_and_actual_npu_id(self):
        lanes = (KnownNPU(41, 9, (1, 1), 1), KnownNPU(17, 2, (9000, 9000), 2))
        choice = choose_candidate(100, lanes, (300, 200), 3, n_layers=8, mode="compute")
        self.assertIsInstance(choice, AssignmentChoice)
        self.assertEqual(choice.npu_id, 17)
        self.assertEqual(choice.scores_ms, (133, 126))

    def test_partial_matches_frozen_strategy_call_signature(self):
        lanes = (KnownNPU(9, 0, (0, 0)), KnownNPU(6, 0, (0, 0)))
        for mode in MODES:
            choose = partial(choose_candidate, mode=mode)
            result = choose(3, lanes, (100, 100), 1, 8, ((4, 7), (1, 6)), 40, 50)
            self.assertEqual(result.npu_id, 6)
            self.assertEqual(len(result.details), 2)
            self.assertTrue(all(math.isfinite(score) for score in result.scores_ms))
            json.dumps(result.details, allow_nan=False)

    def test_fair_pipeline_accounts_for_one_shared_receive_link(self):
        lanes = (KnownNPU(0, 0, (0, 0)),)
        result = choose_candidate(7, lanes, (1000, 1000), 0, n_layers=1)
        disk_ms = 1000 * IO_GIB * 1000 / 40
        link_ms = 1000 * IO_GIB * 2000 / 50
        self.assertGreater(link_ms, disk_ms)
        self.assertAlmostEqual(result.scores_ms[0], 7 + link_ms)
        self.assertEqual(result.details[0]["known_work_lanes_by_ssu"], (1, 1))

    def test_fair_pipeline_ssu_hotspot_and_known_lane_count(self):
        lanes = (KnownNPU(7, 0, (1000, 0)), KnownNPU(9, 0, (0, 500)))
        result = choose_candidate(0, lanes, (100, 0), 0, n_layers=1)
        self.assertEqual(result.npu_id, 9)
        self.assertEqual(result.details[0]["known_work_lanes_by_ssu"], (1, 1))
        self.assertEqual(result.details[1]["known_work_lanes_by_ssu"], (2, 1))
        self.assertAlmostEqual(result.scores_ms[0], 1000 * IO_GIB * 1100 / 40)
        self.assertAlmostEqual(result.scores_ms[1], 1000 * IO_GIB * 500 / 40)
        self.assertAlmostEqual(result.details[1]["fair_disk_ms_by_ssu"][0],
                               1000 * IO_GIB * 100 * 2 / 40)

    def test_fair_pipeline_compute_can_hide_estimated_io(self):
        lanes = (KnownNPU(2, 10, (10, 10)),)
        result = choose_candidate(5, lanes, (100, 100), 3, n_layers=8)
        self.assertEqual(result.scores_ms, (39,))
        self.assertEqual(result.details[0]["projected_compute_ms"], 34)

    def test_guard_excludes_lower_io_score_if_compute_backlog_too_large(self):
        lanes = (KnownNPU(2, 0, (100000, 0)), KnownNPU(6, 10, (0, 0)))
        result = choose_candidate(0, lanes, (1, 0), 1, n_layers=1,
                                  mode="compute_guarded_mix")
        self.assertLess(result.scores_ms[1], result.scores_ms[0])
        self.assertEqual(result.npu_id, 2)
        self.assertTrue(result.details[0]["eligible"])
        self.assertFalse(result.details[1]["eligible"])
        self.assertEqual(result.details[1]["compute_guard_limit_ms"], 1)

    def test_guard_boundary_is_inclusive_and_uses_one_layer_not_whole_request(self):
        lanes = (KnownNPU(2, 0, (100000, 0)), KnownNPU(6, 1, (0, 0)),
                 KnownNPU(8, 2, (0, 0)))
        result = choose_candidate(0, lanes, (1, 0), 1, n_layers=8,
                                  mode="compute_guarded_mix")
        self.assertEqual(result.npu_id, 6)
        self.assertEqual(tuple(row["eligible"] for row in result.details),
                         (True, True, False))

    def test_guarded_mix_projected_demands_follow_documented_formula(self):
        lanes = (KnownNPU(2, 0, (1000,)), KnownNPU(6, 0, (1000,)))
        result = choose_candidate(2, lanes, (1000,), 0, n_layers=1,
                                  mode="compute_guarded_mix")
        own_pipeline = 1000 * IO_GIB * 2000 / 40
        self.assertAlmostEqual(result.details[0]["disk_contention_factor"], 2)
        self.assertAlmostEqual(result.details[0]["projected_demand_gib_s_by_ssu"][0], 80)
        self.assertAlmostEqual(result.scores_ms[0], 2 + own_pipeline * 2)

    def test_sampled_common_queue_does_not_mask_candidate_load_or_mutate_inputs(self):
        lanes = (KnownNPU(8, 9, (100, 30), 1), KnownNPU(3, 1, (20, 300), 1))
        original = tuple(lanes)
        small = ((0, 0), (0, 0))
        large = ((1000000, 2000000), (3000000, 4000000))
        for mode in MODES:
            first = choose_candidate(2, lanes, (64, 64), 1, snapshot_counts_by_ssu=small,
                                     mode=mode)
            second = choose_candidate(2, lanes, (64, 64), 1, snapshot_counts_by_ssu=large,
                                      mode=mode)
            self.assertEqual(first.npu_id, second.npu_id)
            self.assertEqual(first.scores_ms, second.scores_ms)
            self.assertEqual(second.details[0]["sampled_pending_counts_by_ssu"],
                             (3000000, 7000000))
            self.assertFalse(second.details[0]["is_completion_guarantee"])
        self.assertEqual(lanes, original)

    def test_invalid_mode_is_not_silently_another_policy(self):
        with self.assertRaises(ValueError):
            choose_candidate(0, (KnownNPU(0, 0, (0,)),), (1,), 1, mode="unknown")


class PrefetchReleaseTests(unittest.TestCase):
    def test_cached_path_rows_and_ssu_totals_are_equivalent(self):
        nested = prefetch_release_ms(0, 100, (100, 200), ((50, 70), (80, 90)))
        flat = prefetch_release_ms(0, 100, (100, 200), (120, 170))
        self.assertEqual(nested, flat)
        estimated = max(1000 * IO_GIB * 370 / 40, 1000 * IO_GIB * 300 / 50)
        self.assertAlmostEqual(nested, 100 - estimated - 5)

    def test_uses_max_parallel_disks_vs_single_link_not_sum(self):
        disk_component = 1000 * IO_GIB * 1000 / 40
        link_component = 1000 * IO_GIB * 2000 / 50
        release = prefetch_release_ms(0, 100, (1000, 1000), (0, 0), guard_ms=0)
        self.assertGreater(link_component, disk_component)
        self.assertAlmostEqual(release, 100 - link_component)
        self.assertNotAlmostEqual(release, 100 - 2 * disk_component - link_component)

    def test_congested_or_overdue_coflow_is_released_now(self):
        self.assertEqual(prefetch_release_ms(10, 20, (1000,), (10000,)), 10)
        self.assertEqual(prefetch_release_ms(10, 9, (1,), (0,)), 10)

    def test_guard_ten_releases_five_ms_earlier_than_guard_five(self):
        args = (0, 100, (100,), ((20, 30),))
        self.assertAlmostEqual(prefetch_release_ms(*args, guard_ms=5)
                               - prefetch_release_ms(*args, guard_ms=10), 5)

    def test_custom_bandwidths_keep_units_gib_per_second_and_ms(self):
        release = prefetch_release_ms(0, 200, (1000,), (3000,), guard_ms=7,
                                      disk_bw_gib_s=20, link_bw_gib_s=25)
        self.assertAlmostEqual(release, 200 - 1000 * IO_GIB * 4000 / 20 - 7)

    def test_empty_coflow_needs_no_delay(self):
        self.assertEqual(prefetch_release_ms(7, 100, (0, 0), (100, 200)), 7)


if __name__ == "__main__":
    unittest.main()
