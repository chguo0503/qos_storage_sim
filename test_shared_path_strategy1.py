"""Final Strategy 1 arithmetic and equivalence to the selected dev candidate.

Random inputs here test implementation parity, not workload generalization.
Native end-to-end experiments separately test whether these heuristics help.
"""

import random
import unittest

from experimental_shared_assignment import choose_candidate
from experimental_shared_assignment import prefetch_release_ms as development_release
from shared_path_common import IO_GIB
from shared_path_strategy1 import KnownNPU, choose_npu, prefetch_release_ms


class FairPipelineTests(unittest.TestCase):
    def test_copied_policy_matches_selected_development_candidate_exactly(self):
        rng = random.Random(7216)
        for npu_count, ssu_count in ((4, 2), (32, 6), (32, 7)):
            for _ in range(8):
                lanes = tuple(KnownNPU(
                    100 + npu * 3, rng.random() * 100,
                    tuple(rng.randrange(2000) if rng.random() > .25 else 0
                          for _ in range(ssu_count)), rng.randrange(5))
                    for npu in range(npu_count))
                incoming = tuple(rng.randrange(400) for _ in range(ssu_count))
                snapshots = tuple(tuple(rng.randrange(20) for _ in range(256))
                                  for _ in range(ssu_count))
                args = (rng.random() * 100, lanes, incoming, rng.random() * 20, 8,
                        snapshots, 40.0, 50.0)
                self.assertEqual(choose_npu(*args),
                                 choose_candidate(*args, mode="fair_pipeline"))

    def test_score_is_max_of_compute_disk_and_one_receive_link(self):
        lane = (KnownNPU(3, 1, (1000, 1000)),)
        result = choose_npu(7, lane, (100, 200), 2, n_layers=8)
        work = (1800, 2600)
        compute = 17
        link = 1000 * IO_GIB * sum(work) / 50
        disk = 1000 * IO_GIB * max(work) / 40
        self.assertEqual(result.scores_ms, (7 + max(compute, link, disk),))
        self.assertEqual(result.details[0]["projected_compute_ms"], compute)
        self.assertEqual(result.details[0]["known_work_lanes_by_ssu"], (1, 1))
        self.assertEqual(result.details[0]["receive_link_ms"], link)

    def test_known_future_work_count_is_not_actual_ssd_active_count(self):
        # No SSD active-state input exists. These are already-arrived, unsent
        # workloads, so even a lane with zero current compute counts here.
        lanes = (KnownNPU(8, 0, (1000, 0)), KnownNPU(3, 0, (0, 1000)))
        result = choose_npu(0, lanes, (100, 0), 0, n_layers=1)
        self.assertEqual(result.details[0]["known_work_lanes_by_ssu"], (1, 1))
        self.assertEqual(result.details[1]["known_work_lanes_by_ssu"], (2, 1))
        self.assertFalse(result.details[0]["is_completion_guarantee"])

    def test_ties_use_known_request_count_then_compute_then_actual_id(self):
        lanes = (KnownNPU(6, 0, (0,), 1), KnownNPU(4, 0, (0,), 0),
                 KnownNPU(3, 0, (0,), 0))
        result = choose_npu(0, lanes, (0,), 1)
        self.assertEqual(result.npu_id, 3)
        self.assertEqual(result.scores_ms, (8, 8, 8))

    def test_cached_queue_affects_jit_but_does_not_mask_placement(self):
        lanes = (KnownNPU(3, 1, (10,), 1), KnownNPU(9, 5, (10,), 1))
        low, high = ((0,) * 256,), ((4000,) + (0,) * 255,)
        first = choose_npu(0, lanes, (100,), 1, snapshot_counts_by_ssu=low)
        second = choose_npu(0, lanes, (100,), 1, snapshot_counts_by_ssu=high)
        self.assertEqual(first.npu_id, second.npu_id)
        self.assertEqual(first.scores_ms, second.scores_ms)
        self.assertEqual(second.details[0]["sampled_pending_counts_by_ssu"], (4000,))
        self.assertLess(prefetch_release_ms(0, 100, (100,), high),
                        prefetch_release_ms(0, 100, (100,), low))


class FinalPrefetchTests(unittest.TestCase):
    def test_default_guard_is_ten_ms_and_matches_selected_development_candidate(self):
        args = (3, 100, (100, 200), ((20, 30), (50, 60)))
        self.assertEqual(prefetch_release_ms(*args),
                         development_release(*args, guard_ms=10))
        self.assertAlmostEqual(development_release(*args) - prefetch_release_ms(*args), 5)

    def test_final_release_matches_selected_candidate_across_cached_6_7_ssu_inputs(self):
        rng = random.Random(1067)
        for ssu_count in (6, 7):
            for _ in range(20):
                counts = tuple(rng.randrange(1000) for _ in range(ssu_count))
                snapshot = tuple(tuple(rng.randrange(20) for _ in range(256))
                                 for _ in range(ssu_count))
                now = rng.random() * 100
                args = (now, now + rng.random() * 100, counts, snapshot)
                self.assertEqual(prefetch_release_ms(*args),
                                 development_release(*args, guard_ms=10))

    def test_disk_and_link_estimates_use_max_not_sum(self):
        counts = (1000, 1000)
        link_ms = 1000 * IO_GIB * 2000 / 50
        disk_ms = 1000 * IO_GIB * 1000 / 40
        self.assertGreater(link_ms, disk_ms)
        self.assertAlmostEqual(prefetch_release_ms(0, 100, counts, (0, 0)),
                               100 - link_ms - 10)
        # Extra queued I/O may instead make one disk the bottleneck.
        disk_ms = 1000 * IO_GIB * 5000 / 40
        self.assertAlmostEqual(prefetch_release_ms(0, 100, counts, ((4000,), (0,))),
                               100 - disk_ms - 10)

    def test_now_clamp_larger_guard_and_empty_coflow(self):
        self.assertEqual(prefetch_release_ms(10, 9, (1,), (0,)), 10)
        self.assertEqual(prefetch_release_ms(10, 20, (1000,), (10000,)), 10)
        self.assertEqual(prefetch_release_ms(10, 100, (0,), (10000,)), 10)
        args = (0, 100, (100,), (0,))
        self.assertAlmostEqual(prefetch_release_ms(*args, guard_ms=5)
                               - prefetch_release_ms(*args, guard_ms=10), 5)


if __name__ == "__main__":
    unittest.main()
