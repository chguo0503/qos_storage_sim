import math
import random
import unittest
from collections import defaultdict

import sim
from strategy_profiles import CURRENT_STATIC


def reference_uncapped_rates(paths, disk_bw, group_weights):
    assigned = {path: min(path.cir, path.pir) for path in paths}
    remaining = max(0.0, disk_bw - sum(assigned.values()))
    by_group = defaultdict(list)
    for path in paths:
        by_group[path.group_id].append(path)
    group_ids = tuple(by_group)
    group_limits = {
        group_id: sum(
            max(0.0, path.pir - assigned[path])
            for path in by_group[group_id]
        )
        for group_id in group_ids
    }
    group_grants = sim._weighted_capped_split(
        remaining,
        group_ids,
        {group_id: group_weights[group_id] for group_id in group_ids},
        group_limits,
    )
    for group_id, group_paths in by_group.items():
        path_grants = sim._weighted_capped_split(
            group_grants[group_id],
            group_paths,
            {path: path.path_weight for path in group_paths},
            {
                path: max(0.0, path.pir - assigned[path])
                for path in group_paths
            },
        )
        for path, grant in path_grants.items():
            assigned[path] += grant
    return assigned


def flow(path_id, index):
    return sim.BlockIOFlow(
        npu_id=index % 4,
        request_id=index % 4,
        layer=0,
        block_idx=index,
        disk_id=0,
        total_gb=0.001 + (index % 3) * 0.0001,
        queue_id=path_id,
        block_count=1,
        enqueue_time=0.0,
    )


class OptimizationEquivalenceTests(unittest.TestCase):
    def assert_analysis_equal(self, incremental, scanned):
        self.assertEqual(tuple(incremental.counts), tuple(scanned.counts))
        self.assertEqual(incremental.group_io_counts, scanned.group_io_counts)
        self.assertEqual(
            incremental.active_paths_per_group,
            scanned.active_paths_per_group,
        )
        self.assertEqual(
            incremental.active_path_weights,
            scanned.active_path_weights,
        )
        self.assertAlmostEqual(
            incremental.active_group_weight_sum,
            scanned.active_group_weight_sum,
            places=12,
        )
        self.assertAlmostEqual(
            incremental.active_cir_sum,
            scanned.active_cir_sum,
            places=12,
        )

    def test_closed_form_uncapped_rates_match_generic_reference(self):
        rng = random.Random(17)
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0),
            sim.POLICY_QOS_STATIC_CIR,
            sim.DISK_BW,
            CURRENT_STATIC.hardware_config(),
        )
        paths = list(scheduler.paths.values())
        for _ in range(200):
            active = rng.sample(paths, rng.randint(1, len(paths)))
            optimized = sim._static_qos_service_rates(
                active, sim.DISK_BW, scheduler.group_weights
            )
            reference = reference_uncapped_rates(
                active, sim.DISK_BW, scheduler.group_weights
            )
            self.assertTrue(
                all(
                    math.isclose(
                        optimized[path],
                        reference[path],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for path in active
                )
            )

    def test_incremental_pressure_analysis_matches_full_count_scan(self):
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0),
            sim.POLICY_QOS_STATIC_CIR,
            sim.DISK_BW,
            CURRENT_STATIC.hardware_config(),
        )
        for index, path_id in enumerate((0, 33, 66, 99, 132, 165, 198, 231)):
            scheduler.enqueue_many([flow(path_id, index)], 0.0)
            incremental = scheduler.report_path_pressure_analysis(0.0)
            scanned = sim._analyze_qos_counts(
                tuple(path.io_count() for path in scheduler.paths.values()),
                CURRENT_STATIC.hardware_config(),
            )
            self.assert_analysis_equal(incremental, scanned)

        current_time = 0.0
        while scheduler.outstanding_blocks:
            active = scheduler.dispatch(
                current_time, [], schedule_completion=False
            )
            current_time = active.end_time
            scheduler.complete_ready_flows(current_time)
            incremental = scheduler.report_path_pressure_analysis(current_time)
            scanned = sim._analyze_qos_counts(
                tuple(path.io_count() for path in scheduler.paths.values()),
                CURRENT_STATIC.hardware_config(),
            )
            self.assert_analysis_equal(incremental, scanned)
            expected_floor = min(
                (
                    path.virtual_finish
                    for path in scheduler.paths.values()
                    if path.has_work()
                ),
                default=0.0,
            )
            self.assertAlmostEqual(
                scheduler._qos_virtual_floor(), expected_floor, places=12
            )


if __name__ == "__main__":
    unittest.main()
