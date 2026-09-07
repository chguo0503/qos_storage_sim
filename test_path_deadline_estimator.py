from __future__ import annotations

import unittest

from path_deadline_estimator import (
    BYTES_PER_GIB,
    estimate_fcfs_path_deadline,
    estimate_uniform_fcfs_path_deadline,
)


class PathDeadlineEstimatorTest(unittest.TestCase):
    def test_bandwidth_only_uses_total_bytes_and_reports_positive_slack(self):
        required_gib = 0.0011211633682250977
        result = estimate_fcfs_path_deadline(
            path_bytes_ahead=0,
            required_bytes=round(required_gib * BYTES_PER_GIB),
            time_budget_ms=0.5821494916324586,
        )
        self.assertTrue(result.meets_deadline)
        self.assertAlmostEqual(result.estimated_ssd_completion_ms, 0.02802908420562744)
        self.assertAlmostEqual(result.early_by_ms, 0.5541204074268312)
        self.assertEqual(result.shortfall_ms, 0.0)

    def test_backlog_can_turn_the_same_small_layer_into_a_miss(self):
        required_gib = 0.0011211633682250977
        blocking_gib = 0.25732994079589844 + 2 * 0.2566375732421875
        result = estimate_fcfs_path_deadline(
            path_bytes_ahead=round(blocking_gib * BYTES_PER_GIB),
            required_bytes=round(required_gib * BYTES_PER_GIB),
            time_budget_ms=0.5821494916324586,
        )
        self.assertFalse(result.meets_deadline)
        self.assertGreater(result.shortfall_ms, 18.0)
        self.assertEqual(result.early_by_ms, 0.0)

    def test_4k_commands_are_iops_bound_at_100k_iops(self):
        result = estimate_uniform_fcfs_path_deadline(
            path_io_ahead=100,
            required_io_count=10,
            io_size_bytes=4096,
            time_budget_ms=1.0,
            bandwidth_gib_s=40.0,
            iops_limit=100_000.0,
        )
        self.assertAlmostEqual(result.bandwidth_bound_ms, 0.01049041748046875)
        self.assertAlmostEqual(result.iops_bound_ms, 1.1)
        self.assertAlmostEqual(result.estimated_ssd_completion_ms, 1.1)
        self.assertFalse(result.meets_deadline)
        self.assertAlmostEqual(result.shortfall_ms, 0.1)

    def test_count_dependent_model_requires_counts(self):
        with self.assertRaisesRegex(ValueError, "required when using"):
            estimate_fcfs_path_deadline(
                path_bytes_ahead=4096,
                required_bytes=4096,
                time_budget_ms=1.0,
                iops_limit=100_000.0,
            )

    def test_uniform_wrapper_requires_a_nonzero_size_for_nonempty_io(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            estimate_uniform_fcfs_path_deadline(
                path_io_ahead=1,
                required_io_count=1,
                io_size_bytes=0,
                time_budget_ms=1.0,
            )


if __name__ == "__main__":
    unittest.main()
