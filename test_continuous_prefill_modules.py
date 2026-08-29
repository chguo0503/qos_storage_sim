import unittest

from continuous_prefill_workload import _hash_json
from continuous_prefill_client import routing_strategy_specs
from continuous_prefill_workload import prepare_continuous_prefill_workload
import sim


PROFILE_TABLE = {
    (1, 8): (10.0, 1_000.0, 100.0, 0.01),
    (1, 512): (20.0, 2_000.0, 100.0, 0.04),
    (81, 8): (30.0, 3_000.0, 100.0, 0.09),
    (81, 512): (40.0, 4_000.0, 100.0, 0.16),
}

class ContinuousPrefillWorkloadTest(unittest.TestCase):
    def test_formal_category_schedule_arrivals_and_sticky_placement(self):
        workload = prepare_continuous_prefill_workload(
            PROFILE_TABLE,
            num_npu=4,
            batch_size=8,
            new_requests_per_npu=1,
            n_layers=16,
            num_ssu=3,
            seed=42,
        )

        self.assertEqual(len(workload.initial_requests), 32)
        self.assertEqual(len(workload.new_requests), 4)
        for npu_id in range(4):
            categories = [
                request.category
                for request in workload.initial_requests
                if request.npu_id == npu_id
            ]
            self.assertEqual(
                {category: categories.count(category) for category in categories},
                {category: 2 for category in sim.WORKLOAD_CATEGORIES},
            )
        initial_starts = {}
        for request in workload.initial_requests:
            self.assertGreaterEqual(request.arrival_ms, 0.0)
            self.assertLessEqual(request.arrival_ms, 5.0)
            initial_starts.setdefault(request.npu_id, set()).add(
                request.arrival_ms
            )
        self.assertTrue(all(len(starts) == 1 for starts in initial_starts.values()))
        self.assertGreater(
            len({next(iter(starts)) for starts in initial_starts.values()}),
            1,
        )
        self.assertEqual(workload.initial_npu_jitter_ms, (0.0, 5.0))
        self.assertEqual(
            workload.statistics["initial_npu_jitter_window_ms"],
            [0.0, 5.0],
        )
        self.assertTrue(
            all(
                4.0 * workload.t_layer_ms
                <= request.arrival_ms
                <= 12.0 * workload.t_layer_ms
                for request in workload.new_requests
            )
        )
        expanded = workload.expanded_placement()
        for request in workload.requests:
            placement = workload.placement_by_request[request.request_id]
            self.assertTrue(
                all(expanded[request.request_id][layer] is placement for layer in range(16))
            )
            self.assertAlmostEqual(
                sum(block_gb for _, block_gb in placement),
                request.per_layer_kv_gb,
            )

    def test_trace_is_deterministic(self):
        first = prepare_continuous_prefill_workload(
            PROFILE_TABLE, num_npu=2, num_ssu=2, seed=42
        )
        second = prepare_continuous_prefill_workload(
            PROFILE_TABLE, num_npu=2, num_ssu=2, seed=42
        )
        different = prepare_continuous_prefill_workload(
            PROFILE_TABLE, num_npu=2, num_ssu=2, seed=43
        )

        self.assertEqual(first.trace_hash, second.trace_hash)
        self.assertEqual(first.request_dicts(), second.request_dicts())
        self.assertNotEqual(first.trace_hash, different.trace_hash)

    def test_streamed_placement_hash_matches_materialized_reference(self):
        workload = prepare_continuous_prefill_workload(
            PROFILE_TABLE,
            num_npu=3,
            batch_size=2,
            new_requests_per_npu=1,
            n_layers=2,
            num_ssu=5,
            seed=42,
        )
        placement_rows = [
            [request_id, block_index, ssu_id, block_gb]
            for request_id in sorted(workload.placement_by_request)
            for block_index, (ssu_id, block_gb) in enumerate(
                workload.placement_by_request[request_id]
            )
        ]

        self.assertEqual(workload.placement_hash, _hash_json(placement_rows))


class ContinuousPrefillClientTest(unittest.TestCase):
    def test_retained_strategy_configs(self):
        specs = routing_strategy_specs()

        self.assertEqual(
            [spec.name for spec in specs],
            ["baseline", "layer_once", "refresh8"],
        )
        self.assertEqual(
            [spec.pressure_window_io for spec in specs],
            [None, None, 8],
        )
        self.assertEqual(
            [spec.path_selection_mode for spec in specs],
            ["fixed_path_zero", "pressure_aware", "pressure_aware"],
        )


if __name__ == "__main__":
    unittest.main()
