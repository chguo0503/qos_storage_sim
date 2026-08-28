import unittest

from continuous_prefill_workload import _hash_json
from continuous_prefill_client import (
    ActiveRequestSnapshot,
    SCHEME_B_MODES,
    SchemeBController,
    legacy_strategy_specs,
)
from continuous_prefill_workload import prepare_continuous_prefill_workload
import sim


PROFILE_TABLE = {
    (1, 8): (10.0, 1_000.0, 100.0, 0.01),
    (1, 512): (20.0, 2_000.0, 100.0, 0.04),
    (81, 8): (30.0, 3_000.0, 100.0, 0.09),
    (81, 512): (40.0, 4_000.0, 100.0, 0.16),
}


def snapshot(request_id, npu_id, arrival_ms, compute_s, work):
    return ActiveRequestSnapshot(
        request_id=request_id,
        npu_id=npu_id,
        arrival_ms=arrival_ms,
        per_layer_us=compute_s * 1e6,
        work_by_ssu_gb=tuple(work),
    )


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
        specs = legacy_strategy_specs()

        self.assertEqual(
            [spec.name for spec in specs],
            ["baseline", "path_rr", "layer_once", "refresh8", "refresh1"],
        )
        self.assertEqual(
            [spec.pressure_window_io for spec in specs],
            [None, None, None, 8, 1],
        )

    def test_scheme_b_exact_ten_thirty_and_capacity(self):
        controller = SchemeBController(
            num_npu=2,
            num_ssu=1,
            t_layer_ms=10.0,
            mode="once",
        )
        commit = controller.observe(
            0.0,
            (
                snapshot(0, 0, 0.0, 1.0, (10.0,)),
                snapshot(1, 1, 0.0, 1.0, (30.0,)),
            ),
        )

        self.assertEqual(commit.target.demands_gbps, ((10.0,), (30.0,)))
        self.assertEqual(commit.target.grants_gbps, ((10.0,), (30.0,)))
        self.assertEqual(commit.path_write_count, 2)
        self.assertAlmostEqual(sum(commit.target.path_cirs_by_ssu[0]), 40.0)
        self.assertEqual(controller.next_update_time_ms, None)

    def test_periodic_waits_for_layer_equivalent_deadline(self):
        controller = SchemeBController(
            num_npu=2,
            num_ssu=1,
            t_layer_ms=10.0,
            mode="periodic2",
        )
        first = (snapshot(0, 0, 0.0, 1.0, (10.0,)),)
        changed = first + (snapshot(1, 1, 5.0, 1.0, (30.0,)),)

        self.assertIsNotNone(controller.observe(0.0, first))
        self.assertEqual(controller.next_update_time_ms, 20.0)
        self.assertIsNone(controller.observe(5.0, changed))
        commit = controller.observe(20.0, changed)
        self.assertIsNotNone(commit)
        self.assertEqual(commit.reason, "periodic2")
        self.assertEqual(controller.next_update_time_ms, 40.0)

    def test_membership_event_coalesces_for_one_ms(self):
        controller = SchemeBController(
            num_npu=3,
            num_ssu=1,
            t_layer_ms=10.0,
            mode="membership_event",
            membership_debounce_ms=1.0,
        )
        first = (snapshot(0, 0, 0.0, 1.0, (10.0,)),)
        second = first + (snapshot(1, 1, 5.0, 1.0, (10.0,)),)
        third = second + (snapshot(2, 2, 5.5, 1.0, (10.0,)),)

        controller.observe(0.0, first)
        self.assertIsNone(controller.observe(5.0, second))
        self.assertEqual(controller.next_update_time_ms, 6.0)
        self.assertIsNone(controller.observe(5.5, third))
        self.assertEqual(controller.next_update_time_ms, 6.0)
        commit = controller.observe(6.0, third)
        self.assertIsNotNone(commit)
        self.assertEqual(commit.reason, "membership_event")
        self.assertEqual(controller.next_update_time_ms, None)

    def test_future_arrival_is_not_visible(self):
        controller = SchemeBController(
            num_npu=1,
            num_ssu=1,
            t_layer_ms=10.0,
            mode="once",
        )

        with self.assertRaises(ValueError):
            controller.observe(
                0.0,
                (snapshot(0, 0, 1.0, 1.0, (1.0,)),),
            )

    def test_all_update_modes_construct(self):
        for mode in SCHEME_B_MODES:
            controller = SchemeBController(
                num_npu=1,
                num_ssu=1,
                t_layer_ms=10.0,
                mode=mode,
            )
            self.assertIsNotNone(
                controller.observe(
                    0.0,
                    (snapshot(0, 0, 0.0, 1.0, (1.0,)),),
                )
            )


if __name__ == "__main__":
    unittest.main()
