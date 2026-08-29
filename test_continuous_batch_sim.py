import unittest

import sim
from continuous_batch_sim import (
    CIRControlSnapshot,
    ContinuousBatchRequest,
    ControlRequestView,
    MaxMinSchemeBController,
    SteadyStateConfig,
    _steady_block_bounds,
    simulate_continuous_batch,
)
from continuous_prefill_client import static_qos_config
from policy_logic import ManifestDemand, plan_scheme_b
from scheme_b_prefill import PATH_COUNT, dedicated_path_id


def request(request_id, npu_id, arrival_ms=0.0, layers=3, sequence=0):
    return ContinuousBatchRequest(
        request_id=request_id,
        npu_id=npu_id,
        arrival_time_ms=arrival_ms,
        load={
            "request_id": request_id,
            "npu_id": npu_id,
            "stream_id": sequence,
            "category": "SS",
            "per_layer_us": 100_000.0,
            "initial": arrival_ms == 0.0,
        },
        placement=(((0, 0.8),),) * layers,
    )


class ContinuousBatchSimTest(unittest.TestCase):
    def test_steady_blocks_ignore_floating_point_window_drift(self):
        start_ms = 19_403.123456789
        end_ms = start_ms + 2_000.0
        blocks = _steady_block_bounds(start_ms, end_ms, 2_000.0, 500.0)

        self.assertEqual(len(blocks), 4)
        self.assertEqual([duration for _, _, duration in blocks], [500.0] * 4)
        self.assertEqual(blocks[-1][1], end_ms)

    def test_sticky_manifest_preserves_layer_order_float_accumulation(self):
        sticky = ContinuousBatchRequest(
            request_id=0,
            npu_id=0,
            arrival_time_ms=0.0,
            load={
                "request_id": 0,
                "npu_id": 0,
                "category": "SS",
                "per_layer_us": 100_000.0,
                "initial": True,
            },
            placement=(((0, 0.1),),),
        )
        summary = simulate_continuous_batch(
            (sticky,),
            num_npu=1,
            num_ssu=1,
            n_layers=16,
            batch_size=1,
            qos_config=static_qos_config(),
        )
        expected_read_gb = 0.0
        for _ in range(16):
            expected_read_gb += sum(size_gb for _, size_gb in sticky.placement[0])

        self.assertEqual(summary["expected_read_gb"], expected_read_gb)
        self.assertEqual(summary["completed_read_gb"], expected_read_gb)

    def test_runtime_max_min_matches_authoritative_scheme_b(self):
        paths = (dedicated_path_id(0),)
        controller = MaxMinSchemeBController(paths)
        views = (ControlRequestView(
            0, 0, "SS", 1_000.0, -1, 1, (10.0, 100.0), True
        ),)
        snapshot = CIRControlSnapshot(
            time_ms=0.0,
            evaluation=1,
            layer_jobs_since_previous=0,
            num_npu=1,
            num_ssu=2,
            active_requests=views,
            current_path_cirs_by_ssu=((0.0,) * PATH_COUNT,) * 2,
        )
        decision = controller(snapshot)
        target = plan_scheme_b(
            (ManifestDemand(0, 0, 1.0, (10.0, 100.0)),),
            num_npu=1,
            num_ssu=2,
            path_by_npu=paths,
        )
        self.assertEqual(decision.path_cirs_by_ssu, target.path_cirs_by_ssu)
        self.assertAlmostEqual(decision.path_cirs_by_ssu[0][paths[0]], 10.0)
        self.assertAlmostEqual(decision.path_cirs_by_ssu[1][paths[0]], 40.0)

    def test_end_to_end_conservation(self):
        summary = simulate_continuous_batch(
            (request(0, 0), request(1, 0, 50.0)),
            num_npu=1,
            num_ssu=1,
            n_layers=3,
            batch_size=1,
            qos_config=static_qos_config(),
        )
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["submitted_blocks"], 6)

    def test_steady_state_uses_common_window_and_drains_tagged_requests(self):
        requests = tuple(
            request(sequence * 2 + npu_id, npu_id, sequence=sequence)
            for sequence in range(8)
            for npu_id in range(2)
        )
        summary = simulate_continuous_batch(
            requests,
            num_npu=2,
            num_ssu=1,
            n_layers=3,
            batch_size=1,
            qos_config=static_qos_config(),
            cross_request_layer0_prefetch=True,
            steady_state=SteadyStateConfig(
                warmup_requests_per_npu=1,
                settle_ms=0.0,
                measurement_ms=300.0,
                block_ms=100.0,
            ),
        )
        self.assertEqual(summary["mode"], "steady_state_full_load")
        self.assertEqual(summary["measurement_duration_ms"], 300.0)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertGreaterEqual(min(summary["request_counts_by_npu"]), 1)
        self.assertFalse(summary["all_input_requests_completed"])
        self.assertGreaterEqual(summary["drain_stop_ms"], summary["measurement_end_ms"])


if __name__ == "__main__":
    unittest.main()
