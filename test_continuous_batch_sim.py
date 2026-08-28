import unittest

import sim
from continuous_batch_sim import (
    CIRControlConfig,
    CIRControlSnapshot,
    ContinuousBatchRequest,
    ControlRequestView,
    MaxMinSchemeBController,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    ActiveRequestSnapshot,
    build_scheme_b_target,
    legacy_qos_config,
    qos_configs_from_path_cirs,
    scheme_b_client_config,
)
from scheme_b_prefill import PATH_COUNT, dedicated_path_id


def request(request_id, npu_id, arrival_ms=0.0, layers=3):
    return ContinuousBatchRequest(
        request_id=request_id,
        npu_id=npu_id,
        arrival_time_ms=arrival_ms,
        load={
            "request_id": request_id,
            "npu_id": npu_id,
            "category": "SS",
            "per_layer_us": 100_000.0,
            "initial": arrival_ms == 0.0,
        },
        placement=(((0, 0.8),),) * layers,
    )


class ContinuousBatchSimTest(unittest.TestCase):
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
            qos_config=legacy_qos_config(),
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
        target = build_scheme_b_target(
            (ActiveRequestSnapshot(
                0, 0, 0.0, 1_000_000.0, (10.0, 100.0)
            ),),
            num_npu=1,
            num_ssu=2,
        )
        self.assertEqual(decision.path_cirs_by_ssu, target.path_cirs_by_ssu)
        self.assertAlmostEqual(decision.path_cirs_by_ssu[0][paths[0]], 10.0)
        self.assertAlmostEqual(decision.path_cirs_by_ssu[1][paths[0]], 40.0)

    def test_wall_clock_ticks_and_end_to_end_conservation(self):
        calls = []

        def observe(snapshot):
            calls.append(snapshot.time_ms)
            return None

        zero_cirs = ((0.0,) * PATH_COUNT,)
        summary = simulate_continuous_batch(
            (request(0, 0), request(1, 0, 50.0)),
            num_npu=1,
            num_ssu=1,
            n_layers=3,
            batch_size=1,
            qos_configs_by_ssu=qos_configs_from_path_cirs(zero_cirs),
            npu_dedicated_paths=(dedicated_path_id(0),),
            client_io_config=scheme_b_client_config("test"),
            control=CIRControlConfig(callback=observe, interval_ms=10.0),
        )
        self.assertEqual(calls[:3], [0.0, 10.0, 20.0])
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["submitted_blocks"], 6)


if __name__ == "__main__":
    unittest.main()
