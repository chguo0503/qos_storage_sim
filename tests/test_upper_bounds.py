import unittest

import sim
from upper_bounds import isolated_layer_io_time_ms, isolated_no_contention_bound


def contended_prepared():
    loads = []
    placement = {}
    for request_id in range(2):
        loads.append(
            {
                "request_id": request_id,
                "npu_id": request_id,
                "seq_len_k": 3,
                "nql": 0,
                "per_layer_us": 1_000.0,
                "per_layer_kv_gb": 0.004,
                "required_bw_input_gbps": 4.0,
                "category": "SS",
                "arrival_time": 0.0,
            }
        )
        placement[request_id] = {
            0: ((0, 0.004),),
            1: ((0, 0.004),),
        }
    return sim.PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=2,
        num_disk=1,
        placement_mode="manual",
    )


class IsolatedUpperBoundTests(unittest.TestCase):
    def test_layer_bound_serializes_each_ssd_then_single_npu_link(self):
        blocks = ((0, 0.004), (0, 0.004), (1, 0.004))
        self.assertAlmostEqual(isolated_layer_io_time_ms(blocks), 0.340, places=12)

    def test_removing_inter_npu_contention_is_an_upper_bound(self):
        prepared = contended_prepared()
        bound = isolated_no_contention_bound(prepared)
        _, actual = sim.simulate_continuous(
            {},
            policy=sim.POLICY_BASELINE_BYPASS,
            num_npu=2,
            num_disk=1,
            n_layers=2,
            prepared_inputs=prepared,
            submit_order_seed=5,
        )

        expected_per_request = 2.0 / (2.0 + 0.100)
        self.assertEqual(bound["name"], "fluid_no_inter_npu_contention_upper_bound")
        self.assertAlmostEqual(
            bound["avg_request_compute_fraction_upper_bound"],
            expected_per_request,
            places=12,
        )
        self.assertTrue(
            all(
                abs(row["io_wait_lower_bound_ms"] - 0.100) < 1e-12
                for row in bound["request_bounds"]
            )
        )
        self.assertGreater(
            bound["avg_request_compute_fraction_upper_bound"],
            actual["avg_request_compute_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
