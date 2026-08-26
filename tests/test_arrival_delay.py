import unittest

import sim


PROFILE = (5.0, 1_000.0, 10.0, 0.004)
TABLE = {(3, 0): PROFILE}


def manual_prepared(arrival_time):
    loads = [
        {
            "request_id": 0,
            "npu_id": 0,
            "seq_len_k": 3,
            "nql": 0,
            "per_layer_us": 0.0,
            "per_layer_kv_gb": 0.004,
            "required_bw_input_gbps": 5.0,
            "category": "SS",
            "arrival_time": float(arrival_time),
        }
    ]
    placement = {0: {0: ((0, 0.004),)}}
    return sim.PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placement,
        workload_seed=11,
        placement_seed=12,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=1,
        placement_mode="manual",
        arrival_delay_seed=13,
        arrival_delay_max_ms=5.0,
    )


class ArrivalDelayTests(unittest.TestCase):
    def test_every_npu_gets_reproducible_independent_zero_to_five_ms_delay(self):
        common = dict(
            bw_table=TABLE,
            total_requests=128,
            n_layers=1,
            num_disk=1,
            workload_seed=101,
            placement_seed=102,
            arrival_delay_seed=103,
            arrival_delay_max_ms=5.0,
            placement_mode="roundrobin",
        )
        first = sim.prepare_simulation_inputs(**common)
        second = sim.prepare_simulation_inputs(**common)
        delays = [load["arrival_time"] for load in first.request_loads]

        self.assertEqual(delays, [load["arrival_time"] for load in second.request_loads])
        self.assertEqual(len(delays), 128)
        self.assertEqual(len(set(delays)), 128)
        self.assertTrue(all(0.0 <= delay < 5.0 for delay in delays))
        self.assertEqual(first.arrival_delay_seed, 103)
        self.assertEqual(first.arrival_delay_max_ms, 5.0)

    def test_arrival_rng_does_not_change_workload_or_placement(self):
        common = dict(
            bw_table=TABLE,
            total_requests=8,
            n_layers=2,
            num_disk=2,
            workload_seed=201,
            placement_seed=202,
            arrival_delay_max_ms=5.0,
        )
        first = sim.prepare_simulation_inputs(arrival_delay_seed=203, **common)
        second = sim.prepare_simulation_inputs(arrival_delay_seed=204, **common)

        first_profiles = [
            {key: value for key, value in load.items() if key != "arrival_time"}
            for load in first.request_loads
        ]
        second_profiles = [
            {key: value for key, value in load.items() if key != "arrival_time"}
            for load in second.request_loads
        ]
        self.assertEqual(first_profiles, second_profiles)
        self.assertEqual(first.placement_hash, second.placement_hash)
        self.assertNotEqual(first.workload_hash, second.workload_hash)
        self.assertNotEqual(
            [load["arrival_time"] for load in first.request_loads],
            [load["arrival_time"] for load in second.request_loads],
        )

    def test_arrival_delays_release_but_does_not_inflate_request_ttft(self):
        prepared = manual_prepared(arrival_time=2.5)
        _, summary = sim.simulate_continuous(
            {},
            policy=sim.POLICY_BASELINE_BYPASS,
            num_npu=1,
            num_disk=1,
            n_layers=1,
            prepared_inputs=prepared,
            submit_order_seed=7,
        )

        request = summary["request_metrics"][0]
        self.assertAlmostEqual(summary["makespan_ms"], 2.680, places=12)
        self.assertAlmostEqual(request["arrival_delay_ms"], 2.5, places=12)
        self.assertAlmostEqual(request["queueing_delay_ms"], 0.0, places=12)
        self.assertAlmostEqual(request["processing_ttft_ms"], 0.180, places=12)
        self.assertAlmostEqual(request["ttft_ms"], 0.180, places=12)
        self.assertEqual(summary["arrival_delay_seed"], 13)
        self.assertEqual(summary["arrival_delay_max_ms"], 5.0)


if __name__ == "__main__":
    unittest.main()
