import unittest

import capacity_constrained_oracle_experiment as oracle
import sim


def _prepared_single_request():
    loads = (
        {
            "request_id": 0,
            "npu_id": 0,
            "seq_len_k": 3,
            "nql": 0,
            "per_layer_us": 1_000.0,
            "per_layer_kv_gb": 0.004,
            "required_bw_input_gbps": 4.0,
            "category": "SS",
            "arrival_time": 0.25,
        },
    )
    placement = {0: {0: ((0, 0.004),)}}
    return sim.PreparedSimulationInputs(
        request_loads=loads,
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=1,
        placement_mode="manual",
        arrival_delay_seed=3,
        arrival_delay_max_ms=5.0,
    )


class CapacityConstrainedOracleTests(unittest.TestCase):
    def test_client_issue_contract_matches_routing_matrix(self):
        config = oracle.oracle_client_config()
        self.assertEqual(config.submit_batch_size, 1)
        self.assertEqual(config.issue_interval_us, 0.1)

    def test_priority_is_stable_and_favors_lower_weighted_layer_work(self):
        smaller = sim.BlockIOFlow(
            npu_id=0,
            request_id=0,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=0.001,
            queue_id=-1,
            block_count=1,
            enqueue_time=0.0,
            demand_gbps=10.0,
            deadline_time=1.0,
            layer_work_gb=0.01,
        )
        larger = sim.BlockIOFlow(
            npu_id=1,
            request_id=1,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=0.001,
            queue_id=-1,
            block_count=1,
            enqueue_time=0.0,
            demand_gbps=10.0,
            deadline_time=1.0,
            layer_work_gb=0.02,
        )
        self.assertLess(
            oracle.oracle_priority_key(smaller),
            oracle.oracle_priority_key(larger),
        )

    def test_tiny_case_preserves_shared_physical_data_plane(self):
        prepared = _prepared_single_request()
        runtime = {
            "num_npu": 1,
            "n_layers": 1,
            "ls_ratio": None,
            "seed_bundles": {
                "1": {
                    "workload": 1,
                    "placement": 2,
                    "submit_order": 4,
                    "arrival_delay": 3,
                }
            },
        }
        original_priority = sim.omniscient_edf_key
        row = oracle.run_case({}, runtime, 1, 1, prepared)
        self.assertIs(sim.omniscient_edf_key, original_priority)
        summary = row["summary"]
        self.assertEqual(row["kind"], "feasible_oracle_candidate")
        self.assertEqual(summary["policy"], sim.POLICY_PER_SSD_FULL_VISIBLE_EDF)
        self.assertFalse(summary["exact_optimum_proven"])
        self.assertTrue(summary["block_conservation"]["placement_targets_preserved"])
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["backend_capacity_gbps"], 40.0)
        self.assertEqual(summary["npu_bw_limit_gbps"], 50.0)


if __name__ == "__main__":
    unittest.main()
