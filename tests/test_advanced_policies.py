import unittest

from advanced_policies import (
    capped_proportional_demands,
    max_min_work_conserving_rates,
)
from allocation_calibration import run_two_npu_calibration
import sim


def flow(npu_id, index, demand_gbps, size_gb=0.001):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=index,
        disk_id=0,
        total_gb=size_gb,
        queue_id=-1,
        block_count=1,
        enqueue_time=0.0,
        demand_gbps=demand_gbps,
        deadline_time=0.0,
        layer_work_gb=0.040,
    )


def single_io_prepared():
    loads = [
        {
            "request_id": 0,
            "npu_id": 0,
            "seq_len_k": 3,
            "nql": 0,
            "per_layer_us": 1_000.0,
            "per_layer_kv_gb": 0.004,
            "required_bw_input_gbps": 4.0,
            "category": "SS",
            "arrival_time": 0.0,
        }
    ]
    placement = {0: {0: ((0, 0.004),)}}
    return sim.PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=1,
        placement_mode="manual",
    )


class MaxMinTests(unittest.TestCase):
    def test_ten_and_thirty_demands_receive_ten_and_thirty(self):
        rates = max_min_work_conserving_rates(40.0, {0: 10.0, 1: 30.0})
        self.assertEqual(rates, {0: 10.0, 1: 30.0})

    def test_unused_capacity_is_work_conserving_and_oversubscription_is_fair(self):
        self.assertEqual(
            max_min_work_conserving_rates(50.0, {0: 10.0, 1: 30.0}),
            {0: 15.0, 1: 35.0},
        )
        self.assertEqual(
            max_min_work_conserving_rates(40.0, {0: 10.0, 1: 50.0}),
            {0: 10.0, 1: 30.0},
        )

    def test_npu_cap_is_applied_before_splitting_demand_across_ssds(self):
        self.assertEqual(
            capped_proportional_demands(
                50.0,
                0.001,
                {0: 0.020, 1: 0.060},
            ),
            {0: 12.5, 1: 37.5},
        )
        self.assertEqual(
            capped_proportional_demands(
                50.0,
                0.002,
                {0: 0.020, 1: 0.060},
            ),
            {0: 10.0, 1: 30.0},
        )


class AdvancedPolicyIntegrationTests(unittest.TestCase):
    def test_controlled_scheduler_calibration_measures_twenty_twenty_and_ten_thirty(self):
        result = run_two_npu_calibration()
        for actual, expected in zip(
            result["baseline"]["achieved_ssd_service_gbps"],
            (20.0, 20.0),
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(
            result["demand_maxmin"]["achieved_ssd_service_gbps"],
            (10.0, 30.0),
        ):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_demand_scheduler_packetizes_ten_to_thirty_service(self):
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0),
            sim.POLICY_QOS_DEMAND_MAXMIN,
            sim.DISK_BW,
        )
        flows = [flow(0, index, 10.0) for index in range(40)] + [
            flow(1, 40 + index, 30.0) for index in range(40)
        ]
        scheduler.enqueue_many(flows, 0.0)

        selected = []
        current_time = 0.0
        for _ in range(40):
            active = scheduler.dispatch(current_time, [], schedule_completion=False)
            selected.append(active.npu_id)
            self.assertAlmostEqual(active.bw, sim.DISK_BW)
            current_time = active.end_time
            scheduler.complete_ready_flows(current_time)

        self.assertEqual(selected.count(0), 10)
        self.assertEqual(selected.count(1), 30)

    def test_all_policies_share_identical_single_io_two_stage_data_plane(self):
        prepared = single_io_prepared()
        summaries = []
        for policy in (
            sim.POLICY_BASELINE_BYPASS,
            sim.POLICY_QOS_DEMAND_MAXMIN,
            sim.POLICY_OMNISCIENT_EDF,
        ):
            with self.subTest(policy=policy):
                _, summary = sim.simulate_continuous(
                    {},
                    policy=policy,
                    num_npu=1,
                    num_disk=1,
                    n_layers=1,
                    prepared_inputs=prepared,
                    submit_order_seed=4,
                )
                self.assertAlmostEqual(summary["makespan_ms"], 1.180, places=12)
                self.assertAlmostEqual(summary["disk_stats"][0]["busy_time_ms"], 0.100, places=12)
                self.assertAlmostEqual(summary["npu_link_stats"][0]["busy_time_ms"], 0.080, places=12)
                self.assertEqual(summary["disk_stats"][0]["max_backend_active_io"], 1)
                self.assertEqual(summary["npu_link_stats"][0]["max_active_io"], 1)
                self.assertEqual(
                    summary["block_conservation"],
                    {
                        "expected": 1,
                        "submitted": 1,
                        "completed": 1,
                        "placement_targets_preserved": True,
                        "expected_read_gb": 0.004,
                        "ssd_completed_read_gb": 0.004,
                        "completed_read_gb": 0.004,
                    },
                )
                summaries.append(summary)

        self.assertTrue(
            all(
                summary["backend_model"] == summaries[0]["backend_model"]
                and summary["data_plane_stages"] == summaries[0]["data_plane_stages"]
                for summary in summaries[1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
