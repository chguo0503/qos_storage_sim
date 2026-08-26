import unittest

import sim


def candidate(npu_id, block_idx, disk_id, *, size_gb=0.001, deadline=0.0):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=block_idx,
        disk_id=disk_id,
        total_gb=size_gb,
        queue_id=-1,
        block_count=1,
        enqueue_time=0.0,
        deadline_time=deadline,
        layer_work_gb=size_gb,
    )


def coordinator_and_schedulers(num_npu, num_disk):
    npus = [sim.NPUState(npu_id) for npu_id in range(num_npu)]
    for npu in npus:
        npu.pending_blocks = {0: 4}
    disks = [sim.DiskState(disk_id) for disk_id in range(num_disk)]
    coordinator = sim.GlobalLinkCoordinator(npus, disks)
    schedulers = [
        sim.DiskIOScheduler(
            disk,
            sim.POLICY_GLOBAL_LINK_AWARE,
            sim.DISK_BW,
            global_link_coordinator=coordinator,
        )
        for disk in disks
    ]
    return npus, disks, coordinator, schedulers


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


class GlobalLinkAwareSchedulerTests(unittest.TestCase):
    def test_second_disk_avoids_known_same_npu_incast(self):
        _, disks, coordinator, schedulers = coordinator_and_schedulers(2, 2)
        schedulers[0].enqueue_many(
            [candidate(0, 0, 0), candidate(1, 0, 0)], 0.0
        )
        schedulers[1].enqueue_many(
            [candidate(0, 1, 1), candidate(1, 1, 1)], 0.0
        )

        first = schedulers[0].dispatch(0.0, [], schedule_completion=False)
        second = schedulers[1].dispatch(0.0, [], schedule_completion=False)

        self.assertEqual(first.npu_id, 0)
        self.assertEqual(second.npu_id, 1)
        self.assertEqual(len(disks[0].active_flows), 1)
        self.assertEqual(len(disks[1].active_flows), 1)
        self.assertEqual(coordinator.selections, 2)

    def test_only_candidate_dispatches_despite_known_downstream_queue(self):
        _, disks, _, schedulers = coordinator_and_schedulers(1, 2)
        schedulers[0].enqueue_many([candidate(0, 0, 0, size_gb=0.004)], 0.0)
        schedulers[1].enqueue_many([candidate(0, 1, 1, size_gb=0.004)], 0.0)

        first = schedulers[0].dispatch(0.0, [], schedule_completion=False)
        second = schedulers[1].dispatch(0.0, [], schedule_completion=False)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.npu_id, 0)
        self.assertEqual(sum(len(disk.active_flows) for disk in disks), 2)

    def test_only_per_npu_fcfs_heads_are_candidates(self):
        _, _, _, schedulers = coordinator_and_schedulers(2, 1)
        slow_head = candidate(0, 0, 0, deadline=100.0)
        urgent_but_second = candidate(0, 1, 0, deadline=0.0)
        other_head = candidate(1, 0, 0, deadline=50.0)
        schedulers[0].enqueue_many(
            [slow_head, urgent_but_second, other_head], 0.0
        )

        selected = schedulers[0].dispatch(0.0, [], schedule_completion=False)

        self.assertIs(selected, other_head)

    def test_single_disk_single_npu_degenerates_to_shared_three_stage_model(self):
        _, baseline = sim.simulate_continuous(
            {},
            policy=sim.POLICY_BASELINE_BYPASS,
            num_npu=1,
            num_disk=1,
            n_layers=1,
            prepared_inputs=single_io_prepared(),
            submit_order_seed=4,
        )
        _, global_policy = sim.simulate_continuous(
            {},
            policy=sim.POLICY_GLOBAL_LINK_AWARE,
            num_npu=1,
            num_disk=1,
            n_layers=1,
            prepared_inputs=single_io_prepared(),
            submit_order_seed=4,
        )

        self.assertAlmostEqual(global_policy["makespan_ms"], 1.180, places=12)
        self.assertAlmostEqual(
            global_policy["makespan_ms"], baseline["makespan_ms"], places=12
        )
        self.assertAlmostEqual(
            global_policy["disk_stats"][0]["busy_time_ms"], 0.100, places=12
        )
        self.assertAlmostEqual(
            global_policy["npu_link_stats"][0]["busy_time_ms"], 0.080, places=12
        )
        self.assertEqual(
            global_policy["block_conservation"],
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
        self.assertEqual(
            global_policy["global_link_coordination"]["selections"], 1
        )


if __name__ == "__main__":
    unittest.main()
