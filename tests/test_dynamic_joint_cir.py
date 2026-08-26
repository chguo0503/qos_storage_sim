import json
import tempfile
import unittest
from pathlib import Path

from advanced_policies import (
    capped_input_demands,
    proportional_capacity_grants,
    slack_link_guarded_demands,
)
import sim
from dynamic_cir_calibration import run_calibration, write_calibration


def dynamic_flow(npu_id, block_idx, path_id, size_gb=0.001):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=block_idx,
        disk_id=0,
        total_gb=size_gb,
        queue_id=path_id,
        block_count=1,
        enqueue_time=0.0,
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


class DynamicDemandTests(unittest.TestCase):
    def test_input_demand_is_capped_before_proportional_disk_split(self):
        self.assertEqual(
            capped_input_demands(
                50.0,
                80.0,
                {0: 0.020, 1: 0.060},
            ),
            {0: 12.5, 1: 37.5},
        )

    def test_slack_mode_uses_remaining_work_deadline_and_link_backlog(self):
        unguarded = slack_link_guarded_demands(
            50.0,
            now_ms=0.0,
            deadline_ms=2.0,
            link_backlog_gb=0.0,
            work_by_resource={0: 0.020, 1: 0.060},
        )
        guarded = slack_link_guarded_demands(
            50.0,
            now_ms=0.0,
            deadline_ms=2.0,
            link_backlog_gb=0.050,
            work_by_resource={0: 0.020, 1: 0.060},
        )
        layer_zero = slack_link_guarded_demands(
            50.0,
            now_ms=0.0,
            deadline_ms=0.0,
            link_backlog_gb=0.0,
            work_by_resource={0: 0.020, 1: 0.060},
        )
        self.assertEqual(unguarded, {0: 10.0, 1: 30.0})
        self.assertEqual(guarded, {0: 12.5, 1: 37.5})
        self.assertEqual(layer_zero, {0: 12.5, 1: 37.5})
        self.assertLessEqual(sum(guarded.values()), 50.0)

    def test_overload_is_scaled_proportionally_to_one_ssd_capacity(self):
        self.assertEqual(
            proportional_capacity_grants(40.0, {0: 20.0, 1: 60.0}),
            {0: 10.0, 1: 30.0},
        )
        self.assertEqual(
            proportional_capacity_grants(40.0, {0: 10.0, 1: 30.0}),
            {0: 10.0, 1: 30.0},
        )


class DynamicPathTests(unittest.TestCase):
    def test_every_disk_has_two_mutually_exclusive_paths_per_npu(self):
        pools = sim.build_dynamic_npu_path_pools(128, 3)
        for disk_id in range(3):
            disk_pools = [pools[(npu_id, disk_id)] for npu_id in range(128)]
            flattened = [path_id for pool in disk_pools for path_id in pool]
            self.assertTrue(all(len(pool) == 2 for pool in disk_pools))
            self.assertEqual(len(set(flattened)), 256)
            self.assertEqual(set(flattened), set(range(256)))

        small = sim.build_dynamic_npu_path_pools(4, 1)
        self.assertEqual(small[(0, 0)], (0, 128))
        self.assertEqual(small[(3, 0)], (3, 131))

    def test_dynamic_router_never_leaves_its_owned_pool(self):
        counts = [0] * 256
        allowed = (7, 135)
        counts[7] = 3
        selected = sim.client_select_dynamic_owned_paths(
            block_sizes_gb=[0.001] * 8,
            path_io_counts=counts,
            allowed_path_ids=allowed,
            start_offset=0,
        )
        self.assertTrue(set(selected) <= set(allowed))
        self.assertGreater(selected.count(135), selected.count(7))


class DynamicSchedulerTests(unittest.TestCase):
    def build_scheduler(self):
        npus = [sim.NPUState(npu_id) for npu_id in range(2)]
        config = sim.DynamicCIRPolicyConfig(
            mode=sim.DYNAMIC_CIR_DEMAND_PROPORTIONAL
        )
        controller = sim.DynamicCIRController(npus, config)
        pools = sim.build_dynamic_npu_path_pools(2, 1)
        owners = {
            path_id: npu_id
            for (npu_id, _), path_ids in pools.items()
            for path_id in path_ids
        }
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0),
            sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
            sim.DISK_BW,
            dynamic_cir_controller=controller,
            dynamic_path_owners=owners,
            dynamic_cir_config=config,
        )
        return scheduler, controller, pools, owners

    def test_ten_thirty_cir_and_long_run_command_service(self):
        scheduler, controller, pools, owners = self.build_scheduler()
        counts = [0] * 256
        flows = []
        for npu_id, demand_gbps in enumerate((10.0, 30.0)):
            controller.register_layer(
                npu_id=npu_id,
                layer=0,
                input_demand_gbps=demand_gbps,
                deadline_ms=0.0,
                work_by_disk={0: 0.040},
            )
            path_ids = sim.client_select_dynamic_owned_paths(
                block_sizes_gb=[0.001] * 40,
                path_io_counts=counts,
                allowed_path_ids=pools[(npu_id, 0)],
            )
            for block_idx, path_id in enumerate(path_ids):
                counts[path_id] += 1
                flows.append(dynamic_flow(npu_id, block_idx, path_id))
        scheduler.enqueue_many(flows, 0.0)

        scheduler.apply_dynamic_cir_epoch(0.0)
        configured = {
            npu_id: sum(
                path.cir
                for path_id, path in scheduler.paths.items()
                if owners.get(path_id) == npu_id
            )
            for npu_id in range(2)
        }
        self.assertAlmostEqual(configured[0], 10.0, places=12)
        self.assertAlmostEqual(configured[1], 30.0, places=12)
        self.assertLessEqual(scheduler.dynamic_cir_total_gbps, 40.0)

        selected = []
        current_time = 0.0
        for _ in range(40):
            active = scheduler.dispatch(
                current_time, [], schedule_completion=False
            )
            selected.append(active.npu_id)
            self.assertAlmostEqual(active.bw, 40.0, places=12)
            self.assertAlmostEqual(
                active.end_time - current_time,
                0.025,
                places=12,
            )
            current_time = active.end_time
            completed = scheduler.complete_ready_flows(current_time)
            controller.complete_ssd(completed[0])

        self.assertEqual(selected.count(0), 10)
        self.assertEqual(selected.count(1), 30)
        self.assertLessEqual(scheduler.dynamic_cir_max_total_gbps, 40.0)

    def test_cir_reconfiguration_does_not_preempt_active_command(self):
        scheduler, controller, pools, _ = self.build_scheduler()
        for npu_id, demand_gbps in enumerate((10.0, 30.0)):
            controller.register_layer(
                npu_id=npu_id,
                layer=0,
                input_demand_gbps=demand_gbps,
                deadline_ms=0.0,
                work_by_disk={0: 0.001},
            )
        first = dynamic_flow(0, 0, pools[(0, 0)][0])
        scheduler.enqueue_many([first], 0.0)
        active = scheduler.dispatch(0.0, [], schedule_completion=False)
        original_end = active.end_time

        second = dynamic_flow(1, 0, pools[(1, 0)][0])
        scheduler.enqueue_many([second], 0.0)
        scheduler.apply_dynamic_cir_epoch(0.0)
        self.assertIs(active, scheduler.state.active_flows[0])
        self.assertEqual(active.bw, 40.0)
        self.assertEqual(active.end_time, original_end)


class DynamicIntegrationTests(unittest.TestCase):
    def test_fixed_route_uses_one_owned_path_without_changing_dynamic_cir(self):
        _, summary = sim.simulate_continuous(
            {},
            policy=sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
            dynamic_cir_config=sim.DynamicCIRPolicyConfig(
                mode=sim.DYNAMIC_CIR_DEMAND_PROPORTIONAL,
                routing_mode=sim.DYNAMIC_ROUTE_FIXED,
            ),
            num_npu=1,
            num_disk=1,
            n_layers=1,
            prepared_inputs=single_io_prepared(),
            submit_order_seed=4,
        )
        self.assertEqual(
            summary["dynamic_cir_control"]["routing_mode"],
            sim.DYNAMIC_ROUTE_FIXED,
        )
        self.assertAlmostEqual(summary["makespan_ms"], 1.180, places=12)

    def test_dynamic_policy_preserves_shared_three_stage_data_plane(self):
        _, summary = sim.simulate_continuous(
            {},
            policy=sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
            dynamic_cir_config=sim.DynamicCIRPolicyConfig(
                mode=sim.DYNAMIC_CIR_SLACK_LINK_GUARDED
            ),
            num_npu=1,
            num_disk=1,
            n_layers=1,
            prepared_inputs=single_io_prepared(),
            submit_order_seed=4,
        )
        self.assertAlmostEqual(summary["makespan_ms"], 1.180, places=12)
        self.assertAlmostEqual(
            summary["disk_stats"][0]["busy_time_ms"], 0.100, places=12
        )
        self.assertAlmostEqual(
            summary["npu_link_stats"][0]["busy_time_ms"], 0.080, places=12
        )
        self.assertEqual(summary["disk_stats"][0]["max_backend_active_io"], 1)
        self.assertEqual(summary["npu_link_stats"][0]["max_active_io"], 1)
        self.assertEqual(summary["block_conservation"]["expected"], 1)
        self.assertEqual(summary["block_conservation"]["submitted"], 1)
        self.assertEqual(summary["block_conservation"]["completed"], 1)
        self.assertEqual(
            summary["dynamic_cir_control"]["mode"],
            sim.DYNAMIC_CIR_SLACK_LINK_GUARDED,
        )
        self.assertGreater(summary["dynamic_cir_control"]["epochs"], 0)
        self.assertLessEqual(
            summary["dynamic_cir_control"]["max_total_cir_gbps"], 40.0
        )


class DynamicCalibrationTests(unittest.TestCase):
    def test_real_scheduler_calibration_reproduces_20_20_and_10_30(self):
        result = run_calibration()

        self.assertEqual(
            result["baseline"]["measured_ssd_service_gbps"],
            [20.0, 20.0],
        )
        self.assertEqual(
            result["joint_dynamic"]["measured_ssd_service_gbps"],
            [10.0, 30.0],
        )
        self.assertEqual(
            result["joint_dynamic"]["initial_cir_grants_gbps"],
            [10.0, 30.0],
        )
        for policy in ("baseline", "joint_dynamic"):
            self.assertEqual(
                result[policy]["command_service_gbps_unique"],
                [40.0],
            )
            self.assertEqual(result[policy]["ssd_active_io_unique"], [1])
            self.assertEqual(result[policy]["max_ssd_active_io"], 1)
            self.assertEqual(len(result[policy]["command_trace"]), 40)
        self.assertTrue(all(result["checks"].values()))

    def test_calibration_writes_json_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            result, json_path, png_path = write_calibration(Path(directory))

            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                result,
            )
            self.assertGreater(png_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
