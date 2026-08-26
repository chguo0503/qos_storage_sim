import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import marginal_utility_probe as probe
import sim


class MarginalUtilityPureFunctionTests(unittest.TestCase):
    def test_zero_horizon_reaches_cap_and_large_horizon_approaches_floor(self):
        urgent = probe.marginal_desired_cir(
            now_ms=5.0,
            deadline_ms=5.0,
            link_backlog_gb=0.0,
            remaining_work_gb=0.0,
        )
        distant = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=1000.0,
            link_backlog_gb=5.0,
            remaining_work_gb=5.0,
        )

        self.assertEqual(urgent.horizon_ms, 0.0)
        self.assertEqual(urgent.marginal_utility, 1.0)
        self.assertEqual(urgent.desired_cir_gbps, 50.0)
        self.assertGreater(distant.desired_cir_gbps, 5.0)
        self.assertLess(distant.desired_cir_gbps, 5.1)

    def test_horizon_units_formula_and_work_fraction_split(self):
        decision, demands = probe.marginal_disk_demands(
            now_ms=0.0,
            deadline_ms=2.0,
            link_backlog_gb=0.050,
            work_by_disk={0: 0.025, 1: 0.075},
        )

        self.assertEqual(decision.slack_ms, 2.0)
        self.assertEqual(decision.link_service_ms, 1.0)
        self.assertEqual(decision.remaining_work_service_ms, 2.0)
        self.assertEqual(decision.horizon_ms, 5.0)
        self.assertAlmostEqual(decision.marginal_utility, 1.0 / 6.0)
        self.assertAlmostEqual(decision.desired_cir_gbps, 12.5)
        self.assertEqual(demands, {0: 3.125, 1: 9.375})
        self.assertAlmostEqual(sum(demands.values()), 12.5)

    def test_desired_rate_is_monotone_in_slack_queue_and_remaining_work(self):
        base = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=0.0,
            link_backlog_gb=0.0,
            remaining_work_gb=0.010,
        ).desired_cir_gbps
        more_slack = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=1.0,
            link_backlog_gb=0.0,
            remaining_work_gb=0.010,
        ).desired_cir_gbps
        more_queue = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=0.0,
            link_backlog_gb=0.050,
            remaining_work_gb=0.010,
        ).desired_cir_gbps
        more_work = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=0.0,
            link_backlog_gb=0.0,
            remaining_work_gb=0.060,
        ).desired_cir_gbps

        self.assertGreater(base, more_slack)
        self.assertGreater(base, more_queue)
        self.assertGreater(base, more_work)


class MarginalControllerTests(unittest.TestCase):
    def build_controller(self):
        npus = [sim.NPUState(0), sim.NPUState(1)]
        engine_config = probe.MarginalDynamicCIRConfig(
            routing_mode=sim.DYNAMIC_ROUTE_FIXED
        )
        controller = probe.MarginalUtilityCIRController(
            npus,
            engine_config,
            probe.DEFAULT_MARGINAL_CONFIG,
        )
        return npus, controller, engine_config

    def test_controller_uses_current_link_queue_and_remaining_layer_work(self):
        npus, controller, _ = self.build_controller()
        npus[0].link_pending_gb = 0.025
        active_link_flow = sim.BlockIOFlow(
            npu_id=0,
            request_id=0,
            layer=0,
            block_idx=99,
            disk_id=0,
            total_gb=0.001,
            queue_id=-1,
            block_count=1,
            enqueue_time=0.0,
        )
        active_link_flow.link_end_time = 0.5
        npus[0].link_active_flow = active_link_flow
        controller.register_layer(
            npu_id=0,
            layer=0,
            input_demand_gbps=999.0,
            deadline_ms=2.0,
            work_by_disk={0: 0.025, 1: 0.075},
        )

        demands = controller.demands_for_layer(0, 0, 0.0)
        self.assertEqual(demands, {0: 3.125, 1: 9.375})
        self.assertAlmostEqual(
            controller.demand_for_disk(0, 0, 1, 0.0),
            9.375,
        )
        telemetry = controller.probe_telemetry()
        self.assertEqual(telemetry["desired_cir_evaluations"], 2)
        self.assertEqual(telemetry["min_horizon_ms"], 5.0)
        self.assertEqual(telemetry["max_horizon_ms"], 5.0)

        completed_ssd_flow = sim.BlockIOFlow(
            npu_id=0,
            request_id=0,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=0.025,
            queue_id=0,
            block_count=1,
            enqueue_time=0.0,
        )
        controller.complete_ssd(completed_ssd_flow)
        after_completion = controller.demands_for_layer(0, 0, 0.0)
        expected = probe.marginal_desired_cir(
            now_ms=0.0,
            deadline_ms=2.0,
            link_backlog_gb=0.050,
            remaining_work_gb=0.075,
        ).desired_cir_gbps
        self.assertEqual(list(after_completion), [1])
        self.assertAlmostEqual(after_completion[1], expected)

    def test_existing_scheduler_keeps_atomic_40_clamp_and_40_command_rate(self):
        npus, controller, engine_config = self.build_controller()
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
            dynamic_cir_config=engine_config,
        )
        flows = []
        for npu_id in (0, 1):
            controller.register_layer(
                npu_id=npu_id,
                layer=0,
                input_demand_gbps=0.0,
                deadline_ms=0.0,
                work_by_disk={0: 0.001},
            )
            flows.append(
                sim.BlockIOFlow(
                    npu_id=npu_id,
                    request_id=npu_id,
                    layer=0,
                    block_idx=0,
                    disk_id=0,
                    total_gb=0.001,
                    queue_id=pools[(npu_id, 0)][0],
                    block_count=1,
                    enqueue_time=0.0,
                )
            )
        scheduler.enqueue_many(flows, 0.0)

        grants = scheduler.apply_dynamic_cir_epoch(0.0)
        active = scheduler.dispatch(0.0, [], schedule_completion=False)

        self.assertAlmostEqual(sum(grants.values()), 40.0)
        self.assertLessEqual(scheduler.dynamic_cir_total_gbps, 40.0)
        for npu_id in (0, 1):
            configured = sum(
                path.cir
                for path_id, path in scheduler.paths.items()
                if owners.get(path_id) == npu_id
            )
            self.assertAlmostEqual(configured, grants[npu_id])
        self.assertEqual(len(scheduler.state.active_flows), 1)
        self.assertEqual(active.bw, 40.0)
        self.assertAlmostEqual(active.end_time, 0.025)

    def test_temporary_override_restores_sim_and_labels_actual_telemetry(self):
        original = sim.DynamicCIRController
        engine_config = probe.MarginalDynamicCIRConfig(
            routing_mode=sim.DYNAMIC_ROUTE_LEAST_WORK
        )
        with probe.marginal_controller_override() as controllers:
            controller = sim.DynamicCIRController([sim.NPUState(0)], engine_config)
            self.assertIsInstance(
                controller,
                probe.MarginalUtilityCIRController,
            )
        self.assertIs(sim.DynamicCIRController, original)

        controller.register_layer(
            npu_id=0,
            layer=0,
            input_demand_gbps=1.0,
            deadline_ms=0.0,
            work_by_disk={0: 0.001},
        )
        controller.demands_for_layer(0, 0, 0.0)
        telemetry = probe._actual_marginal_telemetry(
            {
                "mode": probe.MARGINAL_MODE,
                "epochs": 3,
                "final_total_cir_gbps": [0.0],
                "max_total_cir_gbps": 40.0,
            },
            controllers[0],
            probe.DEFAULT_MARGINAL_CONFIG,
            sim.DYNAMIC_ROUTE_LEAST_WORK,
        )
        self.assertEqual(telemetry["mode"], probe.MARGINAL_MODE)
        self.assertNotIn("demand_proportional", json.dumps(telemetry))
        self.assertNotIn("slack_link_guarded", json.dumps(telemetry))
        self.assertEqual(telemetry["future_information"], "none")

    def test_tiny_sim_accepts_truthful_marginal_mode_for_both_routes(self):
        loads = [
            {
                "request_id": 0,
                "npu_id": 0,
                "seq_len_k": 3,
                "nql": 0,
                "per_layer_us": 1_000.0,
                "per_layer_kv_gb": 0.001,
                "required_bw_input_gbps": 1.0,
                "category": "SS",
                "arrival_time": 0.0,
            }
        ]
        placement = {0: {0: ((0, 0.001),)}}
        prepared = sim.PreparedSimulationInputs(
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

        for routing_mode in (
            sim.DYNAMIC_ROUTE_FIXED,
            sim.DYNAMIC_ROUTE_LEAST_WORK,
        ):
            with probe.marginal_controller_override() as controllers:
                _, full = sim.simulate_continuous(
                    {},
                    policy=sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
                    num_npu=1,
                    num_disk=1,
                    n_layers=1,
                    submit_order_seed=3,
                    prepared_inputs=prepared,
                    dynamic_cir_config=probe.MarginalDynamicCIRConfig(
                        routing_mode=routing_mode
                    ),
                )
            self.assertEqual(
                full["dynamic_cir_control"]["mode"],
                probe.MARGINAL_MODE,
            )
            self.assertEqual(
                full["dynamic_cir_control"]["routing_mode"],
                routing_mode,
            )
            self.assertLessEqual(
                full["dynamic_cir_control"]["max_total_cir_gbps"],
                40.0,
            )
            self.assertGreater(controllers[0].evaluation_count, 0)
            self.assertTrue(all(probe._summary(full)["invariants"].values()))


class ProbeRunnerPlumbingTests(unittest.TestCase):
    def test_default_plan_is_128_by_16_all_ssus_both_seeds_and_routes(self):
        plan = probe.execution_plan()
        self.assertFalse(plan["execute"])
        self.assertEqual(plan["num_npu"], 128)
        self.assertEqual(plan["n_layers"], 16)
        self.assertEqual(plan["ssu_list"], [40, 56, 80])
        self.assertEqual(plan["seeds"], [42, 43])
        self.assertEqual(plan["case_count"], 12)
        self.assertEqual(
            plan["strategies"],
            ["dynamic_marginal_fixed_path", "joint_marginal_path_cir"],
        )

    def test_cli_without_run_only_prints_plan(self):
        with mock.patch.object(probe, "run_probe") as run_probe:
            with mock.patch("builtins.print") as output:
                status = probe.main([])
        self.assertEqual(status, 0)
        run_probe.assert_not_called()
        rendered = output.call_args.args[0]
        self.assertFalse(json.loads(rendered)["execute"])

    def test_checkpoint_is_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            probe.write_json_checkpoint(path, {"complete": False, "rows": [1]})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"complete": False, "rows": [1]},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
