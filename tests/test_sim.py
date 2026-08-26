import inspect
from types import SimpleNamespace
import unittest

import sim
from experiment import CATEGORY_CIR_GBPS, category_path_cirs, qos_config


def flow(path_id, index, *, npu_id=0, size=0.001):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=index,
        disk_id=0,
        total_gb=size,
        queue_id=path_id,
        block_count=1,
        enqueue_time=0.0,
    )


POLICIES = (
    sim.POLICY_BASELINE_BYPASS,
    sim.POLICY_QOS_STATIC_CIR,
)


def manual_prepared(per_npu_blocks, num_disk):
    loads = []
    placement = {}
    for npu_id, blocks in enumerate(per_npu_blocks):
        loads.append(
            {
                "request_id": npu_id,
                "npu_id": npu_id,
                "seq_len_k": 3,
                "nql": 0,
                "per_layer_us": 0.0,
                "per_layer_kv_gb": sum(size for _, size in blocks),
                "required_bw_input_gbps": 0.0,
                "category": "SS",
                "arrival_time": 0.0,
            }
        )
        placement[npu_id] = {0: tuple(blocks)}

    return sim.PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=num_disk,
        placement_mode="manual",
    )


def paired_summaries(per_npu_blocks, num_disk):
    prepared = manual_prepared(per_npu_blocks, num_disk)
    summaries = {}
    for policy in POLICIES:
        kwargs = {
            "bw_table": {},
            "policy": policy,
            "num_npu": len(per_npu_blocks),
            "num_disk": num_disk,
            "n_layers": 1,
            "prepared_inputs": prepared,
            "submit_order_seed": 0,
        }
        if policy in sim.QOS_POLICIES:
            kwargs["qos_config"] = qos_config()
        _, summaries[policy] = sim.simulate_continuous(**kwargs)
    return summaries


class StaticQoSTests(unittest.TestCase):
    def test_cir_and_category_path_layout(self):
        config = qos_config()
        self.assertEqual(len(category_path_cirs()), 256)
        self.assertAlmostEqual(sum(category_path_cirs()), sim.DISK_BW)
        totals = [
            sum(
                config.path_cirs[path]
                for path in sim.client_category_paths(category, config)
            )
            for category in sim.QOS_ROUTING_CATEGORIES
        ]
        for actual, expected in zip(totals, CATEGORY_CIR_GBPS):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(sim.client_category_paths("LL", config)[:4], (28, 60, 92, 124))

    def test_selector_is_per_io_and_does_not_mutate_snapshot(self):
        config = qos_config()
        allowed = sim.client_category_paths("LL", config)
        counts = [0] * 256
        before = counts.copy()
        selected = sim.client_select_qos_paths(
            block_sizes_gb=[0.001] * 8,
            path_io_counts=counts,
            allowed_path_ids=allowed,
            routing_config=sim.ClientRoutingConfig(config, sim.DISK_BW),
        )
        self.assertEqual(selected, list(allowed[:8]))
        self.assertEqual(counts, before)
        self.assertTrue(set(selected) <= set(allowed))

    def test_256_count_abi_and_path_fcfs(self):
        state = sim.DiskState(0)
        scheduler = sim.DiskIOScheduler(
            state, sim.POLICY_QOS_STATIC_CIR, sim.DISK_BW, qos_config()
        )
        first, second = flow(17, 0), flow(17, 1)
        empty = scheduler.report_path_io_counts(0.0)
        scheduler.enqueue_many([first, second], 0.0)
        queued = scheduler.report_path_io_counts(0.0)
        self.assertEqual(len(queued), 256)
        self.assertEqual(queued[17], empty[17] + 2)
        self.assertEqual(queued[:17] + queued[18:], empty[:17] + empty[18:])

        active = scheduler.dispatch(0.0, [], schedule_completion=False)
        self.assertIs(active, first)
        self.assertEqual(scheduler.report_path_io_counts(0.0)[17], 2)
        scheduler.complete_ready_flows(first.end_time)
        active = scheduler.dispatch(first.end_time, [], schedule_completion=False)
        self.assertIs(active, second)
        self.assertEqual(len(state.active_flows), 1)
        scheduler.complete_ready_flows(second.end_time)
        self.assertEqual(scheduler.report_path_io_counts(second.end_time)[17], 0)

    def test_finite_pir_requires_hardware_token_bucket_spec(self):
        config = qos_config()
        finite_pir = sim.StaticQoSConfig(
            path_cirs=config.path_cirs,
            path_pirs=(5.0,) * 256,
            path_weights=config.path_weights,
            group_weights=config.group_weights,
            category_paths_per_group=config.category_paths_per_group,
        )
        with self.assertRaisesRegex(ValueError, "token-bucket"):
            sim.DiskIOScheduler(
                sim.DiskState(0),
                sim.POLICY_QOS_STATIC_CIR,
                sim.DISK_BW,
                finite_pir,
            )

    def test_cir_controls_service_opportunities_not_command_time(self):
        cirs = [0.0] * 256
        cirs[0] = 30.0
        cirs[1] = 10.0
        config = sim.StaticQoSConfig(
            path_cirs=tuple(cirs),
            path_pirs=(float("inf"),) * 256,
            path_weights=(1.0,) * 256,
            group_weights=(1.0,) * 8,
            category_paths_per_group=(12, 4, 12, 4),
        )
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0), sim.POLICY_QOS_STATIC_CIR, sim.DISK_BW, config
        )
        flows = [flow(0, index, size=0.001) for index in range(40)] + [
            flow(1, 40 + index, size=0.001) for index in range(40)
        ]
        scheduler.enqueue_many(flows, 0.0)

        selected = []
        current_time = 0.0
        for _ in range(40):
            active = scheduler.dispatch(
                current_time, [], schedule_completion=False
            )
            selected.append(active.queue_id)
            self.assertAlmostEqual(active.bw, 40.0)
            self.assertAlmostEqual(active.end_time - current_time, 0.025)
            current_time = active.end_time
            scheduler.complete_ready_flows(current_time)

        self.assertEqual(selected.count(0), 30)
        self.assertEqual(selected.count(1), 10)


class SimulationTests(unittest.TestCase):
    def test_shared_data_plane_service_costs(self):
        self.assertAlmostEqual(sim.ssd_command_service_time_ms(0.004, 40), 0.1)
        self.assertAlmostEqual(sim.npu_link_service_time_ms(0.004, 50), 0.08)

    def test_removed_routing_knobs_are_not_public_arguments(self):
        parameters = inspect.signature(sim.simulate_continuous).parameters
        for name in (
            "pressure_read_interval",
            "path_binding_batch_size",
            "client_submit_batch_size",
            "client_submit_interval_us",
            "npu_bw_limit",
        ):
            self.assertNotIn(name, parameters)

    def test_paired_runs_preserve_blocks_placement_and_cap(self):
        profile = (5.0, 1_000.0, 10.0, 0.017)
        table = {
            (3, 0): profile,
            (3, 1_023): profile,
            (81, 0): profile,
            (81, 80_895): profile,
        }
        prepared = sim.prepare_simulation_inputs(
            table,
            total_requests=4,
            n_layers=2,
            num_disk=4,
            ls_ratio=0.5,
            workload_seed=3,
            placement_seed=5,
        )
        common = dict(
            num_npu=4,
            num_disk=4,
            n_layers=2,
            prepared_inputs=prepared,
            submit_order_seed=9,
        )
        _, baseline = sim.simulate_continuous(
            table, policy=sim.POLICY_BASELINE_BYPASS, **common
        )
        _, qos = sim.simulate_continuous(
            table,
            policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=qos_config(),
            **common,
        )
        self.assertEqual(baseline["workload_fingerprint"], qos["workload_fingerprint"])
        self.assertEqual(baseline["placement_hash"], qos["placement_hash"])
        for summary in (baseline, qos):
            conserved = summary["block_conservation"]
            self.assertEqual(conserved["expected"], conserved["submitted"])
            self.assertEqual(conserved["submitted"], conserved["completed"])
            self.assertTrue(conserved["placement_targets_preserved"])
            self.assertLessEqual(
                summary["npu_link_peak_effective_bw_gbps"],
                sim.NPU_BW_LIMIT + 1e-9,
            )
            self.assertLessEqual(
                max(disk["max_backend_active_io"] for disk in summary["disk_stats"]),
                1,
            )
            self.assertEqual(summary["client_submit_batch_size"], 8)
        self.assertIsNone(baseline["pressure_read_interval"])
        self.assertEqual(baseline["path_selection"], "none")
        self.assertEqual(qos["pressure_read_interval"], 8)
        self.assertEqual(qos["path_selection"], "per_io")


class SharedTwoStageDataPlaneTests(unittest.TestCase):
    def test_simultaneous_ssd_arrivals_use_stable_policy_independent_tie_break(self):
        npu = sim.NPUState(0)
        later_request = flow(0, 0)
        later_request.request_id = 2
        earlier_request = flow(0, 1)
        earlier_request.request_id = 1
        for candidate in (later_request, earlier_request):
            candidate.link_enqueue_time = 1.0
            npu.link_pending.append(candidate)

        context = SimpleNamespace(
            npus=[npu],
            event_heap=[],
            pending_npu_link_start_ids={0},
            global_link_coordinator=None,
        )
        sim._flush_pending_npu_link_starts(context, 1.0)

        self.assertIs(npu.link_active_flow, earlier_request)
        self.assertEqual(list(npu.link_pending), [later_request])

    def test_single_io_has_same_ssd_and_link_cost_for_both_policies(self):
        summaries = paired_summaries([[(0, 0.004)]], num_disk=1)

        for summary in summaries.values():
            self.assertAlmostEqual(summary["makespan_ms"], 0.180, places=12)
            self.assertAlmostEqual(
                summary["request_metrics"][0]["io_wait_L0_ms"], 0.180, places=12
            )
            self.assertAlmostEqual(
                summary["disk_stats"][0]["busy_time_ms"], 0.100, places=12
            )
            self.assertAlmostEqual(
                summary["npu_link_stats"][0]["cap_hit_time_ms"], 0.080, places=12
            )
            self.assertEqual(summary["disk_stats"][0]["max_backend_active_io"], 1)
            self.assertEqual(summary["npu_link_stats"][0]["max_active_io"], 1)

    def test_same_disk_serializes_ssd_but_pipelines_npu_stage(self):
        summaries = paired_summaries(
            [[(0, 0.004)], [(0, 0.004)]],
            num_disk=1,
        )

        for summary in summaries.values():
            waits = sorted(row["io_wait_L0_ms"] for row in summary["request_metrics"])
            self.assertAlmostEqual(waits[0], 0.180, places=12)
            self.assertAlmostEqual(waits[1], 0.280, places=12)
            self.assertAlmostEqual(summary["makespan_ms"], 0.280, places=12)
            self.assertAlmostEqual(
                summary["disk_stats"][0]["busy_time_ms"], 0.200, places=12
            )

    def test_same_npu_serializes_link_and_conserves_blocks_and_bytes(self):
        summaries = paired_summaries(
            [[(0, 0.004), (1, 0.004)]],
            num_disk=2,
        )

        for summary in summaries.values():
            self.assertAlmostEqual(summary["makespan_ms"], 0.260, places=12)
            self.assertEqual(
                [disk["max_backend_active_io"] for disk in summary["disk_stats"]],
                [1, 1],
            )
            link = summary["npu_link_stats"][0]
            self.assertEqual(link["max_active_io"], 1)
            self.assertEqual(link["max_outstanding_io"], 2)
            self.assertEqual(link["dispatches"], 2)
            self.assertAlmostEqual(link["cap_hit_time_ms"], 0.160, places=12)

            conserved = summary["block_conservation"]
            self.assertEqual(
                (
                    conserved["expected"],
                    conserved["submitted"],
                    conserved["completed"],
                ),
                (2, 2, 2),
            )
            self.assertAlmostEqual(conserved["expected_read_gb"], 0.008, places=12)
            self.assertAlmostEqual(conserved["ssd_completed_read_gb"], 0.008, places=12)
            self.assertAlmostEqual(conserved["completed_read_gb"], 0.008, places=12)

    def test_ssd_completion_can_coincide_with_npu_link_completion(self):
        # 第一个 block: SSD 0.100ms + link 0.080ms；第二个 SSD 恰在
        # 0.180ms 完成，必须在同一时间点顺利接续到刚释放的 NPU link。
        summaries = paired_summaries(
            [[(0, 0.004), (1, 0.0072)]],
            num_disk=2,
        )

        for summary in summaries.values():
            self.assertAlmostEqual(summary["makespan_ms"], 0.324, places=12)
            link = summary["npu_link_stats"][0]
            self.assertEqual(link["dispatches"], 2)
            self.assertEqual(link["max_active_io"], 1)
            self.assertAlmostEqual(link["busy_time_ms"], 0.224, places=12)
            self.assertAlmostEqual(link["avg_queue_wait_ms"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
