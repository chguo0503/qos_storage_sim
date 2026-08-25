import inspect
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


class SimulationTests(unittest.TestCase):
    def test_npu_cap_examples(self):
        self.assertEqual(sim.proportional_npu_cap([40]), (40.0,))
        self.assertEqual(sim.proportional_npu_cap([40, 40]), (25.0, 25.0))
        self.assertEqual(sim.proportional_npu_cap([40, 5]), (40.0, 5.0))
        self.assertEqual(sim.proportional_npu_cap([40] * 8), (6.25,) * 8)

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
            self.assertEqual(summary["pressure_read_interval"], 8)
            self.assertEqual(summary["path_selection"], "per_io")
            self.assertEqual(summary["client_submit_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
