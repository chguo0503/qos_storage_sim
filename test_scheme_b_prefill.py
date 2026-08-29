import unittest

import sim
from strategy_profiles import FINAL_STATIC
from scheme_b_prefill import (
    PATH_COUNT,
    build_scheme_b_prefill_plan,
    dedicated_path_id,
)


def prepared_from_work(work_by_npu, compute_s=1.0, n_layers=16):
    loads = []
    placements = {}
    for npu, work_by_ssu in enumerate(work_by_npu):
        loads.append(
            {
                "request_id": npu,
                "npu_id": npu,
                "per_layer_us": compute_s * 1e6,
            }
        )
        blocks = tuple(
            (ssu, float(work))
            for ssu, work in enumerate(work_by_ssu)
            if work > 0.0
        )
        placements[npu] = {layer: blocks for layer in range(n_layers)}
    return sim.PreparedSimulationInputs(
        request_loads=tuple(loads),
        placement_by_request=placements,
        workload_seed=1,
        placement_seed=2,
        workload_hash="workload",
        placement_hash="placement",
        n_layers=n_layers,
        num_disk=len(work_by_npu[0]),
        placement_mode=sim.PLACEMENT_BLOCK_RING_HASH,
    )


class SchemeBPrefillTest(unittest.TestCase):
    def test_ten_plus_thirty_uses_exact_demand(self):
        plan = build_scheme_b_prefill_plan(prepared_from_work(((10.0,), (30.0,))))

        self.assertEqual(plan.demands_gbps, ((10.0,), (30.0,)))
        self.assertEqual(plan.grants_gbps, ((10.0,), (30.0,)))
        self.assertEqual(plan.path_by_npu, (0, 32))
        self.assertEqual(plan.qos_configs[0].path_cirs[0], 10.0)
        self.assertEqual(plan.qos_configs[0].path_cirs[32], 30.0)
        self.assertAlmostEqual(sum(plan.qos_configs[0].path_cirs), 40.0)
        self.assertTrue(plan.summary()["all_constraints_hold"])

    def test_max_min_obeys_ssu_and_npu_caps(self):
        prepared = prepared_from_work(
            ((30.0, 30.0), (30.0, 0.0), (30.0, 0.0))
        )
        plan = build_scheme_b_prefill_plan(prepared)

        self.assertLessEqual(sum(plan.grants_gbps[0]), 50.0 + 1e-9)
        for ssu in range(plan.num_ssu):
            self.assertLessEqual(
                sum(row[ssu] for row in plan.grants_gbps), 40.0 + 1e-9
            )
        self.assertTrue(all(plan.constraints().values()))

    def test_path_mapping_is_unique_and_group_balanced(self):
        paths = tuple(dedicated_path_id(npu) for npu in range(128))

        self.assertEqual(len(set(paths)), 128)
        self.assertLess(max(paths), PATH_COUNT)
        self.assertEqual(
            [sum(path // 32 == group for path in paths) for group in range(8)],
            [16] * 8,
        )

    def test_ring_placement_plan_is_deterministic(self):
        table = {(1, 8): (10.0, 10_000.0, 100.0, 0.1)}
        prepared = sim.prepare_simulation_inputs(
            table,
            total_requests=8,
            n_layers=16,
            num_disk=3,
            workload_seed=42,
            arrival_delay_seed=43,
            arrival_delay_max_ms=5.0,
        )

        first = build_scheme_b_prefill_plan(prepared)
        second = build_scheme_b_prefill_plan(prepared)

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.qos_configs), 3)
        self.assertEqual(len(first.qos_configs[0].path_cirs), 256)
        self.assertEqual(first.summary()["control_plane_updates"], 1)
        self.assertEqual(first.summary()["reuse_layers"], 16)

    def test_per_ssu_config_keeps_legacy_broadcast_exact(self):
        table = sim.load_bw_table_cache(num_npu=4)
        prepared = sim.prepare_simulation_inputs(
            table,
            total_requests=4,
            n_layers=2,
            num_disk=3,
            ls_ratio=0.5,
            workload_seed=31,
            arrival_delay_seed=33,
            arrival_delay_max_ms=5.0,
        )
        qos = FINAL_STATIC.hardware_config()
        common = {
            "policy": sim.POLICY_QOS_STATIC_CIR,
            "num_npu": 4,
            "num_disk": 3,
            "n_layers": 2,
            "client_io_config": sim.ClientIOConfig(
                "test_baseline",
                None,
                submit_batch_size=1,
                issue_interval_us=0.1,
                path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
            ),
            "submit_order_seed": 34,
            "prepared_inputs": prepared,
        }

        _, legacy = sim.simulate_continuous(table, qos_config=qos, **common)
        _, per_ssu = sim.simulate_continuous(
            table, qos_configs_by_disk=(qos,) * 3, **common
        )

        self.assertEqual(legacy, per_ssu)

    def test_plan_runs_on_the_shared_ssd40_npu50_data_plane(self):
        table = sim.load_bw_table_cache(num_npu=4)
        prepared = sim.prepare_simulation_inputs(
            table,
            total_requests=4,
            n_layers=2,
            num_disk=2,
            workload_seed=41,
            arrival_delay_seed=43,
            arrival_delay_max_ms=5.0,
        )
        plan = build_scheme_b_prefill_plan(prepared)
        _, result = sim.simulate_continuous(
            table,
            policy=sim.POLICY_QOS_STATIC_CIR,
            num_npu=4,
            num_disk=2,
            n_layers=2,
            qos_configs_by_disk=plan.qos_configs,
            npu_dedicated_paths=plan.path_by_npu,
            client_io_config=sim.ClientIOConfig(
                "scheme_b_test",
                None,
                submit_batch_size=1,
                issue_interval_us=0.1,
                path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
            ),
            prepared_inputs=prepared,
        )

        self.assertEqual(result["path_pool_mode"], "npu_dedicated")
        self.assertEqual(result["path_selection"], "npu_dedicated")
        self.assertEqual(
            {path for disk in result["disk_stats"] for path in disk["enqueued_path_ids"]},
            set(plan.path_by_npu),
        )
        self.assertEqual(
            sum(disk["pressure_reports"] for disk in result["disk_stats"]), 0
        )
        self.assertLessEqual(
            result["npu_link_peak_effective_bw_gbps"], sim.NPU_BW_LIMIT + 1e-9
        )
        self.assertLessEqual(
            max(disk["max_backend_active_io"] for disk in result["disk_stats"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
