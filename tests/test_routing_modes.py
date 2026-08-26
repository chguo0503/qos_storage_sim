import unittest

import sim


def equal_qos_config():
    return sim.StaticQoSConfig(
        path_cirs=(40.0 / 256,) * 256,
        path_pirs=(float("inf"),) * 256,
        path_weights=(1.0,) * 256,
        group_weights=(1.0,) * 8,
        category_paths_per_group=(8, 8, 8, 8),
    )


def one_layer_prepared(block_count=10):
    block_size = 0.001
    loads = [
        {
            "request_id": 0,
            "npu_id": 0,
            "seq_len_k": 3,
            "nql": 0,
            "per_layer_us": 1_000.0,
            "per_layer_kv_gb": block_count * block_size,
            "required_bw_input_gbps": 10.0,
            "category": "SS",
            "arrival_time": 0.0,
        }
    ]
    placement = {
        0: {0: tuple((0, block_size) for _ in range(block_count))}
    }
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


def demand_load(request_id, demand_gbps):
    return {
        "request_id": request_id,
        "required_bw_input_gbps": float(demand_gbps),
    }


class PressureCadenceTests(unittest.TestCase):
    def test_layer_refresh8_and_per_io_have_exact_report_counts(self):
        prepared = one_layer_prepared(block_count=10)
        variants = (
            (sim.ClientIOConfig("layer", None, 8), 1),
            (sim.ClientIOConfig("refresh8", 8, 8), 2),
            (sim.ClientIOConfig("per_io", 1, 8), 10),
        )
        makespans = []
        for client_config, expected_reports in variants:
            with self.subTest(mode=client_config.name):
                _, summary = sim.simulate_continuous(
                    {},
                    policy=sim.POLICY_QOS_STATIC_CIR,
                    qos_config=equal_qos_config(),
                    client_io_config=client_config,
                    num_npu=1,
                    num_disk=1,
                    n_layers=1,
                    prepared_inputs=prepared,
                    submit_order_seed=3,
                )
                self.assertEqual(summary["disk_stats"][0]["pressure_reports"], expected_reports)
                self.assertEqual(summary["pressure_read_interval"], client_config.pressure_window_io)
                self.assertEqual(summary["block_conservation"]["completed"], 10)
                makespans.append(summary["makespan_ms"])

        self.assertAlmostEqual(max(makespans), min(makespans), places=12)

    def test_live_pressure_analysis_observes_previously_enqueued_io(self):
        scheduler = sim.DiskIOScheduler(
            sim.DiskState(0),
            sim.POLICY_QOS_STATIC_CIR,
            sim.DISK_BW,
            equal_qos_config(),
        )
        before_count = scheduler.report_path_pressure_analysis(0.0).counts[17]
        queued = sim.BlockIOFlow(
            npu_id=0,
            request_id=0,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=0.001,
            queue_id=17,
            block_count=1,
            enqueue_time=0.0,
        )
        scheduler.enqueue_many([queued], 0.0)
        after = scheduler.report_path_pressure_analysis(0.0)

        self.assertEqual(before_count, 0)
        self.assertEqual(after.counts[17], 1)
        self.assertEqual(after.group_io_counts[0], 1)
        self.assertEqual(scheduler.pressure_reports, 2)


class DemandTicketTests(unittest.TestCase):
    def test_ticket_counts_follow_ten_to_thirty_demand_ratio(self):
        loads = (demand_load(0, 10.0), demand_load(1, 30.0))
        tickets = sim.allocate_demand_path_tickets(loads, total_paths=8)
        self.assertEqual(tickets, {0: 2, 1: 6})
        self.assertEqual(sum(tickets.values()), 8)

    def test_each_disk_partitions_all_paths_without_duplicates(self):
        loads = (demand_load(0, 10.0), demand_load(1, 30.0))
        pools = sim.build_demand_ticket_path_pools(
            loads, num_disk=3, total_paths=8
        )

        for disk_id in range(3):
            first = pools[(0, disk_id)]
            second = pools[(1, disk_id)]
            self.assertEqual((len(first), len(second)), (2, 6))
            self.assertEqual(len(set(first + second)), 8)
            self.assertEqual(set(first + second), set(range(8)))


if __name__ == "__main__":
    unittest.main()
