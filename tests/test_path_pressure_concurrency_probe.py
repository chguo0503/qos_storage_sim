from collections import Counter
import unittest
from unittest import mock

import analyze_path_pressure_concurrency as analyzer
import path_pressure_concurrency_probe as probe
import sim
from strategy_profiles import CURRENT_STATIC


def prepared_inputs(*, block_count, arrival_times):
    block_gb = 0.001
    request_loads = []
    placement = {}
    for request_id, arrival_time in enumerate(arrival_times):
        request_loads.append(
            {
                "request_id": request_id,
                "npu_id": request_id,
                "seq_len_k": 3,
                "nql": 0,
                "per_layer_us": 1_000.0,
                "per_layer_kv_gb": block_count * block_gb,
                "required_bw_input_gbps": 2.0,
                "category": "SS",
                "arrival_time": float(arrival_time),
            }
        )
        placement[request_id] = {
            0: tuple((0, block_gb) for _ in range(block_count))
        }
    return sim.PreparedSimulationInputs(
        request_loads=tuple(request_loads),
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(request_loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=1,
        placement_mode="manual",
    )


def run_config(num_npu):
    return {
        "num_npu": num_npu,
        "n_layers": 1,
        "ssu_list": [1],
        "ls_ratio": 0.5,
        "arrival_delay_ms": [0.0, 0.0],
        "seeds": {
            "workload": 1,
            "placement": 2,
            "submit_order": 3,
            "arrival_delay": 4,
        },
    }


def strategy(name):
    return {row.name: row for row in probe.strategies()}[name]


def simulate_with_enqueue_trace(prepared, selected):
    trace = []
    original_enqueue_many = sim.DiskIOScheduler.enqueue_many

    def traced_enqueue_many(scheduler, flows, current_time):
        trace.extend(
            (flow.npu_id, flow.block_idx, flow.queue_id, current_time)
            for flow in flows
        )
        return original_enqueue_many(scheduler, flows, current_time)

    with mock.patch.object(probe, "prepare", return_value=prepared), mock.patch.object(
        sim.DiskIOScheduler, "enqueue_many", new=traced_enqueue_many
    ):
        _, summary, request_metrics = probe.simulate_case(
            {}, run_config(len(prepared.request_loads)), 1, selected
        )
    return trace, summary, request_metrics


class NoPressureRoundRobinTests(unittest.TestCase):
    def test_no_state_reads_nothing_and_round_robins_same_legal_pool(self):
        legal_paths = sim.client_category_paths(
            "SS", CURRENT_STATIC.hardware_config()
        )
        block_count = 2 * len(legal_paths) + 3
        prepared = prepared_inputs(block_count=block_count, arrival_times=(0.0,))
        selected = strategy("no_pressure_rr_issue01us")
        trace = []
        original_enqueue_many = sim.DiskIOScheduler.enqueue_many

        def traced_enqueue_many(scheduler, flows, current_time):
            trace.extend(flow.queue_id for flow in flows)
            return original_enqueue_many(scheduler, flows, current_time)

        with mock.patch.object(probe, "prepare", return_value=prepared), mock.patch.object(
            sim.DiskIOScheduler, "enqueue_many", new=traced_enqueue_many
        ), mock.patch.object(
            sim.DiskIOScheduler,
            "report_path_pressure_analysis",
            side_effect=AssertionError("no-state planner read pressure analysis"),
        ) as pressure_report, mock.patch.object(
            sim.DiskIOScheduler,
            "report_path_io_counts",
            side_effect=AssertionError("no-state planner read Path counts"),
        ) as count_report:
            _, summary, _ = probe.simulate_case(
                {}, run_config(1), 1, selected
            )

        expected = [
            legal_paths[index % len(legal_paths)] for index in range(block_count)
        ]
        self.assertEqual(trace, expected)
        self.assertEqual(set(trace), set(legal_paths))
        counts = Counter(trace)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(pressure_report.call_count, 0)
        self.assertEqual(count_report.call_count, 0)
        self.assertEqual(summary["pressure_reports"], 0)
        self.assertEqual(summary["pressure_telemetry_mb"], 0.0)


class SubmissionAtomicityTests(unittest.TestCase):
    def test_default_zero_issue_interval_preserves_instantaneous_semantics(self):
        prepared = prepared_inputs(block_count=3, arrival_times=(0.0, 0.001))
        default_config = sim.ClientIOConfig("default_zero", 1, 1)
        explicit_config = sim.ClientIOConfig(
            "default_zero", 1, 1, issue_interval_us=0.0
        )
        self.assertEqual(default_config.issue_interval_us, 0.0)

        summaries = []
        for client_config in (default_config, explicit_config):
            _, summary = sim.simulate_continuous(
                {},
                policy=sim.POLICY_QOS_STATIC_CIR,
                qos_config=CURRENT_STATIC.hardware_config(),
                client_io_config=client_config,
                num_npu=2,
                num_disk=1,
                n_layers=1,
                prepared_inputs=prepared,
                submit_order_seed=3,
            )
            summaries.append(summary)

        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["client_submit_interval_us"], 0.0)
        self.assertEqual(
            [
                (row["npu_id"], row["time_ms"])
                for row in summaries[0]["client_submission"]["order_sample"]
            ],
            [(0, 0.0)] * 3 + [(1, 0.001)] * 3,
        )

    def test_batch8_is_atomic_but_batch1_interleaves_equal_ready_npus(self):
        prepared = prepared_inputs(block_count=3, arrival_times=(0.0, 0.0))

        batch8_trace, batch8_summary, _ = simulate_with_enqueue_trace(
            prepared, strategy("per_io_atomic8")
        )
        batch8_npus = [row[0] for row in batch8_trace]
        self.assertEqual(len(set(batch8_npus[:3])), 1)
        self.assertEqual(len(set(batch8_npus[3:])), 1)
        self.assertNotEqual(batch8_npus[0], batch8_npus[3])
        self.assertEqual(batch8_summary["client_submission"]["rounds"], 1)
        self.assertEqual(
            [
                row["io_count"]
                for row in batch8_summary["client_submission"]["order_sample"]
            ],
            [3, 3],
        )

        batch1_trace, batch1_summary, _ = simulate_with_enqueue_trace(
            prepared, strategy("per_io_batch1_zero")
        )
        batch1_npus = [row[0] for row in batch1_trace]
        self.assertEqual(len(batch1_npus), 6)
        for offset in range(0, len(batch1_npus), 2):
            self.assertEqual(set(batch1_npus[offset : offset + 2]), {0, 1})
        submission = batch1_summary["client_submission"]
        self.assertEqual(submission["rounds"], 3)
        self.assertEqual(submission["multi_npu_rounds"], 3)
        self.assertTrue(
            all(row["io_count"] == 1 for row in submission["order_sample"])
        )

    def test_batch1_does_not_interleave_distinct_ready_times(self):
        prepared = prepared_inputs(block_count=3, arrival_times=(0.0, 0.001))
        trace, summary, _ = simulate_with_enqueue_trace(
            prepared, strategy("per_io_batch1_zero")
        )

        self.assertEqual([row[0] for row in trace], [0, 0, 0, 1, 1, 1])
        self.assertEqual([row[3] for row in trace], [0.0] * 3 + [0.001] * 3)
        self.assertEqual(summary["client_submission"]["multi_npu_rounds"], 0)

    def test_finite_issue_interval_interleaves_distinct_ready_times(self):
        prepared = prepared_inputs(block_count=3, arrival_times=(0.0, 0.00005))
        trace, summary, _ = simulate_with_enqueue_trace(
            prepared, strategy("per_io_issue01us")
        )

        self.assertEqual([row[0] for row in trace], [0, 1, 0, 1, 0, 1])
        for actual, expected in zip(
            [row[3] for row in trace],
            [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.00025],
        ):
            self.assertAlmostEqual(actual, expected, places=15)
        self.assertEqual(summary["client_submission"]["multi_npu_rounds"], 0)


class ProbeContractTests(unittest.TestCase):
    def test_formal_runtime_and_strategy_matrix(self):
        runtime = probe.runtime(42)
        self.assertEqual(runtime["num_npu"], 128)
        self.assertEqual(runtime["n_layers"], 16)
        self.assertEqual(runtime["ssu_list"], [40, 56, 80])
        self.assertEqual(runtime["arrival_delay_ms"], [0.0, 5.0])

        rows = probe.strategies()
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row.name for row in rows}), 10)
        expected = {
            "baseline_atomic8": ("none", None, 8, 0.0),
            "no_pressure_rr_atomic8": (
                "no_pressure_round_robin",
                None,
                8,
                0.0,
            ),
            "refresh8_atomic8": ("refresh8_shadow", 8, 8, 0.0),
            "per_io_atomic8": ("per_io_live", 1, 8, 0.0),
            "refresh8_batch1_zero": ("refresh8_shadow", 8, 1, 0.0),
            "per_io_batch1_zero": ("per_io_live", 1, 1, 0.0),
            "baseline_issue01us": ("none", None, 1, 0.1),
            "no_pressure_rr_issue01us": (
                "no_pressure_round_robin",
                None,
                1,
                0.1,
            ),
            "refresh8_issue01us": ("refresh8_shadow", 8, 1, 0.1),
            "per_io_issue01us": ("per_io_live", 1, 1, 0.1),
        }
        self.assertEqual(
            {
                row.name: (
                    row.pressure_mode,
                    row.pressure_window_io,
                    row.submit_batch_size,
                    row.issue_interval_us,
                )
                for row in rows
            },
            expected,
        )
        self.assertTrue(
            all(
                row.policy == sim.POLICY_BASELINE_BYPASS
                if row.pressure_mode == "none"
                else row.policy == sim.POLICY_QOS_STATIC_CIR
                for row in rows
            )
        )

        spec = probe.experiment_spec(runtime, rows)
        self.assertEqual(spec["runtime"], runtime)
        self.assertEqual(spec["backend"]["ssd"], "one nonpreemptive command, size/40GBps")
        self.assertEqual(spec["backend"]["npu"], "one FCFS receive command per NPU, size/50GBps")
        self.assertEqual(
            {row["name"] for row in spec["selected_strategies"]},
            {row.name for row in rows},
        )
        self.assertEqual(set(analyzer.LABELS), {row.name for row in rows})


if __name__ == "__main__":
    unittest.main()
