"""Business and simulator-integration tests for causal NPU assignment."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from unittest.mock import patch

import continuous_batch_sim as batch_sim
from continuous_batch_sim import (
    ContinuousBatchRequest, SteadyStateConfig, simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from npu_request_assignment import (
    IO_SIZE_BYTES, NPUBacklog, assign_requests_on_arrival, choose_npu,
    fixed_window_metrics,
    main,
)


class AssignmentTests(unittest.TestCase):
    def test_176_kib_units_and_least_backlog(self):
        backlogs = [NPUBacklog(i, work, 0) for i, work in enumerate((4, 2, 3, 5))]
        result = choose_npu(10, backlogs, 1, 1, n_layers=8)
        self.assertEqual(IO_SIZE_BYTES, 176 * 1024)
        self.assertEqual(result.npu_id, 1)
        self.assertEqual(result.predicted_finish_ms_by_npu, (22, 20, 21, 23))
        self.assertEqual(result.assumed_share_gib_s, 10)

    def test_io_cost_can_change_choice(self):
        backlogs = [NPUBacklog(0, 1, 1000), NPUBacklog(1, 2, 0)]
        pipeline = choose_npu(0, backlogs, 1, 1)
        compute = choose_npu(0, backlogs, 1, 1, policy="compute")
        self.assertEqual(pipeline.npu_id, 1)
        self.assertEqual(compute.npu_id, 0)

    def test_ids_are_not_indices_and_ties_are_stable(self):
        backlogs = [NPUBacklog(3, 0, 0), NPUBacklog(2, 0, 0)]
        self.assertEqual(choose_npu(0, backlogs, 1, 1).npu_id, 2)

    def test_fluid_records_distinct_fleet_objective(self):
        backlogs = [NPUBacklog(0, 1, 1000), NPUBacklog(1, 2, 0)]
        decision = choose_npu(0, backlogs, 100, 1, policy="fluid")
        self.assertEqual(decision.selection_objective, "estimated_whole_backlog_fluid_makespan")
        self.assertEqual(len(decision.selection_scores_ms_by_npu), 2)
        self.assertNotEqual(decision.predicted_finish_ms_by_npu, decision.selection_scores_ms_by_npu)
        # Keep the heavy-read work together when splitting it creates a
        # larger estimated simultaneous disk overload on the other NPU.
        self.assertEqual(decision.npu_id, 0)
        self.assertEqual(decision.selection_scores_ms_by_npu[0], 9)

    def test_fixed_window_clips_compute_stall_and_reports_idle(self):
        batches = []
        for npu in range(4):
            batches.append({
                "npu_id": npu, "admission_time_ms": 0, "completion_time_ms": 11,
                "layer_metrics": [{"compute_start_ms": 1, "compute_end_ms": 11,
                                   "io_barrier_wait_ms": 1}],
            })
        result = fixed_window_metrics({"microbatch_metrics": batches}, 0, 10)
        self.assertEqual(result["mean_npu_utilization"], 0.9)
        self.assertEqual(result["io_stall_ms_by_npu"], [1] * 4)
        self.assertTrue(result["all_npus_have_active_requests_throughout"])
        batches.pop()
        result = fixed_window_metrics({"microbatch_metrics": batches}, 0, 10)
        self.assertEqual(result["idle_without_active_request_ms_by_npu"], [0, 0, 0, 10])
        self.assertFalse(result["all_npus_have_active_requests_throughout"])

    def test_cli_matches_workload_seed_and_native_submission_seed(self):
        fake_result = {"wall_seconds": 0, "summary": {
            "microbatch_metrics": [], "makespan_ms": 10,
            "fleet_npu_compute_utilization": 0.5, "avg_request_latency_ms": 5,
        }}
        with tempfile.TemporaryDirectory() as directory, \
             patch("sys.argv", ["npu_request_assignment.py", "--seed", "20260908", "--output", directory]), \
             patch("run_stall_policy_experiments.build_variable_requests",
                   return_value=(self.requests(), (), {})) as build, \
             patch("run_stall_policy_experiments.run_case", return_value=fake_result) as run, \
             redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(build.call_args.kwargs["seed"], 20260908)
        self.assertEqual(run.call_args.kwargs["seed"], 20260908)

    @staticmethod
    def requests():
        # Round-robin input assignment, with heterogeneous but causal arrivals.
        return tuple(
            ContinuousBatchRequest(
                request_id=i,
                npu_id=i % 4,
                arrival_time_ms=float(i // 4),
                load={"category": "SS", "per_layer_us": (1 + i % 3) * 1000},
                placement=(((0, IO_SIZE_BYTES / 1024**3),),),
            )
            for i in range(12)
        )

    @staticmethod
    def run_sim(requests):
        return simulate_continuous_batch(
            requests, num_npu=4, num_ssu=1, n_layers=8, batch_size=1,
            qos_config=static_qos_config(),
            client_io_config=routing_strategy_specs()[0].client_config(),
            cross_request_layer0_prefetch=True, disk_bw_gbps=40,
            npu_bw_gbps=50,
        )

    def test_actual_simulator_arrivals_and_input_immutability(self):
        requests = self.requests()
        original_handler = batch_sim._handle_arrival
        with assign_requests_on_arrival() as log:
            result = self.run_sim(requests)
        self.assertIs(batch_sim._handle_arrival, original_handler)
        self.assertEqual(len(log), len(requests))
        by_id = {row["request_id"]: row for row in result["request_metrics"]}
        for request, record in zip(requests, log):
            self.assertEqual(request.npu_id, request.request_id % 4)
            self.assertEqual(record["arrival_time_ms"], request.arrival_time_ms)
            self.assertEqual(by_id[request.request_id]["npu_id"], record["assigned_npu_id"])
            self.assertGreaterEqual(by_id[request.request_id]["admission_time_ms"], request.arrival_time_ms)
        self.assertTrue(all(result["invariants"].values()))
        self.assertTrue(any(row["original_npu_id"] != row["assigned_npu_id"] for row in log))

    def test_future_request_changes_cannot_change_earlier_decisions(self):
        requests = self.requests()
        changed = requests[:8] + tuple(
            replace(request, load={**request.load, "per_layer_us": 1_000_000})
            for request in requests[8:]
        )
        for policy in ("pipeline", "compute", "fluid"):
            with self.subTest(policy=policy):
                with assign_requests_on_arrival(policy=policy) as first:
                    self.run_sim(requests)
                with assign_requests_on_arrival(policy=policy) as second:
                    self.run_sim(changed)
                for a, b in zip(first[:8], second[:8]):
                    self.assertEqual(a["assigned_npu_id"], b["assigned_npu_id"])
                    self.assertEqual(a["selection_scores_ms_by_npu"], b["selection_scores_ms_by_npu"])

    def test_handler_restored_after_failure(self):
        original_handler = batch_sim._handle_arrival
        with self.assertRaisesRegex(RuntimeError, "example"):
            with assign_requests_on_arrival():
                raise RuntimeError("example")
        self.assertIs(batch_sim._handle_arrival, original_handler)

    def test_middle_one_second_steady_state_invariants(self):
        base = self.requests()
        requests = tuple(
            replace(
                base[i % 12], request_id=i, arrival_time_ms=0.0,
                load={**base[i % 12].load, "per_layer_us": 20000.0,
                      "generation": i // 4, "stream_id": i % 4},
            )
            for i in range(120)
        )
        with assign_requests_on_arrival() as log:
            result = simulate_continuous_batch(
                requests, num_npu=4, num_ssu=1, n_layers=8, batch_size=1,
                qos_config=static_qos_config(),
                client_io_config=routing_strategy_specs()[0].client_config(),
                cross_request_layer0_prefetch=True, disk_bw_gbps=40,
                npu_bw_gbps=50,
                steady_state=SteadyStateConfig(
                    warmup_requests_per_npu=2, settle_ms=10,
                    measurement_ms=1000, block_ms=100,
                    timeline_diagnostics=False,
                ),
            )
        self.assertEqual(len(log), 120)
        self.assertTrue(all(result["invariants"].values()))
        self.assertAlmostEqual(result["mean_npu_utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
