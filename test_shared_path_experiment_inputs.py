"""Immutable varying inputs, topology fairness, and uncensored SLO accounting."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_shared_path_experiments import (
    IO_BYTES, IO_GIB, LAYERS, STRATEGIES, build_workload, case_name,
    logical_input_fingerprint, main, save_result, summarize_slo, summarize_window,
)
from sweep_shared_path_experiments import jobs, job_name


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.near = {(ssu, seed): build_workload(num_ssu=ssu, seed=seed)
                    for ssu in (5, 6, 7) for seed in (20260906, 20260907)}
        cls.table, _ = load_authenticated_bw_table(32)

    def test_complete_704_population_near_capacity_and_full_blocks(self):
        for (ssu, seed), (requests, metadata) in self.near.items():
            with self.subTest(ssu=ssu, seed=seed):
                self.assertEqual(IO_BYTES, 180224)
                self.assertEqual(len(requests), 704)
                self.assertEqual(len({r.request_id for r in requests}), 704)
                self.assertEqual(Counter(r.npu_id for r in requests), {n: 22 for n in range(32)})
                self.assertEqual(metadata["initial_arrived_request_count"], 128)
                self.assertLessEqual(max(metadata["input_demand"]["per_ssu_gib_s"]), 39.96)
                self.assertGreater(max(metadata["input_demand"]["per_ssu_gib_s"]), 39)
                for request in requests:
                    profile = self.table[request.load["seq_len_k"], request.load["nql"]]
                    self.assertEqual(request.load["per_layer_us"], profile[1])
                    self.assertAlmostEqual(sum(v for _, v in request.placement[0]), profile[3])
                    self.assertTrue(all(0 <= s < ssu and v == IO_GIB for s, v in request.placement[0]))

    def test_every_lane_changes_profile_and_preserves_declared_quota(self):
        for requests, metadata in self.near.values():
            expected = {(p["seq_len_k"], p["nql"]): p["quota"] for p in metadata["profiles"]}
            for npu in range(32):
                lane = [r for r in requests if r.npu_id == npu]
                counts = Counter((r.load["seq_len_k"], r.load["nql"]) for r in lane)
                self.assertEqual(counts, expected)
                self.assertGreater(len(counts), 1)

    def test_slo_uses_8_layers_not_raw_78_layer_ttft(self):
        for _, metadata in self.near.values():
            for profile in metadata["profiles"]:
                self.assertAlmostEqual(profile["source_ttft_ms_not_used_as_8layer_slo"],
                                       78 * profile["compute_ms"])
                self.assertEqual(profile["ideal_8layer_ms"], 8 * profile["compute_ms"])
                self.assertAlmostEqual(profile["slo_1p5_ms"], 12 * profile["compute_ms"])

    def test_same_trace_topology_changes_only_placement(self):
        examples = [build_workload(num_npu=4, num_ssu=s, regime="same_trace") for s in (5, 6, 7)]
        self.assertEqual(len({m["logical_input_fingerprint"] for _, m in examples}), 1)
        self.assertEqual(len({continuous_batch_input_fingerprint(r) for r, _ in examples}), 3)
        self.assertEqual(len({m["replacements"] for _, m in examples}), 1)
        for index in range(len(examples[0][0])):
            a, b, c = [requests[index] for requests, _ in examples]
            self.assertEqual((a.request_id, a.npu_id, a.arrival_time_ms, a.load),
                             (b.request_id, b.npu_id, b.arrival_time_ms, b.load))
            self.assertEqual(a.load, c.load)

    def test_reproducible_same_strategy_independent_input(self):
        a, ma = build_workload(num_npu=4, num_ssu=6)
        b, mb = build_workload(num_npu=4, num_ssu=6)
        self.assertEqual(continuous_batch_input_fingerprint(a), continuous_batch_input_fingerprint(b))
        self.assertEqual(ma, mb)
        changed = (replace(a[0], arrival_time_ms=a[0].arrival_time_ms + 1),) + a[1:]
        self.assertNotEqual(logical_input_fingerprint(a), logical_input_fingerprint(changed))

    def test_small_keeps_topology_and_raw_profiles_but_is_not_window_benchmark(self):
        requests, metadata = build_workload(num_ssu=5, small=True)
        self.assertEqual(len(requests), 64)
        self.assertTrue(metadata["small"])
        self.assertEqual(metadata["num_npu"], 32)
        self.assertEqual(metadata["initial_backlog"], 2)
        self.assertTrue(case_name(metadata, "strategy1").endswith("_small"))

    def test_artifact_roundtrip_retains_complete_immutable_manifest(self):
        requests, metadata = build_workload(num_npu=2, num_ssu=5, small=True)
        source = "print('frozen source')\n"
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        result = {"strategy": "baseline", "metadata": metadata,
                  "input_fingerprint": continuous_batch_input_fingerprint(requests),
                  "logical_input_fingerprint": metadata["logical_input_fingerprint"],
                  "core_and_policy_sha256": {"example.py": source_hash},
                  "_source_texts": {"example.py": source}}
        with tempfile.TemporaryDirectory() as directory:
            destination = save_result(result, requests, directory)
            saved = json.loads(destination.read_text())
            artifact = json.loads((Path(directory) / saved["input_artifact"]["path_relative_to_data"]).read_text())
            restored = tuple(ContinuousBatchRequest(**row) for row in artifact["requests"])
            self.assertEqual(continuous_batch_input_fingerprint(restored), result["input_fingerprint"])
            self.assertEqual(logical_input_fingerprint(restored), metadata["logical_input_fingerprint"])
            archived_source = Path(directory) / saved["source_artifacts"]["example.py"]
            self.assertEqual(archived_source.read_text(), source)
            self.assertEqual(hashlib.sha256(archived_source.read_bytes()).hexdigest(), source_hash)

    def test_describe_only_does_not_import_adapter_or_run_simulator(self):
        with patch("sys.argv", ["runner", "--small", "--num-npu", "2", "--describe-only"]), \
             patch("run_shared_path_experiments.run_case") as run, redirect_stdout(io.StringIO()) as output:
            main()
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["request_count"], 4)


class MetricTests(unittest.TestCase):
    def test_window_clips_and_distinguishes_idle_from_stall(self):
        summary = {
            "num_npu": 2, "microbatch_metrics": [
                {"npu_id": 0, "admission_time_ms": 900, "completion_time_ms": 2100,
                 "layer_metrics": [{"compute_start_ms": 1100, "compute_end_ms": 2100}]},
                {"npu_id": 1, "admission_time_ms": 1500, "completion_time_ms": 2100,
                 "layer_metrics": [{"compute_start_ms": 1600, "compute_end_ms": 2100}]},
            ], "fleet_npu_compute_utilization": .5, "makespan_ms": 2100,
            "avg_request_latency_ms": 1000, "p99_request_latency_ms": 1100,
            "avg_processing_latency_ms": 1000, "avg_admission_wait_ms": 0,
            "avg_io_stall_ms": 100, "request_count": 2,
        }
        measured = summarize_window(summary)
        self.assertEqual(measured["compute_ms_by_npu"], [900, 400])
        self.assertEqual(measured["active_ms_by_npu"], [1000, 500])
        self.assertEqual(measured["io_stall_ms_by_npu"], [100, 100])
        self.assertEqual(measured["idle_ms_by_npu"], [0, 500])
        self.assertFalse(measured["all_npus_active_whole_window"])
        self.assertEqual(measured["mean_npu_utilization"], .65)

    def test_full_slo_and_window_cohorts_include_completions_after_window(self):
        requests = tuple(ContinuousBatchRequest(i, 0, arrival,
            {"seq_len_k": 32, "nql": nql}, (((0, IO_GIB),),))
            for i, arrival, nql in ((0, 0, 128), (1, 1200, 256)))
        rows = [
            {"request_id": 0, "arrival_time_ms": 0, "admission_time_ms": 1000,
             "completion_time_ms": 2010, "own_compute_ms": 800},
            {"request_id": 1, "arrival_time_ms": 1200, "admission_time_ms": 1500,
             "completion_time_ms": 2005, "own_compute_ms": 200},
        ]
        result = summarize_slo({"request_metrics": rows,
                                "invariants": {"all_requests_completed": True}}, requests)
        self.assertEqual(result["all_requests"]["admission"], {"passed": 1, "count": 2, "rate": .5})
        self.assertEqual(result["all_requests"]["arrival"]["passed"], 0)
        self.assertEqual(result["window_admissions"]["admission"]["count"], 2)
        self.assertEqual(result["window_admissions"]["completion_after_window_end_count"], 2)
        self.assertEqual(result["window_arrivals"]["request_ids"], [1])
        self.assertEqual([p["admission"]["count"] for p in result["per_profile"]], [1, 1])
        with self.assertRaises(AssertionError):
            summarize_slo({"request_metrics": rows[:1],
                           "invariants": {"all_requests_completed": False}}, requests)

    def test_sweep_has_all_five_policies_and_matched_primary_cells(self):
        self.assertEqual(len(jobs("primary")), 30)
        self.assertEqual(len(jobs("same_trace")), 15)
        self.assertEqual(len(jobs("pilot")), 5)
        self.assertEqual({j["strategy"] for j in jobs("pilot")}, set(STRATEGIES))
        self.assertTrue(all(j["num_ssu"] == 6 and j["seed"] == 20260906 for j in jobs("pilot")))
        self.assertEqual(len({job_name(j) for j in jobs("primary")}), 30)
        self.assertTrue(all(j["small"] for j in jobs("smoke")))


if __name__ == "__main__":
    unittest.main()
