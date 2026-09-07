"""Independent workload/provenance/fairness checks for multi-SSU experiments."""

from collections import Counter
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import sim
from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import continuous_batch_input_fingerprint
from run_multi_ssu_stall_experiments import IO_GIB, build_workload, run_case, summarize


class MultiSSUExperimentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.cases = {ssu: build_workload(num_npu=32, num_ssu=ssu,
                                           seed=20260906, regime="near")
                         for ssu in (6, 7)}
            cls.table, cls.source = load_authenticated_bw_table(32)

    def test_initial_backlog_is_exactly_four_per_original_npu(self):
        for ssu, (requests, metadata) in self.cases.items():
            with self.subTest(ssu=ssu):
                initial = Counter(r.npu_id for r in requests if r.arrival_time_ms == 0)
                self.assertEqual(initial, Counter({n: 4 for n in range(32)}))
                self.assertEqual(metadata["initial_backlog"], 4)
                self.assertEqual(len(requests), 32 * 22)
                for n in range(32):
                    arrivals = [r.arrival_time_ms for r in requests if r.npu_id == n]
                    self.assertGreater(arrivals[4], 0)
                    self.assertEqual(arrivals, sorted(arrivals))

    def test_every_lane_has_same_raw_profile_multiset_not_same_order(self):
        for ssu, (requests, metadata) in self.cases.items():
            with self.subTest(ssu=ssu):
                expected = Counter({(p["seq_len_k"], p["nql"]): p["quota"]
                                    for p in metadata["profiles"]})
                orders = []
                for n in range(32):
                    sequence = tuple((r.load["seq_len_k"], r.load["nql"])
                                     for r in requests if r.npu_id == n)
                    self.assertEqual(Counter(sequence), expected)
                    orders.append(sequence)
                self.assertGreater(len(set(orders)), 1)
                self.assertIn("synthetic", metadata["sampling_caveat"])

    def test_data_values_full_176_kib_io_and_native_placement(self):
        for ssu, (requests, metadata) in self.cases.items():
            with self.subTest(ssu=ssu):
                self.assertEqual(metadata["source"]["source_sha256"],
                                 self.source["source_sha256"])
                self.assertEqual(metadata["io_bytes"], 180224)
                self.assertEqual(metadata["n_layers"], 8)
                for request in requests:
                    seq, nql = request.load["seq_len_k"], request.load["nql"]
                    bw, compute_us, ttft, volume = self.table[seq, nql]
                    self.assertEqual(request.load["per_layer_us"], compute_us)
                    self.assertEqual(request.load["required_bw_input_gbps"], bw)
                    self.assertEqual(request.load["source_ttft_ms"], ttft)
                    self.assertFalse(request.load["constructed_profile"])
                    placement = request.placement[0]
                    self.assertEqual(len(placement), (seq * 1024 - nql) // 128)
                    self.assertTrue(all(size == IO_GIB for _, size in placement))
                    self.assertAlmostEqual(sum(size for _, size in placement), volume)
                    for block in (0, len(placement) // 2, len(placement) - 1):
                        self.assertEqual(placement[block][0], sim.block_ring_hash_disk_id(
                            request.request_id, block, ssu))

    def test_near_capacity_checks_each_ssu_not_only_total(self):
        for ssu, (_, metadata) in self.cases.items():
            with self.subTest(ssu=ssu):
                demand = metadata["input_demand"]
                self.assertLessEqual(max(demand["per_ssu_gib_s"]), 39.96 + 1e-9)
                self.assertGreater(demand["hottest_ssu_load_ratio"], .90)
                self.assertLessEqual(demand["aggregate_load_ratio"], 1)
                self.assertLessEqual(max(demand["per_npu_gib_s"]), 50)
                # Each lane has equal aggregate bytes/compute, although native
                # placement makes their per-SSU components different.
                self.assertLess(max(demand["per_npu_gib_s"]) -
                                min(demand["per_npu_gib_s"]), 1e-10)

    def test_seed_reproducibility_and_profile_order_variability(self):
        with redirect_stdout(io.StringIO()):
            a, ma = build_workload(num_npu=4, num_ssu=6, seed=31)
            b, mb = build_workload(num_npu=4, num_ssu=6, seed=31)
            c, _ = build_workload(num_npu=4, num_ssu=6, seed=32)
        self.assertEqual(ma, mb)
        self.assertEqual(continuous_batch_input_fingerprint(a),
                         continuous_batch_input_fingerprint(b))
        self.assertNotEqual(continuous_batch_input_fingerprint(a),
                            continuous_batch_input_fingerprint(c))

    def test_baseline_once_and_controller_receive_same_requests_and_seed(self):
        requests, metadata = self.cases[6]
        fingerprint = continuous_batch_input_fingerprint(requests)
        fake = self.fake_summary(32)
        fake["invariants"] = {"test": True}
        observed = []

        def native_spy(received, **kwargs):
            self.assertIs(received, requests)
            observed.append(kwargs)
            return fake

        with patch("run_multi_ssu_stall_experiments.simulate_continuous_batch", native_spy):
            outputs = [run_case(requests, metadata, strategy=strategy)
                       for strategy in ("baseline", "layer_once", "demand", "deadline")]
        self.assertTrue(all(row["input_fingerprint"] == fingerprint for row in outputs))
        self.assertTrue(all(row["submit_seed"] == metadata["seed"] for row in outputs))
        self.assertTrue(all(kw["submit_order_seed"] == metadata["seed"] for kw in observed))
        for kwargs in observed:
            self.assertEqual((kwargs["num_npu"], kwargs["num_ssu"]), (32, 6))
            self.assertEqual(kwargs["n_layers"], 8)
            self.assertTrue(kwargs["cross_request_layer0_prefetch"])
            self.assertEqual(kwargs["disk_bw_gbps"], 40)
            self.assertEqual(kwargs["npu_bw_gbps"], 50)
        self.assertEqual(continuous_batch_input_fingerprint(requests), fingerprint)

    @staticmethod
    def fake_summary(num_npu=2):
        return {"num_npu": num_npu, "microbatch_metrics": [],
                "fleet_npu_compute_utilization": .5, "makespan_ms": 10,
                "avg_request_latency_ms": 5, "p99_request_latency_ms": 8,
                "avg_admission_wait_ms": 2, "avg_io_stall_ms": 1,
                "request_count": 2}

    def test_fixed_window_distinguishes_io_stall_from_no_request_idle(self):
        summary = self.fake_summary()
        summary["microbatch_metrics"] = [
            {"npu_id": 0, "admission_time_ms": 0, "completion_time_ms": 12,
             "layer_metrics": [{"compute_start_ms": 2, "compute_end_ms": 12}]},
            {"npu_id": 1, "admission_time_ms": 5, "completion_time_ms": 10,
             "layer_metrics": [{"compute_start_ms": 7, "compute_end_ms": 9}]},
        ]
        result = summarize(summary, 0, 10)
        self.assertEqual(result["npu_utilizations"], [.8, .2])
        self.assertEqual(result["mean_npu_utilization"], .5)
        self.assertEqual(result["io_stall_ms_by_npu"], [2, 3])
        self.assertEqual(result["idle_ms_by_npu"], [0, 5])
        self.assertFalse(result["all_npus_active_whole_window"])


if __name__ == "__main__":
    unittest.main()
