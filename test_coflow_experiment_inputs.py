"""Exact postinitial arrival work, unchanged manifests and matched populations."""

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

from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_coflow_experiments import (
    IO_BYTES, LAYERS, REGIME, STRATEGIES, TARGET_ARRIVAL_GIB_S,
    build_reference_workload, build_workload, case_name, logical_input_fingerprint,
    main, placement_fingerprint, request_read_bytes, retime_arrivals, run_case,
    save_result, source_files,
)
from sweep_coflow_experiments import jobs, job_name


class CoflowInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {ssu: build_workload(num_ssu=ssu) for ssu in (5, 6, 7)}

    def test_full_population_raw_quota_and_initial_work(self):
        expected = {(32, 128): 4, (32, 256): 2, (64, 512): 2, (96, 512): 2,
                    (192, 512): 4, (192, 1024): 3, (192, 2048): 5}
        for requests, metadata in self.examples.values():
            self.assertEqual(len(requests), 704)
            self.assertEqual(len({r.request_id for r in requests}), 704)
            self.assertEqual(metadata["initial_arrived_request_count"], 128)
            self.assertEqual(metadata["arrival_workload"]["postinitial_request_count"], 576)
            for npu in range(32):
                lane = [r for r in requests if r.npu_id == npu]
                self.assertEqual(Counter((r.load["seq_len_k"], r.load["nql"]) for r in lane), expected)
                self.assertEqual(sum(r.arrival_time_ms == 0 for r in lane), 4)
                for r in lane:
                    self.assertEqual(r.load["arrival_ms"], r.arrival_time_ms)
                    self.assertEqual(r.load["arrival_time"], r.arrival_time_ms)
                    self.assertEqual(request_read_bytes(r), LAYERS * len(r.placement[0]) * IO_BYTES)
                    self.assertFalse(r.load["constructed_profile"])

    def test_exact_postinitial_rate_and_independent_initial_backlog(self):
        for requests, metadata in self.examples.values():
            a = metadata["arrival_workload"]
            later = [r for r in requests if r.arrival_time_ms > 0]
            initial = [r for r in requests if r.arrival_time_ms == 0]
            seconds = max(r.arrival_time_ms for r in later) / 1000
            actual = sum(request_read_bytes(r) for r in later) / 2**30 / seconds
            self.assertAlmostEqual(actual, TARGET_ARRIVAL_GIB_S, places=9)
            self.assertAlmostEqual(sum(a["postinitial_gib_s_by_ssu"]), actual, places=9)
            self.assertEqual(a["initial_arrived_read_bytes"], sum(request_read_bytes(r) for r in initial))
            self.assertEqual(a["postinitial_read_bytes"], sum(a["postinitial_read_bytes_by_ssu"]))
            self.assertGreater(sum(request_read_bytes(r) for r in requests) / 2**30 / seconds, actual)
            self.assertTrue(a["initial_work_excluded_from_rate"])

    def test_reference_order_and_every_byte_clock_increment(self):
        requests, _ = self.examples[6]
        old_order = sorted((r for r in requests if r.arrival_time_ms > 0),
                           key=lambda r: (r.load["reference_arrival_ms"], r.request_id))
        new_order = sorted((r for r in requests if r.arrival_time_ms > 0),
                           key=lambda r: (r.arrival_time_ms, r.request_id))
        self.assertEqual([r.request_id for r in old_order], [r.request_id for r in new_order])
        previous = 0
        for r in new_order:
            self.assertGreater(r.arrival_time_ms, previous)
            self.assertAlmostEqual((r.arrival_time_ms - previous) * TARGET_ARRIVAL_GIB_S / 1000,
                                   request_read_bytes(r) / 2**30, places=10)
            previous = r.arrival_time_ms

    def test_canonical_grid_resists_picosecond_subdivision_noise(self):
        reference, _ = build_reference_workload(num_npu=2, num_ssu=6, regime="same_trace")
        canonical = tuple(replace(r, arrival_time_ms=round(r.arrival_time_ms, 9))
                          for r in reference)
        expected = retime_arrivals(canonical)
        for noise in (-1e-12, 1e-12):
            # Includes t=0: tiny signed noise must not create extra arrivals
            # or produce a negative-zero field with a different JSON hash.
            perturbed = tuple(replace(r, arrival_time_ms=r.arrival_time_ms + noise)
                              for r in canonical)
            actual = retime_arrivals(perturbed)
            self.assertEqual(continuous_batch_input_fingerprint(actual),
                             continuous_batch_input_fingerprint(expected))
            self.assertEqual(logical_input_fingerprint(actual), logical_input_fingerprint(expected))
            self.assertEqual([r.arrival_time_ms for r in actual],
                             [r.arrival_time_ms for r in expected])

    def test_observed_python_sum_tie_reversal_is_canonicalized(self):
        # Observed Python 3.14 values for these two otherwise tied requests.
        # Without canonicalization the larger rid sorts first, changing the
        # cumulative whole-request-byte clock by milliseconds, not float ulps.
        source, _ = build_reference_workload(num_ssu=6, regime="same_trace")
        local_times = {4000004: 136.99125041335822, 29000004: 136.99125041335822}
        remote_times = {4000004: 136.99125041335822, 29000004: 136.9912504133581}
        local = tuple(replace(r, arrival_time_ms=local_times.get(r.request_id, r.arrival_time_ms))
                      for r in source)
        remote = tuple(replace(r, arrival_time_ms=remote_times.get(r.request_id, r.arrival_time_ms))
                       for r in source)
        self.assertNotEqual(continuous_batch_input_fingerprint(local),
                            continuous_batch_input_fingerprint(remote))
        a, b = retime_arrivals(local), retime_arrivals(remote)
        self.assertEqual(continuous_batch_input_fingerprint(a), continuous_batch_input_fingerprint(b))
        self.assertEqual(logical_input_fingerprint(a), logical_input_fingerprint(b))
        by_id = {r.request_id: r for r in a}
        self.assertLess(by_id[4000004].arrival_time_ms, by_id[29000004].arrival_time_ms)
        self.assertEqual(by_id[4000004].load["reference_arrival_ms"], 136.991250413)

    def test_topologies_change_only_physical_placement(self):
        self.assertEqual(len({m["logical_input_fingerprint"] for _, m in self.examples.values()}), 1)
        self.assertEqual(len({continuous_batch_input_fingerprint(r) for r, _ in self.examples.values()}), 3)
        self.assertEqual(len({m["last_arrival_ms"] for _, m in self.examples.values()}), 1)
        for index in range(704):
            rows = [r[index] for r, _ in self.examples.values()]
            self.assertEqual(len({(r.request_id, r.npu_id, r.arrival_time_ms) for r in rows}), 1)
            self.assertEqual(rows[0].load, rows[1].load)
            self.assertEqual(rows[0].load, rows[2].load)
        self.assertGreater(self.examples[5][1]["arrival_workload"]["aggregate_arrival_load_ratio"], 1)
        self.assertLess(self.examples[6][1]["arrival_workload"]["aggregate_arrival_load_ratio"], 1)

    def test_retime_does_not_change_source_profiles_or_manifest(self):
        old, _ = build_reference_workload(num_npu=2, num_ssu=6, regime="same_trace")
        before = continuous_batch_input_fingerprint(old)
        new = retime_arrivals(old)
        self.assertEqual(continuous_batch_input_fingerprint(old), before)
        self.assertEqual(placement_fingerprint(old), placement_fingerprint(new))
        for a, b in zip(old, new):
            self.assertEqual((a.request_id, a.npu_id, a.placement), (b.request_id, b.npu_id, b.placement))
            for key in ("seq_len_k", "nql", "per_layer_us", "per_layer_kv_gb", "source_ttft_ms"):
                self.assertEqual(a.load[key], b.load[key])

    def test_placement_audit_ignores_assignment_but_detects_ssu_change(self):
        requests, _ = build_workload(num_npu=2, num_ssu=5, small=True)
        moved = (replace(requests[0], npu_id=1),) + requests[1:]
        self.assertEqual(placement_fingerprint(requests), placement_fingerprint(moved))
        row = list(requests[0].placement[0]);ssu, size = row[0];row[0] = ((ssu + 1) % 5, size)
        changed = (replace(requests[0], placement=(tuple(row),)),) + requests[1:]
        self.assertNotEqual(placement_fingerprint(requests), placement_fingerprint(changed))

    def test_seed_changes_order_not_population_or_quota(self):
        a, ma = build_workload(num_npu=2, num_ssu=6, seed=20260906)
        b, mb = build_workload(num_npu=2, num_ssu=6, seed=20260907)
        self.assertEqual({r.request_id for r in a}, {r.request_id for r in b})
        self.assertEqual(ma["profiles"], mb["profiles"])
        self.assertNotEqual(ma["logical_input_fingerprint"], mb["logical_input_fingerprint"])
        self.assertAlmostEqual(mb["arrival_workload"]["actual_postinitial_gib_s"], TARGET_ARRIVAL_GIB_S)

    def test_small_has_actual_positive_arrivals_but_no_window_claim(self):
        requests, metadata = build_workload(num_npu=4, num_ssu=7, small=True)
        self.assertEqual(len(requests), 8)
        self.assertEqual(metadata["initial_arrived_request_count"], 4)
        self.assertEqual(metadata["arrival_workload"]["postinitial_request_count"], 4)
        self.assertTrue(metadata["small_caveat"])
        self.assertTrue(case_name(metadata, "strategy3").endswith("_small"))

    def test_describe_does_not_run_or_import_new_adapter(self):
        with patch("sys.argv", ["runner", "--small", "--num-npu", "2", "--describe-only"]), \
             patch("run_coflow_experiments.run_case") as run, redirect_stdout(io.StringIO()) as output:
            main()
        run.assert_not_called()
        metadata = json.loads(output.getvalue())
        self.assertEqual(metadata["request_count"], 4)
        self.assertEqual(metadata["experiment_id"], "coflow_global_5ms_v1")
        self.assertEqual(metadata["regime"], REGIME)
        self.assertEqual(metadata["arrival_workload"]["reference_arrival_quantum_ms"], 1e-9)
        self.assertEqual(metadata["arrival_workload"]["input_clock_version"], "canonical_reference_ps_v2")

    def test_archive_roundtrip_preserves_retimed_input(self):
        requests, metadata = build_workload(num_npu=2, num_ssu=5, small=True)
        result = {"strategy": "baseline", "metadata": metadata,
                  "input_fingerprint": continuous_batch_input_fingerprint(requests),
                  "logical_input_fingerprint": metadata["logical_input_fingerprint"],
                  "core_and_policy_sha256": {}, "_source_texts": {}}
        with tempfile.TemporaryDirectory() as directory:
            destination = save_result(result, requests, directory)
            saved = json.loads(destination.read_text())
            path = Path(directory) / saved["input_artifact"]["path_relative_to_data"]
            artifact = json.loads(path.read_text())
            restored = tuple(ContinuousBatchRequest(**row) for row in artifact["requests"])
            self.assertEqual(continuous_batch_input_fingerprint(restored), result["input_fingerprint"])
            self.assertEqual(logical_input_fingerprint(restored), metadata["logical_input_fingerprint"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), saved["input_artifact"]["sha256"])

    def test_reference_native_smoke_finishes_and_audits_placement(self):
        requests, metadata = build_workload(num_npu=2, num_ssu=5, small=True)
        result = run_case(requests, metadata, strategy="baseline")
        self.assertEqual(result["slo"]["all_requests"]["admission"]["count"], 4)
        self.assertTrue(result["summary"]["invariants"]["all_requests_completed"])
        self.assertEqual(result["input_placement_fingerprint"], result["execution_placement_fingerprint"])
        self.assertEqual(result["adapter_statistics"]["cir_write_events"], [])
        self.assertFalse(result["common_window"]["all_npus_active_whole_window"])

    def test_sweep_complete_matrix_and_source_archive_set(self):
        self.assertEqual(len(jobs("primary")), 36)
        self.assertEqual(len(jobs("pilot")), 6)
        self.assertEqual(len(jobs("smoke")), 18)
        self.assertEqual(len({job_name(j) for j in jobs("primary")}), 36)
        self.assertEqual({j["strategy"] for j in jobs("pilot")}, set(STRATEGIES))
        self.assertTrue(all(j["num_ssu"] == 6 and j["seed"] == 20260906 for j in jobs("pilot")))
        self.assertIn("run_coflow_experiments.py", source_files())
        self.assertIn("sweep_coflow_experiments.py", source_files())
        self.assertIn("run_shared_path_experiments.py", source_files())
        self.assertIn("sim.py", source_files())


if __name__ == "__main__":
    unittest.main()
