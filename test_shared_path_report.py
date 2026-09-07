"""Independent accounting/audit tests; no policy or simulator run is needed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_shared_path_report import (audit_adapter, audit_assignment, audit_input_artifact, audit_jit,
                                      linear_percentile, manifest_fingerprint, recompute_metrics)


def sample_summary():
    # One layer-like interval per request is repeated eight times to preserve
    # the real execution model. Request 1 arrived early and waited to admit.
    batches, requests = [], []
    for rid, npu, arrival, admission, first_stall in ((0, 0, 0, 0, 4), (1, 1, -20, 0, 0)):
        now, rows = admission, []
        for layer in range(8):
            wait = first_stall if layer == 0 else 0
            start, end = now + wait, now + wait + 1
            rows.append({"layer": layer, "compute_start_ms": start, "compute_end_ms": end,
                         "compute_duration_ms": 1, "io_barrier_wait_ms": wait})
            now = end
        batches.append({"batch_size": 1, "member_request_ids": [rid], "npu_id": npu,
                        "admission_time_ms": admission, "completion_time_ms": now,
                        "layer_metrics": rows})
        requests.append({"request_id": rid, "npu_id": npu, "arrival_time_ms": arrival,
                         "admission_time_ms": admission, "completion_time_ms": now,
                         "own_compute_ms": 8, "io_count": 80})
    return {"num_npu": 2, "num_ssu": 5, "n_layers": 8, "batch_size": 1,
            "request_count": 2, "microbatch_metrics": batches, "request_metrics": requests,
            "makespan_ms": 12, "cir_path_writes": 0, "pressure_reports_by_ssu": [3] * 5}


def sample_adapter():
    return {"collector_times_ms": [0, 5, 10], "fresh_reads_by_ssu": [3] * 5,
            "cache_reads_by_ssu": [7] * 5, "max_snapshot_age_ms": 4.9,
            "cir_write_events": []}


class AccountingTests(unittest.TestCase):
    def test_p99_interpolation_is_explicit(self):
        self.assertEqual(linear_percentile([0, 100], .99), 99)
        self.assertEqual(linear_percentile([3], .99), 3)
        metrics = recompute_metrics(sample_summary())
        self.assertAlmostEqual(metrics["p99_arrival_latency_ms"], 27.84)

    def test_primary_slo_is_admission_not_arrival(self):
        metrics = recompute_metrics(sample_summary(), start_ms=0, end_ms=12)
        self.assertEqual(metrics["admission_slo_passed"], 2)
        self.assertEqual(metrics["arrival_slo_passed"], 1)
        self.assertEqual(metrics["request_count"], 2)

    def test_threshold_equality_passes(self):
        metrics = recompute_metrics(sample_summary(), start_ms=0, end_ms=12)
        first = metrics["records"][0]
        self.assertEqual(first["slo_threshold_ms"], 12)
        self.assertEqual(first["admission_latency_ms"], 12)
        self.assertTrue(first["admission_passed"])

    def test_window_clips_compute_stall_and_idle(self):
        metrics = recompute_metrics(sample_summary(), start_ms=2, end_ms=10)
        self.assertEqual(metrics["compute_ms_by_npu"], [6, 6])
        self.assertEqual(metrics["io_stall_ms_by_npu"], [2, 0])
        self.assertEqual(metrics["idle_ms_by_npu"], [0, 2])
        self.assertEqual(metrics["mean_npu_utilization"], .75)
        self.assertEqual(metrics["active_npu_count"], 1)
        self.assertFalse(metrics["all_npus_active_whole_window"])

    def test_window_never_filters_primary_slo_population(self):
        metrics = recompute_metrics(sample_summary(), start_ms=9, end_ms=10)
        self.assertEqual(metrics["request_count"], 2)
        self.assertEqual(len(metrics["records"]), 2)

    def test_zero_width_window_rejected(self):
        with self.assertRaises(AssertionError):
            recompute_metrics(sample_summary(), start_ms=2, end_ms=2)

    def test_missing_request_not_silently_dropped(self):
        summary = sample_summary()
        summary["request_metrics"].pop()
        with self.assertRaisesRegex(AssertionError, "incomplete request"):
            recompute_metrics(summary)

    def test_duplicate_request_rejected(self):
        summary = sample_summary()
        summary["request_metrics"][1]["request_id"] = 0
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            recompute_metrics(summary)

    def test_negative_compute_interval_rejected(self):
        summary = sample_summary()
        summary["microbatch_metrics"][0]["layer_metrics"][0]["compute_end_ms"] = -1
        with self.assertRaises(AssertionError):
            recompute_metrics(summary)

    def test_nonfinite_data_rejected(self):
        summary = sample_summary()
        summary["microbatch_metrics"][0]["layer_metrics"][0]["compute_start_ms"] = float("nan")
        with self.assertRaises(AssertionError):
            recompute_metrics(summary)

    def test_barrier_accounting_checked(self):
        summary = sample_summary()
        summary["microbatch_metrics"][0]["layer_metrics"][0]["io_barrier_wait_ms"] = 3
        with self.assertRaisesRegex(AssertionError, "barrier"):
            recompute_metrics(summary)

    def test_whole_run_utilization_uses_complete_work(self):
        metrics = recompute_metrics(sample_summary(), start_ms=0, end_ms=12)
        self.assertAlmostEqual(metrics["full_run_mean_npu_utilization"], 16 / 24)

    def test_batch_completion_must_match_last_layer(self):
        summary = sample_summary()
        summary["microbatch_metrics"][0]["completion_time_ms"] = 13
        summary["request_metrics"][0]["completion_time_ms"] = 13
        with self.assertRaisesRegex(AssertionError, "last compute"):
            recompute_metrics(summary)

    def test_makespan_must_match_last_request(self):
        summary = sample_summary()
        summary["makespan_ms"] = 13
        with self.assertRaisesRegex(AssertionError, "makespan"):
            recompute_metrics(summary)


class CollectorAuditTests(unittest.TestCase):
    def test_complete_global_ledger_conservation(self):
        adapter, summary = sample_adapter(), sample_summary()
        summary["completed_blocks"] = 160
        adapter.update(reserved_blocks=160, acknowledged_blocks=160,
                       min_ledger_count=0, ledger_end_counts_by_ssu=[0] * 5)
        self.assertTrue(audit_adapter(adapter, summary)["contract_passed"])
        for field, bad_value in (("reserved_blocks", 161), ("acknowledged_blocks", 159),
                                 ("min_ledger_count", -1), ("ledger_end_counts_by_ssu", [1, 0, 0, 0, 0])):
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, "ledger"):
                audit_adapter({**adapter, field: bad_value}, summary)

    def test_valid_periodic_collector(self):
        result = audit_adapter(sample_adapter(), sample_summary())
        self.assertEqual(result["fresh_read_count"], 15)
        self.assertEqual(result["cached_read_count"], 35)
        self.assertTrue(result["contract_passed"])

    def test_extra_ssu_read_detected(self):
        summary = sample_summary()
        summary["pressure_reports_by_ssu"][0] += 1
        with self.assertRaisesRegex(AssertionError, "outside collector"):
            audit_adapter(sample_adapter(), summary)

    def test_irregular_collection_detected(self):
        adapter = sample_adapter()
        adapter["collector_times_ms"] = [0, 4, 10]
        with self.assertRaisesRegex(AssertionError, "periodic"):
            audit_adapter(adapter, sample_summary())

    def test_stale_snapshot_detected(self):
        adapter = sample_adapter()
        adapter["max_snapshot_age_ms"] = 5.01
        with self.assertRaisesRegex(AssertionError, "snapshot"):
            audit_adapter(adapter, sample_summary())

    def test_future_snapshot_detected(self):
        adapter = sample_adapter()
        adapter["max_snapshot_age_ms"] = -.01
        with self.assertRaisesRegex(AssertionError, "snapshot"):
            audit_adapter(adapter, sample_summary())

    def test_collector_stopped_early_detected(self):
        adapter = sample_adapter()
        adapter["collector_times_ms"] = [0, 5]
        with self.assertRaisesRegex(AssertionError, "stopped early"):
            audit_adapter(adapter, sample_summary())

    def test_static_initialization_is_allowed(self):
        adapter = sample_adapter()
        adapter["cir_write_events"] = [{"time_ms": 0, "ssu_id": 0}]
        self.assertTrue(audit_adapter(adapter, sample_summary())["contract_passed"])

    def test_static_runtime_write_rejected_even_if_slow(self):
        adapter = sample_adapter()
        adapter["cir_write_events"] = [{"time_ms": 0, "ssu_id": 0}, {"time_ms": 100, "ssu_id": 0}]
        with self.assertRaisesRegex(AssertionError, "static"):
            audit_adapter(adapter, sample_summary())

    def test_too_frequent_cir_writes_rejected(self):
        adapter = sample_adapter()
        adapter["cir_write_events"] = [{"time_ms": 0, "ssu_id": 0}, {"time_ms": 1, "ssu_id": 0}]
        with self.assertRaisesRegex(AssertionError, "frequent"):
            audit_adapter(adapter, sample_summary())


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shared-report-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.metrics = recompute_metrics(sample_summary())
        self.requests = [{"request_id": r["request_id"], "npu_id": r["npu_id"],
                          "arrival_time_ms": r["arrival_ms"],
                          "load": {"category": "A", "per_layer_us": 1000.0},
                          "placement": [[[i % 5, 176 * 1024 / 2**30] for i in range(10)]]}
                         for r in self.metrics["records"]]
        source_data = b"fixture data\n"
        (self.folder / "data.txt").write_bytes(source_data)
        self.saved = {"strategy": "baseline", "source_artifacts": {"data": "data.txt"},
                      "core_and_policy_sha256": {"data": hashlib.sha256(source_data).hexdigest()},
                      "metadata": {"source": {"source_sha256": hashlib.sha256(source_data).hexdigest()}}}
        self.write_manifest()

    def write_manifest(self):
        fingerprint = manifest_fingerprint(self.requests)
        logical = [{k: r[k] for k in ("request_id", "npu_id", "arrival_time_ms", "load")}
                   for r in self.requests]
        logical_hash = hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        raw = json.dumps({"input_fingerprint": fingerprint, "logical_input_fingerprint": logical_hash,
                          "requests": self.requests}).encode()
        (self.folder / "input.json").write_bytes(raw)
        self.saved.update(input_fingerprint=fingerprint, logical_input_fingerprint=logical_hash,
                          input_artifact={"path_relative_to_data": "input.json",
                                          "sha256": hashlib.sha256(raw).hexdigest()})

    def test_manifest_and_source_archive_round_trip(self):
        result = audit_input_artifact(self.saved, self.folder, self.metrics)
        self.assertTrue(result["input_manifest_fingerprint_recomputed"])
        self.assertTrue(result["archived_sources_verified"])

    def test_corrupted_manifest_bytes_detected(self):
        (self.folder / "input.json").write_bytes(b"{}")
        with self.assertRaisesRegex(AssertionError, "artifact SHA"):
            audit_input_artifact(self.saved, self.folder, self.metrics)

    def test_placement_is_part_of_fingerprint(self):
        altered = deepcopy(self.requests)
        altered[0]["placement"][0][0][0] = 4
        self.assertNotEqual(manifest_fingerprint(self.requests), manifest_fingerprint(altered))

    def test_request_order_not_part_of_native_fingerprint(self):
        self.assertEqual(manifest_fingerprint(self.requests), manifest_fingerprint(list(reversed(self.requests))))

    def test_updated_hash_cannot_hide_changed_input_compute(self):
        self.requests[0]["load"]["per_layer_us"] = 1001
        self.write_manifest()
        with self.assertRaisesRegex(AssertionError, "input compute"):
            audit_input_artifact(self.saved, self.folder, self.metrics)

    def test_fixed_npu_binding_cannot_change(self):
        self.requests[0]["npu_id"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(AssertionError, "binding"):
            audit_input_artifact(self.saved, self.folder, self.metrics)
        self.saved["strategy"] = "strategy1"
        self.assertTrue(audit_input_artifact(self.saved, self.folder, self.metrics)["archived_sources_verified"])

    def test_non_176kib_block_rejected(self):
        self.requests[0]["placement"][0][0][1] *= 2
        self.write_manifest()
        with self.assertRaisesRegex(AssertionError, "176 KiB"):
            audit_input_artifact(self.saved, self.folder, self.metrics)

    def test_changed_source_archive_rejected(self):
        (self.folder / "data.txt").write_bytes(b"different")
        with self.assertRaisesRegex(AssertionError, "archived source"):
            audit_input_artifact(self.saved, self.folder, self.metrics)

    def test_source_archive_closure_required(self):
        self.saved["core_and_policy_sha256"]["missing.py"] = "not-archived"
        with self.assertRaisesRegex(AssertionError, "closure"):
            audit_input_artifact(self.saved, self.folder, self.metrics)


class AssignmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.metrics = recompute_metrics(sample_summary())
        self.saved = {"strategy": "strategy1", "assignment_log": [
            {"request_id": r["request_id"], "arrival_time_ms": r["arrival_ms"],
             "assigned_npu_id": r["npu_id"], "snapshot_time_ms": r["arrival_ms"]}
            for r in self.metrics["records"]]}

    def test_causal_arrival_assignments(self):
        self.assertEqual(audit_assignment(self.saved, self.metrics)["assignment_decisions_verified"], 2)

    def test_no_assignments_for_fixed_policy(self):
        self.saved["strategy"] = "once"
        with self.assertRaisesRegex(AssertionError, "fixed-NPU"):
            audit_assignment(self.saved, self.metrics)

    def test_future_snapshot_is_not_causal(self):
        self.saved["assignment_log"][0]["snapshot_time_ms"] += .1
        with self.assertRaisesRegex(AssertionError, "future/stale"):
            audit_assignment(self.saved, self.metrics)

    def test_binding_cannot_change_after_arrival(self):
        self.saved["assignment_log"][0]["assigned_npu_id"] = 1
        with self.assertRaisesRegex(AssertionError, "binding changed"):
            audit_assignment(self.saved, self.metrics)

    def test_missing_arrival_decision_is_detected(self):
        self.saved["assignment_log"].pop()
        with self.assertRaisesRegex(AssertionError, "population"):
            audit_assignment(self.saved, self.metrics)


class JITAuditTests(unittest.TestCase):
    def setUp(self):
        estimate = 1000 * (176 * 1024 / 2**30) * 10 / 50
        release = 40 - estimate - 10
        self.statistics = {"jit_prefetch": {"enabled": True, "guard_ms": 10,
            "activation_count": 16, "delay_count": 1,
            "total_delay_ms": release, "max_delay_ms": release, "decision_wall_us": 3,
            "modifies_already_queued_io": False,
            "release_examples": [{"activation_ms": 0, "deadline_ms": 40,
                "release_ms": release, "delay_ms": release, "snapshot_time_ms": 0,
                "layer_io_by_ssu": [2] * 5, "cached_ssu_queue_io": [0] * 5}]}}

    def test_recomputes_logged_jit_formula(self):
        self.assertEqual(audit_jit(self.statistics, sample_summary(), "strategy1")["jit_delayed_layer_count"], 1)

    def test_wrong_release_is_detected(self):
        self.statistics["jit_prefetch"]["release_examples"][0]["release_ms"] += .1
        with self.assertRaisesRegex(AssertionError, "JIT release"):
            audit_jit(self.statistics, sample_summary(), "strategy1")

    def test_wrong_frozen_guard_is_detected(self):
        self.statistics["jit_prefetch"]["guard_ms"] = 5
        with self.assertRaisesRegex(AssertionError, "guard"):
            audit_jit(self.statistics, sample_summary(), "strategy1")

    def test_future_jit_snapshot_is_detected(self):
        self.statistics["jit_prefetch"]["release_examples"][0]["snapshot_time_ms"] = .01
        with self.assertRaisesRegex(AssertionError, "future"):
            audit_jit(self.statistics, sample_summary(), "strategy2")

    def test_missing_jit_rejected_for_final_policy(self):
        with self.assertRaisesRegex(AssertionError, "must include"):
            audit_jit({}, sample_summary(), "strategy1")

    def test_old_reference_without_jit_stats_is_allowed(self):
        self.assertEqual(audit_jit({}, sample_summary(), "once")["jit_activation_count"], 0)
        self.assertEqual(set(audit_jit({}, sample_summary(), "once")),
                         set(audit_jit(self.statistics, sample_summary(), "strategy1")))

    def test_baseline_cannot_enable_jit(self):
        with self.assertRaisesRegex(AssertionError, "fixed-emission"):
            audit_jit(self.statistics, sample_summary(), "baseline")


if __name__ == "__main__":
    unittest.main()
