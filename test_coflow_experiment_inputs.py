"""Shared run wrapper and placement audit used by both retained studies.

The historical module name is retained because the production runner still
uses run_coflow_experiments.run_case for Baseline and Once. The small workload
is only a fixture; this file no longer tests the old coflow arrival experiment.
"""

from dataclasses import replace
import unittest

from run_coflow_experiments import build_workload, placement_fingerprint, run_case


class RetainedExperimentAdapterTests(unittest.TestCase):
    def test_placement_audit_ignores_assignment_but_detects_ssu_change(self):
        requests, _ = build_workload(num_npu=2, num_ssu=5, small=True)
        moved = (replace(requests[0], npu_id=1),) + requests[1:]
        self.assertEqual(placement_fingerprint(requests), placement_fingerprint(moved))
        row = list(requests[0].placement[0]);ssu, size = row[0];row[0] = ((ssu + 1) % 5, size)
        changed = (replace(requests[0], placement=(tuple(row),)),) + requests[1:]
        self.assertNotEqual(placement_fingerprint(requests), placement_fingerprint(changed))

    def test_reference_native_smoke_finishes_and_audits_placement(self):
        requests, metadata = build_workload(num_npu=2, num_ssu=5, small=True)
        result = run_case(requests, metadata, strategy="baseline")
        self.assertEqual(result["slo"]["all_requests"]["admission"]["count"], 4)
        self.assertTrue(result["summary"]["invariants"]["all_requests_completed"])
        self.assertEqual(result["input_placement_fingerprint"], result["execution_placement_fingerprint"])
        self.assertEqual(result["adapter_statistics"]["cir_write_events"], [])
        self.assertFalse(result["common_window"]["all_npus_active_whole_window"])


if __name__ == "__main__":
    unittest.main()
