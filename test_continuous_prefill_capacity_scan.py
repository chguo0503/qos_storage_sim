import json
import unittest

from continuous_prefill_capacity_scan import _critical_path_metrics, _scan_spec


class ContinuousPrefillCapacityScanTest(unittest.TestCase):
    def test_compute_only_bound_uses_external_arrival_not_admission(self):
        summary = {
            "makespan_ms": 25.0,
            "request_metrics": [
                {
                    "npu_id": 0,
                    "arrival_time_ms": 0.0,
                    "admission_wait_ms": 100.0,
                    "own_compute_ms": 10.0,
                },
                {
                    "npu_id": 0,
                    "arrival_time_ms": 5.0,
                    "admission_wait_ms": 100.0,
                    "own_compute_ms": 10.0,
                },
            ],
        }
        metrics = _critical_path_metrics(summary)
        self.assertEqual(metrics["compute_only_lower_bound_ms"], 20.0)
        self.assertEqual(metrics["makespan_gap_above_compute_bound_ms"], 5.0)

    def test_scan_spec_survives_json_round_trip(self):
        spec = _scan_spec()
        self.assertEqual(json.loads(json.dumps(spec)), spec)


if __name__ == "__main__":
    unittest.main()
