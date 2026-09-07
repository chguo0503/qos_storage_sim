"""Math and state-transition tests; full native comparisons have a separate CLI."""
from dataclasses import replace
import unittest

from multi_ssu_stall_predictor import (
    IO_BYTES, LegacyMT19937, NPUState, Request, forecast_baseline,
    fifo_burst_certificate, impact_order_sensitivity,
    io_service_ms, predict_request_impact, screen_current_layer, screen_pending_layer,
)


def request(rid, blocks, compute_ms, disks=2, layers=8):
    return Request(rid, (tuple(i % disks for i in range(blocks)),), compute_ms, layers)


class BoundTests(unittest.TestCase):
    def test_full_io_unit(self):
        self.assertEqual(IO_BYTES, 180224)
        self.assertAlmostEqual(io_service_ms(40), 0.0041961669921875)

    def test_empty_layer(self):
        result = screen_current_layer([0, 0], 0, [100, 100], link_pending_io=200)
        self.assertEqual(result["ready_lower_ms"], 0)
        self.assertEqual(result["verdict"], "unknown")

    def test_missing_ssu_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            screen_current_layer([1, 100], 1, [0])

    def test_unknown_active_is_optimistic(self):
        unknown = screen_current_layer([1], 10, [4])
        known = screen_current_layer([1], 10, [4], disk_active_remaining_bytes=[IO_BYTES / 2])
        self.assertAlmostEqual(known["ready_lower_ms"] - unknown["ready_lower_ms"], io_service_ms(40) / 2)

    def test_receive_link_not_sum_of_ssd_bandwidths(self):
        result = screen_current_layer([256, 256, 256, 256, 256, 255], 4.533469, [0] * 6)
        self.assertEqual(result["verdict"], "must_stall")
        self.assertAlmostEqual(result["ready_lower_ms"], 5.1570892333984375)
        self.assertIsNone(result["ready_upper_ms"])

    def test_link_backlog_count_includes_active(self):
        result = screen_current_layer([1, 1], 0, [0, 0], link_pending_io=10,
                                      link_active_remaining_bytes=IO_BYTES / 2)
        self.assertAlmostEqual(result["ready_lower_ms"], 11.5 * io_service_ms(50))

    def test_burst_certificate_requires_shared_ssu(self):
        shared = fifo_burst_certificate([1534, 0], 0.1533, [7, 0], 0.582149491632)
        separate = fifo_burst_certificate([1534, 0], 0.1533, [0, 7], 0.582149491632)
        self.assertTrue(shared["phase_independent_stall_certified"])
        self.assertFalse(separate["phase_independent_stall_certified"])
        self.assertIn("cumulative", shared["scope"])

    def test_pending_unknown_active_can_be_nearly_done(self):
        result = screen_pending_layer([1, 0], [0, 0], 0)
        self.assertAlmostEqual(result["ready_lower_ms"], io_service_ms(50))

    def test_pending_link_and_unissued_work_not_double_counted(self):
        result = screen_pending_layer([0, 0], [1, 1], 0, link_pending_io=10,
                                       link_active_remaining_bytes=IO_BYTES / 2)
        self.assertAlmostEqual(result["ready_lower_ms"], 11.5 * io_service_ms(50))

    def test_pending_empty_layer(self):
        self.assertEqual(screen_pending_layer([0, 0], [0, 0], 0)["ready_lower_ms"], 0)

    def test_already_ready_is_not_missed_only_because_deadline_is_past(self):
        result = screen_pending_layer([0, 0], [0, 0], -2)
        self.assertEqual(result["verdict"], "already_ready")
        self.assertEqual(result["stall_lower_ms"], 0)


class ForecastTests(unittest.TestCase):
    def test_single_ssd_reduces_to_serial_formula(self):
        result = forecast_baseline([NPUState(request(0, 10, 0.1, disks=1, layers=2))], num_ssu=1)
        first, second = result["layers"]
        self.assertAlmostEqual(first["io_ready_ms"], 10 * io_service_ms(40) + io_service_ms(50))
        self.assertAlmostEqual(second["future_stall_ms"], 0)

    def test_shared_receive_link_bottleneck(self):
        for count in (6, 7):
            result = forecast_baseline([NPUState(request(0, 1535, 4.533469, disks=count))]
                                       + [NPUState()] * 31, num_ssu=count, issue_order_seed=42)
            self.assertEqual(len(result["layers"]), 8)
            self.assertAlmostEqual(result["layers"][0]["io_ready_ms"], 5.1570892333984375)
            for row in result["layers"][1:]:
                self.assertAlmostEqual(row["stall_ms"], 0.6236202333984373)

    def test_empty_request_compute(self):
        result = forecast_baseline([NPUState(request(0, 0, 1, layers=3))], num_ssu=2)
        self.assertEqual(result["request_completion_ms"][0], 3)
        self.assertTrue(all(r["stall_ms"] == 0 for r in result["layers"]))

    def test_placement_changes_by_layer(self):
        job = Request(1, ((0,), (1, 1, 1)), 0.01, 2)
        result = forecast_baseline([NPUState(job)], num_ssu=2)
        self.assertAlmostEqual(result["layers"][1]["io_ready_ms"] - result["layers"][0]["compute_start_ms"],
                               3 * io_service_ms(40) + io_service_ms(50))

    def test_warm_link_only_residual(self):
        job = request(1, 8, 1, layers=1)
        state = NPUState(job, unissued_by_ssu=(0, 0), queued_io_by_ssu=(0, 0),
                         link_pending_io=3, link_active_remaining_bytes=IO_BYTES / 2)
        result = forecast_baseline([state], num_ssu=2)
        self.assertAlmostEqual(result["layers"][0]["io_ready_ms"], 2.5 * io_service_ms(50))

    def test_partial_queued_unissued_not_premature_ready(self):
        state = NPUState(request(0, 8, 0.2, layers=1), queued_io_by_ssu=(2, 1),
                         unissued_by_ssu=(2, 2), link_pending_io=1,
                         next_issue_in_ms=0.1, emitter_ssu_order=(1, 0))
        result = forecast_baseline([state], num_ssu=2)
        self.assertGreater(result["layers"][0]["io_ready_ms"], 0.1)

    def test_cross_request_prefetch_on_and_off(self):
        first, second = request(1, 4, 0.1, layers=1), request(2, 4, 0.1, layers=1)
        states = [NPUState(first, waiting_requests=(second,))]
        on = forecast_baseline(states, num_ssu=2)
        off = forecast_baseline(states, num_ssu=2, cross_request_prefetch=False)
        self.assertGreater(off["request_completion_ms"][2], on["request_completion_ms"][2])
        self.assertEqual(on["layers"][1]["stall_ms"], 0)

    def test_late_candidate_no_retroactive_prefetch(self):
        current = request(1, 4, 1, layers=1)
        state = NPUState(current, next_layer=1, compute_end_in_ms=0.5,
                         unissued_by_ssu=(0, 0))
        result = predict_request_impact([state], request(2, 4, 1, layers=1), 0, num_ssu=2)
        self.assertEqual(result["candidate_layers"][0]["io_release_ms"], 0.5)
        self.assertGreater(result["candidate_layers"][0]["compute_start_ms"], 0.5)

    def test_candidate_preserves_waiting_order(self):
        waiting = request(1, 2, 0.1, layers=1)
        candidate = request(2, 2, 0.1, layers=1)
        states = [NPUState(waiting_requests=(waiting,))]
        result = predict_request_impact(states, candidate, 0, num_ssu=2)
        self.assertEqual([r["request_id"] for r in result["with_candidate"]["layers"]], [1, 2])
        self.assertEqual(states[0].waiting_requests, (waiting,))

    def test_negative_deadline_counts_only_future_stall(self):
        state = NPUState(request(1, 2, 0.1, layers=1), compute_end_in_ms=-2,
                         unissued_by_ssu=(0, 0), queued_io_by_ssu=(1, 1))
        row = forecast_baseline([state], num_ssu=2)["layers"][0]
        self.assertAlmostEqual(row["stall_ms"] - row["future_stall_ms"], 2)

    def test_unknown_fifo_order_is_not_guaranteed(self):
        states = [NPUState(request(i, 4, 0.1, disks=1, layers=1),
                           queued_io_by_ssu=(4,), unissued_by_ssu=(0,)) for i in range(2)]
        a = forecast_baseline(states, num_ssu=1, queue_layout="grouped")
        b = forecast_baseline(states, num_ssu=1, queue_layout="grouped", queue_order=(1, 0))
        ready_a = {r["npu_id"]: r["io_ready_ms"] for r in a["layers"]}
        ready_b = {r["npu_id"]: r["io_ready_ms"] for r in b["layers"]}
        self.assertNotEqual(ready_a[0], ready_b[0])
        self.assertFalse(a["guaranteed"])

    def test_legacy_rng_known_sequence(self):
        generator = LegacyMT19937(42)
        self.assertEqual([generator.uint32() for _ in range(5)],
                         [1608637542, 3421126067, 4083286876, 787846414, 3143890026])

    def test_32_lane_sensitivity_is_four_scenarios_not_factorial(self):
        result = impact_order_sensitivity([NPUState()] * 32, request(1, 2, 0.1, layers=1),
                                          0, num_ssu=7)
        self.assertEqual(len(result["samples"]), 4)
        self.assertFalse(result["range_is_guaranteed"])


if __name__ == "__main__":
    unittest.main()
