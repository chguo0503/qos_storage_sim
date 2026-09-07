"""Pure arithmetic/closed-loop tests; these do not run the project simulator."""

import copy
import unittest

from npu_stall_predictor import (
    IO_BYTES, NPUState, Request, fifo_burst_certificate, forecast_path0,
    impact_order_sensitivity, io_service_ms, predict_request_impact,
    required_rate_gib_s, screen_current_layer,
)


# Teaching units: a full SSD command takes 1 ms and its receive-link tail 0.1 ms.
TOY_BANDWIDTH = IO_BYTES / 2**30 * 1000
TOY_ENV = dict(bandwidth_gib_s=TOY_BANDWIDTH,
               link_gib_s=10 * TOY_BANDWIDTH, issue_interval_us=0)


def lanes(*states):
    return list(states) + [NPUState()] * (4 - len(states))


class CurrentLayerArithmeticTests(unittest.TestCase):
    def test_glm_units_and_exact_empty_queue(self):
        self.assertEqual(IO_BYTES, 176 * 1024)
        self.assertEqual(IO_BYTES, 128 * 1408)
        self.assertEqual(io_service_ms(), 0.0041961669921875)
        expected = 10 * io_service_ms() + io_service_ms(50)
        result = screen_current_layer(10, expected, 0, atomic_enqueue=True)
        self.assertEqual(result["ready_lower_ms"], expected)
        self.assertEqual(result["ready_upper_ms"], expected)
        self.assertEqual(result["verdict"], "meets_deadline")

    def test_unknown_active_is_one_io_of_uncertainty(self):
        result = screen_current_layer(2, 100, 3, atomic_enqueue=True)
        service, tail = io_service_ms(), io_service_ms(50)
        self.assertAlmostEqual(result["ready_lower_ms"], 4 * service + tail)
        self.assertAlmostEqual(result["ready_upper_ms"], 5 * service + tail)
        self.assertAlmostEqual(result["ready_upper_ms"] - result["ready_lower_ms"], service)

    def test_known_active_remainder_counted_once(self):
        result = screen_current_layer(2, 100, 3, atomic_enqueue=True,
                                      active_remaining_bytes=IO_BYTES // 2)
        expected = 4.5 * io_service_ms() + io_service_ms(50)
        self.assertAlmostEqual(result["ready_lower_ms"], expected)
        self.assertEqual(result["ready_lower_ms"], result["ready_upper_ms"])

    def test_uncontrolled_interleaving_does_not_claim_finite_upper(self):
        result = screen_current_layer(1, 100, 0)
        self.assertEqual(result["verdict"], "unknown")
        self.assertIsNone(result["ready_upper_ms"])
        self.assertIsNone(result["stall_upper_ms"])

    def test_receive_tail_can_make_ssd_timely_but_npu_late(self):
        result = screen_current_layer(1, io_service_ms(), 0, atomic_enqueue=True)
        self.assertEqual(result["verdict"], "must_stall")
        self.assertAlmostEqual(result["stall_lower_ms"], io_service_ms(50))

    def test_zero_read_never_waits_for_old_ssd_work(self):
        result = screen_current_layer(0, 0, 100)
        self.assertEqual(result["ready_lower_ms"], 0)
        self.assertEqual(result["ready_upper_ms"], 0)
        self.assertEqual(result["verdict"], "meets_deadline")

    def test_burst_certificate_is_sufficient_not_necessary(self):
        yes = fifo_burst_certificate(10, 0, 1, 2,
                                    bandwidth_gib_s=TOY_BANDWIDTH,
                                    link_gib_s=10 * TOY_BANDWIDTH)
        self.assertTrue(yes["phase_independent_stall_certified"])
        self.assertAlmostEqual(yes["committed_work_lower_ms"], 10)
        self.assertAlmostEqual(yes["certified_excess_lower_ms"], 7.1)
        no = fifo_burst_certificate(10, 10, 1, 2,
                                   bandwidth_gib_s=TOY_BANDWIDTH,
                                   link_gib_s=10 * TOY_BANDWIDTH)
        self.assertFalse(no["phase_independent_stall_certified"])

    def test_zero_read_victim_cannot_receive_a_stall_certificate(self):
        result = fifo_burst_certificate(10000, 0, 0, 0.001)
        self.assertFalse(result["phase_independent_stall_certified"])
        self.assertEqual(result["certified_excess_lower_ms"], 0)

    def test_certificate_excess_is_cumulative_not_one_layer_stall(self):
        states = lanes(
            NPUState(Request(0, 1, 10, layers=3), next_layer=1,
                     compute_end_in_ms=10, queued_io=1, unissued_io=0),
            NPUState(Request(1, 12, 100, layers=1), queued_io=12, unissued_io=0),
            NPUState(Request(2, 13, 100, layers=1), queued_io=13, unissued_io=0),
        )
        result = forecast_path0(states, queue_layout="grouped", queue_order=(1, 0, 2, 3), **TOY_ENV)
        waits = [r["stall_ms"] for r in result["layers"] if r["npu_id"] == 0]
        certificate = fifo_burst_certificate(25, 0, 1, 10,
                                            bandwidth_gib_s=TOY_BANDWIDTH,
                                            link_gib_s=10 * TOY_BANDWIDTH)
        self.assertAlmostEqual(waits[0], 3.1)
        self.assertAlmostEqual(waits[1], 4.0)
        self.assertLess(max(waits), certificate["certified_excess_lower_ms"])
        self.assertGreaterEqual(sum(waits), certificate["certified_excess_lower_ms"])
        self.assertIn("cumulative", certificate["excess_scope"])

    def test_rate_is_work_divided_by_usable_budget(self):
        self.assertAlmostEqual(required_rate_gib_s(10, 2, ahead_io=2, latency_ms=0.5),
                               12 * IO_BYTES / 2**30 / 0.0015)
        self.assertEqual(required_rate_gib_s(0, 0), 0)
        self.assertEqual(required_rate_gib_s(1, 1, latency_ms=1), float("inf"))


class ConditionalRolloutTests(unittest.TestCase):
    def test_completed_initial_io_does_not_finish_unissued_remainder(self):
        state = NPUState(Request(1, 2, 3, layers=2), next_layer=1,
                         compute_end_in_ms=2, queued_io=1, unissued_io=1,
                         next_issue_in_ms=5)
        row = forecast_path0(lanes(state), **TOY_ENV)["layers"][0]
        # The old command completes at 1.1, but the last command is issued at 5.
        self.assertAlmostEqual(row["io_ready_ms"], 6.1)
        self.assertAlmostEqual(row["compute_start_ms"], 6.1)
        self.assertAlmostEqual(row["future_stall_ms"], 4.1)

    def test_known_active_remaining_part_reduces_initial_service(self):
        state = NPUState(Request(1, 2, 3, layers=1), queued_io=2, unissued_io=0)
        result = forecast_path0(lanes(state), active_npu_id=0,
                                active_remaining_bytes=IO_BYTES // 2, **TOY_ENV)
        self.assertAlmostEqual(result["layers"][0]["io_ready_ms"], 1.6)

    def test_zero_path_count_can_still_have_receive_link_tail(self):
        state = NPUState(Request(1, 1, 3, layers=2), next_layer=1,
                         compute_end_in_ms=0.1, unissued_io=0, ready_in_ms=0.3)
        row = forecast_path0(lanes(state), **TOY_ENV)["layers"][0]
        self.assertAlmostEqual(row["compute_start_ms"], 0.3)
        self.assertAlmostEqual(row["future_stall_ms"], 0.2)

    def test_initial_stall_history_is_separate_from_future_wait(self):
        state = NPUState(Request(1, 1, 3, layers=2), next_layer=1,
                         compute_end_in_ms=-3, queued_io=1, unissued_io=0)
        result = forecast_path0(lanes(state), **TOY_ENV)
        row = result["layers"][0]
        self.assertAlmostEqual(row["stall_ms"], 4.1)
        self.assertAlmostEqual(row["future_stall_ms"], 1.1)
        self.assertAlmostEqual(result["future_stall_ms_by_npu"][0], 1.1)

    def test_same_owner_counts_can_have_different_layer_ready_times(self):
        states = lanes(NPUState(Request(1, 2, 3, layers=1), queued_io=2, unissued_io=0),
                       NPUState(Request(2, 2, 3, layers=1), queued_io=2, unissued_io=0))
        round_robin = forecast_path0(states, queue_layout="round_robin", **TOY_ENV)
        grouped = forecast_path0(states, queue_layout="grouped", **TOY_ENV)
        rr_ready = {r["npu_id"]: r["io_ready_ms"] for r in round_robin["layers"]}
        grouped_ready = {r["npu_id"]: r["io_ready_ms"] for r in grouped["layers"]}
        self.assertAlmostEqual(rr_ready[0], 3.1)
        self.assertAlmostEqual(grouped_ready[0], 2.1)
        self.assertFalse(grouped["guaranteed"])

    def test_late_candidate_does_not_retroactively_cross_prefetch(self):
        state = NPUState(Request(1, 1, 10, layers=1), next_layer=1,
                         compute_end_in_ms=5)
        result = predict_request_impact(lanes(state), Request(2, 1, 3, layers=1),
                                        0, **TOY_ENV)
        row = result["candidate_layers"][0]
        self.assertAlmostEqual(row["deadline_ms"], 5)
        self.assertAlmostEqual(row["io_ready_ms"], 6.1)
        self.assertAlmostEqual(row["future_stall_ms"], 1.1)
        self.assertEqual(result["existing_request_completion_delay_ms"], {1: 0})

    def test_known_next_request_prefetches_at_final_compute_start(self):
        state = NPUState(Request(1, 1, 10, layers=1), unissued_io=0,
                         waiting_requests=(Request(2, 1, 3, layers=1),))
        enabled = forecast_path0(lanes(state), **TOY_ENV)
        disabled = forecast_path0(lanes(state), cross_request_prefetch=False, **TOY_ENV)
        enabled_next = next(r for r in enabled["layers"] if r["request_id"] == 2)
        disabled_next = next(r for r in disabled["layers"] if r["request_id"] == 2)
        self.assertAlmostEqual(enabled_next["compute_start_ms"], 10)
        self.assertEqual(enabled_next["future_stall_ms"], 0)
        self.assertAlmostEqual(disabled_next["compute_start_ms"], 11.1)

    def test_closed_loop_can_have_quiet_layers_then_stall_again(self):
        states = lanes(
            NPUState(Request(1, 10, 20), next_layer=1, compute_end_in_ms=20,
                     unissued_io=10),
            NPUState(Request(2, 1, 2), next_layer=1, compute_end_in_ms=2,
                     unissued_io=1),
        )
        result = forecast_path0(states, **TOY_ENV)
        short = {r["layer"]: r for r in result["layers"] if r["npu_id"] == 1}
        self.assertAlmostEqual(short[2]["future_stall_ms"], 8)
        self.assertEqual(short[3]["future_stall_ms"], 0)
        self.assertEqual(short[6]["future_stall_ms"], 0)
        self.assertAlmostEqual(short[7]["future_stall_ms"], 9)
        self.assertEqual(len(result["layers"]), 14)

    def test_zero_ssd_read_layers_compute_without_artificial_tail(self):
        result = forecast_path0(lanes(NPUState(Request(1, 0, 2, layers=3))), **TOY_ENV)
        self.assertEqual([r["compute_start_ms"] for r in result["layers"]], [0, 2, 4])
        self.assertEqual(result["future_stall_ms_by_npu"], [0, 0, 0, 0])


class PairedSnapshotTests(unittest.TestCase):
    def test_paired_forecast_preserves_input_and_identical_without_case(self):
        states = lanes(NPUState(Request(1, 2, 3, layers=3), next_layer=1,
                                compute_end_in_ms=0.5, unissued_io=2))
        preserved = copy.deepcopy(states)
        original = forecast_path0(states, **TOY_ENV)
        impact = predict_request_impact(states, Request(99, 2, 4, layers=2), 1, **TOY_ENV)
        self.assertEqual(states, preserved)
        self.assertEqual(impact["without_candidate"], original)
        self.assertEqual({r["request_id"] for r in impact["candidate_layers"]}, {99})
        self.assertEqual({(r["request_id"], r["layer"]) for r in impact["existing_layer_deltas"]},
                         {(1, 1), (1, 2)})

    def test_new_candidate_stays_behind_already_waiting_request(self):
        state = NPUState(waiting_requests=(Request(1, 1, 3, layers=1),))
        result = predict_request_impact(lanes(state), Request(2, 1, 3, layers=1), 0, **TOY_ENV)
        self.assertEqual([r["request_id"] for r in result["with_candidate"]["layers"]], [1, 2])
        self.assertEqual(result["existing_request_completion_delay_ms"], {1: 0})

    def test_idle_lane_does_not_gift_old_ready_flags_to_candidate(self):
        result = predict_request_impact(lanes(NPUState(unissued_io=0)),
                                        Request(2, 1, 3, layers=1), 0, **TOY_ENV)
        self.assertAlmostEqual(result["candidate_layers"][0]["io_ready_ms"], 1.1)

    def test_final_compute_lane_does_not_gift_ready_flags_to_late_candidate(self):
        state = NPUState(Request(1, 1, 10, layers=1), next_layer=1,
                         compute_end_in_ms=5, unissued_io=0)
        result = predict_request_impact(lanes(state), Request(2, 1, 3, layers=1),
                                        0, **TOY_ENV)
        self.assertAlmostEqual(result["candidate_layers"][0]["io_ready_ms"], 6.1)

    def test_candidate_can_hide_its_later_io_but_delay_existing_request(self):
        state = NPUState(Request(1, 1, 2, layers=3), next_layer=1,
                         compute_end_in_ms=2, unissued_io=0)
        result = predict_request_impact(lanes(state), Request(2, 8, 20, layers=2), 1, **TOY_ENV)
        later = [r for r in result["candidate_layers"] if not r["is_layer0"]]
        self.assertEqual(sum(r["future_stall_ms"] for r in later), 0)
        self.assertGreater(result["existing_request_completion_delay_ms"][1], 0)

    def test_order_sensitivity_explicitly_does_not_claim_bounds(self):
        result = impact_order_sensitivity(lanes(), Request(1, 1, 3, layers=1), 0, **TOY_ENV)
        self.assertEqual(len(result["samples"]), 24)
        self.assertFalse(result["range_is_guaranteed"])


if __name__ == "__main__":
    unittest.main()
