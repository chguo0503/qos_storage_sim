"""Independent checks for the tutorial's small FIFO example."""

import json
import math
import unittest

from toy_staggering_model import clipped_duration, get_story_data, simulate


class ToyStaggeringModelTests(unittest.TestCase):
    def setUp(self):
        self.story = get_story_data()

    def test_story_is_json_serializable(self):
        self.assertEqual(json.loads(json.dumps(self.story)), self.story)

    def test_success_cold_start_and_natural_staggering(self):
        result = self.story["success"]
        orders = {(row["name"], row["dish"]): row for row in result["orders"]}
        self.assertEqual(orders["A", 0]["ready_s"], 1)
        self.assertEqual(orders["B", 0]["ready_s"], 2)
        self.assertEqual(orders["A", 1]["release_s"], 1)
        self.assertEqual(orders["A", 1]["first_service_s"], 2)
        self.assertEqual(orders["A", 1]["ready_s"], 3)
        self.assertEqual(orders["A", 1]["deadline_s"], 5)
        self.assertEqual(orders["B", 1]["release_s"], 2)
        self.assertEqual(orders["B", 1]["ready_s"], 4)
        self.assertEqual(orders["B", 1]["deadline_s"], 6)
        self.assertFalse(result["stalls"])

    def test_failure_is_already_offset_but_misses_deadline(self):
        result = self.story["failure"]
        by_release = {(row["name"], row["release_s"]): row for row in result["orders"]}
        b_order = by_release["B", 27]
        a_order = by_release["A", 28]
        self.assertEqual(b_order["first_service_s"], 27)
        self.assertEqual(b_order["ready_s"], 33)
        self.assertEqual(a_order["first_service_s"], 33)
        self.assertEqual(a_order["ready_s"], 34)
        self.assertEqual(a_order["deadline_s"], 32)
        self.assertEqual(a_order["stall_s"], 2)
        self.assertEqual(a_order["read_latency_s"], 6)

    def test_failure_repeats_beyond_first_plotted_cycle(self):
        result = self.story["failure"]
        for start_s in (27, 37, 47, 57):
            a_orders = [row for row in result["orders"]
                        if row["name"] == "A" and start_s <= row["release_s"] < start_s + 10]
            self.assertEqual([row["release_s"] - start_s for row in a_orders], [1, 7])
            self.assertEqual([row["stall_s"] for row in a_orders], [2, 0])

    def test_every_io_is_one_second_and_fifo_never_overlaps(self):
        for key in ("success", "failure"):
            services = self.story[key]["io_services"]
            for row in services:
                self.assertEqual(row["end_s"] - row["start_s"], 1)
                self.assertGreaterEqual(row["start_s"], row["release_s"])
            for previous, following in zip(services, services[1:]):
                self.assertGreaterEqual(following["start_s"], previous["end_s"])
                self.assertGreaterEqual(following["release_s"], previous["release_s"])

    def test_entire_order_is_contiguous_and_ready_after_last_bag(self):
        for key in ("success", "failure"):
            result = self.story[key]
            for order in result["orders"]:
                services = [row for row in result["io_services"]
                            if (row["name"], row["dish"]) == (order["name"], order["dish"])]
                self.assertEqual(len(services), order["bags"])
                self.assertEqual([row["bag_index"] for row in services], list(range(1, order["bags"] + 1)))
                self.assertEqual(services[0]["start_s"], order["first_service_s"])
                self.assertEqual(services[-1]["end_s"], order["ready_s"])
                self.assertEqual(order["ready_s"] - order["first_service_s"], order["bags"])

    def test_deadline_is_releasing_computations_end(self):
        for key in ("success", "failure"):
            result = self.story[key]
            computes = {(row["name"], row["dish"]): row for row in result["computes"]}
            for order in result["orders"]:
                if order["cold_start"]:
                    self.assertIsNone(order["deadline_s"])
                    self.assertEqual(order["dish"], 0)
                    continue
                releasing_compute = computes[order["name"], order["dish"] - 1]
                self.assertEqual(order["release_s"], releasing_compute["start_s"])
                self.assertEqual(order["deadline_s"], releasing_compute["end_s"])
                self.assertGreater(order["deadline_s"], order["release_s"])

    def test_compute_starts_only_when_data_and_previous_compute_finish(self):
        for key in ("success", "failure"):
            result = self.story[key]
            orders = {(row["name"], row["dish"]): row for row in result["orders"]}
            computes = {(row["name"], row["dish"]): row for row in result["computes"]}
            for compute in result["computes"]:
                name, dish = compute["name"], compute["dish"]
                previous_end = 0 if dish == 0 else computes[name, dish - 1]["end_s"]
                self.assertEqual(compute["start_s"], max(previous_end, orders[name, dish]["ready_s"]))

    def test_stalls_are_after_deadline_not_all_io_waiting(self):
        for key in ("success", "failure"):
            result = self.story[key]
            orders = {(row["name"], row["dish"]): row for row in result["orders"]}
            for stall in result["stalls"]:
                order = orders[stall["name"], stall["dish"]]
                self.assertEqual(stall["start_s"], order["deadline_s"])
                self.assertEqual(stall["end_s"], order["ready_s"])
                self.assertGreater(stall["end_s"], stall["start_s"])

    def test_window_clipping_retains_partial_boundary_computes(self):
        result = self.story["failure"]
        self.assertEqual(clipped_duration(result["computes"], 28, 48, "A"), 16)
        self.assertEqual(clipped_duration(result["stalls"], 28, 48, "A"), 4)
        self.assertEqual(clipped_duration(result["computes"], 28, 48, "B"), 20)
        self.assertEqual(clipped_duration(result["io_services"], 28, 48), 16)
        # Plot begins at 27, while A's carried-in computation starts at 24.
        crossing = [row for row in result["computes"]
                    if row["name"] == "A" and row["start_s"] < 27 < row["end_s"]]
        self.assertEqual(len(crossing), 1)
        self.assertEqual(crossing[0]["start_s"], 24)
        self.assertEqual(crossing[0]["end_s"], 28)

    def test_workload_is_not_nominally_overloaded(self):
        checks = self.story["checks"]
        self.assertEqual(checks["failure_nominal_demand_bags_per_20_s"], 17)
        self.assertEqual(checks["failure_capacity_bags_per_20_s"], 20)
        self.assertLess(checks["failure_nominal_demand_bags_per_s"], 1)
        self.assertEqual(checks["failure_A_utilization"], 0.8)
        self.assertEqual(checks["failure_B_utilization"], 1)
        self.assertEqual(checks["failure_ssu_utilization"], 0.8)

    def test_exact_ready_at_deadline_does_not_duplicate_compute(self):
        result = simulate([{"name": "A", "bags": 1, "compute_s": 1}], horizon=8)
        pairs = [(row["name"], row["dish"]) for row in result["computes"]]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertFalse(result["stalls"])
        self.assertEqual([row["start_s"] for row in result["computes"]], list(range(1, 8)))

    def test_short_horizon_retains_cold_io_without_starting_future_compute(self):
        result = simulate([{"name": "A", "bags": 3, "compute_s": 4}], horizon=0.5)
        self.assertFalse(result["computes"])
        self.assertEqual(result["orders"][0]["ready_s"], 3)

    def test_invalid_parameters(self):
        good = {"name": "A", "bags": 1, "compute_s": 4}
        for horizon in (0, -1, math.inf, math.nan, True, "10"):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                simulate([good], horizon=horizon)
        for configs in ([], "A", [good, good], [{"name": "A", "bags": 0, "compute_s": 4}],
                        [{"name": "A", "bags": 1.5, "compute_s": 4}],
                        [{"name": "A", "bags": True, "compute_s": 4}],
                        [{"name": "", "bags": 1, "compute_s": 4}],
                        [{"name": "A", "bags": 1, "compute_s": -1}]):
            with self.subTest(configs=configs), self.assertRaises(ValueError):
                simulate(configs)


if __name__ == "__main__":
    unittest.main()
