"""Contracts for the experimental client and same-input simulator comparisons."""
from types import SimpleNamespace
import unittest

from run_stall_policy_experiments import (
    CAP_GIB_S, IO_BYTES, IO_GIB, PATHS, ManifestCIRController,
    Profile, build_requests, build_variable_requests, old_rounded_profiles,
    profiles_for_demand, request_weighted_nominal_demand, run_case,
    summarize_complete_run,
)


class ExperimentContracts(unittest.TestCase):
    def test_full_block_input_and_no_silent_padding(self):
        self.assertEqual(IO_BYTES, 176 * 1024)
        profiles = old_rounded_profiles()
        requests = build_requests(profiles, count_per_npu=1)
        for profile, request in zip(profiles, requests):
            self.assertEqual(len(request.placement[0]), profile.blocks)
            self.assertEqual(set(request.placement[0]), {(0, IO_GIB)})
        self.assertLess(sum(p.demand_gib_s for p in profiles), CAP_GIB_S)

    def test_compute_time_encodes_declared_demand(self):
        demands = (0.6, 1.6, 26.0, 11.0)
        profiles = profiles_for_demand((8, 32, 1536, 1536), demands)
        for profile, demand in zip(profiles, demands):
            self.assertAlmostEqual(profile.demand_gib_s, demand)

    def test_arrived_only_controller_and_no_successor_double_count(self):
        def request(rid, lane, rate):
            return SimpleNamespace(request_id=rid, npu_id=lane,
                                   next_layer_work_gb_by_ssu=(rate / 1000,),
                                   per_layer_compute_ms=1.0)
        snapshot = SimpleNamespace(
            time_ms=12.0,
            active_requests=(request(1, 0, 10), request(2, 0, 12), request(3, 1, 8)),
            path_outstanding_io_counts_by_ssu=((0,) * 256,))
        controller = ManifestCIRController()
        result = controller(snapshot)
        self.assertEqual(result.path_cirs_by_ssu[0][PATHS[0]], 12)
        self.assertEqual(result.path_cirs_by_ssu[0][PATHS[1]], 8)
        self.assertEqual(sum(result.path_cirs_by_ssu[0]), 20)
        self.assertEqual(controller.decisions[0]["active_request_ids"], [1, 2, 3])

    def test_overload_scaling_respects_capacity(self):
        snapshot = SimpleNamespace(
            time_ms=0.0,
            active_requests=tuple(SimpleNamespace(
                request_id=i, npu_id=i, next_layer_work_gb_by_ssu=(0.02,),
                per_layer_compute_ms=1.0) for i in range(4)),
            path_outstanding_io_counts_by_ssu=((0,) * 256,))
        result = ManifestCIRController()(snapshot)
        self.assertEqual([result.path_cirs_by_ssu[0][p] for p in PATHS], [10.0] * 4)

    def test_same_input_preserved_and_actual_simulator_invariants(self):
        profiles = tuple(Profile(2 + i, 0.1 * (i + 1)) for i in range(4))
        results = [run_case(profiles, strategy, finite_count=2)
                   for strategy in ("baseline", "layer_once", "dedicated_demand")]
        self.assertEqual(len({r["input_fingerprint"] for r in results}), 1)
        for result in results:
            summary = result["summary"]
            self.assertEqual(summary["submitted_blocks"], 8 * 2 * sum(p.blocks for p in profiles))
            self.assertAlmostEqual(summary["completed_read_gb"], summary["expected_read_gb"])
            self.assertTrue(all(summary["invariants"].values()))
            self.assertLessEqual(max(summary["max_actual_cir_sum_gbps_by_ssu"]), 40.0 + 1e-10)

    def test_variable_requests_reproduce_exact_data_not_arbitrary_compute(self):
        from authenticated_workload_inputs import load_authenticated_bw_table
        table, _ = load_authenticated_bw_table(4)
        for mix in ("long", "broad"):
            requests, pool, metadata = build_variable_requests(mix=mix)
            self.assertTrue(metadata["authenticated_data"]["source_sha256"])
            self.assertGreater(len({r.load["nql"] for r in requests}), 1)
            if mix == "broad":
                self.assertGreater(len({r.load["seq_len_k"] for r in requests}), 1)
            for request in requests:
                key = (request.load["seq_len_k"], request.load["nql"])
                self.assertAlmostEqual(request.load["per_layer_us"], table[key][1])
                self.assertEqual(len(request.placement[0]), (key[0] * 1024 - key[1]) // 128)
                self.assertEqual(set(request.placement[0]), {(0, IO_GIB)})
                self.assertFalse(request.load["constructed_profile"])

    def test_mixed_workload_uses_compute_time_weighted_demand(self):
        requests = [SimpleNamespace(npu_id=0, placement=(((0, IO_GIB),),),
                                    load={"per_layer_us": c}) for c in (1000, 3000)]
        demand = request_weighted_nominal_demand(requests)
        self.assertAlmostEqual(demand["sum_gib_s"], 2 * IO_GIB * 1000 / 4)
        self.assertNotAlmostEqual(demand["sum_gib_s"], (IO_GIB * 1000 + IO_GIB * 1000 / 3) / 2)

    def test_variable_seed_preserves_manifest_and_arrival_rate_contract(self):
        a, _, metadata = build_variable_requests(seed=20260906)
        b, _, _ = build_variable_requests(seed=20260906)
        self.assertEqual(a, b)
        self.assertAlmostEqual(metadata["subsequent_arrival_read_gib"] * 1000 /
                               metadata["subsequent_pacing_budget_end_ms"], 39.2)
        self.assertGreater(metadata["all_input_bytes_over_pacing_budget_gib_s"], 39.2)

    def test_common_window_clips_busy_and_idle_without_hiding_either(self):
        result = run_case(tuple(Profile(2, 0.1) for _ in range(4)),
                          "dedicated_demand", finite_count=2)
        active = summarize_complete_run(result, start_ms=0, end_ms=0.5)
        drained = summarize_complete_run(result, start_ms=1000, end_ms=2000)
        self.assertTrue(active["all_npus_active_whole_window"])
        self.assertLessEqual(active["mean_npu_utilization"], 1.0)
        self.assertFalse(drained["all_npus_active_whole_window"])
        self.assertEqual(drained["mean_npu_utilization"], 0.0)


if __name__ == "__main__":
    unittest.main()
