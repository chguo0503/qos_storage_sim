"""Tests for causal multi-SSU assignment and its unchanged-core adapter."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, simulate_continuous_batch
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from multi_ssu_npu_assignment import (
    IO_SIZE_BYTES, IO_SIZE_GIB, MultiNPUBacklog, assign_requests_on_arrival,
    choose_npu, request_layer_io_by_ssu, snapshot_npu_backlogs,
)


class MultiAssignmentTests(unittest.TestCase):
    def test_32_npus_6_or_7_disks_and_units(self):
        self.assertEqual(IO_SIZE_BYTES, 176 * 1024)
        for ssu in (6, 7):
            backlogs = tuple(MultiNPUBacklog(n, float(32 - n), (0,) * ssu) for n in range(32))
            decision = choose_npu(10, backlogs, (1,) * ssu, 8,
                                  disk_bandwidths_gib_s=(40,) * ssu)
            self.assertEqual(len(decision.selection_scores_ms_by_npu), 32)
            self.assertEqual(decision.npu_id, 31)
            self.assertEqual(decision.estimated_finish_ms_by_npu[31], 19)

    def test_same_total_io_different_ssu_hotspot_changes_score(self):
        common = MultiNPUBacklog(0, 1, (1000, 0))
        hot = (common, MultiNPUBacklog(1, 2, (1000, 0)))
        split = (common, MultiNPUBacklog(1, 2, (0, 1000)))
        a = choose_npu(0, hot, (1, 0), 1, disk_bandwidths_gib_s=(40, 40))
        b = choose_npu(0, split, (1, 0), 1, disk_bandwidths_gib_s=(40, 40))
        self.assertGreater(a.disk_overload_factors_by_candidate[0],
                           b.disk_overload_factors_by_candidate[0])
        self.assertGreater(a.selection_scores_ms_by_npu[0], b.selection_scores_ms_by_npu[0])

    def test_50_gib_link_can_dominate_multiple_40_gib_disks(self):
        decision = choose_npu(0, (MultiNPUBacklog(0, 0, (0, 0)),), (500, 500), 1,
                              disk_bandwidths_gib_s=(40, 40))
        self.assertGreater(decision.link_overload_factors_by_candidate[0],
                           decision.disk_overload_factors_by_candidate[0])
        self.assertAlmostEqual(decision.selection_scores_ms_by_npu[0],
                               1000 * IO_SIZE_GIB / 50 * 1000)

    def test_ssu_hotspot_can_change_selected_npu_with_identical_io_totals(self):
        common = MultiNPUBacklog(0, .2, (10, 0))
        hot = (common, MultiNPUBacklog(1, .2, (100, 0)))
        split = (common, MultiNPUBacklog(1, .2, (0, 100)))
        a = choose_npu(0, hot, (1000, 0), .2, disk_bandwidths_gib_s=(40, 40))
        b = choose_npu(0, split, (1000, 0), .2, disk_bandwidths_gib_s=(40, 40))
        self.assertEqual(a.npu_id, 1)
        self.assertEqual(b.npu_id, 0)

    def test_compute_ablation_and_noncontiguous_ids(self):
        backlogs = (MultiNPUBacklog(7, 2, (10000, 0)), MultiNPUBacklog(3, 2, (0, 0)))
        result = choose_npu(10, backlogs, (0, 0), 1,
                            disk_bandwidths_gib_s=(40, 40), policy="compute")
        self.assertEqual(result.selection_scores_ms_by_npu, (13, 13))
        self.assertEqual(result.npu_id, 3)
        self.assertEqual(result.selection_objective, "remaining_compute_on_selected_npu")
        # This ablation must not use I/O as a tie breaker either.
        equal = (MultiNPUBacklog(0, 2, (10000, 0)), MultiNPUBacklog(1, 2, (0, 0)))
        result = choose_npu(0, equal, (0, 0), 1,
                            disk_bandwidths_gib_s=(40, 40), policy="compute")
        self.assertEqual(result.npu_id, 0)

    def test_per_layer_placements_and_full_io_scope(self):
        manifest = ContinuousBatchRequest(0, 0, 0, {}, (
            ((0, IO_SIZE_GIB),), ((1, IO_SIZE_GIB), (1, IO_SIZE_GIB)),
        ))
        self.assertEqual(request_layer_io_by_ssu(manifest, 2, 2), ((1, 0), (0, 2)))
        partial = replace(manifest, placement=(((0, IO_SIZE_GIB / 2),),))
        with self.assertRaisesRegex(ValueError, "176 KiB"):
            request_layer_io_by_ssu(partial, 2, 2)

    def test_snapshot_counts_completed_client_bytes_not_ssd_queue(self):
        manifest = ContinuousBatchRequest(0, 0, 0, {}, (
            ((0, IO_SIZE_GIB), (1, IO_SIZE_GIB)),
        ))
        request = SimpleNamespace(manifest=manifest, per_layer_compute_ms=2,
                                  io_ready=(False, False),
                                  completed_gb_by_layer_ssu=((IO_SIZE_GIB, 0), (0, 0)))
        batch = SimpleNamespace(member_request_ids=(0,), compute_done_up_to=-1)
        npu = SimpleNamespace(npu_id=0, admission_queue=(), active_batch=batch, compute_active=None)
        # No disks or event heap exist: accessing hidden scheduler state fails.
        context = SimpleNamespace(npus=(npu,), requests={0: request}, n_layers=2, num_ssu=2)
        snapshot = snapshot_npu_backlogs(context, 0)[0]
        self.assertEqual(snapshot.remaining_compute_ms, 4)
        self.assertEqual(snapshot.remaining_io_by_ssu, (1, 2))

    @staticmethod
    def requests(ssu, count=40):
        return tuple(ContinuousBatchRequest(
            request_id=i, npu_id=i % 32, arrival_time_ms=float(i // 32),
            load={"category": "SS", "per_layer_us": 1000 * (1 + i % 3)},
            placement=(tuple((j % ssu, IO_SIZE_GIB) for j in range(4)),),
        ) for i in range(count))

    @staticmethod
    def simulate(requests, ssu):
        return simulate_continuous_batch(
            requests, num_npu=32, num_ssu=ssu, n_layers=8, batch_size=1,
            qos_config=static_qos_config(),
            client_io_config=routing_strategy_specs()[0].client_config(),
            cross_request_layer0_prefetch=True, disk_bw_gbps=40, npu_bw_gbps=50,
        )

    def test_actual_32_npu_simulator_arrivals_and_immutable_ssd_placement(self):
        original = native._handle_arrival
        for ssu in (6, 7):
            with self.subTest(ssu=ssu):
                requests = self.requests(ssu)
                before = tuple((r.npu_id, r.arrival_time_ms, r.placement) for r in requests)
                with assign_requests_on_arrival() as log:
                    result = self.simulate(requests, ssu)
                self.assertIs(native._handle_arrival, original)
                self.assertEqual(before, tuple((r.npu_id, r.arrival_time_ms, r.placement) for r in requests))
                by_id = {r["request_id"]: r for r in result["request_metrics"]}
                self.assertEqual(len(log), len(requests))
                for event in log:
                    rid = event["request_id"]
                    self.assertEqual(event["arrival_time_ms"], requests[rid].arrival_time_ms)
                    self.assertEqual(event["assigned_npu_id"], by_id[rid]["npu_id"])
                    self.assertEqual(sum(event["incoming_total_io_by_ssu"]), 32)
                    self.assertGreaterEqual(event["decision_wall_time_us"], 0)
                self.assertTrue(all(result["invariants"].values()))
                self.assertTrue(any(e["assigned_npu_id"] != e["original_npu_id"] for e in log))

    def test_future_profiles_cannot_change_earlier_decisions(self):
        requests = self.requests(6)
        changed = requests[:32] + tuple(replace(r, load={**r.load, "per_layer_us": 1000000})
                                       for r in requests[32:])
        for policy in ("fluid", "compute"):
            with assign_requests_on_arrival(policy=policy) as a:
                self.simulate(requests, 6)
            with assign_requests_on_arrival(policy=policy) as b:
                self.simulate(changed, 6)
            for first, second in zip(a[:32], b[:32]):
                self.assertEqual(first["assigned_npu_id"], second["assigned_npu_id"])
                self.assertEqual(first["selection_scores_ms_by_npu"], second["selection_scores_ms_by_npu"])

    def test_hook_restored_after_error(self):
        original = native._handle_arrival
        with self.assertRaisesRegex(RuntimeError, "example"):
            with assign_requests_on_arrival():
                raise RuntimeError("example")
        self.assertIs(native._handle_arrival, original)


if __name__ == "__main__":
    unittest.main()
