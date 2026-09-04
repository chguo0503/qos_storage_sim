from __future__ import annotations

from dataclasses import replace
import unittest

from continuous_batch_sim import (
    ContinuousBatchRequest,
    SteadyStateConfig,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config
import sim


class ProfileCycleFrontierTrendTest(unittest.TestCase):
    @staticmethod
    def _requests(
        *, include_generation: bool = True, placement_changes_by_cycle: bool = False
    ):
        requests = []
        for generation in range(12):
            for npu_id in range(2):
                load = {
                    "category": "SS",
                    "per_layer_us": 1_000.0,
                    "stream_id": npu_id,
                }
                if include_generation:
                    load["generation"] = generation
                size_gb = (
                    0.001 * (1 + generation // 2)
                    if placement_changes_by_cycle
                    else 0.001
                )
                requests.append(
                    ContinuousBatchRequest(
                        request_id=1000 + generation * 2 + npu_id,
                        npu_id=npu_id,
                        arrival_time_ms=0.0,
                        load=load,
                        placement=(
                            ((npu_id, size_gb),),
                            ((npu_id, size_gb),),
                        ),
                    )
                )
        return tuple(requests)

    @staticmethod
    def _run(requests, steady):
        baseline = routing_strategy_specs()[0].client_config()
        return simulate_continuous_batch(
            requests,
            num_npu=2,
            num_ssu=2,
            n_layers=2,
            batch_size=1,
            policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=static_qos_config(),
            cross_request_layer0_prefetch=True,
            client_io_config=baseline,
            steady_state=steady,
            submit_order_seed=7,
        )

    def test_probe_is_read_only_and_retains_frontier_inventory(self):
        requests = self._requests()
        disabled = SteadyStateConfig(
            warmup_requests_per_npu=2,
            settle_ms=0.0,
            measurement_ms=6.0,
            block_ms=2.0,
        )
        enabled = replace(
            disabled,
            profile_cycle_probe_npu_ids=(0, 1),
            profile_cycle_period=2,
        )
        without_probe = self._run(requests, disabled)
        with_probe = self._run(requests, enabled)

        frontier = with_probe.pop("profile_cycle_frontier_trend")
        self.assertEqual(with_probe, without_probe)
        self.assertGreaterEqual(frontier["snapshot_count"], 1)
        self.assertTrue(
            frontier[
                "all_snapshots_profile_period_repeats_exact_placement_forcing"
            ]
        )
        previous_completed = None
        for snapshot in frontier["snapshots"]:
            self.assertEqual((snapshot["frontier_generation"] + 1) % 2, 0)
            self.assertEqual(len(snapshot["npu_compute_cumulative_busy_ms_by_npu"]), 2)
            self.assertEqual(len(snapshot["npu_state"]), 2)
            # Independent, identical SSUs make the NPUs exactly symmetric.
            # A direct in-handler capture used to observe NPU 0 already in the
            # next batch while NPU 1 was still between batches.  The normalized
            # post-timestamp cut must expose the same phase on both sides.
            self.assertEqual(
                snapshot["npu_state"][0]["pipeline_state"],
                snapshot["npu_state"][1]["pipeline_state"],
            )
            self.assertEqual(
                snapshot["npu_state"][0]["active_generations"],
                snapshot["npu_state"][1]["active_generations"],
            )
            self.assertEqual(
                snapshot["npu_state"][0]["active_generations"],
                [snapshot["frontier_generation"] + 1],
            )
            self.assertEqual(
                snapshot["relative_generation_by_probe_npu"]["0"],
                snapshot["relative_generation_by_probe_npu"]["1"],
            )
            completed = snapshot["completed_requests_cumulative_by_npu"]
            self.assertEqual(len(completed), 2)
            self.assertTrue(
                all(value >= snapshot["frontier_generation"] + 1
                    for value in completed)
            )
            if previous_completed is not None:
                self.assertTrue(
                    all(end >= start
                        for start, end in zip(previous_completed, completed))
                )
            previous_completed = completed
            inventory = snapshot["inventory"]
            self.assertEqual(
                len(inventory["ssd_submitted_cumulative_gb_by_npu_ssu"]), 2
            )
            stage_sum = sum(
                inventory[name]["fleet"]
                for name in (
                    "ssd_outstanding_gb_totals",
                    "npu_link_outstanding_gb_totals",
                    "client_unissued_gb_totals",
                    "ssd_served_awaiting_link_enqueue_gb_totals",
                )
            )
            self.assertAlmostEqual(
                stage_sum,
                inventory["total_physical_io_outstanding_gb_totals"]["fleet"],
            )
    def test_profile_period_does_not_hide_nonperiodic_placement(self):
        steady = SteadyStateConfig(
            warmup_requests_per_npu=2,
            settle_ms=0.0,
            measurement_ms=6.0,
            block_ms=2.0,
            profile_cycle_probe_npu_ids=(0, 1),
            profile_cycle_period=2,
        )
        summary = self._run(
            self._requests(placement_changes_by_cycle=True), steady
        )
        trend = summary["profile_cycle_frontier_trend"]
        self.assertGreaterEqual(trend["snapshot_count"], 1)
        self.assertFalse(
            trend[
                "all_snapshots_profile_period_repeats_exact_placement_forcing"
            ]
        )
        self.assertTrue(
            all(
                not snapshot["profile_and_placement_forcing"][
                    "profile_period_also_repeats_exact_placement_forcing"
                ]
                for snapshot in trend["snapshots"]
            )
        )

    def test_probe_requires_explicit_generation_metadata(self):
        steady = SteadyStateConfig(
            warmup_requests_per_npu=2,
            settle_ms=0.0,
            measurement_ms=4.0,
            block_ms=2.0,
            profile_cycle_probe_npu_ids=(0, 1),
            profile_cycle_period=2,
        )
        with self.assertRaisesRegex(ValueError, r"load\['generation'\]"):
            self._run(self._requests(include_generation=False), steady)


if __name__ == "__main__":
    unittest.main()
