"""Deterministic regressions for fragmentation-independent SSD accounting.

Run directly with::

    python -B ssd_accounting_checks.py

The checks exercise the SSD scheduler and compact read-only diagnostic
projectors.  They do not run a workload experiment or write artifacts.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import continuous_batch_sim as continuous
from continuous_prefill_client import static_qos_config
import sim


DISK_BW_GBPS = 40.0


def _flow(
    *,
    npu_id: int,
    block_idx: int,
    total_gb: float,
    block_count: int = 1,
    enqueue_time: float = 0.0,
    queue_id: int = -1,
):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        layer=0,
        block_idx=block_idx,
        disk_id=0,
        total_gb=total_gb,
        queue_id=queue_id,
        block_count=block_count,
        enqueue_time=enqueue_time,
    )


def _oracle_scheduler(disk: sim.DiskState):
    return sim.DiskIOScheduler(
        disk,
        sim.POLICY_PER_SSD_FULL_VISIBLE_EDF,
        DISK_BW_GBPS,
        oracle_priority=lambda flow: (flow.block_idx, flow.npu_id),
    )


def _static_scheduler(disk: sim.DiskState):
    return sim.DiskIOScheduler(
        disk,
        sim.POLICY_QOS_STATIC_CIR,
        DISK_BW_GBPS,
        qos_config=static_qos_config(),
    )


def _context(disks, *, num_npu: int):
    return SimpleNamespace(
        disks=disks,
        num_npu=num_npu,
        num_ssu=len(disks),
        disk_bw_gbps=DISK_BW_GBPS,
    )


class SSDAccountingChecks(unittest.TestCase):
    def test_compensated_nonnegative_counter_matches_fsum(self):
        values = [1e8] + [1e-8] * 100_000
        expected = math.fsum(values)
        naive = 0.0
        compensated = 0.0
        correction = 0.0
        for value in values:
            naive += value
            compensated, correction = sim.DiskState._kahan_add(
                compensated,
                correction,
                value,
            )

        self.assertEqual(compensated, expected)
        self.assertGreater(abs(naive - expected), 0.0)

    def test_completion_and_service_are_invariant_to_extreme_fragmentation(self):
        total_gb = 1.23456789
        enqueue_time_ms = 1.0
        activation_time_ms = 3.25
        service_duration_ms = total_gb * 1000.0 / DISK_BW_GBPS
        observation_ms = activation_time_ms + service_duration_ms * 0.5
        completion_ms = activation_time_ms + service_duration_ms
        for fragment_count, policy_name, scheduler_factory, queue_id in (
            (fragment_count, *policy_case)
            for fragment_count in (250_000, 1_000_000)
            for policy_case in (
                ("oracle", _oracle_scheduler, -1),
                ("static_cir", _static_scheduler, 0),
            )
        ):
            with self.subTest(policy=policy_name, fragments=fragment_count):
                disks = [sim.DiskState(0), sim.DiskState(0)]
                schedulers = [scheduler_factory(disk) for disk in disks]
                flows = [
                    _flow(
                        npu_id=0,
                        block_idx=0,
                        total_gb=total_gb,
                        enqueue_time=enqueue_time_ms,
                        queue_id=queue_id,
                    )
                    for _ in disks
                ]
                event_heaps = [[], []]
                for scheduler, flow, event_heap in zip(
                    schedulers,
                    flows,
                    event_heaps,
                ):
                    scheduler.enqueue_many([flow], enqueue_time_ms)
                    scheduler.dispatch(activation_time_ms, event_heap)
                    self.assertEqual(flow.ssd_activation_time, activation_time_ms)
                    self.assertEqual(flow.end_time, completion_ms)
                    self.assertEqual(event_heap[0][0], completion_ms)

                fragmented_scheduler = schedulers[0]
                for fragment in range(1, fragment_count // 2 + 1):
                    fragmented_scheduler.settle(
                        activation_time_ms
                        + service_duration_ms * fragment / fragment_count
                    )

                fragmented_context = _context([disks[0]], num_npu=1)
                unfragmented_context = _context([disks[1]], num_npu=1)
                fragmented_service = continuous._project_ssd_service_by_npu(
                    fragmented_context,
                    observation_ms,
                )[0][0]
                unfragmented_service = continuous._project_ssd_service_by_npu(
                    unfragmented_context,
                    observation_ms,
                )[0][0]
                expected_midpoint_service = total_gb * 0.5
                self.assertEqual(fragmented_service, unfragmented_service)
                self.assertEqual(fragmented_service, expected_midpoint_service)

                for fragment in range(fragment_count // 2 + 1, fragment_count):
                    fragmented_scheduler.settle(
                        activation_time_ms
                        + service_duration_ms * fragment / fragment_count
                    )
                self.assertEqual(
                    fragmented_scheduler.complete_ready_flows(
                        math.nextafter(completion_ms, -math.inf)
                    ),
                    [],
                )
                completed_fragmented = (
                    fragmented_scheduler.complete_ready_flows(completion_ms)
                )
                completed_unfragmented = schedulers[1].complete_ready_flows(
                    completion_ms
                )

                self.assertEqual(completed_fragmented, [flows[0]])
                self.assertEqual(completed_unfragmented, [flows[1]])
                self.assertEqual(flows[0].remaining_gb, 0.0)
                self.assertEqual(flows[1].remaining_gb, 0.0)
                self.assertEqual(disks[0].completed_gb_by_npu[0], total_gb)
                self.assertEqual(disks[1].completed_gb_by_npu[0], total_gb)
                self.assertEqual(disks[0].completed_bytes_gb, total_gb)
                self.assertEqual(disks[1].completed_bytes_gb, total_gb)
                self.assertAlmostEqual(
                    disks[0].busy_time,
                    service_duration_ms,
                    delta=1e-12,
                )
                self.assertAlmostEqual(
                    disks[1].busy_time,
                    service_duration_ms,
                    delta=1e-12,
                )

    def test_physical_outstanding_enumerates_pending_and_active_commands(self):
        disk = sim.DiskState(0)
        scheduler = _oracle_scheduler(disk)
        flows = [
            _flow(npu_id=0, block_idx=0, total_gb=0.8, block_count=3),
            _flow(npu_id=1, block_idx=1, total_gb=0.7, block_count=2),
            _flow(npu_id=0, block_idx=2, total_gb=0.4, block_count=1),
        ]
        scheduler.enqueue_many(flows, 0.0)
        scheduler.dispatch(0.0, [])
        boundary_ms = 5.0
        scheduler.settle(boundary_ms)

        context = _context([disk], num_npu=2)
        outstanding_gb, outstanding_blocks = (
            continuous._project_physical_ssd_outstanding_by_npu(
                context,
                boundary_ms,
            )
        )
        stable_service = continuous._project_ssd_service_by_npu(
            context,
            boundary_ms,
        )

        # The active 0.8-GB command has received 40 GB/s * 5 ms = 0.2 GB.
        self.assertEqual(outstanding_gb, [[1.0], [0.7]])
        self.assertEqual(outstanding_blocks, [[4], [2]])
        self.assertEqual(stable_service, [[0.2], [0.0]])
        self.assertEqual(
            math.fsum(row[0] for row in outstanding_gb),
            1.7,
        )
        self.assertEqual(
            sum(row[0] for row in outstanding_blocks),
            scheduler.outstanding_blocks,
        )

    def test_carry_in_export_is_exact_compact_and_interval_closed(self):
        n_layers = 16
        admission_ms = 0.0
        barrier_start_ms = admission_ms
        layer_metrics = []
        for layer in range(n_layers):
            compute_start_ms = barrier_start_ms + 1.0
            compute_end_ms = compute_start_ms + 2.0
            layer_metrics.append(
                continuous._MicrobatchLayerMetric(
                    layer=layer,
                    io_ready_time_ms=compute_start_ms,
                    compute_start_ms=compute_start_ms,
                    compute_end_ms=compute_end_ms,
                    compute_duration_ms=2.0,
                    io_barrier_wait_ms=1.0,
                )
            )
            barrier_start_ms = compute_end_ms
        carry_batch = continuous._MicrobatchState(
            batch_id=7,
            npu_id=1,
            member_request_ids=(101,),
            admission_time_ms=admission_ms,
            previous_compute_end_ms=barrier_start_ms,
            completion_time_ms=barrier_start_ms,
            layer_metrics=layer_metrics,
        )
        # This later batch is an in-window admission and must remain exclusive
        # to request_rows rather than being duplicated in the carry-in export.
        in_window_batch = continuous._MicrobatchState(
            batch_id=8,
            npu_id=1,
            member_request_ids=(102,),
            admission_time_ms=60.0,
            previous_compute_end_ms=61.0,
            completion_time_ms=62.0,
            layer_metrics=[],
        )
        request = SimpleNamespace(
            batch_id=carry_batch.batch_id,
            admission_time_ms=carry_batch.admission_time_ms,
            completion_time_ms=carry_batch.completion_time_ms,
            per_layer_compute_ms=2.0,
            manifest=SimpleNamespace(npu_id=carry_batch.npu_id),
        )
        context = SimpleNamespace(
            microbatches=[carry_batch, in_window_batch],
            requests={101: request},
            n_layers=n_layers,
        )

        rows, invariants = continuous._timeline_carry_in_batch_rows(
            context,
            window_start_ms=24.0,
        )

        self.assertTrue(all(invariants.values()), invariants)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["batch_id"], 7)
        self.assertEqual(row["request_ids"], [101])
        self.assertEqual(row["npu_id"], 1)
        self.assertEqual(row["layer_count"], n_layers)
        self.assertEqual(row["per_layer_compute_ms"], 2.0)
        self.assertEqual(row["ideal_compute_ms"], 32.0)
        for field in (
            "io_ready_time_ms",
            "compute_start_ms",
            "compute_end_ms",
            "compute_duration_ms",
            "io_barrier_wait_ms",
        ):
            self.assertEqual(len(row[field]), n_layers)

        # Stationarity snapshots are left limits: a completion event exactly
        # at the measurement start has not executed yet, so this zero-overlap
        # batch must still be present for state authentication.
        equality_rows, equality_invariants = (
            continuous._timeline_carry_in_batch_rows(
                context,
                window_start_ms=carry_batch.completion_time_ms,
            )
        )
        self.assertTrue(all(equality_invariants.values()), equality_invariants)
        self.assertEqual([row["batch_id"] for row in equality_rows], [7])

        # A scheduler gap not explained by either I/O readiness or the prior
        # layer's completion must fail the producer contract instead of being
        # silently classified as I/O-barrier time.
        carry_batch.layer_metrics[0].io_ready_time_ms -= 0.25
        _, invalid_invariants = continuous._timeline_carry_in_batch_rows(
            context,
            window_start_ms=24.0,
        )
        self.assertFalse(invalid_invariants["timeline_carry_in_interval_closure"])


if __name__ == "__main__":
    unittest.main()
