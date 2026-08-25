import unittest

import numpy as np

import sim
from experiment import qos_config
from sim import (
    BlockIOFlow,
    ClientRoutingConfig,
    DiskIOScheduler,
    DiskState,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_STATIC_CIR,
    client_category_paths,
    client_select_qos_paths,
    prepare_simulation_inputs,
    proportional_npu_cap,
    simulate_continuous,
)


def _flow(path_id, index, size=0.001, npu_id=0):
    return BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=index,
        disk_id=0,
        total_gb=size,
        queue_id=path_id,
        block_count=1,
        enqueue_time=0.0,
    )


def _brute_scheduler_choice(scheduler):
    paths = [path for path in scheduler.paths.values() if path.pending]
    rates = sim._static_qos_service_rates(
        paths, scheduler.disk_bw, scheduler.group_weights
    )
    candidates = []
    for path in paths:
        rate = rates.get(path, 0.0)
        if rate <= sim._EPS:
            continue
        finish = path.virtual_finish + path.peek().total_gb / rate
        distance = (
            path.path_id - scheduler.qos_rr_cursor
        ) % len(scheduler.paths)
        candidates.append((finish, distance, path.path_id))
    best = min(row[0] for row in candidates)
    return min(
        (row for row in candidates if row[0] <= best + sim._EPS),
        key=lambda row: (row[1], row[2]),
    )[2]


def _brute_client_select(sizes, counts, allowed, routing):
    """优化前算法的紧凑参考实现，只用于等价性测试。"""
    qos = routing.qos_config
    counts = list(counts)
    typical = sorted(sizes)[len(sizes) // 2]
    work = [count * typical for count in counts]
    group_counts = [0] * qos.group_count
    active_paths = [0] * qos.group_count
    active_weights = [0.0] * qos.group_count
    active_group_weight = 0.0
    active_cir = 0.0
    for path_id, count in enumerate(counts):
        group = path_id // qos.paths_per_group
        group_counts[group] += count
        if count:
            if active_paths[group] == 0:
                active_group_weight += qos.group_weights[group]
            active_paths[group] += 1
            active_weights[group] += qos.path_weights[path_id]
            active_cir += min(qos.path_cirs[path_id], qos.path_pirs[path_id])

    offset = routing.start_offset % len(allowed)
    rotated = allowed[offset:] + allowed[:offset]
    tie_rank = {path_id: rank for rank, path_id in enumerate(rotated)}
    binding = routing.path_binding_batch_size
    batches = []
    for start in range(0, len(sizes), binding):
        indices = tuple(range(start, min(start + binding, len(sizes))))
        batches.append((indices, sum(sizes[index] for index in indices)))
    order = sorted(
        range(len(batches)),
        key=lambda index: (-batches[index][1], batches[index][0][0]),
    )
    result = [None] * len(sizes)
    for batch_index in order:
        indices, batch_gb = batches[batch_index]
        best = None
        for path_id in allowed:
            group = path_id // qos.paths_per_group
            was_empty = counts[path_id] == 0
            base = min(qos.path_cirs[path_id], qos.path_pirs[path_id])
            cir = active_cir + (base if was_empty else 0.0)
            group_weight = active_group_weight
            path_weight = active_weights[group]
            if was_empty:
                path_weight += qos.path_weights[path_id]
                if active_paths[group] == 0:
                    group_weight += qos.group_weights[group]
            remaining = max(0.0, routing.disk_bw - cir)
            extra = (
                remaining
                * qos.group_weights[group]
                / group_weight
                * qos.path_weights[path_id]
                / path_weight
            )
            rate = min(qos.path_pirs[path_id], base + extra)
            finish = (work[path_id] + batch_gb) / rate
            tie = (counts[path_id], group_counts[group], tie_rank[path_id])
            candidate = (finish, tie, path_id)
            if best is None or finish < best[0] - sim._EPS:
                best = candidate
            elif abs(finish - best[0]) <= sim._EPS and tie < best[1]:
                best = candidate
        selected = best[2]
        for index in indices:
            result[index] = selected
        group = selected // qos.paths_per_group
        if counts[selected] == 0:
            if active_paths[group] == 0:
                active_group_weight += qos.group_weights[group]
            active_paths[group] += 1
            active_weights[group] += qos.path_weights[selected]
            active_cir += min(qos.path_cirs[selected], qos.path_pirs[selected])
        counts[selected] += len(indices)
        group_counts[group] += len(indices)
        work[selected] += batch_gb
    return result


class NpuCapTests(unittest.TestCase):
    def test_documented_cap_examples(self):
        self.assertEqual(proportional_npu_cap([40.0]), (40.0,))
        self.assertEqual(proportional_npu_cap([40.0, 40.0]), (25.0, 25.0))
        self.assertEqual(proportional_npu_cap([40.0, 5.0]), (40.0, 5.0))
        self.assertEqual(proportional_npu_cap([40.0] * 8), (6.25,) * 8)

    def test_invalid_cap_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            proportional_npu_cap([1.0, -1.0])
        with self.assertRaises(ValueError):
            proportional_npu_cap([1.0], 0.0)


class QoSPathTests(unittest.TestCase):
    def test_256_count_abi_enqueue_activate_complete(self):
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, qos_config()
        )
        flow = _flow(17, 0)
        before = scheduler.report_path_io_counts(0.0)
        scheduler.enqueue_many([flow], 0.0)
        queued = scheduler.report_path_io_counts(0.0)
        self.assertEqual(len(queued), 256)
        self.assertEqual(queued[17], before[17] + 1)
        self.assertEqual(queued[:17] + queued[18:], before[:17] + before[18:])

        scheduler.dispatch(0.0, [], schedule_completion=False)
        self.assertEqual(scheduler.report_path_io_counts(0.0)[17], 1)
        scheduler.complete_ready_flows(flow.end_time)
        self.assertEqual(scheduler.report_path_io_counts(flow.end_time)[17], 0)

    def test_heap_scheduler_matches_full_scan_reference(self):
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, qos_config()
        )
        rng = np.random.RandomState(7)
        path_ids = (0, 1, 12, 16, 28, 32, 60, 92, 124, 156, 188, 220)
        flows = [
            _flow(path_id, index, size=float(rng.choice([0.001, 0.002, 0.004])))
            for index, path_id in enumerate(path_ids * 5)
        ]
        scheduler.enqueue_many(flows, 0.0)
        now = 0.0
        for _ in flows:
            expected = _brute_scheduler_choice(scheduler)
            active = scheduler.dispatch(now, [], schedule_completion=False)
            self.assertEqual(active.queue_id, expected)
            self.assertEqual(len(state.active_flows), 1)
            now = active.end_time
            scheduler.complete_ready_flows(now)
        self.assertFalse(state.active_flows)

    def test_fast_client_selector_matches_brute_reference(self):
        qos = qos_config()
        rng = np.random.RandomState(11)
        for category in ("SS", "SL", "LS", "LL"):
            allowed = client_category_paths(category, qos)
            for binding in (1, 2, 4, 8):
                counts = tuple(int(value) for value in rng.randint(0, 9, 256))
                sizes = tuple(float(value) for value in rng.choice(
                    [0.001, 0.002, 0.004], size=13
                ))
                routing = ClientRoutingConfig(
                    qos_config=qos,
                    disk_bw=40.0,
                    start_offset=17,
                    path_binding_batch_size=binding,
                )
                before = counts
                actual = client_select_qos_paths(
                    block_sizes_gb=sizes,
                    path_io_counts=counts,
                    allowed_path_ids=allowed,
                    routing_config=routing,
                )
                self.assertEqual(
                    actual,
                    _brute_client_select(sizes, counts, allowed, routing),
                )
                self.assertEqual(counts, before)


class IntegrationTests(unittest.TestCase):
    def test_baseline_and_qos_share_inputs_and_obey_cap(self):
        table = {(1, 0): (20.0, 1_000.0, 10.0, 0.02)}
        prepared = prepare_simulation_inputs(
            table,
            total_requests=2,
            n_layers=2,
            num_disk=4,
            workload_seed=3,
            placement_seed=5,
        )
        common = dict(
            num_npu=2,
            num_disk=4,
            n_layers=2,
            prepared_inputs=prepared,
            client_submit_batch_size=8,
            submit_order_seed=9,
            npu_bw_limit=50.0,
        )
        _, baseline = simulate_continuous(
            table, policy=POLICY_BASELINE_BYPASS, **common
        )
        _, qos = simulate_continuous(
            table,
            policy=POLICY_QOS_STATIC_CIR,
            qos_config=qos_config(),
            pressure_read_interval=8,
            path_binding_batch_size=1,
            **common,
        )
        for summary in (baseline, qos):
            conservation = summary["block_conservation"]
            self.assertEqual(
                conservation["expected"],
                conservation["submitted"],
            )
            self.assertEqual(
                conservation["submitted"],
                conservation["completed"],
            )
            self.assertTrue(conservation["placement_targets_preserved"])
            self.assertLessEqual(
                summary["npu_link_peak_effective_bw_gbps"], 50.0 + 1e-9
            )
            self.assertLessEqual(
                max(row["max_backend_active_io"] for row in summary["disk_stats"]),
                1,
            )
        self.assertEqual(
            baseline["workload_fingerprint"], qos["workload_fingerprint"]
        )
        self.assertEqual(baseline["placement_hash"], qos["placement_hash"])


if __name__ == "__main__":
    unittest.main()
