from collections import Counter
from dataclasses import FrozenInstanceError, replace
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import experiment
import sim
from experiment import (
    CATEGORY_CIR_GBPS,
    category_path_cirs,
    plot_routing_comparison,
    qos_config,
    routing_comparison_spec,
    run_routing_comparison_case,
)
from sim import (
    BlockIOFlow,
    ClientRoutingConfig,
    DiskIOScheduler,
    DiskState,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_STATIC_CIR,
    QOS_ROUTING_CATEGORIES,
    SUPPORTED_POLICIES,
    client_category_paths,
    client_select_qos_paths,
    simulate_continuous,
)


def _select(
    counts,
    allowed,
    block_count,
    start_offset=0,
    config=None,
    block_sizes_gb=None,
    path_binding_batch_size=1,
):
    config = qos_config() if config is None else config
    block_sizes_gb = (
        [1.0] * block_count if block_sizes_gb is None else block_sizes_gb
    )
    return client_select_qos_paths(
        block_sizes_gb=block_sizes_gb,
        path_io_counts=counts,
        allowed_path_ids=allowed,
        routing_config=ClientRoutingConfig(
            qos_config=config,
            disk_bw=40.0,
            start_offset=start_offset,
            path_binding_batch_size=path_binding_batch_size,
        ),
    )


def _dispatch_path_sequence(scheduler, state, count):
    """直接驱动整盘离散仲裁，返回每次获选的 Path ID。"""
    selected_paths = []
    current_time = 0.0
    for _ in range(count):
        scheduler.dispatch(current_time, [])
        if len(state.active_flows) != 1:
            raise AssertionError("每次仲裁必须恰好启动一个 I/O")
        active = state.active_flows[0]
        selected_paths.append(active.queue_id)
        current_time = active.end_time
        scheduler.complete_ready_flows(current_time)
    return selected_paths


class StaticCirQoSTests(unittest.TestCase):
    def test_static_cir_is_spread_over_256_paths(self):
        cirs = category_path_cirs()

        self.assertEqual(len(cirs), 256)
        self.assertAlmostEqual(sum(cirs), 40.0)
        expected_group = (
            [20.0 / 96] * 12
            + [4.0 / 32] * 4
            + [12.0 / 96] * 12
            + [4.0 / 32] * 4
        )
        for group_id in range(8):
            actual_group = cirs[group_id * 32 : (group_id + 1) * 32]
            for actual, expected in zip(actual_group, expected_group):
                self.assertAlmostEqual(actual, expected)

        config = qos_config()
        category_totals = [
            sum(cirs[path_id] for path_id in client_category_paths(category, config))
            for category in QOS_ROUTING_CATEGORIES
        ]
        for actual, expected in zip(category_totals, CATEGORY_CIR_GBPS):
            self.assertAlmostEqual(actual, expected)

    def test_static_qos_config_is_immutable_and_minimal(self):
        config = qos_config()

        self.assertEqual(config.path_count, 256)
        self.assertEqual(config.group_count, 8)
        self.assertEqual(config.paths_per_group, 32)
        self.assertEqual(config.path_pirs, (float("inf"),) * 256)
        self.assertEqual(set(config.path_weights), {1.0})
        self.assertEqual(set(config.group_weights), {1.0})
        self.assertEqual(
            set(vars(config)),
            {
                "path_cirs",
                "path_pirs",
                "path_weights",
                "group_weights",
                "category_paths_per_group",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            config.path_cirs = ()

    def test_category_pools_have_the_exact_static_layout(self):
        layouts = (
            ("SS", 0, 12),
            ("SL", 12, 4),
            ("LS", 16, 12),
            ("LL", 28, 4),
        )
        for category, offset, count in layouts:
            with self.subTest(category=category):
                expected = tuple(
                    group_id * 32 + local_id
                    for local_id in range(offset, offset + count)
                    for group_id in range(8)
                )
                self.assertEqual(
                    client_category_paths(category, qos_config()), expected
                )

    def test_selector_stays_inside_each_strict_category(self):
        config = qos_config()
        for category in QOS_ROUTING_CATEGORIES:
            with self.subTest(category=category):
                allowed = client_category_paths(category, config)
                counts = [0] * config.path_count

                # 类别外的 Path 都为空，看起来压力更低；但它们没有出现在
                # allowed 中，因此客户端绝不会跨类别选择这些 Path。
                for path_id in allowed:
                    counts[path_id] = 7
                selected = _select(
                    counts, allowed, 2 * len(allowed), start_offset=5
                )

                self.assertEqual(len(selected), 2 * len(allowed))
                self.assertEqual(set(selected), set(allowed))

    def test_nonempty_path_is_still_selectable(self):
        config = qos_config()
        allowed = client_category_paths("LL", config)
        counts = [0] * config.path_count
        for path_id in allowed:
            counts[path_id] = 9
        target = allowed[7]
        counts[target] = 1
        before = counts.copy()

        self.assertEqual(_select(counts, allowed, 1, start_offset=19), [target])
        self.assertEqual(counts, before)

    def test_one_batch_uses_local_counts_without_changing_report(self):
        allowed = client_category_paths("LL", qos_config())
        counts = [5] * 256
        before = counts.copy()

        selected = _select(counts, allowed, 64, start_offset=3)

        self.assertEqual(
            Counter(selected), Counter({path_id: 2 for path_id in allowed})
        )
        self.assertEqual(counts, before)

    def test_selector_binds_each_configured_io_batch_to_one_path(self):
        allowed = client_category_paths("LL", qos_config())
        counts = [0] * 256
        block_sizes = [0.001] * 17

        selected = _select(
            counts,
            allowed,
            17,
            block_sizes_gb=block_sizes,
            path_binding_batch_size=8,
        )

        self.assertEqual(len(selected), 17)
        self.assertEqual(len(set(selected[:8])), 1)
        self.assertEqual(len(set(selected[8:16])), 1)
        self.assertEqual(len(set(selected[16:])), 1)
        self.assertEqual(counts, [0] * 256)
        with self.assertRaises(ValueError):
            ClientRoutingConfig(
                qos_config=qos_config(),
                disk_bw=40.0,
                path_binding_batch_size=0,
            )

    def test_group_aware_selector_avoids_a_busy_group(self):
        counts = [0] * 256
        for path_id in range(29):
            counts[path_id] = 1
        counts[60] = 1

        self.assertEqual(_select(counts, (28, 60), 1), [60])

    def test_zero_pressure_batch_first_spreads_across_groups(self):
        allowed = client_category_paths("LL", qos_config())
        counts = [0] * 256

        self.assertEqual(_select(counts, allowed, 4), [28, 60, 92, 124])

    def test_selector_is_a_deterministic_four_input_pure_function(self):
        static_config = qos_config()
        routing_config = ClientRoutingConfig(
            qos_config=static_config,
            disk_bw=40.0,
            start_offset=0,
        )
        block_sizes = [1.0, 10.0]
        counts = [0] * 256
        allowed = [28, 60]
        before_sizes = block_sizes.copy()
        before_counts = counts.copy()
        before_allowed = allowed.copy()
        before_static_config = dict(vars(static_config))

        parameters = tuple(
            inspect.signature(client_select_qos_paths).parameters
        )
        first = client_select_qos_paths(
            block_sizes_gb=block_sizes,
            path_io_counts=counts,
            allowed_path_ids=allowed,
            routing_config=routing_config,
        )
        second = client_select_qos_paths(
            block_sizes_gb=block_sizes,
            path_io_counts=counts,
            allowed_path_ids=allowed,
            routing_config=routing_config,
        )

        self.assertEqual(
            parameters,
            (
                "block_sizes_gb",
                "path_io_counts",
                "allowed_path_ids",
                "routing_config",
            ),
        )
        self.assertEqual(first, [60, 28])
        self.assertEqual(second, first)
        self.assertEqual(block_sizes, before_sizes)
        self.assertEqual(counts, before_counts)
        self.assertEqual(allowed, before_allowed)
        self.assertEqual(dict(vars(static_config)), before_static_config)
        self.assertEqual(
            client_select_qos_paths(
                block_sizes_gb=[],
                path_io_counts=counts,
                allowed_path_ids=allowed,
                routing_config=routing_config,
            ),
            [],
        )

    def test_static_group_weights_participate_in_sed(self):
        static_config = qos_config()
        weighted_config = replace(
            static_config,
            group_weights=(1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        counts = [0] * 256
        counts[28] = 1
        counts[60] = 1

        self.assertEqual(
            _select(counts, (28, 60), 1, config=weighted_config),
            [60],
        )

    def test_start_offset_breaks_equal_completion_time_ties(self):
        allowed = client_category_paths("LL", qos_config())
        counts = [1] * 256

        expected = [
            188, 220, 252, 29, 61, 93, 125, 157,
            189, 221, 253, 30, 62, 94, 126, 158,
            190, 222, 254, 31, 63, 95, 127, 159,
            191, 223, 255, 28, 60, 92, 124, 156,
            188, 220, 252,
        ]
        self.assertEqual(_select(counts, allowed, 35, start_offset=5), expected)

    def test_clients_balance_using_new_pressure_reports(self):
        allowed = client_category_paths("LL", qos_config())
        counts = [0] * 256

        for client_id in range(8):
            selected = _select(
                counts, allowed, 8, start_offset=client_id * 7
            )
            for path_id in selected:
                counts[path_id] += 1

        self.assertEqual({counts[path_id] for path_id in allowed}, {2})

    def test_ssd_reports_active_plus_pending_io_for_one_path(self):
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, qos_config()
        )
        flows = [
            BlockIOFlow(
                npu_id=index,
                layer=0,
                block_idx=index,
                disk_id=0,
                total_gb=0.001,
                queue_id=0,
                block_count=1,
                enqueue_time=0.0,
            )
            for index in range(3)
        ]

        scheduler.enqueue_many(flows, 0.0)
        report = scheduler.report_path_io_counts(0.0)

        self.assertEqual(len(report), 256)
        self.assertEqual(report[0], 3)
        self.assertEqual(sum(report), 3)
        self.assertIsNone(scheduler.paths[0].active_flow)
        self.assertEqual(list(scheduler.paths[0].pending), flows)

        scheduler.dispatch(0.0, [])
        report = scheduler.report_path_io_counts(0.0)

        self.assertEqual(report[0], 3)
        self.assertIs(scheduler.paths[0].active_flow, flows[0])
        self.assertEqual(list(scheduler.paths[0].pending), flows[1:])
        self.assertEqual(state.active_flows, [flows[0]])
        self.assertEqual(flows[0].bw, 40.0)

        first_end_time = flows[0].end_time
        self.assertEqual(
            scheduler.complete_ready_flows(first_end_time), [flows[0]]
        )
        scheduler.dispatch(first_end_time, [])
        self.assertEqual(state.active_flows, [flows[1]])
        self.assertIs(scheduler.paths[0].active_flow, flows[1])

    def test_baseline_dispatches_one_full_bandwidth_io_per_npu_rr_turn(self):
        state = DiskState(0)
        scheduler = DiskIOScheduler(state, POLICY_BASELINE_BYPASS, 40.0)
        flows = [
            BlockIOFlow(
                npu_id=npu_id,
                layer=0,
                block_idx=block_idx,
                disk_id=0,
                total_gb=0.004,
                queue_id=-1,
                block_count=1,
                enqueue_time=0.0,
            )
            for block_idx in range(2)
            for npu_id in range(2)
        ]
        scheduler.enqueue_many(flows, 0.0)

        order = []
        current_time = 0.0
        for _ in flows:
            scheduler.dispatch(current_time, [])
            self.assertEqual(len(state.active_flows), 1)
            active = state.active_flows[0]
            self.assertEqual(active.bw, 40.0)
            order.append(active.npu_id)
            current_time = active.end_time
            self.assertEqual(scheduler.complete_ready_flows(current_time), [active])

        self.assertEqual(order, [0, 1, 0, 1])
        self.assertAlmostEqual(current_time, 0.4)
        self.assertEqual(scheduler.max_backend_active_io, 1)

    def test_static_scheduler_packetizes_cir_and_wrr_service_rates(self):
        config = qos_config()
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, config
        )
        flows = [
            BlockIOFlow(
                npu_id=0,
                layer=0,
                block_idx=path_id,
                disk_id=0,
                total_gb=0.001,
                queue_id=path_id,
                block_count=1,
                enqueue_time=0.0,
            )
            for path_id in range(config.path_count)
        ]
        scheduler.enqueue_many(flows, 0.0)
        service_rates = sim._static_qos_service_rates(
            list(scheduler.paths.values()), 40.0, config.group_weights
        )

        self.assertAlmostEqual(sum(service_rates.values()), 40.0)
        for flow in flows:
            self.assertAlmostEqual(
                service_rates[scheduler.paths[flow.queue_id]],
                config.path_cirs[flow.queue_id],
            )
        scheduler.dispatch(0.0, [])
        self.assertEqual(len(state.active_flows), 1)
        self.assertEqual(state.active_flows[0].bw, 40.0)
        self.assertEqual(sum(path.is_active() for path in scheduler.paths.values()), 1)

        single_state = DiskState(1)
        single_scheduler = DiskIOScheduler(
            single_state, POLICY_QOS_STATIC_CIR, 40.0, config
        )
        single_flow = BlockIOFlow(
            npu_id=0,
            layer=0,
            block_idx=0,
            disk_id=1,
            total_gb=0.001,
            queue_id=0,
            block_count=1,
            enqueue_time=0.0,
        )
        single_scheduler.enqueue_many([single_flow], 0.0)
        single_rate = sim._static_qos_service_rates(
            [single_scheduler.paths[0]], 40.0, config.group_weights
        )
        self.assertAlmostEqual(single_rate[single_scheduler.paths[0]], 40.0)
        single_scheduler.dispatch(0.0, [])
        self.assertEqual(single_state.active_flows, [single_flow])
        self.assertEqual(single_flow.bw, 40.0)

    def test_qos_cir_ratio_controls_discrete_io_selection_frequency(self):
        base = qos_config()
        path_cirs = [0.0] * base.path_count
        path_cirs[0], path_cirs[1] = 30.0, 10.0
        config = replace(base, path_cirs=tuple(path_cirs))
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, config
        )
        flows = [
            BlockIOFlow(
                npu_id=path_id,
                layer=0,
                block_idx=index,
                disk_id=0,
                total_gb=0.001,
                queue_id=path_id,
                block_count=1,
                enqueue_time=0.0,
            )
            for index in range(40)
            for path_id in (0, 1)
        ]
        scheduler.enqueue_many(flows, 0.0)

        selected_paths = _dispatch_path_sequence(scheduler, state, 40)

        self.assertEqual(Counter(selected_paths), Counter({0: 30, 1: 10}))
        self.assertEqual(scheduler.max_backend_active_io, 1)

    def test_qos_path_and_group_wrr_control_discrete_selection_frequency(self):
        base = qos_config()
        for weighted_ids, path_weights, group_weights in (
            (
                (0, 1),
                (3.0, 1.0) + (1.0,) * (base.path_count - 2),
                base.group_weights,
            ),
            (
                (0, 32),
                base.path_weights,
                (3.0, 1.0) + (1.0,) * (base.group_count - 2),
            ),
        ):
            with self.subTest(weighted_ids=weighted_ids):
                config = replace(
                    base,
                    path_cirs=(0.0,) * base.path_count,
                    path_weights=path_weights,
                    group_weights=group_weights,
                )
                state = DiskState(0)
                scheduler = DiskIOScheduler(
                    state, POLICY_QOS_STATIC_CIR, 40.0, config
                )
                flows = [
                    BlockIOFlow(
                        npu_id=path_id,
                        layer=0,
                        block_idx=index,
                        disk_id=0,
                        total_gb=0.001,
                        queue_id=path_id,
                        block_count=1,
                        enqueue_time=0.0,
                    )
                    for index in range(40)
                    for path_id in weighted_ids
                ]
                scheduler.enqueue_many(flows, 0.0)

                selected = _dispatch_path_sequence(scheduler, state, 40)

                self.assertEqual(
                    Counter(selected),
                    Counter({weighted_ids[0]: 30, weighted_ids[1]: 10}),
                )
                self.assertEqual(scheduler.max_backend_active_io, 1)

    def test_qos_final_rr_breaks_equal_service_ties(self):
        base = qos_config()
        config = replace(base, path_cirs=(0.0,) * base.path_count)
        state = DiskState(0)
        scheduler = DiskIOScheduler(
            state, POLICY_QOS_STATIC_CIR, 40.0, config
        )
        flows = [
            BlockIOFlow(
                npu_id=path_id,
                layer=0,
                block_idx=index,
                disk_id=0,
                total_gb=0.001,
                queue_id=path_id,
                block_count=1,
                enqueue_time=0.0,
            )
            for index in range(2)
            for path_id in (0, 1)
        ]
        scheduler.enqueue_many(flows, 0.0)

        self.assertEqual(
            _dispatch_path_sequence(scheduler, state, 4), [0, 1, 0, 1]
        )

    def test_routing_comparison_spec_is_a_controlled_two_factor_ablation(self):
        spec = routing_comparison_spec(
            bw_table={(1, 0): (5.0, 1_000.0, 10.0, 0.001)},
            ssu_list=experiment.SSU_LIST,
            n_layers=16,
            seed=42,
        )

        self.assertEqual(spec["ssu_list"], list(experiment.SSU_LIST))
        self.assertEqual(spec["n_layers"], 16)
        self.assertEqual(spec["npu_aggregate_cap_gbps"], 50.0)
        self.assertEqual(
            [
                (
                    row["pressure_read_interval"],
                    row["path_binding_batch_size"],
                )
                for row in spec["routing_variants"]
            ],
            [(0, 1), (8, 1), (8, 8)],
        )
        self.assertNotIn("pressure_read_interval", spec)
        self.assertNotIn("path_binding_batch_size", spec)

    def test_routing_comparison_reuses_workload_and_separates_parameters(self):
        profile = (5.0, 1_000.0, 10.0, 0.017)
        table = {
            (3, 0): profile,
            (3, 1023): profile,
            (81, 0): profile,
            (81, 80_895): profile,
        }
        with patch.object(experiment, "NUM_NPU", 1):
            row = run_routing_comparison_case(
                table,
                num_ssu=1,
                n_layers=1,
                seed=13,
            )

        variants = row["variants"]
        self.assertEqual(
            set(variants),
            {"layer_once_per_io", "refresh8_per_io", "refresh8_bind8"},
        )
        self.assertEqual(variants["layer_once_per_io"]["pressure_reports"], 1)
        self.assertEqual(variants["refresh8_per_io"]["pressure_reports"], 3)
        self.assertEqual(variants["refresh8_bind8"]["pressure_reports"], 3)
        self.assertEqual(
            variants["refresh8_per_io"]["max_path_outstanding_io"], 1
        )
        self.assertEqual(
            variants["refresh8_bind8"]["max_path_outstanding_io"], 8
        )
        for metrics in variants.values():
            self.assertEqual(metrics["backend_dispatches"], 17)
            self.assertEqual(metrics["max_backend_active_io"], 1)
            self.assertIn("comparison_vs_baseline", metrics)

    def test_qos_registers_are_installed_once_per_ssd(self):
        calls = []
        original_init = sim.PathQueue.__init__
        config = qos_config()

        def counted_init(path, path_id, cir, pir, path_weight, group_id):
            calls.append(path_id)
            original_init(path, path_id, cir, pir, path_weight, group_id)

        with patch.object(sim.PathQueue, "__init__", new=counted_init):
            _, summary = simulate_continuous(
                {(1, 0): (5.0, 1_000.0, 10.0, 0.001)},
                policy=POLICY_QOS_STATIC_CIR,
                num_npu=1,
                num_disk=2,
                n_layers=1,
                placement_mode="roundrobin",
                qos_config=config,
                rng=np.random.RandomState(3),
            )

        self.assertEqual(len(calls), 2 * 256)
        self.assertEqual(
            Counter(calls), Counter({path_id: 2 for path_id in range(256)})
        )
        for disk in summary["disk_stats"]:
            for path_id, path_stats in disk["paths"].items():
                self.assertEqual(path_stats["cir_gbps"], config.path_cirs[path_id])
                self.assertEqual(path_stats["pir_gbps"], config.path_pirs[path_id])

    def test_pressure_interval_and_path_binding_are_independent(self):
        table = {(3, 1023): (5.0, 1_000.0, 10.0, 0.017)}
        original_enqueue_many = sim.DiskIOScheduler.enqueue_many

        for read_interval, binding_size, expected_submit_lengths, expected_reports in (
            (4, 1, [4, 4, 4, 4, 1], 5),
            (8, 1, [8, 8, 1], 3),
            (0, 1, [8, 8, 1], 1),
            (8, 8, [8, 8, 1], 3),
        ):
            with self.subTest(
                read_interval=read_interval,
                binding_size=binding_size,
            ):
                submitted_windows = []

                def record_enqueue(scheduler, flows, current_time):
                    flows = tuple(flows)
                    if scheduler.is_qos:
                        submitted_windows.append(
                            tuple(flow.queue_id for flow in flows)
                        )
                    return original_enqueue_many(
                        scheduler, flows, current_time
                    )

                with patch.object(
                    sim.DiskIOScheduler,
                    "enqueue_many",
                    new=record_enqueue,
                ):
                    _, summary = simulate_continuous(
                        table,
                        policy=POLICY_QOS_STATIC_CIR,
                        num_npu=1,
                        num_disk=1,
                        n_layers=1,
                        placement_mode="roundrobin",
                        qos_config=qos_config(),
                        pressure_read_interval=read_interval,
                        path_binding_batch_size=binding_size,
                        rng=np.random.RandomState(13),
                    )

                self.assertEqual(
                    [len(window) for window in submitted_windows],
                    expected_submit_lengths,
                )
                if binding_size == 1:
                    self.assertTrue(
                        any(len(set(window)) > 1 for window in submitted_windows)
                    )
                else:
                    self.assertTrue(
                        all(len(set(window)) == 1 for window in submitted_windows)
                    )
                disk = summary["disk_stats"][0]
                self.assertEqual(disk["pressure_reports"], expected_reports)
                self.assertEqual(disk["blocks_enqueued"], 17)
                self.assertEqual(disk["backend_dispatches"], 17)
                self.assertEqual(disk["max_backend_active_io"], 1)
                self.assertEqual(
                    summary["pressure_read_interval"], read_interval
                )
                self.assertEqual(
                    summary["path_binding_batch_size"], binding_size
                )

    def test_client_submission_uses_seeded_shuffled_npu_rounds(self):
        table = {(3, 1023): (5.0, 1_000.0, 10.0, 0.017)}

        def run(policy, read_interval=8, order_seed=77):
            kwargs = {}
            if policy == POLICY_QOS_STATIC_CIR:
                kwargs = {
                    "qos_config": qos_config(),
                    "pressure_read_interval": read_interval,
                }
            return simulate_continuous(
                table,
                policy=policy,
                num_npu=4,
                num_disk=1,
                n_layers=1,
                placement_mode="roundrobin",
                client_submit_batch_size=8,
                submit_order_seed=order_seed,
                rng=np.random.RandomState(13),
                **kwargs,
            )[1]

        baseline = run(POLICY_BASELINE_BYPASS)
        qos_layer = run(POLICY_QOS_STATIC_CIR, read_interval=0)
        qos_refresh = run(POLICY_QOS_STATIC_CIR, read_interval=8)
        samples = [
            summary["client_submission"]["order_sample"]
            for summary in (baseline, qos_layer, qos_refresh)
        ]

        # 同一 ready_time 内每轮每个 NPU 恰好出现一次，下一轮重新洗牌。
        for sample in samples:
            self.assertEqual(len(sample), 12)
            for round_id in (1, 2, 3):
                rows = [row for row in sample if row["round"] == round_id]
                self.assertEqual({row["npu_id"] for row in rows}, set(range(4)))
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row["io_count"] <= 8 for row in rows))
        self.assertNotEqual(
            [row["npu_id"] for row in samples[0][:4]], list(range(4))
        )

        # Baseline、层级快照和每 8 I/O 刷新共用完全相同的客户端提交顺序。
        order = lambda sample: [
            (row["time_ms"], row["round"], row["npu_id"], row["io_count"])
            for row in sample
        ]
        self.assertEqual(order(samples[0]), order(samples[1]))
        self.assertEqual(order(samples[0]), order(samples[2]))

        different_order_seed = run(POLICY_BASELINE_BYPASS, order_seed=78)
        self.assertNotEqual(
            order(samples[0]),
            order(different_order_seed["client_submission"]["order_sample"]),
        )
        workload = lambda summary: [
            (row["request_id"], row["seq_len_k"], row["nql"])
            for row in sorted(
                summary["request_metrics"], key=lambda row: row["request_id"]
            )
        ]
        self.assertEqual(workload(baseline), workload(different_order_seed))

        self.assertEqual(baseline["disk_stats"][0]["pressure_reports"], 0)
        self.assertEqual(qos_layer["disk_stats"][0]["pressure_reports"], 4)
        self.assertEqual(qos_refresh["disk_stats"][0]["pressure_reports"], 12)
        layer_routes = {
            path_id: stats["activations"]
            for path_id, stats in qos_layer["disk_stats"][0]["paths"].items()
            if stats["activations"]
        }
        refresh_routes = {
            path_id: stats["activations"]
            for path_id, stats in qos_refresh["disk_stats"][0]["paths"].items()
            if stats["activations"]
        }
        self.assertNotEqual(layer_routes, refresh_routes)

    def test_client_submit_interval_advances_ready_time(self):
        _, summary = simulate_continuous(
            {(3, 1023): (5.0, 1_000.0, 10.0, 0.017)},
            policy=POLICY_BASELINE_BYPASS,
            num_npu=2,
            num_disk=1,
            n_layers=1,
            placement_mode="roundrobin",
            client_submit_batch_size=8,
            client_submit_interval_us=2.0,
            submit_order_seed=77,
            rng=np.random.RandomState(13),
        )

        rows = summary["client_submission"]["order_sample"]
        times_by_round = {
            round_id: {row["time_ms"] for row in rows if row["round"] == round_id}
            for round_id in (1, 2, 3)
        }
        self.assertEqual(times_by_round, {1: {0.0}, 2: {0.002}, 3: {0.004}})

    def test_random_placement_does_not_synchronize_all_npus_on_disk_zero(self):
        _, summary = simulate_continuous(
            {(3, 1023): (5.0, 1_000.0, 10.0, 0.017)},
            policy=POLICY_BASELINE_BYPASS,
            num_npu=8,
            num_disk=4,
            n_layers=1,
            placement_mode="random",
            client_submit_batch_size=8,
            submit_order_seed=77,
            rng=np.random.RandomState(13),
        )

        first_round = [
            row
            for row in summary["client_submission"]["order_sample"]
            if row["round"] == 1
        ]
        self.assertEqual({row["npu_id"] for row in first_round}, set(range(8)))
        self.assertGreater(len({row["disk_id"] for row in first_round}), 1)

    def test_group_aware_routing_completes_all_16_layers(self):
        config = qos_config()
        before = dict(vars(config))

        _, summary = simulate_continuous(
            {(1, 0): (5.0, 1_000.0, 10.0, 0.001)},
            policy=POLICY_QOS_STATIC_CIR,
            num_npu=1,
            num_disk=2,
            n_layers=16,
            placement_mode="roundrobin",
            qos_config=config,
            rng=np.random.RandomState(5),
        )

        request = summary["request_metrics"][0]
        self.assertEqual(summary["completed_requests"], 1)
        self.assertEqual(
            set(request["per_layer_kv_actual_dur_ms"]), set(range(16))
        )
        self.assertEqual(
            {key[0] for key in request["qos_route_block_counts"]},
            set(range(16)),
        )
        self.assertEqual(dict(vars(config)), before)

    def test_only_baseline_and_static_qos_complete_end_to_end(self):
        table = {(1, 0): (5.0, 1_000.0, 10.0, 0.001)}
        summaries = {}
        for policy in (POLICY_BASELINE_BYPASS, POLICY_QOS_STATIC_CIR):
            kwargs = (
                {"qos_config": qos_config()}
                if policy == POLICY_QOS_STATIC_CIR
                else {}
            )
            _, summaries[policy] = simulate_continuous(
                table,
                policy=policy,
                num_npu=2,
                num_disk=1,
                n_layers=2,
                placement_mode="roundrobin",
                rng=np.random.RandomState(9),
                **kwargs,
            )

        self.assertIn(POLICY_BASELINE_BYPASS, SUPPORTED_POLICIES)
        self.assertIn(POLICY_QOS_STATIC_CIR, SUPPORTED_POLICIES)
        baseline = summaries[POLICY_BASELINE_BYPASS]
        qos = summaries[POLICY_QOS_STATIC_CIR]
        for summary in summaries.values():
            self.assertEqual(summary["completed_requests"], 2)
            self.assertEqual(summary["total_requests"], 2)
            self.assertGreater(summary["makespan_ms"], 0.0)
            self.assertEqual(
                summary["backend_model"], "one_nonpreemptive_io_per_ssu"
            )
            self.assertEqual(summary["backend_capacity_gbps"], 40.0)
            self.assertEqual(
                summary["disk_stats"][0]["max_backend_active_io"], 1
            )
            self.assertEqual(
                summary["disk_stats"][0]["backend_dispatches"],
                summary["disk_stats"][0]["blocks_enqueued"],
            )
        self.assertEqual(baseline["disk_stats"][0]["queue_count"], 0)
        self.assertEqual(baseline["disk_stats"][0]["pressure_reports"], 0)
        self.assertEqual(qos["disk_stats"][0]["queue_count"], 256)
        used_paths = {
            path_id
            for path_id, path_stats in qos["disk_stats"][0]["paths"].items()
            if path_stats["activations"] > 0
        }
        self.assertTrue(used_paths)
        self.assertLessEqual(
            used_paths, set(client_category_paths("SS", qos_config()))
        )
        self.assertEqual(
            baseline["disk_stats"][0]["blocks_enqueued"],
            qos["disk_stats"][0]["blocks_enqueued"],
        )
        self.assertNotIn("qos_client_path_leasing", qos)
        self.assertFalse(hasattr(sim, "ClientPathLeaseDispatcher"))
        with self.assertRaises(ValueError):
            simulate_continuous(table, policy="old_policy")
        with self.assertRaises(ValueError):
            simulate_continuous(
                table,
                policy=POLICY_BASELINE_BYPASS,
                pressure_read_interval=-1,
            )
        with self.assertRaises(ValueError):
            simulate_continuous(
                table,
                policy=POLICY_BASELINE_BYPASS,
                path_binding_batch_size=0,
            )
        with self.assertRaises(ValueError):
            simulate_continuous(
                table,
                policy=POLICY_BASELINE_BYPASS,
                client_submit_batch_size=0,
            )
        with self.assertRaises(ValueError):
            simulate_continuous(
                table,
                policy=POLICY_BASELINE_BYPASS,
                client_submit_interval_us=-1.0,
            )
        with self.assertRaises(ValueError):
            simulate_continuous(
                table,
                policy=POLICY_BASELINE_BYPASS,
                submit_order_seed=1.5,
            )

    def test_two_panel_routing_ablation_plot_is_written(self):
        rows = []
        for num_ssu in (40, 56):
            rows.append(
                {
                    "num_ssu": num_ssu,
                    "baseline": {
                        "avg_request_compute_fraction": 0.69,
                    },
                    "variants": {
                        name: {
                            "avg_request_compute_fraction": utilization,
                        }
                        for name, utilization in (
                            ("layer_once_per_io", 0.70),
                            ("refresh8_per_io", 0.71),
                            ("refresh8_bind8", 0.68),
                        )
                    },
                }
            )
        data = {
            "experiment": routing_comparison_spec(
                bw_table={(1, 0): (5.0, 1_000.0, 10.0, 0.001)},
                ssu_list=(40, 56),
                n_layers=16,
                seed=42,
            ),
            "results": rows,
        }

        with TemporaryDirectory() as temporary_dir:
            output = plot_routing_comparison(
                data,
                Path(temporary_dir) / "routing.png",
            )
            self.assertTrue(Path(output).read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
