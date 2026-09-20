"""Exact final-block support for the original Baseline / 5-ms Once experiment.

The native SSD, link, admission, and prefetch implementations are unchanged.
This bounded adapter retains the shared experiment's periodic collector and
the original Once routing function, supplying actual command sizes instead of
expanding every command to 176 KiB. Other shared/coflow policies are unsupported.
Run separate processes, not threads: instrumentation is process-local.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import continuous_batch_sim as native
import sim
import shared_path_sim_adapter as shared
from policy_logic import hardware_view, layer_once_path_ids, pressure_snapshot
from shared_path_baseline import baseline_path_ids
from shared_ssu_state import PeriodicSSUState
from run_baseline_npu32_stress import window_evidence
from run_shared_path_experiments import summarize_slo

IO_GIB = 176 * 1024 / 2**30


def validate_blocks(requests):
    """Allow a positive, exact-size tail only as the last block of a layer."""
    tail_count = block_count = 0
    for request in requests:
        for layer in request.placement:
            if not layer:
                raise ValueError("This mixed-workload adapter requires nonempty reads")
            for index, (_ssu, size) in enumerate(layer):
                if not math.isfinite(size) or not 0 < size <= IO_GIB:
                    raise ValueError("Command size must be positive and at most 176 KiB")
                if size != IO_GIB:
                    if index != len(layer) - 1:
                        raise ValueError("Only the final block may be smaller than 176 KiB")
                    tail_count += 1
                block_count += 1
            if not math.isclose(math.fsum(size for _, size in layer),
                                request.load['per_layer_kv_gb'], rel_tol=0., abs_tol=1e-12):
                raise ValueError("Manifest bytes differ from the request's per-layer read volume")
    return {'manifest_block_count': block_count, 'partial_final_block_count': tail_count,
            'all_blocks_equal_176kib': tail_count == 0}


@contextmanager
def exact_tail_adapter(strategy='once', collector_interval_ms=5.0,
                       cir_min_interval_ms=100.0):
    if strategy not in ('baseline', 'once'):
        raise ValueError("Exact-tail compatibility is limited to baseline and once")
    adapter = shared.SharedPathAdapter(strategy, collector_interval_ms,
                                       cir_min_interval_ms)
    original_init = native._Context.__init__
    original_pressure = sim.DiskIOScheduler.report_path_pressure_analysis
    original_complete = native._register_complete

    def collect(context, now_ms):
        started = perf_counter()
        def read(ssu_id):
            scheduler = context.disks[ssu_id].scheduler
            scheduler._pressure_cache = None
            snapshot = original_pressure(scheduler, now_ms)
            return snapshot, tuple(path.cir for path in scheduler.paths.values())
        adapter.collector.collect(now_ms, read)
        context.push_event(adapter.collector.next_collection_ms, native.CIR_CONTROL, 0)
        adapter.collector_wall_us += (perf_counter() - started) * 1e6

    def initialize(context, *args, **kwargs):
        validate_blocks(kwargs['requests'])
        original_init(context, *args, **kwargs)
        if context.control is not None or context.npu_dedicated_paths is not None:
            raise ValueError("Original Baseline/Once uses static shared paths without CIR control")
        adapter.context = context
        adapter.pending = [[0] * 256 for _ in range(context.num_ssu)]
        adapter.qos = tuple(hardware_view(q) for q in context.qos_configs_by_ssu)
        adapter.collector = PeriodicSSUState(context.num_ssu, collector_interval_ms,
                                             cir_min_interval_ms)
        collect(context, 0.0)

    def cached_pressure(scheduler, now_ms):
        scheduler.pressure_cache_hits += 1
        return adapter.collector.get(scheduler.state.disk_id, now_ms)

    def plan(context, state, now_ms):
        started = perf_counter()
        first = len(state.planned_path_ids)
        sizes = tuple(size for _index, size in state.blocks[first:])
        count, s = len(sizes), state.disk_id
        if strategy == 'baseline':
            ids = baseline_path_ids(count)
        else:
            observed = pressure_snapshot(adapter.collector.get(s, now_ms))
            ids = layer_once_path_ids(sizes, observed, state.allowed_path_ids,
                                     adapter.qos[s], start_offset=state.start_offset,
                                     disk_bw_gbps=context.disk_bw_gbps)
        state.planned_path_ids.extend(ids)
        # The original ledger counts commands, including one command per tail.
        # Original Once does not consult this ledger to choose paths.
        for path_id, number in Counter(ids).items():
            adapter.pending[s][path_id] += number
            adapter.max_ledger_count = max(adapter.max_ledger_count,
                                           adapter.pending[s][path_id])
        adapter.reserved_blocks += len(ids)
        adapter.routing_calls += 1
        adapter.routing_wall_us += (perf_counter() - started) * 1e6
        if len(adapter.route_examples) < 12:
            adapter.route_examples.append({'time_ms': now_ms,
                'snapshot_time_ms': adapter.collector.sampled_at_ms,
                'request_id': state.request_id, 'layer': state.layer,
                'ssu_id': s, 'io_count': count, 'path_counts': dict(Counter(ids))})

    def completed(context, flow):
        original_complete(context, flow)
        started = perf_counter()
        adapter.pending[flow.disk_id][flow.queue_id] -= flow.block_count
        adapter.acknowledged_blocks += flow.block_count
        value = adapter.pending[flow.disk_id][flow.queue_id]
        adapter.min_ledger_count = min(adapter.min_ledger_count, value)
        if value < 0:
            raise AssertionError("HBM acknowledgement without a routing reservation")
        adapter.ack_ledger_wall_us += (perf_counter() - started) * 1e6

    with patch.object(native._Context, '__init__', initialize), \
         patch.object(sim.DiskIOScheduler, 'report_path_pressure_analysis', cached_pressure), \
         patch.object(native, '_handle_control', collect), \
         patch.object(native, '_plan_paths', plan), \
         patch.object(native, '_register_complete', completed):
        yield adapter


def run_case(requests, metadata, *, strategy='baseline', assignment='pipeline',
             queue_window_ms=1.0, joint_rule='urgent_short',
             windows=((1000, 2000), (2000, 3000))):
    """Same result/window/SLO interface as run_baseline_npu32_stress.run_case."""
    if strategy not in ('baseline', 'once'):
        raise ValueError("Exact-tail compatibility is limited to baseline and once")
    from run_coflow_experiments import run_case as original_reference_run
    before = native.continuous_batch_input_fingerprint(requests)
    started = perf_counter()
    block_audit = validate_blocks(requests)
    if metadata.get('equal_176kib_blocks') is True and not block_audit['all_blocks_equal_176kib']:
        raise ValueError("Metadata falsely claims equal 176 KiB commands")
    with patch.object(shared, 'shared_path_adapter', exact_tail_adapter):
        result = original_reference_run(requests, metadata, strategy=strategy,
                                        assignment=assignment, queue_window_ms=queue_window_ms,
                                        joint_rule=joint_rule)
    result.pop('_source_texts', None)
    summary = result['summary']
    assert all(summary['invariants'].values())
    assert before == native.continuous_batch_input_fingerprint(requests)
    result['strategy'] = strategy
    result['experiment'] = 'baseline_npu32_stress_v1'
    result['windows'] = [window_evidence(summary, requests, start, end) for start, end in windows]
    result['common_window'] = result['windows'][0]
    result['slo'] = summarize_slo(summary, requests, start_ms=windows[0][0], end_ms=windows[0][1])
    result['python_version'] = sys.version
    original_runner = Path(__file__).with_name('run_baseline_npu32_stress.py')
    result['stress_runner_sha256'] = hashlib.sha256(original_runner.read_bytes()).hexdigest()
    result['wall_seconds_total'] = perf_counter() - started
    result['exact_tail_compatibility'] = {
        **block_audit, 'adapter_file': Path(__file__).name,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Original baseline/once only; exact final block sizes; original periodic 5-ms collector',
        'core_modified': False, 'io_size_padding': False,
        'routing_function': 'policy_logic.layer_once_path_ids(actual_block_sizes, ...)'}
    return result
