#!/usr/bin/env python3
"""One-shot passive block trace of a frozen Ordered B/Once simulation prefix.

No simulator file is edited.  The completion observer copies immutable flow
fields.  The normal COMPUTE_DONE callback runs first, then the first callback
at or after 4200ms raises PrefixStop.  This is not a completed simulation.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import ast
import copy
import gzip
import hashlib
import inspect
import json
import math
import os
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
SOURCE = STUDY / 'long_validation_7200s'
LABEL = 'sustained16_l176_6l_s64_bridge_48s1l_ordered_seed7'
MANIFEST = SOURCE / 'inputs' / (LABEL + '.json.gz')
LEFT, RIGHT, STOP = 3200., 4000., 4200.
EXTRA_TARGET_KEYS = ((22000046, 4), (26000038, 5), (16, 0))
COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
           'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
           'link_start_ms', 'link_end_ms']
sys.path.insert(0, str(ROOT))
import continuous_batch_sim as native
from run_baseline_npu32_stress import load_manifest, run_case


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with (gzip.open if str(path).endswith('.gz') else open)(temporary, 'wt') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  separators=(',', ':') if str(path).endswith('.gz') else None,
                  indent=None if str(path).endswith('.gz') else 2)
    temporary.replace(path)


def completed_metric_serializer():
    """Reuse the frozen summary's read-only row expressions, without finalizing.

    The first four AST statements create batch rows and request rows.  Add only
    guards skipping unfinished objects; preserve every metric expression and
    full context microbatch indexing.  Compile in a separate globals dictionary,
    so neither the native summary function nor native globals are modified.
    """
    source = inspect.getsource(native._build_summary)
    fn = ast.parse(source).body[0]
    assert isinstance(fn.body[0], ast.Assign) and isinstance(fn.body[1], ast.For)
    assert isinstance(fn.body[2], ast.Assign) and isinstance(fn.body[3], ast.For)
    fn = copy.deepcopy(fn)
    fn.name = '_completed_snapshot'
    fn.body = fn.body[:4]
    for index, name in ((1, 'batch'), (3, 'request')):
        fn.body[index].body.insert(0, ast.parse(
            f'if not math.isfinite({name}.completion_time_ms):\n    continue').body[0])
    fn.body.append(ast.parse('return {"microbatch_metrics": microbatch_metrics, '
                             '"request_metrics": request_metrics}').body[0])
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(native.__dict__)
    exec(compile(module, '<passive-completed-prefix-serialization>', 'exec'), namespace)
    return namespace[fn.name], hashlib.sha256(source.encode()).hexdigest()


class PrefixStop(Exception):
    def __init__(self, context, stop_time, payload):
        self.context, self.stop_time, self.payload = context, stop_time, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['baseline', 'once'], required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    out = HERE / 'traces' / args.strategy
    # mkdir is the one-shot lock: incomplete attempts must never be overwritten.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    command = dict(status='starting', strategy=args.strategy, pid=os.getpid(),
        argv=sys.argv, cwd=os.getcwd(), python=sys.version, executable=sys.executable,
        started_utc=datetime.now(timezone.utc).isoformat(),
        observer_source_sha256=sha(__file__), display_window_ms=[LEFT, RIGHT],
        stop_threshold_ms=STOP, completed_simulation=False,
        scope='Exactly one frozen Ordered simulation prefix; no automatic retry.')
    write(out / 'command.json', command)
    try:
        reference_command_path = SOURCE / 'runs' / LABEL / args.strategy / 'command.json'
        refcmd = read(reference_command_path)
        assert refcmd['status'] == 'complete' and refcmd['returncode'] == 0
        reference_path = Path(refcmd['output'])
        reference = read(reference_path)
        assert reference['strategy'] == args.strategy and reference['submit_seed'] == 7
        assert reference['collector_interval_ms'] == 5.0
        assert reference['policy_config']['assignment'] == 'fixed'
        sources = reference['core_and_policy_sha256']
        assert len(sources) == 29 and sources == refcmd['core_source_sha256']
        assert all(sha(ROOT / name) == value for name, value in sources.items())
        assert sha(ROOT / 'run_baseline_npu32_stress.py') == reference['stress_runner_sha256']
        assert sha(reference_path) == refcmd['output_sha256']
        assert sha(MANIFEST) == refcmd['manifest_sha256']
        requests, metadata = load_manifest(MANIFEST)
        assert metadata['input_fingerprint'] == reference['input_fingerprint']
        reqmap = {r.request_id: r for r in requests}
        refb = {b['batch_id']: b for b in reference['summary']['microbatch_metrics']}
        reflayers = {}
        targets = {}
        for batch in refb.values():
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            for metric in batch['layer_metrics']:
                key = (rid, metric['layer'])
                reflayers[key] = metric
                if metric['io_start_time_ms'] < RIGHT and metric['io_ready_time_ms'] >= LEFT:
                    targets[key] = metric
        display_layer_count = len(targets)
        for key in EXTRA_TARGET_KEYS:
            targets[key] = reflayers[key]
        assert targets and max(x['io_ready_time_ms'] for x in targets.values()) < STOP
        command.update(status='running', manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST),
            reference_result=str(reference_path), reference_sha256=sha(reference_path),
            reference_command=str(reference_command_path), reference_command_sha256=sha(reference_command_path),
            input_fingerprint=metadata['input_fingerprint'], request_count=len(requests),
            core_source_sha256=sources, data_sha256=sha(ROOT / 'data'),
            stress_runner_sha256=reference['stress_runner_sha256'],
            original_runtime_runner_sha256=refcmd['runner_sha256'],
            original_runtime_runner=str(STUDY / 'run_candidates.py'),
            current_runtime_runner_sha256=sha(STUDY / 'run_candidates.py'),
            reference_policy_config=reference['policy_config'], collector_interval_ms=5.,
            display_layer_count=display_layer_count, extra_target_keys=EXTRA_TARGET_KEYS,
            target_layer_count=len(targets),
            target_latest_ready_ms=max(x['io_ready_time_ms'] for x in targets.values()))
        write(out / 'command.json', command)
        print(json.dumps({k: command[k] for k in ('status', 'pid', 'strategy', 'target_layer_count',
                                                 'target_latest_ready_ms')}), flush=True)
        rows, calls = [], 0
        original_register, original_compute = native._register_complete, native._handle_compute_done

        def observe(context, flow):
            nonlocal calls
            if (flow.request_id, flow.layer) in targets:
                rows.append([flow.request_id, flow.npu_id, flow.layer, flow.block_idx,
                    flow.disk_id, flow.queue_id, flow.total_gb, flow.block_count,
                    flow.enqueue_time, flow.ssd_activation_time, flow.link_enqueue_time,
                    flow.link_start_time, flow.link_end_time])
            calls += 1
            return original_register(context, flow)

        def stop_after_original(context, npu_id, payload, generation, current_time_ms):
            result = original_compute(context, npu_id, payload, generation, current_time_ms)
            if current_time_ms >= STOP:
                raise PrefixStop(context, current_time_ms, (npu_id, payload, generation))
            return result

        try:
            with patch.object(native, '_register_complete', observe), \
                 patch.object(native, '_handle_compute_done', stop_after_original):
                run_case(requests, metadata, strategy=args.strategy, assignment='fixed',
                    queue_window_ms=reference['policy_config']['queue_window_ms'],
                    joint_rule=reference['policy_config']['joint_rule'], windows=((LEFT, RIGHT),))
        except PrefixStop as stopped:
            context, stop_time, trigger = stopped.context, stopped.stop_time, stopped.payload
        else:
            raise AssertionError('Expected prefix stop was not reached')

        failures, counts = [], defaultdict(int)
        def check(name, ok, detail=None):
            counts['checks'] += 1
            if not ok:
                failures.append(dict(check=name, detail=detail))

        bylayer = defaultdict(list)
        for row in rows:
            bylayer[(row[0], row[2])].append(row)
        check('target_key_set', set(bylayer) == set(targets))
        layer_audits = []
        for key, ref in targets.items():
            blocks = sorted(bylayer[key], key=lambda row: row[3])
            rid, layer = key
            req = reqmap[rid]
            placement = req.placement[0 if len(req.placement) == 1 else layer]
            count_ok = len(blocks) == len(placement) and [r[3] for r in blocks] == list(range(len(placement)))
            byte_ok = count_ok and all(r[1] == req.npu_id and r[4] == s and r[6] == size and r[7] == 1
                for r, (s, size) in zip(blocks, placement))
            disk_bytes = [math.fsum(r[6] for r in blocks if r[4] == d) for d in range(6)]
            expected_bytes = [math.fsum(v for d0, v in placement if d0 == d) for d in range(6)]
            last_end = max((r[12] for r in blocks), default=-1.)
            ready_ok = last_end == ref['io_ready_time_ms'] == context.requests[rid].io_ready_time_ms[layer]
            check('target_blocks_and_bytes', count_ok and byte_ok and disk_bytes == expected_bytes, list(key))
            check('target_last_hbm_equals_reference_ready', ready_ok, list(key))
            check('target_runtime_ready_flag', bool(context.requests[rid].io_ready[layer]), list(key))
            layer_audits.append(dict(request_id=rid, npu_id=req.npu_id, layer=layer,
                reference_metric=ref, captured_blocks=len(blocks), expected_blocks=len(placement),
                per_ssu_gib=disk_bytes, expected_per_ssu_gib=expected_bytes,
                last_link_end_ms=last_end, passed=count_ok and byte_ok and disk_bytes == expected_bytes and ready_ok))

        computed_layers = []
        for batch in context.microbatches:
            rid = batch.member_request_ids[0]
            check('batch_admission_identity', batch.npu_id == refb[batch.batch_id]['npu_id'] and
                list(batch.member_request_ids) == refb[batch.batch_id]['member_request_ids'] and
                batch.admission_time_ms == refb[batch.batch_id]['admission_time_ms'], batch.batch_id)
            for metric in batch.layer_metrics:
                if metric.layer > batch.compute_done_up_to:
                    continue
                actual = {name: getattr(metric, name) for name in reflayers[(rid, metric.layer)]}
                check('computed_layer_all_fields_exact', actual == reflayers[(rid, metric.layer)], [rid, metric.layer])
                computed_layers.append(dict(request_id=rid, npu_id=batch.npu_id, batch_id=batch.batch_id, **actual))
        for key in targets:
            check('target_compute_finished', any(b.member_request_ids[0] == key[0] and b.compute_done_up_to >= key[1]
                for b in context.microbatches), list(key))
        # Compare every submitted/ready layer time in the prefix, not only plotted layers.
        runtime_io = []
        for rid, req in context.requests.items():
            for layer, release in enumerate(req.io_start_time_ms):
                if not math.isfinite(release):
                    continue
                ref = reflayers[(rid, layer)]
                check('submitted_layer_release_exact', release == ref['io_start_time_ms'], [rid, layer])
                ready = req.io_ready_time_ms[layer] if req.io_ready[layer] else None
                if ready is not None:
                    check('ready_layer_time_exact', ready == ref['io_ready_time_ms'], [rid, layer])
                runtime_io.append([rid, layer, release, ready])

        serializer, serializer_sha = completed_metric_serializer()
        snapshot = serializer(context, requests, stop_time, sum(context.event_counts.values()))
        batch_ids = {b['batch_id'] for b in snapshot['microbatch_metrics']}
        request_ids = {r['request_id'] for r in snapshot['request_metrics']}
        # Stopping after a callback may leave other events with the SAME timestamp.
        # All strictly earlier completions must be present; compare all actual
        # completed rows (including the trigger), without claiming the tie drained.
        expected_batches = [b for b in reference['summary']['microbatch_metrics'] if b['batch_id'] in batch_ids]
        expected_requests = [r for r in reference['summary']['request_metrics'] if r['request_id'] in request_ids]
        check('all_strictly_before_stop_batches_present', all(b['batch_id'] in batch_ids
            for b in refb.values() if b['completion_time_ms'] < stop_time))
        check('all_strictly_before_stop_requests_present', all(r['request_id'] in request_ids
            for r in reference['summary']['request_metrics'] if r['completion_time_ms'] < stop_time))
        check('all_completed_microbatch_rows_exact', snapshot['microbatch_metrics'] == expected_batches)
        check('all_completed_request_rows_exact', snapshot['request_metrics'] == expected_requests)
        completed_keys = {(m['request_id'], m['layer']) for m in computed_layers}
        check('all_strictly_before_stop_computed_layers_present', all(key in completed_keys
            for key, m in reflayers.items() if m['compute_end_ms'] < stop_time))
        check('no_after_stop_computed_layers', all(m['compute_end_ms'] <= stop_time for m in computed_layers))
        check('block_observer_matches_completed_count', calls == context.completed_blocks)
        check('physical_ssd_durations', all(abs(r[10] - r[9] - 1000*r[6]/40) < 1e-7 for r in rows))
        check('physical_link_durations', all(abs(r[12] - r[11] - 1000*r[6]/50) < 1e-7 for r in rows))
        check('physical_stage_order', all(r[8] <= r[9]+1e-8 and r[9] < r[10] <= r[11]+1e-8 and r[11] < r[12] for r in rows))
        check('capture_finished_before_stop', bool(rows) and max(r[12] for r in rows) < stop_time)
        check('all_target_rows_are176KiB', all(r[7] == 1 and r[6] == 176*1024/2**30 for r in rows))
        check('baseline_path0', args.strategy != 'baseline' or all(r[5] == 0 for r in rows))
        check('all_29_sources_unchanged', all(sha(ROOT / k) == v for k, v in sources.items()))
        for name, path, expected in (
            ('stress_runner', ROOT / 'run_baseline_npu32_stress.py', command['stress_runner_sha256']),
            ('data', ROOT / 'data', command['data_sha256']),
            ('manifest', MANIFEST, command['manifest_sha256']),
            ('reference', reference_path, command['reference_sha256']),
            ('reference_command', reference_command_path, command['reference_command_sha256']),
            ('observer', Path(__file__), command['observer_source_sha256']),
            ('runtime_runner', STUDY / 'run_candidates.py', command['current_runtime_runner_sha256'])):
            check(name + '_unchanged', sha(path) == expected)
        check('input_fingerprint_unchanged', native.continuous_batch_input_fingerprint(requests) == command['input_fingerprint'])
        check('executed_input_unchanged', native.continuous_batch_input_fingerprint(
            tuple(context.requests[r.request_id].manifest for r in requests)) == command['input_fingerprint'])

        audit = dict(passed=not failures, status='prefix_complete' if not failures else 'audit_failed',
            strategy=args.strategy, completed_simulation=False, stop_time_ms=stop_time,
            stop_trigger=list(trigger), prefix_completed_requests=context.completed_requests,
            prefix_completed_blocks=context.completed_blocks, captured_blocks=len(rows),
            target_layer_count=len(targets), computed_layer_rows_checked=len(computed_layers),
            completed_microbatch_rows_checked=len(expected_batches), completed_request_rows_checked=len(expected_requests),
            total_checks=counts['checks'], failed_checks=failures, hashes=command,
            capture_rule='Whole layers whose reference io_start<4000 and io_ready>=3200, plus three explicitly recorded same-request comparison keys; all blocks retained, including carry-in/out. Extras do not change the display window.',
            stop_rule='Original COMPUTE_DONE callback executes once; first callback time>=4200 then raises observer-only PrefixStop. No new event.',
            stop_tie_boundary='All t<stop events have executed; same-timestamp events after the trigger may remain. Equality covers every actual completed row; display and target completion precede stop strictly.',
            physical_fields='SSD service=ssd_activation_time to link_enqueue_time; link service=link_start_time to link_end_time; enqueue is not SSD start.',
            equivalence_scope='All prefix completed layer fields, completed microbatch/request rows, submitted/ready times, target bytes and HBM completion; not a full-run summary or makespan comparison.',
            intentionally_uncompared='Wall-clock perf_counter overheads, post-stop states and full-run aggregate statistics.',
            full_summary_serializer_source_sha256=serializer_sha,
            observer_semantics='Read flow fields then call original register once; no policy/input/RNG/event/queue/timestamp mutation.',
            wall_seconds=time.perf_counter()-started)
        write(out / 'trace.json.gz', dict(schema_version=1, columns=COLUMNS, rows=rows,
            strategy=args.strategy, display_window_ms=[LEFT, RIGHT], stop_time_ms=stop_time,
            completed_simulation=False, audit_passed=not failures, source=command, audit=audit))
        write(out / 'layer_audit.json.gz', layer_audits)
        snapshot.update(completed_layer_metrics=computed_layers, io_layer_columns=['request_id','layer','io_start_ms','io_ready_ms'],
            io_layer_rows=runtime_io, stop_time_ms=stop_time, completed_simulation=False,
            event_counts=dict(context.event_counts), completed_blocks=context.completed_blocks,
            completed_requests=context.completed_requests)
        write(out / 'prefix_context.json.gz', snapshot)
        write(out / 'audit.json', audit)
        command.update(status=audit['status'], returncode=0 if audit['passed'] else 1,
            stop_time_ms=stop_time, wall_seconds=time.perf_counter()-started,
            finished_utc=datetime.now(timezone.utc).isoformat(), captured_blocks=len(rows),
            trace_sha256=sha(out / 'trace.json.gz'), audit_sha256=sha(out / 'audit.json'))
        write(out / 'command.json', command)
        print(json.dumps({k: command[k] for k in ('status','strategy','stop_time_ms','wall_seconds','captured_blocks')}), flush=True)
        if failures:
            raise AssertionError(str(failures[:8]))
    except BaseException as exc:
        command.update(status='audit_failed' if command.get('status') == 'audit_failed' else 'failed',
            error_type=type(exc).__name__, error=str(exc), returncode=1,
            finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        write(out / 'command.json', command)
        raise


if __name__ == '__main__':
    main()
