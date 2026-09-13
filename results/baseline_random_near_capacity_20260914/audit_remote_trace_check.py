#!/usr/bin/env python3
"""Check the retained physical trace on the remote host, without rendering it."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math

COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
           'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
           'link_start_ms', 'link_end_ms']
BLOCK_GIB = 176 * 1024 / 2**30


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def verify(directory):
    command = read(directory / 'command.json')
    manifest = read(directory / 'manifest.json.gz')
    physical = read(directory / 'physical_service.json')
    path = directory / 'trace.json.gz'
    assert command['status'] == 'complete' and command['strategy'] == 'baseline'
    assert command['trace_window_ms'] == [1800., 4200.]
    assert sha(path) == command['trace_sha256']
    trace = read(path)
    assert trace['schema_version'] == 1 and trace['columns'] == COLUMNS
    assert trace['window_ms'] == [1800., 4200.] and trace['strategy'] == 'baseline'
    assert trace['completed_simulation'] is True and all(value is True for value in trace['checks'].values())
    assert trace['observed_completed_blocks'] == trace['expected_completed_blocks'] == command['expected_blocks'] == physical['expected_blocks'] == physical['observed_completed_blocks']
    assert trace['retained_blocks'] == len(trace['rows']) == command['retained_blocks'] > 0
    source = trace['source']
    assert source['manifest_sha256'] == command['manifest_sha256'] == sha(directory / 'manifest.json.gz')
    assert source['reference_sha256'] == command['output_sha256'] == sha(directory / 'result.json.gz')
    assert source['observer_source_sha256'] == command['runner_sha256']
    assert source['core_source_sha256'] == command['core_source_sha256']
    requests = {row['request_id']: (row['npu_id'], row['load']['ssd_prefix_tokens'] // 128)
                for row in manifest['requests']}
    npu_counts = [0] * manifest['metadata']['num_npu']
    disks = manifest['metadata']['num_ssu']
    keys = set()
    max_ssd_error = max_link_error = 0.0
    for row in trace['rows']:
        assert len(row) == len(COLUMNS)
        rid, npu, layer, block, disk, path_id, size, count, enqueue, ssd_start, ssd_end, link_start, link_end = row
        expected_npu, blocks = requests[rid]
        assert npu == expected_npu and 0 <= layer < manifest['metadata']['n_layers'] and 0 <= block < blocks
        assert disk == (block + npu) % disks and path_id == 0 and size == BLOCK_GIB and count == 1
        assert all(math.isfinite(value) for value in (enqueue, ssd_start, ssd_end, link_start, link_end))
        assert enqueue <= ssd_start + 1e-8 and ssd_start < ssd_end <= link_start + 1e-8 and link_start < link_end
        assert (ssd_start < 4200. and ssd_end > 1800.) or (link_start < 4200. and link_end > 1800.)
        ssd_error = abs(ssd_end - ssd_start - 1000 * size / physical['ssd_gib_per_second'])
        link_error = abs(link_end - link_start - 1000 * size / physical['npu_link_gib_per_second'])
        max_ssd_error = max(max_ssd_error, ssd_error)
        max_link_error = max(max_link_error, link_error)
        assert ssd_error < 1e-6 and link_error < 1e-6
        identity = (rid, layer, block)
        assert identity not in keys
        keys.add(identity)
        npu_counts[npu] += 1
    result = dict(passed=True, verifier_sha256=sha(Path(__file__)),
                  trace_sha256=command['trace_sha256'], trace_file_bytes=path.stat().st_size,
                  source=source, window_ms=trace['window_ms'], columns=COLUMNS,
                  observed_completed_blocks=trace['observed_completed_blocks'],
                  expected_completed_blocks=trace['expected_completed_blocks'],
                  retained_blocks=len(trace['rows']), retained_blocks_by_npu=npu_counts,
                  maximum_ssd_duration_error_ms=max_ssd_error, maximum_link_duration_error_ms=max_link_error,
                  checks=dict(all_saved_rows_checked=True, all_saved_rows_overlap_retention_window=True,
                              no_duplicate_physical_blocks=True, immutable_source_hashes_match=True,
                              all_expected_blocks_observed_in_full_run=True,
                              exact_block_size_and_npu_stripe_placement=True,
                              physical_ssd_and_link_durations_match_capacity=True),
                  scope='All retained rows were checked. The trace is a window subset, not a full-run block log; full-run completed-block conservation is checked separately.')
    target = directory / 'trace_validation.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, retained_blocks=len(trace['rows']), output=str(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
