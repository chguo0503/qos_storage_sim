#!/usr/bin/env python3
"""Replay an immutable saved case in a fresh directory, optionally retaining IO trace."""
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
import sys

import experiment as base


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    source=args.case.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(f'Use a fresh output directory: {output}')
    command=base.read_json(source/'command.json')
    previous=base.read_json(source/'result.json.gz')
    assert command['status']=='complete'
    assert base.sha(source/'result.json.gz')==command['output_sha256']
    assert base.sha(source/'manifest.json.gz')==command['manifest_sha256']
    assert base.source_hashes()==command['core_source_sha256']
    assert base.sha(base.__file__)==command['runner_sha256']
    assert base.sha(base.ROOT/'run_baseline_npu32_stress.py')==previous['stress_runner_sha256']
    requests,metadata=base.load_manifest(source/'manifest.json.gz')
    output.mkdir(parents=True)
    provenance=dict(source_case=str(source),source_manifest_sha256=command['manifest_sha256'],
        source_result_sha256=command['output_sha256'],replay_script_sha256=base.sha(__file__),
        runner_sha256=base.sha(base.__file__),argv=sys.argv,
        note='Only output base and optional trace retention changed; original input and simulator preserved.')
    base.write_json(output/'replay_provenance.json',provenance)
    # execute() reads HERE only for output location. Its frozen code and ROOT,
    # input metadata, physical parameters and event stream remain unchanged.
    base.HERE=output
    base.execute(Namespace(strategy=command['strategy'],trace=args.trace),requests,metadata,source/'manifest.json.gz')
    new=output/'runs'/metadata['label']/command['strategy']
    actual=base.read_json(new/'result.json.gz')
    physics=base.read_json(new/'physical_service.json')
    old_physics=base.read_json(source/'physical_service.json')
    checks={
        'input_fingerprint_equal':actual['input_fingerprint']==previous['input_fingerprint'],
        'summary_exactly_equal':actual['summary']==previous['summary'],
        'windows_exactly_equal':actual['windows']==previous['windows'],
        'physical_service_exactly_equal':physics==old_physics,
        'metadata_exactly_equal':actual['metadata']==previous['metadata'],
        'core_sources_equal':actual['core_and_policy_sha256']==previous['core_and_policy_sha256'],
        'manifest_bytes_equal':base.sha(new/'manifest.json.gz')==command['manifest_sha256'],
    }
    base.write_json(output/'replay_checks.json',dict(checks=checks,passed=all(checks.values()),
        replay_case=str(new),note='Compare all saved summary/layer/window/physical-service fields, not unrecorded per-block timestamps.'))
    assert all(checks.values()),checks
    print(json.dumps(dict(passed=True,replay_case=str(new))))


if __name__=='__main__':main()
