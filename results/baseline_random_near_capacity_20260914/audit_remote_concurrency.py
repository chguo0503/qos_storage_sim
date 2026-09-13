#!/usr/bin/env python3
"""Verify the completed remote queues never exceeded six simultaneous jobs."""
from datetime import datetime, timezone
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    files = {'baseline': 'audit_remote_queue_status.json',
             'once': 'audit_remote_once_status.json',
             'constructed': 'audit_remote_constructed_status.json',
             'context_scale': 'audit_remote_context_scale_status.json',
             'load110_confirmation': 'audit_remote_load110_confirmation_status.json',
             'load105_confirmation': 'audit_remote_load105_confirmation_status.json',
             'context384_confirmation': 'audit_remote_context384_confirmation_status.json',
             'low_b_short': 'audit_remote_low_b_short_status.json',
             'heterogeneity': 'audit_remote_heterogeneity_status.json',
             'context384_once_confirmation': 'audit_remote_context384_once_confirmation_status.json'}
    events, intervals = [], []
    for scope, filename in files.items():
        status = json.loads((HERE / filename).read_text())
        assert status.get('ended_utc') and status['running'] == status['failed'] == 0
        assert status['source_hashes_verified_before'] and status['source_hashes_verified_after']
        for label, job in status['jobs'].items():
            assert job['status'] == 'complete'
            start, end = (datetime.fromisoformat(job[key]) for key in ('started_utc', 'ended_utc'))
            assert start < end
            identity = scope + ':' + label
            events.extend([(start, 1, identity), (end, -1, identity)])
            intervals.append(dict(scope=scope, label=label, start_utc=job['started_utc'], end_utc=job['ended_utc']))
    active, peak, peak_at, peak_jobs = set(), 0, None, []
    validation = json.loads((HERE / 'runtime_validation/remote_main512/command.json').read_text())
    assert datetime.fromisoformat(validation['ended_utc']) <= min(when for when, change, identity in events)
    for when, change, identity in sorted(events):
        if change < 0:
            active.remove(identity)
        else:
            active.add(identity)
        if len(active) > peak:
            peak, peak_at, peak_jobs = len(active), when.isoformat(), sorted(active)
    assert not active and peak <= 6 and len(intervals) == 54
    report = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                  job_count=len(intervals), peak_parallel_jobs=peak, first_peak_utc=peak_at,
                  jobs_at_first_peak=peak_jobs, intervals=intervals,
                  method='Sweep of queue-recorded launch through completion-recognition UTC intervals, ending before starting at exact ties. Completion polling can extend an interval beyond process exit, so the count is conservative.',
                  scope='21 Baseline + 2 exploratory Once + 4 constructed Baseline + 4 context-scale Baseline + 4 predeclared raw load110 confirmation seeds + 4 predeclared raw load105 confirmation seeds + 4 predeclared context384 confirmation seeds + 4 low-B-short exploratory Baseline pilots + 3 heterogeneity Baseline pilots + 4 context384 Once confirmation pairs. The earlier standalone cross-host validation finished before these queues started.')
    output = HERE / 'audit_remote_concurrency_check.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, job_count=len(intervals), peak_parallel_jobs=peak)))


if __name__ == '__main__':
    main()
