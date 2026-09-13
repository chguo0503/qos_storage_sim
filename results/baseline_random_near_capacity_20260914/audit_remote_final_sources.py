#!/usr/bin/env python3
"""Final union audit after all 54 authorized remote queue jobs have completed."""
from datetime import datetime, timezone
import json

from audit_remote_sync import HERE, ROOT, read, sha


def main():
    entries = [('audit_remote_queue_plan.json', 'audit_remote_queue_status.json', 'audit_remote_acceptance.json')]
    for scope in ('once', 'constructed', 'context_scale', 'load110_confirmation',
                  'load105_confirmation', 'context384_confirmation', 'low_b_short', 'heterogeneity',
                  'context384_once_confirmation'):
        entries.append((f'audit_remote_{scope}_plan.json', f'audit_remote_{scope}_status.json',
                        f'audit_remote_{scope}_acceptance.json'))
    frozen = {}; evidence = []; identities = set()
    for plan_name, status_name, acceptance_name in entries:
        plan = read(HERE / plan_name); status = read(HERE / status_name); accepted = read(HERE / acceptance_name)
        assert accepted['passed'] and status['source_hashes_verified_before'] and status['source_hashes_verified_after']
        assert status['ended_utc'] and status['running'] == status['failed'] == 0
        assert status['complete'] == len(plan['jobs']) and status['plan_sha256'] == sha(HERE / plan_name)
        for job in plan['jobs']:
            assert status['jobs'][job['label']]['status'] == 'complete'
            identity = (job['label'], job['strategy'])
            assert identity not in identities
            identities.add(identity)
        for filename, digest in plan['frozen_source_sha256'].items():
            assert filename not in frozen or frozen[filename] == digest
            frozen[filename] = digest
        evidence.append(dict(plan=plan_name, plan_sha256=sha(HERE / plan_name),
            status=status_name, status_sha256=sha(HERE / status_name),
            acceptance=acceptance_name, acceptance_sha256=sha(HERE / acceptance_name),
            jobs=len(plan['jobs']), source_files=len(plan['frozen_source_sha256'])))
    assert len(identities) == 54
    for filename, digest in frozen.items():
        assert sha(ROOT / filename) == digest, filename
    concurrency = read(HERE / 'audit_remote_concurrency_check.json')
    assert concurrency['passed'] and concurrency['job_count'] == 54 and concurrency['peak_parallel_jobs'] <= 6
    report = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
        queue_count=len(entries), unique_queue_jobs=len(identities), source_file_count=len(frozen),
        sources_same_in_every_overlapping_freeze=True, all_local_frozen_sources_unchanged=True,
        each_remote_queue_verified_its_sources_before_and_after=True,
        frozen_source_sha256=frozen, evidence=evidence,
        concurrency_check_sha256=sha(HERE / 'audit_remote_concurrency_check.json'),
        verifier_sha256=sha(HERE / 'audit_remote_final_sources.py'),
        scope='54 unique queued (label,strategy) runs. Earlier duplicate main512 cross-host validation is separate. Scientific result/trace validation is recorded in the bound per-queue acceptance reports. Run this before project source cleanup; use the frozen root-source archive for historical reproduction afterwards.')
    output = HERE / 'audit_remote_final_sources.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, queues=len(entries), jobs=len(identities), sources=len(frozen))))


if __name__ == '__main__':
    main()
