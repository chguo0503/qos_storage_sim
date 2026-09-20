#!/usr/bin/env python3
"""Read-only monitoring and synchronization for discovered search jobs."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def read(path):
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as stream:
        return json.load(stream)


def compact(row):
    roles = row['role_and_stall']
    stalls = roles['io_stall']['by_kind_card_ms']
    return dict(start_ms=row['start_ms'],end_ms=row['end_ms'],U_percent=row['U_percent'],
        SLO1p5=row['slo'],strict_underload=row['demand']['strict_underload_all_disks'],
        per_disk_max_demand_GiB_s=row['demand']['per_disk_max_GiB_s'],
        per_disk_overload_percent=row['demand']['per_disk_overload_percent'],
        all_npus_active=row['all_npus_active'],both_roles_npus=roles['npus_with_A_and_B_compute'],
        stall_card_ms_by_kind=stalls,
        internal_fraction_of_stall_percent=(100*stalls['internal_L1_to_L7']/sum(stalls.values()) if sum(stalls.values()) else 0.0))


def main():
    remote = read(HERE/'remote_environment.json')
    ssh = f"ssh -S {remote['mux']} -o BatchMode=yes"
    remote_here = remote['scratch']+'/results/'+HERE.name
    for child in ('runs','execution_logs'):
        subprocess.run(['rsync','-az','--exclude=*.tmp','-e',ssh,
                        remote['host']+':'+remote_here+'/'+child+'/',str(HERE/child)+'/'],check=True)
    output = dict(updated_utc=datetime.now(timezone.utc).isoformat(),jobs=[],errors=[])
    labels = sorted(p.parent.name for p in (HERE/'runs').glob('*/command.json'))
    for label in labels:
        path = HERE/'runs'/label
        if not (path/'command.json').exists():
            output['jobs'].append(dict(label=label,status='pending_command'))
            continue
        command = read(path/'command.json')
        entry = dict(label=label,status=command['status'],policy=command['policy'],error=command.get('error'),
                     expected_blocks=command['expected_blocks'])
        if (path/'progress.json').exists():
            entry['progress']=read(path/'progress.json')
        if command['status']=='complete':
            entry['wall_seconds']=command['wall_seconds']
            result=read(path/'result.json.gz')
            entry['windows']=[compact(row) for row in result['analysis']]
            entry['warm']=entry['windows'][0]
            entry['result_hash_verified']=(hashlib.sha256((path/'result.json.gz').read_bytes()).hexdigest()==command['result_sha256'])
            entry['input_hash_verified']=(hashlib.sha256((path/'manifest.json.gz').read_bytes()).hexdigest()==command['manifest_sha256'])
            entry['all_source_checks_passed']=all(command[k] for k in ('core_unchanged','extension_unchanged','original_manifest_unchanged'))
            entry['all_result_checks_passed']=all(command['checks'].values())
            if not all(entry[k] for k in ('result_hash_verified','input_hash_verified','all_source_checks_passed','all_result_checks_passed')):
                output['errors'].append(dict(label=label,error='Completed result verification failed'))
        elif (path/'warm_preview.json').exists():
            entry['warm']=compact(read(path/'warm_preview.json'))
        if command['status'].startswith('failed'):
            output['errors'].append(dict(label=label,error=command.get('error')))
        output['jobs'].append(entry)
    output['completed']=sum(row['status']=='complete' for row in output['jobs'])
    output['total_jobs']=len(labels)
    output['all_complete']=output['completed']==len(labels)
    lines=['# Discovered OD/Once screening jobs','',f"Updated UTC: {output['updated_utc']}",'',
           'Warm window: [2,4) seconds. Bandwidth: GiB/s. Stall: summed NPU milliseconds.',
           '', '| Case | Status | U | SLO×1.5 | Strict underload | A/B cards | Internal stall | Cross-request L0 stall | Internal share |',
           '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for row in output['jobs']:
        w=row.get('warm')
        if not w:
            lines.append(f"| {row['label']} | {row['status']} | — | — | — | — | — | — | — |")
            continue
        lines.append(f"| {row['label']} | {row['status']} | {w['U_percent']:.4f}% | {w['SLO1p5']['percent']:.2f}% | {w['strict_underload']} | {w['both_roles_npus']}/32 | {w['stall_card_ms_by_kind']['internal_L1_to_L7']:.3f} | {w['stall_card_ms_by_kind']['cross_request_L0']:.3f} | {w['internal_fraction_of_stall_percent']:.2f}% |")
    lines += ['', 'A/B coverage counts actual overlapping computation, not only admission. '+
              'Strict underload refers to current admitted-request V/C on every disk at every event interval. '+
              'Completed files are verified against recorded input/result hashes and post-run source checks.',
              '',f"Completed: {output['completed']}/{len(labels)}. Errors: {len(output['errors'])}."]
    for name,content in [('screening_progress.json',json.dumps(output,indent=2)+'\n'),
                         ('screening_progress.md','\n'.join(lines)+'\n')]:
        target=HERE/name;temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(content);temporary.replace(target)
    print(json.dumps(dict(completed=output['completed'],errors=output['errors'],
        jobs=[dict(label=r['label'],status=r['status'],sim_ms=r.get('progress',{}).get('simulation_ms')) for r in output['jobs']])),flush=True)


if __name__=='__main__':
    main()
