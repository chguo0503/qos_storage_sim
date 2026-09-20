#!/usr/bin/env python3
"""Audit fully drained phase-lock promotions; never substitute pilot statistics."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
CANDIDATES = dict(native_phase_lock_a102='native_grid1_01',
                  native_phase_lock_a101_b0995='native_grid2_00')


def read(path):
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases=[]; rows=[]; pairs=[]; pending=[]
    for candidate,pilot_name in CANDIDATES.items():
        pilot=read(HERE/'pilots'/pilot_name/'command.json')
        pairs_for_candidate=[]
        source=HERE/'inputs'/f'{candidate}.json.gz'
        manifest=read(source)
        assert manifest['input_fingerprint']==pilot['input_fingerprint']
        assert manifest['metadata']['candidate_spec']['promoted_from_pilot']==pilot_name
        for role in ('A','B'):
            profile=manifest['metadata']['profiles'][role]
            matching=[q for q in manifest['requests'] if q['load']['role']==role]
            assert all(q['load']['per_layer_us']==profile['per_layer_compute_us'] for q in matching)
            assert all(abs(q['load']['per_layer_kv_gb']*1e6/q['load']['per_layer_us']-profile['required_bandwidth_gibps'])<1e-10 for q in matching)
        for suffix,policy in (('_od_local','od_baseline'),('_once_remote','once')):
            label=candidate+suffix; path=HERE/'runs'/label
            if not (path/'command.json').exists() or read(path/'command.json')['status']!='complete':
                pending.append(label);continue
            command=read(path/'command.json');result=read(path/'result.json.gz')
            assert result['strategy']==policy
            assert command['core_unchanged'] and command['extension_unchanged'] and command['original_manifest_unchanged']
            assert all(command['checks'].values())
            assert sha(path/'result.json.gz')==command['result_sha256']
            assert sha(path/'manifest.json.gz')==sha(source)==command['manifest_sha256']
            assert result['input_fingerprint']==pilot['input_fingerprint']
            assert len(result['analysis'])==6
            error=abs(result['analysis'][0]['U_percent']-pilot['measurement']['U_percent']) if policy=='od_baseline' else None
            if error is not None:assert error<1e-7
            for index,w in enumerate(result['analysis']):
                duration=w['end_ms']-w['start_ms'];active=math.fsum(w['per_npu_active_ms'])
                roles=w['role_and_stall'];stall=roles['io_stall'];demand=w['demand']
                row=dict(candidate=candidate,policy=policy,window='full' if index==5 else f"{w['start_ms']/1000:g}-{w['end_ms']/1000:g}s",
                    start_ms=w['start_ms'],end_ms=w['end_ms'],U_percent=w['U_percent'],
                    SLO1p5_percent=w['slo']['percent'],SLO1p5_passed=w['slo']['passed'],SLO1p5_count=w['slo']['count'],
                    strict_underload=demand['strict_underload_all_disks'],max_disk_demand_GiB_s=max(demand['per_disk_max_GiB_s']),
                    all_npus_active=w['all_npus_active'],both_roles_npus=roles['npus_with_A_and_B_compute'],
                    internal_stall_card_ms=stall['by_kind_card_ms']['internal_L1_to_L7'],
                    cross_request_L0_stall_card_ms=stall['by_kind_card_ms']['cross_request_L0'],
                    initial_L0_stall_card_ms=stall['by_kind_card_ms']['initial_L0'],
                    exposed_stall_loss_pp=100*stall['total_card_ms']/(32*duration),
                    inactive_time_loss_pp=100*(32*duration-active)/(32*duration))
                assert abs(100-row['U_percent']-row['exposed_stall_loss_pp']-row['inactive_time_loss_pp'])<1e-7
                for disk in range(3):
                    row[f'SSU{disk}_max_demand_GiB_s']=demand['per_disk_max_GiB_s'][disk]
                    row[f'SSU{disk}_overload_percent']=demand['per_disk_overload_percent'][disk]
                    row[f'SSU{disk}_at_or_above_40_percent']=demand['per_disk_at_or_above_capacity_percent'][disk]
                    row[f'SSU{disk}_supply_GiB_s']=w['SSD_GiB_s'][disk]
                rows.append(row)
            full=result['analysis'][-1]
            cases.append(dict(label=label,input_fingerprint=result['input_fingerprint'],
                manifest_sha256=sha(source),result_sha256=command['result_sha256'],
                all_recorded_checks_passed=True,pilot_OD_warm_U_absolute_error=error,
                requests=command['completed_requests'],blocks=command['observed_blocks'],
                full_underload_audit=full['demand'],
                own_compute_definition='8 × candidate per-layer computation; source_ttft_ms is not used',
                request_identity_matches_promoted_pilot=True))
            pairs_for_candidate.append(label)
        pairs.append(dict(candidate=candidate,completed_policies=len(pairs_for_candidate),
                          paired_manifests_byte_identical=True if len(pairs_for_candidate)==2 else None,
                          labels=pairs_for_candidate))
    output=dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=not pending,pending=pending,
        rows=rows,cases=cases,pairs=pairs,
        caveat='Window admission cohorts differ between policies. Full SLO uses the identical fully drained request population; full U includes inactive drain-tail time.')
    lines=['# Fully drained phase-lock comparisons','',f"Completed policies: {len(cases)}/4.",'',
           '| Candidate | Policy | Window | U | SLO×1.5 | Strict underload | A/B cards |',
           '|---|---|---|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['policy']} | {r['window']} | {r['U_percent']:.5f}% | {r['SLO1p5_percent']:.5f}% | {r['strict_underload']} | {r['both_roles_npus']}/32 |")
    lines += ['',output['caveat']]
    buffer=io.StringIO()
    if rows:
        writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,content in [('native_phase_lock_comparison.json',json.dumps(output,indent=2)+'\n'),
                         ('native_phase_lock_comparison.csv',buffer.getvalue()),
                         ('native_phase_lock_comparison.md','\n'.join(lines)+'\n')]:
        target=HERE/name;temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_text(content);temporary.replace(target)
    print(json.dumps(dict(complete=output['complete'],completed=len(cases),pending=pending)),flush=True)


if __name__=='__main__':
    main()
