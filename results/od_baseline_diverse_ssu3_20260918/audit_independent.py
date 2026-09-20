#!/usr/bin/env python3
"""Independently recalculate completed formal cases without running simulations.

The 36-case final audit additionally requires both queues to have finished
their source-hash checks and every planned job to have exited successfully.
Existing audit reports are never overwritten.
"""
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime,timezone
from itertools import product
import argparse,gzip,hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from inputs.runners.run_baseline_npu32_stress import load_manifest
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint

def read(p):
    with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-7):assert abs(a-b)<tol,(a,b)
def overlap(a,b,l,r):return max(0.,min(b,r)-max(a,l))
def slo(rows):
    p=sum(r['completion_time_ms']-r['admission_time_ms']<=1.5*r['own_compute_ms']+1e-9 for r in rows)
    return {'count':len(rows),'passed':p,'percent':100*p/len(rows) if rows else None}
def audit_case(path,cmd):
    frozen=HERE/'inputs'/f"{cmd['scenario']}_seed{cmd['seed']}_ring_hash.json.gz"
    assert sha(path/'manifest.json.gz')==cmd['manifest_sha256']==cmd['original_manifest_sha256']==sha(frozen)
    assert sha(path/'result.json.gz')==cmd['result_sha256']==cmd['output_sha256']
    for k,v in cmd['core_source_sha256'].items():assert sha(ROOT/k)==v,k
    for k,v in cmd['extension_source_sha256'].items():assert sha(ROOT/k)==v,k
    requests,meta=load_manifest(path/'manifest.json.gz')
    byid={r.request_id:r for r in requests}
    result=read(path/'result.json.gz');summary=result['summary']
    assert cmd['input_fingerprint']==meta['input_fingerprint']==continuous_batch_input_fingerprint(requests)==result['input_fingerprint']==summary['input_fingerprint']
    assert result['input_placement_fingerprint']==result['execution_placement_fingerprint']
    assert meta['layout']=='block_ring_hash' and meta['order']=='random'
    rows=summary['request_metrics'];batch=summary['microbatch_metrics']
    assert len(rows)==len(requests)==summary['request_count']==cmd['completed_requests']
    assert {r['request_id'] for r in rows}==set(byid)
    assert len(set(r['request_id'] for r in rows))==len(rows)
    expected_counts=[[0]*32 for _ in range(3)]
    rates={}
    for q in requests:
        assert len(q.placement)==1 and q.arrival_time_ms==0
        cnt=Counter(d for d,size in q.placement[0])
        for d,v in cnt.items():expected_counts[d][q.npu_id]+=v*8
        rates[q.request_id]=[math.fsum(size for d,size in q.placement[0] if d==disk)*1e6/q.load['per_layer_us'] for disk in range(3)]
    assert expected_counts==result['completed_blocks_by_ssu_npu']
    blocks=sum(map(sum,expected_counts));assert blocks==cmd['expected_blocks']==cmd['observed_blocks']==summary['completed_blocks']==summary['submitted_blocks']
    for r in rows:
        q=byid[r['request_id']]
        assert r['npu_id']==q.npu_id and r['arrival_time_ms']==q.arrival_time_ms
        close(r['own_compute_ms'],8*q.load['per_layer_us']/1000)
        assert math.isfinite(r['completion_time_ms'])
    for npu in range(32):
        assert [r['request_id'] for r in sorted(rows,key=lambda r:r['admission_time_ms']) if r['npu_id']==npu]==[q.request_id for q in requests if q.npu_id==npu]
    adapter=result['adapter_statistics']
    assert adapter['reorder_calls']==adapter['assignment_count']==0 and not adapter['cir_write_events']
    assert all(summary['invariants'].values()) and all(cmd['checks'].values())
    policy=cmd['policy']; ownership=None
    if policy in ('asu_baseline','od_baseline'):
        paths=[0]*32 if policy=='asu_baseline' else [(n%8)*32+n//8 for n in range(32)]
        assert result['baseline_configuration']['npu_path_ids']==paths
        assert result['observed_path_ids_by_ssu_npu']==[[[p] for p in paths] for _ in range(3)]
        independently_routed={(r['ssu_id'],r['npu_id'],r['path_id']):r['blocks'] for r in adapter['routed_blocks_by_ssu_npu_path']}
        assert independently_routed=={(d,n,paths[n]):expected_counts[d][n] for d in range(3) for n in range(32)}
        for r in rows:
            assert all(v['path_id']==paths[r['npu_id']] for v in r['layer0_path_cirs_at_submit'])
        if policy=='od_baseline':
            for q in result['actual_qos_by_ssu']:
                assert [i for i,c in enumerate(q['path_cirs_gib_s']) if c>0]==sorted(paths)
                assert all(q['path_cirs_gib_s'][p]==1.25 for p in paths)
                assert all(q['path_pirs_gib_s'][p]=='unlimited' for p in paths)
                close(sum(q['path_cirs_gib_s']),40.)
        ownership={'verified_blocks':blocks,'paths_per_disk':len(set(paths)),'per_npu_cir_gib_s':1.25 if policy=='od_baseline' else None,'adapter_submit_counts_equal_observer_completed_counts':True}
    analyses=[]
    for idx,saved in enumerate(result['analysis']):
        left,right=saved['start_ms'],saved['end_ms'];duration=right-left;full=left==0
        cohort=rows if full else [r for r in rows if left<=r['admission_time_ms']<right]
        rec=slo(cohort);assert rec==saved['slo']
        u=100*math.fsum(overlap(layer['compute_start_ms'],layer['compute_end_ms'],left,right) for b in batch for layer in b['layer_metrics'])/(32*duration)
        close(u,saved['U_percent'])
        npus=[100*math.fsum(overlap(layer['compute_start_ms'],layer['compute_end_ms'],left,right) for b in batch if b['npu_id']==n for layer in b['layer_metrics'])/duration for n in range(32)]
        for a,b in zip(npus,saved['per_npu_U_percent']):close(a,b)
        activity=[math.fsum(overlap(r['admission_time_ms'],r['completion_time_ms'],left,right) for r in rows if r['npu_id']==n) for n in range(32)]
        if not full:assert all(abs(v-duration)<1e-7 for v in activity)
        for category,values in saved['slo_by_category'].items():assert slo([r for r in cohort if byid[r['request_id']].load['category']==category])==values
        for profile,values in saved['slo_by_profile'].items():
            seq,miss=map(int,profile.split(':'))
            assert slo([r for r in cohort if byid[r['request_id']].load['seq_len_k']==seq and byid[r['request_id']].load['nql']==miss])==values
        after=sum(r['completion_time_ms']>right for r in cohort);assert after==saved['completed_after_window']
        assert sorted(r['request_id'] for r in cohort)==saved['cohort_request_ids']
        # Independent midpoint integration (no use of the saved demand segments
        # or the runner's incremental demand update routine).
        edges=sorted({left,right}|{max(left,r['admission_time_ms']) for r in rows if r['admission_time_ms']<right and r['completion_time_ms']>left}|{min(right,r['completion_time_ms']) for r in rows if r['admission_time_ms']<right and r['completion_time_ms']>left})
        over=[0.,0.,0.];means=[0.,0.,0.];mins=[math.inf]*3;maxs=[0.]*3;allover=anyover=0.
        for a,z in zip(edges,edges[1:]):
            mid=(a+z)/2
            active=[r for r in rows if r['admission_time_ms']<=mid<r['completion_time_ms']]
            demand=[math.fsum(rates[r['request_id']][d] for r in active) for d in range(3)]
            flags=[v>40+1e-8 for v in demand]
            allover+=(z-a)*all(flags);anyover+=(z-a)*any(flags)
            for d in range(3):
                over[d]+=(z-a)*flags[d];means[d]+=(z-a)*demand[d];mins[d]=min(mins[d],demand[d]);maxs[d]=max(maxs[d],demand[d])
        for d in range(3):
            close(100*over[d]/duration,saved['demand']['per_disk_overload_percent'][d])
            close(means[d]/duration,saved['demand']['per_disk_mean_GiB_s'][d])
            close(mins[d],saved['demand']['per_disk_min_GiB_s'][d])
            close(maxs[d],saved['demand']['per_disk_max_GiB_s'][d])
        close(100*allover/duration,saved['demand']['all_disks_overload_percent'])
        close(100*anyover/duration,saved['demand']['any_disk_overload_percent'])
        if idx==0:
            for d in range(3):
                close(math.fsum(result['warm_ssd_10ms_GiB_s'][d])/200,saved['SSD_GiB_s'][d])
                close(math.fsum(result['warm_ssd_GiB_s_by_ssu_npu'][d]),saved['SSD_GiB_s'][d])
                assert max(result['warm_ssd_10ms_GiB_s'][d])<40+1e-7
        if full:
            for d in range(3):
                stats=summary['disk_stats'][d]
                close(stats['completed_gb']*1000/duration,saved['SSD_GiB_s'][d],1e-5)
                close(stats['utilization']*100,saved['SSD_busy_percent'][d],1e-5)
        analyses.append({'start_ms':left,'end_ms':right,'U_percent':u,'slo':rec,'completed_after_window':after,'per_disk_overload_percent':[100*v/duration for v in over],'all_disks_overload_percent':100*allover/duration,'per_disk_mean_demand_GiB_s':[v/duration for v in means],'per_disk_min_demand_GiB_s':mins,'per_disk_max_demand_GiB_s':maxs,'SSD_GiB_s':saved['SSD_GiB_s'],'SSD_busy_percent':saved['SSD_busy_percent'],'category_slo':saved['slo_by_category']})
    return {'case':path.name,'policy':policy,'scenario':cmd['scenario'],'seed':cmd['seed'],'request_count':len(rows),'completed_blocks':blocks,'input_fingerprint':result['input_fingerprint'],'manifest_sha256':cmd['manifest_sha256'],'result_sha256':cmd['result_sha256'],'path_ownership':ownership,'checks_passed':True,'warm_to_full_slo_delta_pp':analyses[-1]['slo']['percent']-analyses[0]['slo']['percent'],'analysis':analyses}

POLICIES=('asu_baseline','od_baseline','once','static','mild','aggressive')
SCENARIOS=('semi','full')
SEEDS=(7,19,43)


def validate_formal_grid(plan):
    """Reject duplicate, missing, substituted or smoke jobs in the formal grid."""
    assert sorted(plan['policies'])==sorted(POLICIES), 'formal policy list changed'
    assert sorted(plan['scenarios'])==sorted(SCENARIOS), 'formal scenario list changed'
    assert sorted(plan['seeds'])==sorted(SEEDS), 'formal seed list changed'
    jobs=plan['jobs']
    assert len(jobs)==36, 'formal plan must contain exactly 36 jobs'
    identities=[(j['policy'],j['scenario'],j['seed']) for j in jobs]
    assert set(identities)==set(product(POLICIES,SCENARIOS,SEEDS)), 'formal grid is incomplete or substituted'
    assert len(identities)==len(set(identities)), 'duplicate formal grid cell'
    labels=[j['label'] for j in jobs]
    assert len(labels)==len(set(labels)), 'duplicate formal label'
    for j in jobs:
        label=j['label']
        assert Path(label).name==label and label not in ('.','..')
        assert not label.startswith('smoke'), 'smoke label in formal plan'
        assert j['host'] in ('local','remote')
        argv=j['argv']
        assert '--smoke' not in argv and '--pilot' not in argv, 'partial run in formal plan'
        for flag,value in (('--label',label),('--policy',j['policy'])):
            assert argv.count(flag)==1 and argv[argv.index(flag)+1]==value, (label,flag)
        assert argv.count('--manifest')==1
        manifest=argv[argv.index('--manifest')+1]
        expected=str((HERE/'inputs'/f"{j['scenario']}_seed{j['seed']}_ring_hash.json.gz").relative_to(ROOT))
        assert manifest==expected, (label,'unexpected input path')
        assert j['manifest_sha256']==plan['input_sha256'][manifest]
    return {j['label']:j for j in jobs}


def validate_queue_evidence(plan, *, require_finished):
    """Check exact queue-plan membership plus durable post-run source checks."""
    evidence={}
    union=set()
    for host in ('local','remote'):
        queue_path=HERE/f'{host}_plan.json'
        status_path=HERE/f'{host}_plan_status.json'
        queue=read(queue_path);status=read(status_path)
        expected=[j for j in plan['jobs'] if j['host']==host]
        assert queue['jobs']==expected, (host,'queue jobs differ from formal plan')
        assert queue['source_sha256']==plan['source_sha256'], (host,'queue source snapshot differs')
        assert status['plan_sha256']==sha(queue_path), (host,'queue plan hash mismatch')
        assert status['queue_source_sha256']==sha(HERE/'run_queue.py'), (host,'queue runner changed')
        assert status['sources_verified_before'] is True, (host,'queue preflight source audit absent')
        labels={j['label'] for j in expected}
        assert not union.intersection(labels), 'job belongs to both queues'
        union.update(labels)
        assert set(status['jobs']).issubset(labels), (host,'unexpected queue status job')
        for j in expected:
            if j['label'] in status['jobs']:
                state=status['jobs'][j['label']]
                assert state['argv']==j['argv'], (j['label'],'queue command changed')
        if require_finished:
            assert status['complete'] is True, (host,'queue is not complete')
            assert status.get('sources_verified_after') is True, (host,'post-run source verification has not finished')
            assert status.get('ended_utc'), (host,'queue has no final timestamp')
            assert set(status['jobs'])==labels, (host,'queue job missing')
            for label,state in status['jobs'].items():
                assert state['status']=='complete' and state.get('returncode')==0, (label,'queue job failed or unfinished')
                assert state.get('ended_utc'), (label,'job has no final timestamp')
        evidence[host]={'plan_sha256':sha(queue_path),'status_sha256':sha(status_path),
                        'queue_source_sha256':status['queue_source_sha256'],
                        'expected_jobs':len(labels),'recorded_jobs':len(status['jobs']),
                        'complete':status['complete'],
                        'sources_verified_before':status['sources_verified_before'],
                        'sources_verified_after':status.get('sources_verified_after',False),
                        'ended_utc':status.get('ended_utc')}
    assert union=={j['label'] for j in plan['jobs']}
    return evidence


def collect_cases(plan, jobs, require_count):
    """Select formal jobs only; smoke runs cannot substitute for missing cells."""
    cases=[]
    ignored_smoke=[]
    for p in sorted((HERE/'runs').glob('*/command.json')):
        c=read(p);label=p.parent.name
        if label not in jobs:
            assert c.get('smoke') is True, (label,'unplanned non-smoke run')
            ignored_smoke.append(label)
            continue
        j=jobs[label]
        assert not c.get('smoke') and not c.get('pilot'), (label,'partial run substituted for formal job')
        for key in ('policy','scenario','seed','manifest_sha256'):
            assert c[key]==j[key], (label,key,'does not match formal plan')
        assert c['argv']==j['argv'][1:], (label,'actual command differs from plan')
        if c['status']=='complete':
            assert c['completed_simulation'] is True, (label,'simulation not drained')
            cases.append((p.parent,c))
    if require_count is not None:
        assert len(cases)==require_count, ('completed formal case count',len(cases),'expected',require_count)
    if require_count==36:
        assert {p.name for p,_ in cases}==set(jobs), 'missing formal case'
        assert {(c['policy'],c['scenario'],c['seed']) for _,c in cases}==set(product(POLICIES,SCENARIOS,SEEDS))
    return cases,ignored_smoke


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='independent_completed.json',help='New filename inside audit/; existing files are preserved')
    ap.add_argument('--require-count',type=int,help='Use 36 for the complete grid and completed queue/source checks')
    ap.add_argument('--check-grid-only',action='store_true',help='Verify plan/queue/case membership only; do not recalculate metrics or write a report')
    args=ap.parse_args()
    assert Path(args.output).name==args.output and args.output not in ('.','..')
    assert args.require_count is None or 0<=args.require_count<=36
    output=HERE/'audit'/args.output
    if not args.check_grid_only:
        assert not output.exists(), f'Preserve existing audit: {output}'
    plan_path=HERE/'formal_plan.json';plan=read(plan_path)
    jobs=validate_formal_grid(plan)
    for p,digest in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha(ROOT/p)==digest, ('formal source/input changed',p)
    final=args.require_count==36
    queues=validate_queue_evidence(plan,require_finished=final)
    cases,ignored_smoke=collect_cases(plan,jobs,args.require_count)
    if args.check_grid_only:
        print(json.dumps({'formal_jobs':len(jobs),'completed_formal_cases':len(cases),
                          'complete_grid_audit':final,'queue_evidence':queues,
                          'ignored_smoke_runs':ignored_smoke},indent=2))
        return
    results=[]
    for p,c in cases:
        results.append(audit_case(p,c));r=results[-1]
        print(r['case'],[(round(x['U_percent'],4),round(x['slo']['percent'],4),[round(v,3) for v in x['per_disk_overload_percent']]) for x in r['analysis']],flush=True)
    groups=defaultdict(list)
    for r in results:groups[(r['scenario'],r['seed'])].append(r)
    for key,rs in groups.items():
        assert len({r['manifest_sha256'] for r in rs})==len({r['input_fingerprint'] for r in rs})==1
        if final:assert {r['policy'] for r in rs}==set(POLICIES), (key,'policy missing')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'case_count':len(results),
            'all_checks_passed':True,'complete_grid_audit':final,
            'formal_plan_sha256':sha(plan_path),'audit_source_sha256':sha(Path(__file__)),
            'queue_evidence':queues,'ignored_smoke_runs':ignored_smoke,
            'method':'Independent raw request/layer overlap and midpoint per-disk demand integration; source/input/result hashes checked; core submitted ownership counts crosschecked against completed-flow observer counts; no simulation or frozen-file mutation.',
            'pairs_share_byte_identical_inputs':True,'cases':results,
            'limits':['Only completed formal cases present when audit started; complete_grid_audit indicates whether all 36 planned cases and finished queue source audits were required.',
                      'Demand is admitted-request layer volume/compute, not the instantaneous device issue rate.',
                      'Warm SLO uses admission cohort and includes later completions; full population has different membership.',
                      'Only a new derived audit JSON is written; existing audits and frozen input, simulator, runner and metrics remain unchanged.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('SAVED',output,'CASES',len(results))


if __name__=='__main__':main()
