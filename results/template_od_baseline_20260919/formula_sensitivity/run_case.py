"""Run archived native simulator with only OD routing and static QoS changed.

ASU parity mode invokes the unmodified archived exact-tail adapter. OD layers
use the current policy's frozen path_ids(), and its actual QoS table exported
without changing units. Native FIFO, WFQ, prefetch, partial IO and arrivals stay.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
import argparse, gzip, hashlib, json, math, sys, time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen_source'))
import sim
import continuous_batch_sim as native
from fixed_total_compat import exact_tail_adapter
from run_baseline_npu32_stress import load_manifest
from continuous_prefill_client import routing_strategy_specs, static_qos_config
import od_policy_snapshot as od

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.name.endswith('.gz'):
        with gzip.open(p, 'wt') as f: json.dump(obj, f)
    else: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def metrics(summary, requests):
    loads = {q.request_id:q.load for q in requests}
    rows = []
    for r in summary['request_metrics']:
        load = loads[r['request_id']]
        group = 'A' if load['role']=='L' else 'B'
        ttft = r['completion_time_ms']-r['admission_time_ms']
        rows.append(dict(request_id=r['request_id'],npu_id=r['npu_id'],group=group,
                         admission_ms=r['admission_time_ms'],completion_ms=r['completion_time_ms'],
                         compute_ms=r['own_compute_ms'],ttft_ms=ttft,ttft_ratio=ttft/r['own_compute_ms']))
    cohorts = {}
    for key,sample in [('all_requests',rows),('warm_admissions',[r for r in rows if 2000<=r['admission_ms']<4000])]:
        cohorts[key] = {}
        for group in ['all','A','B']:
            selected = [r for r in sample if group=='all' or r['group']==group]
            counts = {f'slo{alpha:g}_passed':sum(r['ttft_ratio']<=alpha+1e-10 for r in selected) for alpha in [1,1.5]}
            cohorts[key][group] = dict(count=len(selected), **counts,
                **{k.replace('_passed','_percent'):100*v/len(selected) if selected else None for k,v in counts.items()})
    clip = lambda a,b:max(0.,min(b,4000.)-max(a,2000.))
    compute = [0.]*8
    group_compute = Counter(); group_active = Counter()
    for b in summary['microbatch_metrics']:
        c=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for l in b['layer_metrics'])
        compute[b['npu_id']]+=c
        group='A' if loads[b['member_request_ids'][0]]['role']=='L' else 'B'
        group_compute[group]+=c;group_active[group]+=clip(b['admission_time_ms'],b['completion_time_ms'])
    return dict(warm_U_percent=100*sum(compute)/16000,per_npu_warm_U_percent=[c/20 for c in compute],
                warm_group_U_percent={g:100*group_compute[g]/group_active[g] for g in group_active},
                full_U_percent=100*summary['fleet_npu_compute_utilization'],
                makespan_ms=summary['makespan_ms'],cohorts=cohorts), rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True)
    ap.add_argument('--policy',choices=['asu_baseline','od_baseline'],required=True)
    a=ap.parse_args();out=ROOT/'runs'/f'{a.case}_{a.policy}'
    if (out/'metrics.json').exists(): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'inputs'/f'{a.case}.json.gz'
    requests,meta=load_manifest(manifest)
    source_before={p.name:sha(p) for p in (ROOT/'frozen_source').glob('*.py')}
    fp=native.continuous_batch_input_fingerprint(requests)
    q0=static_qos_config();scale=meta['disk_bw_gib_s']/40
    if a.policy=='asu_baseline':
        qos=replace(q0,path_cirs=tuple(v*scale for v in q0.path_cirs),path_pirs=tuple(v*scale for v in q0.path_pirs))
    else:
        values=json.loads((ROOT/'od_qos_configs.json').read_text())[meta['family']]
        qos=replace(q0,**{k:tuple(v) if isinstance(v,list) else v for k,v in values.items()})
    client=next(s for s in routing_strategy_specs() if s.name=='baseline').client_config()
    paths=Counter();layers0=Counter();started=time.time();last=started;lastsim=0.
    print(json.dumps(dict(event='start',case=a.case,policy=a.policy,requests=len(requests))),flush=True)
    with exact_tail_adapter('baseline',collector_interval_ms=5.) as adapter:
        def plan(context,state,now_ms):
            count=len(state.blocks)-len(state.planned_path_ids)
            ids=od.path_ids(count,state.npu_id,context.num_npu)
            state.planned_path_ids.extend(ids)
            for path,n in Counter(ids).items():
                adapter.pending[state.disk_id][path]+=n
                adapter.max_ledger_count=max(adapter.max_ledger_count,adapter.pending[state.disk_id][path])
            adapter.reserved_blocks+=len(ids);adapter.routing_calls+=1
        original_complete=native._register_complete
        def completed(context,flow):
            nonlocal last,lastsim
            expected=od.od_npu_path_ids(8)[flow.npu_id] if a.policy=='od_baseline' else 0
            assert flow.queue_id==expected,(flow.npu_id,flow.queue_id,expected)
            paths[(flow.npu_id,flow.disk_id,flow.queue_id)]+=flow.block_count
            if flow.layer==0:layers0[(flow.npu_id,flow.queue_id)]+=flow.block_count
            now=time.time()
            if now-last>30:
                last=now;lastsim=context.current_time_ms
                print(json.dumps(dict(event='progress',simulation_ms=lastsim,completed=context.completed_requests)),flush=True)
            return original_complete(context,flow)
        with ExitStack() as stack:
            if a.policy=='od_baseline':stack.enter_context(patch.object(native,'_plan_paths',plan))
            stack.enter_context(patch.object(native,'_register_complete',completed))
            summary=native.simulate_continuous_batch(requests,num_npu=8,num_ssu=1,n_layers=8,batch_size=1,
                policy=sim.POLICY_QOS_STATIC_CIR,qos_config=qos,client_io_config=client,
                cross_request_layer0_prefetch=True,pressure_ttl_ms=5.,disk_bw_gbps=meta['disk_bw_gib_s'],
                npu_bw_gbps=meta['npu_bw_gib_s'],submit_order_seed=meta['seed'],control=None)
        stats=adapter.statistics()
        actual=adapter.context.qos_configs_by_ssu[0]
        assert actual==qos
    assert all(summary['invariants'].values())
    assert summary['request_count']==len(requests)==len(summary['request_metrics'])
    assert native.continuous_batch_input_fingerprint(requests)==fp
    assert source_before=={p.name:sha(p) for p in (ROOT/'frozen_source').glob('*.py')}
    assert sum(paths.values())==summary['completed_blocks']==summary['submitted_blocks']
    assert not stats['cir_write_events']
    m,rows=metrics(summary,requests)
    if a.policy=='asu_baseline':
        ref=json.loads(gzip.decompress((ROOT/'references'/f'{a.case}_asu_summary.json.gz').read_bytes()))
        fields=['request_metrics','microbatch_metrics','makespan_ms','input_fingerprint','events_processed',
                'completed_blocks','completed_read_gb','fleet_npu_compute_utilization','disk_stats']
        parity={k:summary[k]==ref[k] for k in fields}
        save(out/'parity.json',parity)
        assert all(parity.values()),parity
    m.update(case=a.case,policy=a.policy,family=meta['family'],group=meta['group'],seed=meta['seed'],
             input_fingerprint=fp,manifest_sha256=sha(manifest),wall_seconds=time.time()-started,
             npu_count=8,ssu_count=1,n_layers=8,disk_bw_gib_s=meta['disk_bw_gib_s'],
             capacity_display=meta['capacity_display'],cdf_cohort='all same completed requests, includes startup',
             ttft_definition='completion-admission; own_compute is base SLO',
             source_sha256=source_before,runner_sha256=sha(Path(__file__)),
             observed_paths=[dict(npu=n,ssu=s,path=p,blocks=c) for (n,s,p),c in sorted(paths.items())],
             observed_layer0_paths=[dict(npu=n,path=p,blocks=c) for (n,p),c in sorted(layers0.items())],
             cir_sum=sum(qos.path_cirs),active_path_cirs={str(p):qos.path_cirs[p] for p in range(256) if qos.path_cirs[p]>0},
             pir_unlimited=all(math.isinf(v) for v in qos.path_pirs),source_unchanged=True,
             invariants=summary['invariants'])
    save(out/'native_summary.json.gz',summary)
    save(out/'cdf_samples.json.gz',rows)
    save(out/'adapter_statistics.json',stats)
    save(out/'metrics.json',m)
    print(json.dumps(dict(event='done',case=a.case,policy=a.policy,U=m['warm_U_percent'],
                         SLO=m['cohorts']['all_requests']['all']['slo1.5_percent'],wall_seconds=m['wall_seconds'])),flush=True)

if __name__=='__main__':main()
