#!/usr/bin/env python3
"""Offline phase accounting for the ordered mostly-long periodic pilots."""
import argparse, gzip, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

def read(p):
    p=Path(p); b=p.read_bytes(); return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def span(v): return max(v)-min(v)
def arc(v,period):
    x=sorted(t%period for t in v)
    return period-max([b-a for a,b in zip(x,x[1:])]+[x[0]+period-x[-1]])
def ol(a,b,s,e):return max(0.,min(b,e)-max(a,s))
def stats(v):return dict(min=min(v),max=max(v),mean=statistics.mean(v),spread=span(v))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--result',required=True);ap.add_argument('--out',required=True);ap.add_argument('--end-ms',type=int,default=12000)
    a=ap.parse_args(); m=read(a.manifest);r=read(a.result);spec=m['metadata']['construction_spec'];N=spec['long_cards']
    assert a.end_ms>4000
    requests={x['request_id']:x for x in m['requests']}; batches={x['member_request_ids'][0]:x for x in r['summary']['microbatch_metrics']}
    group=defaultdict(dict)
    for rid,x in requests.items():
        if x['npu_id']<N:group[x['load']['original_cycle']][(x['npu_id'],x['load']['original_cycle_position'])]=batches[rid]
    cycle_length=len(spec['long_cycle']);long_positions=[i for i,p in enumerate(spec['long_cycle']) if p==len(m['metadata']['profiles'])-1]
    long_count=len(long_positions);assert long_count>0 and long_positions==list(range(long_count))
    phase_profiles={pos:requests[group[0][0,pos]['member_request_ids'][0]]['load'] for pos in range(cycle_length)}
    assert all(phase_profiles[pos]['role']==('long' if pos<long_count else 'short') for pos in range(cycle_length))
    period=next(x['load']['per_layer_us']/1000 for x in requests.values() if x['load']['role']=='long')
    cycles=[]
    for c,rows in sorted(group.items()):
        assert len(rows)==N*cycle_length
        heads=[rows[n,0] for n in range(N)]; cs=[b['layer_metrics'][0]['compute_start_ms'] for b in heads]
        io=[b['layer_metrics'][0]['io_start_time_ms'] for b in heads]
        wave_arcs=[arc([rows[n,pos]['layer_metrics'][layer]['compute_start_ms'] for n in range(N)],period) for pos in long_positions for layer in range(1,8)]
        record=dict(cycle=c,first_long_L0_compute_start=stats(cs),first_long_L0_release=stats(io),circular_cover_arc_ms=arc(cs,period),
            matched_internal_long_wave_count=len(wave_arcs),matched_internal_long_waves_arc_ms=stats(wave_arcs),shared_long_group_interval_ms=[max(b['admission_time_ms'] for b in heads),min(rows[n,long_count-1]['completion_time_ms'] for n in range(N))])
        if c+1 in group:
            per_card=[]
            for n in range(N):
                long_stall=math.fsum(l['io_barrier_wait_ms'] for pos in long_positions for l in rows[n,pos]['layer_metrics'] if not(pos==0 and l['layer']==0))
                short_stall=math.fsum(l['io_barrier_wait_ms'] for pos in range(long_count,cycle_length) for l in rows[n,pos]['layer_metrics'])
                next_l0=group[c+1][n,0]['layer_metrics'][0]['io_barrier_wait_ms']
                pure=math.fsum(l['compute_end_ms']-l['compute_start_ms'] for pos in range(cycle_length) for l in rows[n,pos]['layer_metrics'])
                actual=group[c+1][n,0]['layer_metrics'][0]['compute_start_ms']-cs[n]
                assert math.isclose(actual,pure+long_stall+short_stall+next_l0,abs_tol=1e-7)
                stages=[]
                for pos in range(long_count,cycle_length+1):
                    b=rows[n,pos] if pos<cycle_length else group[c+1][n,0]
                    rid=b['member_request_ids'][0];load=requests[rid]['load'];ls=b['layer_metrics'] if pos<cycle_length else b['layer_metrics'][:1]
                    exposed=[dict(layer=l['layer'],io_start_ms=l['io_start_time_ms'],io_ready_ms=l['io_ready_time_ms'],compute_start_ms=l['compute_start_ms'],stall_ms=l['io_barrier_wait_ms']) for l in ls if l['io_barrier_wait_ms']>1e-8]
                    stages.append(dict(stage='next_Long_L0' if pos==cycle_length else f'cycle_position_{pos}',request_id=rid,profile=f"{load['seq_len_k']}:{load['nql']}",
                        per_layer_C_ms=load['per_layer_us']/1000,admission_ms=b['admission_time_ms'],
                        first_layer_io_release_ms=ls[0]['io_start_time_ms'],first_layer_ready_ms=ls[0]['io_ready_time_ms'],first_layer_compute_start_ms=ls[0]['compute_start_ms'],
                        L0_stall_ms=ls[0]['io_barrier_wait_ms'],internal_stall_ms=math.fsum(l['io_barrier_wait_ms'] for l in ls[1:]),first_exposed_stall=exposed[0] if exposed else None))
                per_card.append(dict(npu=n,long_stall_after_first_compute_ms=long_stall,short_stall_ms=short_stall,next_long_L0_stall_ms=next_l0,
                    pure_C_ms=pure,next_first_compute_delta_ms=actual,total_added_stall_ms=long_stall+short_stall+next_l0,short_and_return_stages=stages))
            record['return_cycle_account']=dict(per_card=per_card,**{k:stats([x[k] for x in per_card]) for k in ('long_stall_after_first_compute_ms','short_stall_ms','next_long_L0_stall_ms','total_added_stall_ms')})
            stage_summary=[]
            for j in range(cycle_length-long_count+1):
                ss=[x['short_and_return_stages'][j] for x in per_card]
                stage_summary.append(dict(stage=ss[0]['stage'],profile=ss[0]['profile'],per_layer_C_ms=ss[0]['per_layer_C_ms'],
                    L0_stall_ms=stats([x['L0_stall_ms'] for x in ss]),internal_stall_ms=stats([x['internal_stall_ms'] for x in ss]),
                    cards_with_exposed_stall=sum(x['first_exposed_stall'] is not None for x in ss)))
            record['return_cycle_account']['stages']=stage_summary
        cycles.append(record)
    windows=[]
    for start,end in [(0,2000),(2000,a.end_ms),(4000,a.end_ms)]+[(s,min(s+2000,a.end_ms)) for s in range(2000,a.end_ms,2000)]:
        role={q:dict(compute_ms=0.,active_ms=0.,L0_stall_ms=0.,internal_stall_ms=0.) for q in ('short','long')}
        for rid,b in batches.items():
            z=role[requests[rid]['load']['role']];z['active_ms']+=ol(b['admission_time_ms'],b['completion_time_ms'],start,end);prev=b['admission_time_ms']
            for l in b['layer_metrics']:
                z['compute_ms']+=ol(l['compute_start_ms'],l['compute_end_ms'],start,end)
                z['L0_stall_ms' if l['layer']==0 else 'internal_stall_ms']+=ol(prev,l['compute_start_ms'],start,end);prev=l['compute_end_ms']
        for z in role.values():
            assert math.isclose(z['compute_ms']+z['L0_stall_ms']+z['internal_stall_ms'],z['active_ms'],abs_tol=1e-6)
            z['conditional_U_percent']=100*z['compute_ms']/z['active_ms'] if z['active_ms'] else None;z['mean_active_cards']=z['active_ms']/(end-start)
        windows.append(dict(start_ms=start,end_ms=end,U_percent=100*sum(z['compute_ms'] for z in role.values())/(32*(end-start)),roles=role))
    # Count all globally admitted Long requests inside each shared cohort interval.
    events=defaultdict(int)
    for rid,b in batches.items():
        if requests[rid]['load']['role']=='long':events[b['admission_time_ms']]+=1;events[b['completion_time_ms']]-=1
    times=sorted(events);active=0;segments=[]
    for i,t in enumerate(times[:-1]):active+=events[t];segments.append((t,times[i+1],active))
    for rec in cycles:
        s,e=rec['shared_long_group_interval_ms'];vals=[(ol(a,b,s,e),n) for a,b,n in segments if ol(a,b,s,e)>0]
        rec['global_long_count_during_shared_interval']=dict(min=min(n for dt,n in vals),max=max(n for dt,n in vals),mean=math.fsum(dt*n for dt,n in vals)/(e-s)) if e>s else None
    group_stalls=[];initial_stalls=[]
    for rid,x in requests.items():
        if x['npu_id']>=N:continue
        for l in batches[rid]['layer_metrics']:
            initial=x['load']['original_cycle']==x['load']['original_cycle_position']==l['layer']==0
            value=l['io_barrier_wait_ms']
            if initial:initial_stalls.append(value)
            elif value>1e-8:group_stalls.append(dict(npu=x['npu_id'],request_id=rid,cycle=x['load']['original_cycle'],position=x['load']['original_cycle_position'],
                profile=f"{x['load']['seq_len_k']}:{x['load']['nql']}",layer=l['layer'],stall_ms=value,compute_start_ms=l['compute_start_ms']))
    group_stalls.sort(key=lambda x:x['compute_start_ms'])
    out=dict(manifest=a.manifest,result=a.result,source_sha256={str(p):sha(p) for p in (Path(__file__),Path(a.manifest),Path(a.result))},
        long_group_size=N,long_requests_per_cycle=long_count,cycle_profiles=[f"{phase_profiles[pos]['seq_len_k']}:{phase_profiles[pos]['nql']}" for pos in range(cycle_length)],C_long_ms=period,cycles=cycles,windows=windows,
        long_group_full_run=dict(initial_L0_stall_ms=stats(initial_stalls),after_initial_L0_exposed_stall_layer_count=len(group_stalls),
            after_initial_L0_exposed_stall_ms=math.fsum(x['stall_ms'] for x in group_stalls),first_noninitial_exposed_stall=group_stalls[0] if group_stalls else None,
            excludes_only_first_request_first_layer_of_each_long_group_card=True,positive_stall_tolerance_ms=1e-8),
        limits=['Circular arc = C_L minus largest cyclic phase gap; same indexed logical waves across the designated long group, not a global barrier.',
          'The shared group Long interval does not imply that only this group has Long requests; global counts are separately integrated.',
          'IO start-to-ready is queue plus SSD plus link lifetime. This analysis has no block-service trace and does not assign a FIFO predecessor.',
          'Short-segment waits and subsequent Long L0 waits change relative phases; their observed association with higher U is not an isolated causal intervention.'])
    target=Path(a.out);target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    short_names=', '.join(out['cycle_profiles'][long_count:])
    lines=['# mostly/bridge 候选的长组回归相位（离线核验）','',f'长组 {N} 张卡；Long 每层 C = {period:.9f} ms。每轮依次执行 {long_count} 条 Long，随后逐条执行 {short_names}。','',
      '| Cycle | 首 Long L0 C-start 最早–最晚(ms) | 线性跨度(ms) | 周期相位覆盖弧(ms) | 共同长段的全局 Long 数 min–max |','|---:|---:|---:|---:|---:|']
    for c in cycles:
        t=c['first_long_L0_compute_start'];g=c['global_long_count_during_shared_interval'];lines.append(f"| {c['cycle']} | {t['min']:.3f}–{t['max']:.3f} | {t['spread']:.6f} | {c['circular_cover_arc_ms']:.6f} | {g['min']}–{g['max']} |")
    lines+=['','| 窗口(s) | Fleet U(%) | Short C/占用(%) | Long C/占用(%) | 平均 Long 卡数 |','|---|---:|---:|---:|---:|']
    for w in windows:lines.append(f"| {w['start_ms']/1000:g}–{w['end_ms']/1000:g} | {w['U_percent']:.6f} | {w['roles']['short']['conditional_U_percent']:.6f} | {w['roles']['long']['conditional_U_percent']:.6f} | {w['roles']['long']['mean_active_cards']:.6f} |")
    ret=cycles[0]['return_cycle_account'];lines+=['',f"第一次返回时，长组每卡相同 pure C；其间额外等待的卡间差异来自：{long_count}Long 段（不含起始 L0）{ret['long_stall_after_first_compute_ms']['min']:.6f}–{ret['long_stall_after_first_compute_ms']['max']:.6f} ms，Short 段 {ret['short_stall_ms']['min']:.6f}–{ret['short_stall_ms']['max']:.6f} ms，下一 Long L0 接纳后暴露等待 {ret['next_long_L0_stall_ms']['min']:.6f}–{ret['next_long_L0_stall_ms']['max']:.6f} ms。上述逐卡时间账已严格相加复核。",'',*['- '+s for s in out['limits']]]
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(out=str(target),cycles=len(cycles),windows=len(windows))))
if __name__=='__main__':main()
