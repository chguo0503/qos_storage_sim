#!/usr/bin/env python3
"""Read existing Baseline layer logs and quantify phase and exposed waits."""
from __future__ import annotations

from collections import defaultdict
import cmath
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = 2000., 4000.
LABELS = (
    'raw160_one_l16_fixed_seed7', 'raw200_one_l19_fixed_seed7',
    'raw160_three_l20_fixed_seed7', 'raw160_three_l20_mixed_seed7',
    'raw200_three_l20_fixed_seed7', 'raw200_three_l20_mixed_seed7',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt') as stream:
        return json.load(stream)


def overlap(a,b):
    return max(0., min(b,END)-max(a,START))


def quantiles(values):
    values=sorted(values)
    if not values:
        return {'count':0}
    result={'count':len(values), 'mean':math.fsum(values)/len(values)}
    for name,p in [('min',0),('p25',.25),('p50',.5),('p75',.75),('p90',.9),('p95',.95),('p99',.99),('max',1)]:
        pos=(len(values)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
        result[name]=values[lo]+(values[hi]-values[lo])*(pos-lo)
    return result


def modulo_phase(starts,period):
    phases=sorted(t % period for t in starts)
    if not phases:
        return {'count':0}
    gaps=[b-a for a,b in zip(phases,phases[1:])]+[phases[0]+period-phases[-1]]
    resultant=abs(sum(cmath.exp(2j*math.pi*x/period) for x in phases)/len(phases))
    return {'count':len(phases),'period_ms':period,
            'minimal_covering_arc_ms':period-max(gaps),'circular_resultant_length':resultant,
            'definition':'All warm Long compute starts modulo its C; descriptive concentration, not a matched-wave causal comparison.'}


def analyze(label, parent_rows):
    manifest_path=HERE/'inputs'/f'{label}.json.gz'
    paths=list((HERE/'runs'/label/'baseline').glob('*.json.gz'))
    paths=[p for p in paths if '.failure.' not in p.name]
    if len(paths)!=1:
        return {'label':label,'status':'pending' if not paths else 'duplicate','paths':list(map(str,paths))}
    result_path=paths[0]; raw=read(result_path); manifest=read(manifest_path)
    requests={r['request_id']:r for r in manifest['requests']}
    assert raw['strategy']=='baseline' and raw['submit_seed']==7
    assert raw['input_fingerprint']==manifest['input_fingerprint']
    assert all(raw['summary']['invariants'].values())
    reference=parent_rows.get((label,'baseline'))
    if reference:
        assert reference['audit_pass'] and sha(result_path)==reference['result_sha256']
    role_by_card=defaultdict(set)
    long_cs=set()
    for request in requests.values():
        role_by_card[request['npu_id']].add(request['load']['role'])
        if request['load']['role']=='long':long_cs.add(request['load']['per_layer_us']/1000)
    assert len(long_cs)==1
    long_c=next(iter(long_cs))
    fixed=all(len(roles)==1 for roles in role_by_card.values())
    long_cards=sorted(n for n,roles in role_by_card.items() if roles=={'long'}) if fixed else []
    positions={}
    for n in range(32):
        lane=sorted((r for r in requests.values() if r['npu_id']==n),key=lambda r:r['request_id'])
        for pos,r in enumerate(lane):
            assert r['load']['generation']==pos
            positions[r['request_id']]=pos
    compute=[0.]*32; stalls=[0.]*32; active=[0.]*32
    role_compute=defaultdict(float);role_stall=defaultdict(float)
    warm_internal={'short':[],'long':[]}; all_long=[]; waves=defaultdict(dict)
    ledger=[]
    for batch in raw['summary']['microbatch_metrics']:
        assert batch['batch_size']==1
        rid=batch['member_request_ids'][0]; request=requests[rid]; n=batch['npu_id']
        assert n==request['npu_id']
        role=request['load']['role']; C=request['load']['per_layer_us']/1000
        admission,completion=batch['admission_time_ms'],batch['completion_time_ms']
        active[n]+=overlap(admission,completion)
        previous=admission
        for layer in sorted(batch['layer_metrics'],key=lambda x:x['layer']):
            l=layer['layer']; start,end=layer['compute_start_ms'],layer['compute_end_ms']
            release,ready=layer['io_start_time_ms'],layer['io_ready_time_ms']
            wait=start-previous
            assert wait>=-1e-8 and math.isclose(wait,layer['io_barrier_wait_ms'],abs_tol=1e-7)
            assert math.isclose(end-start,C,abs_tol=1e-7)
            cc,ss=overlap(start,end),overlap(previous,start)
            compute[n]+=cc;stalls[n]+=ss;role_compute[role]+=cc;role_stall[role]+=ss
            row={'npu':n,'request_id':rid,'request_position':positions[rid],
                 'logical_layer_index':positions[rid]*8+l,'layer':l,'role':role,
                 'profile':[request['load']['seq_len_k'],request['load']['nql']],
                 'C_ms':C,'io_release_ms':release,'io_ready_ms':ready,
                 'io_lifetime_ms':ready-release,'previous_compute_end_ms':previous,
                 'compute_start_ms':start,'compute_end_ms':end,'exposed_stall_ms':wait,
                 'prefetch_budget_ms':previous-release,'warm_exposed_stall_ms':ss}
            if l>0:
                assert math.isclose(row['prefetch_budget_ms'],C,abs_tol=1e-7)
                assert math.isclose(max(0.,ready-previous),wait,abs_tol=1e-7)
                row['io_lifetime_over_budget']=row['io_lifetime_ms']/row['prefetch_budget_ms']
                if release>=START and end<=END:
                    warm_internal[role].append(row)
            if role=='long':
                all_long.append(row)
                if fixed:waves[row['logical_layer_index']][n]=row
            ledger.append(row)
            previous=end
        assert math.isclose(previous,completion,abs_tol=1e-7)
    assert all(math.isclose(x,2000,abs_tol=1e-7) for x in active)
    assert all(math.isclose(c+s,a,abs_tol=1e-7) for c,s,a in zip(compute,stalls,active))
    U=math.fsum(compute)/64000
    for w in raw['windows']:
        if w['start_ms']==START and w['end_ms']==END:
            assert math.isclose(U,w['mean_npu_utilization'],abs_tol=1e-12)
    if reference:assert math.isclose(U*100,reference['device_utilization_percent'],abs_tol=1e-10)
    wave_rows=[]
    for logical,members in sorted(waves.items()):
        if set(members)!=set(long_cards):continue
        rs=list(members.values())
        if not all(START<=r['io_release_ms']<END and START<=r['compute_start_ms']<END for r in rs):continue
        releases=[r['io_release_ms'] for r in rs]; starts=[r['compute_start_ms'] for r in rs]
        wave_rows.append({'logical_layer_index':logical,'request_position':logical//8,'layer':logical%8,
            'member_count':len(rs),'io_release_min_ms':min(releases),'io_release_max_ms':max(releases),
            'io_release_spread_ms':max(releases)-min(releases),
            'compute_start_min_ms':min(starts),'compute_start_max_ms':max(starts),
            'compute_start_spread_ms':max(starts)-min(starts),
            'long_io_lifetime_min_ms':min(r['io_lifetime_ms'] for r in rs),
            'long_io_lifetime_max_ms':max(r['io_lifetime_ms'] for r in rs),
            'long_exposed_stall_max_ms':max(r['exposed_stall_ms'] for r in rs)})
    per_card_period=[]
    if fixed:
        for n in long_cards:
            rs=sorted((r for r in all_long if r['npu']==n and START<=r['compute_start_ms']<END),key=lambda r:r['logical_layer_index'])
            for a,b in zip(rs,rs[1:]):
                assert b['logical_layer_index']==a['logical_layer_index']+1
                per_card_period.append(abs(b['compute_start_ms']-a['compute_start_ms']-long_c))
    cohorts={}
    for role,rs in warm_internal.items():
        cohorts[role]={'count':len(rs),'fully_hidden_fraction':sum(r['exposed_stall_ms']<1e-8 for r in rs)/len(rs) if rs else None,
            'io_lifetime_ms':quantiles([r['io_lifetime_ms'] for r in rs]),
            'prefetch_budget_ms':quantiles([r['prefetch_budget_ms'] for r in rs]),
            'io_lifetime_over_budget':quantiles([r['io_lifetime_over_budget'] for r in rs]),
            'exposed_stall_ms':quantiles([r['exposed_stall_ms'] for r in rs])}
    example=max(warm_internal['short'],key=lambda r:(r['exposed_stall_ms'],-r['io_release_ms']))
    overlaps=[r for r in all_long if r['io_release_ms']<example['io_ready_ms'] and r['io_ready_ms']>example['io_release_ms']]
    long_reference=max(overlaps,key=lambda r:min(r['io_ready_ms'],example['io_ready_ms'])-max(r['io_release_ms'],example['io_release_ms'])) if overlaps else None
    return {'label':label,'status':'complete','manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
        'result':str(result_path),'result_sha256':sha(result_path),'input_fingerprint':raw['input_fingerprint'],
        'fixed_role_assignment':fixed,'all_32_active':True,'device_U_percent':100*U,
        'compute_npu_ms':math.fsum(compute),'stall_npu_ms':math.fsum(stalls),
        'full_run_capacity_pass_from_parent':reference['full_run_capacity_pass'] if reference else None,
        'roles':{role:{'warm_compute_npu_ms':role_compute[role],'warm_exposed_stall_npu_ms':role_stall[role],
            'warm_L0_exposed_stall_npu_ms':math.fsum(r['warm_exposed_stall_ms'] for r in ledger if r['role']==role and r['layer']==0),
            'warm_L1_7_exposed_stall_npu_ms':math.fsum(r['warm_exposed_stall_ms'] for r in ledger if r['role']==role and r['layer']>0),
            'pooled_admitted_active_U_percent':100*role_compute[role]/(role_compute[role]+role_stall[role])} for role in ('long','short')},
        'fixed_long_card_U_percent':100*sum(compute[n] for n in long_cards)/(len(long_cards)*2000) if fixed else None,
        'matched_long_waves':{'applicable':fixed,'alignment':'request position *8 + layer; retain waves with every Long card release and compute start inside [2000,4000)',
            'wave_count':len(wave_rows),'release_spread_ms':quantiles([r['io_release_spread_ms'] for r in wave_rows]),
            'compute_start_spread_ms':quantiles([r['compute_start_spread_ms'] for r in wave_rows]),
            'per_card_adjacent_layer_period_error_ms':quantiles(per_card_period),'waves':wave_rows},
        'long_modulo_C_concentration':modulo_phase([r['compute_start_ms'] for r in all_long if START<=r['compute_start_ms']<END],long_c),
        'internal_layer_cohorts':cohorts,
        'warm_short_internal_exposed_stall_total_npu_ms':math.fsum(r['warm_exposed_stall_ms'] for r in ledger if r['role']=='short' and r['layer']>0),
        'short_example':example,'simultaneous_long_reference':long_reference,
        'example_selection':'Largest exposed Short internal-layer wait fully contained in warm window, tie earliest release. Long reference maximizes read-lifetime overlap only; not claimed to be the FIFO head/blocker.'}


def main():
    source=HERE/'followup_results.json';source_bytes=source.read_bytes();parent=json.loads(source_bytes)
    parent_rows={(r['label'],r['strategy']):r for r in parent['rows']}
    rows=[analyze(label,parent_rows) for label in LABELS]
    output={'no_simulation_run':True,'script':str(Path(__file__).resolve()),'script_sha256':sha(__file__),
        'parent_analysis_snapshot_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'window_ms':[START,END],'definitions':{
            'internal_layer_cohort':'Layer1..7 only, io_release>=2000 and compute_end<=4000. Full uncensored I/O lifetime and exposed wait; all boundary-overlap waits separately summed.',
            'io_lifetime':'io_ready-io_release, includes queue/service/receive latency; not SSD busy.',
            'prefetch_budget':'previous_compute_end-io_release; equal to preceding same-request layer C under baseline.',
            'exposed_stall':'compute_start-previous_compute_end; matched to io_barrier_wait_ms.',
            'wave_alignment':'Only fixed-role inputs have matched all-Long-card waves by request-position*8+layer; mixed control gets descriptive modulo-C concentration, not false same-wave correspondence.'},
        'cases':rows,'limitations':['Aligned Long releases and simultaneously exposed Short waits support the timing mechanism, but do not identify a physical FIFO head or assign causal blocking to one Long request.',
            'No queue-level counterfactual, policy change, rerun, or new block trace was generated.',
            'Selected seed7 input evidence; replication and whole-population conclusions belong to the parent planned study.']}
    (HERE/'mechanism_evidence.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    md=['# 原始画像固定角色：已有层日志的机制核查','',
        '仅读取已完成Baseline结果，没有新仿真。窗口[2000,4000)ms；同一Long波按“卡内请求位置×8+layer”对齐。只有固定角色配置才把同逻辑层跨所有Long卡的时刻称为一波；每卡混合对照不强行对齐不同画像位置。', '',
        '|case|设备U %|Long暖计算/已接纳active U %|Long暖stall NPU·ms|同波数|Long release spread p50/max ms|Long compute spread p50/max ms|',
        '|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['status']!='complete':md.append(f"|{r['label']}|{r['status']}|—|—|—|—|—|");continue
        w=r['matched_long_waves']; rel=w['release_spread_ms'];comp=w['compute_start_spread_ms']
        fmt=lambda q:f"{q['p50']:.6f}/{q['max']:.6f}" if q['count'] else '不适用'
        md.append(f"|{r['label']}|{r['device_U_percent']:.6f}|{r['roles']['long']['pooled_admitted_active_U_percent']:.6f}|{r['roles']['long']['warm_exposed_stall_npu_ms']:.6f}|{w['wave_count']}|{fmt(rel)}|{fmt(comp)}|")
    md += ['', 'Long内部层的读隐藏证据（同样采用完整暖窗内L1–7）：', '',
        '|case|预算C ms|IO lifetime p95/max ms|IO延迟/预算 max|完全隐藏占比|相邻逻辑层周期误差 max ms|',
        '|---|---:|---|---:|---:|---:|']
    for r in rows:
        if r['status']!='complete':continue
        c=r['internal_layer_cohorts']['long'];life=c['io_lifetime_ms'];budget=c['prefetch_budget_ms']
        period=r['matched_long_waves']['per_card_adjacent_layer_period_error_ms']
        period_text=f"{period['max']:.3g}" if period['count'] else '不适用'
        md.append(f"|{r['label']}|{budget['mean']:.6f}|{life['p95']:.6f}/{life['max']:.6f}|{c['io_lifetime_over_budget']['max']:.6f}|{100*c['fully_hidden_fraction']:.3f}%|{period_text}|")
    md += ['', '顶部Long暖窗总等待包括L0，内部层表只统计L1–7，两者不能互换。混合输入的跨请求Long首层可能只获得前驱Short计算的预取时间；逐层拆分如下。']
    for r in rows:
        if r['status']=='complete' and not r['fixed_role_assignment']:
            x=r['roles']['long']
            md.append(f"- `{r['label']}`：Long暖窗L0等待{x['warm_L0_exposed_stall_npu_ms']:.6f} NPU·ms，L1–7等待{x['warm_L1_7_exposed_stall_npu_ms']:.6f} NPU·ms。")
    md += ['', '短卡内部层统计：仅计L1–7，且IO释放至本层compute_end完整位于暖窗，避免把截断生命周期当完整IO延迟。IO延迟/预算比大于1对应暴露等待；预算按前驱compute_end−本层IO释放计算并逐条核对为C。', '',
        '|case|内部短层数|IO延迟/预算 p50/p95/max|暴露stall p50/p95/max ms|完全隐藏占比|',
        '|---|---:|---|---|---:|']
    for r in rows:
        if r['status']!='complete':continue
        c=r['internal_layer_cohorts']['short'];ratio=c['io_lifetime_over_budget'];stall=c['exposed_stall_ms']
        vals=lambda q:'/'.join(f'{q[k]:.6f}' for k in ('p50','p95','max'))
        md.append(f"|{r['label']}|{c['count']}|{vals(ratio)}|{vals(stall)}|{100*c['fully_hidden_fraction']:.3f}%|")
    md += ['', '每份具体例子（选完整暖窗内最大Short内部层暴露等待；不是人工指定一条Long为blocker）：','']
    for r in rows:
        if r['status']!='complete':continue
        e=r['short_example'];l=r['simultaneous_long_reference'];phase=r['long_modulo_C_concentration']
        md.append(f"- `{r['label']}`：短NPU{e['npu']}、request {e['request_id']}、L{e['layer']}；IO {e['io_release_ms']:.6f}→{e['io_ready_ms']:.6f}ms，预算{e['prefetch_budget_ms']:.6f}ms，前驱计算结束{e['previous_compute_end_ms']:.6f}，实际等待{e['exposed_stall_ms']:.6f}ms。")
        if l:md.append(f"  同时Long参照NPU{l['npu']}、request {l['request_id']}、L{l['layer']}：IO {l['io_release_ms']:.6f}→{l['io_ready_ms']:.6f}，其预算{l['prefetch_budget_ms']:.6f}ms、等待{l['exposed_stall_ms']:.6f}ms。该条只因read lifetime有重叠被选中。")
        md.append(f"  Long计算起点mod C的圆集中度R={phase['circular_resultant_length']:.6f}（1表示集中），覆盖弧{phase['minimal_covering_arc_ms']:.6f}ms / C={phase['period_ms']:.6f}ms；混合对照此项仅作描述，不能替代同逻辑波配对。")
    md += ['', '以上事件可检验“Long相位是否长期集中、其读是否在长C内完成、短层是否超出自身预算”。它们不直接给出物理FIFO队头ID，也不能只凭同时发生证明某条Long造成某条Short等待。IO lifetime含排队、服务和接收过程，不是SSD实际busy。长期统计、完整波记录、例子的原时刻、manifest/result SHA均见[mechanism_evidence.json](mechanism_evidence.json)。']
    (HERE/'mechanism_evidence.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'complete':sum(r['status']=='complete' for r in rows),'planned':len(rows),
        'brief':[{'label':r['label'],'U':r.get('device_U_percent'),'long_waves':r.get('matched_long_waves',{}).get('wave_count')} for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
