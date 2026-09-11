#!/usr/bin/env python3
"""Read three Baseline traces; compare phase, role residence and layer waits.

No simulation, no editing of frozen inputs/results. Labels/directories are configurable
for later binding designs. The warm window remains fixed at [2000,4000) ms.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
START,END=2000.0,4000.0
FRONT=list(range(20))
GUARDS=[17,18,19]


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def clip(a,b,start=START,end=END):return max(0.0,min(b,end)-max(a,start))


def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7)


def quantiles(values):
    values=sorted(values)
    if not values:return {'count':0}
    out={'count':len(values),'mean':math.fsum(values)/len(values)}
    for name,p in [('min',0),('p50',.5),('p90',.9),('p95',.95),('p99',.99),('max',1)]:
        x=(len(values)-1)*p;lo=math.floor(x);hi=math.ceil(x)
        out[name]=values[lo]+(values[hi]-values[lo])*(x-lo)
    return out


def layer_stats(rows):
    return dict(count=len(rows),hidden_count=sum(r['stall_ms']<1e-8 for r in rows),
        fully_hidden_percent=100*sum(r['stall_ms']<1e-8 for r in rows)/len(rows) if rows else None,
        stall_ms=quantiles([r['stall_ms'] for r in rows]),io_lifetime_ms=quantiles([r['io_lifetime_ms'] for r in rows]),
        read_over_budget=quantiles([r['io_lifetime_ms']/r['prefetch_budget_ms'] for r in rows]))


def population_info(manifest):
    data=read(manifest); by_id={};lanes=defaultdict(list);seen=set()
    for r in data['requests']:
        load=r['load'];rid=r['request_id'];origin=load['original_request_id']
        placement=data['placements'][r['placement_index']]
        scientific={k:v for k,v in load.items() if k not in ('request_id','npu_id','generation','source_fixed_request_id','source_fixed_npu_id')}
        assert origin not in seen;seen.add(origin)
        info=dict(rid=rid,npu=r['npu_id'],role=load['role'],C_ms=load['per_layer_us']/1000,
            profile=[load['seq_len_k'],load['nql']],original_id=origin,
            scientific_hash=digest([origin,r['arrival_time_ms'],scientific,placement]),
            per_ssu_rate=[math.fsum(v for s,v in placement[0] if s==d)*1e6/load['per_layer_us'] for d in range(6)])
        by_id[rid]=info;lanes[r['npu_id']].append(info)
    prefixes={}
    for n in range(32):
        lanes[n].sort(key=lambda r:r['rid']);prefix=0
        for pos,r in enumerate(lanes[n]):
            r['position']=pos
            if pos==prefix and r['role']=='long':prefix+=1
        prefixes[n]=prefix
    science=sorted((r['original_id'],r['scientific_hash']) for r in by_id.values())
    binding=sorted((r['original_id'],r['npu'],r['scientific_hash']) for r in by_id.values())
    return dict(metadata=data['metadata'],fingerprint=data['input_fingerprint'],requests=by_id,lanes=lanes,
        initial_long_prefix_count=prefixes,global_population_sha256=digest(science),per_npu_population_sha256=digest(binding),
        manifest_sha256=sha(manifest))


def role_timeline(rows,inputs):
    events=defaultdict(lambda:{'end':[],'start':[]})
    for r in rows:
        events[r['admission_time_ms']]['start'].append(r['request_id'])
        events[r['completion_time_ms']]['end'].append(r['request_id'])
    times=sorted(set(events)|{START,END});active={};segments=[];peaks=[0.0]*6;over=0.0
    for i,t in enumerate(times):
        for rid in events[t]['end']:
            n=inputs[rid]['npu'];assert active[n]==rid;del active[n]
        for rid in events[t]['start']:
            n=inputs[rid]['npu'];assert n not in active;active[n]=rid
        if i+1==len(times):break
        end=times[i+1]
        rates=[math.fsum(inputs[rid]['per_ssu_rate'][s] for rid in active.values()) for s in range(6)]
        peaks=[max(a,b) for a,b in zip(peaks,rates)]
        if max(rates)>40+1e-9:over+=end-t
        a,b=max(t,START),min(end,END)
        if b<=a:continue
        long=sorted(n for n,rid in active.items() if inputs[rid]['role']=='long')
        short=sorted(n for n,rid in active.items() if inputs[rid]['role']=='short')
        if segments and segments[-1]['long_npus']==long and segments[-1]['short_npus']==short and segments[-1]['end_ms']==a:
            segments[-1]['end_ms']=b
        else:segments.append(dict(start_ms=a,end_ms=b,long_npus=long,short_npus=short))
    assert not active
    histogram=defaultdict(float)
    for segment in segments:
        segment['duration_ms']=segment['end_ms']-segment['start_ms']
        segment['long_count']=len(segment['long_npus']);segment['short_count']=len(segment['short_npus'])
        histogram[segment['long_count']]+=segment['duration_ms']
    return dict(segments=segments,long_count_duration_ms=dict(sorted(histogram.items())),
        mean_long_count=sum(k*v for k,v in histogram.items())/(END-START),
        max_ssu_current_V_over_C_gib_s=max(peaks),full_run_per_ssu_peak_gib_s=peaks,any_ssu_over40_ms=over)


def inspect(alias,directory,label):
    manifest=directory/'inputs'/f'{label}.json.gz';info=population_info(manifest)
    candidates=list((directory/'runs'/label/'baseline').glob('*.json.gz'))
    command=directory/'runs'/label/'baseline'/'command.json'
    if len(candidates)!=1 or not command.exists() or read(command).get('status')!='complete':
        return dict(alias=alias,label=label,status='pending',_input=info)
    path=candidates[0];raw=read(path);s=raw['summary'];cmd=read(command)
    assert raw['strategy']=='baseline' and raw['submit_seed']==7 and cmd['returncode']==0
    assert raw['input_fingerprint']==info['fingerprint'] and all(s['invariants'].values())
    assert raw['execution_placement_fingerprint']==raw['input_placement_fingerprint']
    assert all(cmd['source_sha256'].get(name)==value==sha(ROOT/name) for name,value in raw['core_and_policy_sha256'].items())
    inputs=info['requests'];ledger=[];per_npu=[];first_short=[]
    for n in range(32):per_npu.append(dict(npu_id=n,roles={r:dict(compute_ms=0.0,active_ms=0.0,l0_stall_ms=0.0,l1_7_stall_ms=0.0) for r in ('short','long')}))
    for b in s['microbatch_metrics']:
        assert b['batch_size']==1 and len(b['member_request_ids'])==1
        rid=b['member_request_ids'][0];request=inputs[rid];n=b['npu_id'];role=request['role'];C=request['C_ms']
        assert request['npu']==n
        admission,completion=b['admission_time_ms'],b['completion_time_ms'];prev=admission
        per_npu[n]['roles'][role]['active_ms']+=clip(admission,completion)
        for l in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            idx=l['layer'];cs,ce=l['compute_start_ms'],l['compute_end_ms'];release,ready=l['io_start_time_ms'],l['io_ready_time_ms']
            wait=max(0.0,cs-prev)
            assert close(ce-cs,C) and close(wait,l['io_barrier_wait_ms']) and close(wait,max(0.0,ready-prev))
            z=per_npu[n]['roles'][role];z['compute_ms']+=clip(cs,ce);z['l0_stall_ms' if idx==0 else 'l1_7_stall_ms']+=clip(prev,cs)
            row=dict(npu=n,request_id=rid,original_request_id=request['original_id'],request_position=request['position'],
                role=role,profile=request['profile'],layer=idx,C_ms=C,release_ms=release,previous_compute_end_ms=prev,
                ready_ms=ready,compute_start_ms=cs,compute_end_ms=ce,stall_ms=wait,io_lifetime_ms=ready-release,
                prefetch_budget_ms=prev-release,warm_clipped_stall_ms=clip(prev,cs),warm_clipped_compute_ms=clip(cs,ce))
            if idx>0:assert close(prev-release,C)
            ledger.append(row);prev=ce
        assert close(prev,completion)
    role_totals={}
    for role in ('long','short'):
        role_totals[role]={k:math.fsum(p['roles'][role][k] for p in per_npu) for k in ('compute_ms','active_ms','l0_stall_ms','l1_7_stall_ms')}
        r=role_totals[role];r['mean_resident_npu_count']=r['active_ms']/(END-START)
        r['pooled_compute_over_admitted_active_percent']=100*r['compute_ms']/r['active_ms'] if r['active_ms'] else None
    for n,p in enumerate(per_npu):
        a=sum(p['roles'][r]['active_ms'] for r in ('long','short'));c=sum(p['roles'][r]['compute_ms'] for r in ('long','short'))
        st=sum(p['roles'][r][k] for r in ('long','short') for k in ('l0_stall_ms','l1_7_stall_ms'))
        assert close(c+st,a)
        p['active_ms']=a;p['U_percent']=100*c/(END-START)
        rs=sorted((r for r in s['request_metrics'] if r['npu_id']==n),key=lambda r:r['admission_time_ms'])
        shorts=[r['admission_time_ms'] for r in rs if inputs[r['request_id']]['role']=='short']
        longs=[r['admission_time_ms'] for r in rs if inputs[r['request_id']]['role']=='long']
        p['first_short_admission_ms']=min(shorts) if shorts else None;p['first_long_admission_ms']=min(longs) if longs else None
    U=math.fsum(r['compute_ms'] for r in role_totals.values())/64000*100
    raw_window=next(w for w in raw['windows'] if w['start_ms']==START and w['end_ms']==END)
    assert close(U,100*raw_window['mean_npu_utilization'])
    warm=[r for r in s['request_metrics'] if START<=r['admission_time_ms']<END]
    passed=sum(r['completion_time_ms']-r['admission_time_ms']<=1.5*8*inputs[r['request_id']]['C_ms']+1e-9 for r in warm)
    assert passed==raw['slo']['window_admissions']['admission']['passed'] and len(warm)==raw['slo']['window_admissions']['admission']['count']
    internal={r:[x for x in ledger if x['role']==r and x['layer']>0 and x['release_ms']>=START and x['compute_end_ms']<=END] for r in ('short','long')}
    example=max(internal['short'],key=lambda x:(x['stall_ms'],-x['release_ms'])) if internal['short'] else None
    overlap_longs=[x for x in ledger if x['role']=='long' and example and x['release_ms']<example['ready_ms'] and x['ready_ms']>example['release_ms']]
    timeline=role_timeline(s['request_metrics'],inputs)
    return dict(alias=alias,label=label,status='complete',result=str(path),result_sha256=sha(path),manifest=str(manifest),manifest_sha256=sha(manifest),
        input_fingerprint=raw['input_fingerprint'],core_source_sha256=raw['core_and_policy_sha256'],
        device_U_percent=U,warm_admission_count=len(warm),warm_slo_passed=passed,warm_SLO_percent=100*passed/len(warm),
        all_32_active=all(close(p['active_ms'],2000) for p in per_npu),
        all_32_warm_mixed=all(all(p['roles'][r]['compute_ms']>0 for r in ('short','long')) for p in per_npu),
        minimum_per_card_role_compute_ms={r:min(p['roles'][r]['compute_ms'] for p in per_npu) for r in ('short','long')},
        roles=role_totals,per_npu=per_npu,role_timeline=timeline,
        internal_layer_cohorts={r:layer_stats(internal[r]) for r in ('short','long')},
        selected_short_example=example,example_selection='Largest fully-contained warm L1-7 Short exposed wait; ties earliest release. Not selected based on SSD predecessors.',
        simultaneous_long_IO_lifetime_count=len(overlap_longs),
        global_population_sha256=info['global_population_sha256'],per_npu_population_sha256=info['per_npu_population_sha256'],
        _input=info,_ledger=ledger,_internal=internal)


def waves(case,prefix_limit):
    if case['status']!='complete':return {'applicable':False,'reason':'result pending'}
    prefixes=case['_input']['initial_long_prefix_count']
    if not all(prefixes[n]>=prefix_limit for n in FRONT):return {'applicable':False,'reason':'front20 do not share the initial consecutive Long prefix; random layers are not a matched wave'}
    groups=defaultdict(dict)
    for r in case['_ledger']:
        if r['role']=='long' and r['npu'] in FRONT and r['request_position']<prefix_limit:
            groups[(r['request_position'],r['layer'])][r['npu']]=r
    values=[]
    for (pos,l),members in sorted(groups.items()):
        if set(members)!=set(FRONT):continue
        rows=list(members.values())
        if not all(START<=r['release_ms']<END and START<=r['compute_start_ms']<END for r in rows):continue
        values.append(dict(request_position=pos,layer=l,npu_count=20,
            release_min_ms=min(r['release_ms'] for r in rows),release_max_ms=max(r['release_ms'] for r in rows),
            release_spread_ms=max(r['release_ms'] for r in rows)-min(r['release_ms'] for r in rows),
            compute_start_spread_ms=max(r['compute_start_ms'] for r in rows)-min(r['compute_start_ms'] for r in rows),
            max_Long_stall_ms=max(r['stall_ms'] for r in rows)))
    return dict(applicable=True,cards=FRONT,common_initial_long_request_count=prefix_limit,
        alignment='same current-lane ordinal and layer within common initial Long prefix on NPU0-19, not the same original request ID; all releases and compute starts warm',
        wave_count=len(values),release_spread_ms=quantiles([r['release_spread_ms'] for r in values]),
        compute_spread_ms=quantiles([r['compute_start_spread_ms'] for r in values]),waves=values)


def public(value):
    if isinstance(value,dict):return {k:public(v) for k,v in value.items() if not str(k).startswith('_')}
    if isinstance(value,list):return [public(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed-dir',type=Path,default=HERE);p.add_argument('--rebinding-dir',type=Path,default=HERE/'mixed_rebinding')
    p.add_argument('--fixed-label',default='raw200_three_l20_fixed_seed7')
    p.add_argument('--random-label',default='raw200_rebinding_random_seed7');p.add_argument('--ordered-label',default='raw200_rebinding_ordered_seed7')
    p.add_argument('--output-prefix',type=Path,default=HERE/'mixed_rebinding_mechanism')
    p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    cases=[inspect('fixed',args.fixed_dir,args.fixed_label),inspect('random',args.rebinding_dir,args.random_label),inspect('ordered',args.rebinding_dir,args.ordered_label)]
    fixed,random,ordered=cases
    ordered_prefix=min(ordered['_input']['initial_long_prefix_count'][n] for n in FRONT)
    prefix_limit=min(ordered_prefix,min(fixed['_input']['initial_long_prefix_count'][n] for n in FRONT))
    assert prefix_limit>0
    for case in cases:case['common_prefix_front20_waves']=waves(case,prefix_limit)
    guard_times=[]
    if ordered['status']=='complete':
        guard_times=[ordered['per_npu'][n]['first_short_admission_ms'] for n in GUARDS]
        assert all(t is not None for t in guard_times)
        ordered['guard3_exit_initial_long_ms']=dict(by_npu={str(n):ordered['per_npu'][n]['first_short_admission_ms'] for n in GUARDS},earliest=min(guard_times),latest=max(guard_times))
        for case in cases:
            if case['status']!='complete':continue
            rs=case['_internal']['short']
            before=[r for r in rs if r['compute_end_ms']<=min(guard_times)]
            case['selected_short_before_first_guard_exit']=max(before,key=lambda r:(r['stall_ms'],-r['release_ms'])) if before else None
            case['short_internal_by_ordered_guard_epoch']={
                'before_first_guard_exit':layer_stats(before),
                'after_all_guard_exit':layer_stats([r for r in rs if r['release_ms']>=max(guard_times)]),
                'note':'Secondary descriptive split at ordered measured guard exit; original [2000,4000) primary statistics unchanged.'}
    pairing=dict(all_three_global_population_identical=len({c['_input']['global_population_sha256'] for c in cases})==1,
        random_ordered_exact_per_npu_population_and_placement=random['_input']['per_npu_population_sha256']==ordered['_input']['per_npu_population_sha256'],
        excluded_order_encoding_fields=['request_id','npu_id','generation','source_fixed_request_id','source_fixed_npu_id'])
    assert all(v for k,v in pairing.items() if isinstance(v,bool))
    complete=all(c['status']=='complete' for c in cases)
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),no_simulation=True,all_three_complete=complete,
        window_ms=[START,END],common_front20_Long_prefix_count=prefix_limit,input_pairing=pairing,cases=public(cases),
        definitions=dict(warm_layer_cohort='Internal L1-7 with release>=2000 and compute_end<=4000; complete lifecycle and wait, no boundary censoring.',
            role_occupancy='Clipped admission-completion resident NPU time, regardless of computing or waiting.',
            io_lifetime='ready-release includes SSD queue, disk service and NPU receive; not uninterrupted SSD service.',
            wave_scope='Only NPU0-19 in common initial consecutive Long prefix; guard exit is measured. No all32 or arbitrary random-prefix alignment.',
            nominal='Independently scanned currently admitted physical per-disk V/C; exact-time ties batched, no extra L0 nominal term.',
            limitation='Synchronized releases plus simultaneous short waits do not identify physical FIFO predecessors. No block-level causal trace generated. Selected seed7 only.'))
    args.output_prefix.with_suffix('.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    md=['# 重新混合后的定序：长波相位与短层等待','',
        '只读三个 Baseline 运行；主窗固定[2000,4000)ms。Fixed、Random、Ordered保留同一全局原请求人口；新Random与Ordered还保持每卡原请求和实际落盘完全相同，只有卡内顺序不同。Fixed的每卡人口与新混合不同。','',
        f'完成{sum(c["status"]=="complete" for c in cases)}/3；共同初始Long前缀为NPU0–19各{prefix_limit}条。本页不对随机混合的不同逻辑位置强行匹配一波，也不把后续角色切换后的32卡全称为同步Long卡。','',
        '|运行|U %|暖接纳SLO %|Long平均驻留卡数|Long计算/占用 U %|Short计算/占用 U %|每卡都有暖Long/Short计算|',
        '|---|---:|---:|---:|---:|---:|---|']
    for c in cases:
        if c['status']!='complete':md.append(f'|{c["alias"]}|pending|—|—|—|—|—|');continue
        md.append(f'|{c["alias"]}|{c["device_U_percent"]:.6f}|{c["warm_SLO_percent"]:.6f}|{c["roles"]["long"]["mean_resident_npu_count"]:.6f}|{c["roles"]["long"]["pooled_compute_over_admitted_active_percent"]:.6f}|{c["roles"]["short"]["pooled_compute_over_admitted_active_percent"]:.6f}|{c["all_32_warm_mixed"]}|')
    md+=['','SLO按暖窗内admission纳入，计completion−admission≤1.5×8C，跟踪到完成；不含接纳前等待，不是实际端到端首token测量。暖窗接纳人口会随分配和顺序改变，SLO分母不一定是同一批请求ID。','',
        '|运行|共同前20卡波数|release跨度 p50 / max ms|Short完整内部层数|Short stall p50 / p95 / max ms|Short完全隐藏 %|',
        '|---|---:|---|---:|---|---:|']
    for c in cases:
        if c['status']!='complete':continue
        w=c['common_prefix_front20_waves'];q=c['internal_layer_cohorts']['short'];st=q['stall_ms']
        span=f'{w["release_spread_ms"]["p50"]:.6f} / {w["release_spread_ms"]["max"]:.6f}' if w.get('wave_count') else '不作同波配对'
        vals=' / '.join(f'{st[k]:.6f}' for k in ('p50','p95','max'))
        md.append(f'|{c["alias"]}|{w.get("wave_count",0)}|{span}|{q["count"]}|{vals}|{q["fully_hidden_percent"]:.3f}|')
    if guard_times:md+=['',f'Ordered的guard3（NPU17–19）在{min(guard_times):.6f}–{max(guard_times):.6f}ms退出其初始Long段。共同20卡波只在此前的同位置Long前缀对齐；后续驻留变化逐段保存在JSON。']
    md+=['','角色时间账（单位NPU·ms，均裁剪到原主窗）：','',
        '|运行/角色|计算|已接纳占用|L0等待|L1–7等待|','|---|---:|---:|---:|---:|']
    for c in cases:
        if c['status']!='complete':continue
        for role in ('long','short'):
            r=c['roles'][role];md.append(f'|{c["alias"]}/{role}|{r["compute_ms"]:.6f}|{r["active_ms"]:.6f}|{r["l0_stall_ms"]:.6f}|{r["l1_7_stall_ms"]:.6f}|')
    md+=['','三个具体短内部层：Fixed取完整暖窗最大等待；Ordered分别取guard退出前的完整短层最大等待、全暖窗最大等待。原始Random样例另存JSON。以下不是查明FIFO前驱的样本。','',
        '|运行|NPU / request / layer|IO release|前层compute end|IO ready|暴露stall ms|','|---|---|---:|---:|---:|---:|']
    examples=[]
    if fixed['status']=='complete':examples.append(('fixed／全暖窗最大',fixed['selected_short_example']))
    if ordered['status']=='complete':
        examples.extend([('ordered／guard退出前',ordered.get('selected_short_before_first_guard_exit')),('ordered／全暖窗最大',ordered['selected_short_example'])])
    for label,e in examples:
        if e:md.append(f'|{label}|{e["npu"]} / {e["request_id"]} / L{e["layer"]}|{e["release_ms"]:.6f}|{e["previous_compute_end_ms"]:.6f}|{e["ready_ms"]:.6f}|{e["stall_ms"]:.6f}|')
    if complete:
        drop=random['device_U_percent']-ordered['device_U_percent']
        long_C=next(r['C_ms'] for r in ordered['_input']['requests'].values() if r['role']=='long')
        before=ordered['short_internal_by_ordered_guard_epoch']['before_first_guard_exit']['stall_ms']
        after=ordered['short_internal_by_ordered_guard_epoch']['after_all_guard_exit']['stall_ms']
        before_text=f'{before["mean"]:.6f}' if before['count'] else 'N/A'
        after_text=f'{after["mean"]:.6f}' if after['count'] else 'N/A'
        md+=['',f'新绑定内Random减Ordered的设备U为{drop:+.6f}个百分点，正值表示定序后下降。这是同一每卡人口的顺序对照；与Fixed相比还包含人口绑定变化。',
            '',f'Ordered平均Long驻留{ordered["roles"]["long"]["mean_resident_npu_count"]:.6f}张卡，Random为{random["roles"]["long"]["mean_resident_npu_count"]:.6f}；短角色自身的计算/占用比在Random和Ordered下分别为{random["roles"]["short"]["pooled_compute_over_admitted_active_percent"]:.6f}%和{ordered["roles"]["short"]["pooled_compute_over_admitted_active_percent"]:.6f}%。应同时检查角色占比和角色自身等待，不能仅凭设备平均U判定二者的影响。',
            '',f'Ordered完整暖窗内Long内部层最大IO生命周期{ordered["internal_layer_cohorts"]["long"]["io_lifetime_ms"]["max"]:.6f}ms，计算预算{long_C:.6f}ms，内部层完全隐藏占比{ordered["internal_layer_cohorts"]["long"]["fully_hidden_percent"]:.6f}%；Long的暖窗L0/L1–7等待分别为{ordered["roles"]["long"]["l0_stall_ms"]:.6f}/{ordered["roles"]["long"]["l1_7_stall_ms"]:.6f} NPU·ms。Short在guard退出前/后的完整内部层平均等待分别为{before_text}/{after_text}ms；后期角色切换带来不同时间结构，不能再沿用前20卡共同Long波的解释。']
    md+=['','上述跨度与等待只支持时间结构的一致性解释。某条Long的IO生命周期与Short重叠，不足以认定其为物理FIFO前驱；ready−release也不是磁盘连续服务时间。全窗角色账、逐卡计算与占用、共同波记录、来源哈希及guard前后补充统计保存在同名JSON。']
    args.output_prefix.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(dict(all_three_complete=complete,pairing=pairing,brief=[{k:c[k] for k in ('alias','status','device_U_percent','warm_SLO_percent') if k in c} for c in cases]),ensure_ascii=False))
    if args.require_complete and not complete:raise SystemExit(1)


if __name__=='__main__':main()
