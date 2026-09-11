#!/usr/bin/env python3
"""Audit reordered mixed queues, active role counts, and short stalls."""
import csv
import json
import math
from pathlib import Path
import sys

FOLLOWUP=Path(__file__).resolve().parent
HERE=FOLLOWUP/'mixed_rotation'
ROOT=FOLLOWUP.parent.parents[1]
sys.path[:0]=[str(ROOT),str(FOLLOWUP),str(FOLLOWUP.parent)]
import analyze_followup as base
base.HERE=HERE
from run_baseline_npu32_stress import load_manifest,read_json,write_json


def role_counts(raw,requests):
    byid={r.request_id:r for r in requests}
    events={2000.0:[0,0],4000.0:[0,0]}
    for b in raw['summary']['microbatch_metrics']:
        a=max(2000.,b['admission_time_ms']);z=min(4000.,b['completion_time_ms'])
        if z<=a:continue
        idx=0 if byid[b['member_request_ids'][0]].load['role']=='long' else 1
        events.setdefault(a,[0,0])[idx]+=1;events.setdefault(z,[0,0])[idx]-=1
    current=[0,0];hist={};intervals=[]
    times=sorted(events)
    for i,t in enumerate(times[:-1]):
        current=[a+b for a,b in zip(current,events[t])]
        dt=times[i+1]-t
        assert sum(current)==32,(t,current)
        hist[current[0]]=hist.get(current[0],0)+dt
        intervals.append(dict(start_ms=t,end_ms=times[i+1],long_cards=current[0],short_cards=current[1]))
    mean=sum(n*d for n,d in hist.items())/2000
    return dict(mean_long_cards=mean,mean_short_cards=32-mean,min_long_cards=min(hist),max_long_cards=max(hist),
        fraction_long_18_to_22=sum(d for n,d in hist.items() if 18<=n<=22)/2000,
        fraction_long_16_to_24=sum(d for n,d in hist.items() if 16<=n<=24)/2000,
        histogram_ms=hist,intervals=intervals)


def pair_check(requests,source):
    byid={r.request_id:r for r in source}
    seen=[]
    for r in requests:
        old=byid[r.load['source_mixed_request_id']];seen.append(old.request_id)
        assert r.npu_id==old.npu_id and r.arrival_time_ms==old.arrival_time_ms and r.placement==old.placement
        assert all(r.load[k]==v for k,v in old.load.items() if k not in ('request_id','generation'))
    assert sorted(seen)==sorted(byid)


def main():
    jobs={}
    for p in sorted((HERE/'plans').glob('*.json')):
        plan=read_json(p)
        assert base.auditor.sha(plan['spec_file'])==plan['spec_sha256']
        for name,digest in plan['source_sha256'].items():
            snapshot=HERE/'sources'/digest/Path(name).name
            assert base.auditor.sha(snapshot)==digest
            if not name.startswith('results/'):
                assert base.auditor.sha(ROOT/name)==digest
        for j in plan['jobs']:jobs[j['input']['label'],j['strategy']]=j
    rows=[];pending=[];errors=[]
    for (label,strategy),j in jobs.items():
        paths=list((HERE/'runs'/label/strategy).glob('*.json.gz'))
        if not paths or read_json(paths[0].parent/'command.json').get('status') is None:
            pending.append(dict(label=label,strategy=strategy));continue
        assert len(paths)==1
        try:
            item=j['input'];out=base.analyze(item,strategy,paths[0]);row=dict(out['row'])
            requests,meta=load_manifest(item['manifest']);source,_=load_manifest(item['source_manifest'])
            pair_check(requests,source)
            raw=read_json(paths[0]);counts=role_counts(raw,requests)
            write_json(HERE/'role_counts'/label/f'{strategy}.json',counts)
            row.update({k:v for k,v in counts.items() if k not in ('intervals','histogram_ms')})
            cards=out['technical']['windows'][0]['per_npu']
            for role in ('short','long'):
                row[f'min_card_{role}_active_ms']=min(c['by_role'][role]['active_ms'] for c in cards)
                row[f'min_card_{role}_compute_ms']=min(c['by_role'][role]['compute_ms'] for c in cards)
            row['per_npu_population_and_placement_preserved']=True
            row['strict_mixed_underload_valid']=row['active_underload_valid'] and row['warm_mixed_card_count']==32
            row['mean_role_count_near_20_12']=18<=row['mean_long_cards']<=22
            rows.append(row)
        except Exception as e:errors.append(dict(label=label,strategy=strategy,error=f'{type(e).__name__}: {e}'))
    out=dict(completed=len(rows),planned=len(jobs),pending=pending,errors=errors,rows=rows,
        count_definition='Current admitted long/short NPUs per time, integrated over[2000,4000). Near means mean Long18..22, reported alongside full histogram; not a guarantee of constant20/12.')
    write_json(HERE/'results.json',out)
    if rows:
        with (HERE/'per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['**每卡长短混合的定向排列探索**','',
        '同一原始mixed每卡请求人口，只改变卡内次序；不添加延时或缩放C/V。窗口[2,4)s，角色数量按接纳到完成的驻留时间统计。所有候选均保留，不只列出较低U。','',
        '| 排列 | seed | 策略 | U% | SLO% | 平均长卡/短卡 | 长卡数min–max | 18–22长卡时间占比 | 暖混合卡数 | 全程盘峰GiB/s | 每卡混合且欠载 |',
        '|---|---:|---|---:|---:|---|---|---:|---:|---:|---|']
    for r in rows:
        text.append(f"| {r['spec_name']} | {r['seed']} | {r['strategy']} | {r['device_utilization_percent']:.4f} | {r['warm_slo_percent']:.4f} | {r['mean_long_cards']:.3f}/{r['mean_short_cards']:.3f} | {r['min_long_cards']}–{r['max_long_cards']} | {100*r['fraction_long_18_to_22']:.2f}% | {r['warm_mixed_card_count']} | {r['max_ssu_nominal_gib_s']:.6f} | {r['strict_mixed_underload_valid']} |")
    text+=['','两策略同输入但执行速度不同，实际长短驻留卡数可能不同；不能把构造目标20/12冒充共同实现的时序。旧第四请求≤1500、每卡长短最小卡时间、全程超限时间和warm接纳人数在CSV/JSON中逐项保留。','']
    (HERE/'results.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed=len(rows),planned=len(jobs),errors=errors,rows=[{k:r[k] for k in ('spec_name','seed','strategy','device_utilization_percent','warm_slo_percent','mean_long_cards','warm_mixed_card_count','max_ssu_nominal_gib_s','strict_mixed_underload_valid')} for r in rows]),ensure_ascii=False))
    assert not errors and all(r['audit_pass'] for r in rows)


if __name__=='__main__':main()
