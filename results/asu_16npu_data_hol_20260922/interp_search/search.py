#!/usr/bin/env python3
"""Approximate I1/I2 ratio-and-phase selector, never reportable native results."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'phase_search/approx_phase_search.py'
spec=importlib.util.spec_from_file_location('independent_readonly_phase_model',SOURCE)
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
SEEDS=(1,7,19,43,91,137,211)


def main():
    rows=[]
    source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for identity,ma,mb in [('I1',3302,2448),('I2',3328,2500)]:
        ca,va=model.profile(200,ma);cb,vb=model.profile(32,mb)
        for q in range(1,21):
            cycles=math.ceil(12000/(8*(ca+q*cb)))+1
            for seed in SEEDS:
                pats=[]
                for n in range(16):
                    p=[0]*cycles+[1]*(cycles*q)
                    random.Random(seed+100003*n).shuffle(p);pats.append(p)
                r=model.simulate(pats,m=ma,m_b=mb)
                r.update(name=f'{identity}_random_q{q}_seed{seed}',profile=identity,
                         mode='random',q=q,seed=seed,cycles=cycles)
                rows.append(r)
            # I1 periodic cases are covered separately by design16_phases.
            if identity=='I2':
                base=[0]+[1]*q
                designs=[]
                for step in range(q+1):
                    designs.append((f'stride{step}',[(n*step)%(q+1) for n in range(16)]))
                for cohort in (4,8,12):
                    for off in range(1,q+1):
                        designs.append((f'cohort{cohort}_off{off}',[off if n<cohort else 0 for n in range(16)]))
                for phase,offsets in designs:
                    pats=[base[o:]+base[:o] for o in offsets]
                    r=model.simulate(pats,m=ma,m_b=mb)
                    r.update(name=f'{identity}_cyclic_q{q}_{phase}',profile=identity,
                             mode='cyclic',q=q,seed=7,cycle_offsets=offsets,
                             cycle='A'+'B'*q)
                    rows.append(r)
            print(identity,q,len(rows),flush=True)
    rows.sort(key=lambda r:r['U'])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_sha
    # Balance independent-shuffle cases and deliberately chosen periodic phases.
    selected=[]
    for identity in ('I1','I2'):
        candidates=[r for r in rows if r['profile']==identity and r['mode']=='random'
                    and r['every_card_AB_2_4']]
        seen=set()
        for r in candidates:
            if r['q'] in seen:continue
            seen.add(r['q']);selected.append(r)
            if len(seen)==2:break
    candidates=[r for r in rows if r['mode']=='cyclic' and r['every_card_AB_2_4']]
    seen=set()
    for r in candidates:
        if r['q'] in seen:continue
        seen.add(r['q']);selected.append(r)
        if len(seen)==2:break
    for r in selected:
        patterns=[[int(c=='B') for c in p] for p in r['patterns']]
        r['approx_repeat_to_60s']=model.simulate(patterns,m=r['miss'],m_b=r['miss_B'],right=60000)
        # Random finite decks repeated here become a PERIODIC long-run diagnostic.
        r['long_diagnostic_caveat']='The same finite deck is repeated indefinitely; this is not a fresh 60-second independent-shuffle input.'
    report=dict(approximate_only=True,source_sha256=source_sha,
                assumptions=['whole-layer FIFO','ideal striping','approximate NPU link',
                             'unbounded periodic patterns','no native path or block simulation'],
                profiles='grid-interpolated I1/I2, not original data rows',
                seeds=SEEDS,random_cases=280,total_cases=len(rows),rows=rows)
    (HERE/'all.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'shortlist.json').write_text(json.dumps(dict(approximate_only=True,rows=selected),indent=2)+'\n')
    text=['# I1/I2 广比例和相位筛选：仅近似候选','',
          '**这些数字不是原生仿真结果，不可加入正式利用率比较表。** 近似把整层读取原子入队并使用简化接收链路，只用于挑选下一批原生输入。', '',
          f'共 {len(rows)} 个候选，其中 I1/I2 的 q=1..20、7个种子整队列独立洗牌共280个，其余为I2相位选择。随机队列按12秒纯计算工作量准备；短窗口使用[2,4)，主要排序窗口[2,12)。', '',
          '| 输入 | q | seed | 近似[2,4) U | 近似[2,12) U | 同deck周期重复[2,60) U | 每卡warm均含AB |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        text.append(f"| {r['name']} | {r['q']} | {r['seed']} | {r['windows'][0]:.4f}% | {r['U']:.4f}% | {r['approx_repeat_to_60s']['U']:.4f}% | {r['every_card_AB_2_4']} |")
    text+=['','60秒诊断把同一有限deck反复循环，不是重新生成60秒随机输入；只能帮助淘汰短暂低值，不能替代长期原生实验。','',
           '全部 I1/I2 画像在单盘上有与排列无关的任意组合欠载保证。低近似利用率不自动表示原生也低，也不能用候选筛选中的最低seed代替跨seed均值。','',
           '[全部近似记录](all.json) · [六个候选及每卡pattern](shortlist.json) · [复算程序](search.py)','']
    (HERE/'notes.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__=='__main__':main()
