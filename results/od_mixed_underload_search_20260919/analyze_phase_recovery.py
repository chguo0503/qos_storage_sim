#!/usr/bin/env python3
"""Read-only recovery evidence: residence, barriers, and B-layer phase events."""
from __future__ import annotations

import argparse
import bisect
import cmath
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict

import analyze_completed as audit

HERE=Path(__file__).resolve().parent
WINDOWS=((2000.,4000.),(4000.,8000.),(8000.,12000.),(12000.,16000.))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):return math.fsum(values)/len(values) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();directory=args.run_dir.resolve()
    paths={n:directory/n for n in ('result.json.gz','manifest.json.gz','command.json')}
    before={n:sha(x) for n,x in paths.items()}
    result,manifest,command,requests,records,_=audit.load_raw(paths['result.json.gz'],paths['manifest.json.gz'],paths['command.json'])
    B_C=sorted({r['C_ms'] for r in records.values() if r['role']=='B'})
    assert len(B_C)==1;C_B=B_C[0]
    periodic_reference=None
    parameters=manifest['metadata'].get('profiles',{}).get('A',{}).get('construction',{}).get('parameters',{})
    if 'a_run' in parameters and 'b_run' in parameters:
        A_C={r['C_ms'] for r in records.values() if r['role']=='A'};assert len(A_C)==1
        C_A=next(iter(A_C));n_A=parameters['a_run'];n_B=parameters['b_run']
        share=n_A*C_A/(n_A*C_A+n_B*C_B)
        periodic_reference=dict(A_requests=n_A,B_requests=n_B,C_A_ms=C_A,C_B_ms=C_B,
            pure_compute_A_share=share,equivalent_A_cards=32*share,
            note='Periodic repeat only: excludes B prefixes, waiting, and window phase effects; not a warm-window prediction.')
    B_events=[];B_pending=[];A_deadlines=[]
    for batch in result['summary']['microbatch_metrics']:
        rid=batch['member_request_ids'][0];rec=records[rid];layers=batch['layer_metrics']
        for prev,nxt in zip(layers,layers[1:]):
            audit.close(nxt['io_start_time_ms'],prev['compute_start_ms'])
            if rec['role']=='B':
                B_events.append(dict(t_ms=prev['compute_start_ms'],npu_id=rec['npu_id'],request_id=rid,compute_layer=prev['layer']))
                B_pending.append((nxt['io_start_time_ms'],nxt['io_ready_time_ms']))
            else:
                A_deadlines.append(dict(t_ms=prev['compute_end_ms'],stall_ms=nxt['io_barrier_wait_ms']))
    B_starts=sorted(a for a,z in B_pending);B_ends=sorted(z for a,z in B_pending)
    analyses=[]
    for start,end in WINDOWS:
        assert end<=result['summary']['makespan_ms']
        a,_,_,_,_=audit.analyze(result,requests,records,start,end,3)
        events=[e for e in B_events if start<=e['t_ms']<end]
        vectors_by_card=defaultdict(list);hist=[0]*16;harmonics={}
        for e in events:
            phase=(e['t_ms']%C_B)/C_B
            e={**e,'phase_fraction':phase}
            vectors_by_card[e['npu_id']].append(phase)
            hist[min(15,int(phase*16))]+=1
        for harmonic in (1,2,4,8):
            vectors=[cmath.exp(2j*math.pi*harmonic*((e['t_ms']%C_B)/C_B)) for e in events]
            event_R=abs(sum(vectors)/len(vectors))
            card_means=[sum(cmath.exp(2j*math.pi*harmonic*f) for f in phases)/len(phases) for phases in vectors_by_card.values()]
            harmonics[str(harmonic)]=dict(event_weighted_R=event_R,equal_card_R=abs(sum(card_means)/len(card_means)))
        sampled=[]
        for d in A_deadlines:
            t=d['t_ms']
            if start<=t<end:
                count=bisect.bisect_right(B_starts,t)-bisect.bisect_right(B_ends,t)
                sampled.append(dict(**d,B_IO_outstanding_at_deadline=count))
        intervals=[(max(start,s),min(end,z)) for s,z in B_pending if z>start and s<end]
        busy_mean=math.fsum(z-s for s,z in intervals)/(end-start)
        event_changes=defaultdict(int)
        for s,z in intervals:event_changes[s]+=1;event_changes[z]-=1
        count=0;any_busy=0.;previous=start;maximum=0
        for t,change in sorted(event_changes.items()):
            if count:any_busy+=t-previous
            count+=change;maximum=max(maximum,count);previous=t
        active_A=math.fsum(audit.overlap(r['admission_time_ms'],r['completion_time_ms'],start,end) for rid,r in requests.items() if records[rid]['role']=='A')
        active_B=math.fsum(audit.overlap(r['admission_time_ms'],r['completion_time_ms'],start,end) for rid,r in requests.items() if records[rid]['role']=='B')
        stalled=[d for d in sampled if d['stall_ms']>1e-9]
        unstalled=[d for d in sampled if d['stall_ms']<=1e-9]
        analyses.append(dict(window_ms=[start,end],U_percent=a['fleet_U_percent'],all_npus_active=a['demand']['all_npus_active'],
            all_32_npus_have_A_and_B_compute=a['all_32_npus_have_A_and_B_compute'],strict_underload=a['demand']['strict_underload_all_disks'],
            A_residence_card_ms=active_A,B_residence_card_ms=active_B,A_residence_fraction=active_A/(active_A+active_B),
            mean_A_resident_cards=active_A/(end-start),role_summary=a['role_summary'],
            IO_wait_card_ms_by_kind=a['io_stall_card_ms_by_kind'],IO_loss_pp_by_kind=a['io_stall_loss_pp_by_kind'],
            B_phase_event_count=len(events),B_phase_card_count=len(vectors_by_card),B_phase_harmonics=harmonics,
            B_phase_hist16_counts=hist,B_IO_mean_outstanding=busy_mean,B_IO_max_outstanding=maximum,B_IO_any_outstanding_fraction=any_busy/(end-start),
            A_internal_deadline_count=len(sampled),A_internal_stalled_deadline_count=len(stalled),A_internal_stalled_fraction=len(stalled)/len(sampled),
            B_outstanding_at_A_deadline_mean=mean([d['B_IO_outstanding_at_deadline'] for d in sampled]),
            B_outstanding_at_stalled_A_deadline_mean=mean([d['B_IO_outstanding_at_deadline'] for d in stalled]),
            B_outstanding_at_unstalled_A_deadline_mean=mean([d['B_IO_outstanding_at_deadline'] for d in unstalled])))
    assert before=={n:sha(x) for n,x in paths.items()}
    output=HERE/'mechanism_audit'/f'{directory.name}_phase_recovery';output.mkdir(parents=True,exist_ok=True)
    definitions={
        'A_residence':'Overlap of A admission-to-completion intervals / total active NPU card time, not request counts or compute share.',
        'phase_events':'B compute starts for layers L0..L6, equivalently the submissions of next layers of that same B request. L7 cross-request prefetch excluded.',
        'phase':'(absolute time_ms mod fixed C_B_ms) / C_B_ms, with fixed reference t=0. No reference or period fitted to outcomes.',
        'R':'Magnitude of mean(exp(2*pi*i*h*phase)); h=1,2,4,8. event-weighted and equal-card means reported separately.',
        'phase_caution':'Low R1 can also mean several phase clusters, not uniform phases; R2/R4/R8 and histogram provided. C_B is pure compute, not an assumed observed IO period.',
        'B_outstanding':'Half-open io_start_time_ms to io_ready_time_ms of internal B prefetch. Includes queue, SSD, and NPU transfer; not measured SSD service or disk causality.',
        'A_deadline':'A layer compute_end for an internal next-layer prefetch; classified stalled iff its actual next-layer io_barrier_wait_ms >1e-9.',
        'scope':'Fixed reported windows; concurrent changes are mechanism evidence, not proof that one phase statistic caused recovery.'}
    payload=dict(all_checks_passed=True,source_sha256=before,sources_unchanged=True,script_sha256=sha(Path(__file__)),C_B_ms=C_B,periodic_mix_reference=periodic_reference,definitions=definitions,windows=analyses)
    (output/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# 利用率恢复的独立证据','',f'B的纯计算周期 C_B={C_B:.9f} ms。全部统计来自完成后的原始请求、层时刻和manifest；不调用runner/metrics，不拟合相位周期。','',
        '|窗口 秒|U|A驻留份额|平均A卡数|内部等待损失 pp|跨请求等待损失 pp|A内部deadline发生等待比例|B在途平均数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in analyses:
        s,z=a['window_ms'];loss=a['IO_loss_pp_by_kind']
        lines.append(f'|[{s/1000:g},{z/1000:g})|{a["U_percent"]:.6f}%|{100*a["A_residence_fraction"]:.4f}%|{a["mean_A_resident_cards"]:.4f}|{loss.get("internal_L1_to_L7",0):.6f}|{loss.get("cross_request_L0",0):.6f}|{100*a["A_internal_stalled_fraction"]:.3f}%|{a["B_IO_mean_outstanding"]:.3f}|')
    lines+=['','|窗口 秒|B起点事件数|参与卡数|R1|R2|R4|R8|A等待deadline时B在途数|A不等待deadline时B在途数|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in analyses:
        s,z=a['window_ms'];vals=[a['B_phase_harmonics'][str(h)]['event_weighted_R'] for h in (1,2,4,8)]
        x=a['B_outstanding_at_stalled_A_deadline_mean'];y=a['B_outstanding_at_unstalled_A_deadline_mean']
        lines.append(f'|[{s/1000:g},{z/1000:g})|{a["B_phase_event_count"]}|{a["B_phase_card_count"]}|'+ '|'.join(f'{v:.4f}' for v in vals)+f'|{x:.3f}|{y:.3f}|')
    if periodic_reference:
        q=periodic_reference
        lines+=['',f'周期块{q["A_requests"]}A+{q["B_requests"]}B的无等待纯计算A份额={q["A_requests"]}*C_A/({q["A_requests"]}*C_A+{q["B_requests"]}*C_B)={100*q["pure_compute_A_share"]:.6f}%，等效{q["equivalent_A_cards"]:.6f}张A卡。8层在分子分母中约掉。该参考不包括B前缀、等待、窗口相位，不能直接预测warm。']
    lines+=['','A驻留份额包含计算和等待；等待本身也会提高该份额，因此它不是外生固定输入比例。',
        f'把{C_B:.3f}ms看成一圈不断转动的表盘，每次B发起下一层IO就在表盘上点一个点；点挤在一个方向时R1接近1。这里用固定纯计算时间作为表盘周期，没有为了迎合结果拟合周期。',
        '相位取B内部预取提交时刻对纯计算C_B取模。R1越接近1，事件相位越集中于一个方向；R1低也可能是多个对称簇，所以同时给R2/R4/R8、16格直方图，以及按卡等权版本（JSON）。不能把一个R值当成同步程度的充分描述。',
        'B在途从IO提交算到数据到达NPU，含排队、SSD和NPU传输；它不是SSD真实服务并发数。A的deadline是本层计算结束，是否等待直接使用下一层原始io_barrier_wait。',
        '这些统计用于检查恢复是否伴随驻留或相位变化。相关变化不自动证明因果；尤其不能仅凭整机U上升就宣布“自然错开已解决问题”。',
        '', '详细原始统计：[analysis.json](analysis.json)','']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),windows=[{k:a[k] for k in ('window_ms','U_percent','A_residence_fraction','mean_A_resident_cards','B_phase_harmonics','A_internal_stalled_fraction')} for a in analyses]),ensure_ascii=False))


if __name__=='__main__':main()
