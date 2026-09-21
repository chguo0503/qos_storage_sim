#!/usr/bin/env python3
"""Read-only causal evidence from the native I1 1:8 two-cohort experiment."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
CASE=HERE/'runs/I1_s1_r18_two_cohorts_seed7'


def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as stream:return json.load(stream)


def overlap(a,b,lo,hi):return max(0,min(b,hi)-max(a,lo))


def analyze(strategy):
    folder=CASE/strategy
    native=read(folder/'native_summary.json.gz');manifest=read(folder/'manifest.json.gz')
    metrics=read(folder/'metrics.json');config=read(folder/'config.json')
    loads={r['request_id']:r['load'] for r in manifest['requests']}
    layers=[]
    for batch in native['microbatch_metrics']:
        rid=batch['member_request_ids'][0];load=loads[rid]
        barrier=batch['admission_time_ms']
        for l in batch['layer_metrics']:
            layers.append(dict(strategy=strategy,request_id=rid,npu=batch['npu_id'],
                               role=load['profile_group'],layer=l['layer'],
                               io_start_ms=l['io_start_time_ms'],io_ready_ms=l['io_ready_time_ms'],
                               read_latency_ms=l['io_ready_time_ms']-l['io_start_time_ms'],
                               C_ms=load['per_layer_us']/1000,
                               compute_start_ms=l['compute_start_ms'],compute_end_ms=l['compute_end_ms'],
                               stall_start_ms=barrier,stall_end_ms=l['compute_start_ms'],
                               exposed_stall_ms=max(0,l['compute_start_ms']-barrier)))
            barrier=l['compute_end_ms']
    a_layers=[l for l in layers if l['role']=='A']
    b_layers=[l for l in layers if l['role']=='B']
    evidence=[]
    for b in b_layers:
        if b['layer']==0 or not 2000<=b['io_start_ms']<10000:continue
        t=b['io_start_ms']
        prior=[a for a in a_layers if a['io_start_ms']<t<a['io_ready_ms']]
        row=dict(b,prior_unready_A_layers=len(prior),
                 prior_unready_A_layer0=sum(a['layer']==0 for a in prior),
                 prior_unready_A_internal=sum(a['layer']>0 for a in prior))
        evidence.append(row)
    windows=[]
    for name,lo,hi in [('warm',2000,4000),('long',2000,10000)]+[(f'bin{i}',i*1000,(i+2)*1000) for i in (2,4,6,8,10)]:
        compute={'A':0.0,'B':0.0};active={'A':0.0,'B':0.0};stall={g:{'internal':0.0,'layer0':0.0} for g in 'AB'}
        for l in layers:
            compute[l['role']]+=overlap(l['compute_start_ms'],l['compute_end_ms'],lo,hi)
            kind='internal' if l['layer'] else 'layer0'
            stall[l['role']][kind]+=overlap(l['stall_start_ms'],l['stall_end_ms'],lo,hi)
        for b in native['microbatch_metrics']:
            role=loads[b['member_request_ids'][0]]['profile_group']
            active[role]+=overlap(b['admission_time_ms'],b['completion_time_ms'],lo,hi)
        U=sum(compute.values())/(16*(hi-lo))*100
        if name=='warm':assert math.isclose(U,metrics['warm_U_percent'],abs_tol=1e-8)
        # A request-cohort average of its per-block queue statistics, weighted
        # by block count; all of each admitted request's blocks are retained.
        queue={}
        for role in 'AB':
            req=[r for r in native['request_metrics'] if lo<=r['admission_time_ms']<hi
                 and loads[r['request_id']]['profile_group']==role]
            n=sum(r['io_count'] for r in req)
            queue[role]=dict(request_count=len(req),io_count=n,
                block_weighted_ssd_queue_ms=sum(r['io_count']*r['avg_ssd_queue_wait_ms'] for r in req)/n if n else None,
                block_weighted_npu_link_queue_ms=sum(r['io_count']*r['avg_npu_link_queue_wait_ms'] for r in req)/n if n else None)
        bb=[l for l in evidence if lo<=l['io_start_ms']<hi]
        delayed=[l for l in bb if l['exposed_stall_ms']>1e-8]
        windows.append(dict(window=name,start_ms=lo,end_ms=hi,U_percent=U,
            group_U_percent={g:100*compute[g]/active[g] if active[g] else None for g in 'AB'},
            exposed_stall_card_ms=stall,request_cohort_block_queue=queue,
            B_internal_submissions=len(bb),B_internal_delayed=len(delayed),
            B_delayed_with_prior_A=sum(l['prior_unready_A_layers']>0 for l in delayed),
            B_internal_mean_R_ms=statistics.mean(l['read_latency_ms'] for l in bb) if bb else None,
            B_internal_max_R_ms=max((l['read_latency_ms'] for l in bb),default=None)))
    special=max((l for l in evidence if 2000<=l['io_start_ms']<4000),key=lambda l:l['exposed_stall_ms'])
    prior=[a for a in a_layers if a['io_start_ms']<special['io_start_ms']<a['io_ready_ms']]
    expected=(8*config['profile_A']['read_gib']+8*config['profile_B']['read_gib'])*2**30/1e6/40
    result=dict(strategy=strategy,input_fingerprint=native['input_fingerprint'],
                full_SSD_utilization_percent=100*native['ssd_mean_utilization'],
                full_nominal_peak_GB_s=metrics['full']['ordinary_demand']['max_single_ssu_gb_s'],
                full_nominal_overload_ms=metrics['full']['ordinary_demand']['any_ssu_overload_ms'],
                warm_SLO1p5_percent=metrics['warm_SLO_1p5_percent'],
                windows=windows,example=special,example_prior_A=prior,
                eight_A_plus_eight_B_disk_work_ms=expected,
                example_R_minus_eight_A_eight_B_ms=special['read_latency_ms']-expected,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [folder/'native_summary.json.gz',folder/'manifest.json.gz',folder/'metrics.json']})
    return result,evidence


def main():
    results=[];all_layers=[]
    for strategy in ('asu_baseline','once'):
        result,layers=analyze(strategy);results.append(result);all_layers.extend(layers)
    assert results[0]['input_fingerprint']==results[1]['input_fingerprint']
    with (HERE/'I1q8_hol_analysis_layers.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_layers[0]));writer.writeheader();writer.writerows(all_layers)
    (HERE/'I1q8_hol_analysis.json').write_text(json.dumps(dict(
        native_results=True,case=CASE.name,results=results,
        limitation='Native summary has no per-block service log; prior-A means earlier-submitted and not yet IO-ready, not an exact residual-byte FIFO reconstruction.'),indent=2)+'\n')
    lines=['# I1、请求数1:8：原生ASU短请求等待的证据','',
           '这份分析只读取已经完成的原生ASU和Once结果，不使用近似模型预测。两策略输入指纹相同。16卡、单盘40GB/s，A=200K/miss3302，B=32K/miss2448，8层。每卡周期为1个A、8个B，前8卡循环偏移4个请求，后8卡偏移0；每张卡都反复处理A和B。','',
           '| 策略 | [2,4) U | [2,10) U | warm SLO×1.5 | 全程单盘名义峰值GB/s | 全程过载ms |','|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['strategy']} | {r['windows'][0]['U_percent']:.4f}% | {r['windows'][1]['U_percent']:.4f}% | {r['warm_SLO1p5_percent']:.4f}% | {r['full_nominal_peak_GB_s']:.6f} | {r['full_nominal_overload_ms']:.6f} |")
    lines+=['','## 等待发生在哪里','',
            '| 策略/窗口 | A内部层等待(卡ms) | B内部层等待(卡ms) | A首层等待(卡ms) | B首层等待(卡ms) | B类别利用率 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in results:
        for w in r['windows'][:2]:
            a,b=w['exposed_stall_card_ms']['A'],w['exposed_stall_card_ms']['B']
            lines.append(f"| {r['strategy']}/{w['window']} | {a['internal']:.3f} | {b['internal']:.3f} | {a['layer0']:.3f} | {b['layer0']:.3f} | {w['group_U_percent']['B']:.3f}% |")
    r=results[0];example=r['example']
    lines+=['','ASU的主要损失是B后续层读取没有赶上上一层计算结束。Once大幅减少这类等待，同时可能增加A的首层等待；这是真实权衡，不能描述成对所有类别都零成本改善。','',
            '## 一个可以直接代数字的原生层','',
            f"ASU卡{example['npu']}，请求{example['request_id']}，层{example['layer']}（从0编号）：",'',
            f"- 发起读取：{example['io_start_ms']:.6f} ms；数据到齐：{example['io_ready_ms']:.6f} ms。",
            f"- 读取历时：{example['read_latency_ms']:.9f} ms。",
            f"- B上一层可用于预取的计算：{example['C_ms']:.9f} ms。",
            f"- 暴露等待：{example['exposed_stall_ms']:.9f} ms，即读取历时减去计算窗口。",
            f"- 发起时有{example['prior_unready_A_layers']}个更早发起、尚未IO就绪的A层；其中{example['prior_unready_A_layer0']}个是L0、{example['prior_unready_A_internal']}个是后续层。",'',
            f"同一轮8个A、8个B的总盘工作量为 `(8×283.709184 + 8×42.690560)/40 = {r['eight_A_plus_eight_B_disk_work_ms']:.9f} ms`，与这个B层读取历时数值吻合（差{r['example_R_minus_eight_A_eight_B_ms']:.3g} ms）。",'',
            '**不是A的113.966ms计算直接挡住B。** A计算发生在别的NPU；共享的是盘上的单条FIFO路径。此前提交的A读取与同批其他读取仍在服务，B的数据无法在17.147ms内到齐。','',
            '原生摘要没有逐块服务事件，因而上述“先发起尚未就绪A”不是精确的剩余字节队列重放，数值吻合也不应扩展为所有B层都严格按同一整层顺序排队。完整源数据和示例A层时间在JSON中。','',
            '## 重复出现，而不只是启动一次','',
            '| 策略 | [2,4) | [4,6) | [6,8) | [8,10) | [10,12) |','|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('| '+r['strategy']+' | '+' | '.join(f"{w['U_percent']:.3f}%" for w in r['windows'][2:])+' |')
    lines+=['','这里只验证到12秒，不能把有限时长当成无限稳态证明。但等待在后续多个2秒段持续出现，已经排除了只看启动瞬间的解释。','',
            '## 排队发生在盘侧还是NPU接收链路','',
            '以下使用窗口内上卡的完整B请求，按各请求IO块数加权其原生块排队统计；与窗口时间积分的口径不同。','',
            '| 策略/窗口 | B块平均SSD排队ms | B块平均NPU链路排队ms | 内部B层等待/提交 | 等待层中发起时存在更早未就绪A |','|---|---:|---:|---:|---:|']
    for r in results:
        for w in r['windows'][:2]:
            q=w['request_cohort_block_queue']['B']
            lines.append(f"| {r['strategy']}/{w['window']} | {q['block_weighted_ssd_queue_ms']:.6f} | {q['block_weighted_npu_link_queue_ms']:.6f} | {w['B_internal_delayed']}/{w['B_internal_submissions']} | {w['B_delayed_with_prior_A']}/{w['B_internal_delayed']} |")
    lines+=['','欠载定义是当前请求V/C之和低于40GB/s，不是任意瞬间没有字节突发。此处仍沿用历史约定，不把额外跨请求L0预取再计一份需求；而本例的A积压包含L0。这是机制中的实际变量，应明确披露，不能宣称已经排除了跨请求预取的作用。','',
            '[原始数字和示例](I1q8_hol_analysis.json) · [内部B层逐条证据](I1q8_hol_analysis_layers.csv) · [只读复算程序](I1q8_hol_analysis.py)','']
    (HERE/'I1q8_hol_analysis.md').write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
