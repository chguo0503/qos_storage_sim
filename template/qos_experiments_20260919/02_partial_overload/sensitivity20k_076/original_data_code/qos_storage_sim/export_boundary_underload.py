#!/usr/bin/env python3
"""Render frozen native results with an explicit S/L-boundary exemption.

No timeline is removed: total prefetch = continuing + boundary contributions.
The exemption changes the demand acceptance test, never utilization accounting.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
from PIL import Image

import export_strict_random as frozen
import export_random_multitype as inherited
import export_fixed128_32_results as shared
import audit_boundary_underload as boundary

read, write_json, write_csv = shared.read, shared.write_json, shared.write_csv
INK, MUTED = shared.INK, shared.MUTED
CURRENT, CONTINUING, TOTAL, BOUNDARY, PHYSICAL = '#8451a6', '#188369', '#d0711c', '#bb463a', '#0872cd'
KINDS = ('current', 'continuing', 'boundary', 'total_prefetch')


def event_data(case):
    """One exact event partition for all reference contributions and IO debt."""
    built = boundary.build(case['manifest'], case['raw'], case['metadata'])
    s, n = case['num_ssu'], case['metadata']['num_npu']
    width = 4*s+3*n+3
    offsets = {k:i*s for i,k in enumerate(KINDS)}
    offsets.update(total_link=4*s, continuing_link=4*s+n, boundary_link=4*s+2*n)
    events = defaultdict(lambda: [[] for _ in range(width)])
    end = case['raw']['summary']['makespan_ms']
    events[0.]; events[end]

    def add(a,b,offset,rates):
        if b <= a:
            return
        for i,rate in enumerate(rates):
            events[a][offset+i].append(rate)
            events[b][offset+i].append(-rate)

    for key,offset in offsets.items():
        for a,b,rates in built['intervals'][key]:
            add(a,b,offset,rates)
    for job in built['jobs']:
        if job['io_ready_ms'] > job['deadline_ms']:
            which = 0 if job['kind']=='cross_role_boundary' else 1
            add(job['deadline_ms'],job['io_ready_ms'],4*s+3*n+which,[1.])
    for job in built['cold']:
        add(job['release_ms'],job['ready_ms'],4*s+3*n+2,[1.])
    edges = sorted(events)
    active = [0.]*width
    matrix, rows = [], []
    for a,b in zip(edges,edges[1:]):
        active = [math.fsum([active[i],*events[a][i]]) for i in range(width)]
        active = [0. if abs(x)<1e-9 else x for x in active]
        assert min(active) >= -1e-7
        matrix.append(active.copy())
        row = dict(start_ms=a,end_ms=b)
        for key in KINDS:
            offset=offsets[key]
            row[f'fleet_{key}_gib_s']=math.fsum(active[offset:offset+s])
            row.update({f'ssu{i}_{key}_gib_s':active[offset+i] for i in range(s)})
        for key in ('total_link','continuing_link','boundary_link'):
            row[f'max_npu_{key}_gib_s']=max(active[offsets[key]:offsets[key]+n])
        row.update(boundary_overdue_layers=active[-3],continuing_overdue_layers=active[-2],cold_initial_layers=active[-1])
        rows.append(row)
    a=np.asarray(matrix)
    out=dict(edges=np.asarray(edges),rows=rows,built=built,
        boundary_overdue=a[:,-3],continuing_overdue=a[:,-2],cold=a[:,-1])
    out.update({key:a[:,offsets[key]:offsets[key]+s].T for key in KINDS})
    out.update({key:a[:,offsets[key]:offsets[key]+n].T for key in ('total_link','continuing_link','boundary_link')})
    assert np.allclose(out['total_prefetch'],out['continuing']+out['boundary'],atol=1e-7,rtol=0)
    return out


def heading(fig,case):
    fig.text(.060,.978,f'{shared.POLICY[case["policy"]]}：常规欠载与长短交界',fontsize=21,color=INK,va='top')
    fig.text(.060,.948,f'{case["name"]} · 8 NPU / {case["num_ssu"]} SSU × 40 GiB/s · ring hash · seed {case["metadata"]["seed"]}',fontsize=11.5,color=MUTED,va='top')
    for i,line in enumerate(inherited.profile_text_lines(case)):
        fig.text(.060,.924-i*.021,line,fontsize=10.5,color=MUTED,va='top')
    fig.text(.060,.863,f'warm [2,4)秒：NPU U={case["metrics"]["U_percent"]:.2f}% · 交界处只豁免需求约束，全部 IO Stall 仍计入利用率。',fontsize=11,color=INK,va='top')


def draw_bandwidth(case,event,physical,path):
    s=case['num_ssu']
    # Select using all-run total-prefetch peak, not a visually favorable slice.
    hot=int(np.argmax(np.max(event['total_prefetch'],axis=1)))
    count=4 if s==1 else 6
    fig,axes=plt.subplots(count,1,figsize=(19,3.0*count+4),dpi=150)
    fig.subplots_adjust(left=.090,right=.965,top=.815,bottom=.100,hspace=.71)
    heading(fig,case)
    x=event['edges']/1000
    stats=[]

    def demand(ax,disk=None):
        label='整机' if disk is None else f'SSU {disk}（总预取峰值最高）'
        cap=40*s if disk is None else 40.
        curves={key:np.sum(event[key],axis=0) if disk is None else event[key][disk] for key in KINDS}
        ax.axhspan(.9*cap,cap,color='#f5dd98',alpha=.26)
        for key,color,title,lw in [('current',CURRENT,'当前请求 V/C',1.),('continuing',CONTINUING,'非交界预取',1.05),('total_prefetch',TOTAL,'总预取（含交界）',1.0)]:
            ax.stairs(curves[key],x,color=color,lw=lw,label=title)
        ax.axhline(cap,color='#303942',ls='--',lw=1.1,label=f'容量 {cap:g}')
        ax.set_ylim(0,max(cap,max(curves['current']),max(curves['total_prefetch']))*1.19)
        ax.set_xlim(x[0],x[-1]);ax.set_xlabel('仿真时间（秒）')
        ax.set_title(f'{label} · 全部输入 · 峰值：当前 {max(curves["current"]):.2f} / 非交界 {max(curves["continuing"]):.2f} / 总预取 {max(curves["total_prefetch"]):.2f}',loc='left',fontsize=11)
        ax.legend(loc='upper right',ncol=4,fontsize=9,framealpha=.95)
        stats.append(dict(scope=label,capacity_gib_s=cap,**{key:frozen.curve_stats(val,event['edges'],cap) for key,val in curves.items()}))

    demand(axes[0])
    cursor=1
    if s>1:
        demand(axes[cursor],hot);cursor+=1
    ax=axes[cursor];cursor+=1
    v=np.sum(event['boundary'],axis=0)
    ax.stairs(v,x,color=BOUNDARY,lw=1.,fill=True,alpha=.6,label='仅长短交界预取贡献')
    ax.axhline(40*s,color='#303942',ls='--',lw=1.,label=f'整机容量 {40*s:g}')
    ax.set_ylim(0,max(40*s,max(v))*1.19);ax.set_xlim(x[0],x[-1]);ax.set_xlabel('仿真时间（秒）')
    ax.set_title('交界贡献单列 · 总预取 = 非交界预取 + 交界预取',loc='left',fontsize=11)
    ax.legend(loc='upper right',ncol=2,fontsize=9)
    px=physical['edges']/1000

    def actual(ax,disk=None):
        v=np.sum(physical['physical'],axis=0) if disk is None else physical['physical'][disk]
        cap=40*s if disk is None else 40.
        assert max(v)<=cap+1e-6
        label='整机' if disk is None else f'SSU {disk}'
        ax.stairs(v,px,color=PHYSICAL,lw=1.,label='实际 SSD 服务（公共 2 ms）')
        ax.axhline(cap,color='#303942',ls='--',lw=1.,label=f'物理容量 {cap:g}')
        ax.set(xlim=(2,4),ylim=(0,cap*1.18),xlabel='仿真时间（秒）')
        ax.set_title(f'{label} · warm 实际服务 · 峰值 {max(v):.2f} / 平均 {np.mean(v):.2f}',loc='left',fontsize=11)
        ax.legend(loc='upper right',ncol=2,fontsize=9)

    actual(axes[cursor]);cursor+=1
    if s>1:
        actual(axes[cursor],hot);cursor+=1
    ax=axes[cursor]
    ax.stairs(physical['stalled'],px,color='#ba850f',fill=True,alpha=.75,label='等待 IO 的平均 NPU 数 / 2 ms')
    ax.set(xlim=(2,4),ylim=(0,8.5),yticks=[0,2,4,6,8],xlabel='仿真时间（秒）')
    ax.set_title('warm 等待 · 交界及非交界 Stall 全部保留',loc='left',fontsize=11)
    ax.legend(loc='upper right',fontsize=9)
    for axis in axes:
        axis.set_ylabel('带宽（GiB/s）');axis.grid(alpha=.15);axis.spines[['top','right']].set_visible(False)
        limits=axis.get_xlim()
        axis.set_xticks(np.linspace(limits[0],limits[1],9))
        axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel('NPU 数')
    fig.text(.060,.063,'上方需求按精确事件绘制；下方实际服务按公共2 ms统计。需求允许交界峰值，实际 SSD 服务始终受物理容量限制。',fontsize=10.5,color=INK)
    fig.text(.060,.039,'预取参考在计算窗口结束时停止；未完成 IO 和暴露等待仍保留在审计与时间线中。单卡50 GiB/s接收下限另见逐层数据。',fontsize=10.5,color=MUTED)
    result=inherited.save_checked(fig,path)
    result.update(full_demand_window_ms=[0.,event['edges'][-1]],physical_window_ms=[2000.,4000.],physical_bin_ms=2.,
        demand_event_exact=True,no_time_intervals_removed=True,selected_hot_ssu=hot,demand_stats=stats,
        physical_audit=physical['audit'])
    return result


def draw_timebill(case,evidence,path):
    """Show observed service order and the selected short layer's exact bill."""
    assert all(evidence['validation_checks'].values())
    assert Path(evidence['case']).resolve()==case['folder']
    for name,expected in evidence['source_sha256'].items():
        assert hashlib.sha256((case['folder']/name).read_bytes()).hexdigest()==expected
    target=evidence['target_layer'];bill=evidence['time_accounting']
    sequence=evidence['verified_contiguous_ssd_sequence']
    assert not evidence['cross_role_boundaries_active_in_target_read_window']
    assert len(evidence['preceding_long_layers'])==2
    origin=target['io_release_ms'];relative=lambda value:value-origin
    deadline=relative(target['compute_deadline_ms']);ready=relative(target['io_ready_ms'])
    fig,ax=plt.subplots(figsize=(18,8),dpi=150)
    fig.subplots_adjust(left=.11,right=.965,top=.68,bottom=.25)
    fig.text(.055,.951,'真实 FIFO 局部：两层大读取挡住了后续短层',fontsize=22,color=INK)
    fig.text(.055,.902,f'8 NPU / 1 SSU × 40 GiB/s · path0 FIFO · seed {case["metadata"]["seed"]} · 常规需求 {evidence["local_demand_references"]["current"]["fleet_peak_gib_s"]:.3f} GiB/s',fontsize=12,color=MUTED)
    fig.text(.055,.857,f'目标 NPU {target["npu_id"]} / 请求 {target["request_id"]} / L{target["layer"]}：{target["total_k"]:g}K，NQL={target["nql"]}，每层 {target["bytes_gib"]*1024:.3f} MiB',fontsize=12,color=INK)
    fig.text(.055,.810,f'读取含排队 {bill["total_read_latency_ms"]:.3f} ms · 计算掩盖 {bill["hidden_by_current_compute_ms"]:.3f} ms · 暴露 Stall {bill["exposed_stall_ms"]:.3f} ms',fontsize=13,color=INK)
    def bar(a,b,y,color,text=None,textcolor=INK):
        ax.broken_barh([(a,b-a)],(y-.20,.40),facecolors=color,edgecolor='white',linewidth=.8)
        if text:ax.text((a+b)/2,y,text,ha='center',va='center',fontsize=10,color=textcolor)
    bar(0.,deadline,1,inherited.GREEN,f'L5 计算 {deadline:.3f} ms')
    bar(deadline,ready,1,inherited.YELLOW,f'等待 L6 数据 {ready-deadline:.3f} ms')
    bar(0.,relative(sequence[0]['first_ssd_start_ms']),0,'#d7dfe6','前序\nIO')
    for item in sequence:
        a,b=relative(item['first_ssd_start_ms']),relative(item['last_ssd_end_ms'])
        if item['role']=='L':
            bar(a,b,0,'#2166aa',f'NPU {item["npu_id"]} / L{item["layer"]} · 200K\n读 {b-a:.3f} ms','white')
        else:
            is_target=item['request_id']==target['request_id'] and item['layer']==target['layer']
            bar(a,b,0,'#219b9b' if is_target else '#9acbe1','目标' if is_target else '短')
            ax.text((a+b)/2,-.39,f'NPU {item["npu_id"]}\nL{item["layer"]}',fontsize=9,ha='center',va='top',color=INK)
    ax.axvline(deadline,color='#b83738',ls='--',lw=1.3,ymin=.17,ymax=.94)
    ax.text(deadline,1.46,f'短层截止\n{target["compute_deadline_ms"]:.3f} ms',ha='center',fontsize=10,color='#b83738')
    ax.annotate(f'数据到齐\n{target["io_ready_ms"]:.3f} ms',xy=(ready,1.05),xytext=(ready-1.7,1.46),
        ha='center',fontsize=10,color=INK,arrowprops=dict(arrowstyle='->',color=MUTED))
    ax.set(xlim=(-.15,ready+.35),ylim=(-.8,1.8),yticks=[0,1],yticklabels=['SSU 0\npath0 FIFO',f'NPU {target["npu_id"]}\nL{target["current_compute_layer"]} → L{target["layer"]}'],
        xticks=np.arange(0,math.ceil(ready)+1,2),xlabel=f'相对时间（ms）；0 = 原始仿真 {origin:.6f} ms')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.16);ax.tick_params(axis='y',length=0)
    fig.text(.055,.163,f'时间账：前方排队 {bill["waiting_before_first_ssd_ms"]:.3f} + 自身SSD读取 {bill["continuous_own_ssd_service_ms"]:.3f} + 链路尾部 {bill["final_link_tail_ms"]:.3f} = {bill["total_read_latency_ms"]:.3f} ms',fontsize=12,color=INK)
    fig.text(.055,.111,f'该层周期利用率 {bill["single_layer_cycle_utilization_percent"]:.2f}%；完整warm整机利用率 {case["metrics"]["U_percent"]:.2f}%。局部选取短内部层Stall最大者，不能当作平均。',fontsize=11,color=MUTED)
    fig.text(.055,.065,f'本段没有长短交界预取或其逾期IO；所有盘服务段均由真实字节与时间戳核对，蓝色两长层合计占用{bill["two_contiguous_long_layers_service_ms"]:.3f} ms。',fontsize=11,color=MUTED)
    info=inherited.save_checked(fig,path)
    info.update(evidence_sha256=hashlib.sha256(shared.json.dumps(evidence,sort_keys=True).encode()).hexdigest(),
        target_request_id=target['request_id'],target_layer=target['layer'],absolute_origin_ms=origin,
        relative_time_axis=True,counterfactual_drawn=False,all_evidence_checks_passed=True)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--font',type=Path,required=True)
    parser.add_argument('--timebill',type=Path,help='Optional audited native FIFO local time bill JSON')
    args=parser.parse_args()
    for name in ('images','data'):(args.out/name).mkdir(parents=True,exist_ok=True)
    fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),'axes.unicode_minus':False,
        'font.size':11,'figure.facecolor':'white','path.simplify':False})
    cases=[frozen.load_case(p) for p in args.case]
    assert len({c['stem'] for c in cases})==len(cases)
    groups=defaultdict(list);figures=[];allrows=[];aggregate=[];points=[];summaries=[]
    for case in cases:
        stem=case['stem'];groups[case['group']].append(case)
        event=event_data(case);physical=shared.physical_data(case)
        ident=case['folder'].name+'_'+hashlib.sha256(str(case['folder']).encode()).hexdigest()[:8]
        audit_path=args.out/'data'/'audits'/f'{ident}_audit.json'
        audited=read(audit_path) if audit_path.exists() else None
        if audited is not None:
            valid=all(hashlib.sha256((case['folder']/name).read_bytes()).hexdigest()==expected
                for name,expected in audited['source_sha256'].items())
            valid=valid and audit_path.stat().st_mtime>=Path(boundary.__file__).stat().st_mtime
            if not valid:audited=None
        if audited is None:audited=boundary.audit(case['folder'],out=args.out/'data')
        for key in KINDS:
            assert np.allclose(np.max(event[key],axis=1),audited['windows']['full_run'][key]['peak_per_unit_gib_s'],atol=1e-7)
        figures.append(draw_bandwidth(case,event,physical,args.out/'images'/f'{stem}_boundary_bandwidth.png'))
        figures.append(inherited.draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline.png'))
        figures.append(inherited.draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline_zoom.png',True))
        write_csv(args.out/'data'/f'{stem}_demand_event_segments.csv',event['rows'])
        write_csv(args.out/'data'/f'{stem}_physical_2ms.csv',physical['rows'])
        write_csv(args.out/'data'/f'{stem}_requests.csv',case['rows'])
        write_csv(args.out/'data'/f'{stem}_layer_timing_full_run.csv',frozen.layer_rows(case))
        write_csv(args.out/'data'/f'{stem}_layer_cycles.csv',case['analysed']['cycles'])
        write_json(args.out/'data'/f'{stem}_prefetch_jobs_full_run.json',event['built']['jobs'])
        write_json(args.out/'data'/f'{stem}_physical_audit.json',physical['audit'])
        allrows.extend(case['rows']);aggregate.extend(inherited.profile_aggregates(case,case['rows']))
        full=audited['windows']['full_run'];warm=audited['windows']['warm_2000_4000ms']
        summaries.append(dict(name=case['name'],policy=case['policy'],seed=case['metadata']['seed'],num_ssu=case['num_ssu'],
            input_fingerprint=case['fingerprint'],U_percent=case['metrics']['U_percent'],
            warm_slo_1p5_percent=case['metrics']['slo_1p5_percent'],
            full_slo_1p5_percent=100*sum(r['slo_1p5_passed'] for r in case['rows'])/len(case['rows']),
            current_peak_max_ssu_gib_s=max(full['current']['peak_per_unit_gib_s']),
            continuing_peak_max_ssu_gib_s=max(full['continuing']['peak_per_unit_gib_s']),
            total_prefetch_peak_max_ssu_gib_s=max(full['total_prefetch']['peak_per_unit_gib_s']),
            warm_internal_stall_card_ms=warm['utilization']['internal_stall_card_ms'],
            warm_boundary_stall_card_ms=warm['job_stats']['cross_role_boundary']['actual_stall_card_ms'],
            warm_boundary_unavoidable_lower_bound_card_ms=warm['job_stats']['cross_role_boundary']['unavoidable_transfer_stall_lower_bound_card_ms'],
            boundary_exempt_candidate_pass=audited['boundary_exempt_candidate_pass'],request_count=len(case['rows'])))
    for group,items in groups.items():
        ids=[{r['request_id'] for r in c['rows']} for c in items]
        assert all(x==ids[0] for x in ids)
        for cohort in ('warm_admissions','all_requests'):
            figures.append(inherited.draw_cdf(group,items,cohort,args.out/'images'/f'{group}_ttft_slo15_{cohort}.png',points))
        fields=['configuration','scenario_name','seed','npu_id','request_id','input_order_1based','profile_group','profile_tag',
            'role','total_tokens','total_length_k','nql_tokens','hit_prefix_tokens','per_layer_read_gib','per_layer_read_mib',
            'per_layer_compute_ms','bandwidth_demand_gib_s','compute_method','compute_extrapolated','single_ssu_40gib_s_transfer_ms']
        write_csv(args.out/'data'/f'{group}_input_profiles.csv',[{k:r[k] for k in fields} for r in items[0]['rows']])
    if args.timebill:
        evidence=read(args.timebill)
        selected=[c for c in cases if c['folder']==Path(evidence['case']).resolve()]
        assert len(selected)==1,'Time-bill case must be present in --case inputs'
        figures.append(draw_timebill(selected[0],evidence,args.out/'images'/'fifo_local_timebill.png'))
        write_json(args.out/'data'/'local_fifo_timebill.json',evidence)
    write_csv(args.out/'data'/'all_request_execution.csv',allrows)
    write_csv(args.out/'data'/'profile_group_summary.csv',aggregate)
    write_csv(args.out/'data'/'ttft_cdf_points.csv',points)
    write_json(args.out/'data'/'scenario_summary.json',summaries)
    write_csv(args.out/'data'/'scenario_summary.csv',summaries)
    for info in figures:
        path=args.out/'images'/info['file']
        with Image.open(path) as png:png.load()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    write_json(args.out/'data'/'figure_audit.json',dict(figures=figures,all_checks_passed=True,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_simulation_executed=True,
        dependency_sha256={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (frozen,inherited,shared,boundary)}))
    (args.out/'README.md').write_text('''# 长短交界豁免：冻结结果图表

8 NPU，每卡独立随机混合全部画像，ring hash；同一KV block所有层在同一SSU。

- `boundary_bandwidth.png`：完整轨迹的当前请求V/C、非交界预取、总预取。只有S→L或L→S下一请求L0的参考需求贡献属于交界；同角色跨请求不豁免。S/L是输入画像的长短标签，不等于原生QoS分类SS/LL。
- 所有事件区间均保留。总预取严格等于非交界预取加交界贡献，不删除“有交界发生”的整段时间。常规欠载需单独检查当前与非交界需求，图中不把总预取峰值限幅。
- 真实SSD服务使用公共2ms字节桶。每盘物理服务≤40 GiB/s；整机≤40×SSU数。多SSU时仅绘制整机和全程总预取峰值最高的SSU，所有盘数据仍在CSV中。
- 图末行和8NPU时间线包含全部交界、非交界等待，NPU利用率统计完整warm [2,4)秒。缩放图选Stall最多的250ms，只用于观察。
- 预取参考仅在前序计算窗口存在；过期但未完成的IO不会被删除，详情保留在prefetch_jobs_full_run及audits中。首批L0没有前序预取窗口。
- `input_profiles.csv`给出每请求总长、NQL、V、C、V/C与独占单盘40 GiB/s纯传输时间；`layer_timing_full_run.csv`给出实测含排队读取时间、计算时间与暴露Stall。
- 链路下限采用max(max_s V_s/40, sum_s V_s/50)-C的正部。单卡50 GiB/s所必需的交界等待不能直接归因于FIFO；超出这个下限的部分也需要策略对照或IO证据才可确认为FIFO损失。
- TTFT从NPU接纳到请求完成，不含接纳前队列等待；SLO阈值为8层纯计算时间×1.5。全部输入CDF对齐请求ID，warm接纳集合可能随策略不同。
- 此脚本只读取已有原生结果，不启动模拟器，不修改输入或原生源模型。
''',encoding='utf-8')
    print(shared.json.dumps(dict(output=str(args.out.resolve()),cases=len(cases),figures=len(figures),all_checks_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
