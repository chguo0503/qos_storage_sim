#!/usr/bin/env python3
"""Export frozen native runs with event-exact demand curves and true SSD traffic.

This renderer never starts a simulator. Demand references are drawn as exact
steps over the full native makespan; actual SSD service uses common 2-ms bins.
The separate strict audit is the authority for full-trajectory acceptance.
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

import export_random_multitype as inherited
import export_fixed128_32_results as shared
import plot_fifo_mixed8 as accounting
import audit_strict_random as strict

read, write_json, write_csv = shared.read, shared.write_json, shared.write_csv
save_checked = inherited.save_checked
INK, MUTED = shared.INK, shared.MUTED
POLICY = shared.POLICY
CURRENT, PREFETCH, PHYSICAL = '#8e329e', '#cb7619', '#0872cd'
LEFT = 0.


def load_case(folder):
    folder = folder.resolve()
    metadata = read(folder/'metadata.json')
    assert metadata['order'] == 'random'
    assert metadata['num_npu'] == 8
    assert metadata['n_layers'] == 8
    analysed = accounting.analyse(folder)
    name = metadata['case']['name']
    fp = metadata['input_fingerprint']
    group = f'{name}_seed{metadata["seed"]}_s{metadata["num_ssu"]}_{fp[:8]}'
    tags = {f'G{i+1}': gid for i, gid in enumerate(metadata['profiles'])}
    case = dict(folder=folder, name=name, analysed=analysed,
        manifest=read(folder/'manifest.json.gz'), metadata=metadata,
        metrics=read(folder/'metrics.json'), receipt=read(folder/'receipts.json'),
        raw=read(folder/'result.json.gz'), group=group, fingerprint=fp,
        num_ssu=metadata['num_ssu'], tags=tags, group_tags={v:k for k,v in tags.items()})
    assert case['metrics']['window_ms'] == [2000., 4000.]
    case['policy'] = case['metrics']['policy']
    case['stem'] = f'{group}_{case["policy"]}'
    case['rows'] = inherited.request_rows(case)
    for row in case['rows']:
        row['single_ssu_40gib_s_transfer_ms']=row['per_layer_read_gib']/40*1000
    case['stall_intervals'] = []
    for lane in analysed['lanes']:
        for batch in lane['batches']:
            previous = batch['admission_ms']
            for layer in batch['layers']:
                case['stall_intervals'].append((previous, layer['compute_start_ms']))
                previous = layer['compute_end_ms']
    audit_path = folder/'strict_audit.json'
    case['strict_audit'] = read(audit_path) if audit_path.exists() else None
    return case


def event_data(case, left=LEFT, right=None):
    """Preserve the strict auditor's full-precision intervals without binning."""
    if right is None:right=case['raw']['summary']['makespan_ms']
    s, n = case['num_ssu'], case['metadata']['num_npu']
    width = s*2+n+2
    events = defaultdict(lambda: [[] for _ in range(width)])
    events[left]; events[right]
    intervals = strict.build_full_intervals(case['manifest'],case['raw'],case['metadata'])

    def add(start, end, offset, rates):
        start, end = max(start, left), min(end, right)
        if end <= start:
            return
        for i, rate in enumerate(rates):
            events[start][offset+i].append(rate)
            events[end][offset+i].append(-rate)

    for key,offset in [('current',0),('prefetch',s),('link',2*s)]:
        for start,end,rates in intervals[key]:add(start,end,offset,rates)
    for job in intervals['missed_jobs']:add(job['deadline_ms'],job['io_ready_ms'],2*s+n,[1.])
    for job in intervals['cold_jobs']:add(job['io_start_time_ms'],job['io_ready_time_ms'],2*s+n+1,[1.])
    times = sorted(events)
    active = [0.]*width
    matrix = []
    rows = []
    for start,end in zip(times,times[1:]):
        active = [math.fsum([active[i],*events[start][i]]) for i in range(width)]
        assert min(active) >= -1e-7
        # Remove only floating-point cancellation noise, never cap a peak.
        active = [0. if abs(x)<1e-10 else x for x in active]
        matrix.append(active.copy())
        row = dict(start_ms=start,end_ms=end,
            fleet_current_gib_s=math.fsum(active[:s]),
            fleet_prefetch_gib_s=math.fsum(active[s:2*s]),
            max_npu_prefetch_gib_s=max(active[2*s:2*s+n]),
            overdue_layer_count=active[2*s+n],cold_initial_layer_count=active[2*s+n+1])
        row.update({f'ssu{i}_current_gib_s':active[i] for i in range(s)})
        row.update({f'ssu{i}_prefetch_gib_s':active[s+i] for i in range(s)})
        row.update({f'npu{i}_prefetch_gib_s':active[2*s+i] for i in range(n)})
        rows.append(row)
    matrix = np.asarray(matrix)
    return dict(edges=np.asarray(times),current=matrix[:,:s].T,
        prefetch=matrix[:,s:2*s].T,npu=matrix[:,2*s:2*s+n].T,rows=rows,
        overdue=matrix[:,2*s+n],cold=matrix[:,2*s+n+1],
        missed_jobs=intervals['missed_jobs'],cold_jobs=intervals['cold_jobs'])


def curve_stats(values, edges, capacity):
    durations = np.diff(edges)
    return dict(peak_gib_s=float(np.max(values)),
        near90_percent=float(100*np.sum(durations[values>=.9*capacity-1e-8])/sum(durations)),
        over_capacity_ms=float(np.sum(durations[values>capacity+1e-8])),
        weighted_mean_gib_s=float(np.dot(values,durations)/sum(durations)))


def layer_rows(case):
    """Full-run observed layer timing, including cold L0 and request boundaries."""
    qs={q['request_id']:q['load'] for q in case['manifest']['requests']}
    rows=[]
    for batch in case['raw']['summary']['microbatch_metrics']:
        rid=batch['member_request_ids'][0];q=qs[rid]
        previous=batch['admission_time_ms']
        for layer in sorted(batch['layer_metrics'],key=lambda x:x['layer']):
            start=layer['io_start_time_ms'];ready=layer['io_ready_time_ms']
            rows.append(dict(scenario_name=case['name'],policy=case['policy'],seed=case['metadata']['seed'],
                request_id=rid,npu_id=batch['npu_id'],profile_group=q['profile_group'],
                total_tokens=q['total_tokens'],nql_tokens=q['nql'],layer=layer['layer'],
                io_start_time_ms=start,io_ready_time_ms=ready,
                actual_read_span_including_queue_ms=ready-start,
                compute_start_ms=layer['compute_start_ms'],compute_end_ms=layer['compute_end_ms'],
                exposed_io_stall_ms=max(0.,layer['compute_start_ms']-previous),
                per_layer_read_gib=q['per_layer_kv_gb'],
                single_ssu_40gib_s_transfer_ms=q['per_layer_kv_gb']/40*1000))
            previous=layer['compute_end_ms']
    return sorted(rows,key=lambda r:(r['npu_id'],r['io_start_time_ms'],r['layer']))


def heading(fig, case, title, description):
    fig.text(.055,.975,f'{POLICY[case["policy"]]}：{title}',fontsize=21,color=INK,va='top')
    fig.text(.055,.946,
        f'{case["name"]} · 8 NPU / {case["num_ssu"]} SSU × 40 GiB/s · ring hash · seed {case["metadata"]["seed"]}',
        fontsize=11.5,color=MUTED,va='top')
    for i,line in enumerate(inherited.profile_text_lines(case)):
        fig.text(.055,.922-i*.022,line,fontsize=10.5,color=MUTED,va='top')
    fig.text(.055,.850,description,fontsize=10.5,color=INK,va='top')


def draw_demand(case, data, path):
    s = case['num_ssu']
    count = s+3
    fig,axes = plt.subplots(count,1,figsize=(19,3.0*count+5.0),sharex=True,dpi=150)
    fig.subplots_adjust(left=.085,right=.965,top=.815,bottom=.115,hspace=.62)
    right=data['edges'][-1]
    heading(fig,case,f'逐事件需求与容量（全部输入 0–{right/1000:.3f} 秒）',
        '不对需求分桶、平滑或限幅；峰值和接近容量占比均按精确事件区间计算。阴影带为容量的 90%–100%。')
    x = data['edges']/1000
    stats = []

    def panel(ax,current,prefetch,capacity,label):
        own = curve_stats(current,data['edges'],capacity)
        future = curve_stats(prefetch,data['edges'],capacity)
        stats.append(dict(scope=label,current=own,prefetch=future,capacity_gib_s=capacity))
        ax.axhspan(.9*capacity,capacity,color='#f4dc8d',alpha=.28,zorder=0)
        ax.stairs(current,x,color=CURRENT,lw=1.0,label='当前请求 V / 自身 C',zorder=3)
        ax.stairs(prefetch,x,color=PREFETCH,lw=1.0,label='下一层 V / 当前层 C',zorder=4)
        ax.axhline(capacity,color='#333',ls='--',lw=1.05,label=f'容量 {capacity:g}')
        ax.axhline(.9*capacity,color='#ba963d',ls=':',lw=1.05,label=f'90%容量 {capacity*.9:g}')
        for curve,color in [(current,CURRENT),(prefetch,PREFETCH)]:
            i=int(np.argmax(curve))
            ax.plot((x[i]+x[i+1])/2,curve[i],marker='o',markersize=3.8,color=color,zorder=6)
        peak=max(capacity,float(max(current)),float(max(prefetch)))
        ax.set_ylim(0,peak*1.18)
        ax.set_title(f'{label} · 当前峰值 {own["peak_gib_s"]:.2f} / 预取峰值 {future["peak_gib_s"]:.2f} GiB/s'
            f' · ≥90%容量 {own["near90_percent"]:.2f}% / {future["near90_percent"]:.2f}%',
            loc='left',fontsize=11,pad=9)
        ax.legend(loc='upper right',ncol=4,fontsize=8.5,framealpha=.94)
        ax.set_ylabel('GiB/s')

    panel(axes[0],np.sum(data['current'],axis=0),np.sum(data['prefetch'],axis=0),40*s,'整机')
    for disk in range(s):
        panel(axes[disk+1],data['current'][disk],data['prefetch'][disk],40,f'SSU {disk}')
    ax=axes[-2]
    maximum=np.max(data['npu'],axis=0)
    link_stats=curve_stats(maximum,data['edges'],50.)
    ax.stairs(maximum,x,color='#277655',lw=1.05,label='当时最高单卡预取需求')
    ax.axhline(50,color='#333',ls='--',lw=1.05,label='每卡接收上限 50 GiB/s')
    ax.set_ylim(0,max(50.,max(maximum))*1.18)
    ax.set_title(f'NPU 接收约束 · 单卡预取需求峰值 {link_stats["peak_gib_s"]:.2f} GiB/s'
        f' · 超过 50 的时间 {link_stats["over_capacity_ms"]:.3f} ms',loc='left',fontsize=11,pad=9)
    ax.legend(loc='upper right',ncol=2,fontsize=9)
    ax.set_ylabel('GiB/s')
    ax=axes[-1]
    ax.stairs(data['overdue'],x,color='#b63932',lw=1.4,label='截止已过但 IO 未完成的层数')
    ax.stairs(data['cold'],x,color='#6f7783',lw=1.2,ls='--',label='冷启动首层（无预取截止窗口）')
    ax.set(ylim=(-.25,8.4),yticks=[0,2,4,6,8],ylabel='层数',xlabel='仿真时间（秒）')
    ax.set_title(f'未完成读取核对 · 逾期层总数 {len(data["missed_jobs"])}'
        f' · 逾期并发峰值 {int(max(data["overdue"]))} · 冷启动首层 {len(data["cold_jobs"])}',
        loc='left',fontsize=11,pad=9)
    ax.legend(loc='upper right',ncol=2,fontsize=9)
    for ax in axes:
        ax.set_xlim(0,right/1000);ax.set_xticks(np.linspace(0,right/1000,9))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.text(.055,.072,'当前需求含请求的等待区间；预取参考仅在计算窗口存在，包含跨请求首层。冷启动首层没有前序计算窗口。',fontsize=10.5,color=INK)
    fig.text(.055,.049,'两种需求都是参考速率，不是实际 SSD 服务速率。预取窗口结束后需求曲线不再累计，未完成任务仍会计入底部逾期层数。',fontsize=10.5,color=MUTED)
    fig.text(.055,.027,'图中单卡预取峰值超过50时，即使逐盘欠载，也可能受NPU接收链路限制；不能把这种等待归因于FIFO。',fontsize=10.5,color=MUTED)
    audit=save_checked(fig,path)
    audit.update(event_exact=True,demand_bin_width_ms=None,display_window_ms=[LEFT,right],
        demand_stats=stats,npu_link_stats=link_stats,missed_io_deadline_job_count=len(data['missed_jobs']),
        max_concurrent_overdue_layers=int(max(data['overdue'])))
    return audit


def draw_physical(case, data, path):
    s=case['num_ssu'];count=s+2
    fig,axes=plt.subplots(count,1,figsize=(19,2.8*count+5.0),sharex=True,dpi=150)
    fig.subplots_adjust(left=.085,right=.965,top=.810,bottom=.11,hspace=.61)
    m=case['metrics']
    heading(fig,case,'实际 SSD 读取与 IO Stall（完整 warm 2–4 秒）',
        f'整机 NPU 利用率 {m["U_percent"]:.2f}% · 实际蓝线 = 同一个 2 ms 窗口内的真实服务字节 / 2 ms，不做限幅。')
    x=data['edges']/1000

    def panel(ax,values,cap,label):
        ax.stairs(values,x,color=PHYSICAL,lw=1.0,label='实际 SSD 服务 / 公共 2 ms')
        ax.axhline(cap,color='#333',ls='--',lw=1.1,label=f'物理容量 {cap:g} GiB/s')
        ax.set_ylim(0,cap*1.17)
        ax.set_title(f'{label} · 实际峰值 {max(values):.2f} GiB/s · 平均 {np.mean(values):.2f} GiB/s',
            loc='left',fontsize=11,pad=9)
        ax.set_ylabel('GiB/s');ax.legend(loc='upper right',ncol=2,fontsize=9)
    panel(axes[0],np.sum(data['physical'],axis=0),40*s,'整机')
    for disk in range(s):panel(axes[disk+1],data['physical'][disk],40,f'SSU {disk}')
    ax=axes[-1]
    ax.stairs(data['stalled'],x,color='#b77a06',fill=True,alpha=.7)
    ax.set(ylim=(0,8.4),yticks=[0,2,4,6,8],ylabel='NPU 数',xlabel='仿真时间（秒）')
    ax.set_title('同一个 2 ms 窗口内等待 IO 的平均 NPU 数',loc='left',fontsize=11,pad=9)
    for ax in axes:
        ax.set_xlim(2,4);ax.set_xticks(np.arange(2,4.01,.25))
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.text(.055,.064,'SSD 有任务时按40 GiB/s服务；实际服务触顶表示盘正在工作，不表示带宽需求超载。欠载请看单独的逐事件需求图。',fontsize=10.5,color=INK)
    fig.text(.055,.036,'整机容量为 SSU数 × 40 GiB/s。每盘及整机的实际服务都使用相同时间桶直接相加，图中没有突破容量的截断或修饰。',fontsize=10.5,color=MUTED)
    result=save_checked(fig,path)
    result.update(actual_service_bin_width_ms=2.,display_window_ms=[2000.,4000.],physical_audit=data['audit'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--font',type=Path,required=True)
    args=parser.parse_args()
    for name in ('images','data'):(args.out/name).mkdir(parents=True,exist_ok=True)
    fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),
        'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white','path.simplify':False})
    cases=[load_case(p) for p in args.case]
    assert len({c['stem'] for c in cases})==len(cases)
    groups=defaultdict(list)
    figures=[];allrows=[];aggregates=[];points=[];summaries=[]
    for case in cases:
        groups[case['group']].append(case)
        stem=case['stem'];events=event_data(case);physical=shared.physical_data(case)
        if case['strict_audit']:
            reference=case['strict_audit']['windows']['full_run']
            for key in ['current','prefetch']:
                assert np.allclose(np.max(events[key],axis=1),reference[key]['peak_per_unit_gib_s'],atol=1e-7)
            assert max(events['overdue'])==reference['overdue_io_debt']['max_concurrent_jobs']
        figures.append(draw_demand(case,events,args.out/'images'/f'{stem}_strict_demand_events.png'))
        figures.append(draw_physical(case,physical,args.out/'images'/f'{stem}_physical_ssd_bandwidth.png'))
        figures.append(inherited.draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline.png'))
        figures.append(inherited.draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline_zoom.png',True))
        write_csv(args.out/'data'/f'{stem}_demand_event_segments.csv',events['rows'])
        write_csv(args.out/'data'/f'{stem}_prefetch_warm_jobs.csv',physical['jobs'])
        write_json(args.out/'data'/f'{stem}_missed_deadline_jobs.json',events['missed_jobs'])
        write_json(args.out/'data'/f'{stem}_cold_initial_jobs.json',events['cold_jobs'])
        write_csv(args.out/'data'/f'{stem}_physical_2ms.csv',physical['rows'])
        write_csv(args.out/'data'/f'{stem}_requests.csv',case['rows'])
        write_csv(args.out/'data'/f'{stem}_layer_cycles.csv',case['analysed']['cycles'])
        write_csv(args.out/'data'/f'{stem}_layer_timing_full_run.csv',layer_rows(case))
        write_json(args.out/'data'/f'{stem}_physical_audit.json',physical['audit'])
        audit_path=case['folder']/'strict_audit.json'
        if audit_path.exists():write_json(args.out/'data'/f'{stem}_strict_audit.json',read(audit_path))
        allrows.extend(case['rows']);aggregates.extend(inherited.profile_aggregates(case,case['rows']))
        summaries.append(dict(name=case['name'],policy=case['policy'],seed=case['metadata']['seed'],
            num_ssu=case['num_ssu'],input_fingerprint=case['fingerprint'],
            U_percent=case['metrics']['U_percent'],warm_slo_percent=case['metrics']['slo_1p5_percent'],
            full_slo_percent=100*sum(r['slo_1p5_passed'] for r in case['rows'])/len(case['rows']),
            request_count=len(case['rows']),demand_display_window_ms=[LEFT,events['edges'][-1]],
            demand_curve_stats=figures[-4]['demand_stats'],
            npu_link_stats=figures[-4]['npu_link_stats'],source_hashes=case['analysed']['source_hashes']))
    for group,items in groups.items():
        ids=[{r['request_id'] for r in c['rows']} for c in items]
        assert all(x==ids[0] for x in ids)
        for cohort in ('warm_admissions','all_requests'):
            figures.append(inherited.draw_cdf(group,items,cohort,
                args.out/'images'/f'{group}_ttft_slo15_{cohort}.png',points))
        fields=['configuration','scenario_name','seed','npu_id','request_id','input_order_1based',
            'profile_group','profile_tag','role','total_tokens','total_length_k','nql_tokens','hit_prefix_tokens',
            'per_layer_read_gib','per_layer_read_mib','per_layer_compute_ms','bandwidth_demand_gib_s',
            'compute_method','compute_extrapolated']
        rows=[{k:r[k] for k in fields} for r in items[0]['rows']]
        for row in rows:
            row['single_ssu_40gib_s_transfer_ms']=row['per_layer_read_gib']/40*1000
        write_csv(args.out/'data'/f'{group}_input_profiles.csv',rows)
        write_json(args.out/'data'/f'{group}_profile_groups.json',dict(tags=items[0]['tags'],
            actual_profiles=items[0]['metadata']['profiles'],input_fingerprint=items[0]['fingerprint']))
    write_csv(args.out/'data'/'all_request_execution.csv',allrows)
    write_csv(args.out/'data'/'profile_group_summary.csv',aggregates)
    write_csv(args.out/'data'/'ttft_cdf_points.csv',points)
    write_json(args.out/'data'/'scenario_summary.json',summaries)
    write_csv(args.out/'data'/'scenario_summary.csv',
        [{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in summaries])
    for info in figures:
        path=args.out/'images'/info['file']
        with Image.open(path) as png:png.load()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    write_json(args.out/'data'/'figure_audit.json',dict(figures=figures,all_checks_passed=True,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_simulation_executed=True,request_execution_rows=len(allrows),
        dependency_sha256={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (inherited,shared,accounting,strict)}))
    notes='''# 全时段欠载候选：冻结结果图表

每张卡独立随机混合所有画像，8 NPU，ring hash，同一KV block所有层位于同一SSU。图中G标签只是输入画像简称。

- `*_strict_demand_events.png`：0到全部输入完成的精确事件阶梯，显示整机、逐盘当前请求V/C与下一层V/当前层C；不分桶、不平滑、不限幅。36/40 GiB/s是单盘90%/100%容量。倒数第二行显示当时最高单卡预取需求和50 GiB/s接收上限。
- 图覆盖完整轨迹，包括尾段。图中峰值与接近容量比例按精确事件间隔加权。需求参考在计算/请求事件之间恒定；冷启动首层没有前序计算窗口，不能给它构造虚假预取需求。
- 最后一行显示截止时间已过但IO仍未完成的层数，并单独标出没有前序窗口的冷启动首层。这样即使预取计算窗口已经结束，仍能看到未完成的任务。逐区间层数在`*_demand_event_segments.csv`，逾期明细和冷启动明细分别在JSON中。
- `*_physical_ssd_bandwidth.png`：warm [2,4)秒真实服务字节除以公共2 ms，逐盘≤40，整机≤40×SSU数。SSD忙时服务触顶不是需求超载；本图不用于代替逐事件需求判断。
- `*_8npu_timeline*.png`：计算、端到端读取、IO Stall；局部图选择Stall最多的250 ms，只辅助观察，利用率仍来自完整warm窗口。
- `*_input_profiles.csv`：每条请求总长、NQL、每层读取量、计算时间、V/C及独占40 GiB/s下纯传输时间；后者不含排队，也不是多SSU端到端实测延迟。
- `*_requests.csv`：逐请求实测TTFT、SLO、完成时刻与时间账；TTFT从NPU接纳开始，不含接纳前输入队列等待，阈值为8层纯计算时间×1.5。
- `*_layer_timing_full_run.csv`：全部层的IO开始、IO就绪、计算起止时刻，以及含排队的实际读取时间和暴露IO Stall。`single_ssu_40gib_s_transfer_ms`仅为独占单SSU纯传输基准；实际读取时间需看`actual_read_span_including_queue_ms`。
- 相同输入指纹的策略配对绘制TTFT CDF；所有请求图严格对齐请求ID。warm接纳集合可能因策略不同而不同。
- 单卡预取需求超过50 GiB/s时，即便盘侧欠载也可能受到接收链路限制；这类等待不能直接认定为FIFO队头阻塞。当前图包只陈述指定候选的结果，不声称每个候选都通过严格筛选。
- 此脚本只读取冻结结果，源模型与输入不作修改。
'''
    (args.out/'README.md').write_text(notes,encoding='utf-8')
    print(shared.json.dumps(dict(output=str(args.out.resolve()),cases=len(cases),
        input_groups=len(groups),figures=len(figures),all_checks_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
