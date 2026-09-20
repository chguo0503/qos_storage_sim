#!/usr/bin/env python3
"""Render frozen eight-NPU mixed results; never run or replay the simulator.

Purple = active request's V/C. Blue = physically received bytes divided by a
complete internal layer cycle; gray = cross-request / clipped cycles. The
timeline separately shows compute, end-to-end reads, and exposed IO stalls.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fifo_mixed8_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'results/fifo_mixed_unique_20260914'
LEFT, RIGHT, N = 2000., 4000., 8
BLUE, PURPLE, GRAY = '#0068d9', '#a32b91', '#e0e4e9'
GREEN, YELLOW = '#31a354', '#f2c44c'
INK, MUTED = '#172d45', '#546980'


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, context=''):
    assert math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-9), (context, a, b)


def clip(a, b, left=LEFT, right=RIGHT):
    return max(0., min(b, right)-max(a, left))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_case(value):
    path = Path(value)
    for candidate in (path, ROOT/path, DATA/path):
        if (candidate/'result.json.gz').exists():
            return candidate.resolve()
    raise FileNotFoundError(value)


def analyse(folder):
    raw = read(folder/'result.json.gz')
    manifest = read(folder/'manifest.json.gz')
    metric = read(folder/'metrics.json')
    meta = read(folder/'metadata.json')
    physical = read(folder/'receipts.json')
    assert metric['num_npu'] == meta['num_npu'] == physical['num_npu'] == N
    assert physical['observer_checks'] and all(physical['observer_checks'].values())
    assert physical['window_ms'] == metric['window_ms'] == [LEFT, RIGHT]
    assert physical['source_result_sha256'] == sha(folder/'result.json.gz')
    assert physical['source_manifest_sha256'] == sha(folder/'manifest.json.gz')
    assert all(raw['summary']['invariants'].values()) and metric['all_active']
    assert meta['all_length_nql_unique_within_each_npu']
    requests = {q['request_id']: q for q in manifest['requests']}
    layer_stats = {(int(q['request_id']), int(q['layer'])): q for q in physical['layers']}
    received = physical['window_npu_received_gib']
    assert len(received) == N
    lanes, all_cycles, completion_rows, class_rows = [], [], [], []
    for npu in range(N):
        input_loads = [q['load'] for q in requests.values() if q['npu_id'] == npu]
        assert set(q['role'] for q in input_loads) == {'L', 'S'}
        combos = [(q['total_tokens'], q['nql']) for q in input_loads]
        assert len(set(combos)) == len(combos)
        batches = sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id'] == npu),
                         key=lambda b: b['admission_time_ms'])
        flat, demands, active_batches = [], [], []
        roles = {r: dict(active_ms=0., compute_ms=0., active_requests=0,
                         computed_requests=0, admitted_requests=0, completed_requests=0,
                         received_GiB=0., demand_time_GiB=0.) for r in ('L', 'S')}
        for batch in batches:
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            q = requests[rid]['load']
            role = q['role']
            layers = sorted(batch['layer_metrics'], key=lambda l: l['layer'])
            flat.extend(dict(request_id=rid, role=role, **layer) for layer in layers)
            active = clip(batch['admission_time_ms'], batch['completion_time_ms'])
            compute = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms']) for l in layers)
            r = roles[role]
            r['active_ms'] += active
            r['compute_ms'] += compute
            r['active_requests'] += active > 0
            r['computed_requests'] += compute > 0
            r['admitted_requests'] += LEFT <= batch['admission_time_ms'] < RIGHT
            r['completed_requests'] += LEFT <= batch['completion_time_ms'] < RIGHT
            r['received_GiB'] += math.fsum(layer_stats[rid, l['layer']]['window_received_gib'] for l in layers)
            demand = q['per_layer_kv_gb']*1e6/q['per_layer_us']
            r['demand_time_GiB'] += active*demand/1000
            if active > 0:
                demands.append((batch['admission_time_ms'], batch['completion_time_ms'], demand))
                active_batches.append(dict(request_id=rid, role=role,
                    admission_ms=batch['admission_time_ms'], completion_ms=batch['completion_time_ms'],
                    layers=layers))
            if LEFT <= batch['completion_time_ms'] < RIGHT:
                completion_rows.append(dict(npu_id=npu, request_id=rid, role=role,
                    total_tokens=q['total_tokens'], nql=q['nql'],
                    admission_ms=batch['admission_time_ms'], completion_ms=batch['completion_time_ms'],
                    admission_relative_TTFT_ms=batch['completion_time_ms']-batch['admission_time_ms']))
        direct_c = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms']) for l in flat)
        close(direct_c, raw['windows'][0]['compute_ms_by_npu'][npu], 'window compute')
        close(math.fsum(clip(a,b) for a,b,_ in demands), RIGHT-LEFT, 'whole-window active')
        close(math.fsum(r['received_GiB'] for r in roles.values()), received[npu], 'physical class receipt conservation')
        for role, row in roles.items():
            reported = metric['window_by_npu'][npu]['by_role'][role]
            close(row['compute_ms'], reported['compute_ms'], 'class compute')
            close(row['active_ms'], reported['active_ms'], 'class active')
            class_rows.append(dict(npu_id=npu, role=role,
                active_requests=row['active_requests'], computed_requests=row['computed_requests'],
                admitted_requests=row['admitted_requests'], completed_requests=row['completed_requests'],
                active_ms=row['active_ms'], compute_ms=row['compute_ms'], stall_ms=row['active_ms']-row['compute_ms'],
                active_time_U_percent=100*row['compute_ms']/row['active_ms'] if row['active_ms'] else None,
                mean_B_over_whole_window_GiB_s=row['demand_time_GiB']*1000/(RIGHT-LEFT),
                mean_received_b_over_whole_window_GiB_s=row['received_GiB']*1000/(RIGHT-LEFT)))
        cycles = []
        for current, following in zip(flat, flat[1:]):
            start, end = current['compute_start_ms'], following['compute_start_ms']
            if clip(start,end) <= 0:
                continue
            deadline = current['compute_end_ms']
            compute, duration = deadline-start, end-start
            close(following['io_start_time_ms'], start, 'next read starts at compute start')
            close(end, max(deadline,following['io_ready_time_ms']), 'compute/read barrier')
            same = current['request_id'] == following['request_id']
            complete = LEFT <= start < end <= RIGHT
            q = requests[current['request_id']]['load']
            demand = q['per_layer_kv_gb']*1000/compute
            group = q['role'] if same and complete else 'gray'
            actual = ratio = volume = None
            if group != 'gray':
                observed = layer_stats[following['request_id'],following['layer']]
                volume = observed['bytes_gib']
                close(volume, q['per_layer_kv_gb'], 'observed layer bytes')
                placement = manifest['placements'][requests[following['request_id']]['placement_index']]
                assert observed['completed_blocks'] == len(placement[0 if len(placement)==1 else following['layer']])
                assert observed['first_link_start_ms'] >= start-1e-7
                close(observed['last_link_end_ms'], following['io_ready_time_ms'], 'physical last byte')
                assert observed['last_link_end_ms'] <= end+1e-7
                actual = volume*1000/duration
                ratio = actual/demand
                close(ratio, compute/duration, 'same-cycle ratio')
            cycles.append(dict(npu=npu, request_id=current['request_id'], next_request_id=following['request_id'],
                role=q['role'], compute_layer=current['layer'], read_layer=following['layer'],
                start_ms=start, end_ms=end, C_ms=compute, T_ms=duration, deadline_ms=deadline,
                clipped_start_ms=max(LEFT,start), clipped_end_ms=min(RIGHT,end),
                actual_window_compute_ms=clip(start,deadline), group=group,
                B_GiB_s=demand, mean_b_GiB_s=actual, received_GiB=volume, cycle_ratio=ratio))
        edge_start = cycles[0]['clipped_start_ms'] if cycles else RIGHT
        edge_end = cycles[-1]['clipped_end_ms'] if cycles else LEFT
        for a,b in [(LEFT,edge_start),(edge_end,RIGHT)]:
            if b > a:
                cycles.append(dict(npu=npu, request_id=None, next_request_id=None, role=None,
                    compute_layer=None, read_layer=None, start_ms=a, end_ms=b, C_ms=None, T_ms=b-a,
                    deadline_ms=None, clipped_start_ms=a, clipped_end_ms=b,
                    actual_window_compute_ms=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms'],a,b) for l in flat),
                    group='gray', B_GiB_s=None, mean_b_GiB_s=None, received_GiB=None, cycle_ratio=None))
        cycles.sort(key=lambda c:c['clipped_start_ms'])
        cursor = LEFT
        for c in cycles:
            close(c['clipped_start_ms'],cursor,'cycle partition')
            cursor = c['clipped_end_ms']
        close(cursor,RIGHT,'cycle coverage')
        close(math.fsum(c['actual_window_compute_ms'] for c in cycles),direct_c,'cycle compute accounting')
        timeline_stall = 0.
        for batch in active_batches:
            deadline = batch['admission_ms']
            for layer in batch['layers']:
                timeline_stall += clip(deadline,layer['compute_start_ms'])
                deadline = layer['compute_end_ms']
        close(direct_c+timeline_stall,RIGHT-LEFT,'timeline compute plus exposed stalls')
        both = all(r['compute_ms'] > 0 for r in roles.values())
        assert both == metric['window_by_npu'][npu]['both_roles_computed']
        lane = dict(npu=npu, U_percent=100*direct_c/(RIGHT-LEFT), compute_ms=direct_c,
            stall_ms=timeline_stall, both_roles_computed=both,
            input_L_count=sum(q['role']=='L' for q in input_loads), input_S_count=sum(q['role']=='S' for q in input_loads),
            window_L_computed_requests=roles['L']['computed_requests'], window_S_computed_requests=roles['S']['computed_requests'],
            window_L_active_ms=roles['L']['active_ms'], window_S_active_ms=roles['S']['active_ms'],
            mean_demand_GiB_s=math.fsum(clip(a,b)*d for a,b,d in demands)/(RIGHT-LEFT),
            mean_supply_GiB_s=received[npu]*1000/(RIGHT-LEFT), received_GiB=received[npu],
            gray_ms=math.fsum(c['clipped_end_ms']-c['clipped_start_ms'] for c in cycles if c['group']=='gray'),
            cycles=cycles, demands=demands, batches=active_batches, layers=flat)
        close(lane['U_percent'],metric['window_by_npu'][npu]['U_percent'],'per NPU utilization')
        lanes.append(lane)
        all_cycles.extend(cycles)
    close(math.fsum(l['U_percent'] for l in lanes)/N,metric['U_percent'],'fleet utilization')
    assert all(l['both_roles_computed'] for l in lanes) == metric['all_npus_both_roles_computed']
    policy_names = {
        'fifo': 'Baseline FIFO',
        'short_first': '短读取优先（诊断）',
        'once': 'Once per layer（TTL=5ms）',
    }
    policy = policy_names[metric['policy']]
    order = '随机混合' if meta['order']=='random' else '周期混合'
    names = [('result.json.gz'),('manifest.json.gz'),('metadata.json'),('metrics.json'),('receipts.json')]
    return dict(stem=folder.name, policy=policy, order=order, source=str(folder),
        num_ssu=metric['num_ssu'], seed=meta['seed'], metadata=meta, metrics=metric,
        static_underload=meta['static_underload_all_request_combinations'],
        static_max_GiB_s=max(meta['per_ssu_static_upper_bound_gib_s']),
        computed_both_count=sum(l['both_roles_computed'] for l in lanes),
        lanes=lanes, cycles=all_cycles, completion_rows=completion_rows, class_rows=class_rows,
        source_hashes={name:sha(folder/name) for name in names}, all_checks_passed=True)


def short_description(data):
    p = data['metadata']['profiles']
    span = lambda xs: f'{xs[0]:.2f}–{xs[1]:.2f}'
    return (f'L：{span(p["L"]["total_length_k_range"])}K / NQL {p["L"]["nql_range"][0]}–{p["L"]["nql_range"][1]}；'
            f'S：{span(p["S"]["total_length_k_range"])}K / NQL {p["S"]["nql_range"][0]}–{p["S"]["nql_range"][1]}；'
            f'每卡 L:S = 1:{data["metadata"]["case"]["short_per_long"]}')


def add_header(fig, data, view):
    m = data['metrics']
    fig.text(.035,.965,f'{data["policy"]} · 每卡长短请求{data["order"]}：{view}',fontsize=22,color=INK)
    fig.text(.035,.934,f'8 NPU / {data["num_ssu"]} SSU × 40 GiB/s · ring hash · seed {data["seed"]} · warm [2,4)秒'
             f' · 整机U={m["U_percent"]:.2f}% · TTFT SLO×1.5={m["slo_1p5_percent"]:.2f}%',fontsize=13,color=MUTED)
    fig.text(.035,.906,short_description(data),fontsize=12,color=MUTED)
    construction = '小于32K的计算时间含data外推。' if not data['metadata']['all_compute_profiles_within_measured_grid'] else '计算时间由data网格内插。'
    fig.text(.035,.879,f'输入：8/8张卡均含L和S，每卡长度/NQL不重复；统计窗口实际计算过L和S：{data["computed_both_count"]}/8张卡。'+construction,
             fontsize=12,color=MUTED)


def save_checked(fig, output):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width,height = fig.canvas.get_width_height()
    overflow=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            if b.x0 < -2 or b.y0 < -2 or b.x1 > width+2 or b.y1 > height+2:
                overflow.append((artist.get_text(),b.bounds))
    assert not overflow,overflow
    fig.savefig(output,dpi=150)
    plt.close(fig)
    return dict(file=output.name,pixels=[width,height],visible_labels_inside_canvas=True,sha256=sha(output),selected_npus=list(range(N)))


def underload_note(data):
    status='通过：任意各卡请求组合均小于40 GiB/s。' if data['static_underload'] else '未通过静态欠载界；需查看逐盘实际接纳需求审计。'
    return f'逐盘静态需求上界最大值 = {data["static_max_GiB_s"]:.3f} GiB/s；'+status


def policy_note(data):
    return {
        'fifo': 'Baseline使用原始Path0单路FIFO。',
        'short_first': '短读取优先仅为诊断对照，不是Once策略。',
        'once': 'Once per layer：每层更新一次QoS选路，TTL=5ms。',
    }[data['metrics']['policy']]


def bandwidth(ax,lane,ymax):
    blue,purple,points=[],[],[]
    previous=None
    for c in lane['cycles']:
        a,b=c['clipped_start_ms']/1000,c['clipped_end_ms']/1000
        if c['group']=='gray':
            ax.axvspan(a,b,color=GRAY,zorder=0)
            previous=None
            continue
        v=c['mean_b_GiB_s']
        blue.append([(a,v),(b,v)])
        if previous is not None and math.isclose(previous['end_ms']/1000,a,abs_tol=1e-9):
            blue.append([(a,previous['mean_b_GiB_s']),(a,v)])
        points.append(((a+b)/2,v))
        previous=c
    previous=None
    for a,b,v in lane['demands']:
        a,b=max(LEFT,a)/1000,min(RIGHT,b)/1000
        purple.append([(a,v),(b,v)])
        if previous is not None and math.isclose(previous[0],a,abs_tol=1e-9):
            purple.append([(a,previous[1]),(a,v)])
        previous=(b,v)
    ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.2,zorder=3))
    ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.3,linestyles='--',zorder=4))
    if points:
        x,y=zip(*points)
        ax.scatter(x,y,s=12,facecolors='white',edgecolors=BLUE,linewidths=1,zorder=5)
    ax.set(xlim=(2,4),ylim=(0,ymax),xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4],yticks=[0,ymax/2,ymax])
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='both',alpha=.14,lw=.6)


def draw_bandwidth(data,output,ymax):
    fig,axes=plt.subplots(N,1,figsize=(18,12.5),dpi=150,sharex=True,sharey=True,facecolor='white')
    fig.subplots_adjust(left=.115,right=.847,top=.765,bottom=.155,hspace=.39)
    add_header(fig,data,'每层平均带宽与需求（全部8张卡）')
    handles=[Line2D([],[],color=PURPLE,lw=2.3,ls='--',label='需求 B：当前请求每层 V / C'),
        Line2D([],[],color=BLUE,lw=2.2,marker='o',markerfacecolor='white',label='供给 b：完整内部层周期平均带宽'),
        Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.031,.855),frameon=False,ncol=3,fontsize=11.5)
    fig.text(.035,.794,f'左右数字统计完整[2,4)秒，灰区仍计入；所有图纵轴统一0–{ymax:g} GiB/s。',fontsize=12,color=MUTED)
    fig.text(.863,.779,'整窗平均（GiB/s）',fontsize=11,color=INK)
    for npu,ax in enumerate(axes):
        lane=data['lanes'][npu]
        bandwidth(ax,lane,ymax)
        ax.set_ylabel(f'NPU {npu}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=14,color=INK)
        ax.tick_params(axis='y',labelsize=9,length=3)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu==N-1)
        ax.text(1.022,.70,f'需求 {lane["mean_demand_GiB_s"]:.3f}',transform=ax.transAxes,fontsize=11,color=PURPLE,va='center')
        ax.text(1.022,.28,f'供给 {lane["mean_supply_GiB_s"]:.3f}',transform=ax.transAxes,fontsize=11,color=BLUE,va='center')
    axes[-1].set_xlabel('仿真时间（秒）；带宽单位 GiB/s',fontsize=13,labelpad=9)
    fig.text(.035,.092,'周期 T = 当前层开始计算 → 下一层开始计算（包含IO等待）；b = 周期内实际收到的下一层字节量 / T。',fontsize=11.5,color=INK)
    fig.text(.035,.066,'右侧需求按时间加权；供给=2秒内实际收到的字节量/2秒（含灰区、部分IO）。两个整窗均值相除不等于U。',fontsize=11.5,color=MUTED)
    fig.text(.035,.040,underload_note(data),fontsize=11.5,color=MUTED)
    fig.text(.035,.016,'蓝线是完整层周期平均值；物理收字节量来自本次仿真的只读观测。'+policy_note(data),fontsize=11,color=MUTED)
    return save_checked(fig,output)


def draw_timeline(data,output):
    fig,axes=plt.subplots(N,1,figsize=(18,12.5),dpi=150,sharex=True,sharey=True,facecolor='white')
    fig.subplots_adjust(left=.115,right=.947,top=.765,bottom=.140,hspace=.41)
    add_header(fig,data,'读取、计算与IO Stall（全部8张卡）')
    handles=[Patch(facecolor=GREEN,label='绿色：计算'),Patch(facecolor=BLUE,label='蓝色：端到端读取（含排队）'),
        Patch(facecolor=YELLOW,label='黄色：暴露的IO Stall'),Line2D([],[],color='#111111',marker='|',lw=0,markersize=11,label='黑刻线：请求完成')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.031,.855),frameon=False,ncol=4,fontsize=11.5)
    fig.text(.035,.798,'每张卡由上到下：计算 / 读取 / Stall；L和S标记请求类别。完成时刻逐条导出CSV，图中不叠加时间文本。',fontsize=11.5,color=MUTED)
    for npu,ax in enumerate(axes):
        lane=data['lanes'][npu]
        ax.set(xlim=(2,4),ylim=(0,1.12),yticks=[],xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4])
        for batch in lane['batches']:
            a,b=max(LEFT,batch['admission_ms'])/1000,min(RIGHT,batch['completion_ms'])/1000
            ax.axvspan(a,b,color='#e8eef8' if batch['role']=='L' else '#f8f5e9',alpha=.47,zorder=0)
            if (b-a)*1000 >= 35:
                ax.text((a+b)/2,1.055,batch['role'],ha='center',va='center',fontsize=8,color=INK,clip_on=True)
            deadline=batch['admission_ms']
            for layer in batch['layers']:
                for start,end,y,color in [(layer['compute_start_ms'],layer['compute_end_ms'],.71,GREEN),
                        (layer['io_start_time_ms'],layer['io_ready_time_ms'],.40,BLUE),
                        (deadline,layer['compute_start_ms'],.09,YELLOW)]:
                    length=clip(start,end)
                    if length>0:
                        ax.broken_barh([(max(start,LEFT)/1000,length/1000)],(y,.22),facecolors=color,linewidth=0,zorder=2)
                deadline=layer['compute_end_ms']
            end=batch['completion_ms']
            if LEFT <= end < RIGHT:
                ax.plot([end/1000,end/1000],[.96,1.12],color='#111111',lw=1.15,zorder=5)
        # A following request can be prefetched before it is admitted, including
        # past the right edge. Show that physical read even if its active span
        # does not intersect the window; it must not disappear from the chart.
        shown={b['request_id'] for b in lane['batches']}
        for layer in lane['layers']:
            if layer['request_id'] not in shown and clip(layer['io_start_time_ms'],layer['io_ready_time_ms'])>0:
                a,b=layer['io_start_time_ms'],layer['io_ready_time_ms']
                ax.broken_barh([(max(a,LEFT)/1000,clip(a,b)/1000)],(.40,.22),facecolors=BLUE,linewidth=0,zorder=2)
        ax.set_ylabel(f'NPU {npu}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=14,color=INK)
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',alpha=.16,lw=.6)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu==N-1)
    axes[-1].set_xlabel('仿真时间（秒）',fontsize=13,labelpad=9)
    fig.text(.035,.080,'读取可与上一层计算重叠；黄色只统计数据未到齐、导致NPU不能开始计算的时间，不能把蓝色总时长当作Stall。',fontsize=11.5,color=INK)
    fig.text(.035,.054,'逐卡利用率 = 窗口内绿色计算总时长 / 2秒；每卡绿色计算 + 黄色Stall = 2秒，已逐项复核。',fontsize=11.5,color=MUTED)
    fig.text(.035,.028,underload_note(data),fontsize=11.5,color=MUTED)
    return save_checked(fig,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',required=True,help='Case directories, absolute or relative to source / mixed result root')
    parser.add_argument('--out',type=Path,default=ROOT/'results/fifo_mixed8_figures_20260914')
    parser.add_argument('--font',type=Path,required=True,help='Path to an installed font with Chinese glyphs (TTF/OTF/TTC)')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    matplotlib.font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),'axes.unicode_minus':False,'font.size':12})
    data=[analyse(resolve_case(case)) for case in args.cases]
    max_B=max(v for d in data for lane in d['lanes'] for a,b,v in lane['demands'])
    ymax=float(math.ceil(max_B*1.05))
    for item in data:
        stem=item['stem']
        images=[draw_bandwidth(item,args.out/(stem+'_8npu_bandwidth.png'),ymax),
                draw_timeline(item,args.out/(stem+'_8npu_timeline.png'))]
        write_csv(args.out/(stem+'_cycles.csv'),item['cycles'])
        write_csv(args.out/(stem+'_request_completions.csv'),item['completion_rows'])
        write_csv(args.out/(stem+'_per_npu_class.csv'),item['class_rows'])
        per_npu=[{k:v for k,v in lane.items() if k not in ('cycles','demands','batches','layers')} for lane in item['lanes']]
        write_csv(args.out/(stem+'_per_npu.csv'),per_npu)
        summary={k:v for k,v in item.items() if k not in ('lanes','cycles','completion_rows','class_rows','metadata')}
        summary.update(per_npu=per_npu,images=images,shared_ymax_GiB_s=ymax,renderer_sha256=sha(Path(__file__)))
        write_json(args.out/(stem+'_figure_audit.json'),summary)
        print(json.dumps(dict(case=stem,U_percent=item['metrics']['U_percent'],shared_ymax_GiB_s=ymax,
            computed_both_count=item['computed_both_count'],all_checks_passed=True,images=images),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
