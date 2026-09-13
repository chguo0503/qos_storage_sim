#!/usr/bin/env python3
"""One complete same-request short-layer cycle, selected by median positive stall.

Read-only postprocessing of existing immutable block traces; no simulation.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import statistics
import numpy as np
import render as style
import analyze as audit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BLUE, PURPLE, TEAL = '#0068d9', '#a32b91', '#009b97'
LONG, OTHER, COMPUTE, WAIT = '#914776', '#bac3cc', '#168879', '#f0b452'
INK, MUTED = style.INK, style.MUTED


def choose(raw, requests, short_roles, warm):
    all_complete, eligible = [], []
    for batch in raw['summary']['microbatch_metrics']:
        assert len(batch['member_request_ids']) == 1
        rid = batch['member_request_ids'][0]
        if requests[rid]['load']['role'] not in short_roles:
            continue
        layers = batch['layer_metrics']
        for current, following in zip(layers, layers[1:]):
            a, z = current['compute_start_ms'], following['compute_start_ms']
            if not warm[0] <= a < z <= warm[1]:
                continue
            row = dict(request_id=rid, npu=batch['npu_id'], current=current, following=following,
                       wait_ms=z-current['compute_end_ms'])
            all_complete.append(row)
            # Skip the request's first compute layer and final prefetched layer.
            if current['layer'] >= 1 and following['layer'] < len(layers)-1 and row['wait_ms'] > audit.TOL_MS:
                eligible.append(row)
    assert eligible, 'No positive-wait internal short cycle fully inside the requested warm window.'
    median = statistics.median(r['wait_ms'] for r in eligible)
    chosen = min(eligible, key=lambda r: (abs(r['wait_ms']-median), r['request_id'], r['current']['layer']))
    selection = dict(warm_window_ms=list(warm), short_roles=list(short_roles),
                     complete_short_cycles_including_zero_wait=len(all_complete),
                     complete_short_cycles_positive_wait=sum(r['wait_ms'] > audit.TOL_MS for r in all_complete),
                     eligible_cycles=len(eligible), positive_wait_median_ms=median,
                     selected_distance_from_median_ms=abs(chosen['wait_ms']-median),
                     rule='Complete same-request cycles inside warm; current layer index >= 1 and next layer index < n_layers-1; strictly positive exposed wait; minimize absolute distance to median wait, ties by request_id then current layer index.',
                     note='Median of the eligible positive-wait sample, not the mean wait of all short layers; layer indices in evidence are zero-based.')
    return chosen, selection


def cumulative(rows, start_col, end_col, rate, left, right):
    """Exact piecewise-linear integral at every physical interval boundary."""
    changes = defaultdict(float)
    changes[left] += 0.
    changes[right] += 0.
    for row in rows:
        a, z = max(left, row[start_col]), min(right, row[end_col])
        if z > a:
            changes[a] += rate
            changes[z] -= rate
    times = np.asarray(sorted(changes), float)
    values = [0.]
    current = 0.
    for a, z in zip(times, times[1:]):
        current += changes[a]
        assert current >= -1e-7
        values.append(values[-1]+current*(z-a)/1000)
    return times, np.asarray(values)


def prepare(case, warm):
    files = {n: case/n for n in ('manifest.json.gz', 'result.json.gz', 'trace.json.gz', 'command.json', 'analysis.json')}
    man, raw, trace, command, analysis = [audit.read(files[n]) for n in files]
    assert command['status'] == 'complete' and command['completed_simulation']
    for name, key in [('manifest.json.gz', 'manifest_sha256'), ('result.json.gz', 'output_sha256'), ('trace.json.gz', 'trace_sha256')]:
        assert audit.sha(files[name]) == command[key]
    assert analysis['builder_sha256'] == audit.sha(Path(audit.__file__)) and analysis['all_technical_checks_passed']
    for name, digest in analysis['sources'].items():
        assert audit.sha(Path(name)) == digest
    assert man['input_fingerprint'] == raw['input_fingerprint'] == analysis['input_fingerprint']
    assert trace['columns'] == style.COLS and trace['completed_simulation'] and all(trace['checks'].values())
    assert trace['source']['manifest_sha256'] == command['manifest_sha256']
    assert trace['source']['reference_sha256'] == command['output_sha256']
    assert trace['window_ms'][0] <= warm[0] < warm[1] <= trace['window_ms'][1]
    assert raw['strategy'] == trace['strategy'] == 'baseline'
    meta = man['metadata']
    requests = {r['request_id']: r for r in man['requests']}
    request_metrics = {r['request_id']: r for r in raw['summary']['request_metrics']}
    selected, selection = choose(raw, requests, analysis['short_roles'], warm)
    rid, npu = selected['request_id'], selected['npu']
    current, following = selected['current'], selected['following']
    left, deadline, ready = current['compute_start_ms'], current['compute_end_ms'], following['io_ready_time_ms']
    right = following['compute_start_ms']
    q = requests[rid]['load']
    C, V, D = q['per_layer_us']/1000, q['per_layer_kv_gb'], right-left
    audit.close(deadline-left, C)
    audit.close(following['io_start_time_ms'], left)
    audit.close(right, ready)
    audit.close(C+selected['wait_ms'], D)
    info, _, disk_caps, link_cap = audit.profile_info(man, raw['summary'])
    nominal_segments = audit.nominal_segments(raw['summary']['request_metrics'], info, meta['num_ssu'])
    local_nominal = audit.nominal_window(nominal_segments, left, right, disk_caps, link_cap, analysis['roles'])
    local_nominal['segments'] = [dict(start_ms=max(left, s['start_ms']), end_ms=min(right, s['end_ms']),
                                     per_ssu_GiB_s=s['per_ssu_GiB_s'], role_card_count=s['role_card_count'])
                                for s in nominal_segments if s['end_ms'] > left and s['start_ms'] < right]
    rows = np.asarray(trace.pop('rows'), float)
    assert np.all(rows[:, 5] == 0) and np.all(rows[:, 7] == 1)
    target_mask = (rows[:, 0] == rid) & (rows[:, 2] == following['layer'])
    target = rows[target_mask]
    target = target[np.argsort(target[:, 3])]
    placement_set = man['placements'][requests[rid]['placement_index']]
    placement = np.asarray(placement_set[0 if len(placement_set) == 1 else following['layer']], float)
    assert len(target) == len(placement)
    assert np.array_equal(target[:, 3], np.arange(len(placement)))
    assert np.array_equal(target[:, 4], placement[:, 0]) and np.array_equal(target[:, 6], placement[:, 1])
    assert np.all(target[:, 1] == npu)
    audit.close(math.fsum(target[:, 6]), V)
    audit.close(float(target[:, 12].max()), ready)
    assert target[:, 8].min() >= left-audit.TOL_MS and target[:, 12].max() <= right+audit.TOL_MS
    disk_rate, link_rate = float(meta['disk_bw_gib_s']), float(meta['npu_bw_gib_s'])
    own = rows[rows[:, 1] == npu]
    foreign = own[~((own[:, 0] == rid) & (own[:, 2] == following['layer']))]
    for a, z in ((9, 10), (11, 12)):
        audit.close(float(np.maximum(0, np.minimum(foreign[:, z], right)-np.maximum(foreign[:, a], left)).sum()), 0.)
    # Every SSD service block shown must match its original immutable placement.
    shown = rows[(rows[:, 9] < right) & (rows[:, 10] > left)]
    for row in shown:
        request = requests[int(row[0])]
        places = man['placements'][request['placement_index']]
        disk, size = places[0 if len(places) == 1 else int(row[2])][int(row[3])]
        assert row[1] == request['npu_id'] and row[4] == disk and row[6] == size
        audit.close(row[10]-row[9], size*1000/disk_rate)
    for row in target:
        audit.close(row[12]-row[11], row[6]*1000/link_rate)
    physical = []
    short_set = set(analysis['short_roles'])
    for disk in range(meta['num_ssu']):
        disk_rows = shown[shown[:, 4] == disk]
        disk_rows = disk_rows[np.argsort(disk_rows[:, 9])]
        assert np.all(disk_rows[1:, 9] >= disk_rows[:-1, 10]-audit.TOL_MS)
        target_disk = target[target[:, 4] == disk]
        assert len(target_disk)
        first = float(target_disk[:, 9].min())
        ahead = disk_rows[disk_rows[:, 9] < first]
        groups = defaultdict(lambda: dict(ms=0., block_count=0, first_enqueue_ms=math.inf, last_enqueue_ms=-math.inf,
                                         service_start_ms=math.inf, service_end_ms=-math.inf))
        for row in ahead:
            dt = audit.overlap(row[9], row[10], left, first)
            if not dt:
                continue
            key = (int(row[0]), int(row[2]), requests[int(row[0])]['load']['role'])
            groups[key]['ms'] += dt
            groups[key]['block_count'] += 1
            groups[key]['first_enqueue_ms'] = min(groups[key]['first_enqueue_ms'], float(row[8]))
            groups[key]['last_enqueue_ms'] = max(groups[key]['last_enqueue_ms'], float(row[8]))
            groups[key]['service_start_ms'] = min(groups[key]['service_start_ms'], max(left, float(row[9])))
            groups[key]['service_end_ms'] = max(groups[key]['service_end_ms'], min(first, float(row[10])))
        groups = [dict(request_id=k[0], npu=requests[k[0]]['npu_id'], layer=k[1], role=k[2],
                       request_admission_ms=request_metrics[k[0]]['admission_time_ms'], **v)
                  for k, v in sorted(groups.items())]
        long_ms = math.fsum(g['ms'] for g in groups if g['role'] not in short_set)
        short_ms = math.fsum(g['ms'] for g in groups if g['role'] in short_set)
        busy = math.fsum(audit.overlap(r[9], r[10], left, right) for r in disk_rows)
        earlier = all(g['last_enqueue_ms'] <= float(target_disk[:, 8].min())+audit.TOL_MS for g in groups)
        physical.append(dict(ssu=disk, target_blocks=len(target_disk), target_V_GiB=math.fsum(target_disk[:, 6]),
                             target_first_ssd_start_ms=first, target_last_ssd_end_ms=float(target_disk[:, 10].max()),
                             busy_ms=busy, idle_ms=max(0., D-busy),
                             before_target_long_ssd_service_ms=long_ms, before_target_other_short_ssd_service_ms=short_ms,
                             ahead_blocks_enqueued_before_target=earlier, ahead_groups=groups))
    assert all(d['ahead_blocks_enqueued_before_target'] for d in physical), 'Cannot describe all earlier service as already enqueued ahead.'
    ssd_t, ssd_v = cumulative(target, 9, 10, disk_rate, left, right)
    link_t, link_v = cumulative(target, 11, 12, link_rate, left, right)
    audit.close(ssd_v[-1], V)
    audit.close(link_v[-1], V)
    received_deadline = float(np.interp(deadline, link_t, link_v))
    B, average_b, U = V*1000/C, link_v[-1]*1000/D, C/D
    audit.close(average_b/B, U)
    fleet = next(w for w in analysis['windows'] if (w['start_ms'], w['end_ms']) == warm)
    evidence = dict(all_checks_passed=True, no_simulation=True, case=str(case), strategy=raw['strategy'],
                    selection=selection, selected=selected, num_npu=meta['num_npu'], num_ssu=meta['num_ssu'],
                    sources={str(f): audit.sha(f) for f in files.values()},
                    builders={str(p): audit.sha(p) for p in (Path(__file__), Path(audit.__file__), Path(style.__file__))},
                    physical_rows_shown=len(shown), target_blocks=len(target), target_only_own_IO_in_window=True,
                    all_target_and_displayed_placements_verified=True, exact_unbinned_integrals=True,
                    left_ms=left, deadline_ms=deadline, ready_ms=ready, right_ms=right,
                    C_ms=C, wait_ms=selected['wait_ms'], D_ms=D, V_GiB=V, V_MiB=V*1024,
                    B_GiB_s=B, average_b_GiB_s=float(average_b), average_b_over_B=float(average_b/B),
                    local_cycle_U_percent=100*U, fleet_warm_U_percent=fleet['U_percent'],
                    received_at_compute_deadline_MiB=received_deadline*1024,
                    target_first_ssd_service_ms=float(target[:, 9].min()),
                    target_first_NPU_receive_ms=float(target[:, 11].min()),
                    target_ssd_readout_complete_ms=float(target[:, 10].max()),
                    disks=physical, local_nominal=local_nominal,
                    ratio_meaning='Exact accounting identity for this complete C+wait cycle and its fully received next-layer V; not an independent predictor or an instantaneous-utilization identity.')
    return dict(evidence=evidence, meta=meta, q=q, shown=shown, target=target,
                requests=requests, short_roles=short_set, ssd=(ssd_t, ssd_v), link=(link_t, link_v))


def draw(data, output):
    e, meta, q = data['evidence'], data['meta'], data['q']
    left, C, D, V = e['left_ms'], e['C_ms'], e['D_ms'], e['V_MiB']
    selected = e['selected']
    compute_layer, read_layer = selected['current']['layer']+1, selected['following']['layer']+1
    npu, rid, selection = selected['npu'], selected['request_id'], e['selection']
    fig = plt.figure(figsize=(18, 14), dpi=150, facecolor='white')
    fig.text(.06, .963, 'Baseline Random：同一短请求内部，为什么计算结束后还要等 IO？', fontsize=24, color=INK)
    configuration=f'{meta["candidate"]} · {meta["num_npu"]} NPU / {meta["num_ssu"]} SSU × {meta["disk_bw_gib_s"]:g} GiB/s · seed {meta["seed"]} · NPU {npu:02d} · 请求 {rid}'
    notice=style.construction_notice(meta)
    if notice:configuration+=' · '+notice
    fig.text(.06, .935, configuration, fontsize=14, color=MUTED)
    fig.text(.06, .91, f'短请求总长 {q["seq_len_k"]}K / miss {q["nql"]}；计算第 {compute_layer} 层，同时预取同一请求第 {read_layer} 层的 V = {V:.4f} MiB', fontsize=15, color=INK)
    fig.text(.06, .886, f'选择规则：warm 内、避开请求首尾的 {selection["eligible_cycles"]} 个正等待短层中，选最接近等待中位数的层（中位数 {selection["positive_wait_median_ms"]:.3f} ms）。', fontsize=12.5, color=MUTED)
    fig.text(.075, .845, f'B = V/C = {e["B_GiB_s"]:.3f} GiB/s       平均 b = 实际收到 V / 完整周期 D = {e["average_b_GiB_s"]:.3f} GiB/s', fontsize=18, color=INK)
    warm_left,warm_right=selection['warm_window_ms']
    fig.text(.075, .817, f'平均 b / B = C / D = {e["local_cycle_U_percent"]:.2f}%（本层周期利用率）；整机 warm [{warm_left/1000:g},{warm_right/1000:g}) 秒 U = {e["fleet_warm_U_percent"]:.2f}%', fontsize=15, color=INK)
    npu_ax = fig.add_axes([.095, .684, .85, .09])
    disk_ax = fig.add_axes([.095, .459, .85, .159])
    volume_ax = fig.add_axes([.095, .213, .85, .183])
    for ax in (npu_ax, disk_ax, volume_ax):
        ax.set_xlim(0, D)
        ax.set_xticks(np.linspace(0, D, 7))
        ax.set_xticklabels([f'{x:.2f}' for x in np.linspace(0, D, 7)])
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(C, color=INK, ls='--', lw=1.6, zorder=6)
        ax.axvline(D, color=COMPUTE, ls=':', lw=2., zorder=6)
        ax.grid(axis='x', alpha=.13)
    npu_ax.set_title('① NPU：计算结束是截止点；下一层数据没到齐，就暴露出等待 w', loc='left', fontsize=16, pad=13, color=INK)
    npu_ax.set(ylim=(0, 1.4), yticks=[])
    npu_ax.tick_params(labelbottom=False, bottom=False)
    npu_ax.spines[['left', 'bottom']].set_visible(False)
    npu_ax.broken_barh([(0, C)], (.08, .73), facecolor=COMPUTE)
    npu_ax.broken_barh([(C, D-C)], (.08, .73), facecolor=WAIT)
    npu_ax.text(C/2, .445, f'短请求第 {compute_layer} 层计算\nC = {C:.3f} ms', ha='center', va='center', color='white', fontsize=15)
    npu_ax.text((C+D)/2, .445, f'等第 {read_layer} 层数据\nw = {D-C:.3f} ms', ha='center', va='center', color='#684300', fontsize=14)
    npu_ax.annotate('', xy=(0, 1.05), xytext=(D, 1.05), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3))
    npu_ax.text(D/2, 1.08, f'D = C + w = {D:.3f} ms', ha='center', va='bottom', color=INK, fontsize=13)
    npu_ax.text(C, -.13, f'计算截止 +{C:.3f} ms', ha='right', va='top', color=INK, fontsize=12)
    npu_ax.text(D, -.13, f'IO-ready +{D:.3f} ms；开始第 {read_layer} 层', ha='right', va='top', color=COMPUTE, fontsize=12)
    disk_ax.set_title('② SSU：各盘自己的 path 0 上，目标短请求的 IO 前面确实排着别的 IO', loc='left', fontsize=16, pad=16, color=INK)
    disk_ax.set(ylim=(-.55, meta['num_ssu']+.7), yticks=range(meta['num_ssu']),
                yticklabels=[f'SSU {s} / path 0' for s in range(meta['num_ssu'])])
    disk_ax.invert_yaxis()
    disk_ax.tick_params(labelbottom=False, bottom=False)
    disk_ax.spines[['left', 'bottom']].set_visible(False)
    for disk in range(meta['num_ssu']):
        rows = data['shown'][data['shown'][:, 4] == disk]
        by_color = defaultdict(list)
        for row in rows:
            target = row[0] == rid and row[2] == read_layer-1
            color = BLUE if target else OTHER if data['requests'][int(row[0])]['load']['role'] in data['short_roles'] else LONG
            a, z = max(left, row[9])-left, min(left+D, row[10])-left
            by_color[color].append((a, z-a))
        for color, segments in by_color.items():
            disk_ax.broken_barh(segments, (disk-.26, .52), facecolor=color, edgecolor='none')
        for group in e['disks'][disk]['ahead_groups']:
            a, z = group['service_start_ms']-left, group['service_end_ms']-left
            if group['role'] not in data['short_roles'] and z-a > .18*D and abs(z-a-group['ms']) < 1e-7:
                disk_ax.vlines(a, disk-.26, disk+.26, color='white', lw=1.)
                disk_ax.text((a+z)/2, disk, f'NPU {group["npu"]} 长请求 · 第 {group["layer"]+1} 层 IO',
                             ha='center', va='center', fontsize=min(10.5,60/meta['num_ssu']), color='white')
    disk_ax.legend(handles=[Patch(facecolor=LONG, label='其他长请求的 IO'), Patch(facecolor=OTHER, label='其他短请求的 IO'), Patch(facecolor=BLUE, label=f'目标：本卡本请求第 {read_layer} 层 IO')],
                   loc='lower left', bbox_to_anchor=(-.006, -.06), ncol=3, frameon=False, fontsize=12)
    volume_ax.set_title(f'③ NPU 累计收到多少？第 {read_layer} 层必须把 {V:.4f} MiB 全部收齐', loc='left', fontsize=16, pad=18, color=INK)
    volume_ax.plot([0, C, D], [0, V, V], ls='--', color=PURPLE, lw=2.1, label='按 B = V/C 均匀到齐的参考进度')
    volume_ax.plot(data['ssd'][0]-left, data['ssd'][1]*1024, color=TEAL, lw=2.3, label='目标数据：SSD 累计读出')
    volume_ax.plot(data['link'][0]-left, data['link'][1]*1024, color=BLUE, lw=2.7, label='目标数据：NPU 累计收到')
    volume_ax.set(ylim=(-.035*V, 1.38*V), ylabel='累计数据量（MiB）', xlabel=f'从本请求第 {compute_layer} 层开始计算起的时间（毫秒）', yticks=[0, V/2, V])
    volume_ax.legend(loc='upper left', frameon=False, fontsize=11.5)
    missing = V-e['received_at_compute_deadline_MiB']
    volume_ax.annotate(f'计算截止仍缺 {missing:.4f} MiB', xy=(C, e['received_at_compute_deadline_MiB']),
                       xytext=(.50*D, .42*V), ha='center', fontsize=13, color='#9b5420',
                       arrowprops=dict(arrowstyle='->', color='#9b5420', lw=1.2), bbox=dict(facecolor='white', edgecolor='none', alpha=.92))
    longs = [d['before_target_long_ssd_service_ms'] for d in e['disks']]
    first_ssd = e['target_first_ssd_service_ms']-left
    fig.text(.075, .142, f'实际排队证据：目标直到 +{first_ssd:.3f} ms 才开始获得 SSD 服务；此前各盘服务前方长请求约 {min(longs):.3f}～{max(longs):.3f} ms。', fontsize=13.5, color=INK)
    fig.text(.075, .115, f'计算在 +{C:.3f} ms 结束；此时收到 {e["received_at_compute_deadline_MiB"]:.4f}/{V:.4f} MiB；到 +{D:.3f} ms 收齐，等待 {D-C:.3f} ms。', fontsize=13.5, color=INK)
    fig.text(.075, .086, '平均 b 把整段 D（包括零接收的时间）计入分母；它不是 SSD 或链路的限速。这个等式是完整周期的工作量核对。', fontsize=12.5, color=MUTED)
    fig.text(.075, .056, f'绝对时间 [{left/1000:.9f}, {(left+D)/1000:.9f}) 秒；本卡始终处于同一请求，SSU 行显示竞争请求；无平滑或分箱。', fontsize=12, color=MUTED)
    fig.text(.075, .032, f'有等待样本的中位数 ≠ 所有短层的平均等待。warm 内完整短层共 {selection["complete_short_cycles_including_zero_wait"]} 个，其中 {selection["complete_short_cycles_positive_wait"]} 个有等待；首尾过滤后选样。', fontsize=11.5, color=MUTED)
    return style.save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--window', type=audit.parse_window, default=(2000., 4000.))
    args = parser.parse_args()
    case = args.case.resolve()
    assert case.is_relative_to(HERE)
    out = case/'figures'/'internal_median_wait'
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(case, args.window)
    image = draw(data, out/'baseline_random_short_internal_cycle.png')
    evidence = data['evidence']
    evidence['image'] = image
    (out/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (out/'selected_cycle_blocks.json').write_text(json.dumps(dict(columns=style.COLS, target_rows=data['target'].tolist(), physical_ssd_rows=data['shown'].tolist()), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    notes = ['# 同一短请求内部的中位等待样例', '', '[查看 PNG](baseline_random_short_internal_cycle.png)', '',
             '此图根据已有逐块 trace 重建，没有新仿真。选择规则见 evidence.json：只看 warm 内完整、避开请求首尾且发生等待的短层，选其等待中位数附近的层，不挑最坏样例。', '',
             f"本例 NPU {evidence['selected']['npu']}，请求 {evidence['selected']['request_id']}。C={evidence['C_ms']:.9f} ms；w={evidence['wait_ms']:.9f} ms；D={evidence['D_ms']:.9f} ms；V={evidence['V_MiB']:.4f} MiB。", '',
             f"B=V/C={evidence['B_GiB_s']:.9f} GiB/s；平均 b=V/D={evidence['average_b_GiB_s']:.9f} GiB/s；平均 b/B=C/D={evidence['local_cycle_U_percent']:.6f}%。这个等式来自完整周期收齐同一个 V 的工作量核对，不是对调度效果的独立预测。", '',
             'NPU 行始终是同一个请求。SSU 行展示其他卡的竞争请求以及目标请求的磁盘服务，不能把紫色长请求 IO 误读为目标卡执行了另一个请求。累计曲线只统计目标下一层的数据，没有混入竞争请求字节。', '',
             '本例有一个重要限制：它不是全程逐盘瞬时欠载的证明。下表按当前已 admission 请求的逐盘 V/C 统计；下一请求首层预取不再次叠加，但物理读取全部保留。需求降低后，先前排入的 IO 仍然要继续服务。', '',
             '| 从本周期开始的区间（ms） | 角色卡数 | 各盘当前 V/C（GiB/s） |',
             '|---|---|---|']
    for s in evidence['local_nominal']['segments']:
        notes.append(f"| [{s['start_ms']-evidence['left_ms']:.6f}, {s['end_ms']-evidence['left_ms']:.6f}) | {s['role_card_count']} | "+', '.join(f'{v:.6f}' for v in s['per_ssu_GiB_s'])+' |')
    notes += ['', f"各盘容量 {data['meta']['disk_bw_gib_s']:g} GiB/s；周期中存在过载时间 {evidence['local_nominal']['any_ssu_over_capacity_ms']:.6f} ms。本图能证实物理队列与截止失约的关系，但不隔离纯 FIFO 效应和总需求过载效应。", '',
              '前方 IO 的卡号、请求号、层号、enqueue、request admission 和实际磁盘服务时间都在 evidence.json 的 disks/ahead_groups 中；目标逐块行与图中磁盘服务行保存在 selected_cycle_blocks.json。', '',
              '所有原始文件与生成器 SHA-256、完整目标 placement/字节、NPU 无其他请求 IO 混入、未分箱的累计面积校验均保留在 [evidence.json](evidence.json)。']
    (out/'README.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps(dict(image=image, selection=evidence['selection'], npu=evidence['selected']['npu'], request_id=evidence['selected']['request_id'], C_ms=evidence['C_ms'], wait_ms=evidence['wait_ms'], D_ms=evidence['D_ms'], B_GiB_s=evidence['B_GiB_s'], average_b_GiB_s=evidence['average_b_GiB_s'], local_cycle_U_percent=evidence['local_cycle_U_percent']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
