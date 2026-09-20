#!/usr/bin/env python3
"""Plot frozen full-run timelines and exact block-service bandwidth evidence."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

FONT = Path('/home/chguo/.fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'pdf.fonttype': 42,
                     'svg.fonttype': 'path', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
COLOR = {'A': '#277da8', 'B': '#52a68a', 'stall': '#f0a340'}
DESCRIPTION = 'A：128K 总长 / 新增 256；B：32K 总长 / 新增 4K；每卡请求数 A:B=1:2'


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'wt') as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def role(load):
    key = (int(load['seq_len_k']), int(load['nql']))
    return {(128, 256): 'A', (32, 4096): 'B'}[key]


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=155)
    plt.close(fig)


def overlap(a, z, left, right):
    return max(0., min(z, right) - max(a, left))


def collect(case, left, right):
    man, raw = read(case / 'manifest.json.gz'), read(case / 'result.json.gz')
    command = read(case / 'command.json')
    assert command['status'] == 'complete' and command['returncode'] == 0
    assert command['completed_simulation'] is True
    assert sha(case / 'manifest.json.gz') == command['manifest_sha256']
    assert sha(case / 'result.json.gz') == command['output_sha256']
    assert man['input_fingerprint'] == raw['input_fingerprint']
    assert raw['strategy'] == command['strategy'] == 'baseline'
    assert raw['policy_config']['assignment'] == 'fixed'
    assert all(raw['summary']['invariants'].values())
    reqs = {r['request_id']: r for r in man['requests']}
    spans = [{k: [] for k in COLOR} for _ in range(32)]
    sums = {k: np.zeros(32) for k in COLOR}
    active = np.zeros(32)
    lanes = defaultdict(list)
    for batch in raw['summary']['microbatch_metrics']:
        assert batch['batch_size'] == 1
        n = batch['npu_id']; rid = batch['member_request_ids'][0]
        k = role(reqs[rid]['load']); lanes[n].append(batch)
        active[n] += overlap(batch['admission_time_ms'], batch['completion_time_ms'], left, right)
        previous = batch['admission_time_ms']
        for layer in batch['layer_metrics']:
            cs, ce = layer['compute_start_ms'], layer['compute_end_ms']
            assert math.isclose(cs - previous, layer['io_barrier_wait_ms'], abs_tol=1e-6)
            for kind, a, z in (('stall', previous, cs), (k, cs, ce)):
                duration = overlap(a, z, left, right)
                if duration:
                    spans[n][kind].append((max(left, a), duration))
                    sums[kind][n] += duration
            previous = ce
    total = sum(sums.values())
    assert np.allclose(total, active, atol=1e-5)
    assert np.allclose(active, right - left, atol=1e-5), 'Window must precede queue exhaustion.'
    u = 100 * (sums['A'] + sums['B']) / (right - left)
    return dict(man=man, raw=raw, reqs=reqs, spans=spans, sums=sums, lanes=lanes,
                U=float(u.mean()), per_npu_U=u, num_ssu=man['metadata']['num_ssu'])


def timeline(case, out, label, left, right):
    d = collect(case, left, right)
    fig, ax = plt.subplots(figsize=(16, 10.8))
    fig.subplots_adjust(left=.07, right=.96, bottom=.13, top=.81)
    fig.suptitle(f'Baseline {label}：32 张 NPU 的计算与 I/O 等待', y=.974, fontsize=21)
    fig.text(.5, .929, f'32 NPU / {d["num_ssu"]} SSU × 40 GiB/s · '
             f'[{left/1000:g}, {right/1000:g}) 秒 · NPU 平均利用率 {d["U"]:.2f}%',
             ha='center', fontsize=15)
    fig.text(.5, .892, DESCRIPTION, ha='center', fontsize=11)
    fig.legend(handles=[Patch(color=COLOR[k], label=v) for k, v in
                        [('A', 'A 请求计算'), ('B', 'B 请求计算'), ('stall', 'I/O 等待')]],
               loc='upper center', bbox_to_anchor=(.5, .874), ncol=3, frameon=False)
    for n in range(32):
        for kind in COLOR:
            ax.broken_barh(d['spans'][n][kind], (n-.4, .8), facecolors=COLOR[kind], linewidth=0)
        ax.text(right+(right-left)*.007, n, f'{d["per_npu_U"][n]:.1f}%', va='center', fontsize=8)
    ax.set(xlim=(left, right), ylim=(31.8, -.8), yticks=range(32),
           xlabel='仿真时间（秒）', ylabel='NPU 编号')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}'))
    ax.grid(axis='x', alpha=.15)
    both = int(np.count_nonzero((d['sums']['A'] > 0) & (d['sums']['B'] > 0)))
    fig.text(.07, .075, f'全窗有任务：32/32 卡；本窗实际计算过 A 和 B：{both}/32 卡。每请求 8 层，全部计算计入利用率。', fontsize=10)
    fig.text(.07, .042, '橙色为接纳后等待读取；不含到达后等待接纳。图为完整仿真的真实时间区间。', fontsize=10)
    save(fig, out / f'baseline_{label.lower()}')
    return dict(U_percent=d['U'], per_npu_U_percent=d['per_npu_U'].tolist(),
                both_profiles_cards=both, window_ms=[left, right], num_ssu=d['num_ssu'])


def add_interval(row, a, z, value, left, right, dt):
    a, z = max(a, left), min(z, right)
    if z <= a:
        return
    n = len(row); i = min(n-1, int((a-left)//dt)); j = min(n-1, int((z-left)//dt))
    row[i] += value * (min(z, left+(i+1)*dt)-a)/dt
    if j > i:
        row[i+1:j] += value
        row[j] += value * (z-(left+j*dt))/dt


def service_bins(arr, start, end, resource, resources, rate, left, right, dt):
    count = round((right-left)/dt)
    a, z = np.maximum(left, arr[:, start]), np.minimum(right, arr[:, end])
    keep = z > a; a, z = a[keep], z[keep]; r = arr[keep, resource].astype(int)
    assert np.all(z-a < dt), 'Expected sub-bin physical block service.'
    i = np.minimum(count-1, ((a-left)//dt).astype(int))
    j = np.minimum(count-1, ((z-left)//dt).astype(int))
    bins = np.zeros((resources, count))
    np.add.at(bins, (r, i), rate*(np.minimum(z, left+(i+1)*dt)-a)/dt)
    cross = j > i
    np.add.at(bins, (r[cross], j[cross]), rate*(z[cross]-(left+j[cross]*dt))/dt)
    expected = np.bincount(r, weights=(z-a)*rate/1000, minlength=resources)
    assert np.allclose(expected, bins.sum(axis=1)*dt/1000, atol=1e-8, rtol=1e-9)
    return bins


def stage_check(arr, start, end, resource, resources, rate):
    assert np.allclose(arr[:, end]-arr[:, start], arr[:, 6]*1000/rate, atol=1e-8, rtol=1e-9)
    for r in range(resources):
        lane = arr[arr[:, resource] == r]
        lane = lane[np.argsort(lane[:, start], kind='stable')]
        assert np.all(lane[1:, start] >= lane[:-1, end]-1e-8), 'Physical service overlap.'


def bandwidth(case, out, label, left, right, dt):
    d = collect(case, left, right)
    command = read(case / 'command.json')
    trace = read(case / 'trace.json.gz')
    assert trace['completed_simulation'] is True and trace['strategy'] == 'baseline'
    assert trace['checks'] and all(v is True for v in trace['checks'].values())
    assert trace['observed_completed_blocks'] == trace['expected_completed_blocks']
    assert trace['retained_blocks'] == len(trace['rows'])
    assert trace['window_ms'][0] <= left < right <= trace['window_ms'][1], 'Plot exceeds captured service window.'
    assert sha(case / 'trace.json.gz') == command['trace_sha256']
    source = trace['source']
    # Hash local files, rather than dereferencing paths recorded on the remote host.
    assert source['manifest_sha256'] == command['manifest_sha256']
    assert source['reference_sha256'] == command['output_sha256']
    assert source['input_fingerprint'] == d['raw']['input_fingerprint']
    assert source['core_source_sha256'] == d['raw']['core_and_policy_sha256'] == command['core_source_sha256']
    expected = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
                'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
                'link_start_ms', 'link_end_ms']
    assert trace['columns'] == expected
    arr = np.asarray(trace['rows'], dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 13 and len(arr) > 0
    assert np.all(np.isfinite(arr)) and np.all(arr[:, 6] > 0)
    assert np.all(arr[:, 5] == 0) and np.all(arr[:, 7] == 1)
    assert np.all(arr[:, 8] <= arr[:, 9]+1e-8) and np.all(arr[:, 10] <= arr[:, 11]+1e-8)
    stage_check(arr, 9, 10, 4, d['num_ssu'], 40.)
    stage_check(arr, 11, 12, 1, 32, 50.)
    ssd = service_bins(arr, 9, 10, 1, 32, 40., left, right, dt)
    link = service_bins(arr, 11, 12, 1, 32, 50., left, right, dt)
    disks = service_bins(arr, 9, 10, 4, d['num_ssu'], 40., left, right, dt)
    assert disks.max() <= 40+1e-6 and link.max() <= 50+1e-6
    assert np.allclose(ssd.sum(axis=0), disks.sum(axis=0), atol=1e-6)
    nbin = ssd.shape[1]
    demand = np.zeros_like(ssd); stall = np.zeros_like(ssd)
    for n, batches in d['lanes'].items():
        for b in batches:
            q = d['reqs'][b['member_request_ids'][0]]['load']
            nominal = q['per_layer_kv_gb']*1e6/q['per_layer_us']
            add_interval(demand[n], b['admission_time_ms'], b['completion_time_ms'], nominal, left, right, dt)
        for a, duration in d['spans'][n]['stall']:
            add_interval(stall[n], a, a+duration, 1., left, right, dt)
    assert np.allclose(stall.sum(axis=1)*dt, d['sums']['stall'], atol=1e-6)
    edges = left + np.arange(nbin+1)*dt
    fields = ['npu', 'start_ms', 'end_ms', 'nominal_demand_gib_s', 'ssd_service_gib_s', 'npu_receive_gib_s', 'stall_fraction']
    with gzip.open(out / f'bandwidth_{label.lower()}.csv.gz', 'wt', newline='') as stream:
        w = csv.writer(stream); w.writerow(fields)
        for n in range(32):
            for i in range(nbin):
                w.writerow([n, edges[i], edges[i+1], demand[n, i], ssd[n, i], link[n, i], stall[n, i]])
    summary = []
    for n in range(32):
        summary.append(dict(npu=n, U_percent=float(d['per_npu_U'][n]),
                            nominal_demand_gib_s=float(demand[n].mean()),
                            ssd_service_gib_s=float(ssd[n].mean()), npu_receive_gib_s=float(link[n].mean())))
    curves = dict(edges_ms=edges.tolist(), nominal=demand.tolist(), ssd=ssd.tolist(),
                  link=link.tolist(), stall=stall.tolist(), disk=disks.tolist(), summary=summary,
                  num_ssu=d['num_ssu'], U_percent=d['U'], window_ms=[left, right], bin_ms=dt)
    write(out / f'bandwidth_{label.lower()}.json.gz', curves)
    # Select examples only if all blocks of a complete layer were captured.
    candidates = defaultdict(list)
    for n, batches in d['lanes'].items():
        previous = None
        for b in sorted(batches, key=lambda x: x['admission_time_ms']):
            rid = b['member_request_ids'][0]; req = d['reqs'][rid]; k = role(req['load'])
            for layer in b['layer_metrics']:
                if previous is not None and left <= previous['compute_end_ms'] < right:
                    candidates[k].append((layer['io_barrier_wait_ms'], rid, layer, previous, n))
                previous = layer
    examples = {}
    for k in ('A', 'B'):
        for wait, rid, layer, prev, n in sorted(candidates[k], key=lambda x: -x[0]):
            blocks = arr[(arr[:, 0] == rid) & (arr[:, 2] == layer['layer'])]
            req = d['reqs'][rid]
            placement_layers = d['man']['placements'][req['placement_index']]
            placement = placement_layers[0 if len(placement_layers) == 1 else layer['layer']]
            if len(blocks) != len(placement):
                continue
            blocks = blocks[np.argsort(blocks[:, 3], kind='stable')]
            assert np.array_equal(blocks[:, 3], np.arange(len(placement))), 'Duplicate or missing block identity.'
            assert np.all(blocks[:, 1] == n) and np.all(blocks[:, 7] == 1)
            assert all(int(row[4]) == disk and row[6] == volume
                       for row, (disk, volume) in zip(blocks, placement)), 'Block placement/payload mismatch.'
            assert math.isclose(blocks[:, 6].sum(), req['load']['per_layer_kv_gb'], abs_tol=1e-10)
            assert math.isclose(blocks[:, 12].max(), layer['io_ready_time_ms'], abs_tol=1e-7)
            a = layer['io_start_time_ms']; deadline = prev['compute_end_ms']; ready = layer['io_ready_time_ms']
            assert math.isclose(a, prev['compute_start_ms'], abs_tol=1e-7)
            assert math.isclose(max(0., ready-deadline), wait, abs_tol=1e-7)
            times = np.unique(np.r_[blocks[:, 9:13].ravel(), a, deadline, ready])
            payload = blocks[:, 6].sum()*1024
            curves2 = {}
            for name, start, end in [('SSD 已服务', 9, 10), ('NPU 已收到', 11, 12)]:
                changes = np.zeros(len(times))
                rates = blocks[:, 6]*1024/(blocks[:, end]-blocks[:, start])
                np.add.at(changes, np.searchsorted(times, blocks[:, start]), rates)
                np.add.at(changes, np.searchsorted(times, blocks[:, end]), -rates)
                values = np.r_[0., np.cumsum(np.cumsum(changes)[:-1]*np.diff(times))]
                assert math.isclose(values[-1], payload, abs_tol=1e-5)
                curves2[name] = values
            missing = payload-float(np.sum(blocks[:, 6]*1024*np.clip((deadline-blocks[:, 11])/(blocks[:, 12]-blocks[:, 11]), 0, 1)))
            fig, ax = plt.subplots(figsize=(12, 5.8)); fig.subplots_adjust(top=.76, bottom=.18)
            fig.suptitle(f'Baseline {label} · NPU {n} · {k} 请求第 {layer["layer"]} 层', fontsize=17, y=.96)
            fig.text(.5, .89, f'预算 {deadline-a:.3f} ms；读取 {payload:.3f} MiB；I/O 等待 {wait:.3f} ms', ha='center', fontsize=12)
            for name, values in curves2.items():
                ax.plot(times-a, values, label=name, lw=1.7)
            ax.plot([0, deadline-a], [0, payload], '--', color='#ab486d', label='按计算预算均匀读完的参考线')
            ax.axhline(payload, color='#777', lw=.7)
            ax.axvline(deadline-a, color='#bf5b27', ls='--', label='计算结束：需要全部数据')
            if ready > deadline:
                ax.axvspan(deadline-a, ready-a, color=COLOR['stall'], alpha=.2)
            ax.set(xlabel='从本层读取释放起经过的时间（ms）', ylabel='累计数据（MiB）')
            ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=.15)
            fig.text(.10, .065, f'截止时 NPU 还缺 {max(0., missing):.3f} MiB。第 0 层预算来自上一请求的最后一层计算。', fontsize=10)
            save(fig, out / f'deadline_{label.lower()}_{k}')
            examples[k] = dict(npu=n, request_id=rid, original_request_id=req['load'].get('original_request_id'),
                               layer=layer['layer'], stall_ms=wait, release_ms=a, deadline_ms=deadline,
                               ready_ms=ready, payload_MiB=payload, missing_at_deadline_MiB=max(0., missing),
                               captured_blocks=len(blocks), expected_blocks=len(placement),
                               full_layer_identity_and_placement_verified=True)
            break
    write(out / f'deadline_{label.lower()}.json', examples)
    return curves


def pair_bandwidth(random, ordered, out):
    data = {'Random': random, 'Ordered': ordered}
    left, right = random['window_ms']; assert ordered['window_ms'] == [left, right]
    edges = np.array(random['edges_ms'])/1000
    num_ssu = random['num_ssu']; assert ordered['num_ssu'] == num_ssu
    for n in range(32):
        fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [3, .55, 3, .55]})
        fig.subplots_adjust(top=.84, bottom=.12, hspace=.23, left=.07, right=.98)
        fig.suptitle(f'NPU {n:02d}：带宽需求与实际获得的服务', fontsize=20, y=.974)
        fig.text(.5, .93, f'32 NPU / {num_ssu} SSU × 40 GiB/s · 每 {random["bin_ms"]:g} ms 按真实服务时间积分', ha='center', fontsize=12)
        fig.text(.5, .888, DESCRIPTION, ha='center', fontsize=10)
        for j, (label, d) in enumerate(data.items()):
            ax, state = axes[2*j:2*j+2]
            for key, name, color in [('ssd', 'SSD 实际服务', '#718abd'), ('link', 'NPU 实际收到', '#23385c'), ('nominal', '当前请求名义需求 V/C', '#ca4e71')]:
                ax.stairs(d[key][n], edges, label=name, color=color, lw=1.0 if key != 'nominal' else 1.5,
                          alpha=.75 if key == 'ssd' else 1., linestyle='--' if key == 'nominal' else '-')
            ax.set_title(f'{label} · 本卡 U={d["summary"][n]["U_percent"]:.2f}%', loc='left', fontsize=11)
            ax.set_ylabel('GiB/s'); ax.set_ylim(bottom=0); ax.grid(alpha=.12)
            if j == 0:
                ax.legend(loc='upper right', ncol=3, fontsize=8)
            state.stairs(d['stall'][n], edges, fill=True, color=COLOR['stall'])
            state.set(ylim=(0, 1), yticks=[], ylabel='I/O 等待')
            state.spines['left'].set_visible(False)
        axes[-1].set_xlabel('仿真时间（秒）'); axes[-1].set_xlim(left/1000, right/1000)
        fig.text(.07, .065, '名义需求是当前请求的 V/C 参考值；等待期间仍显示它，但其积分不是新增读取量。', fontsize=10)
        fig.text(.07, .03, '等待源于数据没有在截止前到齐；等待时实际带宽也可能高于参考值。请结合累计字节图判断。', fontsize=10)
        save(fig, out / 'per_npu' / f'npu_{n:02d}')
    for label, d in data.items():
        fig, axes = plt.subplots(3, 1, figsize=(15, 9.5), sharex=True)
        fig.subplots_adjust(top=.86, bottom=.12, hspace=.31)
        fig.suptitle(f'Baseline {label} · {num_ssu} SSU：每卡带宽与磁盘服务', y=.97, fontsize=20)
        for ax, key, title in zip(axes[:2], ['nominal', 'link'], ['每卡名义需求 V/C', '每卡实际收到的带宽']):
            im = ax.imshow(d[key], aspect='auto', origin='upper', extent=[left/1000, right/1000, 31.5, -.5], vmin=0, vmax=50, cmap='viridis')
            ax.set(ylabel='NPU', title=title, yticks=[0, 8, 16, 24, 31]); fig.colorbar(im, ax=ax, label='GiB/s', fraction=.025)
        for disk in range(num_ssu):
            axes[2].stairs(d['disk'][disk], edges, label=f'SSU {disk}', lw=.8)
        axes[2].axhline(40, ls='--', color='black', label='每盘上限 40')
        axes[2].set(ylabel='GiB/s', xlabel='仿真时间（秒）', title='逐盘实际服务带宽', ylim=(0, 43))
        axes[2].legend(ncol=min(7, num_ssu+1), fontsize=9)
        fig.text(.125, .048, '实际服务按块的物理传输区间积分；没有把包含排队的整层读取时长当作磁盘服务时长。', fontsize=10)
        save(fig, out / f'bandwidth_overview_{label.lower()}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--random', type=Path, required=True); p.add_argument('--ordered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--start', type=float, default=2000.); p.add_argument('--end', type=float, default=4000.)
    p.add_argument('--bin-ms', type=float, default=.5)
    p.add_argument('--timeline-only', action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    assert math.isfinite(args.start) and math.isfinite(args.end) and 0 <= args.start < args.end
    assert math.isfinite(args.bin_ms) and args.bin_ms > 0
    nbin = (args.end-args.start)/args.bin_ms
    assert nbin >= 1 and math.isclose(nbin, round(nbin), abs_tol=1e-9, rel_tol=0), 'Window must contain a whole number of bins.'
    summaries = {}
    for label, case in [('Random', args.random), ('Ordered', args.ordered)]:
        summaries[label] = timeline(case, args.out, label, args.start, args.end)
    write(args.out / 'timeline_summary.json', summaries)
    if not args.timeline_only:
        r = bandwidth(args.random, args.out, 'Random', args.start, args.end, args.bin_ms)
        o = bandwidth(args.ordered, args.out, 'Ordered', args.start, args.end, args.bin_ms)
        pair_bandwidth(r, o, args.out)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
