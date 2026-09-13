#!/usr/bin/env python3
"""Plot one audited Saturday cycle directly from block arrivals; no simulation."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qos_bi_alignment_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT/'results/baseline_ab128_32_ratio12_20260912'
OUT = HERE/'assets'
SOURCES = {}
FONT_PATH = Path('/home/chguo/.fonts/msyh.ttc')
if not FONT_PATH.exists():
    FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FONT = FontProperties(fname=str(FONT_PATH))
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'pdf.fonttype': 42, 'font.size': 13})
matplotlib.font_manager.fontManager.addfont(str(FONT_PATH))
BLUE, RED, GREEN, INK = '#225bc5', '#be3f65', '#14846b', '#17283d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
        return json.load(f)


def close(a, b):
    assert np.isclose(a, b, rtol=1e-10, atol=1e-7), (a, b)


def main():
    h = read(STUDY/'formula_review/handoff_checks.json')
    assert h['technical_passed']
    case = STUDY/'validation20s/runs/ssu4_ordered_k1_sync_seed7/baseline'
    raw = read(case/'result.json.gz')
    trace = read(case/'trace.json.gz')
    man = read(case/'manifest.json.gz')
    for name in ('result', 'trace', 'manifest'):
        path = case/(name+'.json.gz')
        assert SOURCES[str(path.relative_to(ROOT))] == h['sources'][name]['sha256']
    assert trace['completed_simulation'] and all(trace['checks'].values())
    assert trace['columns'] == ['request_id','npu_id','layer','block_idx','ssu_id','path_id',
        'size_gib','block_count','enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms']
    arr = np.asarray(trace.pop('rows'), dtype=float)
    row = h['A_local']['per_npu_handoffs'][0]
    rid, npu, layer = row['request_id'], row['npu'], row['layer']
    assert (rid, npu, layer) == (9, 0, 2)
    left, deadline, right = (row[k] for k in ('release_ms','deadline_ms','ready_ms'))
    C, R = deadline-left, right-left
    target = arr[(arr[:,0] == rid) & (arr[:,2] == layer)]
    request = next(q for q in man['requests'] if q['request_id'] == rid)
    # This manifest stores one reusable block placement for every layer.
    placements = man['placements'][request['placement_index']]
    assert len(placements) == 1
    placement = placements[0]
    target = target[np.argsort(target[:,3])]
    assert len(target) == len(placement) == 1022
    assert np.array_equal(target[:,3], np.arange(len(placement)))
    for record, (disk, size) in zip(target, placement):
        assert record[1] == npu and record[4] == disk and record[6] == size
    V = float(target[:,6].sum())
    B = V*1000/C
    close(V*1024, h['A_local']['layer_V_MiB'])
    close(target[:,12].max(), right)
    close(C, request['load']['per_layer_us']/1000)
    # Verify the target accounts for all arrivals to this card in this cycle.
    lane = arr[arr[:,1] == npu]
    clipped = np.maximum(0, np.minimum(lane[:,12],right)-np.maximum(lane[:,11],left))
    close(float((clipped*50/1000).sum()), V)
    assert np.allclose(target[:,12]-target[:,11],target[:,6]*1000/50,atol=1e-8)
    assert np.all(target[:,11] >= left-1e-8)
    assert np.all(target[:,12] <= right+1e-8)
    # Event-exact piecewise-constant NPU arrival rate, not bin averages.
    times, inverse = np.unique(np.r_[left, right, target[:,11], target[:,12]], return_inverse=True)
    delta = np.zeros(len(times))
    np.add.at(delta, inverse[2:2+len(target)], 50.)
    np.add.at(delta, inverse[2+len(target):], -50.)
    rates = np.cumsum(delta)[:-1]
    assert rates.min() >= -1e-8 and rates.max() <= 50+1e-8
    received = np.r_[0., np.cumsum(rates*np.diff(times)/1000*1024)]
    close(received[-1], V*1024)
    mean_b = received[-1]/1024*1000/R
    close(mean_b, 4*40/32)
    batch = next(b for b in raw['summary']['microbatch_metrics'] if b['member_request_ids'] == [rid])
    previous, following = batch['layer_metrics'][layer-1:layer+1]
    close(previous['compute_start_ms'], left)
    close(previous['compute_end_ms'], deadline)
    close(following['compute_start_ms'], right)
    close(following['io_barrier_wait_ms'], R-C)
    compute_U = 100*C/R
    model_U = 100*min(1.,mean_b/B)
    close(compute_U,model_U)
    arrived_at_deadline = float(np.sum(target[:,6]*1024*np.clip(
        (deadline-target[:,11])/(target[:,12]-target[:,11]),0,1)))
    close(arrived_at_deadline,0.)

    fig = plt.figure(figsize=(16,11.6), dpi=120, facecolor='white')
    fig.text(.065,.962,'周六真实日志：把 b_i、B_i 和利用率放在同一轮里看',fontsize=24,color=INK)
    fig.text(.065,.927,'32 NPU / 4 SSU × 40 GiB/s · Baseline Ordered · seed 7 · NPU 0 · A 请求 9',fontsize=14,color='#596a80')
    fig.text(.065,.888,f'单层 V = {V*1024:.3f} MiB    C = {C:.3f} ms    B_i = V/C = {B:.2f} GiB/s',fontsize=17,color=INK)
    ax_b = fig.add_axes([.095,.59,.86,.245])
    ax_v = fig.add_axes([.095,.315,.86,.225],sharex=ax_b)
    ax_c = fig.add_axes([.095,.18,.86,.07],sharex=ax_b)
    x = times-left
    xmax = R+1
    for ax in (ax_b, ax_v):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.14)
        ax.axvspan(C,R,color='#f9d9d4',alpha=.37,zorder=0)
        ax.axvline(C,color=RED,linestyle=':',lw=1.5)
        ax.axvline(R,color='#7a8390',linestyle=':',lw=1.)
        ax.tick_params(labelbottom=False)
    ax_b.stairs(rates,x,color=BLUE,lw=1.4,fill=True,alpha=.30)
    ax_b.stairs(rates,x,color=BLUE,lw=1.5,label='b_i(t)：NPU 实际接收速率')
    ax_b.hlines(B,0,R,color=RED,linestyle='--',lw=2,label=f'B_i：参考需求 {B:.2f}')
    ax_b.hlines(mean_b,0,R,color=GREEN,linestyle='-.',lw=2,label=f'本轮平均 b_i：{mean_b:.2f}')
    ax_b.set(xlim=(0,xmax),ylim=(0,66),ylabel='带宽（GiB/s）',yticks=[0,5,28.4759259269,50])
    ax_b.set_yticklabels(['0','5','28.48','50'])
    ax_b.legend(loc='upper left',ncol=3,fontsize=12,frameon=False)
    ax_b.text(11,40,'大部分时间没有数据到达\n最后集中以 50 GiB/s 到达',fontsize=16,color=INK,linespacing=1.5)
    ax_v.plot([0,C,R],[0,V*1024,V*1024],color=RED,ls='--',lw=2,label='按 B_i 均匀读取的参考进度')
    ax_v.plot(x,received,color=BLUE,lw=2,label='实际到达的目标层数据')
    ax_v.set(ylabel='累计到达（MiB）',ylim=(-8,V*1024*1.27),yticks=[0,50,100,175.65625])
    ax_v.set_yticklabels(['0','50','100','175.66'])
    ax_v.legend(loc='upper left',ncol=2,fontsize=12,frameon=False)
    ax_v.scatter([C],[0],color=RED,s=38,zorder=5)
    ax_v.annotate('计算已经结束\n175.66 MiB 还一字节未到',xy=(C,0),xytext=(11,65),
                  fontsize=15,color=INK,arrowprops={'arrowstyle':'->','color':RED})
    ax_c.barh(.5,C,left=0,height=.8,color=BLUE)
    ax_c.barh(.5,R-C,left=C,height=.8,color='#f5c9c1')
    ax_c.text(C/2,.5,f'计算\n{C:.3f} ms',color='white',ha='center',va='center',fontsize=14)
    ax_c.text((C+R)/2,.5,f'IO stall：{R-C:.3f} ms',color='#ac3d32',ha='center',va='center',fontsize=17)
    ax_c.set(ylim=(0,1),yticks=[],xlabel='从本次预取发出起经过的时间（ms）',xticks=[0,C,10,20,30,R])
    ax_c.set_xticklabels(['0',f'{C:.3f}','10','20','30',f'{R:.3f}'])
    ax_c.spines[['top','right','left']].set_visible(False)
    fig.text(.065,.085,f'同一轮：平均 b_i / B_i = 5 / {B:.2f} = {model_U:.2f}%    实测 U = {C:.3f} / {R:.3f} = {compute_U:.2f}%',fontsize=18,color=INK)
    fig.text(.065,.047,'5 是整轮平均，不是每一刻都分到 5；瞬时 b_i(t) / B_i 不能直接当作瞬时利用率。',fontsize=13,color='#596a80')
    fig.text(.065,.018,f'绝对时间 [{left:.6f}, {right:.6f}) ms；第2层计算、预取第3层。17.56% 只对应此局部周期。',fontsize=12,color='#596a80')
    OUT.mkdir(parents=True,exist_ok=True)
    stem = OUT/'bi_Bi_A_local'
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width,height = fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -2 and box.x1 <= width+2 and box.y0 >= -2 and box.y1 <= height+2, artist.get_text()
    for suffix in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+suffix),dpi=120)
    plt.close(fig)
    evidence = dict(no_new_simulation=True, sources=SOURCES, source_builder_sha256=sha(Path(__file__)),
        npu=npu,request_id=rid,read_layer_log=layer,absolute_window_ms=[left,right],
        C_ms=C,R_ms=R,stall_ms=R-C,V_MiB=V*1024,B_GiB_s=B,
        mean_b_GiB_s=mean_b,instant_arrival_rate_minmax_GiB_s=[float(rates.min()),float(rates.max())],
        bytes_arrived_before_deadline_MiB=arrived_at_deadline,model_U_percent=model_U,actual_U_percent=compute_U,
        trace_style='Exact block arrival event intervals; no time bins or smoothing.',
        byte_conservation_verified=True,all_visible_labels_inside_canvas=True,
        files={s:{'path':str(stem.with_suffix('.'+s).relative_to(ROOT)), 'sha256':sha(stem.with_suffix('.'+s))} for s in ('png','pdf','svg')})
    (HERE/'bi_alignment_checks.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('no_new_simulation','B_GiB_s','mean_b_GiB_s','model_U_percent','actual_U_percent','files')},ensure_ascii=False))


if __name__ == '__main__':
    main()
