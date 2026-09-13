#!/usr/bin/env python3
"""Combine the original top bandwidth panels into one aligned 32-card PNG."""
from pathlib import Path
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import render_per_npu_layer_avg_ssu3 as source
import warm_bandwidth_means_ssu3 as means

HERE, ROOT = source.HERE, source.ROOT
OUT = HERE/'figures/ssu3'
PNG = OUT/'random_all_32npu_layer_average.png'


def main():
    original_checks = source.OUT/'layer_average_checks.json'
    audit = json.loads(original_checks.read_text())
    assert source.base.sha(Path(source.__file__)) == audit['builder_sha256']
    assert source.base.sha(source.CACHE) == audit['dependency_checks_sha256']
    originals = {}
    for row in audit['images']:
        path = ROOT/row['path']
        if path.name.startswith('random_npu_'):
            assert source.base.sha(path) == row['sha256']
            originals[path.name] = row
    assert set(originals) == {f'random_npu_{n:02d}.png' for n in range(32)}
    _, datasets = source.verified_data()
    data = datasets['random']
    averages = means.load_verified('random')
    totals = averages['totals']
    assert len(averages['rows']) == 32
    source.base.close(data['U_percent'], audit['U_percent']['random'])
    fig, axes = source.plt.subplots(32, 1, figsize=(18,32), dpi=150,
                                    sharex=True, sharey=True, facecolor='white')
    fig.subplots_adjust(left=.108, right=.848, top=.930, bottom=.054, hspace=.34)
    fig.text(.045,.986,'Baseline Random：32 张 NPU 的每层平均带宽与需求',
             fontsize=25, color=source.INK)
    fig.text(.045,.972,f'32 NPU / 3 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 整机 U={data["U_percent"]:.2f}% · 平均每卡需求={totals["per_card_mean_demand_GiB_s"]:.3f}、供给={totals["per_card_mean_supply_GiB_s"]:.3f} GiB/s',
             fontsize=15, color=source.MUTED)
    handles = [
        Line2D([],[],color=source.PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),
        Line2D([],[],color=source.BLUE,lw=2.3,marker='o',markerfacecolor='white',
               label='平均 b_i：每个完整内部层周期一个值'),
        Patch(facecolor=source.GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.043,.965),
               frameon=False,ncol=3,fontsize=13)
    fig.text(.045,.943,'每行一张卡；左侧 U 和右侧带宽均统计完整 warm [2,4) 秒。蓝线仍按各自层周期取均值。',
             fontsize=13,color=source.MUTED)
    fig.text(.865,.937,'整窗平均（GiB/s）',fontsize=12,color=source.INK)
    rows = []
    for npu, ax in enumerate(axes):
        lane = data['lanes'][npu]
        avg = averages['rows'][npu]
        assert avg['npu'] == npu
        source.base.close(lane['U_percent'],avg['U_percent'])
        source.bandwidth(ax,lane)
        ax.set_yticks([0,10,28.476],['0','10','28.48'])
        ax.set_ylabel(f'NPU {npu:02d}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=16,
                      color=source.INK)
        ax.tick_params(axis='y',labelsize=9,length=3)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu in (7,15,23,31))
        ax.grid(axis='x',alpha=.15,lw=.6)
        demand_label=f'需求 {avg["mean_demand_GiB_s"]:.3f}'
        supply_label=f'供给 {avg["mean_supply_GiB_s"]:.3f}'
        ax.text(1.022,.70,demand_label,transform=ax.transAxes,ha='left',va='center',fontsize=11,color=source.PURPLE)
        ax.text(1.022,.28,supply_label,transform=ax.transAxes,ha='left',va='center',fontsize=11,color=source.BLUE)
        assert list(ax.get_xlim()) == [2.,4.] and list(ax.get_ylim()) == [0.,32.5]
        assert len(ax.collections[2].get_offsets()) == sum(c['group']!='gray' for c in lane['cycles'])
        rows.append(dict(npu=npu,U_percent=lane['U_percent'],U_label=f'U={lane["U_percent"]:.2f}%',
                         source_png=originals[f'random_npu_{npu:02d}.png'],
                         mean_demand_GiB_s=avg['mean_demand_GiB_s'],mean_supply_GiB_s=avg['mean_supply_GiB_s'],
                         mean_demand_label=demand_label,mean_supply_label=supply_label,
                         cycle_count=len(lane['cycles']),
                         internal_cycle_count=sum(c['group']!='gray' for c in lane['cycles'])))
    axes[-1].set_xlabel('仿真时间（秒）；所有行带宽单位均为 GiB/s',fontsize=13,labelpad=10)
    footer = fig.text(.045,.024,'周期 D = 当前层开始计算 → 下一层开始计算（含等待）；平均 b_i = 周期内实际收到的下一层数据量 / D。',
             fontsize=13,color=source.INK)
    fig.text(.045,.013,'右侧需求 = B_i 按时间加权；供给 = 2秒内实际收到的数据量 / 2秒（含灰区）。两个整窗均值相除不等于 U。',
             fontsize=12,color=source.MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width,height = fig.canvas.get_width_height()
    assert footer.get_window_extent(renderer).y1+6 < axes[-1].xaxis.label.get_window_extent(renderer).y0
    for artist in fig.findobj(source.fleet.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b = artist.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=width+2 and b.y1<=height+2,(artist.get_text(),b.bounds)
    fig.savefig(PNG,dpi=150)
    source.plt.close(fig)
    check = dict(all_checks_passed=True,no_new_simulation=True,format='png',
                 num_npu=32,num_ssu=3,order='random',seed=7,window_ms=[2000.,4000.],
                 image=str(PNG.relative_to(ROOT)),sha256=source.base.sha(PNG),pixels=[width,height],
                 original_top_panel_function='render_per_npu_layer_avg_ssu3.bandwidth',
                 shared_xlim_seconds=[2.,4.],shared_ylim_GiB_s=[0.,32.5],
                 builder_sha256=source.base.sha(Path(__file__)),
                 original_checks_sha256=source.base.sha(original_checks),
                 original_renderer_sha256=source.base.sha(Path(source.__file__)),
                 original_cycle_checks_sha256=source.base.sha(source.CACHE),
                 warm_bandwidth_summary_sha256=source.base.sha(means.PATH),
                 warm_bandwidth_builder_sha256=source.base.sha(Path(means.__file__)),
                 whole_window_bandwidth_displayed=True,whole_window_bandwidth_includes_gray=True,
                 bandwidth_totals=totals,
                 only_original_top_panel=True,per_npu_U_displayed=True,
                 per_npu_U_definition='Actual compute time in warm [2000,4000)ms divided by 2000ms',
                 visible_labels_inside_canvas=True,rows=rows,
                 U_percent=data['U_percent'])
    PNG.with_suffix('.checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:check[k] for k in ('image','pixels','U_percent','all_checks_passed')},ensure_ascii=False))


if __name__=='__main__':main()
