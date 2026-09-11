#!/usr/bin/env python3
"""Eight standalone plots from frozen seed-7 Baseline traces; no simulation."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
sys.path.insert(0, str(PARENT))

import plot_case
from plot_separate import configure_font
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

OUT = HERE / "figures" / "separate"
START, END = 2000.0, 4000.0
PROFILE_COUNTS = {(1, 128): 6400, (1, 256): 6400, (1, 384): 6400, (192, 768): 256}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(case, n_long, order, kind):
    timeline = kind == "timeline"
    n_short = 32-n_long
    a = case["audit"]
    name = "Random 短卡乱序" if order == "random" else "RR 短画像轮转"
    subject = "计算与等待时间线" if timeline else "六盘名义需求"
    explanation = ("短卡：分别打乱各自完整请求清单；长卡：固定同一长画像"
                   if order == "random" else
                   "短卡：S1 → S2 → S3 轮转至清单耗尽；长卡：固定同一长画像")
    fig, ax = plt.subplots(figsize=(15, 10.6 if timeline else 8.4))
    fig.subplots_adjust(left=.077, right=.985, top=.741 if timeline else .752,
                        bottom=.188 if timeline else .345)
    title = f"{n_long} 长 / {n_short} 短 · {name}：Baseline {subject}"
    fig.suptitle(title, fontsize=22, y=.976)
    fig.text(.5, .928,
             f"32 NPU / 6 SSU · 每盘 40 GiB/s · [2, 4) 秒 · 设备 U = {100*case['util']:.4f}%",
             ha="center", va="top", fontsize=16)
    fig.text(.5, .883,
             f"长卡 NPU 0–{n_long-1}（仅 LL）  |  短卡 NPU {n_long}–31（仅 SS）",
             ha="center", va="top", fontsize=15)
    fig.text(.5, .844,
             f"名义最热盘峰值：全程 {max(a['full_run_current_profile_peak_by_ssu_gib_s']):.4f} / "
             f"暖窗 {max(a['window_current_profile_peak_by_ssu_gib_s']):.4f} GiB/s",
             ha="center", va="top", fontsize=14)
    fig.text(.5, .806, explanation, ha="center", va="top", fontsize=13.5, color="#454545")
    if timeline:
        fig.legend(handles=[Patch(color=plot_case.COLORS['long'], label="长卡计算（LL）"),
                            Patch(color=plot_case.COLORS['short'], label="短卡计算（SS）"),
                            Patch(color=plot_case.COLORS['stall'], label="接纳后 I/O 等待"),
                            Patch(color="#eceff1", label="空闲（本窗为 0）")],
                   loc="upper center", bbox_to_anchor=(.52, .780), ncol=4,
                   frameon=False, fontsize=12.5, columnspacing=1.5)
        for n, spans in enumerate(case['spans']):
            ax.broken_barh([(START, END-START)], (n-.41, .82), facecolors="#eceff1", edgecolors="none")
            for role in ('long', 'short', 'stall'):
                if spans[role]:
                    ax.broken_barh(spans[role], (n-.41, .82), facecolors=plot_case.COLORS[role],
                                   edgecolors="none", linewidth=0, antialiased=False, rasterized=True)
        ax.axhline(n_long-.5, color="#333333", linewidth=1.2)
        ax.set_ylim(31.8, -.8)
        ax.set_yticks(range(32))
        ax.tick_params(axis='y', labelsize=11, length=3)
        ax.set_ylabel("计算卡编号（NPU）", fontsize=16, labelpad=12)
        fig.text(.077, .104,
                 "短 SS：1K/128、1K/256、1K/384（外推）；长 LL：192K/768（插值）；全局 19,456 请求。",
                 fontsize=12)
        fig.text(.077, .074,
                 "橙色为接纳后的实际 I/O barrier 等待，排除接纳前排队；32 卡整窗 active，区间未移动或加宽。",
                 fontsize=12)
        fig.text(.077, .045,
                 "请求 arrival 均为 0；重绑保留每条原请求的 C/V 和物理 placement；同绑定 random/RR 保持每卡人口一致。",
                 fontsize=11.5)
    else:
        palette = plt.get_cmap('tab10').colors
        for disk in range(6):
            ax.step(case['times'], case['rates'][:, disk], where='post', color=palette[disk],
                    linewidth=1.0, alpha=.85, label=f"盘 {disk}")
        ax.axhline(40, color="#aa2424", linestyle='--', linewidth=1.6, label="每盘容量 40 GiB/s")
        ax.set_ylim(0, 43)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("每盘名义需求（GiB/s）", fontsize=16, labelpad=12)
        ax.grid(axis='y', color="#d9dee1", linewidth=.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .206),
                   ncol=7, frameon=False, fontsize=12, columnspacing=1.3)
        fig.text(.077, .179,
                 "名义需求 = 当前已接纳请求每层盘上读取量 / 每层 C；按全部接纳与完成事件更新，非 SSD 实际吞吐。",
                 fontsize=12)
        fig.text(.077, .143,
                 "全程逐盘超 40 GiB/s 时间为 0；不额外叠加跨请求 L0，欠载不保证每波突发都能及时完成。",
                 fontsize=12)
        fig.text(.077, .105,
                 "短 SS：1K/128、1K/256、1K/384（外推）；长 LL：192K/768（插值）；全局 19,456 请求。",
                 fontsize=12)
        fig.text(.077, .067,
                 "arrival 均为 0；真实物理 placement 从源请求保留，未按新 NPU 编号重新条带化。",
                 fontsize=12)
    ax.set_xlim(START, END)
    ax.set_xticks(range(2000, 4001, 250))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}"))
    ax.set_xlabel("仿真时间（秒）", fontsize=16, labelpad=8)
    fig.text(.077, .018 if timeline else .026,
             f"种子 7 | 输入 {a['input_fingerprint'][:16]} | 仿真结果，非硬件实测",
             fontsize=10.5, color="#5c6164")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            outside.append({'text': text.get_text(), 'bounds': list(box.extents)})
    assert not outside, outside
    if not timeline:
        assert not ax.xaxis.label.get_window_extent(renderer).overlaps(
            fig.legends[0].get_window_extent(renderer)), 'Demand legend overlaps x-axis title'
    suffix = 'baseline' if timeline else 'demand'
    stem = OUT / f"l{n_long}_s{n_short}_{order}_{suffix}"
    description = json.dumps({'source_audit': a, 'script_sha256': sha(__file__),
                              'figure_kind': kind}, ensure_ascii=False, sort_keys=True)
    fig.savefig(stem.with_suffix('.png'), dpi=280, metadata={'Description': description})
    fig.savefig(stem.with_suffix('.svg'), dpi=280, metadata={'Description': description})
    fig.savefig(stem.with_suffix('.pdf'), dpi=280,
                metadata={'Title': title, 'Subject': description, 'Creator': 'plot_role_separate.py / matplotlib'})
    plt.close(fig)
    return {'stem': str(stem), 'kind': kind, 'label': a['label'],
            'all_visible_text_inside_canvas': True,
            'demand_x_axis_title_and_legend_disjoint': True if not timeline else None,
            'files': [{'path': str(stem.with_suffix(ext)), 'sha256': sha(stem.with_suffix(ext))}
                      for ext in ('.png', '.pdf', '.svg')]}


def main():
    font = configure_font()
    # Parent loader reads actual stored placement and current manifest NPU IDs.
    # No striped placement is reconstructed from reassigned NPUs; only root path changes.
    plot_case.BASE = HERE
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE/'plan.json').read_text())
    summary = json.loads((HERE/'summary.json').read_text())
    records = {(r['label'], r['strategy']): r for r in summary['records']}
    with (HERE/'per_seed.csv').open() as stream:
        csv_rows = {(r['label'], r['strategy']): r for r in csv.DictReader(stream)}
    items = [i for i in plan['inputs'] if i['seed'] == 7]
    assert len(items) == 4
    frozen = {str(ROOT/name): digest for name, digest in plan['source_sha256'].items()}
    assert all(sha(path) == value for path, value in frozen.items())
    pairs, cases, outputs = {}, [], []
    for item in items:
        label, n_long, order = item['label'], item['long_npu_count'], item['order']
        assert sha(item['manifest']) == item['manifest_sha256']
        case = plot_case.load_case(label, START, END)
        assert case['audit']['input_fingerprint'] == item['input_fingerprint']
        manifest = plot_case.read_json(Path(item['manifest']))
        counts = Counter()
        for request in manifest['requests']:
            n, load = request['npu_id'], request['load']
            role = 'long' if n < n_long else 'short'
            assert load['role'] == role and load['category'] == ('LL' if role == 'long' else 'SS')
            assert request['arrival_time_ms'] == 0
            assert 'original_request_id' in load and 'source_original_npu_id' in load
            counts[(load['seq_len_k'], load['nql'])] += 1
        assert counts == PROFILE_COUNTS
        a = case['audit']
        r = records[(label, 'baseline')]
        assert r['audit_passed'] and r['all_study_conditions_met']
        parent_scan = r['parent_technical_audit']['nominal_demand_scan']['full_run']
        csv_row = csv_rows[(label, 'baseline')]
        assert csv_row['seed'] == '7'
        assert parent_scan['any_ssu_over_capacity_ms'] == 0
        assert max(a['full_run_current_profile_peak_by_ssu_gib_s']) < 40
        assert math.isclose(case['util'], r['metrics']['device_utilization'], abs_tol=1e-12)
        assert math.isclose(max(a['full_run_current_profile_peak_by_ssu_gib_s']),
                            parent_scan['max_ssu_gib_s'], abs_tol=1e-8)
        assert math.isclose(case['util']*100, float(csv_row['device_utilization_percent']), abs_tol=1e-10)
        assert math.isclose(max(a['full_run_current_profile_peak_by_ssu_gib_s']),
                            float(csv_row['full_run_max_ssu_gib_s']), abs_tol=1e-8)
        assert a['result_sha256'] == r['parent_technical_audit']['file_sha256']
        compute = math.fsum(a['window_compute_ms_by_npu'])
        stall = math.fsum(a['window_stall_ms_by_npu'])
        assert math.isclose(compute+stall, 64000, abs_tol=1e-7)
        a.update(manifest_sha256=item['manifest_sha256'], long_npu_ids=list(range(n_long)),
                 short_npu_ids=list(range(n_long, 32)), short_internal_order=order,
                 total_compute_npu_ms=compute, total_io_stall_npu_ms=stall,
                 compute_plus_stall_npu_ms=compute+stall,
                 full_run_any_ssu_over_capacity_ms=0.0,
                 summary_and_seed7_csv_values_match=True, fixed_role_and_policy_category_verified=True,
                 source_original_manifest=manifest['metadata']['source_manifest'],
                 source_original_manifest_sha256=manifest['metadata']['source_manifest_sha256'])
        assert sha(a['source_original_manifest']) == a['source_original_manifest_sha256']
        pairs.setdefault(n_long, {})[order] = case['population']
        for kind in ('timeline', 'demand'):
            outputs.append(draw(case, n_long, order, kind))
        cases.append(a)
        print(json.dumps({'label': label, 'U_percent': 100*case['util'],
                          'compute_npu_ms': compute, 'stall_npu_ms': stall,
                          'full_peak': max(a['full_run_current_profile_peak_by_ssu_gib_s'])}), flush=True)
    assert all(p['random'] == p['round_robin'] for p in pairs.values())
    assert all(sha(path) == value for path, value in frozen.items())
    audit = {'no_simulation': True, 'window_ms': [START, END], 'seed': 7,
             'script_path': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
             'plan_sha256': sha(HERE/'plan.json'), 'summary_sha256': sha(HERE/'summary.json'),
             'per_seed_csv_sha256': sha(HERE/'per_seed.csv'),
             'python': platform.python_version(), 'font_path': str(font), 'font_sha256': sha(font),
             'pdf_fonttype': plt.rcParams['pdf.fonttype'], 'svg_fonttype': plt.rcParams['svg.fonttype'],
             'reused_sources': {str(p): sha(p) for p in (PARENT/'plot_case.py', PARENT/'plot_separate.py')},
             'frozen_construction_and_simulator_sources': frozen,
             'same_binding_random_RR_complete_population_and_placement_match': True,
             'different_binding_ratios_do_not_keep_same_per_card_population': True,
             'loader_role_reassignment_review': 'Stored placement read verbatim; current manifest NPU checked against execution; source original IDs retain per-new-card matched population; no old stripe-from-NPU assumption.',
             'definitions': {'U': 'Actual clipped layer compute /(32*2000ms)',
                 'stall': 'admission or previous layer compute_end to compute_start, verified against io_barrier_wait_ms',
                 'nominal_demand': 'Current admitted requests physical per-disk V/C; simultaneous admission/completion changes grouped; not SSD throughput or deadline envelope; no extra cross-request L0'},
             'cases': cases, 'figures': outputs}
    (OUT/'figures_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'standalone_figures': len(outputs), 'files': len(outputs)*3, 'audit': str(OUT/'figures_audit.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
