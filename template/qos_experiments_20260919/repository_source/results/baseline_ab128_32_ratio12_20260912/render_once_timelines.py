#!/usr/bin/env python3
"""Render the Baseline timeline style for completed SSU3/SSU4 Once runs."""
from pathlib import Path
import argparse
import math

import numpy as np
import render as reference
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEFT, RIGHT = 2000., 4000.
NPU_COUNT = 32


def collect(directory, num_ssu, order):
    case = directory / 'runs' / order / 'once'
    command = reference.read(case / 'command.json')
    man = reference.read(case / 'manifest.json.gz')
    raw = reference.read(case / 'result.json.gz')
    assert command['status'] == 'complete' and command['returncode'] == 0
    assert command['completed_simulation'] is True
    assert command['strategy'] == raw['strategy'] == 'once'
    assert raw['policy_config']['assignment'] == 'fixed'
    assert reference.sha(case / 'manifest.json.gz') == command['manifest_sha256']
    assert reference.sha(case / 'result.json.gz') == command['output_sha256']
    assert man['input_fingerprint'] == raw['input_fingerprint'] == command['input_fingerprint']
    assert raw['core_and_policy_sha256'] == command['core_source_sha256']
    meta, summary = man['metadata'], raw['summary']
    assert meta['num_npu'] == summary['num_npu'] == NPU_COUNT
    assert meta['num_ssu'] == summary['num_ssu'] == num_ssu
    assert meta['seed'] == 7 and meta['order'] == order
    assert meta['n_layers'] == summary['n_layers'] == 8
    assert meta['disk_bw_gib_s'] == 40 and meta['npu_bw_gib_s'] == 50
    assert all(summary['invariants'].values())
    reqs = {request['request_id']: request for request in man['requests']}
    assert len(reqs) == summary['request_count'] == 3840
    assert {q['request_id'] for q in summary['request_metrics']} == set(reqs)
    spans = [{kind: [] for kind in reference.COLOR} for _ in range(NPU_COUNT)]
    sums = {kind: np.zeros(NPU_COUNT) for kind in reference.COLOR}
    active = np.zeros(NPU_COUNT)
    for batch in summary['microbatch_metrics']:
        assert batch['batch_size'] == len(batch['member_request_ids']) == 1
        assert len(batch['layer_metrics']) == 8
        npu = batch['npu_id']
        request = reqs[batch['member_request_ids'][0]]
        assert request['npu_id'] == npu
        kind = reference.role(request['load'])
        active[npu] += reference.overlap(batch['admission_time_ms'], batch['completion_time_ms'], LEFT, RIGHT)
        previous = batch['admission_time_ms']
        for layer in batch['layer_metrics']:
            start, end = layer['compute_start_ms'], layer['compute_end_ms']
            assert previous <= start + 1e-7 and start < end
            assert math.isclose(start - previous, layer['io_barrier_wait_ms'], abs_tol=1e-6)
            assert math.isclose(end - start, request['load']['per_layer_us'] / 1000, abs_tol=1e-7)
            for state, a, z in (('stall', previous, start), (kind, start, end)):
                duration = reference.overlap(a, z, LEFT, RIGHT)
                if duration > 0:
                    spans[npu][state].append((max(LEFT, a), duration))
                    sums[state][npu] += duration
            previous = end
        assert math.isclose(previous, batch['completion_time_ms'], abs_tol=1e-7)
    assert np.allclose(sum(sums.values()), active, atol=1e-6, rtol=0)
    assert np.allclose(active, RIGHT-LEFT, atol=1e-6, rtol=0)
    # Every plotted lane partitions the same window without gaps or overlaps.
    for lane in spans:
        edge = LEFT
        for a, duration in sorted(piece for pieces in lane.values() for piece in pieces):
            assert math.isclose(a, edge, abs_tol=1e-6), (a, edge)
            edge = a + duration
        assert math.isclose(edge, RIGHT, abs_tol=1e-6)
    U = 100 * (sums['A'] + sums['B']) / (RIGHT-LEFT)
    window = next(w for w in raw['windows'] if w['start_ms'] == LEFT and w['end_ms'] == RIGHT)
    assert np.allclose(sums['A'] + sums['B'], window['compute_ms_by_npu'], atol=1e-7, rtol=0)
    assert math.isclose(float(U.mean()), 100 * window['mean_npu_utilization'], abs_tol=1e-8)
    comparison = reference.read(directory / 'comparison.json')
    old = next(r for r in comparison['results'] if r['strategy'] == 'once' and r['order'] == order
               and r['start_ms'] == LEFT and r['end_ms'] == RIGHT)
    assert math.isclose(float(U.mean()), old['U_percent'], abs_tol=1e-8)
    sources = {str(path.relative_to(ROOT)): reference.sha(path)
               for path in [case / name for name in ('command.json', 'manifest.json.gz', 'result.json.gz')]
               + [directory / 'comparison.json']}
    return dict(spans=spans, sums=sums, U=float(U.mean()), per_npu_U=U,
                sources=sources, input_fingerprint=raw['input_fingerprint'])


def draw(directory, num_ssu, order):
    data = collect(directory, num_ssu, order)
    fig, ax = reference.plt.subplots(figsize=(16, 10.8))
    fig.subplots_adjust(left=.07, right=.96, bottom=.13, top=.81)
    fig.suptitle(f'Once per layer {order.title()}：32 张 NPU 的计算与 I/O 等待', y=.974, fontsize=21)
    fig.text(.5, .929, f'32 NPU / {num_ssu} SSU × 40 GiB/s · '
             f'[2, 4) 秒 · NPU 平均利用率 {data["U"]:.2f}%', ha='center', fontsize=15)
    fig.text(.5, .892, reference.DESCRIPTION, ha='center', fontsize=11)
    fig.legend(handles=[Patch(color=reference.COLOR[kind], label=label) for kind, label in
                        [('A', 'A 请求计算'), ('B', 'B 请求计算'), ('stall', 'I/O 等待')]],
               loc='upper center', bbox_to_anchor=(.5, .874), ncol=3, frameon=False)
    for npu in range(NPU_COUNT):
        for kind in reference.COLOR:
            ax.broken_barh(data['spans'][npu][kind], (npu-.4, .8),
                           facecolors=reference.COLOR[kind], linewidth=0)
        ax.text(RIGHT+(RIGHT-LEFT)*.007, npu, f'{data["per_npu_U"][npu]:.1f}%', va='center', fontsize=8)
    ax.set(xlim=(LEFT, RIGHT), ylim=(31.8, -.8), yticks=range(NPU_COUNT),
           xticks=np.arange(LEFT, RIGHT+1, 250.), xlabel='仿真时间（秒）', ylabel='NPU 编号')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}'))
    ax.grid(axis='x', alpha=.15)
    both = int(np.count_nonzero((data['sums']['A'] > 0) & (data['sums']['B'] > 0)))
    fig.text(.07, .075, f'全窗有任务：32/32 卡；本窗实际计算过 A 和 B：{both}/32 卡。每请求 8 层，全部计算计入利用率。', fontsize=10)
    fig.text(.07, .042, '橙色为接纳后等待读取；不含到达后等待接纳。seed 7，采用完整仿真的真实时间区间。', fontsize=10)
    fig.set_dpi(155)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for artist in fig.findobj(reference.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -2 and box.y0 >= -2 and box.x1 <= width+2 and box.y1 <= height+2, (artist.get_text(), box.bounds)
    stem = directory / 'figures' / f'once_{order}'
    reference.save(fig, stem)
    return dict(strategy='once', order=order, num_npu=NPU_COUNT, num_ssu=num_ssu, seed=7,
                U_percent=data['U'], per_npu_U_percent=data['per_npu_U'].tolist(),
                compute_A_ms=data['sums']['A'].tolist(), compute_B_ms=data['sums']['B'].tolist(),
                io_wait_ms=data['sums']['stall'].tolist(), both_profiles_cards=both,
                window_ms=[LEFT, RIGHT], sources=data['sources'], input_fingerprint=data['input_fingerprint'],
                window_partition_verified=True, utilization_matches_existing_results=True,
                visible_labels_inside_canvas=True, pixels=[width, height],
                artifacts={str(stem.with_suffix('.'+ext).relative_to(ROOT)): reference.sha(stem.with_suffix('.'+ext))
                           for ext in ('png', 'pdf')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-ssu', nargs='+', type=int, choices=(3, 4), default=[3, 4])
    args = parser.parse_args()
    for num_ssu in args.num_ssu:
        directory = HERE / f'once_per_layer_ssu{num_ssu}_seed7'
        results = [draw(directory, num_ssu, order) for order in ('ordered', 'random')]
        reference.write(directory / 'figures/once_timeline_checks.json',
                        dict(all_checks_passed=True, no_new_simulation=True, results=results,
                             builders={str(path.relative_to(ROOT)): reference.sha(path)
                                       for path in (Path(__file__), Path(reference.__file__))}))
        index = directory / 'figures/once_timelines.md'
        lines = ['# Once per layer：32 卡计算与 I/O 等待', '',
                 f'32 NPU、{num_ssu} SSU、seed 7、warm [2,4) 秒。使用已有完整仿真日志，沿用 Baseline 时序图样式。', '',
                 '| 顺序 | NPU 平均利用率 | PNG | PDF |', '|---|---:|---|---|']
        for result in results:
            order = result['order']
            lines.append(f'| {order.title()} | {result["U_percent"]:.4f}% | [PNG](once_{order}.png) | [PDF](once_{order}.pdf) |')
        lines += ['', '蓝色为 A 请求计算，绿色为 B 请求计算，橙色为接纳后 I/O 等待；右侧为每张卡在同一窗口内的计算利用率。', '',
                  '[数值与来源校验](once_timeline_checks.json) · [绘图脚本](../../render_once_timelines.py)', '']
        index.write_text('\n'.join(lines))
        for result in results:
            print(f'SSU{num_ssu} {result["order"]}: U={result["U_percent"]:.4f}%; '
                  f'{directory / "figures" / ("once_"+result["order"]+".png")}', flush=True)


if __name__ == '__main__':
    main()
