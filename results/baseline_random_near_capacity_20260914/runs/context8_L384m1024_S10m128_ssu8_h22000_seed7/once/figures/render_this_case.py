#!/usr/bin/env python3
"""Once case figures, preserving the original input and immutable trace.

Only these new figures are written. Standard fleet calculations reuse the
frozen renderer; the local multi-path view has a separate audited adapter.
"""
from pathlib import Path
import argparse
import gc
import json
import math
import sys

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
STUDY = HERE.parents[3]
sys.path.insert(0, str(STUDY))
import analyze as audit
import render
sys.path.insert(0, str(HERE))
import zoom_once as zoom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet-only', action='store_true')
    parser.add_argument('--zoom-only', action='store_true')
    args = parser.parse_args()
    assert not (args.fleet_only and args.zoom_only)
    command = audit.read(CASE / 'command.json')
    assert command['status'] == 'complete' and command['completed_simulation'] and command['strategy'] == 'once'
    man = audit.read(CASE / 'manifest.json.gz')
    assert audit.sha(CASE / 'manifest.json.gz') == command['manifest_sha256']
    assert audit.sha(CASE.parent / 'baseline/manifest.json.gz') == command['manifest_sha256']
    meta = man['metadata']
    assert meta['candidate'] == 'context8_L384m1024_S10m128'
    assert meta['num_ssu'] == 8 and meta['num_npu'] == 32 and meta['seed'] == 7
    assert meta['count_ratio'] == [1, 24]
    assert all(p['construction']['method'].startswith('affine_extrapolation') for p in meta['profiles'])
    profiles = {p['role']: p for p in meta['profiles']}
    pure_us = sum(n * profiles[r]['per_layer_compute_us'] for n, r in zip(meta['count_ratio'], meta['role_names']))
    volume = sum(n * profiles[r]['per_layer_kv_gib'] for n, r in zip(meta['count_ratio'], meta['role_names']))
    rho = 32 * volume * 1e6 / pure_us / (8 * 40)
    assert math.isclose(rho, meta['ideal_load_ratio'], abs_tol=1e-12)
    notice = f'理想平均负载{100*rho:.2f}%（非逐盘逐时欠载保证）'
    protected = {str(Path(p)): audit.sha(Path(p)) for p in
                 (render.__file__, audit.__file__, STUDY / 'zoom.py', STUDY / 'experiment.py')}
    builders = dict(protected)
    builders.update({str(Path(p)): audit.sha(Path(p)) for p in (Path(__file__), zoom.__file__)})
    original_save = render.save
    display = []

    def annotated_save(fig, path):
        configurations = [t for t in fig.texts if ' NPU / ' in t.get_text()]
        assert len(configurations) == 1
        text = configurations[0]
        text.set_text(text.get_text() + ' · ' + notice)
        for _ in range(48):
            fig.canvas.draw()
            if text.get_window_extent(fig.canvas.get_renderer()).x1 <= fig.canvas.get_width_height()[0] * .97:
                break
            text.set_fontsize(text.get_fontsize() - .25)
        assert text.get_fontsize() >= 10
        image = original_save(fig, path)
        display.append(dict(image=image, subtitle=text.get_text(), subtitle_fontsize=text.get_fontsize(),
                            all_C_extrapolated_explicit=True, mean_load_not_instant_underload_guarantee=True))
        return image

    render.save = annotated_save
    if not args.zoom_only:
        data = render.prepare(CASE, 2000., 4000.)
        assert data['analysis']['short_roles'] == ['S'] and data['strategy'] == 'once'
        images = [render.fleet_bandwidth(data, HERE), render.timeline(data, HERE), render.history(data, HERE)]
        record = dict(all_checks_passed=True, no_new_simulation=True, case=str(CASE), strategy='once',
                      strategy_label='流量分配策略', window_ms=[2000., 4000.],
                      metadata_label=meta['label'], U_percent=data['window']['U_percent'],
                      ideal_load_ratio=rho, evidence=data['evidence'], builders=builders, images=images)
        (HERE / 'checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(fleet_complete=True, warm_U=data['window']['U_percent'],
                              verified_cycles=data['evidence']['verified_complete_internal_cycles']), ensure_ascii=False), flush=True)
        del data
        gc.collect()
    if not args.fleet_only:
        local = zoom.prepare(CASE, (2000., 4000.))
        out = HERE / 'internal_paired_comparison'
        out.mkdir(exist_ok=True)
        image = zoom.draw(local, out / 'allocation_random_short_internal_cycle.png')
        e = local['evidence']
        e['builders'].update(builders)
        baseline_evidence_path = CASE.parent / 'baseline/figures/internal_median_wait/evidence.json'
        baseline_evidence = audit.read(baseline_evidence_path)
        if e['selection']['paired_to_baseline']:
            for key in ('npu', 'request_id'):
                assert e['selected'][key] == baseline_evidence['selected'][key]
            for key in ('current', 'following'):
                assert e['selected'][key]['layer'] == baseline_evidence['selected'][key]['layer']
            audit.close(e['C_ms'], baseline_evidence['C_ms'])
            audit.close(e['V_GiB'], baseline_evidence['V_GiB'])
        e.update(image=image, ideal_load_ratio=rho, load_label=notice,
                 baseline_reference_evidence=str(baseline_evidence_path),
                 baseline_reference_evidence_sha256=audit.sha(baseline_evidence_path))
        (out / 'evidence.json').write_text(json.dumps(e, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (out / 'selected_cycle_blocks.json').write_text(json.dumps(dict(columns=render.COLS,
            target_rows=local['target'].tolist(), physical_ssd_rows=local['shown'].tolist()), ensure_ascii=False, indent=2) + '\n')
        paired = e['selection']['paired_to_baseline']
        text = ['# 流量分配策略：同一请求内部的计算与读取', '',
                '[查看PNG](allocation_random_short_internal_cycle.png)', '',
                '本图复用已完成运行的逐块日志，没有新增仿真。长、短请求C均由data拟合外推；8SSU、理想平均负载99.69%，不保证逐盘逐时欠载。', '',
                ('与Baseline局部图选取相同NPU、相同请求、相同层。绝对时间及竞争卡状态可能不同。' if paired else
                 'Baseline所选请求层不在当前trace范围，因此此图改用本策略warm内正等待中位附近的内部短层；不是配对样本。'), '',
                f'NPU{e["selected"]["npu"]}、请求{e["selected"]["request_id"]}、人类第{e["selected"]["current"]["layer"]+1}层计算，预取第{e["selected"]["following"]["layer"]+1}层。',
                f'绝对时间[{e["left_ms"]/1000:.9f},{e["right_ms"]/1000:.9f})秒；是否完全位于warm：[2,4)秒：{e["selection"]["selected_cycle_in_warm"]}。',
                f'C={e["C_ms"]:.9f}毫秒，w={e["wait_ms"]:.9f}毫秒，D={e["D_ms"]:.9f}毫秒。',
                f'B={e["B_GiB_s"]:.9f}GiB/s；平均b={e["average_b_GiB_s"]:.9f}GiB/s；b/B=C/D={e["local_cycle_U_percent"]:.6f}%。', '',
                'SSU行汇总多个QoS路径的实际磁盘服务。不能把这些服务先后直接理解为同一个FIFO队列，更不能仅凭先服务推断已经在目标请求之前入队。',
                '完整79块目标数据、所有显示竞争块placement及物理时间均核对；NPU行和累计读取只有同一请求的同一目标层，未混入本卡其他请求。',
                '局部b/B是完整周期记账关系，不能作为两种策略整机效果的独立预测；整机差异看相同统计窗口的32卡U。', '',
                '[全部选择、路径、源SHA及物理校验](evidence.json) · [实际显示的块](selected_cycle_blocks.json)']
        (out / 'README.md').write_text('\n'.join(text) + '\n')
        print(json.dumps(dict(zoom_complete=True, paired=paired, npu=e['selected']['npu'],
                              C_ms=e['C_ms'], wait_ms=e['wait_ms'], D_ms=e['D_ms'], cycle_U=e['local_cycle_U_percent']), ensure_ascii=False), flush=True)
    assert all(audit.sha(Path(p)) == digest for p, digest in protected.items())
    suffix = '_zoom' if args.zoom_only else '_fleet' if args.fleet_only else ''
    (HERE / ('display_audit' + suffix + '.json')).write_text(json.dumps(dict(all_checks_passed=True,
        frozen_helpers_unchanged=True, builders=builders, images=display), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
