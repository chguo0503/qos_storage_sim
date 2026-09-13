#!/usr/bin/env python3
"""Two 100% stacked bars from independently checked, identically-input runs."""
from pathlib import Path
import json
import math
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY))
import analyze as audit
import render as style
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CASE = STUDY / 'runs/context8_L384m1024_S10m128_ssu8_h22000_seed7'
COMPUTE, SHORT, FIRST = '#168879', '#e7a028', '#8b658e'


def main():
    sources, commands, results = {}, {}, {}

    def read(path):
        sources[str(path)] = audit.sha(path)
        return audit.read(path)

    for strategy in ('baseline', 'once'):
        case = CASE / strategy
        command = read(case / 'command.json')
        loss = read(case / 'figures/loss_decomposition.json')
        assert command['status'] == 'complete' and command['completed_simulation']
        assert command['strategy'] == strategy
        assert loss['all_checks_passed'] and loss['no_analyzer_import']
        for path, digest in loss['source_sha256'].items():
            assert audit.sha(Path(path)) == digest
            sources[path] = digest
        assert audit.sha(case / 'figures/audit_wait_loss.py') == loss['builder_sha256']
        sources[str(case / 'figures/audit_wait_loss.py')] = loss['builder_sha256']
        command_key = {'manifest.json.gz': 'manifest_sha256', 'result.json.gz': 'output_sha256'}
        for filename, key in command_key.items():
            assert audit.sha(case / filename) == command[key]
        result = next(r for r in loss['results'] if (r['start_ms'], r['end_ms']) == (2000., 20000.))
        analysis = read(case / 'analysis.json')
        window = next(w for w in analysis['windows'] if (w['start_ms'], w['end_ms']) == (2000., 20000.))
        assert window['all_32_active'] and window['long_short_mixed_pass']
        assert window['long_short_mixed_card_count'] == 32
        audit.close(result['U_percent'], window['U_percent'])
        for zero in ('long_internal_loss_pp', 'idle_loss_pp'):
            assert abs(result[zero]) < 1e-9
        audit.close(result['U_percent'] + result['short_internal_loss_pp'] + result['first_layer_loss_pp'], 100.)
        commands[strategy], results[strategy] = command, result
    for key in ('manifest_sha256', 'input_fingerprint', 'core_source_sha256', 'runner_sha256'):
        assert commands['baseline'][key] == commands['once'][key]
    manifest = read(CASE / 'once/manifest.json.gz')
    meta = manifest['metadata']
    assert (meta['num_npu'], meta['num_ssu'], meta['seed'], meta['n_layers']) == (32, 8, 7, 8)
    assert all(p['construction']['method'].startswith('affine_extrapolation') for p in meta['profiles'])
    rho = meta['ideal_load_ratio']
    assert round(rho * 100, 2) == 99.69
    baseline, once = results['baseline'], results['once']
    gain = once['U_percent'] - baseline['U_percent']
    short_gain = baseline['short_internal_loss_pp'] - once['short_internal_loss_pp']
    first_gain = baseline['first_layer_loss_pp'] - once['first_layer_loss_pp']
    audit.close(gain, short_gain + first_gain)
    protected = {str(Path(m.__file__)): audit.sha(Path(m.__file__)) for m in (audit, style)}

    fig = plt.figure(figsize=(16, 10), dpi=150, facecolor='white')
    fig.text(.065, .945, '相同随机输入：流量分配减少短请求内部等待', fontsize=27, color=style.INK)
    fig.text(.065, .895, '32 NPU · 8 SSU × 40 GiB/s · seed 7 · 统计 [2,20) 秒 · 长、短请求 C 均由 data 拟合外推',
             fontsize=15, color=style.MUTED)
    fig.legend(handles=[Patch(facecolor=COMPUTE, label='计算'),
                        Patch(facecolor=SHORT, label='短请求内部等待'),
                        Patch(facecolor=FIRST, label='请求首层交接等待')],
               loc='upper left', bbox_to_anchor=(.065, .86), frameon=False, ncol=3, fontsize=14)
    ax = fig.add_axes([.09, .20, .385, .58])
    for x, r in enumerate((baseline, once)):
        bottom = 0.
        for key, color in (('U_percent', COMPUTE), ('short_internal_loss_pp', SHORT), ('first_layer_loss_pp', FIRST)):
            value = r[key]
            ax.bar(x, value, width=.56, bottom=bottom, color=color, edgecolor='white', linewidth=.8, zorder=3)
            bottom += value
        ax.text(x, r['U_percent']/2, f'{r["U_percent"]:.2f}%', ha='center', va='center',
                fontsize=25, color='white', fontweight='bold', zorder=5)
    ax.set(xlim=(-.55, 1.55), ylim=(0, 100), yticks=range(0, 101, 20),
           xticks=[0, 1], xticklabels=['Baseline', '流量分配策略'])
    ax.set_ylabel('占整窗卡时间（%）', fontsize=14, labelpad=14)
    ax.tick_params(axis='both', labelsize=14, length=0, pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#bac3cc')
    ax.grid(axis='y', color='#e4e8ec', zorder=0)

    fig.text(.55, .765, '平均 NPU 利用率', fontsize=16, color=style.MUTED)
    fig.text(.55, .705, f'{baseline["U_percent"]:.2f}%  →  {once["U_percent"]:.2f}%',
             fontsize=31, color=COMPUTE, fontweight='bold')
    fig.text(.55, .645, f'提高 {gain:.2f} 个百分点', fontsize=24, color=style.INK)
    fig.text(.55, .575, f'其中 {short_gain:.2f} 个百分点来自', fontsize=19, color=style.INK)
    fig.text(.55, .532, '短请求内部等待减少', fontsize=22, color='#a86c0d', fontweight='bold')
    fig.text(.55, .445, '短请求内部等待损失', fontsize=15, color=style.MUTED)
    fig.text(.55, .402, f'{baseline["short_internal_loss_pp"]:.3f}  →  {once["short_internal_loss_pp"]:.3f} pp',
             fontsize=23, color='#a86c0d')
    fig.text(.55, .337, '请求首层交接等待损失', fontsize=15, color=style.MUTED)
    fig.text(.55, .294, f'{baseline["first_layer_loss_pp"]:.3f}  →  {once["first_layer_loss_pp"]:.3f} pp',
             fontsize=21, color=FIRST)
    fig.text(.55, .218, '两策略：长请求内部等待 = 0；空闲 = 0。', fontsize=14, color=style.MUTED)
    fig.text(.065, .115, '首层等待包含预取起点与 50 GiB/s 接收链路带来的固有等待，不能全部归因于 FIFO。',
             fontsize=13, color=style.MUTED)
    fig.text(.065, .075, '理想平均负载 99.69% 不保证逐盘逐时欠载；本图不宣称严格欠载下的纯 FIFO 因果。',
             fontsize=13, color=style.MUTED)
    fig.text(.065, .035, '来源：同一 manifest 与核心源码；两份独立的逐层日志裁剪统计。pp 表示相对整窗卡时间的百分点。',
             fontsize=11, color=style.MUTED)
    image = style.save(fig, HERE / 'context384_baseline_vs_allocation_loss.png')
    assert all(audit.sha(Path(p)) == digest for p, digest in protected.items())
    record = dict(all_checks_passed=True, no_simulation=True, same_manifest_and_core=True,
        each_bar_sums_to_100=True, all_32_active_and_mixed_both=True,
        window_ms=[2000., 20000.], num_npu=32, num_ssu=8, seed=7,
        C_both_extrapolated=True, ideal_load_ratio=rho, instantaneous_underload_not_claimed=True,
        results=results, U_gain_pp=gain, short_internal_gain_pp=short_gain, first_layer_gain_pp=first_gain,
        short_internal_share_of_gain=short_gain/gain,
        manifest_sha256=commands['once']['manifest_sha256'], source_sha256=sources,
        frozen_helpers_unchanged=protected, builder_sha256=audit.sha(Path(__file__)), image=image)
    (HERE / 'checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    notes = ['# 同一随机输入的策略损失对照', '',
             '[查看独立PNG](context384_baseline_vs_allocation_loss.png)', '',
             '32NPU、8SSU、seed7，[2,20)秒。两根柱都按同一32卡×18秒分母归一化为100%。长、短计算时间均为data拟合外推，理想平均负载99.69%并不保证逐盘逐时欠载。', '',
             f'平均U：{baseline["U_percent"]:.9f}% → {once["U_percent"]:.9f}%，提高{gain:.9f}个百分点。',
             f'短请求内部等待损失减少{short_gain:.9f}个百分点；首层交接等待损失减少{first_gain:.9f}个百分点，二者相加与U提高量相等。',
             '长请求内部等待、空闲均为0。首层损失包含链路固有下界，不能全部解释为FIFO。', '',
             '[数值、输入一致性和来源SHA检查](checks.json) · [生成器](render_comparison.py)', '',
             '[Baseline独立损失分解](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/baseline/figures/loss_decomposition.md)',
             '[流量分配策略独立损失分解](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/figures/loss_decomposition.md)']
    (HERE / 'README.md').write_text('\n'.join(notes) + '\n')
    print(json.dumps(dict(image=image, U_gain_pp=gain, short_internal_gain_pp=short_gain,
                          first_layer_gain_pp=first_gain), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
