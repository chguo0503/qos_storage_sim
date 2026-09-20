#!/usr/bin/env python3
"""Read completed native pilots; no simulation, input mutation or core edits."""
import cmath
import gzip
import hashlib
import json
import math
from collections import Counter
from experiment import HERE, BASE, load_manifest


def analyze(label):
    path = HERE / 'inputs' / f'{label}.json.gz'
    requests, meta = load_manifest(path)
    original, _ = load_manifest(BASE / 'manifest.json.gz')
    before = {q.request_id: q for q in original}
    byid = {q.request_id: q for q in requests}
    record = json.loads((HERE / 'runs' / label / 'command.json').read_text())
    with gzip.open(HERE / 'runs' / label / 'live_summary.json.gz', 'rt') as f:
        summary = json.load(f)
    assert record['status'] == 'complete_pilot' and record['source_unchanged']
    assert record['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(requests) == len(original)
    for n in range(32):
        assert [q.request_id for q in requests if q.npu_id == n] == [q.request_id for q in original if q.npu_id == n]
    for q in requests:
        old = before[q.request_id]
        assert (q.npu_id, q.arrival_time_ms, q.placement) == (old.npu_id, old.arrival_time_ms, old.placement)
        assert q.load['per_layer_kv_gb'] == old.load['per_layer_kv_gb']
        assert q.load['role'] == old.load['role']
        if q.load['input_kind'] != 'P':
            assert q.load['per_layer_us'] == old.load['per_layer_us']
        else:
            assert q.load['per_layer_us'] >= old.load['per_layer_us']
    cb = meta['calibration']['C_B_ms']
    anchor = meta['calibration']['target_anchor_ms']
    starts = [l['compute_start_ms'] for b in summary['microbatch_metrics']
              if byid[b['member_request_ids'][0]].load['role'] == 'B'
              for l in b['layer_metrics']]
    phase = []
    for lo, hi in [(0, 800), (800, 1600), (1600, 2400), (2400, 4000), (4000, 8000)]:
        values = [cmath.exp(2j * math.pi * t / cb) for t in starts if lo <= t < hi]
        phase.append(dict(start_ms=lo, end_ms=hi, R=abs(sum(values)/len(values)), count=len(values)))
    first = []
    for n in [0, 8, 16, 24]:
        rows = [b for b in summary['microbatch_metrics'] if b['npu_id'] == n]
        i = next(i for i, b in enumerate(rows) if byid[b['member_request_ids'][0]].load['role'] == 'A')
        j = i
        while byid[rows[j]['member_request_ids'][0]].load['role'] == 'A':
            j += 1
        target = anchor + (i + 1) * 8 * cb
        actual = rows[j]['layer_metrics'][0]['compute_start_ms']
        first.append(dict(npu=n, target_ms=target, actual_ms=actual, target_minus_actual_ms=target-actual))
    windows = []
    for w in record['windows']:
        windows.append(dict(start_ms=w['start_ms'], end_ms=w['end_ms'], U_percent=w['U_percent'],
            strict_under=w['demand']['strict_underload_all_disks'],
            max_demand_GiB_s=w['demand']['per_disk_max_GiB_s'],
            ordinary_A_B_mixed_cards=w['ordinary_A_and_B_cards'], all_active=w['all_active'],
            **{f'min_{k}_compute_ms': min(c[k] for c in w['per_npu_compute_ms']) for k in ['A', 'P', 'B']}))
    ps = [q.load['per_layer_us']/1000 for q in requests if q.load['input_kind'] == 'P']
    return dict(label=label, manifest=str(path), manifest_sha256=record['manifest_sha256'],
        counts=dict(Counter(q.load['input_kind'] for q in requests)), P_C_range_ms=[min(ps), max(ps)],
        unchanged_address_arrival_order_payload=True, only_P_compute_changed=True,
        phase=phase, first_B_after_A_initial=first, windows=windows,
        wall_s=record['wall_s'], source_unchanged=record['source_unchanged'])


def main():
    labels = ['tail_gain_0.5', 'tail_gain_1', 'tail_gain_0.5_group4_refine']
    results = [analyze(label) for label in labels]
    output = dict(pilot_only=True, SLO=None, completed_simulation=False,
        results=results,
        phase_R_definition='abs(mean(exp(2*pi*j*B_layer_compute_start/C_B))); 1 means common phase, 0 means dispersed',
        conclusion='All three frozen calibrations lower warm U temporarily but do not maintain low utilization through 4–8 seconds. No further trials in this subtask.')
    (HERE / 'analysis.json').write_text(json.dumps(output, indent=2))
    lines = ['# 静态尾请求补偿结果', '',
        '三次原模拟器短试验均满足 0–8 秒逐盘严格欠载，但未能持续保持较低利用率。输入和计算时间全部事先冻结；运行时没有控制或暂停 NPU。', '',
        '| 输入 | U：[2,4) 秒 | U：[4,8) 秒 | 逐盘峰值最大值 | 每窗普通 A/B 都执行的卡数 |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        a,b,full=r['windows']
        lines.append(f"| {r['label']} | {a['U_percent']:.4f}% | {b['U_percent']:.4f}% | {max(full['max_demand_GiB_s']):.4f} GiB/s | 32 / 32 |")
    lines += ['', '补偿在 A 段末尾增加一个合成请求 P 的计算时间，意图让下一 B 段回到共同的层周期相位。P 的读取数据与地址保持原 A 不变；它是第三种合成画像，不是纯请求重排，也不是 data 原始画像。', '',
        'gain=0.5 的首轮补偿延缓了相位分散。第三次再利用该冻结回放中 NPU 24–31 的首轮提前量，离线增加其 P 的计算时间，随后从头重放。虽然 warm 利用率降至 93.5859%，4–8 秒已恢复至 97.3481%。增加计算不能按简单比例精确平移 B 的开始时间，因为它也改变共享 I/O 的竞争和等待。', '',
        '| 输入 | B 层相位集中度 R：[0,0.8) 秒 | R：[2.4,4) 秒 | R：[4,8) 秒 |',
        '|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['label']} | {r['phase'][0]['R']:.6f} | {r['phase'][3]['R']:.6f} | {r['phase'][4]['R']:.6f} |")
    lines += ['', '这里 R=1 表示 B 层开始时间集中在同一周期相位，越接近 0 越分散。它是描述同步程度的统计量，不是利用率公式。相位分散与后续等待减少、利用率恢复相符。', '',
        '全部输入保留原始请求数量、顺序、到达时刻、数据读取量和 Ring Hash 地址，仅标记 P 的计算时间改变。核心源码运行前后哈希一致。分析脚本另外核对每请求数据量、地址、到达时间和普通 A/B 的计算时间未变。', '',
        '**限制：** 每次在 8 秒截断，没有排空全部请求，因此不报告 TTFT SLO 或整批全程利用率；[0,8) 的严格欠载也不能外推到尚未运行的后续时间。当前结果只能证明短期降低，不能称为长期低利用率构造。补偿来自一次已知时间线，没有跨种子泛化证据。', '',
        '原生 176 KiB 命令调度和空路径重新入队的虚拟完成时间重置，使短请求的实际服务略快于连续带宽代理。微小的层级提前在多次 A 请求中累积，后续 B 逐渐错开；静态补偿只能部分抵消这一演化。具体测量见相邻 `ideal_search/proxy_native_gap.md`。', '',
        '较大 A 读取量的 10 个原模拟器 4 秒候选见 `../larger_blocks/README.md`：8 个满足 0–4 秒严格欠载，最好 warm U=96.1150%，没有优于已有较小 A 候选。两条探索路径均已停止，不以短窗成功冒充长期成功。', '',
        '可复核文件：`analysis.json`、`inputs/*.json.gz`、`runs/*/command.json`、`runs/*/live_summary.json.gz`。执行 `python finalize_analysis.py` 只读取已有数据并重新生成本汇总，不运行仿真。', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps([dict(label=r['label'], U=[w['U_percent'] for w in r['windows']], R=[p['R'] for p in r['phase']]) for r in results], indent=2))


if __name__ == '__main__':
    main()
