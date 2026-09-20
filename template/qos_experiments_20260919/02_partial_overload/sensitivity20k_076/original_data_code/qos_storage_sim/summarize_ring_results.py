#!/usr/bin/env python3
"""Independently recompute the two completed ring-hash Baseline experiments."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/ring_hash_baseline_random_20260914'
WINDOWS = ((2000.0, 4000.0), (2000.0, 20000.0))
ALPHA = 1.5
HISTORICAL_URL = (
    'https://github.com/chguo0503/qos_storage_sim/blob/main/results/'
    'baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/ttft_slo.csv'
)
HISTORICAL_STRIPE = [
    dict(num_ssu=3, start_ms=2000.0, end_ms=4000.0,
         U_percent=90.67794278896096, passed=264, count=341,
         slo_percent=77.41935483870968),
    dict(num_ssu=3, start_ms=2000.0, end_ms=20000.0,
         U_percent=91.2598858160183, passed=2456, count=3114,
         slo_percent=78.86962106615286),
]


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual, expected, *, tolerance=1e-8):
    assert math.isfinite(actual) and math.isfinite(expected), (actual, expected)
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=tolerance), (actual, expected)


def clipped(start, end, left, right):
    assert math.isfinite(start) and math.isfinite(end) and end >= start
    return max(0.0, min(end, right) - max(start, left))


def slo_stats(sample, requests, *, clock='admission'):
    passed = sum(
        row['completion_time_ms'] - row[clock + '_time_ms']
        <= ALPHA * 8 * requests[row['request_id']]['load']['per_layer_us'] / 1000 + 1e-9
        for row in sample
    )
    return dict(passed=passed, count=len(sample),
                percent=100 * passed / len(sample) if sample else None)


def check_manifest(manifest, reference, num_ssu):
    metadata = manifest['metadata']
    assert metadata['num_npu'] == 32 and metadata['num_ssu'] == num_ssu
    assert metadata['n_layers'] == 8 and metadata['seed'] == 7
    assert len(manifest['requests']) == len(reference['requests']) == 3840
    requests = {row['request_id']: row for row in manifest['requests']}
    assert len(requests) == 3840
    counts = defaultdict(Counter)
    thresholds = {}
    # List order, identities, load values, arrival clocks and NPU binding are fixed.
    for row, original in zip(manifest['requests'], reference['requests']):
        assert {k: v for k, v in row.items() if k != 'placement_index'} == {
            k: v for k, v in original.items() if k != 'placement_index'
        }
        assert row['arrival_time_ms'] == 0.0
        assert row['load']['original_request_id'] == original['load']['original_request_id']
        role = row['load']['role']
        counts[row['npu_id']][role] += 1
        threshold = ALPHA * 8 * row['load']['per_layer_us'] / 1000
        near(thresholds.setdefault(role, threshold), threshold)
        placement = manifest['placements'][row['placement_index']]
        old_placement = reference['placements'][original['placement_index']]
        assert len(placement) == len(old_placement) == 1, 'One block mapping reused by all 8 layers'
        assert len(placement[0]) == len(old_placement[0])
        for (disk, size), (_, old_size) in zip(placement[0], old_placement[0]):
            assert isinstance(disk, int) and 0 <= disk < num_ssu
            assert size == old_size == 176 * 1024 / 2**30
    assert set(counts) == set(range(32))
    assert all(value == Counter(A=40, B=80) for value in counts.values())
    return requests, thresholds


def summarize_case(num_ssu, reference):
    directory = OUT / f'ssu{num_ssu}'
    result_path, manifest_path = directory / 'result.json.gz', directory / 'manifest.json.gz'
    result, manifest = read(result_path), read(manifest_path)
    requests, thresholds = check_manifest(manifest, reference, num_ssu)
    summary = result['summary']
    assert result['strategy'] == 'baseline'
    assert result['input_fingerprint'] == manifest['input_fingerprint']
    assert summary['num_npu'] == 32
    assert summary['invariants'] and all(summary['invariants'].values())
    assert summary['invariants']['all_requests_completed']
    completed = summary['request_metrics']
    assert len(completed) == summary['request_count'] == len(requests) == 3840
    assert {row['request_id'] for row in completed} == set(requests)
    for row in completed:
        source = requests[row['request_id']]
        ideal_ms = 8 * source['load']['per_layer_us'] / 1000
        near(row['own_compute_ms'], ideal_ms)
        assert row['arrival_time_ms'] == 0.0
        assert math.isfinite(row['completion_time_ms'])
        assert 0 <= row['admission_time_ms'] <= row['completion_time_ms']
    batches = summary['microbatch_metrics']
    assert len(batches) == 3840
    assert Counter(rid for batch in batches for rid in batch['member_request_ids']) == Counter(requests.keys())
    assert all(batch['batch_size'] == len(batch['member_request_ids']) == 1 for batch in batches)
    assert all(len(batch['layer_metrics']) == 8 for batch in batches)
    all_layers_ms = math.fsum(
        layer['compute_end_ms'] - layer['compute_start_ms']
        for batch in batches for layer in batch['layer_metrics']
    )
    near(all_layers_ms, math.fsum(row['own_compute_ms'] for row in completed), tolerance=1e-5)
    assert 'placement_audit' in result
    rows = []
    for left, right in WINDOWS:
        computation = [[] for _ in range(32)]
        activity = [[] for _ in range(32)]
        for batch in batches:
            npu = batch['npu_id']
            activity[npu].append(clipped(batch['admission_time_ms'], batch['completion_time_ms'], left, right))
            for layer in batch['layer_metrics']:
                computation[npu].append(clipped(layer['compute_start_ms'], layer['compute_end_ms'], left, right))
        compute_by_npu = [math.fsum(values) for values in computation]
        active_by_npu = [math.fsum(values) for values in activity]
        utilization = math.fsum(compute_by_npu) / (32 * (right - left))
        assert all(-1e-8 <= c <= a + 1e-8 <= right - left + 2e-8
                   for c, a in zip(compute_by_npu, active_by_npu))
        matching = [w for w in result['windows'] if w['start_ms'] == left and w['end_ms'] == right]
        assert len(matching) == 1
        stored = matching[0]
        near(stored['mean_npu_utilization'], utilization)
        for computed, saved in zip(compute_by_npu, stored['compute_ms_by_npu']):
            near(computed, saved)
        for computed, saved in zip(active_by_npu, stored['active_ms_by_npu']):
            near(computed, saved)
        all_active = all(math.isclose(value, right - left, abs_tol=1e-7) for value in active_by_npu)
        assert all_active == stored['all_npus_active_whole_window']
        sample = [r for r in completed if left <= r['admission_time_ms'] < right]
        admission, arrival = slo_stats(sample, requests), slo_stats(sample, requests, clock='arrival')
        by_role = {
            role: slo_stats([r for r in sample if requests[r['request_id']]['load']['role'] == role], requests)
            for role in ('A', 'B')
        }
        after = sum(r['completion_time_ms'] > right for r in sample)
        assert sum(v['count'] for v in by_role.values()) == admission['count']
        assert sum(v['passed'] for v in by_role.values()) == admission['passed']
        if (left, right) == WINDOWS[0]:
            slo = result['slo']
            assert slo['alpha'] == ALPHA
            assert slo['window_start_ms'] == left and slo['window_end_ms'] == right
            stored_slo = slo['window_admissions']
            assert set(stored_slo['request_ids']) == {r['request_id'] for r in sample}
            for clock, stats in (('admission', admission), ('arrival', arrival)):
                assert stats['passed'] == stored_slo[clock]['passed']
                assert stats['count'] == stored_slo[clock]['count']
                near(stats['percent'] / 100, stored_slo[clock]['rate'])
            assert after == stored_slo['completion_after_window_end_count']
        rows.append(dict(
            num_ssu=num_ssu, strategy='baseline', order='random', placement='ring_hash',
            start_ms=left, end_ms=right, U_percent=100 * utilization,
            passed=admission['passed'], count=admission['count'], slo_percent=admission['percent'],
            completed_after_window_count=after, all_npus_active=all_active,
            per_npu_U_percent=[100 * value / (right - left) for value in compute_by_npu],
            compute_ms_by_npu=compute_by_npu, active_ms_by_npu=active_by_npu,
            per_role=by_role, arrival_clock_same_cohort=arrival,
            admitted_request_ids=sorted(r['request_id'] for r in sample),
        ))
    evidence = dict(
        num_ssu=num_ssu, input_fingerprint=result['input_fingerprint'],
        completed_requests=len(completed), makespan_ms=summary['makespan_ms'],
        thresholds_ms=thresholds, placement_audit=result['placement_audit'],
        invariants=summary['invariants'],
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (result_path, manifest_path)},
    )
    return rows, evidence


def report(rows, thresholds):
    tables = []
    for left, right in WINDOWS:
        label = '主窗口 [2,4) 秒' if right == 4000 else '补充窗口 [2,20) 秒'
        lines = [f'## {label}', '', '| SSU 数 | NPU 平均利用率 | TTFT SLO×1.5 达标率 |',
                 '|---:|---:|---:|']
        for row in rows:
            if row['start_ms'] == left and row['end_ms'] == right:
                lines.append(f"| {row['num_ssu']} | {row['U_percent']:.4f}% | "
                             f"{row['slo_percent']:.4f}%（{row['passed']}/{row['count']}） |")
        tables.append('\n'.join(lines))
    historical = '\n'.join(
        f"| [{r['start_ms']/1000:g},{r['end_ms']/1000:g}) 秒 | {r['U_percent']:.4f}% | "
        f"{r['slo_percent']:.4f}%（{r['passed']}/{r['count']}） |"
        for r in HISTORICAL_STRIPE
    )
    return f'''# Ring hash 放置：Baseline random

32 NPU，8 层，seed 7；每卡固定 40A＋80B，A=128K 总长/NQL256，B=32K 总长/NQL4096。每盘 40 GiB/s、每卡接收上限 50 GiB/s，batch=1，Baseline 使用单 Path0，保留跨请求首层预取。

输入来自原实验 `reference_manifest.json.gz`，仅重新映射落盘。请求顺序、原始身份、NPU 绑定、计算时间、块大小和 t=0 到达时间保持一致。使用项目原有 ring hash，每个 SSU 有 256 个虚拟节点；块键为 `(original_request_id, block_index)`，不含层号。**同一个 KV block 的全部 8 层放在同一个盘上；一个请求的不同 block 可分散到不同盘。**

{chr(10).join(chr(10) + table for table in tables)}

## 统计口径

利用率＝窗口内所有 NPU 的实际计算时间之和 ÷（32×窗口时长）。计算区间在窗口边界裁剪，包含 I/O 等待影响。

SLO 样本为接纳时间落在半开窗口内的请求，延迟＝最终完成时间−接纳时间，阈值＝1.5×8×原始每层计算时间；A 阈值 {thresholds['A']:.6f} ms，B 阈值 {thresholds['B']:.6f} ms。这里是 8 层 prefill 完成延迟的 TTFT 代理，**不含接纳前的请求队列等待**。窗末仍未完成的样本跟踪到最终完成，保留在分母；不使用 data 中 78 层 TTFT 直接乘 1.5。

两次仿真均完成全部 3840 请求。统计脚本独立复算两窗口计算时间、主窗口 SLO、每条请求原始计算时间，并与原始结果核对。不同 SSU 数会改变窗口内接纳的请求数和 A/B 比例；本次是单 seed 结果。

## 原条带放置参考

下表引用原实验已发布的 **SSU=3、Baseline random** 结果，**不是本轮重新运行的条带实验**。

| 窗口 | NPU 平均利用率 | TTFT SLO×1.5 达标率 |
|---|---:|---:|
{historical}

[历史结果 CSV]({HISTORICAL_URL})

## 复现

在包含 `run_ring_baseline.py` 和 `reference_manifest.json.gz` 的源码目录执行：

```bash
python run_ring_baseline.py --num-ssu 3
python run_ring_baseline.py --num-ssu 4
python summarize_ring_results.py
```

[本轮精确 CSV](summary.csv) · [复算数据与校验依据](summary.json)
'''


def main():
    reference_path = ROOT / 'reference_manifest.json.gz'
    reference = read(reference_path)
    rows, cases = [], []
    for num_ssu in (3, 4):
        case_rows, evidence = summarize_case(num_ssu, reference)
        rows.extend(case_rows)
        cases.append(evidence)
    assert cases[0]['thresholds_ms'] == cases[1]['thresholds_ms']
    payload = dict(
        all_checks_passed=True, num_npu=32, n_layers=8, seed=7, alpha=ALPHA,
        source_manifest_sha256=sha(reference_path), summarizer_sha256=sha(Path(__file__)),
        metric='8-layer prefill completion minus admission; excludes pre-admission queue',
        cohort='Requests admitted in [start_ms,end_ms), followed to final completion',
        utilization='Clipped total compute card-time / (32 * window duration)',
        results=rows, cases=cases,
        historical_stripe_reference=dict(source=HISTORICAL_URL, rerun_in_this_task=False,
                                         results=HISTORICAL_STRIPE),
    )
    (OUT / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    columns = ('num_ssu', 'strategy', 'order', 'placement', 'start_ms', 'end_ms',
               'U_percent', 'passed', 'count', 'slo_percent',
               'completed_after_window_count', 'all_npus_active')
    with (OUT / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    (OUT / 'README.md').write_text(report(rows, cases[0]['thresholds_ms']), encoding='utf-8')
    print(json.dumps([{key: row[key] for key in columns} for row in rows], ensure_ascii=False))


if __name__ == '__main__':
    main()
