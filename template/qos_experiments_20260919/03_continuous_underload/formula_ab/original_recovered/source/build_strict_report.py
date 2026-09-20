#!/usr/bin/env python3
"""Build the strict-underload report from completed, audited native results."""
import ast
import gzip
import json
import math
from pathlib import Path

from run_fixed128_32 import profile

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'results/strict_random_underload_20260914'
DEST = ROOT.parents[1] / 'strict_underload_deliverables'


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def window_util(result, left, right):
    compute = math.fsum(max(0., min(l['compute_end_ms'], right)-max(l['compute_start_ms'], left))
        for b in result['summary']['microbatch_metrics'] for l in b['layer_metrics'])
    return 100*compute/(8*(right-left))


def main():
    DEST.mkdir(exist_ok=True)
    rows = read(EXP / 'native_strict_summary.json')
    if isinstance(rows, dict):
        raise ValueError('Expected summary list')
    accepted = [r for r in rows if r['strict_candidate_pass']]
    assert {r['name'] for r in accepted} >= {'strict_probe_091', 'strict_probe_096'}
    table = ast.literal_eval((ROOT/'data').read_text())
    lines = ['# 全时段欠载：8 NPU random / ring hash 原生验证', '',
        '日期：2026-09-14。结果基于 seed=7 的冻结输入；不代表所有随机种子，也不是穷尽搜索。', '',
        '**结论：找到全程欠载、仅极少时段接近容量的输入，但本轮没有找到同时令 Baseline FIFO 利用率明显变差的输入。此前约87%–89%的平均欠载案例不符合新条件，不能继续作为推荐。**', '',
        '## 判定口径', '',
        '- 8张NPU，每张卡都独立随机混合全部请求画像，固定分配、batch=1；所有请求到达时间为0。每卡的(总长度,NQL)不重复。',
        '- 每SSU容量40 GiB/s，每NPU接收容量50 GiB/s；这里沿用模拟器的二进制GiB/s，不是十进制GB/s。',
        '- KV block按(request_id,block_index)做ring hash，256虚拟节点/SSU；同一block的8层复用同一盘映射。尾块按实际字节计。',
        '- 当前请求参考需求为Vown/Cown，在请求计算和等待期间均累计；预取参考需求为Vnext/Ccurrent，在当前计算窗口累计，包含下一请求的L0。这两条参考分别检查，不能相加。',
        '- 从t=0一直检查到最后请求完成，任意正时长事件区间内每盘两种参考需求均≤40 GiB/s。整机同时检查≤40×SSU。',
        '- 将“接近容量”具体定为≥90%：每盘36 GiB/s、3盘整机108 GiB/s；任一盘接近容量的时长占比≤5%。完整时段、0–4.5秒、0.5–4秒和2–4秒都检查，不能靠尾部空闲稀释。',
        '- 单张NPU的Vnext/Ccurrent也必须≤50 GiB/s，且不存在单独占有盘和链路仍无法在计算窗口读完的下界冲突。',
        '- 容量峰值由逐事件积分审计；物理吞吐图用相同2ms桶的真实字节，不裁剪、不用平滑隐藏峰值。SSD忙时以40 GiB/s服务，不等于请求需求超载。', '',
        '## 合格的原生结果', '',
        'A=strict_probe_091，B=strict_probe_096；两组均为3 SSU，总容量120 GiB/s。带宽峰值覆盖完整仿真。', '',
        '| 场景/策略 | 当前需求整机峰 | 预取整机峰 | 预取最大单盘峰 | 任一盘≥36占比（全程） | NPU U，0–4.5秒 | NPU U，0.5–4秒 | NPU U，2–4秒 | TTFT SLO×1.5，2–4秒入场 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    enriched = []
    for row in accepted:
        folder = ROOT/row['case']
        result = read(folder/'result.json.gz')
        audit = read(folder/'strict_audit.json')
        metrics = read(folder/'metrics.json')
        row = dict(row, U_first4500_percent=window_util(result,0,4500), U_broad_percent=window_util(result,500,4000))
        enriched.append(row)
        label = ('A' if row['name'].endswith('091') else 'B') + '/' + ('FIFO' if row['policy']=='fifo' else 'Once 5ms')
        lines.append(f"| {label} | {row['full_current_fleet_peak_gib_s']:.3f} | {row['full_prefetch_fleet_peak_gib_s']:.3f} | {row['full_prefetch_peak_per_ssu_gib_s']:.3f} | {row['full_prefetch_any_ssu_near_capacity_percent']:.4f}% | {row['U_first4500_percent']:.4f}% | {row['U_broad_percent']:.5f}% | {row['U_percent']:.2f}% | {metrics['slo_passed']}/{metrics['slo_count']}（{row['slo_1p5_percent']:.2f}%） |")
    lines += ['', '带宽单位均为GiB/s。所有合格场景的盘需求超载时间为0；上表利用率含对应窗口内的真实IO等待，未把空闲NPU移出分母。0–4.5秒含冷启动；完整仿真的容量检查也包括尾段。下文card-ms表示各NPU等待毫秒数之和。', '',
        '**TTFT沿用本实验的admission口径**：完成时间减进入NPU执行队列头的时间；SLO阈值=1.5×8×原始每层计算时间。所有请求t=0到达，arrival到completion还包含卡上等待前序请求的时间，不能把上述通过率解释为arrival口径；逐请求CSV同时给出两者。', '',
        '## 请求选择与读取量', '',
        '总长度固定，NQL在中心附近选取不同整数，以满足每卡画像不重复。下表是中心值，不是所有请求都完全相同；C由data测量网格内插值得到，不做计算缩放或范围外外推。实际NQL和每层数值见CSV。', '']
    for name in ['strict_probe_091','strict_probe_096']:
        folder = EXP/'native'/f'{name}_seed7_fifo'
        meta = read(folder/'metadata.json')
        label = 'A' if name.endswith('091') else 'B'
        lines += [f'### 场景{label}：{name}', '',
            '| 总长度 | NQL中心 | 实际NQL范围 | 每层读取MiB | 每层计算ms | V/C GiB/s | 独占单盘40 GiB/s纯传输ms | 每卡请求数 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
        counts = meta['lane_checks'][0]['profile_group_counts']
        for gid,g in meta['profiles'].items():
            p = profile(table, int(g['total_k']*1024)-g['nql'],g['nql'])
            lines.append(f"| {g['total_k']}K | {g['nql']} | {g['nql_range'][0]}–{g['nql_range'][1]} | {p['read_gib']*1024:.3f} | {p['compute_us']/1000:.3f} | {p['B_gib_s']:.3f} | {p['read_gib']/40*1000:.3f} | {counts[gid]} |")
        a = read(folder/'strict_audit.json')
        lines += ['', f"输入共{sum(g['requests'] for g in meta['profiles'].values())}个请求；每卡独立打乱相同数量的画像。完整仿真{a['makespan_ms']/1000:.4f}秒；单NPU预取需求最大{max(a['max_prefetch_rate_per_npu_gib_s']):.3f} GiB/s。", '',
            f"全程有{a['missed_io_deadline_job_count']}个非冷启动预取层晚于原截止时间到齐，总欠债{a['windows']['full_run']['overdue_io_debt']['card_ms']:.6f} card-ms；这表明欠载不保证零停顿，但本例损失很小。2–4秒内的欠债为{a['windows']['warm_2000_4000ms']['overdue_io_debt']['card_ms']:.6f} card-ms。", '']
    lines += ['单盘40 GiB/s列是V/40纯传输时间，不包含FIFO排队，也不是3盘并行场景下的实际读取耗时。实际每层IO完成、计算和Stall时间在原生结果与导出数据中。', '',
        '## 被排除的案例', '',
        '| 案例 | 排除理由 |', '|---|---|']
    for row in rows:
        if row['strict_candidate_pass']:
            continue
        reason = []
        if not row['storage_references_within_capacity_full_run']:
            reason.append(f"完整轨迹预取单盘峰{row['full_prefetch_peak_per_ssu_gib_s']:.3f}>40；超载{row['full_prefetch_any_ssu_over_capacity_ms']:.3f}ms")
        if not row['npu_prefetch_links_within_capacity_full_run']:
            reason.append(f"单NPU预取峰{row['max_prefetch_npu_gib_s']:.3f}>50，固有链路限制混入Stall")
        lines.append(f"| {row['name']}/{row['policy']} | {'；'.join(reason) or '参见pass_checks'} |")
    lines += ['', '特别是native200_32_s1在前4.5秒合格，完整8.21秒轨迹却超载，因此没有通过截取窗口保留它。', '',
        '## 能说明什么', '',
        '本轮260组均分代理有界筛选中，62组满足其0–4.5秒等窗口的存储参考条件；这不等于62组通过完整原生仿真或链路检查。随后对真实ring分布、50 GiB/s链路及理论候选复查，并完成5组FIFO与1组Once原生验证。代理只用于筛选，最终结论以原生逐事件审计为准。', '',
        '严格参考欠载在理论上仍可能发生FIFO队头阻塞。例如单盘40 GiB/s上，128K/NQL2048长层先发，32K/NQL512短层晚0.2ms发：参考需求合计约15.057 GiB/s，短层仍可能被前序长层IO拖过其截止时间。该局部排队计算的完整假设和源码证据见theory_notes.md；它不是本轮8卡随机混合的原生实验结果，也不能据此声称整机利用率已经很差。', '',
        '当前8卡随机混合的完整实验中，严格欠载案例没有形成频繁且足够长的等待，预取基本被计算掩盖。因此不能继续推荐之前靠运行中需求超载得到低U的输入，也不能据本轮有限搜索断言严格欠载下FIFO永远不会差。', '',
        '冷启动L0没有前一层计算窗口，Vnext/Ccurrent未定义，已单列cold IO；参考在截止时间结束后不再累计，不代表迟到IO消失，因此图和审计另列过期未完成层数。', '',
        '## 复现与文件', '',
        '原始核心来源commit：75e10b8a84d4054921cd3149af40507444efb3fb。本轮未修改原生调度/盘/链路实现，使用现有exact-tail适配与独立输入构造。Once保持原始5ms周期。', '',
        '```bash',
        'python run_strict_random.py --spec results/strict_random_underload_20260914/strict_native_candidates.json --case strict_probe_096 --seed 7 --policy fifo --stage reproduce',
        'python run_strict_random.py --spec results/strict_random_underload_20260914/strict_native_candidates.json --case strict_probe_096 --seed 7 --policy once --stage reproduce',
        'python audit_strict_random.py --case results/strict_random_underload_20260914/reproduce/strict_probe_096_seed7_fifo',
        '```', '',
        'strict_underload_all_images.zip：所有本轮合格场景图片单独打包。strict_underload_data_and_code.zip：原生输入/结果、全部候选审计、逐请求数据、绘图/运行脚本和本报告。', '']
    (DEST/'strict_underload_results.md').write_text('\n'.join(lines),encoding='utf-8')
    (EXP/'accepted_native_results.json').write_text(json.dumps(enriched,ensure_ascii=False,indent=2)+'\n')
    print(DEST/'strict_underload_results.md')


if __name__ == '__main__':
    main()
