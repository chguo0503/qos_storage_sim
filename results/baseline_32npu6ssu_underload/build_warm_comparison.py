#!/usr/bin/env python3
"""Build the cross-study report only from completed, audited per-seed tables."""
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = {7, 19, 43, 67, 101}


def read(name, count):
    path = HERE / name / 'per_seed.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count, (name, len(rows), count)
    assert all(r['status'] == 'complete' for r in rows), name
    assert all(r['audit_passed'].lower() == 'true' for r in rows), name
    return rows


def select(rows, **keys):
    found = [r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items())]
    assert len(found) == 5 and {int(r['seed']) for r in found} == SEEDS, keys
    return found


def stat(rows, field):
    values = [float(r[field]) for r in rows]
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'min': min(values), 'max': max(values), 'n': len(values)}


def fmt(rows, field):
    x = stat(rows, field)
    return f"{x['mean']:.2f} ± {x['sample_sd']:.2f}"


def pair_table(rows, conditions):
    out = ['| 输入 | Baseline U % | Once U % | Baseline SLO % | Once SLO % |',
           '|---|---:|---:|---:|---:|']
    for label, keys in conditions:
        b = select(rows, **keys, strategy='baseline')
        o = select(rows, **keys, strategy='once')
        out.append('| ' + ' | '.join([label, fmt(b, 'device_utilization_percent'),
                   fmt(o, 'device_utilization_percent'), fmt(b, 'warm_admission_slo_percent'),
                   fmt(o, 'warm_admission_slo_percent')]) + ' |')
    return '\n'.join(out)


def raw_category_table(rows):
    b = select(rows, arm='raw84', strategy='baseline')
    o = select(rows, arm='raw84', strategy='once')
    out = ['| 完整原表类别 | Baseline 暖接纳 SLO % | Once 暖接纳 SLO % |',
           '|---|---:|---:|']
    for category in ('SS', 'SL', 'LS', 'LL'):
        field = f'warm_{category}_admission_slo_percent'
        out.append(f'| {category} | {fmt(b, field)} | {fmt(o, field)} |')
    return '\n'.join(out)


def main():
    names = {'paired_once_5seeds': 20, 'role_separated_5seeds': 40,
             'raw_data_random_5seeds': 20, 'raw_quartet_replacement': 22}
    required_flags = {
        'paired_once_5seeds': ('complete_grid', 'all_study_conditions_met'),
        'role_separated_5seeds': ('all_complete', 'all_study_conditions_met'),
        'raw_data_random_5seeds': ('means_published', 'all_audits_passed'),
        'raw_quartet_replacement': ('all_complete', 'all_technical_audits_passed'),
    }
    for name, flags in required_flags.items():
        summary = json.loads((HERE/name/'summary.json').read_text())
        assert all(summary.get(flag) is True for flag in flags), (name, flags)
    rows = {name: read(name, count) for name, count in names.items()}
    paired, role, raw, quartet = (rows[k] for k in names)
    qr = stat(select(quartet, family='raw_feasible_q2', order='random', strategy='baseline'), 'device_utilization_percent')['mean']
    qo = stat(select(quartet, family='raw_feasible_q2', order='ordered', strategy='baseline'), 'device_utilization_percent')['mean']
    qonce = stat(select(quartet, family='raw_feasible_q2', order='ordered', strategy='once'), 'device_utilization_percent')['mean']
    rawb = stat(select(raw, arm='raw84', strategy='baseline'), 'device_utilization_percent')['mean']
    rawo = stat(select(raw, arm='raw84', strategy='once'), 'device_utilization_percent')['mean']
    qmain = [r for r in quartet if r['family'] == 'raw_feasible_q2']
    qvalid = sum(r['all_study_conditions_met'].lower() == 'true' for r in qmain)
    data_path = HERE.parents[1] / 'data'
    catalog = ast.literal_eval(data_path.read_text())
    small_nql = [row for (seq, nql), row in catalog.items() if nql < 512]
    small_nql_request_share = len(small_nql) / len(catalog)
    small_nql_compute_share = sum(row[1] for row in small_nql) / sum(row[1] for row in catalog.values())
    text = [
        '**32 NPU、6 SSU：暖窗对比与原始 data 验证**', '',
        f'原始欠载四画像的 Baseline 利用率为 Random {qr:.4f}%、Ordered {qo:.4f}%；'
        f'全原表随机抽样的 Baseline / Once 利用率为 {rawb:.4f}% / {rawo:.4f}%。'
        '原图构造输入的坏结果需要结合具体画像解释，不能直接套到原始数据。各组输入、约束及 SLO 结果见下表。', '',
        '以下均为本仓库仿真；每盘 40 GiB/s、每卡接收链路 50 GiB/s、每请求 8 层。'
        '主窗固定为 [2000,4000) ms。五个种子为 7、19、43、67、101；表中是种子等权均值 ± 样本标准差，单位为百分数。'
        '表格保留两位小数，显示 100.00% 不必然代表完全没有等待；CSV 保留完整精度。'
        'Once 指 once / Once per layer，使用 5 ms 共享快照、静态 CIR，未开启运行时 CIR 更新；不是 NewOnce。'
        '调度算法在真实客户端上的额外计算开销未建模，这些差值不能直接当作硬件加速比。', '',
        r'\(U=\frac{\text{32 卡在 [2,4)s 内的计算时间之和}}{32\times2\text{ 秒}}\)。'
        '暖窗 SLO 选取在 [2,4)s 被 NPU 接纳的请求，判断“完成时刻−接纳时刻 ≤ 1.5×8×单层计算时间”；'
        '越过 4 秒才完成的请求仍跟踪至完成。', '',
        '**这个 SLO 是接纳后的处理时间代理，不能直接称为用户端 TTFT。** '
        '输入全在 t=0 到达，接纳前等待未计入主表。暖窗到达请求数为零，不能计算暖窗到达队列的达标率；'
        '若对暖窗接纳的同一批请求改用真实到达时刻计时，达标率均为 0%。'
        '各策略在暖窗接纳的请求集合可能不同，详细报告另列完整同人口结果。', '',
        '**一、原图四画像：随机打乱与 Ordered**', '',
        pair_table(paired, [('Random', {'order': 'random'}), ('Ordered', {'order': 'ordered'})]), '',
    ]
    br = stat(select(paired, order='random', strategy='baseline'), 'device_utilization_percent')['mean']
    bo = stat(select(paired, order='ordered', strategy='baseline'), 'device_utilization_percent')['mean']
    bslo = stat(select(paired, order='random', strategy='baseline'), 'warm_admission_slo_percent')['mean']
    text += [f'Baseline 五种子均值由 {br:.4f}% 降到 {bo:.4f}%，下降 {br-bo:.4f} 个百分点、相对 {(br-bo)/br*100:.4f}%。'
             '因此原 seed 7 的相对 10.04% 是一个存在性案例，五种子没有稳定复现超过 10% 的下降。'
             '全部 20 格均通过全程逐盘欠载、32 卡整窗 active、每卡暖窗有长短计算以及原暖机检查。', '',
             '这里的 Random 是“把事先挑选的三种外推短画像和一种插值长画像分别打乱”，并不是“从 data 全表随机抽样”。'
             '改变顺序没有消除这些画像特有的计算时间与读取量差异。'
             f'第一组 Random / Baseline 的暖接纳 SLO 为 {bslo:.2f}%，是该构造人口在上述计时口径下的仿真统计均值。', '',
             'Random 的种子同时影响卡内乱序与同刻提交次序；原图和原始四画像的 Ordered 各自采用同一固定画像模板，'
             '跨种子主要检验提交随机性的影响，不能称为五种独立设计的坏顺序。', '',
             '[完整表、逐种子结果与配对审计](paired_once_5seeds/comparison.md)。', '',
             '**二、同一批请求固定分成长卡和短卡**', '',
             pair_table(role, [(f'{n} 长卡 / {32-n} 短卡，{order}', {'long_npu_count': n, 'order': order})
                               for n in (11, 6) for order in ('random', 'round_robin')]), '',
             '| 绑定与短卡顺序 | Baseline 短卡 U % | Once 短卡 U % | Baseline 长卡 U % | Once 长卡 U % |',
             '|---|---:|---:|---:|---:|']
    for n in (11, 6):
        for order in ('random', 'round_robin'):
            b = select(role, long_npu_count=n, order=order, strategy='baseline')
            o = select(role, long_npu_count=n, order=order, strategy='once')
            text.append('| ' + ' | '.join([f'{n}/{32-n}，{order}', fmt(b, 'short_card_utilization_percent'),
                        fmt(o, 'short_card_utilization_percent'), fmt(b, 'long_card_utilization_percent'),
                        fmt(o, 'long_card_utilization_percent')]) + ' |')
    text += ['', '全局仍是第一组的同一 19,456 条构造请求，保留每条 C、V、到达时刻和逐块落盘位置；改动的是 NPU 绑定与卡内顺序。'
             '同一绑定比例、同一种子内的 random 与 round_robin 保持每张新卡的人口相同；'
             '不同绑定比例之间，以及与原混合实验之间，每卡人口不同。'
             '按用户要求，此组每卡只运行一种角色，故不再要求单卡同时有长短任务。'
             '11/21 和 6/26 绑定的任意画像组合逐盘名义需求上界分别是 31.735895 与 26.559836 GiB/s，均低于 40。', '',
             '两种内部顺序都不能明显修复 Baseline，支持“固定同类卡会持续暴露问题”的判断；'
             '**它还不能证明所有可能的内部排列都一样差。** '
             '长卡使用同一画像，其各层计算相位在暖窗内几乎不漂移，反复形成集中读取；短卡的预取窗口很短，容易等在长读取后面。'
             '长计算本身没有占住别的 NPU，占用共享盘服务时间的是读取。', '',
             '必须区分暖窗和整批结束：6 长卡 / 26 短卡的请求总量绑定不均衡，即使完全没有 I/O 等待，'
             '整批利用率的上界也只有 57.19%。这个尾部损失不能归因于 FIFO；主表的 2 秒窗口中所有卡仍在运行。'
             '6 长卡下 Once 的暖窗利用率虽大幅提高，整批完成时间却微增：random 平均约 1.40 ms，round_robin 约 2.46 ms。', '',
             '[详细结果、有效性与尾部分析](role_separated_5seeds/comparison.md)；'
             '[长卡逐层相位证据](role_separated_5seeds/long_phase_evidence.json)。', '',
             '**三、直接从 data 随机抽样**', '',
             pair_table(raw, [('完整 84 行等概率抽样', {'arm': 'raw84'}),
                              ('保证欠载的 33 行子集等概率抽样', {'arm': 'underload33'})]), '',
             '原表随机抽样下，两种策略的设备 U 确实接近；Baseline 的暖接纳 SLO 也低于原构造输入的 94.80%。'
             '这是更换输入人口后的结果，不能据此认定前一组统计出错。'
             'Once 总体 SLO 更高，也不代表各类别都改善：LL 的暖接纳 SLO 略有下降，见下表。', '',
             raw_category_table(raw), '',
             f'在完整同一批输入请求上，LL 接纳后 SLO 也由 '
             f'{stat(select(raw, arm="raw84", strategy="baseline"), "full_LL_admission_slo_percent")["mean"]:.2f}% '
             f'变为 {stat(select(raw, arm="raw84", strategy="once"), "full_LL_admission_slo_percent")["mean"]:.2f}%。'
             '因此不能宣称 Once 对所有类别都有收益。', '',
             '每卡独立、有放回抽取原始行，使用不同随机序列；不缩放 C，不外推、不插值。'
             '输入长度由每卡累计纯计算时间超过固定阈值决定，以排除 4 秒前跑空。'
             'NQL=64 的原始读取存在 88 KiB 尾块，模拟器以 176 KiB 整块补齐；原始与物理读取量分别保留。', '',
             f'按原目录每行各取一条统计，SS+LS 类别占请求数 {100*small_nql_request_share:.2f}%，'
             f'却只占纯计算时间 {100*small_nql_compute_share:.2f}%。'
             '这说明为何部分请求达标很差时，按计算时间统计的整机 U 仍可能很高。'
             '这里是目录构成统计，不是暖窗实际类别份额；SS/LS 的第二个 S 表示 NQL<512，分类不直接按计算时间 C 切分。', '',
             '**本次完整 84 行抽样的运行未满足之前的逐盘、逐时刻欠载约束；完整目录也不具备任意组合欠载保证。** '
             '有些原始画像的 V/C 很高，不能把其低 SLO 全部解释成 FIFO。'
             '50 GiB/s 单卡链路也是限制：即使首层已提前读完，后七层仍至少需要 7V/50 的传输时间，最后一层还需 C 计算。'
             '原表有 9 种画像满足 7V/50+C > 1.5×8C，哪怕没有盘排队也无法达标；两种策略都受影响。'
             '详细报告列出暖窗实际抽到的这类请求数量，不把原表占比直接当作样本失败率。', '',
             '33 行子集通过“任意 32 卡组合、逐盘都低于 40”的充分条件筛选，只含 SL、LL；'
             '它不代表完整原表，也不覆盖所有可能满足欠载的混合输入。'
             '自然抽样保留原结果，没有为了暖窗单卡同时出现长短任务而重抽种子；'
             '原第四请求完成时刻、每卡混合条件与容量条件在详表分列，不能声称都满足原探索的全部约束。', '',
             '例如，一个时刻的 5 张 32K/128 SS 卡与 27 张 192K/4096 LL 卡，逐盘上界为 38.999454 GiB/s，'
             '说明原始 SS 仍可在受约束的并发组合中合法出现。'
             '这不是整段输入已运行验证的反例，也没有证明每卡暖窗混合或整机 U 下降 10%。'
             '详见 [原始混合输入的数学可行性与限制](raw_mixture_feasibility_note.md)。', '',
             '[抽样方法、四类别 SLO、容量与链路核验](raw_data_random_5seeds/comparison.md)。', '',
             '**四、把 figures 中的四画像换成原始 data 行**', '',
             '原图的 1K 短画像和 192K/768 长画像都不是原表行，因此没有“原封不动换成原表值”的唯一对应关系。'
             '本次提前固定两种替换规则，并保留所有结果。', '',
             '| 角色 | 原图画像 | 保证欠载的原始画像 | 原始单层 C ms | 原始单层 V MiB | 原始 V/C GiB/s |',
             '|---|---|---|---:|---:|---:|',
             '| S1 | 1K/128 | 32K/1024 | 7.257232 | 42.625 | 5.735793 |',
             '| S2 | 1K/256 | 48K/1024 | 9.941586 | 64.625 | 6.348117 |',
             '| S3 | 1K/384 | 64K/1024 | 12.625941 | 86.625 | 6.700073 |',
             '| L | 192K/768 | 160K/1024 | 28.732067 | 218.625 | 7.430756 |', '',
             pair_table(quartet, [('原始四画像 Random', {'family': 'raw_feasible_q2', 'order': 'random'}),
                                  ('原始四画像 Ordered', {'family': 'raw_feasible_q2', 'order': 'ordered'})]), '',
             f'原始可行组的 Baseline 随机均值为 {qr:.4f}%，Ordered 为 {qo:.4f}%，'
             f'随机减 Ordered 的差为 {qr-qo:.4f} 个百分点（相对 {(qr-qo)/qr*100:.4f}%）；'
             f'Ordered 上 Once 减 Baseline 的差为 {qonce-qo:+.4f} 个百分点。'
             '应以这个五种子结果判断原图降幅是否复现，而不是沿用旧画像的结论。', '',
             '该组每卡三种短画像各 14 条、长画像 7 条，共 49 条；全局 1,568 条。'
             'Ordered 使用 L,(S1,S2,S3)×2 的七请求小段，重复七次，并用与原图相同的四组相位构造算法排列。'
             '短任务纯计算份额为 67.490889%，接近原图的 67.467124%；'
             '任意组合逐盘名义需求上界为 39.630698 GiB/s。'
             f'主组 {qvalid}/20 格通过原完整条件：暖机、32 卡整窗 active、每卡有长短计算以及全程逐盘欠载。', '',
             '它改变的不止数据来源：短 C 从 0.528–0.899 ms 增到 7.257–12.626 ms；'
             '长短 C 比由约 28–49 倍缩到约 2.3–4.0 倍，长短读取量差异也明显缩小。'
             '短角色的路由类别由 SS 变为 SL，Once 可选 Path 从 96 条变为 32 条，类别 CIR 由 20 变为 6 GiB/s。'
             '因此这是原始数据的适用性检查，不能叫“只去掉外推”的单因素消融。', '',
             '可以用一个数量级比较理解预取余量：八张长卡各读一层，在单盘上的服务工作量，'
             '原图约需 8.56 ms，而原短 C 最小只有 0.528 ms；原始可行组约需 7.12 ms，短 C 最小已有 7.257 ms。'
             '因此同样规模的长读取突发更容易被计算时间遮住。'
             '这只是由字节量与盘速率得到的机制解释；实际队列还可能有其他工作，不能把它叫最坏排队上界。', '',
             '下面两张图各自独立，均为 seed 7 Baseline；橙色表示接纳后实际 I/O 等待。图按原始时长绘制，极短等待可能不足一个像素。', '',
             '![原始四画像 Random 的32卡暖窗](raw_quartet_replacement/figures/raw_random_baseline_seed7.png)', '',
             '![原始四画像 Ordered 的32卡暖窗](raw_quartet_replacement/figures/raw_ordered_baseline_seed7.png)', '',
             'PDF 与绘图来源见 [图形目录及审计](raw_quartet_replacement/figures/timeline_audit.json)。', '',
             '**最接近原图的替换另做诊断：** S1=32K/128、S2=32K/256、S3=32K/512、L=192K/1024，'
             '保留每一条长任务配 25 组三种短任务。四者最低 V/C 都有 7.520934 GiB/s，'
             '故 32 卡同时 active 时需求至少 240.669886 GiB/s，大于六盘总容量 240 GiB/s；任何排序都无法满足原欠载约束。', '',
             '| 最接近原图的替换，seed 7 Ordered | 暖窗 U % | 暖接纳 SLO % |',
             '|---|---:|---:|']
    for policy in ('baseline', 'once'):
        found = [r for r in quartet if r['family'] == 'raw_nearest_q25' and r['strategy'] == policy]
        assert len(found) == 1
        r = found[0]
        text.append(f"| {policy}（过载诊断） | {float(r['device_utilization_percent']):.2f} | {float(r['warm_admission_slo_percent']):.2f} |")
    text += ['', '这里即使利用率很低，也不能作为“欠载时单 Path 导致低利用率”的证据。'
             '原始可行组没有复现原图降幅，只说明这一替换与排列没有复现，不能推出整个 data 中不存在更坏的合法组合。', '',
             '[原始四画像的逐种子结果、每卡暖窗混合检查与来源审计](raw_quartet_replacement/comparison.md)。', '',
             '**数据与复算**', '',
             '四个子目录分别保存冻结 plan、输入、运行命令、压缩原始结果、独立分析和逐种子 CSV。'
             '本文件只汇总已完成且技术审计通过的 102 个计划格；是否满足欠载、单卡混合等实验条件另行判断，不以技术审计通过代替条件通过。'
             '对均值使用全部预定种子，没有按结果或有效性删掉“不好看”的种子。', '']
    (HERE / 'warm_comparison_tables.md').write_text('\n'.join(text))
    manifest = {'source_tables': {name: {'rows': count, 'sha256': hashlib.sha256((HERE/name/'per_seed.csv').read_bytes()).hexdigest()}
                                 for name, count in names.items()},
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'report_sha256': hashlib.sha256((HERE/'warm_comparison_tables.md').read_bytes()).hexdigest(),
                'aggregation': 'all five predeclared seeds equally weighted; sample SD with n-1',
                'planned_case_count': sum(names.values())}
    (HERE / 'warm_comparison_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(HERE / 'warm_comparison_tables.md')


if __name__ == '__main__':
    main()
