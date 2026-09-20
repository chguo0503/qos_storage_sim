#!/usr/bin/env python3
"""Summarize native runs under the user's request-boundary exception."""
import ast
import json
from pathlib import Path

from audit_boundary_underload import audit, summary_row, read
from run_fixed128_32 import profile

ROOT = Path(__file__).resolve().parent
EXP = ROOT/'results/boundary_exempt_underload_20260914'
OUT = ROOT.parents[1]/'boundary_underload_deliverables'
FIFO = ROOT/'results/random_multitype_search_20260914/screen/strict_extended_seed7_fifo'
ONCE = EXP/'native/strict_extended_seed7_once'
LOW = EXP/'native/boundary_probe_094_seed7_fifo'


def main():
    OUT.mkdir(exist_ok=True)
    cases = [FIFO, ONCE]
    if (LOW/'metrics.json').exists():
        cases.append(LOW)
    cached = [read(p) for p in (EXP/'audits').glob('*_audit.json')]
    audits = []
    for case in cases:
        existing = next((a for a in cached if a['case']==str(case.resolve())), None)
        audits.append(existing if existing is not None else audit(case))
    summaries = [summary_row(a) for a in audits]
    (EXP/'selected_native_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    metas = [read(p/'metadata.json') for p in cases]
    assert metas[0]['input_fingerprint'] == metas[1]['input_fingerprint']
    assert all(a['boundary_exempt_candidate_pass'] for a in audits[:2])
    data = ast.literal_eval((ROOT/'data').read_text())
    lines = ['# 请求内部欠载、SS↔LL交界豁免：原生结果', '',
        '日期：2026-09-14。8 NPU；每卡随机混合；ring hash；8层；固定seed=7。所有结果为有限候选筛选后的单次原生运行，不代表所有随机输入。', '',
        '**按用户澄清，欠载要求不包括短、长请求交界的额外预取需求。此前把交界Vnext/Ccurrent也强制限制在容量以内，筛选过严。新口径下可以恢复一组有FIFO等待损失的输入：整机U 98.31%→99.51%，短请求U 97.08%→99.89%。改善有限，尚未达到87%左右的坏基线。**', '',
        '## 本次如何认定欠载', '',
        '- 当前请求常规需求Vown/Cown在整个admission到completion期间累计，包括Stall。全程每盘不超过40 GiB/s。',
        '- 下一层预取需求分解为“同请求内部层＋同角色下一请求L0”和“跨角色下一请求L0”。前者每盘不超过40；后者属于短↔长交界，允许令总参考需求暂时越过容量。',
        '- 两类参考current和continuing分别检查，不能相加。total_prefetch=continuing+boundary严格守恒。不会因一张NPU处于交界就删除整机该段时间。',
        '- 常规及非豁免预取的整机需求达到90%总容量的时长≤5%，检查完整运行、0–4.5秒、0.5–4秒、2–4秒。逐盘近载占比另报，逐盘峰值仍必须≤40。',
        '- 每NPU真实接收始终≤50 GiB/s。交界预取参考可以>50，此时把独占盘和链路仍不可避免的等待下界单列；内部层不允许这种固有冲突。',
        '- 利用率包含交界和所有IO等待。初始L0计入利用率及物理吞吐，它没有前一层计算窗口，因此没有人为添加Vnext/Ccurrent。',
        '- 本报告“短/长”按输入的S/L读取长度角色识别。原生QoS SS/SL/LS/LL标签仍在CSV，未改动；例如32K/NQL1280在代码分类中为SL，并非原生SS。', '',
        '沿用模拟器单位：每SSU40 **GiB/s**，每NPU50 GiB/s；GiB/s与十进制GB/s不同。', '',
        '## 主对照：1 SSU、同一冻结随机输入', '',
        '| 总长度 | NQL中心 | 实际NQL范围 | 每卡数量 | 每层读取MiB | 每层计算ms | V/C GiB/s | 单盘40 GiB/s纯传输ms |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for gid,g in metas[0]['profiles'].items():
        p = profile(data,int(g['total_k']*1024)-g['nql'],g['nql'])
        count = metas[0]['lane_checks'][0]['profile_group_counts'][gid]
        lines.append(f"| {g['total_k']}K | {g['nql']} | {g['nql_range'][0]}–{g['nql_range'][1]} | {count} | {p['read_gib']*1024:.3f} | {p['compute_us']/1000:.3f} | {p['B_gib_s']:.3f} | {p['read_gib']/40*1000:.3f} |")
    lines += ['', '每卡40条短＋5条长，共360条请求；每卡独立随机打乱，无固定长卡/短卡。同一NPU的(总长度,NQL)不重复。NQL在测量网格内插值，不缩放计算时间，不外推。纯传输时间不含排队，实际逐层读取时间见CSV。', '',
        f"FIFO与Once的完整输入指纹相同：`{metas[0]['input_fingerprint']}`。FIFO保持原生单Path0，Once保持原始5ms周期。", '',
        '统计窗口为2–4秒，8卡全程都有请求执行，各卡均实际计算两类请求。', '',
        '| 指标 | Baseline FIFO | Once 5ms |', '|---|---:|---:|']
    af,ao=audits[:2]
    wf,wo=(a['windows']['warm_2000_4000ms'] for a in (af,ao))
    mf,mo=(read(p/'metrics.json') for p in (FIFO,ONCE))
    for title,a,b in [('整机NPU利用率',mf['U_percent'],mo['U_percent']),
                       ('短请求NPU利用率',mf['short_U_percent'],mo['short_U_percent']),
                       ('长请求NPU利用率',mf['long_U_percent'],mo['long_U_percent']),
                       ('TTFT SLO×1.5通过率',mf['slo_1p5_percent'],mo['slo_1p5_percent'])]:
        lines.append(f'| {title} | {a:.4f}% | {b:.4f}% |')
    for title,key in [('内部层Stall，card-ms','internal_stall_card_ms'),('所有下一请求L0 Stall，card-ms','l0_stall_card_ms')]:
        lines.append(f"| {title} | {wf['utilization'][key]:.6f} | {wo['utilization'][key]:.6f} |")
    lines += [f"| 其中跨短长角色L0 Stall，card-ms | {wf['job_stats']['cross_role_boundary']['actual_stall_card_ms']:.6f} | {wo['job_stats']['cross_role_boundary']['actual_stall_card_ms']:.6f} |", '',
        'card-ms是所有NPU等待毫秒数之和。8卡×2秒=16000 card-ms；FIFO总等待270.260 card-ms、Once总等待78.071 card-ms，对应整机改善约1.201个百分点。内部层等待减少约204 card-ms，L0等待反而增加约11.8 card-ms。长请求利用率也略有下降，不能说各类型都改善。', '',
        '两策略所有非冷启动预取任务的独占读取不可避免等待下界均为0。这说明本例等待不属于“单独占盘也读不完”的固有下限。同输入调度对照表明内部等待有显著可改善空间；不能仅凭某段时间重合把全部等待都认定为同一个长流导致。', '',
        '**TTFT口径为completion−admission**，阈值为1.5×8×原始单层计算时间。admission前卡上排队不计入该口径；逐请求CSV同时提供arrival口径。warm FIFO为124/124、Once为126/126，warm入场请求集合可不同；全部360请求CDF按相同request ID比较。', '',
        '## 带宽检查', '', '| 指标，完整运行 | FIFO | Once |', '|---|---:|---:|']
    for title,key,field in [('当前请求需求峰值，GiB/s','current','fleet_peak_gib_s'),
                             ('非豁免预取峰值，GiB/s','continuing','fleet_peak_gib_s'),
                             ('含交界总预取峰值，GiB/s','total_prefetch','fleet_peak_gib_s'),
                             ('含交界总预取>40时长，ms','total_prefetch','fleet_over_capacity_ms')]:
        lines.append(f"| {title} | {af['windows']['full_run'][key][field]:.6f} | {ao['windows']['full_run'][key][field]:.6f} |")
    lines += ['', '当前及非豁免需求超载时长均为0。完整运行当前需求≥36 GiB/s占比约2.62%/2.64%；四个审计窗口中最大约3.42%，符合“仅偶尔接近总容量”。总预取85.7 GiB/s是交界预算窗口的需求，真实单盘服务没有超过40 GiB/s。', '',
        '图中同时保留current、continuing、total三条参考线，交界额外需求单独画。物理服务使用相同2ms桶统计真实字节；不能把不同NPU不同层周期的平均接收带宽相加当成盘吞吐。', '']
    timebill=EXP/'local_fifo_timebill.json'
    if timebill.exists():
        bill=read(timebill);t=bill['target_layer'];account=bill['time_accounting']
        lines += ['## 直接的FIFO队头阻塞证据', '',
            f"事后选取warm中等待最久的短请求内部层：NPU{t['npu_id']}，request {t['request_id']}，32K/NQL{t['nql']}，读取L{t['layer']}。完整读取窗口内没有任何短↔长交界参考或交界逾期IO。常规需求恒为33.276635/40 GiB/s，预取总需求峰值也仅33.276635。", '',
            '| 时间账 | ms |', '|---|---:|',
            f"| 开始计算L5，同时预取L6 | {t['io_release_ms']:.6f} |",
            f"| 当前计算结束／L6应到齐 | {t['compute_deadline_ms']:.6f} |",
            f"| L6实际首次获得盘服务 | {t['first_ssd_start_ms']:.6f} |",
            f"| L6实际到齐NPU | {t['io_ready_ms']:.6f} |",
            f"| 服务前排队 | {account['waiting_before_first_ssd_ms']:.6f} |",
            f"| 其中前方两整层长读 | {account['two_contiguous_long_layers_service_ms']:.6f} |",
            f"| 本层SSD读取＋尾部链路 | {account['continuous_own_ssd_service_ms']+account['final_link_tail_ms']:.6f} |",
            f"| 可掩盖的计算窗口 | {account['hidden_by_current_compute_ms']:.6f} |",
            f"| 暴露IO Stall | {account['exposed_stall_ms']:.6f} |", '',
            '两层200K长读均经过“首末SSD跨度=该层字节/40”及逐IO无重叠校验，可确认连续占盘，没有把交织读取误算成整层阻塞。对已入队的5层作局部算术重排，三个短读优先、两个长读随后，五个deadline都能满足，盘结束时间不变。这证明该处等待可避免，不是整轮策略收益预测；局部最差层周期利用率也不能当作整机平均。详见local_fifo_timebill.json。', '']
    if len(audits)>2:
        low=audits[2];lm=read(LOW/'metrics.json');lw=low['windows']['warm_2000_4000ms']
        total=lw['utilization']['internal_stall_card_ms']+lw['utilization']['l0_stall_card_ms']
        hard=lw['job_stats']['cross_role_boundary']['unavoidable_transfer_stall_lower_bound_card_ms']
        lines += ['## 更低利用率候选的限制', '',
            f"boundary_probe_094：10 SSU，每卡混合32K/NQL中心128与200K/NQL中心256，数量比3:1。原生FIFO整机U={lm['U_percent']:.4f}%，短请求U={lm['short_U_percent']:.4f}%；新口径审计{'通过' if low['boundary_exempt_candidate_pass'] else '不通过，见pass_checks'}。", '',
            f"warm总Stall={total:.6f} card-ms，其中内部层={lw['utilization']['internal_stall_card_ms']:.6f}、L0={lw['utilization']['l0_stall_card_ms']:.6f}；跨请求独占传输仍不可避免的等待下界={hard:.6f} card-ms，占总Stall下界比例{100*hard/total:.2f}%。这是交界物理下限的限制，不能把这部分损失宣传为FIFO调度造成，也没有为该候选运行Once对照。", '']
    lines += ['## 本轮范围与结论', '',
        '筛选了540组均分代理候选，12组进一步使用实际ring字节分布复查。代理不是原生结果；本报告以原生轨迹审计为准。复用了strict_extended已有FIFO结果，并新运行其Once和boundary_probe_094的FIFO。未挑选新的随机种子，未更改原生盘/链路/仲裁模型。', '',
        '交界豁免后，不必再通过大幅增加短请求NQL强行掩盖下一长请求L0。因此可以出现“内部参考需求欠载，交界突发造成排队，短请求内部等待增加”的案例。当前找到的是有限收益，不是已经找到大幅低利用率且可由Once修复的随机坏例。', '',
        '## 复现', '', '在解压后的qos_storage_sim目录运行：', '', '```bash',
        'python run_boundary_random.py --spec results/boundary_exempt_underload_20260914/native_specs.json --case strict_extended --seed 7 --policy fifo --stage reproduce',
        'python run_boundary_random.py --spec results/boundary_exempt_underload_20260914/native_specs.json --case strict_extended --seed 7 --policy once --stage reproduce',
        'python audit_boundary_underload.py --root results/boundary_exempt_underload_20260914/reproduce', '```', '',
        '原始核心commit为75e10b8a84d4054921cd3149af40507444efb3fb。配套代码和冻结manifest可重放本轮输入。', '',
        'boundary_underload_all_images.zip单独包含本轮所有PNG；boundary_underload_data_and_code.zip包含逐请求、逐层、原生结果、审计与代码。', '']
    (OUT/'boundary_underload_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'boundary_underload_results.md')


if __name__=='__main__':
    main()
