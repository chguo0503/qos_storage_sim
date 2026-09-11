#!/usr/bin/env python3
"""Build a reviewable report from all declared mixed candidates, without filtering failures."""
import argparse,json,statistics,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def avg(rows,key):
    v=[r[key] for r in rows]
    if not v:return '—'
    return f'{statistics.mean(v):.2f}'+(f' ± {statistics.stdev(v):.2f}' if len(v)>1 else '')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    data=read(HERE/'mixed_rebinding/results.json');old=read(HERE/'followup_results.json')
    old_jobs={(j['input']['label'],j['strategy']):j for p in (HERE/'plans').glob('*.json') for j in read(p)['jobs']}
    mixed_jobs={(j['input']['label'],j['strategy']) for p in (HERE/'mixed_rebinding/plans').glob('*.json') for j in read(p)['jobs']}
    if args.require_complete:
        assert data['all_complete'] and data['all_order_pairs_passed'] and not data['errors']
        assert {(r['label'],r['strategy']) for r in data['rows']}==mixed_jobs
        assert not old['pending'] and not old['errors'] and old['completed']==old['planned']==len(old_jobs)
        assert {(r['label'],r['strategy']) for r in old['rows']}==set(old_jobs)
    rows=[r for r in data['rows'] if r['status']=='complete']
    def select(spec,mode,policy):return [r for r in rows if (r['spec_name'],r['mode'],r['strategy'])==(spec,mode,policy)]
    names={'raw176_rebinding':'176K：初版交接','raw200_rebinding':'200K：较晚交接（Once超限）','raw200_earlyguard':'200K：三张卡提前切换','raw200_diversehot':'200K：三种短画像交错','raw176_extendedhot':'176K：延长短请求阻塞段'}
    text=['**每张卡都执行长短请求，是否还能复现固定分卡的阻塞？**','',
      '本报告把“固定长短卡”“每卡随机混合”“每卡定序混合”分开比较。所有输入均来自data原始行，未缩放计算时间、未填充读取量；核心模拟器与策略未修改。所有候选及不满足约束的结果均保留。','',
      '**读表口径**','',
      '32 NPU、6 SSU，每盘40 GiB/s、每卡链路50 GiB/s；每请求8层，warm固定为[2,4)秒。U是窗口内总计算卡时间除以64 NPU·秒。TTFT SLO仍是此前使用的接纳后处理时间代理：窗口内接纳请求跟踪至完成，completion−admission≤1.5×8×单层C。不含接纳前排队，不能当作真实用户端到端TTFT。Once是Once per layer，不是NewOnce。','',
      '表中每卡混合要求32张卡在warm内长、短各有正计算时间；后面的最小值另外检验是否只是贴近窗口边界的一点计算。欠载逐策略、逐盘、全程逐事件核查。需求定义沿用当前接纳请求的每层盘V/C，按此前约定不额外叠加下一请求首层预取项；仿真仍处理所有真实读。这是名义需求约束，不是瞬时I/O到达速率不产生突发的保证。','',
      '**同一每卡人口，只改变请求次序**','',
      '| 原始输入与分配 | 顺序 | seeds数 | Baseline U% | Once U% | Baseline SLO% | Once SLO% | 通过每卡混合、active及欠载的运行 |',
      '|---|---|---:|---:|---:|---:|---:|---|']
    for spec,name in names.items():
      for mode in ('random','ordered'):
        if spec=='raw200_diversehot' and mode=='random':continue
        b=select(spec,mode,'baseline');o=select(spec,mode,'once');both=b+o
        if not both:continue
        if any(r['status']!='complete' for r in data['rows'] if (r['spec_name'],r['mode'])==(spec,mode)):
          text.append(f'| {name} | {mode} | 运行中 | 暂不汇总 | 暂不汇总 | 暂不汇总 | 暂不汇总 | — |');continue
        text.append(f"| {name} | {mode} | {len(b)} | {avg(b,'device_utilization_percent')} | {avg(o,'device_utilization_percent')} | {avg(b,'warm_slo_percent')} | {avg(o,'warm_slo_percent')} | {sum(r['strict_mixed_underload_valid'] for r in both)}/{len(both)} |")
    text+=['','同一行的两策略使用完全相同manifest；同一分配的random与ordered保留每张卡原请求集合、C/V、arrival和逐块SSU位置，只改队列顺序。不同分配版本改变了每卡配额，不能把它们之间的差异都归因于排序。三种短画像交错方案复用“提前切换”方案的随机对照，输入指纹完全相同，不计作额外独立运行。','',
      '若为五seed，使用7、19、43、67、101，数值是逐seed率的均值±样本标准差；seed7是探索，另外四seed是配方冻结后的确认，全部结果都列入，不筛选失败seed。种子改变原请求分配、队列排列及同刻提交的打破并列次序，不代表五份独立业务数据样本。v8的额外四seed在读取seed7首个结果前已声明；输入设计本身来自前期探索，不能把它当成未经挑选的业务分布。单seed行没有稳定性保证。','',
      '**和专用长短卡比较**','',
      '| 长画像 | 分配 | Baseline U% | Once U% | Baseline SLO% | Once SLO% | seeds数 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for spec in ('raw176_three_l20','raw200_three_l20'):
      for mode in ('fixed','mixed'):
        rs=[r for r in old['rows'] if r['spec_name']==spec and r['mode']==mode]
        b=[r for r in rs if r['strategy']=='baseline'];o=[r for r in rs if r['strategy']=='once']
        wanted={j['input']['seed'] for j in old_jobs.values() if j['input']['spec_name']==spec and j['input']['mode']==mode}
        if {r['seed'] for r in b}!=wanted or {r['seed'] for r in o}!=wanted:
            assert not args.require_complete
            text.append(f'| {spec} | {mode} | 运行中，暂不汇总 | 运行中，暂不汇总 | — | — | — |');continue
        text.append(f"| {spec} | {'20长卡/12短卡' if mode=='fixed' else '均匀分配后每卡随机混合'} | {avg(b,'device_utilization_percent')} | {avg(o,'device_utilization_percent')} | {avg(b,'warm_slo_percent')} | {avg(o,'warm_slo_percent')} | {len(b)} |")
    text+=['','176K各版本保留同一全局1212请求人口：420条L、S1/S2/S3各264条。200K各版本保留同一全局1172人口：380条L、三种S各264条。176K和200K之间不是同人口比较。已有五seed的160K及构造输入完整表见[原始数据与固定分卡报告](research_report.md)。','',
      '**怎样把固定分卡的阻塞搬进混合输入**','',
      '这里FIFO指每盘Path0的I/O块队列，不是32张卡共用的一个全局请求队列。其他NPU的长计算本身不会占用短卡的计算资源。主要问题是长读取反复成批到达每块SSU的Path0 FIFO，而短请求用于预取的计算时间较短，读数据不能及时就绪，下一层计算就必须停下来等。','',
      '设一波长读取在单盘前方累计工作约为W，盘速率为B，短层的可隐藏时间为C_S。W/B若大于C_S，就可能出现暴露的等待。这里“可能”很重要：谁先入队、逐块落盘和接收链路都会影响实际ready时刻，W/B不是精确stall公式。真正从日志逐层核对的是：','',
      r'$\mathrm{stall}=\max(0,\ t_{\mathrm{IO\ ready}}-t_{\mathrm{previous\ compute\ end}})$。','',
      '对于连续相同的长请求，只要下一层读取都能藏在计算中，就有下一层计算开始=上一层开始+C_L。因此两张长卡的时间差会保持，不会凭空越跑越散。v8延长短段方案的seed7 Baseline中，长内部层读取生命周期最大19.75ms，小于C_L=31.42ms；前20卡共同13条Long前缀的暖窗同层读波释放跨度约0.05ms。完全落在warm内的短内部层，按层等权计算的stall第95百分位从随机的0升到定序的9.65ms。', '',
      '固定分卡让一批长卡持续以相近相位发读；均匀随机混合常把这些相位打散。因此只匹配长短卡数量还不够，必须尽量保留长读取一起出现的时间结构。早期分组轮转与短前缀交接尝试大多仍在99%以上，全部见[未成功的轮换探索](mixed_rotation/results.md)。','',
      '后来的构造让前20卡从t=0一起执行长请求，让后12卡先执行短请求，目标是在warm内自然交换角色，是否实现逐卡核查。所有arrival仍为0；没有插入sleep、人工等待、admission限速或修改C。三张卡先转短，用来减少交换期间同时活跃的长请求数量。','',
      '**176K延长阻塞段方案的具体输入**','',
      '| 角色 | data键 | 单层C ms | 单层V MiB | 每请求8层纯计算ms |',
      '|---|---|---:|---:|---:|',
      '| S1 | 32K/1024 | 7.257232 | 42.625 | 58.057853 |',
      '| S2 | 48K/1024 | 9.941586 | 64.625 | 79.532692 |',
      '| S3 | 64K/1024 | 12.625941 | 86.625 | 101.007526 |',
      '| L | 176K/1024 | 31.416419 | 240.625 | 251.331352 |','',
      'S1/S2/S3按相对计算时长命名，模拟器原有类别实际均为SL，L属于LL。', '',
      '| NPU | 每卡总人口 | ordered顺序 |',
      '|---|---|---|',
      '| 0–8 | 15L、5S2、4S3 | 15L，然后S2/S3轮转 |',
      '| 9–16 | 15L、4S2、5S3 | 15L，然后S2/S3轮转 |',
      '| 17–19 | 13L、6S2、6S3 | 13L，然后S2/S3轮转 |',
      '| 20–31 | 10或11L、22S1、14或15S2、14或15S3 | 每卡独立打乱9S2+8S3，再22S1，再全部L，再剩余S2/S3 |','',
      '后12卡中S2只有NPU20为15条，其余14条；S3是NPU20–21为15条，其余14条；NPU20–25各11L，26–31各10L。原请求身份由seed分配，但配额不变；random只是打乱这相同的完整每卡队列。','',
      '三个纯计算定位数：17张主长卡的15L为3769.97ms，另外3张卡的13L为3267.31ms，后12卡的9S2+8S3前缀为1523.85ms。它们不等于实际结束时刻；短请求的I/O等待会把后12卡的22S1段推迟、拉长。位置及是否能在4秒前切入L必须实测，不能用纯计算估算冒充实际时序。','',
      '这个配方故意把后12卡在阻塞时段的大部分短请求集中成S1；前20卡转短后执行S2/S3。它不是自然随机业务，也不是每张卡都拥有全部三种短画像。额外的“三种短画像交错”测试属于200K提前切换方案的配对消融：把其热段改成独立打乱的11S1+2S2+2S3，专门检查短请求多样性。它不是176K本方案只改热段的单因素对照，结果单独列出。','',
      '**窗口内的数量和欠载必须看实际执行**','',
      '| 定序方案/策略 | seeds数 | 平均长卡/短卡 | 瞬时长卡min–max（跨seed） | 18–22长卡的时间占比均值 | 每卡较少一类计算的最小值ms | 全程单盘峰GiB/s |',
      '|---|---:|---|---|---:|---:|---:|']
    for spec,name in names.items():
      for policy in ('baseline','once'):
        rs=select(spec,'ordered',policy)
        if not rs:continue
        if any(r['status']!='complete' for r in data['rows'] if (r['spec_name'],r['mode'],r['strategy'])==(spec,'ordered',policy)):continue
        lm=statistics.mean(r['mean_long_cards'] for r in rs);sm=statistics.mean(r['mean_short_cards'] for r in rs)
        text.append(f"| {name}/{policy} | {len(rs)} | {lm:.2f}/{sm:.2f} | {min(r['min_long_cards'] for r in rs)}–{max(r['max_long_cards'] for r in rs)} | {statistics.mean(r['fraction_long_18_to_22'] for r in rs)*100:.2f}% | {min(min(r['min_card_short_compute_ms'],r['min_card_long_compute_ms']) for r in rs):.2f} | {max(r['max_ssu_nominal_gib_s'] for r in rs):.6f} |")
    text+=['','长短卡数按“当前已接纳哪一类请求”统计，包含该请求的I/O等待。固定对照始终20/12；混合只力求时间平均接近，不是任意时刻都相同。Once更快完成短请求，会改变长短驻留比例，不能要求同一输入在两个策略下还保持同一个角色时序。','',
      '**结论边界**','',
      '使用原始data并不保证Baseline接近满利用率；此前接近100%只是特定画像、分配和排序的结果。我们找到了固定分卡的下降，也找到了每卡在warm内都有长短计算的混合下降。与此同时，200K较晚交接那组虽然Baseline本身欠载且下降超过10个百分点，Once却超盘上限，不能把它冒充双方欠载的完整反例。','',
      '所有请求预先排队、长卡重复相同长画像、刻意形成同步读波，都是机制实验的条件。当前结果没有覆盖生产到达分布、长请求计算抖动或调度控制本身的额外开销。短计算不是高带宽的充分条件，带宽还取决于V；本组三短V/C都低于长画像。SLO集合随策略和顺序变化，并非同一批warm请求逐条配对；99%或100%的接纳后SLO也不能说明用户排队时间很好。1.5倍阈值容许相对纯计算多50%的时间，所以有I/O stall的请求仍可能达标；SLO在这里逐请求计算，没有从U反推。','',
      '直接机制证据：[200K较晚交接](mixed_rebinding_mechanism.md)、[200K提前切换](mixed_earlyguard_mechanism.md)、[176K延长阻塞段](mixed_extendedhot_mechanism.md)。这些文件给出层时间、相位和等待；不会把IO生命周期当成连续盘服务，也不会仅凭时间重叠就断言某个长请求是短请求的FIFO前驱。','',
      '**复核文件**','',
      '[seed7完整输入队列CSV](mixed_rebinding/assignments_seed7.csv)、[逐seed CSV](mixed_rebinding/per_seed.csv)、[全部结果与输入配对JSON](mixed_rebinding/results.json)、[原始输入审计](mixed_rebinding/audit.json)、[逐时角色数量](mixed_rebinding/role_counts/)、[计算与等待图](mixed_rebinding/figures/raw176_extendedhot_ordered_seed7_npu_timeline.png)、[长短卡数量图](mixed_rebinding/figures/raw176_extendedhot_ordered_seed7_role_counts.png)。冻结的spec、plan、manifest、核心源码快照、启动命令和原始层日志均随目录保存。','']
    (HERE/'mixed_research_report.md').write_text('\n'.join(text))
    sources=[HERE/'mixed_rebinding/results.json',HERE/'followup_results.json',Path(__file__)]
    (HERE/'mixed_research_report_sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps({'report':str(HERE/'mixed_research_report.md'),'completed':data['completed'],'planned':data['planned']}))

if __name__=='__main__':main()
