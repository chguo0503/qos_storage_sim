#!/usr/bin/env python3
"""Produce the Chinese report from completed, independently audited runs."""
from pathlib import Path
import argparse, csv, gzip, html, json, math, re, statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'

def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(row,key,digits=2): return f'{float(row[key]):.{digits}f}'
def prof(row,role): return f"{float(row[role+'_total_length_k']):g}K/{row[role+'_nql']}"
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def build_markdown():
    rows=read_csv('paired_summary.csv'); allruns=read_csv('all_runs.csv')
    qualified=[r for r in rows if r['series_point_complete_and_eligible']=='True']
    selected={}
    for r in qualified:
        k=(r['id'],r['mode'])
        if k not in selected or int(r['ssu'])>int(selected[k]['ssu']):selected[k]=r
    fixed_order=['D1','D2','C1','E1','F20','E2','E3','Y1','Y2','C2']
    fixed=[selected[(k,'fixed')] for k in fixed_order if (k,'fixed') in selected]
    random_order=['XY12_32','XY12_24','XY12_20','XY12_16','Y1792','Y3072','Y4096','X16']
    randoms=[selected[(k,'random')] for k in random_order if (k,'random') in selected]
    def group(k):return selected.get((k,'random'),{})
    xseries=[group(k) for k in ['XY12_32','XY12_24','XY12_20','XY12_16'] if group(k)]
    yseries=[group(k) for k in ['Y1792','XY12_20','Y3072','Y4096'] if group(k)]
    lines=['# 8 NPU：x、y、并发配比与 FIFO / Once 实验',
        '实验日期：2026-09-15。容量单位统一为十进制 GB/s；主统计窗口为仿真时间 2–4 秒。',
        '## 实验得到的结论',
        f'完成 {len(allruns)} 个原生策略运行。合格主结果包括 {len(fixed)} 组固定并发配置，以及 {len(randoms)} 组每卡随机混合配置；随机组各采用 seed=7、19、43。所有主表中的策略配对都通过完整轨迹逐盘欠载审计，2–4 秒内八张卡均有请求持续执行。',
        '固定并发下能够出现显著 FIFO 损失；每卡随机混合下，增大读取量比也能观察到利用率下降。满足单 A 阻塞不等式只表示存在这种排队机制，损失大小还取决于前方排队量、同时出现的长读取数量及计算/读取相位。',
        ('在 y≈12 的随机系列中，x 依次为 '+ '、'.join(num(r,'x') for r in xseries)+
         '，FIFO 三种子均值依次为 '+'、'.join(num(r,'baseline_fleet_mean_pct')+'%' for r in xseries)+
         '；Once 对应为 '+'、'.join(num(r,'once_fleet_mean_pct')+'%' for r in xseries)+'。') if xseries else '',
        '低于 32K 的 B 画像使用计算时间外推，尚未实测校准。原始数据或其范围内插值的结果与外推结果分别标注；这些数值是此仿真模型内的证据，不能当成实机利用率承诺。',
        '## 配置、欠载与统计口径',
        '- 8 个 NPU；每卡同时运行 1 个请求，batch size=1；每请求 8 层。所有请求 t=0 进入各卡队列，按给定顺序串行接纳。',
        '- ring hash 放置：128 token/block，完整 IO 为 176 KiB；每盘 256 个虚拟节点。request_id 与 block_index 决定放置，同一块跨层同盘；尾块保持真实大小。单盘配置也使用同一函数。',
        '- 每 SSU 为 40 GB/s，每 NPU 接收上限为 50 GB/s。原生代码内部体积单位是 GiB，调用时转换物理容量，并同比例转换静态 CIR；此前 40 GiB/s 实验不能直接与本轮数值混用。',
        '- 计算当前层时预取下一层；上一请求末层计算开始时预取下一请求 L0。两策略使用完全相同的请求、顺序、request_id、盘数与 IO 提交随机种子。',
        '- FIFO 使用原生 path0 的逐 IO FCFS；一层 A 不是原子服务单元。Once 使用原有每层每盘一次路径选择、5 ms 快照与静态 CIR 配置，未修改算法或按结果调参数。',
        '- 主利用率 = 窗口内全部计算区间交集之和 / (8 × 窗口时长)。冷启动不在主窗口内；窗口中的全部层内等待和首层交界等待均保留。',
        '- SLO×1.5 定义为：从 NPU 接纳到完成的耗时 ≤ 1.5 × 该请求八层计算总时长。排在 NPU 输入队列中的等待不在此 SLO 内；全请求与窗口接纳 cohort 的结果分别保留在 CSV。',
        'x = V_A / V_B；y = C_A / C_B。普通需求按当时已接纳、尚未完成的请求逐盘求和：D_s(t) = sum_i V_i,s / C_i。完整执行轨迹上的每个事件间隔都检查 D_s(t) < 40 GB/s。',
        '请求交界的额外 L0 预取突发不加入普通需求；没有删除整个交界时段，也没有删去它对其他内部层和利用率的影响。另检查所有 k≥1 未完成读取的 V_s/C 需求和；以整层到齐时刻结束每盘区间，是保守上界。这里的需求是为在计算窗口内掩盖读取所需的名义速率，实际物理服务速率始终受硬件容量约束。',
        '固定并发首先以 S_min = floor((n_A B_A + n_B B_B)/40) + 1 选最小整数盘数，再按真实 ring hash 与两策略完整轨迹核验。随机系列单盘使用 8×max(B_A,B_B)<40 的保证；输入请求数量比仅用于生成顺序，不用于代替某时刻的并发带宽需求。',
        '## 固定并发：直接检查当前推导',
        'NPU 0 至 n_A−1 连续执行 A，其余卡连续执行 B。主窗口内并发数严格保持表中的 n_A:n_B。每类画像固定，不加入 NQL 抖动；这是固定并发机制实验。',
        mdtable(['ID','A：长度/NQL','B：长度/NQL','n_A:n_B','S','x','y'],[
            [r['id'],prof(r,'A'),prof(r,'B'),r['n_A']+':'+r['n_B'],r['ssu'],num(r,'x'),num(r,'y')] for r in fixed]),
        'D1、D2、C2 的 A/B 均为原始 data 画像。其余含 16K、20K 或 24K 的 B 计算时间外推；非原始 NQL 同时使用插值。',
        mdtable(['ID','FIFO整机%','Once整机%','FIFO B%','Once B%','最大单盘需求 GB/s'],[
            [r['id'],num(r,'baseline_fleet_mean_pct'),num(r,'once_fleet_mean_pct'),num(r,'baseline_B_mean_pct'),num(r,'once_B_mean_pct'),num(r,'worst_full_ordinary_peak_GB_s')] for r in fixed]),
        '![固定并发策略对比](fixed_concurrent_utilization.png)',
        'Y1 与 Y2 是必要的对照：Y1 在 y<8 时仍有明显可减少的等待；Y2 也满足理想单 A 条件，但 FIFO 接近满利用率。因此不能仅凭 x、y 跨过不等式边界，就推断会产生很大的平均损失。固定并发的低结果不能直接当作每卡随机混合的结果。',
        '## 每卡随机混合：保持 y≈12，增大 x',
        '每张卡独立随机打乱 A:B=1:12 的请求数量序列。这不是同时运行的卡数比；实际 n_A(t) 由执行进度决定，已在每次运行的审计中记录。采用三个种子并报告均值和极差，没有选择最差种子作为主结果。',
        mdtable(['ID','A：长度/NQL','B：长度/NQL','S','x','y','FIFO均值%','Once均值%'],[
            [r['id'],prof(r,'A'),prof(r,'B'),r['ssu'],num(r,'x'),num(r,'y'),num(r,'baseline_fleet_mean_pct'),num(r,'once_fleet_mean_pct')] for r in xseries]),
        mdtable(['ID','FIFO极差%','Once极差%','FIFO B均值%','Once B均值%'],[
            [r['id'],num(r,'baseline_fleet_min_pct')+'–'+num(r,'baseline_fleet_max_pct'),num(r,'once_fleet_min_pct')+'–'+num(r,'once_fleet_max_pct'),num(r,'baseline_B_mean_pct'),num(r,'once_B_mean_pct')] for r in xseries]),
        '![随机x系列](random_x_utilization.png)',
        '该系列中 B 的带宽需求保持约 4.3 GB/s，A 的 NQL 用于把 y 保持在约 12。x 增大时，A 的一次读取相对 B 的掩盖窗口更长，实验中内部等待随之增多。由于真实画像与整数 NQL 的限制，这不是所有其他参数严格不变的单变量定理；各组输入长度及有限请求数量也不同。',
        '## 每卡随机混合：x≈10.5，改变 y',
        '保持 B=20K/NQL1236，改变 A 的 NQL。读取量仅有小幅变化，x 约 10.5；A 的计算窗口和读取频率变化较大。输入数量比仍为 1:12，因此实际同时运行 A 的时间占比也会变化。',
        mdtable(['ID','A NQL','x','y','FIFO整机均值%','Once整机均值%'],[
            [r['id'],r['A_nql'],num(r,'x'),num(r,'y'),num(r,'baseline_fleet_mean_pct'),num(r,'once_fleet_mean_pct')] for r in yseries]),
        '![随机y系列](random_y_utilization.png)',
        'y 增大使长请求读取出现得更稀疏。在本系列中不能把“计算时间比更大”理解为“FIFO 必然更差”；要同时看长读取频率、短卡占比及 IO 聚集。',
        '## 欠载余量、被排除的盘数和交界等待',
        '欠载与“很少接近容量”分别统计。达到 36 GB/s 记为接近容量。所有主结果均不超 40；其中有些配置会经常接近 40。若继续采用“接近容量时间不超过 5%”的偏好，应优先查看表中接近时间占比低的组。',
    ]
    nearrows=[]
    for r in fixed+randoms:
        rr=[a for a in allruns if a['id']==r['id'] and a['mode']==r['mode'] and a['ssu']==r['ssu']]
        nearrows.append([r['id'],r['mode'],num(r,'worst_full_ordinary_peak_GB_s'),f"{max(float(a['full_ordinary_demand_any_ge36_pct']) for a in rr):.2f}%"])
    lines.append(mdtable(['ID','模式','最大单盘需求 GB/s','≥36时间占比上界'],nearrows))
    if group('X16'):
        r=group('X16');lines.extend([
            f"更宽松的随机对照 X16：A={prof(r,'A')}，B={prof(r,'B')}，x={num(r,'x')}，y={num(r,'y')}，S=1。任意八卡组合需求不超过 34.41 GB/s；FIFO 三种子均值 {num(r,'baseline_fleet_mean_pct')}%，Once 为 {num(r,'once_fleet_mean_pct')}%。这组全程不进入 36 GB/s 区域，仍有明确调度差异。"
        ])
    lines.extend([
        'E2 的 S=2 配对被排除：FIFO 的普通单盘峰值约39.98 GB/s，但 Once 达到40.28 GB/s；合格版增至 S=3。C2 的 S=3 配对也被排除：FIFO 40.57、Once 40.17 GB/s；合格版增至 S=4。两种策略改变进度，所处 request_id 的放置组合会不同，因此两条轨迹都须通过审计。被排除结果完整保留在 CSV 和原始数据包中。',
        '![完整双盘名义需求](C1_full_nominal_demand.png)',
        '![内部与首层等待分解](stall_decomposition.png)',
        '交界读取允许产生带宽突发，但等待仍计入利用率。对跨类型 L0 另计算物理必要下界：V 以 GB 计、时间以 ms 计，T_min = 1000 × max(max_s(V_s/40), V_total/50)，W = admission − io_start，最低首层等待 = max(T_min − W, 0)。它只是下界；超出下界的部分不能自动全部归因于调度。Once 剩余等待中仍可能包含内部排队与交界排队。逐运行结果见 boundary_bounds.csv。',
        '## 如何结合理论解释结果',
        '在理想共享总带宽池中，单 A 完整排在 B 前、且 B 尚有完整 C_B 窗口时，联合条件为：n_A B_A+n_B B_B < 40S < V_A/C_B。消去连续带宽得到 n_A/y+n_B/x<1；固定整数 S 仍须回到带宽区间检查。',
        '实际多盘读取还受 ring hash 不均衡及每卡 50 GB/s 接收上限影响。C1 的 V_A/(40S C_B)≈0.849，但 V_A/(50 C_B)≈1.359，所以它不能作为“A一定能在 C_B 内传完”的对照。接收端让 A 变慢，也不单独证明 B 必受阻：需要真实 FIFO 队列与等待证据。',
        '原生 FIFO 按小块 IO 排序，多个卡同时提交时会交织；只有位于 B 前方的未完成数据真正构成它的等待。当前结果同时包含“满足条件但几乎不降”的 Y2，以及“短卡明显等待但全系统损失被长卡计算稀释”的 D1/D2。x、y、并发配比和盘数是筛选条件，队列相位与读取聚集决定实际损失。',
        '## 数据来源、复现与检查',
        'data 共84条画像，长度32K–200K。C 在原始长度和 NQL 网格内插值；16K/20K/24K 按32K与48K在相同 NQL 下线性外推。16K的长度权重为2、−1；20K为1.75、−0.75；24K为1.5、−0.5。每个外推/插值画像的锚点与权重均保留在 config 中。',
        'V 始终按 (总 token 数 − NQL) × 1408 byte 计算，未用计算时间缩放伪造新画像，未增加读取 padding。NQL 重复是此实验的固定画像设计；卡内请求仍使用全局唯一 request_id。',
        '数据包保留未修改的原生源码、data、所有输入配置、逐请求盘体积、每层 IO/计算时间戳、策略不变量、输入指纹、完整逐盘欠载审计、三种子结果及复现脚本。另用0.5–4.5秒窗口进行同轨迹敏感性检查，见window_sensitivity.csv；该检查不重新选输入。',
        '执行单组：python run_experiment.py configs/Y1_fixed_s1_seed7.json --strategy baseline --force。将策略改为once即可配对重跑。完整批次命令和文件说明见README.md；绘图用build_plots.py，理论核对用summarize_theory.py，交界下界用boundary_bounds.py。',
        '验证范围是这些完整有限输入与指定种子。固定并发组只采用seed7；随机组使用三个种子，极差不是统计置信区间。结果支持这些配置下的行为趋势，不构成任意随机输入下利用率的数学上界。',
    ])
    path=OUT/'xy_underload_experiment_report.md';path.write_text('\n\n'.join(x for x in lines if x)+'\n')
    return path

def render_pdf(mdpath):
    import fitz
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    tmp=ROOT/'tmp';tmp.mkdir(exist_ok=True);font=tmp/'DroidSansFallback.ttf'
    if not font.exists():font.write_bytes(fitz.Font('china-s').buffer)
    pdfmetrics.registerFont(TTFont('CJK',str(font)))
    pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
    ink=colors.HexColor('#183044');blue=colors.HexColor('#175873');muted=colors.HexColor('#587080')
    styles={
        'body':ParagraphStyle('body',fontName='CJK',fontSize=9.6,leading=15,wordWrap='CJK',textColor=ink,spaceAfter=7,allowWidows=0,allowOrphans=0),
        'h1':ParagraphStyle('h1',fontName='CJK',fontSize=22,leading=30,textColor=blue,spaceAfter=13),
        'h2':ParagraphStyle('h2',fontName='CJK',fontSize=14.5,leading=22,textColor=blue,spaceBefore=16,spaceAfter=9,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8,leading=11.5,wordWrap='CJK',textColor=ink),
        'head':ParagraphStyle('head',fontName='CJK',fontSize=8,leading=11.5,wordWrap='CJK',textColor=colors.white),
    }
    width=A4[0]-80;story=[]
    def p(t,style='body'):return Paragraph(html.escape(t).replace('\n','<br/>'),styles[style])
    blocks=mdpath.read_text().replace('–','-').replace('−','-').split('\n\n')
    for block in blocks:
        block=block.strip()
        if not block:continue
        if block.startswith('# '):story.append(p(block[2:],'h1'))
        elif block.startswith('## '):story.append(p(block[3:],'h2'))
        elif block.startswith('| '):
            data=[[v.strip() for v in line.strip('|').split('|')] for line in block.splitlines() if not re.match(r'^\|[ \-:|]+\|$',line)]
            n=len(data[0]); widths=[width/n]*n
            if n==7:widths=[width*v for v in [.10,.24,.24,.11,.07,.12,.12]]
            if n==8:widths=[width*v for v in [.13,.20,.20,.05,.09,.09,.12,.12]]
            if n==4:widths=[width*v for v in [.24,.18,.29,.29]]
            if n==6:widths=[width*v for v in [.15,.16,.16,.16,.16,.21]]
            table=Table([[p(v,'head' if i==0 else 'cell') for v in row] for i,row in enumerate(data)],colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),blue),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F6F9'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,blue)]))
            story.extend([KeepTogether([table]) if len(data)<=6 else table,Spacer(1,9)])
        elif block.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',block);img=OUT/match.group(2)
            if img.exists():
                from PIL import Image as PILImage
                w,h=PILImage.open(img).size;ih=min(width*h/w,510);iw=ih*w/h
                story.extend([Image(str(img),width=iw,height=ih),Spacer(1,6)])
        else:story.append(p(block))
    dest=OUT/'xy_underload_experiment_report.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D8E2E8'));canvas.line(40,36,A4[0]-40,36)
        canvas.setFont('CJK',8);canvas.setFillColor(muted);canvas.drawString(40,23,'8 NPU | ring hash | 40 GB/s per SSU');canvas.drawRightString(A4[0]-40,23,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=36,bottomMargin=48,title='8 NPU x-y 欠载与FIFO/Once实验',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    print(dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pdf',action='store_true');args=ap.parse_args()
    path=build_markdown();print(path)
    if args.pdf:render_pdf(path)
