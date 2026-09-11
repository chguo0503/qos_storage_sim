#!/usr/bin/env python3
"""Create a Chinese, vector-based explanatory PDF; never runs the simulator."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'prefill_miss_open_closed_explained_v2.pdf'
PAGE_COUNT = 11
DATA = ast.literal_eval((ROOT / 'data').read_text())
COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
GH = f'https://github.com/chguo0503/qos_storage_sim/blob/{COMMIT}/'
W, H = A4
M, CW = 42, W - 84
INK, MUTED = '#172A3A', '#576B7D'
BLUE, GREEN, ORANGE, PURPLE = '#2675C9', '#14816C', '#DA702A', '#7654AB'
PALE, LINE = '#F2F6FA', '#D7E1EA'
pdfmetrics.registerFont(TTFont('CN', '/home/chguo/.fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB', '/home/chguo/.fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNB', italic='CN', boldItalic='CNB')
c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
c.setTitle('看懂 prefill miss、开环闭环与 NPU 利用率（修订版）')
c.setAuthor('qos_storage_sim 项目分析')
c.setSubject('面向初学者的图解；真实 data 数值、源码事实与教学算例分开说明')
page = 0
text_log = []
layout = []


def plain(s):
    return re.sub('<[^>]+>', '', s).replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')


def para(s, x, top, width, size=12.2, color=INK, bold=False, max_h=None, align=0):
    st = ParagraphStyle('p', fontName='CNB' if bold else 'CN', fontSize=size,
                        leading=size * 1.68, textColor=HexColor(color), wordWrap='CJK', alignment=align)
    p = Paragraph(s, st)
    aw, ah = p.wrap(width, H)
    if max_h is not None:
        assert ah <= max_h + .1, (page, plain(s)[:50], ah, max_h)
    assert top + ah <= H - 39, (page, plain(s)[:50], top + ah)
    p.drawOn(c, x, H - top - ah)
    text_log.append(plain(s))
    layout.append({'page':page, 'text':plain(s), 'box':[x,top,width,ah]})
    return ah


def txt(s, x, top, size=11, color=INK, bold=False, align='left'):
    c.setFont('CNB' if bold else 'CN', size)
    c.setFillColor(HexColor(color))
    fn = {'left':c.drawString, 'center':c.drawCentredString, 'right':c.drawRightString}[align]
    fn(x, H - top - size, s)
    text_log.append(s)
    tw = pdfmetrics.stringWidth(s, 'CNB' if bold else 'CN', size)
    left = x if align == 'left' else x-tw if align == 'right' else x-tw/2
    assert left >= 12 and left+tw <= W-12, (page,s,left,tw)
    layout.append({'page':page,'text':s,'box':[left,top,tw,size*1.2]})


def rect(x, top, width, height, fill, radius=8, stroke=None):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke or fill))
    c.roundRect(x, H-top-height, width, height, radius, fill=1, stroke=int(stroke is not None))


def line(x1, y1, x2, y2, color=LINE, dash=None, weight=1):
    c.saveState()
    c.setStrokeColor(HexColor(color)); c.setLineWidth(weight)
    if dash: c.setDash(*dash)
    c.line(x1,H-y1,x2,H-y2)
    c.restoreState()


def arrow(x1,y1,x2,y2,color=MUTED):
    import math
    line(x1,y1,x2,y2,color,weight=1.5)
    a=math.atan2(y2-y1,x2-x1)
    for d in [-.48,.48]:
        line(x2,y2,x2-7*math.cos(a+d),y2-7*math.sin(a+d),color,weight=1.5)


def box(s, top, height, fill=PALE, color=INK, size=12.2, x=M, width=CW):
    rect(x,top,width,height,fill)
    para(s,x+15,top+11,width-30,size,color,max_h=height-18)


def start(title, subtitle, tag):
    global page
    if page: c.showPage()
    page+=1
    c.bookmarkPage(f'p{page}'); c.addOutlineEntry(title,f'p{page}',0)
    text_log.extend(['',f'## {page}. {title}',''])
    rect(0,0,W,7,BLUE,0)
    txt('PREFILL · KV CACHE · NPU',M,24,9.4,MUTED)
    txt(tag,W-M,24,9.4,BLUE,align='right')
    para(title,M,56,CW,23.5,bold=True,max_h=45)
    para(subtitle,M,105,CW,11.3,MUTED,max_h=45)
    line(M,H-45,W-M,H-45)
    txt('qos_storage_sim  |  图解入门 · 修订版 v2  |  2026-09-10',M,H-33,8.5,MUTED)
    txt(f'{page} / {PAGE_COUNT}',W-M,H-33,9,MUTED,align='right')


def source(s, top=755):
    para(s,M,top,CW,8.4,MUTED,max_h=36)


def table(headers, rows, top, widths, fontsize=10.4, rowh=42):
    st = ParagraphStyle('t',fontName='CN',fontSize=fontsize,leading=fontsize*1.45,
                        textColor=HexColor(INK),wordWrap='CJK')
    sh = ParagraphStyle('th',parent=st,fontName='CNB',textColor=white)
    cells=[[Paragraph(v,sh) for v in headers]]+[[Paragraph(str(v),st) for v in r] for r in rows]
    tb=Table(cells,colWidths=widths,rowHeights=[44]+[rowh]*len(rows))
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(INK)),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor(PALE),white]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),9),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,HexColor(LINE))]))
    aw,ah=tb.wrap(CW,H)
    assert top+ah < H-50
    tb.drawOn(c,M,H-top-ah)
    text_log.append(' | '.join(headers))
    text_log.extend(' | '.join(map(str,r)) for r in rows)
    for row in cells:
        for cell,width in zip(row,widths):
            _,hh=cell.wrap(width-16,H)
            assert hh <= rowh-4
    return ah


def timeline_axis(top, xmax, ticks, x=125, width=420, unit='ms'):
    line(x,top,x+width,top,MUTED)
    for t in ticks:
        xx=x+t/xmax*width
        line(xx,top-4,xx,top+4,MUTED)
        txt(str(t),xx,top+7,9.2,MUTED,align='center')
    txt(unit,W-M+5,top+8,8.5,MUTED)
    return lambda t: x+t/xmax*width


def bar(scale,a,b,top,color,label='',size=10,label_color='#FFFFFF'):
    rect(scale(a),top,scale(b)-scale(a),24,color,3)
    if label: txt(label,(scale(a)+scale(b))/2,top+4,size,label_color,align='center')


# 1: cache hit and miss, clearly separating SSD reuse from recomputation.
start('先看懂：哪些 token 要重新计算？',
      '从一个 10,000 token 的 prompt 开始。把 token 理解为模型处理文本的小单位即可。',
      '01 / 基础概念')
box('<b>你说的 miss：</b>prefill 阶段，无法复用已有 KV、需要重新计算的 token 数量或比例。'
    '这里按“可复用前缀 + 新计算后缀”理解。[1][2]',160,78)
txt('教学例子：总长 10,000 token，miss 比例 20%',M,259,14,bold=True)
rect(M,297,CW*.8,58,BLUE,0)
rect(M+CW*.8,297,CW*.2,58,GREEN,0)
txt('命中前缀：8,000 token',M+CW*.4,314,13,'#FFFFFF',align='center')
txt('miss：2,000',M+CW*.9,314,11,'#FFFFFF',align='center')
arrow(M+CW*.4,364,M+CW*.4,391,BLUE)
arrow(M+CW*.9,364,M+CW*.9,391,GREEN)
para('<b>读取已有 KV</b><br/>在本实验模型中，命中部分存于 SSU，需要搬入 NPU。',M+10,404,286,12.5,BLUE)
para('<b>重新计算</b><br/>对应 NQL。',M+CW*.73,404,CW*.27,12.5,GREEN)
box('<b>命中缓存，也可能需要读盘。</b>命中表示“有可复用结果”；结果是否已经在 NPU 内存里，是另一个问题。',497,76)
para('<b>KV 是可复用的中间计算结果。</b>NPU 负责计算；SSU 是存放 KV 的存储设备；I/O 是搬运数据。'
     'prefill 是处理输入、为生成答案做准备的阶段。',M,593,CW,12.2,max_h=66)
para('后面都用：L = 总 token 数；p = miss 比例；M = miss token 数。<br/>'
     '<b>M = p × L = NQL</b>。例如 p = 20% = 0.2。',M,672,CW,12.2,max_h=56)
source('边界：若命中 KV 已驻留 NPU 内存，就不需要全部从 SSU 读取；本图不能直接套用。来源 [1][2] 见第 11 页。')


# 2: exact raw data, recompute all displayed rates rather than assume the field.
start('为什么 miss 越少，搬运反而越吃紧？',
      '比较同样长度的输入：少算了很多，仍要搬运大部分已有 KV。',
      '02 / 真实 data + 简化公式')
box('固定总长和执行配置时，可先近似理解为：<br/>'
    '<b>计算时间 C ≈ 每 token 计算成本 × miss token 数</b><br/>'
    '<b>读取量 D = 每 token 的 KV 大小 × 命中 token 数</b>',156,101)
para('这里 <b>p 是 miss 比例</b>，20% 写成 0.2；D 表示读取的数据量。'
     '要在计算期间读完，需要的平均速率约为 <b>B = D / C</b>。'
     '每 token 计算成本固定时，<b>B ∝ (1 − p) / p</b>：miss 越少，预取时间越短。',M,275,CW,12.2,max_h=66)
rows=[]
audit_rows=[]
for q in [128,512,2048,4096]:
    b,us,_,v=DATA[(32,q)]
    measured=v/(us/1e6)
    assert abs(measured-b)<1e-10
    rows.append([str(q),f'{100*q/32768:.4f}%',f'{us/1000:.3f}',f'{v*1024:.3f}',f'{measured:.3f}'])
    audit_rows.append(dict(L=32768,M=q,p=q/32768,C_ms=us/1000,V_MiB=v*1024,B_GiB_s=measured))
txt('当前 data：总长均为 32K = 32,768 token',M,365,13.4,bold=True)
table(['miss 数\n(NQL)','miss 比例','计算 C\n(ms / 层)','读取 D\n(MiB / 层)','需求 D/C\n(GiB/s)'],rows,397,[94,94,108,108,CW-404],rowh=39)
box('<b>看第一行与第三行：</b>计算从 14.369 ms 缩到 1.178 ms；读取量仍为约 41～44 MiB。'
    '因此需求速率从 2.803 升到 36.333 GiB/s。',618,76,fill='#EAF3FD')
para('“正比”是近似，不是 data 的硬规则：miss 数翻 4 倍时，128→512 的计算时间只变为约 3.14 倍。'
     '上下文长度、固定开销、批处理与执行效率也会影响计算成本。',M,707,CW,10.6,max_h=38)
source('D 取自 Data（数据）；旧版用 V 表示 Volume（数据量），并非 KV 中的 Value。数值来自 data，C 与 D 分别输入；1 GiB = 1,024 MiB。来源 [2][3]。')


# 3: concrete external-arrival example; do not conflate send and execute.
start('开环与闭环：前面慢了，还继续发吗？',
      '这一页只讨论：外部客户端把完整请求送进 NPU 等待队列。先不考虑 I/O 和预取。',
      '03 / 先看外部发送')
box('有 A、B、C 三个请求，原本每个处理 10 ms。现在 <b>A 变慢，需要 30 ms</b>，'
    'B 和 C 仍各需 10 ms。<b>“发送”是进入队列，不是开始执行。</b>',154,77)
txt('开环：按闹钟，每隔 10 ms 发一个',M,256,14,bold=True)
s=timeline_axis(341,50,[0,10,20,30,40,50])
txt('发送时刻',M,315,10.5,MUTED)
for t,label in [(0,'A'),(10,'B'),(20,'C')]:
    txt(label,s(t),291,12,BLUE,bold=True,align='center')
    arrow(s(t),313,s(t),334,BLUE)
para('B 在 10 ms 到达，C 在 20 ms 到达。A 还没做完，B、C 就先排队。'
     '<b>A 变慢，不改变后面的发送时间。</b>',M,376,CW,11.8,max_h=42)
txt('闭环：这个做完，再发下一个',M,445,14,bold=True)
s=timeline_axis(530,50,[0,10,20,30,40,50])
txt('发送时刻',M,504,10.5,MUTED)
for t,label in [(0,'A'),(30,'B'),(40,'C')]:
    txt(label,s(t),480,12,BLUE,bold=True,align='center')
    arrow(s(t),502,s(t),523,BLUE)
para('等 A 在 30 ms 完成才发 B；等 B 在 40 ms 完成才发 C。'
     '<b>前面处理变慢，后面的发送跟着推迟。</b>',M,565,CW,11.8,max_h=42)
box('<b>“有限”只表示清单会用完。</b><br/>'
    '开环有限：按时间表发完这三个，再等它们结束。<br/>'
    '闭环有限：做完一个发一个，第三个完成后结束。',642,95)
source('此处用“一个客户端、每次只等待一条未完成请求”说明闭环。闭环也可以有多个客户端并发；它不意味着全系统一次只能执行一个请求。[4]')


# 4: identify the actual boundary before labelling anything open/closed.
start('先分清：到底是谁在发什么？',
      '完整请求进入等待队列，与 NPU 为某一层发起读取，是两个不同位置的事件。',
      '04 / 当前仿真的两个位置')
box('<b>位置一：外部输入 → 每张 NPU 的等待队列</b><br/>'
    '之前的实验先生成有限清单，所有请求在 t = 0 到达。'
    '每张卡已有自己的输入顺序，后面变慢不会改写这个到达时间。',155,100,fill='#EAF3FD')
txt('执行与排队：',M,278,12,bold=True)
for i,(label,fill) in enumerate([('当前请求',GREEN),('下一请求',BLUE),('后续请求',MUTED)]):
    x=M+111+i*126
    rect(x,272,113,35,fill)
    txt(label,x+56.5,280,11,'#FFFFFF',align='center')
para('“接纳”表示从等待队列取出请求，开始占用本卡的执行位置。'
     '本组实验每卡同时计算一条请求；完整请求有多层，不会在到达时发完所有层的 I/O。',M,327,CW,11.8,max_h=61)
box('<b>位置二：活跃请求 → 各层读取与计算</b><br/>'
    '本层数据到齐，才能开始计算；<b>本层实际开算时，激活下一层预取。</b>'
    '因此一次读取等待，会推迟后续计算和后续读取。',419,102,fill='#EAF6F1')
rect(M,546,147,45,ORANGE); rect(M+177,546,147,45,GREEN); rect(M+354,546,157,45,BLUE)
txt('本层数据迟到',M+73.5,558,11,'#FFFFFF',align='center')
txt('本层计算推迟',M+250.5,558,11,'#FFFFFF',align='center')
txt('下一层预取推迟',M+432.5,558,11,'#FFFFFF',align='center')
arrow(M+150,568,M+174,568); arrow(M+327,568,M+351,568)
para('后面把这条因果关系称为<b>“内部计算—预取反馈”</b>。'
     '它解释了为什么有人把内部推进称为“闭环”，但不能据此认定外部也在等待完成后才发请求。',M,615,CW,12,max_h=63)
box('<b>Baseline 和 Once per layer 共用这套逐层推进机制。</b>'
    '不能用“一个开环、一个闭环”来解释两种策略的利用率差异。',697,56,size=11)
source('源码事实：[3]。外部到达、每卡接纳与逐层预取分别见 6672、3156、3269 行。跨请求首层预取也受“下一请求已到达”限制。',764)


# 5: the pipeline, not serial compute + IO.
start('NPU 为什么会停下来等？',
      '计算当前层时，提前读取下一层；下一层的数据到齐，才可以继续算。',
      '05 / 利用率从哪里来')
box('<b>预取</b> = 提前搬下一层的数据。<b>I/O stall</b> = 算完了，但下一层数据还没到齐，只能等。',158,61)
txt('教学算例：每层计算 C = 2 ms；忽略启动与结束',M,238,13,bold=True)
txt('A  读取只要 1 ms：全部藏在计算里',M,280,12.6,GREEN,bold=True)
s=timeline_axis(395,6,[0,1,2,3,4,5,6])
txt('NPU 计算',M,316,10.5,MUTED); txt('下一层读',M,354,10.5,MUTED)
for a,b in [(0,2),(2,4),(4,6)]: bar(s,a,b,312,GREEN,'计算 2 ms')
for a,b in [(0,1),(2,3),(4,5)]: bar(s,a,b,350,BLUE,'读取',9.2)
txt('B  读取要 3 ms：每轮会露出 1 ms 等待',M,443,12.6,ORANGE,bold=True)
s=timeline_axis(558,6,[0,1,2,3,4,5,6])
txt('NPU 状态',M,479,10.5,MUTED); txt('下一层读',M,517,10.5,MUTED)
for a,b in [(0,2),(3,5)]: bar(s,a,b,475,GREEN,'计算 2 ms')
for a,b in [(2,3),(5,6)]: bar(s,a,b,475,ORANGE,'等 1 ms',9)
for a,b in [(0,3),(3,6)]: bar(s,a,b,513,BLUE,'读取 3 ms')
box('<b>NPU 利用率 = 正在计算的时间 ÷ 观察时间</b><br/>'
    'A：6 / 6 = 100%　　B：4 / 6 ≈ 66.7%',605,70,fill='#EAF3FD')
para('相同 C 和 D、稳定服务速率 b、一层预取的理想情况下：<br/>'
     '<b>U ≈ C / max(C, D/b)</b>。U 是利用率；max 表示取两者较大值。',M,693,CW,11.4,max_h=44)
source('当前代码在本层开算时激活下一层读取；最后一层可预取已到达的下一请求首层。跨请求的预取预算来自前请求末层。来源 [3]。')


# 6: external-arrival invariance is conditional, not a claim of policy equivalence.
start('外部开闭环，一定会改变利用率吗？',
      '不一定。真正进入利用率公式的是计算时间；到达规则只有改变了执行或等待，才会影响它。',
      '06 / 与当前 NPU 利用率的关系')
box('<b>当前统计：32 张卡 × 2 秒 = 64 卡秒。</b><br/>'
    '平均利用率 = 这 64 卡秒里，真正用于计算的时间 / 64 卡秒。'
    '若整窗只有计算与 I/O 等待，也可写成 1 − 总等待 / 64 卡秒。',155,101)
txt('教学例子：实际执行相同，就能得到相同利用率',M,280,13,bold=True)
s=timeline_axis(410,64,[0,16,32,48,64],x=187,width=358,unit='卡秒')
txt('一次全部放好',M,329,11,MUTED)
txt('过程中及时补足',M,373,11,MUTED)
for top in [325,369]:
    bar(s,0,57.6,top,GREEN,'计算：57.6 卡秒',11)
    bar(s,57.6,64,top,ORANGE)
txt('橙色均为 6.4 卡秒等待；两者利用率均为 90%',M,451,11.9,bold=True)
para('<b>这个等价结论有条件：</b>同一策略、相同初始执行状态；每卡任务和顺序相同；整个窗口都有任务；'
     '每次预取时下一请求已到；到达事件本身没有额外改变调度决策。'
     '此时外部到达方式可以不改变计算和等待。',M,492,CW,12,max_h=81)
box('<b>但严格“做完才发下一请求”可能打破这些条件。</b><br/>'
    '当前模型会在前请求末层开算时，预取已到达的下一请求首层。'
    '若等前请求完成才发送，就失去了这段重叠机会。',598,103,fill='#FFF1E7')
para('因此，“队列里一直有活”还不够；也要保证<b>该预取时，下一个活已经可见</b>。',M,722,CW,11.3,max_h=22)
source('时间条为汇总卡时间的教学示意，不是单卡的 64 秒轨迹。只讨论利用率等价；不同到达时间仍可能改变从到达开始计时的延迟。源码 [3]。')


# 7: make cross-NPU relative phase explicit, with one-layer lookahead preserved.
start('自然错开：发生在不同 NPU 之间',
      '改变的是并发请求各层读取的相对时刻；没有重新排列各卡的输入队列。',
      '07 / 内部反馈怎样影响后续等待')
box('教学例子：两卡原本都预计在 <b>1 ms 开始第 3 层计算</b>，同时发起第 4 层读取。'
    '现在 NPU 0 的第 3 层数据晚到；NPU 1 的进度没有改变。',155,78)
txt('原计划：第 4 层读取同时发起',M,254,13,bold=True)
s=timeline_axis(366,4,[0,1,2,3,4])
txt('NPU 0',M,299,11,MUTED); txt('NPU 1',M,336,11,MUTED)
for top in [289,326]:
    arrow(s(1),top,s(1),top+22,BLUE)
txt('蓝箭头 = 发起第 4 层读取',s(1)+14,304,10,BLUE)
txt('实际：一张卡等了 2 ms，后续读取也移到 3 ms',M,410,12.5,bold=True)
s=timeline_axis(543,4,[0,1,2,3,4])
txt('NPU 0',M,454,11,MUTED); txt('NPU 1',M,504,11,MUTED)
bar(s,1,3,450,ORANGE,'等第 3 层数据：2 ms',10)
bar(s,3,4,450,GREEN,'算第 3 层',9.2)
bar(s,1,2,500,GREEN,'算第 3 层',9.2)
arrow(s(3),428,s(3),447,BLUE)
arrow(s(1),478,s(1),497,BLUE)
box('<b>两张卡的第 4 层读取，现在分别在 3 ms 与 1 ms 发起。</b>'
    '单卡的等待改变了跨卡的相对时刻；计算时间不同、后续任务变化也会造成偏移。',586,79,fill='#EAF3FD')
para('<b>错开不保证消除等待。</b>不同时间入队的短读，仍可能遇到同一批未完成的长读取。'
     '后面是否再次聚集，需要继续推演。若仍把 NPU 0 的后续读取固定在 1 ms，分析的就不是当前反馈机制。',M,684,CW,11.7,max_h=59)
source('只画第 3 层计算和第 4 层读取的发起时刻，后续层省略。两卡不必处于相同层号才会竞争。当前计算起点触发后续预取：[3]。')


# 8: a fully specified toy FIFO example, with nominal underload stated precisely.
start('一条 FIFO 队列，短读取怎样被挡住？',
      '教学算例：只看 1 块 SSU 和 2 张 NPU，盘速率 40 GiB/s；忽略接收链路。',
      '08 / 单次阻塞，不是实验结果')
table(['下一层读取','读取量 D','当前层计算 C','名义需求 D/C'],[
    ['长批读取','0.200 GiB','100 ms','2 GiB/s'],
    ['短批读取','0.004 GiB','1 ms','4 GiB/s'],
],154,[130,118,126,CW-374],rowh=40)
para('t = 0：长卡开算，并已把下一层的长批数据排进 FIFO。<br/>'
     't = 0.2 ms：短卡开算并发出下一层读取；它希望在 <b>1.2 ms</b> 前拿到数据。',M,298,CW,11.7,max_h=57)
s=timeline_axis(481,6,[0,1,2,3,4,5,6])
txt('SSU 服务',M,391,10.5,MUTED)
bar(s,0,5,387,PURPLE,'前面长批的连续服务：5 ms',11)
bar(s,5,5.1,387,BLUE)
line(s(1.2),368,s(1.2),471,ORANGE,dash=[3,2])
txt('1.2：短卡要用数据',s(1.2)+5,365,9.1,ORANGE)
txt('短卡状态',M,443,10.5,MUTED)
bar(s,.2,1.2,439,GREEN,'算 1 ms',8.8)
bar(s,1.2,5.1,439,ORANGE,'I/O stall：3.9 ms',11)
arrow(s(5.1),416,s(5.1),435,BLUE)
txt('5.1：短读完成',s(5.1),514,9.6,BLUE,align='center')
box('<b>短读自身只需 0.1 ms，却等到 5.1 ms 才完成。</b><br/>'
    '直接挡住它的是前面的读取工作；其他卡的长计算本身不占用这张短卡。',555,77,fill='#FFF1E7')
para('两卡名义需求合计 <b>2 + 4 = 6 GiB/s &lt; 40 GiB/s</b>，仍可能发生这次等待。'
     '原因是数据成批排队，而短卡要求很快到齐。<b>D/C 欠载，不等于禁止瞬时突发。</b>',M,651,CW,11.8,max_h=64)
source('假设：长批已先入队，短读不能越过它。图中长条代表许多小 I/O 的累计服务，不是一条不可中断 5 ms 的命令。本例不能证明 32 卡长期下降 10%。')


# 9: distinguish all three bandwidth quantities and underload certification.
start('看带宽时，要分清三个数字',
      '想验证“欠载下 FIFO 很差”，需要证明输入有余量，同时追踪数据是否按时到齐。',
      '09 / 回到 32 NPU、6 SSU')
table(['数字','它回答什么问题？'],[
    ['容量：每盘 40 GiB/s','这块盘最多能搬多快？6 块合计 240 GiB/s。'],
    ['名义需求：D/C','如果希望读数藏在计算里，这个请求需要多快？'],
    ['实际吞吐：已搬字节 / 时间','在这段时间里，盘实际上搬了多少数据？'],
],155,[176,CW-176],rowh=50)
box('<b>实际吞吐低，不能独自证明输入欠载。</b><br/>'
    'NPU 等待 → 后面的计算推迟 → 后续读取也推迟 → 盘收到的工作可能减少。'
    '这里说的是内部计算—预取反馈，与外部是否按闹钟发送是两回事。',374,101,fill='#FFF1E7')
para('<b>逐盘检查：</b>此前名义口径是对每块盘，累加<b>当前已接纳请求</b>的 D_s/C'
     '（D_s 是该盘读取量），而非只与总容量 240 比较。'
     '跨请求首层预取的额外释放及前请求末层预算应另列；名义欠载不能保证每次读取都按时完成。',M,497,CW,12,max_h=83)
para('<b>一个反例：</b>第 2 页的 32K / NQL=128，每卡需求约 36.333 GiB/s。'
     '若 32 张卡同时运行这种画像，合计约 <b>1,162.65 GiB/s</b>，已远超 240。'
     '这种情况下利用率低，不能作为“欠载 FIFO 缺陷”的证据。',M,601,CW,12,max_h=83)
box('因此，要同时看 <b>D（读取批量）、C（可等多久）、逐盘分布、请求顺序和释放时刻</b>。'
    '只说“有长流和短流”，条件还不够。',697,56,size=10.9)
source('容量取自此前 32 NPU / 6 SSU 实验配置；每卡接收链路为 50 GiB/s，也会影响到齐时间。数据算术见 [2]；闭环行为见 [3][4]。',761)


# 10: finite windows and the utilization denominator.
start('有限请求：窗口选法也会改变答案',
      'warm 是先让系统运行一段时间再统计；它本身不保证已经进入稳定状态。',
      '10 / 利用率的分子与分母')
box('之前的主统计窗口是 <b>[2 s, 4 s)</b>，共 2 秒。<br/>'
    '<b>32 卡平均利用率 = 窗口内 32 卡计算时间之和 / 64 卡秒</b><br/>'
    '每张卡都有任务时，剩下的时间主要体现等待；队列耗尽后，还会混入无任务空闲。',155,103)
txt('教学例子：同一张卡、同一批任务，总计算都是 100 ms',M,284,12.3,bold=True)
s=timeline_axis(415,200,[0,50,100,150,200])
txt('策略 A',M,330,11,MUTED); txt('策略 B',M,373,11,MUTED)
bar(s,0,40,326,GREEN,'计算',9)
bar(s,40,60,326,ORANGE,'等',9)
bar(s,60,120,326,GREEN,'计算',9)
bar(s,120,200,326,'#DCE4EB','已做完，无任务',9,label_color=MUTED)
bar(s,0,40,369,GREEN,'计算',9)
bar(s,40,140,369,ORANGE,'等待',10)
bar(s,140,200,369,GREEN,'计算',9)
para('<b>都统计 0～200 ms：</b>A 与 B 都是 100 / 200 = 50%。<br/>'
     '<b>但完成时间不同：</b>A 在 120 ms 做完，B 在 200 ms 做完。'
     '按各自完成时间算，分别是 83.3% 和 50%。',M,462,CW,12,max_h=68)
para('所以，扩大窗口时要同时报告：每卡是否还有任务、I/O 等待多少、总完成时间。'
     '不同卡的层读取时刻也可能逐渐分散；一次突发造成等待，不代表以后永远这样。',M,551,CW,12,max_h=66)
box('<b>miss 改变后，也不要只比利用率。</b>命中更多，真正需要的计算减少，即便利用率变低，'
    '请求仍可能更早完成。判断策略时应固定输入，并一起看请求延迟、SLO 达标率与吞吐。',636,99,fill='#EAF3FD')
source('本页时间线为教学算例。之前的 TTFT SLO 口径是“接纳后到完成”，不包含接纳前排队；不能直接等同于用户从到达开始等待首 token 的时间。')


# 11: concise takeaways + traceable sources.
start('读分析时，用这张对照表',
      '先核实假设，再看公式；单次现象、长期结论与真实实验结果需要分开。',
      '11 / 术语速查与来源')
table(['看到的说法','应该怎样理解'],[
    ['miss 长度 / miss 比例','长度 M = pL；同为 10% miss，总长不同，计算量也不同。'],
    ['计算正比 miss 长度','固定条件下的近似；实际 C 查 data，不强行套同一常数。'],
    ['带宽决定利用率','稳定服务的简化模型可估算；FIFO 还要看数据到齐时间。'],
    ['开环 / 闭环','先区分外部请求发送、内部层预取反馈；不能据标签判断 U 高低。'],
    ['有限请求','任务清单会耗尽；扩大窗口可能混入结束后的空闲。'],
],154,[156,CW-156],rowh=45)
box('<b>比较 Baseline 与 Once，应固定输入和逐层推进机制。</b>'
    '研究它们的路径分配与服务顺序，怎样改变各卡的数据到齐时刻、等待和利用率。',447,73)
txt('核实来源（点击可打开）',M,541,13,bold=True)
sources=[
    ('[1] vLLM：Automatic Prefix Caching',
     'https://docs.vllm.ai/en/v0.9.2/features/automatic_prefix_caching.html',
     '命中前缀可复用 KV，跳过该部分的重复计算。'),
    ('[2] 本项目 data 与 token 划分',GH+'data',
     '第 2 页取 data 的 32K 四行；sim.py:231 定义计算与读取 token。'),
    ('[3] 本项目 continuous_batch_sim.py',GH+'continuous_batch_sim.py#L3269',
     '784 行读取 C；3156 行接纳；3269 行逐层预取；6672 行外部到达。'),
    ('[4] Grafana k6：Open and closed models',
     'https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/',
     '开环与闭环到达模型、系统变慢对闭环发送速率的影响。'),
]
for idx,(label,url,desc) in enumerate(sources):
    top=571+idx*41
    para(f'<link href="{url}" color="{BLUE}"><b>{label}</b></link>',M,top,CW,9.9,max_h=18)
    para(desc,M,top+18,CW,8.9,MUTED,max_h=18)
source(f'核对日期：2026-09-10。源码版本：{COMMIT[:12]}。本次仅核对代码与 data、制作教学图，没有重跑仿真。',753)
c.save()

manifest={
    'revision':'v2', 'source_commit':COMMIT, 'pages':page, 'data_rows':audit_rows,
    'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
        for f in ['data','sim.py','continuous_batch_sim.py']},
    'references':[{'label':a,'url':b,'scope':d} for a,b,d in sources],
    'toy_fifo':{'long_GiB':.2,'long_C_ms':100,'short_GiB':.004,'short_C_ms':1,
                'disk_GiB_s':40,'short_release_ms':.2,'short_deadline_ms':1.2,
                'short_ready_ms':5.1,'stall_ms':3.9,'nominal_sum_GiB_s':6},
    'simulations_executed':False,
    'arrival_example':{'A_service_ms':30,'B_service_ms':10,'C_service_ms':10,
                       'open_arrivals_ms':[0,10,20],'closed_arrivals_ms':[0,30,40]},
    'utilization_example':{'total_card_seconds':64,'compute_card_seconds':57.6,
                           'stall_card_seconds':6.4,'utilization':.9},
    'phase_example':{'planned_release_ms':[1,1],'actual_release_ms':[3,1]},
}
assert page==PAGE_COUNT
(HERE/'source_audit_v2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(HERE/'layout_audit_v2.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n')
(HERE/'prefill_miss_open_closed_explained_v2.md').write_text(
    '# 看懂 prefill miss、开环闭环与 NPU 利用率（修订版 v2）\n\n'
    '> PDF 的文字提取稿；图形布局请看同目录 PDF。\n\n'+'\n\n'.join(text_log)+'\n')
print(json.dumps({'pdf':str(OUT),'pages':page,'bytes':OUT.stat().st_size},ensure_ascii=False))
