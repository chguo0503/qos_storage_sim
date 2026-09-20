"""Build a six-page, beginner Chinese guide with native PDF diagrams."""

from pathlib import Path
from html import escape
import json
import hashlib

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.pagesizes import A4


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "KV命中后为什么还要算最后一个token_vLLM图解.pdf"
REF = json.loads((HERE / "sources.json").read_text())
FONT = Path.home() / ".fonts/msyh.ttc"
BOLD = Path.home() / ".fonts/msyhbd.ttc"
pdfmetrics.registerFont(TTFont("Zh", str(FONT), subfontIndex=0))
pdfmetrics.registerFont(TTFont("ZhB", str(BOLD), subfontIndex=0))
pdfmetrics.registerFontFamily("Zh", normal="Zh", bold="ZhB", italic="Zh", boldItalic="ZhB")

W, H = A4
M = 42
CW = W - 2 * M
INK = colors.HexColor("#182A3A")
MUTED = colors.HexColor("#526574")
BLUE = colors.HexColor("#1268A3")
ORANGE = colors.HexColor("#AE5513")
GREEN = colors.HexColor("#157653")
PALE_BLUE = colors.HexColor("#EAF3FA")
PALE_ORANGE = colors.HexColor("#FFF0E3")
PALE_GREEN = colors.HexColor("#EAF5EF")
LINE = colors.HexColor("#D3DCE4")
audit = []
texts = []
c = canvas.Canvas(str(OUT), pagesize=A4)
c.setTitle("KV命中后为什么还要算最后一个token：vLLM通俗图解")
c.setAuthor("QoS Storage Simulator")
c.setSubject("KV cache、最后一个输入token、logits与普通vLLM前缀缓存路径")


def para(text, y, x=M, width=CW, size=11.4, leading=18.5, color=INK, bold=False):
    p = Paragraph(text, ParagraphStyle("p", fontName="ZhB" if bold else "Zh",
        fontSize=size, leading=leading, textColor=color, wordWrap="CJK", spaceAfter=0))
    _, height = p.wrap(width, H)
    assert y + height < H - 42, (page, y, height, text)
    p.drawOn(c, x, H - y - height)
    audit.append(dict(page=page, x=x, top=y, bottom=y+height, width=width))
    texts.append(text)
    return y + height


def label(text, x, y, size=11, color=INK, bold=False):
    c.setFillColor(color)
    c.setFont("ZhB" if bold else "Zh", size)
    c.drawString(x, H-y-size, text)


def box(x, y, width, height, fill, stroke=LINE, radius=7):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, H-y-height, width, height, radius, fill=1, stroke=1)


def arrow(x1, y1, x2, y2, color=MUTED):
    import math
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.35)
    c.line(x1, H-y1, x2, H-y2)
    ang = math.atan2(y2-y1, x2-x1)
    p = c.beginPath()
    p.moveTo(x2, H-y2)
    for a in (ang+2.65, ang-2.65):
        p.lineTo(x2+7*math.cos(a), H-(y2+7*math.sin(a)))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def panel(title, body, y, fill=PALE_BLUE, color=BLUE, height=None):
    head_style = ParagraphStyle("tmp", fontName="Zh", fontSize=11.4, leading=18.5, wordWrap="CJK")
    body_h = Paragraph(body, head_style).wrap(CW-28, H)[1]
    height = height or 47 + body_h
    box(M, y, CW, height, fill)
    para(title, y+12, x=M+14, width=CW-28, color=color, bold=True, size=12)
    para(body, y+36, x=M+14, width=CW-28)
    return y+height


def table(headers, rows, y, widths):
    st = ParagraphStyle("cell", fontName="Zh", fontSize=10.3, leading=16.1, wordWrap="CJK", textColor=INK)
    values = [[Paragraph(str(v), st) for v in headers]] + [[Paragraph(str(v), st) for v in row] for row in rows]
    t = Table(values, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),PALE_BLUE), ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,0),0.8,LINE), ("LINEBELOW",(0,1),(-1,-1),0.45,LINE),
        ("LEFTPADDING",(0,0),(-1,-1),9), ("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    _, ht = t.wrap(CW, H)
    assert y+ht < H-58
    t.drawOn(c, M, H-y-ht)
    audit.append(dict(page=page, x=M, top=y, bottom=y+ht, width=CW))
    return y+ht


def start(number, title, subtitle):
    global page
    page = number
    c.setFillColor(BLUE)
    c.rect(M, H-37, 28, 3, fill=1, stroke=0)
    label("KV CACHE · 通俗图解", M+38, 26, 9, MUTED)
    para(title, 54, size=22, leading=30, bold=True)
    para(subtitle, 91, size=10, leading=15, color=MUTED)
    texts.append(f"PAGE {number}: {title}")


def finish(source_label, source_url=None):
    c.setStrokeColor(LINE)
    c.line(M, 47, W-M, 47)
    label(source_label, M, H-38, size=7.3, color=MUTED)
    if source_url:
        c.linkURL(source_url, (M, 21, W-92, 41), relative=0, thickness=0)
    c.setFillColor(MUTED)
    c.setFont("Zh",9)
    c.drawRightString(W-M, 27, f"{page} / 6")
    c.showPage()


# 1: input token and output token must not be conflated.
start(1, "最后一个输入，还要算一次？", "先认清：最后一个输入 token，与第一个输出 token，是两个位置。")
y = panel("这句话究竟是什么意思", "KV 缓存保留了可复用的中间资料。要决定接下来输出什么，模型还需要最后一个输入位置的最终计算结果。", 126)
para("假设用户输入：今天天气很", y+22, bold=True, size=13)
top = y+61
for i, word in enumerate(("今天", "天气", "很")):
    x=M+i*111
    box(x,top,94,57,PALE_ORANGE if i==2 else PALE_BLUE)
    para(word,top+12,x=x+15,width=70,size=18,bold=True,color=ORANGE if i==2 else BLUE)
arrow(M+320,top+29,M+362,top+29)
box(M+370,top,140,57,PALE_GREEN)
para("好",top+12,x=M+389,width=100,size=18,bold=True,color=GREEN)
para("最后一个输入",top+65,x=M+220,width=116,size=10,color=ORANGE)
para("第一个输出",top+65,x=M+370,width=140,size=10,color=GREEN)
para("为了示意，假设每个方块是 1 个 token；真实分词未必这样切。",top+99,size=9.6,color=MUTED)
y=para("重新算的是用户已经给出的“很”。模型根据“今天＋天气＋很”，为后面可能出现的 token 打分，再按生成规则选出“好”。",top+134)
y=table(["候选输出", "示意评分：logits", "如何理解"], [["好","3.1","当前较倾向这个候选"],["冷","2.0","也是一个候选"],["桌子","−1.4","当前较不倾向这个候选"]],y+23,[100,155,CW-255])
y=para("logits 就是一串原始分数，不是百分比。上表是教学数字；采样规则也会影响最后选哪个 token。",y+18,size=10,color=MUTED)
para("选出“好”后，如果继续生成，下一轮才会把“好”作为新的输入位置，预测它后面的 token。",y+14,size=10.5)
finish("原理参考：vLLM 模型的 hidden states → logits 输出接口（点击可查看）。",REF["llama_model"])


# 2: what state is and is not included in a KV cache.
start(2,"KV 是资料，logits 是候选评分", "可以把 KV 理解成各层留下的“可查阅笔记”；这只是帮助理解的比喻。")
y=table(["名称", "在模型里做什么", "普通 KV 缓存保存吗？"], [
    ["K / V", "K 帮当前 token 找相关信息；V 提供被取出的信息。实际都是数字向量。", "保存每层的 K 和 V"],
    ["最终隐藏状态", "最后一个输入位置走完所有模型层后形成的结果。", "不属于普通 KV 缓存"],
    ["logits", "将最终隐藏状态交给输出层，得到整个词表的候选评分。", "不属于普通 KV 缓存"],
],128,[106,255,CW-361])
para("一层计算里，KV 只是中间的一部分",y+27,bold=True,size=13)
top=y+71
labels=[("进入本层","当前状态",PALE_ORANGE),("生成 K / V","中间资料",PALE_BLUE),("注意力等计算","本层后续处理",PALE_ORANGE),("离开本层","新的状态",PALE_ORANGE)]
for i,(a,b,fill) in enumerate(labels):
    x=M+i*131
    box(x,top,116,70,fill)
    para(a,top+12,x=x+9,width=98,size=10.5,bold=True)
    para(b,top+39,x=x+9,width=98,size=9.2,color=MUTED)
    if i<3:arrow(x+117,top+35,x+128,top+35)
arrow(M+189,top+72,M+189,top+108,BLUE)
box(M+130,top+111,195,58,PALE_BLUE)
para("KV 缓存留下这一部分",top+128,x=M+143,width=170,color=BLUE,bold=True)
para("流程为概念示意；省略残差、归一化等细节。KV 分支不会保存完整的层输出。",top+186,size=9.6,color=MUTED)
y=panel("所以：缓存里有最后位置的 KV，也不等于有最终评分", "下一次请求能够复用前缀的 KV，但普通 KV 缓存没有直接给出这次所需的最终隐藏状态和 logits。vLLM 因此仍安排尾部 token 的前向计算。",top+230,PALE_GREEN,GREEN)
finish("参考：vLLM Llama 层与输出接口；KV cache manager 的 logits 说明。",REF["llama_model"])


# 3: single recomputed token traverses every transformer layer.
start(3,"重算一个 token，也要走完整个模型", "先看理想情况：只复用前 N−1 个 token，最后一个位置单独重新计算。")
para("仍用“今天｜天气｜很”。假设前两项在每一层的 KV 都可复用。下面把模型缩成 3 层示意；实际层数由模型决定。",130)
left=M; mid=M+223; top=208
for i in range(3):
    yy=top+i*108
    box(left,yy,178,73,PALE_BLUE)
    para(f"第 {i+1} 层：缓存资料",yy+10,x=left+12,width=154,bold=True,color=BLUE)
    para("“今天”“天气”的 K / V",yy+37,x=left+12,width=154,size=10)
    arrow(left+180,yy+36,mid-8,yy+36,BLUE)
    box(mid,yy,285,73,PALE_ORANGE)
    para(f"“很”在第 {i+1} 层重新计算",yy+10,x=mid+12,width=260,bold=True,color=ORANGE)
    para("读前缀 KV ＋ 处理自己的当前状态",yy+38,x=mid+12,width=260,size=10)
    if i<2:arrow(mid+145,yy+74,mid+145,yy+103,ORANGE)
arrow(mid+145,top+289,mid+145,top+323,ORANGE)
box(mid,top+325,285,70,PALE_GREEN)
para("最终隐藏状态 → 输出层 → logits",top+337,x=mid+12,width=260,color=GREEN,bold=True)
para("再选择第一个输出 token：“好”",top+363,x=mid+12,width=260,size=10)
para("前缀省下的是重新生成它们的 KV；本次尾部位置仍需在每层读取相关 KV。它还会生成自己的本层 KV，供当前注意力和后续生成使用。",top+419)
para("不是只跑最后一层，也不是先把“好”当输入来重算。",top+484,bold=True,color=ORANGE)
finish("参考：vLLM Llama forward/compute_logits；全注意力读取历史 KV。",REF["llama_model"])


# 4: actual ordinary full-block alignment.
start(4,"vLLM 实际可能重算一段尾巴", "示例限定：普通全注意力、本地前缀缓存，有效命中对齐粒度为 16 token。")
y=para("当前核对的 vLLM 版本会先限制：最多复用到最后一个输入 token 之前。然后按该缓存路径的对齐规则，找可复用前缀。",128)
box(M,y+17,CW,48,PALE_BLUE)
para("max_cache_hit_length = request.num_tokens - 1",y+31,x=M+13,width=CW-26,size=11,bold=True,color=BLUE)
para("下面假设所需前缀缓存都存在。蓝色＝复用，橙色＝本次重新计算。图表示位置范围，不按块内 token 数绘制长度。",y+82,size=10,color=MUTED)
start_y=y+139
cases=[(35,[("1—16",True),("17—32",True),("33—35",False)],32,3),
       (32,[("1—16",True),("17—32",False)],16,16),
       (33,[("1—16",True),("17—32",True),("33",False)],32,1)]
for i,(n,blocks,hit,miss) in enumerate(cases):
    yy=start_y+i*113
    para(f"输入 {n} 个 token",yy,bold=True,size=12)
    for j,(label_,reused) in enumerate(blocks):
        xx=M+j*113
        box(xx,yy+29,102,37,PALE_BLUE if reused else PALE_ORANGE)
        para(label_,yy+37,x=xx+12,width=84,size=11,color=BLUE if reused else ORANGE,bold=True)
    para(f"复用 {hit} 个\n<br/>重算 {miss} 个",yy+28,x=M+366,width=143,size=11,color=ORANGE)
    para(f"最多命中 {n-1} 个；向下对齐到 16 的倍数，得到 {hit}。",yy+73,size=9.8,color=MUTED)
y=start_y+342
para("因此：至少重算最后一个位置，不等于每次只重算一个 token。整块命中时，最后一块也可能被留给本次重算。",y+11,bold=True)
para("16 是教学配置，不是所有设备的默认值。细粒度命中、混合模型、推测解码或外部 KV 连接器可能使用其他规则；不要把这个示例当作所有配置的统一公式。",y+64,size=9.4,color=MUTED,leading=15)
finish("vLLM a8d1aa9c99b8：get_computed_blocks 与普通 full-attention 对齐路径。",REF["kv_manager"]+"#L289-L305")


# 5: connect directly to current user-provided trace and bandwidth semantics.
start(5,"回到我们的 ToolAgent 文件", "文件里的 miss=1 是转换模型的安排，不能直接当成 vLLM 运行测量。")
trace=ROOT/'trace/mooncake/tool_agent/toolagent_data_conversion/toolagent_requests.jsonl'
rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
ones=[r for r in rows if r['u_tokens']==1]
assert len(rows)==23608 and len(ones)==256
example=rows[16797]
assert (example['input_length'],example['hit_tokens'],example['u_tokens'])==(6257,6256,1)
y=para("23,608 条请求中，256 条被转换为只计算 1 个 token。转换器按历史 hash 判断前缀已见过，并把命中上限设成“总长度减 1”；这不是原 trace 记录的实测计算量。",129)
y=table(["同样的输入长度", "当前转换的计算量", "第 4 页的假设：对齐 16"],[
    ["6,257 token", "1 token", "命中 6,256；重算 1"],
    ["78,964 token", "1 token", "命中 78,960；重算 4"],
],y+22,[161,158,CW-319])
para("右列是依据对齐规则推算，不是已经运行 vLLM 的结果；前提仍是相关前缀缓存均可复用。",y+12,size=9.6,color=MUTED)
y=panel("只重算 1 个，为什么仍可能需要读很多？", "request_id=16797：输入 6,257 token，其中 6,256 个命中。当前合成模型给它每层约 53.83 微秒计算、24.44 MiB KV。若要把读取藏进这段计算，需要约 443.37 GiB/s。",y+64)
top=y+25
for i,(title_,desc) in enumerate([("KV 已在本卡 HBM","需要本地读取；不必再从 SSU 取同一份。"),("KV 只在 SSU / 远端","还要搬到执行侧；可能产生外部 I/O 等待。")]):
    xx=M+i*(CW/2+7)
    box(xx,top,CW/2-7,111,PALE_BLUE if i==0 else PALE_ORANGE)
    para(title_,top+13,x=xx+12,width=CW/2-31,bold=True,size=11)
    para(desc,top+43,x=xx+12,width=CW/2-31,size=10.5,leading=17)
para("这 53.83 微秒来自合成公式，不是 vLLM 实测。命中层级、有效块粒度、真实算子时间不同，计算量和外部带宽诉求都会变。",top+138,size=10.4,color=MUTED)
finish("本项目：convert_toolagent.py 与 toolagent_requests.jsonl；未修改原始 trace。")


# 6: recap without treating one implementation as an information-theoretic law.
start(6,"现在可以准确理解原来那句话了", "普通 KV 缓存复用前缀资料，尾部计算负责把资料变成当前需要的输出评分。")
y=panel("完整的一句话", "即使输入对应的 KV 都曾经算过，普通 KV 缓存也没有直接保存所需的最终 logits。vLLM 会保留尾部 token 的计算，以取得最后一个输入位置的输出；实际重算多少，还取决于命中对齐规则。",126,PALE_GREEN,GREEN)
y=table(["常见疑问", "准确理解"],[
    ["重算的是哪个 token？", "至少包括最后一个输入 token“很”；按块对齐时还包括前面的尾段。不是回答“好”。"],
    ["为什么不能直接拿最后一个 V 当答案？", "V 是一层注意力使用的中间向量，不是词表评分；后续计算和最终输出层仍要执行。"],
    ["既然以前算过 logits，为什么不用？", "普通 KV 缓存没有缓存它。另存最终隐藏状态或 logits 是另一种缓存设计，并非理论上绝对禁止。"],
    ["重算 16 个 token，时间就是 16 倍吗？", "不是。多个新 token 通常可以并行处理；时间取决于模型、上下文、硬件和 batch。"],
],y+24,[164,CW-164])
para("核对一下：输入有 32 个 token，有效对齐粒度为 16，且前缀缓存齐全。按第 4 页规则，会复用 16 个、重算 16 个。你能解释为什么不是全部复用吗？",y+20,size=10.5)
y=para("依据与适用范围",y+93,bold=True,size=11.5)
links=[("vLLM：为什么把命中上限设为 N−1",REF['kv_manager']+"#L265-L305"),
       ("vLLM：缓存协调与 full-attention 对齐",REF['coordinator']),
       ("vLLM：模型层与 hidden states → logits",REF['llama_model']),
       ("官方文档：自动前缀缓存",REF['prefix_docs'])]
for text,url in links:
    y=para(f'<link href="{escape(url,quote=True)}" color="#1268A3">{text}（点击）</link>',y+6,size=9.2,leading=14)
para("源码核对：2026-09-19；提交 a8d1aa9c99b8。本文没有运行真实 vLLM 性能测试，示意 token 切分和候选分数也不是模型实测。",y+14,size=9,color=MUTED,leading=14)
finish("源码固定到具体提交；混合模型与外部 KV 传输路径需要按实际配置另外核查。")
c.save()
(HERE/'layout_checks.json').write_text(json.dumps(dict(pages=6,paragraphs=len(audit),
    max_content_bottom=max(r['bottom'] for r in audit),pdf_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
    text_bounds_passed=True,blocks=audit),indent=2)+'\n')
print(OUT)
