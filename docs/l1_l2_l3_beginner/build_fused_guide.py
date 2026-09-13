#!/usr/bin/env python3
"""Integrated beginner manuscript guide. Only publishes existing evidence."""
import copy
import hashlib
import html
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from pypdf import PdfReader, PdfWriter
import build_guide as base
import layerwise_example

HERE, ROOT, STUDY = base.HERE, base.ROOT, base.STUDY
STEM = 'L1_L2_L3_手稿与实验_零基础融合教程'
PAGES = []
LAYERWISE = layerwise_example.evaluate()
note, formula, table, svg, rect, text = base.note, base.formula, base.table, base.svg, base.rect, base.text


def add(key, title, subtitle, body, source='教学示例；不是新的仿真结果。', hand=''):
    PAGES.append(dict(key=key,title=title,subtitle=subtitle,body=body,source=source,hand=hand))


def old(n,key,hand=''):
    p=copy.deepcopy(base.PAGES[n-1]);p.update(key=key,hand=hand)
    p.pop('tag',None)
    PAGES.append(p)


def math(s,caption=''):
    return f'<div class="math">{s}<small>{caption}</small></div>'


def original(s,where):
    return f'<div class="original"><label>手稿定位 · {where}</label><div>{s}</div></div>'


def meaning(s):
    return f'<div class="meaning"><b>换成大白话</b><p>{s}</p></div>'


def check(q,a):
    return f'<div class="check"><b>停一下，自己试一试</b><p>{q}</p><small>核对：{a}</small></div>'


def asset(name,caption):
    return f'<figure><img src="assets/{name}.svg"><figcaption>{caption}</figcaption></figure>'


def photo(n):
    return f'<div class="photo"><img src="../{n}.jpg" alt="第{n}张原始手稿，旋转显示"></div>'


def layerwise_timeline():
    b=''
    names={'A':'甲','B':'乙'}
    for panel,case in enumerate(LAYERWISE['cases']):
        top=panel*200
        b+=text(0,top+25,names[case['priority']]+'的层 IO 优先')
        offset,scale=140,130
        for row,label in enumerate(['SSU','NPU 甲','NPU 乙']):
            y=top+39+row*43
            b+=text(0,y+24,label)
            intervals=case['reads'] if row==0 else [x for x in case['compute'] if x['npu']==('A' if row==1 else 'B')]
            if row:
                cursor=6
                for item in intervals:
                    start,end=max(6,item['start']),min(12,item['end'])
                    if end<=start:continue
                    if start>cursor:
                        b+=rect(offset+(cursor-6)*scale,y,(start-cursor)*scale,33,'#c55a54','等待下一层数据','white')
                    cursor=end
                if cursor<12:
                    b+=rect(offset+(cursor-6)*scale,y,(12-cursor)*scale,33,'#c55a54','等待下一层数据','white')
            for item in intervals:
                start,end=max(6,item['start']),min(12,item['end'])
                if end<=start:continue
                label=('读'+names[item['npu']]+' L'+str(item['layer'])) if row==0 else ('算 L'+str(item['layer'])+'（'+str(item['end']-item['start'])+' 秒）')
                b+=rect(offset+(start-6)*scale,y,(end-start)*scale,33,'#e4e9f0' if row==0 else '#8461bb',label,'#17283d' if row==0 else 'white')
        for t in [6,8,10,12]:
            b+=text(offset+(t-6)*scale-8,top+188,str(t)+(' 秒' if t==12 else ''))
    return svg(b,400)


old(1,'start')
PAGES[-1]['subtitle']='只要会加减乘除，就能顺着例子读懂两张手稿。'
PAGES[-1]['body']=PAGES[-1]['body'].split('<div class="reading">')[0]+'''
<div class="reading"><b>可以分段读，不必一次记住</b><p>1–5 页：认识计算、等待和三层调度。<br>6–18 页：用小数字学求和、速率、积分。<br>19–24 页：理解总工作量、开环与闭环。<br>25–30 页：回到 32 张卡的真实实验。<br>31–35 页：逐组回看手稿、自测与结论。</p><p>每个难公式都配中文读法；原稿条件不完整的地方会直接标出。</p></div>'''
old(2,'profiles','手稿①：基本假设 1、2')
PAGES[-1]['body']+='<p class="small">这些长度、读取量和计算时间合在一起，称为一种请求“画像”。</p>'
old(3,'prefetch','手稿①：每层先到数据，才能计算')
old(6,'levels','手稿①②：L1 / L2 / L3')

add('map','两张手稿，其实在问三个问题','先知道它要说什么，再读符号。',f'''
<div class="cols"><div><b>手稿①：从一层推到利用率</b>{photo(1)}</div><div><b>手稿②：从一段时间推到总工作量</b>{photo(2)}</div></div>
<div class="decision"><b>第一个问题</b><h2>一层算多久、读多久，会等多久？</h2><p>这会用到除法、取大值 max、取小值 min，以及 μ 这个“平均推进速度”。</p></div>
<div class="decision"><b>第二个问题</b><h2>很多张卡、很多小段时间，怎样加起来？</h2><p>Σ 就是按卡或按类别相加；∫ 就是沿着时间相加。下面会从表格一步步过渡。</p></div>
<div class="decision"><b>第三个问题</b><h2>总工作量固定，为什么调度还可能有用？</h2><p>关键是：同样的工作，是否一定在同样长的时间里完成？</p></div>
{note('原图里的 U 有时像“平均利用率”，有时又像“忙卡总数”。我们会把两者分开，避免跟着符号混淆。')}
''','图片：docs/1.jpg、docs/2.jpg。仅在页面中旋转显示，原始文件未改动。')

add('sum','Σ 不难：就是把一项项加起来','先学卡编号 i，再学求和和平均。',f'''
<p>这里有两张卡，编号 i=1、2。<b>i 只是编号</b>，不是乘号。先观察同样长的 10 秒：</p>
{asset('math_sum','教学示意：紫色是计算，橙色是等待；不是实验画像 A/B 的配色。')}
{math('Σᵢ Hᵢ = H₁ + H₂ = 8 + 4 = 12 卡·秒','H 表示这张卡实际计算了多久；Σ 读作“把各项加起来”。')}
{math('平均 U = 12 ÷ (2 × 10) = 60%','N 表示卡数；本例 N=2，本次实验 N=32。')}
<p>也可以看各卡百分比：80%＋40%＝1.2 张等效忙卡。再除以 2，得到整机 60%。<b>1.2 张卡和 60% 是两种不同单位。</b></p>
{note('手稿把各卡利用率直接相加，后面又说满利用率是 1，口径不一致。若要表示平均 U，就要补上“÷N”。')}
{check('32 张卡全部忙，各卡的 100% 相加是什么？','32 张等效忙卡；除以 32，整机利用率才是 100%。')}
''',hand='手稿①：U(t) 的求和；手稿②：总计算量中的 U')

add('proportional','“正比”只是先用乘法估一下','先把手稿的 C、L、m、h 认出来。',f'''
{original('Cᵢ = t<sub>c</sub> mᵢ　　Lᵢ = α hᵢ','① 基本假设 1、2')}
{table(['符号','中文意思'],[['m','要补做计算的 token 数，也就是 miss 的数量'],['h','可复用、但需要从盘读回 KV 的 token 数'],['t_c','假设每个新增 token 要花的计算时间'],['α（读作 alpha）','假设每个复用 token 对应多少数据'],['C / L','这一层的计算时间 / 读取量；本教程先前把读取量写成 V，L=V']])}
<p><b>本页练习：</b>每个新 token 假设花 0.004 秒，1000 个就花 4 秒；每个命中 token 对应 0.01 份材料，4000 个就要读 40 份。</p>
<p class="small">式子里省略了乘号：t_c mᵢ 就是 t_c × mᵢ；α hᵢ 就是 α × hᵢ。</p>
{meaning('“每个要多少 × 一共有多少个 = 总共要多少”。这里故意用秒和份材料，先不做复杂单位换算。')}
<p><b>回到实验：</b>读取确实按固定 KV 大小换算；计算却不能跨 A/B 共用一个 t_c。B 的新增 token 是 A 的 16 倍，计算时间只有约 4.75 倍，因为上下文不同。</p>
{note('因此这些比例式是带条件的模型。实验直接用 data 的真实画像成本，没有强迫计算时间满足同一个比例系数。')}
''','手稿转写与教学算例；实验 data 的固定 KV 系数为每层每命中 token 1408 字节。',hand='手稿①：基本假设 1、2')

add('mu','μ 怎么读？先想“一次要多久”','μ 读作 mu；这里表示平均每秒能推进多少层。',f'''
<p>本页假设每层计算 <b>4 秒</b>，下一层要读 <b>8 份材料</b>；读取一直能得到 <b>1 份/秒</b>，没有额外排队，所以读取要 8 秒。</p>
{math('一次交接用时 = max(4, 8) = 8 秒','max 就是“取两个数中较大的那个”：要等计算和读取都结束。')}
{math('平均推进速度 = 1 ÷ 8 = 0.125 层/秒','倒数就是“一除以它”。8 秒推进 1 层，也就是 80 秒推进 10 层。')}
<p>也能从两个限制分别算：计算最多每秒做 1÷4＝0.25 层；存储最多每秒供应 1÷8＝0.125 层的数据。<b>取较小的速度</b>，因为更慢的一方限制推进。</p>
{original('μᵢ = min(1/Cᵢ, bᵢ/Lᵢ)','① 基本假设 4')}
{meaning('min 读成“取较小值”。b 是实际可获得的服务速率，L 是一层需要的数据量。')}
{note('手稿写“每秒完成数”，没有明确完成的是层还是请求。C、L 若按一层定义，μ 就是<b>层/秒</b>。本次一个请求有 8 层，不能把一层当成一整个请求，也不能直接当 TTFT。')}
<p class="small">本页是固定画像、稳定重叠的理想模型，忽略启动和结尾。若读取含排队，先使用真实读取耗时，不能直接拿盘标称速度代入。</p>
''',hand='手稿①：μ 的 min 公式及“每秒请求数”')

add('util_formula','为什么“每秒几层 × 每层几秒”是利用率？','继续上一页同一个小例子，不换数字。',f'''
<p>80 秒能推进 10 层；每层算 4 秒，所以实际计算 40 秒，其余在等。利用率就是 40÷80＝50%。</p>
{math('uᵢ = μᵢ × Cᵢ = 0.125 × 4 = 0.5 = 50%','小写 u 先表示一张卡的周期平均；不是某个时刻的开关状态。')}
<p>这张卡想在 4 秒内拿到 8 份，需求速率是 8÷4＝2 份/秒。它只得到 1 份/秒，是需求的一半，所以本模型中计算占比为一半。</p>
{original('U(t) = Σᵢ μᵢ(t) Cᵢ = Σᵢ min(1, bᵢ(t)/Bᵢ)','① “每秒 NPU 有效计算时间”一行')}
{math('μᵢCᵢ = Cᵢ × min(1/Cᵢ, bᵢ/Lᵢ)<br>= min(1, bᵢCᵢ/Lᵢ) = min(1, bᵢ/Bᵢ)','第一步代入 μ；C>0，可乘进两项。再用 Bᵢ=Lᵢ/Cᵢ，所以 bᵢCᵢ/Lᵢ=bᵢ/Bᵢ。')}
{math('单卡 uᵢ = min(1, bᵢ/Bᵢ)<br>整机平均 U = (u₁ + u₂ + … + u<sub>N</sub>) ÷ N','“…”只是表示中间还有同样的项。上限 1 就是 100%。')}
{note('这是<b>固定画像、供给规律明确</b>时，把相同周期反复执行后得到的平均模型。手稿把它叫“瞬时利用率”过于直接；真实系统一边预取一边计算，某瞬间的盘速不能直接当某瞬间的计算比例。')}
{check('需求 2 份/秒，供给提高到 4，能得到 200% 利用率吗？','不能。min(1, 4÷2)=1，最多 100%。')}
''',hand='手稿①：U 等式链、结论①')

old(5,'demand','手稿①：Bᵢ=Lᵢ/Cᵢ，需求与供给')
PAGES[-1]['body']+= '<p class="small">手稿的大写 Bᵢ 是第 i 项需求，小写 bᵢ 是获得的供给；没有下标的大写 B 是盘的总容量。它们也都不是实验画像 B 的名字。</p>'
old(4,'cross_request','手稿公式回到真实交接：C 应当取哪一段')

add('underload','“需求不超容量”，到底保证了什么？','先在理想模型中看清，再回到 FIFO。',f'''
{original('Σᵢ Bᵢ ≤ B；若 bᵢ=Bᵢ，则 U=1，否则 U&lt;1','① 情况 1：不过载，以及 1.1 分支')}
<p>两张卡的需求是 2、5 份/秒，盘最多 10 份/秒。2＋5≤10，意思是：<b>理想地分配时，容量足以同时覆盖两项需求。</b></p>
{table(['有数据待读时，两卡可获得的速率','各卡利用率','整机平均'],[['2、5','100%、100%','100%'],['3、6','100%、100%','100%']])}
<p>第二行没有“恰好等于需求”，仍然满利用率。所以手稿的“否则 U&lt;1”需要改：<b>达到或超过需求都可饱和</b>；超过的部分不会让计算超过 100%。</p>
{original('U(t) = Σᵢ bᵢ(t)/Bᵢ …','① 情况 1.2：后接 ≤1 一类上界；括号个别字迹不清')}
<p>这里不能直接去掉 min：只有每个 bᵢ≤Bᵢ 时，min(1,bᵢ/Bᵢ) 才等于 bᵢ/Bᵢ。求平均仍须除以卡数。</p>
{note('上面讲的是理想供给。真实 FIFO 还要问：<b>数据有没有按时送到？</b>总量能分得开，不代表现有队列就会按正确时机送。下一页看反例。')}
<p class="small">更快时会提前读完，不代表整段一直收到表中的速率。原稿写有“待补充”，尚非完整机制；本次 Baseline 也没有不可互借的固定配额。实际还要逐盘、逐时刻检查。</p>
''',hand='手稿①：不过载、1.1/1.2 两个分支')

old(10,'fifo','手稿①②：名义欠载时仍可能因服务先后等待')

add('overload','带宽真的不够时，怎么分也有取舍','用手稿的模型算一个两卡例子。',f'''
<p>两卡每层都算 10 秒；甲每层读 20 份，需求 2 份/秒；乙每层读 100 份，需求 10 份/秒。盘只有 10 份/秒，总需求 12，确实不足。</p>
{table(['理想供给方案','甲 / 乙获得','甲 / 乙利用率','平均 U'],[['按需求比例分','约 1.67 / 8.33','83.33% / 83.33%','83.33%'],['先满足低需求甲','2 / 8','100% / 80%','90%']])}
<p>第二行算起来很简单：甲是 2÷2＝100%；乙是 8÷10＝80%；平均是 90%。总供给两行都是 10，却得到不同平均利用率。</p>
{meaning('没有满利用率时，同样多给一点带宽，低需求卡的计算占比可能增加得更多。达到 100% 后，再给也不能继续增加。')}
{original('Bᵢ 从小到大分配，瞬时利用率最高','① 结论①，按可辨认字迹重排')}
<p>例如两卡的供给都从 0 增加到 1 份/秒：甲的利用率增加 1÷2＝50 个百分点，乙增加 1÷10＝10 个百分点。达到各自需求后就不再增加，因此先把小需求卡填满更有利。</p>
{note('若画像固定、允许任意分配、唯一目标是最大化平均 U，这种从小需求开始满足的办法是该模型的最优分配。<b>它不是瞬时物理规律，也没有同时保证公平、TTFT 或本次 Once 的效果。</b>')}
<p class="small">本页供给是理想稳定模型；甲乙不是本次 A/B。模型用于理解取舍，不是新增仿真结果。</p>
''',hand='手稿①：结论①、情况 2 过载')

add('integral','∫ 看着陌生，其实是沿时间加起来','先不用符号，先把表格算完。',f'''
{asset('math_integral','教学图：实际速率的矩形面积相加，就是已服务的数据量。')}
{table(['时间段','每秒实际送多少','持续多久','这一段送了多少'],[['0–2 秒','6 份/秒','2 秒','6×2＝12 份'],['2–3 秒','0','1 秒','0×1＝0 份'],['3–5 秒','9 份/秒','2 秒','9×2＝18 份']])}
{math('12 + 0 + 18 = 30 份<br>用手稿记法：∫₀⁵ b(t) dt = 30 份','0 到 5 表示从第 0 秒累计到第 5 秒；b(t) 是时刻 t 的速率；dt 是一小段时间。')}
{meaning('一小段里：速率 × 这段有多长＝这一段的数据量。把所有小段相加，就是积分。速率变化更细时，也还是这个想法。')}
{note('Σ 通常按“谁”相加；∫ 在这里按“时间”相加。看到 ∫Σ，就是先把同一时刻各项加起来，再把这段时间累计起来。')}
''',hand='手稿①末部及②：所有时间积分的读法')

add('swap_sum','Σ 和 ∫ 换个顺序，为什么总数一样？','就像一张表，按行加和按列加，最后相同。',f'''
<p>下面两段各长 1 秒，因此每格“每秒速率×1秒”就是这一格的数据量。先只看收到多少份：</p>
{table(['实际数据量','第 1 秒','第 2 秒','各自合计'],[['甲','2 份','4 份','6 份'],['乙','3 份','1 份','4 份'],['每秒合计','5 份','5 份','10 份']])}
<div class="decision"><b>先按列加，再沿时间加</b><h2>(2＋3)＋(4＋1)＝10 份</h2><p>每个时刻先把各项相加，这是 Σ；再沿时间累计，这是外面的 ∫。</p></div>
<div class="decision"><b>先沿时间加，再把各项加起来</b><h2>(2＋4)＋(3＋1)＝10 份</h2><p>先求甲全段收到的 6 份、乙的 4 份，再合在一起。</p></div>
{original('∫₀ᵀ Σᵢ bᵢ(t)dt = Σᵢ ∫₀ᵀ bᵢ(t)dt','② 第一行中间的交换顺序')}
{meaning('同样一批格子，每格都算一次，只是相加顺序不同，所以总数一样。对本次有限卡数和正常服务速率，这一步可以保留。')}
{note('但“交换相加顺序”不等于“可以把不同的换算系数变成同一个”。后面读 W_C 时，还要记住每类自己的 Bᵢ。')}
''',hand='手稿②：交换求和与时间积分')

add('arrival_area','同样送了 8 份，为什么一条会等？','把积分和“截止前到齐”连起来。',f'''
<p>下一层需要 8 份；当前计算从第 0 秒开始，到第 4 秒结束。每一格都长 1 秒：</p>
{table(['实际每秒收到','0–1','1–2','2–3','3–4','4–5'],[['提前送','4','4','0','0','0'],['晚送','0','0','0','4','4']])}
<p><b>提前送：</b>前两格 4＋4＝8，第二秒就收齐。后面带宽为零，卡仍能完成当前计算，并按时开始下一层。</p>
<p><b>晚送：</b>到第 4 秒只累计收到 4；还缺 4。第 5 秒才收齐，所以卡在第 4–5 秒等了 1 秒。</p>
{math('截止前到达量 = ∫<sub>开始预取</sub><sup>计算结束</sup> 实际到达速率 dt','要把目标数据收齐，而不是要求曲线每一刻都高于参考需求。')}
{note('本例需求是 8÷4＝2 份/秒。晚送在<b>等待期间</b>的实际速率是 4，高于需求，卡却仍在等，因为还没收齐。')}
<p>因此，手稿里的“带宽决定瞬时利用率”不能直接当成真实时序的公式。NPU 可以在当前没读数据时计算，也可以在高速补数据时等待。</p>
<p class="small">真实 A→B 首层同样看累计到达：A 算完时 B 的 38.5 MiB 尚未到达，之后补齐才恢复。要检查 NPU 接收端；SSD 读出后还可能经过接收链路。</p>
''',hand='手稿①：瞬时带宽与 U(t) 的使用边界')

old(9,'local_A','手稿①模型的局部实测核验：固定 A 轮次')

add('population','nᵢ Lᵢ：把“一层多少”乘上“做了几层”','这里明确规定 nᵢ 是层数，不是卡数。',f'''
<p>N 是总卡数；nᵢ 是第 i 类一共执行的层数。两个字母很像，但意思不同。这里 i 从卡编号改为<b>画像类别编号</b>，先按同类数据分组。</p>
<p>本次每个请求有 8 层。2 个同类请求，就有 2×8＝16 层；若每层读 3 份，总共就是 16×3＝48 份。</p>
{original('∫₀ᵀ bᵢ(t)dt = Nᵢ Lᵢ','① 下部：有限人口的逐项守恒；原稿记作 Nᵢ')}
{meaning('同一类任务全部做完后，实际累计读出的量，应该等于这类“层数 × 每层读取量”。前提是都读完了，没有重复读或丢弃。')}
{table(['本次完整人口','A','B'],[['全局请求数','32×40＝1280','32×80＝2560'],['全局层数 nᵢ','10240','20480'],['完整读取量','1756.5625 GiB','770 GiB']])}
{math('总读取 W<sub>D</sub> = Σᵢ nᵢLᵢ = 2526.5625 GiB','W 可以读作总工作量，D 在这里表示数据。')}
{note('必须统计同一批任务直到完成。只截 2 秒时，有些任务尚未完成、有些读取跨过边界，就不能直接把整批配额塞进去。')}
<p class="small">原稿 Nᵢ 的计数单位没有充分写明；本文改用 nᵢ 表示层数，避免与卡数 N 混淆。如果用请求数，就必须再乘每请求层数。</p>
''','真实工作量：accounting_checks.json；3840 请求、30720 层完整完成。',hand='手稿①末行；②数据积分与 ΣnᵢLᵢ')

add('weights','为什么要“各自除完，再加起来”？','手稿第二张最容易跳步的地方。',f'''
{asset('math_weights','教学示意：同样叫读取，不同画像对应的计算量不同。先分别换算，再相加。')}
<p>图中的 r 就是手稿的 Bᵢ：Bᵢ=Lᵢ/Cᵢ，因此 Lᵢ/Bᵢ=Cᵢ。把这一类全部层的数据相加后，再除它自己的 Bᵢ，就能换回它对应的总计算时间。</p>
{math('总计算 W<sub>C</sub> = Σᵢ nᵢCᵢ = Σᵢ (nᵢLᵢ / Bᵢ)','按图中数字：40÷2＋200÷10＝20＋20＝40 卡·秒。')}
{original('… = (1/Bᵢ) Σᵢ ∫₀ᵀ bᵢ(t)dt = nᵢLᵢ/Bᵢ','② 第二行后半部分，按可辨认字迹重排')}
<p>若这一行要表示全部类别总量，手稿缺少完整求和。<b>不同 i 的 Bᵢ 不同，不能把一个 1/Bᵢ 拿到对 i 的求和外。</b></p>
{check('图中总数据 240 份，直接全部除以甲的 2，得到 120 卡·秒，为什么错？','乙的数据不是按甲的关系换算；正确做法分别除，再加，得到 40。')}
<p class="small">完整批次里，这个恒等关系可以成立；但盘正在预取 B 时，不能拿卡当前计算 A 的需求去加权 B 的字节。必须按数据真正所属的画像/请求对应。</p>
''',hand='手稿②：W_C 的带宽加权积分、移出系数与末项')

add('compute_integral','把利用率沿时间累加，得到什么？','得到“计算时间账”，不是另一个百分比。',f'''
<p>一张卡：前 2 秒一直计算，后 8 秒只计算其中 2 秒。总计算 4 秒，总观察 10 秒，所以利用率为 40%。</p>
{table(['时间段','这段 U','时间长度','计算时间'],[['前一段','100%','2 秒','1×2＝2 秒'],['后一段','25%','8 秒','0.25×8＝2 秒'],['合计','40%','10 秒','4 秒']])}
<p>不能直接平均 100% 和 25% 得到 62.5%：后一段占的时间更长。按时间加权，就是把每段的“利用率×时长”累加。</p>
{original('W_C = ∫₀ᵀ U(t)dt','② 第二行开头')}
{math('若 U 是整机平均：W<sub>C</sub> = N × ∫₀ᵀ U(t)dt<br>若 U 是等效忙卡数：W<sub>C</sub> = ∫₀ᵀ U(t)dt','两种写法都可以，但同一篇推导必须坚持同一种定义。')}
<p class="small">这里真实 U(t) 是“时刻 t 正在计算的卡数÷总卡数”，不是代入瞬时带宽得到的值。上表先按每段平均核账；直接累加真实计算区间也得到相同时间账。</p>
{note('这能与上一页的总计算量对上：同一批全部完成后，既可按“做了几层”算，也可按“真实计算了多久”数。<b>两本账一致，不代表瞬时盘速与计算状态处处成比例。</b>')}
<p class="small">你现在已经学过相加、累计、逐类换算。到第 {{sheet2}} 页，会把手稿的完整加权积分链逐等号连起来，并保留每类自己的 Bᵢ。</p>
<p class="small">本次完整计算账为 647.267282 卡·秒。公式中的时间单位需要一致：若积分用毫秒，结果是卡·毫秒，除以 1000 后才是卡·秒。</p>
''','真实总计算量来自完整层日志；分段 100% / 25% 为教学算例。',hand='手稿②：W_C=∫U 及与加权读取积分的等号')

add('capacity','手稿的“＝BT”，什么时候能写等号？','把“最多能做多少”和“实际做了多少”分开。',f'''
{original('∫₀ᵀ Σᵢ bᵢ(t)dt = ∫₀ᵀ Bdt = BT = Σᵢ nᵢLᵢ','② 第一行')}
<p>B 是盘容量，T 是观察时长。还记得积分例子吗？盘容量为 10，观察 5 秒，理论最多 50 份；实际有空闲，只送了 30 份。</p>
{math('一般应写：实际累计服务 ≤ B × T','只有整个区间都满速服务，左边才等于右边。')}
{original('∫₀ᵀ Σᵢ Bᵢ(t)dt ≥ BT','① 情况 2：名义需求面积达到或超过容量')}
<p>这是<b>想要的工作面积</b>与能力比较，不能自动证明实际盘一直忙。未来读取可能还没被计算释放出来；当前请求在等时，图上的 V/C 也不会不停生成新数据。</p>
<p class="small">原式使用 ≥。其中等号只是理想需求面积刚好等于容量，并非严格超过；要与真正持续过载区分。</p>
{math('完成时间 ≥ 总读取量 ÷ 总容量','这是“再快也不能低于”的下界，不是实际完成时间的等号。')}
<p>本次 3 盘：2526.5625÷120＝<b>21.0547 秒</b>。它只是总容量下界；更严还要逐盘检查。还没算上读取结束后的最后计算等影响。</p>
{note('手稿把 T 直接写成“总读取÷B”，需要补上持续满速、统计边界合适等条件。<b>全批次平均过载，也不保证每个时刻都在满速读。</b>')}
''','真实总读取量用于下界计算；最热盘约束给 3 盘略强的 21.0574 秒下界。',hand='手稿①：需求面积过载；②：实际服务、BT 与 T 的等号链')

add('finite_counter','逐层预取：同一窗口里，L3 怎样改变 U？','当前层开始计算，才发下一层读取；比较同样的第 6–12 秒。',f'''
<p>两卡各有一个 8 层请求。每层都读 2 份；甲每层算 2 秒，乙算 6 秒；盘速 1 份/秒。初始层数据均已就绪，记作 t=0。每次读完一层后，盘从已提交的 IO 中选优先卡。<b>只改层 IO 优先级。</b></p>
{layerwise_timeline()}
{table(['同一窗口 [6,12)','甲计算','乙计算','整机平均 U'],[['甲的层 IO 优先','6 秒','0 秒','6÷(2×6)＝50%'],['乙的层 IO 优先','4 秒','6 秒','10÷(2×6)＝83.33%']])}
<p class="small">L0 是初始层。甲优先时反复抢到盘，乙在第 6 秒算完 L0 后等 L1；乙优先时，甲在 8–10 秒等待，乙整窗计算。两方案两卡在窗口内都还有未完成请求，无任务结束后的空闲。</p>
{original('开环有限请求：NPU 利用率与 L3 调度无关','① 下部的一条结论')}
{note('外部请求名单固定，不代表后续层 IO 都在起点可读。<b>层服务先后会改变下一层就绪时间，从而改变同一窗口的计算时间。</b>这里乙的 B=2÷6 比甲的 B=2÷2 小；完整周期平均可以与前面的理想模型对上。')}
<p class="small">本例过载：总需求 4/3&gt;容量 1；不是 Baseline/Once 或欠载证据。83.33% 也不是任意短窗的最优上限：t=6 起可读甲→甲→乙，使本窗 100%，但推迟后续计算；详见配套 Markdown。整层不抢占，同刻完成与新提交先于选 IO。</p>
''','逐层教学时序经精确算术核验；非真实实验。修订：2026-09-13。',hand='手稿①：外部有限输入与内部逐层反馈；②：固定窗口内已执行工作量')

add('open_closed','“开闭环”和“有限无限”是两个问题','不要把两组词绑成固定搭配。',f'''
{table(['','到达按预定时间，不看完成：开环','完成一条才补一条：闭环'],[['名单有限','0、1、2、3、4、5 秒各来一条，共 6 条','做完再发下一条，最多共 6 条'],['名单不断补充','每秒来一条，一直继续','做完就补一条，一直继续']])}
<p><b>横着看：</b>下一条什么时候来，要不要看前面完成？<br><b>竖着看：</b>一共就这些任务，还是会一直补？</p>
{original('闭环无限请求：L3 调度影响 L2，可提升利用率；固定时间内读取量不守恒','② 中部')}
{meaning('如果服务变快，后面的任务可能更早开始。同样 10 秒内，可能完成更多层、读更多数据、做更多计算。这里比较的是固定时间内的处理量。')}
{note('“可能改善”不等于“必然改善”。也不是数据违反物理守恒：<b>工作人口可以不同，所以固定窗口处理的总量不必相同。</b>如果两策略都已经 100%，就没有更高的利用率可争取。')}
<p>回到本次：外部请求全在 t=0 入队，是有限名单；但下一层 IO 在当前层开始计算时才发出，内部有反馈。等待会改变之后的读取时刻。只改到达标签、却保持任务可用性和预取条件一样，U 也可能不变。</p>
''','到达表为教学示意；本次固定队列与层 IO 释放规则为源码事实。',hand='手稿①：开环有限；②：闭环无限、窗口工作量与 L2/L3 反馈')

old(7,'input','回到实验：固定 L1、改变 L2')
PAGES[-1]['body']+='<p class="small">手稿说 L1/L2 决定每卡请求数，要区分：本次每卡 40A＋80B 的配额已固定，L2 排序没有改变它；排序可以改变的是固定时间窗内已经推进了多少条。</p>'
old(8,'timelines','回到实验：窗口 U 来自真实计算时间')
PAGES[-1]['body']=PAGES[-1]['body'].replace('<p class="small">在 2–4 秒内，两组全部 32 卡都有任务，且每卡都实际计算过 A、B。当前 27 格全部使用 Baseline，尚未比较本次输入的 Once。</p>','')
PAGES[-1]['source']='真实结果：4 盘长验证 seed7、[2,4) 秒；两组均 32 卡全窗有任务，且逐卡算过 A/B。'
old(11,'recovery','手稿的动态反馈：是否自然错开，要看长期日志')
old(12,'actual_total','手稿②的完整工作量与本次实际完成时间')
old(14,'once','手稿②：L3 能否补救 L2')
PAGES[-1]['body']+= '<p class="small">手稿末尾“L2 不当会降低利用率、L3 可补救”应带条件理解：要指出谁错过截止，以及已有 IO 是否可以更及时服务。名义需求低于容量本身不会制造等待。</p>'
old(15,'slo','避免把利用率当成请求体验或公平性')

add('sheet1','现在回看手稿①：每一组都能读了','保留原意，补齐平均口径和模型条件。',f'''
{table(['手稿位置','现在怎样读','对应教程'],[
['基本假设 1、2','计算/读取各自按数量估算；计算比例有前提。','见 {{proportional}} 页'],
['基本假设 3','Bᵢ=Lᵢ/Cᵢ 是需求，bᵢ 是供给，B 是容量。','见 {{demand}} 页'],
['基本假设 4','一层推进受计算和读取两者中更慢的一方限制。','见 {{mu}} 页'],
['U 的等式链','单卡 uᵢ=μᵢCᵢ；各卡相加后，再除 N 才是平均。','见 {{sum}}、{{util_formula}} 页'],
['结论①','低需求优先是带条件的模型结论；瞬时推广另有限制。','见 {{overload}}、{{arrival_area}}、{{local_A}} 页'],
['不过载 1.1 / 1.2','供给≥需求可饱和；不能随便去掉 min，也不能漏平均。','见 {{underload}} 页'],
['过载的积分','需求面积不是实际到达数据，更不保证盘一直满速。','见 {{overload}}、{{capacity}} 页'],
['有限请求与 L3','有限名单仍有逐层反馈；同窗计算时间可随 L3 改变。','见 {{finite_counter}} 页'],
['逐卡/逐类读取守恒','同一批全部完成后，累计字节=层数×每层字节。','见 {{population}} 页']])}
{note('原稿的 Σmin 可以表示“等效忙卡数”；若称平均利用率，记得 ÷N。原稿 μ 若按每层 C/L 定义，单位就是“层/秒”。')}
<p>这页不是让你背公式。先说中文，再用指向的页码找一个小例子核对。对照原图时，个别字迹不清的词不用猜；关键等式的条件已单独讲解。</p>
''','原图 docs/1.jpg；原式转写、条件和全部定位见 manuscript_inventory.json。')

add('sheet2','现在回看手稿②：把长等式拆成四步','长公式不是新魔法，只是把前面几本账连起来。',f'''
<div class="decision"><b>第一步 · 累计读取</b><h2>∫Σbᵢ dt = ΣnᵢLᵢ</h2><p>同一批全部完成，实际累计服务等于总读取。只有全段满速才再等于 BT；一般只是 ≤BT。见 {{population}}、{{capacity}} 页。</p></div>
<div class="decision"><b>第二步 · 每类换回计算，再相加</b><h2 style="font-size:17px">∫ Σᵢ[bᵢ(t)/Bᵢ]dt = Σᵢ(1/Bᵢ)∫ bᵢ(t)dt<br>= ΣᵢnᵢLᵢ/Bᵢ = ΣᵢnᵢCᵢ = W_C</h2><p>都从 t=0 累计到同一批完成。每类 Bᵢ 固定，可移出它自己的时间积分，但要留在对不同 i 的求和里。这不声称瞬时 U 等于瞬时带宽比。见 {{weights}}、{{compute_integral}} 页。</p></div>
<div class="decision"><b>第三步 · 用真实计算时间核同一本账</b><h2>W_C = N∫U(t)dt</h2><p>这里 U 定义为整机平均。若原稿 U 代表忙卡总数，就不再乘 N。完整批次可以核对两边；瞬时读取与瞬时计算不能随意直接等同。见 {{compute_integral}} 页。</p></div>
<div class="decision"><b>第四步 · 再除观察范围，才得到平均 U</b><h2>U_full = W_C ÷ (N × T_finish)</h2><p>全程见 {{actual_total}} 页：分子相同，结束时间仍可不同。固定窗口见 {{finite_counter}} 页：分母相同，已执行的计算量仍可不同；不要混用两种口径。</p></div>
{note('“L3 可帮助 L2”是一种能力，不是无条件保证。它能调整已提交 IO 的服务机会；不会自动改请求分卡、不会增加盘带宽，也不能直接读取未提交数据。')}
''','原图 docs/2.jpg；四步统一采用整机平均 U 和按画像统计的完整人口。')

add('questions','用六个小问题，检查自己是否真的懂了','答案就在下面；会用自己的话说出来，比记符号重要。',f'''
{check('两张卡都观察 10 秒，各计算 8 秒和 4 秒，平均 U 是多少？','(8＋4)÷(2×10)=60%，不是 120%。')}
{check('当前算 4 秒，下一层第 8 秒才读齐，要等多久？','等 4 秒；总交接周期 8 秒，不是 12 秒。')}
{check('∫b(t)dt 最直白的意思是什么？','每小段实际速率×时长，再相加；得到累计数据量。')}
{check('盘标称 10，观察 5 秒，能不能直接说实际读了 50？','不能。50 是能力上界；中间可能空闲。')}
{check('同一批任务总读取和总计算都相同，U 一定相同吗？','不一定；全部完成时间可能不同。')}
{check('4 盘 Ordered 的一个小段是 17.56%，能说整场就是 17.56% 吗？','不能。2–4 秒为 70.23%，2–20 秒为 71.34%，全程为 71.72%；范围不同。')}
''','题目答案沿用本文教学例子及本次 4 盘 seed7 实验，不新增结果。')

old(16,'conclusions','读完手稿后，区分已证实结果与待测策略')

add('sources','需要复查时，按这张索引找','本版可以单独阅读；原图和精确日志也都保留。',f'''
{table(['想查什么','位置'],[['原始手稿','docs/1.jpg、docs/2.jpg'],['逐条转写、字迹疑点和成立条件','本目录 manuscript_inventory.json'],['每一项手稿在教程哪页','本目录 manuscript_coverage.json'],['本次原始输入和结果','results/baseline_ab128_32_ratio12_20260912/'],['完整公式与实验对照','上述目录 formula_review/README.md'],['字节、计算、窗口和局部交接核验','accounting_checks.json、handoff_checks.json'],['PDF 图源和生成脚本','本目录 assets/、build_fused_guide.py']])}
<p><b>真实实验：</b>均明确标盘数和窗口。核心对照为 32 NPU、4 SSU、seed7，长验证每卡 40A＋80B。3/6 盘用于展示容量与恢复差别；表中 Random 均不是三种子平均。</p>
<p><b>教学例子：</b>刻意用小数字、秒和份材料练习，不冒充真实 NPU 参数。甲乙的紫/橙色与实验 A/B 的蓝/绿色分开。</p>
<p><b>记号约定：</b>实验读量 L=V；Bᵢ 是名义需求，不是画像 B。N 是卡数；nᵢ 明确为层数。求和的 i 会在每个统计范围说明是卡还是画像。整机平均 U 的满值是 1。</p>
{note('这次只重新整理手稿、教学算例和已有日志，没有启动新仿真。本次 AB 的 Once 格仍是待测；没有把旧画像的策略收益搬过来。')}
''','融合版 · 2026-09-12。数字按阅读需要取小数，精确值保存在源审计文件中。')


def publish():
    # Stable conceptual anchors keep page references correct after editing.
    index={p['key']:i for i,p in enumerate(PAGES,1)}
    for p in PAGES:
        p['body']=re.sub(r'\{\{?(\w+)\}?\}',lambda m:str(index[m[1]]),p['body'])
        assert not re.search(r'\{\{?\w+\}?\}',p['body']),p['key']
    inventory=json.loads((HERE/'manuscript_inventory.json').read_text())
    covered={
        'H1-01':['proportional','profiles'],'H1-02':['proportional','profiles'],
        'H1-03':['demand','underload'],'H1-04':['demand','once','capacity'],
        'H1-05':['prefetch','cross_request'],'H1-06':['mu'],
        'H1-07':['sum','util_formula'],'H1-08':['util_formula'],'H1-09':['util_formula'],
        'H1-10':['arrival_area','local_A'],'H1-11':['overload'],
        'H1-12':['underload','fifo'],'H1-13':['underload'],'H1-14':['underload'],
        'H1-15':['underload'],'H1-16':['underload','util_formula'],
        'H1-17':['underload','sum'],'H1-18':['capacity','overload'],
        'H1-19':['finite_counter','open_closed'],'H1-20':['levels','input'],
        'H1-21':['population','open_closed'],
        'H2-01':['capacity'],'H2-02':['integral','capacity'],
        'H2-03':['swap_sum'],'H2-04':['population'],
        'H2-05':['capacity','finite_counter'],'H2-06':['compute_integral'],
        'H2-07':['compute_integral','weights','sheet2','arrival_area'],
        'H2-08':['weights','sheet2'],'H2-09':['weights','sheet2'],
        'H2-10':['open_closed'],'H2-11':['open_closed','levels','once'],
        'H2-12':['open_closed'],'H2-13':['input','timelines','recovery'],
        'H2-14':['fifo','capacity','once'],'H2-15':['once','sheet2','conclusions']}
    assert set(covered)=={item['id'] for item in inventory['items']}
    coverage=[]
    for item in inventory['items']:
        keys=covered[item['id']]
        assert keys and all(k in index for k in keys)
        coverage.append(dict(id=item['id'],source_location=item['source_location'],
                             manuscript=item['manuscript_transcription'],judgment=item['judgment'],
                             pages=[dict(number=index[k],key=k,title=PAGES[index[k]-1]['title']) for k in keys],
                             handwriting_uncertainty=item['handwriting_uncertainty']))
    (HERE/'manuscript_coverage.json').write_text(json.dumps({'all_items_mapped':True,'item_count':len(coverage),
         'page_count':len(PAGES),'note':'逐项内容经独立审稿；本自动检查确认定位完整，不代替数学条件审查。',
         'items':coverage},ensure_ascii=False,indent=2)+'\n')
    css=base.CSS+'''
    .math{font-size:21px;line-height:1.8;text-align:center;background:#f1f5fa;padding:14px;border-radius:9px;margin:15px 0;font-weight:700}
    .math small{display:block;font-size:12px;font-weight:400;line-height:1.65;margin-top:5px}.math sub,.math sup{font-size:12px}
    .original{border:1px solid #d3dbe7;border-radius:8px;padding:11px 16px;background:#faf9f5;margin:14px 0}.original label{font-size:11px;display:block;color:#797166;margin-bottom:5px}.original>div{font-size:17px;line-height:1.8}
    .meaning{padding:13px 17px;background:#f4f7fb;border-radius:8px;margin:14px 0}.meaning>b{font-size:12px;color:#2563eb}.meaning p{margin:5px 0}
    .check{border-left:3px solid #8ba1bc;padding:9px 15px;margin:14px 0;background:#f8fafc}.check>b{font-size:12px;color:#536f8e}.check p{margin:5px 0}
    .photo{position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;background:#f3f2ef;margin:12px 0}.photo img{position:absolute;width:56.25%;height:auto;left:21.875%;top:-38.8889%;transform:rotate(-90deg)}
    .hand{font-size:11px;color:#846d40;border-top:1px solid #e6e0d6;padding-top:8px;margin-top:16px}.source{bottom:15mm}
    #sheet1 table,#sources table{font-size:12px}#sheet1 td,#sheet1 th{padding:9px}
    #questions .check{margin:11px 0;padding:8px 15px}#questions .check p{margin:3px 0}
    #integral td,#integral th{padding:9px 11px}#swap_sum .decision{padding:12px 20px}
    '''
    sections=[]
    for i,p in enumerate(PAGES,1):
        hand=f'<div class="hand">对照：{p["hand"]}</div>' if p['hand'] else ''
        sections.append(f'<section class="page" id="{p["key"]}"><header><div class="eyebrow"><span>手稿 × 实验 · 从加减乘除读起</span><span>{i:02d} / 一次理解一步</span></div><h1>{p["title"]}</h1><p class="sub">{p["subtitle"]}</p></header><main>{p["body"]}{hand}</main><aside class="source">{p["source"]}</aside><footer class="footer"><span>qos_storage_sim · 2026-09-13 · 手稿融合修订版</span><span>{i} / {len(PAGES)}</span></footer></section>')
    js='''<script>window.addEventListener('load',async()=>{await document.fonts.ready;const rows=[...document.querySelectorAll('.page')].map((p,i)=>{const r=p.getBoundingClientRect(),m=p.querySelector('main').getBoundingClientRect(),s=p.querySelector('.source').getBoundingClientRect();return{page:i+1,key:p.id,content_bottom:m.bottom-r.top,source_top:s.top-r.top,overlap:m.bottom>s.top-10}});document.documentElement.dataset.layoutAudit=JSON.stringify(rows);});</script>'''
    out=HERE/(STEM+'.html')
    out.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><title>手稿与实验：零基础融合教程</title><style>'+css+'</style></head><body>'+''.join(sections)+js+'</body></html>')
    (HERE/'fused_page_content.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2)+'\n')
    (HERE/'layerwise_example_checks.json').write_text(json.dumps(LAYERWISE,ensure_ascii=False,indent=2)+'\n')
    for name in re.findall(r'<img src="([^"]+)"',out.read_text()):assert (HERE/name).exists(),name
    browser=['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-pdf-header-footer','--virtual-time-budget=3000']
    dom=subprocess.run(browser+['--dump-dom',out.as_uri()],capture_output=True,text=True,check=True).stdout
    class Parser(HTMLParser):
        rows=None
        def handle_starttag(self,tag,attrs):
            if tag=='html' and 'data-layout-audit' in dict(attrs):self.rows=json.loads(dict(attrs)['data-layout-audit'])
    parser=Parser();parser.feed(dom)
    assert parser.rows and len(parser.rows)==len(PAGES)
    (HERE/'fused_layout_audit.json').write_text(json.dumps(parser.rows,ensure_ascii=False,indent=2)+'\n')
    bad=[r for r in parser.rows if r['overlap']]
    assert not bad, bad
    pdf=HERE/(STEM+'.pdf')
    subprocess.run(browser+[f'--print-to-pdf={pdf}',out.as_uri()],capture_output=True,text=True,check=True)
    reader=PdfReader(pdf);assert len(reader.pages)==len(PAGES)
    checks=[]
    for i,p in enumerate(reader.pages,1):
        value=p.extract_text();assert len(value)>100 and f'{i} / {len(PAGES)}' in value
        checks.append(dict(page=i,text_characters=len(value),footer_verified=True))
    writer=PdfWriter(clone_from=reader)
    for i,p in enumerate(PAGES):writer.add_outline_item(p['title'],i)
    writer.add_metadata({'/Title':'L1 / L2 / L3：手稿与实验，零基础融合教程','/Author':'qos_storage_sim 项目分析','/Subject':'逐项解释手稿，讲清求和积分、模型条件及真实 NPU 实验'})
    temp=pdf.with_suffix('.building.pdf')
    with temp.open('wb') as f:writer.write(f)
    temp.replace(pdf)
    sources=[ROOT/'docs/1.jpg',ROOT/'docs/2.jpg',ROOT/'data',STUDY/'formula_review/accounting_checks.json',STUDY/'formula_review/handoff_checks.json',HERE/'manuscript_inventory.json',HERE/'layerwise_example.py',HERE/'layerwise_example_checks.json']
    (HERE/'fused_pdf_checks.json').write_text(json.dumps({'passed':True,'no_new_simulation':True,'page_count':len(PAGES),'manuscript_item_count':len(coverage),'all_manuscript_items_mapped':True,'pages':checks,'bookmarks':len(PAGES),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf),'pages':len(PAGES),'layout_passed':True},ensure_ascii=False))


if __name__=='__main__':
    publish()
