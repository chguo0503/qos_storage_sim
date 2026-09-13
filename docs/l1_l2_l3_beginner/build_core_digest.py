#!/usr/bin/env python3
"""Four-page recap of the Saturday experiment, using existing audited results."""
import ast
import hashlib
import html
import json
import statistics
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader, PdfWriter
import build_guide as base

HERE, ROOT, STUDY = base.HERE, base.ROOT, base.STUDY
STEM = '周六实验与NPU利用率_四页核心版'
SOURCE_FILES = []


def read_json(path):
    SOURCE_FILES.append(path)
    return json.loads(path.read_text())


COMPARISON = read_json(STUDY/'comparison.json')
HANDOFF = read_json(STUDY/'formula_review/handoff_checks.json')
VERIFY = read_json(STUDY/'verification.json')
assert HANDOFF['technical_passed'] and all(VERIFY['checks'].values())
SOURCE_FILES.append(ROOT/'data')
raw = ast.literal_eval((ROOT/'data').read_text())
raw = {ast.literal_eval(k) if isinstance(k,str) else k:v for k,v in raw.items()}
AUDITS = {}
for disks in (3,4,6):
    for order,seeds in [('random',(7,19,43)),('ordered',(7,))]:
        for seed in seeds:
            AUDITS[disks,order,seed] = read_json(STUDY/f'validation20s/runs/ssu{disks}_{order}_k1_sync_seed{seed}/baseline/audit/audit.json')


def window(disks,order,seed=7,left=2,right=20):
    return next(w for w in AUDITS[disks,order,seed]['windows']
                if w['start_ms']==left*1000 and w['end_ms']==right*1000)


for row in COMPARISON:
    disks=row['num_ssu']
    for end,prefix in [(4,'warm'),(20,'long')]:
        average=statistics.mean(window(disks,'random',s,right=end)['device_utilization_percent'] for s in (7,19,43))
        assert abs(average-row[f'random_{prefix}_mean'])<1e-9
        assert abs(window(disks,'ordered',right=end)['device_utilization_percent']-row[f'ordered_{prefix}'])<1e-9
    assert all(window(disks,o,s)['all_32_active'] and window(disks,o,s)['mixed_card_count']==32
               and not window(disks,o,s)['nominal']['strict_underload']
               for o,seeds in [('random',(7,19,43)),('ordered',(7,))] for s in seeds)

row4=next(r for r in COMPARISON if r['num_ssu']==4)
row6=next(r for r in COMPARISON if r['num_ssu']==6)
classes=window(4,'ordered')['classes']
assert classes['B']['internal_stall_ms']==0
A=HANDOFF['A_local']; H=HANDOFF['B_first_layer']
delta=row4['random_long_mean']-row4['ordered_long']
table=base.table


def code(value):
    return '<pre>'+html.escape(value)+'</pre>'


def box(title,body,cls=''):
    return f'<div class="box {cls}"><b>{title}</b>{body}</div>'


CYCLE_FORMULA=code('U_cycle = C / max(C, V / b)\n        = min(1, b / B)')


profiles=[]
for label,key,desc in [('A',(128,256),'大读取·短计算'),('B',(32,4096),'小读取·长计算')]:
    bw,c_us,_,v_gib=raw[key]
    profiles.append([f'<b>{label}</b> · {desc}',f'{key[0]}K',str(key[1]),f'{v_gib*1024:.2f}',f'{c_us/1000:.3f}',f'{bw:.2f}'])

PAGES=[dict(key='experiment',title='同一批请求，顺序会改变利用率',
subtitle='周六实验 · 2026-09-12 · 32 NPU / 3、4、6 SSU',
body=f'''
{box('最值得记住的实测结果',f'<p>4 盘、同一 <b>2–20 秒</b>窗口：Random <strong>{row4["random_long_mean"]:.2f}%</strong>，Ordered <strong>{row4["ordered_long"]:.2f}%</strong>，相差 <b>{delta:.2f} 个百分点</b>。</p>','lead')}
<p class="label">输入直接取自 data；K=1024，miss 是新增计算的 token 数。</p>
{table(['画像','总输入','miss','每层读<br>MiB','每层算<br>ms','V/C<br>GiB/s'],profiles)}
<p>每卡固定 <b>40A＋80B</b>，每请求 8 层、batch=1。32 卡共 3840 条请求，全部在 t=0 到达；每盘 40 GiB/s，每卡接收链路 50 GiB/s。</p>
<div class="split"><div><b>Random</b><p>每张卡独立打乱自己的同一批请求。</p></div><div><b>Ordered</b><p>每张卡都按 A→B→B 重复，起点相同；没有运行时同步屏障。</p></div></div>
<p class="label">长验证窗口统一为 [2,20) 秒，T=18 秒；以下全部是 Baseline。</p>
{table(['SSU 数','总容量<br>GiB/s','Random<br>平均利用率','Ordered<br>平均利用率'],[[str(r['num_ssu']),str(r['capacity']),f'{r["random_long_mean"]:.2f}%',f'{r["ordered_long"]:.2f}%'] for r in COMPARISON])}
<p class="small">Random 是 seed 7、19、43 的算术平均；Ordered 是 seed 7。每格长窗内所有卡始终有请求，且每卡都计算过 A、B；并非每个更短子窗都逐卡包含两类。</p>
<p><b>4 盘的低利用率延续到长窗；6 盘会恢复。</b>6 盘 Ordered 从 2–4 秒的 {row6['ordered_warm']:.2f}%，逐步恢复到后期约 100%，不能把最初低点当成持续结论。</p>
''',source='[S1] data；comparison.json；12 格 validation20s 的 audit.json。'),
dict(key='stall',title='直接原因：下一层数据没赶上计算',
subtitle='真实逐层时序 · 4 盘 Ordered / seed 7 · 不是教学拼接图',
body=f'''
<p>当前层一开始计算，就预取下一层。<b>只有下一层数据到达 NPU 后，才能继续计算。</b>下面是 NPU 0 的一个真实 A 请求片段。</p>
<figure><img src="assets/local_A_cycle.svg" alt="A 请求一轮计算6.024ms，随后等待28.284ms；合计34.308ms"><figcaption>绝对时间约 2.180–2.214 秒。图中第 2、3 层对应日志 L1、L2。</figcaption></figure>
{table(['这一轮发生了什么','实际时间'],[['当前层的纯计算',f'{A["C_ms"]:.3f} ms'],['下一层从读取发出到 NPU 收齐',f'{A["model_R_ms"]:.3f} ms'],['暴露的 IO 等待',f'{A["model_R_ms"]-A["C_ms"]:.3f} ms']])}
{code(f'该局部周期利用率 = 6.024 / 34.308 = {A["actual_common_window_fleet_U_percent"]:.2f}%')}
<p class="small">17.56% 只对应这一个约 34.3 ms 的局部周期，不是第 1 页的长窗利用率。该小段 32 卡均在 A，4 块盘都忙；整段算出的每卡平均服务速率为 5 GiB/s。</p>
{box('为什么 A 更容易等？','<p>A 的内部层预取，要在前层约 6 ms 的计算期间读回 175.66 MiB。很多卡一起进入 A，读取集中到各盘 FIFO，下一层便可能来不及到齐。B 每层算得更久、读得更少，内部层的等待通常能被计算覆盖。</p>')}
<p><b>A→B 首层是例外：</b>B 的首层在前一个 A 的末层计算时预取，预算只有 A 的 {H['budget_C_ms']:.3f} ms。日志中一次读取历时 {H['actual_R_ms']:.3f} ms，暴露等待 <b>{H['stall_ms']:.3f} ms</b>。</p>
<p class="small">B 自己后续层的 28.593 ms 计算，不能倒过来替首层提供预取时间。其他卡的长计算本身也不占用本卡；共享争用发生在读取服务上。</p>
''',source='[S2] formula_review/handoff_checks.json；local_A_cycle.svg。读取历时含排队、服务和到 NPU 的传输。'),
dict(key='formulas',title='只记三个公式，分清两种时间',
subtitle='全部用纯文本；不需要 μ、积分或长推导',
body=f'''
{box('1 · 实验报告使用：真实窗口平均',code('U_window = 窗口内所有卡累计的计算时间 / (N * T)')+'<p>N=32。t 是某一时刻，例如 3.2 秒；T 是窗口长度：2–4 秒的 T=2 秒，2–20 秒的 T=18 秒。跨边界的计算，只计入窗口内部分。</p>')}
<p class="small">若只看某个 t，真实瞬时利用率就是“此时正在计算的卡数 / N”。窗口平均则累计整段时间；不能用不同结束时刻作分母，冒充同一暖窗对照。</p>
{box('2 · 描述请求：参考带宽需求',code('B_i(t) = V_i(t) / C_i(t)')+'<p>i 是 NPU 编号。V 是当前画像完整的一层读取量；C 是一层纯计算时间，均不是剩余量。画像不变时，B 保持不变；切换请求时可以变化。换算带宽时，统一用 GiB 和秒。</p>')}
<p>请求 A 的参考需求约 <b>28.48 GiB/s</b>，请求 B 约 <b>1.31 GiB/s</b>。计算时间直接查 data；若写 t_c=C/m，它是每 token 的平均分摊时间，会随总长度和 miss 数变化。</p>
{box('3 · 理想模型使用：周期平均',CYCLE_FORMULA+'<p>这里 b 是画像固定时持续可获得的带宽；假设计算与下一层读取理想重叠，忽略启动和结尾。它估计一张卡持续运行的平均水平。</p>')}
<p><b>不要把瞬时读取速率直接代入，就当成真实瞬时利用率。</b>数据提前读完后，实际读取速率可以是 0，卡仍然能计算。混合请求的任意短窗，也不保证与模型相等。</p>
<p class="small">第 2 页选定的完整局部周期，可以核对 5 / 28.48 ≈ 17.56%；这不允许把同一个估计延伸到整段混合窗口。所有比值先得到 0–1，再显示为百分比。</p>
''',source='公式为条件明确的定义或理想模型；实际成本与局部核对来自 [S1]、[S2]。'),
dict(key='conclusions',title='调度要管“分多少”，也要管“何时读”',
subtitle='周六的直接对照是请求顺序；IO 策略效果仍需同输入验证',
body=f'''
{table(['层次','它决定什么','周六对照'],[['L1','请求分给哪张 NPU','保留同一逐卡人口与落盘'],['L2','每张卡按什么顺序执行请求','Random 与 Ordered 的差别'],['L3','已提交的 IO 先服务谁','始终使用 Baseline：每盘 Path0 FIFO']])}
{box('预算与先后，是两个问题','<p><b>分多少：</b>在固定画像、单一带宽池、可自由分配且只求平均 U 的模型里，未达到 100% 时，小 B 的卡每增加一份带宽能支撑更多计算；达到 100% 后不再增加。C 已包含在 V/C 中，不要求计算时长相近。</p><p><b>何时读：</b>下一笔 IO 还要看什么时候必须读齐。给小 B 的卡留足预算，不代表每笔都让它先读；高 B 的卡当前层更紧急时，可以先服务它。</p>')}
<p>本次请求清单预先固定，但后续层 IO 随计算推进才提交。因此，服务顺序会改变未来读取出现的时间；仅凭“外部输入固定”，不能判断利用率不受调度影响。</p>
<div class="boundary"><b>本次已经支持什么？</b><p>相同 A/B 请求，集中进入“大读取·短计算”阶段会暴露等待。4 盘 Ordered 在 2–20 秒内明显较差；6 盘后期读取逐渐错开、利用率恢复。</p><b>还不能推出什么？</b><p>按完整配额且持续计算估出的平均需求约 124.91 GiB/s，低于 4 盘的 160 GiB/s；但所有长验证都曾有逐盘名义需求超限，<b>不是严格逐盘逐时刻欠载的反例</b>。本次 27 格全是 Baseline，<b>没有这批输入的 Once 对照</b>，也没有 A/B 不能互借带宽的配额。</p></div>
<p><b>整机 U 不能代替请求体验。</b>4 盘 Ordered 长窗：A 类利用率仅 {classes['A']['conditional_utilization_percent']:.2f}%，B 类 {classes['B']['conditional_utilization_percent']:.2f}%（各类计算 / 该类占用卡时间）。评估策略还应看分类型结果和 TTFT/SLO；接纳后计时不等于外部到达后的端到端 TTFT。</p>
<div class="references"><b>原始来源</b><br>[S1] <a href="../../results/baseline_ab128_32_ratio12_20260912/report.md">周六实验报告与输入、结果</a> · <a href="../../results/baseline_ab128_32_ratio12_20260912/comparison.json">三种子汇总</a><br>[S2] <a href="../../results/baseline_ab128_32_ratio12_20260912/formula_review/README.md">公式、局部时序与适用条件核验</a><br>完整研究：results/baseline_ab128_32_ratio12_20260912。本文只整理已有证据，没有启动新实验。</div>
''',source='[S1] report.md、12 格长验证 audit.json；verification.json 核验 27 格均为 Baseline。')]


def publish():
    css=base.CSS+'''
    body{font-size:14px;line-height:1.7}h1{font-size:27px}.sub{font-size:13px;margin-bottom:18px}
    p{margin:11px 0}.label{font-size:12px;color:#536a85;margin:13px 0 5px}
    table{font-size:12px;margin:12px 0}td,th{padding:9px 10px}
    .box{padding:13px 16px;border:1px solid #dce5f1;border-radius:10px;background:#f7f9fc;margin:13px 0}
    .box>b{font-size:15px;color:#2457a6}.box p{margin:7px 0 0}.box.lead{background:#edf4ff;border-color:#d8e6ff}
    .lead strong{font-size:23px;color:#2359af}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
    .split>div{padding:10px 15px;background:#f5f8fb;border-radius:8px}.split p{margin:5px 0;font-size:12px}
    pre{font-family:Consolas,'DejaVu Sans Mono',YaHei,monospace;font-size:13px;font-weight:600;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f8;border-radius:7px;padding:12px 14px;margin:10px 0}
    .box pre{background:#eaf0f8}.small{font-size:11px;line-height:1.65}
    figure{margin:16px 0}figcaption{font-size:10px}.boundary{border-left:3px solid #c58a3e;background:#fffaf1;padding:12px 16px;margin:13px 0}.boundary p{margin:5px 0 10px}
    .references{font-size:10px;line-height:1.7;background:#f6f8fb;padding:10px 13px;border-radius:7px;margin-top:14px}a{color:#2359af}
    #formulas .box{margin:14px 0}#conclusions{font-size:13px}#conclusions .box p{font-size:13px}
    '''
    sections=[]
    for i,p in enumerate(PAGES,1):
        sections.append(f'<section class="page" id="{p["key"]}"><header><div class="eyebrow"><span>周六实验 × 核心公式 · 四页速读</span><span>{i:02d} / 04</span></div><h1>{p["title"]}</h1><p class="sub">{p["subtitle"]}</p></header><main>{p["body"]}</main><aside class="source">{p["source"]}</aside><footer class="footer"><span>qos_storage_sim · 实验 2026-09-12 / 整理 2026-09-13</span><span>{i} / 4</span></footer></section>')
    js='''<script>window.addEventListener('load',async()=>{await document.fonts.ready;let rows=[...document.querySelectorAll('.page')].map((p,i)=>{let r=p.getBoundingClientRect(),m=p.querySelector('main').getBoundingClientRect(),s=p.querySelector('.source').getBoundingClientRect();return{page:i+1,key:p.id,content_bottom:m.bottom-r.top,source_top:s.top-r.top,overlap:m.bottom>s.top-10}});document.documentElement.dataset.layoutAudit=JSON.stringify(rows)});</script>'''
    out=HERE/(STEM+'.html')
    out.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>周六实验与 NPU 利用率：四页核心版</title><style>'+css+'</style></head><body>'+''.join(sections)+js+'</body></html>')
    browser=['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-pdf-header-footer','--virtual-time-budget=2500']
    dom=subprocess.run(browser+['--dump-dom',out.as_uri()],capture_output=True,text=True,check=True).stdout
    class Parser(HTMLParser):
        rows=None
        def handle_starttag(self,tag,attrs):
            if tag=='html' and 'data-layout-audit' in dict(attrs):self.rows=json.loads(dict(attrs)['data-layout-audit'])
    parser=Parser();parser.feed(dom)
    assert parser.rows and len(parser.rows)==4
    assert not any(r['overlap'] for r in parser.rows),parser.rows
    pdf=HERE/(STEM+'.pdf')
    subprocess.run(browser+[f'--print-to-pdf={pdf}',out.as_uri()],capture_output=True,text=True,check=True)
    reader=PdfReader(pdf);assert len(reader.pages)==4
    checks=[]
    for i,p in enumerate(reader.pages,1):
        s=p.extract_text();assert len(s)>200 and f'{i} / 4' in s
        checks.append(dict(page=i,text_characters=len(s),footer_verified=True))
    text_all='\n'.join(p.extract_text() for p in reader.pages)
    for value in ['98.90%','71.34%','64.22%','97.24%','83.33%']:
        assert (value in text_all) == (value!='83.33%'),value
    writer=PdfWriter(clone_from=reader)
    for i,p in enumerate(PAGES):writer.add_outline_item(p['title'],i)
    writer.add_metadata({'/Title':'周六实验与 NPU 利用率：四页核心版','/Author':'qos_storage_sim 项目分析','/Subject':'32NPU A/B 输入、逐层预取、核心公式及结论边界'})
    tmp=pdf.with_suffix('.building.pdf')
    with tmp.open('wb') as f:writer.write(f)
    tmp.replace(pdf)
    SOURCE_FILES.extend([HERE/'assets/local_A_cycle.svg',Path(__file__).resolve()])
    audit=dict(passed=True,page_count=4,bookmarks=4,no_new_simulation=True,
               experiment_date='2026-09-12 (Saturday)',compiled_date='2026-09-13',
               summary_recomputed_from_12_audits=True,main_table='Random seeds 7/19/43 arithmetic mean; Ordered seed7; [2,20) seconds.',
               local_figure='4SSU Ordered seed7; one real A-layer cycle; not whole-window U.',
               layout=parser.rows,pdf_checks=checks,pages=PAGES,
               sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_FILES},
               pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    (HERE/'core_digest_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf),'pages':4,'layout_passed':True},ensure_ascii=False))


if __name__=='__main__':
    publish()
