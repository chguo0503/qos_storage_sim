#!/usr/bin/env python3
"""Publish verified comparison PNGs into the original offline HTML index."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
from bs4 import BeautifulSoup
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = ROOT/'template/qos_experiments_20260919'
DEST = TEMPLATE/'od_baseline_comparison'
STRATEGIES = ['asu_baseline','od_baseline','once']
NAMES = {'asu_baseline':'ASU baseline','od_baseline':'OD baseline','once':'流量分配（Once）'}
LABELS = {'full':'full（32卡 / 3盘）','semi':'semi（32卡 / 3盘）','near35':'near35（32卡 / 3盘）',
          'ab128_random':'旧 A/B Random（32卡 / 3盘）','ab128_ordered':'旧 A/B Ordered（32卡 / 3盘）',
          'sensitivity20k_076':'sensitivity20k_076（8卡 / 1盘）',
          **{g:f'{g}（8卡 / 1盘）' for g in ('XY12_32','XY12_24','XY12_20','XY12_16','X16')}}
GROUPS=list(LABELS)


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect_rows():
    rows=[]
    for r in read(HERE/'ab128/summary.json'):
        rows.append(dict(scenario=r['scenario'],strategy=r['strategy'],U_warm_percent=r['U_percent'],
                         SLO_warm_seedmean_percent=r['slo_1p5_percent'],SLO_cdf_percent=r['slo_1p5_percent'],
                         cdf_cohort='warm admissions [2,4)s',seeds='7',num_npu=32,num_ssu=3,capacity='40 GiB/s'))
    for r in read(HERE/'diverse/summary.json')['rows']:
        rows.append(dict(scenario=r['scenario'],strategy=r['strategy'],U_warm_percent=r['U_percent'],
                         SLO_warm_seedmean_percent=r['slo_1p5_percent'],SLO_cdf_percent=r['slo_1p5_percent'],
                         cdf_cohort='warm admissions [2,4)s',seeds=','.join(map(str,r['seeds'])),num_npu=32,num_ssu=3,capacity='40 GiB/s'))
    for r in read(HERE/'formula_sensitivity/summary.json')['rows']:
        rows.append(dict(scenario=r['group'],strategy=r['strategy'],U_warm_percent=r['U_warm_percent'],
                         SLO_warm_seedmean_percent=r['SLO_warm_1p5_seed_mean_percent'],SLO_cdf_percent=r['SLO_full_1p5_percent'],
                         cdf_cohort='all input requests',seeds=r['seeds'],num_npu=8,num_ssu=1,capacity=r['capacity']))
    rows.sort(key=lambda r:(GROUPS.index(r['scenario']),STRATEGIES.index(r['strategy'])))
    assert len(rows)==33 and len({(r['scenario'],r['strategy']) for r in rows})==33
    return rows


def main():
    assert read(HERE/'ab128/checks.json')['all_checks_passed']
    assert read(HERE/'diverse/validation.json')['status']=='complete'
    assert read(HERE/'diverse/render_checks.json')['status']=='complete'
    formula=read(HERE/'formula_sensitivity/summary.json')
    assert len(formula['rows'])==18 and len(formula['figures'])==7
    DEST.mkdir(exist_ok=True)
    published=[]
    for group in ('ab128','diverse','formula_sensitivity'):
        out=DEST/group;out.mkdir(exist_ok=True)
        for p in sorted((HERE/group/'figures').glob('*.png')):
            shutil.copyfile(p,out/p.name)
            with Image.open(out/p.name) as im:im.verify()
            published.append(dict(relative_path=str((out/p.name).relative_to(TEMPLATE)),bytes=p.stat().st_size,sha256=sha(p)))
        for name in ('summary.csv','summary.json','checks.json','render_checks.json','validation.json','verification.json','audit_final.json','original_source_audit.json','source_provenance.json'):
            p=HERE/group/name
            if p.exists():shutil.copyfile(p,out/name)
    assert len(published)==20
    rows=collect_rows();csvout(DEST/'summary.csv',rows);write(DEST/'summary.json',rows)
    bykey={(r['scenario'],r['strategy']):r for r in rows}
    lines=['# 原输入增加 OD baseline：结果对比','',
           '共完成 27 次新增 OD 仿真。另有 7 次原 ASU 一致性复跑，均核对完整请求和层时间。原图中的 Baseline/FIFO 对应 ASU baseline。',
           '', 'OD：每卡在每盘独占一个 QoS Path，CIR = 盘带宽 / NPU 数；PIR 不限，允许借用空闲带宽。',
           '', '**下表每格是：NPU 利用率 / TTFT SLO×1.5 达标率，单位均为 %。两者均采用原 warm [2,4) 秒口径；多种子先逐种子计算再等权平均。**',
           '', '| 输入 | 种子 | ASU baseline | OD baseline | 流量分配（Once） |','|---|---|---:|---:|---:|']
    table=[]
    for group in GROUPS:
        rr=[bykey[group,p] for p in STRATEGIES]
        values=[f'{r["U_warm_percent"]:.2f} / {r["SLO_warm_seedmean_percent"]:.2f}' for r in rr]
        lines.append(f'| {LABELS[group]} | {rr[0]["seeds"]} | '+' | '.join(values)+' |')
        table.append([LABELS[group],rr[0]['seeds'],*values])
    lines+=['','**CDF 口径单独保留原图定义。** full、semi、near35、旧 A/B 为 warm 窗口内接纳请求，并跟踪至完成；五组公式 AB 与 sensitivity 为全输入请求（含启动阶段）。后两类 CDF 的 SLO 数字可能与上表不同。',
            '', '| 全输入 CDF | ASU SLO×1.5 | OD SLO×1.5 | Once SLO×1.5 |','|---|---:|---:|---:|']
    for group in ('sensitivity20k_076','XY12_32','XY12_24','XY12_20','XY12_16','X16'):
        lines.append('| '+group+' | '+' | '.join(f'{bykey[group,p]["SLO_cdf_percent"]:.2f}%' for p in STRATEGIES)+' |')
    lines+=['','说明：',
            '', '- TTFT 是接纳至 prefill 完成的代理指标，不包括接纳前排队；SLO 基准是该请求 8 层纯计算时间。',
            '- 策略会改变接纳时刻，所以 warm CDF 的请求集合可以不同；完整输入、每卡顺序、提交 seed、放置均相同。',
            '- near35 是平均欠载、允许局部过载。负载名称沿用原输入组，不意味着 OD 运行中的逐时刻状态比例与 ASU 相同。',
            '- 五组公式 AB 的盘速为 40 GB/s（十进制），其他组为 40 GiB/s。保留各组原值，不在组间偷换容量。',
            '- 旧 A/B 与 full/semi/near35 保留冻结条带放置；公式 AB 与 sensitivity 保留原 Ring Hash 和精确尾块。',
            '- 层平均带宽的蓝线是完整内部层周期平均，灰区为请求交界或窗口截断；两个整窗带宽均值相除不等于 NPU 利用率。',
            '', '文件入口：', '',
            '- [更新后的图片索引](../index.html) · [原索引备份](../index.before_od_baseline.html)',
            '- [数值 CSV](summary.csv) · [数值 JSON](summary.json) · [新增 PNG 清单](generated_images.csv)',
            '- [运行代码、原始结果和复核](../../../results/template_od_baseline_20260919/README.md)',
            '', '原实验图片和结果没有覆盖。此次不生成八卡 NQL 辨认、简化八卡带宽、固定并发利用率、等待分解和 sensitivity 配对带宽图；索引中也移除了这些图的展示入口。', '']
    (DEST/'README.md').write_text('\n'.join(lines))
    csvout(DEST/'generated_images.csv',published)
    backup=TEMPLATE/'index.before_od_baseline.html'
    if not backup.exists():shutil.copyfile(TEMPLATE/'index.html',backup)
    soup=BeautifulSoup(backup.read_text(),'html.parser')
    soup.title.string='QoS 实验：ASU / OD / 流量分配'
    soup.h1.string='相同输入：ASU / OD / 流量分配'
    intro=soup.h1.find_next_sibling('p')
    intro.clear()
    intro.append('已新增 27 次 OD 仿真并通过原 ASU 一致性复跑核验。请求、每卡顺序、放置、seed 和统计口径均沿用各自原实验。原图中的 Baseline/FIFO 即 ASU baseline；Once 即流量分配。')
    p=soup.new_tag('p')
    p.append('OD 每卡每盘独占一个 Path、等额 CIR，允许借用空闲带宽。')
    a=soup.new_tag('a',href='od_baseline_comparison/README.md');a.string='查看完整数值、统计口径和复核说明';p.append(a)
    intro.insert_after(p)
    for id_ in ('s07','s08'):
        soup.select_one('#'+id_).decompose()
        soup.select_one(f'nav a[href="#{id_}"]').decompose()
    for img in list(soup.select('img')):
        if Path(img['src']).name=='sensitivity20k_bandwidth_pair.png':img.find_parent('figure').decompose()
    prefixes={'ab128':'od_baseline_comparison/ab128/','diverse':'od_baseline_comparison/diverse/','formula':'od_baseline_comparison/formula_sensitivity/'}
    replacements={
        'full_random_ttft_ratio_cdf_two_strategies.png':prefixes['diverse']+'full_random_ttft_ratio_cdf_three_strategies.png',
        'semi_random_ttft_ratio_cdf_two_strategies.png':prefixes['diverse']+'semi_random_ttft_ratio_cdf_three_strategies.png',
        'near35_ttft_slo_cdf.png':prefixes['diverse']+'near35_random_ttft_ratio_cdf_three_strategies.png',
        'AB_random_TTFT_CDF_SLO_1_1p5.png':prefixes['ab128']+'AB_random_TTFT_CDF_three_strategies.png',
        'sensitivity20k_timeline_pair.png':prefixes['formula']+'sensitivity20k_076_timeline_three_strategies.png',
        'sensitivity20k_ttft_cdf_pair.png':prefixes['formula']+'sensitivity20k_076_ttft_cdf_three_strategies.png',
        **{f'ttft_slo_all_{i:02d}_{g}.png':prefixes['formula']+f'formula_{i:02d}_{g}_ttft_cdf_three_strategies.png'
           for i,g in enumerate(('XY12_32','XY12_24','XY12_20','XY12_16','X16'),1)}}
    def figure(src,caption):
        node=soup.new_tag('figure')
        a=soup.new_tag('a',href=src);a.append(soup.new_tag('img',src=src,alt=caption,loading='lazy'));node.append(a)
        c=soup.new_tag('figcaption');c.string=caption;node.append(c)
        return node
    for img in list(soup.select('img')):
        old=Path(img['src']).name
        if old in replacements:
            img.find_parent('figure').replace_with(figure(replacements[old],'三策略对照 · '+Path(replacements[old]).name))
        elif old.startswith('NEW_'):
            img.find_parent('figure').figcaption.append(' · ASU / OD / Once 共用的冻结输入序列；与策略无关。')
    extra={
        's01':[(prefixes['diverse']+'full_od_baseline_per_ssu_seed7.png','OD：逐盘需求与实际供给（seed 7）')],
        's02':[(prefixes['diverse']+'semi_od_baseline_per_ssu_seed7.png','OD：逐盘需求与实际供给（seed 7）'),(prefixes['diverse']+'semi_od_baseline_32npu_timeline_seed7.png','OD：全部 32 卡计算时序（seed 7）')],
        's04':[(prefixes['ab128']+'AB_ordered_TTFT_CDF_three_strategies.png','Ordered：三策略 TTFT CDF')]+[(prefixes['ab128']+f'od_baseline_{o}_timeline.png',f'OD {o.title()}：32 卡计算时序') for o in ('random','ordered')]+[(prefixes['ab128']+f'od_{o}_all_32npu_layer_average.png',f'OD {o.title()}：32 卡层平均带宽') for o in ('random','ordered')],
        's05':[(prefixes['diverse']+'near35_od_baseline_per_ssu_seed7.png','OD：逐盘需求与实际供给（seed 7）')]}
    for section,figures in extra.items():
        for src,caption in figures:soup.select_one('#'+section).append(figure(src,caption))
    para=soup.select_one('#s03').find('p');para.clear();para.append('原 sensitivity20k_076 输入。保留原 Ring Hash、精确尾块、8 NPU / 1 SSU × 40 GiB/s；三策略时序统计 [2,4) 秒，CDF 按原图取全输入。原约束是普通需求欠载、长短请求交界预取豁免；本页不生成配对带宽图。')
    formula_p=soup.new_tag('p');formula_p.string='五组均为 8 NPU / 1 SSU × 40 GB/s（十进制）。下方 CDF 沿用全输入请求，包含启动；NPU 利用率汇总仍取 warm [2,4) 秒。'
    soup.select_one('#s06 h2').insert_after(formula_p)
    s=soup.new_tag('section',id='comparison_summary');h=soup.new_tag('h2');h.string='结果汇总：warm [2,4) 秒';s.append(h)
    p=soup.new_tag('p');p.string='每格为 NPU 利用率 / TTFT SLO×1.5 达标率（%）；多种子等权平均。CDF 的全输入与 warm 区别见各图和完整说明。';s.append(p)
    t=soup.new_tag('table');thead=soup.new_tag('tr')
    for name in ('输入','种子','ASU baseline','OD baseline','流量分配（Once）'):
        th=soup.new_tag('th');th.string=name;thead.append(th)
    t.append(thead)
    for row in table:
        tr=soup.new_tag('tr')
        for value in row:
            td=soup.new_tag('td');td.string=value;tr.append(td)
        t.append(tr)
    s.append(t);soup.nav.insert_after(s)
    soup.style.append('table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #dce3ea;padding:9px 10px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9eff7}tr:nth-child(even){background:#f6f8fb}td:nth-child(4){font-weight:650;color:#a74300}')
    output=soup.prettify()
    (TEMPLATE/'index.html').write_text(output)
    missing=[]
    for elem in soup.select('[src],a[href]'):
        target=elem.get('src') or elem.get('href')
        if target.startswith(('http:','https:','#','data:','mailto:')):continue
        if not (TEMPLATE/target.split('#')[0]).exists():missing.append(target)
    assert not missing,missing
    forbidden={'once_8npu_bandwidth_simple.png','fifo_8npu_bandwidth_simple.png','fixed_concurrent_utilization.png','stall_decomposition.png','sensitivity20k_bandwidth_pair.png'}
    assert not any(Path(i['src']).name in forbidden for i in soup.select('img'))
    image_paths=[str(i['src']) for i in soup.select('img')]
    assert len(image_paths)==34 and len(set(image_paths))==34
    for p in published:assert p['relative_path'] in image_paths
    inventory=list(csv.DictReader((TEMPLATE/'file_inventory.csv').open(encoding='utf-8-sig')))
    for original in inventory:
        path=backup if original['relative_path']=='index.html' else TEMPLATE/original['relative_path']
        assert path.stat().st_size==int(original['bytes']) and sha(path)==original['sha256'],original['relative_path']
    checks=dict(status='complete',new_od_runs=27,asu_parity_runs=7,png_count=20,index_image_count=len(image_paths),
                excluded_images_generated=False,all_local_links_exist=True,all_pngs_readable=True,
                original_inventory_files_preserved=len(inventory),original_index_preserved_as_backup=True,
                original_index_sha256=sha(backup),updated_index_sha256=sha(TEMPLATE/'index.html'),
                all_new_pngs_linked=True,summary_rows=len(rows),generated_images=published)
    write(HERE/'delivery_checks.json',checks);write(DEST/'delivery_checks.json',checks)
    print(json.dumps({k:v for k,v in checks.items() if k!='generated_images'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
