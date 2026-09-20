#!/usr/bin/env python3
"""Create a reviewable report and comparison plot from completed native runs."""
import ast
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from run_fixed128_32 import profile

ROOT=Path(__file__).resolve().parent;E=ROOT/'results/random_multitype_search_20260914'
D=ROOT.parent.parent/'random_multitype_deliverables'
FONT=ROOT.parent.parent/'fifo_figure_reference/fonts/KaiXinSong-Charts.ttf'

def main():
    D.mkdir(exist_ok=True)
    rows=json.loads((E/'native_results.json').read_text())
    groups=json.loads((E/'native_profile_results.json').read_text())
    aggregates=json.loads((E/'native_seed_summary.json').read_text())
    def one(name,seed=7,policy='fifo',stage=None):
        return next(r for r in rows if r['name']==name and r['seed']==seed and r['policy']==policy and (r['stage']==stage if stage else r['window_right_ms']==4000))
    def agg(name):return next(r for r in aggregates if r['name']==name and r['policy']=='fifo' and r['window_right_ms']==4000)
    chosen='robust_036';a=agg(chosen);assert a['n_seeds']==7
    chosen_case=ROOT/one(chosen)['case'];meta=json.loads((chosen_case/'metadata.json').read_text())
    table=ast.literal_eval((ROOT/'data').read_text())
    profile_table=[]
    purpose={'G0':'短计算窗口、小读取，容易暴露等待','G1':'计算较久、读取较少，降低输入平均需求','G2':'大读取、较短计算，使长读取频繁进入队列'}
    for g in meta['specification']['groups']:
        p=profile(table,int(g['total_k']*1024)-g['nql'],g['nql']);pr=meta['profiles'][g['id']]
        profile_table.append(f"|{g['id']}|{g['total_k']}K|{g['nql']}|{p['read_gib']*1024:.4f}|{p['compute_us']/1000:.4f}|{p['B_gib_s']:.4f}|{meta['lane_checks'][0]['profile_group_counts'][g['id']]}|{purpose[g['id']]}|")
    seed_table=[]
    for seed in [7,11,23,41,97,101,211]:
        r=one(chosen,seed);gr={g['profile_group']:g for g in groups if g['case']==chosen and g['seed']==seed and g['policy']=='fifo'}
        seed_table.append(f"|{seed}|{r['U_percent']:.4f}%|{gr['G0']['U_percent']:.4f}%|{gr['G1']['U_percent']:.4f}%|{gr['G2']['U_percent']:.4f}%|{r['slo_1p5_percent']:.4f}%|")
    comparisons=[]
    for name in ['fixed_recommended','strict_multitype','fixed_three','extended_more_headroom',chosen]:
        r=agg(name)
        comparisons.append(f"|{name}|{r['n_seeds']}|{r['input_load_mean']*100:.2f}%|{r['U_mean_percent']:.4f}%|{r['U_min_percent']:.4f}–{r['U_max_percent']:.4f}%|{'是' if r['nominal_static_underload_all_seeds'] else '否'}|")
    paired=[]
    for name in ['fixed_three','extended_more_headroom',chosen]:
        f=one(name);o=one(name,policy='once')
        paired.append(f"|{name}|{f['U_percent']:.4f}%|{o['U_percent']:.4f}%|{o['U_percent']-f['U_percent']:+.4f} pp|{f['slo_1p5_percent']:.4f}%|{o['slo_1p5_percent']:.4f}%|")
    ranges='；'.join(f"{g}: {v['nql_range'][0]}–{v['nql_range'][1]}" for g,v in meta['profiles'].items())
    mmean=meta['input_demand']['per_ssu_gib_s']
    short_internal=[]
    for seed in [7,11,23,41,97,101,211]:
        gr=next(g for g in groups if g['case']==chosen and g['seed']==seed and g['policy']=='fifo' and g['profile_group']=='G0')
        short_internal.append(gr['internal_stall_ms']/(gr['internal_stall_ms']+gr['l0_stall_ms'])*100)
    text=f'''# Random 混排下选择什么输入，能让 baseline NPU 利用率较低？

日期：2026-09-14。本文所有结果表均来自原生逐 IO 仿真。近似模型只用于筛选候选，不作为结果。带宽采用 GiB/s，K=1024 tokens。

## 推荐输入与实际结果

推荐 **8 NPU、2 SSU、每卡三类请求独立随机混排**。三个中心仍只使用 32K 和 128K 两种总长度，通过 NQL 区分计算窗口。每卡 G0/G1/G2 数量 **72/36/18，比例4:2:1**，共126条；整机1008条。7个种子原生 FIFO 平均利用率 **{a['U_mean_percent']:.4f}%**，范围 **{a['U_min_percent']:.4f}–{a['U_max_percent']:.4f}%**。其中101、211是未参与候选筛选的追加检验种子。

|画像|总输入|NQL中心|每层读取 MiB|每层计算 ms|V/C GiB/s|每卡数量|作用|
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(profile_table)}

三个中心均是 `data` 原始记录。实际请求为满足每卡 `(总长度,NQL)` 不重复，在中心附近选择不同整数 NQL，计算时间在 `data` 网格内插值，**不是所有请求都直接来自原始测量点**。seed7实际NQL范围：{ranges}。没有长度外推、计算缩放或尾块填充。逐请求完整值见CSV和manifest。

G0和G1虽然总长度相同，计算窗口相差很大，必须分开统计，不能合并成“32K短请求”后判断效果。

## 为什么选择三类？

1. G0自身读取量较小，计算窗口短。单独服务可以供给，但一旦排队容易超过计算窗口。
2. G2每层读取量约是G0的4倍，而计算窗口只有约11.76ms，能较频繁地下发大读取。之前把长NQL提高到3072/4096，会拉长长请求计算时间，使大读取出现更稀疏。
3. G1计算时间较长但读取少，为总体平均需求留出余量；因此不必用极长计算窗口的长请求来稀释平均负载。三个群体形成不同层周期，随机混排下会反复竞争。

这些是输入设计理由，并不保证任意随机顺序都得到同一个利用率，也不能由输入平均欠载推断全部截止时间都可满足。

## 七个随机种子的原生结果

统计公共窗口[2000,4000)ms。全部NPU保持active；每卡输入都混合三类。每组利用率按该组计算时间/该组active时间计算，整机按所有计算时间/(8×窗口长度)计算。

|seed|整机利用率|G0利用率|G1利用率|G2利用率|窗口接纳 TTFT SLO×1.5|
|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_table)}

G0的等待主要在内部层：七种子中，其L1–L7内部Stall占该组全部Stall的比例为 **{min(short_internal):.2f}%–{max(short_internal):.2f}%**。这排除了“只在请求切换首层等待”的解释，但不等于这些等待全部可以通过FIFO重排消除。

7、11、23、41、97也用于近似候选筛选，属于原生复核；101、211在配置冻结后追加，没有参与候选选择，保留这两个结果，无论是否变差。种子同时影响各卡队列排列与邻近NQL抽样。这里不主张生产随机负载或所有种子都有相同收益。

## 欠载口径必须明确

seed7的事前输入理想平均需求：SSU0 **{mmean[0]:.4f}**、SSU1 **{mmean[1]:.4f} GiB/s**，各低于40；总计 **{sum(mmean):.4f}/80 GiB/s**。计算方法为每卡完整输入的总读取量除以总纯计算时间，再按盘合并，未使用Stall后变低的吞吐证明欠载。

**这组是输入平均欠载，允许运行中需求聚集超载。** 任意当前请求组合的V/C上界不在容量内；下一层预取还需检查V_next/C_current。不能将本结果表述成“每一时刻都欠载，却全因FIFO而损失这些利用率”。当前需求、预取参考需求的峰值与越线时间、单请求供给下限和真实盘速率均在`deadline_audit.json`。

实际物理SSD服务由共同2ms字节桶计算，每盘检查≤40GiB/s、NPU接收≤50GiB/s；没有把各NPU不同层周期的平均带宽相加当成瞬时物理供给，也没有通过限幅制造容量合规。

## 对照和没有达到目标的输入

|方案|原生种子数|平均输入负载率|FIFO平均利用率|种子范围|任意当前组合均欠载|
|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

- `fixed_recommended`：原建议128K/2048 +32K/1280，1:1，1SSU。利用率仍约99.63%，不适合展示明显损失。
- `strict_multitype`：32K/1216、64K/1536、128K/1536、200K/1664，数量8:3:1:1，1SSU；所有当前组合V/C在40以内，但FIFO仍约97.80%。
- `fixed_three`：32K/256、32K/2560、128K/512，数量2:4:1，1SSU。seed7约86.46%，五种子均值91.93%，最坏种子不代表一般水平。
- `extended_more_headroom`：32K/128 +200K/3584，数量6:1，1SSU。平均负载约79.5%，但五种子利用率约93.34%，随机波动更大。

如果要求“任意当前组合甚至所有预取窗口都不过载”，目前没有找到稳定低利用率的多画像方案；严格对照仍接近98%–100%。不要把平均欠载的推荐组替代该更强约束。

## 原 Once 的配对结果

下面均为seed7、完全相同请求与顺序；Once保持原5ms周期状态收集、原静态CIR和每层路径选择，没有为了获得收益改策略。

|输入|FIFO利用率|Once利用率|变化|FIFO SLO×1.5|Once SLO×1.5|
|---|---:|---:|---:|---:|---:|
{chr(10).join(paired)}

不能因为baseline变低就假定Once必然提升整机利用率。各画像的代价和收益分别记录在`native_profile_results.csv`。TTFT从NPU接纳到8层完成计时，阈值为8层纯计算时间×1.5；未计入所有请求t=0到达至接纳的队列等待。不同策略的窗口接纳集合可能不同，另有全输入同集合CDF。

## 一个长读取阻塞一个短读取的实测证据

在1SSU三类对照`fixed_three_seed7_fifo`中找到：短请求32K/NQL262的内部L1，计算窗口2.0374ms、自身盘服务1.0656ms；前方单个128K/NQL513的L6连续占盘4.2801ms，使短请求产生3.7924ms Stall。

在真实IO边界3287.553684ms，两者全部IO都已提交。局部先短后长可使短数据最晚3288.622671ms到齐，早于截止3289.110135ms；长数据最晚3292.902729ms到齐，早于截止3298.685208ms。两者合计服务量和SSD最终完成时刻不变，16项检查通过。

这是该局部两任务存在可行次序的证据，不是完整策略重跑，也不是2SSU推荐组每次Stall都可避免的证明。相关图标明“局部离线可行次序”。

## 较长窗口的复核

1SSU三类输入还构造了较长随机队列，观察[2000,8000)ms：FIFO **89.9288%**；同一条长轨迹的2–4s、4–6s、6–8s分别为 **90.9982%、91.9522%、86.8362%**。因此该轨迹的损失没有随层错位单调消失。

这条长轨迹与最初2s实验的请求数量、随机排列及NQL范围不同；只能在该长轨迹内部比较窗口，不应把它与原短队列当作同一轨迹直接拼接。`long_same_trajectory_windows.csv`保留切窗结果。2SSU最终推荐的七种子验证仍使用2s窗口。

## 实现、筛选和复现

共同配置：8NPU、batch1、8层、所有请求t=0到达、固定NPU分配，各卡以`seed+100003×npu_id`独立shuffle，ring hash每盘256虚拟节点。同一KV block各层复用同一盘映射；不同block可跨盘。尾块按真实剩余tokens×1408字节计算。所有NPU保持有请求执行；低利用率未通过低到达率或人为空闲制造。该负载是固定卡、预排队的串行混排，不是泊松到达。

原GitHub 41个文件哈希核验一致；新增实验构建、记录和审计脚本，原baseline/Once核心未改。`fixed_total_compat.py`沿用上一实验的精确尾块适配及回归验证。

筛选过程完整保留：首先用层级FIFO近似模型探索3274组，选10组做seed7原生测试；对两组做5seed和较长窗口复核。发现seed7偏坏后，再从既有候选中取160组按5seed平均/最高利用率筛选，输入生成与原生逐请求序列校验，并用真实ring字节和50GiB/s链路作近似复核，最终只增加`robust_036`这一个配置的5seed原生验证、2个独立追加种子和1次Once配对。近似值只用于选候选，没有混入原生结果表。

解压数据包进入`source`，安装`requirements.txt`，例如：

```bash
python run_random_multitype.py --spec results/random_multitype_search_20260914/approx_robust_native_plan.json --case robust_036 --seed 7 --policy fifo --stage reproduce
python run_random_multitype.py --spec results/random_multitype_search_20260914/approx_robust_native_plan.json --case robust_036 --seed 7 --policy once --stage reproduce
```

需要原样复现时使用保存的spec/manifest。比例4:2:1描述数量比；保存spec的权重12:6:3和队列扩展规则最终生成每卡72/36/18条。仅改变权重公因数或总队列长度，会改变有限随机排列，不能期望相同seed得到完全相同轨迹。

`native_results.csv`包含全部原生运行，`native_seed_summary.csv`包含跨种子汇总，`native_profile_results.csv`按画像分组；各case的manifest/result/receipts/deadline_audit为原始数据。图片ZIP只含PNG，代码与数据ZIP不含PNG。图片中的单次seed7结果与跨seed均值明确区分。
'''
    (D/'random_multitype_results.md').write_text(text,encoding='utf-8')
    fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(14,5.3),gridspec_kw={'width_ratios':[1.15,1]})
    labels={'fixed_three':'1盘·三类（5种子）','extended_more_headroom':'1盘·两类（5种子）','robust_036':'2盘·推荐三类（7种子）'}
    colors=['#3977b4','#dd8a36','#269071'];seeds=[7,11,23,41,97,101,211]
    for i,(name,color) in enumerate(zip(labels,colors)):
        used=seeds if name==chosen else seeds[:5]
        us=[one(name,s)['U_percent'] for s in used]
        axs[0].plot(range(len(used)),us,'o-',color=color,label=labels[name],lw=1.6)
        avg=statistics.mean(us)
        axs[1].errorbar(avg,i,xerr=[[avg-min(us)],[max(us)-avg]],fmt='o',color=color,markersize=8,capsize=5,lw=2)
        axs[1].text(avg,i+.16,f'{avg:.2f}%',ha='center',color=color,fontsize=12)
    axs[0].set_xticks(range(len(seeds)),[str(s) for s in seeds]);axs[0].set_xlabel('随机种子');axs[0].set_ylabel('整机 NPU 利用率（%）')
    axs[0].set_ylim(75,100);axs[0].grid(axis='y',alpha=.2);axs[0].legend(frameon=False,fontsize=10)
    axs[0].axvspan(4.5,6.5,color='#dcebe3',alpha=.35,zorder=0)
    axs[0].text(5.5,98.2,'未参与筛选',ha='center',fontsize=10,color='#286c4c')
    axs[0].set_title('原生 Baseline FIFO；保留追加检验结果')
    axs[1].set_yticks(range(3),list(labels.values()));axs[1].set_xlim(75,100);axs[1].set_ylim(-.45,2.5)
    axs[1].set_xlabel('整机 NPU 利用率（%）');axs[1].grid(axis='x',alpha=.2)
    axs[1].set_title('点：原生种子平均；横线：最小至最大')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('输入平均欠载，允许运行中需求聚集超载；每卡三类/两类随机混合',fontsize=14,y=1.01)
    fig.tight_layout();fig.savefig(D/'random_multitype_seed_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'report':str(D/'random_multitype_results.md'),'mean_utilization':a['U_mean_percent'],'seeds':a['seeds']}))

if __name__=='__main__':main()
