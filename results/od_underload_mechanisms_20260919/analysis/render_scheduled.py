"""Chinese figures and readable report from an independent completed-run audit.

No simulator is imported. Preview files are segregated from long-run deliverables.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
COLORS={"od_baseline":"#d45e00","once":"#2474b7"}
LABELS={"od_baseline":"OD","once":"Once（流量分配）"}
KINDS={"A":("#d45e00","短计算请求"),"B1":("#2474b7","长计算请求"),
       "B2":("#25a08b","长计算请求（提前选定 miss）"),"stall":("#b0b5bd","I/O 等待")}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next(f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","DejaVu Sans") if f in names)
    plt.rcParams.update({"font.family":font,"font.size":11,"axes.unicode_minus":False,
        "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})


def save(fig,path):
    fig.savefig(path,dpi=180)
    plt.close(fig)
    return path


def mean_window(result,a,z):
    return next((r for r in result["fixed_windows"] if r["start_ms"]==a and r["end_ms"]==z),None)


def utilization(audit,out,preview):
    fig,ax=plt.subplots(figsize=(13,7))
    fig.subplots_adjust(left=.085,right=.975,bottom=.285,top=.80)
    limit=max(r["horizon_ms"] for r in audit["policies"].values())/1000
    for policy,result in audit["policies"].items():
        rows=result["two_second_bins"]
        ax.plot([(r["start_ms"]+r["end_ms"])/2000 for r in rows],
                [r["U_percent"] for r in rows],color=COLORS[policy],marker="o",ms=3.5,lw=1.3,label=LABELS[policy])
        for k,(a,z) in enumerate(((20000,40000),(40000,60000))):
            r=mean_window(result,a,z)
            if r:
                ax.plot([a/1000,z/1000],[r["U_percent"]]*2,color=COLORS[policy],lw=4,alpha=.9)
                offset=1.5 if policy=="once" else -3.0
                ax.text((a+z)/2000,r["U_percent"]+offset,f"{LABELS[policy]} {r['U_percent']:.2f}%",
                    ha="center",va="center",color=COLORS[policy],fontsize=11,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":2})
    ax.set(xlim=(0,limit),ylim=(50,102),xlabel="时间（秒）",ylabel="NPU 平均利用率（%）")
    ax.grid(alpha=.17)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.17),ncol=2,frameon=False)
    fig.suptitle("离线选定请求：长期平均利用率的正式重放" if not preview else "2 周期布局预览：不能用于长期结论",
                 fontsize=19,fontweight="bold",y=.96)
    fig.text(.085,.89,f"32 NPU / 3 SSU × 40 GiB/s · 每卡重复“短—长—长” · seed 7 · 同一冻结输入",fontsize=11)
    fig.text(.085,.155,"细线点 = 连续 2 秒的计算占比（受它与 1.65 秒周期错位影响）；粗横线 = 预先固定的 20 秒均值。",fontsize=10.5)
    fig.text(.085,.108,"各策略只显示首卡排空前的完整 2 秒窗口；全部 32 卡持续活跃。曲线可提前结束，排空之后不补零。",fontsize=10.5)
    fig.text(.085,.062,"这是针对 OD 时序提前挑选 miss 的对抗性构造；不是随机抽样结论。请求长度、读取量和计算均按输入模型计入。",fontsize=10.3,color="#555555")
    if preview:fig.text(.085,.02,"本预览只有 2 周期，没有 20–40 / 40–60 秒观测；本图仅检查版式。",fontsize=10,color="#a52916")
    return save(fig,out/("preview_long_term_utilization.png" if preview else "scheduled_long_term_utilization.png"))


def timeline(audit,out,preview,policy="od_baseline"):
    result=audit["policies"][policy];selection=result["selected_two_cycles"]
    if policy=="once":
        other=audit["policies"]["od_baseline"]["selected_two_cycles"]
        assert (selection["start_ms"],selection["end_ms"])==(other["start_ms"],other["end_ms"])
    a,z=selection["start_ms"]/1000,selection["end_ms"]/1000
    assert selection["timeline"]
    fig,ax=plt.subplots(figsize=(14,12))
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.835)
    for kind,(color,label) in KINDS.items():
        for n in range(32):
            seg=[(r["start_ms"]/1000,(r["end_ms"]-r["start_ms"])/1000)
                 for r in selection["timeline"] if r["npu_id"]==n and r["kind"]==kind]
            if seg:ax.broken_barh(seg,(n-.34,.68),facecolors=color,linewidth=0)
    ax.set(xlim=(a,z),ylim=(31.7,-.7),xlabel="时间（秒）",ylabel="NPU 编号",yticks=range(32))
    ax.tick_params(axis="y",labelsize=9)
    ax.grid(axis="x",alpha=.18)
    handles=[Patch(facecolor=c,label=l) for c,l in KINDS.values()]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.89),ncol=2,frameon=False,fontsize=11)
    title="OD：完整重复单元中的计算与 I/O 等待" if policy=="od_baseline" else "Once（流量分配）：同一时间段的执行"
    fig.suptitle(title+("（2 周期预览）" if preview else ""),fontsize=19,fontweight="bold",y=.968)
    subtitle=(f"完整输入提前冻结；尾部长请求的 miss 已选定 · 展示 {a:.1f}–{z:.1f} 秒，未截取单独低谷" if policy=="od_baseline" else
              f"与 OD 图使用相同墙钟区间 {a:.1f}–{z:.1f} 秒；此时正在执行的请求身份和轮次可与 OD 不同")
    fig.text(.075,.923,subtitle,fontsize=11)
    fig.text(.075,.105,"橙色是短计算请求；蓝色、绿色都是长计算请求。绿色只表示这一位置的 miss 在离线规划时选定。",fontsize=10.5)
    wait_note=("短计算块很窄，反复出现的大灰块才是利用率损失。没有把人为暂停记成计算；计算时长按原始 data 或插值独立核验。" if policy=="od_baseline" else
               "灰色表示实际 I/O 等待，橙色表示短请求的计算；两者分别统计。所有计算时长均按原始 data 或插值独立核验。")
    fig.text(.075,.070,wait_note,fontsize=10.5)
    footer=("各卡依次运行短、长、长请求。该构造利用输入时长维持相位接近；不能推断普通随机输入也会如此，解释见报告。" if policy=="od_baseline" else
            "输入及每卡请求顺序完全相同，绿色请求的 miss 没有为 Once 重新规划；执行速度改变会使卡间相位不同。")
    fig.text(.075,.035,footer,fontsize=10.3,color="#555555")
    key="od" if policy=="od_baseline" else "once"
    return save(fig,out/(("preview_" if preview else "scheduled_")+key+"_two_cycles.png"))


def demand(audit,out,preview):
    result=audit["policies"]["od_baseline"];selection=result["selected_two_cycles"]
    a,z=selection["start_ms"],selection["end_ms"]
    # Full event trace must extend past the last selected plot endpoint.
    rows=[r for r in result["per_disk_demand_segments"] if r[0]<z and r[1]>a]
    plotted_end=min(z,rows[-1][1])
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    fig.subplots_adjust(left=.085,right=.978,bottom=.135,top=.85,hspace=.23)
    maxvalue=max(max(r[2:5]) for r in rows)
    for d,ax in enumerate(axes):
        edges=[max(a,r[0])/1000 for r in rows]+[plotted_end/1000]
        values=[r[2+d] for r in rows]
        ax.stairs(values,edges,baseline=None,color="#d45e00",lw=1.6,label="逐事件需求 D_s(t)")
        for r in rows:
            if r[2+d]>40:ax.axvspan(max(a,r[0])/1000,min(plotted_end,r[1])/1000,color="#e76f51",alpha=.045,lw=0)
        ax.axhline(40,color="#525252",ls="--",lw=1.2,label="物理容量 40 GiB/s")
        ax.set(ylabel=f"SSU {d}\n需求（GiB/s）",ylim=(0,maxvalue*1.08),xlim=(a/1000,plotted_end/1000))
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("时间（秒）")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.912),ncol=2,frameon=False)
    fig.suptitle("OD：需求在完整重复单元内发生强烈波动"+("（2 周期预览）" if preview else ""),fontsize=19,fontweight="bold",y=.973)
    fig.text(.085,.931,"D_s(t) = 32 卡当前请求在该盘的 V/C 总和；V 是每层读取量，C 是每层纯计算时间",fontsize=11)
    fig.text(.085,.037,"灰线为 40 GiB/s；橙线高于灰线才是过载。等待中的请求也计入需求。这是间歇性强过载，不能称为逐时欠载。",fontsize=10.5)
    return save(fig,out/("preview_od_demand.png" if preview else "scheduled_od_demand.png"))


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
        ["| "+" | ".join(map(str,row))+" |" for row in rows])


def span(a,b,spec):
    return format(a,spec) if a==b else format(a,spec)+"–"+format(b,spec)


def report(audit,path,figures):
    inp=audit["input"];policies=audit["policies"];lines=[]
    od1=mean_window(policies["od_baseline"],20000,40000);od2=mean_window(policies["od_baseline"],40000,60000)
    on1=mean_window(policies["once"],20000,40000);on2=mean_window(policies["once"],40000,60000)
    if all(r is not None for r in (od1,od2,on1,on2)):
        lines.append(f"同一冻结输入、seed 7 下，OD 在 20–40 秒和 40–60 秒的平均 NPU 利用率分别为 {od1['U_percent']:.4f}%、{od2['U_percent']:.4f}%；Once（流量分配）分别为 {on1['U_percent']:.4f}%、{on2['U_percent']:.4f}%。这两个预先固定的中后段窗口内，32 张卡持续活跃，并且每张卡都运行了长短请求，没有使用排空尾部。")
    lines.append("这是一次针对 OD 提前选定输入的正式重放。请求在运行前已经全部冻结，正式指标没有使用离线规划的额外事件钩子；标准 API 的策略适配器照常工作。")
    lines.append("比较的是当前两套完整策略配置。OD 每盘使用 32 条 NPU 独占 Path，每卡 CIR=40/32=1.25 GiB/s，空闲带宽可借用。Once 配置为 256 条可用 Path，保留原来的类别候选池和静态 CIR：SS/SL/LS/LL 的每盘类别总预算分别为 20/6/8/6 GiB/s，候选 Path 数分别为 96/32/96/32；本输入的 A 属于 LS、B 属于 LL。Once 每层、每盘使用一次拥塞快照来规划这一层各 I/O 的 Path，不是整层只选一条 Path。因此结果包含路径数量、类别 CIR/候选池与选路的共同作用，不能说成“只改变选路”的单变量实验。")
    lines.append("它检验的是“是否存在让 OD 长时间平均利用率偏低的输入”，不是普通 data 随机输入发生问题的概率，也不是每一时刻严格欠载的实验。每盘的瞬时需求是否超限，以下面逐事件重算为准。")
    rows=[]
    for a,z in ((20000,40000),(40000,60000),(20000,60000)):
        for p,r in policies.items():
            w=mean_window(r,a,z)
            if w:rows.append([f"{a/1000:g}–{z/1000:g} 秒",LABELS[p],f"{w['U_percent']:.4f}%",f"{w['slo_1p5_percent']:.4f}%",w["slo_1p5_count"],"32 / 32" if w["all_npus_active"] else "未满足",f"{w['mixed_npus']} / 32"])
    lines.append(table(["统计区间","策略","NPU 平均利用率","SLO ×1.5 达标率","窗口接纳请求数","持续活跃卡","都计算过长短请求的卡"],rows))
    periodic=[r for r in policies["od_baseline"]["cycle_checks"] if r["earliest_A_compute_start_ms"]>=20000 and r["latest_tail_end_ms"]<=60000 and r["ends_before_first_card_finishes"]]
    if periodic:
        values=[r["complete_cycle_card_time_weighted_U_percent"] for r in periodic]
        lines.append(f"补充核对了全部 {len(periodic)} 个完全落在 20–60 秒内的完整 ABB 周期：OD 的逐卡完整周期加权计算占比为 {min(values):.4f}%–{max(values):.4f}%，周期末卡间完成时刻最大跨度为 {max(r['tail_end_spread_ms'] for r in periodic):.6f} ms。这个周期指标使用每卡自己的周期区间，不等于同一个墙钟窗口的整机利用率，不能替换上表；它用于检查现象是否反复出现。")
        lines.append("图上的 2 秒窗口与 1.65 秒请求周期并不对齐，有些窗口多包含一次等待段，所以细线高低会变化；不能把这种采样后的波形直接当成请求本身的周期。长期判断用预先固定的宽窗口均值，同时保留所有 2 秒窗口，没有挑选低谷。")
    lines.append("利用率按窗口内真实计算时间 / (32 × 窗口时长) 计算。SLO 按窗口内上卡的请求计算，并等待这些请求真正完成后统计；耗时从上卡开始到 prefill 完成，不含卡内排队，阈值是该请求自身 8 层纯计算时间的 1.5 倍。这不是完整端到端 TTFT。没有丢弃窗口末尾未完成的请求。")
    role_rows=[]
    for policy,result in policies.items():
        window=mean_window(result,20000,60000)
        if window:
            for role,r in window["by_role"].items():
                role_rows.append([LABELS[policy],"短计算请求" if role=="A" else "长计算请求",
                    f"{r['active_U_percent']:.4f}%",f"{r['share_of_window_card_time_percent']:.4f}%",
                    f"{r['contribution_to_fleet_U_percentage_points']:.4f}"])
    if role_rows:
        lines.append(table(["20–60 秒策略","请求类型","该类型驻留期间的计算占比","占全部窗口卡时间","对整机 U 的贡献（百分点）"],role_rows))
        lines.append("整机 U 是最后一列相加，不是把两个类别各取一半，也不是按请求数 1:2 加权。OD 的长请求内部各层读得及，不等于整个长请求零等待：它的首层仍可能等前一短请求最后一层没能完全藏住的读取。")
    lines.append("窗口中的请求身份会因策略推进速度而变化；因此还应看同一批全部请求：")
    shared=audit["window_cohort_overlap"].get("20000:60000")
    if shared:
        lines.append(f"例如 20–60 秒的上卡队列：OD 包含 {shared['OD_count']} 个请求，Once 包含 {shared['Once_count']} 个，身份交集为 {shared['shared_count']} 个；这不是完全相同的窗口样本。下表使用全部 {inp['request_count']} 个相同请求，避免把窗口推进差异混入请求集合。")
    rows=[]
    for p,r in policies.items():
        q=r["same_population_slo_1p5"]
        rows.append([LABELS[p],q["count"],q["passed"],f"{q['percent']:.4f}%",f"{q['p95_normalized']:.4f}"])
    lines.append(table(["策略","相同完整请求人口","达标数","SLO ×1.5 达标率","归一化耗时 p95"],rows))
    lines.append("完整人口 SLO 包含启动和排空期的请求，只用于同一请求集合的补充核对。长期利用率结论使用前表的固定中段窗口，不使用带排空尾部的全程平均。p95 为排序后第 ceil(0.95 × n) 个请求。")
    lines.append("每卡输入为“短计算请求 A → 长计算请求 B1 → 长计算请求 B2”，重复 "+str(inp["metadata"]["cycles"])+" 次。全部 arrival=0；32 NPU、3 SSU、每盘 40 GiB/s、8 层、batch=1、seed 7、原始 RingHash。每个请求的物理身份唯一，没有为落盘比例搜索地址。")
    rows=[]
    for label,g in inp["profile_groups"].items():
        rows.append([label,g["count"],span(g['total_tokens_min']/1024,g['total_tokens_max']/1024,"g"),span(g['miss_min'],g['miss_max'],"d"),
            span(g['per_layer_compute_ms_min'],g['per_layer_compute_ms_max'],".6f"),span(g['per_layer_read_MiB_min'],g['per_layer_read_MiB_max'],".5f")])
    lines.append(table(["请求位置","整机数量","总输入长度 K","miss tokens","每层计算 ms","每层读取 MiB"],rows))
    lines.append("A 直接使用 data 的 128K / miss 256。B 的计算时间来自 data 中长度 96/128/160K 与 miss 2048/4096 网格的双线性内插，无外推；固定命中 127744 tokens，miss 是整数，总长度随 miss 改变，读取量保持一致。它不是每个点都已有原始实测的 workload。")
    lines.append("通俗地说，每轮预先定为 1.65 秒：第一轮目标在 1650 ms 结束，第二轮在 3300 ms 结束，以此类推。先在离线规划中看每张卡开始计算 B2 的时间 s，再在允许的整数 miss 中选最接近 C=(E-s)/8 的请求，E 就是预先固定的周期末时刻，E、s、C 都用毫秒。这样尾部长请求完成时间会重新靠拢，下一轮短计算请求又容易一起到来。正式实验重新读取冻结输入，策略无法临时延长计算或等待。")
    arithmetic=inp["frozen_input_arithmetic"]
    lines.append(f"还可以用冻结后的输入反算一次：每卡平均共需纯算 {arithmetic['mean_pure_compute_ms_per_npu']/1000:.6f} 秒；若全部卡按设计在 {arithmetic['designed_end_ms']/1000:g} 秒完成，则利用率应为两者之比，即 {arithmetic['U_percent_if_all_finish_at_designed_end']:.6f}%。这是看过规划后、对冻结输入的条件式算术核对，不是独立的事前预测；上表仍以正式重放的逐层积分为准。")
    lines.append(f"每个 B2 完成时刻到目标时刻的量化误差上界为 {inp['quantization_individual_end_bound_ms']:.6f} ms，两个卡之间的理论量化差上界为 {inp['quantization_pair_spread_bound_ms']:.6f} ms。并非所有 B2 都增加计算：有 {inp['reduced_tail_request_count']} 个缩短；相对基准 B，净计算变化 {inp['signed_compute_change_card_ms']/1000:.6f} 卡秒、绝对变化 {inp['absolute_compute_change_card_ms']/1000:.6f} 卡秒。所有这些计算都进入利用率分子和自身 SLO 基线。")
    rows=[]
    for p,r in policies.items():
        for a,z in ((20000,40000),(40000,60000),(20000,60000)):
            w=mean_window(r,a,z)
            if not w:continue
            for d in range(3):rows.append([LABELS[p],f"{a/1000:g}–{z/1000:g}",d,f"{w['per_disk_mean_GiB_s'][d]:.4f}",f"{w['per_disk_peak_GiB_s'][d]:.4f}",f"{w['per_disk_overload_percent'][d]:.4f}%"])
    lines.append(table(["策略","时间段（秒）","SSU","时间平均需求 GiB/s","最高需求 GiB/s","需求 >40 的时间占比"],rows))
    lines.append("这里的需求 D_s(t) 是当前每张卡正在处理的请求的 V_i,s / C_i 之和，I/O 等待中的卡也计入；下一请求首层预取不再额外叠加一个请求画像。OD 在这里是间歇性强过载，Once 也仍有超限时段；都不能称为任意时刻逐盘欠载。超限占比只数持续时间，不数超限有多高或哪些请求赶不上读取期限，因此不能单独用它预测利用率；Once 某盘的超限占比虽更高，但峰值和时间平均需求已经明显降低。")
    refs=", ".join(f"SSU {d}: {v:.4f}" for d,v in enumerate(inp["pure_compute_reference_rate_GiB_s_by_ssu"]))
    lines.append("另外一种容易混淆的数字是“按每卡总读取量 / 每卡总纯计算时间，再把各卡相加”的工作量参考速率："+refs+" GiB/s。它没有描述请求在墙钟时间上的同时出现情况，即使低于 40，也不能证明执行中欠载。")
    bound=inp["finite_input_capacity_bound"]
    lines.append(f"只按这批工作的总量看，计算最多的卡至少要 {bound['max_per_npu_pure_compute_ms']/1000:.4f} 秒；数据最多的盘以 40 GiB/s 读完至少要 {bound['bottleneck_disk_service_ms']/1000:.4f} 秒。因此全程完成时间至少为两者较大值 {bound['makespan_lower_bound_ms']/1000:.4f} 秒，全程利用率的相应宽松上界为 {bound['full_U_upper_bound_percent']:.4f}%。这是工作总量给出的必要界，不是最优策略可达到的保证，更不是 20–40 秒窗口的上界。")
    rows=[]
    for p,r in policies.items():
        for d in r["full_physical_SSD_supply"]:rows.append([LABELS[p],d["ssu_id"],f"0–{d['end_ms']/1000:.6f}",f"{d['mean_GiB_s']:.6f}"])
    lines.append(table(["策略","SSU","供给统计时间（秒）","全程真实物理供给均值 GiB/s"],rows))
    lines.append("上述供给是每盘真实完成字节 / 全程时长，包含启动与排空；不能直接减去 20–60 秒的需求均值解释成同一窗口的缺口。raw 没有任意窗口的物理服务事件，因此不虚构这类供给曲线。")
    lines.append("[layer_cycle_supply.csv](layer_cycle_supply.csv) 则记录完整层周期的真实读取量与平均供给：从本层计算开始到下一层计算开始，下一层全部读取在这个周期内完成，所以平均 b=V_next/周期时长。已逐周期检查 I/O 提交和就绪端点。跨请求使用下一请求首层的 V；异步周期均值不是瞬时速率，裁窗口边界也不能更换平均的分母。")
    example=next(iter(policies["od_baseline"]["representative_internal_A_cycles"]),None)
    if example:
        lines.append(f"一个真实例子：预先选定图示区间内，NPU 0 的首个完整短请求内部周期（请求 {example['request_id']}，第 {example['layer']+1} 层到下一层）读取 {example['per_layer_read_MiB']:.5f} MiB；计算 C={example['C_ms']:.6f} ms，随后等待 I={example['I_ms']:.6f} ms，完整周期 T={example['T_ms']:.6f} ms。所以需求 B=V/C={example['B_GiB_s']:.4f} GiB/s，完整周期供给 b=V/T={example['b_GiB_s']:.4f} GiB/s，b/B=C/T={example['cycle_U_percent']:.4f}%。这是同一个请求内部的完整周期等式，不能机械套到跨请求边界或任意瞬时。")
    lines.append("需要保留的限制：这个输入是专门针对 OD、seed 7 和当前配置挑选的，对 Once 没有重新选择 miss；不能推断其他种子或真实随机 workload。离线同步依赖确定性的计算/服务时序；真实系统的抖动可能破坏它，本轮没有验证这类鲁棒性。两策略虽运行同一冻结请求集合，但完成速度改变当前活跃请求身份，因此每时刻的需求曲线不必相同。OD 仍为每卡每盘独立 Path、每盘固定队深 8192（每卡 256）；Once 使用此前原始策略默认的无限队深。CIR 是可借用空闲带宽的保底，不是每卡永远只有 1.25 GiB/s 的上限；队深不能借用也不等于带宽不能借用。")
    if audit.get("depth_control",{}).get("passed"):
        lines.append("队深差异也做了同输入消融：把 OD 的每卡每盘 256 深度改成无限，全部 4800 个请求的时间字段、38400 层的完整时序记录都与有限深度相同。因此这次 OD 利用率损失不是该队深上限造成的。只有从实际入盘开始计的等待诊断变化，反映等待位置在主机与盘侧之间移动；不能把诊断缩短当作请求加速。这项结论只针对本输入。")
    if audit.get("seed_control",{}).get("passed"):
        lines.append("另把 OD 的同刻提交顺序 seed 从 7 改为 19，输入字节不变，全部请求和层记录仍严格相同。这个种子变化在本例中没有有效扰动 OD 时序，因此不能包装成另一批随机工作负载，也不能当作抗计算/SSD 服务抖动的证据。")
    lines.append("独立验证已覆盖：manifest 的全部块落盘、数据内插与整数 miss、来源文件 SHA 链、规划/正式 OD 的所有请求和层时序严格相同、每层计算时间、每卡计算+I/O 等待守恒、所有连续 2 秒窗口的活跃情况、完整人口 SLO、逐事件需求以及层周期供给端点。正式数据都已排空，OD 和 Once 的输入 manifest SHA 完全相同。")
    lines.append("原始数值见 [audit.json](audit.json)、[windows.csv](windows.csv)、[cycles.csv](cycles.csv)、[population_slo.csv](population_slo.csv)。\n\n"+"\n".join(f"- [{p.stem}]({Path('../../figures')/p.name})" for p in figures))
    path.write_text("\n\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--name",default="abb_interp_unique_50");p.add_argument("--preview",action="store_true");a=p.parse_args()
    folder=HERE/("scheduled_"+a.name);path=folder/"audit.json";audit=json.loads(path.read_text())
    assert audit["all_policies_complete"] and all(r["passed"] for r in audit["policies"].values())
    for filename,digest in audit["source_sha256"].items():assert sha(HERE/filename)==digest
    for filename,digest in audit["derived_csv_sha256"].items():assert sha(folder/filename)==digest
    if not a.preview:assert audit["input"]["metadata"]["cycles"]>=50
    setup();out=folder if a.preview else STUDY/"figures";out.mkdir(parents=True,exist_ok=True)
    figures=[utilization(audit,out,a.preview),timeline(audit,out,a.preview),demand(audit,out,a.preview)]
    if not a.preview:
        od=mean_window(audit["policies"]["od_baseline"],20000,60000)
        once=mean_window(audit["policies"]["once"],20000,60000)
        if od and once and once["U_percent"]-od["U_percent"]>=3:
            figures.append(timeline(audit,out,False,policy="once"))
    if not a.preview:report(audit,folder/"README.md",figures)
    checks=dict(status="complete",preview=a.preview,audit_sha256=sha(path),source_sha256=sha(Path(__file__)),
                png_sha256={str(p.relative_to(STUDY)):sha(p) for p in figures},
                visual_review={"status":"pending"},all_both_replays_complete=True)
    (folder/("preview_render_checks.json" if a.preview else "render_checks.json")).write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"figures":[str(p) for p in figures],"preview":a.preview}))


if __name__=="__main__":main()
