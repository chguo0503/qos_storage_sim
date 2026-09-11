#!/usr/bin/env python3
"""Explain the finished physical bandwidth diagnostic with links to every NPU."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent

def main():
    d=json.loads((HERE/'analysis.json').read_text());b=d['policies']['baseline'];o=d['policies']['once']
    assert all(b['checks'].values()) and all(o['checks'].values())
    counter=b['examples']['high_bandwidth_during_stall']
    npu_rows=[]
    for n in range(32):
        x,y=b['npu_summary'][n],o['npu_summary'][n]
        npu_rows.append(f'| {n} | {x["demand_mean_gib_s"]:.3f} / {x["ssd_mean_gib_s"]:.3f} / {x["link_mean_gib_s"]:.3f} | {x["stall_ms"]:.3f} | {y["demand_mean_gib_s"]:.3f} / {y["ssd_mean_gib_s"]:.3f} / {y["link_mean_gib_s"]:.3f} | {y["stall_ms"]:.3f} | [PNG](figures/npu_{n:02d}.png) · [PDF](figures/npu_{n:02d}.pdf) |')
    example_rows=[]
    for policy,name in [(b,'Baseline'),(o,'Once per layer')]:
        x=policy['examples']['same_time_short']
        example_rows.append(f'| {name} | {x["release_ms"]:.6f} | {x["deadline_ms"]:.6f} | {x["ready_ms"]:.6f} | {x["ssd_before_deadline_MiB"]:.3f} | {x["link_before_deadline_MiB"]:.3f} | {x["stall_ms"]:.6f} |')
    long=b['examples']['long_l0']
    contents=f'''**每张 NPU 的带宽诉求、SSU 实际供给与 I/O 等待**

已经生成32张独立NPU图，每张图上半部是Ordered Baseline，下半部是相同输入的Once per layer。配置沿用32 NPU、6 SSU、每盘40 GiB/s、NPU链路50 GiB/s；长请求、短请求和桥接请求的输入均未改变。

**先看这三处**

- [为什么stall期间带宽高于诉求，仍然会等待](figures/stall_despite_high_bandwidth.png)
- [同一短请求和长请求：截至计算截止时，数据是否读齐](figures/cumulative_examples.png)
- [Baseline的32卡总览](figures/overview_baseline.png) · [Once的32卡总览](figures/overview_once.png)

你说的“不匹配”方向正确，但应比较**需要数据的截止时刻**，不能只比较stall期间的瞬时速率。对具体下一层：

`无等待所需条件：截至前一层计算结束，目标层数据已经全部进入NPU。`

`I/O stall = max(0，目标层最后一个块进入NPU的时刻 − 前一层计算结束时刻)。`

当前每卡串行、batch_size=1的实验满足这个等式。它检查的是正确目标层的累计数据，不能用这张卡收到的其他层数据代替。

**一个实际反例：stall时供给高于诉求，仍然来不及**

NPU{counter['npu']}的请求`{counter['request_id']}`、层L{counter['layer']}，需要 **{counter['payload_MiB']:.3f} MiB**，前一层计算留出的预算是 **{counter['budget_C_ms']:.6f} ms**。要在预算里完成，平均需要 **{counter['budget_gib_s']:.6f} GiB/s**。

| 时段 | 真实记录 |
|---|---|
| 从读取释放到计算预算截止 | SSU提供了{counter['ssd_before_deadline_MiB']:.3f} MiB，进入NPU的数据也是{counter['link_before_deadline_MiB']:.3f} MiB |
| 预算截止时 | NPU还缺{counter['link_deficit_at_deadline_MiB']:.3f} MiB，必须等待 |
| 随后的stall | 持续{counter['stall_ms']:.6f} ms；SSU对这一层的平均供给{counter['ssd_mean_during_stall_gib_s']:.6f} GiB/s |

供给速率虽然在stall期间超过了{counter['budget_gib_s']:.6f} GiB/s，却错过了前面的计算预算，已经发生的等待无法消除。这不是虚构例子，数据来自该层全部真实块的服务起止记录。

**另一个短请求，以及Once对同一层的结果**

NPU22、请求`22000046`的L4是32K/1024短请求的一层。两策略的目标量都为42.625 MiB，计算预算都为7.257232 ms。下表时间均是原仿真的绝对时间，单位ms。

| 策略 | 读取释放 | 计算预算截止 | 数据全部进入NPU | 截止时SSU已读出MiB | 截止时已进入NPU MiB | stall ms |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(example_rows)}

同一请求在Once下推进得更早，两个绝对释放时间不同。累计图把各自释放时刻记为横轴0，比较相同请求、相同层的数据进度，并在图中保留各自绝对释放时间。

同图中的长请求首层需要 **{long['payload_MiB']:.3f} MiB**，预算来自前一个桥接请求最后一层的 **{long['budget_C_ms']:.6f} ms**，所以这次实际预取需要的预算速率是 **{long['budget_gib_s']:.6f} GiB/s**。Baseline在截止前已把这一层数据准备好，stall为0。由此可以同时看到短请求错过预算、长请求预取被计算遮住的情形。

**每张带宽图的线是什么意思**

| 图中内容 | 定义 |
|---|---|
| 当前请求D/C，名义参考 | 沿用原实验：当前已接纳请求单层读取量÷它自己的单层计算时间。它是参考速率，不能把延伸经过stall的曲线面积当作新增读取字节。 |
| SSU实际读服务 | 六盘实际为该NPU读取的字节之和÷分箱时长。按每个块真正的SSD服务区间积分，包含跨箱边界的部分字节。 |
| 进入NPU的带宽 | 数据在NPU链路上真正传输的字节÷分箱时长。SSD读完后尚在链路队列里的数据，不计作已经进入NPU。 |
| 橙色区间 | 原日志中的实际I/O stall。 |

曲线使用 **0.5 ms分箱**；SSD供给是六盘合计，可短暂高于50 GiB/s，但进入NPU的链路带宽不能超过50 GiB/s。实际服务没有用“读取量÷层读取生命周期”倒算成一条平滑线。

名义参考与具体预取预算在请求切换处有区别。例如桥接请求仍在计算时，其自身D/C为 **1.314932 GiB/s**；此时预取的却是下一长请求首层，真正需要的预算是 **8.218328 GiB/s**。因此累计图使用**下一层的实际读取量÷前驱计算时间**，包括所有跨请求首层读取。它用于解释截止时的数据缺口，没有更改之前欠载测试的约定口径。

累计图中的红色斜线只是“均匀完成”的参考路径，达到目标量后封顶；实际曲线中途低于它，不必然产生stall。关键是到截止竖线时，进入NPU的数据有没有达到完整目标量。若截止时SSD已读齐而NPU仍没收齐，问题还涉及链路；若SSD本身也没读齐，则SSD服务已经来晚。

**取样窗口与数据可靠性**

本次带宽图取原完整运行的 **[3.2,4.0)秒**，能看到短请求、桥接请求和下一长请求的交接。这个800ms局部窗口的整机U为Baseline **{b['window_U_percent']:.6f}%**、Once **{o['window_U_percent']:.6f}%**；它与[2,60)长窗平均是不同统计，长窗仍分别为89.964747%与99.961111%。

原60秒结果只保存层级时序，没有逐块物理服务明细。因此本次使用完整冻结输入、相同种子和策略，从t=0被动复现到约4.2秒。观察函数只复制已经发生的服务记录，未增加业务事件或修改调度。展示窗涉及的每层全部块已核齐；原始结果、输入、源码哈希不变，所有已完成层和请求的前缀时序与原60秒结果严格一致。未把这次局部停止标作完整60秒仿真。

[Baseline采样审计](traces/baseline/audit.json) · [Once采样审计](traces/once/audit.json) · [服务与截止口径独立审查](review/method_review.md) · [逐块物理服务独立复核](review/check_trace.md)

独立复核覆盖Baseline的813,656个块和Once的896,904个块：每盘实际服务不超过40 GiB/s、每条NPU链路不超过50 GiB/s，逐块时长与分箱字节守恒均通过。这里的容量检查针对物理服务；不能据此倒推“所有请求同时希望得到的带宽”也必然低于容量。

**32张卡的独立图与窗口均值**

带宽列依次为“名义诉求 / SSU实际读服务 / 进入NPU”，单位GiB/s，全部是相同800ms窗口均值。均值会掩盖先等待、后补读的先后顺序，也受窗口边界影响；判断stall原因请结合曲线和逐层截止数据。

| NPU | Baseline带宽 | Baseline stall ms | Once带宽 | Once stall ms | 独立图 |
|---|---:|---:|---:|---:|---|
{chr(10).join(npu_rows)}

每张图还有同名SVG版本，位于`figures/`。

可复算数据：[逐NPU汇总CSV](per_npu_summary.csv) · [逐NPU每0.5ms带宽CSV.gz](per_npu_bandwidth_bins.csv.gz) · [逐层截止与缺口CSV](per_layer_deadlines.csv) · [分析JSON](analysis.json)。

复现脚本：[被动采样](replay_trace.py)、[带宽与截止分析](analyze_delivery.py)、[绘图](plot_bandwidth_delivery.py)。
'''
    (HERE/'README.md').write_text(contents)
    print(HERE/'README.md')

if __name__=='__main__':main()
