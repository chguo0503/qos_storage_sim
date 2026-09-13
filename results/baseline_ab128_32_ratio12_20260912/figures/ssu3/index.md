# 3 SSU 图表入口

32 NPU、3 SSU × 40 GiB/s，长验证 seed 7，同一个 warm `[2,4)` 秒。Random 真实整机 U=**90.67794%**，Ordered=**71.00895%**；这是单次运行结果，不是三种子均值。使用已有日志，新增图只生成 PNG。

**全32卡带宽总图：[Random PNG](random_all_32npu_layer_average.png) · [Ordered PNG](ordered_all_32npu_layer_average.png)。** 分别将该顺序下 `npu_00`～`npu_31` 原图的最上方带宽面板汇成32行，共用时间轴和纵轴尺度，只保留每层平均 `b_i`、`B_i` 和灰区标记。[Random 来源校验](random_all_32npu_layer_average.checks.json) · [Ordered 来源校验](ordered_all_32npu_layer_average.checks.json)

每行左侧另标该卡的平均利用率 `U`：该卡在 warm `[2,4)` 秒内的实际计算时间除以 2 秒。Random 每卡为 78.19%～99.80%，Ordered 每卡均为 71.01%。

每行右侧显示同一个完整2秒窗口内的平均带宽需求与供给，单位GiB/s。需求为当前请求 `B_i` 按时间加权的参考值；供给按实际收到的数据量计算，包含图中灰区。两个整窗均值相除不等于利用率。[完整均值表与口径](warm_bandwidth_means.md) · [CSV](warm_bandwidth_means.csv)

先看整体 32 卡的周期比值：[Random PNG](fleet_cycle_ratio/fleet_cycle_ratio_random.png) · [Ordered PNG](fleet_cycle_ratio/fleet_cycle_ratio_ordered.png) · [公式与汇总说明](fleet_cycle_ratio/README.md)。

| 看什么 | NPU 15 Random 示例 | NPU 15 Ordered 示例 | 全部卡入口 |
|---|---|---|---|
| 每层周期平均 `b_i` 与 `B_i` | [PNG](per_npu/random_npu_15.png) | [PNG](per_npu/ordered_npu_15.png) | [逐卡索引](per_npu/index.md) |
| 同一 A 请求内部的实际接收与等待 | [PNG](internal_layer_zoom/random_a_npu_15.png) | [PNG](internal_layer_zoom/ordered_a_npu_15.png) | [内部局部图](internal_layer_zoom/index.md) |
| 同一 B 请求内部的实际接收与等待 | [PNG](internal_layer_zoom/random_b_npu_15.png) | [PNG](internal_layer_zoom/ordered_b_npu_15.png) | [内部局部图](internal_layer_zoom/index.md) |
| A→B 跨请求首层预取 | [PNG](per_npu_zoom/random_npu_15.png) | [PNG](per_npu_zoom/ordered_npu_15.png) | [交接局部图](per_npu_zoom/index.md) |

周期平均图从开始算第 k 层到开始算第 k+1 层划分周期，包含等待。`B_i` 仍是当前已接纳请求的 `V/C`；灰区涉及跨请求或窗口截断，不套内部周期比值，真实计算时间仍计入利用率。局部图只统计其标出的片段，不能冒充上述两秒整机平均。

原 `per_npu/npu_XX.png` 已更新为周期平均的 Random/Ordered 合并对照；同名 PDF 保留历史 0.5 ms 带宽采样口径，本次未生成或更新 PDF。

[Random 两秒计算时序](baseline_random.png) · [Ordered 两秒计算时序](baseline_ordered.png) · [Random 18 秒计算时序](long/baseline_random.png) · [Ordered 18 秒计算时序](long/baseline_ordered.png)

[返回全部拓扑图表](../index.md) · [实验报告](../../report.md)
