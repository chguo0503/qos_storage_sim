# 逐卡带宽：按完整内部层周期平均的 b_i

32 NPU、4 SSU × 40 GiB/s、seed 7、同一warm [2,4) 秒。使用既有物理接收日志的逐层积分结果，没有新增仿真。

紫虚线B_i保持原义：当前已接纳请求每层读取量V / 每层纯计算时间C，在请求切换时改变。蓝线改为按层周期的实际平均接收速率。

```text
周期D：开始计算第k层 → 开始计算第k+1层
D = 第k层计算C + 等待第k+1层数据的时间
平均b_i = 这整个周期内实际收到的第k+1层数据量 / D
```

例如Ordered NPU15的一个A内部周期：收到175.65625MiB，计算6.024ms，等待28.284ms，完整周期34.308ms。因此蓝线画为5GiB/s，紫线仍为28.476GiB/s。没有把实际50GiB/s的短暂传输画成密集尖峰，也没有只对传输中的时间取平均。

每个完整内部周期是一段水平蓝线，圆点标记这个周期；相邻周期均值相同会连成较长水平线。若b_i与B_i相等，蓝线和紫虚线重叠，蓝色圆点仍可见。

灰区表示跨请求或被warm窗口边缘截断的周期，蓝线不填值，不表示零带宽。跨请求时B_i可能在周期中途变化，不能直接套用内部的b_i/B_i=C/D；边界也不能拿完整V除以裁剪时长。所有灰区的真实计算和等待仍计入原来的warm利用率。

每张独立PNG下方还放大A、B各一个请求的3个完整内部周期（第2～4层计算），标明均值和周期耗时；按warm内最早符合条件选取，没有选择最大等待。两局部的横轴范围和纵轴刻度不同。

当前生成64张分开的PNG，并更新32张原npu_XX.png合并对照，方便旧链接继续使用。同名npu_XX.pdf是历史瞬时带宽版本，本次没有生成或修改PDF。

| NPU | Random：每层平均 PNG | Ordered：每层平均 PNG | 合并对照 PNG |
|---:|---|---|---|
| 0 | [Random](random_npu_00.png) | [Ordered](ordered_npu_00.png) | [对照](npu_00.png) |
| 1 | [Random](random_npu_01.png) | [Ordered](ordered_npu_01.png) | [对照](npu_01.png) |
| 2 | [Random](random_npu_02.png) | [Ordered](ordered_npu_02.png) | [对照](npu_02.png) |
| 3 | [Random](random_npu_03.png) | [Ordered](ordered_npu_03.png) | [对照](npu_03.png) |
| 4 | [Random](random_npu_04.png) | [Ordered](ordered_npu_04.png) | [对照](npu_04.png) |
| 5 | [Random](random_npu_05.png) | [Ordered](ordered_npu_05.png) | [对照](npu_05.png) |
| 6 | [Random](random_npu_06.png) | [Ordered](ordered_npu_06.png) | [对照](npu_06.png) |
| 7 | [Random](random_npu_07.png) | [Ordered](ordered_npu_07.png) | [对照](npu_07.png) |
| 8 | [Random](random_npu_08.png) | [Ordered](ordered_npu_08.png) | [对照](npu_08.png) |
| 9 | [Random](random_npu_09.png) | [Ordered](ordered_npu_09.png) | [对照](npu_09.png) |
| 10 | [Random](random_npu_10.png) | [Ordered](ordered_npu_10.png) | [对照](npu_10.png) |
| 11 | [Random](random_npu_11.png) | [Ordered](ordered_npu_11.png) | [对照](npu_11.png) |
| 12 | [Random](random_npu_12.png) | [Ordered](ordered_npu_12.png) | [对照](npu_12.png) |
| 13 | [Random](random_npu_13.png) | [Ordered](ordered_npu_13.png) | [对照](npu_13.png) |
| 14 | [Random](random_npu_14.png) | [Ordered](ordered_npu_14.png) | [对照](npu_14.png) |
| 15 | [Random](random_npu_15.png) | [Ordered](ordered_npu_15.png) | [对照](npu_15.png) |
| 16 | [Random](random_npu_16.png) | [Ordered](ordered_npu_16.png) | [对照](npu_16.png) |
| 17 | [Random](random_npu_17.png) | [Ordered](ordered_npu_17.png) | [对照](npu_17.png) |
| 18 | [Random](random_npu_18.png) | [Ordered](ordered_npu_18.png) | [对照](npu_18.png) |
| 19 | [Random](random_npu_19.png) | [Ordered](ordered_npu_19.png) | [对照](npu_19.png) |
| 20 | [Random](random_npu_20.png) | [Ordered](ordered_npu_20.png) | [对照](npu_20.png) |
| 21 | [Random](random_npu_21.png) | [Ordered](ordered_npu_21.png) | [对照](npu_21.png) |
| 22 | [Random](random_npu_22.png) | [Ordered](ordered_npu_22.png) | [对照](npu_22.png) |
| 23 | [Random](random_npu_23.png) | [Ordered](ordered_npu_23.png) | [对照](npu_23.png) |
| 24 | [Random](random_npu_24.png) | [Ordered](ordered_npu_24.png) | [对照](npu_24.png) |
| 25 | [Random](random_npu_25.png) | [Ordered](ordered_npu_25.png) | [对照](npu_25.png) |
| 26 | [Random](random_npu_26.png) | [Ordered](ordered_npu_26.png) | [对照](npu_26.png) |
| 27 | [Random](random_npu_27.png) | [Ordered](ordered_npu_27.png) | [对照](npu_27.png) |
| 28 | [Random](random_npu_28.png) | [Ordered](ordered_npu_28.png) | [对照](npu_28.png) |
| 29 | [Random](random_npu_29.png) | [Ordered](ordered_npu_29.png) | [对照](npu_29.png) |
| 30 | [Random](random_npu_30.png) | [Ordered](ordered_npu_30.png) | [对照](npu_30.png) |
| 31 | [Random](random_npu_31.png) | [Ordered](ordered_npu_31.png) | [对照](npu_31.png) |

[完整周期定义与整机汇总](../fleet_cycle_ratio/README.md) · [逐周期数据和来源校验](../fleet_cycle_ratio/checks.json) · [本次生成校验](layer_average_checks.json) · [生成代码](../../../render_per_npu_layer_avg.py)
