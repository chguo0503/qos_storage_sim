# 时序与逐卡带宽图

最终图使用长验证人口、seed 7、[2,4) 秒；新增图仅生成 PNG，旧图保留已有 PDF。

**3 SSU 新图：[统一入口](ssu3/index.md)。** 相同两秒 warm 窗口、seed 7，Random 实测 U=90.67794%，Ordered=71.00895%；新增图只生成 PNG。

**逐卡看周期平均带宽：[4 SSU 逐卡 PNG 索引](ssu4/per_npu/index.md)。** 每卡分别提供 Random、Ordered。蓝线改为每层周期的平均实际接收带宽；内部周期从开始计算当前层，到开始计算下一层，包含中间等待，收到的是下一层数据。`B_i` 保持原义：当前已接纳请求的 `V/C`。灰色区间涉及跨请求或窗口截断，不套内部周期的 `b_i/B_i` 比值。仅生成 PNG；原 `npu_XX.png` 更新为周期平均的合并对照，原 `npu_XX.pdf` 保留历史 0.5 ms 带宽采样图，两者口径不同。

**看 b_i/B_i 怎样汇总成整机利用率：[Random 全32卡PNG](ssu4/fleet_cycle_ratio/fleet_cycle_ratio_random.png) · [Ordered 全32卡PNG](ssu4/fleet_cycle_ratio/fleet_cycle_ratio_ordered.png) · [公式、数值和读图说明](ssu4/fleet_cycle_ratio/README.md)。** 每个色块是同一请求内部的完整层周期，按周期平均实际接收带宽计算比值；按周期时长加权，并补回跨请求和窗口边界的真实计算时间，得到同一warm窗口整机U。4 SSU、seed 7：Random 99.48%，Ordered 70.23%。

**看单请求细节：[内部逐层带宽 PNG 索引](ssu4/internal_layer_zoom/index.md)。** 4 SSU、32 卡，每卡分成 Random A、Ordered A、Random B、Ordered B 四张图。每张只含同一请求的3个完整内部层周期，读取量和 B_i 不变，没有跨请求或读取边缘裁剪。NPU15：[Random A](ssu4/internal_layer_zoom/random_a_npu_15.png) · [Ordered A](ssu4/internal_layer_zoom/ordered_a_npu_15.png) · [Random B](ssu4/internal_layer_zoom/random_b_npu_15.png) · [Ordered B](ssu4/internal_layer_zoom/ordered_b_npu_15.png)。

此前的 [A→B 交接 PNG](ssu4/per_npu_zoom/index.md) 保留用于专门检查跨请求首层预取，不作为基础带宽图的主入口。

<a id="ssu3"></a>
## 3 SSU × 40 GiB/s

[逐卡周期平均带宽](ssu3/per_npu/index.md) · [同请求内部局部图](ssu3/internal_layer_zoom/index.md) · [A→B 交接局部图](ssu3/per_npu_zoom/index.md) · [周期比值与整机汇总说明](ssu3/fleet_cycle_ratio/README.md)

NPU 15 周期平均带宽：[Random PNG](ssu3/per_npu/random_npu_15.png) · [Ordered PNG](ssu3/per_npu/ordered_npu_15.png)。整体 32 卡周期比值：[Random PNG](ssu3/fleet_cycle_ratio/fleet_cycle_ratio_random.png) · [Ordered PNG](ssu3/fleet_cycle_ratio/fleet_cycle_ratio_ordered.png)。

[Random 2秒时序](ssu3/baseline_random.png) · [Ordered 2秒时序](ssu3/baseline_ordered.png) · [Ordered 局部](ssu3/ordered_B_zoom.png)

[Random 18秒时序](ssu3/long/baseline_random.png) · [Ordered 18秒时序](ssu3/long/baseline_ordered.png)

[Random 带宽总览](ssu3/bandwidth_overview_random.png) · [Ordered 带宽总览](ssu3/bandwidth_overview_ordered.png)

| NPU | Random PNG | Ordered PNG | 历史瞬时 PDF |
|---:|---|---|---|
| 0 | [Random](ssu3/per_npu/random_npu_00.png) | [Ordered](ssu3/per_npu/ordered_npu_00.png) | [PDF](ssu3/per_npu/npu_00.pdf) |
| 1 | [Random](ssu3/per_npu/random_npu_01.png) | [Ordered](ssu3/per_npu/ordered_npu_01.png) | [PDF](ssu3/per_npu/npu_01.pdf) |
| 2 | [Random](ssu3/per_npu/random_npu_02.png) | [Ordered](ssu3/per_npu/ordered_npu_02.png) | [PDF](ssu3/per_npu/npu_02.pdf) |
| 3 | [Random](ssu3/per_npu/random_npu_03.png) | [Ordered](ssu3/per_npu/ordered_npu_03.png) | [PDF](ssu3/per_npu/npu_03.pdf) |
| 4 | [Random](ssu3/per_npu/random_npu_04.png) | [Ordered](ssu3/per_npu/ordered_npu_04.png) | [PDF](ssu3/per_npu/npu_04.pdf) |
| 5 | [Random](ssu3/per_npu/random_npu_05.png) | [Ordered](ssu3/per_npu/ordered_npu_05.png) | [PDF](ssu3/per_npu/npu_05.pdf) |
| 6 | [Random](ssu3/per_npu/random_npu_06.png) | [Ordered](ssu3/per_npu/ordered_npu_06.png) | [PDF](ssu3/per_npu/npu_06.pdf) |
| 7 | [Random](ssu3/per_npu/random_npu_07.png) | [Ordered](ssu3/per_npu/ordered_npu_07.png) | [PDF](ssu3/per_npu/npu_07.pdf) |
| 8 | [Random](ssu3/per_npu/random_npu_08.png) | [Ordered](ssu3/per_npu/ordered_npu_08.png) | [PDF](ssu3/per_npu/npu_08.pdf) |
| 9 | [Random](ssu3/per_npu/random_npu_09.png) | [Ordered](ssu3/per_npu/ordered_npu_09.png) | [PDF](ssu3/per_npu/npu_09.pdf) |
| 10 | [Random](ssu3/per_npu/random_npu_10.png) | [Ordered](ssu3/per_npu/ordered_npu_10.png) | [PDF](ssu3/per_npu/npu_10.pdf) |
| 11 | [Random](ssu3/per_npu/random_npu_11.png) | [Ordered](ssu3/per_npu/ordered_npu_11.png) | [PDF](ssu3/per_npu/npu_11.pdf) |
| 12 | [Random](ssu3/per_npu/random_npu_12.png) | [Ordered](ssu3/per_npu/ordered_npu_12.png) | [PDF](ssu3/per_npu/npu_12.pdf) |
| 13 | [Random](ssu3/per_npu/random_npu_13.png) | [Ordered](ssu3/per_npu/ordered_npu_13.png) | [PDF](ssu3/per_npu/npu_13.pdf) |
| 14 | [Random](ssu3/per_npu/random_npu_14.png) | [Ordered](ssu3/per_npu/ordered_npu_14.png) | [PDF](ssu3/per_npu/npu_14.pdf) |
| 15 | [Random](ssu3/per_npu/random_npu_15.png) | [Ordered](ssu3/per_npu/ordered_npu_15.png) | [PDF](ssu3/per_npu/npu_15.pdf) |
| 16 | [Random](ssu3/per_npu/random_npu_16.png) | [Ordered](ssu3/per_npu/ordered_npu_16.png) | [PDF](ssu3/per_npu/npu_16.pdf) |
| 17 | [Random](ssu3/per_npu/random_npu_17.png) | [Ordered](ssu3/per_npu/ordered_npu_17.png) | [PDF](ssu3/per_npu/npu_17.pdf) |
| 18 | [Random](ssu3/per_npu/random_npu_18.png) | [Ordered](ssu3/per_npu/ordered_npu_18.png) | [PDF](ssu3/per_npu/npu_18.pdf) |
| 19 | [Random](ssu3/per_npu/random_npu_19.png) | [Ordered](ssu3/per_npu/ordered_npu_19.png) | [PDF](ssu3/per_npu/npu_19.pdf) |
| 20 | [Random](ssu3/per_npu/random_npu_20.png) | [Ordered](ssu3/per_npu/ordered_npu_20.png) | [PDF](ssu3/per_npu/npu_20.pdf) |
| 21 | [Random](ssu3/per_npu/random_npu_21.png) | [Ordered](ssu3/per_npu/ordered_npu_21.png) | [PDF](ssu3/per_npu/npu_21.pdf) |
| 22 | [Random](ssu3/per_npu/random_npu_22.png) | [Ordered](ssu3/per_npu/ordered_npu_22.png) | [PDF](ssu3/per_npu/npu_22.pdf) |
| 23 | [Random](ssu3/per_npu/random_npu_23.png) | [Ordered](ssu3/per_npu/ordered_npu_23.png) | [PDF](ssu3/per_npu/npu_23.pdf) |
| 24 | [Random](ssu3/per_npu/random_npu_24.png) | [Ordered](ssu3/per_npu/ordered_npu_24.png) | [PDF](ssu3/per_npu/npu_24.pdf) |
| 25 | [Random](ssu3/per_npu/random_npu_25.png) | [Ordered](ssu3/per_npu/ordered_npu_25.png) | [PDF](ssu3/per_npu/npu_25.pdf) |
| 26 | [Random](ssu3/per_npu/random_npu_26.png) | [Ordered](ssu3/per_npu/ordered_npu_26.png) | [PDF](ssu3/per_npu/npu_26.pdf) |
| 27 | [Random](ssu3/per_npu/random_npu_27.png) | [Ordered](ssu3/per_npu/ordered_npu_27.png) | [PDF](ssu3/per_npu/npu_27.pdf) |
| 28 | [Random](ssu3/per_npu/random_npu_28.png) | [Ordered](ssu3/per_npu/ordered_npu_28.png) | [PDF](ssu3/per_npu/npu_28.pdf) |
| 29 | [Random](ssu3/per_npu/random_npu_29.png) | [Ordered](ssu3/per_npu/ordered_npu_29.png) | [PDF](ssu3/per_npu/npu_29.pdf) |
| 30 | [Random](ssu3/per_npu/random_npu_30.png) | [Ordered](ssu3/per_npu/ordered_npu_30.png) | [PDF](ssu3/per_npu/npu_30.pdf) |
| 31 | [Random](ssu3/per_npu/random_npu_31.png) | [Ordered](ssu3/per_npu/ordered_npu_31.png) | [PDF](ssu3/per_npu/npu_31.pdf) |

[Random 原始带宽分箱](ssu3/bandwidth_random.csv.gz) · [Ordered 原始带宽分箱](ssu3/bandwidth_ordered.csv.gz)

<a id="ssu4"></a>
## 4 SSU × 40 GiB/s

[Random 2秒时序](ssu4/baseline_random.png) · [Ordered 2秒时序](ssu4/baseline_ordered.png) · [Ordered 局部](ssu4/ordered_B_zoom.png)

[Random 18秒时序](ssu4/long/baseline_random.png) · [Ordered 18秒时序](ssu4/long/baseline_ordered.png)

[Random 带宽总览](ssu4/bandwidth_overview_random.png) · [Ordered 带宽总览](ssu4/bandwidth_overview_ordered.png)

| NPU | Random PNG | Ordered PNG | 历史瞬时 PDF |
|---:|---|---|---|
| 0 | [Random](ssu4/per_npu/random_npu_00.png) | [Ordered](ssu4/per_npu/ordered_npu_00.png) | [PDF](ssu4/per_npu/npu_00.pdf) |
| 1 | [Random](ssu4/per_npu/random_npu_01.png) | [Ordered](ssu4/per_npu/ordered_npu_01.png) | [PDF](ssu4/per_npu/npu_01.pdf) |
| 2 | [Random](ssu4/per_npu/random_npu_02.png) | [Ordered](ssu4/per_npu/ordered_npu_02.png) | [PDF](ssu4/per_npu/npu_02.pdf) |
| 3 | [Random](ssu4/per_npu/random_npu_03.png) | [Ordered](ssu4/per_npu/ordered_npu_03.png) | [PDF](ssu4/per_npu/npu_03.pdf) |
| 4 | [Random](ssu4/per_npu/random_npu_04.png) | [Ordered](ssu4/per_npu/ordered_npu_04.png) | [PDF](ssu4/per_npu/npu_04.pdf) |
| 5 | [Random](ssu4/per_npu/random_npu_05.png) | [Ordered](ssu4/per_npu/ordered_npu_05.png) | [PDF](ssu4/per_npu/npu_05.pdf) |
| 6 | [Random](ssu4/per_npu/random_npu_06.png) | [Ordered](ssu4/per_npu/ordered_npu_06.png) | [PDF](ssu4/per_npu/npu_06.pdf) |
| 7 | [Random](ssu4/per_npu/random_npu_07.png) | [Ordered](ssu4/per_npu/ordered_npu_07.png) | [PDF](ssu4/per_npu/npu_07.pdf) |
| 8 | [Random](ssu4/per_npu/random_npu_08.png) | [Ordered](ssu4/per_npu/ordered_npu_08.png) | [PDF](ssu4/per_npu/npu_08.pdf) |
| 9 | [Random](ssu4/per_npu/random_npu_09.png) | [Ordered](ssu4/per_npu/ordered_npu_09.png) | [PDF](ssu4/per_npu/npu_09.pdf) |
| 10 | [Random](ssu4/per_npu/random_npu_10.png) | [Ordered](ssu4/per_npu/ordered_npu_10.png) | [PDF](ssu4/per_npu/npu_10.pdf) |
| 11 | [Random](ssu4/per_npu/random_npu_11.png) | [Ordered](ssu4/per_npu/ordered_npu_11.png) | [PDF](ssu4/per_npu/npu_11.pdf) |
| 12 | [Random](ssu4/per_npu/random_npu_12.png) | [Ordered](ssu4/per_npu/ordered_npu_12.png) | [PDF](ssu4/per_npu/npu_12.pdf) |
| 13 | [Random](ssu4/per_npu/random_npu_13.png) | [Ordered](ssu4/per_npu/ordered_npu_13.png) | [PDF](ssu4/per_npu/npu_13.pdf) |
| 14 | [Random](ssu4/per_npu/random_npu_14.png) | [Ordered](ssu4/per_npu/ordered_npu_14.png) | [PDF](ssu4/per_npu/npu_14.pdf) |
| 15 | [Random](ssu4/per_npu/random_npu_15.png) | [Ordered](ssu4/per_npu/ordered_npu_15.png) | [PDF](ssu4/per_npu/npu_15.pdf) |
| 16 | [Random](ssu4/per_npu/random_npu_16.png) | [Ordered](ssu4/per_npu/ordered_npu_16.png) | [PDF](ssu4/per_npu/npu_16.pdf) |
| 17 | [Random](ssu4/per_npu/random_npu_17.png) | [Ordered](ssu4/per_npu/ordered_npu_17.png) | [PDF](ssu4/per_npu/npu_17.pdf) |
| 18 | [Random](ssu4/per_npu/random_npu_18.png) | [Ordered](ssu4/per_npu/ordered_npu_18.png) | [PDF](ssu4/per_npu/npu_18.pdf) |
| 19 | [Random](ssu4/per_npu/random_npu_19.png) | [Ordered](ssu4/per_npu/ordered_npu_19.png) | [PDF](ssu4/per_npu/npu_19.pdf) |
| 20 | [Random](ssu4/per_npu/random_npu_20.png) | [Ordered](ssu4/per_npu/ordered_npu_20.png) | [PDF](ssu4/per_npu/npu_20.pdf) |
| 21 | [Random](ssu4/per_npu/random_npu_21.png) | [Ordered](ssu4/per_npu/ordered_npu_21.png) | [PDF](ssu4/per_npu/npu_21.pdf) |
| 22 | [Random](ssu4/per_npu/random_npu_22.png) | [Ordered](ssu4/per_npu/ordered_npu_22.png) | [PDF](ssu4/per_npu/npu_22.pdf) |
| 23 | [Random](ssu4/per_npu/random_npu_23.png) | [Ordered](ssu4/per_npu/ordered_npu_23.png) | [PDF](ssu4/per_npu/npu_23.pdf) |
| 24 | [Random](ssu4/per_npu/random_npu_24.png) | [Ordered](ssu4/per_npu/ordered_npu_24.png) | [PDF](ssu4/per_npu/npu_24.pdf) |
| 25 | [Random](ssu4/per_npu/random_npu_25.png) | [Ordered](ssu4/per_npu/ordered_npu_25.png) | [PDF](ssu4/per_npu/npu_25.pdf) |
| 26 | [Random](ssu4/per_npu/random_npu_26.png) | [Ordered](ssu4/per_npu/ordered_npu_26.png) | [PDF](ssu4/per_npu/npu_26.pdf) |
| 27 | [Random](ssu4/per_npu/random_npu_27.png) | [Ordered](ssu4/per_npu/ordered_npu_27.png) | [PDF](ssu4/per_npu/npu_27.pdf) |
| 28 | [Random](ssu4/per_npu/random_npu_28.png) | [Ordered](ssu4/per_npu/ordered_npu_28.png) | [PDF](ssu4/per_npu/npu_28.pdf) |
| 29 | [Random](ssu4/per_npu/random_npu_29.png) | [Ordered](ssu4/per_npu/ordered_npu_29.png) | [PDF](ssu4/per_npu/npu_29.pdf) |
| 30 | [Random](ssu4/per_npu/random_npu_30.png) | [Ordered](ssu4/per_npu/ordered_npu_30.png) | [PDF](ssu4/per_npu/npu_30.pdf) |
| 31 | [Random](ssu4/per_npu/random_npu_31.png) | [Ordered](ssu4/per_npu/ordered_npu_31.png) | [PDF](ssu4/per_npu/npu_31.pdf) |

[Random 原始带宽分箱](ssu4/bandwidth_random.csv.gz) · [Ordered 原始带宽分箱](ssu4/bandwidth_ordered.csv.gz)

<a id="ssu6"></a>
## 6 SSU × 40 GiB/s

[Random 2秒时序](ssu6/baseline_random.png) · [Ordered 2秒时序](ssu6/baseline_ordered.png) · [Ordered 局部](ssu6/ordered_B_zoom.png)

[Random 18秒时序](ssu6/long/baseline_random.png) · [Ordered 18秒时序](ssu6/long/baseline_ordered.png)

[Random 带宽总览](ssu6/bandwidth_overview_random.png) · [Ordered 带宽总览](ssu6/bandwidth_overview_ordered.png)

| NPU | 带宽对照 PNG | PDF |
|---:|---|---|
| 0 | [PNG](ssu6/per_npu/npu_00.png) | [PDF](ssu6/per_npu/npu_00.pdf) |
| 1 | [PNG](ssu6/per_npu/npu_01.png) | [PDF](ssu6/per_npu/npu_01.pdf) |
| 2 | [PNG](ssu6/per_npu/npu_02.png) | [PDF](ssu6/per_npu/npu_02.pdf) |
| 3 | [PNG](ssu6/per_npu/npu_03.png) | [PDF](ssu6/per_npu/npu_03.pdf) |
| 4 | [PNG](ssu6/per_npu/npu_04.png) | [PDF](ssu6/per_npu/npu_04.pdf) |
| 5 | [PNG](ssu6/per_npu/npu_05.png) | [PDF](ssu6/per_npu/npu_05.pdf) |
| 6 | [PNG](ssu6/per_npu/npu_06.png) | [PDF](ssu6/per_npu/npu_06.pdf) |
| 7 | [PNG](ssu6/per_npu/npu_07.png) | [PDF](ssu6/per_npu/npu_07.pdf) |
| 8 | [PNG](ssu6/per_npu/npu_08.png) | [PDF](ssu6/per_npu/npu_08.pdf) |
| 9 | [PNG](ssu6/per_npu/npu_09.png) | [PDF](ssu6/per_npu/npu_09.pdf) |
| 10 | [PNG](ssu6/per_npu/npu_10.png) | [PDF](ssu6/per_npu/npu_10.pdf) |
| 11 | [PNG](ssu6/per_npu/npu_11.png) | [PDF](ssu6/per_npu/npu_11.pdf) |
| 12 | [PNG](ssu6/per_npu/npu_12.png) | [PDF](ssu6/per_npu/npu_12.pdf) |
| 13 | [PNG](ssu6/per_npu/npu_13.png) | [PDF](ssu6/per_npu/npu_13.pdf) |
| 14 | [PNG](ssu6/per_npu/npu_14.png) | [PDF](ssu6/per_npu/npu_14.pdf) |
| 15 | [PNG](ssu6/per_npu/npu_15.png) | [PDF](ssu6/per_npu/npu_15.pdf) |
| 16 | [PNG](ssu6/per_npu/npu_16.png) | [PDF](ssu6/per_npu/npu_16.pdf) |
| 17 | [PNG](ssu6/per_npu/npu_17.png) | [PDF](ssu6/per_npu/npu_17.pdf) |
| 18 | [PNG](ssu6/per_npu/npu_18.png) | [PDF](ssu6/per_npu/npu_18.pdf) |
| 19 | [PNG](ssu6/per_npu/npu_19.png) | [PDF](ssu6/per_npu/npu_19.pdf) |
| 20 | [PNG](ssu6/per_npu/npu_20.png) | [PDF](ssu6/per_npu/npu_20.pdf) |
| 21 | [PNG](ssu6/per_npu/npu_21.png) | [PDF](ssu6/per_npu/npu_21.pdf) |
| 22 | [PNG](ssu6/per_npu/npu_22.png) | [PDF](ssu6/per_npu/npu_22.pdf) |
| 23 | [PNG](ssu6/per_npu/npu_23.png) | [PDF](ssu6/per_npu/npu_23.pdf) |
| 24 | [PNG](ssu6/per_npu/npu_24.png) | [PDF](ssu6/per_npu/npu_24.pdf) |
| 25 | [PNG](ssu6/per_npu/npu_25.png) | [PDF](ssu6/per_npu/npu_25.pdf) |
| 26 | [PNG](ssu6/per_npu/npu_26.png) | [PDF](ssu6/per_npu/npu_26.pdf) |
| 27 | [PNG](ssu6/per_npu/npu_27.png) | [PDF](ssu6/per_npu/npu_27.pdf) |
| 28 | [PNG](ssu6/per_npu/npu_28.png) | [PDF](ssu6/per_npu/npu_28.pdf) |
| 29 | [PNG](ssu6/per_npu/npu_29.png) | [PDF](ssu6/per_npu/npu_29.pdf) |
| 30 | [PNG](ssu6/per_npu/npu_30.png) | [PDF](ssu6/per_npu/npu_30.pdf) |
| 31 | [PNG](ssu6/per_npu/npu_31.png) | [PDF](ssu6/per_npu/npu_31.pdf) |

[Random 原始带宽分箱](ssu6/bandwidth_random.csv.gz) · [Ordered 原始带宽分箱](ssu6/bandwidth_ordered.csv.gz)
