# 三类负载：ASU、OD、流量分配的 TTFT/SLO CDF

复用之前已完成的27次正式仿真。32 NPU、3 SSU×40 GiB/s、Ring hash、Random、每请求8层；主窗口为 [2,4) 秒。没有运行新仿真，旧图保持原样。

| 负载 | ASU SLO×1.5 | OD SLO×1.5 | 流量分配（Once）SLO×1.5 | 图片 |
|---|---:|---:|---:|---|
| 持续欠载 | 100.00% | 100.00% | 100.00% | [完整CDF](figures/under_ttft_slo_cdf.png) · [阈值附近放大](figures/under_ttft_slo_cdf_zoom.png) |
| 持续过载 | 38.70% | 34.62% | 71.61% | [完整CDF](figures/full_ttft_slo_cdf.png) · [阈值附近放大](figures/full_ttft_slo_cdf_zoom.png) |
| 局部过载（间歇过载） | 93.20% | 93.53% | 98.99% | [完整CDF](figures/semi_ttft_slo_cdf.png) · [阈值附近放大](figures/semi_ttft_slo_cdf_zoom.png) |

```text
横轴 = (prefill 完成时刻 - 请求接纳时刻) / 本请求 8 层纯计算时间
纵轴 = 三个 seed 各自经验 CDF 的等权平均
横轴 1.5 处的纵轴值 = SLO×1.5 达标率
```

seed 为7、19、43。每个seed内，三策略使用逐字节相同的冻结输入；只统计窗口内接纳的请求，并一直跟踪至完成，不删除窗后完成或不达标请求。不同策略推进速度不同，窗口内具体请求和数量可以不同。三个seed先各自计算CDF再等权平均，不能将全部请求简单拼接。

SLO沿用接纳后prefill完成的代理口径，不含接纳前队列等待，也未模拟真正的首token事件。分母是每个请求自己的纯计算时间，不是除以总体平均，也没有再除以1.5。数值边界沿用1e-9毫秒容差。

局部过载对应此前的semi输入，旧报告称“局部欠载”或“间歇过载”，表示有些阶段欠载、有些阶段至少一盘过载。负载名称沿用原组分类，不能自动推广为Once也在每个时刻满足同一状态。这里不是上一轮新构造的94% OD输入。

持续欠载三条曲线在归一化耗时1处重合，重合与达标率100%已明确标注。放大图仅改变显示范围，不改变CDF分母；完整图保留长尾。

- full/semi：`results/od_baseline_diverse_ssu3_20260918/runs/` 中对应三策略与三个seed。
- under ASU/OD：`results/continuous_underload_asu_od_20260918/runs/random_*`。
- under Once：`results/od_vs_once_three_loads_20260918/runs/under_once_seed*_local/`。
- [逐请求样本](request_samples.csv) · [画图数据](plot_data.json) · [绘图校验](render_checks.json) · [独立复核](independent_audit.md)

复现：`python results/od_vs_once_three_loads_20260918/three_strategy_cdf/render_cdf.py`
