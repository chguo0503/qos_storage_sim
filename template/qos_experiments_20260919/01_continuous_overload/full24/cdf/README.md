# 持续过载：两条策略CDF

按原图数据重绘：删除固定候选池，仅保留Baseline与原始Once per layer，后者展示名改为“流量分配”。两条CDF、颜色、线型、横轴范围0.9–17与统计口径保持原图。

数据来源：https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/figures/ttft_cdf_three_strategies/cdf_points.csv

warm接纳窗口[2,4)秒；seed7/19/43的CDF等权平均；保留窗后完成的请求。SLO×1.5：Baseline40.2954308715%，流量分配76.1302681992%。本次仅重绘，没有重新仿真。

运行python render_cdf.py可重新生成PNG/SVG；依赖Python、NumPy、Matplotlib。字体随包附带。
