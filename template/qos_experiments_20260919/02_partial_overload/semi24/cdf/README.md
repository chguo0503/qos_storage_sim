# 间歇过载：Baseline 与流量分配 CDF

按原GitHub semi图的数据重绘：删除固定候选池，仅保留Baseline与原始Once per layer，后者显示为“流量分配”。沿用原曲线、线型、颜色、横轴0.9–3和全部长尾。

数据来源：https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/figures/ttft_cdf_three_strategies/cdf_points.csv

窗口为[2,4)秒内接纳的全部请求，跟踪至最终完成；seed7/19/43的CDF等权平均。SLO×1.5达标率：Baseline96.2705497503%，流量分配99.8095238095%。本次仅重绘，没有重跑仿真。

运行python render_cdf.py即可复现PNG和SVG；依赖Python、NumPy、Matplotlib。字体随包附带。
