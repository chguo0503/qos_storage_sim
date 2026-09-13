# 32 卡平均需求与供给：两条曲线

独立新增的PNG，Random、Ordered分开。32 NPU、3 SSU × 40 GiB/s、seed 7、warm [2,4) 秒。

- [Random PNG](random_mean_demand_supply.png)
- [Ordered PNG](ordered_mean_demand_supply.png)

横轴为仿真时间，纵轴为每卡平均带宽（GiB/s）。每张图只有两条曲线：

1. 需求：同一时刻32张卡各自当前请求的B_i=V/C相加，再除以32。
2. 供给：先把每张卡实际收到的数据量摊到自己的层周期（当前层开始计算到下一层开始计算，包含等待），再在同一时刻对32张卡求平均。

完整跨请求周期也按实际收到的字节数计算。窗口两端不完整的周期只使用窗内字节量除以窗内片段长度，避免漏计或额外计入窗外数据。因此新曲线面积可精确对齐原整窗统计。两图使用相同的坐标尺度。

蓝线不是原始瞬时吞吐，也不是存储承诺的可用带宽；不同卡的周期并不对齐。不要用它的局部峰值判断磁盘是否超速，也不要把两线之比当作瞬时或整窗NPU利用率。

所有旧PNG/PDF/SVG均保留，没有改写。

[Random曲线CSV](random_curves.csv) · [Ordered曲线CSV](ordered_curves.csv) · [来源与校验](checks.json)
