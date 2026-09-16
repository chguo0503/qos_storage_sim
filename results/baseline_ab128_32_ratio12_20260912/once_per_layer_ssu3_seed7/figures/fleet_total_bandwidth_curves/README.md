# Once per layer：32 卡总带宽需求与供给

32 NPU、3 SSU × 40 GiB/s、seed 7，warm [2,4) 秒。使用完整 Once 仿真的原始日志。

| 顺序 | NPU 利用率 | 总需求均值 GiB/s | 总供给均值 GiB/s | 图 |
|---|---:|---:|---:|---|
| Ordered | 83.4170% | 254.197483 | 104.603186 | [PNG](ordered_total_demand_supply.png) · [PDF](ordered_total_demand_supply.pdf) |
| Random | 91.2428% | 188.887587 | 111.674396 | [PNG](random_total_demand_supply.png) · [PDF](random_total_demand_supply.pdf) |

紫色虚线：同一时刻 32 张卡当前请求的 Bi=V/C 之和。蓝线：先把每张卡实际收到的数据量按自己的层周期平均，再在同一时刻对 32 张卡求和。

层周期从当前层开始计算到下一层开始计算，包含等待。跨请求周期计入；窗口边界按窗内收到量和片段时长计算。曲线面积已与整窗实际收到量核对一致，完整内部周期另外核对读取字节及计算时间。

两条线之比不能作为 NPU 利用率。蓝线为各卡周期平均后的总量，各卡周期并不对齐，局部峰值不能用来判断磁盘是否超速。四张图均采用 0–1000 GiB/s 纵轴，与参考图一致。

[Ordered 曲线 CSV](ordered_curves.csv) · [Random 曲线 CSV](random_curves.csv) · [来源与校验](checks.json) · [绘图脚本](../../../render_once_fleet_total_bandwidth_curves.py)
