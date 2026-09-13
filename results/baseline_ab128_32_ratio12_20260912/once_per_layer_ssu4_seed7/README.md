# SSU4：Once per layer 利用率、TTFT SLO×1.5 和整机带宽

配置：32 NPU、4 SSU × 40 GiB/s、每卡接收链路 50 GiB/s、seed 7。每卡 40A＋80B、8 层、batch=1；全部请求在 t=0 到达。Random 为各卡独立洗牌，Ordered 为 ABB 重复 40 轮。

本次新增 Random、Ordered 两次完整 Once 仿真，每次完成 3840 个请求、15052800 个 I/O 块。每组直接复制对应 SSU4 Baseline 的冻结输入，输入 SHA 和核心代码 SHA 均一致，固定 NPU 绑定、落盘与请求顺序。所有数值为 seed 7。

Once per layer 沿用 `strategy='once'`：最近一次 5 ms 共享采样快照，每请求/层/SSU 一次规划全部 I/O 块路径。沿用类别允许路径、静态 CIR 和跨请求首层预取。控制通信和规划 CPU 的仿真延迟均为 0；Python 运行时间不计入 NPU 时序。

利用率 = 窗口内真实计算卡时间 / (32 × 窗口时长)。TTFT 沿用项目的接纳计时口径：8 层 prefill 完成时间 − 接纳时间，是首 token 延迟的代理，不含接纳前排队。

SLO×1.5 达标条件：延迟 ≤ 1.5 × 8 × 原始每层计算时间。A 阈值为 72.288139 ms；B 阈值为 343.114104 ms。样本为窗口内接纳的请求，每请求等权，跨窗完成的请求跟踪到最终完成并保留在分母中。

## 主窗口 [2,4) 秒

| 顺序 | 策略 | NPU 平均利用率 | TTFT SLO×1.5 达标率 | A 类达标率 | B 类达标率 |
|---|---|---:|---:|---:|---:|
| Random | Baseline | 99.4849% | 100.00%（373/373） | 100.00%（124/124） | 100.00%（249/249） |
| Random | Once per layer | 99.3008% | 100.00%（371/371） | 100.00%（120/120） | 100.00%（251/251） |
| Ordered | Baseline | 70.2349% | 62.50%（160/256） | 0.00%（0/96） | 100.00%（160/160） |
| Ordered | Once per layer | 93.5429% | 85.96%（306/356） | 55.75%（63/113） | 100.00%（243/243） |

## 补充窗口 [2,20) 秒

| 顺序 | 策略 | NPU 平均利用率 | TTFT SLO×1.5 达标率 | A 类达标率 | B 类达标率 |
|---|---|---:|---:|---:|---:|
| Random | Baseline | 98.9567% | 98.03%（3292/3358） | 94.00%（1034/1100） | 100.00%（2258/2258） |
| Random | Once per layer | 98.7922% | 97.73%（3277/3353） | 93.08%（1022/1098） | 100.00%（2255/2255） |
| Ordered | Baseline | 71.3373% | 66.23%（1632/2464） | 0.00%（0/832） | 100.00%（1632/1632） |
| Ordered | Once per layer | 97.9395% | 94.32%（3156/3346） | 82.88%（920/1110） | 100.00%（2236/2236） |

## 带宽图

- [四种情况总览 PNG](../figures/ssu4/fleet_total_bandwidth_curves.png)
- [Once Random PNG](../figures/ssu4/once_per_layer/random_total_demand_supply.png)
- [Once Ordered PNG](../figures/ssu4/once_per_layer/ordered_total_demand_supply.png)
- [Baseline Random PNG](../figures/ssu4/fleet_total_bandwidth_curves/random_total_demand_supply.png)
- [Baseline Ordered PNG](../figures/ssu4/fleet_total_bandwidth_curves/ordered_total_demand_supply.png)

图形沿用 SSU3 的定义与 0–1000 GiB/s 纵轴。紫色虚线为当前请求 Bi=V/C 对 32 卡求和；蓝线先按每卡层周期平均实际收到的数据量，再对 32 卡求和。周期从本层开始计算到下一层开始计算，包含 I/O 等待。跨请求周期计入，窗边片段按窗内收到量计算。

蓝线为周期平均供给，各卡周期不对齐。其局部峰值不能用于判断磁盘是否超速，两线之比也不能用作 NPU 利用率。已核对曲线面积与整窗实际收到字节量、逐盘/逐卡物理服务不重叠、完整内部周期字节和计算时间。

## 统计口径与复查

相同完整输入不代表窗口内接纳到同一批请求；应连同类别达标率与分母一起比较。每卡实际计算、活跃时长及窗口内计算过的类别保存在逐卡 CSV 中。

若对同一窗口内接纳样本改用“完成 − 外部到达”，达标率均为 0%，因为所有请求 t=0 到达。若选取窗口内新到达请求，则样本数为 0，达标率为 N/A。

[精确 CSV](comparison.csv) · [逐卡利用率 CSV](per_npu_utilization.csv) · [统计与校验 JSON](comparison.json) · [原始运行](runs/) · [复算脚本](analyze.py)

复现：先运行 `python run_once.py --order random` 和 `python run_once.py --order ordered`（运行目录已存在时拒绝覆盖）；再运行 `python analyze.py --cases baseline_random baseline_ordered once_random once_ordered --assemble`。
