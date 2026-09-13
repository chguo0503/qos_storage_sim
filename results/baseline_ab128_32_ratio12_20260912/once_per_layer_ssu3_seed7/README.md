# 相同输入下的 Baseline / Once per layer 对照

配置：32 NPU、3 SSU × 40 GiB/s、每卡接收链路50 GiB/s、seed 7。每卡40A＋80B，8层，batch=1，全部请求t=0到达，固定NPU绑定和数据落盘。Random各卡独立洗牌；Ordered各卡按ABB重复40轮。沿用跨请求首层预取。

Once per layer对应现有`strategy='once'`：使用最近一次5ms共享采样快照，每请求/层/SSU一次规划全部I/O块路径，沿用类别允许路径及静态CIR。不使用`once_native`或`new_once`；不改变L1分卡或L2请求顺序。

两种策略均未计入控制通信和规划CPU的仿真延迟（设为0）；Python运行耗时不计入NPU计算时序。静态CIR不是不能借用的带宽上限，空闲服务份额可分配给有积压的Path；Once没有额外物理带宽。

每种顺序的Once直接复制原Baseline的manifest，字节级SHA相同；核心源码也相同。两次完整有限请求仿真均执行到全部块完成，主图只显示预定warm窗口[2,4)秒。所有数字均为seed 7，不能当作多种子均值。

## 主统计窗口 [2,4) 秒

| 请求顺序 | Baseline U | Once per layer U | Once − Baseline，百分点 |
|---|---:|---:|---:|
| Random | 90.6779% | 91.2428% | +0.5649 |
| Ordered | 71.0089% | 83.4170% | +12.4081 |

## 补充长窗口 [2,20) 秒

| 请求顺序 | Baseline U | Once per layer U | Once − Baseline，百分点 |
|---|---:|---:|---:|
| Random | 91.2599% | 90.8643% | -0.3956 |
| Ordered | 64.2162% | 90.4310% | +26.2147 |

利用率统一按窗口内真实计算卡时间 / (32 × 窗口时长)，包含I/O等待与窗口边界。差值使用百分点，不是相对百分比。

## 两张32卡带宽图

- [Random — Once per layer](figures/random_all_32npu_layer_average.png)
- [Ordered — Once per layer](figures/ordered_all_32npu_layer_average.png)
- [原Random — Baseline](../figures/ssu3/random_all_32npu_layer_average.png)
- [原Ordered — Baseline](../figures/ssu3/ordered_all_32npu_layer_average.png)

每行一张卡。紫虚线为当前请求的Bi=V/C；蓝线为同一请求内部完整层周期内实际收到的下一层数据量/周期时长，周期包含等待。跨请求和窗口截断段标灰、不填蓝线，不代表供给为零。左右的利用率与带宽数值完整统计[2,4)秒；右侧供给包含灰区真实收到的字节。两个整窗带宽均值相除不等于利用率。

- Random Once：全32卡整窗有任务=True；窗口内计算过A和B的卡数=32/32。
- Ordered Once：全32卡整窗有任务=True；窗口内计算过A和B的卡数=32/32。

## 主窗口的请求类别统计

此处类别利用率=该类实际计算时间/该类已接纳占用卡时间，包括首层交接等待；不是图中的完整内部周期比值。

| 顺序 | 策略 | 类别 | 类别利用率 | 窗口卡时间占比 |
|---|---|---|---:|---:|
| Random | baseline | A | 49.1291% | 16.9956% |
| Random | baseline | B | 99.1853% | 83.0044% |
| Random | once | A | 49.9659% | 16.8912% |
| Random | once | B | 99.6320% | 83.1088% |
| Ordered | baseline | A | 15.1393% | 31.8325% |
| Ordered | baseline | B | 97.0986% | 68.1675% |
| Ordered | once | A | 32.7346% | 24.4054% |
| Ordered | once | B | 99.7796% | 75.5946% |

输入保持一致不代表窗口内执行到同一批请求：调度改变推进速度和A/B驻留比例。此实验存在名义需求超限，不能称为逐盘逐时刻欠载对照；结果不能推广为任意输入下某策略更优。

[精确数值CSV](comparison.csv) · [来源和校验JSON](comparison.json) · [运行脚本](run_once.py) · [绘图脚本](render_fleet.py)
