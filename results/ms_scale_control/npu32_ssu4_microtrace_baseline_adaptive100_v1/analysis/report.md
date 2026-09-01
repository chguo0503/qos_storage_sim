# 32 NPU / 4 SSU 微观仿真审计

## 结论

精确微观 trace 内的 NPU 平均利用率分别为 Baseline **86.994%**、Adaptive 100 ms **58.611%**（差 -28.383 个百分点）。

这段 50 ms trace 的 SSD 平均利用率差异更大：Baseline **99.192%**、Adaptive **95.748%**。这不是稳态均值的逐点证据，而是两个流水线在各自 measurement 起点的短窗口相位切片。

完整 measurement summary 的 NPU 平均利用率分别为 **77.994%** 和 **82.550%**（差 +4.555 个百分点）；3 s SSD 平均利用率分别为 **94.874%** 和 **93.863%**。

从 `request_rows` 后处理重算的按 NPU 等权 TTFT SLO 为：α=1.5 时 **51.517%** 和 **64.105%**；α=2 时 **56.892%** 和 **85.010%**。

本次仿真的 primary SLO 是 α=2；α=1.5 与旧 SSU 曲线相同，都是在不改变控制器和仿真事件的前提下，使用 raw `ttft_ms <= 1.5 * ideal_ttft_ms` 后处理得到。

## 为什么 50 ms 不能作逐点相同比较

Baseline 与 Adaptive 的稳态 measurement 分别从绝对仿真时间 **10136.618341 ms** 和 **10906.062312 ms** 开始。warmup 达标时间由策略决定，因此两个相对时间 0 ms 并非同一流水线相位；compute、SSD dispatch 和 link completion 的短周期峰谷可以错开。

`04_full_3s_timeseries.png` 同时保留原生 5 ms 点、100 ms trailing rolling 和从 measurement 起点累计的 duty。最后 500 ms 内累计 NPU 曲线的摆幅仅为 Baseline **1.933 pp**、Adaptive **1.800 pp**；累计 SSD 曲线摆幅为 **0.473 pp** 和 **0.187 pp**，并在 3 s 终点精确闭合 summary。

因此，0.5 ms/50 ms trace 用来证明事件因果关系和计数是否真实闭合；策略的平均资源占用应看完整 3 s 积分，而不是要求两个短 trace 逐点相同。

## 3 s 与 frozen formal 8 s 的关系

新 3 s 结果不是一条不同随机轨迹：它与 frozen formal artifact 的 input fingerprint、measurement start 完全一致，而且 NPU 利用率数值精确等于旧 8 s 结果前 6 个 500 ms block 的累计值。

在 3 s 截点，Baseline 为 **77.994%**、Adaptive 为 **82.550%**，差 **+4.555 pp**；因此这 +4.555 pp 是选取前 3 s 窗口的结果，不是新 trace 改变了仿真。

继续积分到 formal 8 s，Baseline 收敛到 **78.639%**、Adaptive 收敛到 **78.186%**，差变为 **-0.452 pp**。`06_formal_8s_convergence.png` 展示 500 ms 原始波动及累计平均如何改变排序。

## Fleet 平均值掩盖的逐 NPU 重分配

Adaptive 相对 Baseline 的 fleet 平均变化是 **+4.555 pp**，但逐 NPU 变化的平均绝对值为 **9.865 pp**，最大绝对变化为 **54.590 pp**。

32 个 NPU 中，19 个利用率上升、13 个下降、0 个不变。`05_per_npu_measurement_utilization.png` 展示了这种重分配，说明相近的 fleet mean 不代表每个 NPU 行为相同。

微观记录能够闭合仿真内部的资源统计：精确服务区间重新积分后，与 5 ms summary block 及独立的 0.5 ms 分箱一致；瞬时每个 SSU 不超过 40 GB/s、每个 NPU link 不超过 50 GB/s、每个 NPU 同时最多一个 compute 区间。因而“利用率接近”不是 summary 把策略写成同一个值造成的。

但这不是对真实硬件计数器的验证。此 DES 把每个 SSU 建模为一次只运行一个、不可抢占、以 40 GB/s 执行的命令，把每个 NPU link 建模为一次只运行一个、以 50 GB/s 执行的流；CIR 改变离散命令的排序，而不是把同时在途 I/O 按比例限速。连续满负载下，各策略的总字节量和计算量接近，所以 fleet 平均利用率天然接近；策略更明显地改变的是谁先得到 I/O、单请求 barrier 和 TTFT/SLO。这个抽象内部自洽，但不能单独证明与真实 SSD/NVMe 并发和带宽整形足够接近。

Baseline 使用 category/generic Path，没有 NPU→dedicated Path 映射，因此不能定义逐 NPU 的 installed CIR；报告保留逐 SSU×Path 的 CIR、每条 flow 的实际 path_id 以及 dispatch 时该 Path 的 CIR。Adaptive 100 ms 才额外报告 dedicated Path 映射下的逐 NPU CIR 总和。分析没有为 Baseline 虚构 NPU CIR。

## 请求级因果链示例

Baseline 与 Adaptive 各自从本策略 trace 自动选取一个完整 layer；两者 measurement 起点和 active cohort 不同，因此不把它们称为同一请求配对。

`03`/`03b` 顶图只画所选 request/layer 对各资源的服务贡献，并按 0.02 ms interval overlap 求 bin 平均；它不是整台 SSD 或整个 NPU link 的瞬时总带宽。原始模型中，一个正在服务的 SSD 命令固定为 40 GB/s，一个正在服务的 NPU link flow 固定为 50 GB/s。

### Baseline：request=549，NPU=5，layer=7

- `io_ready = max(link_end) = 10178.556632544 ms`，与 simulator 字段的误差为 0.000e+00 ms。
- `compute_start = max(io_ready, previous_compute_end) = 10178.556632544 ms`，重建误差为 0.000e+00 ms。
- `barrier = compute_start - previous_compute_end = 14.995135933 ms`，与 simulator 字段的误差为 0.000e+00 ms。

### Adaptive: interval 100 ms：request=710，NPU=6，layer=10

- `io_ready = max(link_end) = 10933.781441867 ms`，与 simulator 字段的误差为 0.000e+00 ms。
- `compute_start = max(io_ready, previous_compute_end) = 10933.781441867 ms`，重建误差为 0.000e+00 ms。
- `barrier = compute_start - previous_compute_end = 15.584846433 ms`，与 simulator 字段的误差为 0.000e+00 ms。

## 校验

- Baseline：capacity checks=PASS；50 ms closure=PASS；full 3 s/5 ms reconstruction=PASS；最大绝对闭合残差=3.638e-12。
- Adaptive: interval 100 ms：capacity checks=PASS；50 ms closure=PASS；full 3 s/5 ms reconstruction=PASS；最大绝对闭合残差=4.547e-12。
- Frozen formal 8 s prefix：PASS；新 3 s 与旧前六 block 最大残差=0.000e+00。

## 文件

- `service_intervals.csv`：每条 SSD、NPU link、compute 的原始精确起止时间、实际速率、请求/NPU/SSU/layer 标识。
- `npu_timeseries.csv`：0.5 ms、逐 NPU 的 compute 利用率和实际 link 带宽。
- `ssu_timeseries.csv`：0.5 ms、逐 SSU 的实际带宽和占用率。
- `fleet_timeseries.csv`：0.5 ms 的 fleet 聚合值。
- `cir_timeseries.csv`：0.5 ms、逐 SSU×相关 Path 的已安装 CIR 时间加权值；Adaptive 的 dedicated Path 行带 NPU ID，Baseline 的 NPU ID 为 N/A。
- `request_details.csv`、`layer_details.csv`：请求级与 layer 因果分解。
- `selected_layer_service_intervals.csv`：自动选出的 Baseline/Adaptive 两个完整 layer 的轻量原始 SSD/link/compute 区间。
- `03_selected_layer_micro_timeline.png`、`03b_baseline_selected_layer_micro_timeline.png`：Adaptive 与 Baseline 各自的请求级因果时间线；带宽面板是 selected-layer contribution 的 0.02 ms bin 平均。
- `case_summary.csv`：新 3 s measurement 的利用率，以及从 raw request rows 重算的 α=1.5/α=2 两组 SLO。
- `full_measurement_5ms_timeseries.csv`：完整 3 s 的 fleet 原生 5 ms、100 ms rolling 与累计资源占用。
- `full_measurement_npu_5ms_timeseries.csv`：完整 3 s 的逐 NPU 5 ms、rolling 与累计 compute duty。
- `validation.json`：容量、输入配对和两级积分闭合的机器可读证据。
- `historical_8s_convergence.csv`：frozen formal 的 16×500 ms 原始利用率与累计利用率。
- `06_formal_8s_convergence.png`：3 s 截点到 8 s 终点的窗口收敛图。

时间区间统一采用半开语义 `[start, end)`；图中的 0.5 ms 点是精确 interval overlap 的平均值，不是采样时刻的瞬时猜测。
