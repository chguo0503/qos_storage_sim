# 4 NPU / 1 SSU Baseline Path0 最低利用率实验

## 结论

在统一的 1 秒中间窗口内，Baseline Path0 的平均 NPU 利用率为 **76.2261%**，四张卡分别为 **21.6560%, 83.2484%, 100.0000%, 100.0000%**。四条固定流的需求和为 **39.886158 GiB/s**，小于 40 GiB/s；所有 NPU 都有持续的 active request，且计算或 I/O barrier 覆盖完整 1 秒。

这不是已证明的连续参数全局最小值，而是报告所列有界搜索中的最低候选。最终统一口径又对短流的整数 NQL=153..192 共 40 个点逐一复跑；NQL=169 是该有限网格最低点。`data` 没有概率权重，因此“满足 data 分布”不能解释成统计抽样；这里保留 GLM-5.1 KV 关系，并在相邻 NQL 点间线性插值。NPU 0 的 1K 长度还使用了 32K/48K 计算时间斜率外推，所以它是模型构造，不是实测 profile。

## 输入与带宽定义

| NPU | role | total tokens | NQL | KV/layer (GiB) | compute/layer (us) | B (GiB/s) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | short_victim | 1024 | 169 | 0.001121163 | 582.149492 | 1.925903 |
| 1 | large_fast | 196608 | 368 | 0.257329941 | 12390.164802 | 20.768888 |
| 2 | large_moderate_a | 196608 | 896 | 0.256637573 | 29856.562816 | 8.595684 |
| 3 | large_moderate_b | 196608 | 896 | 0.256637573 | 29856.562816 | 8.595684 |

单流定义为 `B = 每层 SSD KV GiB / 每层理想计算秒数`。由于每个请求固定运行 8 层，分子和分母同时乘 8，B 不变。四条流各自固定 profile，因此任意时刻的输入需求和就是四个 B 的和，不使用周期平均来绕过 40 GiB/s 约束。代码沿用历史 `_gb/_gbps` 字段名，但 KV 由 `2^30` 归一化，实际单位是 GiB/GiB/s。

这也意味着字面量 **40 GB/s（十进制）只等于 37.252903 GiB/s**，主候选不满足这个更窄的上限。单位审计另给出一组 Baseline 输入：需求和 **37.172558 GiB/s = 39.913730 GB/s**，平均 NPU 利用率仍为 **77.7745%**。它证明结论不依赖把 GB 与 GiB 混用，但不是主搜索网格的最低点。

## 为什么 Path0 会降低利用率

Baseline 将全部物理块放入同一个 Path0 FCFS 队列，命令不可抢占。三个 192K 流每层约有 1,500 个物理块；NPU 0 每层只有数个块，但它们一旦排在已入队的大流块之后就不能越过。跨请求 Layer-0 预取已经开启，但模拟器仍只提前一层，无法为短流建立多层缓冲。

NPU 0 在窗口中的 barrier 为 **783.440 ms**；其中 SSD 正在服务其他 NPU 的重叠时间为 **779.590 ms**。NPU 1 也出现 **167.516 ms** barrier。因此平均值低不是因为某张卡没有请求，而是 active request 被 I/O barrier 阻塞。

只用于因果诊断的四 Path 对照保持输入、SSD、QoS 表和预取不变，只把 NPU 映射到 Path 0--3。其平均利用率为 **95.8124%**，比 Path0 高 **19.5863 个百分点**；NPU 0 从 **21.6560%** 恢复到 **100.0000%**。这支持“单 Path FCFS 队头阻塞是关键原因”。但 NPU 1 在四 Path 下仍约 83.25%，说明物理 SSD 突发与单层预取深度也是共同原因，不能把全部损失都归因于 Path 数量。

## 审计边界

- GLM-5.1 的 78 层只用于保持 `data` 中 source TTFT = 78 × per-layer compute 的关系；本实验按要求实际执行 8 层。中间窗口消除了首次冷启动 Layer-0 的影响，但每个新请求仍有 Layer-0，跨请求预取会改变 Path0 排队，所以不能说 Layer0 完全无关。
- `/home/chguo/work/last_code/qos_storage_sim/data` 的最大 200K 行超过 [GLM-5.1 官方 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json) 中的 202,752 token 上限（200×1024=204,800），本实验没有使用 200K。
- 当前结果是离散事件模型预测。1K profile 超出 `data` 的 32K--200K 长度网格，真实 GLM-5.1 硬件结论必须补测其 per-layer compute 和 KV 读流量。
- SSD 实际服务率为 **34.764181 GiB/s**，不是名义需求和；二者分别回答输入理想需求与物理执行吞吐。

## 文件

- [input_profiles.csv](../data/input_profiles.csv)：四条固定流。
- [request_layer_timeline.csv](../data/request_layer_timeline.csv)：窗口相交的逻辑 I/O、barrier 与 compute。
- [physical_block_trace.csv](../data/physical_block_trace.csv)：窗口相交的物理 Path0 块服务。
- [01_npu_io_compute_timeline.png](../figures/01_npu_io_compute_timeline.png)、[02_ssu_path0_enqueue_order.png](../figures/02_ssu_path0_enqueue_order.png)、[03_ssu_path0_service_timeline.png](../figures/03_ssu_path0_service_timeline.png)、[04_100ms_npu_utilization.png](../figures/04_100ms_npu_utilization.png)：与历史 timeline 方法对应的图。
- [simulator_summary.json](../data/simulator_summary.json)：未经裁剪的 Baseline 稳态摘要。
- [result.json](../data/result.json)：审计、归因和四 Path 机制对照。
- [search_summary.csv](../data/search_summary.csv)、[search_result.json](../data/search_result.json)：统一口径的 NQL=153..192 有界细网格。
