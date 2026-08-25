# 仅客户端 Path 租约约束下的静态 CIR QoS 探索

日期：2026-08-24

## 范围与硬性约束

- 128 个 NPU；每个 SSU 带宽为 40 GB/s；每个 SSU 有 256 个 QoS Path，分为 8 组，每组 32 个。
- 调度顺序为 CIR 保证、固定权重的分层 WRR，最后由每个 Path 以 FCFS 方式提供单活跃 I/O 服务。PIR 不限速。
- Baseline 绕过 QoS，所有 block I/O 在工作保持（work-conserving）的 RR 调度下共同竞争。
- 主实验中的 `static-CIR` 只改变启动时配置的 CIR，Path 和组级 WRR 权重均固定为 1。
- 客户端只有在获得一个未被其他 NPU 租用的 Path 后，才能提交 block。尚未派发的 block 留在客户端；SSU 内部不维护 owner 或 lease 表。
- 路由只使用请求元数据（`required_bw`、类别、layer）、静态配置和各 Path 的 outstanding I/O 数量。
- 硬性门槛是平均单请求 NPU utilization 至少提升 10 个百分点，同时所有请求类别的 p95 TTFT 均不得恶化，且不能出现饥饿。

除非另有说明，下文所有结果均使用 seed 42、4 个仿真 layer、`ls_ratio=0.5`、随机 placement 和 128 个请求。

## 为什么淘汰此前 40 SSU、+10.40pp 的结果

此前 `[SS,SL,LS,LL]=[20,3,13,4]` GB/s 的配置点，主要依靠为 SS 启动流量提供大容量、相互隔离的 CIR 池来获得增益。SS utilization 从约 20.3% 上升到 58.8%，从而拉高了四种等量请求类别的非加权平均值。但它不能作为最终有效结果：

1. LS p95 TTFT 增加约 3.71 ms，LL 增加约 1.01 ms，因此未通过“尾延迟不得恶化”的硬性门槛。
2. 当时的 eager 路由可能把一个 I/O 排到已被其他 NPU 占用的 Path 后面，违反了后来明确的空闲 Path 租约规则。

旧的排队模型还在 84 SSU 下得到过一个看似有希望的合并结果：6 个 seed 的平均增益为 +10.17pp。但 LL 的合并 p95 增加了 0.88 ms，而且仍然存在相同的租约违规问题。因此，这个结果只能用于诊断性能上界，不能作为符合约束的结果。

## 严格租约实现

`ClientPathLeaseDispatcher` 是位于未改动的 SSU QoS 调度器之上的 NPU 端逻辑。它会合并同一时间戳到达的竞争者，每个 Path 最多只授予一个 I/O，将未派发的 block 留在客户端，并且仅在处理完 I/O completion 后才释放临时租约。一个 Path 必须同时满足可见 I/O 数量为 0 且不存在有效的客户端租约，才能被再次分配。采用 round-robin 获取方式，避免事件插入顺序让某一个 coflow（协同流）占用所有空闲 Path。

测试验证了：

- 不会出现跨 NPU 的 Path 占用；
- 每个 Path 的 active depth 永远不超过 1；
- 等待中的 block 不会进入 SSU 队列，因此 Path queue wait 为 0；
- 同一时间戳下的 Path 获取不会导致饥饿；
- 单 I/O 和绑定式 coflow 两种租约生命周期；
- 基于 required bandwidth 和 pending byte 的仲裁消融；
- 每个 block 都能准确完成，且非租约模式的行为保持不变。

## 找到的最佳符合约束的静态 CIR 配置点

目前找到的最佳严格租约规则，在每个包含 32 个 Path 的组内采用：

~~~text
Path 数量 [SS, SL, LS, LL] = [12, 4, 12, 4]
每个 SSU 的 CIR 预算 [SS, SL, LS, LL] = [20, 4, 12, 4] GB/s
PIR = 不限速；所有 Path/组级 WRR 权重 = 1
第 0 层路由 = 严格按类别；租约生命周期 = 单个 block I/O
~~~

| SSU 数量 | Baseline 利用率 | QoS 利用率 | 增益 | SS p95 变化 | SL p95 变化 | LS p95 变化 | LL p95 变化 | 硬性门槛 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 40.74% | 45.46% | +4.73pp | -70.73 ms | +6.94 ms | -17.46 ms | +29.87 ms | 未通过 |
| 28 | 51.64% | 58.24% | +6.60pp | -35.48 ms | +4.08 ms | -4.28 ms | +17.77 ms | 未通过 |
| 40 | 58.80% | 66.63% | +7.83pp | -20.64 ms | +3.63 ms | +0.80 ms | +11.22 ms | 未通过 |
| 56 | 65.61% | 74.26% | +8.65pp | -9.37 ms | +1.09 ms | +1.95 ms | +6.90 ms | 未通过 |
| 84 | 73.33% | 82.20% | +8.87pp | -5.42 ms | +0.06 ms | +0.65 ms | +4.75 ms | 未通过 |
| 112 | 78.47% | 85.91% | +7.44pp | -3.02 ms | +0.24 ms | +0.55 ms | +3.53 ms | 未通过 |

六个 SSU 点均没有发生饥饿，SSU Path queue wait 的最大值也严格为 0。利用率增益随 SSU 数量先上升，在 84 SSU 达到峰值 +8.87pp，随后在 112 SSU 回落到 +7.44pp。84 SSU 下，QoS 各类别的 p95 分别为：SS 18.0169 ms、SL 249.0108 ms、LS 42.5575 ms、LL 507.0992 ms。

这个配置点不应进入多 seed 验收：在搜索 seed 上，它既没有达到 utilization 目标，也有三个类别未通过 p95 门槛。多 seed 验证无法修复这些硬门槛失败，只会增加针对随机噪声调参的风险。

### SSU 扫描图

主图沿用原始 I/O scheduling sweep 的单面板样式，只显示 Baseline 与 QoS 两条 utilization 曲线；顶部横轴给出对应的 NPU/SSU 比例。六个点的 QoS utilization 均高于 Baseline，但没有点达到 +10pp。

![静态 CIR 严格租约下的平均 NPU utilization](../results/qos_static_cir_utilization_ls0p5.png)

各类别 p95 TTFT 的正值表示相对 Baseline 恶化。SS 在所有点均明显改善；LL 在所有点均恶化，是当前静态资源划分的主要尾延迟瓶颈。

![各请求类别的 p95 TTFT 变化](../results/qos_static_cir_p95_delta_ls0p5.png)

硬门槛图的目标区域要求横轴不大于 0 且纵轴不小于 10pp；当前六个点均未进入该区域。

![利用率增益与尾延迟硬门槛图](../results/qos_static_cir_gate_pareto_ls0p5.png)

## 未能弥合差距的消融实验

| 84 SSU 下的变体 | 增益 | 关键 p95 结果 |
|---|---:|---|
| 四类请求各使用 64 个 Path，单 I/O RR 租约 | +7.86pp | SL +0.71、LS +2.88、LL +2.17 ms |
| 按 required bandwidth 加权分配租约 | +7.24pp | LS 相比 RR 有改善，但 SL +4.67、LL +5.58 ms |
| 按 pending byte 加权分配租约 | +6.71pp | SL/LS/LL 均恶化超过 2 ms |
| 绑定式 coflow 租约，允许非 SS 请求借用 | +8.21pp | SL/LS/LL 均恶化 1.13--3.03 ms |
| 所有类别共享第 0 层 Path | +3.91pp | 各类别尾延迟更接近，但 SS 加速效果消失 |
| 按 required bandwidth 划分高、低速 Path 层级 | 最高 +3.36pp | LS 可以通过，但 SL/LL 明显恶化 |
| 每个 NPU 静态拥有两个 Path | 约 +2.6pp | 隔离语义清晰，但 Path 并行度不足 |

进一步的 CIR 和 Path 数量扫描呈现出相同的权衡：把资源转移给 LL，只能小幅缓解其排队压力，却会从 SS/LS 移走足够多的服务能力，导致总体 utilization 下降。固定 WRR 权重的消融也没有带来改善。

## 负结果的结论与适用边界

在“仅客户端持有空闲 Path 租约”、WRR 权重固定、只允许配置静态 CIR、且没有 admission control 的约束下，已测试的 16/28/40/56/84/112 SSU 配置点均无法在所有类别 p95 不恶化的同时达到 +10pp。最佳配置点的增益为 +8.87pp，且仍有三个类别未通过 p95 门槛。这一结果来自多种由机制分析驱动的路由方案，而不是针对 seed 42 拟合出的精细网格搜索结果。

参考设计中剩余的机制，是由剩余 deadline slack 驱动的 URGENT 逃生通道，并配合 admission/rate limiting，且通常需要更高的 WRR 权重。但该机制超出了本实验的信息与控制边界：当前客户端策略只能使用类别、layer、`required_bw` 和 I/O 数量，主实验的 WRR 权重固定，而且不允许全局 admission controller。若要严谨测试该机制，需要显式加入 residual-slack 状态，并增加客户端 admission 协调或文档所述的控制面 lease allocator；不能把它包装成仅搜索 CIR 的结果。

## 复现与验证

实验入口为 `qos_static_experiment.py`。可通过 `static_cir_config` 使用以下参数复现最接近的严格租约配置：

~~~python
static_cir_config(
    (20, 4, 12, 4),
    client_path_leasing=True,
    layer0_nonexpress_mode="strict_category",
    category_paths_per_group=(12, 4, 12, 4),
)
~~~

`plot_qos_static_cir.py` 的默认参数就是本节使用的 `SSU_LIST=[16,28,40,56,84,112]` 和 `LS_RATIO_LIST=[0.5]`。运行以下命令会断点续跑缺失点，保存完整 JSON，并重新生成三张图：

~~~bash
python plot_qos_static_cir.py
~~~

数值结果保存在 `results/qos_static_cir_strict_lease_sweep.json`。完整测试套件已通过。曾尝试过一种失败的 free-path cache 优化：虽然最终指标完全相同，但它改变了内部事件数量，因此已经撤回。目前保留的快速路径，只会跳过“缓存的 eligible-path 集合与当前 free-path 集合没有交集”的 coflow。
