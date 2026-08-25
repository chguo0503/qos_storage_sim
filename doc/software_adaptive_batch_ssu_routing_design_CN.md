# 软件侧自适应分批 SSU 选路：实现设计与审计记录

日期：2026-08-25

> **归档说明（2026-08-25）**：用户已决定停止这条自适应 K/q/refill
> 路线并放弃粗搜索。对应实现已从当前 `sim.py`、`experiment.py` 和测试中移除；
> 当前主线只比较统一受 NPU 50 GB/s 聚合限速的 `baseline_bypass` 与
> `qos_static_cir`（0/1、8/1、8/8）。下文保留为历史设计记录，其中“当前代码
> 事实”和“将新增”均不得再当作现版本说明。需求原文仍保存在同目录的
> `software_adaptive_batch_ssu_routing_requirements_CN.md`。

本文记录 `software_adaptive_batch_ssu_routing_requirements_CN.md` 落地前的代码审计、
实现边界和算法定义。需求文档与用户本轮明确给出的约束不一致时，以用户本轮约束为准。

## 1. 已核实的当前代码事实

1. 当前仅有 `baseline_bypass` 和 `qos_static_cir` 两个公开 policy。
2. 两种 policy 都使用“一块 SSU 最多一个不可抢占 active I/O”的后端近似。
3. Baseline 在每个 NPU 内 FCFS、不同 NPU 间逐 I/O RR；QoS 使用静态 CIR、
   Path WRR、Group WRR 和最终 RR 的离散虚拟完成标签仲裁。
4. 当前客户端提交状态是 `(request, layer, SSU)` 级；每个状态独立维护 cursor，
   不存在 `(request, layer)` 全局 UNSENT/IN_FLIGHT/COMPLETED 状态、全局窗口或
   completion 驱动的低水位 refill。
5. QoS 客户端能取得完整 256 项 `active + pending I/O-count`；选择器不要求空 Path，
   不维护 owner、lease 或 reservation，并使用本轮 shadow count。
6. 当前代码把一次选中的后端 I/O 直接设为 40 GB/s。一个 NPU 同时从多块 SSU
   返回时没有共同 50 GB/s 数据面上限，因此历史结果不能作为本任务的合规结果。
7. 输入画像的 `required_bw` 在 `_load_from_key()` 中被丢弃。
8. 请求抽样和 placement 消耗同一个 RNG；submit order 已使用独立 RNG，但 workload、
   placement 尚未分离，也没有显式 placement artifact/hash。
9. 当前 `avg_npu_utilization` 实际是逐请求
   `compute_ms / (compute_ms + io_wait_ms)` 的平均值，不是 fleet NPU 硬件利用率。
10. 当前结果没有完整输出类别 p50、fleet compute utilization、NPU link 积分、
    SSU active/effective 两种利用率、layer-global K/q/refill 和 placement 守恒指标。
11. 当前工作区在本任务修改前已有大量用户修改和删除；这些内容不会被回滚或覆盖。
12. 本任务修改前测试为 30/31 通过。唯一既有失败是历史
    `routing_comparison_spec()` 当前返回七个 SSU 点，而旧测试仍断言 `[40, 56]`。

## 2. 已发现的需求/历史文档冲突

1. 早期设计包含盘框 CPU、Path lease、byte counter、动态 QoS、复制热点 block、
   重发或迁移等机制；本任务明确禁止把这些作为新策略输入或控制手段。
2. 早期文档把具体硬件描述成 token bucket/WRR；当前模拟器实际是 byte-based、
   packetized virtual-finish 近似，不能声称复刻具体硬件实现。
3. 需求文件中的早期草案偏向 `q=1` 和较小搜索；本轮约束明确要求比较
   fixed q `{1,2,4,8}`、adaptive q、三种评分和三种 refill 分配策略。
4. 历史 +8pp 左右结果没有 NPU 50 GB/s 聚合限速，并且使用 per-SSU 固定 batch；
   它们只用于说明潜在机制，不进入本轮验收。
5. 当前 `experiment.py` 的旧扫描列表与注释/旧测试不一致。本任务会新增独立的
   adaptive 实验 spec，主验收固定为 40/56 SSU，不修改旧历史扫描的含义。

这些冲突不妨碍在锁定约束内实现，因此无需改变 placement、遥测权限或硬件模型。

## 3. 新增策略和静态配置

新增独立 policy/schema，不覆盖旧名称或缓存：

- `legacy_baseline_capped`
- `legacy_qos_capped`
- `matched_baseline_layer_global`
- `software_adaptive_batch_qos`

其中 legacy 两组保留旧提交行为但共同应用 50 GB/s NPU cap；matched 和 adaptive
共用 layer-global 状态机。新配置使用 frozen dataclass，至少记录：

- K mode、`K_min`、`K_max`；
- q mode、fixed q、adaptive-q 公式版本；
- headroom、low-water fraction 和 `ceil` 取整；
- SSU score、refill allocation；
- pressure refresh、Path binding；
- NPU cap、同 timestamp 可见性和 tie-break hash 版本。

## 4. 配对输入与数据守恒

工作负载、placement 和提交顺序分别使用独立 RNG。每个配对 case 先生成只读输入：

```text
workload_seed  -> request loads
placement_seed -> (request_id, layer, block_idx) -> (ssu_id, block_gb)
submit_seed    -> 同 timestamp 的 NPU 顺序
```

placement 以 `request_id` 为键，不依赖请求后来被哪个空闲 NPU 执行。规范化后计算
workload fingerprint 和 placement SHA-256；配对运行开始及结束都验证二者一致。

每个 adaptive/matched layer 为每个 block 保存且只保存一个状态：

```text
UNSENT -> IN_FLIGHT -> COMPLETED
```

任何时刻断言三类数量之和等于总 block 数；只允许一次 submit 和一次 complete；
flow 中携带 request ID、原 placement SSU 和 block ID，完成时再次校验。

## 5. Layer-global 状态机

一个 `(request, layer)` 状态覆盖该 layer 的全部固定目标 SSU：

- `unsent_by_ssu`：每块目标 SSU 的未提交 block deque；
- `inflight_blocks` 和 `per_ssu_inflight`；
- `completed_blocks`；
- 当前 K、q、`B_target`、`W_low`、planning round；
- refill pending/generation；
- deadline、动态需求和诊断计数。

每轮定义保持不变：

```text
K_effective = min(K, 拥有 IN_FLIGHT 或 UNSENT block 的不同 SSU 数)
B_target    = K_effective * q
W_low       = ceil(B_target * low_watermark_fraction)
top_up      = max(0, B_target - current_global_inflight)
```

已有 in-flight SSU 必须保留。K/q 缩小时不取消或迁移；只停止向超额 SSU/深度继续
提交。q=1 时提交后逐 SSU in-flight 必须不超过 1。

初始 plan 和 refill 都是独立客户端事件；一次 plan/enqueue 后立即返回事件循环。
只有真实 completion 能预约 refill。由于 completion 事件优先级高于客户端 refill，
同 timestamp 的多个 completion 先聚合；pending flag/generation 保证只执行一次。
如果 `top_up=0`，本轮结束并等待下一次 completion，不创建同时间自旋。

## 6. 动态需求带宽

L0 使用 50 GB/s。L1 及以后使用：

```text
D_rem = 所有 UNSENT 和 IN_FLIGHT block 的完整 block_gb 之和
T_rem_ms = previous_layer_compute_expected_end_ms - now_ms
R_req = min(50, 1000 * D_rem / max(T_rem_ms, epsilon))
```

客户端不读取 flow 的 `remaining_gb`。`T_rem<=0` 且数据未完成时使用 50 GB/s，并
记录 late-prefetch/deadline miss。画像同时保留 `required_bw_input_gbps`、动态重算值、
source 和差异；动态值是运行时 source of truth。

## 7. 候选 SSU/Path 估计

每个 fresh planning round 对所有仍有 UNSENT block 的目标 SSU 读取一份长度严格为
256 的快照。选择器只使用：请求/层/剩余数据、block placement、静态 QoS 镜像和
256-count 快照。

对每个候选 SSU，以当前类别合法 Path 池和现有 Group-aware SED 估计加入下一条
block 后的最佳 Path。旧积压字节只能用代表性 block 大小乘 count 近似：

```text
F_s = (estimated_old_backlog_gb + next_block_gb) / long_term_path_rate_gbps
A_s = min(SSU_BW, next_block_gb / max(F_s, epsilon))
```

`F_s` 包含静态 CIR、全局剩余带宽回收、Group WRR、Path WRR 和新 I/O 自身。
同一 round 每规划一条就更新本地 shadow count；原始 snapshot 保持不可变。

相同分数使用 BLAKE2 稳定哈希打散，输入至少包含
`(request_id, layer, planning_round, ssu_id)`，不依赖 dict 顺序或最低 SSU ID。

三种评分定义：

- `predicted_finish_time`：最小 `F_s`；
- `estimated_effective_rate`：最大 `A_s`；
- `hybrid`：最小
  `0.5 * F_s + 0.5 * (next_block_gb / max(A_s, epsilon))`。

hybrid 两项单位都为秒，系数和公式版本写入 spec。

matched baseline 不读取 Path pressure。它使用相同 layer-global 状态机和 K/q
配置函数，但 SSU 排序只依赖 placement 与稳定哈希；中性容量估计使用固定 SSU
物理速率。这样不会把 adaptive 的硬件遥测偷偷泄漏给 baseline。

## 8. 动态 K 和 q

动态 K：候选不足 8 时使用全部；否则从 `K_min` 开始，按配置 score 依次加入，直到：

```text
min(50, sum(A_s)) >= headroom * R_req
```

K 不超过 K_max、候选数和剩余 block 数。全部候选仍不足时记录 capacity shortfall。

fixed q 直接取 `{1,2,4,8}`。adaptive q 使用可复现的带宽时延积公式 v1：

```text
F_ref_s = selected SSU 的 F_s 的 p75
per_ssu_demand_gbps = R_req / max(selected_K, 1)
q_raw = ceil(per_ssu_demand_gbps * F_ref_s / representative_block_gb)
q = {1,2,4,8} 中不小于 q_raw 的最小值，超出时取 8
```

该公式只使用允许输入；q 增大可 top-up，缩小时不取消已有 I/O。

三种 refill 分配：

- `distinct_first`：先给不同 selected SSU 各一条，再按评分增加深度；
- `weighted_by_A_s`：用累计 `allocated/A_s` 最小者近似加权轮转；
- `iterative_min_F_s`：每加入一条后更新 shadow，再重算下一条的 `F_s`。

所有方式都只从对应 SSU 的 UNSENT deque 取 block，绝不改变 placement。

## 9. NPU 50 GB/s 数据面限速

每个 active flow 先取得 SSU 侧 raw rate。任一 NPU 的 active 集合或 raw rate 改变时：

1. 把该 NPU 当前跨所有 SSU 的 active flow 按旧有效速率 settle 到当前时刻；
2. 若 `sum(raw)>50`，统一乘 `50/sum(raw)`，否则不缩放；
3. 更新每条 flow 的有效速率、预计完成时刻和所在 SSU generation；
4. 旧 completion event 失效，重新安排新 event；
5. flow 完成释放 credit 后立即重算其余 active flow。

`40+40 -> 25+25`、`40+5 -> 40+5`、`8*40 -> 8*6.25` 都由同一函数实现。
每个 NPU 独立拥有 50 GB/s cap。Baseline、旧 QoS、matched 和 adaptive 共用该逻辑。

## 10. 指标与正确性门禁

结果中明确区分：

- `avg_request_compute_fraction`；
- `fleet_npu_compute_utilization`；
- `npu_link_utilization`；
- SSU active-time utilization；
- SSU effective-bandwidth utilization。

此外记录类别 p50/p95/p99/max、L0/L1/L2+ stall、deadline miss、Jain、makespan、
吞吐、NPU cap peak/integral/hit time、queue wait、Path 最大 outstanding、每轮 K/q/window、
refill/pressure/shortfall/候选不足/去重、SSU/Path 分布和全部 fingerprints。

仿真结束前检查：所有 block 完成；UNSENT/IN_FLIGHT/outstanding 为 0；每盘最多一个
active；Path FCFS；目标 SSU 等于 placement；NPU 峰值不超过 `50+epsilon`；无 NaN/Inf；
事件堆未提前耗尽；忙闲和字节积分守恒。

## 11. 待实验验证的假设

以下不是结论：

1. layer-global window 可能减少一次性深队列带来的 HoL，但 matched baseline 也会获得
   这部分收益。
2. fresh 256-count + Group-aware SED 可能改善 adaptive 相对 matched 的关键层完成顺序；
   单 active SSU 和 count-only 估计也可能让收益很小。
3. q>1 可能提高供给连续性，也可能增加 Path/SSU 队列深度和尾延迟。
4. 50 GB/s NPU cap 会削弱“同时选更多 SSU”的吞吐收益；K 的主要作用可能转为绕开
   队列和降低慢盘概率，而非叠加物理带宽。

最终只依据多 seed、held-out、严格配对实验报告是否达到 +10pp 与所有硬约束。

## 12. 必须披露的模型边界

256-count 是零延迟无丢失遥测；count 没有 bytes/age/剩余服务时间；旧积压字节由代表
block 估算；每 SSU 单 active 会放大 HoL；NPU 比例限速会让 active I/O 长时间占盘；
QoS 是 byte-based packetized WFQ 近似；同 timestamp 顺序可见会产生顺序敏感性；
免费 fresh telemetry 会高估真实收益。仿真搜索结果不能推广成真实硬件全局最优。
