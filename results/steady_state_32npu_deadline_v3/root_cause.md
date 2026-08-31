# Deadline/barrier Scheme B V3：32-NPU 结论

## 结论

当前 EDF/barrier-first V3 **不值得进入 128-NPU 长跑**。在完全配对的真实
`data`、32 NPU、SSU10、batch=1、16 层 warm/full-load 正式窗口中：

| 策略 | Mean NPU util | Equal-NPU TTFT SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 57.24% | 75.00% | 75.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| admission25 | 57.00% | 84.57% | 84.59% | 43.75% | 100.00% | 95.06% | 100.00% |
| deadline/barrier V3 | 56.68% | 74.92% | 74.92% | 24.39% | 98.85% | 77.78% | 100.00% |

V3 与 baseline 的 assignment/workload/placement/trace/simulator-input 五个
指纹完全相同。V3 相对 admission25 为 `-0.32 pp` mean NPU util、
`-9.65 pp` equal-NPU SLO；相对 baseline 也没有 SLO 改善。

## V3 原型做了什么

控制器只使用客户端可见信息：全剩余 manifest、下一层 demand、
`waiting_for_io`、`compute_done_up_to`、每层 compute 和 snapshot 时钟。

1. 每 25 ms 估计请求距 `2 × ideal TTFT` 截止线的剩余时间；
2. 正在 barrier 等待者优先，其次按剩余 laxity/deadline；
3. 为入选请求放置 deadline floor；未入选请求保留 5% background pool；
4. 最后用 absolute max-min spill 消耗可用 SSD/NPU tail；
5. 每 SSD 不超过 40 GB/s，每 NPU 不超过 50 GB/s，grant 有限、非负且
   demand-capped；设置周期和 controller 内部最小决策间隔都为 25 ms。

静态/runner 回归共 17 个测试通过，覆盖阶段恒等式、SSD40/NPU50、非负
有限值、demand cap、priority packing、25-ms rate limit、真实 steady
summary schema 和 source/config fingerprint。

## 根因一：EDF 优先级不等于 binary-SLO cardinality

真实 sequence-0 的同一 32-NPU × SSU10 快照给出决定性反例：

该快照可由 `analyze_deadline_v3_static.py` 和
`results/steady_state_32npu_deadline_v3/static_snapshot.json` 重现。

| Allocator | 入选数 | 入选类别 | 未入选类别 | SSD 列利用 |
|---|---:|---|---|---|
| EDF/barrier-first V3 | 8 | 6 SS + 2 LS | 2 SS + 6 LS + 8 SL + 8 LL | 10/10 列均 40 GB/s |
| admission25 | 26 | 8 SL + 8 LS + 8 LL + 2 SS | 6 SS | 平均 37.76 GB/s CIR；tail 由 WRR 使用 |
| cardinality-first + explicit spill | 26 | 与 admission25 相同 | 与 admission25 相同 | 10/10 列均 40 GB/s |

SS 的截止线最早，但其 raw row demand 约 83.35 GB/s，deadline floor 会被
NPU50 顶住；它是昂贵的 SLO admission item。EDF 先装 SS 后只能保护 8 个
请求。admission25 的目标是通过数量，先装单位 SSD footprint 小的长请求，
因此同一容量可保护 26 个。对 binary pass/fail 指标，后者才是正确一级
目标。

V3 并没有浪费 SSU 容量：正式窗口 SSD mean utilization 为 95.32%，与
baseline 95.38% 和 admission25 95.15% 接近；NPU link mean utilization
也约 23.83%。失败来自“服务给了哪些 coflow、能否跨过完整请求阈值”，
不是总带宽没有用满。

## 根因二：25 ms 对 SS 是请求级时间尺度，reactive barrier boost 太晚

SS 每层 compute 约 1.288 ms，16 层 ideal TTFT 约 20.606 ms，2×SLO
deadline 约 41.212 ms。因此 25 ms：

- 跨过约 19.4 个 SS layer-compute，超过整条 16-layer ideal TTFT；
- 占 SS SLO deadline 的约 60.7%；
- 新请求平均等待半个周期就损失约 12.5 ms slack。

更关键的是当前 simulator/QoS 数据面在命令入队时按当时 CIR 写 virtual
finish tag；CIR 更新不会重写已排队命令的 tag。一个 layer 的小命令约在
0.1 ms 量级内全部提交。控制器 25 ms 后观察到 `waiting_for_io` 时，当前
barrier 的命令早已带旧 tag 入队；boost 主要只能影响未来 layer。周期性
重选又会撤销上一请求的 whole-request floor，形成混合 tag generation。

V3 的 149 次 evaluation/149 次 commit/46,945 次 Path write 与 admission25
的 154/151/48,010 同量级，所以问题不是“更新次数太少”或软件开销，而是
更新相位和目标函数不匹配。

## 可部署的下一版应如何定义

在设置间隔不能低于 25 ms 时，控制必须是 **predictive request-lifetime
admission**，不能依赖 SS 请求内的 reactive barrier correction：

1. 一级目标固定为 maximum/greedy SLO cardinality：按 0.50–0.52 floor
   的 normalized SSD footprint 装包；
2. 入选 request ID pin 到完成，避免每个控制周期重新把 floor 分散给不同
   请求；
3. deadline、`waiting_for_io` 和 age 只在量化后的同成本候选之间 tie-break，
   或只分配 selected tail，不能越过 cardinality objective；
4. 使用 V2 的 `selected floor → rejected background → selected-first tail →
   absolute spill`，显式填满 placement 允许使用的列；
5. 以 batch-membership/request-admission event 触发，25 ms 为全局最小写间隔；
   如果硬件确实能重排已排队命令，才重新评估更细粒度 barrier feedback。

静态修正（0.52 cost-first + explicit spill）已在同一真实快照上恢复 26 个
入选请求，10 个 SSD 列均为 40 GB/s，最大 NPU grant 为 45.10 GB/s。500-ms
短 smoke 的 request-weighted SLO 从 EDF V3 的 70.67% 提升到 76.92%，并
恢复 `SL/LS/LL=100%`；但短窗不足以证明能超过 admission25 的正式
84.57%。该修正实质上收敛到已隔离测试的 V2，而不是需要再发明一套并行
V3。

## 128-NPU 决策

- 不运行当前 EDF/barrier-first V3 的 128-NPU 实验；32-NPU 已有配对负结果。
- 先用正式 128-NPU V2 结果验证 explicit tail 是否解决尺度下的 CIR 列欠填。
- 只有 cost-first/pinned 版本在正式 32-NPU 窗口同时超过 admission25 的
  SLO 且 mean util 不下降，才值得增加 128-NPU 验证。
