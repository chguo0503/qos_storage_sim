# 32 NPU / SSU=4 / 32 秒收敛与机制审计

## 结论

这组数据证明 3 秒的领先没有持续到 8/16 秒；其主要来源是该短窗口准入了 compute 更重、IO 更轻的请求画像。3 秒 matched cohort 也观察到较低 stall，但长窗口的方向和幅度并不稳定。

短窗口和长窗口并不是两种利用率定义：它们都是 compute interval 在窗口内的积分，只是积分边界不同。3 秒时 Adaptive 相对 Baseline 为 **+4.555 pp**，32 秒时为 **+0.423 pp**，最后一个 trailing-8s 窗口为 **+4.812 pp**。Adaptive 的累计 compute-busy 优势在 3s 达到峰值 **+4.373 NPU-s**；峰值随后回落，是与“前移工作而非创造长期算力”一致的积分证据。由于边界 Q 未独立记录，它不是单独充分的因果证明。

3 秒 actual busy 差为 **+4.373 NPU-s**，admitted compute 差为 **+4.090 NPU-s**，比例为 **93.5%**。这表示短期差的主体来自 measurement window 选中了不同请求画像；它不等价于 Adaptive 增加了硬件能力。

## 不同前缀的闭合量

| 前缀 | Baseline NPU | Adaptive NPU | A-B | A-B actual busy | A-B admitted compute | A-B inferred inventory drop | matched stall A-B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2s | 81.841% | 83.425% | +1.584 pp | +1.014 NPU-s | +5.139 NPU-s | -4.126 NPU-s | -23.516 ms (n=64) |
| 3s | 77.994% | 82.550% | +4.555 pp | +4.373 NPU-s | +4.090 NPU-s | +0.283 NPU-s | -27.669 ms (n=121) |
| 8s | 78.639% | 78.186% | -0.452 pp | -1.158 NPU-s | -1.280 NPU-s | +0.122 NPU-s | -0.723 ms (n=410) |
| 16s | 78.564% | 77.739% | -0.825 pp | -4.224 NPU-s | -3.741 NPU-s | -0.483 NPU-s | +3.916 ms (n=880) |
| 32s | 76.739% | 77.162% | +0.423 pp | +4.332 NPU-s | +6.736 NPU-s | -2.403 NPU-s | -1.303 ms (n=1839) |

`actual busy` 是仿真实际 compute interval 的严格积分。`admitted compute` 是该前缀新准入请求的 `sum(16 × per-layer compute)`；`admitted IO` 则由 `raw_demand_gbps × ideal_ttft_s` 重建。对每个策略和每个窗口都有守恒式 `Q(start)-Q(end) = actual busy - admitted compute`，其中 Q 是 active request 的剩余 compute 库存。

当前 schema v2 **没有独立记录**每个边界的 Q，因此只能由上式报告 `inferred inventory drop`，不能拿它反过来宣称库存被独立验证。表中的 A-B inventory drop 是两策略该推断量之差；SSD queue 或 request count 没有被冒充为 compute 库存。

## 四个互不重叠的 8 秒窗口

| 独立窗口 | Baseline NPU | Adaptive NPU | A-B | A-B admitted compute | A-B admitted IO |
|---:|---:|---:|---:|---:|---:|
| 0–8s | 78.639% | 78.186% | -0.452 pp | -1.280 NPU-s | -3.385 GB |
| 8–16s | 78.489% | 77.292% | -1.198 pp | -2.461 NPU-s | -15.953 GB |
| 16–24s | 71.778% | 70.308% | -1.470 pp | -0.259 NPU-s | -0.330 GB |
| 24–32s | 78.048% | 82.861% | +4.812 pp | +10.735 NPU-s | -20.924 GB |

四段 A-B 分别为 **-0.452 pp, -1.198 pp, -1.470 pp, +4.812 pp**。最后一段突然变为 +4.812 pp，而不是围绕一个小值稳定波动，因此 **32 秒累计均值虽只差 +0.423 pp，仍不足以证明已经收敛**。它说明更长窗口平均掉了大部分画像/相位差；单 seed 的末段波动本身不能证明底层非稳态，只能阻止我们宣称已经达到稳态。

## 为什么短期可能高、长期却不高

1. Adaptive 改的是 SSU 命令的 CIR 排队优先级。它可以更早把 IO 送到原本等待的 NPU，减少某一段时间内的 compute starvation；3 秒内有 **19/32** 个 NPU 的累计利用率上升，Baseline 初始利用率最低四分之一 NPU 的平均变化为 **+20.729 pp**。
2. 控制器没有改变每层固定 compute 时间，也没有增加 SSD 的 40 GB/s 或 NPU-link 的 50 GB/s 容量。被提前完成的工作会改变后续队列和可运行层的数量；如果没有新增长期吞吐，累计优势会持平或回落。
3. 两策略使用相同 workload、placement 和 trace 指纹，但 warmup 完成时刻和 measurement 起始动态状态并不相同。相对时间 0 对齐的是“各自 warmup+settle 后”，不是完全相同的 SSD/link/active-compute 库存快照。
4. 这是 closed-loop 连续输入。不同策略在同一 3 秒内准入的 request 数和 profile mix 可以不同。因此必须同时看 admitted compute/IO，而不能只看请求个数。详细结果在 `profile_diagnostics.csv` 和图 05。
5. `matched_request_stall.csv` 只比较两个策略都出现的同一 request ID，并使用 `stall = TTFT - ideal_ttft`，从而排除 request identity 差异；但两边的绝对时刻和全局队列状态仍不同，因此它不是纯调度效率指标。3 秒 matched cohort 的平均 stall 变化为 **-27.669 ms**（负数表示 Adaptive 更好），32 秒为 **-1.303 ms**，且中间窗口会换符号；见图 06。
6. NPU utilization 在这里是二值 compute busy-duty，不是 tensor-core、FLOPS 或 HBM 硬件计数器。

## IO 与队列检查

每个 500ms block 已独立验证：逐 NPU compute busy、逐 SSU SSD busy/served、逐 NPU link busy/served均与 32 秒汇总闭合；SSD 以 40 GB/s、link 以 50 GB/s 的固定活动速率换算也逐块闭合。SSD/link outstanding 的块数和 GB 在相邻边界连续。资源曲线和队列见图 04；它们能解释 IO 何时可用，但不能替代缺失的 active compute 库存。

## 有效性与限制

- 输入严格为 32 NPU、4 SSU、16 层、warmup=8、settle=500ms、32 秒、64×500ms。
- baseline 与 Adaptive 100ms 的 source/config、十个输入指纹、prefix-32 materialization 和 simulator input fingerprint 完全一致。
- 每个 case 的 29 个 simulator invariants 全部为真，并由本分析再次重建关键资源与请求统计。
- 前 8 秒聚合利用率桥接旧正式实验：Baseline 78.638773%，Adaptive 78.186478%；measurement start 也精确一致。
- 这仍是单 seed。32 秒能检验窗口收敛，但不能替代多 seed，也不能仅凭累计均值宣称已经达到稳态；应结合 rolling-8s/16s 和队列漂移判断。

## 输出

- `summary.csv`：2/3/8/16/32 秒前缀的主指标与 work-mix 残差
- `blocks_500ms.csv`、`cumulative_prefix.csv`、`rolling_8s.csv`、`rolling_16s.csv`、`disjoint_windows.csv`
- `per_npu_prefix_utilization.csv`
- `resource_diagnostics.csv`、`profile_diagnostics.csv`、`matched_request_stall.csv`
- `validation.json`：输入哈希、严格校验和机制判据
- 图 01–06：利用率收敛、差值积分、逐 NPU、IO/queue、admitted work mix、matched-request stall
