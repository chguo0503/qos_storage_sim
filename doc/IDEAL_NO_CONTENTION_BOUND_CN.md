# “理想策略”与无竞争理论上界说明

## 结论先行

仓库里容易被统称为“理想策略”的对象其实有三类，含义完全不同：

| 对象 | 是否运行真实事件仿真 | NPU 间是否仍竞争 SSD | 是否保留 SSD40→NPU50 单命令数据面 | 能否报告 makespan / fleet | 用途 |
|---|---:|---:|---:|---:|---|
| 实际 QoS / baseline 策略 | 是 | 是 | 是 | 是 | 可实现策略对比 |
| `per_ssd_full_visible_edf`、`global_link_aware_online` | 是 | 是 | 是 | 是 | 信息更强的可运行 heuristic |
| `fluid_no_inter_npu_contention_upper_bound` | 否 | 否 | 否；改成 fluid 下界 | 否 | request 指标的乐观理论天花板 |

所以，“理想策略是否实现了所有 NPU 都无竞争”需要分两种回答：

- 两个可运行的理想化 heuristic **没有消除竞争**。它们仍让 128 个 NPU 共享真实 SSD，每块 SSD 仍只有一个 40 GB/s 非抢占命令槽，每个 NPU 仍只有一个 50 GB/s 接收槽。
- `fluid_no_inter_npu_contention_upper_bound` **在数学上消除了所有跨 NPU 竞争**。但它不是一个调度策略，也没有生成联合命令序列；它是把每个 NPU 分别放进一个“独占全部 SSD 的平行世界”后算出的不可运行上界。

正式分析中的关系是：

```text
最佳可运行策略 <= 未知的真实最优策略 <= fluid 无竞争上界
```

## 1. 正式上界使用什么输入

上界与可运行策略复用完全相同的：

- 128 个 request/NPU；
- 16 层；
- 每个 request 的 `per_layer_us`、KV 字节数和类别；
- 每个 block 固定落在哪块 SSD；
- 每个 NPU 独立的 0–5 ms 到达时刻；
- 一层前预取规则：第 0 层到达时开始读，第 `l+1` 层只能在第 `l` 层开始计算时开始读。

入口是 [`isolated_no_contention_bound`](../upper_bounds.py)，它直接读取 `PreparedSimulationInputs`，不调用 `simulate_continuous`。

## 2. “无竞争”是如何实现的

### 2.1 不是通过一个更聪明的全局仲裁器

上界没有模拟：

- baseline 的 NPU-RR；
- QoS Path / Group / CIR / PIR / WRR；
- SSD 命令队列；
- NPU 接收 FCFS 队列；
- 动态选路或动态 CIR；
- NPU 间通信、锁、时间片或发送时机编排。

它没有尝试在同一个物理系统中避免冲突，而是直接从问题定义中删除跨 NPU 资源共享。

### 2.2 每个 request 都被单独计算

对 request `i`、layer `l`，定义：

- `B(i,l,d)`：该 request 该层固定落在 SSD `d` 上的总 GB；
- `B(i,l) = Σd B(i,l,d)`：该层全部 SSD 的总 GB；
- 每盘名义带宽 `D = 40 GB/s`；
- 该 NPU 的接收带宽 `N = 50 GB/s`。

上界独立计算每个 request，没有任何其他 request 的队列或负载进入公式。等价地说：

```text
request 0 独占一套“所有 SSD 都可给它 40 GB/s”的世界
request 1 也独占另一套相同世界
...
request 127 同样独占自己的世界
```

这些世界可以在时间上重叠。它不是把 128 个 request 依次串行运行，而是把同一块物理 SSD 的 40 GB/s 容量乐观地复制了 128 份。因此这里保留的是“每个 request 独立世界内的名义 40 GB/s”，不是物理 SSD 的全局共享 40 GB/s。

## 3. 每层 I/O 下界公式

对一个 request 的一层，代码计算：

```text
SSD 工作量下界 = max_d B(i,l,d) / 40
NPU link 工作量下界 = B(i,l) / 50

L(i,l) = max(
    max_d B(i,l,d) / 40,
    B(i,l) / 50
)
```

时间单位在代码中换算为毫秒。

含义是：

- 即使完全没有其他 NPU，最忙的那块 SSD 也至少要花 `bytes/40`；
- 该 NPU 的单接收链路至少要花 `total_bytes/50`；
- 两项可以被乐观地完美重叠，所以只取较大者。

这比当前真实后端更乐观。真实后端是：

```text
SSD 整条命令完成
        ↓
block 才进入 NPU FCFS 接收队列
        ↓
NPU 以 50 GB/s 完成整条命令
```

而 fluid 上界允许字节从层 I/O release 时刻就连续流动，相当于理想 cut-through；它不等待一个 block 在 SSD 端完整完成，也不保留非抢占命令边界。

## 4. 16 层 pipeline 如何计算

令：

- `C(i)` 为 request `i` 每层固定计算时间；
- `A(i)` 为其 0–5 ms 到达时刻；
- `E(i,l)` 为第 `l` 层计算结束时刻；
- `S(i,l)` 为第 `l` 层计算开始时刻；
- `R(i,l)` 为该层 I/O release 时刻。

初始化：

```text
E(i,-1) = A(i)
```

逐层递推：

```text
R(i,0) = A(i)
R(i,l) = S(i,l-1), l > 0

IOReady(i,l) = R(i,l) + L(i,l)
S(i,l) = max(E(i,l-1), IOReady(i,l))
Wait(i,l) = S(i,l) - E(i,l-1)
E(i,l) = S(i,l) + C(i)
```

`R(i,l)=S(i,l-1)` 正是“一层前预取”：上一层开始计算时，下一层才能开始读。

最后对每个 request 计算：

```text
U_bound(i) = 16*C(i) / (16*C(i) + Σ_l Wait(i,l))
```

正式上界指标是 128 个 `U_bound(i)` 的算术平均。这与实际仿真中的 `avg_request_compute_fraction` 定义对齐。

## 5. 被删除或放宽的约束

### 5.1 删除所有跨 NPU SSD 竞争

不同 NPU 即使读取同一块 SSD，也互不排队。每个 request 都可把该 SSD 当作自己独占的 40 GB/s。

### 5.2 删除 Path/CIR/仲裁约束

上界没有 Path 数、类别隔离、CIR 预算、PIR、Group WRR、Path WRR，也没有 baseline RR。固定 placement 仍保留，但 placement 只决定单个 request 自身哪块盘最忙。

### 5.3 删除命令边界与 store-and-forward barrier

真实的单命令、非抢占 SSD 服务被放宽为 fluid 字节流；SSD 和 NPU link 的工作量可从 release 时刻完美重叠。

### 5.4 删除离散 incast/FCFS 排队

同一 NPU 的多个 SSD 完成可能同时汇入一个 50 GB/s FCFS 接收队列。上界不模拟这些离散到达，只保留该层总字节除以 50 GB/s 的必要工作量。

### 5.5 删除所有跨 request 尾部耦合

一个 request 的完成时间不会被另一个 request 延长。正因为没有联合时间线，上界没有一个有意义的全局最后完成时刻。

## 6. 仍然保留的约束

上界并不是把 I/O 时间设为零。它仍保留：

- 每个 block 的原始 SSD placement 和每层总字节数；
- 每个 request 独立世界内，每盘最多 40 GB/s 的必要工作量；
- 每个 NPU 的总接收工作量最多 50 GB/s；
- 每层固定计算时间和 16 层严格计算顺序；
- 一层前预取的 release 时机；
- 一层所有数据 ready 后才能开始该层计算；
- 每个 request 的 0–5 ms 到达绝对时刻。

它不允许把 block 迁移到更空闲的 SSD，也不允许在上一层开始计算之前读取未来层。

## 7. 为什么它一定是上界

任何符合当前真实数据面的 schedule，对 request `i` 的 layer `l` 都至少需要：

```text
max(
    该层最忙 SSD 的字节 / 40,
    该层总字节 / 50
)
```

真实 schedule 还可能增加：

- 其他 NPU 的 SSD 排队；
- 非抢占命令造成的次序等待；
- SSD 整块完成后才可进入 NPU 的 barrier；
- 同一 NPU 接收队列中的 incast 等待；
- 调度器并不知道未来而造成的选择损失。

因此，对每个 request 都有：

```text
Wait_actual(i) >= Wait_fluid(i)
U_actual(i) <= U_bound(i)
```

逐 request 成立后，128 个 request 的算术平均也成立。

注意：这是 relaxed feasible set 的上界证明，不是“已经找到真实硬件上的最优命令顺序”。

## 8. 为什么不报告 makespan 和 fleet utilization

### request compute fraction 可以报告

每个 request 都有自身的计算时间和暴露 I/O wait，所以可以计算 `U_bound(i)`，再求 request 平均。

### makespan 不能报告

`makespan` 是同一物理系统中 128 个 request 的最后完成时刻。上界把每个 request 放在独立平行世界，并复制共享 SSD 容量，没有生成一个联合时间线，因此没有与真实系统可比的 makespan。

### fleet utilization 也不能报告

实际 fleet 指标是：

```text
Σ_i compute_ms(i) / (128 * global_makespan_ms)
```

它依赖 global makespan。既然 fluid bound 没有联合 makespan，就不能给出合法的 fleet 上界。把各 request 的独立完成时刻强行取最大值会把复制出来的 SSD 容量当成真实容量，得到的不是原系统的 fleet 指标。

## 9. 正式 128 NPU / 16 层结果诊断

seed 42 的正式输入中，每个 SSU 数量都有 `128 × 16 = 2048` 个 request-layer。逐层检查发现：

- 40、56、80 SSU 的所有 2048 层，`B(i,l)/50` 都严格大于 `max_d B(i,l,d)/40`；
- 因而每一层的 fluid 下界都由单 NPU 50 GB/s link 工作量决定；
- 三个 SSU 数量的正式 bound 都是 `91.231809%`。

有限发行四策略复验又加入 seed 43。该 seed 的三个 SSU bound 都是
`92.451995%`；两个 seed 等权均值为 `91.841902%`。上界不模拟客户端命令
提交事件，所以 batch1/0.1 us 不会改变同一 seed 的 bound；seed 间差异来自
workload 本身。

这只说明当前这个很宽松的上界已经进入 per-NPU-link-bound，不说明真实系统增加 SSU 没有价值。真实策略仍会受到 SSD 队列、block 命令顺序和 transient hotspot 影响。

## 10. `isolated_layer_io_time_ms` 与正式 fluid bound 的区别

[`isolated_layer_io_time_ms`](../upper_bounds.py) 还提供了另一种“单 request 独占资源”计算：

- 每盘按输入 block 顺序串行执行 `size/40`；
- 不同盘并行；
- block 在 SSD 完成后才进入单 NPU link；
- link 按 SSD 完成顺序串行执行 `size/50`。

它保留离散命令和 store-and-forward，是一条 isolated 可行 schedule，但不保证是 isolated exact optimum。当前正式报告没有用它；正式上界用的是更乐观的 `fluid_layer_io_lower_bound_ms`。

## 11. 正确解读方式

可以说：

> 91.23% 是在保持 workload、placement、每层 release 规则以及每请求名义 SSD40/NPU50 工作量的前提下，删除跨 NPU 竞争并允许 fluid cut-through 后得到的 request compute fraction 乐观上界。

不应说：

- “已经实现一个 91.23% 的可部署策略”；
- “全局最优调度器已被求解”；
- “所有 NPU 在真实 SSD 上可以同时独占 40 GB/s”；
- “这个上界证明 50 GB/s 是真实系统唯一瓶颈”；
- “它的 fleet utilization 是 91.23%”。

91.23% 只属于 request 平均指标；实际可运行策略是否改善 fleet，仍必须回到共享 SSD40→NPU50 事件仿真，用真实 makespan 单独验证。
