# S1/S2/S3 启动成本与等价优化条件

2026-09-07；只读源码、结果与独立内存对象微基准，未修改任何根目录 Python 文件，未中断已有进程。这里分析的是执行仿真的 Python 成本，不是向模拟时间注入的新控制开销。

## 直接原因

`coflow_sim_adapter.py:113–139` 的 arrival 回调在判断 assignment 前，无条件执行 `shared.backlogs(now_ms)`。随后才区分 pipeline、fixed、compute。**因此现有 `--assignment fixed` 仍然计算全部 NPU、全部已到达排队请求、全部层、全部 SSU 的 backlog。** S1/S2/S3 共用此 arrival 回调；更换后端策略不会绕过它。

`SharedPathAdapter.backlogs()`（`shared_path_sim_adapter.py:73–105`）每次做：

1. 将每卡 `admission_queue` 复制成列表，重新 `sum(8*C)`。
2. 加上 active batch 的剩余计算；加入其请求 ID。
3. 对上述每个请求的每层、每个 SSU 重新计算未到 HBM 的块数，包括尚未发出的未来层；即使某个 SSU 的块数为零，也遍历该维度。
4. 返回 `KnownNPU`，包含 remaining compute、unfinished I/O、unfinished request count。

`layer_counts_cache` 只避免反复遍历物理 placement 来求每层块数，**不缓存所有已知请求的聚合 backlog**。每次命中 cache 后，三重请求/层/SSU 循环照常执行。这里的 backlog 是客户端已知工作，不是 SSD 队列深度。

本轮大输入全请求 t=0。原事件类型 `REQUEST_ARRIVAL=-1`、`BATCH_DISPATCH=-0.5`，因此整个同刻 arrival 波次先执行完，才开始第一次接纳、发 I/O。第 k 次 arrival 会重新扫描此前 k−1 个请求，而期间一个请求都不能完成。模拟时钟可一直保持 0，同时解释器持续消耗 CPU；这不等于没有运行，也不构成死锁证据。

相关位置：`continuous_batch_sim.py:33–34,3141–3153,3218–3224,6672–6681,6691–6713`。同刻顺序仍是 `(arrival, 原NPU, request_id)`；不能为了提速跳过 arrival、交错 dispatch 或改变 ties，这会改变算法结果。

## 量级与一次微基准

设请求数 M、层数 L=8、SSU 数 S。在全 t=0、无 active batch 的初始波次，累计扫描此前请求恰好为 `M(M−1)/2`；I/O 内层为 `L*S*M(M−1)/2`。这还不含 compute 求和、列表复制、冷 cache、选卡与日志开销。

| 本轮正式输入 | 请求数 M | 累计此前请求访问 | 层访问 | layer×SSU 内层访问 |
|---|---:|---:|---:|---:|
| strong32_stripe6_feasible，6 SSU | 19,392 | 188,015,136 | 1,504,121,088 | **9,024,726,528** |
| strong32_local/hash，8 SSU | 24,024 | 288,564,276 | 2,308,514,208 | **18,468,113,664** |

只读微基准直接调用原 `SharedPathAdapter.backlogs()`：由正式 manifest 建立 32 张卡的内存队列，使用实际 C、实际每层 SSU 块数；所有请求 `io_ready=False`、未 materialize、无 active batch，placement 计数 cache 已预热。每档三次，取 `process_time()` 中位数；对象用 `SimpleNamespace` 表达，只用于操作量估计，不是一次完整 simulator profile。

| 输入 | 排队请求数 | 原方法单次 CPU 时间 | 单次墙钟 |
|---|---:|---:|---:|
| 6 SSU | 1,024 | 3.360 ms | 3.470 ms |
| 6 SSU | 4,096 | 14.207 ms | 14.272 ms |
| 6 SSU | 19,392 | 65.524 ms | 65.880 ms |
| 8 SSU | 1,024 | 4.108 ms | 4.109 ms |
| 8 SSU | 4,096 | 16.838 ms | 16.905 ms |
| 8 SSU | 24,024 | 100.315 ms | 100.990 ms |

按单次时间/请求访问数线性外推全部前缀，单案例仅 backlog 扫描约 **617–652 CPU 秒（10.3–10.9 分钟）**、**1158–1205 CPU 秒（19.3–20.1 分钟）**。这是本机当前解释器的粗估，不能当成远端 Python 3.14、多进程争用下的墙钟承诺；还未加入其他仿真工作。

完整结果也有独立可读证据：raw32_local S3 fixed 的 `assignment_wall_us` 合计 **18.279 s**，全运行 **1077.980 s**；raw32_stripe S3 fixed 分别 **18.248 s / 980.568 s**（两者均 2736 请求）。旧四卡 padded S1 fixed 为 **0.860 s / 150.478 s**，strong4 S3 fixed 为 **4.370 s / 114.217 s**。所以小输入中的 assignment 占比不高，不能外推到 2.4 万请求；反过来，消除启动扫描也不会消除数百万物理 I/O 事件、grant 规划、后端仲裁、ACK 账本和结果序列化的成本。计时统计是 `perf_counter`，会含线程/进程被系统暂停的时间。

## 什么条件下可跳过，同时保持决定不变

| 调用位置/配置 | 可以省略什么 | 必须保留什么 |
|---|---|---|
| coflow S1/S2/S3，assignment=fixed | 若只要求模拟决定与时间线相同，可省略整个 backlog；assigned 永远是 original NPU。若还要保持现有 `scores_ms` 日志，则只省略 I/O backlog。 | 原 arrival/日志顺序、原 NPU、原事件行为；保留日志时原顺序计算的 remaining compute。 |
| coflow S1/S2/S3，assignment=compute | 可完全省略未被消费的 unfinished I/O 计算。 | 每卡 remaining compute、unfinished request count、数组顺序；选择 key 是 `(compute, count, n != original, n)`。 |
| coflow S1/S2/S3，assignment=pipeline | 不能普遍省略 I/O backlog，它直接进入 disk/link score。 | 精确 I/O 向量、compute、count，以及 `(score, count, compute, n != original, n)` 的 tie 顺序。可用等价的整数聚合代替重扫。 |
| 原 shared_path_adapter 的 strategy1/strategy2 | 不能照抄 coflow compute/fixed 的剪枝；这里直接调用 shared `choose_npu`，使用 I/O backlog。 | 原 fair-pipeline 所需全部字段。 |
| 原 shared Baseline/Once/New once | 这些 arrival 分支原本就不调用 backlogs。 | 无此项收益。 |

coflow 的 `scores_ms` 在 fixed 下只是诊断，后续路由/发射不消费它。`backlogs` 的额外副作用只是 memoize `layer_counts_cache`，不会发事件、取压力、推进时钟或使用 RNG；在经过原输入验证的 immutable manifest 上，略去未使用的 cache 填充只改变运行时间。若希望整个结果除计时/源码哈希外保持一致，应保留原 compute score 日志，不要简单把 scores 填零或删除。

“队列很长”“仍处于 t=0”“预计 compute 会占主导”都不是 pipeline 可以传入零 I/O 的充分条件；第二次到达开始就已有真实的已知未来工作。即使所有候选都 compute-dominated，也需要保留 `remaining_compute + incoming_compute` 的实际浮点 score 和全部 tie 项，不能直接替换成看似等价的另一个 `min`。

## 建议的未来独立优化，按风险从低到高

**第一步：coflow fixed/compute 只计算 compute/count。** 抽取 `backlogs` 的 compute 部分，保留同一个 `sum`、同一队列次序、同样的 active 层 remainder 加法以及 count。完全不进入每请求×每层×SSU 部分。此步尚有 O(M²) 的轻量 compute 求和，但移除了上述 48/64 倍内层维度，可先独立证明。

**第二步：pipeline 的初始纯 arrival 波次使用整数 I/O 前缀账本。** 精确快路径条件为：初始波次尚未发生 batch dispatch/IO activation、所有已知请求仍在对应 admission_queue、无 active batch、无 IO-ready/ACK 进展、历史请求没有重绑或出队。每个 incoming 请求在其 arrival 时按原方法获得 layer counts；评分时账本只含此前已处理的请求，选好卡并执行原 arrival 后，再按其实际 assigned NPU 加入 `sum(all-layer counts)`。按卡/SSU 累加整数可与重扫结果完全相同。进入任何可能改变已知工作状态的事件后退出快路径，先回到原实现；不要仅凭 `now_ms==0` 猜测状态。输入支持 repeated/specific layer layouts 时应准确求和，不能始终假定首层×8。

这一步保留原 compute 求和以避免浮点变化，同时将 I/O 聚合从初始 O(M²LS) 降为 O(MLS+MNS) 量级（N 为 NPU 数，固定 32）。不能预先读取尚未到达请求来填写策略可见状态。

**第三步才考虑一般增量 backlog。** 需要同时处理到达/分卡、排队 L0 预取、接纳、层计算、请求完成、HBM ACK，以及每卡 active/queued 归属。I/O 减量必须按 HBM ACK，而不是 SSD service completion；backlog 包含未发出的未来层，也包含已发出但未到 HBM 的块。仅缓存队列长度、只按 SSD 完成减量、忽略尚未接纳的 L0，都会改变 pipeline 决策。每次重绑只允许影响这条新到达请求，不能改变既有请求。

**不要随意把 compute 的 `sum(...)` 改成常驻浮点 `+=`。** 这两种求和不保证逐位一致；Python 从 3.12 起还改变了 float `sum` 算法，见 [Python 官方 sum 文档](https://docs.python.org/3.14/library/functions.html#sum)。本项目已经出现过极小浮点/tie 顺序差异被调度放大的情况。若要优化此部分，必须另做跨版本的前缀与出队状态等价验证；使用更准确的算法也可能改变原决定，因此不是自动成立的纯性能修复。

## 精确等价验收标准

优化应在隔离目录与独立进程运行，使用同一冻结 manifest、同一 seed、同一 Python/平台先做 old/new 配对，然后在本机和远端解释器分别重复。不可拿 old 本机与 new 远端的差异混为优化结果。

1. 每个 arrival 的 request ID、模拟时刻、原/执行 NPU、collector snapshot 时刻、完整 scores、所有 tie-breaking 结果逐项完全相同；需要时把原/新 `KnownNPU` 在同一状态并行求值验证。除 wall timing 外，不用宽松 epsilon 放过 score 差异。
2. 每个事件的时间、类型、资源、request/layer/block 身份及消费顺序一致；RNG 状态或生成顺序、event 序号也应一致。优化不应添加/删除任何模拟事件。
3. 所有 grant 批次与发射顺序、Path 路由、SSD enqueue/service 选择与时间、HBM 完成顺序与时间一致；可在小型回归用例流式 hash，而非保留不可控的大 trace。
4. 所有 request/microbatch/layer 的 admission、IO activation/ready、compute start/end、completion、stall，以及逐卡/盘字节、事件计数、collector/CIR 日志和 invariants 一致。只允许明确白名单中的 Python wall timings、创建时间和源码哈希变化。
5. 同时覆盖 fixed/compute/pipeline × S1/S2/S3，全 t=0、不同原 NPU 分组、1 ns 交错到达、非零时间后续 arrival、同时 compute/ACK/arrival、queued L0 已预取、near-tie 异构 C、空 SSU 分量、多个布局。已有 `test_fixed_assignment_option...` 只验证不重绑，并未验证性能分支未计算 backlog 或完整 event trace 等价。

确认逐事件等价后再用逐步增长 M 的基准报告加速比。仅比较平均利用率相同不够：不同分卡与相位可能碰巧得到相同均值。

## 与模拟利用率严格分开

现有 adapter 明确报告 `modeled_control_cpu_latency_ms=0`、通信延迟为 0；`perf_counter()` 仅用于统计，没有加入 `context.current_time_ms`。减少 backlog 计算可使仿真更快结束，若满足上述等价标准，**模拟 U、stall 和 makespan 必须不变**。它不会修复 Baseline 的 FIFO 机制或 S1–S3 的策略失效，也不能据此推断真实部署的控制器有零成本。今后若向模拟时间注入 CPU/通信成本，那是另一个模型实验，不是纯性能重构。

审阅源码哈希：

```text
shared_path_sim_adapter.py 174103c5c17d5f75ccdfe41a2415cbe93a0b6cfe9858bdeed2b6c1628c0bebf2
coflow_sim_adapter.py 5f65a1392e4387d35bdc637faf7e136bfe4e73de36e812c75a13bc822bd364c6
coflow_client_policy.py 7a79e24ad380c91dc97f9e2c3173aabf9dfa2b64111d5b9836c6a7de6154378f
continuous_batch_sim.py c652bfb22fb6c3d219d7b8f78e172ca5808c55c5e929547ece5da0680129f9b1
```
