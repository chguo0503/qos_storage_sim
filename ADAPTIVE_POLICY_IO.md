# Adaptive Admission Scheme B V2.1 —— 输入输出详解

> 配套代码：`adaptive_policy_standalone.py`（可直接 `python3` 运行，不依赖仿真器）
>
> 正式实现：`adaptive_admission_scheme_b_v2_1.py` + `slo_admission_scheme_b.py` + `slo_admission_scheme_b_v2.py`

---

## 0. 先破除一个误解：它其实只有一个输入

策略看起来输入很多，是因为把**参数**和**输入**混在一起了。分开看：

```
   真正的输入（每次调用都变）：
      demand[N][S]        一个矩阵。就这一个。

   参数（配置一次，运行期不变）：
      explicit_spill_threshold = 0.75    切换 V1/V2 的阈值
      target_ratio             = 0.52    SLO floor 目标
      required_ratio           = 0.50    SLO floor 底线
      background_reserve_...   = 0.05    给落选者留的余量
      ssd_caps                 = 40.0    每块盘带宽
      npu_caps                 = 50.0    每个 NPU 链路带宽

   状态（跨调用记忆，只有一项）：
      pinned_npu_ids                     上次选中的 NPU，尽量保持
```

**输出也只有一个矩阵：`grants[N][S]`。**

```
        demand[N][S]  ──►  策略  ──►  grants[N][S]
                            ▲
                     6 个常数参数 + 1 个 pinned 列表
```

所以它不是「输入好多」，而是**一个矩阵进、一个矩阵出**，其余都是配置。

---

## 1. 输入：demand 矩阵

### 定义

```
   demand[n][s] = NPU n 想从 SSU s 上，以多快的速率读数据（GB/s）
```

### 重要：它是「剩余全部层」的聚合，不是单层

```
                   该请求【所有剩余未就绪层】要从 SSU s 读的字节总和 (GB)
   demand[n][s] = ────────────────────────────────────────────────────────
                        剩余 IO 层数 × 每层 compute 时间 (秒)
```

分子分母**都是聚合量**，量纲仍是 GB/s。报告里称之为 **"完整 remaining manifest"**。

**含义**：如果盘能按这个速率供数据，该请求直到结束都不用等 I/O。

### 一个具体例子

某请求还剩 8 层未就绪，每层 compute 1.288 ms，每层要从 SSU 3 读 0.0045 GB：

```
   剩余字节  = 8 × 0.0045 = 0.036 GB
   剩余计算  = 8 × 1.288  = 10.304 ms = 0.010304 s
   demand    = 0.036 / 0.010304 = 3.494 GB/s
```

### 为什么不用单层

```
   单层 demand：  每层字节 / 每层 compute       <- 只看眼前一步
   聚合 demand：  剩余总字节 / 剩余总 compute   <- 看到请求结束
```

**chguo: 代码这方面可以精简，因为一直是同构层**

**当前配置下两者数值几乎相同**，但语义不同。

先说清楚当前配置：

- **层是严格同构的**（`sim.py:288`）：16 层共享同一个 `layer_blocks` 对象，
  block 大小、SSU 分布、per-layer compute 全部相同。不是近似，是构造上的恒等。
- **请求内预取深度为 1**（`continuous_batch_sim.py:1324`）/：算第 L 层时只提交
  第 L+1 层的 I/O。唯一的例外是最后一层结束时触发的
  `cross_request_layer0_prefetch`，它预读的是**下一个请求**的第 0 层。

因此 `remaining_io_layers ∈ {remaining_layers, remaining_layers - 1}`，两者最多
差 1 层。层同构又让分子分母同乘 L 约掉，所以**聚合 demand 与单层 demand 在数值上
基本相等**。

那为什么还要用聚合量？因为**分子携带了额外信息**：

```
   单层：  0.0045 GB / 1.288 ms        只说"这一步要多快"
   聚合：  0.036 GB  / 10.304 ms       还说了"总共还剩 0.036 GB"
```

Stage A 的背包按 `normalized.sum()`（即该请求在所有 SSU 上的归一化总占用）排序，
挑"最便宜的先进"。这是一个**请求生命周期**决策——需要知道保住这个请求总共要花
多少资源。单层量的分子丢掉了"还剩多少层"，无法支撑这个排序。

> 如果未来引入**异构层**（不同层字节不同）或**更深的预取**，两者会真正分离，
> 届时聚合量是唯一正确的选择。当前配置下它主要是为了 Stage A 的排序语义。

### 三个实现细节

来自 `continuous_batch_sim.py:1537-1547`：

```python
remaining_work = [0.0] * context.num_ssu
remaining_io_layers = 0
for layer in range(max(0, compute_done_up_to + 1), context.n_layers):
    if request.io_ready[layer]:
        continue                    # (1) 已读好的层不再占带宽
    remaining_io_layers += 1
    for ssu_id, _, work_gb in _state_placement_groups(request, layer):
        remaining_work[ssu_id] += work_gb        # 累加，不是单层
```

1. **已就绪的层被跳过，分子分母配套。** 同一个循环里，跳过 `io_ready` 的层时
   既不累加它的字节，也不累加它的 compute。所以分母必须是 `remaining_io_layers`
   而非 `remaining_layers`——否则分子是 k 层的量、分母是 k+1 层的时间，需求会被
   低估。当前预取深度为 1，两者最多差 1 层。

2. **进行中的层保留完整 manifest。** 即使某层已读了一半也按整层计入。源码注释：

   > Future layers and an in-progress barrier retain their full manifest work so
   > the policy does not depend on simulator-private partial-command progress.

   这是**刻意设计**——让策略不依赖仿真器内部的部分命令进度，从而可移植到真实
   系统（真盘也很难精确报告"这条命令读了 37%"）。代价是需求被略微高估。

3. **有 fallback 路径**（`adaptive_admission_scheme_b_v2_1.py:161-167`）：当
   `remaining_work` 为空时，退化成「下一层形状 × 剩余层数」外推。

### 矩阵长什么样

128 NPU × 24 SSU，每行一个 NPU（batch=1 时也就是一个请求）：

```
            SSU0    SSU1    SSU2   ...   SSU23
   NPU0  [  3.49    3.51    3.47   ...    3.48  ]   <- 一行加起来 = 该请求总需求
   NPU1  [  0.58    0.57    0.59   ...    0.58  ]
   NPU2  [  1.20    1.19    1.21   ...    1.20  ]
    ...
   NPU127[  3.50    3.48    3.52   ...    3.49  ]
            ↑
            一列加起来 = 该盘被要求的总速率（可能远超 40）
```

### 三个关键性质

1. **一行 = 一个 coflow。** 该请求必须**所有** SSU 的数据都到齐，才能算下一层
   （layer barrier）。所以只喂饱一行的一部分是没用的。

2. **需求差异主要来自 compute，不是字节。** 真实 `data` 的四类请求：

   | 类别 | 每层字节 | 每层 compute | raw demand |
   |---|---:|---:|---:|
   | SS | 0.107338 GB | 1.288 ms | **83.34 GB/s** |
   | SL | 0.106750 GB | 7.729 ms | **13.81 GB/s** |
   | LS | 0.193024 GB | 6.695 ms | 28.83 GB/s |
   | LL | 0.192688 GB | 13.097 ms | 14.71 GB/s |

   **SS 和 SL 的字节只差 0.55%，需求差 6.03 倍。** 盘只看得见字节，看不见
   compute —— 这正是需要显式 demand 输入的根本原因。

3. **矩阵通常严重超售。** SSU24 时 fleet demand ≈ 2665 GB/s，容量只有 960 GB/s
   （36%）。策略的工作就是决定这 64% 的缺口让谁承担。

---

## 2. 参数

| 参数 | 默认 | 含义 | 怎么定的 |
|---|---:|---|---|
| `target_ratio` | 0.52 | SLO floor 目标：拿到需求的 52% 就算达标 | SLO 定义 α=2（允许 2× 理想 TTFT），留 2% 余量 |
| `required_ratio` | 0.50 | 数学底线，`target` 放不下时退到这里 | 严格对应 α=2 |
| `background_reserve_fraction` | 0.05 | 每块盘留 5% 给落选请求 | 防止落选者完全饿死 |
| `explicit_spill_threshold` | **0.75** | selected fraction 低于它就用 V2 | **标定值，非最优** |
| `ssd_caps` | 40.0 | 每块盘带宽 GB/s | 硬件 |
| `npu_caps` | 50.0 | 每个 NPU 链路带宽 GB/s | 硬件 |

### 关于 0.75 —— 最需要注意的一个数

模块 docstring 自己写明：

> The 0.75 default is an operating-point heuristic, not a universal optimum.

它是在 4 个观测点之间"切一刀"切出来的：

```
   严重过载点：  0.6875 (32×6)    0.6875 (128×24)
                        ↓
                 ─── 0.75 ───     <- 阈值放在空隙里
                        ↑
   中等负载点：  0.8125 (32×10)   0.8047 (128×40)
```

**换工作负载必须重新标定。** 这是报告 §8.1 限制表的第一条。

---

## 3. 状态：pinned_npu_ids

唯一的跨调用记忆，作用是**防止抖动**。

如果每次决策都重新挑一批 NPU，某个请求可能这次被选中、下次落选，结果两头不
讨好：既没跑完，又占了资源。pinning 让上次选中的请求**优先**保持选中。

```python
# 控制器内部（adaptive_admission_scheme_b_v2_1.py:192）
pinned = tuple(
    npu for npu, request_id in self.selected_request_by_npu.items()
    if request_by_npu.get(npu) == request_id     # 必须还是同一个请求
)
```

注意 `== request_id` 这个检查：如果该 NPU 换了新请求，pin 自动失效。

**pinning 是"优先尝试"，不是硬承诺**——容量或 manifest 变化时旧请求仍可能掉选。

---

## 4. 输出：grants 矩阵

```
   grants[n][s] = 给 NPU n 在 SSU s 上的 CIR（GB/s）
```

保证三条：

```
   1. grants[n][s] <= demand[n][s]        不会给用不掉的带宽
   2. sum_n grants[n][s] <= 40            每块盘不超售
   3. sum_s grants[n][s] <= 50            每个 NPU 链路不超售
```

### 怎么落到硬件

每个 NPU 在每块 SSU 上占一个专属 Path，把 grants 写成 Path CIR：

```
   CIR[s][path(n)] = grants[n][s]
```

**重要**：CIR 是仲裁保证/长期虚拟速率，**不是**把命令的物理执行速率改成 G。
命令仍以 40 GB/s 非抢占执行，新 CIR 只影响后续仲裁顺序。未用带宽由
work-conserving 规则再分配。

---

## 5. 算法：三步

```
  demand[N][S]
       │
       ▼
  ┌─────────────────────────────────────────────────┐
  │ Stage A：谁进 SLO floor                          │
  │   1. 所有人都能拿 52%？   -> 全员入选，收工        │
  │   2. 所有人都能拿 50%？   -> 全员入选，收工        │
  │   3. 都不行 -> 贪心背包：                         │
  │        先 pinned，再按归一化 SSD 占用从小到大      │
  │        （便宜的先进，最大化"达标请求数"）           │
  └─────────────────────────────────────────────────┘
       │
       ├──► selected_fraction = 选中数 / 活跃数
       │
       ▼
  ┌─────────────────────────────────────────────────┐
  │ Stage B：floor 之后剩下的字节怎么花                │
  │                                                 │
  │   fraction >= 0.75  ->  V1 coflow residual      │
  │   fraction <  0.75  ->  V2 explicit spill       │
  └─────────────────────────────────────────────────┘
       │
       ▼
   grants[N][S]
```

### V1 和 V2 只差 Stage B

**选谁进 floor 是完全一样的**——正式代码里有断言：

```python
if v2.selected_npu_ids != v1.selected_npu_ids:
    raise AssertionError("V1 and V2 admission sets diverged")
```

| | V1 coflow residual | V2 explicit spill |
|---|---|---|
| 做法 | 剩余容量对**所有**请求做一次统一残量分配 | 三段：落选者 background → 喂满选中者 → 溢出 |
| 理由 | 避免把稀缺份额花在碎片上，而请求仍被别的 SSU 卡住 | 重过载下更坚决地保证被选中的那批 |
| 适合 | 大部分请求能选上 | 只能选上一小部分 |

### 实测：三个工作点各走了哪条路

来自 `results/steady_state_128npu_adaptive_v2_1/report.md`：

| SSU | selected fraction | explicit/coflow 次数 | 实际分支 |
|---:|---:|---|---|
| 24 | 0.5859 | **248 / 1** | 几乎全走 V2 |
| 40 | 0.9219 | 0 / 166 | **全部 V1** |
| 70 | 1.0000 | 0 / 131 | **全部 V1** |

**所以 SSU40 和 SSU70 完全等价于纯 V1**，一次 V2 都没走。这解释了为什么
`admission_25ms`（纯 V1）和 `adaptive_v2_1` 的结果逐位相同：

```
   SSU40   纯V1 util=51.57%   adaptive util=51.57%
   SSU70   纯V1 util=90.85%   adaptive util=90.85%
```

---

## 6. 关于 "once per layer"：它比你想的更省

你担心它像 `layer_once` 那样每层都跑。**不是的**，它是**事件触发 + 最小间隔**：

```python
"trigger": "batch_membership_event",     # 请求加入/离开才触发
"min_interval_ms": 25.0,                 # 两次决策至少隔 25 ms
"semantics": "event-gated minimum spacing, not periodic ticks"
```

**不是每 25 ms 无条件轮询**，而是「有事件才算，且两次之间至少隔 25 ms」。

实测调用次数（2 秒测量窗）：

| SSU | 决策次数 | 平均间隔 |
|---:|---:|---:|
| 24 | 249 | 8.0 ms |
| 40 | 166 | 12.0 ms |
| 70 | 131 | 15.3 ms |

对比 `layer_once`：128 NPU × 16 层 × 每层触发 ≈ **每秒上万次**。V2.1 少了两个
数量级。

### 单次决策的耗时

`adaptive_policy_standalone.py` 实测（本机）：

```
   128 NPU × 70 SSU     1.4 ms
   256 NPU × 128 SSU   89.3 ms
```

128×70 时 1.4 ms / 15.3 ms 间隔 ≈ **9% 的一个核**，完全可接受。

### 真正的成本不在计算，在写寄存器

报告 §7.4 指出：主要成本是 **70 × 256 Path 表的寄存器扇出**。实测 Path 写入次数：

| SSU | Path writes |
|---:|---:|
| 24 | 754,209 |
| 40 | 844,631 |
| 70 | 611,236 |

**优化应该往这里使劲，不是往算法复杂度上使劲。**

---

## 7. 优化建议

按性价比排序：

### 高性价比

1. **只写变化的 CIR。** 相邻两次决策的 grants 大部分格子不变，加一个 delta
   阈值（比如变化 < 0.1 GB/s 就不写），能砍掉大部分 Path 写入。这是最直接的
   收益，且不改变算法语义。

2. **V1 每次都算，即使最后用 V2。** 因为决策信号 `selected_fraction` 来自 V1
   的选择结果。但 Stage A 在 V1/V2 里是**完全相同**的（正式代码用断言保证），
   所以可以把 Stage A 抽出来只算一次，再只跑选中的那个 Stage B。

   实测这笔浪费（128×24 过载工作点，会走 V2 分支）：

   ```
      纯 V1        1.35 ms
      V2.1         3.52 ms      <- 多花 161%
   ```

   非过载点（走 V1 分支）没有这个开销，因为 V2 根本不会被调用。所以这个优化
   **只在过载工作点有收益**，但那恰恰是决策最频繁的时候（SSU24 平均间隔
   8.0 ms，比 SSU70 的 15.3 ms 密一倍）。

3. **增大 `min_interval_ms`。** 实测 50 ms 与 25 ms 的结果几乎一样：

   ```
      SSU40   25ms util=51.57%  |  50ms util=51.70%
      SSU70   25ms util=90.85%  |  50ms util=90.78%
   ```

   但 SSU70 的 SLO 从 98.88% 掉到 92.26%，所以**过载点不能放太松**。

### 中等

4. **SSU40/70 直接用纯 V1。** 既然实测从不触发 V2，在这些工作点上可以跳过
   adaptive 层。但这依赖当前 workload，换负载要重测。

5. **增量式 Stage A。** 大部分决策之间 active 集合只变一两个请求，贪心背包
   可以从上次结果增量修正，而不是从头排序。

### 需要先验证

6. **降低决策频率到「只在 selected_fraction 跨过阈值时」。** 风险是错过需要
   重新分配的时刻，必须先做敏感性实验。

---

## 8. 直接上手

```bash
# 跑内置演示（128 NPU × 24/40/70 SSU）
python3 adaptive_policy_standalone.py

# 只看某个规模
python3 adaptive_policy_standalone.py --npu 128 --ssu 24

# 与正式实现对拍（验证等价）
python3 adaptive_policy_standalone.py --selftest
```

演示输出示例：

```
=== 128 NPU x 24 SSU ===
  fleet demand     4502.14 GB/s
  fleet capacity    960.00 GB/s   (21.3%)
  selected       90/128  fraction=0.7031  ->  v2_explicit_selected_spill
  SSD 用量  max 40.000 / 40   NPU 用量 max 15.542 / 50
  各类别拿到的需求满足率：
    SS  n= 32  mean=  1.62%   <- 需求 83 GB/s，被牺牲
    SL  n= 32  mean= 55.98%   <- 需求 14 GB/s，达标
    LS  n= 32  mean= 44.10%
    LL  n= 32  mean= 55.74%
```

**注意 SS 那一行**：它需求最高（83.34 GB/s），被策略判定为"太贵"而牺牲掉，
换取其余三类达标。这就是 admission control 的本质——**在放不下所有人时，
最大化能达标的请求数，而不是让所有人一起不达标。**

> **演示矩阵的两点简化**（不影响算法验证，但不可直接对比正式结果）：
>
> 1. `demo_matrix()` 按**单层**构造 demand（`每层字节 / 每层 compute`）。当前层
>    严格同构、预取深度为 1，所以它与聚合量数值基本等价；但它丢掉了"还剩多少层"
>    这一信息，而 Stage A 的背包排序依赖该信息。真实控制器用完整 remaining manifest。
> 2. 字节按 Dirichlet 散布到各 SSU，与正式实验的 ring placement 不同。

### 在自己代码里用

```python
import numpy as np
from adaptive_policy_standalone import allocate

demand = np.array([...])            # (N, S) GB/s
result = allocate(demand,
                  explicit_spill_threshold=0.75,
                  ssd_caps=40.0, npu_caps=50.0,
                  pinned_npu_ids=last_selected)

grants = result.grants_gbps         # (N, S) 写成 CIR
last_selected = result.selected_npu_ids   # 存起来，下次传进去
```

---

## 9. 代码等价性验证

`adaptive_policy_standalone.py` 与正式实现的对拍结果：

```
   300 组随机矩阵（均匀 / 稀疏 / 指数 / 低载 × 随机 pinned）
   最大逐元素偏差 = 0.000e+00      不一致 0 组
   分支覆盖：V2 走了 95 组，V1 走了 205 组

   边界情况：全零 / 单 NPU / 单 SSU / 巨大需求  -> 全部 gap = 0
```

之所以能做到完全一致，是因为它复用了 `continuous_batch_control.py` 里那两个
分配原语（该模块只依赖 math/numpy，不含任何仿真器代码），只重写了上层的决策
逻辑。**"脱离仿真"和"与正式实现等价"两点同时成立。**

---

## 10. 已知限制

摘自报告 §8.1，与本文档相关的几条：

| 限制 | 影响 |
|---|---|
| 0.75 / 0.52 / 0.05 是当前 trace 的启发式 | 非通用最优；异构 SLO 无 per-request ratio |
| Greedy 非最优 knapsack | 不保证最大达标数；同质 tie 按 NPU ID |
| Pinning 是优先尝试，非硬承诺 | 容量/manifest 变化时旧请求仍可能掉选 |
| 不读 pressure / 实际到达字节 | manifest 或 compute 估计错时 grant 会偏 |
| CIR 不是精确瞬时速率 | 只有 backlogged 且 surplus 语义匹配时才接近长期 grant |
| batch=1，每 NPU 一个 active coflow | 多请求并发无法直接映射「一行一个 NPU」 |
| finite PIR 未接线 | `path_pirs` 有结构但运行期不生效，硬隔离无法表达 |
