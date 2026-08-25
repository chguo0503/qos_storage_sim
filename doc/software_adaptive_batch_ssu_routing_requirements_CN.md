# 软件侧自适应分批 SSU 选路需求

> 状态：已完成需求确认，等待实现
>
> 日期：2026-08-25
>
> 用途：交接给另一个 AI 实现。本文件描述目标、约束、算法口径、实验方案和验收条件；当前尚未实现本文方案。

> **后续状态（2026-08-25）**：用户已决定停止该方案的粗搜索并从当前主线移除
> 自适应 K/q/refill 实现。本文件作为原始需求归档保留；当前可执行代码只包含统一
> NPU 50 GB/s 限速的 baseline 与静态 QoS 路由对比，不能把下文“等待实现”理解为
> 当前仓库仍计划自动运行这套搜索。

---

## 0. 一句话目标

保持 block 的既有 SSU 放置和 SSD 静态 QoS 配置不变，在 NPU 客户端增加一个
`(request, layer)` 级的跨 SSU 分批提交器：它只能读取各候选 SSU 的 256 项
Path I/O-count，优先从压力较小且确实持有本层剩余 block 的 SSU 提交 I/O，
并根据剩余数据、剩余 layerwise 时间和真实的 NPU 50 GB/s 聚合上限动态调整
SSU fan-out 与批大小。

这里的“批发送”只表示**分批进行选路和提交**，不表示把多个 I/O 合并成一个
硬件命令，也不研究 doorbell、CPU 调用或网络消息的批处理开销。

本文中的 SSU 对应当前仿真器里由一套 `DiskState/DiskIOScheduler` 表示的存储单元。
文中“当前代码事实”是已核对的现状，“必须/不得”是目标设计约束；性能改善只是
待实验验证的假设，不是预设结论。动态 fan-out 也可能因为错误的 count-to-time
估计、额外排队或当前单-active-SSU 模型的 HoL 偏差而变差。

---

## 1. 当前代码事实与本次新增目标

### 1.1 当前代码已经具备的能力

当前 `sim.py` 已经实现：

- 一层 I/O 先按目标 SSU 拆成多个 `(request, layer, SSU)` 提交状态；
- 每个状态使用固定的 `client_submit_batch_size`，默认一次最多提交 8 个 I/O；
- 全局按 `ready_time` 调度，同一时刻随机排列 NPU，每轮每个 NPU 最多提交一个
  固定 batch；
- QoS 模式可按 `pressure_read_interval` 读取某一 SSU 的 256 项 Path count；
- 在目标 SSU 内使用 `client_select_qos_paths()` 选择本类别允许的 Path；
- Path 可非空、无 owner/lease，入队后按 Path 内 FCFS；
- 每块 SSU 在当前模型中最多只有一个不可抢占的 active I/O。

### 1.2 当前固定 batch 不等于本次需求

本次需求不是简单修改 `client_submit_batch_size=8`，也不是继续逐个轮转每个
`(request, layer, SSU)` 状态。两者差异如下：

| 维度 | 当前实现 | 本次目标 |
|---|---|---|
| 决策作用域 | 单个 `(request, layer, SSU)` | 整个 `(request, layer)` 的所有目标 SSU |
| SSU 选择 | 按已有状态队列轮转 | 根据各候选 SSU 压力动态选择 |
| fan-out | 由 placement 和轮转顺序隐式决定 | 动态 `K`，正常情况下不少于 8 |
| batch 大小 | 固定最多 8 个、发往一个 SSU | 基线策略为 `B=K`、每个所选 SSU 一个 I/O |
| 下一轮触发 | `ready_time`，可在同一时刻继续轮转 | 当前 layer 的全局 in-flight 降到半批低水位 |
| NPU 50 GB/s | 当前没有真正限速 | 必须跨全部 SSU 统一执行 |

因此实现时需要新增 layer 级协调状态，不能只在现有
`_ClientSubmissionState` 外面再套一个同步 `for` 循环。

---

## 2. 已确认、不得擅自改变的需求

### 2.1 数据放置

- block 到 SSU 的放置由现有 placement 阶段确定，运行期间不迁移。
- 这里的“固定”精确指：在同一个实验 cell 内，已经生成的
  `(request_id, layer, block_idx) -> SSU` 映射对所有配对策略相同。当前随机
  placement 可以逐 layer 生成不同 SSU；本需求不要求不同 layer 复用同一组 SSU。
- NPU 只能在**实际持有当前 layer 尚未完成 block 的 SSU**中选择。
- 不增加副本、EC 备选位置或跨 SSU 重定向。
- 跨 SSU 可以重排尚未提交 block 的先后顺序；I/O 一旦入队，不再迁移。
- 因此“选择 SSU”只是在既有 `unsent_by_ssu` 队列之间选择本轮先提交谁，绝不
  表示选路器可以改写一个 block 的目标 SSU。

### 2.2 硬件动态信息边界

每块 SSU 只能提供完整的 256 项 Path I/O-count：

```text
path_io_count[path_id] = active I/O 数 + pending I/O 数
```

不能假设客户端还能获得：

- queued bytes；
- active I/O 剩余字节或剩余服务时间；
- 实测 Path/Group 带宽；
- 其他 NPU 的身份、需求或 deadline；
- Path owner、租约或原子预留结果。

第一版继续采用当前的零延迟、无丢失压力读取假设，不模拟压力读取或提交的软件
开销。该假设只能视为软件算法上界，结果中必须明确披露。

### 2.3 SSD QoS 配置

- SSD 初始化时配置一次 CIR、PIR、Path WRR 和 Group WRR；运行期间不改写。
- 客户端严格使用请求类别对应的静态 Path 池，不跨 SS/SL/LS/LL 类别。
- 不引入 token bucket、动态 CIR/PIR/WRR、中央分配器、Path lease 或 admission
  controller。
- 主实验继续使用 uncapped PIR；有限 PIR 在当前实现中存在长期服务率与单 I/O
  实际速度重复限速的问题，不应混入本次实验。

### 2.4 NPU 聚合带宽

- `NPU_BW_LIMIT = 50 GB/s` 是真实物理聚合上限，不是仅用于计算 ideal 指标的常量。
- 同一个 NPU 从所有 SSU 同时返回的数据总速率必须始终不超过 50 GB/s。
- 该物理约束必须同时应用于 Baseline、现有 QoS 和新策略，避免只限制实验组。

### 2.5 当前队列语义

- 同一 Path 内严格 FCFS。
- Path 可被多个 NPU 共享，非空 Path 仍可继续入队。
- 无 Path owner、lease、reservation 或迁移。
- 保持当前“每块 SSU 合计最多一个不可抢占 active I/O”的仿真假设。
- 后查询者可以看到同一仿真时刻此前已经完成 enqueue 的 I/O；不恢复
  timestamp-wide 冻结快照或统一提交 barrier。

最后一条是后续确认的口径，覆盖早期“同一时刻所有 NPU 先看共同旧快照、再统一
commit”的草案。当前实现是 seeded shuffle 后依次 plan+enqueue；保留它意味着
同刻后决策者确实可能看到先决策者的新 I/O。这会带来事件顺序影响，结果中应披露，
但本次不要同时引入两阶段 barrier 作为隐藏改动；如需比较，应另列消融实验。

---

## 3. 术语和状态定义

### 3.1 关键符号

| 符号 | 含义 |
|---|---|
| `E` | 当前 layer 仍持有未提交 block 的候选 SSU 集合 |
| `K` | 当前规划轮选择的 SSU fan-out 数，按整个 `(request, layer)` 定义 |
| `q` | 每个所选 SSU 在一个目标窗口中的 I/O 数；主方案 `q=1` |
| `B_target` | layer 级目标 in-flight I/O 数，`B_target = K × q` |
| `W_low` | 低水位，`ceil(B_target / 2)` |
| `D_rem` | 当前 layer 尚未完成的总数据量，单位 GB |
| `T_rem` | 数据需要就绪前的剩余时间，单位 ms |
| `R_req` | 当前 layer 所需的 NPU 聚合读取速率，单位 GB/s |
| `A_s` | 仅根据 256 Path count 和静态配置估计的 SSU `s` 可用服务率 |
| `F_s` | 向 SSU `s` 再提交一个本类别 block 后的预计完成时长分数 |

### 3.2 layer 级 block 状态

每个 `(request, layer)` 至少需要显式区分：

```text
UNSENT       已放置但尚未提交
IN_FLIGHT    已提交但尚未完成
COMPLETED    已完成
```

`D_rem` 必须包含 `UNSENT + IN_FLIGHT` 的数据量。软件侧不知道 active I/O 的
部分完成字节，因此一个尚未完成的 I/O 按完整 block 大小计入 `D_rem`；这是保守
估计，不能偷看仿真器内部 `remaining_gb` 作为客户端输入。

---

## 4. deadline 与动态带宽需求

### 4.1 L0

L0 前面没有可用于隐藏读取的 NPU 计算窗口。第一版不把画像中的 `ttft_ms`
擅自解释为硬 deadline，而是把 L0 目标带宽设为 NPU 上限：

```text
R_req(L0) = 50 GB/s
```

L0 的目标是尽快完成，同时仍受每块 SSU 能力、排队和 NPU 聚合上限约束。

### 4.2 L1 及以后

计算第 `k-1` 层时预取第 `k` 层。第 `k` 层数据的就绪时限是当前计算层的
预计完成时刻：

```text
T_rem_ms = current_compute_expected_end_ms - now_ms

R_req = min(
    50,
    1000 × D_rem_GB / max(T_rem_ms, epsilon)
)
```

不得使用“第二批固定乘 1.5”之类的规则。第二轮需求可能升高，也可能降低：

- 剩余时间下降得比剩余数据快，`R_req` 升高；
- 第一批进展较快，剩余数据下降得更快，`R_req` 降低。

若 `T_rem <= 0` 且仍有未完成数据，则：

- `R_req` 记为 50 GB/s；
- 记录一次 layer deadline miss/late-prefetch 状态；
- 继续完成 I/O，不能丢弃请求。

### 4.3 `required_bw` 字段

当前 `_load_from_key()` 会丢弃四元组画像中的显式 `required_bw`。新实现必须保留
该字段，至少用于记录和校验 nominal demand；不得继续让输入字段悄无声息地失效。
运行时批次选择以本节的 `D_rem / T_rem` 为动态需求，不使用任意固定倍率提高
第二批需求。

---

## 5. 只使用 256 Path count 的 SSU 压力估计

“压力小”不是硬件直接提供的 bandwidth 数值，而是客户端估计结果。文档和字段
命名必须区分 `reported_io_count` 与 `estimated_available_rate`。

每次规划轮对 `E` 中每块候选 SSU：

1. 读取该 SSU 的完整 256 Path count；
2. 取得当前请求类别允许的 Path 池；
3. 假设把该 SSU 上的下一个未提交 block 加入候选 Path；
4. 使用现有 Group-aware SED/静态 CIR+WRR 长期服务率逻辑，估计最佳合法 Path；
5. 得到该 SSU 的 `F_s` 和 `A_s`；
6. 以 `F_s` 小者优先排列候选 SSU。

为避免实现者任意发明 `A_s`，第一版使用同一个 SED 预测量换算：令 `b_s` 为该
SSU 下一条待提交 block 的大小，`F_s` 为它从现在起的预计完成时长，则：

```text
A_s = min(SSU_BW, b_s / max(F_s, epsilon))
```

单位必须统一为 GB 和秒，结果为 GB/s。`F_s` 必须包含 count 推出的预计旧积压和
这条新 I/O 的服务时间，不能使用绝对仿真时间戳。`A_s` 只是近端有效贡献的启发式
估计，不是硬件报告的空闲带宽，也不是 SLO 保证；实现版本和公式必须写入结果 spec。

估算旧积压字节时只能使用本请求剩余 block 的代表性大小（例如中位数）乘以
报告的 I/O 数。这不是对真实剩余时间的精确预测；异构 I/O 大小时会有系统性
误差，必须作为结果限制说明。

完全平分时使用基于 `(request_id, layer, planning_round, ssu_id)` 的固定哈希或
稳定轮转打散。不得永远选择最低编号 SSU，也不得增加中央协调。

同一规划轮选中一个 SSU/Path 后，应把本轮已计划提交的 I/O 加入客户端本地
shadow count，避免同一 NPU 在一轮内反复选择同一热点。

---

## 6. 动态 K、初始批次和半批低水位

### 6.1 候选不足时的确定行为

令：

```text
candidate_count = min(
    有 UNSENT block 的 SSU 数,
    UNSENT block 数
)

K_floor = min(8, candidate_count)
```

若实际候选不足 8，使用全部候选；不能创建副本或选择没有数据的 SSU 来凑 8。

### 6.2 动态选择 K

按 `F_s` 从小到大选择候选 SSU：

1. 先选择至少 `K_floor` 个；
2. 继续加入候选，直到：

   ```text
   min(50, sum(A_s for selected SSUs)) >= R_req
   ```

3. 若全部候选的估计能力仍不足，选择全部候选，并记录
   `predicted_capacity_shortfall=True`；
4. 第一版不增加未经验证的 headroom，默认比较阈值就是 `R_req`。headroom 可作为
   后续独立实验参数，不能藏在算法常量中。

### 6.3 主方案的 batch 定义

主方案：

```text
q = 1
B_target = K
```

初始规划轮向每个选中 SSU 提交一个 I/O。因此负载较高、剩余时间更短或候选 SSU
更拥挤的 NPU 可能得到更大的 `K` 和 batch；负载较轻的 NPU 通常停在最小 fan-out。

### 6.4 半批低水位补充

低水位按整个 `(request, layer)` 的全局 in-flight 数量计算，不是按单个 SSU，
也不是现有 `(request, layer, SSU)` 固定 8-I/O 状态的 cursor。

```text
W_low = ceil(B_target / 2)
```

当一次真实 I/O completion 使该 layer 的 `IN_FLIGHT <= W_low` 且仍有 `UNSENT`
block 时：

1. 进入下一规划轮；
2. 重新读取所有当前候选 SSU 的 256 Path count；
3. 重新计算 `D_rem`、`T_rem`、`R_req`、`K_new` 和 `B_target_new`；
4. 提交：

   ```text
   top_up = max(0, B_target_new - current_in_flight)
   ```

5. 最多提交 `top_up` 个 block，并重新设置低水位；
6. 若 `top_up=0`，等待后续 completion，不能在同一时刻自旋；
7. 一个 completion 时间点只能为同一 layer 预约一轮有效 refill，使用 generation
   或 pending 标志去重。

这是一种 sliding-window top-up，而不是每次低水位再额外叠加完整新 batch。

规划和提交一轮后必须把控制权交回事件循环。严禁在一个同步 Python 调用中连续
规划完所有后续 batch，否则时间不推进、没有 completion，也无法体现本方案。

### 6.5 后续探索的每 SSU 深度

主方案先使用 `q=1`。实验中再独立测试：

```text
q ∈ {2, 4, 8}
B_target = K × q
```

额外 I/O 按估计 `A_s` 加权分配，但必须满足 block 确实位于目标 SSU。该实验用于
判断更深客户端 outstanding 是否减少控制轮次，不能预设 `q=8` 更好。

---

## 7. NPU 50 GB/s 聚合限速

### 7.1 必须实现的比例缩放

对于同一个 NPU 当前从不同 SSU 返回的所有 active I/O，先取得 SSU 侧给出的
原始速率 `r_i_raw`。若总和超过 50 GB/s：

```text
scale = 50 / sum(r_i_raw)
r_i_effective = r_i_raw × scale
```

否则：

```text
r_i_effective = r_i_raw
```

示例：

```text
40 + 40 GB/s  -> 25 + 25 GB/s
8 × 40 GB/s   -> 8 × 6.25 GB/s
40 + 5 GB/s   -> 保持 40 + 5 GB/s
```

### 7.2 事件循环要求

当前调度器按 SSU 独立计算 completion。增加 NPU 聚合上限后，一个 NPU 在任意
SSU 上 active 集合发生变化时，都可能改变该 NPU 其他 SSU I/O 的有效速率。
因此实现必须：

1. 先把所有受影响 active I/O settle 到当前时刻；
2. 跨 SSU 重新计算该 NPU 的比例缩放；
3. 更新所有受影响 I/O 的剩余完成时刻；
4. 使旧 completion event 失效，并安排新 event；
5. 保证任何时刻同一 NPU 的有效速率之和不超过 `50 + epsilon`。

不得只在创建 I/O 时算一次上限，也不得把每条 I/O 分别 cap 到 50，因为每块 SSU
本来只有 40 GB/s，那样不会产生聚合限制。

### 7.3 必须披露的模型偏差

当前模型每块 SSU 只有一个不可抢占 active I/O。若某个 NPU 同时从 8 块 SSU
读取，比例限速可能让每个 active I/O 以约 6.25 GB/s 长时间占住各自 SSU，产生
比真实多 outstanding SSD 更强的 Head-of-Line blocking。

本次不擅自修改这个既定后端假设，但结果必须报告该偏差，不能直接推广为真实
SSD 行为。

---

## 8. 建议的实现结构

名称可调整，但职责必须分开。

### 8.1 layer 级客户端协调状态

建议新增类似：

```python
AdaptiveLayerSubmissionState
```

至少保存：

- `request_id`, `npu_id`, `layer`, `category`；
- 按 SSU 分组的 `UNSENT` block 队列；
- `IN_FLIGHT` block 集合及其大小；
- completed/remaining bytes；
- 当前 planning round；
- `K`, `q`, `B_target`, `W_low`；
- 当前 compute deadline/expected end；
- refill pending/generation；
- pressure-read、selected-SSU、capacity-shortfall 和 deadline-miss 统计。

现有 `_ClientSubmissionState` 可以继续服务旧策略，但新策略不能把每个 SSU 状态
当作彼此独立的固定 8-I/O producer。

### 8.2 推荐的职责拆分

建议拆成可单测的纯计算与事件副作用：

```text
compute_required_rate(...)
estimate_ssu_candidates(...)
select_adaptive_ssus(...)
plan_layer_top_up(...)
submit_planned_blocks(...)
rebalance_npu_bandwidth(...)
```

纯函数不得读取事件循环全局状态；事件层负责读取报告、enqueue、更新 generation
和安排 completion/refill。

### 8.3 Path 选择

SSU 选定后继续复用 `client_select_qos_paths()`：

- 使用同一规划轮刚读取的压力快照，避免无意义地重复读取；
- 严格限制在请求类别 Path 池；
- 每个 I/O 默认独立选择 Path；
- 计入本规划轮已计划提交的 block；
- 保持 Path 内实际 enqueue 顺序和 FCFS。

---

## 9. 实验矩阵

### 9.1 必须保留的对照组

所有策略必须使用相同请求、相同 block 大小、相同 block→SSU placement 和相同
随机输入。至少比较：

| 组 | 说明 |
|---|---|
| `baseline_bypass` | 现有 Baseline，但加入共同的 NPU 50 GB/s 物理上限 |
| `qos_static_cir_current` | 当前固定 per-SSU batch/Path SED，同样加入 NPU 上限 |
| `fixed_k8_q1` | layer 级固定 `K=8, q=1`，候选不足时用全部 |
| `fixed_k16_q1` | layer 级固定 `K=16, q=1` |
| `fixed_k32_q1` | layer 级固定 `K=32, q=1` |
| `adaptive_k_q1` | 本文主方案 |
| `adaptive_k_q2/q4/q8` | 后续每 SSU 深度消融 |

固定 K 也必须按压力选择 SSU；它们用于判断收益来自“动态 K”还是仅来自跨 SSU
压力排序。

### 9.2 负载维度

结果至少按以下维度分层，而不是只报告总体均值：

- SS、SL、LS、LL 类别；
- `R_req` 区间；
- 每层剩余 block 数/GB；
- 实际候选 SSU 数；
- 所选 `K` 和 `q`；
- L0 与 L1+；
- 当前实验的 40/56 SSU 点。

seed 42 只能用于 smoke/debug。形成性能结论前应运行多个独立 workload/placement
seed，报告逐 seed 结果和不确定性；不能只挑有利 seed。

### 9.3 不得混淆的控制变量

- pressure-report 延迟与提交软件开销在第一版固定为 0；
- CIR/PIR/WRR 配置固定；
- placement 固定且配对；
- NPU 50 GB/s 上限对所有组相同；
- submit-order RNG 必须与 workload/placement RNG 独立；
- 改变 `K` 时不能同时偷偷修改 Path pool、类别 CIR 或后端 active 数。

---

## 10. 结果指标

### 10.1 正确性与资源约束

- 所有请求和所有 block 完成；
- 无 block 丢失、重复提交或重复完成；
- block 实际提交 SSU 与原 placement 一致；
- QoS Path 始终位于请求类别合法池；
- Path 内 FCFS 不被破坏；
- 每块 SSU `max_backend_active_io <= 1`；
- 每个 NPU 任意时刻聚合有效速率 `<= 50 GB/s + epsilon`；
- 仿真结束时所有 SSU outstanding 和所有 layer UNSENT/IN_FLIGHT 均为 0；
- 运行期间没有修改静态 CIR/PIR/WRR。

### 10.2 性能与 SLO

- 每类 TTFT：p50/p95/p99/max；
- layer deadline miss 次数和比例；
- L0、L1、L2+ I/O stall；
- 最大单请求 stall/queue wait，用于识别饥饿；
- makespan 和请求吞吐；
- 每请求 compute fraction；
- 真正的 fleet/makespan NPU utilization，不能与上一项混称；
- NPU 聚合带宽峰值、均值和 cap-hit 时间比例；
- SSU busy utilization、queue wait、Path 最大 outstanding；
- 各类别与请求的 Jain/公平性，但 Jain 不能替代 p99/max 饥饿检查。

### 10.3 新策略专用遥测

- 每 layer planning/refill 轮数；
- 256-count pressure-report 次数；
- `K`、`q`、`B_target` 分布；
- 候选不足 8 的 layer 数；
- `predicted_capacity_shortfall` 次数；
- `R_req` 变化轨迹或分桶；
- 被选 SSU/Path 分布与热点程度；
- 低水位触发到 top-up 的事件数，确认不存在同时间自旋。

历史 `+10pp 平均单请求 compute fraction、各类别 p95 不恶化、公平性不下降且无
饥饿` 可以继续作为性能目标，但它不是代码正确性的验收条件。实现即使没有性能
收益，也应如实保留失败结果，不能为通过门槛而修改输入或隐藏 p99/max。

---

## 11. 测试清单

### 11.1 纯函数单测

- `R_req` 随 `D_rem/T_rem` 正确升高或降低；
- L0 恒以 50 GB/s 为目标；
- `T_rem<=0` 记录 miss 且不除零；
- 候选大于等于 8 时 `K>=8`；
- 候选不足 8 时使用全部且不选无数据 SSU；
- 达到 `R_req` 后不无故继续扩大 K；
- 全部候选仍不足时标记 capacity shortfall；
- 平分使用稳定 hash/轮转，不总取最低 SSU ID；
- 仅使用合法类别 Path。

### 11.2 分批事件单测

- 初始 `q=1` 时每个所选 SSU 恰好提交一个 I/O；
- in-flight 高于低水位时不 refill；
- completion 跨过 `ceil(B/2)` 时恰好触发一次 refill；
- refill 是 top-up 到新 `B_target`，不是再叠加完整 batch；
- `top_up=0` 时不会在同一 timestamp 自旋；
- 两个 NPU 同时 ready 时仍能交错，后查询者能看到此前 enqueue；
- 未提交 block 不计入 SSU hardware pressure，但计入客户端 `D_rem`；
- layer 只有全部 block 完成后才 ready。

### 11.3 NPU 聚合带宽单测

- 单 active 40 GB/s 不缩放；
- `40+40` 缩放为 `25+25`；
- `40+5` 保持不变；
- `8×40` 缩放为每条 6.25；
- active 集合变化时旧 completion event 失效并正确重排；
- Baseline 和所有 QoS 策略都满足相同 50 GB/s 上限。

### 11.4 端到端不变量

- 多 seed、四类别、不同 pressure interval、不同 K/q 下全部完成；
- `backend_dispatches == completed block I/O 数`；
- 所有 SSU outstanding 最终为 0；
- busy+idle 与 makespan 守恒；
- `TTFT = queueing + processing`；
- placement、workload fingerprint 在配对策略间一致。

---

## 12. 当前代码中实现前必须处理或明确隔离的问题

1. **`required_bw` 被丢弃**：保留进 request/NPU 状态，避免输入字段失效。
2. **NPU 50 GB/s 未限速**：必须先作为所有策略共享的物理约束实现。
3. **利用率命名混淆**：现有 `avg_npu_utilization` 是平均每请求 compute fraction，
   不是 fleet/makespan utilization；新结果同时输出并明确命名两者。
4. **结果 cache 缺少输入/代码指纹**：新 schema/spec 至少包含所有新参数、数据
   hash 和实现版本；不能因忘记 bump 手工 schema 而复用旧结果。
5. **placement 缺少结构性配对证明**：当前请求 fingerprint 不包含实际
   block→SSU 映射。新实验应先生成不可变 placement artifact，保存 placement seed、
   算法版本和完整映射 hash，并在配对策略间 assert 相同。
6. **默认输出覆盖历史文件**：新实验使用独立输出目录/文件名，不覆盖旧 schema。
7. **输入校验不足**：校验 `ls_ratio∈[0,1]`、合法 token partition、正数 NPU/SSU/
   layer、合法 placement、有限非负带宽和合法 QoS 权重。
8. **有限 PIR 重复限速**：本实验保持 PIR uncapped，不顺带声称已经支持有限 PIR。
9. **完整 summary 非标准 JSON**：新专用统计使用字符串键/嵌套对象，避免 tuple key。

---

## 13. 非目标

本次不要实现：

- block 副本、EC、多位置选择或重新放置；
- 动态修改 SSD CIR/PIR/WRR；
- Path lease/owner/reservation；
- 中央协调器或全局最优 SSU 分配；
- queued-byte、实际带宽或剩余服务时间等新硬件遥测；
- doorbell/CPU/网络提交开销的批处理收益；
- 修改当前每 SSU 一个 active I/O 的后端模型；
- 针对单个 seed、类别或 SSU 点硬编码选路规则；
- 为达到目标而迁移已经入队的 I/O。

---

## 14. 建议的配置字段

字段名可调整，但结果 spec 必须记录等价信息：

```text
client_ssu_submission_policy = current | fixed_fanout | adaptive_fanout
min_ssu_fanout = 8
fixed_ssu_fanout = 8 | 16 | 32 | null
io_per_selected_ssu = 1 | 2 | 4 | 8
refill_low_watermark_fraction = 0.5
npu_bw_limit_gbps = 50.0
npu_bw_cap_policy = proportional_share
pressure_signal = full_256_active_plus_pending_io_count
pressure_latency_us = 0
client_submit_overhead_us = 0
```

建议为新方案增加独立 policy/strategy 名称，不要改变现有
`qos_static_cir` 的历史含义后仍复用旧 cache。

---

## 15. 交付要求

另一个 AI 的完成物至少应包括：

1. layer 级跨 SSU 自适应提交实现；
2. 所有策略共享的 NPU 50 GB/s 聚合限速；
3. 固定 K 与 adaptive K 实验入口；
4. 本文测试清单中的核心单测和端到端不变量测试；
5. 新 schema、独立结果路径和完整参数记录；
6. README 中对新旧 batch 语义的区分；
7. 对零延迟压力、I/O-count 估计误差、单 active SSU 和单 seed 偏差的明确说明；
8. 如实报告正结果、负结果和容量不足情况，不只报告平均值。

实现过程中若代码事实与本文冲突，应先报告冲突，不得自行扩大数据位置、硬件
遥测或动态 QoS 权限。本文“已确认、不得擅自改变”的部分优先级高于历史 lease
探索文档。
