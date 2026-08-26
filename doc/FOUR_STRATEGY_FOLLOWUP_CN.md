# 四策略结果与四个后续问题

## 当前正式范围

本轮只比较用户最后指定的四个方案：

| 图例 | 实际含义 | 是否运行共享 SSD40→NPU50 数据面 |
|---|---|---:|
| Baseline (NPU-RR) | 每块 SSD 在活跃 NPU 源队列之间逐命令 RR | 是 |
| Static CIR before | Path=`12/4/12/4`，CIR=`20/4/12/4`，refresh8 | 是 |
| Static CIR after | Path=`12/4/12/4`，CIR=`20/6/8/6`，refresh8 | 是 |
| Ideal fluid bound | 删除跨 NPU 竞争并允许 fluid cut-through 的乐观上界 | 否 |

正式配置是 128 NPU、每个 NPU 一个请求、16 层、SSU=`40/56/80`、seed=`42/43`。每个 NPU 的 release delay 独立采样于 `[0,5 ms)`；同一 `(seed, SSU)` 下四个方案复用相同 workload、placement 和 release delay。三个可运行方案均使用 batch=1、每条 I/O 0.1 us 的有限发行敏感性模型。

0.1 us 只用来打破历史零耗时原子提交并检查因果关系，不是某款 NPU 的实测命令发行延迟。

主图 [01_strategy_overview_finite_issue.png](../results/four_strategy_concurrency/01_strategy_overview_finite_issue.png) 严格只有上述四条线。它画的是 `avg_request_compute_fraction`，即平均“每请求 NPU 计算占比”，不是 fleet 指标。

## 1. `-0.077 pp`、fleet 和 makespan 应该如何理解

### 两种利用率不是同一个指标

对请求 `i`，令 `C_i` 是 16 层计算时间，`W_i` 是实际暴露给计算流水线的 I/O stall：

```text
request compute fraction = mean_i(C_i / (C_i + W_i))
```

128 个请求等权；release 前的 0–5 ms 不进入单请求分母。它回答的是“一个典型请求从开始处理到完成期间，有多大比例在计算”。

```text
makespan = 从全局时刻 0 到最后一个请求完成

fleet compute utilization = sum_i(C_i) / (128 * makespan)
```

fleet 把 release 前、等待 I/O、以及较早完成 NPU 的尾部空闲都算进 128 台 NPU 的全局机器时间。配对策略的 `sum_i(C_i)` 完全相同，所以 fleet 与 makespan 单调反向。

### 旧的 `-0.077 pp` 不是“平均请求更慢”

`-0.077 pp` 来自历史 seed 42、atomic-8/零发行时间模型中，优化后 Static 相对 baseline 的三个 SSU fleet 差值均值。相同结果里，request 指标实际提高约 `+5.719 pp`。原因是许多请求变快，但决定全局尾部的一个 LL 请求变慢，使 makespan 略长。

在当前更严格的双 seed、batch1/0.1 us 正式结果中：

| SSU | Static after − baseline request | Static after − baseline fleet | Static after − baseline makespan |
|---:|---:|---:|---:|
| 40 | +7.943 pp | -0.043 pp | +6.741 ms |
| 56 | +5.594 pp | -0.033 pp | +4.966 ms |
| 80 | +1.076 pp | -0.008 pp | +1.272 ms |

跨三个 SSU 平均是 request `+4.871 pp`、fleet `-0.028 pp`、makespan `+4.326 ms`。因此旧的 `-0.077 pp` 不能当成固定结论；当前双 seed 后差距缩小到 `-0.028 pp`。

更重要的是 seed 方向：seed 42 的尾部 LL 请求在 Static after 下慢 `+19.029/+10.528/+2.550 ms`，seed 43 却快 `-5.547/-0.596/-0.006 ms`。目前可以确认 Static after 的平均请求指标在 6/6 个配对点都高于 baseline，但不能只用两个 seed 断言它的 fleet 必然更低。

详细的逐类输入和尾请求表位于 [四策略正式报告](../results/four_strategy_concurrency/report.md)。

## 2. 实时读取 Path 状态是否曾被提交原子性掩盖

### 历史模型确实存在一个 8-I/O 原子窗口

旧默认不是“一整个 NPU、请求或层都原子写入”，而是：

```text
一个 (request, layer, SSU) 提交状态
        ↓
同一个客户端事件内最多 enqueue 8 条 I/O，发行耗时为 0
        ↓
事件返回后，其他 NPU/SSD 事件才可能运行
```

所以其他 NPU 无法插进这 8 条 I/O 中间。refresh8 读取一次硬件 Path 压力，并用本地 shadow 把本批后续 7 条新分配计入；per-I/O 虽然调用 8 次读取，但中间没有其他 NPU 或 SSD completion 改变状态，看到的信息与 shadow 等价。这解释了历史结果中 refresh8 与 per-I/O 逐请求完全相同；它不能证明真实并发下实时读取没有价值。

### 当前有限发行模型如何允许插入

当前三个可运行方案统一使用：

```text
submit_batch_size = 1
issue_interval_us = 0.1
```

一个 NPU 发完一条后，下一条最早只能在 0.1 us 后提交；在这两个时刻之间，其他已到达 NPU、SSD completion、NPU link completion 和仲裁事件都能进入事件队列。同一时间戳有多个 ready NPU 时，提交器对 NPU 做 seeded shuffle，每个 NPU 本轮只发一条。因此不存在旧的 8 条跨 NPU 不可插入窗口。

注意：本轮两个 Static 方案仍都使用 refresh8，以隔离 CIR 差异。refresh8 会提前规划接下来 8 条 Path ID，因此它虽然逐条发行，却不会在第 2–8 条前重新读其他 NPU 的新状态。

### 40 SSU 的因果探针

在 seed 42、40 SSU 的独立敏感性探针中，Static profile 固定为优化前的 `20/4/12/4`：

| 提交模型 | 不读 Path 表 | refresh8 | per-I/O live |
|---|---:|---:|---:|
| atomic8、零发行时间 | 71.885594% | 81.952211% | 81.952211% |
| batch1、零发行时间 | — | 81.966372% | 81.963771% |
| batch1、0.1 us | 71.578645% | 80.797008% | 81.944874% |

在真正允许事件插入后，per-I/O 比 refresh8 高 `+1.147866 pp`；per-I/O 比完全不读状态高 `+10.366229 pp`。完全不读状态的实现只在同一类别合法 Path 池中按稳定起点轮转，Path-table 读取数为 0。

这组结果支持两个因果判断：

1. 历史 refresh8=per-I/O 很大程度上是客户端原子窗口造成的信息等价，不是实时状态本身无价值。
2. 不读取 Path 压力会显著损失当前负载下的选路质量；本地 shadow 只能反映本 NPU 已规划内容，无法替代其他 NPU 和服务完成带来的新状态。

该探针只有一个 seed、一个 SSU 点，而且 0.1 us 未做硬件标定，所以它是机制证据，不应作为三 SSU 的正式收益曲线。用户最后要求主测试只保留四方案，因此它没有加入主图。

## 3. Ideal 是否真的让所有 NPU 无竞争

`fluid_no_inter_npu_contention_upper_bound` 在数学上删除了所有跨 NPU 竞争，但它不是一个可运行调度器。

对请求 `i` 的层 `l`，令 `B(i,l,d)` 是落在 SSD `d` 上的 GB，`B(i,l)` 是本层总 GB。它使用：

```text
L(i,l) = max(max_d(B(i,l,d) / 40), B(i,l) / 50)
```

然后按一层前预取递推每层可见的 I/O stall。每个请求都在一个独立平行世界里独占全部 SSD 的名义 40 GB/s 和自己的 50 GB/s link；同一块物理 SSD 的容量被乐观复制给 128 个请求。它还删除了命令边界、Path/CIR、SSD 队列、NPU FCFS 离散 incast，并允许 SSD 与 NPU 工作量 fluid cut-through 完美重叠。

因此它能报告逐请求上界及其平均值，但没有原物理系统的一条联合时间线，不能合法报告 makespan 或 fleet。当前两个 seed 的 bound 分别为 `91.231809%` 和 `92.451995%`，均值 `91.841902%`。

完整定义、递推公式、上界证明和限制见 [IDEAL_NO_CONTENTION_BOUND_CN.md](IDEAL_NO_CONTENTION_BOUND_CN.md)。

## 4. 为什么 SSU 有效利用率只有约 4%–8%，50 GB/s 又有什么关系

这里的 SSU effective utilization 是覆盖全局 makespan 的容量积分：

```text
U_ssu = total_read_GB / (num_ssu * 40 GB/s * makespan_s)
```

当前双 seed baseline 均值为：

| SSU | SSU effective | 全局 NPU-link utilization | 平均 NPU link queue wait |
|---:|---:|---:|---:|
| 40 | 7.995% | 1.999% | 0.954 ms/I/O |
| 56 | 5.727% | 2.005% | 1.263 ms/I/O |
| 80 | 4.013% | 2.007% | 1.533 ms/I/O |

这不表示获胜 I/O 只获得 4%–8% 的盘带宽。所有可运行策略中：

- 每块 SSD 最多一条 active 命令；获胜后始终按 `size / 40 GB/s` 非抢占服务。
- 总读取量在同一 seed 的 40/56/80 SSU 间基本固定，盘数增加只是把相同工作摊到更多已安装容量。
- 约 2.2 s makespan 的大部分时间由 16 层计算、逐层依赖和长 LL 请求占据；SSD 有工作时跑满 40 GB/s，但在整个全局窗口内大部分时间没有可运行命令。

50 GB/s 是每个 NPU 独立的接收上限，不是 128 个 NPU 共享的 50 GB/s。全局 link 利用率约 2% 仍不排除局部 incast：多块 SSD 若在相近时刻向同一个 NPU 完成数据，只有一条能进入其 50 GB/s FCFS 服务，其他 I/O 会排队。随着 SSU 从 40 增到 80，baseline 的平均 link queue wait 从 0.954 ms 增到 1.533 ms，正是“全局平均很低、局部瞬时有拥塞”。

当前模型在 SSD 与 NPU link 之间使用无限 store-and-forward buffer；SSD 完成后立即释放槽位，NPU link 拥塞不会反压占住 SSD。因此 50 GB/s 不直接进入 SSU effective 公式，但会通过层 ready 时间和后续 I/O release 间接影响 makespan。

## 5. 当前 CIR/Path 是否合理，怎样继续改进

### 优化后的固定 CIR 明确优于优化前

Path 数不变时，`20/6/8/6` 相对 `20/4/12/4` 在 40/56/80 SSU 的 request 指标分别提高 `+0.074/+0.609/+0.565 pp`，fleet 也分别提高 `+0.043/+0.032/+0.017 pp`，makespan 分别缩短 `6.367/4.712/2.535 ms`。六个 `(seed, SSU)` request 配对点方向全部为正。

CIR 调整把 4 GB/s 从 LS 转给 SL 和 LL：

- SL 在三个 SSU 点提高 `+2.645/+3.338/+0.700 pp`；
- LL 提高 `+0.974/+0.888/+0.395 pp`；
- LS 回退 `-0.994/-1.786/-1.949 pp`；
- SS 的 CIR 未变，但会因活跃 Path 集合和命令顺序产生间接变化。

所以优化前确实低估了 SL/LL 的保护需求；优化后总体更好，但并非所有类别都更快，也不是全局最优证明。

### Path=`12/4/12/4` 本轮没有重新优化

本轮为了只识别 CIR 因果，两套 Static 保持相同 Path 数。Path 数决定可分散队列和活跃 Path 竞争的粒度，CIR 决定保证服务机会；两者可能交互。下一步若允许扩大范围，合理做法是：

1. 先固定多个训练 seed，联合搜索离散 Path tickets 与总和为 40 的类别 CIR。
2. 目标函数同时包含 request 均值和 LL 尾部/makespan 约束，避免只靠 SS 巨大收益掩盖 LL 尾部。
3. 用未参与选择的 seed 验证，不能在同一个 seed 上选参数并宣称泛化。
4. 若硬件允许更细粒度所有权，按 NPU 当前需求分配 Path/CIR 比四个静态类别更接近用户希望的 10/30 GB/s。

### 动态 Path+CIR 的已有结论

在用户缩小到四策略之前，项目已测试 NPU 同时决定 owned Path 和 desired CIR 的动态方案。两个 seed 中最好的动态方案都是 `joint_slack_path_cir`：它在 80 SSU 相对最佳固定 Static 分别提高 `+0.563/+0.437 pp`，但在 40 SSU 分别回退 `-1.647/-0.725 pp`；跨两个 seed、三个 SSU 的平均 request 指标仍略低于最佳固定 Static。

所以“同时选路+CIR”在容量充足时有收益，但当前零成本、逐命令重配的实现尚未形成总体更优策略。本轮按最新要求没有把它放进正式曲线。

### 增加层数或 batch 会怎样

- 增加层数会增加稳态预取阶段的占比，并稀释 L0 启动偶然性；但长计算也可能覆盖更多 I/O，结果不保证让 Static 相对 baseline 的差异变大。
- 增加每个 NPU 的请求 batch 或持续到达负载，会让 SSD 队列更长期保持 backlog，更接近 CIR/WFQ 的稳态带宽分配场景，也更容易显现 10/30 相对 20/20 的价值。
- 当前合同固定为每 NPU 一个请求、16 层，因此上述是机制推断，不是本轮实测结论。若要验证，应保持总字节或到达率可比，分别扫描层数与并发 batch，不能同时改变后再归因。

## 结果入口

- [四策略正式报告](../results/four_strategy_concurrency/report.md)
- [四策略主图](../results/four_strategy_concurrency/01_strategy_overview_finite_issue.png)
- [四策略聚合 JSON](../results/four_strategy_concurrency/analysis.json)
- [理想无竞争上界详解](IDEAL_NO_CONTENTION_BOUND_CN.md)
