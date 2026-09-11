# 每 NPU 带宽诉求、真实服务与 stall：只读方法审查

本审查没有启动仿真，也没有修改模拟器。以下定位对应当前正式长实验冻结源码；`sim.py` SHA256 为 `5f3a9bb82c5740a86c9c916e01495326a0bf4363fb6a5312cb99ac747e8600fd`，`continuous_batch_sim.py` 为 `c652bfb22fb6c3d219d7b8f78e172ca5808c55c5e929547ece5da0680129f9b1`。

## 1. 可以精确记录哪些物理服务

一个块有以下时间：进入 SSD 队列 (e_b)、SSD 激活 (a_b)、SSD 完成并进入链路队列 (z_b)、NPU 链路开始 (l_b)、HBM 就绪 (h_b)。

| 阶段 | 源码字段 | 含义 |
|---|---|---|
| SSD 排队 | `enqueue_time` → `ssd_activation_time` | 尚未得到 SSD 物理读服务 |
| SSD 服务 | `ssd_activation_time` → `link_enqueue_time` | 单个 SSD 正在读取该块 |
| 链路排队 | `link_enqueue_time` → `link_start_time` | 已读出，等待此 NPU 的串行链路 |
| NPU 链路服务 | `link_start_time` → `link_end_time` | 该块向 NPU/HBM 传输 |

定位：`sim.py:514` 定义所有字段，`:1005` 激活并记录不可变 `ssd_activation_time`，`:1113` 分派后以整盘带宽服务一个不可抢占块，`:1167` 完成；`continuous_batch_sim.py:3977` 核算链路、`:3997` 启动链路、`:4017` 接收 SSD 完成、`:4060` 处理 SSD 完成、`:4079` 处理 HBM 完成。

**不能用 `flow.start_time` 当 SSD 起点**：`sim.py:982` 的 `settle` 会改写它。当前冻结源码已经有 `ssd_activation_time`，旧版本“该字段不存在”的历史说明不适用。

本实验每 SSD 每次只服务一个块，服务时速率 40 GiB/s；每 NPU 的链路每次只服务一个块，速率 50 GiB/s。CIR/路径仲裁改变谁先获得服务，不把单块的物理服务速率改成 CIR。以代码中实际二进制容量换算，单位应写 GiB/s、MiB；某些源码变量的 `gb/gbps` 名称不能直接当十进制单位。

## 2. 推荐同时展示两种“实际提供”

令块 (b) 的实际读取量为 (v_b) GiB。对于 NPU (n)、SSD (s)：

\[
B^{SSD}_{n,s}(t)=40\sum_{b:n_b=n,s_b=s}\mathbf1_{[a_b,z_b)}(t),
\qquad B^{SSD}_{n}(t)=\sum_s B^{SSD}_{n,s}(t).
\]

它表示 SSD 当时把多少物理读服务分配给该 NPU。单盘对全 NPU 求和不超过 40；单 NPU 跨六盘求和可以短暂超过 50，理论上最多 240，因为六盘可并行读出并排入链路队列。这不是链路超容量。

\[
B^{HBM}_{n}(t)=50\sum_{b:n_b=n}\mathbf1_{[l_b,h_b)}(t).
\]

它表示实际进入该 NPU 的链路服务，始终不超过 50。累计 HBM 完成字节的阶梯曲线也可用于核对层是否全部就绪，但不应把“完成瞬间把整块计入”的阶梯差分误当均匀物理服务。

图上建议给同一 NPU 的两条带宽线，名称分别写“SSD 读服务”和“进入 NPU 的带宽”。如采用 1 ms 或更宽分箱，必须用服务区间与箱子的精确交集计算字节再除箱宽，不能按块完成时刻把整块丢入一个箱。标明分箱宽度；瞬时单块图和平均带宽图用途不同。

## 3. 带宽诉求的预算、截止时刻与 stall

对一次内部层预取 (j\to j+1)，令 (r) 为第 (j) 层计算开始，(d) 为其计算结束，(C_j=d-r)，下一层实际落在 SSD (s) 的读取量为 (V_{j+1,s})。预算速率是：

\[
B^{budget}_{n,s}=\frac{1000V_{j+1,s}}{C_j},\qquad t\in[r,d).
\]

时间使用 ms，速率为 GiB/s。该量表达“希望在这段计算时间内准备好下一层”的平均预算，不是实时排队量、实时 SSD 发命令速率，也不是设备保证。

定位：`continuous_batch_sim.py:3269` 在计算开始调用下一层预取，截止时间就是本层计算结束（`:3304`）；具体块分批提交在 `:3002`，同刻 NPU 提交顺序由固定 `submit_order_seed` 驱动的 RNG 决定（`:3049`）。层级 release 不等于每块的实际 enqueue。

跨请求 L0 同样使用上一请求最后一层计算作为可隐藏预算，但分子必须换成**下一请求 L0 的实际读取量**，分母必须是**上一请求最后一层 C**。用目标请求自己的 C 会在短/桥接/长切换处错配。定位 `:3227` 和 `:3321`。第一条请求 L0 没有前一层计算预算，应独列；不能制造一个 C 长度的前置区间。

令该目标层最后一个块的 HBM 就绪时刻为

\[
H=\max_{b\in\text{目标层}} h_b.
\]

本次 fixed、batch size 1、控制开销为 0 的内部层有：

\[
\text{下一层 compute start}=\max(d,H),\qquad
\text{stall}=\max(0,H-d).
\]

源码 `:2635` 要求所有块就绪，`:2651` 记录 ready，`:3293`–`:3298` 记录计算与暴露等待，`:4106`–`:4109` 在最后一个 HBM 完成块处标记层就绪。初始 L0 与接纳前等待须分开，不套内部层公式。

不要把 (B^{budget}) 延长到整个 stall 再把面积叫“实际需求字节”：延期面积会重复增加同一批已固定的字节量。图上宜让预算线在 (d) 结束，标截止竖线，并用背景阴影标 ([d,H))。若附“逾期预算线”，必须标为解释用代理且不用于字节守恒。

## 4. 为什么 stall 时实际带宽仍可能很高

stall 表示截止时刻尚未拿齐**正确的目标层**，不是要求此刻服务速率为 0。先排队、后集中获得高带宽，仍会晚于截止；一个短尾块未到即可阻止整层计算。SSD 六盘合计服务很多，也可能只有某一盘的最后一块迟到。SSD 已全部读完时，NPU 还可能等链路队列或链路服务。

建议为具体层附累计服务检查：截至 (d)，目标层在每盘的 SSD 已服务字节、HBM 已收到字节分别是多少，并对照目标量 (V_s)。用 ((request\_id,layer,block\_idx,ssu\_id)) 归属，而非用“当时正在计算的请求 ID”归属：跨请求 L0 本来可能属于尚未接纳的下一请求。

找到使 (H) 最大的真实最后块，展示其 (e_b,a_b,z_b,l_b,h_b)。总块排队时间相加不等于 NPU 暴露等待，因为多个块和多盘同时在等/服务。若要断言具体谁阻塞了该块，还需核该 SSD 的实际前驱服务片段；层 IO 生命周期本身不能证明 FIFO 前驱因果。

## 5. 可复用 observer 与短前缀重放要求

现成参考：`results/baseline_32npu6ssu_underload/short_ge10k_validation_20260911/trace_ordered.py`，以及更早的 `trace_ordered_zoom.py`。前者在调用 `run_case` 之前，用 `unittest.mock.patch` 包住 `continuous_batch_sim._register_complete`；只复制完成 flow 字段、原函数严格调用一次，不改流对象、事件、队列或 RNG。

Once 的 `shared_path_sim_adapter.py:148` 在进入上下文时捕获 `original_complete`，`:219` 原回调后更新确认账，因此 observer 必须包在 adapter 外层/进入之前（`:323`–`:327`），不能替换掉它的确认账逻辑。物理服务口径对 Baseline 与 Once 一致。

以下是复核通过前必须明确的范围条件：

1. 完整 manifest、原策略参数与提交随机种子保持不变。`run_baseline_npu32_stress.py:289` 的 `windows` 只控制事后统计，输入 metadata 的 horizon 也不是运行停止参数。默认事件循环跑完整批次（`continuous_batch_sim.py:6691`）；若仅诊断到 4.2 s，须有独立明确的进程内停止实现及停止记录，不能把局部中止输出当完整仿真成功。
2. 只收集最终 HBM 完成回调时，运行到 4.2 s **不能先验保证**已捕获所有在 ([3.2,4.0)) 有 SSD 服务的块：个别块可能仍在链路队列。须检查停止上下文的 active/pending flows，证明其中没有与观察窗重叠的遗漏；否则增加 SSD 完成钩子 `_enqueue_link_io`、链路启动钩子，或继续到观察对象完成。不能把尚未完成的块静默当 0。
3. 捕获必须含 carry-in/carry-out。旧 observer 的 `enqueue < right and link_end >= left` 是包容性过滤，之后各物理阶段仍要按自己的服务区间裁剪；不能只保留 enqueue 或 completion 位于窗内的块。
4. 到 4.2 s 的重放不要求整个 60 s summary 相等。应独立逐字段对齐原完整结果中截止观察终点的 layer release、ready、compute start/end、admission/completion 前缀；跨终点的字段仅比较已发生值及已确定的 compute duration。特别同时核 workload fingerprint、strategy、assignment、5 ms collector、submit seed、全部 core/policy SHA，排除额外 observer wall-time 字段。
5. 对捕获区间核 SSD 单盘服务不重叠、NPU 链路服务不重叠；每个完整块 SSD 时长 (1000v_b/40)、链路时长 (1000v_b/50)；阶段先后顺序、ID/placement/字节量、分箱字节守恒、层最后 HBM 时刻与原 ready 精确相符。对未完整捕获的边界层明确标 partial。

本方法审查没有推断新 trace 的实际数值，也不把已有 60 s 层摘要当作逐块物理服务记录。只有被动重放和前缀等价审核通过后，新增图才能称正式冻结运行的局部物理诊断。
