**这是先行保存的 Baseline 结果。全部 Baseline / Once 长测现已完成，见 [完整报告](report.md)。**

| 统计窗口 | 随机 Baseline | Ordered Baseline |
|---|---:|---:|
| [2, 60) 秒 | 99.9979% | 89.9647% |
| [2, 4) 秒 | 99.9805% | 89.8044% |
| [50, 60) 秒 | 99.9994% | 89.8629% |

同一批请求只改变卡内顺序，主窗相差 **10.0331 个百分点**。Ordered 的 29 个连续 2 秒窗口为 **89.6098%–90.8979%**。这次低值持续到了第 60 秒，没有在后半段恢复至接近 100%。这仍是有限仿真，不是无限期稳定性的证明。

[Ordered Baseline 长时间线](figures/separate/ordered_baseline_long.png) · [局部时间线](figures/separate/ordered_baseline_detail.png) · [Random Baseline 长时间线](figures/separate/random_baseline_long.png)

32 NPU、6 SSU、每盘 40 GiB/s。总输入全部来自原始 data，快短总长 32K/48K/64K，长请求总长 176K；每请求 8 层，没有缩放计算或读取量。NQL=1024 是新计算/miss 数，不是总输入。

- NPU 0–15：循环执行 **6 个长请求 → 1 个 64K 快短 → 1 个 32K/NQL4096 辅助请求**。
- NPU 16–31：循环执行 **46 个 32K 快短 → 1 个 48K 快短 → 1 个 64K 快短 → 1 个长请求**；每卡从 49 项环的不同位置开始。
- 辅助请求是真实 data 行，单层计算 28.593ms，图中单独用紫色表示；它不计入快短。它为随后长请求的首层预取提供了时间。

每卡都有超过 62 秒的纯计算输入，主窗内全部 32 卡有任务，利用率下降来自等待。每卡在 `[2,60)` 内都实际计算了快短和长请求；最低快短占自身计算 **5.3987%**，最低长请求占 **7.8699%**。这里保证的是整个长窗混合；各 2 秒小窗只有 24–27 张卡同时包含快短和长请求，不能说每 2 秒每卡都混合。

按原约定的“当前请求逐盘读取量/单层计算时间”口径，全程最热盘峰值：Ordered **36.011188**、Random **36.428824 GiB/s**，均低于 40。补充核查把跨请求预取改成真实下一请求的读取量：Ordered 在 `[2,60)` 的峰值 **39.676099 < 40**；Random 为 **44.344865 > 40**，因此不能声称 Random 也通过了这个新增口径。所有实际 I/O 都完整执行。

接纳后 SLO×1.5 达标率：Ordered **99.9492%**，Random **100.0000%**。这是“完成−接纳”不超过纯计算时间 1.5 倍，排除了接纳前的排队，不是端到端 TTFT。

[Ordered 原始结果审计](audit_existing/long_main_ordered_baseline_validation.json) · [Random 原始结果审计](audit_existing/long_main_random_baseline_validation.json) · [长读取同步情况](audit_existing/long_main_ordered_baseline_phase.json)

[第51秒附近的局部预取与等待](figures/separate/ordered_baseline_late_handoff.png)：快短计算预算7.257ms，读取生命周期15.643ms，产生8.386ms真实等待；同窗辅助请求遮住了下一长请求的首层读取。读取生命周期包含排队和传输，不等于磁盘独占服务时间。
