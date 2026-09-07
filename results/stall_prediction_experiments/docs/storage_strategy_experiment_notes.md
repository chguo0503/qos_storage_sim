# 单 SSU 存储策略实验记录

## 结论边界

本轮实验确认：单 Path0 的队头阻塞可以很严重，隔离 Path 可以消除相当多损失；但“为每张卡设置 `CIR = 每层数据量 / 每层计算时间`”不是对任意变化输入都优于 once-per-layer 的策略。固定输入得到接近 100% 的结果，不能外推到逐请求变化的输入。

尤其要区分三种输入来源：

1. `mechanism_synthetic_not_data_calibrated`：用于隔离机制的自选 K、C。默认 NQL=128，计算时间不一定与 GLM 表一致，所以路由类别与计算需求的组合也不一定真实。
2. 校准构造：计算时间来自现有 data 的 NQL 插值或 1K 序列外推，来源写在每个 Profile 的 `provenance`；这是模型假设，不是实测硬件数据。
3. `direct_data_*`：K、C、NQL、类别都按 data 的原始行生成；没有改变计算时间，所有 I/O 恰好是完整 176 KiB。

## 不变的硬件和输入契约

- 4 NPU 独立请求、1 SSU、每请求 8 层、跨请求 Layer0 预取开启。
- 每个 I/O 是 1 个 GLM KV block，即 128 token 的单层 KV，共 176 KiB。
- SSD 服务带宽 40 GiB/s，每卡独立接收链路 50 GiB/s。
- 保留核心 `sim.py`、`continuous_batch_sim.py`、`policy_logic.py`；新策略只调用既有动态 CIR 接口。
- 256 条可用 Path，专用路径使用 0、32、64、96；PIR 一律无限，未假装实现有限 PIR。
- 对比组使用相同 request ID、到达时间、计算时间、KV 量和固定 NPU 分配；`input_fingerprint` 核验相等。
- 调度决策只消费已到达请求的 `CIRControlSnapshot`。未访问 FIFO 顺序、后端活动 I/O 剩余量或未来到达。

## 存储策略 A 的初版公式与消融

令第 i 张卡当前需要隐藏的整层读取量为 V_i，计算预算为 C_i，则规划需求为：

\[
d_i=V_i/C_i.
\]

`dedicated_demand` 使用共同数学文件中的 `required_rate_gib_s`。如果总需求超过 40，则先按比例分配：

\[
r_i=d_i\min(1,40/\sum_jd_j).
\]

这是服务速率规划，不是当前虚拟完成标签仲裁器上的 deadline 保证。CIR 重写保留历史 virtual-finish 状态；队列空闲时还会重新分配剩余带宽。尤其在请求切换时，下一请求 Layer0 的正确时间预算来自前一请求最后一层，而不是下一请求自己的计算时间。初版以同 NPU 的 current/prefetch manifest 最大需求处理切换，可能过度或不足分配。

该实现是项目既有 dedicated-Path Scheme B 思路的简化、可审计实验变体，不应表述为从零发明新 SSD 调度器。

消融 `dedicated_equal` 把每张卡设为 10 GiB/s，以区分“Path 隔离”与“按计算预算分配 CIR”的贡献。

## 固定流结果：能解释机制，但不是逐请求变化的证明

下表为各策略完成规定 warmup、settle 后的中间 1 秒平均 NPU 利用率。不同策略的窗口起点可能不同；`two_short_098` 另做同一绝对时间 1000–2000 ms 的完整请求集检查，结论保持。

| 输入 | 来源 | Baseline | Once/layer | 等 CIR | 按需求 CIR |
|---|---|---:|---:|---:|---:|
| rounded_old | 旧例计算插值/外推，I/O 整块重构 | 75.9973% | 99.9918% | 未测 | 99.9989% |
| two_short_098 | 纯机制，非 GLM 校准 | 76.3319% | 96.1204% | 96.1238% | 100% |
| tiny_victims_098 | 纯机制，非 GLM 校准 | 56.4596% | 96.9768% | 96.9764% | 100% |
| calibrated_two_ls | NQL 插值 + 1K 序列外推 | 65.1259% | 98.9899% | 99.9974% | 100% |
| direct_data | 原始 data 四行 | 91.0248% | 100% | 未测 | 100% |
| raw_sticky | 原始 data 四行 | 91.0877% | 97.8735% | 未测 | 100% |

`raw_sticky` 的四卡配置为 `(32K,512)、(32K,512)、(192K,512)、(192K,4096)`，名义总需求 39.7234035 GiB/s。Baseline 没有降到 85% 以下，因此没有继续对此输入进行大量参数搜索。

机制例 `two_short_098` 的同一绝对 1000–2000 ms 窗口分别为 76.1972%、96.1238%、100%。整批 295 个相同请求的平均到达延迟分别为 2228.27、1403.49、1373.57 ms，但按需求 CIR 的 makespan 为 3959.88 ms，略差于 once 的 3951.50 ms。这说明平均利用率、平均延迟和最后一个请求完成时间不是同一个目标。

## 逐请求变化的 raw-data 输入

`build_variable_requests` 的两个池均只使用原始 data：

| 池 | 原始 `(总长 Ki-token, NQL)` | 每卡完整循环中的配额 |
|---|---|---|
| long | (192,512)、(192,1024)、(192,2048) | 17、14、1 |
| broad | (32,128)、(32,256)、(64,512)、(96,512)、(192,512)、(192,1024) | 8、4、4、4、8、16 |

每张卡分别打乱配额循环；新的请求可以有不同长度、NQL、计算时间和读取量。选择这些配额是为了构造接近容量的实验，不意味着这些频率来自真实线上流量统计。

混合输入的名义需求必须按计算时间加权，而不是平均单请求 `V/C`：

\[
d_i^{\rm mix}=\frac{\sum_{r\to i}8V_r}{\sum_{r\to i}8C_r},
\qquad \rho=\frac{\sum_i d_i^{\rm mix}}{40}.
\]

循环理论值和有限抽样值都记录在 JSON。随机截取部分循环会改变有限样本需求；例如 long/20260907 有 40.649996 GiB/s，因此只能作为略微过载对照，不能算“名义需求不超过 40”的成功案例。

每卡在开始时显式有 6 个已到达排队请求，随后按完整请求读取字节数，以 39.2 GiB/s 的节奏产生新到达。这些初始 backlog 是额外启动突发：若把它们也除以整个到达跨度，输入速率会高于 39.2。JSON 同时保存启动字节量、后续节奏、有限到达跨度口径，不能混为一谈。

所有策略完整跑完同一组请求，再裁剪共同绝对 `[1000,2000] ms` 窗口。逐卡 active 时间核验为整整 1000 ms，不通过挑选不同窗口或截掉空闲来抬高利用率；同时保留完整请求集的到达延迟与 makespan。

### 动态输入结果（存储策略 A 初版）

| 输入 | 有限名义需求 GiB/s | Baseline | Once/layer | 等 CIR | 按需求 CIR |
|---|---:|---:|---:|---:|---:|
| long seed20260906、逐个到达 | 38.0058 | 95.6846% | 95.7851% | 94.8778% | 95.1981% |
| long seed20260910、4 请求一批 | 39.7713 | 94.4660% | 94.4590% | 未测 | 93.8787% |
| broad seed20260906、逐个到达 | 38.5680 | 93.5645% | 92.9023% | 95.6763% | 95.4881% |
| broad seed20260908、逐个到达 | 39.6923 | 90.9392% | 91.2930% | 未测 | 90.7967% |
| long seed20260907、4 请求一批（略过载） | 40.6500 | 93.5822% | 93.5813% | 未测 | 93.9252% |

这组反例很重要：按需求 CIR 在两个 long 样本和 broad 留出样本上均不如 once；broad 首个样本虽好于 once，又略差于等 CIR。仅根据固定流的漂亮结果，不能声称初版 A 已完成优化目标。原始 data 的动态输入中，本轮也没有找到 Baseline 低于 80% 的例子。

## 使用方式

在项目根目录运行：

```bash
python3 -m unittest test_stall_policy_experiments -v
python3 run_stall_policy_experiments.py --case calibrated_two_ls
python3 run_stall_policy_experiments.py --case variable_raw --mix broad --seed 20260906 --strategies baseline layer_once dedicated_equal dedicated_demand
```

函数接口：

```python
requests, pool, metadata = build_variable_requests(seed=20260906, mix="broad")
result = run_case(pool, "dedicated_demand", requests=requests, complete_all=True)
metrics = summarize_complete_run(result, start_ms=1000, end_ms=2000)
```

每份结果记录实际请求指纹、源码 hash、控制决策、仿真 invariants、逐请求与逐层明细。没有修改旧低利用率目录中的数据。

## 本轮审计中发现的问题

- 纯机制样本的 C/NQL 组合不能默认为 GLM 真实组合；已明确标记来源，并追加 raw-data 变化输入。
- 初始积压和后续 arrival bandwidth 是两个口径，已同时披露。
- 只改变随机种子不一定改变结果：固定流 seed42 与 seed7 相同，因此另做真实初始相位扰动和变化输入。
- 远端系统 Python 没有 numpy；复用了已有虚拟环境 `/home/chguo/qos-search.EHBmAM/venv/bin/python3`，没有安装系统软件或修改旧实验目录。
- 当前 CIR 回调虽然收到公开队列计数，初版公式没有使用这些计数。不能把它包装成已经精确感知 deadline 的队列预测器。
