# 合成带宽满足度实验

这个实验有意义，但它不是正式 128-NPU 结果的替代品。它的作用是把“输入本身不可能”、
“Scheme B 的 allocator/目标有问题”和“CIR、SSD、NPU 链路数据面没有兑现”拆开验证。

## 为什么不能只看 SSU 总带宽

总带宽小于 `SSU 数 × 40 GB/s` 并不代表请求一定可满足。固定 placement 下必须同时满足：

```text
对每块 SSU s：Σn D[n,s] <= 40 GB/s
对每个 NPU n：Σs D[n,s] <= 50 GB/s
```

例如两块 SSU 的总容量是 80 GB/s，`60/10 GB/s` 的 placement 总需求只有 70 GB/s，
但第一块盘已经超过 40 GB/s；任何只改 Path/CIR、不移动 placement 的策略都无法把第二块盘
的空闲 30 GB/s 借给第一块盘。相反，单 NPU 的 `[30,30] GB/s` 在两块盘上都没有打满，
但会受到该 NPU 的 50 GB/s 接收链路限制。

## 两级实验

### 1. 输入与 allocator audit

直接构造 NPU×SSU demand matrix。固定单层计算预算 `C`，每层在 `(n,s)` 上的 placement
数据量严格设为：

```text
W[n,s] = D[n,s] × C
```

这一层不运行事件仿真，检查：

- raw full-hide 是否满足每盘 SSD40 和每 NPU 50；
- 16 层 warm、2×TTFT SLO 的 fluid target 是否可行；跨请求 Layer0 只提前一个 compute
  interval 开始，过载时仍有残余 Layer0 I/O，因此重复模板所需服务比例为 `1/2`；
- Scheme B 的 demand-capped equal max-min grant 是否正确；
- `grant/demand` 是否已经预示某个 NPU 无法达到 2×SLO。

Baseline、Layer-once、Refresh8 只有选路逻辑，没有 grant allocator。这三者在 audit JSON 中
明确标为 `not_applicable:routing_policy_without_grant_allocator`，不会人为伪造 grant。

### 2. 六策略完整数据面

把同一矩阵转成每个 NPU 无限重复同一模板的有限饱和前缀，然后复用当前完整仿真器：

```text
placement → SSD40 单命令不可抢占仲裁 → NPU50 FCFS 链路 → layer barrier → compute
```

六个策略为：

- `baseline`；
- `layer_once`；
- `refresh8`；
- causal `scheme_b`；
- full-manifest、request/coflow normalized 的 `scheme_b_slo`。
- full-manifest、SLO-threshold admission 的 `scheme_b_admission`。

`scheme_b_slo` 先尝试让同一请求的所有 SSU flow 达到统一的 `1/2` 进度，
再向 full-hide demand 分配余量。它解决旧 Scheme B 按绝对 GB/s progressive fill 时
“小 flow 已经给满、同一请求的大 flow 仍低于 SLO 门槛”的错配；但在全体 `1/2`
target 本来就不可行时，它选择 normalized fairness，不等价于最大化 SLO pass 数。

`scheme_b_admission` 在全体 `0.52` 目标可行时先保障全部请求；只有目标集合不可行时，
才用稳定 greedy packing 选择一部分请求并 pin 到完成，同时为未选择请求保留 coflow
progress。它优化的是 processing-SLO 达标数量，不是严格的最大基数求解器，也不把
admission 前排队时间算入 TTFT。

每个正的 `(NPU,SSU)` flow 默认拆成 9 个 block。9 是有意选择的最小值：Layer-once
读取一次 pressure，而 Refresh8 会在第 9 个 I/O 前再次读取；少于 9 个 block 时两者可能
退化成同一种行为。

默认每个 NPU 预生成 32 个请求，warmup 后 settle 50 ms，再使用共同 1,000 ms 窗口。
该长度覆盖大多数默认 case，但不能先验保证任意策略的 tail-drain 都有足够 backlog；每个
run 仍须由 `all_npus_sampled_for_slo` 和 `no_backlog_exhaustion` 两个 invariant 验证。

## 合成 case

| Case | Demand | 用途 |
|---|---|---|
| Uniform ramp | 4 NPU×2 SSU，每个 flow=`L/4`，`L=5,20,39,40,41,60,80,84,90` | 精确跨过单盘 40 GB/s knee；NPU50 不干扰 |
| Heterogeneous raw feasible | `[[5,25],[15,5],[20,0],[0,10]]` | 两盘都恰为 40；若 Scheme B 不给满，属于 allocator 问题 |
| Deadline-feasible objective | 单盘 `[60,10,10]` | raw 80 不可行，但 warm 2×目标可行；Scheme B 给 `[20,10,10]`，大流可能失败，隔离策略目标问题 |
| Hotspot raw-infeasible | `[[20,0],[20,0],[20,0],[0,10]]` | fleet 70<80，但 SSU0=60；隔离 placement/input 问题 |
| Hotspot deadline-infeasible | 热点三条改成 30 | warm 2×目标在 SSU0 也超过 40，100% SLO 本来就不可能 |
| Single flow | 单盘 `60` 和 `90` | 验证单流超过 SSD40 的上限 |
| NPU50 control | 单 NPU、两盘 `[30,30]` | 两盘均有余量，只有 NPU50 是瓶颈 |
| NPU50 link stress | 单 NPU、八盘各 `10` | 盘均远低于40，使 NPU link 在完整数据面中接近50 GB/s |

Uniform ramp 的理论结果是：

```text
L <= 40：raw satisfaction = 1
L > 40：raw satisfaction 上限 = 40/L
16-layer warm 2×SLO fluid knee = 40 / (1/2) = 80 GB/s/SSU
```

因此 `L=80` 恰好是零裕量的 warm SLO 边界，离散命令、排队顺序和浮点误差都会使实际
达标率低于 100%；`L=84/90` 则在容量上不可能全部达标。若 Layer0 在 admission 前已经
完整读完，才会得到更宽松的 `15/31`，但这不符合当前只提前一个 compute interval 的实现。

### 纯 SSU capacity-knee profile

`--pure-ssu` 使用固定 32 NPU / 8 SSU、每盘 4 个 striped NPU，并把物理 NPU link 与
Scheme-B allocator 的 NPU cap 都提高到有限的 `1e6 GB/s`。这不是把 link 设为 100；
100 GB/s 仍有相当于 SSD40 的 40% 串行服务时间，不能视为“忽略 NPU 接收速度”。

主 sweep 为每盘 `32,36,39,40,41,42,44 GB/s`，另含每盘总计 41 GB/s、按
`40/30/20/10%` 分给 4 个 NPU 的 skewed case。正式结果中 48 个数据面窗口全部通过
invariant：

```text
L <= 40: SSD satisfaction ~= 1, SSD utilization ~= L/40
L = 41:  SSD satisfaction = 0.9756, SSD utilization = 1
L = 42:  SSD satisfaction = 0.9524, SSD utilization = 1
L = 44:  SSD satisfaction = 0.9091, SSD utilization = 1
```

六个策略在 uniform case 都落在相同物理曲线上。这证明受控输入下 SSU40 仲裁、动态
CIR 提交与稳态记账没有隐藏的容量损失；动态策略只能改善异构 flow 的分配和尾延迟，
不能提高单盘 40 GB/s 物理上限。

## 窗口指标

稳态窗口现在同时记录两张精确矩阵：

1. `measurement_npu_ssu_ssd_served_gbps`：SSD 后端在窗口内实际服务给每个
   `(NPU,SSU)` 的速率；
2. `measurement_npu_ssu_link_served_gbps`：这些数据在窗口内获得 NPU50 链路服务的速率。

两者都按服务区间与共同 `[start,end)` 窗口的重叠积分，包含跨越窗口边界的部分命令；
没有使用“只统计窗口内完成命令”的有偏方法。这里的 link service 包含 active command
已经传输但尚未整条完成的部分字节；这些字节只有在命令完成后才对 layer compute 可见，
端到端效果仍以 TTFT 和 layer barrier 为准。

主要报告：

- 每盘 offered、SSD served、utilization；
- SSD service satisfaction：`Σmin(ssd_service,D)/ΣD`；
- NPU-link service satisfaction：`Σmin(link_service,D)/ΣD`；
- 每个 active flow 的 mean、p10、min satisfaction；
- 每个 NPU 的 coflow satisfaction：其所有正 demand flow 的最小满足度；
- NPU compute utilization、TTFT processing SLO@2×、mean/p99 TTFT；
- pressure read、controller evaluation、CIR commit；
- 窗口开始/结束的 SSD outstanding blocks。

CIR 是仲裁保证，不是硬限速。SSU 未满时，一个 flow 可以拿到超过 CIR 的 surplus。因此
`actual == grant` 不是通用正确性条件；只有相关 Path 在整个窗口持续 backlogged、且该盘的
CIR 总和等于 40 时，才应期待长期实际份额接近 grant。

## 归因规则

按以下顺序判断：

1. 重建 `D=W/C`，确认六策略 input fingerprint 完全相同。
2. 若 raw D 可行但 Scheme B 的 `G != D`，是 allocator/control-plane 问题。
3. 若 G 正确，但 SSD service 明显低于可兑现份额，并且 Path 持续 backlogged，是
   CIR 提交、仲裁或测量问题。
4. 若 SSD service 正常、NPU-link service 受限，并且某 NPU 行和超过 50，是 NPU50
   链路瓶颈。
5. 若 delivered 符合当前 grant、warm 2×容量可行，但 SLO 仍差，重点检查 Scheme B
   的等权 max-min、deadline awareness 和 causal update，而不是增加 SSU。
6. 若某盘的 warm 2×需求本身超过 40，100% SLO 不可能；这是 input/placement 给出的
   上限，但仍应与可达到的最优 SLO 比较，量化策略的额外损失。

若某策略在严格 tail-drain 期间耗尽有限饱和前缀，runner 将该 run 记录为 `invalid`，保存
失败的 invariant 和 `completed_by_npu_at_stop`，但不填任何吞吐或 SLO 数值。默认热点
`60/10` 的 Layer-once case 中，32-request 前缀结束时完成数为 `[32,5,32,32]`，把前缀
增加到 128 和 512 后仍失败；这已是确定性的持续流饥饿证据，而不只是短窗口噪声。仍不能
把“等其他 NPU 停止后才完成”当作无限请求场景下的有效结果，具体选路/仲裁根因应单独诊断。
JSON 同时写入 `complete=false`、配对 input fingerprint、失败 invariants 和逐 NPU drain
diagnostics。默认 CLI 在写完产物后返回非零；只有显式传入 `--allow-invalid` 才返回成功，
避免自动化把不完整策略矩阵误当作完整实验。

## 运行

只跑全部数学 audit，速度很快：

```bash
python synthetic_bandwidth_experiment.py --mode audit
```

跑默认小型、可手算的完整六策略诊断：

```bash
python synthetic_bandwidth_experiment.py --mode both --allow-invalid
```

运行正式的纯 SSU 32×8 knee sweep（同时关闭物理和 allocator NPU 约束）：

```bash
python synthetic_bandwidth_experiment.py --pure-ssu --mode both
```

输出为：

```text
results/synthetic_bandwidth_satisfaction/pure_ssu_32npu_8ssu.json
results/synthetic_bandwidth_satisfaction/pure_ssu_32npu_8ssu.md
```

只跑 40 GB/s knee 附近的完整数据面：

```bash
python synthetic_bandwidth_experiment.py \
  --mode both \
  --uniform-only \
  --uniform-loads 5,20,39,40,41,60,80,84,90
```

对 128 NPU、指定 SSU 数先做纯矩阵 audit：

```bash
python synthetic_bandwidth_experiment.py \
  --mode audit \
  --uniform-only \
  --uniform-num-npu 128 \
  --uniform-num-ssu 70 \
  --uniform-layout dense
```

验证物理 NPU50 的配对反事实（只把 Baseline 的链路提高到100）：

```bash
python synthetic_bandwidth_experiment.py \
  --mode both \
  --case npu50_link_stress \
  --strategies baseline \
  --physical-npu-bw-gbps 100 \
  --output results/synthetic_bandwidth_satisfaction/npu_link100_counterfactual.json
```

这个参数只改变物理 NPU 接收链路；Scheme B allocator 的部署约束仍冻结为50 GB/s，避免
把硬件反事实误写成当前控制策略。

128×70 dense 的完整 DES 会产生大量 block 事件；应先用 audit 和默认小矩阵定位转折点，
再只选择必要的 `--uniform-loads` 做长跑。不要为了加速把 `--blocks-per-flow` 降到 8 以下，
否则 Refresh8 与 Layer-once 的比较会失去意义。

默认输出：

```text
results/synthetic_bandwidth_satisfaction/results.json
results/synthetic_bandwidth_satisfaction/results.md
```

小矩阵用于机制诊断，不直接宣称是 128-NPU trace 的性能结果。只有在诊断确认 allocator、
CIR、SSD 与 NPU link 都符合预期后，才把同样的负载点带回正式 128-NPU steady 实验。
