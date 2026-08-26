# QoS + SSD 离散事件仿真

本项目比较 Baseline、静态 QoS Path 选路、需求感知仲裁和理想化协调策略。
正式实验固定为 **128 NPU、16 层、40/56/80 SSU**；同一 SSU 点的所有
策略共享完全相同的 workload、block→SSU placement、release delay 和 seed，
并使用同一套 SSD→NPU 数据面。

## 正式实验参数

- 128 个 NPU，每个 NPU 一个请求，每个请求 16 层。
- SSU 数分别为 40、56 和 80，每块 SSU 的服务能力为 40 GB/s。
- 每个 NPU 有独立的 50 GB/s 接收链路，不与其他 NPU 共享该上限。
- 每个 NPU 的 release time 由独立 seed 从 `Uniform[0, 5 ms)` 抽样。
  release jitter 决定第 0 层何时开始提交，不会伪装成 SSD/NPU 服务时间；
  配对策略使用相同的 128 个 delay。
- `LS_RATIO=0.5`，默认 workload seed 为 42；workload、placement、提交顺序
  和 release delay 使用相互独立的 seed。

小规模 `--quick` 只用于检查运行器和绘图管线，不是正式性能结果。

## 共享数据面

每条 I/O 的完整路径为：

```text
策略仲裁选中一条 I/O
        ↓
SSD 单命令、不可抢占服务：t = size / 40 GB/s
        ↓ SSD 完成并释放该 SSU 服务位
目标 NPU 的独立 FCFS 接收队列：t = size / 50 GB/s
        ↓
block 才对计算流水线可见
```

- 每块 SSD 最多一条 active I/O；命令获胜后以 40 GB/s 服务，不是
  获得一段可与其他命令并行分享的 fluid 带宽。
- 每个 NPU 接收队列最多一条 active I/O，同一 NPU 来自多块 SSD 的
  数据会在这里 FCFS 排队，因此 incast 会产生真实的 link queue wait。
- SSD 完成后立即释放服务位。模型假设 SSD 与 NPU 之间有无限
  store-and-forward buffer，所以 NPU 拥塞不会反向占住 SSD。
- Baseline 与其他可实现策略的差异只在队列和仲裁顺序。SSD/NPU
  服务时间、block 完成条件和 placement 都不变。

## 策略

下面反引号中的首个名称是 `analysis_experiment.py --select` 使用的 formal
strategy ID；括号内是仿真器 summary 中的 policy/name。

- `baseline`（policy=`baseline_bypass`）：绕过 QoS Path；每块 SSD 对有积压的 NPU 源队列
  逐 I/O round-robin。
- `current_layer_snapshot`、`current_refresh8`、`current_per_io`、静态调优候选
  （policy=`qos_static_cir`）：客户端选择 Path，SSD 内 Path FCFS，并按 CIR、
  组间/组内 WRR 和最终 RR 进行命令级仲裁。正式矩阵包含当前静态
  配置、固定候选 CIR/Path 布局以及不同遥测/提交粒度。
- 需求比例 Path ticket：仍使用 `qos_static_cir` 硬件，但将 256 个
  等 CIR Path 按每个 NPU 的 capped demand 静态分配，且每个 NPU 至少一个
  Path。它用离散 Path 数近似 10/30 之类的需求比，不会运行时改寄存器。
- `demand_maxmin`（policy=`qos_demand_maxmin`）：每块 SSD 使用 capped per-NPU demand 计算工作保守的
  max-min 目标，再以 packetized virtual-finish 顺序执行单命令。当两个 NPU
  的需求为 10/30 GB/s 且共享一块 40 GB/s SSD 时，该策略的设计目标
  是 10/30，而不是 NPU-RR 的 20/20。
- `dynamic_demand_fixed_path`、`joint_demand_path_cir`、
  `dynamic_slack_fixed_path`、`joint_slack_path_cir`
  （policy=`qos_dynamic_joint_cir`）：每个 NPU 在每块 SSU 上拥有两条互斥
  Path，NPU 根据当前已释放层计算 desired CIR，SSU 在下一条命令前统一
  clamp 并安装最终 Path CIR。`fixed_path` 与 `joint` 分别固定使用一条
  owned Path，或在两条 owned Path 中动态选择，用于分离 CIR 与 Path
  选路的收益。具体公式和硬件边界见下节。
- `per_ssd_full_visible_edf`：让每块 SSD 看到已释放当前层的全部 I/O，
  按 deadline/layer work 的 EDF/SRPT 启发式顺序仲裁。各 SSD 仍独立决策，
  不感知 NPU link 积压，也不能看未释放层；它不是 Oracle。
- `global_link_aware_online`：在每块空闲 SSD 的已提交 per-NPU FCFS
  队头中选择命令。协调器只使用当前 NPU active/pending 工作和已经
  dispatch 的 SSD 命令，预测候选命令的 SSD→NPU 完成时间，从而减少
  多盘同时向一个 NPU 灌入。它不读取未提交 I/O、未释放层或未来
  placement，不会空转有可服务命令的 SSD。该策略是有全局已知状态的
  online heuristic，不是最优性证明。
- `isolated_no_contention_bound`
  （summary name=`fluid_no_inter_npu_contention_upper_bound`）：不是可执行调度策略。它
  移除 NPU 之间的 SSD 竞争，允许流式数据从每层 I/O release 时刻就
  同时经过 SSD/NPU，
  并忽略不可抢占命令边界。它是故意乐观、不可实现的 relaxation，只用作
  request compute fraction 的理论上界，不应与真实策略的成本并列宣称。

## 当前静态 QoS 布局

每块 SSU 有 8 个 Group，每组 32 个 Path：

| 类别 | 每组 Path | 全盘 Path | 全盘 CIR |
|---|---:|---:|---:|
| SS | 12 | 96 | 20 GB/s |
| SL | 4 | 32 | 4 GB/s |
| LS | 12 | 96 | 12 GB/s |
| LL | 4 | 32 | 4 GB/s |

PIR uncapped，Path/Group 权重均为 1。CIR/PIR/WRR 只在初始化时配置，
运行期间不改动。CIR/WRR 只决定命令服务机会，不会将获胜命令的
SSD 服务速率改成 CIR。由于缺少目标硬件的 token-bucket 桶深和 burst
规格，命令级模型明确拒绝有限 PIR。

## 联合动态 Path + CIR

动态策略不修改上面的类别共享 Path。它在每块 SSU 上把 256 个 Path
静态划成 128 组，每个 NPU 独占两个 Path：`npu_id` 和
`npu_id + 128`。Path 所有权在运行期间不变；动态变化的只有当前层 I/O
选择哪条 owned Path，以及 SSU 在命令边界安装的 Path CIR。

令 `W_i` 为 NPU `i` 当前已释放层尚未完成的 SSD 字节数，`W_i,d` 为其中
位于 SSU `d` 的字节数，`Q_i` 为该 NPU 已经 active/pending 的 50 GB/s
接收链路字节数。两种 desired CIR 只使用这些当前可见状态，不读取未来层：

```text
# demand_proportional
D_i   = min(input_demand_i, 50)
D_i,d = D_i × W_i,d / W_i

# slack_link_guarded
slack_i         = max(0, (layer_deadline_i - now) / 1000)
guarded_slack_i = max(0, slack_i - Q_i / 50)
D_i             = 50                         if guarded_slack_i == 0
                  min(W_i / guarded_slack_i, 50) otherwise
D_i,d           = D_i × W_i,d / W_i
```

第 0 层的 deadline 是 release 时刻；后续预取层的 deadline 是当前计算层
预计结束时刻。总需求先在 NPU 侧 cap 到 50 GB/s，再按当前剩余字节拆到
各 SSU，不能在每块 SSU 上各生成一份 50 GB/s。SSU `d` 汇总所有当前
有积压 NPU 的 desired CIR，并原子执行：

```text
G_i,d = D_i,d × min(1, 40 / sum_j(D_j,d))
```

所以每个控制 epoch 都满足 `sum_i(G_i,d) <= 40 GB/s`。一个 NPU 的 grant
再按两条 owned Path 当前 pending bytes 的比例写入 Path CIR。CIR 仍只控制
长期命令服务机会；获胜命令保持单命令 40 GB/s、不可抢占，正在执行的
命令不会因下一份 CIR 改变结束时间。

每种 CIR 公式都有一组配对消融：

- `*_fixed_path`：一个 `(NPU, layer, SSU)` 固定使用两条 owned Path 中的一条。
- `joint_*_path_cir`：每 8 条 I/O 读取一次 count-only 压力，在两条
  owned Path 中选择 projected work 较小者；CIR 公式和数据面不变。

当前控制模型把配置传播延迟设为 0，并允许每条不可抢占命令结束后立即
产生下一次 CIR epoch。正式运行约产生每百万 block 101–104 万次配置变化，
因此这些数字是“零成本、逐命令重配置”的乐观 online heuristic，不代表
已经验证某款硬件能够以该周期写寄存器。

受控校准使用一块 40 GB/s SSD、两个持续积压且 I/O 等大的 NPU。Baseline
实测为 20/20 GB/s，`demand_proportional` 动态 CIR 实测为 10/30 GB/s；
每条被选中命令仍以 40 GB/s 服务，最大 active SSD I/O 为 1，最大
`sum(CIR)` 为 40 GB/s。该校准验证仲裁语义，不是多 SSU、NPU incast 或
16 层端到端收益证明。证据位于
`results/joint_dynamic_cir/calibration.json` 和 `calibration.png`。

## Path 压力遥测与选路频率

`qos_static_cir` 的客户端只读取 256 个 Path 的 `active + pending`
I/O count，不读取队列字节、age、剩余服务时间或其他 NPU 身份。选择器使用
静态 CIR/WRR 预测清空时间，并对同一次规划内已分配的 I/O 更新本地
shadow，不修改硬件快照。

正式矩阵分别测试：

- `per_layer_snapshot`：每个 `(request, layer, SSU)` 只读一次压力，
  该 SSU 上本层后续 I/O 都由这份快照加本地 shadow 规划。
- `refresh8`：每规划 8 条 I/O 重新读取压力，默认提交 batch 也是 8。
- `per_io_live`：每条 I/O 分配 Path 前都读取一次当前压力；该 I/O
  入队后，下一条 I/O 的读取能看到它。

三者使用同一选择函数和同一静态硬件。历史默认提交器以零耗时发布
8 条原子 batch；在这个模型里，`refresh8` 的本地 shadow 已包含同一 NPU
窗口内的新分配，而 `per_io_live` 在 batch 中看不到其他 NPU 插入，因此
两者的信息实际上等价，不能把该结果外推为真实并发下的“信息上界”。
`path_pressure_concurrency_probe.py` 另设 batch1/零耗时对照和每条命令
0.1 us 的有限发行敏感性对照；后者允许不同到达时刻的 NPU 在两条命令间
插入。0.1 us 是因果探针参数，不代表某款 NPU 的实测发行延迟。

## 两种 NPU 利用率指标

报告同时给出两个含义不同的指标，不应混用：

1. `avg_request_compute_fraction`：先对每个请求计算

   ```text
   16 层计算时间 / (16 层计算时间 + 该请求的所有 I/O stall)
   ```

   再对 128 个请求做等权平均。它表示“一个典型请求在处理期间有多少
   比例真正用于计算”，release jitter 之前的时间不计入分母。策略对比、
   输入分类归因和 Jain fairness 主要使用该指标。
2. `fleet_npu_compute_utilization`：

   ```text
   所有 NPU 计算时间之和 / (128 × 全局 makespan)
   ```

   它表示整个机群在全局时间窗口内的计算占用，因此会受 0–5 ms
   release 错峰和最慢请求对 makespan 的影响。

`npu_link_utilization`、SSD busy/effective utilization、queue wait 和 makespan 是定位原因的
辅助指标，不是上述两个计算利用率的同义词。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 正式配对矩阵：默认最多使用 10 个进程
python analysis_experiment.py --workers 10 --rerun

# 解析 JSON，生成 analysis.json、report.md 和全部结果图
python analyze_results.py \
  --input results/full_analysis/results.json \
  --output-dir results/full_analysis

# 独立 seed 复验所选的五种策略，并生成第 12 张图
python analysis_experiment.py --workers 9 --seed 43 \
  --select baseline,current_refresh8,tune__low_protect_cir_20_6_8_6_current_paths,tune__low_protect_cir_20_5_10_5_paths_12_5_10_5,demand_maxmin \
  --output-dir results/seed43_validation --rerun
python validation_analysis.py \
  --seed42 results/full_analysis/results.json \
  --seed43 results/seed43_validation/results.json \
  --output-dir results/full_analysis

# 联合动态 Path+CIR：两个完整 seed，每个都是 9 策略 × 3 SSU
python joint_dynamic_experiment.py --workers 9 --seed 42 --rerun
python joint_dynamic_experiment.py --workers 9 --seed 43 --rerun

# 校准 20/20 baseline 与 10/30 动态 CIR，并生成 JSON/PNG
python dynamic_cir_calibration.py \
  --output-dir results/joint_dynamic_cir

# 严格校验两个 seed，生成 analysis.json、report.md 和两张图
python analyze_joint_dynamic.py \
  --seed42 results/joint_dynamic_cir/results_seed42.json \
  --seed43 results/joint_dynamic_cir/results_seed43.json \
  --output-dir results/joint_dynamic_cir

# NPU 本地边际效用 CIR 探针：两个 seed × 三个 SSU × fixed/joint Path
python marginal_utility_probe.py --run --workers 10 --rerun
python analyze_marginal_probe.py \
  --probe results/marginal_utility_probe/results.json \
  --joint-seed42 results/joint_dynamic_cir/results_seed42.json \
  --joint-seed43 results/joint_dynamic_cir/results_seed43.json \
  --output-dir results/marginal_utility_probe

# 快速管线验证（不可作为正式结论）
python analysis_experiment.py --quick --workers 2 \
  --output-dir results/full_analysis_quick --rerun

# 提交原子性、有限发行时间和完全不读 Path 表的双 seed 消融
python path_pressure_concurrency_probe.py --seed 42 --workers 10
python path_pressure_concurrency_probe.py --seed 43 --workers 10
python analyze_path_pressure_concurrency.py --seeds 42 43

# 最终限定的四策略有限发行矩阵：双 seed × 三个 SSU × 四策略
python four_strategy_concurrency_experiment.py --workers 10 --rerun
python analyze_four_strategy_concurrency.py
```

`analysis_experiment.py` 会按 `(SSU, strategy)` 分解为独立任务并行运行，
定期 checkpoint `results.json`。缓存 spec 包含代码、数据、参数、seed 和
placement 指纹；它们变化时不会复用旧数字。

`four_strategy_concurrency_experiment.py` 同样使用多进程和逐案例 checkpoint；
固定只允许 Baseline、Static 优化前、Static 优化后和 fluid bound 四个方案，
避免分析器意外把其他策略画入最终曲线。

## 双 seed 有限发行四策略结果

最终敏感性矩阵使用 batch=1、每条 I/O 0.1 us 发行间隔，使其他 NPU 和
设备事件可以在同一 NPU 的相邻命令之间插入。0.1 us 尚未由目标硬件标定，
因此用于检查历史 atomic-8 结论是否稳健，不应解读为硬件时延预测。

每格为 `avg_request_compute_fraction / fleet_npu_compute_utilization`；fluid
bound 没有联合 makespan，所以 fleet 为 N/A。数值是 seed 42/43 均值。

| SSU | Baseline | Static before `20/4/12/4` | Static after `20/6/8/6` | Fluid bound |
|---:|---:|---:|---:|---:|
| 40 | 73.420% / 15.293% | 81.289% / 15.207% | 81.363% / 15.250% | 91.842% / N/A |
| 56 | 81.806% / 15.339% | 86.791% / 15.274% | 87.400% / 15.306% | 91.842% / N/A |
| 80 | 88.645% / 15.354% | 89.156% / 15.329% | 89.721% / 15.346% | 91.842% / N/A |

Static after 相对 before 的跨 SSU request 增益为 **+0.416 pp**，fleet
提高 **+0.031 pp**，makespan 平均缩短 **4.538 ms**。相对 baseline，
request 提高 **+4.871 pp**，但 fleet 仍低 **0.028 pp**；这个 fleet 差异
由 seed 42 的 LL 尾请求主导，在 seed 43 上方向相反，不能视为稳定退化。

严格四曲线图片、逐类输入和尾部归因见
`results/four_strategy_concurrency/`；四个后续问题的完整解释见
`doc/FOUR_STRATEGY_FOLLOWUP_CN.md`。

## 正式 16 层结果

完整 24×3 矩阵已生成并通过合同、指纹、逐请求配对、数据面 invariant 和
逐请求 fluid-bound 校验。表中每格为
`avg_request_compute_fraction / fleet_npu_compute_utilization`；fluid bound 只定义
第一个指标。

| SSU | Baseline | 当前 static CIR | 最佳固定 static `20/6/8/6` | Global link-aware online | Fluid upper bound |
|---:|---:|---:|---:|---:|---:|
| 40 | 72.87% / 14.55% | 81.95% / 14.40% | 82.44% / 14.42% | 75.82% / 14.54% | 91.23% / n/a |
| 56 | 81.27% / 14.59% | 86.87% / 14.49% | 87.03% / 14.52% | 84.93% / 14.62% | 91.23% / n/a |
| 80 | 88.29% / 14.62% | 89.84% / 14.58% | 90.11% / 14.59% | 90.60% / 14.63% | 91.23% / n/a |

跨三个 SSU，最佳固定 static 的平均 request compute fraction 为 86.53%，
baseline 为 80.81%，提升 **+5.72 pp**。它的 CIR 从当前
`(20,4,12,4)` 改为 `(20,6,8,6)`，Path 数仍为 `(12,4,12,4)`；主要是
保护 SL/LL，而不是继续从长计算类抽走服务。对应 fleet 指标没有超过
baseline，因为最慢 LL 请求仍决定全局 makespan。

在历史零耗时 atomic-8 提交模型中，`refresh8` 与 per-I/O live 在三个正式点
的结果逐 request、逐位相同，而 per-I/O 读取
170.5 万次计数器、约 1.746 GB 遥测；refresh8 只需约 24.9–28.5 万次、
255–292 MB。这只能说明同一 NPU 本地 shadow 正确复现了该原子窗口内的
变化，不能证明有限命令发行时间下读取其他 NPU 的新状态没有价值。

独立 workload seed 43 复验中，`20/6/8/6` 的平均增益为 +6.154 pp，并在复验的
4 个非 baseline 候选中仍排名第一（seed 42 为 +5.719 pp）。seed 43 没有重跑
全部 24 个正式策略，因此这不是全策略排名结论；它只是两 seed 敏感性检查，
也不是统计显著性证明。
完整归因、12 张图和跨 seed 报告位于 `results/full_analysis/`。

### 联合动态 Path+CIR 结果

独立的联合动态矩阵对 seed 42 和 43 都运行了相同的 9 个策略、128 NPU、
16 层和 40/56/80 SSU，并通过 workload/placement 配对、数据面 invariant
以及动态 `sum(CIR) <= 40 GB/s` 校验。下表是 request compute fraction；
最后一列是三个 SSU 等权平均后相对同 seed baseline 的增益。

| Seed | Strategy | 40 SSU | 56 SSU | 80 SSU | Mean gain |
|---:|---|---:|---:|---:|---:|
| 42 | Baseline | 72.87% | 81.27% | 88.29% | +0.000 pp |
| 42 | Best fixed static | 82.44% | 87.03% | 90.11% | +5.719 pp |
| 42 | Joint demand Path+CIR | 80.37% | 86.34% | 90.36% | +4.885 pp |
| 42 | Joint slack Path+CIR | 80.79% | 86.87% | 90.68% | +5.306 pp |
| 43 | Baseline | 73.98% | 82.36% | 89.05% | +0.000 pp |
| 43 | Best fixed static | 83.47% | 88.82% | 91.55% | +6.154 pp |
| 43 | Joint demand Path+CIR | 82.28% | 88.50% | 91.78% | +5.726 pp |
| 43 | Joint slack Path+CIR | 82.74% | 88.89% | 91.99% | +6.083 pp |

两个 seed 的最佳动态策略都是 `joint_slack_path_cir`，但总体 request 指标
仍由 `best_fixed_static` 获胜。动态 slack 在容量较充足时更有优势：相对
best fixed，80 SSU 上 seed 42/43 分别为 +0.563/+0.437 pp；40 SSU 上则
分别为 -1.647/-0.725 pp。

least-work Path 相对同 CIR 的 fixed-Path 消融，在 40/56 SSU 通常提高
request 指标约 0.2–0.6 pp，而在 80 SSU 两个 seed 都轻微回退。这说明
更鲜的双 Path 分流主要在 SSD 竞争较强时有用，并不是无条件收益。

完整输入级归因、fixed/joint 消融、控制 epoch 和两 seed 报告位于：

- `results/joint_dynamic_cir/results_seed42.json`
- `results/joint_dynamic_cir/results_seed43.json`
- `results/joint_dynamic_cir/analysis.json`
- `results/joint_dynamic_cir/report.md`
- `results/joint_dynamic_cir/01_joint_dynamic_strategy_comparison.png`
- `results/joint_dynamic_cir/02_category_path_cir_epochs.png`

### NPU 本地边际 CIR 探针

为了检查 demand/slack 公式是否已经足够好，另行测试了一个不读取未来信息的
bounded marginal-utility 公式。NPU 只使用当前层 deadline、当前层剩余 SSD
字节 `W_i` 和自身 50 GB/s 接收链路积压 `Q_i`：

```text
H_i = max(deadline_i - now, 0) + 1000 * Q_i / 50 + 1000 * W_i / 50
u_i = 1 ms / (1 ms + H_i)
D_i = 5 + (50 - 5) * u_i
```

`D_i` 仍按 `W_i,d / W_i` 拆盘，并由每块 SSU 原子比例缩放到总 CIR 不超过
40 GB/s。这个公式会保护 deadline 近、剩余工作小且 link 不拥塞的请求，
同时用 5 GB/s floor 防止大层饿死。

完整 12 点配对结果中，marginal joint 在 6/6 场景高于 baseline、3/6
场景略高于 best-fixed；联合选路相对 fixed 在 5/6 场景提高 request 指标。
但跨两个 seed 和三个 SSU 等权平均，marginal joint 为 86.824%，低于
joint-slack 的 86.994% 和 best-fixed 的 87.237%。它的 fleet 为 15.311%，
比 joint-slack 的 15.269% 更接近 baseline 15.329%，说明它更偏向抑制
incast/尾部，而不是最大化平均 request compute fraction。因此它保留为
目标函数权衡探针，不替代推荐的 slack 动态 CIR。

结果位于 `results/marginal_utility_probe/`，其中包含严格配对
`analysis.json`、中文 `report.md` 和六面板 `analysis.png`。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 0–5 ms release delay 配对、256-count ABI、Path 选路频率、Path FCFS、
CIR 服务机会、每盘/每 NPU 单 active、40/50 GB/s 精确服务时间、block/
placement/bytes 守恒、需求 max-min、global link-aware 可见性边界、fluid
upper bound 和分析管线。

## 主要文件

- `sim.py`：离散事件引擎、命令仲裁、静态 Path 选路和 SSD→NPU 数据面。
- `strategy_profiles.py`：预先声明的 CIR/Path 候选和客户端遥测/提交变体。
- `advanced_policies.py`：demand max-min、per-SSD EDF 和 global online 的纯调度函数。
- `upper_bounds.py`：独立无竞争参考与 fluid 理论上界。
- `analysis_experiment.py`：正式配对策略矩阵、多进程运行和 checkpoint。
- `analyze_results.py`：固定策略选择、归因分析、报告和结果图。
- `validation_analysis.py`：seed 42/43 严格合同检查、敏感性表和第 12 张图。
- `allocation_calibration.py`：受控的 10/30 GB/s 双 NPU 校准。
- `joint_dynamic_experiment.py`：动态 Path+CIR 的双 seed 配对矩阵和 checkpoint。
- `analyze_joint_dynamic.py`：动态策略合同检查、消融/类别归因和结果图。
- `dynamic_cir_calibration.py`：真实命令调度器上的 20/20 与 10/30 校准。
- `marginal_utility_probe.py`：NPU 本地边际 desired CIR 的独立双 seed 探针。
- `analyze_marginal_probe.py`：边际探针的严格配对分析、报告和结果图。
- `path_pressure_concurrency_probe.py`：atomic-8、batch1/零耗时、有限发行和
  no-telemetry Path 轮转的配对消融。
- `analyze_path_pressure_concurrency.py`：提交交错证据、逐请求配对和消融图。
- `four_strategy_concurrency_experiment.py`：最终限定四策略的双 seed、有限发行
  正式矩阵，多进程运行并逐案例 checkpoint。
- `analyze_four_strategy_concurrency.py`：严格四曲线、双 seed 聚合、类别与尾部
  归因，以及历史零发行模型配对比较。
- `doc/IDEAL_NO_CONTENTION_BOUND_CN.md`：可运行理想化 heuristic 与不可运行
  fluid 无竞争上界的数学定义、约束和正确解读。
- `doc/FOUR_STRATEGY_FOLLOWUP_CN.md`：fleet/makespan、客户端原子性、无状态
  Path 选路、低 SSU 利用率和 CIR/Path 合理性的集中回答。
- `tests/`：语义、守恒、策略和管线测试。

## 模型限制

- QoS 256-count 遥测被视为零延迟且无丢失；count 在 SSD 命令完成时
  减一，不等待 NPU link 完成。
- 历史完整主矩阵的客户端命令发行时间为 0；最终四策略矩阵和有限发行探针
  使用 0.1 us/命令，只用于判断并发插入是否改变结论，尚未用实测硬件
  发行/遥测延迟标定。
- 中间 buffer 无限，没有反压；尚未建模固定命令/NAND 延迟、QD 曲线、
  channel/die 并行、流式传输或 PCIe/网络协议开销。
- packetized-WFQ 是对 CIR/WRR 服务机会的命令级近似，不能声称复刻
  某款 SSD 的具体 WRR/token-bucket 微架构。
- 动态 CIR 假设 desired 上报和寄存器配置零延迟，并允许逐命令更新；尚未
  建模配置写入周期、控制报文成本、版本传播延迟或较慢 epoch 对结果的影响。
- block placement 在策略之间固定；调度器不能将 I/O 改发到另一块 SSD。
- `global_link_aware_online` 的全局信息仅限于已提交/已承诺工作，
  并使用贪心 downstream-aware EDF；它不是 offline optimum。
- fluid upper bound 放宽了关键硬件约束，所以只能表示理论天花板，
  不是对可部署策略的性能承诺。
