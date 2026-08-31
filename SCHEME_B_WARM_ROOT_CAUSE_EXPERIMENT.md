# Warm 满载下 Scheme B 的根因与优化结果

## 结论先行

这次问题不是单一的 “SSU 总带宽不够”，也不是单一的 NPU 50 GB/s 接收上限：

- **受控合成输入下，SSU allocator、动态 CIR、SSD40 数据面和窗口记账都正确。**
  32 NPU / 8 SSU、关闭 NPU 限制后，六个策略在每盘 40 GB/s 以下都满足 100%；
  轻微过载时都精确落在 `40 / offered` 的物理曲线上。
- **SSU6 是输入容量和固定 placement 主导。** 即使 2× TTFT 所需的半带宽，热点盘仍需
  60.933 GB/s，100% SLO 不可能。任何策略只能选择牺牲哪些请求。
- **SSU10 是输入瞬时热点和策略目标共同造成。** 长期平均的 2× 目标可行，但固定 seed 的
  20k 去同步 phase 抽样仍有 16.315% 出现至少一块盘的半带宽需求超过 40 GB/s。
- **SSU18（约等效 128 NPU / 72 SSU）容量基本可行，原 Scheme B 的主要问题是目标函数。**
  旧的 flow-level absolute max-min 和后来的全体 normalized fairness 都没有最大化
  `TTFT <= 2×compute` 的请求数；过载时会出现 “人人公平地低于 0.5，人人 miss”。
- **NPU50 是次要限制，不是主因。** SS 请求 raw demand 为 83.346 GB/s，确实超过50；
  但其 2× target 只有41.673 GB/s。SSU18 下移除 NPU50 只让 baseline SLO 增加约5点，
  而优化策略相对 baseline 可增加约19点。

因此，单纯 “按 raw 请求带宽比例分 SSU CIR” 不够。策略必须显式理解 request barrier 和
SLO threshold，并在目标集合不可行时做 admission，而不是把所有 coflow 一起压到 threshold
以下。

## 实验口径

- 32 NPU、16层、batch 1；每个 NPU 有确定性的32请求饱和前缀。
- 每 NPU 至少完成4个请求，额外 settle 500 ms，再测固定2,000 ms 窗口。
- 窗口中所有 NPU 必须有样本；窗口内 admission 的请求全部 drain 到完成；任一 NPU
  提前耗尽前缀则整行 invalid。
- 主 SLO 是各 NPU 先算达标率、再等权平均；`processing TTFT = completion-admission`，
  不包含外部 arrival-to-admission 排队。
- SSD 是每盘40 GB/s、单命令不可抢占；物理 NPU link 是每 NPU 50 GB/s。
- 32×17 与32×18分别等效128×68与128×72，用来夹住目标128×70。它们保持容量比例，
  但不声称有限 fleet 的调度方差完全相同。

所有正式行保存 workload、placement、trace、simulator-input 和源码指纹。诊断 runner 在
启动时冻结源码/配置指纹，每行记录该指纹，并在结束时再次校验，不能用旧缓存重新盖章。

## 1. 纯 SSU 合成带宽实验

物理 NPU link 和 Scheme-B allocator cap 都提高到有限的 `1e6 GB/s`，因此只保留 SSU40。
每盘由4个 NPU striped 地持续请求，另有一个40/30/20/10%异构分布点。

| 每盘 offered | 理论 satisfaction | 六策略实测 | SSD util | 2× SLO |
|---:|---:|---:|---:|---:|
| 32 | 1.0000 | 1.0000 | 80.0% | 100% |
| 36 | 1.0000 | 1.0000 | 90.0% | 100% |
| 39 | 1.0000 | 1.0000 | 97.5% | 100% |
| 40 | 1.0000 | 1.0000 | 100% | 100% |
| 41 | 0.9756 | 0.9756 | 100% | 100% |
| 42 | 0.9524 | 0.9524 | 100% | 100% |
| 44 | 0.9091 | 0.9091 | 100% | 100% |

六策略是 baseline、layer_once、refresh8、causal Scheme B、normalized Scheme B 和最终的
SLO-admission Scheme B。48/48 数据面窗口通过全部 invariant。这个实验说明：策略能正确
兑现可用 SSU 带宽；真实 trace 的差异不是隐藏的 CIR/SSD 实现错误。

## 2. 真实输入的容量边界

| 类别 | Raw demand | 2× target | NPU50 是否限制 raw |
|---|---:|---:|:---:|
| SS | 83.346 GB/s | 41.673 GB/s | yes |
| SL | 13.812 GB/s | 6.906 GB/s | no |
| LS | 28.831 GB/s | 14.415 GB/s | no |
| LL | 14.712 GB/s | 7.356 GB/s | no |

Fleet 长期 raw demand 为666.236 GB/s，raw knee 为16.656 SSU，2× fluid knee 为8.328
SSU。即使 SSD 无限，NPU50 给出的理想 fleet mean NPU utilization 上限也只有97.105%。

| SSU | 长期最大 raw/盘 | 长期最大 2× target | 同步 sequence 2×可行 | 20k去同步抽样不可行 |
|---:|---:|---:|---:|---:|
| 6 | 121.866 | 60.933 | 0/32 | 100.000% |
| 10 | 70.349 | 35.175 | 0/32 | 16.315% |
| 18 | 39.187 | 19.593 | 32/32 | 0.000% |

去同步数字是固定 seed 的 fluid phase 抽样，不是 DES 结果；用途只是证明 “长期平均可行”
不等于每个瞬时 placement 都可行。

仅按 fleet 总容量计算，平均 NPU 利用率的乐观上限约为：SSU6 36.02%、SSU10 60.04%、
SSU18 97.105%（此点由 NPU50 限制）。Baseline 已分别达到32.69%、57.24%、93.29%，
所以平均利用率本来只剩约3个百分点的理论空间，不可能远超 baseline。

## 3. 优化后的 Scheme B

`SLOAdmissionSchemeBController` 使用客户端可见的所有未 ready 层 manifest，并给每个 NPU
一个 dedicated Path：

1. 将 warm 2× 的严格进度 floor 设为0.50，工程目标设为0.52。
2. 若全体0.52可行，先保障所有请求；若只有全体0.50可行，则至少保障严格 floor。
3. 真正过载时，稳定 greedy packing 选择一组请求，已选择 request pin 到完成。
4. 未选择请求仍获得后台 coflow progress，避免只读完小分片却卡在另一块 SSU 的 barrier。
5. 全体都已获得 floor 后，剩余容量改为 work-conserving throughput fill。
6. 只更新 CIR；PIR 保持无限。控制由 batch-membership 事件触发，并设置最小更新时间。

这是一个有5%后台 reserve 和2% margin 的启发式 threshold admission，不是精确的
maximum-cardinality knapsack。它优化的是 processing-SLO 达标数量；在严重过载时仍存在
公平性/尾部取舍。

## 4. 10 ms 基准结果

| SSU | 策略 | Mean NPU util | Equal-NPU SLO | Requests/s |
|---:|---|---:|---:|---:|
| 6 | baseline | 32.69% | 25.00% | 96.0 |
| 6 | current Scheme B | 32.57% | 9.15% | 91.0 |
| 6 | normalized Scheme B | 32.76% | 0.00% | 92.0 |
| 6 | SLO admission | 32.40% | 67.35% | 92.0 |
| 10 | baseline | 57.24% | 75.00% | 160.0 |
| 10 | current Scheme B | 56.94% | 50.77% | 156.5 |
| 10 | normalized Scheme B | 56.69% | 61.51% | 164.0 |
| 10 | SLO admission | 57.13% | 82.08% | 159.5 |
| 18 | baseline | 93.29% | 79.38% | 258.0 |
| 18 | current Scheme B | 92.78% | 82.28% | 254.5 |
| 18 | normalized Scheme B | 91.14% | 89.00% | 247.0 |
| 18 | SLO admission | 92.21% | 98.44% | 255.0 |

相同 request-ID 交集仍保持同方向，收益不是 measurement cohort 漂移。SSU18 中 admission
把最难的 SS 类 SLO 从 baseline 的17.19%提高到93.75%。有效的 released-I/O oracle 为
93.05% utilization / 100% SLO，说明 admission 已接近可见 I/O 下的 threshold 上界。

SSU6 不能当成全面胜利：其最小 NPU 利用率从 baseline 的30.33%降到23.05%，normalized
TTFT P99 从16.95×恶化到53.18×，SS 类仍是0% SLO。它是容量不足时提高通过数量的
cardinality tradeoff。

## 5. 控制频率消融

以下仍是 batch-boundary 事件触发；`25/50/100 ms` 是最小间隔，不是固定周期 timer。

| SSU | Min interval | Mean NPU util | Equal-NPU SLO | Control evals | Path writes |
|---:|---:|---:|---:|---:|---:|
| 10 | 25 ms | 57.00% | 84.57% | 154 | 48,010 |
| 10 | 50 ms | 57.13% | 82.77% | 78 | 24,650 |
| 10 | 100 ms | 56.89% | 76.03% | 38 | 11,850 |
| 18 | 25 ms | 93.38% | 98.85% | 126 | 37,158 |
| 18 | 50 ms | 94.23% | 89.60% | 65 | 26,498 |
| 18 | 100 ms | 93.26% | 86.49% | 33 | 15,265 |

推荐：

- **25 ms：SLO-first。** SSU18 相对 baseline 同时为利用率 `+0.08 pp`、SLO
  `+19.47 pp`；SSU10 为 `-0.24 pp / +9.57 pp`。
- **50 ms：低写入、利用率优先。** SSU18 同时为 `+0.93 pp / +10.22 pp`；SSU10
  为 `-0.11 pp / +7.77 pp`，控制评估约减半。
- **100 ms：不推荐作为通用默认。** SS 的 compute-only TTFT 只有约20.6 ms；100 ms
  会跨过多个短请求，SSU10 的 SLO 收益只剩约1点。

因此百毫秒控制在这份 trace 上过慢；25–50 ms 是已验证的范围，且远高于1 ms下限。

## 6. SSU70 比例夹逼

32×17（等效128×68）的配对结果：

| 策略 | Mean NPU util | Equal-NPU SLO |
|---|---:|---:|
| baseline | 91.61% | 80.76% |
| 10-ms SLO admission | 90.57% | 98.12% |

结合32×18（等效128×72），SSU70附近的稳定结论是：10-ms admission 相对 baseline 提高
约17–19个 SLO 点，平均利用率代价约1点。32×17 baseline 的 SSD outstanding block 数从
6492降到74，虽然所有既有 invariant 通过且四个 utilization block 只波动0.67点，这仍提示
SSD队列并非严格平稳；不能把比例夹逼冒充精确的128×70正式结果。

## 7. NPU50 反事实

SSU18 下只移除 baseline 的物理 NPU50：utilization 从93.29%升到94.84%，SLO 从79.38%
升到84.38%。因此 NPU50 解释约1.55点 utilization 和5点 SLO，不足以解释 admission 的
约19点收益。

`new_scheme_b_npu_bypass` 同时放开物理和 allocator NPU cap，只是敏感性检查，不是
admission controller 的严格单变量2×2消融。

## 最终判断

- **策略实现/SSU带宽分配本身：** 受控输入下正确。
- **真实负载输入：** 固定 placement、类别异构和瞬时 phase hotspot 是 SSU6/10 的主要
  容量问题；只看总带宽会漏掉单盘约束。
- **旧 Scheme B：** 目标函数是主要策略问题。绝对 flow fairness 或全体 normalized
  fairness 都不等价于最大化 binary SLO。
- **NPU接收：** NPU50 限制利用率上限并影响 SS，但不是 SSU70 附近 SLO 差距的主因。
- **其他遗漏点：** request-level barrier、非抢占命令、控制陈旧、CIR surplus 的 WRR
  分配、以及 measurement cohort/队列平稳性都会让 “grant 看起来合理” 与端到端 TTFT
  不完全等价。

优化后可以诚实地说：**SSU18/SSU70附近 TTFT processing SLO 有很大数值提升；在25或
50 ms配置下也能让平均 NPU利用率不低于 baseline。** 不能说利用率“远超”baseline，原因
不是策略失败，而是 baseline 已接近 SSD/NPU 的物理上限。当前结论来自单 seed，不构成
统计显著性证明。

## 复现入口

```bash
python synthetic_bandwidth_experiment.py --pure-ssu --mode both
python steady_state_32npu_experiment.py --workers 3
python steady_state_32npu_diagnostics.py --workers 3 --rerun
python analyze_steady_state_32npu_root_cause.py
python analyze_steady_state_32npu_capacity.py
python steady_state_32npu_admission_interval_ablation.py --workers 3
python steady_state_32npu_ratio70_bracket.py --workers 2 --rerun
```

对应细表位于：

- `results/synthetic_bandwidth_satisfaction/pure_ssu_32npu_8ssu.md`
- `results/steady_state_32npu_normalized_slo/root_cause_report.md`
- `results/steady_state_32npu_normalized_slo/capacity_analysis.md`
- `results/steady_state_32npu_normalized_slo/admission_interval_ablation.md`
- `results/steady_state_32npu_normalized_slo/ratio70_bracket.md`
