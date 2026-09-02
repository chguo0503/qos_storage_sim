# 32 NPU / SSU 3–5 / 64 秒微观时间线审计

## 验收结论

9 个点（3 个策略 × SSU 3/4/5）全部通过独立验收：每点 128 个 500 ms 块、129 个左极限边界，仿真器报告的全部 invariant 均为 `true`。分析器又独立重算了 NPU×SSU 的 interval-average attributed SSD service、NPU-link delivery、资源容量和 `C = activated + Q(start) - Q(end)` 闭合。

严格配对范围是同一 SSU 内三策略的完整输入 fingerprint 和科学运行时；跨 SSU 则严格要求 profile catalog、采样 recipe、schedule 和 assignment 相同。placement/trace/simulator fingerprint 随 SSU 数变化是预期行为。报告没有写入 hostname、PID、token 或私有绝对路径。
三个 SSU shard 的完整 hostname/platform/thread/CPython/NumPy/BLAS runtime signature 只在内存中比较，公开结果仅保留不可逆 SHA-256。原始 runner shard 不复制进公开目录；每个公开文件必须严格小于 50 MiB。

## 领导汇报摘要

- Adaptive 本次配置是 **α=2 tuned**：`required_ratio=0.5` 是 α=2 容量可行性目标，不是运行时读取 TTFT deadline/slack。α=1.5 图仅是同一条 α=2-tuned 轨迹的补充敏感性统计，不是另跑的 α=1.5-tuned 控制器。
- 64 秒全窗内，同一 SSU 三策略的 NPU 利用率最大精确跨度为 `0.736348` 个百分点；不是逐位完全相等。
- 完整 state partition 的最大 `other` 占比仅 `0.000000000%`，且所有策略相对 baseline 的 compute 差均由反向 IO-barrier 差精确闭合；因此该模型里的利用率差来自 IO barrier，而不是未解释的 idle。
- 三个 SSU 的正式 measurement 均观测到 `640 evaluations / 640 commits`，相邻 evaluation 均为 `100 ms`。这是持续 batch-boundary 事件让 100-ms 最小间隔在这条高负载轨迹上饱和；trigger 仍只有事件来源，并不存在 wall-clock 周期 trigger，不能把一般语义写成‘每 100 ms 定时改 CIR’。

| SSU | measurement evaluations | commits | gap min/median/max | trigger reasons |
|---:|---:|---:|---:|---|
| 3 | 640 | 640 | 100.000000/100.000000/100.000000 ms | `batch_boundary` |
| 4 | 640 | 640 | 100.000000/100.000000/100.000000 ms | `batch_boundary` |
| 5 | 640 | 640 | 100.000000/100.000000/100.000000 ms | `batch_boundary` |

## 64 秒总体结果

| SSU | 策略 | NPU 利用率 | equal-NPU TTFT SLO α=1.5 | equal-NPU TTFT SLO α=2 | 平均 IO barrier | 请求数 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Baseline | 58.19% | 34.25% | 40.02% | 296.553 ms | 2877 |
| 3 | Layer once: TTL 5 ms | 58.32% | 53.90% | 66.34% | 296.844 ms | 2887 |
| 3 | Adaptive: event-driven, min 100 ms | 58.93% | 31.38% | 72.50% | 292.760 ms | 2875 |
| 4 | Baseline | 76.47% | 47.10% | 53.06% | 126.682 ms | 3808 |
| 4 | Layer once: TTL 5 ms | 76.33% | 63.24% | 75.32% | 127.329 ms | 3805 |
| 4 | Adaptive: event-driven, min 100 ms | 76.07% | 50.88% | 78.72% | 129.648 ms | 3790 |
| 5 | Baseline | 89.92% | 63.39% | 70.00% | 46.289 ms | 4459 |
| 5 | Layer once: TTL 5 ms | 89.20% | 74.42% | 85.07% | 49.866 ms | 4445 |
| 5 | Adaptive: event-driven, min 100 ms | 89.61% | 69.56% | 84.96% | 47.771 ms | 4456 |

## TTFT 窗口与启动阶段口径

`TTFT = request completion_time - admission_time`，不包含 arrival→admission 的等待。正式 measurement 在每个 NPU 已 warm up 8 个请求后再 settle 500 ms 才开始；SLO cohort 只包含 admission 落在 `[measurement_start, measurement_end)` 的请求，因此不包含仿真最初请求的启动瞬态。measurement start 已在途的 carry-in 不进入 SLO cohort，但其与窗口的交集会完整进入 utilization/state partition。这里描述的是仿真口径，不能称为真实硬件 cold-start 测量。

layer 0 I/O 若在 admission 后才 ready，其等待会进入 TTFT 的 IO barrier；cross-request prefetch 若在 admission 前已经完成，则提前完成的那一段不会计入 TTFT。逐层导出的 `compute_start=max(previous_compute_end, io_ready)` 与 barrier 之和已经逐请求闭合到 TTFT。

## 低利用率的主证据：完整 64 秒状态时间分解

下面直接对每个 NPU 的所有 microbatch layer 区间与严格 64 秒窗口求交，包含 measurement start 时已经在途的 carry-in。每个 NPU 都精确满足 `compute + io_barrier + other = 64 s`，其中 compute 与利用率分子独立一致；因此这张表覆盖完整窗口的 idle，而不是只看 measurement 内新 admission 的请求 cohort。

| SSU | 策略 | compute | IO-barrier idle | other complement | barrier / all idle | vs baseline: Δbarrier | vs baseline: Δother |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | Baseline | 58.19% | 41.81% | 0.00% | 100.00% | +0.000000 NPU-s | +0.000000 NPU-s |
| 3 | Layer once: TTL 5 ms | 58.32% | 41.68% | 0.00% | 100.00% | -2.651024 NPU-s | +0.000000 NPU-s |
| 3 | Adaptive: event-driven, min 100 ms | 58.93% | 41.07% | 0.00% | 100.00% | -15.080398 NPU-s | +0.000000 NPU-s |
| 4 | Baseline | 76.47% | 23.53% | 0.00% | 100.00% | +0.000000 NPU-s | +0.000000 NPU-s |
| 4 | Layer once: TTL 5 ms | 76.33% | 23.67% | 0.00% | 100.00% | +2.911616 NPU-s | +0.000000 NPU-s |
| 4 | Adaptive: event-driven, min 100 ms | 76.07% | 23.93% | 0.00% | 100.00% | +8.232317 NPU-s | -0.000000 NPU-s |
| 5 | Baseline | 89.92% | 10.08% | 0.00% | 100.00% | +0.000000 NPU-s | +0.000000 NPU-s |
| 5 | Layer once: TTL 5 ms | 89.20% | 10.80% | 0.00% | 100.00% | +14.764293 NPU-s | -0.000000 NPU-s |
| 5 | Adaptive: event-driven, min 100 ms | 89.61% | 10.39% | 0.00% | 100.00% | +6.365424 NPU-s | -0.000000 NPU-s |

## 为什么 SLO 可以更高，但平均 NPU 利用率不一定更高

两者统计对象不同。上面的完整时间分解给出 idle 究竟落在 IO barrier 还是 other；NPU 利用率本身是 64 秒内 compute busy time 的积分，而 TTFT SLO 是逐请求的阈值计数。策略可能把等待在请求之间重新分布，从而增加 fail→pass 个数而不增加总 compute busy time；但这句话本身不是整体 SLO 差异的因果证明。闭环运行中，各策略按半开测量窗 admission 得到的整体 SLO cohort 成员和数量可能不同，因此不能仅凭整体 SLO 直接归因。matched common cohort 只佐证其覆盖到的交集请求；零 common NPU 和非交集请求均不支持该因果外推。完整 64 秒 state partition 才是利用率/idle 根因的全窗证据。

下面的 Q/activated 等式进一步解释 compute busy 总量为什么变化，但它不是 idle 类型的替代指标；等式由逐 NPU、逐 500 ms 边界精确闭合后再汇总：

`Δ compute_busy = Δ activated_compute + Δ Q(start) - Δ Q(end)`

| SSU | 策略 vs baseline | Δcompute busy | Δactivated | ΔQ(start) | -ΔQ(end) | 闭合误差 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Layer once: TTL 5 ms | +2.651024 NPU-s | +1.469937 | +1.253036 | -0.071948 | -4.619e-14 |
| 3 | Adaptive: event-driven, min 100 ms | +15.080398 NPU-s | +6.129228 | +8.609839 | +0.341332 | +6.573e-14 |
| 4 | Layer once: TTL 5 ms | -2.911616 NPU-s | -6.686182 | +2.358530 | +1.416035 | +2.736e-13 |
| 4 | Adaptive: event-driven, min 100 ms | -8.232317 NPU-s | -7.354942 | -0.183433 | -0.693942 | +3.730e-13 |
| 5 | Layer once: TTL 5 ms | -14.764293 NPU-s | -10.348282 | -3.076922 | -1.339089 | -3.340e-13 |
| 5 | Adaptive: event-driven, min 100 ms | -6.365424 NPU-s | -5.739686 | -1.574394 | +0.948656 | -1.954e-13 |

matched-request 只作为 TTFT cohort 的佐证：它比较三策略测量窗中共同的 `(NPU, sequence)` 请求，可展示阈值重分配，但不包含所有 carry-in，也绝不用于声称覆盖完整 64 秒 idle：

| SSU | 每NPU common请求 min/median/max | 零 common coverage NPU数 | 每NPU IoU min/median/max |
|---:|---:|---:|---:|
| 3 | 61/85.5/95 | 0 | 0.716/0.883/0.968 |
| 4 | 84/115.5/143 | 0 | 0.840/0.945/0.992 |
| 5 | 102/139.0/167 | 0 | 0.926/0.979/1.000 |

| SSU | 策略 vs baseline | common 请求 | barrier 改善/恶化 | α1.5 fail→pass / pass→fail | α2 fail→pass / pass→fail | barrier 总变化 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Layer once: TTL 5 ms | 2669 | 1612/1034 | 891/364 | 1019/317 | +13.224813 s |
| 3 | Adaptive: event-driven, min 100 ms | 2669 | 1551/1118 | 267/351 | 868/1 | -8.680399 s |
| 4 | Layer once: TTL 5 ms | 3664 | 2149/1358 | 1110/511 | 1234/408 | +1.301349 s |
| 4 | Adaptive: event-driven, min 100 ms | 3664 | 1866/1789 | 445/301 | 951/8 | +8.814001 s |
| 5 | Layer once: TTL 5 ms | 4395 | 2201/1640 | 876/380 | 877/204 | +14.763146 s |
| 5 | Adaptive: event-driven, min 100 ms | 4395 | 1904/2200 | 507/231 | 725/61 | +6.077942 s |

## 带宽时间线应如何读

- `controller_demand` 是 remaining manifest IO / remaining controller compute。它是 Adaptive 的输入；对 baseline/layer-once 只作为同口径反事实诊断。
- `physical_remaining_demand` 是 producer 直接导出的物理未交付工作除以同一 compute denominator，只是诊断，不是控制器输入。分析器独立验证 activated IO 各 stage 守恒、direct SSD/link queue、`physical >= activated-undelivered`、`controller >= physical` 及 demand 除法，但现有导出不足以从零完全重建全部未来尚未 activate 的 physical work，因此不把它冒充成独立 oracle。
- `installed CIR` 是 dedicated Path 的仲裁保证，不是实际吞吐。主图没有把 CIR 画成 actual。
- `interval-average attributed SSD service` 是相邻 500 ms 边界间 NPU×SSU SSD-served 累计字节差除以 0.5 s；它不是某一时刻正在执行命令的瞬时速率。同一窗口内可先后服务多个 cell。
- 只有 `active_command_by_ssu.physical_service_gbps` 与 dispatch probe 的 `physical_command_service_gbps` 描述当时唯一在途非抢占命令的服务速率（本配置为 40 GB/s）。`NPU-link delivered` 才是该 NPU 从该 SSU 在区间内平均实际收到的带宽。二者的区间平均差刻画 SSD→link 管线填充/排空与排队。
- v3 边界把层状态拆成 `current_compute_layer`、`next_compute_layer`、`next_compute_layer_io_ready` 和 `waiting_on_io_layer`；不再把正在计算的当前层误写成 next-layer IO 状态。SSD scientific service 只采用 compensated completed-command 累计加 immutable active-prefix；历史 fragmented-settle 累计只留作 residual 诊断，绝不用于带宽图或结论。SSD outstanding 则直接枚举 physical pending/active，不由两个累计数相减构造。
- Adaptive 的每个旧 CIR 都由前一次实际安装值重建（首次为 0），threshold=0 时按 `abs(target-old)>1e-12` 重放。重建后的 commit/transaction/path-write 计数和 129 个边界的 installed CIR 已全部一致。

## Adaptive 是否因为‘快到 TTFT SLO’才分配带宽

不是。当前控制器的因果输入没有 admission time、elapsed TTFT、deadline 或 slack。`required_ratio=0.5` 是 α=2 的容量可行性配置，不是实时 deadline 检查。`min_interval_ms=100` 是 event-driven controller evaluation/commit 的最小间隔（节流），并不表示每 100 ms 周期性必改一次 CIR；只有事件触发、距上次 evaluation 至少 100 ms，并且 `abs(target-old)>threshold` 时对应 CIR entry 才写入。`adaptive_decisions.csv` 的相邻 time/trigger/changed-entry 可逐次核对。图里同时给出 `deadline_slack = α×ideal - elapsed` 与 `slack_after_remaining_compute = deadline_slack - Q(t)`；它们均由边界状态事后派生，并明确标注 diagnostic-only。decision-time 的 prefetch-only/尚未 admission 请求没有 TTFT 时钟，因此 elapsed、slack 和 slack-Q 一律为 NA，也不进入 slack 曲线。若 selected 与 slack 有相关性，也不能解释为控制器读取了 slack。

Adaptive 真正的选择理由记录在 decision CSV：controller demand、normalized total/dominant score、candidate order、pinned 状态、admission stage、capacity rejection、violating SSU 和最终 grant。admission 决定‘谁被选中’，grant decomposition 决定‘每个 NPU×SSU 最终给多少’；explicit-spill/V2 使用 floor/background/selected-tail/spill-tail 四项，coflow/V1 则单列 V1 grant，所有 cell 均逐元素闭合到 final grant。Layer-once 的理由则由每块 route plan、fresh pressure read、TTL cache hit 和 dispatch probe 的 virtual-finish/RR 选择共同刻画。

## Layer-once 的 route plan 为什么选择这些 Path/group

route probe 保留了纯策略函数所需的完整输入：allowed Path 的 pressure count/CIR/PIR/weight/group、全 group 活跃聚合、block 大小和 start offset。分析器用 `policy_logic.pressure_aware_path_ids` 独立重放，要求 selected Path 逐 block 完全相同；`fresh` 必须 age=0，`cache` 必须仍在 5 ms TTL 内。这里同样没有 TTFT deadline/slack 输入。

| SSU | 策略 | route records | planned blocks | fresh/cache/none | selected groups | portable exact replay | truncated |
|---:|---|---:|---:|---|---|---:|---:|
| 3 | Baseline | 99 | 33049 | `{"none":99}` | `{"0":33049}` | True | False |
| 3 | Layer once: TTL 5 ms | 123 | 35566 | `{"cache":99,"fresh":24}` | `{"0":4422,"1":4438,"2":4461,"3":4503,"4":4401,"5":4535,"6":4523,"7":4283}` | True | False |
| 3 | Adaptive: event-driven, min 100 ms | 150 | 37137 | `{"none":150}` | `{"0":3688,"1":2992,"2":2936,"3":3232,"4":4732,"5":12025,"6":2424,"7":5108}` | True | False |
| 4 | Baseline | 188 | 46502 | `{"none":188}` | `{"0":46502}` | True | False |
| 4 | Layer once: TTL 5 ms | 288 | 56963 | `{"cache":248,"fresh":40}` | `{"0":7335,"1":6979,"2":7102,"3":7024,"4":7314,"5":6763,"6":7316,"7":7130}` | True | False |
| 4 | Adaptive: event-driven, min 100 ms | 184 | 45319 | `{"none":184}` | `{"0":4928,"1":6872,"2":2080,"3":3424,"4":2584,"5":12225,"6":9038,"7":4168}` | True | False |
| 5 | Baseline | 295 | 61475 | `{"none":295}` | `{"0":61475}` | True | False |
| 5 | Layer once: TTL 5 ms | 235 | 48816 | `{"cache":190,"fresh":45}` | `{"0":5990,"1":6186,"2":6128,"3":6076,"4":6128,"5":6238,"6":6045,"7":6025}` | True | False |
| 5 | Adaptive: event-driven, min 100 ms | 245 | 40843 | `{"none":245}` | `{"0":2031,"1":9162,"2":3552,"3":2384,"4":6888,"5":13082,"6":1504,"7":2240}` | True | False |

## 可逐式复算的 Adaptive 决策样例

下面每个 SSU 确定性取测量窗内最早一个有 admission attempt 的候选（若该模式无需 attempt，则取最早 selected NPU）。`target = effective_ratio × manifest demand`；`normalized_total = Σ(target_ssu / 40)`，`normalized_dominant = max(target_ssu / 40)`。attempt 的 before/after 和 violating SSU 解释‘为何选中/拒绝’，grant 分量解释‘最终给多少’。slack 不参与任何一步。

### SSU 3

evaluation `131`，t=`0.034s`，NPU `10` / request `426`；mode=`greedy_overload`，residual=`v1_coflow_residual`，selected=`True`。

- demand=`[2.269339689193726,2.346080644866944,2.3131916638641363]` GB/s；ratio=`0.52`；target=`[1.1800566383807376,1.2199619353308109,1.2028596652093508]` GB/s。
- normalized total=`0.090071956`，dominant=`0.0304990484`；candidate rank=`0`。
- final grant=`[1.2350824504672142,1.2768485236714195,1.2589487780124742]` GB/s；逐 cell decomposition closure 见 `adaptive_grant_components.csv`。
- attempt stage=`greedy_candidate`，accepted=`True`，before=`[27.019224967425416,25.892324848947624,26.340300228483898]`，after=`[25.83916832904468,24.672362913616812,25.137440563274545]`，violating SSU=`none`。

### SSU 4

evaluation `111`，t=`0.094s`，NPU `10` / request `522`；mode=`greedy_overload`，residual=`v1_coflow_residual`，selected=`True`。

- demand=`[0.419049557876587,0.3989351790985108,0.37882080032043464,0.41234476495056155]` GB/s；ratio=`0.52`；target=`[0.21790577009582524,0.20744629313122562,0.19698681616662603,0.214419277774292]` GB/s。
- normalized total=`0.0209189539`，dominant=`0.00544764425`；candidate rank=`0`。
- final grant=`[0.24811430532630907,0.23620481867064624,0.22429533201498342,0.2441444764410881]` GB/s；逐 cell decomposition closure 见 `adaptive_grant_components.csv`。
- attempt stage=`greedy_candidate`，accepted=`True`，before=`[19.907188561271624,20.720113231254395,22.082852215899653,21.810754059421622]`，after=`[19.6892827911758,20.512666938123168,21.885865399733028,21.59633478164733]`，violating SSU=`none`。

### SSU 5

evaluation `101`，t=`0.137s`，NPU `16` / request `624`；mode=`greedy_overload`，residual=`v1_coflow_residual`，selected=`True`。

- demand=`[1.688283285095215,1.4799419435302734,1.2356796810058595,1.4655735751464845,1.4296526541870118]` GB/s；ratio=`0.52`；target=`[0.8779073082495118,0.7695698106357423,0.6425534341230469,0.7620982590761719,0.7434193801772462]` GB/s。
- normalized total=`0.0948887048`，dominant=`0.0219476827`；candidate rank=`0`。
- final grant=`[1.0829858680584001,0.949340803489491,0.7926534864087013,0.9401239024847388,0.9170816499728579]` GB/s；逐 cell decomposition closure 见 `adaptive_grant_components.csv`。
- attempt stage=`greedy_candidate`，accepted=`True`，before=`[33.21148467179346,33.51414681429102,33.82896643146362,33.702755673432996,33.20951599011266]`，after=`[32.333577363543945,32.74457700365527,33.186412997340575,32.94065741435683,32.46609660993542]`，violating SSU=`none`。


## 窗口效应

0.5/1/2/3/8/16/32/64 秒均同时输出 cumulative 和 disjoint 统计。利用率是该时间窗内严格 resource-time 积分；SLO 则按 admission time 落在半开窗内选择 cohort，TTFT 允许在该窗结束之后完成，所以它不是‘截至 3 秒已完成请求’的在线 SLO。3 秒不能整除 64 秒，CSV 保留最后 1 秒并标记 `full_requested_window=false`；min–max 图只使用完整 disjoint 窗，避免把 1 秒尾窗伪装成 3 秒窗。短窗差异是系统相位、carry-in 库存 Q、请求 profile 和排队状态的混合；不能仅凭一个短窗断言长期吞吐改变。

## 微观 dispatch 证据与限制

共导出 90000 条 first-50-ms capped dispatch prefix 记录；每个 case 最多 10000 条，`probe_stream_truncated=True` 仅表示 record cap reached，不能据此声称观察到了第 10001 条，也不能把该前缀称为完整 50 ms dispatch 序列。这些记录是 simulator 在 dispatch 前的 read-only winner prediction，随后由 runtime assertion 与实际 Path 核对，并不是分析器独立重演完整调度器。CSV 保留 cap flag、candidate path count、virtual finish、RR cursor、estimated arbitration rate、CIR、group、queue wait、request/layer/block 和 physical command rate。
另导出 1807 条 first-50-ms capped route prefix 记录；每条已记录 route plan 的纯策略函数重放均逐 block exact-match，fresh/cache age 与 selected Path/group 映射均已硬验收。具体 Path 的 portable replay 只覆盖 measurement 开头 50 ms 内已记录、最多 10000 条的前缀；若 cap reached 则不声称完整覆盖这 50 ms，更不能外推为 64 秒逐 dispatch Path 重放。全 64 秒仅有每 500 ms 的 group/fresh/cache 累计计数及其差分证据。

## 模型真实性边界

这里的 NPU utilization 是仿真器对 compute interval 的 binary duty-cycle 积分，不是芯片性能计数器。每个 SSD 同时最多执行一个 40 GB/s 非抢占 command；CIR 只影响下一条离散 command 的仲裁，不是多个 IO 并发时按比例持续限速。每个 NPU link 也是单流 50 GB/s。层计算时长与 profile 固定，模型没有真实 HBM、网络、缓存和内核执行抖动。因而这些时间线可以严格证明仿真内部的状态、守恒和决策链，却不能替代真实硬件的外部测量；这些离散且固定的结构也会让不同策略的长期总 NPU duty cycle 更容易接近。

本报告能严格证明这一个固定 workload/seed/64 秒轨迹中的守恒关系和调度机制；它不能把单 seed 结果提升为总体概率结论。若用于最终对外结论，仍应追加多 seed 置信区间和更长稳态窗。

## 产物索引

- `00a_mean_npu_utilization_vs_ssu.png`、`00b_ttft_slo_alpha1p5_vs_ssu.png`、`00c_ttft_slo_alpha2_vs_ssu.png`: 64 秒全窗的三策略 SSU3/4/5 汇总曲线；SLO 是 equal-observed-NPU 口径。
- `timeline_npu_ssu_500ms_ssu{3,4,5}_{case}.csv`: 每个策略、NPU、SSU、500 ms 的 demand/CIR/interval-average attributed SSD service/link delivery/slack/queue/route 证据；按 SSU×策略拆分以保持单文件适合 GitHub。
- `window_metrics.csv`: cumulative + disjoint 多尺度利用率与 α1.5/α2 SLO。
- `state_durations_per_npu.csv`、`state_duration_summary.csv` 与 `state_durations_500ms.csv`: 包含 carry-in 的完整窗口及 128 个精确块 compute/IO-barrier/other 主根因分解。
- `compute_inventory_decomposition.csv`: Q/activated compute 精确分解。
- `matched_requests.csv` 与 `matched_request_layers_ssu{N}_{case}_npu{lo}_{hi}.csv`: 同请求及逐层 barrier 时间；逐 SSU×策略×8-NPU 分片，保证 GitHub 单文件体积。
- `forensic_selection.csv`、`forensic_timeline.csv` 与 `09_forensic_zoom_ssu{3,4,5}.png`: 确定性 full-lifecycle matched 请求的 16 层 Gantt 与 request-ID 绑定后的局部证据；SSD/link 明确是 NPU aggregate，不能误称单请求归因。
- `adaptive_decisions.csv`, `adaptive_decision_npu.csv`, `adaptive_admission_attempts_ssu{3,4,5}.csv`, `adaptive_grant_components.csv`, `adaptive_boundary_slack.csv`: 决策因果输入、candidate order/rank、attempt/capacity rejection、V1/V2 grant 分量、重建 CIR 和 diagnostic-only slack。
- `path_state_500ms_ssu{3,4,5}_{case}_disk{physical_ssu}.csv`: 按拓扑×策略×物理 SSU 拆分的 500-ms 左边界 sparse Path queue/CIR/virtual-finish/下一次仲裁速率估计；这是边界状态与 next-arbitration estimate，不是前一个区间历史 winner。拆分后即使 256 Path 全部活跃，单文件也有严格行数/体积预算。
- `dispatch_probe.csv`: 开头 bounded 窗内 simulator pre-dispatch read-only prediction 与 runtime winner assertion。
- `route_probe_ssu{3,4,5}.csv`: 开头 50 ms 内、最多 10000 条的 capped route prefix，含 cap flag、路径规划输入、fresh/cache TTL、selected Path/group 与逐记录纯策略函数 exact replay；若 cap reached，不声称完整覆盖该 50 ms。
- `ssd_accounting_residuals.csv`: v3 stable-service/direct-physical-queue 的逐边界逐 SSU residual 及容差证明。
- `analyze_npu32_ssu345_timeline64.py`: 与 validation/results/manifest 中 analyzer SHA-256 完全一致的分析器字节。

验证指纹：source `76658d5aac81f11a8f9fddcfeaed6692821aefde3892cdc0d7dd564563242568`；config `0d85b34ec9d1044d2c52e496b0f4e4a8b4ab13e0598c21db7e7621848e04c288`；analyzer `bfcbec54b2f101e5970074b1b717e1544d900e0a3beb67656c265dce4630fa94`。
