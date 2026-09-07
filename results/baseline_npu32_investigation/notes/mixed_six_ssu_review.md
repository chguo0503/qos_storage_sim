# 32 NPU / 6 SSU 多短画像收尾复核

按限定范围完成 **2 输入 × Baseline / New Once（5 ms）× seed 7，共 4 jobs**，本机 Python 3.10.10、2 workers。没有扩展策略/seed，没有修改原 helper 或任何 root `.py`。4/4 成功，所有结果 invariants、完整 request-ID cohort、原 NPU 绑定、实际读取量、同机配对和冻结源码哈希审核通过。

这次补充支持两个不同结论：短计算占比较高的 E80 在 6 盘下仍可明显降低 Baseline 的整体利用率，New Once 能恢复大部分损失；原始 data 的多长度输入则再次出现“整体利用率高，但最短类服务差”。New Once 的改善存在代价，包含长类以及 short 池内相对较长的 64K/512 类。

## 输入与公平性

新 [wrapper](../mixed_six_ssu/run_probe.py) 只在自己的 Python 进程中设置 `run_mixed_sustained_probe.NUM_SSU=6`，调用原 `build_input`；manifest 的硬编码 placement 描述已更正为 `ssu=(block_index+original_npu_id//4)%6`。使用原 stress runner 执行，保留每个 job 的 command、Python/NumPy/platform、PID、返回码、wall time 和原始压缩结果。

| 输入 | 每卡画像与请求配额（内部单位） | 请求总数 | 每卡理想计算量 | short 计算份额 | 最热盘 ρ | 最大接收链路 ρ |
|---|---|---:|---:|---:|---:|---:|
| A：aligned_short080_ssu6_seed7 | 1K/128, 1K/256, 1K/384, 192K/256 = 65:65:65:4 | 19,104 | 4150.393 ms | 79.857256% | 0.952049 | 0.141822 |
| B：raw_size_varied_ssu6_seed7 | 32K/128, 48K/256, 64K/512, 192K/1024, 192K/2048 = 6:6:6:1:3 | 2,112 | 7192.028 ms | 20.488849% | 0.968765 | 0.145158 |

两组每卡均完成 3 个配额单位，随后按 `seed+npu*100003` 独立 shuffle **整个完整人口**，32 卡有 32 种不同完整顺序；每张卡都会运行全部画像。所有请求 t=0 到达，满足每卡完整理想计算量 `≥ 4000 ms + 2×最大单请求8层计算量`。这些是合成的有限饱和 backlog，不是生产到达记录。所有 C 均保持 helper 原始输出、`compute_scale_actual=1.0`。A 沿用既有 aligned 构造画像，不能称为原始 data 行；B 的五个 C/V 均直接取原始 data，没有插值、缩放或 padding。

逐盘理想平均读取需求（GiB/s）独立重算为：

- A：`[37.814092, 38.081975, 37.829622, 37.829622, 37.814092, 37.546210]`。
- B：`[38.701299, 38.750588, 38.743867, 38.723703, 38.681135, 38.652009]`。

32 卡仍分成 8 个 4 卡 group，在 6 盘下 offset 0、1 被重复，因此盘间并非完全等量；表中使用最热盘而非只看总量。所有盘都低于 40 GiB/s，接收链路也低于 50 GiB/s。通过完整输入的平均容量条件不保证各层 deadline，不构成某有限窗口利用率上界。

**A 是相同人口的拓扑对照。** 与原 8 盘 E80 逐条比较了 ID、原 NPU、arrival、完整 load 字典、列表顺序和全部 per-NPU deck hash，全部精确一致；logical fingerprint 相同，physical fingerprint 改变。**B 是重新配额的新人口**，不能说与 8 盘 raw size-varied 仅改变盘数。

构造审核见 [input_audit.json](../mixed_six_ssu/input_audit.json)；另一 agent 的独立逐块审核见 [independent_wrapper_review.json](../mixed_six_ssu/independent_wrapper_review.json)。

## 6 盘实测结果

下面的窗口 U 是真实 compute 时间 / `(32×窗口长度)`。全部 8 个结果窗口均为 **32 张卡全窗 active、idle=0**，因此窗口 U 的缺口是真实 exposed stall，不是队列耗尽。完整 cohort 的 compute/active 和 admission SLO 则是另一类指标，单独列在后文。

| 输入 | 策略 | [1000,2000] ms U | [2000,3000] ms U | 完整 makespan |
|---|---|---:|---:|---:|
| A：E80 | Baseline | 74.530090% | 80.112303% | 5391.765722 ms |
| A：E80 | New Once | 91.885123% | 97.609653% | 4473.272457 ms |
| B：raw 多长度 | Baseline | 95.717147% | 96.253392% | 7778.426334 ms |
| B：raw 多长度 | New Once | 98.412930% | 96.681138% | 7734.469367 ms |

A 的 New Once 分别改善 17.355033、17.497350 个百分点；B 分别改善 2.695783、0.427746 个百分点。B 的完整 makespan 只缩短约 43.957 ms，不宜把最短类的明显改善描述为整体完成时间大幅改善。

## 相同请求 cohort：改善与代价

admission SLO 定义为 `completion−admission ≤ 1.5×自身计算时间`，不是 arrival-to-completion SLO。每个策略使用完全相同的完整 request-ID cohort，包含尾部所有请求，不是只挑窗口内完成者。

| 输入/画像 | 该画像占全输入理想 C | Baseline compute/active | New Once compute/active | Baseline admission SLO | New Once admission SLO |
|---|---:|---:|---:|---:|---:|
| A：1K/128 | 19.842% | 66.337% | 99.993% | 3973/6240 | 6240/6240 |
| A：1K/256 | 26.207% | 74.485% | 99.995% | 4909/6240 | 6240/6240 |
| A：1K/384 | 33.808% | 81.662% | 99.995% | 5555/6240 | 6240/6240 |
| A：长 192K/256 | 20.143% | 93.255% | 77.938% | 384/384 | 297/384 |
| B：32K/128 | 2.359% | 43.884% | 88.418% | 178/576 | 552/576 |
| B：48K/256 | 5.343% | 73.525% | 97.779% | 433/576 | 569/576 |
| B：64K/512 | 12.787% | 94.778% | 87.942% | 559/576 | 520/576 |
| B：长 192K/1024 | 11.380% | 98.831% | 92.218% | 96/96 | 96/96 |
| B：长 192K/2048 | 68.132% | 99.435% | 97.756% | 288/288 | 288/288 |

A 的 short 池 compute/active 为 74.986%→99.995%，SLO 14437/18720→18720/18720；长类平均 stall 从 5.039→19.721 ms。B 的 short 池合计为 78.401%→90.369%，SLO 1170/1728→1641/1728，但 **64K/512 虽被归为 short，实际变差**：平均 stall 从 2.815→7.005 ms。B 两个长类平均 stall 也分别从 3.226→23.022 ms、3.094→12.498 ms。

B 的 Baseline 整体约 96% U 与最短类 43.884% compute/active 并不矛盾：最短 32K/128 只占全输入 C 的 2.359%，而两个 192K 长类合计约 79.511%。不能把整个 20.489% 的 short 计算份额都称为最短请求。New Once 明显改善最短两类，但不保证 short 池内每一类和所有长类均改善。

## A：与原 8 盘 E80 的同机拓扑对照

原 8 盘的两个参考结果也来自本机 `mixed_varied/local`。除逐条输入的非 placement 字段精确一致外，逐策略核对了 core/policy hash、stress runner hash、Python/NumPy/platform、submit seed、collector interval 和 policy_config，全相同。

| 策略 | 8 盘两窗 U | 6 盘两窗 U | 6−8 盘 U（百分点） | makespan 8→6 盘 |
|---|---|---|---|---|
| Baseline | 86.191154%, 91.963843% | 74.530090%, 80.112303% | −11.661063, −11.851541 | 4657.310→5391.766 ms |
| New Once | 98.426767%, 98.901822% | 91.885123%, 97.609653% | −6.541644, −1.292169 | 4236.997→4473.272 ms |

这说明该 E80 人口的 Baseline 损失不依赖于使用 8 盘拓扑，换到 6 盘反而更大。该变化同时改变盘数、总容量及具体 block-to-SSU 映射，所以只能归因于这里实现的整体拓扑/placement 变化，不能单独分离“容量减少”和“布局变化”各占多少。它也没有证明任何 32/6 输入都会产生同样损失。

## 完整运行时间账与实现耗时

下表单位均为每卡平均 ms，满足 `C + stall + idle = makespan`。本轮完整运行的 idle 都在各卡最后完成之后，故等于 tail idle；不能把它算作 stall。

| 输入/策略 | 平均 C | 平均 stall | 平均 idle (=tail idle) | makespan |
|---|---:|---:|---:|---:|
| A Baseline | 4150.393 | 1166.059 | 75.313 | 5391.766 |
| A New Once | 4150.393 | 236.822 | 86.057 | 4473.272 |
| B Baseline | 7192.028 | 443.484 | 142.915 | 7778.426 |
| B New Once | 7192.028 | 338.598 | 203.843 | 7734.469 |

B 的平均 stall 降约 104.885 ms，但平均尾部 idle 增约 60.928 ms，因此 makespan 只减少约 43.957 ms。这里不需要把窗口 idle 误算成 stall，也不需要假设每类请求都改善。

四格墙钟运行时间依次为 A Baseline 173.548 s、A New Once 524.417 s、B Baseline 249.801 s、B New Once 694.909 s。它们是本机执行离散事件程序的实际耗时，不是模拟控制延迟或 NPU 利用率损失。只跑了上述四格，没有增加仿真或修改核心以加速。

## 可复查产物

- [plan.json](../mixed_six_ssu/plan.json)：输入、四格计划、Python 环境、全部冻结 source/data hash。
- [analysis/audit.json](../mixed_six_ssu/analysis/audit.json)：重算输入、完整 cohort、逐画像、逐窗口、逐卡与 invariants。
- [analysis/pair_and_topology_audit.json](../mixed_six_ssu/analysis/pair_and_topology_audit.json)：两组策略同机配对、A 的两组 8→6 盘桥接、每卡完整运行时间账。
- [analysis/profiles.csv](../mixed_six_ssu/analysis/profiles.csv)、[windows.csv](../mixed_six_ssu/analysis/windows.csv)、[window_profiles.csv](../mixed_six_ssu/analysis/window_profiles.csv)、[cards.csv](../mixed_six_ssu/analysis/cards.csv)：全部结果含变差画像，不筛选有利结果。
- [source_integrity_after.json](../mixed_six_ssu/source_integrity_after.json)：107 项 source/data 记录运行前后未变。

wrapper SHA256：`18523ec0c454afa83f2bb81b58ab37ebeba4b993f333576c5a3ada9ce0328746`；原 stress runner SHA256：`94704edf908fbf90b9e39f5b26f94ad59c0c5dcfc4c1b981e1a2860921c2946d`；data SHA256：`fd197b79865b4c1f42d400100c5e05349ca1ba5f2d42b904af8a1759aabeb04b`。

A 6 盘 input fingerprint：`4d2fc8f78a881b92ecca84b6bf0764276f986c9bf1727a1309774176f9f5089a`；A 原 8 盘：`7f5f1d09bb5db576ec94ca5ef6addd94072c21d3a35dfae76375303769dbc938`；B 6 盘：`d5653af784d581fb61c67501f5b105a93d790fb506a1fa88ecbc51aad1b84195`。

可复算分析命令：`python -B results/baseline_npu32_investigation/mixed_six_ssu/analyze_probe.py`。该命令只读既有结果，不启动仿真。本次仅 seed 7 和两个策略，结论应限于这些人口与策略，不外推成六种策略的 6 盘全面排名。
