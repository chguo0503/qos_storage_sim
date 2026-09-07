# 每卡多种短画像持续混合：独立审计与结果解读

本审计读取 `mixed_varied/plan.json` 与冻结 manifest、完整结果，不运行仿真，不修改根目录代码。最终 **12 输入、36 策略作业全部完成并独立审计通过**，0 pending/failed/invalid；36 份结果共完成316992请求，得到168个画像cohort、112个同画像策略对照。72个公共窗口全部32卡全程active。全部数字、源文件及源码SHA见 `mixed_profile_audit/summary.md`、CSV/JSON，全部图输出PNG/SVG/PDF。24组相对baseline的整机对照中，7组至少一个窗口U更低或完整makespan更长；`fleet_pairs.csv`保留全部24组，无阈值筛去小负例。

## 输入符合什么要求

12 个输入分成 6 族，每族 seed7、123。每卡完整请求多重集先生成，再用 `random.Random(seed+npu_id*100003).shuffle` 独立打乱整个序列；不是打乱一个小 deck 后重复。独立逐卡重新生成序列、核对全部 request ID、画像配额和每一块的 placement，32 条 lane 顺序均互不相同，全部匹配。不同策略处理同一个 frozen input fingerprint 及每类完全相同的 request-ID cohort。

|输入族|每卡画像|短组计算份额|最热SSU mean rho|每卡最少 ideal compute ms|
|---|---|---:|---:|---:|
|aligned_short072|1K/128、256、384 + 192K/256|71.8164%|0.935417|4449.417|
|aligned_short080|同上|79.8573%|0.709111|4150.393|
|raw_equal_rho090|32K/128、256、512 + 192K/1024、2048|32.9176%|0.899037|5683.027|
|raw_equal_rho097|同上|37.7399%|0.967915|6123.196|
|raw_skew_rho096|同上，偏重32K/512|58.1766%|0.961327|5862.194|
|raw_size_varied_rho097|32K/128、48K/256、64K/512 + 192K/1024、2048|37.5494%|0.973032|6104.520|

`aligned` 的三个 1K 画像由 data 外推；其 192K 长画像及全部 `raw` 画像直接使用 data 行。独立读取 data 原始字典核对 direct 行的 C/V/带宽/TTFT。全部计算缩放系数为 1，128-token 整块、无 padding；逐卡总 C、逐块总读量、各 SSD/link mean rho 均独立重新求和通过。

“短组”是 spec 指定的较短画像集合，不等于模拟器 SS 类别。32K/128、48K/256 属 SS；32K/512、64K/512 属 SL。对比报告始终保留具体 seq/NQL 和 C 份额，不能把分析分组当作同一 IO 路径类别。

所有输入是人工频率和排序、arrival=0 的有限积压；不是生产流量分布。Mean rho 可行只排除了平均容量不足，不保证每次层截止期可行。两个种子提供有限的复现证据，不代表总体统计显著性。

## 结论确实需要随着新输入更新

此前固定角色输入约 52% 的 fleet U，不能直接推广到“每卡不停切换多种短画像并混入长画像”的新输入。在 aligned_short072 中，baseline 两个种子、两个窗口为 80.292–82.724%；全部卡整个窗口 active，损失已分布到多数卡，不再是固定短卡/长卡的两群角色。三种短画像的完整 cohort U 仍约 70%、78–79%、85–86%。

在真实 data 参数的 size-varied 族，baseline 两个种子两个窗 fleet U 为 95.884–96.662%，但最短 32K/128 的完整 cohort U 只有 57.803/58.104%，admission SLO 为 53.906/49.888%。Seed7 甚至两个窗最差 NPU U 仍有 92.387/92.646%：每卡都穿插长 C 请求，查看最差卡仍无法揭示每类短请求的损失。

这不是仅仅把所有短流合并就能消除的掩蔽。raw_skew096 的短组虽然占 58.18% 总 C，其中 32K/512 一类就占 52.54%，128/256 两类合计仅 5.63%；seed7 baseline 短组整体 U95.509%，最短 32K/128 却只有 61.377%、SLO52.163%。需要逐画像，而不只是 fleet 或“所有短请求”两种平均。

## 策略保护了什么，代价在哪里

同一个 aligned_short072 seed7 输入，baseline / Once / New Once 两个窗口 U 分别为 80.292/82.724、90.677/92.904、91.620/93.804%。三种短画像 U 从 69.911/77.994/84.781% 提升到 Once 的 97.189/98.592/99.058%，New Once 约 100%；seed123 复现。New Once 三短的 warm-layer stall 为零、admission SLO 全部通过。

但长 192K/256 有实际代价：seed7 的 cohort U 从 baseline93.404% 降为 Once81.217%、New Once82.302%；SLO 从 576/576 降为 500/576、516/576。两种策略完整 makespan 仍从 5489.518ms 降为 4891.038/4785.350ms。应同时报告总体收益和长画像退步。

真实 size-varied seed7 则出现**短组内部**的代价：

|画像|baseline U|Once U|New Once U|baseline SLO|Once SLO|New Once SLO|
|---|---:|---:|---:|---:|---:|---:|
|32K/128|57.803%|76.039%|92.446%|483/896|709/896|870/896|
|48K/256|86.925%|94.505%|99.488%|817/896|878/896|894/896|
|64K/512|98.261%|89.215%|86.528%|896/896|843/896|808/896|

Seed123 的 64K/512 同样从 U98.096%、SLO100% 降为 Once90.443%、95.424%，New Once87.636%、89.063%。所以“保护短流”不等于每一种较短画像都改善。

64K/512 seed7 的完整请求平均 stall 为：

|策略|总stall ms|L0 stall ms|后7层warm stall ms|
|---|---:|---:|---:|
|baseline|0.904136|0.788123|0.116013|
|Once|6.176463|1.873801|4.302662|
|New Once|7.954475|2.410770|5.543705|

新增等待约 79.4%/77.0% 发生在 warm 阶段。与此相对，baseline 的最短 32K/128 原本 6.880ms stall 中，6.436ms（93.55%）就是 warm 等待。逐层分解定位了等待发生阶段；这些摘要本身不足以判定某一具体路径或队列机制的因果贡献。

真实 size-varied seed7 的 full makespan 6520.496→6520.297→6489.609ms，总体收益很小；raw_equal090 seed7 的 Once 还有小负例：1–2s U93.217→92.861%，full makespan6054.620→6060.963ms，虽然 2–3s U97.624→98.601%。不应只挑获益的窗口。

raw_skew096 seed7 的 Once 是更明显的总体负例：两个窗 U96.848/98.379→95.613/96.440%，full makespan6088.923→6193.890ms（增加1.724%），短组整体 cohort U95.509→93.370%。最短128/256分别提升至97.401/99.669%，但占52.54%总C的512从98.497降为92.822%。New Once 的第二窗也略低于baseline（97.874%），完整 makespan 则稍快至6084.129ms。raw_equal097 seed7 Once 两个窗口 U 都改善，完整 makespan 却6523.849→6545.316ms（增加0.329%）。这些结果直接否定“只要baseline受损，其余策略所有指标都会变好”的概括。

## 全给短流也必须先看物理容量

独立 `mixed_controls` 中只有三种 1K 短画像，baseline/Once/New Once 两公共窗均 100%、admission SLO 均 100%；该控制 mean disk rho 仅 0.1422。因此随机多短画像本身未必导致低利用率。它改变了完整配额和负载，不是相同负载下只移除长流的严格因果实验，详见 `mixed_control_analysis.md`。

相反，若从真实 size-varied 输入删除两个长画像，保留原三种短画像等配额，则独立按 data 与 frozen manifest 逐块求和得到 mean rho=1.87807。保留的 2688 请求总读量为 1377.578125GiB，总 C 为 73350.728ms，40GiB/s×8 SSD 给完整 makespan 下界 4304.932ms，完整 fleet U 上界 53.2461%。这是任何策略均受约束的完整运行总字节界；不是任意局部窗口的界，也不是新仿真。证据与独立复算脚本为 `raw_allshort_capacity_independent.json`、`audit_raw_short_capacity.py`。

## 可复算与图

额外严格三SS语义控制的6份结果另见 `mixed_strict_ss_analysis.md` 与 `mixed_strict_ss_audit/`；S1–S3选卡机制扩展另见 `mixed_policy_audit/`。它们不加入本节36作业分母。

```bash
python results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py
python results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py --study results/baseline_npu32_investigation/mixed_controls --short-only-control
python results/baseline_npu32_investigation/notes/audit_raw_short_capacity.py
```

主实验输出 `mixed_profile_audit/`：输入审计、逐画像完整 cohort、同 ID 配对、窗口画像时间权重、每卡数据、全部计划状态 CSV、完整 JSON 与源文件 SHA256。每一个已完成结果还逐项核对 core_and_policy_sha256、stress runner、提交种子、5ms、实际 fixed NPU 绑定、完整请求数、总读量、所有 invariants，以及逐层独立积分的 U 与原结果一致。

`*_short_profiles` 图给出三短的 U/SLO/stall；`*_all_profiles_tradeoff` 图包含每种画像及总 C 份额，显示长类或短组内部代价。全部输出 PNG/SVG/PDF；图中缺策略明确 pending。完整 request-ID cohort 避免按各策略窗口完成样本筛选；两个窗口始终保持 1–2s、2–3s。
