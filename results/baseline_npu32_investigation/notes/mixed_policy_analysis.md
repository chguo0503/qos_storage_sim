# 每卡混合画像的 S1–S3 补充：独立最终审核

新增 8/8 作业全部完成并通过独立结果审核，另读取主实验的 6 份 Baseline/Once/New Once 参照；没有将这 8 份加入原 mixed_varied 的 36 作业分母。完整表和原始结果 SHA256 在 [mixed_policy_audit/summary.md](mixed_policy_audit/summary.md)、[runs.csv](mixed_policy_audit/runs.csv) 与 [audit.json](mixed_policy_audit/audit.json)。

两个输入为 E72=`aligned_short072_seed7`、V38=`raw_size_varied_rho097_seed7`。每个输入分别运行 S1/S2/S3 pipeline 和 S3 fixed。E72 的 1K 短画像使用原模型的外推/NQL 插值，V38 的画像直接来自 data；两者都不额外缩放 C。请求频率、完整乱序序列和全体 t=0 到达是实验构造，不是观测到的生产轨迹。V38 的 64K/NQL512 属模拟器 SL；分析用的“较短画像组”不能全部称为 SS。

## 比较和完整性

每个新增策略分别对原 Baseline、Once、New Once 三份参照配对，共 24 个完整运行比较、108 个同画像完整 cohort 比较。两组所有 14 份结果的两个公共窗口均为全体 32 卡持续 active，共 28 窗。窗口固定为 1–2s、2–3s，不另选有利区间。

独立检查包含：原 manifest 和复制件的完整请求/placement 精确相等、输入 fingerprint、所有标记 direct_data_row 的 C/V/TTFT 与原 data 精确一致、每卡完整 shuffle 重放、完整 ID population、逐项 core_and_policy_sha256 与现冻结源码一致、stress runner SHA、策略与 5ms collector、提交种子、完整读量与全部 invariants。外推画像保留其 construction 来源并核实跨策略 C/V 相同，不把它们称为原 data 直接测量行。窗口由实际执行 NPU 的逐层时间段重新积分；画像按原 manifest 的完整 request-ID cohort 分组，包含所有请求，不筛选某个策略先完成的样本。

Pipeline 允许改变执行 NPU，逐条 assignment log 与实际 request/batch NPU 一致；原始数据在 SSD 上的 placement 保持。V38 的 S1/S2/S3 pipeline **各自都是 2870/2944 请求重分卡**，E72 各自都是 18029/18624；fixed 策略均为 0。计数相同本身不是逐请求分配映射相同的证明。实际分卡后的 mean 最热盘 rho：V38 三 pipeline 均 0.973333738，fixed 0.973031931；E72 分别为 0.935338212、0.935417087。这仍只是完整输入平均容量的必要条件，不能保证局部突发或截止期均可满足。

## 两个输入的全策略结果

表中 U 为两公共窗整机利用率，T 为完整批次 makespan。完整运行的 fleet U 与各类请求的 compute/active 另列于 CSV，不能混用分母。

|策略|E72 U1/U2|E72 T ms|V38 U1/U2|V38 T ms|
|---|---:|---:|---:|---:|
|Baseline|80.292% / 82.724%|5489.518|96.235% / 96.661%|6520.496|
|Once|90.677% / 92.904%|4891.038|96.365% / 97.028%|6520.297|
|New Once|91.620% / 93.804%|4785.350|97.676% / 97.416%|6489.609|
|S1 pipeline|93.478% / 94.424%|4817.828|95.675% / 96.697%|6740.878|
|S2 pipeline|92.851% / 93.748%|4850.463|95.092% / 97.210%|6749.121|
|S3 pipeline|93.977% / 93.620%|4831.173|94.940% / 97.262%|6700.928|
|S3 fixed|90.171% / 93.450%|4839.875|97.832% / 98.129%|6465.262|

E72 完整批次由快到慢：New Once、S1 pipeline、S3 pipeline、S3 fixed、S2 pipeline、Once、Baseline。四个补充策略都改善 Baseline，但均比 New Once 完整排空慢 0.679%–1.361%。S2/S3 pipeline 的第二窗还略低于 New Once，S3 fixed 两窗都低于 New Once。不能用第一窗优势代替完整批次排名。

V38 完整批次由快到慢：S3 fixed、New Once、Once、Baseline、S3 pipeline、S1 pipeline、S2 pipeline。三个 pipeline 第一窗 U 都低于 Baseline，完整批次也分别慢 3.380%、3.506%、2.767%；但三个 pipeline 的完整短组 compute/active 与短组接纳 SLO 均提高。完整批次速度、活动期效率和短流 SLO 在这里给出不同排序。

## 同画像的收益和代价

E72 的三种 SS 在四个补充策略下接纳 SLO 全部 100%，compute/active 各约 99.69%–99.95%。长 192K/NQL256 则由 Baseline 的 93.404% 降到 81.057%–82.647%，接纳 SLO 从 576/576 降到 518/576–538/576。对短流的保护伴随长流等待增多；原始逐类 L0/warm stall 均保存在 profiles.csv。

V38 各短画像请求数均为 896。下表为完整相同 ID cohort 的 compute/active；括号为接纳 SLO 通过数，分母均 896。

|策略|32K/128，SS|48K/256，SS|64K/512，SL|
|---|---:|---:|---:|
|Baseline|57.803% (483)|86.925% (817)|98.261% (896)|
|Once|76.039% (709)|94.505% (878)|89.215% (843)|
|New Once|92.446% (870)|99.488% (894)|86.528% (808)|
|S1 pipeline|87.531% (837)|96.477% (882)|92.031% (869)|
|S2 pipeline|88.458% (852)|96.185% (883)|92.215% (869)|
|S3 pipeline|88.964% (855)|95.345% (883)|92.747% (868)|
|S3 fixed|88.308% (839)|94.248% (874)|91.261% (824)|

S3 fixed 比 New Once 完整排空更快，两个窗口也更高，但前两种短画像的计算比和 SLO 更差，第三种更好；192K/NQL1024 的接纳 SLO 为 61/64，而 New Once 为 64/64。因此最快策略没有同时保护所有画像。各 pipeline 相比 New Once 也在这三类之间重新分配等待，且其长画像 compute/active 均下降。全部正负比较保留于 profile_pairs.csv。

## V38 的排空代价来自哪段时间

独立脚本 [decompose_mixed_policy_idle.py](decompose_mixed_policy_idle.py) 按实际执行卡对完整批次验证：

`32 × makespan = compute + IO stall + startup idle + request-gap idle + tail idle`。

14 份结果的 startup idle 和 request-gap idle 均为 0；所有完整运行 idle 都发生在某张卡最后一个请求完成以后、整批结束以前。该结论来自完整每卡事件序列，不由公共窗 U 推断。明细与结果 SHA 在 [full_idle_decomposition.csv](mixed_policy_audit/full_idle_decomposition.csv) 和相邻 JSON。

|V38 策略|总 compute NPU·ms|总 IO stall NPU·ms|尾部 idle NPU·ms|
|---|---:|---:|---:|
|Baseline|195344.630|10573.021|2738.212|
|Once|195344.630|10262.190|3042.681|
|New Once|195344.630|9212.016|3110.831|
|S1 pipeline|195344.630|8436.231|11927.243|
|S2 pipeline|195344.630|7847.779|12779.469|
|S3 pipeline|195344.630|7736.792|11348.277|
|S3 fixed|195344.630|8481.232|3062.528|

S3 pipeline 相比 S3 fixed 的总 IO stall **减少 744.439481 NPU·ms**，尾部 idle **增加 8285.749208 NPU·ms**，合计增加 7541.309727 NPU·ms，恰为 `32 × 235.665929 ms` 的完整 makespan 差。不能把这项排空退化说成总 IO stall 上升。

三个 pipeline 中实际各卡的完整纯计算量均在 5949.332–6441.238ms，原 fixed 每卡均为 6104.520ms，显示重分卡后完整负载失衡。尾部 idle 时间账能定位差额；它不单独证明全部尾差都由这项纯计算失衡造成，也不能隔离 I/O 排序、入队相位、选卡三者的独立因果贡献。两公共窗全部 active/idle=0，完整尾部现象也不能用来解释它们的所有局部窗口差异。

## 执行记录与复算

原 sweep 状态保存 8/8 complete、0 failed；其实现只在仿真子进程返回 0 且完整结果存在后写 complete。外层工具会话 73185 收尾返回 143，终止来源未确定；command.json 不保存 returncode，不能声称该文件直接证明返回码。该外层执行通道事件及独立源码补核见 [execution_incidents.json](../ops/execution_incidents.json)。八个结果均独立验证通过，无重跑，不因此推断外层会话成功退出。

```bash
python results/baseline_npu32_investigation/notes/aggregate_mixed_policy_profiles.py
python results/baseline_npu32_investigation/notes/decompose_mixed_policy_idle.py
```

两输入的完整七策略画像图均提供 PNG、SVG、PDF：`mixed_policy_audit/*_all_policy_profiles.*`。本记录和脚本只分析已完成结果，不新增模拟，也不修改任何 root Python 源码。
