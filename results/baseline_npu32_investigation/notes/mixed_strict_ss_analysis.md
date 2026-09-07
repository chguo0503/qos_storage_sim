# 三种真实 SS 短画像的独立语义控制

两个 frozen inputs、baseline/Once/New Once 三策略，**6/6 完成并独立审计通过**。每卡32K/NQL128、48K/NQL256、64K/NQL128各27请求，三者按模拟器seq<=80K且NQL<512的规则均为SS；另有192K/NQL1024、2048各3、9请求，均为LL。每卡5画像、93请求，32卡共2976请求；seed7与123独立打乱完整序列，全部arrival=0。

所有画像均直接来自原data，未缩放C、未补齐尾块。逐请求身份、完整配额、逐块placement、完整shuffle重放、原始data数值、源码每项SHA、提交种子、5ms与实际fixed分卡核验通过。Mean最热盘rho=0.951899847、linkrho=0.190379969；每卡ideal compute6948.737ms。三SS占总C17.704988%，192K2048单类占70.5171%。12个公共窗口全部32卡全程active。

这不是原V38输入只改标签的实验：64K的NQL/C/V、配额、短C份额及完整序列均变化。因此它验证的是同属SS时是否仍存在服务差异及策略取舍，不能独立归因原V38结果是SS/SL分类所致。

|seed|策略|1–2s fleet U|2–3s fleet U|full makespan ms|完整SS组cohort U|SS组admission SLO|
|---|---|---:|---:|---:|---:|---:|
|7|baseline|96.5079%|94.2346%|7519.960|75.3800%|1939/2592|
|7|Once|98.0039%|95.1858%|7434.959|81.1943%|2148/2592|
|7|New Once|97.8237%|94.5049%|7461.440|86.8384%|2306/2592|
|123|baseline|95.7758%|97.7350%|7613.860|72.3270%|1878/2592|
|123|Once|97.3397%|98.1085%|7541.353|76.8962%|2020/2592|
|123|New Once|96.7247%|99.0898%|7585.642|80.4592%|2147/2592|

逐SS画像的完整同request-ID cohort compute/active：

|seed|画像|baseline|Once|New Once|
|---|---|---:|---:|---:|
|7|32K128|64.1745%|69.0435%|75.7393%|
|7|48K256|89.1201%|90.2786%|95.4483%|
|7|64K128|67.8330%|78.5929%|83.7544%|
|123|32K128|60.6170%|64.0780%|67.5742%|
|123|48K256|87.2844%|87.6583%|91.2026%|
|123|64K128|64.3344%|73.2527%|76.7357%|

因此 baseline 的画像服务差异和长C时间掩蔽，在真正三SS输入里仍然存在。Seed7的32K128平均stall5.261ms中4.858ms在warm阶段，64K128的7.015ms中6.160ms在warm阶段；两个LL长画像warm stall均0。这些结果定位等待阶段，不单凭摘要断言某条路径的因果机制。

两策略都改善了三SS各自的**cohort U**，但这不保证各类SLO同时提高：seed123的48K256 baseline通过803/864，Once降为786/864，New Once为792/864，分别少17、11请求（-1.9676/-1.2731个百分点）；同时该类cohort U反而从87.2844%提高到87.6583%/91.2026%。均值与阈值尾部的方向并不必然一致，这个同SS小负例完整保留。

New Once的三SS均值优于Once，但两个seed完整makespan都慢于Once；seed7两公共窗U也都低于Once，seed123第一窗低于Once、第二窗更高。相应长LL有代价：seed7 192K1024/2048的cohort U从baseline98.6907/99.3054%降为New Once93.5043/97.7416%，seed123亦类似。长LL的admission SLO仍全100%，并不表示它们没有新增等待。

可复算：

```bash
python results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py --study results/baseline_npu32_investigation/mixed_strict_ss --output results/baseline_npu32_investigation/notes/mixed_strict_ss_audit --require-short-category SS
```

全部30画像行、20画像配对、12窗口、384个卡×窗口观测、源码及源文件SHA、完整SLO/分位数/L0-warm分解，在 `mixed_strict_ss_audit/`；其中PNG/SVG/PDF全画像图明确保留LL代价。该控制独立于主实验36作业。
