# 原始四画像替换：独立输入与结果审计

已完成 22/22 格，技术审计通过 22 格。主组 raw_feasible_q2 为预定五 seed × random/ordered × Baseline/Once，共20格；raw_nearest_q25 仅 seed7 ordered 两策略，是过载诊断。其 random manifest 只作排列基准，未计划运行，不计为缺失实验。

主组结果：Baseline五seed平均设备U由random的99.9993%变为ordered的99.8116%；ordered−random为-0.1876个百分点，逐seed相对下降的均值为0.1876%。本组通过全部研究条件的运行是20/20，全部预定seed均纳入统计，不按有效性筛选。

32 NPU、6 SSU、每请求8层；固定暖窗[2000,4000) ms。设备U=真实计算区间与暖窗交集总时长/(32×2000)，请求等权U另列。active包括计算与接纳后的I/O等待，接纳前排队不是I/O stall。
主SLO人口为暖窗内接纳的请求，逐条跟到完成；completion−admission ≤1.5×8C。这里是接纳后处理时间的TTFT代理，不含入卡前排队，也不使用原始数据的78层TTFT。所有arrival=0，因此暖窗arrival人口为空；完整输入人口结果另存CSV/JSON。不同顺序/策略的暖窗接纳人口可能不同。

主组每卡[14,14,14,7]共49请求、全局1568，纯C4949.368761 ms；packet=L+(S1 S2 S3)×2，纯C707.052680 ms、7 cycles。旧组每卡[200,200,200,8]共608请求：本组把每卡long数从8改为7，并改变short配额与总人口，不能声称两组是同一请求集合。短类纯C份额67.490889%，旧构造为67.467124%，差0.023766个百分点。原始C/V均未缩放、未填充；只在新组的random/ordered之间逐块保留原NPU的176KiB striping。
主组任意当前画像组合的逐盘名义V/C上界为39.630697580 GiB/s <40。诊断组四画像最小V/C=7.520933953，因此32卡active时总量至少240.669886488>240；任何排序都不可能满足原六盘欠载约束。这些是当前已接纳画像的名义需求，不是实际吞吐、突发I/O到达包络或deadline保证；跨请求L0不另叠加到该名义定义中。诊断组画像依次为32K/128、32K/256、32K/512、192K/1024，类别是SS/SS/SL/LL：第三短画像也已跨入SL，不能暗示最近原始键保留了原SS/LL分类。

本组是原始画像下的外部有效性对照，不是只移除外推的消融：短C增至旧值13.75–14.26倍，短V增至35.43–100.8倍；long/short C比由28.48–48.52降至2.276–3.959。长类占字节比例由77.27%降至36.05%，纯C加权rho6由0.562846升至0.893343。旧三短为SS，新三短为SL，Once合法Path池由96缩至32；LL仍32，Baseline仍Path0。5ms快照相对短C也从旧5.56–9.47倍变成0.40–0.69倍。

exact_cohort4_p1沿用同一构造规则，新实际切点[0,1,3,5]，纯C相位[0,229.856539,367.447084,526.512462] ms。组内8卡使用人工同步模板，不代表真实业务到达分布；五个ordered manifest的模拟输入指纹相同，不能称为五种独立坏顺序。随机组为五种完整人口独立shuffle。新短类别仍叫short仅为分析角色，其实际单层C为7–13ms。

有效性逐格报告：技术通过、第四请求在1500ms前完成、暖窗32卡全active、每卡都有short/long计算、全程逐事件盘/链路名义欠载。未满足warm mixed的seed也保留；不重采样、不按有效性筛选均值。每卡纯C>4s能排除提前跑空，不能保证暖窗出现long。

五seed等权均值 ± 样本标准差（n−1），包含全部预定seed，条件不通过也不剔除。

|顺序|策略|设备U %|暖接纳SLO %|全约束通过|
|---|---|---:|---:|---:|
|random|baseline|99.9993 ± 0.0006|100.0000 ± 0.0000|5/5|
|random|once|99.9021 ± 0.0555|100.0000 ± 0.0000|5/5|
|ordered|baseline|99.8116 ± 0.0173|100.0000 ± 0.0000|5/5|
|ordered|once|99.6422 ± 0.1560|100.0000 ± 0.0000|5/5|

逐格结果（诊断组不与主组混合）：

|组|seed|顺序|策略|状态|设备U %|暖接纳SLO %|暖混合卡|全active|第四完成≤1500|全程欠载|
|---|---:|---|---|---|---:|---:|---:|---|---|---|
|raw_feasible_q2|7|random|baseline|complete|99.9997|100.0000 (640/640)|32/32|True|True|True|
|raw_feasible_q2|7|random|once|complete|99.8564|100.0000 (640/640)|32/32|True|True|True|
|raw_feasible_q2|7|ordered|baseline|complete|99.8156|100.0000 (625/625)|32/32|True|True|True|
|raw_feasible_q2|7|ordered|once|complete|99.6880|100.0000 (635/635)|32/32|True|True|True|
|raw_feasible_q2|19|random|baseline|complete|99.9990|100.0000 (629/629)|32/32|True|True|True|
|raw_feasible_q2|19|random|once|complete|99.8501|100.0000 (631/631)|32/32|True|True|True|
|raw_feasible_q2|19|ordered|baseline|complete|99.8260|100.0000 (625/625)|32/32|True|True|True|
|raw_feasible_q2|19|ordered|once|complete|99.5882|100.0000 (635/635)|32/32|True|True|True|
|raw_feasible_q2|43|random|baseline|complete|99.9988|100.0000 (639/639)|32/32|True|True|True|
|raw_feasible_q2|43|random|once|complete|99.8835|100.0000 (641/641)|32/32|True|True|True|
|raw_feasible_q2|43|ordered|baseline|complete|99.7969|100.0000 (625/625)|32/32|True|True|True|
|raw_feasible_q2|43|ordered|once|complete|99.8320|100.0000 (635/635)|32/32|True|True|True|
|raw_feasible_q2|67|random|baseline|complete|99.9988|100.0000 (632/632)|32/32|True|True|True|
|raw_feasible_q2|67|random|once|complete|99.9746|100.0000 (630/630)|32/32|True|True|True|
|raw_feasible_q2|67|ordered|baseline|complete|99.7905|100.0000 (625/625)|32/32|True|True|True|
|raw_feasible_q2|67|ordered|once|complete|99.6927|100.0000 (635/635)|32/32|True|True|True|
|raw_feasible_q2|101|random|baseline|complete|100.0000|100.0000 (638/638)|32/32|True|True|True|
|raw_feasible_q2|101|random|once|complete|99.9462|100.0000 (637/637)|32/32|True|True|True|
|raw_feasible_q2|101|ordered|baseline|complete|99.8293|100.0000 (625/625)|32/32|True|True|True|
|raw_feasible_q2|101|ordered|once|complete|99.4102|100.0000 (635/635)|32/32|True|True|True|
|raw_nearest_q25|7|ordered|baseline|complete|46.6644|7.9875 (102/1277)|24/32|True|True|False|
|raw_nearest_q25|7|ordered|once|complete|45.7428|14.7059 (190/1292)|24/32|True|True|False|

所有输入与身份置换证据见audit.json；逐请求/层的来源位于runs，主表及全部人口、角色、类别SLO见summary.json与per_seed.csv。mean/SD用fraction保存于JSON，用明确percent字段保存于CSV。名义欠载不证明FIFO等待的唯一原因；任何机制判断仍需结合真实层时序及队列证据。
