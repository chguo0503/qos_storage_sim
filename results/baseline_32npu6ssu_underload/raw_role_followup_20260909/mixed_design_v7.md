# v7：同v6绑定与random，对热段改用三种Short

冻结模块[mixed_design_v7.py](mixed_design_v7.py)，SHA-256 `ea6b0d834c8deae671f802e094affc28021cf1c418dd8bf1ce8a0b11b8f5a1bd`。API、全局1172人口、各卡配额和请求物理放置均与v6一致，不再重绑新人口。

唯一有意变化是ordered后12卡的热段：

`9S2+8S3独立洗牌 → 11S1+2S2+2S3每卡独立洗牌 → 全部Long → 剩余Short轮转`。

v6相应热段是连续15S1。v7保留完整每卡人口，把少用的4条S1移到尾部，并将尾部的2条S2、2条S3移到热段，C/V完全不改。前20卡Long与后续Short顺序不变；后12卡最初9S2+8S3前缀也不变。

绑定和random的随机流故意保留`raw200_rebinding_v6`命名，仅热段新增`ordered_hot_short_v7_{npu}`随机流。真实五seed共160卡逐项验证：**v7random与v6random的原对象顺序完全相同**，不仅人数、画像配额相同，因此可以复用v6random策略结果作为准确对照，不重复模拟。证据：[mixed_design_v7_checks.json](mixed_design_v7_checks.json)，由[mixed_design_v7_check.py](mixed_design_v7_check.py)复算并记录v6参考SHA。

新热段15条请求纯C999.716814470ms，整个首次Long前缀纯C2523.571248189ms，比v6更长。任何热段最终画像都满足首次Long L0不会早于`2523.571248189 − max(C_S1,C_S2,C_S3)=2510.945307428ms`；这比v6的Guard Baseline上界2300.405643ms更宽松。实际seed7所有卡的末条画像给出的更紧最早下界为2513.629662ms，见[v6/v7共同FIFO数值证明](mixed_design_v6_fifo_bound.json)。这个论证仍只限Baseline ordered，不是Once多Path服务保证。

旧200K固定输入Short均值给出整个前缀约3676.431198ms，仅作为时间尺度参考。更长C与更大I/O同时进入热段，且每卡画像顺序不同，可能改变停顿与相位；不能预设U下降仍超过10个百分点。实际暖窗是否覆盖三种Short、每卡Long/Short正计算时长、名义容量与设备U需从新ordered日志报告。
