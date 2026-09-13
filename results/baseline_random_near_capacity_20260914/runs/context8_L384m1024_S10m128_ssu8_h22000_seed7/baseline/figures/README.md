# 384K主组：Baseline Random图

32 NPU、8 SSU×40 GiB/s、seed7；L384K/miss1024与S10K/miss128，每卡L:S=1:24，独立完整随机队列。两类C均外推。理想平均负载99.69%不保证逐盘逐时欠载。

- [32卡层周期平均带宽](baseline_random_all_32npu_layer_average.png)
- [32卡计算与IO等待](baseline_random_timeline.png)
- [连续两秒窗口利用率](baseline_random_window_U.png)
- [同一短请求内部中位等待](internal_median_wait/baseline_random_short_internal_cycle.png)
- [等待损失与首层链路固有下界](loss_decomposition.md)

warm [2,4)秒U=87.483765%；[2,20)秒U=87.486597%。两窗口均32卡全程有请求，32卡实际运行过长短两类。

统计来自原始运行；逐块日志来自同输入、同seed的等价重播。绘图前重新核验了双方SHA、完整summary（包括所有层）、windows、metadata与physical_service相等。重播不作为新增种子或独立重复。14,449个完整内部层周期通过实际收到字节核验。

原case日志与原renderer均未修改。复用到重播目录的analysis是原文件逐字节副本，具有独立analysis_reuse.json来源说明。

[周期与重播证据](checks.json) · [副标题及生成器核验](display_audit.json) · [局部选择与逐块证据](internal_median_wait/evidence.json)。
