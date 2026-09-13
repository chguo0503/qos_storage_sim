# 512K长读取尺度：Baseline Random图

32 NPU、8 SSU×40 GiB/s、seed7；L512K/miss1024与S10K/miss128，每卡L:S=1:31，独立完整随机队列。两类计算时间均为原始data区间外拟合，不能称为实测512K硬件结果。理想平均负载99.74%不代表逐盘逐时欠载。

- [32卡层周期平均带宽](baseline_random_all_32npu_layer_average.png)
- [32卡计算与IO等待](baseline_random_timeline.png)
- [连续两秒窗口利用率](baseline_random_window_U.png)
- [同一短请求内部的中位等待样例](internal_median_wait/baseline_random_short_internal_cycle.png)
- [内部等待、首层等待与链路固有下界](loss_decomposition.md)

warm [2,4)秒 U=87.253128%；[2,20)秒 U=86.317495%。这两个窗口均为32卡全程有请求，且32卡均实际计算过长、短两类。18秒观察窗不是无限时长极限。

蓝线只填完整同请求内部周期；逐块核验12,531个周期收齐相同下一层V。请求交接和窗口截断处留灰，但仍计入整窗真实利用率及接收量。

局部图从2,624个避开请求首尾、完整且有等待的短层中选择等待中位数附近，未挑最坏层。该局部名义需求全程超限；它说明排队与错过截止的对应关系，不能单独证明严格欠载或纯FIFO额外损失。

[绘图与逐周期核验](checks.json) · [标题负载/外推标注](display_audit.json) · [局部逐块证据](internal_median_wait/evidence.json) · [局部选择说明](internal_median_wait/README.md)。原始render/zoom/analyze及旧图未修改，没有新仿真。
