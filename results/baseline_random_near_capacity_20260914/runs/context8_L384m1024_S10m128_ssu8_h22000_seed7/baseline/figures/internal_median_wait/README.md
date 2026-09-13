# 384K案例：同请求内部中位等待

[查看PNG](baseline_random_short_internal_cycle.png)

原实验seed7没有保留逐块日志，因此使用同输入、同seed的重播补充trace。完整summary（含所有层）、窗口、metadata、实际服务统计与原运行完全一致，并逐项重新核验SHA。重播没有作为新seed或新独立样本。

32 NPU、8 SSU×40 GiB/s；L384K/miss1024、S10K/miss128，两类C均外推。理想平均负载99.69%不保证逐盘逐时欠载。

选择NPU12、请求12000055。从3391个warm内完整、避开首尾且有等待的短层中选择中位附近；没有选择最坏层。
C=0.716652510 ms；w=1.154051548 ms；D=1.870704058 ms；V=13.578125 MiB。
B=18.502534376 GiB/s；平均b=7.088180325 GiB/s；b/B=C/D=38.309240%。这是完整周期记账关系，不是瞬时利用率或排队预测。

本卡NPU行与累计接收始终属于同一请求。SSU行包含其他卡的竞争IO；所有显示块与原manifest placement一致。
局部名义逐盘需求超限1.870704058 ms；不能只凭此图宣称严格欠载下FIFO损失。
首层固有链路等待另见[损失分解](../loss_decomposition.md)，不把全部首层损失称作FIFO。

[选择规则、重播校验及全部来源SHA](evidence.json) · [实际块记录](selected_cycle_blocks.json)
