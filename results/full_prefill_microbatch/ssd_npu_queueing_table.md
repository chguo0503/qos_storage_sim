# SSD 与 NPU50 排队对比

配置：128 NPU、28 SSU、batch 8、16 层。下表统计初始满 microbatch，按照
I/O 数量加权计算每条 I/O 的平均排队时间。

| 策略 | SSD 排队 / I/O | NPU50 排队 / I/O | NPU 占排队比例 | 主要瓶颈 |
|---|---:|---:|---:|---|
| Baseline Path0 | 23.128 ms | 5.612 ms | 19.5% | SSU 明显更重 |
| Path RR | 8.343 ms | 8.248 ms | 49.7% | 两边基本相同 |
| Layer once | 7.992 ms | 8.135 ms | 50.4% | 两边相同，NPU 略重 |
| Refresh 8 | 9.199 ms | 8.140 ms | 46.9% | 两边接近，SSU 略重 |
| Refresh 1 | 9.199 ms | 8.147 ms | 47.0% | 两边接近，SSU 略重 |
| Legacy Scheme B once | 12.194 ms | 6.250 ms | 33.9% | SSU 更重 |
| Causal previous-layer Scheme B | 8.990 ms | 8.259 ms | 47.9% | 两边接近，SSU 略重 |
| Full-info EDF reference | 35.132 ms | 5.524 ms | 13.6% | SSU 明显更重 |

NPU 占排队比例的计算方式是：

\[
\frac{\text{NPU50 排队时间}}
{\text{SSD 排队时间}+\text{NPU50 排队时间}}
\]

所有策略都至少一次在全部 128 条 NPU 接收链路上产生排队，并在全部 28 块
SSU 上产生 backlog。因此“SSU 更重”只表示 SSD 排队贡献更大，不代表可以
删除 NPU 50 GB/s 接收限制。
