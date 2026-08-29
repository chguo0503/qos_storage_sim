# Causal Scheme B after Layer 0

配置：128 NPU、batch 8、16 layers、28 SSUs；三种策略使用同一请求、放置、microbatch membership、SSD40 与 NPU50 数据面。

Layer 0 固定使用 baseline Path0 与基础 CIR。每个 NPU 完成自己的上一层 I/O 后，上报该层实际 bytes-by-SSU；控制器不接收仿真时钟、未来 placement 或 SSD 队列状态，原子更新 max-min CIR 后才提交下一层。cold Path0 使用固定 baseline CIR，不设置全局 fence。

| Strategy | Fleet util. | Makespan (ms) | Initial L0 (ms) | Initial L1-15 (ms) | Avg batch barrier (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Path 0) | 44.9426% | 6453.774 | 79.165 | 65.549 | 131.870 | 5317.867 |
| Original Scheme B | 43.5526% | 6659.748 | 161.658 | 34.715 | 175.391 | 5570.189 |
| Causal Layer 0 Path0 -> Scheme B | 44.9426% | 6453.774 | 215.910 | 1.063 | 193.550 | 5510.491 |

## Causal result

Original Scheme B makespan minus baseline: +205.974 ms.

Hybrid makespan minus baseline: +0.000 ms.

The causal policy recovered 100.00% of the original critical-path makespan gap; this does not mean its tail or mean barrier matches baseline.

Critical NPU 119 initial Layer-0 barriers: baseline 22.911 ms, original Scheme B 226.243 ms, hybrid 22.911 ms.

For that NPU, the causal hybrid leaves exposed wait in Layer 1--15: baseline 0.000 ms, original Scheme B 2.641 ms, hybrid 0.000 ms. This is physical cold/warm overlap, not a simulator fence.

Across all initial full batches, Layer-0 barrier is 215.910 ms versus 79.165 ms for baseline. Early warm dedicated Paths consume grants while late cold commands retain only the fixed baseline Path0 CIR, so the tail moves into Layer 0 even though the critical NPU is unchanged.

Compared with original Scheme B, the hybrid changes fleet utilization by +1.390 pp, makespan by -205.974 ms, and P99 by -59.698 ms.

Compared with baseline, it changes fleet utilization by +0.000 pp, makespan by +0.000 ms, and P99 by +192.624 ms.
