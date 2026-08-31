# Baseline 2 NPU × 2 SSU 教学实验

## 先明确 baseline 是什么

Baseline 把所有 I/O 固定送到每块 SSD 的 **Path0**，并没有按 NPU 显式均分带宽。Path0 的 PIR 无上限；当它是唯一活跃 Path 时，会工作保持地拿到 SSD40 的全部剩余带宽。因此，对称 trace 中接近均分只是 FIFO、提交顺序和逐层 barrier 共同产生的结果，不是 baseline 的公平性保证。

## 可直接引用的结果表

固定 8 层、10 ms/层、SSD40、NPU50、batch=1；所有利用率均为同一个 19,920 ms 饱和窗口内的精确事件重叠。TTFT/Barrier 列顺序为 NPU0/NPU1。

| Case | D[NPU0; NPU1] GB/s | Cmd/positive flow | SSD util 0/1 | NPU link util 0/1 | NPU compute util 0/1 | Mean NPU util | TTFT ms 0/1 | Barrier ms 0/1 | 2× SLO 0/1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 均匀满载 | `[20,20]; [20,20]` | 4 | 100.00%/100.00% | 80.00%/80.00% | 100.00%/100.00% | 100.00% | 80.00/80.00 | 0.00/0.00 | 100.00%/100.00% |
| 同盘热点 | `[30,0]; [30,0]` | 4 | 100.00%/0.00% | 40.00%/40.00% | 66.67%/66.67% | 66.67% | 120.00/120.00 | 40.00/40.00 | 100.00%/100.00% |
| 互补放置 | `[38,0]; [0,38]` | 16 | 95.00%/95.00% | 76.00%/76.00% | 100.00%/100.00% | 100.00% | 80.00/80.00 | 0.00/0.00 | 100.00%/100.00% |
| 异构大命令 | `[30,0]; [10,0]` | 1 | 78.31%/0.00% | 43.37%/19.28% | 72.29%/96.39% | 84.34% | 110.67/83.00 | 30.67/3.00 | 100.00%/100.00% |
| 异构小命令对照 | `[30,0]; [10,0]` | 4 | 100.00%/0.00% | 60.00%/20.00% | 100.00%/100.00% | 100.00% | 80.00/80.00 | 0.00/0.00 | 100.00%/100.00% |

## 结论

- **均匀满载不等于 baseline 做了均分。** 两块盘都满载时，对称小命令恰好让两个 NPU 连续计算，所以 mean NPU util 是 100%；这是输入和时间线的结果。
- **全局总容量足够仍可能下降。** 同盘热点只有 60 GB/s fleet demand，低于两盘合计 80 GB/s，但 SSD0 的 60 GB/s 超过单盘 40 GB/s；SSD1 的空闲不能搬运 SSD0 上的数据，两个 NPU util 都降到 66.67%。
- **mean 会掩盖逐 NPU 差异。** 异构大命令中 NPU0/NPU1 util 是 72.29%/96.39%，mean 仍有 84.34%；相应 TTFT 是 110.67/83.00 ms。
- **barrier 不只由平均带宽决定。** 异构输入的每盘平均需求恰好等于 40 GB/s，但 0.3-GB 非抢占命令会阻塞 0.1-GB 命令。保持 demand 和 placement 完全相同，仅拆成 4 个命令后，两个 NPU 都恢复为 100% util、80 ms TTFT。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 python baseline_2npu_teaching_experiment.py
PYTHONDONTWRITEBYTECODE=1 pytest -q test_baseline_2npu_teaching_experiment.py
```

JSON 保留逐盘 busy time、逐 NPU compute/link busy time、逐 NPU TTFT/barrier、输入和源码指纹，以及所有 steady-state invariants。

Source fingerprint: `007742936bea4947e3f42d6cb18a5acbc9e216bc531d250613decc2dab9d2791`
Experiment fingerprint: `76bc5022f5314ec13f7d0ec66f8c10137f0db274002daada68639f4b8fb76dd4`
