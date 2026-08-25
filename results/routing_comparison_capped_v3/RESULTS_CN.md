# 16 层 NPU 50 GB/s 限速后的路由对比

日期：2026-08-25

## 结论

加入真实的“每个 NPU 跨全部 SSU 合计不超过 50 GB/s”数据面限速后，当前
静态 QoS 路由在 40 和 56 SSU 都明显差于 baseline。更频繁地读取 Path 压力
几乎没有帮助；8-I/O Path 绑定能缓解逐 I/O 分散产生的长尾，但仍未超过
baseline，且 p95 和 Jain fairness 都恶化。

因此，历史 uncapped 图中的 79%/85% 和约 +8pp 不能继续用于回答当前问题。
本次没有发现 capped 条件下可用的正收益方案，更没有达到 +10pp。

## 实验口径

- 128 NPU，16 layers，`LS_RATIO=0.5`，workload seed 42；
- SSU 数量：40、56；每块 SSU raw 上限 40 GB/s；
- 每个 NPU 跨所有 SSU 的有效接收带宽上限：50 GB/s；
- 每块 SSU 最多一个不可抢占 active I/O；
- 相同 workload、block 大小和 block→SSU placement；
- Baseline 与 QoS 都按 ready time 提交，同 timestamp 使用独立 seed shuffle；
- 客户端提交节奏固定为每轮最多 8 个 I/O；
- 静态 CIR `[SS,SL,LS,LL]=[20,4,12,4] GB/s`，PIR uncapped，所有 WRR
  权重为 1；
- schema v3，结果 spec 中记录了 NPU cap、三个独立 seed、代码和数据指纹。

四个对比组：

- `capped baseline`：QoS bypass；
- `0/1`：每个 `(request, layer, SSU)` 只读一次 256-count，每个 I/O 独立选 Path；
- `8/1`：每 8 个 I/O 读一次 256-count，每个 I/O 独立选 Path；
- `8/8`：每 8 个 I/O 读一次，并让本窗口最多 8 个 I/O 共用一个 Path。

## 主结果

这里的 request compute fraction 是
`compute_ms / (compute_ms + io_wait_ms)` 的逐请求平均值，不是真实整机 NPU
利用率。

| SSU | 策略 | request compute fraction | 相对 baseline | fleet compute util | Jain | p95 通过 | fairness 通过 |
|---:|---|---:|---:|---:|---:|:---:|:---:|
| 40 | baseline | 61.76% | — | 14.53% | 0.7682 | — | — |
| 40 | 0/1 | 29.06% | -32.70pp | 12.19% | 0.5956 | 否 | 否 |
| 40 | 8/1 | 29.07% | -32.69pp | 12.18% | 0.5966 | 否 | 否 |
| 40 | 8/8 | 46.08% | -15.68pp | 14.19% | 0.6587 | 否 | 否 |
| 56 | baseline | 64.65% | — | 14.57% | 0.7935 | — | — |
| 56 | 0/1 | 29.96% | -34.70pp | 12.29% | 0.5958 | 否 | 否 |
| 56 | 8/1 | 30.35% | -34.30pp | 12.40% | 0.5950 | 否 | 否 |
| 56 | 8/8 | 39.60% | -25.05pp | 14.23% | 0.5879 | 否 | 否 |

`0/1 → 8/1` 只改善 0.01pp（40 SSU）和 0.39pp（56 SSU）。这说明在当前
模型中，是否每 8 个 I/O 刷新一次压力不是主导因素。

## 各类别 p95 TTFT

单位为 ms。

| SSU | 策略 | SS | SL | LS | LL |
|---:|---|---:|---:|---:|---:|
| 40 | baseline | 151.27 | 984.39 | 259.23 | 2008.87 |
| 40 | 0/1 | 436.17 | 1353.80 | 740.26 | 2433.79 |
| 40 | 8/1 | 434.24 | 1352.52 | 718.90 | 2437.22 |
| 40 | 8/8 | 227.27 | 1004.98 | 356.92 | 2059.91 |
| 56 | baseline | 131.65 | 983.00 | 230.31 | 2004.77 |
| 56 | 0/1 | 425.69 | 1270.59 | 713.74 | 2411.61 |
| 56 | 8/1 | 410.91 | 1271.30 | 701.62 | 2369.55 |
| 56 | 8/8 | 374.05 | 1006.67 | 480.85 | 2055.23 |

所有类别在所有 QoS 变体下都高于 paired baseline，因此不存在隐藏某一类别失败
后宣称平均值成功的问题。完整 p50/p95/p99/max 保存在 JSON 中。

## 为什么加 cap 后反而很差

### 1. 单 active、不可抢占与 NPU cap 形成 HoL

一条 I/O 一旦进入某块 SSU 的唯一后端槽位，即使因为同一 NPU 同时从多块 SSU
读取而被比例缩到很低的有效速率，它仍长期占住该 SSU，后面的 I/O不能抢占。
这会让“SSU 看起来 active”与“真正传输了多少字节”严重分离。

| SSU | 策略 | SSU active-time util | SSU effective-BW util |
|---:|---|---:|---:|
| 40 | baseline | 35.49% | 8.12% |
| 40 | 0/1 | 47.24% | 6.81% |
| 40 | 8/1 | 47.32% | 6.81% |
| 40 | 8/8 | 38.63% | 7.94% |
| 56 | baseline | 34.07% | 5.82% |
| 56 | 0/1 | 45.97% | 4.91% |
| 56 | 8/1 | 47.06% | 4.95% |
| 56 | 8/8 | 41.44% | 5.68% |

逐 I/O QoS 的 active-time 更高，传输的有效字节比例却更低，不能把前者当成性能
提升。

### 2. 逐 I/O 分散到许多 Path 放大 layer coflow 尾部

Baseline 以 NPU 源队列为 FCFS/RR 单位。QoS 则先把一个 layer 的 block 分散到
许多 Path，再按 Path 的 CIR/WRR 虚拟完成标签服务。一个 layer 必须等待最后一个
block；Path 分散越广，越容易出现少数 straggler Path 拖住整个 layer。

| SSU | 策略 | 平均 queue wait / block | 最大 queue wait |
|---:|---|---:|---:|
| 40 | baseline | 4.40 ms | 46.36 ms |
| 40 | 0/1 | 14.33 ms | 452.58 ms |
| 40 | 8/1 | 16.70 ms | 451.77 ms |
| 40 | 8/8 | 9.77 ms | 80.47 ms |
| 56 | baseline | 3.86 ms | 37.71 ms |
| 56 | 0/1 | 11.74 ms | 429.69 ms |
| 56 | 8/1 | 13.02 ms | 404.05 ms |
| 56 | 8/8 | 10.27 ms | 88.39 ms |

8/8 把相关 I/O 集中到更少 Path，显著降低最大等待，因此比 0/1、8/1 好；但它
仍受 Path 级仲裁和单 active HoL 影响，无法超过 baseline。

### 3. 压力刷新不能修复调度单位不匹配

8/1 比 0/1 多读取约 3 倍 256-count，但两者的 request compute fraction、
makespan、fairness 和长尾都非常接近。更新后的 count 能改变具体 Path，却不能
改变“layer 等待所有 block”“Path 级公平”和“被限速 active I/O 不可抢占”这三个
核心约束。

以上是当前仿真数据支持的因果解释；它不是对真实 SSD 硬件的证明。尤其是“一块
SSU 只有一个 active I/O”会放大 HoL，真实设备的内部并行和抢占/切片行为需要硬件
验证。

## 正确性与上限检查

- 两点各策略都完成 128/128 请求；
- 每组均有 1,704,832 个 block，恰好 submit 和 complete 一次；
- placement target 与 bytes 全部守恒；
- 每块 SSU 的 `max_backend_active_io <= 1`；
- 仿真结束 outstanding 为 0；
- 所有策略的 NPU 有效峰值不超过 `50 + 1e-9 GB/s`；
- 所有 paired workload fingerprint 和 placement hash 相同；
- JSON 使用 `allow_nan=False` 成功写出。

## 仿真速度优化

在不改变结果的前提下做了三项优化：

1. QoS 后端不再每完成一条 I/O 扫描全部非空 Path；只在非空集合变化时重建
   CIR/WRR 服务率，并按 finish-tag bucket + heap 选下一 Path；
2. 客户端 Path 选择复用 count 聚合和等价候选投影；
3. 四个 paired 策略共享同一份只读 workload/placement artifact，避免重复生成和
   深拷贝。

128 NPU、40 SSU、1 layer 的完整四组 smoke 从 46.00 秒降到 24.87 秒，优化前后
`results` JSON 逐字段完全相同（约 1.85×）。完整 40/56 SSU、16-layer 两 worker
运行耗时 780.82 秒，单 worker 峰值 RSS 约 1.47 GB。

## 复现

```bash
python -m unittest discover -s tests -v

python experiment.py \
  --layers 16 \
  --ssu-list 40,56 \
  --workers 2 \
  --rerun \
  --output-dir results/routing_comparison_capped_v3
```

测试结果：35/35 通过。

机器可读结果：`routing_comparison_capped.json`；绘图：
`routing_comparison_capped.png`。
