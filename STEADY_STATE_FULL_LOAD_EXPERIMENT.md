# 稳态满负载实验规范

本文档冻结当前正式实验要求。该实验用于回答：当 128 个 NPU 始终有请求可执行、不会因
有限 trace 提前退出时，Baseline、每层读取一次 Path 状态和 Scheme B 在相同物理数据面上的
平均 NPU 利用率与 TTFT processing SLO 有何差异。

## 正式矩阵

| 参数 | 固定值 |
|---|---|
| NPU | 128 |
| 推理层数 | 16 |
| batch | 1 |
| SSU | `16 / 24 / 40 / 70` |
| 策略 | `baseline / layer_once / scheme_b` |
| seed | 42 |
| SSD 后端 | 每块 SSD 单命令、不可抢占，40 GB/s |
| NPU 接收链路 | 每个 NPU 独立 FCFS，50 GB/s |
| 客户端提交 | 每次 1 条 I/O，相邻提交间隔 0.1 us |

`Refresh8` 和 `Best feasible` 明确不进入本实验，也不进入最终结果图。

## 三个策略

### Baseline

- 不读取 Path pressure；
- 所有 I/O 固定进入 Path 0；
- 使用与 `layer_once` 完全相同的 Static QoS 硬件配置。

### Read once per layer

- 每个 request-layer-SSU 读取一次目标 SSD 的 Path pressure；
- 用该 snapshot 和本地 shadow pressure 一次性规划这一层发往该 SSU 的全部 block；
- 与 Baseline 的区别只应是 NPU 选路，SSD QoS 仲裁器和物理数据面相同。

### Scheme B

- 首次冷 Layer 0 使用公共 Path 0；
- 后续层只使用已完成前层的 bytes-by-SSU、计算预算和当前活跃请求信息；
- 为 NPU 分配专属 Path，并用 demand-capped max-min grant 动态更新每块 SSU 的 CIR；
- 不读取仿真未来完成时间，也不在不同仿真 run 之间复用 controller 状态。

## 请求流与 placement

- 每个 NPU 使用确定性的饱和请求流；正式输入前缀为 32 个请求/NPU；
- 请求类别按 `SS / SL / LS / LL` 四类循环，不同 NPU 使用确定性轮转，使 fleet 输入均衡；
- 同一 SSU 点的三个策略使用相同 request、profile、arrival、request ID 和 placement；
- 输入配对由 `workload_hash / placement_hash / trace_hash` 强制校验；
- 每个 token block 用 `(request_id, block_index)` 做一致性 ring hash；同一 block 的所有
  16 层复用同一个 SSU，不同 block 可以位于不同 SSU；
- 32 请求只是有限的预生成前缀。若任何 NPU 在正式窗口结束前耗尽 backlog，
  `no_backlog_exhaustion` invariant 必须令该 case 失败，不能输出指标。

## 跨请求 Layer 0 预取

三个策略都启用完全相同的跨请求预取规则：当请求 k 的最后一层开始计算，且请求 k+1
已经到达时，可以开始读取请求 k+1 的 Layer 0；请求 k+1 的 admission 仍需等待请求 k
完成。这样，下一请求 Layer 0 的 I/O 可以被上一请求最后一层计算掩盖。

## 稳态时间线

```text
128 个 NPU 始终 backlogged
          ↓
每个 NPU 至少完成 4 个请求
          ↓
额外 settle 500 ms
          ↓
共同的 2,000 ms 测量窗口
          ↓
停止接纳新的测量样本，但继续满负载运行
直到窗口内 admission 的请求全部完成
```

- 每个策略在自身所有 NPU 达到 warmup 条件后开启测量，因此三个策略的绝对窗口起点
  不要求相同；
- 128 个 NPU 在单个策略内共享同一个测量窗口；
- TTFT cohort 是半开窗口 `[start, end)` 内 admission 的请求；
- 窗口关闭后必须把所有 tagged 请求 drain 到完成，避免右截尾和幸存者偏差；
- 不要求三个策略恰好测量相同 request ID。它们比较的是同一确定性平衡请求流在各自
  稳态窗口中的表现；报告必须保留这一限制。

## 两个主指标

### 平均 NPU 利用率

对每个 NPU，计算其所有 layer compute interval 与共同 2,000 ms 窗口的精确交集：

```text
NPU utilization[n] = 窗口内 compute busy ms / 2,000 ms
Mean NPU utilization = 128 个 NPU utilization 的等权平均
```

已经在窗口开始前 admission、但仍在窗口内计算的请求会贡献利用率。某个 NPU 在窗口
结束后 drain tagged 请求的计算时间不进入利用率分子。

### TTFT processing SLO

对窗口内 admission 的每个请求：

```text
processing TTFT = completion - admission
compute-only TTFT = 16 × 该请求的单层计算时间
SLO 达标 = processing TTFT <= 2 × compute-only TTFT
```

先在每个 NPU 内计算达标率，再对 128 个 NPU 等权平均，作为主 SLO 指标。另存的
request-weighted SLO 只用于诊断。

该指标不是用户外部 arrival-to-completion TTFT：饱和队列中 admission 之前的等待不计入，
而 admission 前已经完成的跨请求 Layer 0 预取会缩短 processing TTFT。最终图和报告应写
`TTFT processing SLO`，避免将其误称为端到端 TTFT。

## 正确性要求

每个 case 必须同时满足：

- 所有 NPU 均达到 warmup；
- 测量窗口严格关闭且总长为 2,000 ms；
- 所有 NPU 都有 SLO 样本；
- 窗口内 admission 无漏标，所有 tagged 请求均完成；
- 测量前和测量期间没有 NPU 耗尽 backlog；
- 每块 SSD 同时最多执行一条命令；
- 任意 SSU 的 CIR 总和不超过 40 GB/s；
- 同一 SSU 下三个策略的 workload、placement 和 trace hash 完全一致。

任一 case 抛出异常时，runner 必须立即打印 `FAILED strategy/SSU`、终止其他 worker 并
返回失败，不能等待其余长任务全部结束后才显示错误。运行期间每个 case 每 60 秒输出一次
heartbeat。

## 运行与产物

```bash
# 12 个正式 case；可根据本机 CPU 调整 workers
python steady_state_experiment.py --workers 9

# 完成后校验配对关系并生成双子图、JSON 和 Markdown 报告
python analyze_steady_state.py
```

输出目录：`results/steady_state_full_load_layer16/`

- `results.json`：逐 case checkpoint；
- `analysis.json`：配对验证、主指标和相对 Baseline 的百分点差；
- `report.md`：结果表与口径说明；
- `01_steady_state_full_load.png`：横轴 SSU 数，左图平均 NPU 利用率，右图 TTFT
  processing SLO @ 2x，只包含三个正式策略。
