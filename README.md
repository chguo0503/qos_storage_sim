# 静态 QoS 路由仿真

项目只比较两个严格配对的策略：

- `baseline_bypass`：绕过 QoS Path，各 NPU 源队列逐 I/O RR。
- `qos_static_cir`：固定 CIR、PIR 和 WRR；每 8 条 I/O 读取一次 256-Path
  `active + pending` count，每条 I/O 独立选择合法 Path。

两组使用相同请求、block→SSU placement、8-I/O 客户端提交节奏和物理限制。
所有参数都固定在代码中，不再保留 `0/1`、`8/8`、Path lease、动态 K/q 或
动态硬件配置分支。

## 模型

- 128 个 NPU，每个 NPU 一个请求。
- 每块 SSU 40 GB/s，最多一个不可抢占 active I/O。
- 每个 NPU 从所有 SSU 接收的有效带宽总和不超过 50 GB/s。
- 同一时刻按独立 seed 随机排列就绪 NPU；每个 NPU 提交后立即入队。
- Path 内严格 FCFS；QoS 按 CIR、组内 WRR、组间 WRR、最终 RR 选择队头。
- CIR/PIR/WRR 只在初始化时配置，运行期间不修改。

NPU 限速按 raw rate 等比例缩放且保持 work-conserving。例如 `40+40` 变成
`25+25`，`40+5` 保持不变。baseline 和 QoS 经过同一套限速代码。

## 静态 Path 布局

每块 SSU 有 8 个 Group，每组 32 个 Path：

| 类别 | 每组 Path | 全盘 Path | 全盘 CIR |
|---|---:|---:|---:|
| SS | 12 | 96 | 20 GB/s |
| SL | 4 | 32 | 4 GB/s |
| LS | 12 | 96 | 12 GB/s |
| LL | 4 | 32 | 4 GB/s |

PIR uncapped，Path/Group 权重均为 1。客户端只使用请求类别对应的 Path 池。

## 客户端选路

`client_select_qos_paths()` 只接收四项客户端可见信息：本窗口 block 大小、
完整 256-count 快照、合法 Path 池和静态 QoS 镜像。它不会读取队列字节、
I/O age、剩余服务时间或其他 NPU 身份。

选择器用本窗口 block 中位大小估算旧积压，根据静态 CIR 和两级 WRR 估算
长期服务率，再选择预计清空时间最短的 Path。每规划一条 I/O 就更新本地
shadow count，但不修改硬件快照。

```python
from experiment import qos_config
from sim import ClientRoutingConfig, client_category_paths, client_select_qos_paths

config = qos_config()
paths = client_select_qos_paths(
    block_sizes_gb=[0.001] * 8,
    path_io_counts=[0] * 256,
    allowed_path_ids=client_category_paths("LL", config),
    routing_config=ClientRoutingConfig(config, disk_bw=40.0),
)
```

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python experiment.py \
  --layers 16 \
  --ssu-list 40,56 \
  --workers 2 \
  --rerun
```

默认输出：

- `results/routing_comparison/results.json`
- `results/routing_comparison/comparison.png`

图中只有 Baseline 和 QoS 8/1 两条线。缓存包含代码、数据、placement、seed、
NPU cap 和静态 QoS spec；spec 变化时不会复用旧结果。

## 当前 16 层结果

128 NPU、seed 42、`LS_RATIO=0.5` 的实测结果如下：

| SSU | Baseline | QoS 8/1 | 差值 |
|---:|---:|---:|---:|
| 40 | 61.76% | 29.07% | -32.69pp |
| 56 | 64.65% | 30.35% | -34.30pp |

QoS 在两个点都降低了 request compute fraction，同时类别 p95 和 Jain fairness
也未通过 baseline 门槛。代码没有隐藏这一负结果；完整指标见
[`results.json`](results/routing_comparison/results.json)，两线图见
[`comparison.png`](results/routing_comparison/comparison.png)。

双进程完整运行用时 276.93 秒、峰值内存 294,000 KiB。相同机器上，精简前的
对应两策略运行用时 332.06 秒、峰值 1,446,456 KiB；调度结果不变。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 256-count ABI、逐 I/O 合法选路、Path FCFS、每盘单 active、50 GB/s
限速、block/placement/bytes 守恒、两策略严格配对和实验结果回归。

## 文件

- `sim.py`：离散事件引擎、硬件调度、NPU 限速和客户端选路。
- `experiment.py`：两策略配对运行、缓存、摘要和两线绘图。
- `tests/`：核心语义与集成测试。
- `data`：请求画像输入。

## 模型限制

256-count 遥测被视为零延迟且无丢失；count 不含字节或 age。每盘单 active
会放大 HoL，NPU 限速也可能让 active I/O 长时间占住后端。本仿真是静态 QoS
的 packetized WFQ 近似，不能宣称复刻具体 SSD 的 WRR/token-bucket 实现。
