# QoS + SSD 路由仿真

这是一个聚焦于 **NPU Path 选路频率** 的离散事件仿真项目。项目只保留
最终实验链路：五种可执行路由策略，以及一个保留全部物理容量约束的
best-feasible 参考策略。

## 物理模型

每条 I/O 都经过同一个两级数据面：

```text
NPU 选择 Path ID
        ↓
SSD 仲裁：单命令、不可抢占，服务时间 = size / 40 GB/s
        ↓ SSD 完成立即释放槽位
目标 NPU 的独立 FCFS 接收队列，服务时间 = size / 50 GB/s
        ↓
block 对计算侧可见
```

- 每块 SSD 同时最多服务一条命令。
- CIR 只控制命令获得服务的长期机会；获胜命令仍以完整 40 GB/s 执行。
- 每个 NPU 有独立的 50 GB/s 单服务器接收队列，多 SSD 同时完成会形成
  incast 排队。
- SSD 完成后不等待 NPU 接收，因此模型假设中间 buffer 足够大、无反压。
- block→SSD placement、工作负载、到达延迟和提交顺序在策略间严格配对。

## 固定实验配置

| 参数 | 值 |
|---|---|
| NPU | 128 |
| 层数 | 16 |
| SSU | 8, 16, 28, 40, 56, 80, 112 |
| Seed | 42, 43 |
| NPU 到达延迟 | 每个 NPU 独立从 `[0, 5 ms)` 抽样 |
| SSD 带宽 | 40 GB/s/盘 |
| NPU 接收带宽 | 50 GB/s/NPU |
| I/O 发行 | batch=1，间隔 0.1 µs |
| Static CIR（SS/SL/LS/LL） | 20/6/8/6 GB/s |
| 每组 Path（SS/SL/LS/LL） | 12/4/12/4，共 8 组、256 Path |

## 对比策略

五个普通策略全部调用相同的 `qos_static_cir` 实现，SSD/NPU 数据面、CIR、
Path 布局完全相同，唯一变量是 NPU 写入命令的 Path ID：

1. `baseline`：所有 NPU 的所有 I/O 固定进入 Path 0，不读取 Path 状态。
2. `path_rr_baseline`：不读状态，在类别合法 Path 内逐 I/O round-robin。
3. `refresh1`：每规划一条 I/O 前读取一次 Path pressure。
4. `refresh8`：每规划 8 条 I/O 读取一次。
5. `layer_once`：每个 `(request, layer, SSD)` 只读取一次。

如果 baseline 与 Static QoS 被强制产生完全相同的 Path-ID 序列，两者的
request、SSD、NPU link、makespan 和利用率结果必须逐项相同；测试会验证
这个合同。

第六条曲线 `capacity_constrained_oracle` 是每个 `(seed, SSU)` 上，从五个
普通策略和 demand-weighted SJF 候选中选择出的最佳可行结果。它保留原始
placement、SSD40、NPU50、到达时间和层依赖，但没有数学最优性证明，不能
称为 exact optimum 或理论上界。

## 当前结果

表中数值为两个 seed 的平均 `avg_request_compute_fraction`：

| SSU | Path0 | No-state RR | Refresh1 | Refresh8 | Layer once | Best feasible |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 31.219% | 34.517% | 35.342% | 35.338% | 35.336% | 48.202% |
| 16 | 46.376% | 50.302% | 52.796% | 52.578% | 52.757% | 65.097% |
| 28 | 60.294% | 63.971% | 71.123% | 72.623% | 72.610% | 77.513% |
| 40 | 70.265% | 72.652% | 82.811% | 81.363% | 81.684% | 83.871% |
| 56 | 80.329% | 80.404% | 87.852% | 87.400% | 87.231% | 87.943% |
| 80 | 88.274% | 87.666% | 90.562% | 89.721% | 89.716% | 90.964% |
| 112 | 91.343% | 90.791% | 90.998% | 90.507% | 90.795% | 91.763% |

最终图片：

![最终路由策略曲线](results/routing_refresh_concurrency/01_routing_refresh_finite_issue.png)

## 指标

`avg_request_compute_fraction` 先计算每个请求的：

```text
16 层计算时间 / (16 层计算时间 + 该请求暴露的 I/O stall)
```

再对 128 个请求等权平均。主图将它显示为 `Average NPU Utilization (%)`。
0–5 ms release delay 发生在请求开始前，不进入这个分母。

`fleet_npu_compute_utilization` 为：

```text
全部 NPU 计算时间 / (128 × 全局 makespan)
```

它包含 release 错峰、I/O stall 和已完成 NPU 的空闲时间，因此会被最慢请求
主导。它与主图指标不是同一个量。

## 运行

需要 Python 3.9+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 五个路由策略：2 seed × 7 SSU × 5 策略，共 70 个 case
python routing_refresh_concurrency_experiment.py --workers 10 --rerun

# 容量约束参考候选：2 seed × 7 SSU，共 14 个 case
python capacity_constrained_oracle_experiment.py --workers 10

# 校验配对结果、生成报告和图片
python analyze_routing_refresh_concurrency.py

# 快速测试，不运行正式大矩阵
python -m unittest discover -s tests -v
```

正式仿真耗时较长。runner 会逐 case checkpoint；不传 `--rerun` 时，只有
代码、数据和配置指纹完全一致才会复用缓存。原始 JSON 被保留在最终结果
目录中，方便只修改分析或绘图时避免重跑。

## 项目结构

- `sim.py`：离散事件引擎、Static QoS 仲裁、SSD→NPU 两级数据面。
- `advanced_policies.py`：容量约束候选所需的调度优先级函数。
- `strategy_profiles.py`：唯一的最终 Static CIR/Path 配置。
- `experiment.py`：结果压缩和原始 Matplotlib 画图函数。
- `routing_refresh_concurrency_experiment.py`：五策略正式矩阵和多进程 checkpoint。
- `capacity_constrained_oracle_experiment.py`：可行的容量约束参考候选。
- `analyze_routing_refresh_concurrency.py`：严格校验、双 seed 聚合、报告和绘图。
- `results/routing_refresh_concurrency/`：原始结果、聚合结果、报告和图片。
- `tests/`：物理数据面、路由合同、Oracle 和分析管线测试。

## 模型限制

- Path pressure 读取被视为零延迟、无丢失。
- 0.1 µs 是用于允许真实事件插入的因果参数，不是特定 NPU 的实测时延。
- 未建模固定 NAND/协议延迟、QD 吞吐曲线、channel/die 并行和 buffer 反压。
- packetized WFQ 是 CIR/WRR 服务机会的命令级近似，不对应某款 SSD 的完整
  token-bucket 微架构。
- 容量约束参考是 best observed feasible schedule，不是带最优性证书的上界。
