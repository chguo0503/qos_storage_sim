# 仿真器

这个目录不依赖 `inputs/`、`results/` 或根目录 `data`。调用方提供请求，仿真器返回完整时序和统计。

```text
simulator/
├── api.py                统一运行入口 run_simulation
├── config.py             QoS 和客户端提交配置
├── contracts.py          策略输入快照、控制决策等公共类型
├── telemetry.py          周期性 SSU 拥塞采样
├── core/
│   ├── sim.py            SSU、QoS 硬件仲裁、NPU 接收链路、Ring hash
│   └── continuous_batch_sim.py
│                         请求/层事件、预取、计算和统计
├── policies/             策略决策；每个 .py 模块有 main 自检
└── adapters/             将核心状态转成策略输入，再执行策略决定
```

调用关系：`输入构造器 → api.run_simulation → adapter/核心 → 策略文件`。输入构造器负责选请求、到达时间、分卡和落盘；核心负责推进时钟；策略负责根据提供的状态选择 Path 或带宽配置。核心原有的 Scheme B 控制器已从事件引擎提取，核心导入独立策略实现。

## 主要策略

| 策略 | 实现 | 行为 |
|---|---|---|
| ASU baseline | [asu_baseline.py](policies/asu_baseline.py) | 每盘所有 NPU 共用 Path0 |
| OD baseline | [od_baseline.py](policies/od_baseline.py) | 每盘每个 NPU 独占一个 Path，CIR 均分，允许借用空闲带宽 |
| Once／流量分配 | [once.py](policies/once.py) | 每层按已有采样快照规划各块 Path，保留原类别合法候选池 |
| NewOnce | [new_once.py](policies/new_once.py) | 在 Once 上使用全客户端预留/完成账本 |
| SLO 候选池 | [slo_pool.py](policies/slo_pool.py) | 按请求等待预算选候选池，再调用原 Once；含 static/mild/aggressive |
| Scheme B | [scheme_b.py](policies/scheme_b.py)、[scheme_b_slo.py](policies/scheme_b_slo.py)、[scheme_b_causal.py](policies/scheme_b_causal.py) | 独立的 CIR 规划控制器 |

`policies/` 还保留之前实验使用的 coflow、分卡、带宽分配与 strategy1/2 算法。它们不是新入口的默认策略，历史 runner 可继续调用。`policy_logic.py` 和 `allocation.py` 提供可复用的纯计算函数；`baselines.py` 是两个基线的兼容分发接口，不包含第二份算法。

## 单独验证策略

在项目根目录执行，不需要输入文件或完整实验：

```bash
python -m simulator.policies.asu_baseline
python -m simulator.policies.od_baseline
python -m simulator.policies.once
python -m simulator.policies.new_once
python -m simulator.policies.slo_pool
```

每个 `main()` 用小规模确定性状态检查路径合法性、配置或决策守恒，成功输出 `PASS`。这些自检说明实现满足相应约束，不证明策略一定优于其他策略。完整回归测试运行 `python -m pytest -q`。

## 调用接口

```python
from simulator.api import run_simulation

# requests 是调用方已构造的 ContinuousBatchRequest 序列。
result = run_simulation(
    requests, strategy="once", num_npu=32, num_ssu=3,
    n_layers=8, seed=7,
)
summary = result["summary"]
```

请求类型在 [core/continuous_batch_sim.py](core/continuous_batch_sim.py)。`placement` 中的 `(ssu_id, size)` 使用 GiB，可以提供一份布局复用于所有层，也可以逐层提供布局。当前共享 Path adapter 要求每块都是 176 KiB；它不偷偷补齐数据。

默认仍为每盘 40 GiB/s、每卡接收链路 50 GiB/s、8 层、batch size=1、5 ms 采样，并开启已到达下一请求的首层预取。非 OD 策略沿用总 CIR=40 GiB/s 的类别配置；不能把物理盘速率降到其 CIR 总额以下而仍声称是原配置。

仿真完整执行输入到结束。`fleet_npu_compute_utilization` 是全程真实计算卡时间占比；历史 warm `[2,4)` 秒利用率和接纳窗口 SLO 应从返回的层/请求时序另行统计。新接口没有把小示例的全程利用率当成原实验结果。

## 保留的运行限制

- adapter 使用原来的上下文补丁连接事件引擎，异常退出时恢复。并行运行请用独立进程；同进程多线程并发安装 adapter 不受支持。
- 全部代码只使用规范的包导入，避免同一个核心以 `sim` 和 `simulator.core.sim` 被加载两遍。根目录不保留另一份实现。
- 策略真实 CPU 耗时单独记录，当前不计入仿真时钟；部署开销不能从模拟利用率直接推断。
- 此次保持物理服务、层预取、事件顺序与数值模型；旧函数名中的 GB/s 字段仍按项目的 GiB/s 约定解释。

输入示例和冻结清单见 [inputs/README.md](../inputs/README.md)，迁移及验证见 [维护记录](../docs/maintenance_20260919/README.md)。
