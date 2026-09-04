# V/B 连续推理实验复现说明

本文档用于在另一台机器或由本地 AI 独立复现以下两个实验：

1. **4 NPU / 1 SSU 受控机制实验**：对比固定分离与串行混排；
2. **32 NPU / 5 SSU 稳态实验**：只对比固定分离下的 Baseline 与 V/B。

这两个实验的 Baseline 和 V/B 配置不同，不能直接比较两组实验的绝对利用率。
4-NPU 实验用于展示机制；32-NPU 实验用于在真实 `data` 画像随机输入下验证固定分离的稳态收益。

---

## 1. 代码和环境

在 `qos_storage_sim` 项目根目录运行。

最低依赖：

```text
Python 3.10+
numpy>=1.24
matplotlib>=3.7
```

安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要文件：

```text
data
authenticated_workload_inputs.py
continuous_batch_sim.py
continuous_prefill_client.py
continuous_prefill_workload.py
random_steady_state_workload.py
sim.py
strategy_profiles.py
vb_pool_policy.py

# 4-NPU 实验
run_vb_continuous_nplus1_npu4.py
run_vb_queue_mixing_causal_test.py
plot_vb_queue_mixing_timelines_8s.py

# 32-NPU / 5-SSU 实验
run_vb_pool_steady_experiment.py
run_vb_queue_layout_npu32_ssu5_8s.py
```

本次使用的 `data` 文件校验值：

```text
SHA256(data) = fd197b79865b4c1f42d400100c5e05349ca1ba5f2d42b904af8a1759aabeb04b
画像数 = 84
```

`data` 中每个画像的语义为：

```python
table[(seq_len_k, nql)] = (
    required_bw_input_gbps,   # 名称沿用源码；本实验按 GiB/s 解释
    per_layer_us,
    source_ttft_ms,
    per_layer_kv_gib,
)
```

仿真时间由 `per_layer_us` 和 `per_layer_kv_gib` 决定。`source_ttft_ms` 只作为
来源元数据保存，**不能直接作为本实验的请求完成时间**。

---

## 2. 公共执行模型

### 2.1 逐层执行

每个请求按以下顺序执行：

```text
读取 L0 KV -> 计算 L0
计算 L0 时读取 L1 KV
计算 L1 时读取 L2 KV
...
计算倒数第二层时读取最后一层 KV
计算最后一层时预取同一 NPU 下一个请求的 L0 KV
```

关键参数：

```python
batch_size = 1
cross_request_layer0_prefetch = True
disk_bandwidth_gibps = 40.0       # 每个 SSU
npu_bandwidth_gibps = 50.0        # 每个 NPU 的接收上限
submit_batch_size = 1
client_issue_interval_us = 0.1
```

一个 NPU 同一时刻只计算一个请求。下一个请求可以提前读取 L0，但 admission 和
计算仍必须等待当前请求完成。因此低 V 请求若处在某个 NPU 的串行链中，后续高 V
请求不能越过它。

### 2.2 层内 I/O 与计算屏障

对请求 `r` 的第 `l` 层：

```text
io_start[r,l]       KV 读取开始
io_ready[r,l]       本层所有 KV 就绪
compute_start[r,l]  max(前一层计算结束, io_ready[r,l])
compute_end[r,l]    compute_start[r,l] + per_layer_compute_ms
```

暴露在 NPU 时间线上的 I/O 等待为：

```text
layer 0: admission -> compute_start[0]
layer l: compute_end[l-1] -> compute_start[l]
```

KV I/O 可以与上一层计算重叠；已经被计算掩盖的 I/O 不计入 NPU stall。

### 2.3 KV 放置

- 使用 token-block ring hash；
- 每个请求先生成一次物理 KV block 放置；
- 相同放置在所有层复用；
- 比较策略或重排 NPU 时，必须保留原 `request_id` 和 `placement`；
- **不能在固定分离后重新生成 placement**，否则布局处理会同时改变 SSU 热点分布。

---

## 3. V、B 与分类规则

对一个请求，定义：

- `C`：每层计算时间，单位秒；
- `D`：每层 KV 数据量，单位 GiB。

```text
B = D / C                 GiB/s
V = C / B = C^2 / D       s^2/GiB
```

32-NPU 实验使用固定阈值：

```python
V_CUTOFF = 0.00031
high_v = V > V_CUTOFF     # 注意是严格大于，不是 >=
low_v  = V <= V_CUTOFF
```

84 个画像中有：

```text
高 V 画像：48
低 V 画像：36
```

---

# 实验 A：4 NPU / 1 SSU 机制实验

## 4. 输入参数

```python
NUM_NPU = 4
NUM_SSU = 1
N_LAYERS = 8
DISK_BW_GIBPS = 40.0
NPU_BW_GIBPS = 50.0
SEED = 42
REQUESTS_PER_NPU = 249
WINDOW = [2000.0, 10000.0) ms
WINDOW_DURATION = 8000.0 ms
BLOCK_MS = 500.0
```

这里只使用 `data` 中两个真实画像：

| 类别 | Profile | C/层 | D/层 | B | V |
|---|---|---:|---:|---:|---:|
| 低 V | `(200,256)` | 9.043910535 ms | 0.268218994 GiB | 29.657413 GiB/s | 0.0003049460 |
| 高 V | `(48,512)` | 5.044309981 ms | 0.063781738 GiB | 12.644294 GiB/s | 0.0003989396 |

所有请求在 `t=0` 已经进入各 NPU 的串行后备队列。有限的 249 条/NPU 只是为了表示
无限输入，最终必须检查最后一个请求的 admission 晚于 10000 ms。

## 5. 两种请求布局

### 5.1 固定分离 `segregated`

```text
NPU0: L, L, L, L, ...
NPU1: H, H, H, H, ...
NPU2: H, H, H, H, ...
NPU3: H, H, H, H, ...
```

### 5.2 串行混排 `mixed_hhhl`

画像选择代码：

```python
profile = L if (sequence + npu_id) % 4 == 0 else H
request_id = sequence * 4 + npu_id
```

这会使每个 sequence 的全体 4 NPU 中恰好出现 1 个 L、3 个 H；每个 NPU 自己的
串行链都是相位错开的 `H/H/H/L` 周期。两个布局的全局 H/L 工作量完全相同。

## 6. 4-NPU Baseline

Baseline 是 Equal-CIR，不是 32-NPU 实验中的固定 Path-0 Baseline：

```python
path_cirs = (10.0, 10.0, 10.0, 10.0) + (0.0,) * 252
npu_dedicated_paths = (0, 1, 2, 3)
path_pirs = infinity
cross_request_layer0_prefetch = True
```

每个 NPU 有一个独立 Path，CIR 为 10 GiB/s；QoS 保持 work-conserving。

## 7. 4-NPU V/B

用每个 NPU 的第一个代表请求构造计划，参数：

```python
alpha = 1.02
V_CUTOFF = 0.00031
```

按 V 从高到低选择保护对象，三个高 V NPU 进入 LL 池，低 V NPU 进入 LS 池：

```text
LL protected CIR = 38.691539555 GiB/s
LS background CIR = 1.308460445 GiB/s
```

V/B 使用 `layer_once` 压力感知选路：每个“请求-层-SSU”读取一次最新 Path 压力，
`pressure_ttl_ms=0`，只在请求所属 LL/LS 池内选 Path。

## 8. 运行 4-NPU 实验

```bash
python -m py_compile \
  run_vb_continuous_nplus1_npu4.py \
  run_vb_queue_mixing_causal_test.py \
  plot_vb_queue_mixing_timelines_8s.py

python plot_vb_queue_mixing_timelines_8s.py
```

输出目录：

```text
results/vb_queue_mixing_timeline_8s/
```

预期结果：

| 布局 | Baseline | V/B | V/B-Baseline |
|---|---:|---:|---:|
| 固定分离 | 67.760346% | 76.751512% | +8.991166 pp |
| 串行混排 | 52.666371% | 52.590200% | -0.076171 pp |

固定分离的逐 NPU 利用率：

```text
Baseline: [33.665603%, 79.125260%, 79.125260%, 79.125260%]
V/B:      [ 7.009031%, 99.997016%, 100.000000%, 100.000000%]
```

---

# 实验 B：32 NPU / 5 SSU 固定分离稳态实验

## 9. 拓扑和稳态窗口

```python
NUM_NPU = 32
NUM_SSU = 5
N_LAYERS = 16
BATCH_SIZE = 1
DISK_BW_GIBPS = 40.0
NPU_BW_GIBPS = 50.0
SEED = 42
MASTER_REQUESTS_PER_NPU = 128

WARMUP_REQUESTS_PER_NPU = 8
SETTLE_MS = 500.0
MEASUREMENT_MS = 8000.0
BLOCK_MS = 500.0
```

测量窗口不是固定绝对时刻。步骤是：

1. 等待每个 NPU 都完成至少 8 个请求；
2. 再等待 500 ms；
3. 打开严格的半开窗口 `[window_start, window_start+8000)`；
4. 窗口内 admission 的请求可以在窗口外完成，但利用率只裁剪窗口内计算区间。

本次实际窗口：

```text
Baseline: [10723.912291509036, 18723.912291509034) ms
V/B:      [10559.020898966803, 18559.020898966803) ms
```

绘图时两者分别归一化为相对时间 `[0,8)` 秒，不能把两张图的相对 `t=0` 当成同一个
绝对仿真时刻。

## 10. 随机请求生成

模式：

```python
IID_UNIFORM_PROFILE_CATALOG_V1
```

对原始布局中的每个 `(npu_id, sequence)`：

- 从排序后的 84 个画像中独立、均匀、有放回采样；
- 随机流由 `(seed, npu_id, sequence)` 确定；
- `request_id = sequence * 32 + npu_id`；
- 每个 NPU 的初始启动抖动由独立确定性随机流从 `[0,5] ms` 生成；
- 同一 NPU 的后续请求使用相同初始 arrival 值进入串行队列。

不要用普通 `random.choice` 替换仓库中的 `_rng`，否则不会得到相同 trace 和指纹。

本次 4,096 条 master 请求实际包含：

```text
高 V：2346
低 V：1750
```

输入指纹：

```text
catalog hash  = d659bbd33b43cee791885f93b22a5cbf46244356c54835b0d1d06b6a7c9a25fc
schedule hash = d9e630cce5628b47336d2182a164bbdcf0b96480f880c475d0e88a0aedbd4d9e
workload hash = 5a8f2df85915fcbad4e8ebe874d66598eb563aaf3a634d1d46d745e2ae8038c3
placement hash= 921a020b04599be808e3c62d4872be736582bc9f2865b2e0a675dd5d3e0e9923
trace hash    = 062694374dedd96e3b12a90bf8ba0e2a2087e0e1b7d9956930145124f2f5bdeb
```

## 11. 固定分离重排

固定 NPU 分组：

```python
LOW_NPUS  = range(0, 14)   # NPU0--13
HIGH_NPUS = range(14, 32)  # NPU14--31
```

精确重排过程：

```python
low_jobs  = sorted(all low-V jobs,  key=request_id)
high_jobs = sorted(all high-V jobs, key=request_id)

for index, job in enumerate(low_jobs):
    target_npu = LOW_NPUS[index % 14]

for index, job in enumerate(high_jobs):
    target_npu = HIGH_NPUS[index % 18]
```

对每个目标 NPU，重新设置：

```text
request.npu_id
load["npu_id"]
load["stream_id"]        # 该目标 NPU 内从 0 开始的序号
load["arrival_ms"]
load["arrival_time"]
load["initial"]
```

必须保留：

```text
request_id
profile_key
per_layer_us
load["per_layer_kv_gb"]       # 名称沿用源码；数值按 GiB 解释
source_ttft_ms
placement
```

最后按 `(stream_id, npu_id)` 排序，保证所有请求 arrival 相同时，仿真器仍按目标 NPU
上的预期串行顺序 admission。

得到的队列深度：

```text
NPU0--13:  每个 125 个低 V 请求
NPU14--19: 每个 131 个高 V 请求
NPU20--31: 每个 130 个高 V 请求
```

Baseline 和 V/B 的固定布局指纹必须完全相同：

```text
physical job multiset hash = a7182afd3f5ccd645f3310daacb9785df69faedd4c96210c468707bbc92f1e4e
layout assignment hash      = d0cd92fb88a13d151c166fc83ddaf8c20cb4d67d7d1f6db7b7cfe0b8ab562e79
```

## 12. 32-NPU Baseline

Baseline 调用：

```python
qos_config = static_qos_config()
client_io_config = routing("baseline").client_config()
pressure_ttl_ms = 0.0
```

真实含义：

- 所有 I/O 固定进入 Path 0；
- 不读取 Path 压力；
- 256 Path、8组；
- 每组四类 Path 数量为 `(12,4,12,4)`；
- `(SS,SL,LS,LL)` 四类静态 CIR 总预算为 `(20,6,8,6)` GiB/s；
- Path PIR 为无限；所有 Path 和组权重均为 1；
- 数据面保持 work-conserving。

## 13. 32-NPU V/B

### 13.1 分类

每条请求独立计算 V：

```python
category = "LL" if V > 0.00031 else "LS"
```

### 13.2 池级 CIR

池预算只由完整 84 画像目录的分布计算，不读取未来随机 trace：

```text
E[fleet high-V B]
  = 32 * sum(B of 48 high-V catalog profiles) / 84
  = 120.42930154077847 GiB/s

uncapped_LL_per_ssu
  = 1.02 * E[fleet high-V B] / 5
  = 24.567577514318806 GiB/s

LL_CIR_per_ssu
  = min(0.98 * 40, uncapped_LL_per_ssu)
  = 24.567577514318806 GiB/s

LS_CIR_per_ssu
  = 40 - LL_CIR_per_ssu
  = 15.432422485681194 GiB/s
```

五个 SSU 使用相同池预算。

### 13.3 Path 配置与在线选路

- LL 使用保护池，LS 使用背景池；
- 8组，每组类别 Path 数量仍为 `(12,4,12,4)`；
- 池级 CIR 均匀拆分到对应类别的 Path；
- PIR 为无限，Path/组权重均为 1；
- 使用 `layer_once` 压力感知路由；
- 每个请求、每层、每个涉及的 SSU 读取一次最新 Path 压力；
- `pressure_ttl_ms=0`；
- 只能在请求所属 LL 或 LS 池内部选 Path，不跨池。

重要限制：本对比同时改变了**池级 CIR/请求分类**和**Path 选路方式**。因此
`V/B - Baseline` 不是只改变优先级的纯消融实验。

## 14. 运行 32-NPU 固定分离实验

先做静态检查：

```bash
python -m py_compile run_vb_queue_layout_npu32_ssu5_8s.py
python run_vb_queue_layout_npu32_ssu5_8s.py --dry-run
```

建议使用一个新的输出目录，避免错误复用旧 checkpoint：

```bash
python run_vb_queue_layout_npu32_ssu5_8s.py \
  --fixed-only \
  --max-workers 2 \
  --output-dir results/repro_npu32_ssu5_fixed_8s
```

两个 case 分别处理约 8644万和9079万个离散事件，不建议直接使用4个 worker。

如果中途被打断，用相同目录恢复已经完成的 case：

```bash
python run_vb_queue_layout_npu32_ssu5_8s.py \
  --fixed-only \
  --max-workers 2 \
  --resume \
  --output-dir results/repro_npu32_ssu5_fixed_8s
```

只从已有 CSV/JSON 重新绘图：

```bash
python run_vb_queue_layout_npu32_ssu5_8s.py \
  --plot-only \
  --output-dir results/repro_npu32_ssu5_fixed_8s
```

## 15. 利用率的精确统计方法

窗口为：

```text
[Ws, We), We - Ws = 8000 ms
```

任意计算区间 `[s,e)` 在窗口内的贡献：

```text
overlap(s,e) = max(0, min(e,We) - max(s,Ws))
```

每个 NPU：

```text
C_i = sum(overlap(layer_compute_start, layer_compute_end))
U_i = C_i / 8000
```

全体平均利用率：

```text
U_mean = sum(C_i) / (32 * 8000)
```

必须包括：

- 在 `Ws` 前 admission、但在窗口开始时仍活跃的 carry-in 请求；
- 窗口内 admission 的所有请求；
- 在 `We` 后才完成的 carry-out 请求，但只截取其窗口内部分。

不能使用：

```text
窗口内完成请求数 * 每请求纯计算时间
```

因为这种算法会漏掉 carry-in/carry-out 的部分计算，也会错误处理不同画像的计算时间。

请求级和逐层级导出必须分别满足：

```text
sum(request compute overlap) == simulator compute_ms_by_npu
sum(layer compute overlap)   == simulator compute_ms_by_npu
compute + io_barrier + other == 8000 ms, for every NPU
```

## 16. 请求完成点规则

一个完成点必须在**请求级**生成，不能从16层记录展开，否则同一请求会重复16次。

纳入条件：

```python
window_start <= completion_time_ms < window_end
```

唯一键：

```text
(layout, policy, npu_id, request_id)
```

检查规则：

- 同一个 NPU 的完成时间必须严格递增；
- 不允许同一 NPU 出现重复完成时间；
- 不同 NPU 在同一时刻完成在物理上是合法的，不能人为给时间加 jitter；
- 图中的每个标记必须对应 `request_completions_8s.csv` 的一行。

## 17. 32-NPU 预期结果

| 策略 | 8秒计算总量 | 平均利用率 | 高V完成 | 低V完成 | 事件数 |
|---|---:|---:|---:|---:|---:|
| Baseline | 156387.567551 ms*NPU | 61.088894% | 196 | 378 | 90,796,159 |
| V/B | 165214.211352 ms*NPU | 64.536801% | 198 | 377 | 86,441,034 |

```text
V/B - Baseline = +3.447908 percentage points
```

手算：

```text
Baseline = 156387.567551 / (32 * 8000)
         = 0.610888935746
         = 61.088894%

V/B      = 165214.211352 / (32 * 8000)
         = 0.645368013094
         = 64.536801%
```

按固定 NPU 组分解：

| NPU组 | Baseline | V/B | 变化 |
|---|---:|---:|---:|
| NPU0--13，低V专属 | 19.455145% | 18.953651% | -0.501494 pp |
| NPU14--31，高V专属 | 93.470698% | 99.990363% | +6.519665 pp |

计算时间变化：

```text
高 V NPU 总计算时间：134597.805138 -> 143986.122632 ms*NPU
高 V 收益：                              +9388.317495 ms*NPU

低 V NPU 总计算时间： 21789.762413 ->  21228.088720 ms*NPU
低 V 代价：                               -561.673693 ms*NPU

净收益：                                 +8826.643801 ms*NPU
净利用率变化：8826.643801 / 256000 = +3.447908 pp
```

## 18. “无限输入”验证

本实验用有限 trace 表示无限后备队列。除了仿真器的
`no_backlog_exhaustion=True`，还必须检查窗口排空结束时：

```text
queue_depth[npu] - completed_by_npu_at_stop[npu] >= 2
```

保留至少两个未完成请求，意味着既有当前活跃请求，也至少有一个后继请求。

本次最小剩余数：

```text
Baseline: 55
V/B:      61
```

因此8秒窗口内没有有限输入耗尽造成的伪空闲。

## 19. 输出文件

```text
01_npu32_ssu5_fixed_separation_baseline_vs_vb_timeline_8s.png
01_npu32_ssu5_fixed_separation_baseline_vs_vb_timeline_8s.pdf
02_fixed_separation_baseline_timeline_8s.png/.pdf
03_fixed_separation_vb_pool_timeline_8s.png/.pdf

case_summary_8s.csv
per_npu_window_accounting_8s.csv
request_completions_8s.csv
request_window_accounting_8s.csv
layer_intervals_8s.csv
result_8s.json
```

CSV 用途：

| 文件 | 用途 |
|---|---|
| `case_summary_8s.csv` | 两个策略的总结果和手算分子 |
| `per_npu_window_accounting_8s.csv` | 每个 NPU 的 compute/stall/other，可直接手算平均利用率 |
| `request_completions_8s.csv` | 所有请求完成点、精确时间和 NPU 内序号 |
| `request_window_accounting_8s.csv` | carry-in/out 后的逐请求窗口贡献 |
| `layer_intervals_8s.csv` | 每层 I/O barrier 和 compute 的原始时间区间 |
| `result_8s.json` | 参数、指纹、不变量和完整汇总 |

## 20. 自动复核脚本

在结果目录生成后运行：

```python
import csv
import math
from collections import defaultdict

root = "results/repro_npu32_ssu5_fixed_8s"

with open(f"{root}/per_npu_window_accounting_8s.csv", newline="") as f:
    accounting = list(csv.DictReader(f))

for policy in ("baseline", "vb_pool"):
    rows = [r for r in accounting if r["policy"] == policy]
    assert len(rows) == 32
    compute_ms = math.fsum(float(r["compute_ms"]) for r in rows)
    utilization = compute_ms / (32 * 8000)
    partition_error = max(
        abs(float(r["partition_sum_ms"]) - 8000) for r in rows
    )
    print(policy, compute_ms, utilization, partition_error)
    assert partition_error < 1e-6

with open(f"{root}/request_completions_8s.csv", newline="") as f:
    completions = list(csv.DictReader(f))

identity = {
    (r["layout"], r["policy"], r["npu_id"], r["request_id"])
    for r in completions
}
assert len(identity) == len(completions)

by_lane = defaultdict(list)
for r in completions:
    by_lane[(r["layout"], r["policy"], r["npu_id"])].append(
        float(r["completion_time_ms"])
    )

for lane, times in by_lane.items():
    times.sort()
    assert all(b > a for a, b in zip(times, times[1:])), lane

print("completion rows:", len(completions))
```

本次应得到：

```text
completion rows = 1149
唯一 request completion identity = 1149
同一 NPU 内重复完成时间 = 0
每 NPU 分区最大误差 <= 1.82e-12 ms
```

---

## 21. 解释结果时必须保留的限制

1. **4-NPU 与 32-NPU 的 Baseline 不同。** 前者是每 NPU 独立 10 GiB/s CIR，
   后者是仓库原始静态 QoS + 全部固定 Path 0。
2. **32-NPU 的 V/B 与 Baseline 不只是 CIR 不同。** V/B 还启用了池内压力感知
   多 Path 选路。
3. **固定分离是人为布局。** 它证明消除跨类别串行阻塞后，V/B 的收益可以转化为
   更高平均利用率；它不代表真实调度器一定能自由把请求迁移到指定 NPU。
4. **连续运行中两个策略消费的请求前缀不同。** Master trace 相同，但策略速度不同，
   8秒内实际完成的请求子集自然不同。这是稳态吞吐实验的正常现象。
5. **当前结果是单个 seed=42。** 若要形成统计结论，应增加多个 seed，报告均值、
   标准差和置信区间。
6. **不要把完成请求数直接当作 token 吞吐。** 不同画像的请求长度、每层计算量和
   KV 量不同，应同时统计请求数、token 数、TTFT/SLO 和队列稳定性。

## 22. 本地 AI 的验收标准

本地 AI 只有在以下条件全部满足时才能声明“复现成功”：

- `data` SHA256、画像数和输入指纹一致；
- Baseline/V/B 使用同一个固定布局和同一个物理请求多重集；
- 重排只改变 NPU 归属和串行序号，不改变 request ID、profile 或 placement；
- 两个策略都没有后备队列耗尽；
- 每个 NPU 的8秒状态分区严格闭合；
- 请求级、层级和仿真器计数器得到相同 compute 总量；
- 完成事件按请求生成，同一 NPU 内无重复时间；
- 利用率结果分别接近 61.088894% 和 64.536801%；
- 固定分离下的 V/B 增益接近 +3.447908 pp；
- 图上的完成标记数量与 `request_completions_8s.csv` 行数一致。
