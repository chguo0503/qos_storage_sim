# 任务：把控制频率降到毫秒级，并量化它与 SLO / NPU 利用率的关系

> 工作目录：本仓库根目录
> 背景文档：`CIR_UPDATE_PERIOD.md`、`ADAPTIVE_POLICY_IO.md`
> **全部在 32 NPU 上做**（单次 3~5 分钟，128 NPU 要 17~36 分钟）

---

## 1. 动机

当前两条控制路径都跑在**微秒级**，真实硬件做不到：

| 机制 | 动作 | 当前频率（每块盘） | 目标 |
|---|---|---:|---|
| `layer_once` | **读** Path 压力表 | 63~173 µs 一次 | **≥ 1 ms** |
| Adaptive V2.1 | **写** CIR 寄存器 | 8~15 ms 一次，每次写 127/128 个 Path | 保持毫秒级，**减少每次写的个数** |

两者机制不同：layer_once 只读不写（`cir_path_writes == 0`），V2.1 只写不读
（`pressure_reports == 0`）。**开销不可直接比较，要分别处理。**

需要回答的核心问题：**把频率降下来，TTFT SLO 和 NPU 利用率会掉多少？**

---

## 2. 三项改动

### 2.1 压力读取的 TTL 缓存（针对 layer_once）

**文件**：`sim.py`，`DiskIOScheduler.report_path_pressure_analysis`（约 738 行）

```python
def report_path_pressure_analysis(self, current_time):
    self.pressure_reports += 1
    self.settle(current_time)
    return PathPressureSnapshot(...)
```

改成带 TTL 的缓存。**关键陷阱**：`settle()` 有副作用（推进 flow 剩余字节和
时间戳），**不能连它一起跳过**，否则改变仿真语义。正确做法：

```python
def report_path_pressure_analysis(self, current_time):
    self.settle(current_time)                      # 永远执行，保持语义
    ttl = self.pressure_ttl_ms
    if (ttl > 0.0 and self._pressure_cache is not None
            and current_time < self._pressure_cache_time + ttl):
        return self._pressure_cache                # 命中缓存，不计数
    self.pressure_reports += 1                     # 只有真读才计数
    snap = PathPressureSnapshot(...)
    self._pressure_cache = snap
    self._pressure_cache_time = current_time
    return snap
```

`__init__` 加字段 `pressure_ttl_ms=0.0`（默认 0 = 关闭缓存 = 当前行为）。

> `PathPressureSnapshot` 的 `counts=self._path_io_counts` 传的是**可变列表引用**。
> 缓存时必须存**快照副本**（`tuple(...)`），否则缓存会跟着变，等于没缓存。
> 这一点务必确认。

### 2.2 CIR 写入阈值（针对 V2.1）

**文件**：`sim.py`，`DiskIOScheduler.update_path_cirs`（约 772 行）

```python
        changes = [
            (path_id, cir)
            for path_id, cir in enumerate(path_cirs)
            if abs(self.paths[path_id].cir - cir) > _EPS      # _EPS = 1e-12,等于没过滤
        ]
```

改为：

```python
        threshold = max(_EPS, self.cir_write_threshold_gbps)
        changes = [... if abs(...) > threshold]
```

`__init__` 加字段 `cir_write_threshold_gbps=0.0`（默认 0 = 当前行为）。

比较基准是 `self.paths[path_id].cir`（**实际写入值**），所以偏差不会累积——
累计变化超过阈值就会被写入。**不要改成与理论值比较。**

### 2.3 参数透传

两个构造点：`sim.py:2172`、`continuous_batch_sim.py:667`。在
`simulate_continuous_batch` 加同名 kwarg，默认 0.0，透传到每块盘。

### 2.4 硬约束

**两个默认值都是 0.0 时，必须与改动前逐位相同。** 先验证这一点（见 5.1）。

---

## 3. 工作负载改造：随机请求类型

### 3.1 当前的问题

`steady_state_workload.py` 给每个 NPU 一个相位偏移，`REQUESTS_PER_NPU = 32`、
4 个类别，`32 % 4 == 0`，导致**每个 NPU 都拿到严格的 8/8/8/8**：

```
   per-NPU category mix distinct patterns: 1  ->  (LL:8, LS:8, SL:8, SS:8)
   per-NPU ms/GB : min = 48.0310  max = 48.0310   spread = 0.0000%
```

**128 个 NPU 的长期需求完全相同**，"按需分配"无从谈起。

而且只用了 4 个固定 profile，但 `data` 里有 **84 个**：

```
   全部 84 个 profile:  raw demand 1.31 ~ 105.44 GB/s   (跨度 80.2 倍)
   当前用的 4 个:       raw demand 13.81 ~  83.35 GB/s  (跨度  6.0 倍)

   SS (80,64)   bw=83.35 GB/s   SL (80,512)  bw=13.81 GB/s
   LS (144,256) bw=28.83 GB/s   LL (144,512) bw=14.71 GB/s
```

84 个 profile = `seq_len_k` ∈ {32,48,...,200} × `nql` ∈ {64,128,...,4096}。

### 3.2 要求

新建一个 workload 构造函数（**不要改 `steady_state_workload.py`**，正式结果依赖它）：

- 每个 NPU 连续输入请求，**类型从 84 个 profile 中随机抽取**，不是固定 4 个；
- **每个 NPU 的请求序列独立随机**，不再是相同配比；
- 保持可复现：用固定 seed，记录 workload hash；
- 保持每 NPU 的请求数不变（32），便于与现有基线对比。

### 3.3 需要报告的工作负载特征

改造后必须报告（这些数字本身就是结论的一部分）：

```
   per-NPU raw demand:  min / max / mean / 变异系数
   per-NPU ms/GB:       min / max / spread
   fleet raw demand D:  总量
```

**预期**：per-NPU demand 不再全相同。如果仍然接近相同，说明随机化没生效。

> **重要**：改了 workload 之后，**不能再和现有的 32-NPU 基线直接对比**
> （`results/steady_state_32npu_adaptive_v2_1/`），因为输入变了。必须在新
> workload 上重跑一个 `ttl=0, threshold=0` 的基线作为对照。

---

## 4. Warm 稳态：这次实验最容易出错的地方

**所有数据必须在 warm 满载稳态下采集。** 现有实验用 `STEADY_CONFIG` 保证这点：

```
   warmup_requests_per_npu = 4      每个 NPU 先跑完 4 个请求
   settle_ms               = 500    再静置 500 ms
   measurement_ms          = 2000   然后才开始 2 秒测量窗
   slo_alpha               = 2.0
```

四个阶段：`warmup → settle → measurement(取数) → drain`。实测 warmup 时长
**随负载差 5 倍**：

```
   32 NPU            warmup 到达      测量窗            tail drain
     SSU6  (过载)     2521.0 ms    [3021, 5021]       1375.5 ms
     SSU10 (中等)     1010.3 ms    [1510, 3510]        325.3 ms
     SSU18 (轻载)      524.4 ms    [1024, 3024]        197.8 ms
```

### 为什么这次特别危险

本任务的两项改动**都会改变 warmup 时长**：

1. **随机 workload**：每个 NPU 的请求组成不同，完成 4 个请求的耗时不同。
   现在同构所以齐步走；随机化后 `min(completed_by_npu) >= 4` 会被**最慢的
   NPU** 拖住，warmup 可能显著变长。
2. **降低控制频率**：控制变慢会拖慢整体推进，warmup 跟着变长。

**如果沿用固定 warmup 参数而不检查，低频档次可能在尚未进入稳态时就开始测量。**
那样测到的 SLO 下降会被误判成"降频的代价"，实际是"没等 warm"。
这个混淆会直接导致错误结论，且事后很难发现。

### 强制要求

1. **每次运行报告四段时间**，不能只报最终指标：
   `warmup_reached_ms`、`measurement_start_ms`、`measurement_end_ms`、
   `drain_stop_ms`、`tail_drain_ms`。

2. **三条 invariant 必须全为 True**，否则该次结果作废：

   | invariant | 含义 |
   |---|---|
   | `no_backlog_exhaustion` | 测量窗内每个 NPU 始终有请求排队（**最关键**：为 False 说明负载不饱和，不是满载稳态） |
   | `all_tagged_requests_completed` | 被标记的请求全部完成 |
   | `measurement_window_closed` | 窗口正常闭合 |

3. **随机 workload 下重新校准 `warmup_requests_per_npu`。** 4 个是为同构负载
   定的，异构下最慢的 NPU 可能需要更多。做法：先跑一次新 workload 基线，
   检查 `no_backlog_exhaustion` 和 warmup 时长是否合理；若临界，把该参数调大
   （如 6 或 8）并**在所有档次统一使用**。

4. **跨档次对比前确认 warmup 可比。** 若某档的 `warmup_reached_ms` 比同 SSU
   的基线长出 **50% 以上**，该档结果需单独复核——这是"没等 warm"最容易
   察觉的信号。

5. **不要为了缩短实验而削减 `settle_ms` 或 `measurement_ms`。** 32 NPU 单次
   只要 3~5 分钟，没有必要。

## 5. 实验

全部 32 NPU，SSU ∈ {6, 10, 18}。现有同构基线供参考（旧 workload）：

```
  SSU6   util=0.3301 slo=0.7524 writes=39378  wall=188s
  SSU10  util=0.5700 slo=0.8457 writes=48010  wall=206s
  SSU18  util=0.9338 slo=0.9885 writes=37158  wall=310s
```

### 5.1 回归验证（先做）

用旧 workload + 两个参数都为 0.0，重跑上面三个点，**三项指标必须完全一致**：
`mean_npu_utilization`、`ttft_slo_attainment`、`cir_path_writes`。
不一致就停下排查。

### 5.2 新 workload 基线

用新的随机 workload，参数仍为 0.0，跑 SSU ∈ {6,10,18}。这是后续所有对比的基准。
同时报告 §3.3 的工作负载特征。

### 5.3 主扫描

新 workload 上，两个维度**分别**扫（不要一开始就交叉，先看各自的曲线）：

**A. 压力读取 TTL**（只对 layer_once 有意义）

| ttl_ms | 0.0（基线） | 0.25 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|

**B. CIR 写入阈值**（只对 Adaptive V2.1 有意义）

| threshold_gbps | 0.0（基线） | 0.05 | 0.25 | 0.5 |
|---|---|---|---|---|

**C. Adaptive 的决策间隔**（已有 25/50ms 的 128-NPU 数据，补 32-NPU 的毫秒级点）

| min_interval_ms | 25（基线） | 50 | 100 | 200 |
|---|---|---|---|---|

每档 × 3 个 SSU。A 有 4 个非基线档 ×3 = 12 次，B 有 3×3 = 9 次，C 有 3×3 = 9 次，
加基线共约 **36 次运行 ≈ 2.5~3 小时**（可并行）。

### 5.4 每次运行必须记录

| 指标 | 用途 |
|---|---|
| `mean_npu_utilization` | 核心 |
| `ttft_slo_attainment` | 核心 |
| `request_weighted_slo_attainment` | 核心 |
| 按 profile 分组的 SLO | 高 demand 的请求最先被牺牲，重点看 |
| `pressure_reports` | A 组的核心开销指标 |
| `cir_path_writes` / `cir_commits` | B、C 组的核心开销指标 |
| `control_evaluations` | **自检**：B 组必须与基线相同 |
| `warmup_reached_ms` | **warm 校验**（见 §4）：比基线长 50%+ 需复核 |
| `measurement_start_ms` / `end_ms` / `drain_stop_ms` / `tail_drain_ms` | 四段时间 |
| `invariants` 全部（尤其 `no_backlog_exhaustion`） | 任一为 False 则该次结果作废 |

---

## 6. 要回答的问题

1. **每块盘的压力读取间隔提到 1 ms、2 ms、5 ms，SLO 和利用率各掉多少？**
   拐点在哪？1 ms 是否可接受？
2. **CIR 写入阈值 0.05 / 0.25 / 0.5，SLO 掉多少？** 换来多少写入减少？
3. **决策间隔 50 / 100 / 200 ms，SLO 掉多少？**（128-NPU 上 25→50ms 掉了
   6.6~7.2 pp，32-NPU 是否同样敏感？）
4. **三个手段里，哪个"每省 1% 开销"的 SLO 代价最小？** 这是最终选型依据。
5. **随机化 workload 后，结论是否改变？** 特别是：需求异构之后，
   NPU 利用率是否开始对分配策略敏感（旧 workload 下它被吞吐钉死，不敏感）。
6. 三个 SSU 工作点（过载 6 / 中等 10 / 轻载 18）的敏感度差异。
7. **随机 workload 下 `warmup_requests_per_npu=4` 是否仍然够用？** 若不够，
   最终采用的值是多少，依据是什么（见 §4.3）。

### 已有的部分线索（128 NPU，旧 workload，只读观测）

CIR 写入阈值能省多少（**只说明能少写多少，不说明 SLO 会怎样**）：

```
   阈值(GB/s)     SSU24 省      SSU70 省
      0.010        -66%          -36%
      0.050        -91%          -57%
      0.250        -92%          -83%
      0.500        -97%          -93%
```

**两个工作点差异很大，SSU24 在 0.05~0.25 有平台期而 SSU70 没有。**
不要假设一个阈值适用所有场景。

---

## 7. 判据与纪律

- **推翻信号**：某个手段让 SLO 掉超过 2 pp 而开销只降一点，说明它不划算。
  **如实报告，不要调参去凑好看的结果。**
- **自检**：B 组（写入阈值）的 `control_evaluations` 必须与基线相同。若变了，
  说明改动泄漏到了控制路径。
- **不要覆盖 `results/` 下的冻结结果**，输出到新目录如
  `results/ms_scale_control/`。源码改动会让 `source_fingerprint` 变化、
  runner 可能报 `source_stable: false`，这是预期的。
- **不要动 `_EPS` 本身**，它在 `sim.py` 多处复用。
- **任何 `no_backlog_exhaustion == False` 的运行都不得进入结论**，即使它的
  util/SLO 看起来"正常"。那不是满载稳态，数字没有意义。
- **不要把 warm 不足造成的 SLO 下降报告成降频的代价**——这是本任务最主要的
  误报风险，见 §4。
- 项目约定不保留 `test_*.py`（见 `README.md`），验证靠 fingerprint 和 invariant。

---

## 8. 交付物

1. 代码改动（`sim.py`、`continuous_batch_sim.py`、新 workload 文件），默认行为不变。
2. `results/ms_scale_control/results.json` + `report.md`。
3. 分析报告，回答 §6 的六个问题，并给出**推荐配置**：
   压力读取 TTL、CIR 写入阈值、决策间隔各取多少，理由是什么。
4. 新 workload 的特征统计（§3.3）。

---

## 9. 快速上手

```bash
cd /path/to/qos_storage_sim

python3 adaptive_policy_standalone.py --selftest      # 策略本体，秒级
python3 steady_state_32npu_adaptive_v2_1_experiment.py --help
cat results/steady_state_32npu_adaptive_v2_1/report.md
```

关键位置：

```
   sim.py:738                     report_path_pressure_analysis —— 加 TTL 缓存
   sim.py:772                     update_path_cirs —— 加写入阈值
   sim.py:677                     DiskIOScheduler.__init__ —— 加两个字段
   continuous_batch_sim.py:667    scheduler 构造点 —— 透传
   steady_state_workload.py       现有 workload（不要改，照着写新的）
   six_request_workload.py:28     BALANCED_PROFILES —— 当前只用的 4 个 profile
```
