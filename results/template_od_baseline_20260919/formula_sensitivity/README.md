# 原输入新增 OD：五组公式实验与 sensitivity20k_076

本目录保存 16 次 OD 运行：五组公式实验各 seed 7 / 19 / 43，以及 sensitivity20k_076 的 seed 7。仅输出五张公式 CDF、一张 sensitivity CDF、一张 sensitivity 计算时序 PNG。未运行 mixed8、E1 独立实验，也未生成 sensitivity bandwidth_pair。

## 输入和版本

使用模板归档的原始请求，未改队列、NPU 分配、到达时间、读取量、计算时间或 Ring Hash。全部为 8 NPU、1 SSU、8 层、batch size 1，所有请求 t=0 到达；允许最后一层计算时预取下一请求的第 0 层。176 KiB 命令末尾保留原来的精确尾块，不补齐读取量。

- 五组公式：每卡 A:B=1:12，三个 seed；每 seed 分别 416、520、520、624、624 请求。原物理盘容量 **40 GB/s（十进制）**，NPU 链路 50 GB/s，转换为 GiB/s 后传入原引擎。
- sensitivity20k_076：392 请求，每卡 7 A + 42 B，seed 7；盘容量 **40 GiB/s**，NPU 链路 50 GiB/s。A 为 200K / miss 2042–2054，B 为 20K / miss 1126–1178。该输入不是固定 20K / miss 1024。
- 原实验部分画像使用 NQL 插值与小于 32K 的计算时间外推。本次沿用原值，不将其标成实测。

`frozen_source/` 是未修改的原始 flat 版源码副本，两族核心和 exact-tail adapter 的 SHA 相同。ASU 先各选一例完整复跑，逐请求、逐层、事件计数、盘统计及总量与归档**完全一致**。证明见两份 `runs/*_asu_baseline/parity.json`。

## OD 接入

`od_policy_snapshot.py` 保存当前 `simulator/policies/od_baseline.py` 原样副本；`od_qos_configs.json` 由其 `qos_config(8, capacity)` 导出。

`run_case.py` 在原 `exact_tail_adapter('baseline')` 上仅替换 `_plan_paths`：执行 NPU n 使用路径 `32*n`，每卡跨层、跨请求第 0 层始终保持独占路径。每盘 8 条可用路径分别位于 8 个硬件组，CIR=该盘容量/8、PIR 无限；闲置份额可以按原生仲裁借用。没有动态 CIR 更新，没有请求重排，也没有修改盘内 FIFO、WFQ、链路或事件时序。

每个完成 I/O 都核对执行 NPU 与路径归属，单独核对第 0 层；`metrics.json` 记录实际路径计数、CIR 和输入指纹。保留原 5 ms 采集事件。

## 指标口径

- `U_warm_percent`：逐层计算区间与 [2,4) 秒相交的时间 / (8 卡 × 2 秒)。多 seed 取均值。
- `U_full_percent`：每 seed 从 0 到最后完成的整机利用率，再取 seed 均值。
- `SLO_full_1p5_percent`：合并 seed 的**全部相同输入请求**。TTFT=完成−NPU 接纳；阈值为各请求自身 8 层纯计算时间 ×1.5，包含冷启动、不含接纳前排队。所有 CDF 使用这个群体。
- `SLO_warm_1p5_percent`：仅接纳时间位于 [2,4) 的请求，保留窗口之后的最终完成，再合并分子分母。策略间 warm 入场集合可以不同。
- `SLO_warm_1p5_seed_mean_percent` 额外提供各 seed 达标率的简单均值，避免与合并计数混淆。
- 类别 U 为该类计算时间 / 该类占卡时间，不能把它等同于该类对整机 U 的贡献。

时序图画 [2,4)；CDF 画全输入。原 `summary.csv`/`summary.json` 明确区分两者。

CDF 保留各自原绘图的精确浮点运算：公式组分母为 raw 的 `own_compute_ms`；sensitivity 为逐层 `compute_end_ms-compute_start_ms` 的顺序和。两者代表同一纯计算时间，但约 1e-14 的差异会影响相等点的阶梯合并，因此不擅自统一运算。ASU/Once 的公式逐请求 CSV 和 sensitivity 原 CDF 坐标均逐点核对；1.5 附近没有影响达标计数的浮点边缘样本。

## 文件及复现

- `summary.csv` / `summary.json`：六组 × 三策略，含 warm/full 两种统计。
- `per_seed_metrics.csv`：每 seed 明细。
- `cdf_samples.csv.gz` / `cdf_points.csv.gz` / `cdf_statistics.csv`：绘图样本、经验 CDF 点、类别分位数。
- `runs/`：16 次 OD raw summary、CDF 样本和指标，加两次 ASU parity。
- `inputs/`：冻结 manifest；`jobs.json` 和 `source_provenance.json` 记录指纹与归档 SHA。
- `verification.json`：独立逐层积分、SLO、路径归属、I/O 总量、原档未改与 CDF 请求集合审计。

```bash
python results/template_od_baseline_20260919/formula_sensitivity/prepare.py
python results/template_od_baseline_20260919/formula_sensitivity/run_case.py --case XY12_32_random_s1_seed7 --policy od_baseline
python results/template_od_baseline_20260919/formula_sensitivity/build_figures.py
python results/template_od_baseline_20260919/formula_sensitivity/audit.py
```

运行器拒绝覆盖已经完成的结果；上面的单例运行命令用于新副本或未完成目录。ASU/Once 曲线直接读取原归档，原归档从未覆盖。
