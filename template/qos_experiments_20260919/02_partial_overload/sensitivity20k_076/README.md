# 2026-09-14：sensitivity20k_076 核对说明

用户指定的实验是 `lower_fifo_followup_20260914/native20/sensitivity20k_076_seed7_{fifo,once}`。此前将其匹配为 `mixed8` 是错误的，两个实验不可合并。此次仅核对、汇总原有代码与结果，没有重跑仿真。

## 直接打开

- [带宽需求与实际SSD供给原图](original_figures/sensitivity20k_bandwidth_pair.png)
- [Baseline/Once八卡计算时序原图](original_figures/sensitivity20k_timeline_pair.png)
- [整体及A/B的TTFT CDF原图](original_figures/sensitivity20k_ttft_cdf_pair.png)
- [结果表](verified_metrics.csv) · [请求画像表](verified_profiles.csv)
- [原始报告](original_data_code/lower_fifo_followup.md)
- [运行代码](original_data_code/qos_storage_sim/run_20k_sensitivity.py) · [绘图代码](original_data_code/qos_storage_sim/newexport_lower_fifo_sensitivity.py)
- [输入配置](original_data_code/qos_storage_sim/results/lower_fifo_followup_20260914/native_20k_specs.json)

上述三图均与历史原图SHA256相同。当前尚未找到可验证属于该实验的独立逐NPU需求/供给PNG；原数据含每NPU的2ms实际接收字节，保存在各策略的receipts.json。本次没有新生成图片或重新仿真，也没有用mixed8的逐卡图替代。

## 配置与命名

- 8 NPU、1 SSU、8 层；SSU 容量 40 GiB/s，NPU 接收上限 50 GiB/s。
- 每卡 7 个 A 长请求 + 42 个 B 短请求，A:B=1:6；8 张卡共 392 个请求。
- 每卡独立随机混排，seed=7；每卡随机种子为 `7+100003*npu_id`。全部请求在 t=0 到达，固定分卡、batch size=1。
- A 总长度 200K、NQL 中心 2048，实际 NQL 2042–2054。
- B 总长度 20K、NQL 中心 **1152**，实际 NQL 1126–1178。该实验不是固定 1024。
- ring hash、每盘 256 虚拟节点；同一请求同一 KV block 的各层使用相同盘映射。Baseline 为 FIFO Path0。
- `076` 是 `fifo_20k_sensitivity.py` 中 `enumerate(grid)` 的候选编号（从 0 开始，第 77 个组合），不是 0.76 的计算时间缩放系数。
- 20K 单层计算时间由原有 32K/48K 数据按权重 1.75/−0.75 外推，未实测校准；这是一组原生仿真器执行的敏感性实验。

## 结果（2–4 秒）

| 指标 | Baseline/FIFO | Once |
|---|---:|---:|
| 整机 NPU 利用率 | 94.6147% | 99.2567% |
| A 长请求利用率 | 99.4464% | 98.9465% |
| B 短请求利用率 | 86.7022% | 99.8375% |
| SSU实际平均带宽 | 30.0216 GiB/s | 31.3784 GiB/s |
| SSU实际带宽利用率 | 75.0540% | 78.4460% |
| 全部 warm 入场请求 TTFT SLO×1.5 | 98.4375%（126/128） | 100%（137/137） |
| A 的 warm TTFT SLO×1.5 | 100%（18/18） | 100%（18/18） |
| B 的 warm TTFT SLO×1.5 | 98.1818%（108/110） | 100%（119/119） |

利用率统计按 [2,4) 秒裁剪计算与 active 区间；整机分母是 8×2000 ms，逐类利用率为该类计算时间除以该类 active 时间。SLO 统计的是在 [2,4) 秒被 NPU 接纳的请求，TTFT=完成−接纳，不包括接纳前排队；阈值为 1.5×8×原始单层计算时间。

原始 CDF 使用相同的全部 392 个输入请求，Baseline 达标率为 **387/392=98.7245%**，Once 为 **392/392=100%**，不能与 warm 的 98.4375% 混用。全输入 A：56/56→56/56；B：331/336→336/336。`verified_metrics.csv` 同时保留 warm 和全输入口径，以及 SLO×1 和 SLO×1.5；全输入行未填写未核对的全程利用率。

## 带宽与负载口径

原代码的带宽单位是 **GiB/s**。

| 请求 | 名义中心每层读取 | 名义中心每层计算 | 名义中心需求 | 实际需求范围 |
|---|---:|---:|---:|---:|
| A：200K / 2048 | 272.250000 MiB | 70.740538 ms | 3.758370 GiB/s | 3.747303–3.769502 GiB/s |
| B：20K / 1152 | 25.953125 MiB | 5.881291 ms | 4.309402 GiB/s | 4.210916–4.412321 GiB/s |

因此“4.6/4.5”不能作为本实验精确参数写回索引，实验 ID 和保存的配置是匹配依据。输入平均总需求 31.534335 GiB/s；全请求组合静态上界 35.298572 GiB/s；两策略全程当前请求名义需求峰值均为 34.153895 GiB/s，小于 40。

若把短↔长交界预取的 deadline 需求纳入，则存在局部突发：Baseline 总预取需求峰值 117.138593 GiB/s，Once 114.635804 GiB/s。原实验采用“普通需求全程欠载，交界预取豁免”的口径；实际 SSD 服务按 2 ms 真实字节统计，峰值两策略均约 40 GiB/s（浮点误差范围内），没有物理供给超过容量。

## 原始文件定位

以下路径相对于 `original_data_code/qos_storage_sim/`：

- 执行代码：`run_20k_sensitivity.py`。
- 输入规格：`results/lower_fifo_followup_20260914/native_20k_specs.json`。
- 原始数据：`results/lower_fifo_followup_20260914/native20/sensitivity20k_076_seed7_fifo/` 与对应 `once/`，每个目录含 manifest、result、metadata、metrics、receipts。
- 逐请求/逐层表：`results/lower_fifo_followup_20260914/csv/sensitivity20k_076_seed7_{fifo,once}_{requests,layers}.csv`。
- 绘图代码：`newexport_lower_fifo_sensitivity.py`。
- 原始合图：`sensitivity20k_bandwidth_pair.png`（需求与真实 SSD 服务）、`sensitivity20k_timeline_pair.png`（8 卡时序）、`sensitivity20k_ttft_cdf_pair.png`（全部/A/B CDF）。
- 图数据及审计：`results/lower_fifo_followup_20260914/figures/data/`。

`verification.json` 记录原生数据与图片审计中保存的 SHA256 核对结果。FIFO/Once 的输入指纹完全相同；warm 的计数及达标数已由逐请求 CSV 重算并与原始 metrics 匹配。

SSU实际平均带宽与利用率由原2ms物理供给CSV在[2,4)s按时间加权得到，并与receipts.json中的window_ssu_read_bandwidth_gib_s核对。完整带宽值见 [verified_bandwidth.csv](verified_bandwidth.csv)。
