# A/B 请求的 TTFT CDF 与 SLO×1 / ×1.5

本图采用同一组 A/B 画像的后续 **1:2 配比、32 NPU、3 SSU、Random、seed 7** 对照。数据来自已有运行日志，未重新运行仿真。

| 画像 | 总输入 | NQL | 单层读取 | 单层计算 | 带宽需求 |
|---|---:|---:|---:|---:|---:|
| A | 128K | 256 | 175.65625 MiB | 6.02401155 ms | 28.475926 GiB/s |
| B | 32K | 4096 | 38.5 MiB | 28.59284200 ms | 1.314932 GiB/s |

每盘 40 GiB/s，每 NPU 接收上限 50 GiB/s，每请求 8 层、batch=1。每张 NPU 共 40A+80B，完整输入共 3840 条请求，全部 t=0 到达，各卡随机打乱队列。

## 统计口径

- 统计 [2,4) 秒内接纳的请求，并跟踪到最终完成，包含窗后完成的请求。
- TTFT 口径为 `prefill 完成时间 − NPU 接纳时间`，不含接纳前排队；它是仿真的 prefill 完成延迟代理，并非实际首 token 事件时间。
- `SLO = 8 × 本请求单层纯计算时间`。A 的 SLO×1 / ×1.5 = 48.192092 / 72.288139 ms；B = 228.742736 / 343.114104 ms。
- 横轴 `x=TTFT/SLO`；纵轴 `F(x)=满足 TTFT ≤ x×SLO 的请求数 / 样本总数`。因此 F(1) 和 F(1.5) 分别为两个达标率。
- 每条请求等权，使用原始经验阶梯 CDF，不平滑、不拟合，展示完整长尾。
- 使用 1e-9 ms 浮点容差；与阈值的绝对耗时误差不超过此容差的绘图点吸附到阈值，避免无等待请求因数值舍入误判。
- 各策略推进速度不同，窗口接纳的请求集合及 A/B 比例可能不同；每个策略均保留了 32 个窗后完成请求。

## 全部请求

| 策略 | 请求数 | SLO×1 | SLO×1.5 |
|---|---:|---:|---:|
| Baseline | 341 | 53.67% (183/341) | 77.42% (264/341) |
| 流量分配（原 Once） | 343 | 60.64% (208/343) | 79.59% (273/343) |

## 分别统计 A、B

| 策略 | 类别 | 请求数 | SLO×1 | SLO×1.5 |
|---|---|---:|---:|---:|
| Baseline | A | 108 | 12.04% | 28.70% |
| Baseline | B | 233 | 72.96% | 100.00% |
| 流量分配（原 Once） | A | 106 | 9.43% | 33.96% |
| 流量分配（原 Once） | B | 237 | 83.54% | 100.00% |

根据更新要求，本图仅展示 Baseline 和流量分配；原始数据中的固定候选池记录保留用于溯源，不参与当前图和汇总表。

## 来源与复现

[原始逐请求样本](https://github.com/chguo0503/qos_storage_sim/blob/main/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/ttft_cdf_random_seed7/request_samples.csv)

[原始汇总](https://github.com/chguo0503/qos_storage_sim/blob/main/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/ttft_cdf_random_seed7/summary.csv)

SLO×1.5 的计数与原始汇总逐项核对一致；×1 根据同一份逐请求样本重新计算。来源版本和 Git blob SHA 记录在 `source_metadata.json`。

`plot_ab_ttft_cdf.py` 读取本目录的 `source_samples.csv` 和 `source_summary.csv`，重新生成 PNG、SVG、逐请求结果与汇总表。依赖 Python、NumPy 和 Matplotlib；中文字体子集及其许可随附在 fonts/。

```bash
python3 plot_ab_ttft_cdf.py
```
