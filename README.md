# QoS Storage Simulator — Adaptive V2.1 steady-state package

本目录只保留当前 warm/full-load 调查的核心仿真器、真实 `data`、Adaptive
V2.1 策略、正式对照结果及其分析证据。历史 cold/warm、routing 和
full-prefill 实验已经移出项目。

## 最终结论与结果

- [完整中文报告](BASELINE_QOS_STEADY_STATE_REPORT.pdf)
- [报告 Markdown](BASELINE_QOS_STEADY_STATE_REPORT.md)
- [128-NPU Adaptive 结果](results/steady_state_128npu_adaptive_v2_1/report.md)
- [32→128 联合分析](results/steady_state_128npu_admission_v1/scale_analysis.md)
- [2 NPU × 2 SSU baseline 实验](results/baseline_2npu_teaching/report.md)
- [Ring placement 审计](results/ring_ownership_scale/analysis.md)
- [合成带宽满足度审计](results/synthetic_bandwidth_satisfaction/results.md)
- [32-NPU CIR 控制：平均 NPU 利用率](results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_plots/01_mean_npu_utilization_vs_ssu.png)
- [32-NPU CIR 控制：TTFT SLO 1.5×](results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_plots/02_ttft_slo_alpha1p5_vs_ssu.png)
- [32-NPU CIR 控制：TTFT SLO 2×](results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_plots/03_ttft_slo_alpha2_vs_ssu.png)
- [32-NPU CIR 控制：49 行汇总](results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_analysis/summary.csv)

128 NPU、16 层、batch=1、真实 `data` 的最终 Adaptive V2.1 结果：

| SSU | Baseline util / SLO | Adaptive util / SLO |
|---:|---:|---:|
| 24 | 32.76% / 25.00% | 32.51% / 52.10% |
| 40 | 51.60% / 50.00% | 51.57% / 82.45% |
| 70 | 88.93% / 77.33% | 90.85% / 98.88% |

## 当前实现

- `sim.py`：SSD40 命令级仲裁和基础数据面。
- `continuous_batch_sim.py`：warm/full-load、NPU50 和测量窗口。
- `policy_logic.py`、`continuous_batch_control.py`：策略与 grant 分配基础。
- `slo_admission_scheme_b.py`：V1 request admission/coflow residual。
- `slo_admission_scheme_b_v2.py`：V2 selected-first explicit spill。
- `adaptive_admission_scheme_b_v2_1.py`：按当前 selected fraction 自适应选择
  V1/V2 residual，25 ms event-gated 控制。

主要 runner：

```bash
python steady_state_128npu_adaptive_v2_1_experiment.py --help
python steady_state_32npu_adaptive_v2_1_experiment.py --help
python steady_state_128npu_admission_experiment.py --help
python steady_state_128npu_admission_v2_experiment.py --help
```

重新生成联合分析：

```bash
python analyze_steady_state_128npu_scale.py
python analyze_ring_ownership_scale.py
```

## 保留边界

`steady_state_32npu_experiment.py`、`steady_state_experiment.py` 和
`steady_state_128npu_admission_experiment.py` 虽然名称像旧 runner，但仍是当前
Adaptive runner 的运行依赖，不可删除。

冻结的 128-NPU Adaptive runner 还存在一个已记录的 provenance 缺口：它间接导入
`steady_state_32npu_experiment.py`，但没有将该文件计入自己的 `SOURCE_FILES`。
现有正式结果因此保持原样；下一版本应先补齐完整传递 import 指纹再重跑。

按当前维护约定，项目不保留 `test_*.py`。正式结果依靠 source/config/case
fingerprint、跨策略输入配对哈希和运行时 invariant 保留审计信息。
