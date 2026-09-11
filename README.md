# QoS Storage Simulator

研究 NPU 计算、KV 预取与共享 SSD 调度的离散事件模拟器。仓库保留核心模型、策略、回归测试、实验复现入口，以及冻结输入、原始结果和独立审计。

## 最新结果

- **[持续混合长短请求：60 秒实验报告](results/baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/report.md)**：32 NPU、6 SSU，原始 `data` 画像；对比同一逐卡人口的 Random / Ordered 与 Baseline / Once per layer，主窗为 `[2,60)` 秒。包含分段统计、未达目标的前序尝试，以及人为构造输入的适用边界。
- **[每卡带宽诉求、SSD 真实服务、链路交付与 stall](results/baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/bandwidth_delivery/README.md)**：32 张逐卡对照图、两张概览与单层累计字节证据；局部观察窗为 `[3.2,4.0)` 秒，与长窗平均分开解释。
- [正式长窗时间线与需求图](results/baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/figures/separate/README.md) · **[全部实验与教学资料导航](results/README.md)**。

这些结果是特定输入与模型下的实验，不是所有工作负载的策略排名。不同实验的画像、分卡、顺序、到达规则、盘数和统计窗口不同，不能直接混排利用率。最新长测的主四格使用 seed 7；较早的五种子短窗研究是另外一组实验。

## 历史研究入口

| 主题 | 入口 |
|---|---|
| 原始画像、固定分卡与每卡混合，五种子对照 | [最终短窗对照](results/baseline_32npu6ssu_underload/raw_role_followup_20260909/final_comparison.md) · [扩大窗口后的变化](results/baseline_32npu6ssu_underload/raw_role_followup_20260909/window_sensitivity/report.md) |
| 固定长卡/短卡，Random 与 Round-robin | [角色分离结果](results/baseline_32npu6ssu_underload/role_separated_5seeds/comparison.md) · [输入分配和独立图](results/baseline_32npu6ssu_underload/role_separated_5seeds/figures/separate/README.md) |
| 从原始 `data` 抽样，而非仅打乱选定画像 | [原表随机输入](results/baseline_32npu6ssu_underload/raw_data_random_5seeds/comparison.md) · [原始四画像替换](results/baseline_32npu6ssu_underload/raw_quartet_replacement/comparison.md) |
| 早期 32 卡混合输入与多策略调查 | [完整报告](results/baseline_npu32_investigation/docs/report.md) · [可编辑演示](results/baseline_npu32_investigation/presentation/README.md) |
| 5 ms 共享状态、coflow 与多 SSU 预测 | [Coflow](results/coflow_global_5ms_experiments/docs/coflow_global_5ms_report.md) · [Shared-path](results/shared_path_5ms_experiments/docs/shared_path_5ms_report.md) · [Multi-SSU](results/multi_ssu_stall_experiments/docs/multi_ssu_experiment_report.md) |
| 两份 Baseline 高/低利用率报告 | [报告及原始证据](baseline_two_reports/README.md) · [输入与容量复核](results/baseline_two_reports_analysis/README.md) |
| 预取、miss、外部到达与内部反馈教学 | [修订版解释 PDF](results/prefill_miss_open_closed_guide/prefill_miss_open_closed_explained_v2.pdf) · [早期 4 卡 Path0 教程](results/baseline_4npu_ssu1_low_utilization/baseline_path0_low_utilization_tutorial_v3_2.pdf) |
| 早期固定 28:4 V/B 实验 | [结果与统计边界](VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md) · [策略与硬件假设](VB_POLICY_IMPLEMENTATION_AND_HARDWARE_FEASIBILITY.md) |

## 读数时的约定

- 设备利用率通常是窗内计算时间总和除以 `NPU 数×窗口长度`；逐请求等权利用率、画像条件利用率与完整批次利用率是其他口径。是否全卡 active、是否提前耗尽输入，需看对应审计。
- **Admission SLO 是接纳后的处理时间代理，不等于端到端 TTFT。** `completion−admission` 不含接纳前排队；从外部到达计时要使用 `completion−arrival`。具体门限、纳入人口和跨窗完成的处理方式以各报告为准。
- 当前请求的单层读取量/计算时间 `D/C` 是约定的名义需求。它与 SSD 实际服务、链路交付、下一层在截止前应读齐的字节量不同；平均或逐时刻名义欠载不保证 FIFO 无等待。
- `once` / Once per layer 与 `new_once` 是不同策略。画像的“短/长”描述也不自动等于模拟器的 SS/SL/LS/LL 分类。

## 源码与依赖

源码位置保留，以兼容既有导入、实验命令与来源哈希。

| 文件/前缀 | 职责 |
|---|---|
| `sim.py`、`continuous_batch_sim.py` | SSD 调度、链路与计算/预取事件模型 |
| `coflow_*`、`shared_path_*`、`*_predictor.py`、`*_assignment.py` | 控制策略、状态适配、预测与分卡 |
| `data`、`authenticated_workload_inputs.py`、`*_workload.py` | 原始 84 画像表及输入构造/核验 |
| `run_*`、`sweep_*` | 单格实验、冻结输入与批量编排；可能运行较久 |
| `build_*`、`plot_*`、`make_*`、`validate_*` | 报告、图表和离线复算；具体读写路径看各入口 |
| `test_*.py`、`ssd_accounting_checks.py` | 规则、因果性、指标与存储记账回归 |
| `results/` | 按研究保存的报告、输入、结果、复现脚本和审计 |

核心 Python 依赖见 [requirements.txt](requirements.txt)，测试依赖见 [requirements-dev.txt](requirements-dev.txt)：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

仅运行核心可改为安装 `requirements.txt`。完整报告重建还可能需要 ReportLab、pandoc、XeLaTeX 和中文字体，依各生成器而定；部分图稿使用原机器的字体或绝对路径。现有 PDF/PNG 可直接阅读，不能据此宣称在任意机器上可一键重建全部报告。跨环境复现须核对冻结输入与事件结果，不能只比较随机种子或图片二进制哈希。

## 检查与复现

以下检查覆盖纯规则与指标核算，不启动实验 sweep：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  test_layer_budget_controller.py test_path_deadline_estimator.py \
  test_coflow_client_policy.py test_coflow_disk_policy.py \
  test_coflow_capacity_analysis.py test_shared_ssu_state.py \
  test_shared_path_report.py test_coflow_report.py
```

完整根目录回归还包含小规模仿真、适配器与发布资料检查，需保留对应输入/图片等 fixture；`ssd_accounting_checks.py` 应显式列入。运行范围可限定为根目录，避免递归收集 `results/` 中的测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  test_*.py ssd_accounting_checks.py
```

正式重跑以各实验目录内的 plan、manifest、command 和源码快照为准，使用新输出目录，勿覆盖已审计结果。构造新输入与读取冻结 manifest 是不同操作；修改统计窗口也不能自动保证所有卡仍 active。

## 整理与历史恢复

[2026-09-11 整理清单与恢复说明](docs/cleanup_20260911/README.md) 记录本轮的外部归档、文件核验及发布范围。纯开发扫描和部分旧 pilot 移出发布工作树；正式图、原始数据、负对照与回归测试保留。已入库的历史内容可从提交 `e56b1f95bca63b0f500fb114d056611de310e567` 恢复；未入库资料以清理清单中的项目外归档为准。旧 coflow/shared-path 报告若要完整重建，须先恢复其开发目录，不能把“可直接阅读报告”当作“精简树可完整重跑”。

早期欠载研究的 [analysis.json.gz](results/baseline_32npu6ssu_underload/analysis.json.gz) 是原 `analysis.json` 的无损压缩，数值与解压原字节保留；相关读取入口兼容旧明文文件。旧 V/B 的 [CURRENT_PROJECT_MANIFEST.md](CURRENT_PROJECT_MANIFEST.md) 与 [SHA 清单](CURRENT_PROJECT_SHA256SUMS) 仅覆盖那次最小实验，不是扩展后全仓的当前 manifest。本地缓存、环境、凭据与代理笔记不属于发布内容。
