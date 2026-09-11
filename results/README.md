# 实验与资料导航

先看最新的 **[32 NPU / 6 SSU 持续混合请求长测](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/report.md)**，以及 **[每卡带宽诉求、SSD 真实服务、链路交付与 stall](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/bandwidth_delivery/README.md)**。前者主窗是 `[2,60)` 秒，后者取同一主实验的 `[3.2,4.0)` 秒作物理服务解释；两者的平均值不能互换。

各研究依次回答不同问题，不是一张可直接横向排名的性能榜。输入画像、长短比例、每卡绑定、到达方式、排序、随机种子、盘数与窗口须匹配后才能解释策略差值。未达目标、容量不合格及尾部耗尽的案例也作为证据保留。

## 最新长期实验与真实交付证据

| 内容 | 入口 |
|---|---|
| 原始画像、每卡反复运行长短请求，Random/Ordered × Baseline/Once | [60 秒报告](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/report.md) |
| 完整长窗、细节窗、名义需求与第51秒交接 | [独立图及口径](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/figures/separate/README.md) |
| 32 张逐卡带宽图、累计截止缺口、stall期间高带宽反例 | [带宽交付报告与审计入口](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/bandwidth_delivery/README.md) |
| 主结果、输入与每种画像的独立核验 | [审计目录](baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/audit_existing/) |

最新长期输入使用原始 `data` 画像并保留真实辅助桥接计算，主四格为 seed 7。整个长窗内每卡混合，不意味着每个2秒子窗内每卡都包含两类；报告同时列出这项更强条件的实际结果。

## 32 卡欠载、输入顺序与分卡研究

| 研究 | 要回答的问题 | 报告/图 |
|---|---|---|
| 最初的欠载排序构造 | 同一人口只改顺序，Baseline是否下降；含外推短画像及插值长画像 | [原报告](baseline_32npu6ssu_underload/report.md) · [全部案例](baseline_32npu6ssu_underload/all_results.md) · [原独立图](baseline_32npu6ssu_underload/figures/separate/README.md) |
| 原构造五种子配对 | seed7存在性结果是否能在其余种子保持同样降幅 | [Baseline/Once配对](baseline_32npu6ssu_underload/paired_once_5seeds/comparison.md) |
| 固定长卡与短卡 | 11长/21短或6长/26短；短卡Random/Round-robin、两策略、五种子 | [角色分离结果](baseline_32npu6ssu_underload/role_separated_5seeds/comparison.md) · [输入分配与图](baseline_32npu6ssu_underload/role_separated_5seeds/figures/separate/README.md) |
| 原表随机输入 | 从84画像或容量充分条件子集抽样，与选定画像后洗牌区分 | [随机data研究](baseline_32npu6ssu_underload/raw_data_random_5seeds/comparison.md) |
| 原始四画像替换 | 原始画像替代早期构造画像，保留成功与未复现结果 | [四画像研究](baseline_32npu6ssu_underload/raw_quartet_replacement/comparison.md) |
| 固定与每卡混合的原始画像对照 | 同一全局人口重新分卡；两个混合顺序还保持逐卡人口 | [五种子最终表](baseline_32npu6ssu_underload/raw_role_followup_20260909/final_comparison.md) · [完整方法与探索](baseline_32npu6ssu_underload/raw_role_followup_20260909/mixed_research_report.md) |
| 改变观察窗/延长有限队列 | 短窗低利用率是否随时间恢复，区分尾部空闲与相位变化 | [窗口敏感性](baseline_32npu6ssu_underload/raw_role_followup_20260909/window_sensitivity/report.md) |
| 短请求总输入超过10K的核查 | 明确原始画像与旧1K外推的区别，补局部物理FIFO证据 | [验证报告](baseline_32npu6ssu_underload/short_ge10k_validation_20260911/report.md) · [独立图](baseline_32npu6ssu_underload/short_ge10k_validation_20260911/figures/separate/README.md) |
| 四组暖窗汇总 | 统一定义下阅读构造、角色分离、原表随机与四画像结果 | [暖窗汇总](baseline_32npu6ssu_underload/warm_comparison_tables.md) |

早期主研究的逐事件审计为 [analysis.json.gz](baseline_32npu6ssu_underload/analysis.json.gz)，完整窗口表为 [summary.csv](baseline_32npu6ssu_underload/summary.csv)。JSON仅改变压缩容器；解压内容与原始明文相同。子研究各自同名的 `analysis.json` 不因此自动迁移，按各报告链接读取。

## 调度策略与历史实验

| 研究 | 入口 |
|---|---|
| 32卡/6或8盘混合输入的大规模调查 | [报告](baseline_npu32_investigation/docs/report.md) · [PDF](baseline_npu32_investigation/report.pdf) · [演示与输入说明](baseline_npu32_investigation/presentation/README.md) |
| 全局 coflow，5ms状态采集 | [报告](coflow_global_5ms_experiments/docs/coflow_global_5ms_report.md) · [PDF](coflow_global_5ms_experiments/coflow_global_5ms_report.pdf) · [演示](coflow_global_5ms_experiments/presentation/) |
| 共享 Path 状态与选路 | [报告](shared_path_5ms_experiments/docs/shared_path_5ms_report.md) · [PDF](shared_path_5ms_experiments/shared_path_5ms_report.pdf) |
| 多SSU的等待预测与分卡 | [实验报告](multi_ssu_stall_experiments/docs/multi_ssu_experiment_report.md) · [数学说明](multi_ssu_stall_experiments/docs/multi_ssu_math.md) |
| 单组等待预测与预算策略 | [实验报告](stall_prediction_experiments/docs/stall_policy_experiment_report.md) · [数学说明](stall_prediction_experiments/docs/stall_prediction_math.md) |
| 4NPU/1SSU低利用率 | [原报告](baseline_4npu_ssu1_low_utilization/docs/report.md) · [教程PDF](baseline_4npu_ssu1_low_utilization/baseline_path0_low_utilization_tutorial_v3_2.pdf) |
| 两份Baseline高/低利用率报告 | [报告与逐请求证据](../baseline_two_reports/README.md) · [输入/容量复核](baseline_two_reports_analysis/README.md) |
| 32NPU/5SSU、固定28:4 V/B | [结果](../VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md) · [实现与硬件边界](../VB_POLICY_IMPLEMENTATION_AND_HARDWARE_FEASIBILITY.md) · [原始结果目录](vb_fixed_split_npu32_ssu5_28high_4low_4s/) |

## 教学资料

- [Prefill miss、开环/闭环与NPU利用率：修订PDF](prefill_miss_open_closed_guide/prefill_miss_open_closed_explained_v2.pdf) · [文字稿](prefill_miss_open_closed_guide/prefill_miss_open_closed_explained_v2.md)。区分外部请求发送、NPU接纳与内部计算/预取反馈。
- [为什么自然错开有时有效、有时仍反复等待](path0_staggering_story/why_staggering_sometimes_works.md) · [PDF](path0_staggering_story/why_staggering_sometimes_works.pdf)。这是教学小模型，应按其假设阅读。
- [Baseline/NewOnce 原构造入门PDF](baseline_32npu6ssu_underload/tutorial/baseline_newonce_beginner_guide.pdf)。对应早期构造画像与2–4秒研究，不替代最新原始画像长测。
- [4卡Path0截止与错开说明](baseline_4npu_ssu1_low_utilization/docs/path0_deadline_and_staggering_notes_v3.md)。

## 指标与复现边界

**接纳后 SLO 不等于端到端 TTFT。** 许多研究判断 `completion−admission <= 1.5×理想计算时间`；这不包含接纳前的排队。到达计时须使用 `arrival`，阈值和请求人口也须与报告一致。按窗口接纳选择的请求应跟踪至完成，而不同策略选中的请求集合可能不同。

设备U、请求等权U、画像自身U和有限批次的完整运行U各有分母；不得混用。`D/C`名义欠载与“实际SSD/链路服务不超过物理能力”也不是同一断言，前者不证明截止必然满足。跨请求L0预取、桥接请求的真实计算及各次统计边界均应保留在解释中。

主源码与回归测试仍保留原位置，安装和分层检查见[仓库README](../README.md)。报告生成器不都只依赖NumPy/Matplotlib：ReportLab、pandoc/XeLaTeX和字体按具体脚本准备；历史绝对路径需按环境适配。正式复现使用对应冻结manifest与命令，并写到新目录。

[本轮清理与恢复说明](../docs/cleanup_20260911/README.md) 列出项目外归档和保留范围。部分coflow/shared-path纯开发扫描与旧pilot移出发布树，完整重建旧报告须先恢复这些目录。已跟踪历史可从提交 `e56b1f95bca63b0f500fb114d056611de310e567` 恢复，其他文件依外部归档清单恢复；不能仅凭此索引推定所有历史文件都存在Git中。正式图、原始数据和解释结论所需的负对照保留。
