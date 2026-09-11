# 项目整理记录：2026-09-11

本次整理保留模拟器、全部调度策略、实质回归测试、正式报告/图片、正式输入/完成运行、负对照与失败原因记录。没有新增性能实验，也没有按结果是否有利筛选数据。当前项目入口见[根README](../../README.md)和[实验目录](../../results/README.md)。

## 清理范围

| 处理 | 文件数 | 原大小 MiB |
|---|---:|---:|
| 删除可再生缓存、重复分析缓存与被替代预览 | 200 | 35.379 |
| 删除与展开目录相同的ZIP | 3 | 33.275 |
| 移出工作树、保留外部归档 | 753 | 670.436 |
| 用无损gzip替代原始大JSON | 1 | 255.900 |

共从原位置移除957个文件、清除718个空目录。压缩文件新增14.272 MiB，合计减少工作树约980.717 MiB（未扣除本次新增的小型说明文件）。其中历史开发资料移到项目外，不代表永久删除磁盘上的这部分内容；Git历史也没有被重写。

每个原路径、字节数、SHA256、处理原因及其是否曾在Git中跟踪，见[逐文件清单](manifest.json)。`.learnings`、本地工具配置与运行环境不上传。

## 保留了什么

- 最新总输入超过10K、32 NPU / 6 SSU的长期混合实验：五个完整长运行、全部输入、正式图、补充约束审计及未成功的机制对照。
- 最新[逐NPU带宽交付研究](../../results/baseline_32npu6ssu_underload/sustained_mixed_ge10k_20260911/bandwidth_delivery/README.md)：32张单卡图、4张说明/总览图及三种格式、真实SSD/链路trace、CSV和独立审计。
- 固定选卡、卡内混合、原始data随机抽样等全部正式结果，包括Baseline表现很好或Once没有改善的结果。
- 原coflow/shared-path正式矩阵、旧32卡研究的screen/formal/holdout/grid、跨环境和机制对照、旧4卡/1盘、多SSU实验。
- 完整`baseline_two_reports/`及冻结的原版本源码。只删除重复ZIP，没有用当前源码替换旧执行版本。
- 所有根目录`test_*.py`及`ssd_accounting_checks.py`。它们校验队列、存储记账、输入公平性和统计口径，未发现可以安全删除的冗余测试源码。
- 原取消/中断记录及审计所依赖的plan/status。取消的任务不算策略性能失败。

原始冻结结果和图的数值不改动。个别报告下载链接、读取脚本和导航文案随包装调整；历史PDF及其原交付审计保留，不把其中的历史文件SHA改写成新版本SHA。检查历史交付时应使用对应版本，当前路径变化记录在本清单中。

## 历史开发资料如何恢复

七个完整阶段被归档，范围不是按胜负挑选的个别运行：

- `results/coflow_global_5ms_experiments/data/exploratory_precanonical/`
- `results/shared_path_5ms_experiments/data/pilot_v1/`
- `results/_archive/baseline_4npu_ssu1_low_utilization_docs_20260906.uMi5Hm/`
- `results/baseline_npu32_investigation/mixed_sustained/inputs/`
- `results/baseline_npu32_investigation/mixed_varied_short/candidates/`
- `results/coflow_global_5ms_experiments/data/development/`
- `results/shared_path_5ms_experiments/data/development/`

完整本地归档位于项目同级目录 `qos_storage_sim_cleanup_20260911T003141Z/files/`，里面保持原相对路径，移走文件的SHA未改变。`inventory.before.json`和`cleanup.completed.json`分别记录清理前状态与完成清单。该本地归档不随本次提交上传。

已跟踪文件还保留在Git提交 [`e56b1f9`](https://github.com/chguo0503/qos_storage_sim/tree/e56b1f95bca63b0f500fb114d056611de310e567) 中。要重建旧coflow/shared完整报告，先在仓库根恢复两组开发扫描：

```bash
git restore --source=e56b1f95bca63b0f500fb114d056611de310e567 --worktree -- \
  results/coflow_global_5ms_experiments/data/development \
  results/shared_path_5ms_experiments/data/development
```

随后按原实验的构建脚本执行。浅克隆若缺少该提交，需要先获取历史。其他历史阶段可把上述路径换成清单内对应目录；未曾跟踪的草稿/运行日志需从本地归档取回。

旧开发扫描的所有选择指标和失败候选表仍保留在原正式Markdown/PDF及`data/report_audit.json`；coflow的`selection.json`、`canonical_grid/development_candidates.json`也留在原位。当前精简工作树可直接阅读正式报告，但**不能在未恢复开发扫描时声称可完整重算这两份旧报告的开发选择过程**。

## 大JSON的无损保存

原文件：`results/baseline_32npu6ssu_underload/analysis.json`，255.900 MiB。

当前文件：[analysis.json.gz](../../results/baseline_32npu6ssu_underload/analysis.json.gz)，14.272 MiB。

解压后字节SHA256仍为：`452d09f3115e5104a156b70431efd572b07b2e2d5ed3c1e0458b0e793ca05628`。没有删除字段、舍入数值或重算统计。四个分析/读取脚本已支持gzip，读取时优先gzip并兼容旧plain JSON；教程审计中的`analysis_sha256`继续对解压后的原字节计算。

如需恢复旧文件名供外部工具读取：

```bash
gzip -dc results/baseline_32npu6ssu_underload/analysis.json.gz > /tmp/qos_analysis.json
```

三个重复ZIP合计169个文件条目均与整理前展开文件的SHA逐项一致。当前通过展开目录分发；ZIP可按需重新生成，已加入忽略规则。

## 验证

清理前建立了逐文件SHA清单；发布前核对正式证据保留、压缩内容一致、文档入口、待提交文件体积和规则/指标回归。完成后的具体检查结果记录在[validation.json](validation.json)。此次不以重新生成旧图来替代来源核查，也没有运行完整实验扫描。
