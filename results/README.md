# 实验结果导航

这里汇总各次实验。数字、画像、分卡、顺序、盘数、placement 和窗口必须成套阅读，不能把不同输入的利用率直接当成策略排名。

| 研究 | 内容 | 入口 |
|---|---|---|
| 公式 A/B 扩至32卡：`formula_ab_32npu_20260921` | 五组原画像的6盘对照、第四组4盘容量控制、另一个3盘候选；ASU/OD/Once、三个种子，区分逐盘欠载与过载、warm与全输入 | [结果与CDF](formula_ab_32npu_20260921/README.md) · [数学说明](formula_ab_32npu_20260921/math_notes.md) · [独立审计](formula_ab_32npu_20260921/AUDIT.md) |
| 保留实验模板与新增 OD 对照 | 按负载浏览原图、冻结资料与原输入 ASU/OD/Once 对照；模板缺失项单独标注 | [图片索引](../template/qos_experiments_20260919/index.html) · [模板说明](../template/qos_experiments_20260919/README.md) · [三策略对照](../template/qos_experiments_20260919/od_baseline_comparison/README.md) |
| 周六：`baseline_ab128_32_ratio12_20260912` | A总128K/miss256，B总32K/miss4096，每卡A:B=1:2；Random/Ordered及盘数、策略对照；当前从模板读取保留资料 | [模板中保留的报告](../template/qos_experiments_20260919/repository_source/results/baseline_ab128_32_ratio12_20260912/report.md) · [历史公式核对](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_ab128_32_ratio12_20260912/formula_review/README.md) |
| 历史：`baseline_random_near_capacity_20260914` | 各卡独立Random；按V/C、纯计算权重和等待阈值寻找低利用率；原始data与外推分列；当前目录已删除，链接固定到已提交历史 | [历史报告](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/report.md) · [历史结果表](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/comparison.md) · [历史数学说明](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/report_core.md) |
| 多样 data：`diverse_data_ssu3_l3_20260916` | 24 种画像、两组配比；共享 Path Baseline、Once 与候选池策略；该目录保存的历史结果使用条带 placement | [实验目录](diverse_data_ssu3_l3_20260916/) |
| ASU / OD：`od_baseline_diverse_ssu3_20260918` | 同一批多样 data 请求，使用 Ring hash 重跑；新增每盘每卡独占 Path、CIR 均分的 OD Baseline；两组配比、三个种子、六种策略 | [报告与复现](od_baseline_diverse_ssu3_20260918/README.md) · [策略定义](../docs/baseline_strategies.md) |
| 文档输入复核：`continuous_underload_asu_od_20260918` | 10 种原始画像，每卡20条，共640条；固定请求身份与Ring hash，比较Random/Ordered的ASU与OD；如实记录超限区间 | [报告与图表](continuous_underload_asu_od_20260918/README.md) · [文档核对](continuous_underload_asu_od_20260918/doc_review.md) |
| 三类负载对比：`load_regimes_asu_od_comparison_20260918` | 持续过载、持续欠载、局部欠载；复用18次Random结果，对比ASU/OD的CDF、NPU利用率及总带宽/逐盘带宽 | [对比总览与独立PNG](load_regimes_asu_od_comparison_20260918/README.md) |
| OD / 原始Once：`od_vs_once_three_loads_20260918` | 三类输入的利用率与SLO对照；补齐持续欠载Once三个种子，核对full/semi收益、LL代价与完整人口 | [对照与输入](od_vs_once_three_loads_20260918/README.md) · [机制与代价](od_vs_once_three_loads_20260918/mechanism_notes.md) |
| 每卡混合、持续欠载：`od_mixed_underload_search_20260919` | 32 NPU / 3 SSU，原始data与合成输入分列；逐盘逐事件检查，OD/Once同输入对照；找到warm差距，但后期恢复 | [结果报告](od_mixed_underload_search_20260919/README.md) · [输入与数学解释](od_mixed_underload_search_20260919/experiment_design.md) · [全部筛选](od_mixed_underload_search_20260919/search_log.md) |
| OD 固定队深：`od_fixed_qdepth_full_20260919` | 原 full24 冻结输入；每盘8192槽、每卡每盘256槽，与不限队深OD逐请求／层对照；区分主机等待与盘内等待 | [报告、图表与复核](od_fixed_qdepth_full_20260919/README.md) |
| OD 长期机制：`od_underload_mechanisms_20260919` | 原始data候选与插值构造分列；50周期的间歇性强过载输入、Once对照、队深消融和提交种子检查；不属于持续欠载 | [研究报告](od_underload_mechanisms_20260919/README.md) · [数学解释](od_underload_mechanisms_20260919/math_explanation.md) · [正式结果与审计](od_underload_mechanisms_20260919/analysis/scheduled_abb_interp_unique_50/README.md) |

学习材料见[手稿与实验通俗教程](../docs/l1_l2_l3_beginner/README.md)。上述两个已删除的旧结果目录可在[提交1c2bb30的历史结果](https://github.com/chguo0503/qos_storage_sim/tree/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results)查看；模板保留范围见[完整性说明](../template/qos_experiments_20260919/FILE_COMPLETENESS.md)。更早的实验仍可从[整理前项目](https://github.com/chguo0503/qos_storage_sim/tree/38edfa31cb5b61da02f4d198e97dc09d18356ad6)查阅。

各报告分别标明主窗口、长期观察窗与完整输入范围，有限观察不等于无限时间极限；SLO是接纳后prefill完成代理。局部坏层不代表整机同样低，理想平均负载接近容量也不代表逐盘逐时欠载。
