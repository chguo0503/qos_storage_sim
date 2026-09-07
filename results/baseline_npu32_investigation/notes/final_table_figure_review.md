# 最终表格与图04复核

已重新运行 `analysis/plot_arrival_counterfactual.py`，四格、输入保留字段、逐层时间线与独立汇总交叉检查全部通过。四格仍取本机 Python 3.10 的原生 Baseline / Deadline + Fluid 结果。[图04审计](../analysis/04_arrival_counterfactual_audit.json)

已打开重生成的 [PNG](../analysis/04_arrival_counterfactual_home_ssu.png) 检查：两个 `B (deadline + fluid)` panel 标题完整、无挤压或遮挡，图例、坐标和说明文字均未裁切。无需改布局或字号；未修改图脚本、根目录源码或主报告。

`docs/report.md` 的主实验12行表，全部72个窗口利用率与 `notes/mixed_profile_audit/windows.csv` 按报告保留两位小数后逐项一致。六盘4行表的8个窗口数值、4个完整批次时间、2个最热盘负载和2个短计算时间份额，分别与独立 `mixed_six_ssu/analysis/{windows,status,inputs}.csv` 一致。合计 **88/88 个数值核对通过**；全部80个窗口均32卡持续active且idle为0。

这里的一致指报告展示精度一致，不是将已舍入的数值宣称为底层浮点逐位相等。逐项结果、报告和表格内容哈希、CSV/图/脚本哈希保存在 [final_table_figure_review.json](final_table_figure_review.json)。
