# 实验结果导航

当前保留周六实验和本次Random研究。数字、画像、分卡、顺序、盘数和窗口必须成套阅读，不能把不同输入的利用率直接当成策略排名。

| 研究 | 内容 | 入口 |
|---|---|---|
| 周六：`baseline_ab128_32_ratio12_20260912` | A总128K/miss256，B总32K/miss4096，每卡A:B=1:2；Random/Ordered及盘数、策略对照；已有32卡带宽图、zoom和PDF | [报告](baseline_ab128_32_ratio12_20260912/report.md) · [公式核对](baseline_ab128_32_ratio12_20260912/formula_review/README.md) |
| 本次：`baseline_random_near_capacity_20260914` | 只运行各卡独立Random；按V/C、纯计算权重和等待阈值寻找长期80几；原始data与外推分列，同输入策略配对 | [报告](baseline_random_near_capacity_20260914/report.md) · [结果表](baseline_random_near_capacity_20260914/comparison.md) · [数学说明](baseline_random_near_capacity_20260914/report_core.md) |

学习材料见[手稿与实验通俗教程](../docs/l1_l2_l3_beginner/README.md)。旧实验由Git历史保留：[整理前项目](https://github.com/chguo0503/qos_storage_sim/tree/38edfa31cb5b61da02f4d198e97dc09d18356ad6)。

本次报告明确区分2秒主窗口与18秒观察窗，后者不等于无限时间极限；SLO是接纳后prefill完成代理。局部坏层不代表整机同样低，理想平均负载接近容量也不代表逐盘逐时欠载。
