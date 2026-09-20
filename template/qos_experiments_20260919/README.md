# QoS 实验完整资料包

解压后打开 [index.html](index.html) 浏览图片；总体、类别、带宽指标及输入配比见 [实验总索引](qos_load_regime_experiments_index.md)。

本版以此前完整总包为基础，只调整以下两处，其余实验结果、代码和图片继续保留：

- 用 **2026-09-16 near35** 替换持续欠载的旧10画像随机输入实验。near35为32 NPU、3 SSU、24画像、每卡42条，固定配比随机混排；负载性质是**平均欠载，允许局部过载**。Baseline的NPU利用率/SLO×1.5为 **99.25% / 96.20%**，流量分配为 **99.40% / 100%**。
- 将 **2026-09-14 sensitivity20k_076** 加入“局部过载”部分。保留原配置、原始数据、三张原图、整体/A/B类别指标和代码；其原约束为普通需求欠载、长短请求交界预取豁免，具体口径见独立说明。

| 分类 | 内容 | 目录或入口 |
|---|---|---|
| 持续过载 | full 24画像 | `01_continuous_overload/full24/` |
| 局部过载 | semi 24画像 | `02_partial_overload/semi24/` |
| 局部过载补充AB | sensitivity20k_076 | [说明与原图](02_partial_overload/sensitivity20k_076/README.md) |
| 旧AB | A128K/NQL256、B32K/NQL4096的Random与Ordered；包括71.01%/71.43% → 83.42%/74.61% | `02_partial_overload/ab128_32/` 和 `repository_source/results/baseline_ab128_32_ratio12_20260912/` |
| 欠载随机输入 | near35：每盘平均需求约35 GiB/s | [原报告](03_continuous_underload/near35/near35_random_results/near35_report.md) |
| 持续欠载AB | 公式推导的五组XY及相关E1材料 | `03_continuous_underload/formula_ab/` |
| 其他历史结果 | mixed8逐卡带宽、原数据与代码 | `04_ab8_bandwidth_original/` |
| 其他历史结果 | 精确200K/2048＋20K/1024的E1和随机筛选 | `05_exact_ab_200K2048_20K1024/` |
| 公共源码与汇总 | 原仓库副本、原始路径索引、总体/类别/带宽表、画像混排图 | `repository_source/`、`tables/` |

near35归入本包的“欠载随机输入”位置，目录名沿用 `03_continuous_underload`；这不表示其全程每盘需求都低于容量。五组公式推导AB继续单独保留。

本次仅整理既有结果，未重新仿真或重画原图。此前从冻结输入补画的full/semi/near35画像混排图保留NEW前缀。此前要求删除的类别混排图片及对应生成分支继续排除。

请参阅 [文件完整性与复现说明](FILE_COMPLETENESS.md)：此前尚缺的七份原始大结果及五组AB附加脚本仍如实标注。[file_inventory.csv](file_inventory.csv)列出本包全部文件的大小和SHA256。
