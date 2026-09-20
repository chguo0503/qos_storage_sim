# 文件完整性与复现说明

本包整理截至2026-09-19已经保存的实验。指标来自原始结果、CSV或对原始样本的明确补算；没有重新运行仿真。仓库来源的副本版本固定为 `1c2bb30acdf8c721476b0294cb31079f0fab6fba`。

## 已保留的内容

| 实验 | 已归档材料 | 主入口 |
|---|---|---|
| full / semi 24画像 | 原始CDF、指标与逐请求样本、逐盘带宽图、冻结输入、脚本；semi另有32卡时序和等待解释 | `repository_source/results/diverse_data_ssu3_l3_20260916/` |
| 欠载随机输入near35：平均欠载、允许局部过载 | 六次完整运行的输入、结果、校验、全套代码、总体/分类表及CDF/带宽图 | `03_continuous_underload/near35/` |
| 旧AB128K/32K | Random/Ordered四张时序、四张逐卡层带宽、四张总带宽图、CDF及样本、指标、输入与命令、代码 | `repository_source/results/baseline_ab128_32_ratio12_20260912/` |
| 五组XY及E1 | 恢复的原始配置/结果/trace、100份校验一致的源码、理论表和最终版五组CDF | `03_continuous_underload/formula_ab/` |
| 精确200K/2048＋20K/1024 | 32卡4盘随机混排四组Baseline短窗筛选源码/冻结输入/结果；8卡E1双策略源码/trace/汇总原图 | `05_exact_ab_200K2048_20K1024/` |
| 局部过载补充AB：sensitivity20k_076 | 完整原代码/输入/双策略结果/receipts、逐请求/层CSV、三张原始配对图及校验 | `02_partial_overload/sensitivity20k_076/` |
| mixed8逐8卡原图 | 原图、2ms物理带宽、576条输入/执行数据、CDF点、完整原代码 | `04_ab8_bandwidth_original/` |
| 从冻结输入补画 | full、semi、near35的配比、原顺序和3张详细画像混排图，NEW前缀 | `tables/derived/` |

near35为2026-09-16补跑的固定配比随机混排实验：32 NPU、3 SSU、24种画像、每卡42条，原实验目录`results/diverse_near35_20260916`。其每盘平均带宽需求约35 GiB/s，允许局部过载；warm 2–4秒Baseline的NPU利用率/SLO×1.5为99.25%/96.20%，流量分配为99.40%/100%。sensitivity20k_076按本次整理要求加入局部过载目录；原约束仍是普通需求全程欠载、交界预取突发豁免，详见该实验说明。

总体表、类别表和五组AB汇总在 `tables/`。`index.html` 是精选原图浏览页；`file_inventory.csv` 列出全部归档文件、字节数和SHA256。`repository_files_index.csv` 保留仓库每个文件的固定版本链接，并标明是否已保存本地副本。

## 仅保留在线路径的原始大文件

以下七份原始压缩事件结果没有取得可验证字节，故未放入离线包；没有用空文件或其它运行代替。对应汇总、逐请求CDF样本、配置、输入、代码和已生成图片均已保留。

在仓库 `results/diverse_data_ssu3_l3_20260916/` 下：

- `runs/full_once_seed7_remote/result.json.gz`
- `runs/full_once_seed19_remote/result.json.gz`
- `runs/full_once_seed43_local/result.json.gz`

在仓库 `results/baseline_ab128_32_ratio12_20260912/` 下：

- `validation20s/runs/ssu3_random_k1_sync_seed7/baseline/result.json.gz`
- `validation20s/runs/ssu3_ordered_k1_sync_seed7/baseline/result.json.gz`
- `once_per_layer_ssu3_seed7/runs/random/once/result.json.gz`
- `once_per_layer_ssu3_seed7/runs/ordered/once/result.json.gz`

精确可点击路径见 [repository_files_index.csv](repository_files_index.csv)。这些缺失不影响已经归档表格和图片的核对，但离线重新分析完整事件轨迹时需要先取得相应原文件，或按冻结输入重新仿真。

## 五组AB原ZIP的恢复

原 `xy_underload_data_code.zip` 缺少ZIP中央目录。恢复过程保留原数据内容：469个原始文件全部通过CRC和解压长度检查，72份压缩trace逐一通过原记录SHA256核验。缺失源码只从其它原始包按完全相同SHA256补回，共补56份；最终100份已恢复源码均与原记录一致。

`run_experiment.py` 及关键依赖已恢复，导入检查通过；没有实际重跑。原README提及的根目录 `verify_and_export.py`、`summarize_theory.py`、`theory_summary.json` 尚未找到，另有13份附加实验脚本缺失。因此“主实验入口已补齐”和“原完整报表批处理尚不完整”需区别理解。精确清单见 `03_continuous_underload/formula_ab/hash_matched_source_recovery.json` 与 `03_continuous_underload/formula_ab/original_recovered/RECOVERY_NOTICE.md`。

独立CDF绘图包完整，位于 `03_continuous_underload/formula_ab/` 下的 `ttft_slo_cdf_bundle/` 及 `xy_ttft_cdf_data_code/`，可直接从已存逐请求数据重绘。E1原trace保留了层读取起止与到齐时间，但没有完整逐IO物理服务区间；未用“读取量/层耗时”假冒真实逐时刻盘供给图。

## 使用代码

- 所有旧实验的具体参数优先采用对应 `command.json`、配置JSON和冻结manifest，不以简化输入生成脚本代替原顺序。
- full/semi入口为 `repository_source/results/diverse_data_ssu3_l3_20260916/run_trial.py`、`run_once_control.py`；CDF为同目录 `render_threeway_ttft_cdf.py`。
- 旧AB入口为 `repository_source/results/baseline_ab128_32_ratio12_20260912/experiment.py`，其余Once与绘图入口在同目录。
- near35从子目录运行 `report_near35.py` 即可重绘已存结果；输入/仿真脚本在其 `source/results/diverse_near35_20260916/`。
- 固定配额随机混排图由 `tables/derived/plot_other_saved_manifests.py` 从full、semi、near35的冻结输入生成，默认路径按本资料包定位。
- sensitivity20k_076的代码、冻结输入、双策略原始结果和绘图入口见 `02_partial_overload/sensitivity20k_076/README.md`。

请保持原有结果目录，使用新的输出目录或先保存原目录副本后再重跑；本次整理没有覆盖任何原仿真结果。
