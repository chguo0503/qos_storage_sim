# 五组 AB：最终核对结果与文件索引

本次只整理原实验与原始trace，未重新仿真。

五组均8NPU、1SSU=40GB/s十进制、8层，A:B输入数量比1:12，每卡独立打乱，seed7/19/43。下表U是2–4秒三种子简单平均，SLO是全请求合并（含冷启动）对应用户后来的CDF口径，两列cohort不同。SLO1本次从原始逐请求CSV补统计，沿用CDF脚本1e-10浮点容差。

|组|策略|整机U%|A类U%|B类U%|全请求SLO1%|全请求SLO1.5%|全程盘利用率%|
|---|---|---:|---:|---:|---:|---:|---:|
|XY12_32|baseline|99.74|99.93|99.48|78.53|100.00|65.47|
|XY12_32|once|99.85|99.75|100.00|90.62|100.00|65.55|
|XY12_24|baseline|98.28|99.78|96.43|48.91|100.00|71.90|
|XY12_24|once|99.13|99.12|99.14|78.40|100.00|73.17|
|XY12_20|baseline|95.03|99.22|90.25|28.08|99.10|76.26|
|XY12_20|once|98.62|98.27|99.05|70.64|100.00|78.96|
|XY12_16|baseline|90.83|99.23|83.81|6.62|93.43|81.17|
|XY12_16|once|97.69|97.34|98.10|65.44|99.68|87.82|
|X16|baseline|92.56|99.34|85.19|9.56|95.89|76.18|
|X16|once|98.33|98.05|98.68|72.92|99.73|80.83|

盘利用率是native_summary原记录ssd_mean_utilization的三种子均值，统计全程含冷启动与排空，不是2–4秒窗口。每类SLO完整值、warm SLO pooled与seedmean均在CSV。类别U分母为该类请求的接纳至完成区间在warm窗中的card-time，不能把A/B类U直接按输入数量比平均。

第一、二组满足任意8卡欠载但不满足单个完整A读挡住完整B计算窗口的带宽联合区间；第三至五组均满足。详见原request_bandwidth_and_S.csv。所有五组NQL包含插值，不满足此前NQL必须512倍数的另一次约束；B<32K计算时间为外推。

## 可拼接的汇总CSV

- five_ab_tables/five_ab_summary.csv：5行双策略总指标（warm U、全请求SLO1/1.5、warm SLO1.5）。
- five_ab_tables/five_ab_results_long.csv：30行，5组×2策略×All/A/B；另有名义需求、盘利用率及来源字段。
- five_ab_tables/E1_fixed_reference.csv：E1固定4A+4B对照，不能混成随机输入。

## 原始输入、结果与图索引

相对于source_five_ab：

|内容|路径|
|---|---|
|五组原输入|xy_underload_data_code_recovered/configs/{ID}_random_s1_seed{7,19,43}.json|
|每策略原trace|xy_underload_data_code_recovered/results/{ID}_random_s1_seed{seed}/{baseline,once}/native_summary.json.gz|
|每策略请求清单与画像|同目录requests.json、config.json|
|每策略精确统计与审计|同目录analysis.json|
|原运行脚本|xy_underload_data_code_recovered/run_experiment.py、run_batch.py、audit_results.py|
|所有seed原结果表|xy_underload_data_code_recovered/outputs/all_runs.csv、paired_summary.csv|
|最终全部请求SLO横轴CDF|ttft_slo_all_requests_images/ttft_slo_all_01_XY12_32.png 至 ttft_slo_all_05_X16.png|
|原SLO横轴All/A/B三面板CDF|ttft_slo_cdf_bundle/figures/ttft_slo_cdf_01_XY12_32.png 至 ttft_slo_cdf_05_X16.png|
|SLO CDF独立绘图代码|ttft_slo_cdf_bundle/build_slo_cdf.py|
|SLO CDF原逐请求数据|ttft_slo_cdf_bundle/cdf_requests.csv.gz|
|SLO CDF原统计|ttft_slo_cdf_bundle/slo_cdf_statistics.csv|
|原ms横轴CDF（早期版）|xy_ttft_cdf_figures/figures/ttft_cdf_*.png|
|x/y/S和名义带宽完整精度|xy_ttft_cdf_data_code/request_bandwidth_and_S.csv|
|整机及B类别U对比图|xy_underload_figures/outputs/random_x_utilization.png、random_y_utilization.png|
|原理论实验报告|xy_underload_data_code_recovered/outputs/xy_underload_experiment_report.md|

## E1：200K/NQL2048 + 20K/NQL1024 的确定已核验对照

这是固定NPU0–3持续A、NPU4–7持续B，1SSU，seed7，8层。与用户提及随机200K/20K图是不是同一个版本尚未证实；不要冒充随机实验。

|指标|Baseline|Once|
|---|---:|---:|
|2–4s整机U|83.50%|96.92%|
|2–4s A类U|100.00%|100.00%|
|2–4s B类U|67.00%|93.85%|
|2–4s接纳请求SLO1.5|22.22%|100.00%|
|2–4s A类SLO1.5|100.00%|100.00%|
|2–4s B类SLO1.5|12.50%|100.00%|
|全请求SLO1.5|39.83%|100.00%|
|全请求 A类SLO1.5|100.00%|100.00%|
|全请求 B类SLO1.5|34.86%|100.00%|

- 配置：xy_underload_data_code_recovered/configs/E1_fixed_s1_seed7.json。
- 原始数据：xy_underload_data_code_recovered/results/E1_fixed_s1_seed7/{baseline,once}/ 下的native_summary.json.gz、requests.json、analysis.json。
- 原汇总图：xy_underload_figures/outputs/fixed_concurrent_utilization.png 包含E1。
- 没有在此包中找到E1专门的逐NPU带宽供给图。native_summary只有层读取起止/到齐与disk_stats全程聚合，没有逐IO服务区间，无法不重仿真还原原来的真实2ms供给曲线。

## 原包恢复状态

xy_underload_data_code.zip缺中央目录，文件在source/fifo_20k_sensitivity.py的完整entry结束后截停。保留原件；从local headers恢复469个entry，每个CRC/长度正确；72份native_summary均与包内原verification.json的SHA256完全相同并成功解gzip。

另从其它已保存原包找回56份hash与本实验source_sha256完全相同的源码；记录见hash_matched_source_recovery.json。当前共100份记录源码已通过hash核验，仍缺13份附加实验/报表源码。主实验run_experiment.py及其导入依赖（包括fixed_total_compat.py）已补齐，模块导入检查成功；source/data原始画像存在。本次未运行仿真，不把导入检查等同完整重跑验证。文档提及的verify_and_export.py、summarize_theory.py与theory_summary.json仍不在恢复内容中，原README完整矩阵报表重建命令不完整。CDF独立包完好，可不依赖核心模拟器直接重画CDF。

缺失13份记录源码：

- plot_abcd_8npu.py
- plot_abcd_layer_demand_supply.py
- plot_abcd_total_demand.py
- proxy_4npu_agent.py
- report_4npu_exact_data.py
- report_ratio8_4npu.py
- report_ratio_sweep_4npu.py
- report_ratio_sweep_r40_4npu.py
- run_4npu_exact_data.py
- run_4npu_ratio8.py
- run_abcd_8npu.py
- run_abcd_bd_heavy_8npu.py
- run_abcd_fixed_8npu.py

有效可解压恢复包：xy_underload_data_code_recovered.zip。审计：xy_underload_data_code_recovery_audit.json、hash_matched_source_recovery.json。
