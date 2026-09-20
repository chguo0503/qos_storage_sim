# 8 NPU 带宽图归属核对（原实验，未重跑）

`fifo_8npu_bandwidth_simple.png` 和 `once_8npu_bandwidth_simple.png` 已与原始 `mixed8_bandwidth_simple_images.zip` 内同名文件核对，SHA256 完全一致。**这两张图属于 mixed8：A≈200K/NQL1664，B≈20K/NQL1024，带窄幅扰动；不属于 A200K/NQL2048 的组。**

| 原实验 | 实际请求参数与配比 | FIFO NPU 利用率 | Once NPU 利用率 | FIFO TTFT SLO×1.5 | Once TTFT SLO×1.5 |
|---|---|---:|---:|---:|---:|
| mixed8 随机混排；本次找到的逐 8 卡简图 | 每卡 6A+66B；A 199.9873–200K、NQL1651–1664；B 20–20.0801K、NQL1024–1106 | 92.523977% | 97.155743% | 98.214286%（165/168） | 100%（170/170） |
| lower_fifo_followup 的 sensitivity20k_076 | 每卡 7A+42B；A200K、NQL2042–2054（中心2048）；B20K、NQL1126–1178（中心1152） | 94.6147% | 99.2567% | 98.4375%（126/128） | 100%（137/137） |
| XY fixed E1 | A200K/NQL2048，B20K/NQL1024，4A卡+4B卡固定绑定；非随机混排 | 83.50% | 96.92% | 参见 E1 原表 | 参见 E1 原表 |

mixed8 均为 8 NPU、1 SSU（40 GiB/s）、8层、batch=1、seed=7，统计窗口 [2,4)s。TTFT 是 admission 到完成的服务耗时，阈值为 1.5×8层纯计算；窗口内 admission 的请求跟踪到完成。全部输入 576 请求的 SLO 是 FIFO 550/576=95.486111%、Once 576/576=100%，不可与窗口口径混用。20K 的计算时间含原实验的长度外推。

## 目标归属更正

用户已明确所指是9月14日 `sensitivity20k_076`（94.6147%→99.2567%），对应上表第二行。本目录mixed8图片属于第一行，不作为该目标的原图。此前以近似带宽4.6/4.5判定候选的表述已撤回。

## mixed8 类别指标

| 策略 | 短请求 SL NPU 利用率 | 长请求 LL NPU 利用率 | 短请求窗口 SLO | 长请求窗口 SLO | SSU 实际平均带宽 | SSU 带宽利用率 |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 85.265191% | 99.289555% | 98%（147/150） | 100%（18/18） | 34.652844 GiB/s | 86.632111% |
| Once | 97.543475% | 96.869026% | 100%（149/149） | 100%（21/21） | 36.871341 GiB/s | 92.178353% |

常规 V/C 任意组合上界 38.916459 GiB/s < 40 GiB/s，但它不包括跨请求的下一请求 L0 预取截止时间。预取窗口仍会局部超供给，因此此组可以作为“常规带宽欠载但局部预取拥塞”的历史补充，不能仅据简图把它等同于后来的三场景 semi 基线。

## 图表与代码位置

以下主要材料位于本附录子目录：

- `mixed8_bandwidth_simple_images_extracted/`：FIFO/Once 各自的整机供给图、逐8卡带宽图、两策略整机对比；还含 short_first 诊断策略（不等于 Once）。
- `mixed8_bandwidth_plot_data_extracted/`：`plot_bandwidth.py`、README、2ms 原始逐卡/整机 CSV、`data/audit.json`（直接记载 `L200n1664_S20n1024_r11_random_b1_sync_jnarrow_seed7` 及结果 SHA256）。
- `mixed8_request_data_and_code_extracted/source/results/fifo_mixed_unique_20260914/formal_random_s1_once/comparison_summary.csv`：整体和长短类别 NPU 利用率、窗口 SLO。
- `mixed8_request_data_and_code_extracted/source/results/mixed8_complete_figures_20260914/data/`：`request_profiles.csv`（576行原输入）、`request_execution.csv`、`slo_1p5_summary_from_details.csv`、`ttft_slo15_cdf_points.csv` 等。
- `mixed8_request_data_and_code_extracted/source/results/mixed8_physical_bandwidth_20260914/data/physical_bandwidth_summary.csv`：物理供给均值、峰值、容量核验；本表的带宽数字以此为准。
- `mixed8_request_data_and_code_extracted/source/`：`run_fifo_mixed_unique.py`、`plot_mixed8_physical.py`、`plot_mixed8_slo15.py`、`export_mixed8_request_details.py` 与原引擎依赖。

简图橙线为含 L0 的“下一层读取量/当前计算窗口”需求，在统一2ms桶平均；蓝线是实际字节量/2ms。整机图蓝线用 SSD 服务，逐卡图蓝线用 NPU 实际接收，存在传输时差。它们的比值不等于 NPU 利用率。

两张逐卡图 SHA256：

- FIFO：`cb4f07285bad72d7039ee89294a300e01656f0005e1bd163a1ba180837d47c14`
- Once：`c2347c9b0d6c4d5374b9071782647298695331dd0f47e7f92ebeeb333ccbb3b8`

