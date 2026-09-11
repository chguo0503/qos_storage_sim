# 两份 Baseline 分析报告及证据

请保留本目录结构，解压后打开下面两份 Markdown，即可显示配套图片。

- [原 Baseline 高利用率的含义与反证](01_baseline_high_utilization_rebuttal.md)：原始 704 条混排输入、全部 32 卡顺序、1 秒全卡时序、分类与画像利用率、8 层请求放大图。
- [低利用率输入的精确构造](02_baseline_low_utilization_inputs.md)：32 卡、4/5/6 SSU 三个案例，全部输入数量、分卡、类别、1 秒时序和固定长卡的短族对照。

两份正文分别讨论历史原输入与本次对话中此前已完成的新输入实验。本次制图和写作只读取已保存事件，没有新跑仿真。

## 目录与单位

| 路径 | 内容 |
|---|---|
| `images/` | 14 张图，各含 PNG 和可缩放 SVG；正文引用 PNG |
| `data/high/` | 原混排案例的完整输入、事件和重新聚合结果 |
| `data/low4/`、`low5/`、`low6/` | 三个低利用率案例的完整输入、事件和重新聚合结果 |
| `sources/` | 原始结果、输入元数据及来源源码；固定长卡对照的原指标 |
| `audits/` | 独立重算与最终交付校验结果 |
| `code/` | 从事件重算、绘图和生成正文的 Python 代码 |

每个案例的 `all_inputs_and_times.csv.gz` 包含所有输入，不只包括 warm 窗口内的请求。gzip 解压后为 UTF-8 CSV，可按 `npu_id,input_order` 阅读。`request_id` 唯一标识请求，`input_order` 从 0 起，文件保留原有输入顺序。

- `arrival_ms`：外部输入到达时间；低利用率三例全部为 0。
- `admission_ms`：NPU 开始处理该请求的时间。
- `first_layer_io_start_ms`：第 0 层预取发出时间，可能早于接纳。
- `completion_ms`：请求完成时间。
- 低案例空观察时刻表示捕获尚未执行到，不能按 0 处理；完整输入仍保留。
- 时间字段以 ms 为单位；`layer_kv_gib` 以 GiB 为单位，1 GiB=1024 MiB；`seq_len_k` 的 1K=1024 token。
- `utilization` 及其他利用率字段为 0–1；正文和图乘 100 显示百分比。
- `profile_id` 是画像/画像族；`category` 是模拟器 SS/SL/LS/LL 分类，二者不能混用。

`warm_requests.csv` 给出逐请求窗口内的计算、占用与等待时长；`warm_layers.csv.gz` 给出与窗口重叠请求的逐层原始时刻。`normalized.json.gz` 同时保留完整输入和已捕获的请求层事件，是绘图输入。窗口边界在 `statistics.json` 中明确记录。

## 复现统计和图片

需要 Python 3、NumPy、Matplotlib。在本目录运行，例如：

```bash
python3 code/render_evidence.py data/high/normalized.json.gz --output regenerated --key high --title "Original mixed input"
python3 code/render_evidence.py data/low4/normalized.json.gz --output regenerated --key low4 --title "Fixed 6 long + 26 short"
python3 code/render_evidence.py data/low5/normalized.json.gz --output regenerated --key low5 --title "Fixed 9 long + 23 short"
python3 code/render_evidence.py data/low6/normalized.json.gz --output regenerated --key low6 --title "Fixed 11 long + 21 short"
```

以上命令写入 `regenerated/`，不重跑仿真。以当前已复核的 `data/` 和 `sources/` 重建正文、执行交付校验：

```bash
python3 code/build_documents.py
python3 code/verify_delivery.py
```
