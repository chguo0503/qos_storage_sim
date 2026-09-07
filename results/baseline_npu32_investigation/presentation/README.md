# 混合输入下的 NPU 利用率与 QoS 策略

[单页 PPTX（当前交付）](mixed_input_summary.pptx) · [单页 PDF](mixed_input_summary.pdf) · [单页预览](mixed_input_summary.png) · [输入明细](input_details.md)

当前交付为一页：用具体输入描述替换 E72/V38 等内部代号，表格明确区分 6 盘和 8 盘，下方列每卡条数。6 盘仅测 Baseline 与 New Once，其余单元格用“—”表示未测。六盘、八盘配额不同，不构成仅改变盘数的对照。

单页源文件：`source/build_one_page.py`；运行 `python -B results/baseline_npu32_investigation/presentation/source/build_one_page.py` 生成 PPTX，再按下方同样的 LibreOffice / pdftoppm 命令导出（文件名改为 `mixed_input_summary`）。

[六页详细版本 PPTX](mixed_input_report.pptx) · [六页 PDF](mixed_input_report.pdf)

参照 `coflow_global_5ms_experiments/presentation` 的白底、灰色表头、策略对比风格制作，共六页。全部文字、表格、时间占比条和输入顺序色格均可在 PowerPoint 中编辑。

|页码|内容|
|---|---|
|1|32 NPU / 8 SSU，七种策略在两组输入上的利用率与完整输入 SLO|
|2|Baseline 利用率 96.23% 的原 data 参数混合输入|
|3|Baseline 利用率 80.29% 的 1K 外推参数混合输入|
|4|六盘的两组输入、具体配额及利用率|
|5|32 张 NPU 各自完整的 92 条输入顺序，全部 2,944 条可见|
|6|96% 整机平均下各类请求的等待，以及 New Once 的改善和代价|

## 利用率与输入对应

以下每个数量均为**每张 NPU 的请求条数**；`32K/128` 表示序列长度 32×1024 token、NQL 128 token。

|输入|SSU|每卡输入|Baseline 利用率|New Once 利用率|
|---|---:|---|---:|---:|
|V38|8|32K/128 ×28；48K/256 ×28；64K/512 ×28；192K/1024 ×2；192K/2048 ×6|96.23%|97.68%|
|E72|8|1K/128 ×188；1K/256 ×188；1K/384 ×188；192K/256 ×18|80.29%|91.62%|
|A6 / E80|6|1K/128 ×195；1K/256 ×195；1K/384 ×195；192K/256 ×12|74.53%|91.89%|
|B6|6|32K/128 ×18；48K/256 ×18；64K/512 ×18；192K/1024 ×3；192K/2048 ×9|95.72%|98.41%|

利用率均为种子 7、[1,2] 秒内 32 卡平均。每组使用同一冻结输入比较策略。八盘与六盘的配额不同，不能将两者差值归因于盘数。V38/B6 的五种画像计算参数直接取原 data；E72/A6 的 1K 画像使用外推，NQL 384 还含插值。频率和顺序都是合成输入，并非生产 trace。

所有请求 t=0 到达，每请求 8 层、batch=1。各卡配额相同，整个列表分别独立打乱一次：`Random(7+npu_id*100003)`。V38 的 32 份完整顺序互不相同，并非每张卡连续重复一种请求。各卡仍共享同一种画像分布，尚不代表真实业务的卡间分布差异或在线到达过程。

V38 中两类 192K 长请求占总纯计算时间的 62.45%，32K/128 仅占 4.32%。因此 Baseline 整机利用率虽然为 96.23%，32K/128 完整请求群的计算时间/接纳至完成时间仅为 57.80%，SLO 为 483/896；New Once 分别为 92.45% 和 870/896。这是两种统计口径，不能直接把逐类占比与整机窗口利用率相减。

## 数据与口径

- [画像参数 CSV](input_profiles.csv)：四组输入、18 行参数，包含每层计算时间和跨盘总读取量。
- [完整请求 CSV](input_sequences.csv)：42,784 条请求的原 NPU、顺序、ID、序列长度、NQL 和到达时间。
- [输入明细](input_details.md)：冻结输入路径、指纹、参数来源、时间占比、每卡前 12 条顺序。
- [完整研究报告](../report.pdf)：本次演示对应的完整研究背景和更大规模验证。
- [数值复核](source/numeric_review.md)与[版面复核](source/layout_review.md)。

表中的 C/V 是**每层**数值；V 为 SSD→HBM 跨全部 SSD 的总读取量。每请求纯计算时间为 8C、读取量为 8V。NQL 是本次新增 query token 数，SSD 历史前缀为序列总 token 数减 NQL。每盘带宽 40 GiB/s，每卡接收链路 50 GiB/s，I/O 命令 176 KiB。

U1/U2 分别取 [1,2] 秒和 [2,3] 秒。SLO 覆盖完整输入，条件为接纳至八层完成时间 ≤ 1.5×该请求八层纯计算时间，不含接纳前排队。S1–S3 pipeline 允许重分 NPU；Baseline、Once、New Once 和 S3 fixed 保留原绑定。

## 复现演示文件

在仓库根目录执行以下命令，仅读取已有冻结实验，不启动模拟：

```bash
python -B results/baseline_npu32_investigation/presentation/source/build_input_slides.py
libreoffice -env:UserInstallation=file:///tmp/qos-mixed-presentation-lo --headless --convert-to pdf --outdir results/baseline_npu32_investigation/presentation results/baseline_npu32_investigation/presentation/mixed_input_report.pptx
pdftoppm -png -scale-to-x 1800 -scale-to-y -1 results/baseline_npu32_investigation/presentation/mixed_input_report.pdf results/baseline_npu32_investigation/presentation/slides/slide
cp results/baseline_npu32_investigation/presentation/slides/slide-1.png results/baseline_npu32_investigation/presentation/mixed_input_report.png
```
