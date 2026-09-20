# 精确AB原始资料：200K/NQL2048 + 20K/NQL1024

这里确实有代码、原始结果和图片，本目录对应 **E1固定并发欠载实验**。同一对画像另有32卡4盘随机混排筛选，见上级目录的说明；两组配置与统计窗口不同。此次仅整理既有资料，没有重跑仿真，也没有生成或替换真实带宽供给曲线。

## 已证实的配置

- 8 NPU，1 SSU；每盘40 GB/s十进制，NPU接收50 GB/s。
- NPU0–3连续执行A，NPU4–7连续执行B；固定并发4A+4B，**每张卡没有混合AB，也不是随机混排**。
- A总长度200K、NQL2048；B总长度20K、NQL1024。B的计算时间来自32K/48K外推。
- 8层；seed7；主统计窗2–4秒。所有请求t=0进入各自队列。
- 普通并发名义需求为37.0377518053 GB/s，低于40；两策略完整轨迹名义需求均无过载。跨请求L0预取突发不加入名义需求，但等待仍计入利用率。

## 原结果

|指标|Baseline|Once|
|---|---:|---:|
|2–4s整机NPU利用率|83.50%|96.92%|
|2–4s A类利用率|100.00%|100.00%|
|2–4s B类利用率|67.00%|93.85%|
|2–4s接纳请求SLO×1.5|22.22%|100.00%|
|全请求SLO×1.5|39.83%|100.00%|

SLO=接纳到完成TTFT≤1.5×本请求8层纯计算时间，不包含NPU输入队列中接纳前等待。完整分类值见E1_fixed_reference.csv。

## 原图在哪里

- original_figures/fixed_concurrent_utilization.png 和.pdf：原始固定并发实验汇总，包含E1的整机/短类利用率。还含其他固定组，保留原样。
- original_figures/stall_decomposition.png 和.pdf：原始等待分解图，E1与C1、XY12_16各一栏，保留原样。

本包没有E1专门的逐NPU真实供给图或E1专门TTFT CDF。不能把其它画像/策略顺序的图混作本配置的原图。

## 代码和原始trace

- configs/E1_fixed_s1_seed7.json：完整输入和画像来源。
- run_experiment.py：原实验运行器。
- source/：逐文件按原实验source_sha256核验的核心源码与原始data。
- results/E1_fixed_s1_seed7/baseline/ 和once/：native_summary.json.gz、analysis.json、requests.json、config.json、adapter_statistics.json。
- audit_results.py：原统计代码。
- build_plots.py：原汇总绘图代码。本包只选出E1；若重画，其结果只含可读取的E1，原多组图片保留在original_figures。

原始压缩包曾缺中央目录，trace已通过原验证清单SHA256与gzip检查；源码仅以完全相同SHA256从其它原包补回，没有换版本。run_experiment.py导入检查通过，但本次没有运行仿真。

进入此目录，安装requirements后可使用原入口（会重算并写入相应结果目录，请先复制本包保留原结果）：

```bash
python run_experiment.py configs/E1_fixed_s1_seed7.json --strategy baseline --force
python run_experiment.py configs/E1_fixed_s1_seed7.json --strategy once --force
```

不加--force会复用已完成结果。source_recovery_provenance.json记录严格同hash源码补回来源；附加其它实验脚本仍有未恢复项，但主运行器的导入依赖已齐备。
