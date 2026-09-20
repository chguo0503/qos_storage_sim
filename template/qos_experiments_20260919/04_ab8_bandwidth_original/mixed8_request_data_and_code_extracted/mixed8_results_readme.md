# 8 NPU 混合请求：明细、带宽与 TTFT SLO×1.5

2026-09-14。本次覆盖当前正式实验的三个场景：Baseline FIFO、原始Once per layer（5ms）和短读取优先诊断。均为8NPU、1SSU×40GiB/s、ring hash、seed7，每卡独立随机混合长短请求。数据直接来自已封存的正式运行，本次未重新仿真或更换输入。

## 文件

- `mixed8_request_details.xlsx`：两张可筛选工作表，完整列出576条请求画像、1728条（三策略×576）执行记录。
- `mixed8_all_images.zip`：**仅包含12张PNG图片**，不混入代码、数据或Excel预览。
- `mixed8_request_data_and_code.zip`：全部请求CSV/JSON、带宽曲线与层周期数据、CDF点、独立校验、原始manifest/result/receipts和复现代码。此包不含PNG，图片另包。

## 每条请求提供了什么

`request_profiles.csv`按NPU及输入顺序排列，包括request_id、NPU编号、输入顺序、实验L/S类、源码QoS类别、总tokens、总长度K、NQL、命中前缀tokens、每层读取GiB/MiB、每层计算ms、需求GiB/s、8层总计算与总读取量、计算数据的插值/外推说明及原始锚点。

总长度 = 命中前缀 + NQL；K=1024 tokens，GiB=2^30 bytes，MiB=2^20 bytes。每层带宽需求为：

`B (GiB/s) = 每层读取量 (GiB) / [每层计算时间 (ms) / 1000]`。

每卡完整输入72条：6个约200K长请求和66个约20K短请求。每卡内(总长度,NQL)组合均唯一，不要求不同卡之间画像也唯一。三策略的全部576条请求、NPU绑定、输入顺序和KV盘映射完全相同，因此共用一份画像表。

`request_execution.csv`为每条请求分别列出三策略的admission、completion、TTFT、8层纯计算时间、SLO×1.5阈值、归一化TTFT、是否达标、是否在warm内接纳/活动/计算/完成，以及窗口内compute、stall、L0及后7层stall。窗口内无活动时，窗口内利用率为空，而不是0。

Excel用公式保留读取量换算、V/C、TTFT、SLO阈值和达标判定；CSV/JSON保存原始精度数值。`request_field_definitions.json`解释每个字段及时间口径。

## 场景结果

| 场景 | warm整机U | 平均总需求 GiB/s | 平均总供给 GiB/s | warm接纳请求 SLO×1.5 | 相同576条全集 SLO×1.5 |
|---|---:|---:|---:|---:|---:|
| Baseline FIFO | 92.52% | 37.334 | 34.653 | 165/168 = 98.21% | 550/576 = 95.49% |
| Once per layer（5 ms） | 97.16% | 37.323 | 36.871 | 170/170 = 100.00% | 576/576 = 100.00% |
| 短读取优先（诊断） | 98.15% | 37.321 | 37.056 | 167/167 = 100.00% | 576/576 = 100.00% |

warm为[2,4)s。所有8卡在该窗口始终有请求，并且每卡都实际计算过L和S。整机U由8卡计算时间/16000card-ms得到。

## 整机带宽图的参考口径

本次3张`fleet_total_bandwidth/*_total_demand_supply.png`采用用户指定的[参考图定义](https://github.com/chguo0503/qos_storage_sim/blob/main/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/fleet_total_bandwidth_curves/README.md)：紫线是同一时刻8卡当前请求V/C之和，不除以8；蓝线先将各卡实际收到的下一层字节量摊入各自层周期，再对8卡求和。

层周期从当前层开始计算，到下一层开始计算，包含等待；完整跨请求周期也纳入。窗口两端的周期只用窗内实际收到的字节量/窗内片段时长。每卡曲线面积及整机曲线面积都已与同一次正式运行的物理receipt字节核对一致，需求均值/峰值也与原审计一致。

**蓝线是按各卡层周期摊平均值后的和，不是瞬时盘吞吐，也不是承诺带宽。**三场景蓝线峰值约为FIFO83.884、Once58.425、短优先66.071GiB/s；超过40不表示盘超速。不同卡的周期不齐，某卡将在周期后段收到的字节可能被摊入周期前段。例如FIFO在3758.915ms附近的高峰，包含NPU4直到3765.033ms才开始接收、但已被摊入该周期的字节。

因此蓝线可局部超过40，但整窗实际收字节量守恒；也不能把蓝紫两线比值当作瞬时或整窗NPU利用率。三张整机图共同使用0–90GiB/s纵轴，便于比较。真正用于本例欠载约束的是各卡当前请求的名义V/C需求：任意组合上界38.916459<40GiB/s。它不保证瞬时突发或跨请求L0切换时完全无等待。

`npu_bandwidth`中的3张旧逐卡图在跨请求/边缘周期用灰区不画蓝线；本次整机图按指定参考把这些周期计入，因此不能简单只把旧图可见的蓝线相加得到新图。

## TTFT SLO×1.5

TTFT采用原实验的admission-relative口径：`完成时刻 - NPU接纳时刻`。阈值为每请求`1.5 × 8 × 自身每层计算时间`。不包括t=0到NPU接纳前的队列等待；执行CSV另保留arrival-relative延迟字段，避免混淆。

CDF横轴为`TTFT / (8 × 每层计算时间)`，纵轴为累计请求比例。竖虚线在1.5，虚线处累计比例即SLO达标率；不是用同一个毫秒阈值衡量不同长度的请求。

- `ttft_slo15_cdf_window_admissions.png`：统计[2,4)s内接纳的请求，使用完整运行中它们的完成时间。三策略集合分别为168、170、167条，随执行进度不同。
- `ttft_slo15_cdf_all_requests.png`：统计三策略完全相同的576条请求，包含预热与运行尾段，用于同一请求全集的比较。
- `ttft_slo15_rates_by_class.png`：并列展示上述两种集合的整体、短流S、长流L达标率与实际分子/分母。

warm FIFO短流147/150=98.00%、长流18/18=100%；Once短流149/149、长流21/21均100%；诊断短流147/147、长流20/20均100%。全集FIFO短流502/528=95.08%、长流48/48=100%，其余两策略两类均100%。

## 图片清单（12张）

| ZIP内目录 | 数量 | 内容 |
|---|---:|---|
| fleet_total_bandwidth | 3 | 三场景各一张8卡总需求/总供给曲线 |
| npu_bandwidth | 3 | 三场景各一张完整8卡逐层带宽图 |
| timelines | 3 | 三场景各一张绿色计算/蓝色读取/黄色stall时序图 |
| ttft_slo | 3 | warm CDF、同一请求全集CDF、分长短类达标率对照 |

## 数据来源与复现

画像在约20K和约200K附近小幅变化；类内保持对齐命中前缀不变，因此类内每层读取量相同。20K的计算时间由原始`data`在32K/48K长度方向外推，长画像插值；没有新增硬件实测，没有计算缩放或padding。所有请求t=0到达，各NPU沿队列串行执行，输入顺序random不等同于泊松到达。1SSU下ring hash全部映射到唯一盘，同一个block所有层复用同一映射。短读取优先只是诊断，不能视为Once。

源快照为`chguo0503/qos_storage_sim` commit `75e10b8a84d4054921cd3149af40507444efb3fb`，原始41文件保留。明细与图都基于已封存的3个正式运行，未使用早期筛选窗口结果。

解压数据包后进入`source`目录，已有输入和receipt，可直接导出与绘图，无需重跑仿真：

```bash
python3 -m pip install -r requirements.txt
python3 export_mixed8_request_details.py
python3 plot_mixed8_fleet_total.py --font /path/to/chinese-font.ttf
python3 plot_mixed8_slo15.py --font /path/to/chinese-font.ttf
```

字体使用本机含中文字符的TTF/OTF/TTC文件；图片已经生成，包内不附字体。`data/image_manifest.json`列出全部PNG的SHA256；每条曲线的分段/层周期CSV与`*_audit.json`支持独立核算。图片ZIP与数据ZIP均经过CRC检查。
