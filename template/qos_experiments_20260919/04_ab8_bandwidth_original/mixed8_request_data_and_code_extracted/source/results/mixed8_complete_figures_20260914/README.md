# 8 NPU 混合请求：明细、带宽与 TTFT SLO×1.5

2026-09-14。本次覆盖当前正式实验的三个场景：Baseline FIFO、原始Once per layer（5ms）和短读取优先诊断。均为8NPU、1SSU×40GiB/s、ring hash、seed7，每卡独立随机混合长短请求。数据直接来自已封存的正式运行，本次以原输入只读重放补齐物理IO时间桶，调度与执行结果完全一致。

## 文件

- `mixed8_request_details.xlsx`：两张可筛选工作表，完整列出576条请求画像、1728条（三策略×576）执行记录。
- `mixed8_all_images.zip`：**仅包含13张PNG图片**，不混入代码、数据或Excel预览。
- `mixed8_request_data_and_code.zip`：全部请求CSV/JSON、带宽曲线与层周期数据、CDF点、独立校验、原始manifest/result/receipts和复现代码。此包不含PNG，图片另包。

## 每条请求提供了什么

`request_profiles.csv`按NPU及输入顺序排列，包括request_id、NPU编号、输入顺序、实验L/S类、源码QoS类别、总tokens、总长度K、NQL、命中前缀tokens、每层读取GiB/MiB、每层计算ms、需求GiB/s、8层总计算与总读取量、计算数据的插值/外推说明及原始锚点。

总长度 = 命中前缀 + NQL；K=1024 tokens，GiB=2^30 bytes，MiB=2^20 bytes。每层带宽需求为：

`B (GiB/s) = 每层读取量 (GiB) / [每层计算时间 (ms) / 1000]`。

每卡完整输入72条：6个约200K长请求和66个约20K短请求。每卡内(总长度,NQL)组合均唯一，不要求不同卡之间画像也唯一。三策略的全部576条请求、NPU绑定、输入顺序和KV盘映射完全相同，因此共用一份画像表。

`request_execution.csv`为每条请求分别列出三策略的admission、completion、TTFT、8层纯计算时间、SLO×1.5阈值、归一化TTFT、是否达标、是否在warm内接纳/活动/计算/完成，以及窗口内compute、stall、L0及后7层stall。窗口内无活动时，窗口内利用率为空，而不是0。

Excel用公式保留读取量换算、V/C、TTFT、SLO阈值和达标判定；CSV/JSON保存原始精度数值。`request_field_definitions.json`解释每个字段及时间口径。

## 场景结果

| 场景 | warm整机U | 平均总需求 GiB/s | 盘侧平均吞吐 GiB/s | warm接纳请求 SLO×1.5 | 相同576条全集 SLO×1.5 |
|---|---:|---:|---:|---:|---:|
| Baseline FIFO | 92.52% | 37.334 | 34.653 | 165/168 = 98.21% | 550/576 = 95.49% |
| Once per layer（5 ms） | 97.16% | 37.323 | 36.871 | 170/170 = 100.00% | 576/576 = 100.00% |
| 短读取优先（诊断） | 98.15% | 37.321 | 37.056 | 167/167 = 100.00% | 576/576 = 100.00% |

warm为[2,4)s。所有8卡在该窗口始终有请求，并且每卡都实际计算过L和S。整机U由8卡计算时间/16000card-ms得到。

## 更新：真实盘侧带宽与截止时间容量不足

`fleet_total_bandwidth/*_total_demand_supply.png` 已全部更新为真实盘侧统计，取代旧版“不同 NPU 层周期平均接收速率之和”。当前三张图每张分为三部分：

1. **物理供给**：蓝线为同一个2ms窗口内，SSD实际服务的所有IO字节总量/0.002s。每条IO按真实服务区间与时间桶的交集分摊，排队不产生字节。紫线保留当前请求常规V/C的时间平均，虚线为40GiB/s容量。没有人为截断、限幅或平滑。
2. **截止时间参考需求**：当前层计算期间，需要把下一层（含下一请求L0）的数据读完，参考速率为 `V_next / C_current`。只在当前层计算区间累加，所有NPU再在同一2ms时间窗内平均；已过期读取不把分母改成负数或继续计入，实际等待在下图展示。红叉表示某一单项预取的读取量本身就超过其计算窗口内盘最多能服务的量，且发起与截止均在warm内。
3. **实际IO等待**：每个2ms内等待IO的NPU数量的时间平均。积分为实际等待卡·ms，核对后与原利用率完全一致。

| 场景 | 真实SSD平均 GiB/s | 真实SSD峰值 GiB/s | 截止需求2ms均值峰 GiB/s | 单项无法按时预取的完整窗口数 |
|---|---:|---:|---:|---:|
| FIFO | 34.652844 | 40 | 121.142919 | 15 |
| Once | 36.871341 | 40 | 82.978693 | 19 |
| 短读取优先诊断 | 37.055689 | 40 | 83.506504 | 18 |

三场景每场观测1,251,456条SSD IO，20项重放校验全部通过。summary/windows/slo的canonical SHA256与原结果完全一致；无物理服务重叠、无字节丢失，核心代码与原输入/结果不变。原始最大浮点值40.00000000000001属于舍入，判定容差1e-7GiB/s；数据未被截到40。SSU读服务与NPU链路接收分开记录，画盘容量图只使用SSU服务桶。

橙线超过40表示“同时按各自窗口匀速完成”的参考需求超线，**单凭它不能一般性证明截止时间集合不可行**，因为不同窗口可能错位并提前服务。红叉则满足单项 `V_next > 40 × C_current秒`，因此即使独占盘也来不及。策略在warm内的执行进度不同，15/19/18不是同一请求集合，不能单凭次数比较策略好坏。

新增 `capacity_overload/fifo_capacity_overload_zoom.png` 给出更严格的共同窗口证据：NPU7的L0预取发起于3757.731721ms，需在3763.243552ms前就绪；NPU4发起于3758.914535ms，需在3764.333545ms前就绪。两项各需0.266372681GiB，且都必须完全在[3757.731721,3764.333545]ms内完成。

`必须读取0.532745361GiB > 40GiB/s × 0.006601825秒 = 0.264072985GiB`。

因此即使其他六卡停止读取，也无法同时满足这两个截止时间，至少需要80.696685GiB/s。这是**短截止窗口内真实需求超过容量**的证据，实际盘吞吐仍不超过40。

这里揭示了原“欠载”结论的边界：任意当前请求常规V/C组合的上界38.916459<40，只证明该常规需求口径欠载；它没有保证短→长L0预取的所有截止窗口也欠载。本次未调整输入来制造过载，而是补充原输入已有的截止时间约束。

逐NPU旧图仍使用完整内部层周期的平均接收量V/(C+stall)，不等同于本次2ms SSD物理吞吐；灰区没有蓝线，其右侧实际接收均值包括灰区。新旧带宽口径不要混用。旧 `mixed8_complete_figures_20260914/data/*fleet_total*` CSV/JSON仅保留为历史核对；新图的权威数据位于 `mixed8_physical_bandwidth_20260914/data`。

数据目录含共同2ms的physical_demand CSV、原始physical_bins及重放audit JSON、逐项预取deadline CSV、精确事件需求分段CSV、容量不足见证JSON。精确需求越线时长与“2ms平均值越线桶数×2ms”是不同量，summary分别保存。

## TTFT SLO×1.5

TTFT采用原实验的admission-relative口径：`完成时刻 - NPU接纳时刻`。阈值为每请求`1.5 × 8 × 自身每层计算时间`。不包括t=0到NPU接纳前的队列等待；执行CSV另保留arrival-relative延迟字段，避免混淆。

CDF横轴为`TTFT / (8 × 每层计算时间)`，纵轴为累计请求比例。竖虚线在1.5，虚线处累计比例即SLO达标率；不是用同一个毫秒阈值衡量不同长度的请求。

- `ttft_slo15_cdf_window_admissions.png`：统计[2,4)s内接纳的请求，使用完整运行中它们的完成时间。三策略集合分别为168、170、167条，随执行进度不同。
- `ttft_slo15_cdf_all_requests.png`：统计三策略完全相同的576条请求，包含预热与运行尾段，用于同一请求全集的比较。
- `ttft_slo15_rates_by_class.png`：并列展示上述两种集合的整体、短流S、长流L达标率与实际分子/分母。

warm FIFO短流147/150=98.00%、长流18/18=100%；Once短流149/149、长流21/21均100%；诊断短流147/147、长流20/20均100%。全集FIFO短流502/528=95.08%、长流48/48=100%，其余两策略两类均100%。

## 图片清单（13张）

| ZIP内目录 | 数量 | 内容 |
|---|---:|---|
| fleet_total_bandwidth | 3 | 三场景真实盘吞吐、截止参考需求与IO等待 |
| capacity_overload | 1 | FIFO两个预取共同截止窗口的严格容量不足证据 |
| npu_bandwidth | 3 | 三场景各一张完整8卡逐层带宽图 |
| timelines | 3 | 三场景各一张绿色计算/蓝色读取/黄色stall时序图 |
| ttft_slo | 3 | warm CDF、同一请求全集CDF、分长短类达标率对照 |

## 数据来源与复现

画像在约20K和约200K附近小幅变化；类内保持对齐命中前缀不变，因此类内每层读取量相同。20K的计算时间由原始`data`在32K/48K长度方向外推，长画像插值；没有新增硬件实测，没有计算缩放或padding。所有请求t=0到达，各NPU沿队列串行执行，输入顺序random不等同于泊松到达。1SSU下ring hash全部映射到唯一盘，同一个block所有层复用同一映射。短读取优先只是诊断，不能视为Once。

源快照为`chguo0503/qos_storage_sim` commit `75e10b8a84d4054921cd3149af40507444efb3fb`，原始41文件保留。明细与图都基于3个正式运行；新增只读重放严格复现原执行，未使用早期筛选窗口结果。

解压数据包后进入`source`目录，已有输入和physical_bins，可直接导出与绘图；只有需要重建IO观测时才运行replay_mixed8_physical.py：

```bash
python3 -m pip install -r requirements.txt
python3 export_mixed8_request_details.py
python3 plot_mixed8_physical.py --font /path/to/chinese-font.ttf
python3 plot_mixed8_slo15.py --font /path/to/chinese-font.ttf
```

字体使用本机含中文字符的TTF/OTF/TTC文件；图片已经生成，包内不附字体。`data/image_manifest.json`列出全部PNG的SHA256；每条曲线的分段/层周期CSV与`*_audit.json`支持独立核算。图片ZIP与数据ZIP均经过CRC检查。
