# 正式60秒结果：独立图索引

本目录含14张独立图，每张提供PNG、PDF和SVG；均取自已完成的seed 7结果，没有额外仿真或时间线拼接。32 NPU、6 SSU，每盘40 GiB/s。随机与定序使用同一每卡请求人口和物理放置，改变请求顺序。

主时间线覆盖[2,60)秒；细节时间线覆盖[2,12)秒。蓝色为NQL1024快短计算，紫色为32K/NQL4096桥接计算，绿色为Long计算，橙色为接纳后的真实I/O等待。桥接的每层计算时间28.593ms单独列示，全部真实计算均计入设备利用率。

| 输入 | 策略 | U[2,60) | U[2,12) | 全程名义最热盘峰值 GiB/s |
|---|---|---:|---:|---:|
| 随机 | Baseline | 99.997867% | 99.992793% | 36.428824 |
| 随机 | Once per layer | 99.987211% | 99.976792% | 36.586561 |
| 定序 | Baseline | 89.964747% | 89.794543% | 36.011188 |
| 定序 | Once per layer | 99.961111% | 99.967428% | 36.021873 |

主窗设备U的分母固定为32卡×58秒；全部32卡全窗有任务，且每卡均有实际快短和Long计算。更细窗口的计算占比以该图自己的窗口为准。

| 输入与策略 | 图内容 | PNG | PDF | SVG | 审计 |
|---|---|---|---|---|---|
| 随机 / Baseline | 32卡时间线 [2,60)s | [PNG](random_baseline_long.png) | [PDF](random_baseline_long.pdf) | [SVG](random_baseline_long.svg) | [JSON](random_baseline_long.audit.json) |
| 随机 / Baseline | 32卡细节 [2,12)s | [PNG](random_baseline_detail.png) | [PDF](random_baseline_detail.pdf) | [SVG](random_baseline_detail.svg) | [JSON](random_baseline_detail.audit.json) |
| 随机 / Baseline | 6盘当前请求名义需求 [2,60)s | [PNG](random_baseline_demand.png) | [PDF](random_baseline_demand.pdf) | [SVG](random_baseline_demand.svg) | [JSON](random_baseline_demand.audit.json) |
| 随机 / Once per layer | 32卡时间线 [2,60)s | [PNG](random_once_long.png) | [PDF](random_once_long.pdf) | [SVG](random_once_long.svg) | [JSON](random_once_long.audit.json) |
| 随机 / Once per layer | 32卡细节 [2,12)s | [PNG](random_once_detail.png) | [PDF](random_once_detail.pdf) | [SVG](random_once_detail.svg) | [JSON](random_once_detail.audit.json) |
| 随机 / Once per layer | 6盘当前请求名义需求 [2,60)s | [PNG](random_once_demand.png) | [PDF](random_once_demand.pdf) | [SVG](random_once_demand.svg) | [JSON](random_once_demand.audit.json) |
| 定序 / Baseline | 32卡时间线 [2,60)s | [PNG](ordered_baseline_long.png) | [PDF](ordered_baseline_long.pdf) | [SVG](ordered_baseline_long.svg) | [JSON](ordered_baseline_long.audit.json) |
| 定序 / Baseline | 32卡细节 [2,12)s | [PNG](ordered_baseline_detail.png) | [PDF](ordered_baseline_detail.pdf) | [SVG](ordered_baseline_detail.svg) | [JSON](ordered_baseline_detail.audit.json) |
| 定序 / Baseline | 6盘当前请求名义需求 [2,60)s | [PNG](ordered_baseline_demand.png) | [PDF](ordered_baseline_demand.pdf) | [SVG](ordered_baseline_demand.svg) | [JSON](ordered_baseline_demand.audit.json) |
| 定序 / Once per layer | 32卡时间线 [2,60)s | [PNG](ordered_once_long.png) | [PDF](ordered_once_long.pdf) | [SVG](ordered_once_long.svg) | [JSON](ordered_once_long.audit.json) |
| 定序 / Once per layer | 32卡细节 [2,12)s | [PNG](ordered_once_detail.png) | [PDF](ordered_once_detail.pdf) | [SVG](ordered_once_detail.svg) | [JSON](ordered_once_detail.audit.json) |
| 定序 / Once per layer | 6盘当前请求名义需求 [2,60)s | [PNG](ordered_once_demand.png) | [PDF](ordered_once_demand.pdf) | [SVG](ordered_once_demand.svg) | [JSON](ordered_once_demand.audit.json) |

| 补充图 | PNG | PDF | SVG | 数据与审计 |
|---|---|---|---|---|
| 四格各29个独立2秒窗口的设备U | [PNG](rolling_2s_utilization.png) | [PDF](rolling_2s_utilization.pdf) | [SVG](rolling_2s_utilization.svg) | [CSV](rolling_2s_utilization.csv) · [JSON](rolling_2s_utilization.audit.json) |
| 定序Baseline后半段真实交接 [51100,51520)ms | [PNG](ordered_baseline_late_handoff.png) | [PDF](ordered_baseline_late_handoff.pdf) | [SVG](ordered_baseline_late_handoff.svg) | [JSON](ordered_baseline_late_handoff.audit.json) · [说明](../../notes/late_handoff.md) |

名义需求按当前已接纳请求的每盘每层D/C相加；跨请求下一层L0按约定不额外加入此项，但原模拟执行全部预取。该曲线不是瞬间释放量、设备吞吐或SSD busy。后半段图的“读取生命周期”包含排队与传输，不是单盘独占服务时间，也不把某条同时发生的Long读取认定为短层的具体FIFO队头。

这是一批有限、全部在t=0到达的构造请求，图覆盖本次完整长窗，不能仅凭图推广为无限稳态或所有混合输入。58秒全景中的微小等待可能小于像素，请结合细节图、局部交接和精确数值阅读。

[最终视觉与数值审阅](final_layout_review.md) · [全部42个图文件哈希及逐项检查](final_layout_review.json) · [绘图计划](figure_plan.json) · [四格生成脚本](../../build_final_figures.py) · [四颜色时间线脚本](../../draw_final_timeline.py) · [后半段交接脚本](../../build_late_handoff.py)
