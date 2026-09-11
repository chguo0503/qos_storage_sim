**总输入超过 10K：独立图片**

32 NPU、6 SSU；每盘 40 GiB/s，每卡接收链路 50 GiB/s。短画像总输入是 32K、48K、64K，长画像是 176K，全部直接来自 `data`；NQL=1024 是需要新计算的 token 数，不是完整输入长度。

以下主图统一使用 seed 7、原定 `[2,4)` 秒窗口。每张卡都有长短两类的正计算，32 卡全窗有任务。颜色：蓝色短计算、绿色长计算、橙色接纳后 I/O 等待。每个文件只放一张图。

| 图 | 对应整机 U | PNG | 单页 PDF | SVG |
|---|---:|---|---|---|
| Random Baseline | 99.9825% | [查看](random_baseline.png) | [下载](random_baseline.pdf) | [下载](random_baseline.svg) |
| Ordered Baseline | 89.5726% | [查看](ordered_baseline.png) | [下载](ordered_baseline.pdf) | [下载](ordered_baseline.svg) |
| Random Once per layer | 99.9361% | [查看](random_once.png) | [下载](random_once.pdf) | [下载](random_once.svg) |
| Ordered Once per layer | 99.2081% | [查看](ordered_once.png) | [下载](ordered_once.pdf) | [下载](ordered_once.svg) |
| Ordered 局部计算/等待/真实服务 | 89.5726% 为完整主窗值 | [查看](ordered_zoom.png) | [下载](ordered_zoom.pdf) | [下载](ordered_zoom.svg) |
| Random Baseline 六盘需求 | — | [查看](random_baseline_demand.png) | [下载](random_baseline_demand.pdf) | [下载](random_baseline_demand.svg) |
| Ordered Baseline 六盘需求 | — | [查看](ordered_baseline_demand.png) | [下载](ordered_baseline_demand.pdf) | [下载](ordered_baseline_demand.svg) |
| Random Once 六盘需求 | — | [查看](random_once_demand.png) | [下载](random_once_demand.pdf) | [下载](random_once_demand.svg) |
| Ordered Once 六盘需求 | — | [查看](ordered_once_demand.png) | [下载](ordered_once_demand.pdf) | [下载](ordered_once_demand.svg) |
| 三倍新队列：Ordered Baseline `[2,12)` | 97.8355% | [查看](ordered_baseline_long_window.png) | [下载](ordered_baseline_long_window.pdf) | [下载](ordered_baseline_long_window.svg) |

最后一行是已有扩展输入的单次结果：每卡完整队列重复三遍，按真实执行进度推进，不是拉长原本已经耗尽的输入，也不是拼接旧日志。后续恢复表明，这个混合构造不能证明长期损失 10 个百分点。

六盘需求 = 当前已接纳请求的单层逐盘读取量 / 单层计算时间。沿用原约定，不额外叠加下一请求首层预取；所有实际 I/O 仍完整执行。它不是实际 SSD 吞吐，也不是没有突发的保证。

[完整报告与输入分配](../../report.md) · [逐种子 CSV](../../per_seed.csv) · [20 格独立审计](../../audit.json) · [局部数值证据](ordered_zoom_evidence.json)
