# Once per layer：32 卡计算与 I/O 等待

32 NPU、3 SSU、seed 7、warm [2,4) 秒。使用已有完整仿真日志，沿用 Baseline 时序图样式。

| 顺序 | NPU 平均利用率 | PNG | PDF |
|---|---:|---|---|
| Ordered | 83.4170% | [PNG](once_ordered.png) | [PDF](once_ordered.pdf) |
| Random | 91.2428% | [PNG](once_random.png) | [PDF](once_random.pdf) |

蓝色为 A 请求计算，绿色为 B 请求计算，橙色为接纳后 I/O 等待；右侧为每张卡在同一窗口内的计算利用率。

[数值与来源校验](once_timeline_checks.json) · [绘图脚本](../../render_once_timelines.py)
