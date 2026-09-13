# 3 SSU：每张 NPU 的带宽局部放大（只生成 PNG）

本目录展示跨 A→B 请求的首层交接。首层使用前一个 A 末层的计算预算，不能直接用 B 自身层内 V/C 代替。

使用周六长验证 seed 7 原日志，32 NPU、3 SSU × 40 GiB/s，总盘容量 120 GiB/s，NPU 接收上限 50 GiB/s。每张卡分别从完整图窗位于 warm [2,4) 秒内的 A→B 交接中选择首层等待最长的一次；并列取最早。图是局部案例，不是整窗或整机平均。

所有图统一横轴 [-8,65] ms，0 为目标 B 首层预取发出。紫虚线是当前请求自身的 V/C，蓝线是 NPU 实际接收，橙线是 SSD 为本卡读出。两种实际带宽分别绘图；不要把不同纵轴的视觉高度直接比较。

蓝色笔画统一为 1.9 pt；看到的宽窄主要是接收脉冲持续时间。高度=速率，横向宽度=接收时间，曲线下方面积=数据量。A 每层 175.65625 MiB，在 50 GiB/s 下有效接收 3.430786 ms；B 每层 38.5 MiB，对应 0.751953 ms。同一画像各层读取量相同；有间隙时加总各段面积，边缘裁剪按图中说明理解。

累计面板只统计目标 B 首层，其预算来自前一个 A 的 6.024 ms，对应约 6.24 GiB/s，不是 B 内部层的 1.31 GiB/s。计算与等待条和带宽曲线使用同一条时间轴。

推荐先看 NPU 15：Random 本次首层等待 14.402 ms，Ordered 等待 13.600 ms。它们是各自挑出的局部事件，不是同一绝对时刻，也不代表整体优劣。逐块服务重建、无分箱或平滑，没有运行新仿真。

已核实所选 64 次首层读取均在统一 [-8,65] ms 图窗中完整到齐，不需要扩大窗口；下一层提前或晚到根据各图日志计算，不能由盘数预设。

| NPU | Random PNG | Ordered PNG |
|---:|---|---|
| 0 | [Random](random_npu_00.png) | [Ordered](ordered_npu_00.png) |
| 1 | [Random](random_npu_01.png) | [Ordered](ordered_npu_01.png) |
| 2 | [Random](random_npu_02.png) | [Ordered](ordered_npu_02.png) |
| 3 | [Random](random_npu_03.png) | [Ordered](ordered_npu_03.png) |
| 4 | [Random](random_npu_04.png) | [Ordered](ordered_npu_04.png) |
| 5 | [Random](random_npu_05.png) | [Ordered](ordered_npu_05.png) |
| 6 | [Random](random_npu_06.png) | [Ordered](ordered_npu_06.png) |
| 7 | [Random](random_npu_07.png) | [Ordered](ordered_npu_07.png) |
| 8 | [Random](random_npu_08.png) | [Ordered](ordered_npu_08.png) |
| 9 | [Random](random_npu_09.png) | [Ordered](ordered_npu_09.png) |
| 10 | [Random](random_npu_10.png) | [Ordered](ordered_npu_10.png) |
| 11 | [Random](random_npu_11.png) | [Ordered](ordered_npu_11.png) |
| 12 | [Random](random_npu_12.png) | [Ordered](ordered_npu_12.png) |
| 13 | [Random](random_npu_13.png) | [Ordered](ordered_npu_13.png) |
| 14 | [Random](random_npu_14.png) | [Ordered](ordered_npu_14.png) |
| 15 | [Random](random_npu_15.png) | [Ordered](ordered_npu_15.png) |
| 16 | [Random](random_npu_16.png) | [Ordered](ordered_npu_16.png) |
| 17 | [Random](random_npu_17.png) | [Ordered](ordered_npu_17.png) |
| 18 | [Random](random_npu_18.png) | [Ordered](ordered_npu_18.png) |
| 19 | [Random](random_npu_19.png) | [Ordered](ordered_npu_19.png) |
| 20 | [Random](random_npu_20.png) | [Ordered](ordered_npu_20.png) |
| 21 | [Random](random_npu_21.png) | [Ordered](ordered_npu_21.png) |
| 22 | [Random](random_npu_22.png) | [Ordered](ordered_npu_22.png) |
| 23 | [Random](random_npu_23.png) | [Ordered](ordered_npu_23.png) |
| 24 | [Random](random_npu_24.png) | [Ordered](ordered_npu_24.png) |
| 25 | [Random](random_npu_25.png) | [Ordered](ordered_npu_25.png) |
| 26 | [Random](random_npu_26.png) | [Ordered](ordered_npu_26.png) |
| 27 | [Random](random_npu_27.png) | [Ordered](ordered_npu_27.png) |
| 28 | [Random](random_npu_28.png) | [Ordered](ordered_npu_28.png) |
| 29 | [Random](random_npu_29.png) | [Ordered](ordered_npu_29.png) |
| 30 | [Random](random_npu_30.png) | [Ordered](ordered_npu_30.png) |
| 31 | [Random](random_npu_31.png) | [Ordered](ordered_npu_31.png) |

[来源与数值校验](checks.json) · [生成代码](../../../render_bandwidth_zoom_ssu3.py)
