# 单个请求内部的逐层带宽图（PNG）

主入口：32 NPU、4 SSU × 40 GiB/s、长验证 seed 7。每卡有 Random A、Ordered A、Random B、Ordered B 四张图。每图只包含同一个请求的第2～4层计算及第3～5层预取，均完整、无跨请求、无读取边缘裁剪。

选 warm [2,4) 秒内最早能完整包含上述片段的对应画像请求，不选最大等待。A 各层175.65625 MiB / 6.024012 ms / B_i=28.475926 GiB/s；B各层38.5 MiB / 28.592842 ms / B_i=1.314932 GiB/s。

紫虚线是恒定需求 B_i，蓝线是 NPU 实际接收，橙线是 SSD 为这张卡读出。蓝色每层总面积相同；实际预取的层比当时计算的层晚一层。黑虚线/绿点线只标记第一次交接的截止/到齐。

各图时间跨度可能不同：比较时看毫秒刻度，不能只看屏幕上的宽度。图内的利用率、平均带宽只统计这3轮，不是整机或两秒warm统计。

完整3轮恰好计算3C、收到3V，因此 mean(b)/B=(3V/T)/(V/C)=3C/T=U 是这里的工作量核对，不能称为独立预测。平均到达带宽包含未接收时段，不表示链路限速。

| NPU | Random A | Ordered A | Random B | Ordered B |
|---:|---|---|---|---|
| 0 | [Random A](random_a_npu_00.png) | [Ordered A](ordered_a_npu_00.png) | [Random B](random_b_npu_00.png) | [Ordered B](ordered_b_npu_00.png) |
| 1 | [Random A](random_a_npu_01.png) | [Ordered A](ordered_a_npu_01.png) | [Random B](random_b_npu_01.png) | [Ordered B](ordered_b_npu_01.png) |
| 2 | [Random A](random_a_npu_02.png) | [Ordered A](ordered_a_npu_02.png) | [Random B](random_b_npu_02.png) | [Ordered B](ordered_b_npu_02.png) |
| 3 | [Random A](random_a_npu_03.png) | [Ordered A](ordered_a_npu_03.png) | [Random B](random_b_npu_03.png) | [Ordered B](ordered_b_npu_03.png) |
| 4 | [Random A](random_a_npu_04.png) | [Ordered A](ordered_a_npu_04.png) | [Random B](random_b_npu_04.png) | [Ordered B](ordered_b_npu_04.png) |
| 5 | [Random A](random_a_npu_05.png) | [Ordered A](ordered_a_npu_05.png) | [Random B](random_b_npu_05.png) | [Ordered B](ordered_b_npu_05.png) |
| 6 | [Random A](random_a_npu_06.png) | [Ordered A](ordered_a_npu_06.png) | [Random B](random_b_npu_06.png) | [Ordered B](ordered_b_npu_06.png) |
| 7 | [Random A](random_a_npu_07.png) | [Ordered A](ordered_a_npu_07.png) | [Random B](random_b_npu_07.png) | [Ordered B](ordered_b_npu_07.png) |
| 8 | [Random A](random_a_npu_08.png) | [Ordered A](ordered_a_npu_08.png) | [Random B](random_b_npu_08.png) | [Ordered B](ordered_b_npu_08.png) |
| 9 | [Random A](random_a_npu_09.png) | [Ordered A](ordered_a_npu_09.png) | [Random B](random_b_npu_09.png) | [Ordered B](ordered_b_npu_09.png) |
| 10 | [Random A](random_a_npu_10.png) | [Ordered A](ordered_a_npu_10.png) | [Random B](random_b_npu_10.png) | [Ordered B](ordered_b_npu_10.png) |
| 11 | [Random A](random_a_npu_11.png) | [Ordered A](ordered_a_npu_11.png) | [Random B](random_b_npu_11.png) | [Ordered B](ordered_b_npu_11.png) |
| 12 | [Random A](random_a_npu_12.png) | [Ordered A](ordered_a_npu_12.png) | [Random B](random_b_npu_12.png) | [Ordered B](ordered_b_npu_12.png) |
| 13 | [Random A](random_a_npu_13.png) | [Ordered A](ordered_a_npu_13.png) | [Random B](random_b_npu_13.png) | [Ordered B](ordered_b_npu_13.png) |
| 14 | [Random A](random_a_npu_14.png) | [Ordered A](ordered_a_npu_14.png) | [Random B](random_b_npu_14.png) | [Ordered B](ordered_b_npu_14.png) |
| 15 | [Random A](random_a_npu_15.png) | [Ordered A](ordered_a_npu_15.png) | [Random B](random_b_npu_15.png) | [Ordered B](ordered_b_npu_15.png) |
| 16 | [Random A](random_a_npu_16.png) | [Ordered A](ordered_a_npu_16.png) | [Random B](random_b_npu_16.png) | [Ordered B](ordered_b_npu_16.png) |
| 17 | [Random A](random_a_npu_17.png) | [Ordered A](ordered_a_npu_17.png) | [Random B](random_b_npu_17.png) | [Ordered B](ordered_b_npu_17.png) |
| 18 | [Random A](random_a_npu_18.png) | [Ordered A](ordered_a_npu_18.png) | [Random B](random_b_npu_18.png) | [Ordered B](ordered_b_npu_18.png) |
| 19 | [Random A](random_a_npu_19.png) | [Ordered A](ordered_a_npu_19.png) | [Random B](random_b_npu_19.png) | [Ordered B](ordered_b_npu_19.png) |
| 20 | [Random A](random_a_npu_20.png) | [Ordered A](ordered_a_npu_20.png) | [Random B](random_b_npu_20.png) | [Ordered B](ordered_b_npu_20.png) |
| 21 | [Random A](random_a_npu_21.png) | [Ordered A](ordered_a_npu_21.png) | [Random B](random_b_npu_21.png) | [Ordered B](ordered_b_npu_21.png) |
| 22 | [Random A](random_a_npu_22.png) | [Ordered A](ordered_a_npu_22.png) | [Random B](random_b_npu_22.png) | [Ordered B](ordered_b_npu_22.png) |
| 23 | [Random A](random_a_npu_23.png) | [Ordered A](ordered_a_npu_23.png) | [Random B](random_b_npu_23.png) | [Ordered B](ordered_b_npu_23.png) |
| 24 | [Random A](random_a_npu_24.png) | [Ordered A](ordered_a_npu_24.png) | [Random B](random_b_npu_24.png) | [Ordered B](ordered_b_npu_24.png) |
| 25 | [Random A](random_a_npu_25.png) | [Ordered A](ordered_a_npu_25.png) | [Random B](random_b_npu_25.png) | [Ordered B](ordered_b_npu_25.png) |
| 26 | [Random A](random_a_npu_26.png) | [Ordered A](ordered_a_npu_26.png) | [Random B](random_b_npu_26.png) | [Ordered B](ordered_b_npu_26.png) |
| 27 | [Random A](random_a_npu_27.png) | [Ordered A](ordered_a_npu_27.png) | [Random B](random_b_npu_27.png) | [Ordered B](ordered_b_npu_27.png) |
| 28 | [Random A](random_a_npu_28.png) | [Ordered A](ordered_a_npu_28.png) | [Random B](random_b_npu_28.png) | [Ordered B](ordered_b_npu_28.png) |
| 29 | [Random A](random_a_npu_29.png) | [Ordered A](ordered_a_npu_29.png) | [Random B](random_b_npu_29.png) | [Ordered B](ordered_b_npu_29.png) |
| 30 | [Random A](random_a_npu_30.png) | [Ordered A](ordered_a_npu_30.png) | [Random B](random_b_npu_30.png) | [Ordered B](ordered_b_npu_30.png) |
| 31 | [Random A](random_a_npu_31.png) | [Ordered A](ordered_a_npu_31.png) | [Random B](random_b_npu_31.png) | [Ordered B](ordered_b_npu_31.png) |

[数据和来源校验](checks.json) · [生成代码](../../../render_internal_layer_zoom.py)
