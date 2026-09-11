# 两张原始四画像时间线的视觉核验

已分别查看 Random、Ordered 最终 PNG；标题、图例、32 卡编号、坐标轴及脚注完整，无裁切或重叠。两张图独立输出，未覆盖父研究图。

- Random：设备 U=99.9997319413%，compute=63999.828442450 NPU·ms，IO stall=0.171557550 NPU·ms。
- Ordered：设备 U=99.8155782852%，compute=63881.970102512 NPU·ms，IO stall=118.029897488 NPU·ms。
- 两者均为真实 [2000,4000) ms，32 卡整窗 active，compute+stall=64000 NPU·ms；暖窗利用率与原结果 windows 字段相符。
- 共用父 loader 逐层核对 compute duration、io_barrier_wait_ms 和前驱计算结束/接纳时刻，等待不包含接纳前排队或预取 read lifetime。
- 通过 original_request_id 核对完整人口、原 NPU、C/V、arrival 和实际 placement 一致。原始短画像均为策略 SL，已在图中明示。
- 真实极短等待保留原宽度，不加宽到最小像素，因此 Random 图上橙色可能不可见。

生成命令：`python -B results/baseline_32npu6ssu_underload/raw_quartet_replacement/plot_raw_timelines.py`。
Python：3.10.10。图源、原输入/结果 SHA、复用绘图源码 SHA 和输出 SHA 均存于 [timeline_audit.json](timeline_audit.json)。没有运行仿真。
