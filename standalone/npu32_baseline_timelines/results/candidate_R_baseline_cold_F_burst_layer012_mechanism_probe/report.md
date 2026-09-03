# Candidate R：Baseline 三层同步 F-burst 冷启动机制探针

> **同步 cold F-burst 显微探针：30×F + 2×V；不含 S，非 warm formal R。**

本实验不能替代或证明 formal-R 的长窗口利用率结果。

- 配置：32 NPU、4 SSU、3 层、30 个 F=`(200,256)`；NPU 24/26 为 V=`(32,2048)`。
- 每 NPU 一个请求，全部在 `t=0` 到达；无 ingress network；无跨请求 Layer0 prefetch。
- makespan：`167.052257093 ms`；物理 block：`145,260`。
- 仿真全程 NPU compute utilization：`16.839165%`；SSD mean utilization：`91.219243%`。
- Layer0：新增 I/O Stall `1586.012471 NPU·ms`；累计利用率 `15.909051%`；超 deadline NPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31]`。
- Layer1：新增 I/O Stall `877.860402 NPU·ms`；累计利用率 `19.585972%`; warm 累计 NPU 利用率 `40.603697%`；超 deadline NPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`。
- Layer2：新增 I/O Stall `1137.926790 NPU·ms`；累计利用率 `19.994965%`; warm 累计 NPU 利用率 `30.870399%`；超 deadline NPU `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`。

## 图片

1. [NPU I/O/compute timeline](01_npu_layer012_io_compute_timeline.png)
2. [Same-run Layer0/1 simplified timeline](01b_npu_layer01_io_compute_timeline_simplified.png)
3. [Per-SSU Path0 enqueue order](02_ssu_path0_enqueue_order_layer012.png)
4. [Per-SSU physical service timeline](03_ssu_path0_service_timeline_layer012.png)
5. [Physical enqueue time by layer](04_ssu_path0_enqueue_time_layer012.png)

蓝色仅表示 deadline 前 I/O，红色斜线仅表示 deadline 后 I/O，绿色表示 compute。
Layer0 deadline 只是比较预算；Layer1/2 红段才等于真实本层 barrier。

## 数据与验证

- [Small per-request timeline](request_layer_timeline.csv)
- [Machine-readable summary](summary.json)
- 完整 `physical_block_trace.csv` 默认省略；使用 `--write-physical-trace` 生成。
- 校验：`30/30` 为真。
- 观察器非干扰：插桩/无插桩完整 summary SHA-256 一致：`True`。
