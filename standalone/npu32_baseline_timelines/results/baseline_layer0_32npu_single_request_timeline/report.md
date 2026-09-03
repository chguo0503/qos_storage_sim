# Baseline 单请求 Layer0 时间线

- 配置：32 NPU、11 SSU、1 层、全部 `(48,512)`。
- 每 NPU 一个请求，全部在 `t=0` 到达；无 ingress network；无跨请求 Layer0 prefetch。
- makespan：`10.125828975 ms`；物理 block：`12,160`。
- 仿真全程 NPU compute utilization：`49.816267%`；SSD mean utilization：`45.810293%`。
- Layer0：新增 I/O Stall `156.371957 NPU·ms`；累计利用率 `50.793915%`；超 deadline NPU `[1, 14, 17, 19, 23, 29, 30]`。

## 图片

1. [NPU I/O/compute timeline](01_npu_layer0_io_compute_timeline.png)
2. [Per-SSU Path0 enqueue order](02_ssu_path0_enqueue_order.png)
3. [Per-SSU physical service timeline](03_ssu_path0_service_timeline.png)

蓝色仅表示 deadline 前 I/O，红色斜线仅表示 deadline 后 I/O，绿色表示 compute。
Layer0 deadline 只是比较预算；Layer1/2 红段才等于真实本层 barrier。

## 数据与验证

- [Small per-request timeline](request_timeline.csv)
- [Machine-readable summary](summary.json)
- 完整 `physical_block_trace.csv` 默认省略；使用 `--write-physical-trace` 生成。
- 校验：`30/30` 为真。
- 观察器非干扰：插桩/无插桩完整 summary SHA-256 一致：`True`。
