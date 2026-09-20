# 模板实验：增加 OD baseline

来源：`template/qos_experiments_20260919/index.html` 及其冻结输入、原始结果和绘图代码。

已完成。入口：[更新后的图片索引](../../template/qos_experiments_20260919/index.html) · [完整三策略对比表](../../template/qos_experiments_20260919/od_baseline_comparison/README.md) · [交付校验](delivery_checks.json)。

本次直接回放保存的请求，不重新随机抽样，也不重新生成放置。ASU 是原图中的 Baseline/FIFO；OD 每张 NPU 在每个 SSU 上使用一个独立 QoS Path，CIR 为盘带宽除以 NPU 数，PIR 不限，仍允许原生仲裁借用空闲带宽。流量分配仍指原始 Once。

## 实验范围

| 输入 | 新增 OD 次数 | 原配置与保留的图 |
|---|---:|---|
| full / semi / near35 | 9 | 各 3 种子；32 卡、3 盘、8 层；三策略 CDF、OD 逐盘带宽，semi 另有 OD 32 卡时序 |
| 128K/256 + 32K/4096 | 2 | Random / Ordered、seed 7；32 卡、3 盘、8 层；三策略 CDF、OD 时序、OD 32 卡层平均带宽 |
| 五组公式 AB | 15 | 各 3 种子；8 卡、1 盘、8 层；三策略 CDF |
| sensitivity20k_076 | 1 | seed 7；8 卡、1 盘、8 层；三策略 CDF、时序 |

上述共 27 次新增 OD 仿真，另外重跑原 ASU 用于版本一致性检查。原结果和原图保留，新增结果分目录存放。运行完成和校验状态以各目录的 `command.json` / `checks.json` 为准。

## 必须区分的统计口径

- NPU 利用率：原 warm `[2,4)` 秒内的实际计算卡时间，除以 NPU 数 × 2 秒。
- full / semi / near35 与旧 A/B 的 CDF：取 `[2,4)` 秒内接纳的请求，跟踪到完成。多种子曲线按原口径对种子等权。
- 五组公式 AB 的原 CDF：所有输入请求，包括启动阶段；不能与 warm 接纳 cohort 混称。汇总同时列出两种 SLO。
- TTFT 在这些仿真中是“接纳到 prefill 完成”的代理指标，不含接纳前排队，也不是真实首输出 token 的测量。
- SLO×1.5：上述时间不超过该请求 8 层纯计算时间的 1.5 倍。
- near35 是平均欠载、允许局部过载；沿用原目录分类并不证明持续欠载。
- 五组公式 AB 的盘速为十进制 **40 GB/s**；其他上述实验为 **40 GiB/s**。每一组均保留自己的原值。
- full / semi / near35 与旧 A/B 回放旧条带放置；五组公式 AB 与 sensitivity 保留其原 Ring Hash 及精确尾块。这不改变当前新输入只生成 Ring Hash 的接口。

## 文件

- `ab128/`：冻结旧 A/B 的 OD 与 ASU 一致性复跑、逐层 I/O 守恒审计、绘图和数值。
- `diverse/`：full / semi / near35 的输入核验、隔离运行源码、原曲线核对、OD 结果和绘图。
- `formula_sensitivity/`：五组 AB 与 sensitivity 的原源码、OD 接入、结果和绘图。

不生成用户排除的八卡带宽原图、NQL 辨认图、简化八卡带宽图、固定并发利用率图、等待分解图、sensitivity 配对带宽图。历史归档中的这些文件不删除。
