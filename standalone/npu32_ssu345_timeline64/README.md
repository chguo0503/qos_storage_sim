# 32-NPU / SSU 3–5 / 64 秒时间线独立复现实验

这个目录是本次实验的独立最小项目。它只包含：

- 正式仿真的 17 个传递依赖源码、冻结分析器和认证 workload `data`；
- 3 个策略 × SSU 3/4/5 的完整正式派生结果；
- 观测器等价性验证；
- 依赖声明和离线完整性检查器。

没有复制仓库中的旧实验、失败校准、`*_checks.py`、128-NPU 脚本、source
snapshot、tar 包或历史结果。冻结分析器内部的 self-test/oracle 仍然保留，因为
删除它们会改变已经审计并写入正式结果的 analyzer SHA-256。

## 最重要的结论和证据

正文只采用 α=2；α=1.5 是同一轨迹上的补充统计。

| SSU | Baseline util / α2 | Layer-once TTL5 util / α2 | Adaptive 100ms util / α2 |
|---:|---:|---:|---:|
| 3 | 58.19% / 40.02% | 58.32% / 66.34% | 58.93% / 72.50% |
| 4 | 76.47% / 53.06% | 76.33% / 75.32% | 76.07% / 78.72% |
| 5 | 89.92% / 70.00% | 89.20% / 85.07% | 89.61% / 84.96% |

以下文件共同证明“策略改变的是请求之间的等待分布，而不是增加总计算能力”：

1. `formal_results/state_duration_summary.csv`、
   `state_durations_per_npu.csv` 和 `state_durations_500ms.csv`：完整 64 秒、
   逐 NPU、逐 500ms 验证 `compute + IO barrier + other = 100%`，其中
   `other≈0`；策略间 utilization 差与 IO-barrier 差反向等量闭合。
2. `formal_results/matched_requests.csv`、`matched_request_summary.csv` 和
   `matched_request_coverage.csv`：在三策略共同的 `(NPU, sequence)` 请求上，
   直接统计 barrier 改善/恶化以及 α2 fail→pass / pass→fail。以 SSU4
   Adaptive 为例，α2 为 `951/8`，但 common cohort 的 barrier 总量反而增加，
   说明提升来自阈值附近的等待重分布，而不是所有请求的等待都下降。
3. `formal_results/compute_inventory_decomposition.csv`：逐式闭合
   `compute busy = activated compute + Q(start) - Q(end)`，解释短窗口的
   carry-in/库存效应，没有把策略描述成创造额外算力。
4. `formal_results/08_matched_request_barrier_ecdf.png`、
   `05_state_duration_partition.png`、`05b_state_duration_timeline.png` 和
   `03_cumulative_multiscale_util_slo.png` 是上述证据的图形化版本。

完整说明见 `formal_results/report.md`。

## 环境

- Linux；runner 使用 `spawn`，并读取 `/proc/self/status`/`fcntl`。
- Python 3.10+。
- `numpy>=1.24`、`matplotlib>=3.7`。
- 正式 simulation 环境记录在结果中：CPython 3.14.4、NumPy 2.5.2、
  OpenBLAS 0.3.34，线程数限制为 1。

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 重新运行正式仿真

三个 SSU 建议逐个运行，避免三个进程池同时占满内存。每个 shard 内的三个策略
由 `--max-workers 3` 并行：

```bash
mkdir -p raw-private

for ssu in 3 4 5; do
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python ms_scale_control_experiment.py \
      --definition legacy32 \
      --case baseline \
      --case layer_once_ttl_5ms \
      --case adaptive_t0_i100ms \
      --ssu "$ssu" \
      --max-workers 3 \
      --mp-start-method spawn \
      --seed 42 \
      --requests-per-npu 256 \
      --warmup-requests 8 \
      --settle-ms 500 \
      --measurement-ms 64000 \
      --block-ms 500 \
      --timeline-diagnostics \
      --timeline-dispatch-probe-ms 50 \
      --timeline-dispatch-probe-limit 10000 \
      --fresh \
      --output "raw-private/shard_ssu${ssu}.json"
done
```

正式 raw SHA-256 应为：

| shard | SHA-256 |
|---|---|
| SSU3 | `82fe27388c49353e552ecadca22458714771f659f463c0290062997118878bab` |
| SSU4 | `475496b1839f07494b3289dbca4a8a3b756139c0134e9f9b0a0e2d71cac9c930` |
| SSU5 | `28e06f0ef8005c0dc62b775f72f9ecc845d3d80cac5ef18c5941b7dec28ed12a` |

raw shard 包含运行主机名、PID 等 provenance，因此 `raw-private/` 被明确忽略，
不会上传。公开结果只保留 raw basename 和不可逆 SHA-256。

## 重新生成分析结果

输出目录必须不存在或为空：

```bash
mkdir -p /tmp/qos-timeline-mpl reproduced_results
env MPLCONFIGDIR=/tmp/qos-timeline-mpl \
  python analyze_npu32_ssu345_timeline64.py \
    --input raw-private/shard_ssu3.json \
    --input raw-private/shard_ssu4.json \
    --input raw-private/shard_ssu5.json \
    --output-dir reproduced_results
```

## 验证上传包

```bash
python verify_bundle.py
```

检查器会验证整个独立目录的 bundle manifest、正式结果内部 manifest、所有文件
大小/SHA-256、42 个正式 validation checks、冻结 analyzer，以及 9 个 α2/util
核心点。

正式结果目录共有 126 个文件；其中 manifest 管理 125 个文件，另一个是 manifest
自身。所有单文件严格小于 50 MiB。源代码基线 commit 为 `79212f1`，冻结 analyzer
SHA-256 为 `bfcbec54b2f101e5970074b1b717e1544d900e0a3beb67656c265dce4630fa94`。
