# 持续欠载输入实验复现说明

## 1. 实验目标

复现 `diverse_data_ssu3_l3_20260916` 中的“持续欠载”输入组，并与同批请求的 Random/Ordered 两种排列进行比较。

这里的“持续欠载”是指：在仿真统计期间，各 SSU 的当前名义带宽需求持续低于其容量上限；它不是低到达率或泊松到达实验。

## 2. 固定系统配置

| 参数 | 值 |
|---|---:|
| NPU 数量 | 32 |
| SSU 数量 | 3 |
| 单 SSU 带宽上限 | 40 GiB/s |
| 请求执行方式 | 每个 NPU 串行执行自己的请求队列 |
| NPU 分配 | 每张卡使用同样的请求画像配比 |
| Random 含义 | 固定配比，各 NPU 独立随机排列 |
| Ordered 含义 | 固定配比，按统一画像顺序排列 |
| 到达模型 | 非泊松；预先生成有限请求集合 |
| 存储放置 | 沿用仿真器的 ring-hash 放置 |
| Baseline 仲裁 | 单 Path FIFO（Path0） |

> 注意：不要把这组输入写成“随机分布到达”。准确说法是“固定配比、随机顺序的有限请求输入”。

## 3. 请求画像

请求画像由两个字段确定：

- `total_tokens_k`：总输入长度，单位为 K tokens；
- `nql`：新增 token 数量，取 2048 或 4096。

命中 token 数量按下式计算：

```text
total_tokens = total_tokens_k × 1024
hit_tokens   = total_tokens - nql
```

每个 NPU 包含 10 种画像，每种画像重复 2 次，共 20 条请求。

| 画像编号 | 总输入长度 | NQL | 命中 token | 每 NPU 数量 |
|---:|---:|---:|---:|---:|
| 1 | 32K | 2048 | 30,720 | 2 |
| 2 | 32K | 4096 | 28,672 | 2 |
| 3 | 64K | 2048 | 63,488 | 2 |
| 4 | 64K | 4096 | 61,440 | 2 |
| 5 | 80K | 2048 | 79,872 | 2 |
| 6 | 80K | 4096 | 77,824 | 2 |
| 7 | 128K | 2048 | 129,024 | 2 |
| 8 | 128K | 4096 | 126,976 | 2 |
| 9 | 160K | 2048 | 161,792 | 2 |
| 10 | 160K | 4096 | 159,744 | 2 |

因此，每个 NPU 的输入统计为：

- 20 条请求；
- 五种总长度各 4 条，各占 20%；
- NQL=2048 与 NQL=4096 各 10 条，各占 50%；
- 总输入量为 `1,856K tokens`；
- 32 张卡合计 640 条请求、`59,392K tokens`。

## 4. 从 `data` 映射画像

对表中每个 `(total_tokens_k, nql)`，从仿真器现有 `data` 中读取对应画像参数，包括：

- 每层 KV 读取量；
- 每层计算时间；
- 层数；
- 理想 TTFT 或计算基线。

必须直接使用 `data` 中的数值，不要只根据 token 数线性估算读取量或计算时间。若 `data` 的 key 使用 K-token 表示，则画像 key 为：

```text
(32, 2048), (32, 4096),
(64, 2048), (64, 4096),
(80, 2048), (80, 4096),
(128, 2048), (128, 4096),
(160, 2048), (160, 4096)
```

运行前应逐项检查这 10 个 key 均存在；缺少画像时应立即报错，不能静默替换成邻近配置。

## 5. 生成可复现的输入 CSV

下面的脚本只生成请求队列，不修改核心仿真器。保存为 `generate_continuous_underload_input.py` 后运行即可。

```python
#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

NPU_COUNT = 32
LENGTHS_K = (32, 64, 80, 128, 160)
NQLS = (2048, 4096)
REPEAT_PER_PROFILE = 2


def base_profiles():
    profiles = []
    for total_k in LENGTHS_K:
        for nql in NQLS:
            profiles.extend([(total_k, nql)] * REPEAT_PER_PROFILE)
    assert len(profiles) == 20
    return profiles


def generate(order: str, seed: int):
    rows = []
    request_id = 0

    for npu_id in range(NPU_COUNT):
        queue = base_profiles()

        if order == "random":
            # 各 NPU 使用独立但可重复生成的随机序列。
            random.Random(seed + npu_id).shuffle(queue)
        elif order == "ordered":
            # canonical order：长度由短到长，同长度下 NQL 由小到大。
            queue.sort(key=lambda x: (x[0], x[1]))
        else:
            raise ValueError(order)

        for sequence, (total_k, nql) in enumerate(queue):
            total_tokens = total_k * 1024
            rows.append({
                "request_id": request_id,
                "npu_id": npu_id,
                "sequence": sequence,
                "total_tokens_k": total_k,
                "total_tokens": total_tokens,
                "nql": nql,
                "hit_tokens": total_tokens - nql,
            })
            request_id += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("random", "ordered"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = generate(args.order, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
```

生成两个输入文件：

```bash
python generate_continuous_underload_input.py \
  --order random --seed 7 \
  --output inputs/continuous_underload_random_seed7.csv

python generate_continuous_underload_input.py \
  --order ordered --seed 7 \
  --output inputs/continuous_underload_ordered.csv
```

`seed=7` 用于生成一个确定性、可重复的 Random 输入。如果原实验目录中已有归档请求 CSV，应优先使用归档 CSV，以复现完全相同的逐卡顺序；上述脚本保证画像和配比等价，但只有 seed 规则一致时才保证排列逐条相同。

## 6. 接入核心仿真器

读取 CSV 后，按以下规则构造仿真请求：

1. 根据 `npu_id` 将请求放入对应 NPU 队列；
2. 按 `sequence` 升序执行；
3. 使用 `(total_tokens_k, nql)` 查询 `data` 中的请求画像；
4. 使用画像内原始的逐层读取量、逐层计算时间和层数；
5. Random 与 Ordered 必须使用完全相同的 640 条请求，只允许队列顺序不同；
6. 两种顺序必须使用相同 placement、Path、带宽、层数和统计口径；
7. 不要为 Random 重新采样画像，也不要为不同策略重新生成请求集合。

伪代码如下：

```python
for row in csv_rows:
    profile_key = (row.total_tokens_k, row.nql)
    assert profile_key in data
    profile = data[profile_key]

    request = Request(
        request_id=row.request_id,
        npu_id=row.npu_id,
        sequence=row.sequence,
        profile=profile,
    )
    npu_queues[row.npu_id].append(request)

for queue in npu_queues.values():
    queue.sort(key=lambda req: req.sequence)
```

## 7. 欠载校验

仅检查整个运行的平均带宽不够。必须在事件边界或足够细的时间片上验证每个 SSU：

```math
D_s(t)=\sum_{i\in A(t)}\frac{V_{i,s}}{C_i}<40\ \text{GiB/s}
```

其中：

- `A(t)`：时刻 `t` 正在执行的请求集合；
- `V_{i,s}`：请求 `i` 每层从 SSU `s` 读取的数据量；
- `C_i`：请求 `i` 的单层计算时间；
- `D_s(t)`：SSU `s` 在该时刻的名义带宽需求。

建议在每次请求 admission、完成或当前请求切换时重新计算，而不是只用 100 ms 平均曲线判断。

参考校验代码：

```python
for event_time in sorted(all_request_switch_times):
    for ssu_id in range(3):
        demand = sum(
            current_req[npu].read_gib_per_layer[ssu_id]
            / current_req[npu].compute_s_per_layer
            for npu in range(32)
            if current_req[npu] is not None
        )
        assert demand < 40.0, (event_time, ssu_id, demand)
```

这里的 `read_gib_per_layer[ssu_id]` 必须来自实际 placement 后落到该 SSU 的数据量。不能把全请求读取量直接重复计入每个 SSU。

## 8. 运行与统计要求

至少分别运行：

```text
Baseline FIFO + Random
Baseline FIFO + Ordered
```

若同时比较 Once，则继续运行：

```text
Once + Random
Once + Ordered
```

每组应导出：

- 整机 NPU 利用率；
- 逐 NPU 利用率；
- 总体 TTFT；
- TTFT SLO ×1 与 ×1.5 达标率；
- 按总长度或请求类别拆分的 TTFT/SLO；
- 每个 SSU 的需求带宽和实际供给带宽；
- 每个时间区间的欠载校验结果。

Random 与 Ordered 应使用相同统计窗口。若使用 warm 窗口，应在结果中同时记录：

```text
warm_start_s
warm_end_s
统计窗口内接纳的请求数
统计窗口内完成的请求数
```

不要只写“warm=2s”，必须明确统计区间，例如 `[2s, 8s)`。

## 9. 复现前验收清单

- [ ] 32 个 NPU、3 个 SSU、每盘 40 GiB/s；
- [ ] 每个 NPU 恰好 20 条请求；
- [ ] 每种 `(长度, NQL)` 画像每卡恰好 2 条；
- [ ] 全局共 640 条请求；
- [ ] Random 与 Ordered 的请求多重集合完全相同；
- [ ] 10 种画像全部从 `data` 成功加载；
- [ ] 各 NPU 的 Random 顺序独立打乱；
- [ ] Baseline 使用 Path0 FIFO；
- [ ] placement、策略参数和统计窗口保持一致；
- [ ] 对每个 SSU 做逐事件欠载检查，需求始终小于 40 GiB/s；
- [ ] 保存输入 CSV、运行配置、seed、代码版本和结果目录。

## 10. 与其他实验的区别

不要与以下实验混用：

1. **9 月 14 日 A/B 全程欠载实验**：8 NPU、1 SSU，使用 20K/200K 等两类请求；
2. **间歇过载**：每卡 30 条请求，部分时间超过容量；
3. **持续过载**：每卡 42 条请求，统计区间长期超过容量；
4. **早期 16K/160K 欠载实验**：32 NPU、6 SSU，不是本说明中的 3 SSU 多画像输入。

本说明唯一对应的是：**32 NPU、3 SSU、五种长度 × 两种 NQL、每画像每卡两条、每卡共 20 条的固定配比输入。**
