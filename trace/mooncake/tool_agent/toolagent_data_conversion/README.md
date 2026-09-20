# ToolAgent trace → qos_storage_sim data

本转换将 `QiliangLi/status_aware_network/mooncake_trace/toolagent_trace.jsonl` 转成 `chguo0503/qos_storage_sim/data` 的字典结构。保留全部 23,608 条请求；按 `(input_length, u_tokens)` 去重后得到 11,672 种画像。

**这是 synthetic-v1 合成画像转换，计算时间不是 ToolAgent 实测值，也不是原 data 的 GLM 画像。** 原始 trace 没有计算时间；本转换沿用上一项目已定义的时间公式，不对原 data 做插值或外推。

## 文件

| 文件 | 用途 |
|---|---|
| `data_toolagent` | 与 data 相同的 Python 字典文本，可用 `ast.literal_eval` 读取 |
| `toolagent_requests.jsonl` | 全部请求、原始到达时间、画像引用、hash_ids、命中字段与合成画像值 |
| `convert_toolagent.py` | 标准库转换脚本，从原始 trace 重建命中并输出画像 |
| `load_toolagent.py` | 显式加载转换文件，保留小数 K 和零读取画像 |
| `conversion_report.json` | 输入哈希、参数、数量统计、守恒校验与派生字段对照结果 |
| `validation_independent.json` | 全部画像对照源项目 Fraction 公式，并通过目标核心 loader/token partition 校验 |
| `source/toolagent_trace.jsonl` | 本次转换的原始输入 |

## 字段映射与单位

```python
{
    (input_length / 1024, u_tokens): (
        required_bw_GiB_per_s,
        per_layer_compute_us,
        source_compute_ttft_ms,
        per_layer_kv_GiB,
    ),
    ...
}
```

| 目标字段 | 转换规则 |
|---|---|
| `seq_len_k` | `input_length / 1024`，保留小数，不向上/向下取整 |
| `nql` | `u_tokens`，即未命中、需要计算的 token；不是输出 token 数 |
| `per_layer_us` | `c * 1e6` |
| `per_layer_kv_gb` | `hit_tokens * 4096 / 2**30`，实际数值单位 GiB |
| `required_bw_input_gbps` | `per_layer_kv_GiB / c`，实际数值单位 GiB/s |
| 第三个 tuple 值 | `32 * c * 1000`，源模型 32 层的纯计算参考时间，单位 ms |

原 data 的 `_gb` / `_gbps` 是历史字段名，实际数值采用 GiB / GiB/s。本转换显式将源项目的字节量转换为 GiB，不能直接复制其十进制 GB 数值。

原 data 的第三字段满足 `78 * per_layer_us / 1000`，源画像对应 78 层。本转换采用源 synthetic-v1 的 **32 层**，第三字段为 `32 * per_layer_us / 1000`；不虚构成原 data 的 78 层模型。仿真使用多少层仍由仿真配置决定，读取此文件不会自动修改层数。

## 计算画像

令 `h=hit_tokens`、`u=u_tokens`，每条请求单独构成 batch：

```text
eta = min(1, max(1/16, u/512))
A = u*h + u*(u+1)/2
c = 50e-6 + 2e-7*u/eta + 1e-10*A      # 秒/层
KV_bytes_per_layer = 4096*h
```

该转换保留请求的精确 token 数，不把请求吸附到原 data 的 32K–200K 或 NQL 离散网格。尤其不强行把 NQL 改成 512 的倍数。

这是 **singleton 画像表**。如果未来让多个请求联合组成 batch，源模型应按整个 batch 的 `N=Σu` 和 `A=Σ[u*h+u*(u+1)/2]` 重新计算；直接相加本表的 c 无法重现源项目的合批加速。

## 命中与时间

命中沿用源项目的 first_seen 模型：按 `(timestamp, 原始行号)` 稳定排序；连续前缀遇到首个未见 hash 即停止；同请求内重复块、同 timestamp 的其他请求不产生新增可见命中；同 timestamp 组处理完后才更新 seen。尾块参与，至少保留一个 token 计算，所以 `h <= input_length - 1`、`u >= 1`。

该命中是离线假设，不包含缓存容量、淘汰或实际计算/写入完成时间。若提供 `--derived`，脚本会与上游派生文件逐条对照。

`toolagent_requests.jsonl` 保留原始 `timestamp`，`arrival_time` 与 `arrival_time_ms` 同为毫秒，`request_id` 按排序后的请求从 0 编号；没有缩放、泊松替换、随机打乱或全部归零。`output_length` 和 `hash_ids` 保留为原始信息，但 output_length 不参与 Prefill 时间计算。

逐请求清单另存源项目参考时间：

```text
r = h * 4096 / (80 * 10**9)          # 秒；源参考带宽80 GB/s为十进制
T0 = r + 32*c + 31*max(0, r-c)
source_T0_ms = T0 * 1000
source_slo4_ms = 4 * T0 * 1000
```

这里 `T0` 包含空系统单请求的暴露读取时间，不能与 data 的第三字段 `32*c` 混淆。实际共享存储下的 TTFT、排队和利用率尚未运行仿真。

## 使用

从本包原始 trace 重新生成：

```bash
python convert_toolagent.py --input source/toolagent_trace.jsonl --output-dir regenerated
```

用 Python 直接读画像：

```python
import ast
from pathlib import Path

table = ast.literal_eval(Path("data_toolagent").read_text(encoding="utf-8"))
key = (13544 / 1024, 13032)
bw_gib_s, c_us, compute32_ms, kv_gib = table[key]
```

该例 key 为 `(13.2265625, 13032)`：h=512，单层计算 11815.9412 μs，单层读取 0.001953125 GiB，32 层纯计算 378.1101184 ms。

## 接入现有仿真器的边界

画像格式转换已经完成，但**原 trace 回放还需要把逐请求清单接入到达队列，并明确 NPU 分配规则**。本转换不擅自选择 NPU 数、路由、盘数、策略或实验层数。

| 现有入口 | 注意事项 |
|---|---|
| `sim.load_bw_table_cache` | 核心解析保留小数 K；但优先读取旧 NPZ 缓存，建议显式读取本文件 |
| `sim.calculate_token_partition` | `round(seq_len_k*1024)` 可精确恢复本文件的输入 token 数 |
| `authenticated_workload_inputs.load_authenticated_bw_table` | 会把 K 强转整数，并拒绝 0 带宽/0 KV；不能直接用于本转换文件 |
| 部分历史实验脚本 | 会把 K 强转整数，或拒绝零命中请求，接入前需改相应入口 |
| `continuous_batch_sim._request_profile_metadata` | 会将画像键转成整数，需修正才能精确保存小数 K 标签 |
| 随机画像采样器 | 从 11,672 种画像均匀抽样会改变原 trace 的频率与顺序，回放必须使用 23,608 条请求清单 |
| 现有按 request_id 的 ring hash 放置 | 不会自动使用保留的原 hash_ids 复用关系，若需要按 hash 身份放置须另接入 |

核心模拟器本身可处理零命中请求：无需 SSD 读取，直接开始该层计算。转换不能为了绕过某个入口的校验而丢弃这些请求。

原 data 使用约 1408 Bytes/token/layer 的 KV 画像，本文件使用源模型的 4096 Bytes/token/layer；二者的计算公式与层数也不同。不能把两组策略结果的变化全部归因于输入 trace 差异。

## 来源

- 源仓库读取版本：`240a882862fbea77e42207e4860f3fd1494faf3e`。
- [原始 ToolAgent trace](https://github.com/QiliangLi/status_aware_network/blob/240a882862fbea77e42207e4860f3fd1494faf3e/mooncake_trace/toolagent_trace.jsonl)
- [源命中算法](https://github.com/QiliangLi/status_aware_network/blob/240a882862fbea77e42207e4860f3fd1494faf3e/tools/cq_derive_hits.py)
- [源计算画像](https://github.com/QiliangLi/status_aware_network/blob/240a882862fbea77e42207e4860f3fd1494faf3e/sim/cq/profile.py) · [参数](https://github.com/QiliangLi/status_aware_network/blob/240a882862fbea77e42207e4860f3fd1494faf3e/sim/cq/config.py)
- [目标 data](https://github.com/chguo0503/qos_storage_sim/blob/main/data) · [目标核心加载代码](https://github.com/chguo0503/qos_storage_sim/blob/main/sim.py)
- [目标单位、78层与KV大小说明](https://github.com/chguo0503/qos_storage_sim/blob/main/run_baseline_4npu_ssu1_low_utilization.py)

本次只制作独立转换文件，没有覆盖任何仓库原 data，也没有修改远端代码。
