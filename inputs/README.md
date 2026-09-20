# 输入与实验入口

`inputs/` 负责构造请求、读取画像和保存输入清单。`simulator/` 接收已经构造好的请求并执行仿真。实验输出继续保存在 `results/`；原始画像 [data](../data) 和用户提供的 [trace/](../trace/) 保留在项目根目录。

## 先运行一个小例子

以下命令在项目根目录运行。示例会完整执行一个有限请求集合；终端中的 NPU 利用率是**从开始到最后请求完成的全程利用率**，不是历史实验的 warm `[2,4)` 秒利用率。

```bash
python -m inputs --input synthetic --num-npu 4 --num-ssu 3 \
  --n-layers 8 --requests-per-npu 2 --strategy od_baseline \
  --output /tmp/qos_synthetic_od.json
```

从原 `data` 的精确画像构造请求：

```bash
python -m inputs --input data --keys 32:256,128:4096 \
  --num-npu 4 --num-ssu 3 --requests-per-npu 2 --seed 7 \
  --strategy once --output /tmp/qos_data_once.json
```

`32:256` 表示总输入 `32 × 1024` token，其中 256 个未命中 token 需要计算；不是命中长度 32K。示例 data 输入从给定画像中随机抽样，不保证每张卡一定抽到每种画像。合成输入使用明确标注的自造参数；两者都不代表真实业务到达分布。

输出路径由调用者指定；新入口拒绝覆盖已有输出。重复运行示例时请换一个文件名。

## 三种输入来源

| 来源 | 入口 | 保留或构造的内容 |
|---|---|---|
| 人工合成 | [synthetic.py](synthetic.py) | 小规模、可重复的两种画像；每卡独立打乱顺序，所有到达时间为 0 |
| 原始画像表 | [from_data.py](from_data.py)、[authenticated.py](authenticated.py) | 精确读取根目录 `data` 中的计算时间和读取量，不插值；请求数量、顺序及到达模式由构造器决定 |
| 冻结请求清单 | [manifest.py](manifest.py) | 保存每个请求的卡号、到达时间、画像和落盘结果；加载时验证完整输入指纹 |

公开入口使用 Ring hash。每个请求按其 ID 和块序号确定落盘，再将该层落盘复用于后续层；不依赖策略执行时临时随机换盘。策略对照应加载同一份清单，而不是分别生成“看起来相似”的请求。

目前通用的共享 Path 入口要求等大的 176 KiB I/O。`from_data.py` 会拒绝不足整块的画像，不会悄悄补齐读取量。历史 stress runner 支持精确尾块，但是否能运行还取决于所选 native/共享 Path 策略的 I/O 限制。

## 画像的单位

根目录 `data` 是 Python 字典文本，由 `ast.literal_eval` 读取。键为 `(总输入K, miss token数)`，值为：

```text
(参考带宽 GiB/s, 每层纯计算时间 us, 源模型纯计算参考时间 ms, 每层读取量 GiB)
```

部分旧记录只有前三项，此时按 `读取量 = 带宽 × 每层计算秒数` 还原第四项。代码中的部分旧字段仍叫 `_gb`、`_gbps`，实际数值单位是 **GiB、GiB/s**。

`authenticated.load_authenticated_bw_table()` 始终返回直接解析的 `data`。若 `results/` 存在可选 NPZ 缓存，必须逐画像与原数据一致才允许继续。普通 [catalog.py](catalog.py) loader 会优先读取缓存；需要认证来源的实验应使用 authenticated loader。

原 `data` 的第三项对应源模型 **78 层**的纯计算参考时间。仿真层数由配置决定，例如 8 层实验的纯计算基准为 `8 × 每层计算时间`，不能直接把源模型第三项当成该实验的 TTFT SLO。

## 冻结输入，再比较策略

```python
from inputs.from_data import build_requests
from inputs.manifest import save_manifest, load_manifest
from simulator.api import run_simulation

requests, source = build_requests(
    num_npu=4, num_ssu=3, requests_per_npu=2, seed=7,
    keys=((32, 256), (128, 4096)),
)
metadata = dict(num_npu=4, num_ssu=3, n_layers=8, seed=7, source=source)
save_manifest("/tmp/qos_input.json.gz", requests, metadata)

requests, metadata = load_manifest("/tmp/qos_input.json.gz")
result = run_simulation(
    requests, strategy="od_baseline", num_npu=metadata["num_npu"],
    num_ssu=metadata["num_ssu"], n_layers=metadata["n_layers"],
    seed=metadata["seed"],
)
```

同一输入可以再交给 `asu_baseline` 或 `once`。原始 Once 在报告中称为“流量分配策略”。`run_simulation()` 返回完整的请求和层时序；warm 窗口、按接纳或到达计时的 SLO，需要在结果上明确统计。

也可以从命令行加载：

```bash
python -m inputs --input manifest --manifest /tmp/qos_input.json.gz \
  --strategy asu_baseline --seed 7 --output /tmp/qos_input_asu.json
```

这里 `--seed` 控制仿真同刻提交顺序；不会重新打乱已经冻结的请求。没有显式指定的 seed、NPU 数、SSU 数、层数从清单 metadata 读取；缺少这些字段时使用入口默认值，因此自建清单应写齐配置。

清单 schema v1 包含：

```text
metadata             实验配置和来源说明
placements           去重后的逐层落盘列表，元素为 (ssu_id, 读取GiB)
requests             request_id、npu_id、arrival_time_ms、load、placement_index
input_fingerprint    请求内容和落盘的完整指纹
```

支持 `.json` 和 `.json.gz`。已有文件的请求指纹相同时保留原文件；不同时拒绝覆盖。指纹验证请求内容，不代替对 metadata、源码哈希及实验参数的核查。部分早期 `results/` 使用逐请求内嵌 `placement` 的其他 JSON 结构，它们由原实验 runner 读取，不应直接当成此 schema v1。

## 历史构造器与 runner

| 模块 | 保留用途 |
|---|---|
| [continuous_prefill.py](continuous_prefill.py) | 多卡 Prefill 请求及固定落盘，包含初始请求和后续到达 |
| [random_steady_state.py](random_steady_state.py) | 分类别分层抽样、类别均匀抽样、画像均匀抽样；三者分布不同 |
| [six_request.py](six_request.py) | 每卡六请求的历史平衡画像实验 |
| [runners/run_baseline_npu32_stress.py](runners/run_baseline_npu32_stress.py) | 固定请求清单、原 data 或合成画像的有限队列压力实验 |
| [runners/run_shared_path_experiments.py](runners/run_shared_path_experiments.py) | 共享 Path 策略、同输入比较及统计 |
| [runners/run_coflow_experiments.py](runners/run_coflow_experiments.py) | 按累计请求字节确定外部到达时刻的历史 coflow 实验 |
| [runners/run_multi_ssu_stall_experiments.py](runners/run_multi_ssu_stall_experiments.py) | 6/7 SSU 的历史负载与等待实验 |
| [runners/run_baseline_4npu_ssu1_low_utilization.py](runners/run_baseline_4npu_ssu1_low_utilization.py) | 4 NPU / 1 SSU 历史案例及图表 |
| `runners/sweep_coflow_*.py` | 独立子进程执行的批量实验 |

这些 runner 保留各自的历史默认值。不能把某个 runner 的默认拓扑、到达模型、统计窗口或 `baseline` 别名直接套到另一个实验。复现时以该实验的配置和原始清单为准。

只构造、描述输入而不仿真：

```bash
python -m inputs.runners.run_baseline_npu32_stress \
  --family raw --raw-keys 32:256,128:4096 --num-npu 4 --num-ssu 3 \
  --horizon-ms 1 --describe-only --manifest-out /tmp/qos_stress_input.json.gz
```

`horizon-ms` 控制该历史构造器的理想计算覆盖长度，并非仿真截止时间；它会加入覆盖余量，最终仍应让所有请求完成。

## 从其他目录调用

在项目根目录可直接使用 `python -m inputs ...`。若要在任意目录使用 `-m`，先在源码 checkout 中执行 `python -m pip install -e .`，或设置 `PYTHONPATH` 为项目根目录。当前根目录 `data` 随源码 checkout 使用，不承诺普通 wheel 安装后自动携带该数据文件。

历史 runner 也支持绝对路径直接执行，例如：

```bash
python /path/to/qos_storage_sim/inputs/runners/run_shared_path_experiments.py --help
```

直接执行会将项目根目录加入导入路径；所有代码仍使用同一套规范包名，避免出现两份 `sim` 模块导致策略 monkeypatch 不生效。并行仿真应使用独立进程，不要在同一个进程的多个线程中同时安装策略 adapter。

## 源码校验和 trace 边界

[provenance.py](provenance.py) 的 `source_files()` 返回根 `data`、`simulator/**/*.py`、`inputs/**/*.py` 的排序相对路径。历史 runner 继续暴露这个接口，新运行会记录真正的包内源码并可保存嵌套快照。迁移前的实验哈希和原始结果保持原样；新源码路径及哈希发生变化是正常的，不能重写旧哈希来宣称代码未变。实验目录自己的扩展策略仍须单独记录源码。

用户提供的 [toolagent trace](../trace/mooncake/tool_agent/toolagent_data_conversion/README.md) 保持原样。其 loader 保留小数 K、零读取画像和原始到达时间；它明确**不是现有仿真调度或请求路由 adapter**。其中计算时间是另一套合成公式，源模型为 32 层，不能当成原 `data` 的实测画像，也不能直接用上面的整数 K authenticated loader 替代。后续接入应明确转换规则并验证 token、字节与到达时刻守恒。
