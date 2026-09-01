# Final-source timeline observer 不扰动验证

结论：硬门槛通过。在 `legacy32 / baseline / SSU=4 / seed=42` 的 8 秒测量回归中，开启 timeline observer 没有改变仿真轨迹。最终源码指纹是：

```text
88161e7a08d573d0ceea29e29f8a456136ee39709800d7e6e34b0e4cfc8a6ced
```

剔除完整、明确列出的 observer-only 字段后，两边 `steady_summary` 严格对象相等，规范化 SHA256 同为：

```text
35e5ab3a13d55caa1435563193fc2097a727b9a5a28c9459c90cb9c01082ebb2
```

再剔除 12 项不可避免的执行元数据或由 observer 配置派生的 fingerprint 后，完整科学 artifact 也严格对象相等，SHA256 同为：

```text
ff8e48c86e146b477211f9b22355e8b201024f70c3d04d8df85515e499f3481a
```

这次回归实际处理了 `73,711,592` 个离散事件，因此证据不只是“最终平均值接近”。但它仍是这个配置和 seed 的严格回归证明，不是覆盖所有策略、SSU 和 seed 的数学证明。

## 配置与身份

| 项目 | 值 |
|---|---:|
| definition / case | `legacy32 / baseline` |
| NPU / SSU | 32 / 4 |
| layers / batch size | 16 / 1 |
| seed | 42 |
| backing requests per NPU | 64 |
| warmup requests per NPU | 8 |
| settle | 500 ms |
| measurement / block | 8,000 ms / 500 ms |
| observer off | `timeline_diagnostics=false` |
| observer on | `timeline_diagnostics=true`, probe 5 ms, limit 500 |
| selected run status | 两边均为 `selected_complete=true`, result `status=ok` |

两次运行的源码在运行前后均保持稳定；definition、输入、materialized prefix、workload statistics 和 measurement cohort 指纹也完全相同。三个核心文件的 SHA256 为：

- `continuous_batch_sim.py`: `f57719541dbad034778388b6271d5a011e9c05e68ccf1608e38d9b5ec65cef0c`
- `ms_scale_control_experiment.py`: `a6ffbe5aa45e3116356ff92deae5ab59ca705d5551aa804b7f5496bb3ef3fb6b`
- `adaptive_admission_scheme_b_v2_1.py`: `ee39343a2aee2b9901d12c5c043e67b46f7753e0e54b72bb7092fcf322909f97`

## 严格比较方法

第一阶段只删除 observer 新增内容：

- 任意名为 `timeline` 的 mapping 字段；
- 任意以 `timeline_` 开头的字段；
- request row 的 `timeline_layers`、`io_barrier_ms`、`ttft_accounting_error_ms`；
- observer-only invariant `request_ttft_compute_barrier_decomposition`。

这里特别补全了三个不以 `timeline_` 开头的诊断字段。schema 差异也单独审核过：每条 on-side request 恰好多出上述三个 request 字段；每个 on-side boundary 恰好多出 `timeline`；on-side invariants 恰好多出 15 个诊断检查，没有漏删或宽泛删除业务字段。

第一阶段之后，完整 artifact 仍有且仅有 12 个递归差异，全部属于四类：observer 配置派生的 config/case fingerprint、创建时间、driver/worker 的进程与内存观测、worker wall time。第二阶段仅删除这 12 项后，完整 artifact 严格相等。没有使用浮点容差，也没有删除任何仿真结果。

规范化 JSON 统一采用 UTF-8、字典键排序、紧凑分隔符 `(',', ':')` 和 `ensure_ascii=false`。

## 逐层 exact-equal 证据

| 层次 | 数量/形状 | 结论 | SHA256 |
|---|---:|---|---|
| 完整 steady summary | 1 | 全部科学字段 exact-equal | `35e5ab3a13d55caa1435563193fc2097a727b9a5a28c9459c90cb9c01082ebb2` |
| request rows 核心字段 | 471 | 逐请求 exact-equal | `f2889e58fb90e766f86fc176e2cd105ebc94e64539b834674f1410f067b1500a` |
| 0.5 秒 measurement blocks | 16 | 完整旧 block 对象 exact-equal | `18202541efe001c305b622fda852a2a97515190a78b3967a4c4b42b1e03e2007` |
| stationarity boundaries 旧字段 | 17 | 删除 on-side `timeline` 后 exact-equal | `a0986589bbc56db32734610789a29046800b616a5796abfc1f1c870c54c68f07` |
| pre-instrumentation invariants | 29 | 全部相同且为 true | `52970d1767344a20710f582188dc05700de3b2d21a5838d95a2a6bcde91067db` |
| NPU utilization | 32 | 逐元素 exact-equal | `5c84a92a2f4935866af9a9faa775ab4a569a320b1cbdb6b3c53e5292babae8c9` |
| NPU compute time | 32 | 逐元素 exact-equal | `e799bbf33c10adf9417c8243d5d28b960fbfdd21280504bf9a73bb43590ed999` |
| NPU×SSU SSD service | 32×4 | GB 和 GB/s 均逐元素 exact-equal | `405ea73283e3ad378c1de8eaa79f26d5490d5e882d5002ec4bf5123a0a98e4e5` |
| NPU×SSU link service | 32×4 | GB 和 GB/s 均逐元素 exact-equal | `5b0cbdd2f66af5b7adc6004366f7c6006e573caea228764b587f7f48c7836c4f` |

两边相同的关键结果包括：

- measurement start/end：`10136.618340562687 / 18136.618340562687 ms`；
- drain stop：`20278.423522693287 ms`；
- mean NPU utilization：`0.7863877341361004`；
- TTFT SLO：`0.5639746542211228`；
- mean / p99 TTFT：`546.8147936050225 / 2215.4127624296416 ms`；
- measurement request count：471；
- events processed：两边均为 73,711,592。

这直接排除了“observer 改变了请求完成顺序、0.5 秒窗口、逐 NPU compute、逐 NPU×SSU 服务量，最后恰好平均值接近”的可能性。

## Observer 自校验

最终源码的 17 项 observer 检查全部通过，覆盖：snapshot shape/nonnegative、独立服务和排队归因、I/O stage 守恒、remaining-work bounds、累计量单调、SSD/link 队列守恒、compute inventory 守恒、request TTFT 分解、全窗和逐 block 状态时长分解，以及 dispatch replay。

- dispatch probe 达到配置上限：500 条；预测 Path 与实际调度 Path 为 `500/500`，mismatch 为 0，故 `truncated=true`；
- route probe 共 16 条，未达到上限，`truncated=false`；
- 471 条 request 的 `TTFT = ideal compute + I/O barrier` accounting error 均不超过 `1e-7 ms`。

## 观测开销

两次 single-worker 运行并行执行，因此这里只报告观测值，不把它解释成严格性能 benchmark：

- wall time：off `294.5564 s`，on `303.5786 s`，增加约 `3.0630%`；
- worker peak RSS：off `969,256,960 bytes`，on `974,393,344 bytes`，增加 `5,136,384 bytes`，约 `4.8984 MiB` 或 `0.5299%`。

仓库只保留这份 sanitized 派生证据；源结果文件、机器/进程标识值和私有绝对路径均未写入本报告或配套 JSON。

## 可复现命令模板

关闭 observer：

```bash
python3 ms_scale_control_experiment.py --definition legacy32 --output observer_off.json --case baseline --ssu 4 --max-workers 1 --seed 42 --requests-per-npu 64 --warmup-requests 8 --settle-ms 500 --measurement-ms 8000 --block-ms 500 --fresh
```

开启 observer：

```bash
python3 ms_scale_control_experiment.py --definition legacy32 --output observer_on.json --case baseline --ssu 4 --max-workers 1 --seed 42 --requests-per-npu 64 --warmup-requests 8 --settle-ms 500 --measurement-ms 8000 --block-ms 500 --timeline-diagnostics --timeline-dispatch-probe-ms 5 --timeline-dispatch-probe-limit 500 --fresh
```
