# Final-source timeline observer 不扰动验证

结论：硬门槛通过。在 `legacy32 / baseline / SSU=3 / seed=42` 的 8 秒测量回归中，开启 timeline observer 没有改变仿真轨迹。最终源码指纹是：

```text
76658d5aac81f11a8f9fddcfeaed6692821aefde3892cdc0d7dd564563242568
```

剔除完整、明确列出的 observer-only 字段后，两边 `steady_summary` 严格对象相等，规范化 SHA256 同为：

```text
c92db3680a22c67ba331b0db348083bc7f153646b7da5577481e08422e93b7a3
```

再剔除 12 项不可避免的执行元数据或由 observer 配置派生的 fingerprint 后，完整科学 artifact 也严格对象相等，SHA256 同为：

```text
eae58c99b99732952651df432997f3b75a0621f5261cdb98d639600a4164ca87
```

这次回归实际处理了 `57,196,749` 个离散事件，因此证据不只是“最终平均值接近”。但它仍是这个配置和 seed 的严格回归证明，不是覆盖所有策略、SSU 和 seed 的数学证明。

两份未入库原始 artifact 的 SHA256 分别为 `f1badf1e4974e82c276507169f9dec7e442579e29893feab7390f564edb7038e`（off）和 `8ef93bea2c74afabbc4443ed3e72156a8df4e11f17ecaa6b0a1e2a8f62ff9477`（on），便于对私有原始文件复核。

## 配置与身份

| 项目 | 值 |
|---|---:|
| definition / case | `legacy32 / baseline` |
| NPU / SSU | 32 / 3 |
| layers / batch size | 16 / 1 |
| seed | 42 |
| backing requests per NPU | 256 |
| warmup requests per NPU | 8 |
| settle | 500 ms |
| measurement / block | 8,000 ms / 500 ms |
| observer off | `timeline_diagnostics=false` |
| observer on | `timeline_diagnostics=true`, probe 50 ms, limit 10,000 |
| selected run status | 两边均为 `selected_complete=true`, result `status=ok` |

两次运行的源码在运行前后均保持稳定；definition、输入、materialized prefix、workload statistics 和 measurement cohort 指纹也完全相同。四个核心文件的 SHA256 为：

- `sim.py`: `5f3a9bb82c5740a86c9c916e01495326a0bf4363fb6a5312cb99ac747e8600fd`
- `continuous_batch_sim.py`: `ae5c37aa6733e81fcc0e624404a89a154594448712a9b1ff94c39a2ba36c162e`
- `ms_scale_control_experiment.py`: `a6ffbe5aa45e3116356ff92deae5ab59ca705d5551aa804b7f5496bb3ef3fb6b`
- `adaptive_admission_scheme_b_v2_1.py`: `ee39343a2aee2b9901d12c5c043e67b46f7753e0e54b72bb7092fcf322909f97`

## 严格比较方法

第一阶段只删除 observer 新增内容：

- 任意名为 `timeline` 的 mapping 字段；
- 任意以 `timeline_` 开头的字段；
- request row 的 `timeline_layers`、`io_barrier_ms`、`ttft_accounting_error_ms`；
- observer-only invariant `request_ttft_compute_barrier_decomposition`。

这里特别补全了三个不以 `timeline_` 开头的诊断字段。schema 差异也单独审核过：每条 on-side request 恰好多出上述三个 request 字段；每个 on-side boundary 恰好多出 `timeline`；on-side invariants 恰好多出 21 个 `timeline_` 检查和一个 request TTFT 分解检查，没有漏删或宽泛删除业务字段。

第一阶段之后，完整 artifact 仍有且仅有 12 个递归差异，全部属于四类：observer 配置派生的 config/case fingerprint、创建时间、driver/worker 的进程与内存观测、worker wall time。第二阶段仅删除这 12 项后，完整 artifact 严格相等。没有使用浮点容差，也没有删除任何仿真结果。

规范化 JSON 统一采用 UTF-8、字典键排序、紧凑分隔符 `(',', ':')` 和 `ensure_ascii=false`。

## 逐层 exact-equal 证据

| 层次 | 数量/形状 | 结论 | SHA256 |
|---|---:|---|---|
| 完整 steady summary | 1 | 全部科学字段 exact-equal | `c92db3680a22c67ba331b0db348083bc7f153646b7da5577481e08422e93b7a3` |
| request rows 核心字段 | 353 | 逐请求 exact-equal | `d38b06f27ba693cfaa78f862ef423e942948e0975547734b48eba7b90c436538` |
| 0.5 秒 measurement blocks | 16 | 完整旧 block 对象 exact-equal | `576164d829d129ce0cbc4ada211d3250fe4f823364d503b3641640b5c063e365` |
| stationarity boundaries 旧字段 | 17 | 删除 on-side `timeline` 后 exact-equal | `6a1d0462417fa4c76e5eab452bccc4f4855d54783d2be7c1991cb42a6a9e7459` |
| pre-instrumentation invariants | 29 | 全部相同且为 true | `52970d1767344a20710f582188dc05700de3b2d21a5838d95a2a6bcde91067db` |
| NPU utilization | 32 | 逐元素 exact-equal | `2aa3fa684ee787bd5a4f49fbcf011ff971b88920e23eca179aaba5b5233f1a0f` |
| NPU compute time | 32 | 逐元素 exact-equal | `e6453ad782596623c69ac11505a41eaf80cb99c80907c8c782d538025021c8c0` |
| NPU×SSU SSD service | 32×3 | GB 和 GB/s 均逐元素 exact-equal | `458403307beaad80b9281b34a19a3388ea49bc8a8840565f00b67e4197d5cad8` |
| NPU×SSU link service | 32×3 | GB 和 GB/s 均逐元素 exact-equal | `3eb39a8d876c2ea56159efa3a01c0f08c0d5e2c4db3de8e92b758972fcdbf003` |

两边相同的关键结果包括：

- measurement start/end：`10797.499240789299 / 18797.4992407893 ms`；
- drain stop：`20767.34298871763 ms`；
- mean NPU utilization：`0.5790995496872885`；
- TTFT SLO：`0.40616198038073037`；
- mean / p99 TTFT：`734.5470947671338 / 2284.708109817942 ms`；
- measurement request count：353；
- events processed：两边均为 57,196,749。

这直接排除了“observer 改变了请求完成顺序、0.5 秒窗口、逐 NPU compute、逐 NPU×SSU 服务量，最后恰好平均值接近”的可能性。

## Observer 自校验

最终源码的 24 项 observer 检查全部通过，覆盖：snapshot shape/nonnegative、独立服务和排队归因、I/O stage 守恒、remaining-work bounds、累计量单调、SSD/link 队列守恒、compute inventory 守恒、request TTFT 分解、全窗和逐 block 状态时长分解、measurement-start carry-in 的定义/身份/逐层闭合，以及 dispatch replay。

- dispatch probe 达到配置上限：10,000 条；预测 Path 与实际调度 Path 为 `10,000/10,000`，mismatch 为 0，故 `truncated=true`；
- route probe 共 99 条，未达到上限，`truncated=false`；
- 353 条 request 的 `TTFT = ideal compute + I/O barrier` accounting error 最大为 `2.831e-11 ms`，均不超过 `1e-7 ms`。

## 观测开销

两次 single-worker 运行并行执行，因此这里只报告观测值，不把它解释成严格性能 benchmark：

- wall time：off `278.1987 s`，on `291.6058 s`，增加约 `4.8193%`；
- worker peak RSS：off `1,909,141,504 bytes`，on `1,921,343,488 bytes`，增加 `12,201,984 bytes`，约 `11.6367 MiB` 或 `0.6391%`。

仓库只保留这份 sanitized 派生证据；源结果文件、机器/进程标识值和私有绝对路径均未写入本报告或配套 JSON。

## 可复现命令模板

关闭 observer：

```bash
python3 ms_scale_control_experiment.py --definition legacy32 --output observer_off.json --case baseline --ssu 3 --max-workers 1 --seed 42 --requests-per-npu 256 --warmup-requests 8 --settle-ms 500 --measurement-ms 8000 --block-ms 500 --fresh
```

开启 observer：

```bash
python3 ms_scale_control_experiment.py --definition legacy32 --output observer_on.json --case baseline --ssu 3 --max-workers 1 --seed 42 --requests-per-npu 256 --warmup-requests 8 --settle-ms 500 --measurement-ms 8000 --block-ms 500 --timeline-diagnostics --timeline-dispatch-probe-ms 50 --timeline-dispatch-probe-limit 10000 --fresh
```
