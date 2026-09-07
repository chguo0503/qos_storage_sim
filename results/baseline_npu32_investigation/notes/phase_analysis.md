# 32卡强反例：随机启动相位复核

2026-09-07。4/4个补充case全部完成，0个仿真失败。每个case使用7,209,024条物理IO、24024个请求；4次仿真的8个固定窗口均32卡全active。

**随机错开启动能缓解Baseline，但没有消除该输入下的FIFO损失。** 未错开时全卡U约52.21%；按每卡0–10ms随机offset启动后，两种offset种子均约75%。相同随机启动输入上的Once（5ms collector）达到95.00%–96.32%，仍比Baseline高20.03–21.32个百分点。

## 唯一的输入干预

源文件固定为 `screen/inputs/strong32_stripe.json.gz`。32 NPU、8 SSU，每盘40 GiB/s；24张卡固定为1K/NQL128，8张卡固定为192K/NQL256。原24024个请求的request_id、NPU绑定、画像、计算时间、精确块尺寸、SSU placement、请求数、horizon=4000ms全部保留。每盘名义ρ=0.9060869284，容量条件可行，compute_scale不变。1K短画像仍是外推构造，不是data原始行。

对jitter种子7和123，依NPU ID顺序用 `random.Random(jitter_seed).uniform(0,10)` 生成32个offset。某卡的**全部**请求统一在其offset到达；不是给每个请求逐次增加随机抖动。只改manifest的arrival字段、load中的arrival_time/arrival_ms及对应metadata/fingerprint。

提交顺序的原生PRNG种子始终保留42；offset RNG种子7/123是另外一个参数。两份新输入的offset范围分别0.374957–9.762551ms和0.016706–9.111505ms，全部具体值保存在spec及phase_metrics.json。

两策略都使用现成coflow wrapper：真实5ms周期collector、相同静态CIR、无运行期CIR写。这里的Once与4卡格点的Once native（TTL=0）信息条件不同。

## 两个窗口的完整结果

| 启动输入 | 窗口/s | Baseline U% | Once U% | Once−Baseline/pp |
|---|---|---:|---:|---:|
| 原始同刻到达（参考） | [1,2] | 52.210 | 94.466 | +42.256 |
| 原始同刻到达（参考） | [2,3] | 52.216 | 93.571 | +41.355 |
| offset seed7 | [1,2] | 75.016 | 95.321 | +20.305 |
| offset seed7 | [2,3] | 74.976 | 95.005 | +20.029 |
| offset seed123 | [1,2] | 75.003 | 96.321 | +21.317 |
| offset seed123 | [2,3] | 74.976 | 95.364 | +20.388 |

每行Baseline与Once严格共用同一份完整新manifest；原始同刻到达行引用已保存screen结果。**不同启动行的arrival及input_fingerprint不同**，它们属于输入干预对照，不能称为跨行相同输入。

两个随机启动baseline的第二窗口U同为74.975696%，24张短卡均值约66.634261%，8张长卡100%。这说明低利用率在启动结束约2秒后仍存在，且不是某几张卡没有请求。不能由两个offset种子推导任意启动分布都会得到同值，也不能断言已经证明任意系统的稳态。

| Offset seed | 策略 | 短卡U [1,2]s/% | 长卡U [1,2]s/% | 整批U% | 完整makespan/ms |
|---:|---|---:|---:|---:|---:|
| 7 | baseline | 66.688 | 100.000 | 74.917 | 5542.480 |
| 7 | once | 93.762 | 100.000 | 93.329 | 4449.071 |
| 123 | baseline | 66.671 | 100.000 | 74.920 | 5542.299 |
| 123 | once | 95.094 | 100.000 | 94.906 | 4375.157 |

本补充case中Once同时改善完整排空时间：seed7从5542.480降到4449.071ms（缩短19.73%），seed123从5542.299降到4375.157ms（缩短21.06%）。这里有同输入完整makespan证据，不只是中间一秒变好。

## 解释范围

- 同刻启动确实放大了原强反例：Baseline从52%到75%的差异必须承认。
- 一次随机错开启动后，Baseline仍比同输入Once低约20个百分点；因此该强输入的调度差距不只存在于“所有卡t=0同时发起”这一初始条件。
- 这里只检验同一固定画像、条带化8盘、两种一次性启动抖动。持续变化画像、逐请求到达抖动、其他SSU数、其他抖动幅度需要另测；不能把当前结果泛化到生产分布。
- 全部请求仍构成每卡一次性有限backlog，arrival SLO主要包含人工构造的入口排队。本报告以固定窗口I/O stall和同manifest完整makespan作主要比较。

## 完整性与中断记录

首次编排在最后180秒进度记录后收到SIGTERM，返回143，两子进程一同结束；当时没有任何完成结果或Python失败trace。父agent确认没有用户或父agent停止请求，具体信号来源仍未知。原日志、状态、环境、命令及说明保存在 `phase/interrupted_attempt_1/`，未删除或挑选其运行结果。

随后仅针对本study使用nohup/setsid重启，相同已冻结manifest不变。重启后的4个case全部完成，编排elapsed1085.3秒；两条baseline分别207.3/209.2秒，两条Once分别673.2/649.4秒。运行结束和独立复核均确认根目录源码SHA256没有改变，重启日志无traceback。

## 复算与产物

自包含生成和运行（已有结果会进行指纹检查后复用）：

```bash
python results/baseline_npu32_investigation/phase/run_phase_robustness.py --workers 2
```

只复核和汇总：

```bash
python results/baseline_npu32_investigation/phase/aggregate_phase.py
```

- `phase/inputs/strong32_stripe_jitter_seed{7,123}.json.gz`：完整新输入，`.spec.json`记录32个offset及源文件SHA。
- `phase/runs/<input>/{baseline,once}/`：全部4个完整压缩结果、命令及过程日志。
- `phase/phase_metrics.csv`：8行策略/窗口指标，含短长卡、整批U、makespan和输入fingerprint。
- `phase/phase_metrics.json`：逐请求干预复核、完整offset、结果SHA、core/runner源码SHA、原始同步输入参考结果。
- `phase/phase_status.json`：4/4、0失败。`phase/sweep_restart.log`保存结束日志，`phase/phase_environment.json`保存环境。

汇总脚本检查所有24024请求仅arrival字段改变，其他load字段和有序placement逐请求相等；同offset种子两个策略指纹相等、submit_seed=42、collector=5ms；直接从全部layer时间线重新计算窗口U并核对每卡active=1000ms。CSV的utilization字段为0–1，本文乘100。
