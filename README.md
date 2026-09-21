# QoS Storage Simulator

模拟NPU的prefill计算、KV读取预取和共享SSU调度。项目保存各项研究的输入、仿真结果、图片、源码与审计记录；教程和原始`data`同时保留。主要实验使用32张NPU，具体配置以各实验目录为准。

## 实验与教程

| 内容 | 入口 |
|---|---|
| 16卡 ASU 长短读取阻塞：data 插值、名义欠载、完整输入与随机控制 | [输入、原生结果与图](results/asu_16npu_data_hol_20260922/README.md) · [独立审计](results/asu_16npu_data_hol_20260922/AUDIT.md) |
| 五组公式 A/B 扩至32卡：6盘欠载对照、4盘容量控制与3盘新候选 | [结果与CDF](results/formula_ab_32npu_20260921/README.md) · [数学说明](results/formula_ab_32npu_20260921/math_notes.md) |
| 保留实验的图片与资料模板；原输入增加 OD 对照 | [图片索引](template/qos_experiments_20260919/index.html) · [模板说明](template/qos_experiments_20260919/README.md) · [ASU/OD/Once 对照](template/qos_experiments_20260919/od_baseline_comparison/README.md) |
| 周六：128K/32K画像，A:B=1:2，Random/Ordered，盘数与策略对照 | [模板中保留的报告](template/qos_experiments_20260919/repository_source/results/baseline_ab128_32_ratio12_20260912/report.md) |
| 历史：接近容量时的低利用率Random输入、输入配比与每层平均b/B | [历史研究报告](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/report.md) · [历史统计](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/comparison.md) · [历史通俗方法](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/report_core.md) · [历史89%/85%目标核对](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/goal_80s_update.md) |
| 多样data画像：间歇过载、持续过载与L3策略对照 | [实验目录](results/diverse_data_ssu3_l3_20260916/) |
| ASU/OD基线：Ring hash、32 NPU、3 SSU、多样画像对照 | [实验说明](results/od_baseline_diverse_ssu3_20260918/README.md) · [基线定义与调用方法](docs/baseline_strategies.md) |
| 文档10画像输入：Random/Ordered的ASU、OD对照及逐事件欠载核验 | [实验报告](results/continuous_underload_asu_od_20260918/README.md) · [文档核对](results/continuous_underload_asu_od_20260918/doc_review.md) |
| OD 固定队深：每盘8192槽、每卡每盘256槽，与原 full24 对照 | [实验报告与等待计时](results/od_fixed_qdepth_full_20260919/README.md) |
| OD 长期低利用率：插值构造、间歇性强过载、Once 对照与验证边界 | [研究报告](results/od_underload_mechanisms_20260919/README.md) · [数学解释](results/od_underload_mechanisms_20260919/math_explanation.md) · [正式结果与独立审计](results/od_underload_mechanisms_20260919/analysis/scheduled_abb_interp_unique_50/README.md) |
| 小白教程、两份手稿与实验解释 | [教程导航](docs/l1_l2_l3_beginner/README.md) · [四页核心PDF](docs/l1_l2_l3_beginner/周六实验与NPU利用率_四页核心版.pdf) |
| KV 全命中为什么仍重算尾部：结合当前 vLLM | [六页 PDF](docs/kv_cache_last_token/KV命中后为什么还要算最后一个token_vLLM图解.pdf) · [说明与来源](docs/kv_cache_last_token/README.md) |
| 所有保留资料的入口 | [results导航](results/README.md) |

## 读数约定

- NPU平均利用率是窗口内真实计算卡时间除以`NPU数×窗口长度`，IO等待留在分母。每卡有待处理任务、每卡窗口内都实际算过两类请求，需要分别核验。
- `B=V/C`是该画像每层参考带宽。完整内部周期的`平均b=V/(C+等待)`包含零接收的时间，因此`平均b/B=C/(C+等待)`；它不代表瞬时利用率，也不能直接预测排队。
- `rho≈1`是整批输入的理想平均需求接近容量，不保证逐盘逐时欠载。SSU真实服务、NPU链路接收和名义需求分开统计。
- 表中的SLO是接纳后prefill完成代理，不含接纳前排队；没有真实首token事件，不能当成端到端TTFT。
- `once`（图中“流量分配策略”）与`new_once`不同。原始data和区间外构造画像分别标注。

## 代码结构与运行

```text
simulator/             独立仿真器
  core/                SSU/NPU、层事件、预取、统计
  policies/            独立策略，每个模块提供 main() 自检
  adapters/            核心与策略之间的状态/执行接口
  api.py               统一运行入口
inputs/                合成、data、manifest 输入与历史实验 runner
tests/                 核心与策略回归测试
results/               保留的实验结果、输入、绘图和审计
template/              保留实验的图片索引与资料包
docs/                  教程与维护记录
trace/                 用户提供的原始 trace 及转换工具
data                   原始请求画像表
```

[仿真器与策略说明](simulator/README.md) · [输入说明和示例](inputs/README.md) · [本次迁移、验证与清理](docs/maintenance_20260919/README.md)

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m simulator.policies.once
python -m inputs --input synthetic --num-npu 4 --num-ssu 3 \
  --requests-per-npu 2 --strategy od_baseline --output /tmp/qos_example_od.json
```

命令在项目根目录运行。新入口拒绝覆盖已有输出；再次运行请换文件名。示例打印的是有限输入的**全程利用率**，不是原实验 warm `[2,4)` 秒利用率。使用 `--input data` 从原始画像构造请求，或使用 `--input manifest --manifest 路径` 读取同一冻结输入比较策略。manifest 中保存的 seed 默认沿用，可用 `--seed` 显式覆盖。

当前基线为 **`asu_baseline`** 和 **`od_baseline`**：前者共用每盘 Path0；后者每盘每个 NPU 独占一个 Path，CIR 均分，允许借用空闲带宽。`once` 是原始“流量分配策略”，与 `new_once` 区分。旧名称 `baseline` 在历史 runner 中保留为 ASU 别名。详见[基线说明](docs/baseline_strategies.md)。

新生成输入统一使用 **Ring hash**。不保留其他 placement 策略。已有实验的输入和图表仍反映原运行时的 placement；严格按旧源码 SHA 回放时，必须使用记录中的源码版本，不能用当前重构后的源码冒充旧版本。详见[placement说明](docs/ring_hash_placement.md)。

旧根目录命令已迁入 `inputs/runners/`，例如：

```bash
python -m inputs.runners.run_shared_path_experiments --help
python -m inputs.runners.run_baseline_npu32_stress --help
```

现存实验 Python 脚本已更新为包导入；旧冻结 JSON、CSV、图片和输入未重写。新运行的源码指纹覆盖真正执行的 `simulator/`、`inputs/` 模块和原始 `data`。完整仿真仍可能花数分钟，绘图和指标分析可直接复用已有结果。并行仿真用独立进程。

默认测试只收集 `tests/`，不遍历整个实验目录。历史图表重绘可能还需要中文字体、ReportLab 或浏览器；测试通过不代表所有 PDF 在任何机器都能直接重建。

## 保留范围与历史恢复

[此前清理时的文件范围](docs/maintenance_20260914/project_manifest_20260914.md) · [对应SHA256清单](docs/maintenance_20260914/project_sha256sums_20260914.txt) · [历史清理记录](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/cleanup_execution.md) · [历史清理依赖与阅读审计](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_random_near_capacity_20260914/cleanup_readability_audit.md)。新增研究以各自目录中的输入与源码校验记录为准。

旧 `baseline_ab128_32_ratio12_20260912`、`baseline_random_near_capacity_20260914` 目录已不在当前 `results/` 下。周六实验保留资料从上述模板进入；未保留在模板中的完整历史资料可在提交 [1c2bb30](https://github.com/chguo0503/qos_storage_sim/tree/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results) 查看。模板的文件范围及缺失项见[完整性说明](template/qos_experiments_20260919/FILE_COMPLETENESS.md)，不把模板视为全部旧结果的完整副本。

`inputs/runners/` 中部分模块沿用旧实验名，是保留实验入口的导入依赖。它们的旧默认命令可能写入旧结果目录，应使用新的输出位置，历史回放则以对应研究里的冻结命令为准。

更早的已入库实验可从提交[38edfa31](https://github.com/chguo0503/qos_storage_sim/tree/38edfa31cb5b61da02f4d198e97dc09d18356ad6)查看。历史完整源码校验可能包含已精简的旧脚本，需使用对应提交或研究保存的源码包在独立目录恢复。原教程与PDF生成审计所对应的文档字节保存在[929102a7](https://github.com/chguo0503/qos_storage_sim/tree/929102a70c2316789003f4969e12a9fc92397f8a)，后续仅修复导航，不重写历史审计。
