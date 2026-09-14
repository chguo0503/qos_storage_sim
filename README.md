# QoS Storage Simulator

模拟32张NPU的prefill计算、KV读取预取和共享SSU调度。当前工作树围绕两项研究保存输入、完整仿真结果、图片、源码与独立审计；教程和原始`data`同时保留。

## 实验与教程

| 内容 | 入口 |
|---|---|
| 周六：128K/32K画像，A:B=1:2，Random/Ordered，盘数与策略对照 | [实验报告](results/baseline_ab128_32_ratio12_20260912/report.md) |
| 本次：用数学寻找接近容量时的低利用率Random输入 | [研究报告](results/baseline_random_near_capacity_20260914/report.md) · [全部统计](results/baseline_random_near_capacity_20260914/comparison.md) |
| 本次输入配比、等待阈值、每层平均b/B | [通俗方法](results/baseline_random_near_capacity_20260914/report_core.md) · [89%/85%目标核对](results/baseline_random_near_capacity_20260914/goal_80s_update.md) |
| 小白教程、两份手稿与实验解释 | [教程导航](docs/l1_l2_l3_beginner/README.md) · [四页核心PDF](docs/l1_l2_l3_beginner/周六实验与NPU利用率_四页核心版.pdf) |
| 所有保留资料的入口 | [results导航](results/README.md) |

## 读数约定

- NPU平均利用率是窗口内真实计算卡时间除以`NPU数×窗口长度`，IO等待留在分母。每卡有待处理任务、每卡窗口内都实际算过两类请求，需要分别核验。
- `B=V/C`是该画像每层参考带宽。完整内部周期的`平均b=V/(C+等待)`包含零接收的时间，因此`平均b/B=C/(C+等待)`；它不代表瞬时利用率，也不能直接预测排队。
- `rho≈1`是整批输入的理想平均需求接近容量，不保证逐盘逐时欠载。SSU真实服务、NPU链路接收和名义需求分开统计。
- 表中的SLO是接纳后prefill完成代理，不含接纳前排队；没有真实首token事件，不能当成端到端TTFT。
- `once`（图中“流量分配策略”）与`new_once`不同。原始data和区间外构造画像分别标注。

## 环境和运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider test_*.py
```

当前保留4个测试模块、10个已有测试例，覆盖两组结果使用的Baseline/Once路由、I/O守恒、5ms采样、运行入口与SSU放置校验，全部通过。12个旧测试文件及保留文件中的无关用例已删除，见[逐文件依据与验证](docs/maintenance_20260914/README.md)。此前159项测试的[验证记录](results/baseline_random_near_capacity_20260914/retained_tests_check.json)保留为历史证据。历史图表重绘还可能需要中文字体、ReportLab或浏览器，不能由单元测试通过推断所有PDF可在任意机器一键重建。

新实验和历史复现均应使用新输出位置，避免覆盖冻结结果。复现本次已保存输入可用：

```bash
python results/baseline_random_near_capacity_20260914/replay_case.py \
  --case results/baseline_random_near_capacity_20260914/runs/reference_ssu3_h22000_seed7/baseline \
  --output /tmp/qos_reference_fresh_replay
```

输出目录必须不存在；脚本核验输入、核心源码及完整结果。需要逐块图时加`--trace`。每次完整仿真可能需要数分钟以上；目录内数学分析和绘图脚本仅读取已有结果。

本次384K的Once原始trace约122.76MiB，发布时按原始字节无损分片，未重压缩或抽样。需要重新绘制它的逐块图时先恢复：

```bash
python results/baseline_random_near_capacity_20260914/publish_large_artifact.py \
  --restore results/baseline_random_near_capacity_20260914/runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/trace.json.gz.parts/manifest.json
```

脚本逐项验证分片与完整SHA，恢复后的trace与原科学证据字节相同；现有PNG和统计表可以直接阅读。

## 保留范围与历史恢复

[现行文件范围](CURRENT_PROJECT_MANIFEST.md) · [SHA256清单](CURRENT_PROJECT_SHA256SUMS) · [已执行清理记录](results/baseline_random_near_capacity_20260914/cleanup_execution.md) · [清理依赖与阅读审计](results/baseline_random_near_capacity_20260914/cleanup_readability_audit.md)。

部分保留运行模块沿用旧文件名，是两项研究的导入依赖，不应按名称再次删除。它们的旧默认命令可能重新生成旧结果目录，应以研究目录里的冻结命令为准。

更早的已入库实验可从提交[38edfa31](https://github.com/chguo0503/qos_storage_sim/tree/38edfa31cb5b61da02f4d198e97dc09d18356ad6)查看。历史完整源码校验包含已精简的旧脚本，须用本次研究的`audit_remote_sources.tar.gz`在独立目录恢复；新运行需要的29项核心源码保留在根目录。原教程与PDF生成审计所对应的文档字节保存在[929102a7](https://github.com/chguo0503/qos_storage_sim/tree/929102a70c2316789003f4969e12a9fc92397f8a)，后续仅修复导航，不重写历史审计。
