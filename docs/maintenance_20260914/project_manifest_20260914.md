> 2026-09-14 的历史清理快照。下文路径和 SHA 清单以当时的项目根目录为准；当前结构见[项目说明](../../README.md)，不能用此旧清单校验重构后的源码。

# 当前项目保留范围

项目聚焦两项研究：周六的 128K/32K、A:B=1:2 对照，以及本次接近容量时的 Baseline Random 输入搜索。`data`、教程、可复现输入、完整结果、绘图脚本和审计证据均保留。

| 内容 | 位置 |
|---|---|
| 周六结果 | `results/baseline_ab128_32_ratio12_20260912/` |
| 本次研究 | `results/baseline_random_near_capacity_20260914/` |
| 教程与手稿 | `docs/` |
| 原始请求画像 | `data` |
| 核心运行模块及回归测试 | 根目录 Python 文件 |

本次研究内部既保留有利候选，也保留高利用率候选、窗口长短覆盖失败的种子和外推画像的失败方向；它们是判断选择偏差的证据，不属于可删的临时结果。等价 trace 重播与长窗口不同人口分别归档，不计为新增随机种子。未完成的中断尝试与原输入恢复记录也保留，但中断尝试不进入性能均值，恢复不计为新种子。

## 运行依赖

根目录保留35项运行文件（含`data`）和4个回归测试模块，共10个已有测试例。部分运行模块使用历史实验名，是实际导入依赖；名称相似不表示可以删除。测试则按现有两组结果实际使用的功能精简，见[本轮维护记录](docs/maintenance_20260914/README.md)。两项研究的运行命令以各自目录里的冻结命令为准。

```text
authenticated_workload_inputs.py
baseline_path0_layout.py
coflow_capacity_analysis.py
coflow_client_policy.py
coflow_disk_adapter.py
coflow_disk_policy.py
coflow_joint_policy.py
coflow_sim_adapter.py
continuous_batch_control.py
continuous_batch_sim.py
continuous_prefill_client.py
continuous_prefill_workload.py
data
multi_ssu_npu_assignment.py
multi_ssu_qos_controller.py
policy_logic.py
random_steady_state_workload.py
run_baseline_4npu_ssu1_low_utilization.py
run_baseline_npu32_stress.py
run_coflow_experiments.py
run_multi_ssu_stall_experiments.py
run_shared_path_experiments.py
shared_path_baseline.py
shared_path_common.py
shared_path_new_once.py
shared_path_once.py
shared_path_sim_adapter.py
shared_path_strategy1.py
shared_path_strategy2.py
shared_ssu_state.py
sim.py
six_request_workload.py
strategy_profiles.py
sweep_coflow_development.py
sweep_coflow_experiments.py
test_coflow_experiment_inputs.py
test_shared_path_policies.py
test_shared_path_sim_adapter.py
test_shared_ssu_state.py
```

本轮精简后的10项测试全部通过，记录见[当前验证](docs/maintenance_20260914/root_test_cleanup.json)。核心模拟器、策略及`data`保持不变。此前159项测试的[验证](results/baseline_random_near_capacity_20260914/retained_tests_check.json)、[清理记录](results/baseline_random_near_capacity_20260914/cleanup_execution.md)与[90例核验](results/baseline_random_near_capacity_20260914/final_research_audit.json)保留为历史证据。

## 历史与完整性

- 更早已入库内容见提交 [`38edfa31`](https://github.com/chguo0503/qos_storage_sim/tree/38edfa31cb5b61da02f4d198e97dc09d18356ad6)。
- 周六研究、原始教程及其历史审计版本在提交 [`929102a7`](https://github.com/chguo0503/qos_storage_sim/tree/929102a70c2316789003f4969e12a9fc92397f8a)。之后的教程修改仅修复导航，旧审计不会被改写成新文档的校验结果。
- 本次运行最初冻结的完整源码副本保存在 `results/baseline_random_near_capacity_20260914/audit_remote_sources.tar.gz`；原始106项源码审计包含后来精简的历史文件。需要重查该原始快照时，应解压到独立目录，不能要求清理后的根目录仍包含全部旧文件。
- `CURRENT_PROJECT_SHA256SUMS` 对保留交付文件列出 SHA256，清单不包含自身、Git元数据、缓存和运行日志。最终交付后可在项目根目录运行 `sha256sum -c CURRENT_PROJECT_SHA256SUMS`。
- 保留科学结果的输入指纹、核心源码SHA、结果SHA和独立分析分别记在case与审计JSON中；不以图像标题代替原始数据验证。
- 本轮按要求更新周六研究`presentation_pdf`下四张PDF的顶部文字，并精简根目录测试；旧审计对应修改前的文件版本，最新修改见[本轮记录](docs/maintenance_20260914/README.md)。

物理带宽观测器只保存0–20秒；65秒工作量实验中20秒以后只有完整计算/等待日志，不声称有未记录的实际SSD带宽。

384K seed7 Once的122.76MiB原始压缩trace以64MiB无损分片发布。使用研究目录内的`publish_large_artifact.py`和对应`trace.json.gz.parts/manifest.json`恢复；已实际往返重建并验证整体SHA一致。Git中的分片包含原始全部字节，原trace本地副本不重复入库。原命令和绘图审计的SHA不改写。
