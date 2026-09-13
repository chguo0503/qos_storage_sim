# 清理依赖审查（只读建议，未删除任何文件）

审查目的：最终仅保留周六研究与当前研究的results目录，同时保留两组研究复现所需的模拟器和工具。运行中的源码不得删除或修改。

## 必须保留的根文件

以下35项来自Sat/本次脚本的根模块导入闭包（含惰性分支），并补上动态source_files()要求的哈希文件。缺一个哈希清单文件也可能使入口在运行前直接失败。

- `authenticated_workload_inputs.py`
- `baseline_path0_layout.py`
- `coflow_capacity_analysis.py`
- `coflow_client_policy.py`
- `coflow_disk_adapter.py`
- `coflow_disk_policy.py`
- `coflow_joint_policy.py`
- `coflow_sim_adapter.py`
- `continuous_batch_control.py`
- `continuous_batch_sim.py`
- `continuous_prefill_client.py`
- `continuous_prefill_workload.py`
- `data`
- `multi_ssu_npu_assignment.py`
- `multi_ssu_qos_controller.py`
- `policy_logic.py`
- `random_steady_state_workload.py`
- `run_baseline_4npu_ssu1_low_utilization.py`
- `run_baseline_npu32_stress.py`
- `run_coflow_experiments.py`
- `run_multi_ssu_stall_experiments.py`
- `run_shared_path_experiments.py`
- `shared_path_baseline.py`
- `shared_path_common.py`
- `shared_path_new_once.py`
- `shared_path_once.py`
- `shared_path_sim_adapter.py`
- `shared_path_strategy1.py`
- `shared_path_strategy2.py`
- `shared_ssu_state.py`
- `sim.py`
- `six_request_workload.py`
- `strategy_profiles.py`
- `sweep_coflow_development.py`
- `sweep_coflow_experiments.py`

## 回归测试与可删脚本的区别

没有test_*被正常仿真入口导入；这不代表它们全都无用。推荐保留以下几个体积小、直接保护当前策略/数据面的回归测试：

- `test_shared_path_sim_adapter.py`
- `test_shared_path_policies.py`
- `test_shared_ssu_state.py`
- `test_continuous_batch_profile_cycle_frontier.py`

test_coflow_sim_adapter.py imports example_trace from test_shared_path_sim_adapter.py; keeping the former requires the latter. Runtime-unreachable does not mean a regression test is useless.

旧教程截图、排版、旧策略报告和已删除旧实验的专属测试可作为清理候选；删前应检查它们是否仍被文档或其他保留测试调用。

## 旧results依赖与链接修复

Sat和本次Python文件没有依赖其他旧results目录的import或路径常量；不需要把旧results里的脚本提取到新研究才能复现这两组实验。

发现的具体文档活链接：

- 周六 `formula_review/README.md:220` 引用旧 sustained_mixed_ge10k_20260911/report.md。
- 原始 `docs/L1_L2_L3_NPU_图解教程.md:503–509` 的GitHub main链接指向旧underload图表/证据/报告。
- `docs/cleanup_20260911/README.md` 中两处本地活链接与若干历史路径会变旧；可加归档说明或切为历史commit链接。

上述被引用旧报告都已经git跟踪，因此可在最终清理前记录commit，并改成该commit的GitHub永久链接。这样保留历史来源，又不必把旧数据重新放入results。历史审计JSON中的来源路径是当时事实，宜保留其内容和hash并在新清理说明中交代归档位置，不应批量改写所有历史JSON。

`source_files()`以glob枚举coflow_*.py和shared_path_*.py，移动/删改这些文件会改变科学源码指纹。为了仅清理测试，不应借机改造运行时导入结构。

本次 `audit_remote_sources.tar.gz` 约0.4MiB，保存全部106个运行时冻结源文件，建议保留。它比日后追溯已删旧测试脚本更直接。最终先在清理前完成 `audit_remote_acceptance.py`；清理后若要复核全量历史源hash，可将此快照解压到独立目录，再给验收脚本传 `--source-root <该目录>`，不必把旧脚本重新放回项目。`runtime_validation/remote_incoming/` 中的传输tar与解包残留在验收后可删；保留正式 `runs`、输入、队列/导入/验收JSON即可。

## 清理后的新实验与历史复现

已在独立 `/tmp` 目录实际验证，只复制上面的35个根文件（含data）和本次冻结runner，执行默认main512 seed7的 `--prepare-only` 成功；没有运行仿真。它枚举的29个核心/策略hash全部与历史一致，生成manifest解压后的字节与历史文件完全一致。证据见 `audit_minimal_prepare_check.json`。

必须区分三种核验：

- runner自身的 `source_hashes()` 枚举29个核心/策略文件，保证一次运行前后这些源文件不变；独立的 `study_plan.json` 与远端plan才冻结全部106个源文件。删无关测试不会自动改变29个hash，但会使106个历史源清单在当前目录中不完整。
- `load_manifest()` 会重建请求并核验 `input_fingerprint`；旧manifest本身没有核心源码hash，不能阻止用户用变更后的模拟器运行。历史结果的严格复现需显式核对外部plan中的源码hash。
- 重新生成 `.json.gz` 的gzip头含生成时间。本次验证只有mtime这4字节不同，解压后的内容完全相同，仍会得到不同的压缩包SHA256。这不是请求变更，却无法通过历史文件传输hash检查。

因此最终README应明确：**做新实验可用清理后的依赖生成新manifest；复现历史结果请在独立目录解压冻结源码快照、使用Python3.10与numpy2.2.6，并原样复制保留的 `inputs/*.json.gz`，不要重新生成或重新压缩旧manifest。** 原样复制后用该副本runner的 `--manifest <原始输入副本>` 运行，输出写入独立目录，避免碰已有历史结果。科学结果可比较；主机名、路径、Python补丁版本及真实执行耗时另行说明，不要求这些运行环境元数据逐字相同。

本审查尚未执行清理、提交或上传；由根代理在全部实验结束后统一处理。
