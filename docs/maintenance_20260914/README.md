# 四张PDF文字调整与根目录测试精简

按本轮要求，`presentation_pdf` 下四张PDF去掉顶部的配置/整机统计行，以及“每行一张卡……”说明。标题、图例、每卡U、需求/供给数值和时间轴保持原样；四张仍是单页矢量PDF。

[四张PDF所在目录](../../results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/presentation_pdf/) · [逐字文字对照](pdf_text_comparison.json)

根目录原16个测试文件包含旧实验功能，不能只因为生产模块仍被导入，就认为其中所有测试都用于现有两组结果。本轮删除12个文件，并精简另外4个文件，只保留10个已有测试例：

| 保留文件 | 与现有结果的关系 |
|---|---|
| `test_coflow_experiment_inputs.py` | 两组实验实际复用的运行入口与SSU放置指纹校验；旧名称保留，但已删除旧coflow输入配额和到达率实验用例 |
| `test_shared_path_policies.py` | Baseline/Once路由，以及Once路由单测所用快照夹具的校验 |
| `test_shared_path_sim_adapter.py` | Baseline/Once实际仿真适配：I/O守恒、5ms采样、Baseline观测透明性 |
| `test_shared_ssu_state.py` | 5ms周期快照和最小采样间隔 |

删除的范围包括旧coflow容量证书和全局调度、动态NPU分配、动态CIR、JIT、重排、周期前沿探针等。两组结果均采用固定选卡的Baseline/Once路径；逐文件调用依据见[精简记录](root_test_cleanup.md)。这不表示上述功能本身无用，只是当前两组结果未使用。

运行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_*.py`，10个测试全部通过。没有新增测试输入或断言。35项生产运行文件（含data）逐字节保持不变；原956张图表中，仅本轮指定4张PDF变化，其余952张不变。[测试结果及SHA](root_test_cleanup.json)

此前“51文件、159项测试、956张图片不变”的记录对应[43ab395版本](https://github.com/chguo0503/qos_storage_sim/tree/43ab395ba92be984ccd7318f7809ddf2775a3274)及其清理过程，作为历史证据保留。本轮单独记录已授权修改，不覆盖旧审计。该次文件清单现已归档为 `project_manifest_20260914.md` 和 `project_sha256sums_20260914.txt`；当前结构见[项目说明](../../README.md)。
