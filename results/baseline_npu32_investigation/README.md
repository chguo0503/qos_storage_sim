# Baseline 在 32 NPU 下的利用率调查

本目录保存 2026-09-07 的原结果审计、新输入、配对实验和报告。原模拟器与策略源码未修改。固定画像输入用于机制控制；每卡混合多种画像的输入用于检验这一机制能否推广。随机构造不等于生产到达统计。

## 阅读入口

- `report.pdf`、`docs/report.md`：中文主报告。
- `notes/mixed_profile_audit/summary.md`：每卡多种不同短流主实验，六种配额、两个种子、三策略共36作业。
- `notes/mixed_strict_ss_audit/summary.md`：三个短画像真正同属SS的六作业复核。
- `notes/mixed_six_ssu_review.md`：六盘混合流四作业，含同人口八盘对照和变差画像。
- `notes/mixed_policy_audit/summary.md`：混合输入S1–S3、S3固定分卡的独立配对。
- `notes/all_strategy_table.md`：screen/formal/holdout 计划全表；未完成显示 pending。
- `analysis/summary.csv`、`roles.csv`、`matched_comparisons.csv`：可进一步分析的完整与窗口指标。
- `analysis/matched_audit.json`：冻结输入、请求、字节守恒、源文件和时间账目审计。
- `notes/existing_audit.md`：原 coflow / multi-SSU 结果的同画像重聚合。
- `notes/grid_analysis.md`、`notes/phase_analysis.md`：固定流格点与随机启动相位控制。
- `notes/native_assignment_findings.md`：选卡制造热点与到达顺序反事实。
- `notes/startup_cost_review.md`：原仿真器二次复杂度启动开销，未计入模拟时间。

## 复现一个冻结输入

在仓库根目录执行；Python 环境需有 NumPy 和 Matplotlib。写入新输出目录以保留本轮证据。

```bash
python -B run_baseline_npu32_stress.py \
  --manifest results/baseline_npu32_investigation/mixed_varied/inputs/raw_size_varied_rho097_seed7.json.gz \
  --strategy baseline --output /tmp/qos_reproduce_varied32_baseline

python -B run_baseline_npu32_stress.py \
  --manifest results/baseline_npu32_investigation/mixed_varied/inputs/raw_size_varied_rho097_seed7.json.gz \
  --strategy once --output /tmp/qos_reproduce_varied32_once
```

这里每张卡包含三种体积和计算预算不同的短画像及两种长画像，各卡独立打乱全部92条请求，参数完全取自data。每次结果保留完整输入指纹、提交种子、策略配置、两公共窗口、全部请求/层时间线和源码 SHA256。

`mixed_varied/selected_spec.json` 保存首五种配额，`mixed_varied/raw_size_varied_spec.json` 保存短流体积变化的第六种配额。需要重新生成输入并运行时可用：

```bash
python -B results/baseline_npu32_investigation/run_mixed_sustained_probe.py \
  --spec results/baseline_npu32_investigation/mixed_varied/raw_size_varied_spec.json \
  --output /tmp/qos_reproduce_varied_mixed --workers 4
```

## 复现原冻结计划

```bash
python -B sweep_baseline_npu32_stress.py \
  --plan results/baseline_npu32_investigation/screen_plan.json \
  --output /tmp/qos_reproduce_screen --workers 4 --timeout 7200 --label screen
```

同理可运行 `formal_plan.json` 和 `holdout_plan.json`。先冻结每个输入，再对全部策略读取同一 manifest。正式 S1–S3 的 pipeline 与 fixed 变体明确分开；fixed 为选卡消融。原生 Once 与 5 ms Once 的观测权限也分开，不合并成同一个算法排名。

`offload_plan.json` 是 formal 中部分作业的本机副本，用于并行提速，不增加唯一实验数量。完成结果先原子安装到远端尚未开始的作业目录，远端原调度器再次检查输入/配置/源码后作为缓存接受。运行中的远端作业不被覆盖。

部分 S1–S3 输入有两万余条同时到达请求；原适配器反复扫描不断增长的 backlog，即使 fixed 也会产生很高的 Python 运行开销。该开销不进入模拟时间。完整复现应为单作业预留足够墙钟时间；不得把墙钟超时当成数据面低利用率。

## 只读复算与制图

```bash
python -B results/baseline_npu32_investigation/notes/reaggregate_existing.py
python -B results/baseline_npu32_investigation/analyze_results.py
python -B results/baseline_npu32_investigation/notes/build_all_strategy_table.py
python -B results/baseline_npu32_investigation/analysis/plot_arrival_counterfactual.py
python -B results/baseline_npu32_investigation/grid/plot_grid.py
python -B results/baseline_npu32_investigation/analysis/plot_mixed_input.py
python -B results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py
python -B results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py \
  --study results/baseline_npu32_investigation/mixed_strict_ss --require-short-category SS
python -B results/baseline_npu32_investigation/mixed_six_ssu/analyze_probe.py
python -B results/baseline_npu32_investigation/notes/aggregate_mixed_policy_profiles.py
```

主分析只读取已经完成的 `.json.gz` 结果；运行过程中导出的汇总带 snapshot 时间，最终报告使用全部已完成结果重新生成。

报告 PDF 在 `docs/` 中用 Pandoc/XeLaTeX 构建：

```bash
pandoc report.md --pdf-engine=xelatex -o ../report.pdf
```

六盘或严格SS的单项复现同样使用根目录`run_baseline_npu32_stress.py --manifest ... --strategy ... --output ...`，分别选择`mixed_six_ssu/inputs/`或`mixed_strict_ss/inputs/`中的冻结输入。其他策略的八作业计划在`mixed_policy/plan.json`，沿用同一sweep程序；`variant`区分pipeline和fixed，不能只按strategy名称合并。

最终`completion.json`记录实际完成时间、逐阶段结果数、计数范围与源包SHA256；`final_verification.json`重新核对全部结果和初始103项源码/data。`final_source_and_inputs.tar.gz`包含根目录Python源码、data、新增分析/运行脚本、冻结输入和计划，可在新目录解包；原始仿真结果仍完整保存在本目录各`runs/`中，源包不重复收纳结果。

## 口径与证据边界

- 主窗口固定为 `[1000,2000] ms`，第二窗口为 `[2000,3000] ms`；另报告完整批次 makespan。短 pilot 有明确独立窗口，不能替换正式窗口。
- 整机利用率分母是卡数乘窗口长度；画像计算比的分母是同请求群体接纳到完成总历时。两者不能混称。
- 短请求 SLO 从接纳计时，与 arrival-to-completion 分开；t=0 饱和积压的后者不代表线上体验。
- 所有逐卡窗口均核对 compute、暴露 I/O stall 与 idle；动态选卡后的物理卡不再对应同一请求。
- 逐盘平均容量可行是必要条件，不保证每个层期限可满足。过载输入明确保留并标注。
- 请求读取生命周期会含并发等待，不能相加后称为 SSD 忙时间。
- `data` 原始行、1K 外推、NQL 插值、尾块补齐、计算时间缩放均在 manifest 中明确区分。
- `initial_source_hashes.json` 和 `initial_source.tar.gz` 保存开始时源码；运行器、策略、环境及实际命令随每阶段或结果保留。
- 本机与授权远端使用不同 Python 版本；冻结输入避免生成差异，有限桥接结果见 `native_assignment/raw32_baseline_wrapper_bridge.json` 与跨环境复核。未声称所有策略跨版本逐位等价。
- 所有数据传输只使用本次授权连接；本目录不保存 SSH 密码。
