# 2026-09-15 首轮远端预览采集器错误

首轮远端四组实验中，Baseline 在模拟约 4.5 秒进入 warm 预览采集时退出：

```text
AttributeError: 'list' object has no attribute 'values'
run_trial.py / warm_preview / context.microbatches.values()
```

原因：运行时 `context.microbatches` 是列表，预览采集代码误当作字典。正式仿真内核、策略和输入均未因该错误改变。此前 smoke 测试刻意跳过 warm 预览，因此没有覆盖这一分支。

处理：确认异常后终止同一批尚未进入该分支的三个自有实验进程；不触碰其他进程。失败及中断输出保留在远端原目录。修复后使用新的 `*_remote_v2` 标签从同一完整输入重新运行，首轮数据不进入结果对比。

- 远端隔离目录：`/home/chguo/work/qos_l3_slo_routing_20260915_1789483894695997467`
- 首轮标签：`baseline_seed7_remote`、`once_seed7_remote`、`mild_seed7_remote`、`aggressive_seed7_remote`
- 首轮队列 PID：600394；终止的自有实验 PID：600399、600400、600401。
- 防止复发：单独验证 warm 预览分支的数据结构与计算口径，不能只依赖跳过预览的 smoke 测试。

此记录按 self-improvement 技能要求保留错误及修复过程；与最终实验结论无关。

修复验证：v2 将采集计算提取为可直接验证的 `warm_statistics`，并使用真实结果重建运行时结构核对。远端 v2 Baseline 首次预览成功，warm NPU 利用率90.67794278896096%、SLO达标264/341，与历史基准精确一致。新策略预览也已成功生成；完整仿真仍继续执行。

最终验证：修复后的四组主实验与两组静态对照均完整结束，六组远端输出通过全部不变量和源码哈希核查；预览指标与最终统计一致。
