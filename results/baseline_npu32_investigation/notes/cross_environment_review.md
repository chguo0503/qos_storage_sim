# 有限跨环境复核

完整配对 7/7；只读比较相同冻结 manifest 与原策略。

已完成配对的全部配置检查通过：True；全部请求和微批/层时间线逐项精确相等：True。最大保存 U 差 2.22044604925e-14 个百分点，同解释器重算 U 最大差 0，makespan 最大差 0 ms。

同输入、同提交 seed、同核心和策略源码；本机 shared Baseline/Once 的用途是为本机执行策略提供锚点。不能预先假定跨 Python 的相位或逐请求时间线相同。

| 输入 | 策略 | 窗口/ms | 远端 U% | 本机 U% | 差/百分点 | makespan 差/ms | 全请求/层时间线精确相同 |
|---|---|---|---:|---:|---:|---:|---|
| raw32_shuffled | baseline | 1000–2000 | 98.476700623 | 98.476700623 | 0 | 0 | True |
| raw32_shuffled | baseline | 2000–3000 | 98.660833789 | 98.660833789 | 0 | 0 | True |
| raw32_shuffled | once | 1000–2000 | 97.801703592 | 97.801703592 | 1.11022302e-14 | 0 | True |
| raw32_shuffled | once | 2000–3000 | 98.139584806 | 98.139584806 | 0 | 0 | True |
| strong32_local | baseline | 1000–2000 | 52.209627191 | 52.209627191 | -2.22044605e-14 | 0 | True |
| strong32_local | baseline | 2000–3000 | 52.216004354 | 52.216004354 | 2.22044605e-14 | 0 | True |
| strong32_local | once | 1000–2000 | 100.000000000 | 100.000000000 | 0 | 0 | True |
| strong32_local | once | 2000–3000 | 100.000000000 | 100.000000000 | 0 | 0 | True |
| strong32_stripe6_feasible | baseline | 1000–2000 | 47.692779812 | 47.692779812 | -1.66533454e-14 | 0 | True |
| strong32_stripe6_feasible | baseline | 2000–3000 | 47.684851622 | 47.684851622 | -2.22044605e-14 | 0 | True |
| strong32_stripe6_feasible | once | 1000–2000 | 93.316692412 | 93.316692412 | -1.11022302e-14 | 0 | True |
| strong32_stripe6_feasible | once | 2000–3000 | 96.312694925 | 96.312694925 | 2.22044605e-14 | 0 | True |
| raw32_local | baseline | 1000–2000 | 86.993842660 | 86.993842660 | 0 | 0 | True |
| raw32_local | baseline | 2000–3000 | 86.738583737 | 86.738583737 | 1.11022302e-14 | 0 | True |

数值详细表：`cross_environment/cross_environment_comparison.csv`。逐请求差：`matched_request_differences.csv`；指纹、参数、源码与递归结果对比：`cross_environment_audit.json`。

保存的 U 可能仅因 Python 求和舍入而有微差，因此另在同一个本机解释器重算两套事件时间线的 U；同时单独比较全部 request_metrics、microbatch_metrics 与每个记录层事件，不能把均值相近当成时间线相同。

raw32_local 本机 shared Baseline 复用 design 代理的 `native_assignment/runs/raw32_local/baseline_shared_local`，未重复运行。所有 5 ms 设置和物理位置保留。运行时长受机器、Python 和并行负载共同影响，不能据此单独给实现加速比。
