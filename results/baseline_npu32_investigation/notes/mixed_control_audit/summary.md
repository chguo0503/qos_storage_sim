# 持续混合多短画像独立重聚合

计划：`results/baseline_npu32_investigation/mixed_controls`；生成 UTC：2026-09-06T17:25:51.091505+00:00。

所有画像统计使用完整同 request-ID cohort；U 是 compute/active；SLO 是 admission 后 1.5×自身计算时间。图与表包含尚缺策略的 pending 标记。

|输入|策略|状态|1–2s U|2–3s U|两窗 all-active|full makespan ms|短类 U|短类 SLO|
|---|---|---|---:|---:|---|---:|---:|---:|
|aligned_varied_short_only_seed7|baseline|complete|100.000%|100.000%|True|4028.372|99.998%|22752/22752|
|aligned_varied_short_only_seed7|once|complete|100.000%|100.000%|True|4028.452|99.997%|22752/22752|
|aligned_varied_short_only_seed7|new_once|complete|100.000%|100.000%|True|4028.468|99.997%|22752/22752|

## 逐短画像的同 cohort 指标

|输入|策略|短画像 seqK/NQL|完整请求数|输入 compute 占比|compute/active|admission SLO|平均 stall ms|
|---|---|---|---:|---:|---:|---:|---:|
|aligned_varied_short_only_seed7|baseline|1/128|7584|24.847%|99.996%|7584/7584|0.000|
|aligned_varied_short_only_seed7|baseline|1/256|7584|32.818%|99.998%|7584/7584|0.000|
|aligned_varied_short_only_seed7|baseline|1/384|7584|42.335%|99.999%|7584/7584|0.000|
|aligned_varied_short_only_seed7|once|1/128|7584|24.847%|99.994%|7584/7584|0.000|
|aligned_varied_short_only_seed7|once|1/256|7584|32.818%|99.997%|7584/7584|0.000|
|aligned_varied_short_only_seed7|once|1/384|7584|42.335%|99.998%|7584/7584|0.000|
|aligned_varied_short_only_seed7|new_once|1/128|7584|24.847%|99.994%|7584/7584|0.000|
|aligned_varied_short_only_seed7|new_once|1/256|7584|32.818%|99.997%|7584/7584|0.000|
|aligned_varied_short_only_seed7|new_once|1/384|7584|42.335%|99.998%|7584/7584|0.000|
