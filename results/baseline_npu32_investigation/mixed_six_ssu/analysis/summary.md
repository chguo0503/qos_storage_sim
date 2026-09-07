# 持续混合多短画像独立重聚合

计划：`/home/chguo/work/last_code/qos_storage_sim/results/baseline_npu32_investigation/mixed_six_ssu`；生成 UTC：2026-09-06T17:48:57.692481+00:00。

所有画像统计使用完整同 request-ID cohort；U 是 compute/active；SLO 是 admission 后 1.5×自身计算时间。图与表包含尚缺策略的 pending 标记。

|输入|策略|状态|1–2s U|2–3s U|两窗 all-active|full makespan ms|短类 U|短类 SLO|
|---|---|---|---:|---:|---|---:|---:|---:|
|aligned_short080_ssu6_seed7|baseline|complete|74.530%|80.112%|True|5391.766|74.986%|14437/18720|
|aligned_short080_ssu6_seed7|new_once|complete|91.885%|97.610%|True|4473.272|99.995%|18720/18720|
|raw_size_varied_ssu6_seed7|baseline|complete|95.717%|96.253%|True|7778.426|78.401%|1170/1728|
|raw_size_varied_ssu6_seed7|new_once|complete|98.413%|96.681%|True|7734.469|90.369%|1641/1728|

## 逐短画像的同 cohort 指标

|输入|策略|短画像 seqK/NQL|完整请求数|输入 compute 占比|compute/active|admission SLO|平均 stall ms|
|---|---|---|---:|---:|---:|---:|---:|
|aligned_short080_ssu6_seed7|baseline|1/128|6240|19.842%|66.337%|3973/6240|2.143|
|aligned_short080_ssu6_seed7|baseline|1/256|6240|26.207%|74.485%|4909/6240|1.911|
|aligned_short080_ssu6_seed7|baseline|1/384|6240|33.808%|81.662%|5555/6240|1.616|
|aligned_short080_ssu6_seed7|new_once|1/128|6240|19.842%|99.993%|6240/6240|0.000|
|aligned_short080_ssu6_seed7|new_once|1/256|6240|26.207%|99.995%|6240/6240|0.000|
|aligned_short080_ssu6_seed7|new_once|1/384|6240|33.808%|99.995%|6240/6240|0.000|
|raw_size_varied_ssu6_seed7|baseline|32/128|576|2.359%|43.884%|178/576|12.051|
|raw_size_varied_ssu6_seed7|baseline|48/256|576|5.343%|73.525%|433/576|7.687|
|raw_size_varied_ssu6_seed7|baseline|64/512|576|12.787%|94.778%|559/576|2.815|
|raw_size_varied_ssu6_seed7|new_once|32/128|576|2.359%|88.418%|552/576|1.234|
|raw_size_varied_ssu6_seed7|new_once|48/256|576|5.343%|97.779%|569/576|0.485|
|raw_size_varied_ssu6_seed7|new_once|64/512|576|12.787%|87.942%|520/576|7.005|
