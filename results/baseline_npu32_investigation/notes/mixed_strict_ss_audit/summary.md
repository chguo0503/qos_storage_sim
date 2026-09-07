# 持续混合多短画像独立重聚合

计划：`results/baseline_npu32_investigation/mixed_strict_ss`；生成 UTC：2026-09-06T18:00:52.673550+00:00。

所有画像统计使用完整同 request-ID cohort；U 是 compute/active；SLO 是 admission 后 1.5×自身计算时间。图与表包含尚缺策略的 pending 标记。

|输入|策略|状态|1–2s U|2–3s U|两窗 all-active|full makespan ms|短类 U|短类 SLO|
|---|---|---|---:|---:|---|---:|---:|---:|
|raw_strict_ss_rho095_seed7|baseline|complete|96.508%|94.235%|True|7519.960|75.380%|1939/2592|
|raw_strict_ss_rho095_seed7|once|complete|98.004%|95.186%|True|7434.959|81.194%|2148/2592|
|raw_strict_ss_rho095_seed7|new_once|complete|97.824%|94.505%|True|7461.440|86.838%|2306/2592|
|raw_strict_ss_rho095_seed123|baseline|complete|95.776%|97.735%|True|7613.860|72.327%|1878/2592|
|raw_strict_ss_rho095_seed123|once|complete|97.340%|98.109%|True|7541.353|76.896%|2020/2592|
|raw_strict_ss_rho095_seed123|new_once|complete|96.725%|99.090%|True|7585.642|80.459%|2147/2592|

## 相对baseline的窗口或完整完工负例

正Δmakespan表示变慢；全部配对（含正例）在fleet_pairs.csv。

|输入|策略|ΔU1 pp|ΔU2 pp|Δmakespan %|
|---|---|---:|---:|---:|

## 逐短画像的同 cohort 指标

|输入|策略|短画像 seqK/NQL|完整请求数|输入 compute 占比|compute/active|admission SLO|平均 stall ms|
|---|---|---|---:|---:|---:|---:|---:|
|raw_strict_ss_rho095_seed7|baseline|32/128|864|3.662%|64.174%|544/864|5.261|
|raw_strict_ss_rho095_seed7|baseline|48/256|864|8.295%|89.120%|817/864|2.606|
|raw_strict_ss_rho095_seed7|baseline|64/128|864|5.748%|67.833%|578/864|7.015|
|raw_strict_ss_rho095_seed7|once|32/128|864|3.662%|69.043%|604/864|4.225|
|raw_strict_ss_rho095_seed7|once|48/256|864|8.295%|90.279%|820/864|2.299|
|raw_strict_ss_rho095_seed7|once|64/128|864|5.748%|78.593%|724/864|4.029|
|raw_strict_ss_rho095_seed7|new_once|32/128|864|3.662%|75.739%|714/864|3.019|
|raw_strict_ss_rho095_seed7|new_once|48/256|864|8.295%|95.448%|841/864|1.018|
|raw_strict_ss_rho095_seed7|new_once|64/128|864|5.748%|83.754%|751/864|2.869|
|raw_strict_ss_rho095_seed123|baseline|32/128|864|3.662%|60.617%|524/864|6.123|
|raw_strict_ss_rho095_seed123|baseline|48/256|864|8.295%|87.284%|803/864|3.110|
|raw_strict_ss_rho095_seed123|baseline|64/128|864|5.748%|64.334%|551/864|8.201|
|raw_strict_ss_rho095_seed123|once|32/128|864|3.662%|64.078%|561/864|5.283|
|raw_strict_ss_rho095_seed123|once|48/256|864|8.295%|87.658%|786/864|3.006|
|raw_strict_ss_rho095_seed123|once|64/128|864|5.748%|73.253%|673/864|5.401|
|raw_strict_ss_rho095_seed123|new_once|32/128|864|3.662%|67.574%|655/864|4.522|
|raw_strict_ss_rho095_seed123|new_once|48/256|864|8.295%|91.203%|792/864|2.059|
|raw_strict_ss_rho095_seed123|new_once|64/128|864|5.748%|76.736%|700/864|4.485|
