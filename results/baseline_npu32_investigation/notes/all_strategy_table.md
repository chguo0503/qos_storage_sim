# 当前全部计划策略表

生成时间UTC：2026-09-06T18:08:21.901897+00:00。计划106作业：complete=106，pending=0，failed=0，invalid=0。

本表只读取本地已同步的完整结果。pending表示未完成或尚未同步，不赋值为0；完整结果若缺匹配Baseline，差值也保持空。按完整冻结input_fingerprint、submit seed和四个核心源码哈希配对，不按名称猜测相同输入。

模式：5ms/fixed保留原分卡，5ms/pipeline仅S1–S3实际启用到达选卡；baseline/Once/New once忽略CLI中的assignment，仍属fixed。native/fixed为原生即时压力/控制分支，deadline等不会因命令写assignment=pipeline而重分卡。跨模式差异包含不同遥测/控制权限。

U1=[1000,2000]ms，U2=[2000,3000]ms，单位%；接纳/到达SLO均为完整请求集completion−对应起点≤1.5×8C。SLO列为通过数/总数；接纳排队未计入接纳SLO。active为两个窗口分别全程有已接纳工作的卡数。

所有输入的配额、到达和placement均为构造；“原始data参数”仅表示画像数值未经外推/插值/C缩放/尾块补齐，不表示生产到达分布。

## 已完成的负例：任一窗口低于同输入Baseline

只筛选已完成的严格配对；负差全部保留，包括小差异，不据此做统计显著性推断。Δmakespan正值表示完整排空更慢。

| 输入 | 策略/模式 | ΔU1/pp | ΔU2/pp | Δmakespan/% | active |
|---|---|---:|---:|---:|---|
| raw32_local | S1 / 5ms/pipeline | -76.053 | -73.925 | +160.29 | 32,32/32 |
| raw32_local | S2 / 5ms/pipeline | -76.053 | -73.926 | +160.89 | 32,32/32 |
| raw32_local | S3 / 5ms/pipeline | -76.053 | -73.901 | +160.13 | 32,32/32 |
| strong32_local | S1 / 5ms/pipeline | -47.984 | -38.686 | +159.66 | 32,32/32 |
| strong32_local | S2 / 5ms/pipeline | -47.983 | -38.647 | +185.67 | 32,32/32 |
| strong32_local | S3 / 5ms/pipeline | -47.982 | -38.461 | +185.96 | 32,32/32 |
| legacy4_padded | S2 / 5ms/pipeline | -22.216 | -27.873 | -30.86 | 4,4/4 |
| legacy4_padded | S3 / 5ms/pipeline | -22.216 | -27.873 | -30.86 | 4,4/4 |
| legacy4_padded | S1 / 5ms/pipeline | -22.219 | -27.871 | -30.86 | 4,4/4 |
| raw32_shuffled | Once / 5ms/fixed | -0.675 | -0.521 | +0.71 | 32,32/32 |
| raw32_shuffled | New once / 5ms/fixed | -0.468 | -0.398 | -0.14 | 32,32/32 |
| raw32_shuffled_seed7 | New once / 5ms/fixed | -0.465 | -0.014 | -0.37 | 32,32/32 |
| raw32_shuffled | S1 / 5ms/fixed | -0.276 | -0.403 | -0.86 | 32,32/32 |
| raw32_shuffled_seed123 | New once / 5ms/fixed | -0.009 | -0.379 | -0.78 | 32,32/32 |
| raw32_shuffled | S2 / 5ms/fixed | -0.355 | +0.088 | -1.06 | 32,32/32 |
| raw32_shuffled_seed123 | Once / 5ms/fixed | -0.114 | -0.351 | +0.29 | 32,32/32 |
| raw32_shuffled_seed7 | Once / 5ms/fixed | -0.294 | +0.114 | +0.42 | 32,32/32 |
| raw32_allshort_overload | Once / 5ms/fixed | -0.012 | -0.012 | +0.18 | 32,32/32 |

## 主窗口U更高但完整排空更慢

同样只列已完成的同输入配对；窗口U收益不能替代完整吞吐证据。

| 输入 | 策略/模式 | ΔU1/pp | ΔU2/pp | Δmakespan/% |
|---|---|---:|---:|---:|
| raw32_shuffled_seed7 | S3 / 5ms/pipeline | +1.280 | +1.234 | +2.86 |
| raw32_shuffled | S1 / 5ms/pipeline | +1.068 | +0.883 | +2.37 |
| raw32_shuffled | S3 / 5ms/pipeline | +1.186 | +0.948 | +2.25 |
| raw32_shuffled | S2 / 5ms/pipeline | +1.198 | +0.935 | +2.22 |
| raw32_shuffled_seed123 | S3 / 5ms/pipeline | +1.803 | +0.944 | +1.43 |
| raw32_allshort_overload | S3 / 5ms/fixed | +0.007 | +0.012 | +0.02 |

## legacy4_exact

冻结指纹：`3e7e333d365c9138906ba021a68e32f303bed6df22433b50748ea0b9921e448b`。外推/插值画像；4NPU/1SSU；ρmax=0.9972；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / native/fixed | complete | 76.23 | 76.00 | 8391.0 | 53.97 | 792/1046 | 3/1046 | 4,4/4 |
| screen | Once / native/fixed | complete | 99.99 | 100.00 | 4578.4 | 98.92 | 1046/1046 | 4/1046 | 4,4/4 |

## legacy32_local_exact

冻结指纹：`d0dabc13e665c177e3eb93d96dcde30a0fa993c895b0a12567b700a17a6e85b7`。外推/插值画像；32NPU/8SSU；ρmax=0.9972；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / native/fixed | complete | 76.23 | 76.00 | 8391.0 | 53.97 | 6336/8368 | 24/8368 | 32,32/32 |
| screen | Once / native/fixed | complete | 99.98 | 99.97 | 4587.6 | 98.72 | 8368/8368 | 32/8368 | 32,32/32 |

## legacy32_stripe_exact

冻结指纹：`5c5e8b2b6751e426f4612096f49564895b885d431b753a62f1fa3f9d30457d31`。外推/插值画像；32NPU/8SSU；ρmax=0.9972；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / native/fixed | complete | 77.19 | 76.99 | 7927.2 | 57.13 | 6704/8368 | 24/8368 | 32,32/32 |
| screen | Once / native/fixed | complete | 100.00 | 99.99 | 4589.3 | 98.69 | 8368/8368 | 32/8368 | 32,32/32 |

## legacy4_padded

冻结指纹：`849d6c9b398354fe6acff4d7bbbc35d49df401beaf9c1da47d524d445e229911`。外推/插值画像+尾块补齐；4NPU/1SSU；ρmax=0.9998；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 76.23 | 76.00 | 8391.0 | 53.97 | 773/1046 | 3/1046 | 4,4/4 |
| screen | Once / 5ms/fixed | complete | 99.78 | 100.00 | 4588.7 | 98.70 | 1046/1046 | 4/1046 | 4,4/4 |
| screen | S3 / 5ms/fixed | complete | 95.93 | 95.77 | 5349.4 | 84.66 | 1045/1046 | 3/1046 | 4,4/4 |
| formal | New once / 5ms/fixed | complete | 99.99 | 99.97 | 4576.1 | 98.97 | 1046/1046 | 4/1046 | 4,4/4 |
| formal | S1 / 5ms/fixed | complete | 95.68 | 96.12 | 5335.8 | 84.88 | 1046/1046 | 4/1046 | 4,4/4 |
| formal | S1 / 5ms/pipeline | complete | 54.01 | 48.13 | 5801.8 | 78.06 | 1000/1046 | 4/1046 | 4,4/4 |
| formal | S2 / 5ms/fixed | complete | 95.93 | 95.77 | 5349.4 | 84.66 | 1045/1046 | 3/1046 | 4,4/4 |
| formal | S2 / 5ms/pipeline | complete | 54.01 | 48.12 | 5801.8 | 78.06 | 1000/1046 | 4/1046 | 4,4/4 |
| formal | S3 / 5ms/pipeline | complete | 54.01 | 48.12 | 5801.8 | 78.06 | 1000/1046 | 4/1046 | 4,4/4 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4581.8 | 98.85 | 1045/1046 | 3/1046 | 4,4/4 |

## strong4

冻结指纹：`a1aea8062ed1f739974bc8c0bbb783d7b2ed249fcabe355fd8628a277868f915`。外推/插值画像；4NPU/1SSU；ρmax=0.9061；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 52.21 | 52.22 | 6802.6 | 61.04 | 1923/3003 | 1/3003 | 4,4/4 |
| screen | Once / 5ms/fixed | complete | 100.00 | 100.00 | 4187.9 | 99.15 | 3003/3003 | 4/3003 | 4,4/4 |
| screen | S3 / 5ms/fixed | complete | 100.00 | 100.00 | 4187.9 | 99.15 | 3003/3003 | 4/3003 | 4,4/4 |

## strong32_local

冻结指纹：`0873cdcc471ab53d1aa61be51c14452c1b6e7c3b1ecb2ea09329754d736980d0`。外推/插值画像；32NPU/8SSU；ρmax=0.9061；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 52.21 | 52.22 | 6802.6 | 61.04 | 15384/24024 | 8/24024 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 100.00 | 100.00 | 4187.9 | 99.15 | 24024/24024 | 32/24024 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 100.00 | 100.00 | 4188.0 | 99.15 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 100.00 | 100.00 | 4187.9 | 99.15 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 100.00 | 100.00 | 4188.0 | 99.15 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 4.23 | 13.53 | 17663.5 | 23.51 | 23566/24024 | 0/24024 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 100.00 | 100.00 | 4188.0 | 99.15 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 4.23 | 13.57 | 19432.8 | 21.37 | 20687/24024 | 0/24024 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 4.23 | 13.76 | 19452.6 | 21.35 | 20710/24024 | 0/24024 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4186.5 | 99.18 | 24000/24024 | 8/24024 | 32,32/32 |
| formal | Stall interchange / native/fixed | complete | 100.00 | 100.00 | 4187.9 | 99.15 | 24006/24024 | 14/24024 | 32,32/32 |

## strong32_stripe

冻结指纹：`dac07f416a477ffdca7980699bd5ef835ff131b4c404a0aeefdd1dafe0ae75a4`。外推/插值画像；32NPU/8SSU；ρmax=0.9061；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 52.21 | 52.22 | 6802.6 | 61.04 | 15384/24024 | 8/24024 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 94.47 | 93.57 | 4486.7 | 92.55 | 23345/24024 | 19/24024 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 99.98 | 99.98 | 4191.8 | 99.06 | 24024/24024 | 32/24024 | 32,32/32 |

## strong32_hash_feasible

冻结指纹：`e63d7e5eaf8627fc5c2cddf097a6b5b33b2af98c440a964ad3fe25d182fc3822`。外推/插值画像；32NPU/8SSU；ρmax=0.9797；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 50.96 | 50.70 | 6904.5 | 60.14 | 15909/24024 | 8/24024 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 96.02 | 96.20 | 4422.8 | 93.88 | 23479/24024 | 26/24024 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 99.91 | 99.93 | 4195.1 | 98.98 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 100.00 | 100.00 | 4193.2 | 99.02 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 99.93 | 99.92 | 4196.1 | 98.95 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 73.58 | 82.38 | 5386.0 | 77.09 | 23732/24024 | 32/24024 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 99.98 | 99.98 | 4195.8 | 98.96 | 24024/24024 | 32/24024 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 74.01 | 80.07 | 5371.6 | 77.30 | 23731/24024 | 32/24024 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 74.29 | 83.83 | 5355.8 | 77.53 | 23754/24024 | 32/24024 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4190.4 | 99.09 | 24010/24024 | 18/24024 | 32,32/32 |

## strong32_stripe6_feasible

冻结指纹：`4166ee5fd62b3cb65b9a95841205996e34d5a68a94ececfddaf6434b25fb8b0f`。外推/插值画像+C缩放；32NPU/6SSU；ρmax=0.9800；C倍率=1.249321；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 47.69 | 47.68 | 7090.5 | 58.93 | 13632/19392 | 8/19392 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 93.32 | 96.31 | 4512.8 | 92.59 | 18818/19392 | 20/19392 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 100.00 | 99.99 | 4194.1 | 99.63 | 19392/19392 | 32/19392 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 100.00 | 100.00 | 4189.0 | 99.75 | 19392/19392 | 32/19392 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 99.98 | 99.98 | 4194.9 | 99.61 | 19392/19392 | 32/19392 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 70.78 | 64.83 | 5806.2 | 71.97 | 19035/19392 | 32/19392 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 99.99 | 99.99 | 4192.6 | 99.66 | 19392/19392 | 32/19392 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 70.68 | 64.63 | 5811.5 | 71.90 | 19032/19392 | 32/19392 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 70.92 | 67.16 | 5729.5 | 72.93 | 19047/19392 | 32/19392 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4190.7 | 99.71 | 19368/19392 | 8/19392 | 32,32/32 |
| formal | Stall interchange / native/fixed | complete | 100.00 | 100.00 | 4191.8 | 99.68 | 19368/19392 | 8/19392 | 32,32/32 |

## raw32_local

冻结指纹：`acb863fc25c304971f5d93a81f91921c4745f75defee563e26da2ac630046fa7`。原始data参数；32NPU/8SSU；ρmax=0.9473；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 86.99 | 86.74 | 5779.8 | 79.58 | 2408/2736 | 16/2736 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 96.36 | 96.46 | 4932.6 | 93.24 | 2736/2736 | 32/2736 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 97.67 | 97.73 | 4845.3 | 94.93 | 2731/2736 | 27/2736 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 98.41 | 98.45 | 4718.8 | 97.47 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 98.53 | 98.51 | 4835.8 | 95.11 | 2733/2736 | 29/2736 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 10.94 | 12.81 | 15044.2 | 30.57 | 906/2736 | 0/2736 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 97.80 | 97.96 | 4828.0 | 95.26 | 2733/2736 | 29/2736 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 10.94 | 12.81 | 15078.8 | 30.50 | 887/2736 | 0/2736 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 10.94 | 12.84 | 15034.8 | 30.59 | 915/2736 | 0/2736 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4652.6 | 98.85 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | Stall interchange / native/fixed | complete | 100.00 | 100.00 | 4664.3 | 98.61 | 2736/2736 | 32/2736 | 32,32/32 |

## raw32_stripe

冻结指纹：`f8cefaef0685136f96a6be162c07fd931e2b3f24b8f64d73f22d80585a8e86f1`。原始data参数；32NPU/8SSU；ρmax=0.9473；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 87.57 | 87.26 | 5731.8 | 80.24 | 2419/2736 | 16/2736 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 92.68 | 92.67 | 5305.7 | 86.69 | 2710/2736 | 19/2736 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 100.00 | 100.00 | 4671.8 | 98.45 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 99.99 | 100.00 | 4663.6 | 98.62 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 100.00 | 100.00 | 4675.3 | 98.38 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 94.47 | 92.62 | 5153.2 | 89.25 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 100.00 | 100.00 | 4671.0 | 98.47 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 94.40 | 92.83 | 5151.9 | 89.27 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 94.52 | 93.05 | 5151.4 | 89.28 | 2736/2736 | 32/2736 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 100.00 | 100.00 | 4657.8 | 98.75 | 2736/2736 | 32/2736 | 32,32/32 |

## raw32_hash_feasible

冻结指纹：`87ccb9fb93f6574de285788b32f98775f7100b5cc29efb429b7a546a83d313c3`。原始data参数+C缩放；32NPU/8SSU；ρmax=0.9799；C倍率=1.040801；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 87.69 | 87.79 | 5788.2 | 81.37 | 2364/2656 | 16/2656 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 93.19 | 93.25 | 5314.0 | 88.63 | 2613/2656 | 26/2656 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 100.00 | 100.00 | 4861.6 | 96.88 | 2656/2656 | 32/2656 | 32,32/32 |

## raw32_allshort_overload

冻结指纹：`acd13805de760c420d39d7e223b418156ac6c949a2280f1435d29607c20eac0e`。原始data参数；32NPU/6SSU；ρmax=4.8479；C倍率=1.000000；容量条件=False。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 20.63 | 20.63 | 15124.0 | 20.63 | 0/10592 | 0/10592 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 20.62 | 20.62 | 15151.8 | 20.59 | 44/10592 | 0/10592 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 20.63 | 20.64 | 15126.7 | 20.62 | 3/10592 | 0/10592 | 32,32/32 |

## raw32_synchronized

冻结指纹：`6411f736cb51cd78111f4dd9888088204b34cc307a7f09ca317515f6181c85c4`。原始data参数；32NPU/6SSU；ρmax=0.7438；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 96.33 | 97.32 | 5527.2 | 92.95 | 659/768 | 32/768 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 96.38 | 97.91 | 5437.8 | 94.48 | 740/768 | 32/768 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 97.21 | 97.70 | 5334.4 | 96.31 | 720/768 | 32/768 | 32,32/32 |

## raw32_shuffled

冻结指纹：`94a6a48ff5017acea0fd831e36b59be0ee356ef9166ace5c437fde9a1e4e9bee`。原始data参数；32NPU/6SSU；ρmax=0.7438；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| screen | Baseline / 5ms/fixed | complete | 98.48 | 98.66 | 5362.0 | 95.82 | 678/768 | 32/768 | 32,32/32 |
| screen | Once / 5ms/fixed | complete | 97.80 | 98.14 | 5399.8 | 95.15 | 734/768 | 37/768 | 32,32/32 |
| screen | S3 / 5ms/fixed | complete | 98.55 | 98.67 | 5290.4 | 97.11 | 749/768 | 36/768 | 32,32/32 |
| formal | New once / 5ms/fixed | complete | 98.01 | 98.26 | 5354.5 | 95.95 | 746/768 | 37/768 | 32,32/32 |
| formal | S1 / 5ms/fixed | complete | 98.20 | 98.26 | 5315.8 | 96.65 | 736/768 | 36/768 | 32,32/32 |
| formal | S1 / 5ms/pipeline | complete | 99.54 | 99.54 | 5488.8 | 93.60 | 755/768 | 34/768 | 32,32/32 |
| formal | S2 / 5ms/fixed | complete | 98.12 | 98.75 | 5305.2 | 96.84 | 731/768 | 36/768 | 32,32/32 |
| formal | S2 / 5ms/pipeline | complete | 99.67 | 99.60 | 5480.9 | 93.74 | 756/768 | 35/768 | 32,32/32 |
| formal | S3 / 5ms/pipeline | complete | 99.66 | 99.61 | 5482.4 | 93.71 | 756/768 | 35/768 | 32,32/32 |
| formal | A deadline / native/fixed | complete | 99.09 | 99.31 | 5237.9 | 98.09 | 740/768 | 32/768 | 32,32/32 |

## raw32_shuffled_seed7

冻结指纹：`44354531755633a190984abc2ff04218acab280b1fad2aed11d4b2e2b395399b`。原始data参数；32NPU/6SSU；ρmax=0.7438；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| holdout | Baseline / 5ms/fixed | complete | 98.40 | 98.38 | 5367.0 | 95.73 | 660/768 | 32/768 | 32,32/32 |
| holdout | Once / 5ms/fixed | complete | 98.11 | 98.49 | 5389.4 | 95.33 | 735/768 | 34/768 | 32,32/32 |
| holdout | New once / 5ms/fixed | complete | 97.94 | 98.37 | 5347.0 | 96.09 | 748/768 | 34/768 | 32,32/32 |
| holdout | S3 / 5ms/pipeline | complete | 99.68 | 99.61 | 5520.4 | 93.07 | 756/768 | 36/768 | 32,32/32 |

## raw32_shuffled_seed123

冻结指纹：`b5e74fda88c0242e779343a68b47d7c9ae872344f2ae757706a80bb6a260efc1`。原始data参数；32NPU/6SSU；ρmax=0.7438；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| holdout | Baseline / 5ms/fixed | complete | 97.85 | 98.60 | 5403.5 | 95.08 | 654/768 | 31/768 | 32,32/32 |
| holdout | Once / 5ms/fixed | complete | 97.74 | 98.25 | 5419.3 | 94.81 | 730/768 | 32/768 | 32,32/32 |
| holdout | New once / 5ms/fixed | complete | 97.85 | 98.22 | 5361.2 | 95.83 | 749/768 | 32/768 | 32,32/32 |
| holdout | S3 / 5ms/pipeline | complete | 99.66 | 99.55 | 5480.5 | 93.75 | 754/768 | 33/768 | 32,32/32 |

## small_alone32

冻结指纹：`d53462b296757e15986d2c894fc39457389035db93cb70e99ca0d44af88b3041`。外推/插值画像；32NPU/8SSU；ρmax=0.2226；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| holdout | Baseline / 5ms/fixed | complete | 100.00 | 100.00 | 3112.7 | 100.00 | 23584/23584 | 32/23584 | 32,32/32 |
| holdout | Once / 5ms/fixed | complete | 100.00 | 100.00 | 3112.8 | 99.99 | 23584/23584 | 32/23584 | 32,32/32 |

## long_alone32

冻结指纹：`dc9840036d0317355f9d24bdf22d9c53701b424922822db4dcab4997e93a687e`。原始data参数；32NPU/8SSU；ρmax=0.7521；C倍率=1.000000；容量条件=True。

| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |
|---|---|---|---:|---:|---:|---:|---|---|---|
| holdout | Baseline / 5ms/fixed | complete | 100.00 | 100.00 | 3844.9 | 99.33 | 448/448 | 32/448 | 32,32/32 |
| holdout | Once / 5ms/fixed | complete | 100.00 | 100.00 | 3845.1 | 99.33 | 448/448 | 32/448 | 32,32/32 |

复算：`python results/baseline_npu32_investigation/notes/build_all_strategy_table.py`。CSV中U/rate为0–1，delta_u为百分点；完整路径、结果SHA及源代码指纹保存在CSV/JSON。
