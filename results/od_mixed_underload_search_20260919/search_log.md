# 全部正式结果与短试验索引

完整排空 25 次；原仿真 4 秒筛选 61 次（60 种输入）；另有 1 次截断口径校验。

正式结果均核对结果/输入文件 SHA、输入指纹与运行器完整校验。筛选不提供 SLO，不代表长期结果。

“全程欠载”包括启动和排空，不表示全程所有卡活跃。利用率和 SLO 以下均为 warm [2,4)；全程 U 不用于证明稳态退化。

| 完整实验 | U | SLO×1.5 | warm欠载 | 全程欠载 | 每卡混合 | 输入 |
|---|---:|---:|---|---|---:|---|
| boundary_od_seed7_local | 97.9693% | 100.0000% | True | True | 32/32 | data原行 |
| fixedssu1_safe38_od_local | 97.2811% | 96.3724% | True | True | 32/32 | 合成+选址 |
| fixedssu1_safe38_once_remote | 99.5461% | 100.0000% | True | True | 32/32 | 合成+选址 |
| fixedssu1_safe38_ordinary_addresses_od_remote | 97.5349% | 96.0736% | False | False | 32/32 | 合成 |
| fixedssu1_safe38_random_od_remote | 99.4861% | 100.0000% | False | False | 32/32 | 合成+选址 |
| fixedssu1_safe38_spread_od_remote | 97.5070% | 100.0000% | True | False | 32/32 | 合成+选址 |
| fixedssu1_safe39_od_local | 97.0112% | 98.8138% | True | True | 32/32 | 合成+选址 |
| fixedssu1_tight40_od_local | 96.9371% | 98.9461% | True | True | 32/32 | 合成+选址 |
| native_phase_lock_a101_b0995_od_local | 94.0580% | 73.7530% | True | True | 32/32 | 合成+选址 |
| native_phase_lock_a101_b0995_once_remote | 99.4978% | 100.0000% | True | True | 32/32 | 合成+选址 |
| native_phase_lock_a102_od_local | 95.9035% | 86.8794% | True | True | 32/32 | 合成+选址 |
| native_phase_lock_a102_once_remote | 99.0648% | 100.0000% | False | False | 32/32 | 合成+选址 |
| raw32_128_a3b1_p3_od_local | 98.8225% | 100.0000% | True | False | 32/32 | data原行 |
| raw32_128_a4b1_p3_od_remote | 99.1395% | 100.0000% | True | True | 32/32 | data原行 |
| raw32_128_a5b1_p3_od_remote | 99.0421% | 100.0000% | True | True | 32/32 | data原行 |
| raw32_200_a5b1_p3_od_remote | 98.5624% | 100.0000% | True | True | 32/32 | data原行 |
| raw32_200_a7b1_p3_od_remote | 98.0448% | 100.0000% | True | True | 32/32 | data原行 |
| raw32_200_proxy_best_od_local | 99.0575% | 100.0000% | True | True | 32/32 | data原行 |
| raw32_32_a3b1_p4_od_local | 99.6835% | 100.0000% | False | False | 32/32 | data原行 |
| raw_boundary_32m2048_200m4096_sync_once_remote | 98.4911% | 100.0000% | True | True | 32/32 | data原行 |
| synth_phase180_1039_c103_od_remote | 98.7913% | 100.0000% | False | False | 32/32 | 合成 |
| synth_phase180_1039_od_local | 99.0127% | 100.0000% | False | False | 32/32 | 合成 |
| synth_phase80_1210_c103_od_remote | 98.0730% | 97.8622% | False | False | 32/32 | 合成 |
| synth_phase80_1210_od_local | 98.1512% | 98.7500% | False | False | 32/32 | 合成 |
| synth_phase80_1210_once_remote | 98.7862% | 100.0000% | False | False | 32/32 | 合成 |

## 4秒原仿真筛选

“通过”要求前4秒逐盘严格欠载、warm全部32卡活跃且都实际计算过A/B。先后调参属于自适应搜索，不能把最优值当随机总体均值。

| 短试验 | warm U | warm欠载 | 前4秒欠载 | warm混合卡数 | 通过 |
|---|---:|---|---|---:|---|
| native_grid1_00 | 97.8274% | True | False | 32 | False |
| native_grid1_01 | 95.9035% | True | True | 32 | True |
| native_grid1_02 | 97.3822% | True | False | 32 | False |
| native_grid1_03 | 98.3164% | True | True | 32 | True |
| native_grid1_04 | 97.4607% | True | False | 32 | False |
| native_grid1_05 | 97.3993% | True | True | 32 | True |
| native_grid1_06 | 97.3795% | True | False | 32 | False |
| native_grid1_07 | 98.0375% | True | False | 32 | False |
| native_grid1_08 | 95.0113% | True | False | 32 | False |
| native_grid1_09 | 97.3729% | True | False | 32 | False |
| native_grid1_10 | 97.2509% | True | False | 32 | False |
| native_grid1_11 | 97.5049% | True | False | 32 | False |
| native_grid1_12 | 95.8465% | False | False | 32 | False |
| native_grid1_13 | 97.3018% | True | False | 32 | False |
| native_grid1_14 | 93.5863% | False | False | 32 | False |
| native_grid1_15 | 97.1396% | True | False | 32 | False |
| native_grid1_16 | 95.3041% | False | False | 32 | False |
| native_grid1_17 | 96.6539% | True | False | 32 | False |
| native_grid1_18 | 96.9253% | True | True | 32 | True |
| native_grid1_19 | 97.9907% | True | False | 32 | False |
| native_grid1_20 | 96.9275% | True | True | 32 | True |
| native_grid1_21 | 98.2216% | True | True | 32 | True |
| native_grid2_00 | 94.0580% | True | True | 32 | True |
| native_grid2_01 | 97.2784% | True | True | 32 | True |
| native_grid2_02 | 97.0039% | True | True | 32 | True |
| native_grid2_03 | 94.2659% | True | True | 32 | True |
| native_grid2_04 | 98.1763% | True | True | 32 | True |
| native_grid2_05 | 96.1770% | True | True | 32 | True |
| native_grid2_06 | 98.7024% | True | True | 32 | True |
| native_grid2_07 | 97.0897% | True | True | 32 | True |
| native_grid2_08 | 97.4093% | True | True | 32 | True |
| native_grid2_09 | 97.8576% | True | True | 32 | True |
| native_grid2_10 | 96.9241% | True | True | 32 | True |
| native_grid2_11 | 97.1316% | True | False | 32 | False |
| native_grid3_00 | 96.1389% | True | True | 32 | True |
| native_grid3_01 | 95.8362% | True | True | 32 | True |
| native_grid3_02 | 96.1571% | True | True | 32 | True |
| native_grid3_03 | 97.3155% | True | True | 32 | True |
| native_grid3_04 | 96.3712% | True | True | 32 | True |
| native_grid3_05 | 95.8746% | True | True | 32 | True |
| native_grid3_06 | 96.7813% | True | True | 32 | True |
| native_grid3_07 | 96.3332% | True | True | 32 | True |
| native_grid3_08 | 94.2067% | True | True | 32 | True |
| native_grid3_09 | 97.6646% | True | True | 32 | True |
| native_grid3_10 | 97.4526% | True | True | 32 | True |
| native_grid3_11 | 97.0039% | True | True | 32 | True |
| native_grid3_12 | 96.2072% | True | True | 32 | True |
| native_grid3_13 | 98.0839% | True | True | 32 | True |
| native_grid3_14 | 96.3055% | True | True | 32 | True |
| native_grid3_15 | 96.4471% | True | True | 32 | True |
| native_grid3_16 | 95.8711% | True | True | 32 | True |
| validation_safe38 | 97.2811% | True | True | 32 | True |
| larger180_grid_00 | 97.1062% | True | True | 32 | True |
| larger180_grid_01 | 96.9560% | True | True | 32 | True |
| larger180_grid_02 | 96.7220% | True | True | 32 | True |
| larger180_grid_03 | 97.0038% | True | True | 32 | True |
| larger180_grid_04 | 96.6015% | True | True | 32 | True |
| larger180_grid_05 | 96.1322% | True | True | 32 | True |
| larger180_grid_06 | 96.1150% | True | True | 32 | True |
| larger180_grid_07 | 98.6485% | True | True | 32 | True |
| larger180_grid_08 | 98.0505% | True | False | 32 | False |
| larger180_grid_09 | 96.9761% | True | False | 32 | False |

连续服务代理是候选生成工具，单独保存在 ideal_search；其预测不计入正式结果。额外的静态尾请求校准概念验证如存在，单独报告，不混入上述4秒筛选表。
