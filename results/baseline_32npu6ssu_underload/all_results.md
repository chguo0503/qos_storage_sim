**全部结果：固定 [2000,4000) ms；32 NPU / 6 SSU**

由 `build_overview.py` 从 `analysis.json.gz` 生成。利用率单位为 %，相对下降以同人口随机 Baseline 为参照；负数代表重排后提高。

“有效”要求源码/输入审计、主窗暖机与每卡混合、全运行逐事件名义容量检查全部通过。不能只看利用率一列。不同种子可能共享同一输入指纹，仍分别保留各自提交种子的运行。

| 输入标签 | 策略 | 设备 U | 请求等权 U | 相对随机下降 | 全程逐盘峰值 GiB/s | 暖窗/混合 | 欠载 | 有效 |
|---|---|---:|---:|---:|---:|---|---|---|
| [concurrency_l1024_seed7](runs/concurrency_l1024_seed7/baseline/concurrency_l1024_seed7_6aaac9eaa305_baseline_104671470b.json.gz) | baseline | 92.6319 | 89.8189 | — | 30.735146 | 通过 | 通过 | 通过 |
| [concurrency_l1024_seed7__exact_cohort2_p1](runs/concurrency_l1024_seed7__exact_cohort2_p1/baseline/concurrency_l1024_seed7__exact_cohort2_p1_2deab240d672_baseline_104671470b.json.gz) | baseline | 85.6352 | 84.9301 | 7.5532% | 40.124773 | 通过 | **失败** | **失败** |
| [concurrency_l1024_seed7__exact_cohort4_p1](runs/concurrency_l1024_seed7__exact_cohort4_p1/baseline/concurrency_l1024_seed7__exact_cohort4_p1_4132463058f4_baseline_104671470b.json.gz) | baseline | 83.6752 | 84.5811 | 9.6691% | 33.899092 | 通过 | 通过 | 通过 |
| [concurrency_l1024_seed7__exact_cohort4_p2](runs/concurrency_l1024_seed7__exact_cohort4_p2/baseline/concurrency_l1024_seed7__exact_cohort4_p2_0ef0671df434_baseline_104671470b.json.gz) | baseline | 85.0381 | 81.3441 | 8.1978% | 33.899092 | 通过 | 通过 | 通过 |
| [concurrency_l512_seed7](runs/concurrency_l512_seed7/baseline/concurrency_l512_seed7_315424332b28_baseline_104671470b.json.gz) | baseline | 95.7783 | 95.7463 | — | 33.620391 | 通过 | 通过 | 通过 |
| [concurrency_l512_seed7__exact_cohort4_p1](runs/concurrency_l512_seed7__exact_cohort4_p1/baseline/concurrency_l512_seed7__exact_cohort4_p1_c6da472dae99_baseline_104671470b.json.gz) | baseline | 87.3031 | 90.6352 | 8.8487% | 29.475179 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed19](runs/concurrency_l768_seed19/baseline/concurrency_l768_seed19_d4ec996aae12_baseline_104671470b.json.gz) | baseline | 92.0434 | 90.6480 | — | 33.265417 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed19__exact_cohort4_p1](runs/concurrency_l768_seed19__exact_cohort4_p1/baseline/concurrency_l768_seed19__exact_cohort4_p1_6dd2d87260e7_baseline_104671470b.json.gz) | baseline | 85.4746 | 85.7123 | 7.1366% | 33.393753 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed43](runs/concurrency_l768_seed43/baseline/concurrency_l768_seed43_6bc80f6ed00c_baseline_104671470b.json.gz) | baseline | 91.2970 | 90.1135 | — | 34.201465 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed43__exact_cohort4_p1](runs/concurrency_l768_seed43__exact_cohort4_p1/baseline/concurrency_l768_seed43__exact_cohort4_p1_6dd2d87260e7_baseline_104671470b.json.gz) | baseline | 85.5117 | 84.8788 | 6.3368% | 34.105710 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7](runs/concurrency_l768_seed7/baseline/concurrency_l768_seed7_42376d7b73a5_baseline_104671470b.json.gz) | baseline | 92.9229 | 91.5348 | — | 30.381565 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort2_p1](runs/concurrency_l768_seed7__exact_cohort2_p1/baseline/concurrency_l768_seed7__exact_cohort2_p1_ff68c7b52562_baseline_104671470b.json.gz) | baseline | 85.6778 | 88.0867 | 7.7969% | 33.096599 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort4_p1](runs/concurrency_l768_seed7__exact_cohort4_p1/baseline/concurrency_l768_seed7__exact_cohort4_p1_6dd2d87260e7_baseline_104671470b.json.gz) | baseline | 83.5963 | 84.5983 | 10.0369% | 34.136730 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort4_p1](runs/concurrency_l768_seed7__exact_cohort4_p1/new_once/concurrency_l768_seed7__exact_cohort4_p1_6dd2d87260e7_new_once_ea64cc31a1.json.gz) | new_once | 99.1409 | 99.9443 | — | 32.944469 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort4_p2](runs/concurrency_l768_seed7__exact_cohort4_p2/baseline/concurrency_l768_seed7__exact_cohort4_p2_645a54e5d482_baseline_104671470b.json.gz) | baseline | 87.6862 | 87.2767 | 5.6355% | 34.368384 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort4_p3](runs/concurrency_l768_seed7__exact_cohort4_p3/baseline/concurrency_l768_seed7__exact_cohort4_p3_a27586023d38_baseline_104671470b.json.gz) | baseline | 85.4416 | 87.9278 | 8.0510% | 34.237047 | 通过 | 通过 | 通过 |
| [concurrency_l768_seed7__exact_cohort4_p4](runs/concurrency_l768_seed7__exact_cohort4_p4/baseline/concurrency_l768_seed7__exact_cohort4_p4_1813c13e64bf_baseline_104671470b.json.gz) | baseline | 83.7939 | 85.0085 | 9.8243% | 34.368384 | **失败** | 通过 | **失败** |
| [raw_s2_l1_seed7](runs/raw_s2_l1_seed7/baseline/raw_s2_l1_seed7_921465fb2ec4_baseline_104671470b.json.gz) | baseline | 99.8775 | 99.8502 | — | 38.806344 | 通过 | 通过 | 通过 |
| [raw_s4_l1_seed7](runs/raw_s4_l1_seed7/baseline/raw_s4_l1_seed7_57c56a2e6655_baseline_104671470b.json.gz) | baseline | 99.9556 | 99.9732 | — | 37.691788 | 通过 | 通过 | 通过 |
| [raw_s4_l1_seed7__burst_all_p1](runs/raw_s4_l1_seed7__burst_all_p1/baseline/raw_s4_l1_seed7__burst_all_p1_aba426eb7641_baseline_104671470b.json.gz) | baseline | 98.1155 | 98.0019 | 1.8409% | 39.630698 | 通过 | 通过 | 通过 |
| [raw_s4_l1_seed7__cohort2_p1](runs/raw_s4_l1_seed7__cohort2_p1/baseline/raw_s4_l1_seed7__cohort2_p1_72896842779c_baseline_104671470b.json.gz) | baseline | 99.6814 | 99.6326 | 0.2743% | 39.630698 | 通过 | 通过 | 通过 |
| [raw_s4_l1_seed7__cohort4_p1](runs/raw_s4_l1_seed7__cohort4_p1/baseline/raw_s4_l1_seed7__cohort4_p1_4e4a0d6cc17a_baseline_104671470b.json.gz) | baseline | 99.5229 | 99.4118 | 0.4329% | 37.981991 | 通过 | 通过 | 通过 |
| [raw_s8_l1_seed7](runs/raw_s8_l1_seed7/baseline/raw_s8_l1_seed7_4a727938e3b9_baseline_104671470b.json.gz) | baseline | 99.9960 | 99.9987 | — | 36.554103 | 通过 | 通过 | 通过 |
| [raw_varied_seed7](runs/raw_varied_seed7/baseline/raw_varied_seed7_d806a3042657_baseline_104671470b.json.gz) | baseline | 99.9976 | 99.9990 | — | 37.422034 | 通过 | 通过 | 通过 |
| [synthetic_s150_l1_seed7](runs/synthetic_s150_l1_seed7/baseline/synthetic_s150_l1_seed7_d8d9f56d5f2b_baseline_104671470b.json.gz) | baseline | 98.2691 | 98.1790 | — | 23.289321 | **失败** | 通过 | **失败** |
| [synthetic_s60_l1_seed7](runs/synthetic_s60_l1_seed7/baseline/synthetic_s60_l1_seed7_32d4a460e2aa_baseline_104671470b.json.gz) | baseline | 95.3204 | 94.0035 | — | 28.911563 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__burst_all_p1](runs/synthetic_s60_l1_seed7__burst_all_p1/baseline/synthetic_s60_l1_seed7__burst_all_p1_b747370fae1c_baseline_104671470b.json.gz) | baseline | 96.4345 | 96.4007 | -1.1688% | 39.630698 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__cohort2_p1](runs/synthetic_s60_l1_seed7__cohort2_p1/baseline/synthetic_s60_l1_seed7__cohort2_p1_084bc0c08be2_baseline_104671470b.json.gz) | baseline | 94.8903 | 93.8475 | 0.4512% | 33.483107 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__cohort4_p1](runs/synthetic_s60_l1_seed7__cohort4_p1/baseline/synthetic_s60_l1_seed7__cohort4_p1_e292d6cda939_baseline_104671470b.json.gz) | baseline | 95.2212 | 94.1955 | 0.1041% | 31.092481 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_burst_p1](runs/synthetic_s60_l1_seed7__exact_burst_p1/baseline/synthetic_s60_l1_seed7__exact_burst_p1_ee9011a35697_baseline_104671470b.json.gz) | baseline | 94.5000 | 94.9816 | 0.8607% | 39.630698 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_burst_p2](runs/synthetic_s60_l1_seed7__exact_burst_p2/baseline/synthetic_s60_l1_seed7__exact_burst_p2_1e51e1119ed1_baseline_104671470b.json.gz) | baseline | 97.4742 | 98.8393 | -2.2595% | 39.630698 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort2_p1](runs/synthetic_s60_l1_seed7__exact_cohort2_p1/baseline/synthetic_s60_l1_seed7__exact_cohort2_p1_934fea821da3_baseline_104671470b.json.gz) | baseline | 89.4309 | 88.3709 | 6.1786% | 26.174275 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort2_p2](runs/synthetic_s60_l1_seed7__exact_cohort2_p2/baseline/synthetic_s60_l1_seed7__exact_cohort2_p2_90b194c99482_baseline_104671470b.json.gz) | baseline | 90.2334 | 90.7702 | 5.3368% | 26.174275 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort4_p1](runs/synthetic_s60_l1_seed7__exact_cohort4_p1/baseline/synthetic_s60_l1_seed7__exact_cohort4_p1_1b5f382b15b4_baseline_104671470b.json.gz) | baseline | 87.0264 | 86.7807 | 8.7012% | 27.446061 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort4_p2](runs/synthetic_s60_l1_seed7__exact_cohort4_p2/baseline/synthetic_s60_l1_seed7__exact_cohort4_p2_c4398688a698_baseline_104671470b.json.gz) | baseline | 85.9027 | 85.1020 | 9.8800% | 27.368843 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort4_p3](runs/synthetic_s60_l1_seed7__exact_cohort4_p3/baseline/synthetic_s60_l1_seed7__exact_cohort4_p3_247ae4b15bfb_baseline_104671470b.json.gz) | baseline | 87.7548 | 87.9217 | 7.9370% | 27.368843 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__exact_cohort4_p4](runs/synthetic_s60_l1_seed7__exact_cohort4_p4/baseline/synthetic_s60_l1_seed7__exact_cohort4_p4_1d2041527aa1_baseline_104671470b.json.gz) | baseline | 86.4459 | 87.4250 | 9.3102% | 27.446061 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__phase_cohort3_p1](runs/synthetic_s60_l1_seed7__phase_cohort3_p1/baseline/synthetic_s60_l1_seed7__phase_cohort3_p1_42663127091a_baseline_104671470b.json.gz) | baseline | 89.3084 | 88.2914 | 6.3072% | 30.776840 | 通过 | 通过 | 通过 |
| [synthetic_s60_l1_seed7__phase_cohort4_bias05_p1](runs/synthetic_s60_l1_seed7__phase_cohort4_bias05_p1/baseline/synthetic_s60_l1_seed7__phase_cohort4_bias05_p1_ae44b97cb448_baseline_104671470b.json.gz) | baseline | 88.6671 | 87.5187 | 6.9799% | 30.667450 | 通过 | 通过 | 通过 |

共 38 份输入 manifest，39 次策略运行，36 次满足全部主窗与名义容量条件。源码/执行审计通过 39 次。

仅 `concurrency_l768_seed7__exact_cohort4_p1` 的 Baseline 同时满足所有条件与相对随机降低至少 10% 的目标。所有结果并不构成生产分布的独立随机样本。

[主报告](report.md) · [详细审计](analysis.json.gz) · [全部 250 ms 子窗口及指标](summary.csv)
