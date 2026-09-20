# 为什么盘满时，2秒窗口的NPU利用率仍不同

对象是原full持续过载实验，而非后续near35实验。三种子等权平均：

| 统计范围 | Baseline U | 流量分配 U | 差值（百分点） |
|---|---:|---:|---:|
| [2,4)秒 | 62.384978% | 64.440747% | +2.055768 |
| [2,6)秒 | 66.185247% | 67.291082% | +1.105835 |
| 完整相同请求集合 | 63.496313% | 63.728274% | +0.231961 |

两策略的[2,4)与[2,6)窗口均为三盘各40GiB/s满速。2秒内总盘读取量固定为240GiB，但窗口NPU利用率是窗口内真正计算卡时间除以32×2秒；并不是盘吞吐利用率。

不同请求的C/V不同。L3改变各层的数据到齐、计算启动和后续请求接纳时刻，同一短窗内执行的请求组合以及窗界已读未算/前窗已预取的数据可能不同。因此相同窗口读出字节不要求计算卡时间相同。这里说明可导致差异的机制，并未将2.055768个百分点逐项归因。

完整固定请求集合的计算卡时间为274.8702965卡秒，总读取量1579.703125GiB。三盘总容量120GiB/s，对应理想读完下限13.16419秒，整批平均U的容量上界约65.25046%。实际三种子平均总完成时间为Baseline13.527906秒、流量分配13.479954秒，差约48ms。

所以2秒窗口的+2.06个百分点不能解读为突破带宽容量上限，也不能直接视为长期稳定吞吐增益。完整集合仅+0.23个百分点。稳态下若请求完成组合固定、在途数据不持续积累且盘始终满速，平均计算吞吐会受相同的读量/计算工作比约束。

来源：
https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/threeway_macro_summary.csv
https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/input_profile_summary.csv
https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/metrics.py
https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/run_trial.py
