# 五种子策略对比

- [2–4 秒 warm 窗口](warm_2_4s_npu_utilization.png)
- [2–20 秒长期窗口](long_2_20s_npu_utilization.png)

两张独立 PNG，横轴为 seed 7/19/43/67/101。蓝色圆点为 Baseline，橙色方点为流量分配策略（内部目录名 once）。每个点标明实际利用率；灰色线只连接同一 seed 的两个策略。

32 NPU，8 SSU，每盘 40 GiB/s，8 层。长请求总长 384K、miss 1024；短请求总长 10K、miss 128；请求数之比 1:24。每卡使用完整独立 Random 队列，两策略使用相同 manifest。长、短请求的 C 均由 data 线性拟合外推，属于模型构造，不是这些输入长度的硬件实测。

利用率 = 该窗口全部 NPU 的实际计算时间 /（32 × 窗口长度）。两个窗口的全部10个运行均满足32卡全程活跃、每卡都计算过长短请求。

图上统计为5个种子等权均值、样本标准差（分母 n−1=4），以及同种子策略差的均值。百分点和百分比不同：例如87%→94%为增加7个百分点。阴影80–90%是本研究关注区间，不是SLO或置信区间；纵轴放大为78–100%。

seed7为探索案例，19/43/67/101为事先固定的确认种子；Once确认是在看到单独seed7收益后启动的阶段。五种子汇总保留全部种子，不声称整个搜索过程从未适应性选候选。近容量理想平均负载不等于逐盘逐时严格欠载。

| 窗口 | Baseline均值 ± 样本SD | 流量分配策略均值 ± 样本SD | 同seed差均值 |
|---|---:|---:|---:|
| [2, 4) 秒 | 87.1287% ± 0.3784 pp | 94.7598% ± 1.0389 pp | +7.6311 pp |
| [2, 20) 秒 | 87.3140% ± 0.1103 pp | 94.7048% ± 0.1315 pp | +7.3907 pp |

重建：`python results/baseline_random_near_capacity_20260914/figures/five_seed_strategy_comparison/render_five_seed.py`。

绘图脚本只读取10份canonical `analysis.json`，在 [plot_data.json](plot_data.json) 和 [checks.json](checks.json) 绑定各文件SHA，并验证绘图前后不变。未读取或绑定会继续更新的key_results.json。脚本从每卡计算时间独立复核所有20个运行/窗口利用率，也检查成对manifest摘要相同。独立洗牌属于已冻结实验输入协议；本脚本不再次解析请求队列。

来源：
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/baseline/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/baseline/analysis.json) — SHA256 `ff6818f42a5e246c1a09d03d08122a33c41b69c87a5c0c06dbaed36e3421a394`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/analysis.json) — SHA256 `7f41313d90196d382dce6fa57bf7102922ac4aff1acb439f0f1e07019a99bc25`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed19/baseline/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed19/baseline/analysis.json) — SHA256 `94e17cb4af9a249fefdb7ba789a38ce5cab1e9830a299afd4d903287170a45fb`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed19/once/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed19/once/analysis.json) — SHA256 `561c734c61f3a1f395161ebc64d57ea6abbd8d82ce1c5032b58698e6f4fc3dfb`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed43/baseline/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed43/baseline/analysis.json) — SHA256 `ea174892a13f3a3792cf851f6324c5dafa969182a571f9180eb3cf969450ead5`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed43/once/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed43/once/analysis.json) — SHA256 `7bc25c45208adace0b082b70b37998824c1ae48a6a1bcab5140ad3e428cd729b`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed67/baseline/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed67/baseline/analysis.json) — SHA256 `70b7ff79b787313765cd84d15a0eee964a73cc16c7f27127ebbc275c6e7ee73c`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed67/once/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed67/once/analysis.json) — SHA256 `0d465ea34e22ccc388fec0b82cdc9ca7cef10483da3f3f4ad85d2987454d4b38`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed101/baseline/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed101/baseline/analysis.json) — SHA256 `c7634e67fce128b702c6b391e5bc28006a923cfd8757406f44fa91d54b9e8b5e`
- [runs/context8_L384m1024_S10m128_ssu8_h22000_seed101/once/analysis.json](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed101/once/analysis.json) — SHA256 `96a5099396f931f05b701208b78e0bad39e6ea77741aac244bed772b301a1795`
