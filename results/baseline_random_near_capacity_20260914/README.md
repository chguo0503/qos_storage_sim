# 接近容量时，Baseline Random 是否会持续低利用率

本次完成90个科学案例，只测试 Random，没有新增 Ordered 实验。主构造组五种子在2–20秒的Baseline平均利用率为87.31%，同输入流量分配策略为94.70%。输入为每卡长384K/miss1024与短10K/miss128按1:24独立完整打乱；长、短计算时间均由data外推，不能当作对应长度的硬件实测。

32 张 NPU，主要构造结果使用8块SSU，原始data负载对照使用3块，另有4/6块探索；每盘 40 GiB/s，每卡接收链路 50 GiB/s，每请求 8 层。沿用原有跨请求首层预取、FIFO 数据面及 5 ms 共享采样。所有请求在仿真 t=0 到达并固定绑定 NPU；每卡准备至少 22 秒纯计算工作，独立打乱整个有限队列。各卡没有运行时同步、迁移、延时注入或种子筛除。

主窗口沿用 `[2,4)` 秒，同时报告 `[2,20)` 秒及连续 2 秒分窗，另有扩大工作量后的 `[2,60)` 和 `[20,60)` 秒验证。请求一直可用与 NPU 一直计算是不同条件；已从日志逐卡核验实际长、短计算，所有不满足覆盖条件的案例原样保留。

这里“接近容量”指整批输入按纯计算时间折算的平均需求接近相应拓扑的总容量（3/4/6/8盘为120/160/240/320 GiB/s），不表示每盘、每个时刻都欠载。实际 current V/C 峰值及超限时间、真实 SSD 忙时分别统计。若仍要求严格逐盘逐时欠载，不能把本轮整批均值接近容量的结果作为证明。

- [当前研究报告与主要输入说明](report.md)
- [重点结果：五种子、同输入策略配对、65秒工作量](key_results.md)
- [五种子策略对照：warm与长窗两张独立图](figures/five_seed_strategy_comparison/README.md)
- [主构造组延长输入：全部29个两秒分窗](long_horizon/context384/figures/long_horizon/random_utilization_all_2s_windows.png)
- [主构造组延长输入：逐窗U、SLO与覆盖](long_horizon/context384/long_horizon_comparison.md)
- [与教程公式逐步对应](report_core.md)
- [同输入下，短内部等待减少如何提高整机U](figures/context384_strategy_comparison/README.md)
- [384K主组：Baseline的32卡带宽、时序与内部层放大](runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/baseline/figures/README.md)
- [384K主组：流量分配策略的对应图](runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/figures/README.md)
- [读取尺度、B、等待与U的四张独立图](figures/context_scale/README.md)
- [固定研究计划与源文件哈希](study_plan.json)
- [候选配方和数学筛选](candidate_math.md)
- [已完成结果：逐组、逐种子、SLO](comparison.md)
- [公式与实际等待核对](math_result_table.md)
- [10K–16K短请求的透明外推说明](constructed_candidates.md)
- [原始数据搜索的失败与后续推导](math_followup.md)
- [真实短层FIFO等待局部图](runs/main512_ssu3_h22000_seed7/baseline/figures/internal_median_wait/baseline_random_short_internal_cycle.png)
- [前三组输入独立审计](input_audit.json)
- [模拟器与运行环境审阅](audit_plan.md)
- [通用分析方法](analysis说明.md)
- [输入生成与完整仿真脚本](experiment.py)
- [90个科学案例的最终核验](final_research_audit.json)
- [数学复核与最终确认](mathematical_review.md)
- [项目清理记录](cleanup_execution.md)

种子 7 用于首轮探索，19、43、67、101 用于已预定主组的确认。所有已运行案例、科学条件不满足项和运行失败均保留。扩展候选单独标记，避免把探索挑出的最低值当成不偏的平均结果。

两组优先原始数据配方：

| 名称 | 每卡请求数量比 | 理想平均需求 | 理想负载率 |
|---|---|---:|---:|
| main512 | 192K/miss4096 : 32K/miss512 = 1:9 | 119.6775 GiB/s | 99.7313% |
| main1024 | 192K/miss4096 : 32K/miss1024 = 1:18 | 120.2364 GiB/s | 100.1970% |

以上总输入均来自 `data` 原始行，不是命中长度加 miss。每层读取量和计算时间保持原值，没有读取填充或人为缩放计算时间。

上述两组用原始数据。新增 `fit*` 组的长请求仍来自原始 data；10K、12K、16K短请求的计算时间由 data 固定 miss=128 的32K–200K行作线性外推，明确标记为构造画像，不能称为真实10K硬件测量。原始和外推结果分别报告，不混算均值。

SLO 沿用此前的接纳后 prefill 完成代理：窗口内接纳的请求跟踪至完成，检查 `完成时间−接纳时间 <= 倍数×8×该画像单层计算时间`，分别报告 ×1.5 和 ×2。它不包含接纳前排队，不是用户到达后实际首 token 的端到端 TTFT。

新增context组保持8盘和平均负载接近容量，改变长请求读取尺度；384K组进行原定四个种子确认和同输入Once对照。新增lowB4组改为4盘、低B短计算，是独立机制假设。所有外推边界见报告，不能把它们称为data现成行。约105%/110%原始data负载敏感性另外标出容量因素。更长验证使用新>=65秒纯计算人口的完整shuffle，不视作22秒队列的同一前缀。

异质输入另外用10K/12K/14K短请求和352K/384K/416K长请求做总C/V配平对照。短异质组的后续四个种子及seed7 Once、主组的后续Once四种子，均在看到对应pilot后才启动，保留这种自适应研究顺序，不冒称所有矩阵在开头预注册。

原始trace超过100MiB的一例使用[无损分片与实际往返校验](large_artifact_roundtrip_audit.json)发布。已有统计和PNG可直接读；恢复逐块trace的方法见项目根[README](../../README.md)。
