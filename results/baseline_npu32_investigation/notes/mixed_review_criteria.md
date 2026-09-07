# 多短画像随机序列：独立审计与结论更新标准

本节预先约定口径，避免看见结果后只挑支持旧结论的指标。新输入回答“每卡都连续处理多种不同短流与长流”的问题；不能把它称作之前固定角色低利用率输入的同输入复测。

## 输入有效性

- 每个 NPU 至少 4 种 `(seq_len_k, nql)` 画像，其中至少 3 种被明确列为短画像。短类同时报告各自计算时间与读量，不能只用统一的 `short` 标签掩盖内部差别。
- 这里 `short` 是 spec 指定的较短画像集合，不等于模拟器的 `SS` 类别：其阈值为 seq<=80K、NQL>=512 视作长NQL，所以 32K512、64K512 实为 `SL`，32K128、48K256 为 `SS`。分组标签不作为策略路径因果证据。
- 每卡完整请求多重集具有相同画像配额；使用独立 RNG 打乱整个序列，不能只打乱一副小牌后循环。独立重新生成每卡序列并核对元数据中的 SHA256。
- 固定请求 ID、原始 NPU、画像、计算时间、到达时间、逐块 SSU placement。策略间按完整 input fingerprint 配对，并检查每种画像的完整请求 ID 集相同。随机种子不同属于不同输入。
- 计算、KV 体积与 padding 不因策略改变；来源为原始 data 行与 1K 外推的输入分开呈现。原始画像参数不等于真实生产分布，画像频率、排序与 t=0 积压仍是人工构造。
- 独立按 manifest 累加 `C_i = sum(C_request)` 与 `V_is = sum(read GiB on disk s)`，复算 `rho_s = sum_i(1000 V_is/C_i)/40` 和 `rho_link_i = 1000 sum_s(V_is)/C_i/50`。按 `sum(short C)/sum(all C)` 算计算占比，另给请求数占比。均值条件可行只排除平均容量不足，不能保证瞬时截止期可行。
- 使用同一冻结核心代码、策略参数、5ms 收集器和 fixed 绑定。原生 native、pipeline 重绑定属于另一个机制对照，不能混为仅 IO 排序不同。

## 输出有效性

- 所有输入的两公共窗口保持 `[1000,2000]`、`[2000,3000]` ms。独立积分每层 compute 区间和每请求 active 区间；计算 `U=sum(compute)/(32*1000)`，同时记录每卡 U、active、idle。仅在所有卡整个窗口 active 时，低 U 才直接解释为等待/阻塞而非输入耗尽。
- 不通过挑“恰好在窗口完成”的请求比较策略。画像与短类指标使用 manifest 确定的**完整同请求 ID cohort**；每策略必须全部完成。报告 `sum(own compute)/sum(completion-admission)`、每请求比例均值/分位、stall、admission SLO 与 arrival SLO。
- SLO 固定为 `completion-admission <= 1.5*own_compute + 1e-9 ms`，arrival 版本仅把 admission 换为 arrival。t=0 积压下 arrival SLO 会随序列位置恶化，不代替调度后服务品质指标。
- 窗口画像时间权重独立积分：各画像占 fleet time 的 compute 与 active 份额、其 active 内 compute 比。全 fleet U 较高时仍检查每一种短画像。
- full makespan 使用完整 population。报告公共窗 U 与 makespan 的方向冲突，不能只取有利项。missing / incomplete 明确 pending，不当作零。

## 如何更新结论

1. 若新输入 baseline 在两个种子、两个 all-active 窗口都较高，就明确承认“固定长短卡坏例不能直接推广到这类持续混合输入”，并给出实际 U；不能因旧坏例存在而坚持新输入必差。
2. 若 fleet U 较高而某些短画像的完整 cohort compute fraction / SLO 很低，则结论拆开写：总体计算忙碌较好，短请求服务质量仍被时间权重掩盖。必须列画像、计算份额与受损请求数。
3. 若 mean rho 可行、all-active 的新混合输入依旧较低，并且至少两个固定种子同向，才把“baseline 会在每卡多画像混合下受损”作为该输入族的复现结论。数值描述优先，不把任意 U 阈值当定理。
4. 策略结论逐输入、逐指标配对：改善 fleet U 不自动表示每短画像改善；改善短请求不自动表示 makespan 更短。Once / New Once 低于 baseline 的结果同样保留。两个种子仅说明有限的复现性，不等同总体统计显著性。
5. 过载输入保留作边界对照，并明确物理容量约束；不把过载造成的低 U 单独归因于 baseline 调度。
6. 另设 `mixed_strict_ss` 语义控制时，要求三个短画像实际均为 SS（32K128、48K256、64K128），两长均为 LL，并独立输出。其新画像/配额/短C份额不同，不能称作原 size-varied 输入的只改标签对照。若其中不同 SS 画像仍表现不同或产生策略代价，则说明原观察不只来自分析组混含 SS/SL；若差异消失，也应明确更新结论。无论结果如何，都不能仅凭类别标签断言具体队列机制。

上述主实验标准于2026-09-07首批结果出现前约定；严格SS条款在该控制结果出现前补充。最终主实验36/36与严格SS6/6结果、完整CSV/JSON/图分别见 `mixed_independent_review.md`、`mixed_strict_ss_analysis.md` 及对应 audit 目录。
