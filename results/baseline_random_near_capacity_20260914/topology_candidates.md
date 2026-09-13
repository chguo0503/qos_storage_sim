# 改变盘数的二级候选：6 SSU 与 8 SSU

主实验仍为 3 SSU。本文件将盘数变化作为单独的拓扑敏感性实验；同时改变了容量与请求配比，不能把其收益或损失解释成只改变 QoS 策略。没有运行新仿真，筛选没有使用已观测 U 排名。

固定 32 NPU、每盘 40 GiB/s、NPU 链路 50 GiB/s、8 层、batch=1、原有跨请求首层预取。所有画像直接来自 data，最小总输入 32K，miss 至少 128，单层按 176 KiB 整块读取，Bi<50。每卡相同数量比例，再独立完整打乱队列。

## 先算比例和需要发生的等待

```text
S = 盘数 * 40
rho = 32*(nL*VL+nS*VS) / [S*(nL*CL+nS*CS)]
f_short = nS*CS / (nL*CL+nS*CS)
若长类不 stall：U ≈ 1 / (1 + f_short*w/CS)
要 U≈80%：所有短层平均额外等待 w≈0.25*CS/f_short
```

w 包括不等待的短层。这个近似要求长期完成比例稳定，忽略启动与请求边界；它回答需要多大等待，不预言实际会出现这么多等待。

扫描范围是 2 种盘数×2 种长画像×4 种短画像，共 16 个组合；整数配比搜索 nL=1…64、nS=1…512、互质。若存在 |rho−1|≤0.005 的配比，先取计数和最小的，再取误差小的；否则取最接近 rho=1 的。先用代数验证正比例可行，避免把搜索上限误当成数学不可行。

## 最多八组可运行候选

| ID | 盘数 | L miss / S 总K及miss | L:S | rho | f_short | U80 所需平均等 ms |
|---|---:|---|---|---:|---:|---:|
| topo6_m2048_s32m128 | 6 | 200K/2048；32K/128 | 4:31 | 0.997578 | 11.43% | 2.576 |
| topo8_m2048_s32m128 | 8 | 200K/2048；32K/128 | 3:43 | 1.003530 | 19.27% | 1.528 |
| topo6_m2048_s32m256 | 6 | 200K/2048；32K/256 | 2:19 | 0.997041 | 21.15% | 2.361 |
| topo8_m2048_s32m256 | 8 | 200K/2048；32K/256 | 2:39 | 1.000273 | 35.51% | 1.406 |
| topo6_m2048_s32m512 | 6 | 200K/2048；32K/512 | 1:18 | 0.996972 | 48.51% | 1.908 |
| topo8_m2048_s48m128 | 8 | 200K/2048；48K/128 | 1:9 | 1.000960 | 16.15% | 2.343 |
| topo6_m4096_s32m256 | 6 | 200K/4096；32K/256 | 1:29 | 1.003394 | 29.07% | 1.718 |
| topo8_m4096_s32m256 | 8 | 200K/4096；32K/256 | 1:51 | 1.002219 | 41.89% | 1.192 |

选择顺序覆盖同一画像在 6/8 盘的变化、较紧/较松短截止时间、48K 高带宽短类、两种长计算时间。没有声称这是全空间中最差的八组。

200K/2048 的整请求纯计算约 565.924 ms；200K/4096 为 1130.686 ms。后者连续两个长请求就可能覆盖整个 warm，所以重点先验证前者。即使前者也必须实测每卡 warm 长短覆盖，不能保证独立随机一定满足。

8 盘搭 200K/2048 与 32K/512 可以让短计算贡献达到约 81%，但平均每个长请求对应的纯计算工作超过 3 秒，warm 内很多卡可能没有长类；因此它保留在完整扫描记录中，而不列入优先启动的八组。

## 盘数变多，有两个相反作用

为了仍然 rho≈1，需要更多短请求，因此 f_short 增大：短类一旦等待，对整机影响更明显。但每个长读取跨更多盘并行处理，其服务突发缩短：同样的排前读取量可能更容易被计算隐藏。所以不能只看 f_short 增大就断言 U 会更低。

| ID | 长层盘侧服务 ms | 短层 C ms | 短层自身链路最低 ms | 可容忍首次链路延迟 ms |
|---|---:|---:|---:|---:|
| topo6_m2048_s32m128 | 1.107788 | 1.178026 | 0.856018 | 0.322008 |
| topo8_m2048_s32m128 | 0.830841 | 1.178026 | 0.856018 | 0.322008 |
| topo6_m2048_s32m256 | 1.107788 | 1.997480 | 0.852661 | 1.144819 |
| topo8_m2048_s32m256 | 0.830841 | 1.997480 | 0.852661 | 1.144819 |
| topo6_m2048_s32m512 | 1.107788 | 3.702133 | 0.845947 | 2.856185 |
| topo8_m2048_s48m128 | 0.830841 | 1.513570 | 1.285706 | 0.227865 |
| topo6_m4096_s32m256 | 1.096598 | 1.997480 | 0.852661 | 1.144819 |
| topo8_m4096_s32m256 | 0.822449 | 1.997480 | 0.852661 | 1.144819 |

这里显式补上 50 GiB/s 的 NPU 链路。仅用 V/(盘数×40) 算的是盘侧时间，不能直接当完整读取完成时间。

```text
短层第一次开始进入 NPU 链路，相对本层计算开始晚了 tau：
整层 IO 完成时间至少为 tau + VS/50
如果 tau > CS - VS/50，就一定不能在本层计算内藏住读取。
```

这是假设同请求内部层在本层计算开始时预取下一层的时长下界。第一次进入链路后，如果还有供给间隙，实际完成会更晚；磁盘与链路可以流式重叠，不应把全部盘服务时间和链路服务时间直接相加。

例如 32K/128 的 C≈1.178 ms，自身链路接收至少≈0.856 ms，只剩≈0.322 ms 的首次供给余量。48K/128 自身接收至少≈1.286 ms，而 C≈1.514 ms，余量更小。这个变量比单看长层总服务时间更接近可以验证的截止条件。

但真正排前的是剩余 IO 块。Baseline 逐个 176 KiB 块服务；同时释放的层可能交错提交，长卡也可能早已读完。必须通过 trace 量到 tau、FIFO 前缀和最终 IO-ready，再判断这些候选为什么成功或失败。

## 统计口径和成本

- 新 metadata 分别记录 num_ssu=6/8；文件名和下列命令带不同候选名。主报告应与 3 盘分表。
- 新输入每卡纯计算至少 22 秒；只 Random，pilot=7，确认=19/43/67/101。不能复制所有卡的排列、引入同步屏障或按坏 seed 重抽。
- warm[2,4)、long[2,20) 均报告 U；每卡 warm 是否两类都计算必须单独验证。失败结果保留，但不能当满足混合约束的最终证据。
- 平均 rho≈1 不是逐盘逐时欠载。整个有限批次相同分母下仍有 SSU忙率=rho×NPU U；若 Random 使 SSU一直忙，U仍难很低。
- 增盘并保持接近满载，会增加完成同样计算时长需要模拟的读取块数。下面记录完整预计块数，不能把六小时预算只按候选个数估算。

| ID | 每卡 L/S 数量 | 每卡纯计算 s | 总 IO 块数（百万） | warm 风险标记 |
|---|---|---:|---:|---|
| topo6_m2048_s32m128 | 36/279 | 23.003 | 32.811 | 仍需日志验证 |
| topo8_m2048_s32m128 | 33/473 | 23.133 | 44.259 | 仍需日志验证 |
| topo6_m2048_s32m256 | 32/304 | 22.967 | 32.743 | 仍需日志验证 |
| topo8_m2048_s32m256 | 26/507 | 22.816 | 43.510 | 仍需日志验证 |
| topo6_m2048_s32m512 | 21/378 | 23.080 | 32.901 | 仍需日志验证 |
| topo8_m2048_s48m128 | 33/297 | 22.272 | 42.502 | 仍需日志验证 |
| topo6_m4096_s32m256 | 14/406 | 22.317 | 32.019 | 长请求超过1秒 |
| topo8_m4096_s32m256 | 12/612 | 23.348 | 44.612 | 长请求超过1秒 |

## 可运行命令（由主任务决定启动；本脚本没有执行）

下列初筛均为 Baseline、无 trace；若发现值得复验的低 U，再由主任务补充 trace 和确认种子。

```bash
python results/baseline_random_near_capacity_20260914/experiment.py --name topo6_m2048_s32m128 --profiles 200:2048,32:128 --counts 4,31 --roles L,S --num-ssu 6 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo8_m2048_s32m128 --profiles 200:2048,32:128 --counts 3,43 --roles L,S --num-ssu 8 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo6_m2048_s32m256 --profiles 200:2048,32:256 --counts 2,19 --roles L,S --num-ssu 6 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo8_m2048_s32m256 --profiles 200:2048,32:256 --counts 2,39 --roles L,S --num-ssu 8 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo6_m2048_s32m512 --profiles 200:2048,32:512 --counts 1,18 --roles L,S --num-ssu 6 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo8_m2048_s48m128 --profiles 200:2048,48:128 --counts 1,9 --roles L,S --num-ssu 8 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo6_m4096_s32m256 --profiles 200:4096,32:256 --counts 1,29 --roles L,S --num-ssu 6 --horizon-ms 22000 --seed 7 --strategy baseline
python results/baseline_random_near_capacity_20260914/experiment.py --name topo8_m4096_s32m256 --profiles 200:4096,32:256 --counts 1,51 --roles L,S --num-ssu 8 --horizon-ms 22000 --seed 7 --strategy baseline
```

[完整扫描与参数](topology_candidates.json) · [数学生成器](topology_candidates.py) · [主 3 盘候选](candidate_math.md)
