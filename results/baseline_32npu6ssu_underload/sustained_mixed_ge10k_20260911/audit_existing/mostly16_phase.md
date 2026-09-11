# mostly/bridge 候选的长组回归相位（离线核验）

长组 16 张卡；Long 每层 C = 31.416419055 ms。每轮依次执行 8 条 Long，随后逐条执行 32:1024, 48:1024, 64:1024。

| Cycle | 首 Long L0 C-start 最早–最晚(ms) | 线性跨度(ms) | 周期相位覆盖弧(ms) | 共同长段的全局 Long 数 min–max |
|---:|---:|---:|---:|---:|
| 0 | 18.432–18.450 | 0.017324 | 0.017324 | 16–32 |
| 1 | 2269.745–2278.411 | 8.666450 | 8.666450 | 16–32 |
| 2 | 4518.994–4529.425 | 10.431386 | 10.431386 | 16–32 |
| 3 | 6768.242–6779.022 | 10.779429 | 10.779429 | 16–32 |
| 4 | 9017.491–9028.436 | 10.944458 | 10.944458 | 16–32 |
| 5 | 11266.774–11283.351 | 16.576512 | 16.576512 | 16–32 |
| 6 | 13516.023–13538.654 | 22.631094 | 21.450171 | 16–32 |

| 窗口(s) | Fleet U(%) | Short C/占用(%) | Long C/占用(%) | 平均 Long 卡数 |
|---|---:|---:|---:|---:|
| 0–2 | 86.712156 | 73.975364 | 99.080823 | 16.234611 |
| 2–12 | 97.054810 | 94.090996 | 99.848663 | 16.472303 |
| 4–12 | 97.286490 | 94.603892 | 99.876180 | 16.281953 |
| 2–4 | 96.128087 | 91.907180 | 99.744672 | 17.233702 |
| 4–6 | 97.227371 | 94.389533 | 99.800152 | 16.783810 |
| 6–8 | 97.199055 | 94.497808 | 99.859077 | 16.123032 |
| 8–10 | 97.288581 | 94.675270 | 99.861845 | 16.123543 |
| 10–12 | 97.430955 | 94.843651 | 99.986939 | 16.097428 |

第一次返回时，长组每卡相同 pure C；其间额外等待的卡间差异来自：8Long 段（不含起始 L0）0.000000–0.000000 ms，Short 段 2.063242–6.688619 ms，下一 Long L0 接纳后暴露等待 0.000000–4.023749 ms。上述逐卡时间账已严格相加复核。

- Circular arc = C_L minus largest cyclic phase gap; same indexed logical waves across the designated long group, not a global barrier.
- The shared group Long interval does not imply that only this group has Long requests; global counts are separately integrated.
- IO start-to-ready is queue plus SSD plus link lifetime. This analysis has no block-service trace and does not assign a FIFO predecessor.
- Short-segment waits and subsequent Long L0 waits change relative phases; their observed association with higher U is not an isolated causal intervention.
