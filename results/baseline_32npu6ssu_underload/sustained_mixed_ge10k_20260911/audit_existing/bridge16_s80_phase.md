# mostly/bridge 候选的长组回归相位（离线核验）

长组 16 张卡；Long 每层 C = 31.416419055 ms。每轮依次执行 6 条 Long，随后逐条执行 80:1024, 32:4096。

| Cycle | 首 Long L0 C-start 最早–最晚(ms) | 线性跨度(ms) | 周期相位覆盖弧(ms) | 共同长段的全局 Long 数 min–max |
|---:|---:|---:|---:|---:|
| 0 | 18.613–18.626 | 0.013528 | 0.013528 | 16–19 |
| 1 | 1877.826–1877.839 | 0.013528 | 0.013528 | 16–19 |
| 2 | 3737.039–3737.052 | 0.013528 | 0.013528 | 16–20 |
| 3 | 5596.252–5596.266 | 0.013528 | 0.013528 | 16–20 |
| 4 | 7455.465–7455.479 | 0.013528 | 0.013528 | 16–20 |
| 5 | 9314.679–9314.692 | 0.013528 | 0.013528 | 16–20 |
| 6 | 11173.892–11173.905 | 0.013528 | 0.013528 | 16–20 |
| 7 | 13033.105–13033.119 | 0.013528 | 0.013528 | 17–20 |

| 窗口(s) | Fleet U(%) | Short C/占用(%) | Long C/占用(%) | 平均 Long 卡数 |
|---|---:|---:|---:|---:|
| 0–2 | 90.246353 | 82.439976 | 98.964822 | 15.116877 |
| 2–12 | 90.855794 | 82.732858 | 99.946616 | 15.100360 |
| 4–12 | 90.871190 | 82.762526 | 99.947719 | 15.098885 |
| 2–4 | 90.794209 | 82.614135 | 99.942209 | 15.106258 |
| 4–6 | 90.801955 | 82.641742 | 99.974984 | 15.065088 |
| 6–8 | 90.988706 | 83.020130 | 99.891089 | 15.114400 |
| 8–10 | 90.881656 | 82.778262 | 99.950686 | 15.100291 |
| 10–12 | 90.812444 | 82.610298 | 99.974205 | 15.115761 |

第一次返回时，长组每卡相同 pure C；其间额外等待的卡间差异来自：6Long 段（不含起始 L0）0.000000–0.000000 ms，Short 段 0.000000–0.000000 ms，下一 Long L0 接纳后暴露等待 0.000000–0.000000 ms。上述逐卡时间账已严格相加复核。

- Circular arc = C_L minus largest cyclic phase gap; same indexed logical waves across the designated long group, not a global barrier.
- The shared group Long interval does not imply that only this group has Long requests; global counts are separately integrated.
- IO start-to-ready is queue plus SSD plus link lifetime. This analysis has no block-service trace and does not assign a FIFO predecessor.
- Short-segment waits and subsequent Long L0 waits change relative phases; their observed association with higher U is not an isolated causal intervention.
