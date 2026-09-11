# mostly/bridge 候选的长组回归相位（离线核验）

长组 16 张卡；Long 每层 C = 31.416419055 ms。每轮依次执行 6 条 Long，随后逐条执行 64:1024, 32:4096。

| Cycle | 首 Long L0 C-start 最早–最晚(ms) | 线性跨度(ms) | 周期相位覆盖弧(ms) | 共同长段的全局 Long 数 min–max |
|---:|---:|---:|---:|---:|
| 0 | 18.613–18.626 | 0.013528 | 0.013528 | 16–19 |
| 1 | 1856.351–1856.364 | 0.013528 | 0.013528 | 16–20 |
| 2 | 3694.089–3694.103 | 0.013528 | 0.013528 | 16–20 |
| 3 | 5531.828–5531.841 | 0.013528 | 0.013528 | 16–19 |
| 4 | 7369.566–7369.580 | 0.013528 | 0.013528 | 16–19 |
| 5 | 9207.304–9207.318 | 0.013528 | 0.013528 | 16–19 |
| 6 | 11045.043–11045.056 | 0.013528 | 0.013528 | 17–19 |
| 7 | 12882.781–12882.795 | 0.013528 | 0.013528 | 17–19 |

| 窗口(s) | Fleet U(%) | Short C/占用(%) | Long C/占用(%) | 平均 Long 卡数 |
|---|---:|---:|---:|---:|
| 0–2 | 90.090285 | 81.967089 | 98.970787 | 15.287396 |
| 2–12 | 90.708554 | 82.274674 | 99.950492 | 15.268554 |
| 4–12 | 90.696450 | 82.255171 | 99.946480 | 15.268567 |
| 2–4 | 90.756970 | 82.352685 | 99.966540 | 15.268499 |
| 4–6 | 90.549210 | 82.008990 | 99.949987 | 15.232544 |
| 6–8 | 90.744547 | 82.348397 | 99.926862 | 15.284428 |
| 8–10 | 90.802642 | 82.441021 | 99.961292 | 15.272131 |
| 10–12 | 90.689403 | 82.222902 | 99.947800 | 15.285167 |

第一次返回时，长组每卡相同 pure C；其间额外等待的卡间差异来自：6Long 段（不含起始 L0）0.000000–0.000000 ms，Short 段 0.000000–0.000000 ms，下一 Long L0 接纳后暴露等待 0.000000–0.000000 ms。上述逐卡时间账已严格相加复核。

- Circular arc = C_L minus largest cyclic phase gap; same indexed logical waves across the designated long group, not a global barrier.
- The shared group Long interval does not imply that only this group has Long requests; global counts are separately integrated.
- IO start-to-ready is queue plus SSD plus link lifetime. This analysis has no block-service trace and does not assign a FIFO predecessor.
- Short-segment waits and subsequent Long L0 waits change relative phases; their observed association with higher U is not an isolated causal intervention.
