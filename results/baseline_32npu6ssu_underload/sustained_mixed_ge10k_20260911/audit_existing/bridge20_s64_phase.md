# mostly/bridge 候选的长组回归相位（离线核验）

长组 20 张卡；Long 每层 C = 31.416419055 ms。每轮依次执行 6 条 Long，随后逐条执行 64:1024, 32:4096。

| Cycle | 首 Long L0 C-start 最早–最晚(ms) | 线性跨度(ms) | 周期相位覆盖弧(ms) | 共同长段的全局 Long 数 min–max |
|---:|---:|---:|---:|---:|
| 0 | 21.736–21.761 | 0.024877 | 0.024877 | 20–22 |
| 1 | 1859.474–1859.499 | 0.024877 | 0.024877 | 20–22 |
| 2 | 3697.212–3697.237 | 0.024877 | 0.024877 | 20–23 |
| 3 | 5534.951–5534.976 | 0.024877 | 0.024877 | 20–23 |
| 4 | 7372.689–7372.714 | 0.024877 | 0.024877 | 20–23 |
| 5 | 9210.428–9210.452 | 0.024877 | 0.024877 | 21–23 |
| 6 | 11048.166–11048.191 | 0.024877 | 0.024877 | 20–22 |
| 7 | 12885.904–12885.929 | 0.024877 | 0.024877 | 20–22 |

| 窗口(s) | Fleet U(%) | Short C/占用(%) | Long C/占用(%) | 平均 Long 卡数 |
|---|---:|---:|---:|---:|
| 0–2 | 90.760859 | 80.433041 | 98.701999 | 18.090260 |
| 2–12 | 91.720953 | 81.092008 | 99.916498 | 18.068284 |
| 4–12 | 91.764267 | 81.187077 | 99.909752 | 18.078084 |
| 2–4 | 91.547696 | 80.713069 | 99.943556 | 18.029084 |
| 4–6 | 91.594197 | 80.797409 | 99.932756 | 18.055446 |
| 6–8 | 91.713506 | 81.058582 | 99.924679 | 18.072502 |
| 8–10 | 91.916749 | 81.543165 | 99.902892 | 18.080589 |
| 10–12 | 91.832615 | 81.350201 | 99.878759 | 18.103797 |

第一次返回时，长组每卡相同 pure C；其间额外等待的卡间差异来自：6Long 段（不含起始 L0）0.000000–0.000000 ms，Short 段 0.000000–0.000000 ms，下一 Long L0 接纳后暴露等待 0.000000–0.000000 ms。上述逐卡时间账已严格相加复核。

- Circular arc = C_L minus largest cyclic phase gap; same indexed logical waves across the designated long group, not a global barrier.
- The shared group Long interval does not imply that only this group has Long requests; global counts are separately integrated.
- IO start-to-ready is queue plus SSD plus link lifetime. This analysis has no block-service trace and does not assign a FIFO predecessor.
- Short-segment waits and subsequent Long L0 waits change relative phases; their observed association with higher U is not an isolated causal intervention.
