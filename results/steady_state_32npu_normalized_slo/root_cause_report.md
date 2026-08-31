# 32-NPU warm/full-load root-cause analysis

本报告由主实验 `results.json` 和诊断实验 `diagnostics.json` 独立重算。所有利用率都来自各策略自身固定 2,000-ms 测量窗；SLO 同时给出 Equal-NPU 和 request-weighted 口径。

输入 artifact 的 stored source fingerprint 与当前源码一致。

## Baseline / Current / New / Admission

| SSU | 策略 | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | 测量请求 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | baseline | 30.33% | 30.41% | 32.69% | 25.00% | 25.00% | 0.00% | 0.00% | 0.00% | 100.00% | 192 |
| 6 | current_scheme_b | 30.40% | 31.07% | 32.57% | 9.15% | 9.34% | 0.00% | 29.79% | 0.00% | 6.52% | 182 |
| 6 | new_scheme_b | 32.20% | 32.27% | 32.76% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 184 |
| 6 | admission | 23.05% | 23.21% | 32.40% | 67.35% | 67.93% | 0.00% | 92.68% | 87.23% | 100.00% | 184 |
| 10 | baseline | 54.79% | 55.00% | 57.24% | 75.00% | 75.00% | 0.00% | 100.00% | 100.00% | 100.00% | 320 |
| 10 | current_scheme_b | 53.32% | 55.04% | 56.94% | 50.77% | 50.80% | 0.00% | 100.00% | 1.27% | 100.00% | 313 |
| 10 | new_scheme_b | 55.39% | 55.94% | 56.69% | 61.51% | 61.59% | 0.00% | 100.00% | 52.50% | 100.00% | 328 |
| 10 | admission | 52.30% | 54.69% | 57.13% | 82.08% | 82.45% | 38.75% | 100.00% | 91.67% | 100.00% | 319 |
| 18 | baseline | 91.97% | 92.01% | 93.29% | 79.38% | 79.46% | 17.19% | 100.00% | 100.00% | 100.00% | 516 |
| 18 | current_scheme_b | 89.37% | 91.18% | 92.78% | 82.28% | 82.51% | 27.05% | 100.00% | 100.00% | 100.00% | 509 |
| 18 | new_scheme_b | 90.35% | 90.44% | 91.14% | 89.00% | 89.07% | 54.62% | 100.00% | 100.00% | 100.00% | 494 |
| 18 | admission | 90.96% | 91.47% | 92.21% | 98.44% | 98.43% | 93.75% | 100.00% | 100.00% | 100.00% | 510 |

### Admission 相对收益（原始固定窗口）

下表是 admission 减去 reference 的百分点。请求 cohort 会随策略变化，因此 TTFT 的严格同请求比较见下一节。

| SSU | Reference | Mean NPU util | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | baseline | -0.30 pp | +42.35 pp | +42.93 pp | +0.00 pp | +92.68 pp | +87.23 pp | +0.00 pp |
| 6 | current_scheme_b | -0.17 pp | +58.20 pp | +58.59 pp | +0.00 pp | +62.90 pp | +87.23 pp | +93.48 pp |
| 10 | baseline | -0.11 pp | +7.08 pp | +7.45 pp | +38.75 pp | +0.00 pp | -8.33 pp | +0.00 pp |
| 10 | current_scheme_b | +0.20 pp | +31.31 pp | +31.65 pp | +38.75 pp | +0.00 pp | +90.40 pp | +0.00 pp |
| 18 | baseline | -1.09 pp | +19.05 pp | +18.97 pp | +76.56 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 18 | current_scheme_b | -0.58 pp | +16.16 pp | +15.92 pp | +66.70 pp | +0.00 pp | +0.00 pp | +0.00 pp |

## Paired request intersection：baseline/current/admission

每个 SSU 仅保留三个策略测量 cohort 中 request ID 的交集；NPU、sequence、category 和 ideal TTFT 也逐请求核对一致。该表用于排除不同 admission 时刻造成的 cohort 偏差。

### SSU 6

原 cohort：baseline=192，current=182，admission=184；交集=35。

| 策略 | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 27.78% | 25.71% | 0.00% | 0.00% | 0.00% | 100.00% |
| current_scheme_b | 6.94% | 8.57% | 0.00% | 30.00% | 0.00% | 0.00% |
| admission | 70.83% | 71.43% | 0.00% | 90.00% | 87.50% | 100.00% |

| Admission minus | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +43.06 pp | +45.71 pp | +0.00 pp | +90.00 pp | +87.50 pp | +0.00 pp |
| current_scheme_b | +63.89 pp | +62.86 pp | +0.00 pp | +60.00 pp | +87.50 pp | +100.00 pp |

### SSU 10

原 cohort：baseline=320，current=313，admission=319；交集=274。

| 策略 | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 74.86% | 74.82% | 0.00% | 100.00% | 100.00% | 100.00% |
| current_scheme_b | 50.04% | 50.00% | 0.00% | 100.00% | 1.45% | 100.00% |
| admission | 82.72% | 82.48% | 39.13% | 100.00% | 91.30% | 100.00% |

| Admission minus | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +7.86 pp | +7.66 pp | +39.13 pp | +0.00 pp | -8.70 pp | +0.00 pp |
| current_scheme_b | +32.67 pp | +32.48 pp | +39.13 pp | +0.00 pp | +89.86 pp | +0.00 pp |

### SSU 18

原 cohort：baseline=516，current=509，admission=510；交集=485。

| 策略 | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 79.83% | 80.00% | 18.49% | 100.00% | 100.00% | 100.00% |
| current_scheme_b | 82.06% | 82.27% | 27.73% | 100.00% | 100.00% | 100.00% |
| admission | 98.33% | 98.35% | 93.28% | 100.00% | 100.00% | 100.00% |

| Admission minus | Equal-NPU SLO | Request SLO | SS | SL | LS | LL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +18.51 pp | +18.35 pp | +74.79 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| current_scheme_b | +16.27 pp | +16.08 pp | +65.55 pp | +0.00 pp | +0.00 pp | +0.00 pp |

## SSU18 NPU-link bypass

Bypass 将物理 NPU link 提升到 1,000,000 GB/s；new Scheme B 的 allocator NPU cap 同时提升，baseline 没有动态 allocator cap。

| SSU | 策略 | 物理 NPU GB/s | 分配器 NPU cap | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | 测量请求 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | baseline | 50 | — | 91.97% | 92.01% | 93.29% | 79.38% | 79.46% | 17.19% | 100.00% | 100.00% | 100.00% | 516 |
| 18 | baseline_npu_bypass | 1e+06 | — | 91.55% | 91.58% | 94.84% | 84.38% | 84.47% | 36.92% | 100.00% | 100.00% | 100.00% | 528 |
| 18 | new_scheme_b | 50 | 50 | 90.35% | 90.44% | 91.14% | 89.00% | 89.07% | 54.62% | 100.00% | 100.00% | 100.00% | 494 |
| 18 | new_scheme_b_npu_bypass | 1e+06 | 1e+06 | 90.27% | 90.46% | 90.83% | 91.67% | 91.68% | 64.29% | 100.00% | 100.00% | 100.00% | 481 |

| 策略 | Bypass mean util | Bypass Equal-NPU SLO | Bypass Request SLO |
|---|---:|---:|---:|
| baseline | +1.55 pp | +4.99 pp | +5.01 pp |
| new_scheme_b | -0.31 pp | +2.66 pp | +2.62 pp |

## Valid released-I/O oracle

| SSU | 策略 | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | 测量请求 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | released_io_oracle | 87.30% | 88.72% | 93.05% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 508 |

SSU18 oracle 是有效 steady-state row，可作为已释放 I/O 可见条件下的调度参考；它不是不可实现的全未来信息最优解。

## Invalid oracle diagnostics

以下 row 只报告科学无效原因，不进入上面的任何利用率或 SLO 数字表。

| SSU | Failed invariants | Zero-sample NPU IDs | Finite-prefix exhausted NPU IDs |
|---:|---|---|---|
| 6 | all_npus_sampled_for_slo, no_backlog_exhaustion | 2,3,7,9,11,13,14,17,18,20,21,22,23,24,25,27,28,29,30 | 1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31 |
| 10 | all_npus_sampled_for_slo, no_backlog_exhaustion | 4,28,30,31 | 1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,19,20,23,24,25,26,28,29,30,31 |

## Root-cause reading

- Admission 与 baseline/current 在相同 workload、placement、trace 和 simulator input 上配对；其 SLO 改善不能解释为输入变化。
- Admission 的 SLO 收益应优先使用 paired-request 表；固定窗口利用率仍是正确的系统时间积分，但不能按 request ID 配对。
- SSU18 的 NPU bypass 只给出移除 50-GB/s link/cap 后的敏感性，不能单独证明所有差距都由 NPU link 导致。
- SSU6/10 oracle 因有限前缀耗尽且部分 NPU 无 SLO 样本而无效，不能引用其 SLO/利用率，也不能把它们当上界。
