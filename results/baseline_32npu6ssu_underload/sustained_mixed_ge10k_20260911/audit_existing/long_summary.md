# 60秒验证：独立审计汇总

完整审计 5/5。所有行均为 seed7；空值表示等待，未用 0 或 pilot 数字补齐。

| 输入 | 策略 | 状态 | U[2,60) | U[4,60) | 2秒窗 min–max | 暖接纳SLO | 每卡快短+长均≥5% |
|---|---|---|---:|---:|---:|---:|---:|
| 主方案定序 | baseline | complete | 89.964747% | 89.970475% | 89.609821–90.897879% | 99.949193% | 32/32 |
| 主方案定序 | once | complete | 99.961111% | 99.961077% | 99.920192–99.988018% | 100.000000% | 32/32 |
| 主方案随机 | baseline | complete | 99.997867% | 99.998488% | 99.980480–100.000000% | 100.000000% | 32/32 |
| 主方案随机 | once | complete | 99.987211% | 99.987699% | 99.913566–100.000000% | 100.000000% | 32/32 |
| 备选12L定序 | baseline | complete | 89.858274% | 89.831246% | 88.513297–90.927106% | 99.218113% | 16/32 |

SLO从接纳计时，并随访完整完成；不同策略的窗口接纳人口可能不同。两种预取需求代理与原始容量判据分别保留在JSON/CSV中；次级检查不改变原接受条件。

- U clips actual layer compute into the declared window and divides by 32 times window duration.
- SLO cohort: admission in [2000,60000), follow full completion, completion-admission <= 1.5*8*raw per-layer C. It is not arrival-clock or hardware TTFT; cohorts can differ by policy and order.
- Fast short is short role with NQL1024; NQL4096 bridge is excluded from its per-card C share. This secondary >=5% count does not change original role-based acceptance.
- Prefetch variants are additional demand proxies with actual next payload, not actual throughput or an all-deadline feasibility proof.
- Only seed7 and a finite 60-second horizon: no multi-seed mean or infinite-horizon stationary claim. Cancelled predecessor runs are excluded.
