# Cross-request Layer0 prefetch: result interpretation

## Verdict against the +10 pp requirement

The modified Scheme B reaches the requested mean-NPU-utilization gain in two
warm configurations and one cold configuration:

- warm, 16 layers / 40 SSUs: `+24.50 pp`;
- warm, 24 layers / 40 SSUs: `+17.62 pp`;
- cold, 16 layers / 40 SSUs: `+11.27 pp`.

No 56- or 80-layer point reaches `+10 pp`. The complete paired deltas are in
`report.md`.

## Why short requests improve most

Every post-cold request is known before the previous request's final compute
layer. Both policies start its Layer0 read at that point. Scheme B additionally
uses the ring-hash manifest to install the next request's dedicated Path/CIR
before issuing the read, so it never falls back to the 0.208333 GB/s public
cold Path.

At 40 SSUs this cuts mean exposed warm-request I/O stall as follows:

| Layers | Baseline | Scheme B | Reduction |
|---:|---:|---:|---:|
| 16 | 109.54 ms | 39.35 ms | 70.19 ms |
| 24 | 164.36 ms | 82.68 ms | 81.68 ms |
| 56 | 383.64 ms | 283.63 ms | 100.01 ms |
| 80 | 548.11 ms | 432.51 ms | 115.60 ms |

The absolute saving grows mildly, but compute work grows directly with layer
count. The same request-boundary saving therefore occupies a much larger share
of a 16/24-layer request than a 56/80-layer request.

## Why the SSU count changes the result

- At 40 SSUs, baseline is bandwidth constrained and stays near 51.24% warm
  utilization. Demand-aware CIR and early Layer0 service remove large exposed
  gaps, especially for short requests.
- At 56 SSUs, baseline already reaches about 72.7%. Scheme B improves mean
  stall, but its max-min objective redistributes service between request
  classes; the remaining utilization gains are only 2.19--9.43 pp.
- At 70 SSUs, baseline is already around 93%. There is too little unused
  compute time for a 10 pp utilization gain, although the additional capacity
  lets Scheme B improve SLO without displacing as many feasible requests.

## Why SLO gets worse at 56 SSUs for long requests

For 80 layers / 56 SSUs, baseline's warm SLO by request class is
`SS=0%, SL=100%, LS=100%, LL=100%`, giving 74.84% overall. Scheme B changes it
to `SS=9.3%, SL=100%, LS=78.0%, LL=100%`, giving 71.72% overall. The bandwidth
spent rescuing a small fraction of SS requests causes more LS requests to cross
their SLO threshold. The controller optimizes instantaneous max-min bandwidth,
not the number of requests that can still meet their deadline.

At 80 layers / 70 SSUs, the extra capacity avoids that tradeoff: Scheme B raises
SS from 4.3% to 51.6% while SL/LS/LL remain at 100%, so warm SLO improves by
11.88 pp even though utilization rises by only 0.51 pp.

## Remaining cold-start and tail limitation

The first request still uses the public cold Path. At 40 SSUs its mean exposed
stall is worse under Scheme B: 276.93 vs 112.26 ms at 16 layers and 690.32 vs
571.15 ms at 80 layers. This is why cold gains are consistently smaller than
warm gains.

Scheme B also widens progress dispersion. For example, at 80 layers / 56 SSUs
it improves mean warm stall from 214.20 to 194.12 ms, but p99 rises from 627.52
to 773.78 ms and full-run makespan rises from 5006.7 to 5643.2 ms. The current
policy therefore meets the requested average-utilization target only for short,
40-SSU workloads; it is not yet a universally better tail/fairness policy.

## Validation

All 24 paired cases use the same request assignment, ring-hash placement,
arrival sequence, SSD40 service, and per-NPU NPU50 receive queue. Every case has
640 cross-request prefetches. Scheme B has 640 manifest-controlled dedicated
Path prefetches, baseline has zero, and every simulator invariant passes.
