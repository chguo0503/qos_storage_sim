# SSU3 64-second SSD accounting failure diagnosis

This is a sanitized, read-only traceback extraction from the old immutable
source at commit `71e5f06577f6f8c4826ddb38e5029e5f5a3a97d4`, source fingerprint
`88161e7a08d573d0ceea29e29f8a456136ee39709800d7e6e34b0e4cfc8a6ced`.
It contains no hostname, PID, or private filesystem path.

## Exact failed case

- Baseline, 32 NPUs, 3 SSUs, 16 layers, seed 42.
- 256 backing requests/NPU, warmup 8 requests/NPU, settle 500 ms.
- Measurement 64,000 ms in 500-ms blocks, with timeline diagnostics.
- 209,888,095 events, 2,877 measured requests, 129 boundaries and
  387 boundary/SSU rows.
- Exactly two gates failed: `ssd_service_attribution` and
  `timeline_independent_queue_attribution`. Every other timeline gate passed.

The old whole-window NPU-attributed service minus busy-time service was
`[1.0983058018609881e-08, 1.0190888133365661e-08,
1.0106759873451665e-08] GB`, or `[10.9831, 10.1909, 10.1068]` decimal bytes.
All three SSUs therefore crossed the unchanged absolute `1e-8 GB` service
gate. Replacing built-in `sum` with `math.fsum` only at the final reduction
produced the same three numbers, so this is not a final-reduction artifact.

## Worst boundary discrepancies

At boundary 109 (54.5 s), SSU 0:

- fragmented per-NPU service sum: `2373.6932250774075 GB`;
- independent busy-time service: `2373.693225060778 GB`;
- difference: `1.6629655874567106e-08 GB = 16.6296558746 bytes`.

At boundary 112 (56.0 s), SSU 1:

- counter-derived queue: `1.4796358528855933 GB`;
- directly observed physical queue: `1.479635866272729 GB`;
- difference: `-1.338713562226701e-08 GB = -13.3871356223 bytes`.

There were 126 of 387 rows where service or queue residual exceeded
`1e-8 GB`. The integer queue-block residual was exactly zero at every row.

## Physical-chain proof

For every boundary and SSU, independently recompute

```text
R_physical = enqueued_gb - busy_service_gb - physical_outstanding_gb
```

where busy service is the independently maintained per-SSD scheduler busy-time
counter times 40 GB/s, and physical outstanding is the pending command
inventory plus the active command's remainder. The tested old source did not
yet use compensated busy-time accumulation; compensation is part of the v3
replacement. The largest absolute residual over all 387 rows was only
`2.6353919047039653e-13 GB = 0.00026353919047 byte`, at boundary 109, SSU 0.
The old fragmented-service error is therefore about **63,101.26 times** larger
than the worst independent physical-chain closure residual.

The defect is the representation, not missing physical work: the old observer
adds `bw * settle_delta / 1000` once per settle edge for every NPU/SSU cell,
then obtains a small queue by subtracting two large cumulative counters.
Event/observer fragmentation changes that floating-point path. The replacement
uses compensated whole-command completions plus one active prefix reconstructed
from immutable command activation, and directly enumerates pending plus active
physical commands for queue state. The old fragmented chain remains diagnostic
only; no tolerance is widened.

## Separate mutable-progress completion risk

The old source also repeatedly subtracted every settle fragment from mutable
`remaining_gb`. This is a separate latent correctness defect. With one
1.23456789-GB command deliberately split into 1,000,000 settle fragments, both
the oracle and static-CIR schedulers retained `2.1120623475921188e-12 GB` at the
theoretical completion event. That exceeded the internal `1e-12 GB` epsilon,
so both returned no completed command. The v3 replacement reconstructs
progress from immutable activation, total size, bandwidth, and theoretical end:
both policies complete at the exact end, retain zero bytes, and refuse to
complete at the immediately preceding representable float.

This artificial failure did **not** occur in the realistic old-source Baseline
8-second SSU3 diagnostic. It completed 14,286,656 commands with zero empty
completion events. Settle fragments per command were min/p50/p99/max
`1 / 1 / 15 / 71`. Of 14,286,653 instrumented due-settle observations,
3,140,249 had a positive raw pre-clamp remainder, but none exceeded the
internal `1e-12 GB = 0.001 byte` epsilon. The maximum was only
`1.818993089833243e-14 GB = 0.0000181899308983 byte`, occurred with one settle
fragment, and was exactly the same as the immutable-formula rounding residual.
The old clamp therefore left no positive expiry remainder. This is a real
latent numerical correctness bug, but it was not triggered by that measured
Baseline trace. Cross-strategy old/new trajectory equivalence is still required
before declaring every old formal strategy trace unaffected.

Machine-readable values and equations are in
`ssd_accounting_failure_diagnosis.json`.
