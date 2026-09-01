# SSD accounting patch validation

This is a sanitized validation record. It contains no hostname, process ID, or
private filesystem path. The immutable old source is commit
`71e5f06577f6f8c4826ddb38e5029e5f5a3a97d4`, fingerprint
`88161e7a08d573d0ceea29e29f8a456136ee39709800d7e6e34b0e4cfc8a6ced`.
The final frozen replacement fingerprint is
`76658d5aac81f11a8f9fddcfeaed6692821aefde3892cdc0d7dd564563242568`.

## Old/new trajectory equivalence

The comparison hashes the complete request rows, bounded dispatch probes,
route probes, control-trigger records, Adaptive decision records, discrete
event/control counters, CIR boundary state, and their combined payload.

| Policy | Window | Events | Requests | Control decisions | Combined SHA256 | Result |
|---|---:|---:|---:|---:|---|---|
| Baseline | 2 s | 41,394,489 | 86 | 0 | `ce01dc7ae13b075e89e5f40ebd8ffb0d9915456146755ab6219c6e0a3d77b999` | exact |
| Layer-once TTL 5 ms | 2 s | 38,254,224 | 96 | 0 | `819cb6e4c0ad6c77bcf82889d9f0eeae55375bcf755c36171eb261f31b031182` | exact |
| Adaptive 100 ms | 8 s | 66,643,422 | 352 | 243 | `ab1efa593698d71b961523517565e5924475ff64217d5b4a4e9c4af533ac2be9` | exact |

Every component hash, every discrete counter, and the NPU-utilization/TTFT
scalars are bit-for-bit equal between old and patched sources for all three
traces. Adaptive additionally matches 3,265 control triggers, 243 decisions,
729 CIR write transactions, and 21,679 Path writes.

Source fingerprints and the intended new accounting fields are deliberately
not compared: the old source has no v3 stable/fragmented/physical-queue
semantics, compensated busy diagnostics, or carry-in export. Baseline and
Layer-once were compared before the carry-in output was added; Adaptive used
the complete accounting replacement before the final inclusive-left-limit
wording change. Those later changes execute only while building the returned
summary after the event loop, and a final-source actual DES check is recorded
below.

## Completion-fragmentation validation

The old mutable subtraction fails an artificial 1,000,000-fragment command:
both oracle and static-CIR leave `2.1120623475921188e-12 GB` at the theoretical
end, above the internal `1e-12 GB` epsilon, and return no completed command.
The frozen regression runs both policies at 250,000 and 1,000,000 fragments.
All four cases refuse completion at the immediately preceding float, complete
at the exact theoretical end, leave zero remainder, and give exactly the same
stable midpoint service as an unfragmented command.

This is a real latent numerical-correctness fix, not a globally observer-only
change. It was not triggered in the realistic old 8-second Baseline trace:
14,286,656 commands completed, no due completion returned empty, fragmentation
was min/p50/p99/max `1/1/15/71`, and the largest raw pre-clamp remainder was
only `1.818993089833243e-14 GB = 0.0000181899308983 byte`.

## Actual DES accounting closure

A patched 32-NPU, 3-SSU, 16-layer Baseline DES ran 8 seconds, processed
57,196,749 events and 353 measured requests, with no failed invariant.

- Whole-window stable service minus compensated busy service was exactly
  `[0, 0, 0] GB`.
- Maximum boundary stable-service/busy residual was
  `1.1368683772161603e-13 GB = 0.000113686837722 byte`.
- Maximum direct-physical-queue/scheduler residual was
  `2.1649348980190553e-15 GB`.
- Maximum `enqueue - stable service - physical queue` residual was
  `5.4539706084710815e-14 GB`.
- Both physical and counter queue block residuals were exactly zero.
- The retained fragmented observer differed from stable whole-window service
  by `[0.238174, 0.600323, 0.565421]` decimal bytes, demonstrating why it is
  diagnostic only.

## Carry-in and left-limit proof

The final source also ran an actual 2-NPU, 2-SSU, 16-layer, 4-second DES:
2,433,867 events, nine stationarity boundaries, and no failed invariant. It
exported two authenticated carry-in rows, each with one request and 16 complete
layer intervals. One batch strictly crossed the measurement start; the other
completed exactly at it.

The producer therefore uses the left-limit definition
`admission < measurement_start <= completion`. A completion-equality row is
still active in the snapshot taken before same-time workload events, but its
half-open interval contributes zero elapsed time. All seven definition,
uniqueness, batch-size, layer-shape, identity, compute-budget, and interval
closure gates passed.

## Frozen source and tests

- `sim.py`: `5f3a9bb82c5740a86c9c916e01495326a0bf4363fb6a5312cb99ac747e8600fd`
- `continuous_batch_sim.py`: `ae5c37aa6733e81fcc0e624404a89a154594448712a9b1ff94c39a2ba36c162e`
- `ssd_accounting_checks.py`: `7b444b82b674b65d06ce21240ca954372c33ff259e55cbdb765328843ad91b6c`

Final verification passed: compile, 4/4 SSD/carry checks, 9/9 Adaptive checks,
4/4 forecast checks, 10/10 protected-floor checks, actual-DES invariants,
remote mirror compile/checks, and the whitespace diff check. No tolerance was
widened and no commit was created.

Machine-readable values and every component trajectory hash are in
`ssd_accounting_patch_validation.json`.
