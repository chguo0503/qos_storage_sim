# Pre-migration simulator regression fixture

These files preserve a deterministic execution captured on 2026-09-19 before
the flat modules moved into `simulator/` and `inputs/`.

- `pre_refactor_trace.json.gz`: immutable request manifest for 32 NPUs, 3 SSUs,
  8 layers and 4 requests per NPU. All arrivals are zero, assignment is fixed,
  placement is native Ring Hash and each I/O block is 176 KiB. The four
  constructed profiles rotate across NPUs and queue positions.
- `pre_refactor_expected.json.gz`: pre-migration ASU, OD and original Once
  completion times, layer I/O/compute times, utilization, block/request totals
  and signatures. Each policy completes 128 requests and 47,104 blocks.

This small constructed workload verifies simulation behavior; it is not a
measurement of real workload performance. Tests compare simulated timing with
an absolute tolerance of 1e-9 ms. Host CPU execution times are not deterministic
simulation outputs and are not compared.

The one-time migration audit also verified byte-identical request, layer and
ordered block-completion signatures before and after migration. Normal tests
need only these files and do not depend on the temporary migration snapshot or
historical result directories.
