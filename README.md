# QoS Storage Simulator — current minimal project

This checkout intentionally contains one experiment only:

```text
32 NPU / 5 SSU / 16 layers / seed 42
warm-up + 500 ms settle, followed by a 4 s measurement window
28 fixed high-V NPU lanes + 4 fixed low-V NPU lanes
Baseline versus four static V/B LL/LS allocations
TTFT / ideal TTFT guard = 8
```

Historical campaigns, intermediate plots, abandoned strategies, and local
environments were moved outside this repository on 2026-09-04. They are not
needed to understand or reproduce the retained result.

## Read this first

1. [`VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md`](VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md)
   contains the result and its statistical limits.
2. [`VB_POLICY_IMPLEMENTATION_AND_HARDWARE_FEASIBILITY.md`](VB_POLICY_IMPLEMENTATION_AND_HARDWARE_FEASIBILITY.md)
   states exactly what the retained V/B policy does and does not model.
3. [`CURRENT_PROJECT_MANIFEST.md`](CURRENT_PROJECT_MANIFEST.md) lists the complete
   dependency closure and hashes. The JSON files record only a smaller legacy
   source-hash subset.

The most important result is not “V/B wins.” For this one seed, Baseline is
already at 92.908964% fleet NPU utilization. The only V/B setting that both
passes `TTFT/ideal <= 8` and improves utilization reaches 93.009984%, a gain of
only 0.101019 percentage points. One seed and eight 500 ms blocks do not prove
that this small gain is stable.

## Repository map

```text
run_vb_fixed_split_npu32_ssu5_experiment.py  matched runner and CLI
plot_vb_fixed_split_npu32_ssu5_4s.py         regenerates summary + 01–04 PNG
vb_pool_policy.py                            V/B values and LL/LS QoS tables
continuous_batch_sim.py, sim.py              event model and SSU scheduler
authenticated_workload_inputs.py             validates the profile table
continuous_prefill_*.py                       workload and routing adapters
random_steady_state_workload.py               deterministic catalog draw
six_request_workload.py                       request construction helper
policy_logic.py, continuous_batch_control.py  Path planning/control primitives
strategy_profiles.py                          Baseline static QoS table
data                                          authenticated 84-profile input
ssd_accounting_checks.py                      storage-accounting regressions
test_continuous_batch_profile_cycle_frontier.py  steady-window regression
results/vb_fixed_split_npu32_ssu5_28high_4low_4s/
```

Only NumPy and matplotlib are third-party dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The retained artifacts were verified with Python 3.10.10, NumPy 2.2.6, and
matplotlib 3.10.9. `requirements.txt` gives compatible lower bounds rather than
claiming bit-for-bit reproducibility across library versions.

## Fast checks

These do not rerun the hour-long experiment:

```bash
python3 -m unittest -v \
  ssd_accounting_checks.py \
  test_continuous_batch_profile_cycle_frontier.py

python3 run_vb_fixed_split_npu32_ssu5_experiment.py \
  --case baseline --dry-run

MPLBACKEND=Agg python3 plot_vb_fixed_split_npu32_ssu5_4s.py
```

The plot command writes four PNG files and `summary.csv`. Add `--pdf` only when
PDF copies are actually needed.

## Full matched rerun

```bash
for case in baseline vb_catalog vb_duration_aware vb_ll36 vb_split_aware; do
  python3 run_vb_fixed_split_npu32_ssu5_experiment.py \
    --case "$case" \
    --requests-per-npu 64 \
    --measurement-ms 4000 \
    --block-ms 500 \
    --output-dir results/vb_fixed_split_npu32_ssu5_28high_4low_4s
done

MPLBACKEND=Agg python3 plot_vb_fixed_split_npu32_ssu5_4s.py
```

The saved run took about 56 minutes sequentially on the original host. Cases
are independent and may be run in parallel if each process has adequate memory
and writes a different case JSON. `route_only` is intentionally absent: with
64 source requests per original NPU its four low-V lanes exhaust their finite
backlog, so it is recorded as an invalid ablation in `route_only.error.log`.

## Artifact contract

Each valid JSON contains the exact transformed-input fingerprint, workload and
placement hashes, metrics, 500 ms measurement blocks, request rows, and 31
invariant checks. All five valid JSON files must have the same
`fixed_split_input_fingerprint`; all invariants must be true and
`no_backlog_exhaustion` must be true.

The cleanup did not delete historical work. A local sibling archive holds the
old untracked/ignored files and a pre-cleanup Git patch; tracked history is also
recoverable from commit `9b7e3320551732856d2715a607a24284b68778f2`.
