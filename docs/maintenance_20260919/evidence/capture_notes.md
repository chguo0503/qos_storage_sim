# 2026-09-19 package migration regression evidence

## Outcome

All three policies have byte-identical request metrics, layer metrics, ordered block completion timestamps, and admission/completion timestamps before and after migration. Makespan, full-fleet utilization and both benchmark windows also agree exactly. The original source snapshot was captured before root modules moved.

- Before core tests: 33 passed, 1 deselected in 1.73 s. The deselected archived placement test needs a historical result outside the snapshot.
- After complete suite: 59 passed in 16.65 s. This includes the archived placement test and the new migration tests.
- Every benchmark policy fully drains 128 requests and 47,104 blocks (7.90625 GiB), with 96 cross-request L0 prefetches.

## Small deterministic workload

32 NPUs, 3 SSUs, 8 layers, 4 requests/NPU, all arrivals at zero, fixed FIFO assignment and native Ring Hash placement. Four constructed profiles rotate across NPU/queue position: (24 blocks, 120 us/layer), (96,1500), (48,550), (16,300); every block is 176 KiB. This is a regression fixture, not a workload claim or a replacement for the prior study. Benchmark windows are **2–12 ms and 8–20 ms**, not seconds.

| Policy | Makespan (ms) | Full U (%) | Before wall (s) | After wall (s) |
|---|---:|---:|---:|---:|
| asu_baseline | 70.295418603516 | 28.109940011100 | 1.216 | 1.248 |
| od_baseline | 72.032372338867 | 27.432110533638 | 1.333 | 1.361 |
| once | 72.169326074219 | 27.380053375694 | 3.368 | 3.417 |

Wall timings are single unprofiled executions, not a speedup measurement. Differences of a few percent may reflect host scheduling/load. The initial profiled Once run is intentionally excluded from that comparison.

## Replay commands

Run with `/home/chguo/miniforge3/bin/python` (Python 3.10.10). Output directories must not already exist; use a new suffix when replaying.

Before core tests (from `snapshot/`):

```bash
python -m pytest -q -p no:cacheprovider --basetemp /tmp/qos_refactor_before/pytest_tmp test_od_baseline.py test_shared_path_sim_adapter.py test_shared_path_policies.py test_shared_ssu_state.py test_coflow_experiment_inputs.py test_ring_hash_placement.py -k 'not archived_stripe'
```

Before capture with Once CPU profile:

```bash
timeout 55s python /tmp/qos_refactor_before/benchmark.py --source-root /tmp/qos_refactor_before/snapshot --output /tmp/qos_refactor_before/baseline
```

Before repeat without profiling:

```bash
timeout 55s python /tmp/qos_refactor_before/benchmark.py --source-root /tmp/qos_refactor_before/snapshot --output /tmp/qos_refactor_before/unprofiled --profile-policy none --compare-to /tmp/qos_refactor_before/baseline/benchmark.json
```

After capture:

```bash
timeout 55s python /tmp/qos_refactor_before/benchmark_after.py --source-root /home/chguo/work/last_code/qos_storage_sim --output /tmp/qos_refactor_before/after --profile-policy none --compare-to /tmp/qos_refactor_before/unprofiled/benchmark.json
```

After complete tests (from project root):

```bash
python -m pytest -q -p no:cacheprovider --basetemp /tmp/qos_refactor_before/pytest_after_final_tmp
```

## Artifacts

- `snapshot.json`: original root source/data SHA-256 and capture metadata; `snapshot/`: actual original files.
- `migration_map.json`: parent agent's old-module to package-module mapping.
- `tests_before.{json,log}`, `tests_after.{json,log}`: exact argv, cwd, exit status, timing and console output.
- `baseline/`, `unprofiled/`, `after/`: frozen input, per-policy complete raw results, signatures, source hashes, timings and benchmark summary.
- `comparison.json`: paired deterministic results and individual elapsed host times.
- `baseline/once.prof`, `baseline/once.profile.txt`: cProfile data and readable ranking.
- `historical_runtime_evidence.json`: comparable-input historical run times with concurrency caveats.
- Repository `tests/fixtures/pre_refactor_trace.json.gz`, `pre_refactor_expected.json.gz`: compact immutable input and expected per-layer/request times; no dependency on /tmp for normal tests.

## What the new tests cover

The scoped shared adapter restores every patched event method after nested exceptions. The SLO extension also restores its adapter factory and Once routing hook after exceptions. Controller contracts re-export the same classes, preserving type identity. AST and actual subprocess execution verify the simulator does not import inputs/results; the subprocess receives a physical copy of simulator alone. Six policy modules have isolated executable examples. Manifest legacy exports are the same function objects as the new module; save/load preserves input identity. Source provenance covers every nested simulator/input module. Catalog convenience execution matches explicit prepared-input execution. The public API matches the old wrapper's complete simulated summary and routing/ownership counters.

The first After test run had 55 passes and four failures in newly written assertions: three incorrectly compared host CPU timing counters; one compared mutable NPUState object identity. The tests were corrected to exclude only the three explicit host timer fields and compare actual NPU values/summary. No simulator/input source change was necessary. Final suite passes.

## Measured runtime hotspots

The Once profile has 38,000,592 calls in 7.647 s (profiling overhead included). The routing path is prominent: 47,104 `_projection_choices` calls consume 3.269 s cumulative; 3,072 `plan` calls consume 3.598 s. `paths_per_group` is read 2,734,855 times (1.052 s cumulative). The event plane also matters: `_build_qos_arbitration_cache` has 28,743 calls/1.369 s cumulative; `_static_qos_service_rates` 28,743/0.510 s. These costs overlap and must not be added together.

Historical same-input remote examples: full OD 323.984 s versus Once 768.557 s for 9,411,584 blocks; semi OD 216.872 s versus Once 454.003 s for 6,661,120 blocks. Concurrent host load prevents interpreting these as controlled speed ratios. The small unprofiled fixture independently shows that Once spends more host time, but migration alone did not optimize that algorithm.


## Final CLI/facade follow-up

After the complete 59-test run, the parent corrected the new CLI seed default and restored an archived SLO facade export. The CLI test verifies a frozen manifest's seed 19 is used by default, explicit `--seed 7` overrides submission seed, both preserve the manifest fingerprint, and their full summaries equal direct API execution with the selected seed. The original diverse SLO policy's four tests also pass. Targeted follow-up: **5 passed in 0.41 s**, recorded in `tests_followup.{json,log}`. There are now 60 tests under `tests/` (59 previously passed plus this one additional passing test); the four archived SLO tests are outside that default test path.

```bash
python -m pytest -q -p no:cacheprovider tests/test_package_regressions.py::test_manifest_cli_inherits_seed_and_allows_explicit_override results/diverse_data_ssu3_l3_20260916/test_policy.py
```

The benchmark's source hashes describe its exact capture time. Later CLI default handling, API input validation and the facade re-export were checked with the targeted tests; no assertion is made that the earlier benchmark hashes include those later source edits. Their valid default simulation scheduling is unchanged.
