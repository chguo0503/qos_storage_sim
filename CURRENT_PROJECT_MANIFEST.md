# Current project manifest

## Scope

This is the complete retained project for the 32-NPU/5-SSU fixed 28:4 V/B
experiment. Historical material is intentionally outside the checkout.

The saved result was produced before cleanup at Git commit
`9b7e3320551732856d2715a607a24284b68778f2`. Cleanup did not change the runner,
simulator, policy, workload input, or JSON artifacts. It changed only project
organization, documentation, and the plotter's default output format.

## Runtime dependency closure

The runner's complete local dependency closure is:

```text
run_vb_fixed_split_npu32_ssu5_experiment.py
authenticated_workload_inputs.py
continuous_batch_control.py
continuous_batch_sim.py
continuous_prefill_client.py
continuous_prefill_workload.py
policy_logic.py
random_steady_state_workload.py
sim.py
six_request_workload.py
strategy_profiles.py
vb_pool_policy.py
data
```

The plotter additionally requires `plot_vb_fixed_split_npu32_ssu5_4s.py` and
matplotlib. NumPy and matplotlib are the only non-standard dependencies.

## Frozen input identity

All five valid case JSON files agree on:

```text
fixed split input  66c60cc004653aafdd1ecb7fd0fe932b74d69cbb4845006c5d98260c26cdf388
raw input          50751065918710c1015a6a885da8eed1a99d3f7ad45d0bd534ed14c2e8393a9a
workload           ea0d14d517bfcc1154742c7c7e35fccb040cf3ea02584ead4b62e1f80f5ab636
placement          e8612c481608b06d12f7228639a31df98c70fc9901222c66d9f7c6ad17e9024b
trace              2d47bc9108bbc2152404362ab3304e2ff20e6c56e4590cb3046c0abf0a134f22
```

Each has 31/31 true simulator invariants, all 32 NPUs represented in the SLO
sample, and `no_backlog_exhaustion=true`.

## Integrity check

Run:

```bash
sha256sum --check CURRENT_PROJECT_SHA256SUMS
```

This covers the full runtime source closure, input, tests, plotter, raw JSON,
summary, failure log, and retained PNGs. A regenerated PNG can have a different
binary hash under another matplotlib version even when its plotted data are
identical; the JSON and input fingerprints are the authoritative evidence.

The JSONs contain legacy `source_sha256` entries for only five files. Use this
manifest, not that smaller subset, when auditing the cleaned project.

## Local archive and Git recovery

The pre-cleanup local archive is:

```text
/home/chguo/work/last_code/qos_storage_sim_legacy_20260904_X2ASSo
```

It contains old results and top-level files that were never committed, four
regenerable PDFs, and `pre_cleanup_tracked.patch`. It is not a runtime
dependency and is deliberately outside the Git repository. Tracked history is
recoverable from the commit named above.
