# 32-NPU Baseline layer timelines

This directory is a self-contained, reproducible runner for the three Baseline
micro-timeline cases discussed in the report.  It can be copied out of the
repository and run on its own.  The included `sim.py`,
`continuous_batch_sim.py`, `policy_logic.py`, `continuous_batch_control.py`
and `data` are frozen snapshots of the authoritative simulator inputs; the
package does not contain a second, simplified storage scheduler.

## Fixed cases

| case | NPU | SSU | layers | request profile |
|---|---:|---:|---:|---|
| `homogeneous-layer0` | 32 | 11 | 1 | every NPU uses `(48,512)` |
| `homogeneous-layer012` | 32 | 11 | 3 | every NPU uses `(48,512)` |
| `mixed-layer012` | 32 | 4 | 3 | 30 F=`(200,256)` plus two V=`(32,2048)` on NPU 24/26 |

Every NPU submits exactly one request at `t=0`.  Ingress networking and
cross-request Layer-0 prefetch are disabled.  All physical I/O uses Baseline
Path0; every SSU is a 40 GB/s non-preemptive server and every NPU receive link
is 50 GB/s.

The mixed case is a **synchronized cold F-burst microscope**.  It contains no
S phase, does not replay the F×5→S×11 cycle, is not the warm formal-R run, and
cannot by itself support the formal-R long-window utilization result.

## Run

From the repository root:

```bash
python -m pip install -r standalone/npu32_baseline_timelines/requirements.txt
MPLCONFIGDIR=/tmp/qos-baseline-timeline-mpl \
  python standalone/npu32_baseline_timelines/run_timelines.py --case all
```

Run one fixed case or choose another output root:

```bash
python standalone/npu32_baseline_timelines/run_timelines.py \
  --case mixed-layer012 \
  --output-dir /tmp/reproduced-baseline-timelines
```

The compact default omits the large per-block CSV while retaining it in
memory for every validation and plot.  Add `--write-physical-trace` when the
full auditable transition table is needed:

```bash
python standalone/npu32_baseline_timelines/run_timelines.py \
  --case mixed-layer012 \
  --write-physical-trace
```

## Outputs and color semantics

Each case gets its own directory below
`standalone/npu32_baseline_timelines/results/` by default (or below
`--output-dir`).  It contains the small per-request timeline, `summary.json`,
`report.md`, and:

- `01_*`: per-NPU I/O/compute timeline;
- `02_*`: exact combined Path0 enqueue rank in every SSU;
- `03_*`: physical non-preemptive SSD service timeline;
- `04_*`: physical enqueue time by layer (three-layer cases only);
- `01b_*`: L0/L1 view of the unchanged three-layer mixed run.

The `results/` stored beside this runner are its canonical figures.  Earlier
figures under the repository-level `results/` used several legacy plotters and
therefore differ in layout even though the authenticated simulator summaries
and physical event traces are identical.  This runner deliberately normalizes
all deadline coloring to the semantics below.

In every `01` timeline, blue is only I/O before that NPU-layer's deadline,
red hatching is only I/O beyond the deadline, and green is NPU compute.  The
dashed mark is the deadline and the black tick is I/O Ready.  For Layer0 the
deadline is a comparison budget; the whole cold read is the real barrier.
For later layers, the red interval equals actual I/O barrier wait.

`01b` is a view of the same `n_layers=3` simulation; it does **not** rerun a
two-layer model and therefore introduces no early-release/truncation effect.

Starting at Layer1, "warm cumulative utilization" retains compute from Layer0
but excludes only the Layer0 cold-read barrier.  It is therefore not a metric
restricted to Layer1 and later compute.

## Reproducibility checks

The runner executes each simulator case twice: once with read-only physical
transition instrumentation and once without it.  It rejects the result unless
the two complete simulator summaries have identical SHA-256 hashes.  It also
checks request placement, Path0 ordering, SSD/link rates, non-overlap,
I/O-ready causality, layer release/deadline semantics, and frozen reference
metrics from the original runs.

Python 3.10 or newer is required.  The recorded reference environment is
CPython 3.10.10, NumPy 2.2.6 and Matplotlib 3.10.9.  A full three-case run takes
about 20 seconds and the largest case needs roughly 1 GB peak memory in that
environment.  Numerical/event results are the scientific reproduction target.
PNG byte hashes may differ across font/FreeType/zlib builds even when the
underlying event trace is identical.
