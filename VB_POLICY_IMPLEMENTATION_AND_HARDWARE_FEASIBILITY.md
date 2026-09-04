# V/B policy used by the retained 28:4 experiment

This document describes only the code that remains in this checkout. It
separates simulator facts from hardware assumptions and avoids mixing older
V/B, AreaGuard, or dynamic-CIR experiments into the current result.

## 1. What is actually compared

All cases use the same 2,048-request master draw, KV placement, 32 NPUs, five
40 GiB/s SSUs, 16 layers, and one request at a time per NPU. The runner first
classifies every request by V, then deterministically redistributes requests:

- NPU 0–3 receive only low-V requests;
- NPU 4–31 receive only high-V requests;
- each group is sorted by `request_id` and assigned round-robin to its lanes;
- every case reuses the same physical transformed trace and placement.

The stored `fixed_split_input_fingerprint` is identical across cases. The
later `simulated_input_fingerprint` intentionally differs between Baseline and
V/B because that fingerprint includes the LL/LS category relabel. This is not
physical input drift: request ID, NPU, arrival, profile, and placement remain
identical.

This fixed separation is an experimental workload construction, not an online
scheduler that migrates live model state between NPUs.

Baseline and V/B differ in two coupled ways:

| Case | Path routing | QoS table |
|---|---|---|
| Baseline | every I/O uses Path 0; no pressure read | four-category static table 20/6/8/6 GiB/s per SSU |
| V/B | one pressure snapshot per request-layer-SSU, then choose within its pool | static LL/LS split selected before the run |

Therefore the measured delta cannot be attributed solely to the V formula or
solely to multi-Path routing. The failed `route_only` run was intended to
separate those effects, but its finite low-V input was exhausted.

## 2. Request values and classification

For a request profile:

- `C = per_layer_us / 1e6` seconds of layer compute;
- `D = per_layer_kv_gb` GiB read per layer;
- `B = D / C` GiB/s is its ideal masking-bandwidth demand;
- `V = C / B = C² / D` seconds²/GiB.

The retained cutoff is `V > 0.00031`. Above it, the request is relabeled LL;
otherwise it is relabeled LS. Relabeling preserves request ID, NPU, arrival,
profile, layer count, and physical KV placement. The implementation is in
[`vb_pool_policy.py`](vb_pool_policy.py) and the materialization step in
[`run_vb_fixed_split_npu32_ssu5_experiment.py`](run_vb_fixed_split_npu32_ssu5_experiment.py).

V is a heuristic measure of compute window per unit masking bandwidth. It is
not a proof of marginal fleet-utilization gain: it omits current backlog,
coflow barriers, request age, placement hotspots, and harm to the other pool.

## 3. Static LL/LS bandwidth tables

Each SSU has 256 Paths arranged as eight Groups. Per Group, the category layout
is `SS×12, SL×4, LS×12, LL×4`, hence 96 LS Paths and 32 LL Paths per SSU.
The requested pool CIR is divided evenly among its Paths. SS and SL receive
zero CIR in a V/B case; PIR is unlimited; Path and Group weights are one.

The runner evaluates four fixed per-SSU splits, all summing to 40 GiB/s:

| Case | LL | LS |
|---|---:|---:|
| `vb_catalog` | 24.567578 | 15.432422 |
| `vb_duration_aware` | 21.998287 | 18.001713 |
| `vb_ll36` | 36.000000 | 4.000000 |
| `vb_split_aware` | 37.619103 | 2.380897 |

These CIRs are installed once before simulation. There is no periodic or
millisecond-level CIR update in this experiment. Unused capacity is
work-conservingly redistributed by the simulator's Group/Path weights.

## 4. Path selection and SSU service

For each request-layer-SSU, `layer_once` reads one immutable pressure snapshot.
The ABI exposes queued-plus-active outstanding-I/O counts and aggregate
active-Path state; it does not expose queued bytes, I/O age, or deadline. The
planner approximates each Path's old work as `count × median block size of the
current planning batch`. It then adds each newly planned block's real size to
a software shadow, processing larger new blocks first, and chooses only within
the request's LL or LS category. With `pressure_ttl_ms=0`, every query is fresh.

The SSU model is command-level and non-preemptive:

1. each Path is FCFS;
2. CIR establishes a base service share for backlogged Paths;
3. unused capacity is distributed by Group and Path weights;
4. those rates determine virtual completion order;
5. one physical command at a time occupies each SSU backend at 40 GiB/s.

An in-flight command is not preempted. The model has no device queue-depth
limit, NAND/media tail latency, PCIe transaction overhead, finite buffer
backpressure, or cost for reading pressure. NPU link capacity is modeled as
50 GiB/s.

## 5. Inputs and outputs

Inputs are the authenticated 84-profile `data` table, seed 42, fixed hardware
constants, and runner arguments. The default master draw is 64 requests per
original NPU. Its high/low split is then reassigned into 28/4 persistent lanes.

Each case JSON records:

- source, workload, placement, raw-input, and transformed-input fingerprints;
- the exact configuration and static pool split;
- fleet/cohort NPU utilization and NPU·ms compute area;
- fleet/per-SSU utilization;
- per-request `ttft_ms` and `ideal_ttft_ms`, from which the plotter derives the
  TTFT/ideal ratio;
- eight 500 ms measurement blocks;
- simulator invariants and backlog-exhaustion status.

The four plots are projections of those JSON files; they are not independent
measurements. `summary.csv` is regenerated by the same plotting script.

## 6. What the result supports

For this one seed and 4 s window:

- Baseline: 92.908964% fleet NPU utilization;
- best unconstrained V/B: 93.285583%, but max TTFT/ideal is 9.474589;
- only guard-feasible non-degrading V/B: 93.009984%, max TTFT/ideal 7.935386;
- guard-feasible gain: +0.101019 percentage points.

The eight block-wise deltas change sign, so +0.101019 pp is an observation,
not a statistically established improvement. The fixed 28:4 workload also
makes the 28 high-V lanes almost fully utilized under Baseline, leaving little
compute area for any policy to recover.

See [`VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md`](VB_FIXED_SPLIT_28_HIGH_4_LOW_4S_REPORT.md)
for the complete table and plots.

## 7. Hardware feasibility boundary

The following mechanisms are plausible to implement, but are not validated by
this simulator result:

- compute C, D, B, and V from known model/profile metadata;
- attach an LL/LS tag at request admission;
- install a static two-pool CIR table;
- read SSU Path pressure and select a legal Path at a layer boundary.

Before claiming production feasibility, hardware measurements must establish:

- exact GB/GiB and time units;
- real Path/Group/CIR/PIR/weight and idle-bandwidth borrowing semantics;
- pressure snapshot contents, atomicity, p50/p99 read latency, and host cost;
- races between multiple clients reading the same snapshot and enqueueing I/O;
- CIR write/ack/readback latency and in-flight-command behavior;
- queue-depth, media-tail, PCIe, link, and buffer effects missing here;
- prediction error in C/D across layer, batch, model version, and NPU load.

The current result does not prove that V/B improves other seeds, that every
protected request receives B, that `TTFT/ideal <= 8` holds outside this window,
or that simulator gains transfer unchanged to real SSUs.

## 8. Current implementation map

- [`run_vb_fixed_split_npu32_ssu5_experiment.py`](run_vb_fixed_split_npu32_ssu5_experiment.py): input construction, cases, guard, output.
- [`vb_pool_policy.py`](vb_pool_policy.py): B/V computation and LL/LS QoS tables.
- [`policy_logic.py`](policy_logic.py): pressure-aware Path projection.
- [`sim.py`](sim.py): Path queues and static-CIR/WRR command scheduler.
- [`continuous_batch_sim.py`](continuous_batch_sim.py): NPU/layer lifecycle, measurement window, metrics, and invariants.
- [`CURRENT_PROJECT_MANIFEST.md`](CURRENT_PROJECT_MANIFEST.md): full retained dependency hashes.
