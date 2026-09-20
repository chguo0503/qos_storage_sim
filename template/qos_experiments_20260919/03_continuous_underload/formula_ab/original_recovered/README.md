# 8 NPU x/y underload experiments

All capacities in this experiment are **decimal GB/s**: 40 per SSU and 50 per NPU. Native volumes are GiB and converted explicitly. There are 8 layers/request and batch size 1.

The primary measurement interval is 2000–4000 ms. `window_sensitivity.csv` also reports 500–4500 ms and 4000–4500 ms. Every NPU remains active throughout every reported interval. All full traces run to completion; drain-period utilization is stored separately and is not the primary comparison.

## Results and scope

- 72 unique native strategy runs / 36 paired cases.
- 10 valid fixed-composition configurations (seed 7).
- 8 valid random-profile configurations, each with seeds 7, 19, 43.
- 34 pairs meet both full-run per-SSU underload audits and identical-input checks.
- E2/S2 and C2/S3 are excluded paired cases due to ordinary ring-hash demand exceeding 40 GB/s in at least one policy. Qualified replacements use E2/S3 and C2/S4. The excluded raw results are retained.
- A 16K/20K/24K compute time is a linear extrapolation from the data's 32K/48K lengths, with NQL interpolation as needed. Every configuration records its anchors and weights. This requires hardware calibration before treating simulated results as measured performance.

Fixed mode: NPUs 0 through n_A-1 run A repeatedly; the remaining NPUs run B. Random mode: every NPU independently shuffles its A:B=1:12 count mixture. This count mixture is not substituted for instantaneous concurrency in the bandwidth audit. Current request occupancy and actual per-request ring-hash volume determine the full-run audit.

L0 boundary prefetch demand is exempted from the ordinary profile demand. The entire boundary interval is never removed. All observed boundary waits remain in NPU utilization. The pending-demand audit covers k>=1 using each layer's all-SSU completion time as a conservative per-SSU endpoint.

## Files

- `source/`: unchanged native simulator Python sources and original `data` (84 profiles).
- `configs/`: the 36 actually simulated inputs. Paired policies use the same config.
- `results/<case>/<strategy>/native_summary.json.gz`: complete native request/layer timestamps and native invariants.
- `requests.json`: every request, NPU, generation, A/B profile and per-SSU volume.
- `analysis.json`: exact window compute/active accounting, group/per-NPU utilization, SLO, full/warm demand audits, input fingerprint and source hashes.
- `outputs/paired_summary.csv`: means/ranges and paired qualification; includes rejected pairs.
- `outputs/all_runs.csv`: individual seed results, full demand peaks, overload and near-capacity time fractions.
- `outputs/per_npu.csv`: each NPU's A/B input counts, actual warm roles and utilization.
- `outputs/window_sensitivity.csv`: alternate windows from the same complete trajectories.
- `outputs/boundary_bounds.csv`: cross-type L0 waits and a necessary physical lower bound; excess over that lower bound is not automatically all avoidable.
- `outputs/verification.json`: completed runs, paired input checks, gzip decoding checks, accounting and source-artifact hashes.
- `theory_summary.json`: x/y, ideal integer S, ideal pooled-read ratio and isolated-A physical lower bounds under actual ring placement and the receive cap.
- `screen/`: analytical candidate selection, including an unrun Y3 candidate clearly separated from simulated configurations.
- `outputs/xy_underload_experiment_report.md/.pdf`: Chinese report; plots have matching PNG/PDF versions.

## Reproduce

Python package versions used here are in `requirements.txt`. Run commands from this directory.

```bash
python run_experiment.py configs/Y1_fixed_s1_seed7.json --strategy baseline --force
python run_experiment.py configs/Y1_fixed_s1_seed7.json --strategy once --force
```

`--force` recomputes that local result. Without it, a completed run is cached. To regenerate the complete matrix while preserving delivered results, first rename `results` to a separate archive directory, then run:

```bash
python run_batch.py all_jobs.json --workers 6
python verify_and_export.py
python build_plots.py
python summarize_theory.py
python boundary_bounds.py
python build_report.py --pdf
```

Simulations run in separate processes because the existing exact-tail adapter patches module-level functions. Do not parallelize simulations as threads in one Python interpreter. Raw summary writes are closed, decoded for verification, and atomically placed before analysis is saved.

The input fingerprint and byte-identical request metadata were verified within every FIFO/Once pair. The preserved native source files were not edited. Two raw compressed outputs were regenerated after an incomplete-file check; the repeated input fingerprints, fleet utilization and total stall matched exactly. The final archive contains only the restored complete native summaries; `recovery/` retains the prior derived metrics for the repeat check.

The SLO×1.5 metric measures NPU admission to completion against 1.5 times eight layers' compute time. Initial queue wait is excluded. This definition differs from arrival-to-completion latency.
