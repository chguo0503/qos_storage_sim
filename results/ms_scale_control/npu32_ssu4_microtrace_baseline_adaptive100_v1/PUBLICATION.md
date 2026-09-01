# 32-NPU / 4-SSU microtrace publication

This directory publishes the lightweight, privacy-safe outputs for the paired
`baseline` and `adaptive_t0_i100ms` diagnostic.

Configuration: seed 42, 32 NPUs, 4 SSUs, 128 input requests/NPU, warmup 8
requests/NPU, 500 ms settle, 3 s measurement, native 5 ms accounting blocks,
and an exact 50 ms service trace.  The runner authenticates the historical
input/workload/placement/trace fingerprints and the policy-specific
measurement start before accepting a result.

The `analysis/` publication contains:

- all seven comparison and causal-timeline figures;
- `report.md`, `case_summary.csv`, and machine-readable `validation.json`;
- 0.5 ms fleet/NPU/SSU resource time series and full-measurement 5 ms series;
- both selected layers' exact SSD/link/compute service intervals;
- the frozen formal 8 s convergence series.

The two complete raw directories are retained locally at `raw_baseline/` and
`raw_adaptive100/` (about 140 MiB each).  The 52 MiB all-request
`analysis/service_intervals.csv` is also retained locally.  They are excluded
from the Git publication to avoid adding roughly 330 MiB of reproducible bulk
data; `analysis/selected_layer_service_intervals.csv` is the publishable exact
request-level subset.  The full raw manifest SHA-256 values are:

- Baseline: `4d11c657ae9c839bae64246e9187e7b9eb689bd94d746d6a4e756210b8a66f5f`
- Adaptive 100 ms: `1c1bcec34725ab232365c714e92948f9b9a91c05da62c977d430fc648d1378bc`

Reproduce the analysis with:

```bash
python analyze_npu32_ssu5_microtrace.py \
  --input-root results/ms_scale_control/npu32_ssu4_microtrace_baseline_adaptive100_v1/raw_baseline \
  --input-root results/ms_scale_control/npu32_ssu4_microtrace_baseline_adaptive100_v1/raw_adaptive100 \
  --historical-formal-json results/ms_scale_control/frozen32_ssu4_selected7_measure8_backing128.json \
  --output-dir results/ms_scale_control/npu32_ssu4_microtrace_baseline_adaptive100_v1/analysis
```

Despite the legacy filenames, both diagnostic scripts authenticate and support
SSU counts 4 and 5.
