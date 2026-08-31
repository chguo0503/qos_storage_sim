# selected128 α=1.5 calibration v2 decision evidence

> Mechanical application of the preregistered v2 rule. The decision is reported as evidence and is not applied to the formal campaign.

## Mechanical decision

- Outcome: `TARGET_SELECTED`
- Selected target ratio: `0.6866666666666666`
- Selection branch: `no_qualified_challenger`
- Qualified challengers: `[]`
- Validity gate passed: `True`
- Formal campaign modified: `False`

## Contract

- Case: `adaptive_a1p5_t0_i25ms`
- Seed: `43`
- Measurement duration: `4000 ms`
- SSUs: `16, 20, 24`
- Candidate target ratios: `0.68, 0.6866666666666666, 0.7`
- Current formal target shown for context only: `0.6866666666666666`
- SLO aggregation: equal weight per NPU; request-weighted values are secondary evidence.
- Preregistered rule: `campaigns/selected128_alpha1p5_calibration_rule_v2.json` (`7ed0a55ee970dbeeb3f5c37ed0d8d5bf4193a120712113cd9c7b10fd1323fd96`)

## Per-cell evidence

| target | SSUs | mean NPU util % | α1.5 equal-NPU SLO % | α2 equal-NPU SLO % | requests |
|---:|---:|---:|---:|---:|---:|
| 0.68 | 16 | 73.933545 | 83.015137 | 86.824106 | 962 |
| 0.68 | 20 | 92.697990 | 88.196523 | 90.999830 | 1160 |
| 0.68 | 24 | 97.958817 | 88.970328 | 95.522317 | 1238 |
| 0.6866666666666666 | 16 | 74.196139 | 85.785297 | 88.397890 | 956 |
| 0.6866666666666666 | 20 | 92.913127 | 88.263340 | 91.391966 | 1161 |
| 0.6866666666666666 | 24 | 97.851989 | 88.777553 | 95.437940 | 1235 |
| 0.7 | 16 | 74.022071 | 83.381579 | 86.356133 | 957 |
| 0.7 | 20 | 92.952690 | 88.087057 | 90.304075 | 1159 |
| 0.7 | 24 | 97.871589 | 88.547434 | 95.318485 | 1236 |

## Statistics across SSUs 16/20/24

| target | formal context | util mean/min/max % | α1.5 SLO mean/min/max % | α2 SLO mean/min/max % |
|---:|:---:|:---|:---|:---|
| 0.68 | no | 88.196784/73.933545/97.958817 | 86.727329/83.015137/88.970328 | 91.115418/86.824106/95.522317 |
| 0.6866666666666666 | yes | 88.320418/74.196139/97.851989 | 87.608730/85.785297/88.777553 | 91.742599/88.397890/95.437940 |
| 0.7 | no | 88.282117/74.022071/97.871589 | 86.672024/83.381579/88.547434 | 90.659565/86.356133/95.318485 |

## Challenger qualification and deltas

| challenger | qualified | min Δα1.5 | points Δ≥0.005 | mean Δα1.5 | mean Δutil |
|---:|:---:|---:|---:|---:|---:|
| 0.68 | False | -0.027701595 | 0 | -0.008814005 | -0.001236340 |
| 0.7 | False | -0.024037172 | 0 | -0.009367062 | -0.000383015 |

Deltas above are unrounded fractions (challenger minus default); thresholds are applied before display rounding.

## Tiebreak evidence

```json
{
  "applied": false,
  "stages": []
}
```

## Provenance

- Source fingerprint: `7fc63b4110c9a7161be79945e03fa06c037f883ead61379b96cb51c7cc3ec900`
- Runtime identity: `{"blas_name": "scipy-openblas", "blas_version": "0.3.34.0.0", "numpy_version": "2.5.2", "openblas_configuration": "OpenBLAS 0.3.34.0.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64", "python_implementation": "CPython", "python_version": "3.14.4"}`
- Thread limits: `{"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}`
- JSON evidence: `results/ms_scale_control/selected128_alpha1p5_calibration_v2_analysis/report.json` (`bb6833a468562731b16945c70f15ea9b2fa64907ae5a008599d2005beee50332`)
- CSV rows: `results/ms_scale_control/selected128_alpha1p5_calibration_v2_analysis/rows.csv` (`92415b0cddfb1c81a92671063e36344d760cec5d8b06ab3097051c045bbc2f11`)

The mechanical v2 decision is not a campaign mutation; applying any change to the formal campaign requires a separate explicit action.
