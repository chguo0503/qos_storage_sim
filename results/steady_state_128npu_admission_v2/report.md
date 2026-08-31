# 128-NPU real-data explicit-tail admission V2

This isolated run uses the frozen seed-42 real-data trace, 16 layers, batch 1, four warmup completions per NPU, 500-ms settle, and the same 2,000-ms measurement window. The 25-ms controller is event-gated minimum spacing, not a periodic timer.

Default complete: `true`; optional SSU70 complete: `true`; selected complete: `true`; source stable: `true`; config stable: `true`.

Source fingerprint: `4328c6d73e413d75104cbd9ef48a24897f2ff2f1b4e6a797975a75fdd6cb2c37`

Config fingerprint: `38b983ce35441f6f671740d2e55d3f58d047f6764e7388dde332d26132a4886b`

| SSU | Strategy | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Requests | Evals | Commits | Path writes | Wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | admission_v2_25ms | 16.66% | 21.18% | 32.51% | 52.10% | 53.88% | 8.33% | 90.91% | 59.09% | 55.87% | 722 | 249 | 248 | 754209 | 1020.2s |
| 40 | admission_v2_25ms | 43.95% | 47.75% | 51.66% | 79.20% | 79.81% | 31.56% | 100.00% | 87.29% | 100.00% | 1139 | 162 | 162 | 821977 | 1150.8s |
| 70 | admission_v2_25ms | 88.16% | 89.23% | 90.89% | 98.55% | 98.57% | 94.20% | 100.00% | 100.00% | 100.00% | 1964 | 131 | 130 | 585135 | 2213.3s |

## Input fingerprints by SSU

- SSU24: assignment=`f426e5cf68b8ad7209cc475e3d7a59e04b0ecc81bc86fb69db2049d305aa5289`, workload=`7b0afef5684d25f0fb2dbec043fb9e31113cda97319a54ff086f885fd07367a0`, placement=`a3651593bbaec14465431cbb921a9e1cf9b831b0980f355554653ba007aedc58`, trace=`6b076d2c8ec5d898f9d1f7fc77cf78a5396a86932fc768a0caa42acc121289f1`, simulator=`60fecc129ab712d41a2c7622de246b1a9265d5b9d37de61b43e50954c8bb4f8d`
- SSU40: assignment=`f426e5cf68b8ad7209cc475e3d7a59e04b0ecc81bc86fb69db2049d305aa5289`, workload=`b25e12ce0e8a459c9d787cdaf24d22f6e53faae3ae84d44f87a31e31b24398b5`, placement=`b7c84e5692ac5166edd7e8f30db60495b3ad1c68eafd08bc106f11b39c151d6a`, trace=`cf8f41c66beea221cad59f56eb5f5e504bbcc834cde34ed909016f68d6b436c3`, simulator=`18612c10e17c1df8d2949bd605073cc4d7a5133b41215c58aea98ac7f522669e`
- SSU70: assignment=`f426e5cf68b8ad7209cc475e3d7a59e04b0ecc81bc86fb69db2049d305aa5289`, workload=`4d5cc0de548f64277aff8aee3be3a40605a53654437c178cbb9ae768de6d18eb`, placement=`50f39faebc120d3d06d211da3ea1f3028ab5a09d21935dab850e4fcb5196d546`, trace=`3a4aaa42d54b5c0c0fc7f48468eded6cc87868b394d30e64570fa3120a7281ae`, simulator=`42194116b32574c92b8b46ac560799e56bc7183d7390da168ed8a86cbbfa2e15`
