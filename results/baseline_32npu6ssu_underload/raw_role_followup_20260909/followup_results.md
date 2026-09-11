**原始 data 固定与混合：补测结果**

[2000,4000) ms U; admission-clock TTFT proxy <=1.5*8C, all warm admissions followed to completion. All predeclared seeds retained; 4th by1500 reported separately.

已完成 60/60；解析错误 0。单seed候选探索与五seed确认分开阅读，不删除表现不佳的种子。

| 配置 | 模式 | seed | 策略 | NPU U% | 暖接纳SLO% | 短角色U% | 全程逐盘峰值GiB/s | 全32卡active | 第4条≤1500ms | 欠载 |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| raw176_three_l20 | fixed | 19 | baseline | 91.0564 | 92.1717 | 76.1503 | 38.021409 | True | True | True |
| raw176_three_l20 | fixed | 19 | once | 100.0000 | 100.0000 | 100.0000 | 38.339599 | True | True | True |
| raw176_three_l20 | fixed | 43 | baseline | 91.4207 | 94.1176 | 77.1220 | 38.127473 | True | True | True |
| raw176_three_l20 | fixed | 43 | once | 99.9930 | 100.0000 | 99.9814 | 38.233536 | True | True | True |
| raw176_three_l20 | fixed | 67 | baseline | 91.3779 | 94.1026 | 77.0078 | 38.233536 | True | True | True |
| raw176_three_l20 | fixed | 67 | once | 99.9982 | 100.0000 | 99.9952 | 38.233536 | True | True | True |
| raw176_three_l20 | fixed | 101 | baseline | 91.2365 | 89.4207 | 76.6306 | 38.127473 | True | True | True |
| raw176_three_l20 | fixed | 101 | once | 99.9986 | 100.0000 | 99.9963 | 38.180504 | True | True | True |
| raw176_three_l20 | fixed | 7 | baseline | 91.3233 | 92.0918 | 76.8621 | 38.127473 | True | True | True |
| raw176_three_l20 | fixed | 7 | once | 99.9975 | 100.0000 | 99.9935 | 38.180504 | True | True | True |
| raw176_three_l20 | mixed | 7 | baseline | 99.9610 | 100.0000 | 99.9714 | 39.107430 | True | True | True |
| raw176_three_l20 | mixed | 7 | once | 99.8072 | 100.0000 | 99.8546 | 39.333192 | True | True | True |
| raw160_three_l20 | fixed | 19 | baseline | 91.3665 | 100.0000 | 76.9774 | 37.957206 | True | True | True |
| raw160_three_l20 | fixed | 19 | once | 99.9976 | 100.0000 | 99.9937 | 38.116301 | True | True | True |
| raw160_three_l20 | mixed | 19 | baseline | 99.9759 | 100.0000 | 99.9910 | 38.717406 | True | True | True |
| raw160_three_l20 | mixed | 19 | once | 99.8772 | 100.0000 | 99.9891 | 38.756637 | True | True | True |
| raw160_three_l20 | fixed | 43 | baseline | 91.4889 | 100.0000 | 77.3037 | 37.957206 | True | True | True |
| raw160_three_l20 | fixed | 43 | once | 100.0000 | 100.0000 | 100.0000 | 38.063269 | True | True | True |
| raw160_three_l20 | mixed | 43 | baseline | 99.9829 | 100.0000 | 99.9755 | 39.090544 | True | True | True |
| raw160_three_l20 | mixed | 43 | once | 99.9730 | 100.0000 | 100.0000 | 38.915732 | True | True | True |
| raw160_three_l20 | fixed | 67 | baseline | 91.9422 | 100.0000 | 78.5124 | 38.063269 | True | True | True |
| raw160_three_l20 | fixed | 67 | once | 100.0000 | 100.0000 | 100.0000 | 38.063269 | True | True | True |
| raw160_three_l20 | mixed | 67 | baseline | 99.9880 | 100.0000 | 99.9959 | 39.037513 | True | True | True |
| raw160_three_l20 | mixed | 67 | once | 99.8535 | 100.0000 | 99.9811 | 38.915732 | True | True | True |
| raw160_three_l20 | fixed | 101 | baseline | 91.7151 | 100.0000 | 77.9070 | 37.957206 | True | True | True |
| raw160_three_l20 | fixed | 101 | once | 99.9913 | 100.0000 | 99.9769 | 38.010237 | True | True | True |
| raw160_three_l20 | mixed | 101 | baseline | 99.9959 | 100.0000 | 100.0000 | 38.753554 | True | True | True |
| raw160_three_l20 | mixed | 101 | once | 99.9718 | 100.0000 | 100.0000 | 38.809669 | True | True | True |
| raw200_three_l20 | fixed | 19 | baseline | 88.2672 | 81.2834 | 68.7125 | 38.232777 | True | True | True |
| raw200_three_l20 | fixed | 19 | once | 99.9862 | 100.0000 | 99.9632 | 38.484135 | True | True | True |
| raw200_three_l20 | mixed | 19 | baseline | 99.8973 | 100.0000 | 99.8969 | 39.376479 | True | True | True |
| raw200_three_l20 | mixed | 19 | once | 99.7951 | 100.0000 | 99.9693 | 39.503381 | True | True | True |
| raw200_three_l20 | fixed | 43 | baseline | 88.4601 | 84.7411 | 69.2269 | 38.285809 | True | True | True |
| raw200_three_l20 | fixed | 43 | once | 100.0000 | 100.0000 | 100.0000 | 38.431103 | True | True | True |
| raw200_three_l20 | mixed | 43 | baseline | 99.9640 | 100.0000 | 99.9915 | 39.618772 | True | True | True |
| raw200_three_l20 | mixed | 43 | once | 99.9357 | 100.0000 | 99.9632 | 39.480494 | True | True | True |
| raw200_three_l20 | fixed | 67 | baseline | 88.5480 | 84.0220 | 69.4613 | 38.431103 | True | True | True |
| raw200_three_l20 | fixed | 67 | once | 100.0000 | 100.0000 | 100.0000 | 38.431103 | True | True | True |
| raw200_three_l20 | mixed | 67 | baseline | 99.8222 | 100.0000 | 99.8169 | 39.609301 | True | True | True |
| raw200_three_l20 | mixed | 67 | once | 99.7981 | 100.0000 | 99.9588 | 39.376479 | True | True | True |
| raw200_three_l20 | fixed | 101 | baseline | 88.4453 | 79.6247 | 69.1875 | 38.325040 | True | True | True |
| raw200_three_l20 | fixed | 101 | once | 99.9985 | 100.0000 | 99.9959 | 38.378072 | True | True | True |
| raw200_three_l20 | mixed | 101 | baseline | 99.9136 | 100.0000 | 99.9269 | 39.284449 | True | True | True |
| raw200_three_l20 | mixed | 101 | once | 99.8591 | 100.0000 | 100.0000 | 39.284449 | True | True | True |
| raw160_three_l20 | fixed | 7 | baseline | 91.7349 | 100.0000 | 77.9599 | 37.957206 | True | True | True |
| raw160_three_l20 | fixed | 7 | once | 99.9988 | 100.0000 | 99.9969 | 38.010237 | True | True | True |
| raw160_three_l20 | mixed | 7 | baseline | 99.9970 | 100.0000 | 99.9971 | 39.037513 | True | True | True |
| raw160_three_l20 | mixed | 7 | once | 99.9497 | 100.0000 | 99.9659 | 38.823469 | True | True | True |
| raw200_three_l20 | fixed | 7 | baseline | 88.5527 | 82.9670 | 69.4738 | 38.325040 | True | True | True |
| raw200_three_l20 | fixed | 7 | once | 99.9985 | 100.0000 | 99.9961 | 38.378072 | True | True | True |
| raw200_three_l20 | mixed | 7 | baseline | 99.9433 | 100.0000 | 99.9869 | 39.646394 | True | True | True |
| raw200_three_l20 | mixed | 7 | once | 99.8635 | 100.0000 | 99.9522 | 39.604565 | True | True | True |
| raw160_three_l16 | fixed | 7 | baseline | 93.6455 | 100.0000 | 87.2910 | 37.377821 | True | True | True |
| raw160_three_l16 | fixed | 7 | once | 99.9316 | 100.0000 | 99.8631 | 37.470084 | True | True | True |
| raw160_one_l16 | fixed | 7 | baseline | 87.8193 | 100.0000 | 75.6387 | 35.172471 | True | True | True |
| raw160_one_l16 | fixed | 7 | once | 99.9999 | 100.0000 | 99.9998 | 35.172471 | True | True | True |
| raw200_one_l19 | fixed | 7 | baseline | 84.4489 | 58.4475 | 61.7205 | 36.394418 | True | True | True |
| raw200_one_l19 | fixed | 7 | once | 99.9998 | 100.0000 | 99.9996 | 36.394418 | True | True | True |
| raw200_fast_l17 | fixed | 7 | baseline | 89.7528 | 51.3326 | 78.1392 | 39.211552 | True | False | True |
| raw200_fast_l17 | fixed | 7 | once | 92.8394 | 91.3700 | 84.7241 | 39.211552 | True | False | True |

fixed/mixed每个配置和seed使用同一批原请求，保留C、V、到达和实际落盘。固定模式长卡只处理长画像，短卡独立打乱其短画像；mixed把同一批请求按画像均匀分给32卡后独立打乱。

名义需求沿用当前接纳请求逐盘V/C，不额外叠加下一请求首层预取；真实预取仍完整仿真。4th≤1500ms是此前附加的暖机判据，2048类长计算可能在纯计算上就无法满足，不能把这类结果悄悄并入满足旧判据的结果。
