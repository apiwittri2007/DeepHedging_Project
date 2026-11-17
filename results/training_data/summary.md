## CVaR₉₅ of hedging P&L on the real test set (n = 2650, mean ± sd over seeds)

| Arm | 0 bps | 5 bps | 10 bps | 15 bps | 25 bps | 50 bps |
|---|---:|---:|---:|---:|---:|---:|
| A0 GBM, known vol | 5.825 ± 0.061 | 5.948 ± 0.074 | 6.120 ± 0.077 | 6.210 ± 0.083 | 6.505 ± 0.080 | 7.160 ± 0.067 |
| A1 GBM, estimated vol | 5.856 ± 0.089 | 5.959 ± 0.080 | 6.074 ± 0.069 | 6.240 ± 0.052 | 6.512 ± 0.074 | 7.161 ± 0.114 |
| A2 GARCH-t | 6.532 ± 1.063 | 10.262 ± 12.744 | 7.833 ± 4.493 | 7.510 ± 1.902 | 6.984 ± 0.367 | 10.580 ± 9.124 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 5.997 ± 0.261 | 6.160 ± 0.215 | 6.227 ± 0.236 | 6.453 ± 0.323 | 6.773 ± 0.317 | 7.405 ± 0.202 |
| A3 Block bootstrap | 5.837 ± 0.095 | 5.974 ± 0.109 | 6.115 ± 0.094 | 6.276 ± 0.150 | 6.519 ± 0.177 | 7.225 ± 0.198 |
| A4 Real windows | 5.897 ± 0.089 | 6.021 ± 0.099 | 6.172 ± 0.110 | 6.285 ± 0.105 | 6.579 ± 0.120 | 7.220 ± 0.108 |
| A5 GBM → real fine-tune | 5.815 ± 0.077 | 5.947 ± 0.086 | 6.070 ± 0.067 | 6.214 ± 0.067 | 6.466 ± 0.083 | 7.098 ± 0.059 |
| L Linear correction | 5.768 | 5.916 | 6.064 | 6.214 | 6.516 | 7.286 |
| HW Min-variance delta (Hull–White form) | 6.029 | 6.187 | 6.345 | 6.504 | 6.824 | 7.632 |
| BS delta | 5.886 | 6.041 | 6.196 | 6.352 | 6.666 | 7.465 |
| Delta + band | 5.912 | 6.060 | 6.208 | 6.358 | 6.659 | 7.426 |
| No hedge | 34.431 | 34.431 | 34.431 | 34.431 | 34.431 | 34.431 |

## CVaR difference vs BS delta (negative = less tail risk), month-clustered 95% CI

Single-test percentile bootstrap (not corrected for multiple testing; the corrected, primary
inference is the multiple-testing table below, which uses a slightly different bootstrap convention).

| Arm | Cost (bps) | Diff | 95% CI | p(not better) |
|---|---:|---:|---|---:|
| A0 GBM, known vol | 0 | -0.061 | [-0.162, +0.030] | 0.103 |
| A1 GBM, estimated vol | 0 | -0.030 | [-0.153, +0.086] | 0.299 |
| A2 GARCH-t | 0 | +0.646 | [+0.231, +1.098] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 0 | +0.111 | [-0.092, +0.345] | 0.836 |
| A3 Block bootstrap | 0 | -0.049 | [-0.280, +0.206] | 0.348 |
| A4 Real windows | 0 | +0.011 | [-0.145, +0.190] | 0.546 |
| A5 GBM → real fine-tune | 0 | -0.071 | [-0.194, +0.070] | 0.164 |
| HW Min-variance delta (Hull–White form) | 0 | +0.143 | [+0.037, +0.240] | 0.994 |
| L Linear correction | 0 | -0.118 | [-0.217, -0.024] | 0.004 |
| A0 GBM, known vol | 5 | -0.093 | [-0.220, +0.029] | 0.072 |
| A1 GBM, estimated vol | 5 | -0.083 | [-0.226, +0.050] | 0.122 |
| A2 GARCH-t | 5 | +4.221 | [+1.314, +6.782] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 5 | +0.119 | [-0.120, +0.401] | 0.798 |
| A3 Block bootstrap | 5 | -0.067 | [-0.323, +0.212] | 0.318 |
| A4 Real windows | 5 | -0.020 | [-0.188, +0.173] | 0.413 |
| A5 GBM → real fine-tune | 5 | -0.094 | [-0.230, +0.060] | 0.114 |
| HW Min-variance delta (Hull–White form) | 5 | +0.146 | [+0.040, +0.243] | 0.994 |
| L Linear correction | 5 | -0.126 | [-0.225, -0.032] | 0.003 |
| A0 GBM, known vol | 10 | -0.077 | [-0.205, +0.043] | 0.108 |
| A1 GBM, estimated vol | 10 | -0.123 | [-0.247, -0.004] | 0.020 |
| A2 GARCH-t | 10 | +1.636 | [+0.506, +2.699] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 10 | +0.030 | [-0.200, +0.281] | 0.586 |
| A3 Block bootstrap | 10 | -0.081 | [-0.338, +0.208] | 0.285 |
| A4 Real windows | 10 | -0.024 | [-0.205, +0.186] | 0.397 |
| A5 GBM → real fine-tune | 10 | -0.127 | [-0.286, +0.048] | 0.080 |
| HW Min-variance delta (Hull–White form) | 10 | +0.149 | [+0.041, +0.245] | 0.995 |
| L Linear correction | 10 | -0.132 | [-0.232, -0.040] | 0.002 |
| A0 GBM, known vol | 15 | -0.143 | [-0.276, -0.028] | 0.012 |
| A1 GBM, estimated vol | 15 | -0.112 | [-0.250, +0.018] | 0.049 |
| A2 GARCH-t | 15 | +1.158 | [+0.560, +1.803] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 15 | +0.100 | [-0.129, +0.338] | 0.794 |
| A3 Block bootstrap | 15 | -0.076 | [-0.378, +0.254] | 0.328 |
| A4 Real windows | 15 | -0.067 | [-0.257, +0.147] | 0.272 |
| A5 GBM → real fine-tune | 15 | -0.139 | [-0.303, +0.040] | 0.069 |
| HW Min-variance delta (Hull–White form) | 15 | +0.152 | [+0.043, +0.248] | 0.995 |
| L Linear correction | 15 | -0.139 | [-0.239, -0.048] | 0.002 |
| A0 GBM, known vol | 25 | -0.161 | [-0.312, -0.019] | 0.012 |
| A1 GBM, estimated vol | 25 | -0.153 | [-0.322, +0.005] | 0.029 |
| A2 GARCH-t | 25 | +0.318 | [+0.109, +0.500] | 0.999 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 25 | +0.107 | [-0.180, +0.395] | 0.751 |
| A3 Block bootstrap | 25 | -0.147 | [-0.507, +0.233] | 0.230 |
| A4 Real windows | 25 | -0.087 | [-0.340, +0.181] | 0.267 |
| A5 GBM → real fine-tune | 25 | -0.200 | [-0.398, +0.005] | 0.029 |
| HW Min-variance delta (Hull–White form) | 25 | +0.158 | [+0.047, +0.253] | 0.995 |
| L Linear correction | 25 | -0.150 | [-0.252, -0.063] | 0.001 |
| A0 GBM, known vol | 50 | -0.306 | [-0.519, -0.104] | 0.001 |
| A1 GBM, estimated vol | 50 | -0.304 | [-0.536, -0.090] | 0.004 |
| A2 GARCH-t | 50 | +3.115 | [+1.996, +3.816] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 50 | -0.060 | [-0.457, +0.350] | 0.372 |
| A3 Block bootstrap | 50 | -0.240 | [-0.718, +0.259] | 0.173 |
| A4 Real windows | 50 | -0.246 | [-0.628, +0.151] | 0.118 |
| A5 GBM → real fine-tune | 50 | -0.367 | [-0.668, -0.066] | 0.007 |
| HW Min-variance delta (Hull–White form) | 50 | +0.167 | [+0.056, +0.265] | 0.998 |
| L Linear correction | 50 | -0.179 | [-0.284, -0.094] | 0.000 |

## CVaR difference vs A1 GBM, estimated vol (negative = less tail risk), month-clustered 95% CI

Single-test percentile bootstrap (not corrected for multiple testing; the corrected, primary
inference is the multiple-testing table below, which uses a slightly different bootstrap convention).

| Arm | Cost (bps) | Diff | 95% CI | p(not better) |
|---|---:|---:|---|---:|
| A0 GBM, known vol | 0 | -0.031 | [-0.082, +0.022] | 0.120 |
| A2 GARCH-t | 0 | +0.676 | [+0.326, +1.072] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 0 | +0.141 | [-0.021, +0.353] | 0.949 |
| A3 Block bootstrap | 0 | -0.019 | [-0.250, +0.216] | 0.461 |
| A4 Real windows | 0 | +0.041 | [-0.140, +0.223] | 0.689 |
| A5 GBM → real fine-tune | 0 | -0.041 | [-0.186, +0.101] | 0.314 |
| HW Min-variance delta (Hull–White form) | 0 | +0.173 | [+0.034, +0.310] | 0.992 |
| L Linear correction | 0 | -0.088 | [-0.201, +0.016] | 0.051 |
| A0 GBM, known vol | 5 | -0.011 | [-0.075, +0.061] | 0.396 |
| A2 GARCH-t | 5 | +4.304 | [+1.478, +6.805] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 5 | +0.202 | [+0.013, +0.443] | 0.985 |
| A3 Block bootstrap | 5 | +0.015 | [-0.231, +0.264] | 0.572 |
| A4 Real windows | 5 | +0.062 | [-0.110, +0.249] | 0.776 |
| A5 GBM → real fine-tune | 5 | -0.012 | [-0.153, +0.129] | 0.466 |
| HW Min-variance delta (Hull–White form) | 5 | +0.228 | [+0.080, +0.374] | 0.999 |
| L Linear correction | 5 | -0.043 | [-0.159, +0.067] | 0.214 |
| A0 GBM, known vol | 10 | +0.046 | [-0.001, +0.102] | 0.973 |
| A2 GARCH-t | 10 | +1.759 | [+0.693, +2.763] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 10 | +0.153 | [-0.022, +0.370] | 0.956 |
| A3 Block bootstrap | 10 | +0.041 | [-0.209, +0.306] | 0.641 |
| A4 Real windows | 10 | +0.098 | [-0.083, +0.295] | 0.858 |
| A5 GBM → real fine-tune | 10 | -0.004 | [-0.147, +0.150] | 0.491 |
| HW Min-variance delta (Hull–White form) | 10 | +0.272 | [+0.131, +0.412] | 1.000 |
| L Linear correction | 10 | -0.010 | [-0.121, +0.090] | 0.427 |
| A0 GBM, known vol | 15 | -0.030 | [-0.087, +0.027] | 0.150 |
| A2 GARCH-t | 15 | +1.271 | [+0.718, +1.878] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 15 | +0.213 | [+0.065, +0.399] | 0.998 |
| A3 Block bootstrap | 15 | +0.036 | [-0.250, +0.331] | 0.616 |
| A4 Real windows | 15 | +0.045 | [-0.142, +0.247] | 0.699 |
| A5 GBM → real fine-tune | 15 | -0.026 | [-0.163, +0.122] | 0.386 |
| HW Min-variance delta (Hull–White form) | 15 | +0.265 | [+0.122, +0.405] | 1.000 |
| L Linear correction | 15 | -0.026 | [-0.138, +0.073] | 0.314 |
| A0 GBM, known vol | 25 | -0.008 | [-0.067, +0.055] | 0.421 |
| A2 GARCH-t | 25 | +0.471 | [+0.318, +0.613] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 25 | +0.261 | [+0.081, +0.486] | 0.999 |
| A3 Block bootstrap | 25 | +0.006 | [-0.284, +0.315] | 0.540 |
| A4 Real windows | 25 | +0.067 | [-0.134, +0.273] | 0.750 |
| A5 GBM → real fine-tune | 25 | -0.046 | [-0.179, +0.089] | 0.260 |
| HW Min-variance delta (Hull–White form) | 25 | +0.311 | [+0.154, +0.469] | 1.000 |
| L Linear correction | 25 | +0.003 | [-0.115, +0.115] | 0.510 |
| A0 GBM, known vol | 50 | -0.002 | [-0.075, +0.066] | 0.484 |
| A2 GARCH-t | 50 | +3.419 | [+2.319, +4.089] | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 50 | +0.244 | [+0.028, +0.508] | 0.991 |
| A3 Block bootstrap | 50 | +0.064 | [-0.307, +0.446] | 0.635 |
| A4 Real windows | 50 | +0.058 | [-0.219, +0.345] | 0.660 |
| A5 GBM → real fine-tune | 50 | -0.063 | [-0.233, +0.109] | 0.223 |
| HW Min-variance delta (Hull–White form) | 50 | +0.471 | [+0.242, +0.707] | 1.000 |
| L Linear correction | 50 | +0.125 | [-0.062, +0.303] | 0.898 |

## Realism of training data (mean over folds)

| Source | Excess kurtosis | ACF of abs returns (lags 1–10) | Hill tail index |
|---|---:|---:|---:|
| Real (pre-year) | 6.69 | 0.039 | 3.48 |
| A1 GBM, estimated vol | 0.11 | -0.012 | 6.93 |
| A2 GARCH-t | 135.98 | 0.113 | 3.19 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 24.33 | 0.066 | 3.31 |
| A3 Block bootstrap | 7.74 | 0.043 | 3.41 |

## Folds

| Test year | Test windows | Real training windows | Stocks | GARCH stocks | Unavailable |
|---:|---:|---:|---:|---:|---|
| 2016 | 104 | 22308 | 9 | 8 | — |
| 2017 | 110 | 24531 | 9 | 9 | — |
| 2018 | 144 | 26810 | 12 | 9 | — |
| 2019 | 183 | 29730 | 15 | 9 | — |
| 2020 | 215 | 33448 | 18 | 12 | — |
| 2021 | 289 | 37915 | 22 | 15 | — |
| 2022 | 332 | 43702 | 28 | 18 | — |
| 2023 | 333 | 50623 | 29 | 21 | — |
| 2024 | 341 | 57670 | 29 | 28 | — |
| 2025 | 341 | 64804 | 29 | 29 | — |
| 2026 | 258 | 71938 | 29 | 29 | — |

## Multiple testing: every arm × cost vs BS delta as one family (m = 60)

Month-clustered bootstrap shared by all tests; one-sided (arm has lower CVaR).

| Arm | Cost (bps) | Diff | 95% CI | p raw | p Holm | p Romano–Wolf |
|---|---:|---:|---|---:|---:|---:|
| A0 GBM, known vol | 0 | -0.061 | [-0.153, +0.040] | 0.106 | 1.000 | 0.584 |
| A1 GBM, estimated vol | 0 | -0.030 | [-0.146, +0.093] | 0.325 | 1.000 | 0.887 |
| A2 GARCH-t | 0 | +0.646 | [+0.194, +1.061] | 0.997 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 0 | +0.111 | [-0.123, +0.314] | 0.840 | 1.000 | 0.998 |
| A3 Block bootstrap | 0 | -0.049 | [-0.304, +0.182] | 0.350 | 1.000 | 0.912 |
| A4 Real windows | 0 | +0.011 | [-0.169, +0.166] | 0.556 | 1.000 | 0.977 |
| A5 GBM → real fine-tune | 0 | -0.071 | [-0.211, +0.053] | 0.139 | 1.000 | 0.682 |
| HW Min-variance delta (Hull–White form) | 0 | +0.143 | [+0.046, +0.250] | 0.996 | 1.000 | 1.000 |
| L Linear correction | 0 | -0.118 | [-0.212, -0.019] | 0.014 | 0.728 | 0.094 |
| Delta + band | 0 | +0.026 | [-0.061, +0.109] | 0.725 | 1.000 | 0.996 |
| A0 GBM, known vol | 5 | -0.093 | [-0.215, +0.034] | 0.071 | 1.000 | 0.455 |
| A1 GBM, estimated vol | 5 | -0.083 | [-0.215, +0.061] | 0.136 | 1.000 | 0.622 |
| A2 GARCH-t | 5 | +4.221 | [+1.660, +7.128] | 1.000 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 5 | +0.119 | [-0.163, +0.358] | 0.822 | 1.000 | 0.997 |
| A3 Block bootstrap | 5 | -0.067 | [-0.347, +0.188] | 0.318 | 1.000 | 0.887 |
| A4 Real windows | 5 | -0.020 | [-0.213, +0.148] | 0.417 | 1.000 | 0.940 |
| A5 GBM → real fine-tune | 5 | -0.094 | [-0.249, +0.042] | 0.095 | 1.000 | 0.577 |
| HW Min-variance delta (Hull–White form) | 5 | +0.146 | [+0.049, +0.252] | 0.997 | 1.000 | 1.000 |
| L Linear correction | 5 | -0.126 | [-0.219, -0.026] | 0.009 | 0.513 | 0.068 |
| Delta + band | 5 | +0.018 | [-0.070, +0.101] | 0.650 | 1.000 | 0.990 |
| A0 GBM, known vol | 10 | -0.077 | [-0.196, +0.051] | 0.123 | 1.000 | 0.591 |
| A1 GBM, estimated vol | 10 | -0.123 | [-0.241, +0.002] | 0.027 | 1.000 | 0.226 |
| A2 GARCH-t | 10 | +1.636 | [+0.573, +2.767] | 1.000 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 10 | +0.030 | [-0.220, +0.260] | 0.611 | 1.000 | 0.979 |
| A3 Block bootstrap | 10 | -0.081 | [-0.371, +0.176] | 0.287 | 1.000 | 0.861 |
| A4 Real windows | 10 | -0.024 | [-0.234, +0.156] | 0.400 | 1.000 | 0.937 |
| A5 GBM → real fine-tune | 10 | -0.127 | [-0.302, +0.032] | 0.067 | 1.000 | 0.454 |
| HW Min-variance delta (Hull–White form) | 10 | +0.149 | [+0.053, +0.258] | 0.997 | 1.000 | 1.000 |
| L Linear correction | 10 | -0.132 | [-0.225, -0.032] | 0.006 | 0.336 | 0.049 **✓** |
| Delta + band | 10 | +0.012 | [-0.077, +0.094] | 0.603 | 1.000 | 0.982 |
| A0 GBM, known vol | 15 | -0.143 | [-0.258, -0.009] | 0.018 | 0.943 | 0.142 |
| A1 GBM, estimated vol | 15 | -0.112 | [-0.242, +0.025] | 0.061 | 1.000 | 0.370 |
| A2 GARCH-t | 15 | +1.158 | [+0.514, +1.756] | 1.000 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 15 | +0.100 | [-0.137, +0.330] | 0.819 | 1.000 | 0.997 |
| A3 Block bootstrap | 15 | -0.076 | [-0.407, +0.226] | 0.326 | 1.000 | 0.888 |
| A4 Real windows | 15 | -0.067 | [-0.281, +0.123] | 0.251 | 1.000 | 0.837 |
| A5 GBM → real fine-tune | 15 | -0.139 | [-0.318, +0.026] | 0.057 | 1.000 | 0.408 |
| HW Min-variance delta (Hull–White form) | 15 | +0.152 | [+0.056, +0.261] | 0.998 | 1.000 | 1.000 |
| L Linear correction | 15 | -0.139 | [-0.229, -0.039] | 0.004 | 0.228 | 0.033 **✓** |
| Delta + band | 15 | +0.005 | [-0.082, +0.088] | 0.541 | 1.000 | 0.977 |
| A0 GBM, known vol | 25 | -0.161 | [-0.302, -0.010] | 0.020 | 1.000 | 0.158 |
| A1 GBM, estimated vol | 25 | -0.153 | [-0.312, +0.015] | 0.037 | 1.000 | 0.285 |
| A2 GARCH-t | 25 | +0.318 | [+0.136, +0.528] | 1.000 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 25 | +0.107 | [-0.181, +0.395] | 0.797 | 1.000 | 0.996 |
| A3 Block bootstrap | 25 | -0.147 | [-0.527, +0.213] | 0.218 | 1.000 | 0.786 |
| A4 Real windows | 25 | -0.087 | [-0.354, +0.166] | 0.253 | 1.000 | 0.835 |
| A5 GBM → real fine-tune | 25 | -0.200 | [-0.404, -0.001] | 0.025 | 1.000 | 0.236 |
| HW Min-variance delta (Hull–White form) | 25 | +0.158 | [+0.064, +0.269] | 0.999 | 1.000 | 1.000 |
| L Linear correction | 25 | -0.150 | [-0.236, -0.048] | 0.002 | 0.118 | 0.018 **✓** |
| Delta + band | 25 | -0.006 | [-0.092, +0.082] | 0.471 | 1.000 | 0.947 |
| A0 GBM, known vol | 50 | -0.306 | [-0.508, -0.092] | 0.003 | 0.174 | 0.034 **✓** |
| A1 GBM, estimated vol | 50 | -0.304 | [-0.518, -0.072] | 0.006 | 0.336 | 0.063 |
| A2 GARCH-t | 50 | +3.115 | [+2.414, +4.234] | 1.000 | 1.000 | 1.000 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 50 | -0.060 | [-0.471, +0.337] | 0.402 | 1.000 | 0.931 |
| A3 Block bootstrap | 50 | -0.240 | [-0.739, +0.238] | 0.181 | 1.000 | 0.730 |
| A4 Real windows | 50 | -0.246 | [-0.642, +0.137] | 0.106 | 1.000 | 0.584 |
| A5 GBM → real fine-tune | 50 | -0.367 | [-0.667, -0.066] | 0.011 | 0.583 | 0.094 |
| HW Min-variance delta (Hull–White form) | 50 | +0.167 | [+0.069, +0.278] | 1.000 | 1.000 | 1.000 |
| L Linear correction | 50 | -0.179 | [-0.264, -0.075] | 0.001 | 0.060 | 0.004 **✓** |
| Delta + band | 50 | -0.040 | [-0.117, +0.051] | 0.226 | 1.000 | 0.746 |

## Robustness of the corrected p-values to the bootstrap cluster

Pre-registered: 1-month clusters. A 21-day window can span two months, so 2- and 3-month
clusters are shown too. Rows with any Romano–Wolf p < 0.10.

| Arm | Cost (bps) | Diff | p RW 1m | p RW 2m | p RW 3m | p Holm 1m | p Holm 3m |
|---|---:|---:|---:|---:|---:|---:|---:|
| L Linear correction | 0 | -0.118 | 0.094 | 0.048 | 0.170 | 0.728 | 1.000 |
| L Linear correction | 5 | -0.126 | 0.068 | 0.034 | 0.125 | 0.513 | 1.000 |
| L Linear correction | 10 | -0.132 | 0.049 | 0.024 | 0.091 | 0.336 | 0.837 |
| L Linear correction | 15 | -0.139 | 0.033 | 0.017 | 0.064 | 0.228 | 0.632 |
| L Linear correction | 25 | -0.150 | 0.018 | 0.010 | 0.042 | 0.118 | 0.541 |
| A0 GBM, known vol | 50 | -0.306 | 0.034 | 0.053 | 0.042 | 0.174 | 0.174 |
| A1 GBM, estimated vol | 50 | -0.304 | 0.063 | 0.061 | 0.049 | 0.336 | 0.118 |
| A5 GBM → real fine-tune | 50 | -0.367 | 0.094 | 0.076 | 0.109 | 0.583 | 0.541 |
| L Linear correction | 50 | -0.179 | 0.004 | 0.004 | 0.017 | 0.060 | 0.060 |

## Economic size at 15 and 50 bps (vs BS delta; average premium HK$6.05 per HK$100)

| Arm | Cost (bps) | CVaR change, HK$ per HK$1m notional | % of premium | Mean P&L change, HK$ per HK$1m |
|---|---:|---:|---:|---:|
| A0 GBM, known vol | 15 | -1,428 | -2.4% | -67 |
| A1 GBM, estimated vol | 15 | -1,123 | -1.9% | -14 |
| A2 GARCH-t | 15 | +11,582 | +19.2% | +46 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 15 | +1,004 | +1.7% | +91 |
| A3 Block bootstrap | 15 | -763 | -1.3% | +282 |
| A4 Real windows | 15 | -670 | -1.1% | +369 |
| A5 GBM → real fine-tune | 15 | -1,386 | -2.3% | +469 |
| HW Min-variance delta (Hull–White form) | 15 | +1,522 | +2.5% | +138 |
| L Linear correction | 15 | -1,387 | -2.3% | +240 |
| Delta + band | 15 | +52 | +0.1% | +317 |
| No hedge | 15 | +280,790 | +464.5% | -2,141 |
| A0 GBM, known vol | 50 | -3,057 | -5.1% | +1,066 |
| A1 GBM, estimated vol | 50 | -3,040 | -5.0% | +1,264 |
| A2 GARCH-t | 50 | +31,148 | +51.5% | -1,182 |
| A2b GARCH-t, ν ≥ 4 (sensitivity) | 50 | -601 | -1.0% | +1,569 |
| A3 Block bootstrap | 50 | -2,399 | -4.0% | +1,594 |
| A4 Real windows | 50 | -2,456 | -4.1% | +1,649 |
| A5 GBM → real fine-tune | 50 | -3,668 | -6.1% | +1,696 |
| HW Min-variance delta (Hull–White form) | 50 | +1,671 | +2.8% | -64 |
| L Linear correction | 50 | -1,793 | -3.0% | +254 |
| Delta + band | 50 | -396 | -0.7% | +917 |
| No hedge | 50 | +269,660 | +446.1% | +6,272 |

Primary cost: 15 bps.
