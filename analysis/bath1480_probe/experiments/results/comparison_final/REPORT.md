# Phase 1/2/3/4/5 Comparison (per-model bars)

Dataset: bath_1480, 3504 rxns  

- **Phase 1** = paper reproduction (Espley 46 features, Espley DFT labels)
- **Phase 2** = xTB-augmented, **VarianceThreshold τ=1e-10** (61 features)
- **Phase 3** = xTB-augmented, **VarianceThreshold τ=0.05** (46 features)
- **Phase 5** = xTB-augmented, **no filter** (63 features, paper's exact methodology)
- **Phase 4** = XGB + MACE residual, our own labels + EDA (independent architecture)

Phase 2/3/5 use identical HPs (from hps_phase23.pkl), same targets (our self-consistent d1_own/d2_own + 6 EDA channels), same 5 seeds (22, 23, 14, 1, 2). Only feature count differs by τ.

## Test MAE (mean ± std across 5 seeds) per channel × model × phase

### d1

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.313 ± 0.222 | 0.085 |
| Phase 1 (paper) | krr | 2.986 ± 0.122 | 0.077 |
| Phase 1 (paper) | svr | 2.643 ± 0.150 | 0.068 |
| Phase 1 (paper) | rf | 2.952 ± 0.123 | 0.076 |
| Phase 2 (τ=1e-10) | ridge | 2.564 ± 0.022 | 0.047 |
| Phase 2 (τ=1e-10) | krr | 2.714 ± 0.090 | 0.050 |
| Phase 2 (τ=1e-10) | svr | 2.210 ± 0.113 | 0.041 |
| Phase 2 (τ=1e-10) | rf | 2.206 ± 0.055 | 0.041 |
| Phase 2 (τ=1e-10) | xgb | 1.791 ± 0.062 | 0.033 |
| Phase 3 (τ=0.05) | ridge | 2.576 ± 0.016 | 0.048 |
| Phase 3 (τ=0.05) | krr | 2.655 ± 0.104 | 0.049 |
| Phase 3 (τ=0.05) | svr | 2.167 ± 0.134 | 0.040 |
| Phase 3 (τ=0.05) | rf | 2.186 ± 0.074 | 0.040 |
| Phase 3 (τ=0.05) | xgb | 1.823 ± 0.056 | 0.034 |
| Phase 5 (no filter) | ridge | 2.565 ± 0.022 | 0.048 |
| Phase 5 (no filter) | krr | 2.718 ± 0.098 | 0.050 |
| Phase 5 (no filter) | svr | 2.238 ± 0.124 | 0.041 |
| Phase 5 (no filter) | rf | 2.213 ± 0.071 | 0.041 |
| Phase 5 (no filter) | xgb | 1.815 ± 0.057 | 0.034 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.333 ± 0.128 | 0.046 |

### d2

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.156 ± 0.131 | 0.087 |
| Phase 1 (paper) | krr | 2.765 ± 0.155 | 0.077 |
| Phase 1 (paper) | svr | 2.370 ± 0.125 | 0.066 |
| Phase 1 (paper) | rf | 2.610 ± 0.113 | 0.072 |
| Phase 2 (τ=1e-10) | ridge | 2.093 ± 0.088 | 0.046 |
| Phase 2 (τ=1e-10) | krr | 2.467 ± 0.088 | 0.055 |
| Phase 2 (τ=1e-10) | svr | 1.879 ± 0.046 | 0.042 |
| Phase 2 (τ=1e-10) | rf | 1.795 ± 0.086 | 0.040 |
| Phase 2 (τ=1e-10) | xgb | 1.469 ± 0.059 | 0.033 |
| Phase 3 (τ=0.05) | ridge | 2.092 ± 0.093 | 0.046 |
| Phase 3 (τ=0.05) | krr | 2.470 ± 0.106 | 0.055 |
| Phase 3 (τ=0.05) | svr | 1.900 ± 0.071 | 0.042 |
| Phase 3 (τ=0.05) | rf | 1.874 ± 0.076 | 0.042 |
| Phase 3 (τ=0.05) | xgb | 1.569 ± 0.042 | 0.035 |
| Phase 5 (no filter) | ridge | 2.093 ± 0.088 | 0.046 |
| Phase 5 (no filter) | krr | 2.489 ± 0.109 | 0.055 |
| Phase 5 (no filter) | svr | 1.922 ± 0.052 | 0.043 |
| Phase 5 (no filter) | rf | 1.820 ± 0.082 | 0.040 |
| Phase 5 (no filter) | xgb | 1.458 ± 0.053 | 0.032 |
| Phase 4 (XGB+MACE) | XGB+MACE | 1.923 ± 0.046 | 0.040 |

### interaction

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.101 ± 0.216 | 0.089 |
| Phase 1 (paper) | krr | 2.918 ± 0.151 | 0.084 |
| Phase 1 (paper) | svr | 2.400 ± 0.156 | 0.069 |
| Phase 1 (paper) | rf | 2.742 ± 0.168 | 0.079 |
| Phase 2 (τ=1e-10) | ridge | 2.250 ± 0.263 | 0.050 |
| Phase 2 (τ=1e-10) | krr | 2.342 ± 0.116 | 0.052 |
| Phase 2 (τ=1e-10) | svr | 1.858 ± 0.101 | 0.041 |
| Phase 2 (τ=1e-10) | rf | 2.543 ± 0.111 | 0.056 |
| Phase 2 (τ=1e-10) | xgb | 1.833 ± 0.108 | 0.040 |
| Phase 3 (τ=0.05) | ridge | 2.363 ± 0.222 | 0.052 |
| Phase 3 (τ=0.05) | krr | 2.419 ± 0.125 | 0.053 |
| Phase 3 (τ=0.05) | svr | 1.964 ± 0.119 | 0.043 |
| Phase 3 (τ=0.05) | rf | 2.559 ± 0.110 | 0.057 |
| Phase 3 (τ=0.05) | xgb | 1.883 ± 0.121 | 0.042 |
| Phase 5 (no filter) | ridge | 2.251 ± 0.264 | 0.050 |
| Phase 5 (no filter) | krr | 2.363 ± 0.112 | 0.052 |
| Phase 5 (no filter) | svr | 1.887 ± 0.104 | 0.042 |
| Phase 5 (no filter) | rf | 2.526 ± 0.103 | 0.056 |
| Phase 5 (no filter) | xgb | 1.840 ± 0.137 | 0.041 |

### pauli

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 7.361 ± 0.508 | 0.035 |
| Phase 2 (τ=1e-10) | krr | 10.100 ± 0.658 | 0.048 |
| Phase 2 (τ=1e-10) | svr | 5.732 ± 0.171 | 0.027 |
| Phase 2 (τ=1e-10) | rf | 8.727 ± 0.275 | 0.041 |
| Phase 2 (τ=1e-10) | xgb | 5.298 ± 0.178 | 0.025 |
| Phase 3 (τ=0.05) | ridge | 8.215 ± 0.432 | 0.039 |
| Phase 3 (τ=0.05) | krr | 9.906 ± 0.743 | 0.047 |
| Phase 3 (τ=0.05) | svr | 6.085 ± 0.225 | 0.029 |
| Phase 3 (τ=0.05) | rf | 9.029 ± 0.266 | 0.043 |
| Phase 3 (τ=0.05) | xgb | 5.705 ± 0.167 | 0.027 |
| Phase 5 (no filter) | ridge | 7.362 ± 0.514 | 0.035 |
| Phase 5 (no filter) | krr | 10.216 ± 0.725 | 0.048 |
| Phase 5 (no filter) | svr | 5.886 ± 0.200 | 0.028 |
| Phase 5 (no filter) | rf | 8.752 ± 0.256 | 0.041 |
| Phase 5 (no filter) | xgb | 5.346 ± 0.165 | 0.025 |
| Phase 4 (XGB+MACE) | XGB+MACE | 10.122 ± 0.602 | 0.045 |

### oi

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 3.966 ± 0.226 | 0.040 |
| Phase 2 (τ=1e-10) | krr | 4.476 ± 0.278 | 0.045 |
| Phase 2 (τ=1e-10) | svr | 3.057 ± 0.133 | 0.031 |
| Phase 2 (τ=1e-10) | rf | 5.178 ± 0.252 | 0.052 |
| Phase 2 (τ=1e-10) | xgb | 2.930 ± 0.084 | 0.029 |
| Phase 3 (τ=0.05) | ridge | 4.191 ± 0.134 | 0.042 |
| Phase 3 (τ=0.05) | krr | 4.480 ± 0.236 | 0.045 |
| Phase 3 (τ=0.05) | svr | 3.176 ± 0.176 | 0.032 |
| Phase 3 (τ=0.05) | rf | 5.146 ± 0.247 | 0.052 |
| Phase 3 (τ=0.05) | xgb | 3.155 ± 0.112 | 0.032 |
| Phase 5 (no filter) | ridge | 3.968 ± 0.229 | 0.040 |
| Phase 5 (no filter) | krr | 4.542 ± 0.324 | 0.046 |
| Phase 5 (no filter) | svr | 3.104 ± 0.151 | 0.031 |
| Phase 5 (no filter) | rf | 5.127 ± 0.277 | 0.051 |
| Phase 5 (no filter) | xgb | 2.920 ± 0.060 | 0.029 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.118 ± 0.300 | 0.047 |

### elst

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 4.267 ± 0.202 | 0.038 |
| Phase 2 (τ=1e-10) | krr | 5.216 ± 0.244 | 0.047 |
| Phase 2 (τ=1e-10) | svr | 3.769 ± 0.182 | 0.034 |
| Phase 2 (τ=1e-10) | rf | 5.395 ± 0.106 | 0.048 |
| Phase 2 (τ=1e-10) | xgb | 3.680 ± 0.083 | 0.033 |
| Phase 3 (τ=0.05) | ridge | 4.791 ± 0.144 | 0.043 |
| Phase 3 (τ=0.05) | krr | 5.314 ± 0.336 | 0.048 |
| Phase 3 (τ=0.05) | svr | 4.048 ± 0.225 | 0.036 |
| Phase 3 (τ=0.05) | rf | 5.536 ± 0.113 | 0.050 |
| Phase 3 (τ=0.05) | xgb | 3.951 ± 0.126 | 0.035 |
| Phase 5 (no filter) | ridge | 4.267 ± 0.202 | 0.038 |
| Phase 5 (no filter) | krr | 5.267 ± 0.240 | 0.047 |
| Phase 5 (no filter) | svr | 3.808 ± 0.156 | 0.034 |
| Phase 5 (no filter) | rf | 5.382 ± 0.124 | 0.048 |
| Phase 5 (no filter) | xgb | 3.718 ± 0.091 | 0.033 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.692 ± 0.279 | 0.048 |

### disp

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 0.012 ± 0.001 | 0.000 |
| Phase 2 (τ=1e-10) | krr | 0.867 ± 0.037 | 0.028 |
| Phase 2 (τ=1e-10) | svr | 0.365 ± 0.021 | 0.012 |
| Phase 2 (τ=1e-10) | rf | 0.127 ± 0.020 | 0.004 |
| Phase 2 (τ=1e-10) | xgb | 0.082 ± 0.019 | 0.003 |
| Phase 3 (τ=0.05) | ridge | 0.011 ± 0.000 | 0.000 |
| Phase 3 (τ=0.05) | krr | 0.840 ± 0.050 | 0.027 |
| Phase 3 (τ=0.05) | svr | 0.347 ± 0.024 | 0.011 |
| Phase 3 (τ=0.05) | rf | 0.063 ± 0.014 | 0.002 |
| Phase 3 (τ=0.05) | xgb | 0.098 ± 0.012 | 0.003 |
| Phase 5 (no filter) | ridge | 0.012 ± 0.001 | 0.000 |
| Phase 5 (no filter) | krr | 0.882 ± 0.043 | 0.028 |
| Phase 5 (no filter) | svr | 0.380 ± 0.021 | 0.012 |
| Phase 5 (no filter) | rf | 0.131 ± 0.018 | 0.004 |
| Phase 5 (no filter) | xgb | 0.088 ± 0.007 | 0.003 |
| Phase 4 (XGB+MACE) | XGB+MACE | 0.061 ± 0.015 | 0.002 |

### cpcm

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 2.606 ± 0.155 | 0.060 |
| Phase 2 (τ=1e-10) | krr | 2.798 ± 0.095 | 0.065 |
| Phase 2 (τ=1e-10) | svr | 2.401 ± 0.100 | 0.056 |
| Phase 2 (τ=1e-10) | rf | 3.073 ± 0.153 | 0.071 |
| Phase 2 (τ=1e-10) | xgb | 2.399 ± 0.043 | 0.056 |
| Phase 3 (τ=0.05) | ridge | 2.797 ± 0.169 | 0.065 |
| Phase 3 (τ=0.05) | krr | 2.861 ± 0.089 | 0.066 |
| Phase 3 (τ=0.05) | svr | 2.525 ± 0.065 | 0.059 |
| Phase 3 (τ=0.05) | rf | 3.071 ± 0.154 | 0.071 |
| Phase 3 (τ=0.05) | xgb | 2.434 ± 0.072 | 0.056 |
| Phase 5 (no filter) | ridge | 2.606 ± 0.155 | 0.060 |
| Phase 5 (no filter) | krr | 2.802 ± 0.089 | 0.065 |
| Phase 5 (no filter) | svr | 2.418 ± 0.111 | 0.056 |
| Phase 5 (no filter) | rf | 3.068 ± 0.145 | 0.071 |
| Phase 5 (no filter) | xgb | 2.428 ± 0.051 | 0.056 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.852 ± 0.085 | 0.060 |

### cds

| Phase | Model | MAE | NMAE |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 0.378 ± 0.029 | 0.061 |
| Phase 2 (τ=1e-10) | krr | 0.367 ± 0.017 | 0.059 |
| Phase 2 (τ=1e-10) | svr | 0.318 ± 0.015 | 0.051 |
| Phase 2 (τ=1e-10) | rf | 0.375 ± 0.017 | 0.060 |
| Phase 2 (τ=1e-10) | xgb | 0.279 ± 0.012 | 0.045 |
| Phase 3 (τ=0.05) | ridge | 0.391 ± 0.021 | 0.063 |
| Phase 3 (τ=0.05) | krr | 0.389 ± 0.017 | 0.063 |
| Phase 3 (τ=0.05) | svr | 0.341 ± 0.017 | 0.055 |
| Phase 3 (τ=0.05) | rf | 0.389 ± 0.013 | 0.063 |
| Phase 3 (τ=0.05) | xgb | 0.296 ± 0.011 | 0.048 |
| Phase 5 (no filter) | ridge | 0.378 ± 0.029 | 0.061 |
| Phase 5 (no filter) | krr | 0.369 ± 0.017 | 0.059 |
| Phase 5 (no filter) | svr | 0.320 ± 0.014 | 0.051 |
| Phase 5 (no filter) | rf | 0.374 ± 0.018 | 0.060 |
| Phase 5 (no filter) | xgb | 0.280 ± 0.015 | 0.045 |
| Phase 4 (XGB+MACE) | XGB+MACE | 0.422 ± 0.006 | 0.059 |

