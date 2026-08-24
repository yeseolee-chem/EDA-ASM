# Phase 1/2/3/4/5 v3 (oracle-clean + MAD-normalized)

Dataset: bath_1480, 3504 rxns  

## Methodology notes
- HPs (`hps_phase23_v2.pkl`) are **aliased from paper Arm A tuning** (46 Espley features). No per-arm re-tuning against v2 60-feature space; sklearn arms are therefore slightly HP-underfit for their actual feature set. XGB uses fixed defaults + early-stopping-on-val, so any XGB advantage reported here is conservative under this HP handicap.
- Control arms (Phase 2ctl/3ctl/5ctl) use armB-v2 features paired with Espley original DFT labels — the clean feature-only ablation vs Phase 1.

## Fixes applied
- Removed oracle features `disp_xtb`, `dispd4_xtb`, `eint_total_xtb` from all datasets.
- Fixed strain frag index swap: `strain_1_xtb ↔ strain_2_xtb`.
- NMAE = MAE / meanAD (v9 house standard; label meanAD across cohort, seed-invariant).
- `disp` channel declared **ORACLE** (analytical D4 dispersion) — reported but excluded from best-model ranking.

## Phase definitions
- Phase 1: paper reproduction (Espley features + labels, unchanged)
- Phase 2: xTB-aug, τ=1e-10, our own labels + 6 EDA channels
- Phase 3: xTB-aug, τ=0.05
- Phase 5: xTB-aug, no filter (paper methodology)
- Phase 4: XGB + MACE residual (5-fold CV, different protocol)

## Test MAE per channel × model × phase

### d1

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.313 ± 0.222 | 0.592 |
| Phase 1 (paper) | krr | 2.986 ± 0.122 | 0.533 |
| Phase 1 (paper) | svr | 2.643 ± 0.150 | 0.472 |
| Phase 1 (paper) | rf | 2.952 ± 0.123 | 0.527 |
| Phase 2 (τ=1e-10) | ridge | 2.575 ± 0.033 | 0.335 |
| Phase 2 (τ=1e-10) | krr | 2.696 ± 0.076 | 0.350 |
| Phase 2 (τ=1e-10) | svr | 2.222 ± 0.108 | 0.289 |
| Phase 2 (τ=1e-10) | rf | 2.190 ± 0.059 | 0.285 |
| Phase 2 (τ=1e-10) | xgb | 1.817 ± 0.041 | 0.236 |
| Phase 3 (τ=0.05) | ridge | 2.606 ± 0.014 | 0.339 |
| Phase 3 (τ=0.05) | krr | 2.638 ± 0.085 | 0.343 |
| Phase 3 (τ=0.05) | svr | 2.185 ± 0.107 | 0.284 |
| Phase 3 (τ=0.05) | rf | 2.194 ± 0.068 | 0.285 |
| Phase 3 (τ=0.05) | xgb | 1.806 ± 0.024 | 0.235 |
| Phase 5 (no filter) | ridge | 2.576 ± 0.033 | 0.335 |
| Phase 5 (no filter) | krr | 2.701 ± 0.083 | 0.351 |
| Phase 5 (no filter) | svr | 2.249 ± 0.115 | 0.292 |
| Phase 5 (no filter) | rf | 2.208 ± 0.055 | 0.287 |
| Phase 5 (no filter) | xgb | 1.814 ± 0.023 | 0.236 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.397 ± 0.103 | 0.311 |
| Phase 2ctl (Espley y) | ridge | 3.054 ± 0.243 | 0.545 |
| Phase 2ctl (Espley y) | krr | 2.865 ± 0.118 | 0.512 |
| Phase 2ctl (Espley y) | svr | 2.448 ± 0.114 | 0.437 |
| Phase 2ctl (Espley y) | rf | 2.800 ± 0.072 | 0.500 |
| Phase 2ctl (Espley y) | xgb | 2.224 ± 0.120 | 0.397 |
| Phase 3ctl (Espley y) | ridge | 3.160 ± 0.199 | 0.564 |
| Phase 3ctl (Espley y) | krr | 2.894 ± 0.109 | 0.517 |
| Phase 3ctl (Espley y) | svr | 2.465 ± 0.104 | 0.440 |
| Phase 3ctl (Espley y) | rf | 2.910 ± 0.058 | 0.520 |
| Phase 3ctl (Espley y) | xgb | 2.305 ± 0.087 | 0.412 |
| Phase 5ctl (Espley y) | ridge | 3.057 ± 0.241 | 0.546 |
| Phase 5ctl (Espley y) | krr | 2.887 ± 0.116 | 0.516 |
| Phase 5ctl (Espley y) | svr | 2.482 ± 0.093 | 0.443 |
| Phase 5ctl (Espley y) | rf | 2.801 ± 0.056 | 0.500 |
| Phase 5ctl (Espley y) | xgb | 2.255 ± 0.123 | 0.403 |

### d2

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.156 ± 0.131 | 0.611 |
| Phase 1 (paper) | krr | 2.765 ± 0.155 | 0.535 |
| Phase 1 (paper) | svr | 2.370 ± 0.125 | 0.459 |
| Phase 1 (paper) | rf | 2.610 ± 0.113 | 0.505 |
| Phase 2 (τ=1e-10) | ridge | 2.133 ± 0.088 | 0.321 |
| Phase 2 (τ=1e-10) | krr | 2.460 ± 0.083 | 0.370 |
| Phase 2 (τ=1e-10) | svr | 1.902 ± 0.046 | 0.286 |
| Phase 2 (τ=1e-10) | rf | 1.799 ± 0.071 | 0.270 |
| Phase 2 (τ=1e-10) | xgb | 1.437 ± 0.053 | 0.216 |
| Phase 3 (τ=0.05) | ridge | 2.129 ± 0.098 | 0.320 |
| Phase 3 (τ=0.05) | krr | 2.485 ± 0.097 | 0.373 |
| Phase 3 (τ=0.05) | svr | 1.954 ± 0.071 | 0.294 |
| Phase 3 (τ=0.05) | rf | 1.859 ± 0.064 | 0.279 |
| Phase 3 (τ=0.05) | xgb | 1.553 ± 0.067 | 0.233 |
| Phase 5 (no filter) | ridge | 2.135 ± 0.089 | 0.321 |
| Phase 5 (no filter) | krr | 2.481 ± 0.106 | 0.373 |
| Phase 5 (no filter) | svr | 1.954 ± 0.048 | 0.294 |
| Phase 5 (no filter) | rf | 1.795 ± 0.079 | 0.270 |
| Phase 5 (no filter) | xgb | 1.439 ± 0.068 | 0.216 |
| Phase 4 (XGB+MACE) | XGB+MACE | 1.971 ± 0.035 | 0.296 |
| Phase 2ctl (Espley y) | ridge | 2.766 ± 0.095 | 0.536 |
| Phase 2ctl (Espley y) | krr | 2.584 ± 0.127 | 0.500 |
| Phase 2ctl (Espley y) | svr | 2.134 ± 0.142 | 0.413 |
| Phase 2ctl (Espley y) | rf | 2.374 ± 0.065 | 0.460 |
| Phase 2ctl (Espley y) | xgb | 1.893 ± 0.059 | 0.367 |
| Phase 3ctl (Espley y) | ridge | 2.770 ± 0.126 | 0.536 |
| Phase 3ctl (Espley y) | krr | 2.565 ± 0.124 | 0.497 |
| Phase 3ctl (Espley y) | svr | 2.207 ± 0.150 | 0.427 |
| Phase 3ctl (Espley y) | rf | 2.509 ± 0.101 | 0.486 |
| Phase 3ctl (Espley y) | xgb | 2.037 ± 0.092 | 0.394 |
| Phase 5ctl (Espley y) | ridge | 2.766 ± 0.095 | 0.536 |
| Phase 5ctl (Espley y) | krr | 2.600 ± 0.134 | 0.503 |
| Phase 5ctl (Espley y) | svr | 2.164 ± 0.139 | 0.419 |
| Phase 5ctl (Espley y) | rf | 2.383 ± 0.077 | 0.461 |
| Phase 5ctl (Espley y) | xgb | 1.890 ± 0.042 | 0.366 |

### interaction

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.101 ± 0.216 | 0.647 |
| Phase 1 (paper) | krr | 2.918 ± 0.151 | 0.609 |
| Phase 1 (paper) | svr | 2.400 ± 0.156 | 0.501 |
| Phase 1 (paper) | rf | 2.742 ± 0.168 | 0.572 |
| Phase 2 (τ=1e-10) | ridge | 2.444 ± 0.205 | 0.513 |
| Phase 2 (τ=1e-10) | krr | 2.488 ± 0.137 | 0.522 |
| Phase 2 (τ=1e-10) | svr | 2.020 ± 0.131 | 0.424 |
| Phase 2 (τ=1e-10) | rf | 2.835 ± 0.166 | 0.595 |
| Phase 2 (τ=1e-10) | xgb | 2.040 ± 0.124 | 0.428 |
| Phase 3 (τ=0.05) | ridge | 2.549 ± 0.177 | 0.535 |
| Phase 3 (τ=0.05) | krr | 2.556 ± 0.122 | 0.537 |
| Phase 3 (τ=0.05) | svr | 2.083 ± 0.117 | 0.437 |
| Phase 3 (τ=0.05) | rf | 2.802 ± 0.153 | 0.588 |
| Phase 3 (τ=0.05) | xgb | 2.042 ± 0.100 | 0.429 |
| Phase 5 (no filter) | ridge | 2.445 ± 0.207 | 0.513 |
| Phase 5 (no filter) | krr | 2.510 ± 0.134 | 0.527 |
| Phase 5 (no filter) | svr | 2.052 ± 0.134 | 0.431 |
| Phase 5 (no filter) | rf | 2.823 ± 0.155 | 0.593 |
| Phase 5 (no filter) | xgb | 2.040 ± 0.102 | 0.428 |
| Phase 2ctl (Espley y) | ridge | 3.041 ± 0.218 | 0.634 |
| Phase 2ctl (Espley y) | krr | 2.928 ± 0.171 | 0.611 |
| Phase 2ctl (Espley y) | svr | 2.345 ± 0.142 | 0.489 |
| Phase 2ctl (Espley y) | rf | 2.758 ± 0.166 | 0.575 |
| Phase 2ctl (Espley y) | xgb | 2.127 ± 0.137 | 0.443 |
| Phase 3ctl (Espley y) | ridge | 3.221 ± 0.122 | 0.672 |
| Phase 3ctl (Espley y) | krr | 3.063 ± 0.138 | 0.639 |
| Phase 3ctl (Espley y) | svr | 2.532 ± 0.089 | 0.528 |
| Phase 3ctl (Espley y) | rf | 2.942 ± 0.121 | 0.614 |
| Phase 3ctl (Espley y) | xgb | 2.350 ± 0.140 | 0.490 |
| Phase 5ctl (Espley y) | ridge | 3.042 ± 0.217 | 0.634 |
| Phase 5ctl (Espley y) | krr | 2.951 ± 0.174 | 0.615 |
| Phase 5ctl (Espley y) | svr | 2.355 ± 0.128 | 0.491 |
| Phase 5ctl (Espley y) | rf | 2.755 ± 0.148 | 0.574 |
| Phase 5ctl (Espley y) | xgb | 2.132 ± 0.137 | 0.445 |

### pauli

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 7.463 ± 0.600 | 0.378 |
| Phase 2 (τ=1e-10) | krr | 10.021 ± 0.656 | 0.507 |
| Phase 2 (τ=1e-10) | svr | 5.909 ± 0.222 | 0.299 |
| Phase 2 (τ=1e-10) | rf | 8.752 ± 0.208 | 0.443 |
| Phase 2 (τ=1e-10) | xgb | 5.461 ± 0.176 | 0.276 |
| Phase 3 (τ=0.05) | ridge | 8.410 ± 0.467 | 0.425 |
| Phase 3 (τ=0.05) | krr | 9.922 ± 0.715 | 0.502 |
| Phase 3 (τ=0.05) | svr | 6.567 ± 0.292 | 0.332 |
| Phase 3 (τ=0.05) | rf | 9.017 ± 0.251 | 0.456 |
| Phase 3 (τ=0.05) | xgb | 6.074 ± 0.232 | 0.307 |
| Phase 5 (no filter) | ridge | 7.465 ± 0.606 | 0.378 |
| Phase 5 (no filter) | krr | 10.137 ± 0.734 | 0.513 |
| Phase 5 (no filter) | svr | 6.074 ± 0.257 | 0.307 |
| Phase 5 (no filter) | rf | 8.742 ± 0.206 | 0.442 |
| Phase 5 (no filter) | xgb | 5.566 ± 0.174 | 0.282 |
| Phase 4 (XGB+MACE) | XGB+MACE | 10.320 ± 0.555 | 0.522 |

### oi

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 3.996 ± 0.221 | 0.392 |
| Phase 2 (τ=1e-10) | krr | 4.427 ± 0.287 | 0.434 |
| Phase 2 (τ=1e-10) | svr | 3.102 ± 0.128 | 0.304 |
| Phase 2 (τ=1e-10) | rf | 5.167 ± 0.189 | 0.507 |
| Phase 2 (τ=1e-10) | xgb | 3.009 ± 0.059 | 0.295 |
| Phase 3 (τ=0.05) | ridge | 4.244 ± 0.140 | 0.416 |
| Phase 3 (τ=0.05) | krr | 4.425 ± 0.256 | 0.434 |
| Phase 3 (τ=0.05) | svr | 3.249 ± 0.172 | 0.319 |
| Phase 3 (τ=0.05) | rf | 5.152 ± 0.209 | 0.506 |
| Phase 3 (τ=0.05) | xgb | 3.197 ± 0.144 | 0.314 |
| Phase 5 (no filter) | ridge | 3.998 ± 0.223 | 0.392 |
| Phase 5 (no filter) | krr | 4.486 ± 0.339 | 0.440 |
| Phase 5 (no filter) | svr | 3.161 ± 0.155 | 0.310 |
| Phase 5 (no filter) | rf | 5.156 ± 0.200 | 0.506 |
| Phase 5 (no filter) | xgb | 3.012 ± 0.078 | 0.296 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.219 ± 0.244 | 0.512 |

### elst

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 4.269 ± 0.216 | 0.423 |
| Phase 2 (τ=1e-10) | krr | 5.173 ± 0.254 | 0.512 |
| Phase 2 (τ=1e-10) | svr | 3.812 ± 0.199 | 0.378 |
| Phase 2 (τ=1e-10) | rf | 5.348 ± 0.106 | 0.530 |
| Phase 2 (τ=1e-10) | xgb | 3.763 ± 0.121 | 0.373 |
| Phase 3 (τ=0.05) | ridge | 4.816 ± 0.159 | 0.477 |
| Phase 3 (τ=0.05) | krr | 5.290 ± 0.323 | 0.524 |
| Phase 3 (τ=0.05) | svr | 4.131 ± 0.229 | 0.409 |
| Phase 3 (τ=0.05) | rf | 5.535 ± 0.147 | 0.548 |
| Phase 3 (τ=0.05) | xgb | 3.986 ± 0.106 | 0.395 |
| Phase 5 (no filter) | ridge | 4.269 ± 0.216 | 0.423 |
| Phase 5 (no filter) | krr | 5.219 ± 0.246 | 0.517 |
| Phase 5 (no filter) | svr | 3.853 ± 0.184 | 0.382 |
| Phase 5 (no filter) | rf | 5.365 ± 0.084 | 0.531 |
| Phase 5 (no filter) | xgb | 3.722 ± 0.124 | 0.369 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.666 ± 0.284 | 0.561 |

### disp ⚠ ORACLE

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 1.348 ± 0.047 | 0.361 |
| Phase 2 (τ=1e-10) | krr | 1.337 ± 0.051 | 0.358 |
| Phase 2 (τ=1e-10) | svr | 1.022 ± 0.057 | 0.274 |
| Phase 2 (τ=1e-10) | rf | 1.564 ± 0.057 | 0.419 |
| Phase 2 (τ=1e-10) | xgb | 1.009 ± 0.049 | 0.270 |
| Phase 3 (τ=0.05) | ridge | 1.446 ± 0.050 | 0.387 |
| Phase 3 (τ=0.05) | krr | 1.479 ± 0.051 | 0.396 |
| Phase 3 (τ=0.05) | svr | 1.197 ± 0.060 | 0.321 |
| Phase 3 (τ=0.05) | rf | 1.600 ± 0.058 | 0.429 |
| Phase 3 (τ=0.05) | xgb | 1.116 ± 0.025 | 0.299 |
| Phase 5 (no filter) | ridge | 1.348 ± 0.048 | 0.361 |
| Phase 5 (no filter) | krr | 1.354 ± 0.059 | 0.363 |
| Phase 5 (no filter) | svr | 1.037 ± 0.060 | 0.278 |
| Phase 5 (no filter) | rf | 1.579 ± 0.044 | 0.423 |
| Phase 5 (no filter) | xgb | 1.010 ± 0.045 | 0.271 |
| Phase 4 (XGB+MACE) | XGB+MACE | 1.526 ± 0.087 | 0.409 |

### cpcm

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 2.677 ± 0.155 | 0.538 |
| Phase 2 (τ=1e-10) | krr | 2.784 ± 0.105 | 0.559 |
| Phase 2 (τ=1e-10) | svr | 2.411 ± 0.093 | 0.484 |
| Phase 2 (τ=1e-10) | rf | 3.067 ± 0.140 | 0.616 |
| Phase 2 (τ=1e-10) | xgb | 2.391 ± 0.039 | 0.480 |
| Phase 3 (τ=0.05) | ridge | 2.952 ± 0.159 | 0.593 |
| Phase 3 (τ=0.05) | krr | 2.874 ± 0.106 | 0.577 |
| Phase 3 (τ=0.05) | svr | 2.563 ± 0.056 | 0.515 |
| Phase 3 (τ=0.05) | rf | 3.078 ± 0.135 | 0.618 |
| Phase 3 (τ=0.05) | xgb | 2.441 ± 0.047 | 0.490 |
| Phase 5 (no filter) | ridge | 2.677 ± 0.155 | 0.538 |
| Phase 5 (no filter) | krr | 2.792 ± 0.096 | 0.561 |
| Phase 5 (no filter) | svr | 2.418 ± 0.098 | 0.486 |
| Phase 5 (no filter) | rf | 3.054 ± 0.139 | 0.613 |
| Phase 5 (no filter) | xgb | 2.382 ± 0.060 | 0.479 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.899 ± 0.089 | 0.582 |

### cds

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 0.409 ± 0.020 | 0.714 |
| Phase 2 (τ=1e-10) | krr | 0.380 ± 0.017 | 0.664 |
| Phase 2 (τ=1e-10) | svr | 0.338 ± 0.019 | 0.590 |
| Phase 2 (τ=1e-10) | rf | 0.384 ± 0.016 | 0.670 |
| Phase 2 (τ=1e-10) | xgb | 0.291 ± 0.017 | 0.509 |
| Phase 3 (τ=0.05) | ridge | 0.423 ± 0.009 | 0.738 |
| Phase 3 (τ=0.05) | krr | 0.410 ± 0.016 | 0.715 |
| Phase 3 (τ=0.05) | svr | 0.369 ± 0.019 | 0.644 |
| Phase 3 (τ=0.05) | rf | 0.401 ± 0.011 | 0.699 |
| Phase 3 (τ=0.05) | xgb | 0.309 ± 0.012 | 0.538 |
| Phase 5 (no filter) | ridge | 0.410 ± 0.020 | 0.715 |
| Phase 5 (no filter) | krr | 0.383 ± 0.016 | 0.669 |
| Phase 5 (no filter) | svr | 0.340 ± 0.018 | 0.593 |
| Phase 5 (no filter) | rf | 0.384 ± 0.016 | 0.669 |
| Phase 5 (no filter) | xgb | 0.294 ± 0.014 | 0.514 |
| Phase 4 (XGB+MACE) | XGB+MACE | 0.446 ± 0.013 | 0.779 |

